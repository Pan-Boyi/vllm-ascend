# CANN 官方 DeepSeek-V4.1-Flash day-0 支持 深度调研报告

调研时间：2026-09-10
调研目标：判断 `cann/cann-recipes-infer` 的 V4.1 day-0 支持对我们在 vllm-ascend + A5 上适配 V4.1 有多大帮助。

> ## ⚠️ 本报告的两条结论已被实测修正（`dsv41-kernel` 交叉验证，2026-09-10）
>
> **1.「MXFP8 离线 broadcast scale 展开成 1×32」—— 这是 CANN 转换后 checkpoint 的状态，
> 不是我们手上这份。**
>
> 官方那句 `# Scales are repeated offline`（`raw/dsv41/models/modeling_deepseek.py:2209`）
> 服务的是**已转换**的变体。区分器是 `weight_block_size`：
>
> | | DeepSeek 原生发布版（**我们手上的**） | CANN `convert_model.py` 转换后 |
> |---|---|---|
> | `weight_block_size` | `[32, 32]` | `[1, 32]`（`convert_config.py:136` 写死）|
> | `wkv.scale`（对 `[512, 5120]` 的权重） | **`[16, 160]`** 每 32×32 一个 | `[512, 160]` 每行每 32 列一个 |
> | 加载时 | 需要 broadcast 展开 | 只需 `view(uint8)` |
>
> 而这两个变体在 vllm-ascend 里**各有一条现成的路**，终点是同一个布局
> （`w8a8_mxfp8.py:79` 已展开变体 / `:523` 块变体，后者在
> `process_weights_after_loading` 里展开）。**我们走块变体那条，正确，不需要改。**
>
> 这也和原 tilelang kernel 对得上（见 `ALIGNMENT_PROTOCOL.md` 第 10.6 节）：
> `fp8_gemm` 的权重 scale 本来就是 **2D block×block**（`inference/kernel.py:231-232`），
> `fp4_gemm` 的才是 **1D `[N, K/32]`**（`:514-515`）。两条路要两种形状。
>
> **2.「`hc_pre` 的 d=5120 被官方算子拒绝，华为自己也退回纯 PyTorch」—— 对 CANN 的
> `MhcPreSinkhorn` 成立，对我们仓内的 `HcPre` 不成立。**
>
> 仓内 `csrc/moe/hc_pre/` 是**另一个算子**，实测改一行 `TORCH_CHECK` 就接受 d=5120
> （重编 42.8 秒），且 4096/5120/7168 全部工作，对照 CPU golden 的 `post` 误差在
> d=5120 时（3.68e-6 / 9.44e-6）比 d=4096 还好。**我们不需要 fallback。**
> 别被同名的 CANN 算子文档（「尾轴仅 4096/7168」）误导。

**所有原始文件已存于本目录 `raw/` 下**：
- `raw/tech_report.md` — 华为官方技术报告全文（23KB / 286 行）
- `raw/dsv41/` — V4.1 参考实现全部源码（8753 行，22 个文件）
- `raw/ElasticBuffer.md` — Engram host offload 的官方算子 API 文档
- `raw/op_*.md` — ops-transformer 相关算子 README
- `tree_dsv41.txt` — V4.1 目录树

---

## 0. 一句话结论

**CANN 给的是「一份可运行但功能受限的 Apache-2.0 参考实现 + 一份设计说明」，不是可直接复用的高性能方案。**
最大的四个价值是：
1. **权威解答了 MXFP8 32×32 问题**（答案：离线 broadcast scale 展开成 1×32，不做融合算子；官方注释原话「Scales are repeated offline」）；
2. **官方代码里硬编码 `attn_tp_size > 1` 直接 raise**，等于官方承认 TP=1 是当前唯一路径，我们的判断被证实；
3. **Engram 的完整 Python 参考实现（521 行）+ host offload 官方算子 `ElasticBuffer` 已开源**，这块能省掉最多工作量；
4. **两个我们卡住的算子问题拿到官方答案**：`cmp_ratio=2` 有现成算子（`MixedQuantSparseFlashMla`，1~128 全支持）；`hc_pre` 的 d=5120 **确实被官方算子拒绝**（950 上只认 4096/7168），华为自己也只能退回纯 PyTorch。

**同时必须清醒**：这次 day-0 里 **V4.1 专属的融合算子（sparse_attn / mhc_pre / compressor）无一例外被注释或 fallback 关掉**，KV cache 被强制改回 BF16，只支持 eager / TP=1 / bs-per-rank=1 / 无 MTP。**它是「正确性参考 + API 契约」，不是性能实现。**

---

## A. 这是什么东西

### A1. 定位、官方性、license

| 项 | 结论 | 出处 |
|---|---|---|
| 官方性 | 华为官方。所有文件头 `Copyright (c) 2026 Huawei Technologies Co., Ltd.` | `raw/dsv41/models/modeling_deepseek.py:3`、`utils/convert_model.py:4` |
| License | **Apache License 2.0** | `raw/dsv41/LICENSE.txt`（201 行标准 Apache-2.0 全文）；`ops/LICENSE` 同 |
| 血统 | 从 DeepSeek 开源代码迁移 + CANN 平台优化 | README：「本实践基于DeepSeek开源代码进行迁移，并在CANN平台上完成性能优化」 |
| 维护活跃度 | repo 覆盖 27 个模型（kimi_k3、glm_5_2、qwen3_5、longcat-2.0 等），V4.1 与 V4 并存，day-0 节奏明显 | `models/` 目录列表 |

### A2. 是文档还是代码？—— **两者都有，且代码是真的**

`docs/models/deepseek_v4_1/` 只有 2 项：`figures/` + `deepseek_v4.1_flash_cann_tech_report.md`。
但 `models/deepseek_v4_1/` 下有 **22 个文件 / 8753 行真实 Python 代码**（`tree_dsv41.txt`）：

```
models/modeling_deepseek.py          2450 行   主干模型（40层 MoE + CSA2 + Engram 挂载）
models/modules/attention_data.py     1073 行   attn metadata / slot_mapping / block_table 构建
utils/encoding_dsv4.py                971 行   多模态 encoding
models/modules/common_modules.py      648 行
models/modules/engram.py              521 行   ← Engram 完整实现
utils/mx_quantize.py                  450 行   ← MX 量化 (改自 microxcaling + torchao)
models/modeling_deepseek_vision.py    337 行   ViT
models/configuration_deepseek.py      313 行
models/modules/compressor.py          256 行
models/modules/indexer.py             214 行   ← 两级召回
utils/convert_model.py                180 行   ← 权重转换（32×32 → 1×32）
models/modules/op_impls/mhc.py        130 行   ← mHC
```

### A3. 跑在什么框架上 —— **自研推理栈，不是 vLLM 也不是 MindIE**

repo 根目录有 `executor/`、`module/`、`accelerator/`、`benchmark/`、`integration/`。
V4.1 代码 import 的是 `executor.core.config.InferenceConfig`、`module.linear.ReplicatedLinear`、`module.fuse_moe_gmm.FusedMoEGMM`、`module.quantization.fp8` 等（`raw/dsv41` 全局 import 列表）。
启动入口：`bash executor/scripts/infer.sh --model deepseek_v4_1 --yaml deepseek_v4_1_flash_rank_8_8ep.yaml`（README「拉起多卡推理」）。

> **对我们的含义**：模型定义无法直接 copy 到 vLLM，但**算法逻辑（Engram 哈希/门控、indexer 两级召回、mHC sinkhorn、cache 分层共享）是框架无关的，可以直接照抄**。这些正是我们估工作量最大的部分。

### A4. 硬件 / CANN 版本要求 —— **⚠️ 我们的 CANN 9.1.0 不够**

| 项 | 官方要求 | 我们 | 出处 |
|---|---|---|---|
| 芯片 | **Ascend 950 系列**（950PR / 950DT） | A5 = 950PR/DT ✅ | README「硬件要求：产品型号：Ascend 950 系列」 |
| 驱动 | **Ascend HDK 25.5.1** | 待核 ⚠️ | README「确认版本是否为 `25.5.1`」 |
| CANN | **支持 CANN 9.2.0**，且要 **weekly 版本** | 9.1.0 ❌ | README「本实践依赖 CANN 开发套件包（toolkit）和与目标硬件匹配的二进制算子包（ops），**支持 CANN 9.2.0**。请从 CANN 下载页面选择对应架构的 **weekly 版本**」 |
| torch | torch 2.9.0 + torch_npu 2.9.0.post7.**dev20260903**（daily 包） | ⚠️ | README 环境准备第 1 步 |
| 最小部署单元 | **8 chips** | ⚠️ 我们是单卡目标 | README 末表：「Ascend 950 / DeepSeek-V4.1 Flash / Hybrid MXFP8-MXFP4 / **8**」 |
| A2 / A3 | Engram 的 `ElasticBuffer` **A3/A2 明确不支持** | — | `raw/ElasticBuffer.md`：「Atlas A3 训练系列产品/Atlas A3 推理系列产品：**不支持**」「Atlas A2 …：**不支持**」 |

**结论：这套 recipe 依赖 CANN 9.2.0 weekly + HDK 25.5.1 + torch_npu daily 包，是一条比我们当前基线更新的软件栈。**

---

## B. 🔴 最高优先问题：MXFP8 32×32 二维 block 的权威答案

### 官方答案：**A5 只吃 1×32，32×32 靠离线 broadcast 展开，不写融合算子。**

**证据 1 — README「转换权重」章节（原文）：**
> 「DeepSeek-v4.1的原始权重中使用了 **32x32-block 的量化，通过 broadcast scale 可以将其转化为 1x32-group 的标准 MXFP8 量化**。……使用 `utils/convert_model.py` 脚本完成 Hybrid FP8-MXFP4 到 Hybrid MXFP8-MXFP4 的权重转换。」

**证据 2 — `raw/dsv41/utils/convert_model.py:39-52` 实现（原文）：**
```python
BLOCK = 32
def expand_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    Expands a 32x32-block FP8 scale to 1x32-group format by repeating each
    row 32 times along dim 0.
    Args:  scale: shape (N // 32, K // 32), dtype float8_e8m0fnu
    Returns: Expanded scale tensor of shape (N, K // 32).
    """
    return scale.view(torch.uint8).repeat_interleave(BLOCK, dim=0).view(scale.dtype)
```

**证据 3 — 触发条件（`convert_model.py:130-141`）完全对应你实测的权重 header：**
```python
if (weight_name.endswith(".scale") and weight.dtype == torch.float8_e8m0fnu and weight.dim() == 2):
    w = current_state_dict[base + ".weight"]
    if (w.dtype == torch.float8_e4m3fn and w.dim() == 2
            and w.shape[0] % 32 == 0 and w.shape[1] % 32 == 0
            and weight.shape[0] == w.shape[0] // 32      # ← 两维都 /32 才算 2D block
            and weight.shape[1] == w.shape[1] // 32):
        new_scale = expand_scale(weight)
```
- `wkv [512,5120]` + scale `[16,160]`：`16 == 512/32` 且 `160 == 5120/32` → **命中，展开成 `[512,160]`**。
- routed experts `[2304,2560]` + scale `[2304,160]`：`2304 != 2304/32` → **不命中，字节原样拷贝**（docstring 明写：*"Routed experts and engram embed (already 1x32) ... are copied byte-identically"*）。

这与你从真实权重 header 得出的结论 **100% 吻合**。

**证据 4 — 转换后的 config 声明（`utils/convert_config.py:139` 与 `convert_model.py:88`）：**
```python
quantization_config["weight_block_size"] = [1, BLOCK]     # [1, 32]
weight_quant_strategy = "group" if is_mx else "block"
weight_group_size = 32 if is_mx else None
input_group_size = 32 if is_mx else 128
```

**证据 5 — 运行期 GEMM 的 scale 语义（`modeling_deepseek.py:891-895`）：**
```python
o, o_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
o = torch_npu.npu_transpose_quant_batchmatmul(
        o, self.wo_a.weight, dtype=torch.bfloat16,
        x1_scale=o_scale.view(torch.float8_e8m0fnu),
        x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
        group_sizes=(0, 0, 32), ...)     # ← group 只在最后一维，前两维为 0
```
`group_sizes=(0, 0, 32)` 表示 **分组只发生在最后一维、粒度 32**，前两维不分组 —— 即硬件 GEMM 接口只暴露一维 block-32。

### 判定
| 问题 | 权威答案 |
|---|---|
| A5 的 MXFP8 GEMM 支持哪些 scale 布局？ | **只有 1×32（沿收缩轴 group-32，E8M0）**。recipe 侧三处互证（`group_sizes=(0,0,32)`、`weight_block_size=[1,32]`、`strategy="group"/group_size=32`）+ 算子库侧硬编码 `[groupSizeM,groupSizeN,groupSizeK] = [1,1,32]`（§B5c）。**A5 唯一的二维 block scale 是 perblock 128×128 + FLOAT32，不承载 e8m0。** |
| 2D 32×32 怎么处理？ | **加载期/离线展开**（`repeat_interleave(32, dim=0)`）。 |
| 华为写融合算子了吗？ | **没有。** 官方选择的是权重转换，一次性离线做完。 |
| 硬件直接支持吗？ | **不支持。** 若支持就不需要这个转换脚本。 |
| 代价 | 只有 scale 存储膨胀 32×（脚本会打印 `+X GiB`）。**数值完全无损**——32 行本来共享同一个 scale，展开只是复制。 |

> **对我们的建议**：直接照搬 `expand_scale`（4 行代码），放到 vllm-ascend 的 weight loader 里做**加载期展开**，或离线跑一次转换脚本。**「MXFP8 2D 32×32 待验」这一项可以直接关闭，工作量 ≈ 0.5 人天。**
> 注意 scale 显存膨胀：attention/shared-expert/indexer 的 scale 从 `[N/32, K/32]` 变 `[N, K/32]`，多占 31/32 倍的 scale 空间（相对权重本体是 1/32 量级，可接受）。


### B5. 第二路独立证据：**算子库文档侧全面复核（76+ 份 README/aclnn 文档）**

上面的证据链来自 **recipe 侧**（权重转换 + 模型代码）。另有一路从 **`ops-transformer` 算子库文档侧**独立复核，结论完全一致，且给出了更硬的证据。

#### (a) `(…, ceil(K/64), 2)` 的尾维是「2 个 MX group」，**不是第二个 block 轴**

这是最容易被误读成「2D block」的形状。官方明确定义：

`gmm/grouped_matmul_activation_quant/docs/aclnnGroupedMatmulActivationQuantWeightNz.md`：
> L52：「$blocksize$ 表示MX量化结果的存储block大小，当前仅支持64，**对应2个MX量化group，每个group包含32个元素**。」
> L614：「表中 xScale、weightScale、outputScale 的 shape 最后一维为 2，**表示每个64元素的存储block中包含2个MX量化group，每个group覆盖32个元素**。」

→ `(m, ceilDiv(k,64), 2)` 展平就是 `(m, ceilDiv(k,32))`，**仍是一维 per-32**，只是按 64 元素打包存储。

#### (b) 决定性artifact：**同一个算子里 MX 与 perblock 并列，互斥**

`mc2/matmul_reduce_scatter_v2/README.md` L290-291：
> 「x1Scale：……在 **perblock** 场景下，shape为 **[ceildiv(m, 128), ceildiv(k, 128)]**……在 pertensor 和 perblock 场景下，数据类型支持 **FLOAT**。**在 mx 量化场景下，数据类型为 FLOAT8_E8M0，shape 为 (m, ceilDiv(k, 64), 2)。**」
> 「x2Scale：……在 perblock 场景下，shape为 [ceildiv(k,128), ceildiv(n,128)]。……**在 mx 场景下……当 x2 转置时 shape 为 (n, ceilDiv(k,64), 2)，不转置时为 (ceilDiv(k,64), n, 2)。**」

**同一算子的两种 scale 布局，显式互斥**：
- **2D block scale ⇒ 只能是 perblock 128×128 + FLOAT32**
- **MX（E8M0）⇒ 只能是一维 per-32**

**A5 上唯一存在的二维 block scale 是 128×128 FP32，它在物理上就不承载 e8m0。**

#### (c) 硬件 groupSize 魔数直接把答案编码进去了

`mc2/quant_grouped_mat_mul_allto_allv/README.md` L415（`mc2/allto_allv_quant_grouped_mat_mul/README.md` L259 同文）：
> 「当 gmmXScale/gmmWeightScale/mmXScale/mmWeightScale 输入是2维及以上时，且数据类型都为 FLOAT8_E8M0 时，**[groupSizeM，groupSizeN，groupSizeK] 取值组合会推导为 [1, 1, 32]**，对应 groupSize 的值为 4295032864。」

对比 `mc2/matmul_all_reduce/README.md` L286：「group_size 在 **perblock** 场景下，只支持 **549764202624**」（= [128,128,128]）。

→ **硬件层面 MX 就是字面意义的 `M=1, N=1, K=32`。** 这与我们在 recipe 里看到的 `group_sizes=(0, 0, 32)` 和 NPU 单测报的 `EZ0027: group_sizes must be [1, 1, 32]` **三方吻合**。

#### (d) scale 永远沿「被规约的那根轴」，语义最清晰的表述

`attention/block_sparse_attention/docs/aclnnBlockSparseAttentionV3.md` L49-75：
> qDequantScale / kDequantScale：「MX 量化模式，**仅支持 Rowwise E8M0 Micro Scaling**」— shape `[B, Nq, Sq, CeilDiv(D,64), 2]`
> vDequantScale：「MX 量化模式，**仅支持 Columnwise E8M0 Micro Scaling**」— shape `[B, Nkv, CeilDiv(Skv,64), D, 2]`

Q/K 用 Rowwise（规约轴 = D），V 用 Columnwise（规约轴 = S）。**永远一维，永远沿收缩轴。**

#### (e) ⚠️ 实操坑：**存在两种不同的物理打包，rank 不统一**

大多数算子用 `(…, ceil(axis/64), 2)`，但也有算子用**裸 `ceil(D/32)`**：

`posembedding/qkv_rms_norm_rope_cache_with_k_scale/docs/aclnnQkvRmsNormRopeCacheWithKScale.md`：
> 「`qScaleOptional` 为 FLOAT8_E8M0 `[T, Nq, ceil(D/32)]`；`kScaleCacheRef` 为 FLOAT8_E8M0 `[BlockNum, Nk, BlockSize, ceil(D/32)]`」

`attention/indexer_quant_cache` 同样是裸形式：「cacheScale.headDim ≥ scaleCol，scaleCol：MX-FP8/MX-FP4（quantMode=0/3）为 ⌈d/32⌉」。

> **语义相同（每 32 元素一个 e8m0），但张量 rank 不同。接管道时不要假设全链路 scale rank 一致——逐算子核对。** 这条对我们把 MXFP8 权重/激活/KV scale 串起来时特别容易踩。

#### (f) 其它一致的 MX 算子（均为一维 per-32 e8m0）

`gmm/grouped_matmul_swiglu_quant_v2`（「dequantMode……仅支持取值2，2表示MX量化」）、`gmm/grouped_matmul_finalize_routing`（V3 + WeightNzV2，含 MxA8W4，`x2` 为 NZ_C0_32 且 `k % 32 == 0`）、`gmm/quant_grouped_matmul_inplace_add`、`mc2/quant_all_reduce`、`mc2/quant_reduce_scatter`、`attention/fused_infer_attention_score`（V5，含 NZ 变体 `(Bn, KV_N, Bs/16, D/64, 16, 2)`）。

`mc2/mega_moe/README.md`（如果我们要动 A5 的 MoE，这条相关）：
> L420：「$\mathrm{Q}_{\text{MX}}$ 表示 MX 逐组量化（**group size = 32**）……同时输出 FLOAT8_E8M0 缩放因子」
> L593：「其中所有 MX 缩放因子的类型均为 `FLOAT8_E8M0`，**量化粒度均为 32 个连续元素**。」
> L924：「当前版本**仅支持 MXFP 量化模式**（`dispatchQuantMode` = 4）」

#### (g) 负结果（明确「未获取到」，非推测）

- 对全部 76+ 份已抓取文档执行 `grep -rniE "32x32|32\*32|32×32|\(32, 32\)"` → **0 命中**。
- 所有 `perblock` 命中项全部是 **128**（`gsN=gsK=128`、`ceildiv(·,128)`、groupSize 549764202624）且 scale 为 **FLOAT32**，**从无 e8m0**。
- `experimental/gmm/` 只有 `CMakeLists.txt`，无算子。
- `attention/quant_flash_attn/` 与 `mc2/all_gather_matmul_v3/` **无 README**（raw 抓取返回 404）。MxFP4 文档只存在于 `experimental/attention/quant_flash_attn/README.md`。
- `ffn/ffn/README.md` 完全没有 MX/E8M0，只有 per-channel / per-group 伪量化配 FP16/BF16 `antiquantScale` —— **不是 A5 的 MX 路径**，别误用。

#### 🔒 最终裁定（两路独立证据一致）

> **昇腾 950 上没有任何 GEMM 接受二维 32×32 block scale。**
> MXFP8 的 scale **必须**按「沿收缩轴一维 per-32 的 E8M0」排布。
> A5 上唯一的二维 block scale 是 **perblock 128×128 + FLOAT32**，那是一个独立模式，物理上无法承载 e8m0。
> **V4.1 权重里的 32×32 只能靠 broadcast 展开成 1×32 —— 这正是华为 `convert_model.py` 做的事。**

---

## C. 算子清单（B5/B6）

### C1. 参考实现里**实际调用**的算子

| 算子 | 来源 | 状态 |
|---|---|---|
| `torch_npu.npu_dynamic_mx_quant` | torch_npu | CANN 已有 |
| `torch_npu.npu_transpose_quant_batchmatmul`（MXFP8 GEMM，`group_sizes=(0,0,32)`） | torch_npu | CANN 已有 |
| `torch_npu.npu_transpose_batchmatmul` / `npu_fused_matmul` | torch_npu | CANN 已有 |
| `torch_npu.npu_rms_norm` / `npu_add_rms_norm` / `npu_swiglu` | torch_npu | CANN 已有 |
| `torch_npu.npu_moe_init_routing_v2` / `npu_moe_re_routing` / `npu_moe_gating_top_k` / `npu_moe_finalize_routing` | torch_npu | CANN 已有 |
| `torch_npu.npu_moe_distribute_dispatch_v2` / `combine_v2`（含 `y_dtype=torch_npu.float4_e2m1fn_x2`） | torch_npu | CANN 已有（MXFP4 EP dispatch） |
| `torch.ops.cann_ops_nn.swiglu_group_quant` | **cann_ops_nn**（独立开源 repo） | 需装 `cann_ops_nn` |
| `torch.ops.cann_ops_transformer.inplace_partial_rotary_mul` | **ops-transformer** | 需装 |
| `cann_ops_transformer.ops.ElasticBuffer`（Engram host offload） | **ops-transformer / mc2** | 需装，**仅 950PR/DT** |
| `torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata` | ops-transformer | 代码里引用，但调用点被注释 |
| `torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata` | ops-transformer | 同上 |
| `cann_ops_transformer.compressor` / `flash_attn` / `flash_attn_metadata` | ops-transformer | import 了 |

### C2. ⚠️ 关键发现：**核心融合算子在这次 day-0 里是被关掉的**

三处「TODO / 注释掉」，全部指向同一件事——**V4.1 的融合稀疏注意力 / mHC kernel 还没 ready**：

1. `modeling_deepseek.py:750-751`
   ```python
   # TODO: remove the following code after the new kernel is ready
   # self.sparse_attn_ops = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla
   ```
   → 实际走的是 `Attention.sparse_attn()`：**纯 PyTorch 的 gather + einsum + 手写 softmax（含 attn_sink）**（`modeling_deepseek.py:809-860`）。

2. `modeling_deepseek.py:2125`
   ```python
   # self.generate_kernel_metadata(attn_metadata, is_prefill)
   ```
   → metadata 生成整个被注释；且 `generate_kernel_metadata` 里写的还是 V4 的 **c1a / c4a / c128a**（`cmp_ratio: 4` / `128`），**没有 V4.1 的 c2a**（`modeling_deepseek.py:2068-2078`）。这段是从 V4 抄过来的、当前不生效的残留代码。

3. `models/modules/op_impls/mhc.py:84-100 / 116-119`
   ```python
   # TODO: change to mhc_pre fusion kernel
   # @register_op_impl(op_type="hc_pre", func_key="hc_pre_ascendc")
   # def hc_pre_ascendc(...):
   #     y, post, comb, pre, _, _, _, _, _ = torch.ops.cann_ops_transformer.mhc_pre_sinkhorn(...)
   # @register_op_impl(op_type="hc_post", func_key="hc_post_ascendc")
   # def hc_post_ascendc(x, residual, post, comb):
   #     y = torch.ops.cann_ops_transformer.mhc_post(residual, comb, x, post)
   ```
   → 只注册了 `hc_pre_native` / `hc_post_native`（纯 PyTorch）。

4. KV cache 全是 **BF16**，不是技报说的 FP8/FP4：
   `modeling_deepseek.py:1252-1253`
   ```python
   self.cache_dtype = torch.bfloat16
   self.state_cache_dtype = torch.float32
   ```

> **结论：技术报告描述的是「目标设计」，开源代码是「功能正确的 BF16 参考实现」。两者有明显落差，引用技报数据时必须区分。**

### C3. ops-transformer 里已开源的相关算子（含完整 AscendC 源码 op_host/ + op_kernel/）

`experimental/attention/` 下与我们相关的：
`quant_sparse_flash_mla`、`kv_quant_sparse_attn_sharedkv`(+`_metadata`)、`sparse_attn_sharedkv`(+`_metadata`)、`turbo_quant_sparse_attn_sharedkv`、`sparse_flash_attention_antiquant`(+`_metadata`)、`quant_lightning_indexer`(+`_metadata`)、`quant_compressor`、`compressor`、`quant_sals_indexer`、`stem_indexer`、`typhoon_mla`、`nsa_compress_attention_infer`、`blitz_sparse_attention` 等。


### C5. ⚠️ experimental 算子**不随 CANN 二进制发布，必须自己编译**

`raw/experimental_usage.md`（`experimental/attention/Attention融合算子Experimental使用说明.md`）：
- 前提条件：`python >= 3.10.0`、`torch >= 2.7.0`、`gcc >= 9.0.0`、`torch_npu >= 2.7.0`
- 算子表里每个条目的「说明」都带同一句：**「算子 torch 接口调用依赖 torch_ops_extension，具体安装方法见 [cann-recipes-infer/ops/ascendc 安装指导]」**

该文档列出的 experimental attention 算子（**注意：这是一份比目录列表更短的「已文档化」清单**）：
`compressor`、`kv_quant_sparse_attn_sharedkv`(+`_metadata`)、`sparse_attn_sharedkv`(+`_metadata`)、`quant_lightning_indexer`(+`_metadata`)、`stem_indexer`(+`_metadata`)、`blitz_sparse_attention`、`typhoon_mla`、`fused_infer_attention_score`、`fused_gdn_decode`。

- **`mixed_quant_sparse_flash_mla` 在这份说明里出现 0 次**（`grep -c "mixed_quant" = 0`），进一步佐证 §H 的「未获取到」。
- 图模式支持：这些算子 **Ascend IR 一栏全是 ×，aclgraph 一栏是 √**。
- 执行单元：AI Core。

> **风险升级**：这些算子要通过 `cann-recipes-infer/ops/ascendc` 的 `torch_ops_extension` **源码编译安装**，不是 `Ascend-cann-kernels-*.run` 里现成的二进制。我们要用 `quant_sparse_flash_mla` / `compressor` 就得引入这条自编译链路。

### C4. 补：Compressor 也是「有融合算子但默认关掉」

`raw/dsv41/models/modules/compressor.py:38`：
```python
FALL_BACK_COMPRESSOR = 1
```
`compressor.py:216-226`：`if FALL_BACK_COMPRESSOR: kv = self._native_compressor_fallback(...)  else: kv = compressor(**cmpr_input_kwargs)`
→ 默认走 **native PyTorch fallback**，`cann_ops_transformer.compressor` 融合算子被绕过。

但调用参数表本身很有价值（`compressor.py:200-215`），它暴露了融合 Compressor 的**目标 API 且明确带 `cmp_ratio=2`**：
```python
cmpr_input_kwargs = {
    "x": x.view(-1, self.dim), "wkv": self.wkv.weight, "wgate": self.wgate.weight, "ape": None,
    "state_block_table": state_block_table, "seqused": seq_used_q, "start_pos": start_pos,
    "cmp_ratio": self.compress_ratio,          # V4.1 = 2
    "coff": 1,
    "cache_mode": 1,                            # 1: contiguous buffer; 2: ring buffer
    "cu_seqlens": cu_seqlens,
    "state_cache": kv_cache.state_cache.flatten(-2),
}
```

> **模式总结：这次 day-0 里，V4.1 专属的融合算子（sparse_attn / mhc_pre / compressor）无一例外被注释或 fallback 关掉，全部走纯 PyTorch。**开源的是「正确性参考 + 目标 API 契约」，不是性能实现。

---

## D. 🔴 第二个权威结论：TP=1 是官方也没绕过去的硬约束

### D1. 官方 yaml 全线 TP=1

`raw/dsv41/config/deepseek_v4_1_flash_rank_8_8ep.yaml`（原文）：
```yaml
parallel_config:
  world_size: 8
  attn_tp_size: 1
  dense_tp_size: 1
  moe_tp_size: 1
  embed_tp_size: 1
  lmhead_tp_size: 1
  o_proj_tp_size: 1
```

### D2. 官方代码**硬 raise**

`raw/dsv41/models/modeling_deepseek.py:1675-1676`（原文）：
```python
if parallel_config.attn_tp_size > 1:
    raise ValueError(f"{parallel_config.attn_tp_size=} is not supported yet!")
```

### D3. 算子文档给出根因，与我们实测的 `EZ1008` 完全一致

`raw/op_kv_quant_sparse_attn_sharedkv.md` 约束说明（原文）：
> 「参数q中的 **N1 当前支持 64/128**，ori_kv、cmp_kv 中的 KV_N 仅支持 1。」

`raw/op_quant_sparse_flash_mla.md` 约束说明（原文，同样）：
> 「参数q中的 **N1 当前支持 64/128**，ori_kv、cmp_kv 中的 KV_N 仅支持 1。」

heads=64，TP=2 → N1=32 → 不在 {64,128} 内。**华为官方没有解决这个约束，他们也是 TP=1 + DP/EP。**

> 我们的 memory note「A5 上 DSV4 必须 TP=1」得到官方源码级确认。**不需要再花时间找「TP>1 的隐藏路径」了——不存在。**

### D6. 补：V4 在 950 上**同样**是 TP=1 —— 这是 A5 代际通例，不是 V4.1 特例

`models/deepseek_v4/config/ci_950/deepseek_v4_flash_rank_16_16ep.yaml`（原文）：
```yaml
parallel_config:
  world_size: 16
  cp_size: 1
  attn_tp_size: 1
  dense_tp_size: 1
  moe_tp_size: 1
  embed_tp_size: 1
  lmhead_tp_size: 1
  o_proj_tp_size: 1
```
V4（16 卡 16EP）与 V4.1（8 卡 8EP）在 950 上的并行配置**完全同构：全线 TP=1，靠 EP/DP 扩展**。

### D4. 官方并行策略全貌

- 部署：`world_size=8`，**纯 EP（8ep）+ DP**，attention 侧 TP=1。技报 §4.1：Prefill「可使用 **CP + EP**」、Decode「可使用 **DP + EP**」、Engram「多机多卡切分，host DDR 内存作为扩展」。
- **但 CP 在开源代码里也没实现**：`modeling_deepseek.py:1679-1680`
  ```python
  if parallel_config.cp_size > 1:
      raise ValueError(f"{parallel_config.cp_size=} is not supported yet!")
  ```
- Engram 独立 TP：`engram_tp_size=8`，README 明写「当前支持 `tp_size`=`world_size`」。

### D5. 其它「not supported yet」清单（`modeling_deepseek.py:1657-1690`）

| 限制 | 原文 |
|---|---|
| 只支持 eager | `if exe_mode != "eager": raise ValueError(f"{exe_mode=} does not supported!")` |
| **不支持 MTP / 投机解码** | `if model_config.next_n > 0: raise ValueError(... "Please set next_n = 0")` |
| 每 DP rank batch 只能 1 | `if scheduler_config.batch_size_per_dp_rank > 1: raise ...` |
| Engram 不支持投机解码 | `engram.py:238`：`input_ids.view(-1, 1) # not support speculative decoding` |

---

## E. Engram（B7）—— 覆盖度最高的一块

### E1. 参考实现：**完整、可抄**（`raw/dsv41/models/modules/engram.py`，521 行）

包含：
- `build_compressed_token_map()`：NFKC → NFD → StripAccents → Lowercase → 空白折叠 → 压缩词表（99,092）。**含一个私有区哨兵字符 `` 的坑**（防止单空格 token 被 Strip 吃掉后与别的 token 合并）——这种细节我们自己踩会花很久。
- `compute_hash_multipliers()`：per-layer RNG `np.random.default_rng(10007 * layer_id)`，奇数乘子，且 `multiplier_bound = (int64_max // vocab_size) // 2` 防溢出。
- `EngramLayout.from_args()`：从 `engram_vocab_size=16,000,000` 起依次取**互不重复的素数**做每个 (ngram_size, head) 的桶模数。
- `NgramHashState._get_ngram_hashes()`：滚动 XOR 累积 2/3/4-gram，`% primes` 后 `+ offsets`。
- `MultiHeadEmbedding`：TP 切分 + offload/非 offload 两条路径 + allgather/reduce_scatter 处理 attn_dp ≠ engram_dp。
- `Engram.forward()`：`wkv` Linear (6144→25600) → split 出 hc_mult 路 key + 共享 value → **带符号 sqrt + sigmoid 门控** → 加回残差流；`image_mask` 处 gate 置 0。

关键配置（`configuration_deepseek.py:207-214`，与你的估算完全对上）：
```python
engram_layer_ids=[1, 14]
engram_num_embeddings=[384008192, 384018432]
engram_max_ngram_size=4
engram_vocab_size=16000000
engram_n_heads=8
engram_head_dim=256
engram_compressed_vocab_size=99092
```

### E2. Host offload：官方走 **ElasticBuffer / URMA-RDMA**，不是 UVA

**技报 §2.3.2**（原文）：
> 「Engram Table offload 到 host 侧，通过 UB 能力从分布式的 host memory 中 fetch engram 表。通过 **URMA 转 UBMem** 的方式进行互通：1）将 host memory 注册到 device 侧…… 2）使用 hcomm 的能力建立 urma 通信连接 3）使用 AIV 组网通信任务放入到 URMA 通信队列中，后续 urma 后台执行，**不占用计算核与计算进行 overlap**」

**算子文档 `raw/ElasticBuffer.md`**（原文）：
> 产品支持情况：**Ascend 950PR/Ascend 950DT：支持**；Atlas A3/A2/310：**不支持**。
> 「**推理**（`with_grad=False`）：将 `storage` 数据拷贝到内部 **host pinned 共享内存段**。」
> 「host pinned 共享内存段：指通过 **`aclrtMallocHost`** 分配的页锁定（pinned）主机内存，并经 **`aclrtHostRegisterV2`** 映射后获得设备可访问地址。该内存段既可被本卡NPU直接访问，也可被**远端 rank 的 NPU 通过 RDMA 读取**」
> `engram_fetch(indices) -> Callable`：「通过 **RDMA one-sided read** 从远端 rank 的 host pinned 共享内存抓取数据。接口采用**异步设计**：调用后立即返回一个 callable，执行该 callable 时阻塞等待 RDMA 传输完成并返回结果 tensor。」
> 索引映射：`rank_id = ⌊global_idx / num_entries⌋`，`local_idx = global_idx mod num_entries`。**越界元素返回 0**（不报错）。

代码调用（`engram.py:426-432`）：
```python
if self.engram_offload:
    wait_callable = self.engram_buffer.engram_fetch(flat_input_ids.reshape(-1).to(torch.int32))
    output = wait_callable()
    return output.view(*shape, self.embedding_dim)
```

> **回答你的问题「华为用不用 `pinned_mem_register:True` 这条 UVA 路？」——不用。**
> 他们用的是 `aclrtMallocHost` + `aclrtHostRegisterV2` + HCCL/URMA 的 **RDMA one-sided read**，而且是**跨 rank 分布式**的（表按 rank 切，任一 rank 可读任一 rank 的 host 段）。这是比 UVA 更强的能力，但**绑死 950PR/DT + `cann_ops_transformer` 的 mc2 模块**。
> 如果我们要在 vllm-ascend 上做，两条路：(a) 直接调 `ElasticBuffer`（需要装 ops-transformer，且要接 HCCL group）；(b) 走我们已确认的 `PYTORCH_NPU_ALLOC_CONF=pinned_mem_register:True` UVA 单卡路径（更简单，但没有跨卡 RDMA、也没有后台 overlap）。**单卡场景 (b) 完全够用。**

### E3. ⚠️ 落差：开源路径里 Engram 表是 **BF16**，不是 FP8

- 技报 §2.2.1 说：「**Engram 表**：大表保持 **FP8 存储**，查询时按行反量化到 BF16。」§2.3.1 说「量化时，采用 mxfp8 量化，其 scale factor 数据相对较小，全量存储到 device 内存中」，还有专门的 `ops_engram_quant.png` 图。
- **但代码不是这么做的**：`engram.py:325-355` 的 `fp8_weight_loader()` 在**加载期就把 FP8 反量化成 BF16**：
  ```python
  dequant_weight = (quant_weight[shard_start:shard_end].to(torch.float32)
                    .unflatten(-1, (-1, block_size)) * local_scale.unsqueeze(-1)
                   ).flatten(-2).to(torch.bfloat16)
  ```
  且 `offload_weights()` 强制检查：
  ```python
  if self.weight.dtype != torch.bfloat16:
      raise TypeError("ElasticBuffer only accepts BF16 Engram embeddings, ...")
  ```
- `ElasticBuffer.md` 的 `engram_write` 参数说明佐证：「数据类型支持 **`bfloat16`、`float16`、`float32`**」——**没有 FP8**。

> **含义**：技报里「FP8 存表 + AIV gather scale 反量化」的优化**没有出现在开源接口里**。开源路径的 host 内存占用是技报数字的 **2 倍**（2×384M×256×2B ≈ 394 GB，8 卡切分后 ≈ 49 GB/rank host DDR）。
> **这是一个我们没意识到的坑**：如果按技报的 197GB 规划 host 内存，实际会翻倍。

---

## F. 稀疏注意力 / CSA2 / 两级召回（B8）

### F1. CSA2 跨层 KV 共享：**框架层实现完整，可抄**

`modeling_deepseek.py:1225-1320` 建了三类共享 cache（`_Scope.SHARED` / `SHARED_HEAD` / `PER_LAYER`）：
- `win_kv`（每层私有，`sliding_window=128`）
- `cmp_cache` + `indexer_cache`（**只在 `kv_source_layers=[2,8,14,20]` 建，其余层共享**，`block_size = block_size * ratio`，`manager_key=f"c{ratio}a_cmp_kv"`）
- `state_cache`（**只在 `cmp_ratio > 1` 的源层建**，`num_head=2`（kv/score 两路）、`dtype=float32`、`sliding_window=ratio`）

分发逻辑（`modeling_deepseek.py:794-807`）非常干净：
```python
if self.is_kv_source:            latent = self.compressor(...); comp_idx = self.indexer(..., latent, ...)
elif self.is_index_source_kv_share: comp_idx = self.indexer(..., None, ...)   # 只跑 indexer 不产 KV
else:                            comp_idx = kv_cache.shared_idx              # 纯消费
```
配置（`configuration_deepseek.py:165,174,175`）：
```python
kv_source_layers=[2, 8, 14, 20]
compress_ratios=(0, 0) + (2,) * 18 + (1,) * 20
index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36)
num_hidden_layers=40, num_attention_heads=64, num_key_value_heads=1, head_dim=512
```

### F2. 两级候选筛选：**纯 PyTorch 参考实现，可抄，但没有融合算子**

`raw/dsv41/models/modules/indexer.py:189-214` `select_candidate_blocks()`：
- 一级：`logits` 按 `candidate_block_size` 分块 → `amax` → `topk(candidate_topk_blocks)` → bool mask；**把 query 最新位置所在的半满块用 `+inf` 强行钉住**（注释解释得很清楚：「it holds the most recent tokens but could otherwise be outscored by an older, full block」）。
- 二级：非源层 `index_score.masked_fill(~candidates, -inf)` 后再 `topk(index_topk)`。
- 源层 `is_candidate_source` 产出 `kv_cache.candidates`，后续层 `uses_candidates` 消费。

`candidate_source_layer` / `candidate_topk_blocks` / `candidate_block_size` 三个字段**不在 `configuration_deepseek.py` 的默认值里**，只从 HF `config.json` 读——说明真实权重的 config.json 带这些字段。**未获取到其具体数值**（我们没有权重）。

### F3. ⚠️ `cmp_ratio=2`：老算子不支持，新算子支持

| 算子 | cmp_ratio 支持 | 出处 |
|---|---|---|
| `kv_quant_sparse_attn_sharedkv`（我们现在用的） | 「kv压缩场景**支持 4/128**，非kv压缩场景仅支持传1」 | `raw/op_kv_quant_sparse_attn_sharedkv.md` 参数表 |
| `sparse_attn_sharedkv` | 「仅支持输入 **4 或 128**」；且 **950PR/DT 标 ×（不支持）** | `raw/op_sparse_attn_sharedkv.md` |
| `turbo_quant_sparse_attn_sharedkv` | 「Host 保留原实现对 `cmp_ratio=4/128` 的支持」 | `raw/op_turbo_quant_sparse_attn_sharedkv.md` |
| **`quant_sparse_flash_mla`** | **「kv压缩场景支持 1~128」** ✅，950PR/DT **√** | `raw/op_quant_sparse_flash_mla.md` 参数表 |

> **这是一个可以直接省工作量的发现**：`quant_sparse_flash_mla` 已经把 `cmp_ratio` 泛化到 **1~128 任意值**，`cmp_ratio=2` 天然支持。
> 但注意两点落差：
> 1. `quant_sparse_flash_mla` 的 KV 是 **HIFLOAT8 per-tensor**（`qkv_quant_mode` 仅支持 1 = HIFLOAT8 per_tensor），**不是 V4.1 要的 FP8-SWA + FP4-压缩混合精度**。
> 2. V4.1 真正要的 `mixed_quant_sparse_flash_mla`（recipe 里 import 的那个）在开源 ops-transformer 里**未找到同名算子**——见下方 §H「未获取到」。
>
> 另：`cmp_sparse_indices` 的 `K2` **仅支持 512/1024**（两处 README 一致），恰好对上 `index_topk=512`。

---

## G. 🎯 交付表：我们的工作量项 × CANN 官方覆盖 × 可复用性 × 修正人天

| # | 工作量项 | 原估 | CANN 覆盖情况 | 形态 | 可复用性 | **修正人天** | 依据 |
|---|---|---|---|---|---|---|---|
| 1 | **Engram**（2 层 ×3.84 亿行 + n-gram 哈希 + gather + 反量化 + host offload） | 12~18 | **大部分覆盖** | ✅ **521 行 Apache-2.0 Python 源码**；offload 有官方算子 `ElasticBuffer`（950PR/DT 专属，二进制+文档） | 算法部分**直接抄**（哈希/素数桶/压缩词表/门控 100% 可移植）；offload 二选一：调 `ElasticBuffer` 或走我们的 UVA | **5~8** | `engram.py`、`ElasticBuffer.md` |
| 2 | **CSA2 跨层 KV 共享**（40层只有4层存KV、8层跑indexer） | 10~12 | **覆盖（框架层）** | ✅ 源码（cache 分配 + 源层/消费层分发逻辑） | 逻辑直接抄，但要重写成 vLLM 的 KVCacheManager 形态 | **6~8** | `modeling_deepseek.py:1225-1320, 794-807` |
| 3 | **两级候选筛选**（2048 blocks 粗筛 → 512 token 精筛） | 6 | **覆盖（参考实现）** | ✅ 26 行纯 PyTorch（`select_candidate_blocks`） | **直接抄**；无融合算子，性能自负 | **1.5~2** | `indexer.py:189-214` |
| 4 | **`cmp_ratio=2` kernel 支持** | 6~7 | ✅ **完全覆盖** | ✅ **`MixedQuantSparseFlashMla`**（`attention/`，950PR/DT 专属）`cmp_ratio` **支持 1~128**，且原生支持 SWA+CSA+HCA 与 per-token-group 量化 KV（`quant_mode=2`：nope FP8_E4M3 + rope bf16 + **scale FLOAT8_E8M0**） | 直接调用；主要成本是对齐 `metadata`（必传 [1024]）与 KV 打包布局 | **2~3** | §M2 / `raw/ops_survey/attention_mixed_quant_sparse_flash_mla_README.md` |
| 5 | **Single-Pass mHC**（`hc_pre` 要多返回 `pre`） | 1 | **覆盖（签名已有）** | ⚠️ `mhc_pre_sinkhorn` 的输出**第 4 个就是 `pre`**：`y, post, comb, pre, _, _, _, _, _ = ...`；且 `n=4`/`numIters=20` 正是 V4.1 的 `hc_mult`/`hc_sinkhorn_iters`。但受 #6 的 c=5120 限制，实际用不上 | 签名确认可用；有 native 兜底 | **0.5** | `op_impls/mhc.py:90-100`、§N1 |
| 6 | **`hc_pre` 硬拒 d=5120** | 1~2 | 🔴 **官方证实：算子确实不支持 5120** | ❌ `MhcPreSinkhorn` 在 **950PR/DT 上「c 仅支持 4096、7168」**（A2/A3 反而只要 128 对齐）。官方 recipe 因此整段注释掉 AscendC 路径，退回 native PyTorch | 无 kernel 可用；照官方做法走 native 兜底，同时向华为提 issue 要 c=5120 | **0.5（native 兜过）** | §M1 / `raw/ops_survey/mhc_mhc_pre_sinkhorn_README.md` |
| 7 | **DSpark drafter 3 block** | 8~12 | **完全没覆盖** | ❌ 官方**显式禁止**：`if model_config.next_n > 0: raise ...`；Engram 也注明 `not support speculative decoding` | 无 | **8~12（不变）** | `modeling_deepseek.py:1683-1684`、`engram.py:238` |
| 8 | **MXFP8 2D 32×32 block** | 待验 | **完全覆盖，答案明确** | ✅ 4 行 `expand_scale()` 离线/加载期 broadcast；`weight_block_size=[1,32]` | **直接抄** | **0.5** | `convert_model.py:39-52`、README「转换权重」 |
| | **合计** | **44~58+** | | | | **24~34** | |

**净省：约 20~24 人天。**

### 额外净收益（不在原表里的）
| 项 | 价值 |
|---|---|
| 完整模型 config 常量 | `kv_source_layers` / `compress_ratios` / `index_source_layers` / engram 全套超参，省掉逆向权重的时间 |
| `mx_quantize.py`（450 行） | MXFP4/FP8 量化-反量化参考（改自 microxcaling + torchao），可作对拍基准 |
| MoE MXFP4 EP dispatch 用法 | `npu_moe_distribute_dispatch_v2(y_dtype=torch_npu.float4_e2m1fn_x2, quant_mode=4)` 的正确调用姿势 |
| `expand_scale` + `generate_quant_config` | compressed-tensors 格式的 quant_config 生成，vLLM 侧可直接对接 |

---

## H. 明确「未获取到」的项

- ~~**`mixed_quant_sparse_flash_mla` 算子本体**：未获取到。~~ **→ 已在 §N2 找到并修正**：它不在 `experimental/attention/` 而在 `attention/` 正式目录下，README 完整。（`experimental/attention/` 与 `Attention融合算子Experimental使用说明.md` 里确实都没有它，所以最初的检索方向本身是错的。）
- **`candidate_source_layer` / `candidate_topk_blocks` / `candidate_block_size` 的具体数值**：只从 HF `config.json` 读，不在开源默认值里。**未获取到。**
- **精度数据 / 性能数据（B12）**：技报**没有给任何 accuracy 或 tokens/s 数字**。只给了 KV 存储对比表（1M 序列：V4 ≈3.0 GB → V4.1 ≈0.8 GB，「降低约 3.9× KV存储」）。README 指定了数据集（InfiniteBench `longbook_qa_eng` + MMMU）但**没有跑分结果**。**未获取到。**
- **CANN 9.1.0 能否跑**：README 只说「支持 CANN 9.2.0」，**未说明 9.1.0 是否可用**。
- **`hc_pre` 对 d=5120 的支持**：AscendC 路径在 recipe 里被注释，**官方文档未获取到 hidden dim 约束**（`experimental/mhc/` 的 README 调研另行进行）。
- **git clone 全仓失败**（网络超时），本次调研全部通过 gitcode API + raw 文件抓取完成；`ops/`（`ascendc`/`tilelang`/`pypto`/`shmem`/`cannbot_dsl`）、`executor/`、`module/` 的具体内容**未逐一展开**。

---

## I. 风险

| 风险 | 级别 | 说明 |
|---|---|---|
| **依赖 CANN 9.2.0 weekly** | 🔴 **最高** | 我们是 9.1.0。**实测确认 9.1.0 分支缺 `MixedQuantSparseFlashMla` / `quant_sparse_flash_mla` / `quant_lightning_indexer_v2` / `ElasticBuffer` / `quant_compressor`（全部 404）**，见 §N。不升级 = Engram offload + cmp_ratio=2 稀疏注意力 + 量化 KV 三块算子全无 |
| **依赖 torch_npu daily 包** | 🟠 中 | `torch_npu-2.9.0.post7.dev20260903`，从 OBS 直下的日构建，不是正式 release |
| **依赖 HDK 25.5.1 驱动** | 🟠 中 | 需核对我们的驱动版本 |
| **核心融合算子未 ready** | 🔴 高 | `mixed_quant_sparse_flash_mla` / `mhc_pre_sinkhorn` 在 day-0 里全被注释掉；技报的性能设计**尚无可运行实现** |
| **`ElasticBuffer` 绑死 950 + HCCL** | 🟠 中 | A3/A2 明确不支持；需要 process group，单卡场景等于不可用 → 走 UVA |
| **Engram host 内存翻倍** | 🟠 中 | 开源路径 BF16 存表，非技报的 FP8。~394 GB 总量 |
| **最小部署 8 卡** | 🟠 中 | 与我们「单卡 vllm」目标冲突；单卡下 Engram 表（BF16 全量 394 GB）无法容纳，必须自己做 FP8 存表 + on-the-fly 反量化，**这正是技报描述但未开源的那条路** |
| **License** | 🟢 低 | Apache-2.0，可商用、可修改，保留版权声明即可 |
| **experimental 算子需自编译** | 🟠 中 | 不随 CANN kernels 包发布，需通过 `cann-recipes-infer/ops/ascendc` 的 `torch_ops_extension` 源码构建（`raw/experimental_usage.md`） |
| 非开源算子 | 🟢 低 | ops-transformer 是**全源码开源**（`op_host/` + `op_kernel/` AscendC 源码都在），不是黑盒二进制 |

---

## J. 我们没意识到的坑（B16）

1. **Engram 表在开源路径是 BF16 不是 FP8** → host/HBM 预算翻倍（详见 §E3）。单卡场景这是致命的，必须自己实现 FP8 存表 + gather 时反量化。
2. **`sliding_window=128` 的 win_kv 不持久化**，PrefixCache 命中时要**向前补算 128 token 重建**（技报 §4.2）。这条影响 vLLM 的 prefix caching 语义，我们原表里没有这一项。
3. **压缩 KV 的 block_size 是 `block_size * ratio`**（`modeling_deepseek.py:1300`），不是 `block_size` —— PagedAttention block 管理要按 ratio 缩放，容易写错。
4. **`state_cache` 是 `num_head=2` 的 FP32 中间状态**（kv/score 两路），`sliding_window=ratio` —— cmp_ratio=2 的 compressor 需要跨 step 保留半个压缩组的状态。原表里「cmp_ratio=2 kernel 支持」只算了 kernel，**没算这个 state cache 的 slot 管理**。
5. **`ori_kv`/`cmp_kv` 的 D 必须是 640 而不是 512**（`kv_quant_sparse_attn_sharedkv` 约束）：「按 kv_rope、kv_nope 及 nope_quant_scale 顺序拼接，尾部 pad 128B 对齐至 640。其中 kv_rope 为 bfloat16、rope_head_dim=64；kv_nope 为 float8_e4m3fn/hifloat8、nope_head_dim=448；nope_quant_scale 为 float8_e8m0fnu，dim = 448/64 = 7」。**量化 KV cache 的物理 layout 是 rope+nope+scale 打包的，不是简单的 512 维 FP8。**
6. **`cmp_sparse_indices` 的有效值必须前置**：「需要保证每行有效值均在前半部分，无效值均在后半部分」。indexer 输出的 `-1` 填充位置有严格顺序要求。
7. **`attn_sink`**：V4.1 的 sparse attention 带 attention sink（`sinks` 在算子里是「当前必须传入」的 FLOAT32 输入，`modeling_deepseek.py:846-852` 的 native 实现也有 `self.attn_sink`）。原表没提。
8. **官方 day-0 是 eager-only、bs/rank=1、无 MTP、无 CP** —— 别把这份 recipe 当性能基线。
9. **MX scale 的张量 rank 全链路不统一**（§B5e）：多数算子是 `(…, ceil(axis/64), 2)`，但 `qkv_rms_norm_rope_cache_with_k_scale`、`indexer_quant_cache` 用裸 `ceil(D/32)`。语义相同、rank 不同。**串权重/激活/KV 三段 scale 时逐算子核对，别假设一致。**
10. **`ffn/ffn` 不是 A5 的 MX 路径**：它只有 per-channel/per-group 伪量化配 FP16/BF16 `antiquantScale`，无 E8M0。选算子时容易误用。


---

## K. 补充：repo 定位与相邻线索（`raw/repo_README.md`）

repo 自述：「基于 CANN 平台的大模型推理优化实践，覆盖 DeepSeek、Qwen、GLM、Hunyuan 等主流模型，克隆即用……复现生产级高性能」，badge 标 `License Apache 2.0` / `Models 23+`。

「最新动态」时间线（原文摘录），可用于判断维护节奏与 day-0 惯例：
- **[2026/09] DeepSeek-V4.1-Flash 模型在昇腾 950PR/DT 系列上已支持 FP8-FP4 推理部署，支持多模理解、Engram offload等特性**
- [2026/08] **DeepSeek-V4-Flash 模型在昇腾 910B/A3 上已支持单卡推理部署**：attention 与热专家基于NPU计算，其余 MoE 专家 offload 到 Kunpeng CPU 并以原生 MXFP4 计算，**decode 约 19–22.5 tok/s**
- [2026/06] DeepSeek-V4 模型在昇腾 950PR/DT 系列上已支持 HiF8 推理部署
- [2026/04] DeepSeek-V4 模型在昇腾 Atlas A3 系列和 950PR/DT 系列上已 0day 支持推理部署

> 💡 **额外线索（与我们「单卡」目标高度相关）**：`models/deepseek_v4_flash_tilelang_and_inductor_af/` 是 **V4 的单卡方案**（910B/A3，MoE 专家 offload 到鲲鹏 CPU 跑原生 MXFP4，decode 19–22.5 tok/s）。虽然是 V4 + A2/A3，但**「专家 offload 到 host CPU 计算」的工程范式**可能对我们单卡跑 V4.1 有直接借鉴价值。本次未展开调研，建议后续单独看。

---

## L. `convert_model.py` 精读 + 量化路径分发（应协调者追加要求）

### L1. 转成什么格式？scale 布局怎么处理？

**Hybrid FP8-MXFP4（HF 原始） → Hybrid MXFP8-MXFP4（昇腾）**，唯一的实质改动就是 **scale 布局**：
`(N/32, K/32)` 二维 block scale → `(N, K/32)` 一维 group scale（沿 dim 0 `repeat_interleave(32)`）。
权重本体（`float8_e4m3fn` 的 bit）**一个字节都不动**。

### L2. ⚠️ 是的 —— **华为在离线阶段就把 32×32 展开成了 1×32。「离线/加载期展开」有官方背书。**

三处独立互证（全部原文）：
1. README「转换权重」：「原始权重中使用了 32x32-block 的量化，**通过 broadcast scale 可以将其转化为 1x32-group 的标准 MXFP8 量化**」
2. `convert_model.py:39-52` 的 `expand_scale()`：`scale.view(torch.uint8).repeat_interleave(32, dim=0).view(scale.dtype)`，docstring 写死 `(N//32, K//32) -> (N, K//32)`
3. **运行时加载器里的官方注释**（`modeling_deepseek.py:2206`）：
   ```python
   if ".ffn.shared_experts." in name or ".attn." in name:
       name = name.replace(".scale", ".weight_scale")
       if name.endswith(".weight_scale"):
           # Scales are repeated offline; preserve their E8M0 bits when loading.
           loaded_weight = loaded_weight.view(torch.uint8)
   ```
   「**Scales are repeated offline**」——官方明确表述 scale 是离线重复的。

> 这与你们 NPU 单测的 `EZ0027: group_sizes must be [1, 1, 32]` / `QuantMatmulChecker::CheckGroupSize()` 被 `DAV_3510` 门控**完全一致**。
> 文档侧的对应证据是 `modeling_deepseek.py:892-895` 传的 `group_sizes=(0, 0, 32)`——只有最后一维分组。
> **两路独立验证（NPU 单测 + CANN 官方源码）结论相同：A5 只吃 1×32，官方解法是离线展开，不写融合算子。你们「加载期一行修复」的方案与华为官方做法本质相同，只是把 offline 挪到 load 时，完全站得住。**

### L3. MoE expert 的 MXFP4 怎么转？—— **完全不动，零 repack**

`convert_model.py` 的展开条件要求 `weight.shape[0] == w.shape[0] // 32`。
routed experts 是 `[2304, 2560]` 配 scale `[2304, 160]`：`2304 != 2304/32 = 72` → **不命中，走 `new_state_dict[weight_name] = weight` 原样拷贝**。
docstring 明写（`convert_model.py:112-114`）：
> 「**Routed experts and engram embed (already 1x32) and all bf16/fp32 tensors are copied byte-identically.**」

**结论：MXFP4 专家本来就是 1D block-32，不需要 repack，转换脚本对它们是 no-op。**

### L4. Engram 表怎么处理？—— 转换脚本**不碰**，但加载期会**反量化成 BF16**（体积翻倍）

- **转换阶段**：engram embed 已是 1×32，`copied byte-identically`（同上 docstring）。
- **加载阶段**：`modeling_deepseek.py:2212-2235` 把 `.engram.embed.` 的 `weight` + `scale` 配对缓存，凑齐后调 `engram_fp8_weight_loader`；该 loader（`engram.py:325-355`）**在 host 上直接反量化成 BF16**：
  ```python
  block_size = cols // scale_cols          # engram scale 是 [rows, cols/32] → 1D block-32
  dequant_weight = (quant_weight[shard_start:shard_end].to(torch.float32)
                    .unflatten(-1, (-1, block_size)) * local_scale.unsqueeze(-1)
                   ).flatten(-2).to(torch.bfloat16)
  ```
  且**只反量化本 rank 的分片**（`shard_start = engram_tp_rank * input_size_per_partition`），避免单卡装全表。
- 然后 `offload_weights()` → `ElasticBuffer.engram_write(self.weight)`，而 `ElasticBuffer` 的 `engram_write` **只接受 bfloat16/float16/float32**（`ElasticBuffer.md` 参数说明），FP8 不支持。
- 特殊布局：无。就是 `(num_entries, hidden)` 的二维连续 tensor，`hidden=256`。索引是**全局线性索引**，`rank_id = ⌊global_idx / num_entries⌋`。

> ⚠️ **体积**：FP8 存 197 GB，BF16 存 **≈394 GB**。开源路径是后者。8 卡切分后 ≈49 GB/rank host DDR。**单卡跑 V4.1 时这条路走不通，必须自己实现「FP8 存表 + gather 时反量化」——而那正是技报描述了但没开源的部分。**

### L5. 离线还是运行时？体积变多少？

**纯离线产物。** `main()` 读 `fp8_path` 下所有 `*.safetensors`，逐个 `load_file` → 改 scale → `save_file` 到 `output_path`，并重写 `model.safetensors.index.json`（`metadata["total_size"] += grand_added`）和 `config.json`。收尾打印：
```python
print(f"Done. scales expanded: {grand_trans} (+{grand_added / 2**30:.3f}GiB)")
```
**体积增量 = 命中的每个 scale 张量元素数 ×31/32 字节**（E8M0 = 1 byte/元素），只涉及 attention / shared_experts / indexer 这些 2D-block 权重，相对权重本体是 **1/32 量级**，绝对值应在数 GB 内（脚本会精确打印，**未获取到实测数值**——我们没有权重）。
另有 `--is_mx` 模式：只改 `config.json` 的 `quantization_config`，不动权重（`convert_config.py:main`）。

### L6. 🔴 CANN 侧的量化路径分发 —— **数据驱动，不是 model_type 白名单**

对照你们发现的上游 vLLM 死代码 bug（`deepseek_v4_1/quant_config.py` 从未注册 → V4.1 静默掉进 `AscendFp8Config` → 专家 MXFP4 变 MXFP8 → 显存翻倍）：

**CANN 官方的做法里根本不存在「model_type → quant scheme」这层代码映射**，因此也就不存在「白名单漏了一个 model_type」这类失效模式。它是**把量化方案写进权重的 `config.json`，运行期照单执行**：

`convert_model.py:convert_quantization_config()` 在转换时**直接重写** `config.json`：
```python
old_quant = config.pop("quantization_config", None)          # 丢掉 HF 原始的 fp8 config
quant_ignore_layers = generate_ignore_item(num_layers, compress_ratios,
                                           kv_source_layers, index_source_layers)
quantization_config = generate_quant_config(cache_scheme, quant_ignore_layers,
                                            w4a8=True, is_mx=True)
quantization_config["weight_block_size"] = [1, BLOCK]        # [1, 32]
config["quantization_config"] = quantization_config
```
生成的结构（`convert_config.py:generate_quant_config` / `generate_quant_group`）：
```python
{"quant_method": "compressed-tensors", "format": "float-quantized",
 "quantization_status": "compressed",
 "config_groups": {
   "group_0": {"targets": ["Linear"],  # w8a8 MXFP8：weights group_size=32, input group_size=32
               "weights": {"num_bits": 8, "strategy": "group", "group_size": 32,
                           "block_structure": None, "type": "float"},
               "input_activations": {"num_bits": 8, "strategy": "group", "group_size": 32,
                                     "dynamic": True, "type": "float"}},
   "group_1": {"targets": ["MoEGMM"], # w4a8 MXFP4：注释写死 "only support mxfp4 for w4a8 with float type"
               "weights": {"num_bits": 4, "strategy": "group", "group_size": 32, ...}}},
 "ignore": [...],                       # compressor.wkv / wgate / indexer.weights_proj / head 不量化
 "weight_block_size": [1, 32],
 "kv_cache_scheme":  {"num_bits": 8, "type": "float"},
 "comp_cache_scheme":{"num_bits": 4, "type": "float"},
 "li_cache_scheme":  {"num_bits": 4, "type": "float"}}
```
关键点：
- **`group_0` targets `Linear`（→ MXFP8 w8a8）与 `group_1` targets `MoEGMM`（→ MXFP4 w4a8）是按模块类型分派的**，与 model_type 无关。专家走 MXFP4 是由 `targets: ["MoEGMM"]` + `num_bits: 4` 保证的，**不可能因为 model_type 判断失误而退化成 MXFP8**。
- 运行期消费点：`config.quant_config.mm_quant_mode`（→ `"w8a8mxfloat8"`）与 `.gmm_quant_mode`（→ `"w4a8mxfloat4"`），见 `modeling_deepseek.py:172-173, 246, 296-301`。
- 并且有**平台兜底纠偏**（`attention_data.py:154-158`）：
  ```python
  def update_gmm_quant_mode(self):
      if self.platform_version == PlatformVersion.ASCEND_950 and \
          "w4" in self.config.quant_config.gmm_quant_mode and "mx" not in ...gmm_quant_mode:
          self.config.quant_config.gmm_quant_mode = ...replace("float", "mxfloat")
  ```
  **在 950 上，只要是 w4 就强制升级成 mx 变体** —— 一道防止「专家掉进非 MX 路径」的显式保险。

> **给我们的直接建议**：
> 1. 修 vLLM 那个死代码 bug 时，**别只补白名单**——照 CANN 的做法，把分派锚定在 `quantization_config.config_groups[*].targets`（`Linear` vs `MoEGMM`/`FusedMoE`）+ `num_bits`，而不是 model_type 字符串。这样再来 V4.2 也不会漏。
> 2. 抄一条 CANN 的**平台兜底**：在 A5 上检测到 `w4 && !mx` 就强制转 `mxfloat`，把「专家显存翻倍」变成不可能发生，而不是靠 review 发现。
> 3. `ignore` 列表值得照抄（`convert_config.py:generate_ignore_item`）：`layers.{i}.attn.compressor.wkv/wgate`、`layers.{i}.attn.indexer.weights_proj`、`layers.{i}.attn.indexer.compressor.wkv/wgate`、`head` **不参与量化**。漏掉会直接掉精度。
> 4. 另注意 CANN 自己也有一处**硬覆盖**（`attention_data.py:151-152`）把 KV/LI cache 量化强行关掉：
>    ```python
>    def update_kv_quant_settings(self):
>        self.config.quant_config.set_quant_mode("kv_cache_quant_mode", "unquant")
>        self.config.quant_config.set_quant_mode("li_cache_quant_mode", "unquant")
>    ```
>    即 config.json 里写着 KV FP8 / comp FP4 / LI FP4，**运行期被无条件改回 unquant（BF16）**。这正是 §E3/§C2 观察到的「技报 vs 代码」落差的机制来源。

---

## M. 算子仓深挖结果（`raw/ops_survey/`）—— 三个关键算子找到了

### M1. 🔴 `MhcPreSinkhorn`：**官方文档确认 950 只认 4096/7168，5120 被拒**

`raw/ops_survey/mhc_mhc_pre_sinkhorn_README.md` 约束说明（**原文**）：
> - **n 目前支持 4。**
> - **numIters 目前仅支持 20。**
> - Atlas A2 / A3 系列：
>   - 输入x仅支持4维BSND格式，shape为(bs, seq_len, n, c)。
>   - **输入x的最后一维 c 需满足 128 对齐，且取值范围为 [1, 100000]。**
> - **Ascend 950PR/Ascend 950DT**：
>   - 输入x支持4维BSND格式和3维TND格式……
>   - **输入x的最后一维 c 仅支持 4096、7168。**
>   - 输入phi的数据范围限定在 ±1/√(nc)，eps 限定在 1e-6。

产品支持：950PR/DT √、A3 √、A2 √。

**这是我们「`hc_pre` 硬拒 d=5120（只认 4096/7168）」判断的官方文档级证实。**
两条额外信息：
1. `n=4` 与 `numIters=20` 恰好等于 V4.1 的 `hc_mult=4` / `hc_sinkhorn_iters=20` —— 这个算子就是为这类模型做的，唯独 hidden 维不匹配。
2. **同一算子在 A2/A3 上只要求 128 对齐（5120 合法），偏偏在 950 上被收窄到 {4096, 7168}。** 这是 A5 代际专有的收窄，不是通用限制。
3. **这正是官方 recipe 把 `hc_pre_ascendc` 整段注释掉、退回 `hc_pre_native` 的原因**（V4.1 hidden=5120）。华为撞了同一堵墙，选择了纯 PyTorch 兜底。

> **修正估计**：这一项**不是我们做错了什么，是算子当前不支持**。短期只能照官方做法走 native；要 kernel 就得等 CANN 放开 c=5120 或自己改。**建议：0.5 人天（走 native 兜底）+ 向华为提 issue 要 5120 支持。**

### M2. 🟢 `MixedQuantSparseFlashMla` **确实存在且已开源**（修正 §H 的「未获取到」）

`raw/ops_survey/attention_mixed_quant_sparse_flash_mla_README.md`（在 `attention/` 正式目录下，不在 experimental）：
- 产品：**Ascend 950PR/DT √**，A3/A2 全 ×（**A5 专属**）
- 功能：「支持 SWA（Sliding Window Attention）、CSA（Compressed Sparse Attention）、HCA（Heavily Compressed Attention）三类 Attention 计算场景。与 `SparseFlashMla` 的区别在于，**本算子支持 KV 的 per-token-group 量化输入**。」
- **`cmp_ratio` 支持 1 到 128** ✅（原文：「`cmp_ratio` 表示 `cmp_kv` 相对于压缩前KV长度的压缩倍率……**支持1到128**」）→ **cmp_ratio=2 原生支持**
- `quant_mode` 两种布局（原文）：
  > 「量化模式1：K、V 依次由 **rope（64，bfloat16）、nope（448，FLOAT8_E4M3FN）、scale（7，bfloat16）、pad（18B）** 拼接而成；
  > 量化模式2：K、V 依次由 **nope（448，FLOAT8_E4M3FN）、rope（64，bfloat16）、scale（7，FLOAT8_E8M0）、pad（1B）** 拼接而成。**量化模式2仅支持 layout_kv 为 PA_BBND**。」
- 其它：`kv_n 仅支持1`、`q_d 仅支持512`、`rope_head_dim 仅支持64`、`ori_mask_mode ∈ {0,3,4}`、`cmp_mask_mode ∈ {0,3}`、PA block_size 1~1024、`metadata` 必传 shape [1024]。

⚠️ **一个值得实测的线索（未证实，勿当结论）**：
`MixedQuantSparseFlashMla` 的约束说明里**只写了 `kv_n 仅支持1、q_d 仅支持512`，没有出现 `q_n 仅支持 64/128` 这一条**——而 `kv_quant_sparse_attn_sharedkv` 和 `quant_sparse_flash_mla` 两个 README **都明确写了**「参数q中的 N1 当前支持 64/128」。
> 这**可能**意味着新算子放开了 head 数约束（→ TP>1 有戏），也**可能**只是文档没写全。**官方 recipe 依然硬 raise `attn_tp_size > 1`（§D2），所以在实测之前不能改变「TP=1」的结论。** 建议交给正在跑 NPU 单测的同事顺手验一下 `q_n=32`。

### M3. `QuantLightningIndexerV2`：topk 放宽到 8192，但**没有两级召回**

`raw/ops_survey/attention_quant_lightning_indexer_v2_README.md`：
- 产品：950PR/DT √、**A3 √、A2 √**（比 v1 多支持 A2）
- 「`topk` 支持 **[1, 8192]**」（另一处布局下 `topk` 支持 [1, 2048]），默认值 2048
- `cmp_ratio` 为可选属性、默认 1，无 4/128 限制
- 有 `cmp_residual_k`（「需满足 0 <= cmp_residual_k[i] < cmp_ratio」）、`block_table`（PagedAttention）

**但计算公式仍是单级 Top-k**（`out = Top-k{...}`），**没有「候选块粗筛 → 块内精筛」的两级结构**。
→ **§F2 结论不变：两级候选筛选没有对应融合算子，只有 `indexer.py` 里 26 行的 PyTorch 参考实现。**

---

## N. 🔴 CANN 版本矩阵：**我们的 9.1.0 缺一半关键算子**（实测 HTTP 探测）

`ops-transformer` README 明确警告（原文）：
> 「本项目源码会跟随CANN软件版本发布……请选择配套的CANN版本与Gitcode标签源码，**使用 master 分支可能存在版本不匹配的风险**。」

该仓有对应 CANN 版本的分支：`9.2.0`、`9.1.0`、`9.0.0`、`8.5.0` 等。
**逐个探测同一路径在 9.1.0 / 9.2.0 分支下是否存在**（HTTP 200 = 有，404 = 无）：

| 算子路径 | **9.1.0（我们）** | **9.2.0（官方要求）** |
|---|:---:|:---:|
| `attention/mixed_quant_sparse_flash_mla/` ← **V4.1 主稀疏注意力** | ❌ 404 | ✅ 200 |
| `attention/quant_sparse_flash_mla/` | ❌ 404 | ✅ 200 |
| `attention/quant_lightning_indexer_v2/` | ❌ 404 | ✅ 200 |
| `mc2/common/docs/torchapi_ElasticBuffer.md` ← **Engram host offload** | ❌ 404 | ✅ 200 |
| `experimental/attention/quant_compressor/` | ❌ 404 | ✅ 200 |
| `attention/sparse_flash_mla/` | ✅ 200 | ✅ 200 |
| `attention/quant_lightning_indexer/`（v1） | ✅ 200 | ✅ 200 |
| `mhc/mhc_pre_sinkhorn/` | ✅ 200 | ✅ 200 |
| `mhc/mhc_post/` | ✅ 200 | ✅ 200 |
| `experimental/attention/kv_quant_sparse_attn_sharedkv/` | ✅ 200 | ✅ 200 |

### N1. mHC 算子在 9.1.0 上**根本不支持 950**

同一路径 `mhc/mhc_pre_sinkhorn/README.md` 在两个分支上是两个不同的算子：

**9.1.0 分支**（`raw/mhc_pre_sinkhorn_910.md`，算子名 **`MhcPre`**）：
> 产品支持情况：**Ascend 950PR/Ascend 950DT：x（不支持）**；Atlas A3 √；Atlas A2 √
> 约束说明：
> - n目前支持4。
> - **输入x的最后一维需要满足128对齐。**

**9.2.0 / master 分支**（算子名 **`MhcPreSinkhorn`**）：
> 产品支持情况：**Ascend 950PR/Ascend 950DT：√**
> 约束说明：n 支持 4；numIters 仅支持 20；
> - A2/A3：**最后一维 c 需满足 128 对齐，取值范围 [1, 100000]**（→ 5120 合法）
> - **950PR/DT：最后一维 c 仅支持 4096、7168**（→ **5120 非法**）

> **结论：d=5120 在 A5 上无论 9.1.0 还是 9.2.0 都用不了 mHC 融合算子。**
> - 9.1.0：算子对 950 整体不支持；
> - 9.2.0：950 支持了，但把 c 收窄到 {4096, 7168}，恰好排除 V4.1 的 5120。
> - A2/A3 上 5120 反而合法 —— 说明**不是算法限制，是 950 kernel 的 tiling 只做了两档**。这是一个**明确、可提 issue 的诉求**。

### N2. 升级到 CANN 9.2.0 能买到什么

| 能力 | 9.1.0 | 9.2.0 |
|---|---|---|
| `cmp_ratio=2` 的稀疏 MLA | ❌ 无算子 | ✅ `MixedQuantSparseFlashMla`（1~128） |
| per-token-group 量化 KV（E8M0 scale 打包进 KV） | ❌ | ✅ `quant_mode=2` |
| Engram host offload（RDMA/URMA） | ❌ 无 `ElasticBuffer` | ✅ |
| Indexer topk 上限 | 2048（v1） | 8192（v2） |
| 量化 Compressor | ❌ | ✅ `quant_compressor` |
| mHC 融合算子（A5, d=5120） | ❌（950 不支持） | ❌（c 只认 4096/7168） |
| TP>1 | ❌（N1∈{64,128}） | ❓ `MixedQuantSparseFlashMla` 文档未声明 q_n 约束，**待实测** |

> **这条是本次调研对排期影响最大的发现**：如果不升 CANN 9.2.0，**Engram offload、cmp_ratio=2 稀疏注意力、量化 KV 三块的算子全都不存在**，只能全走纯 PyTorch —— 那基本等同于把官方 day-0 recipe 的「关掉融合算子」状态原样复制一遍。
> **建议把「升级到 CANN 9.2.0（weekly）」列为 V4.1 适配的前置阻塞项，而不是可选优化。**

---

## O. 行动建议（优先级排序）

0. **🔴 先决条件：把「升级到 CANN 9.2.0 weekly + HDK 25.5.1」当成阻塞项排期**（§N）。9.1.0 上 `MixedQuantSparseFlashMla` / `ElasticBuffer` / `quant_compressor` / `quant_lightning_indexer_v2` 全部不存在（实测 404）。
1. **立即照抄 `expand_scale()`**（4 行）到我们的 weight loader → 关闭「MXFP8 2D 32×32」这一项。**MXFP8 32×32 争议就此定论：A5 只支持 1×32，官方做法是离线 broadcast，不需要写融合算子。**
2. **停止寻找 TP>1 的路径**：官方 yaml 全 TP=1 + 官方代码 `attn_tp_size > 1` 直接 raise + 算子 README 明写 `N1 仅支持 64/128`。三重确认。
3. **把 `engram.py` 整体拿过来**做算法基准（哈希、素数桶、压缩词表、门控），offload 走我们已验证的 UVA (`pinned_mem_register:True`)，不追 `ElasticBuffer`（绑 HCCL + 8 卡）。
4. **评估从 `kv_quant_sparse_attn_sharedkv` 切到 `quant_sparse_flash_mla`**（`cmp_ratio` 1~128），能直接吃下 cmp_ratio=2；但要先确认其 HIFLOAT8 per-tensor 量化模式对 V4.1 的精度影响。
5. **核对我们的 CANN / HDK / torch_npu 版本**与官方要求（9.2.0 weekly / 25.5.1 / dev20260903）的差距，决定是否升级。
6. **重新核算单卡 Engram 内存预算**：开源路径 BF16 = 394 GB，技报的 FP8 = 197 GB 但**未开源**。单卡必须自己做 FP8 存表 + gather 反量化。

7. **mHC 的 d=5120**：短期照官方做法走 `hc_pre_native`（纯 PyTorch），同时**向华为提 issue**：`MhcPreSinkhorn` 在 950 上把 c 收窄到 {4096, 7168}，而 A2/A3 只要 128 对齐 —— 请求放开 5120。这是有明确文档依据的合理诉求。
8. **交给正在跑 NPU 单测的同事顺手验一条**：`MixedQuantSparseFlashMla`（需 CANN 9.2.0）在 `q_n=32` 下是否可用 —— 它的 README **没有**像另外两个算子那样声明「N1 仅支持 64/128」。若真放开，TP=2 就有了可能性。**在实测出结果前，仍按 TP=1 规划。**
