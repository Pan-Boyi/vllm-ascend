# DeepSeek-V4.1-Flash — CPU golden ↔ Ascend NPU 逐层对齐协议 v1.0

本文档给昇腾实现方直接使用。照做即可产出可比对的 dump，并用同一个脚本判定通过与否。

CPU golden 的定位：**数值参考**，不是硬件模拟器。它跑的是模型自带的官方参考实现
（`/workspace/models/DeepSeek-V4.1-Flash/inference/`，逐字复制），只把 tilelang 写的 6 个
GPU kernel 换成纯 PyTorch CPU 版。量化语义（哪个值落到哪个 fp8/fp4 码、block scale 怎么算、
clamp 和 amax 下限在哪）与原 kernel **逐条对齐**；归约（GEMM 累加、softmax）改为 fp32 全精度，
**比硬件更准**。

---

## 0. 先读这一节：本模型对齐的头号难点是 MoE 路由，不是算子精度

实测（本仓库 `dumps/smoke3` vs `dumps/smoke3_fp64`）：**只把 GEMM 累加器从 fp32 换成 fp64**
——一个让结果严格更准、单个 GEMM 上相对误差只有 ~1e-7 的改动——就造成：

| 位置 | 效果 |
|---|---|
| layer 0 输出 | rel_l2 = 1.0e-3（正常） |
| layer 1 专家选择 | **2.47% 的 token 选中了不同专家** |
| layer 1 输出 | rel_l2 = 2.9e-2 |
| layer 2 专家选择 | **16.67% 的 token 选中了不同专家** |
| layer 2 输出 | rel_l2 = 1.0e-1 |
| 最终 logits | rel_l2 = 5.6e-2 |

原因：`Gate` 用 `(scores + bias).topk(k)` 选专家，这是**连续值上的离散判决**。当第 k 名和
第 k+1 名分数接近时，任意微小扰动都会翻转选择，而两个不同专家的输出差异是 O(1) 的。

### 而且它会滚雪球

同一次 fp32↔fp64 实验，按 **token** 统计（27 个 token，`moe_routing_margin` 实测）：

| 层 | 路由被翻转的 token | 翻转 token 的 margin | 未翻转 token 的 margin |
|---|---|---|---|
| layer 0 | 0 / 27 (0.0%) | — | min 4.7e-4, 中位 4.3e-3 |
| layer 1 | 2 / 27 (7.4%) | max 1.5e-2, 中位 1.2e-3 | min 9.4e-5, 中位 3.7e-3 |
| layer 2 | 15 / 27 (**55.6%**) | max 6.0e-3, 中位 2.1e-3 | min 1.4e-3, 中位 6.6e-3 |

两点必须注意：

1. **layer 0 完全一致，layer 1 就有 7%，layer 2 已经过半。** 路由一旦在某层分叉，
   下游层的 gate 输入本身就带了 3e-2 量级的误差，翻转不再是"平局"而是误差累积的结果。
2. **到 layer 2，翻转 token 的 margin（≤6.0e-3）和未翻转 token 的 margin（≥1.4e-3）已经完全重叠。**
   也就是说 **margin 阈值只在最前面一两层能区分"平局"和"真 bug"，再往后就失效了。**

**结论（协议的核心）：**

1. **专家路由不一致时，隐藏态的 rel_l2 数字没有意义。** 不要用它判定算子对错。
2. 判定顺序永远是：**先比路由 → 路由一致再比数值**。
3. 只有**第一个**出现路由分歧的层是有诊断价值的。在那一层看 `moe_routing_margin`：
   margin 极小 = 数值近似平局，属于可接受差异；margin 不小却选错 = **真 bug**。
4. **因此第 7 节的"强制路由"调试开关不是可选项，是逐层比对能否进行下去的前提。**
   没有它，layer 2 之后的所有数值比对都是在测噪声。

---

## 1. 环境

```bash
unset LD_PRELOAD      # 必须。jemalloc 在 ~/.bashrc 里，不 unset 会 segfault
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
```

CPU golden 侧：`torch 2.10.0+cpu`、`safetensors>=0.8`、`transformers 5.14`、`numpy`、`sympy`。
不需要 NPU，**不要设 `ASCEND_RT_VISIBLE_DEVICES`，不要 import torch_npu**。

模型目录 `/workspace/models/DeepSeek-V4.1-Flash` 只读。

---

## 2. 固定输入

**用固定文本，不要用随机 token。** 随机 token 会落到语义上不可能的组合，把模型推到激活分布的
尾部，放大 MoE 近似平局的比例，让对齐结果比真实推理更糟、也更不可复现。

标准输入（`run_golden.py` 里的 `GOLDEN_PROMPT`，逐字使用）：

```
The capital of France is Paris. Ascend NPU kernels must match this reference bit for bit in structure and within tolerance in value.
```

用模型自带 tokenizer 编码，得到 **27 个 token**：

```
[671, 6102, 294, 8760, 344, 11111, 16, 48548, 523, 39675, 55, 87375, 2231,
 6614, 566, 6915, 4669, 362, 4669, 295, 4456, 305, 2549, 24030, 295, 1990, 16]
```

- `batch = 1`，`start_pos = 0`（纯 prefill，无 decode）
- 27 > `candidate_block_size`(8)、> `compress_ratio`(2)，所以 layer 2 的压缩器会产出 13 个
  latent、indexer 真正被触发；同时 27 < `window_size`(128)，滑窗路径也走到。
- **NPU 侧必须直接喂这 27 个 token id**，不要重新分词（避免 tokenizer 差异污染比对）。

如果要额外验证长序列/decode 路径，另开一组，不要替换这一组。

---

## 3. Dump 点

每层一个文件 `layer_{NN}.safetensors`（NN 两位，00..39），另加 `prologue` / `epilogue`。
张量名必须**完全一致**，否则比对脚本会报 missing。

### prologue.safetensors
| 名字 | 形状 | dtype | 含义 |
|---|---|---|---|
| `input_ids` | [1, 27] | int32 | 输入 token |
| `embed.out` | [1, 27, 5120] | bf16 | 词嵌入输出（**扩成 hc 4 份之前**） |
| `engram_hash_ids` | [1, 27, 2, 24] | int64 | n-gram 哈希 id，[B,L,engram层数,n_hash_cols] |

### layer_NN.safetensors（每层）
| 名字 | 形状 | dtype | 含义 |
|---|---|---|---|
| `layer.in` | [1, 27, 4, 5120] | bf16 | 进入本层的 hyper-connection 残差流 |
| `pre_mix.in` | [1, 27, 4] | fp32 | 上一层 FFN 交下来的 pre 系数 |
| `engram.out` | [1, 27, 4, 5120] | bf16 | **仅 layer 1 / 14**：engram 写入后的残差流 |
| `hc_mixes.0.0/.1/.2` | [1,27,4] / [1,27,4] / [1,27,4,4] | fp32 | attention 段的 pre / post / comb（Sinkhorn 后） |
| `hc_pre.0` | [1, 27, 5120] | bf16 | hc 4 份塌缩成 1 份，attention 的输入 |
| `attn_norm.out` | [1, 27, 5120] | bf16 | attention 前 RMSNorm |
| `attn.in` | [1, 27, 5120] | bf16 | 同上（冗余，用于确认 hook 挂对位置） |
| `compressor.out` | [1, 13, 512] | bf16 | **仅 kv_source 层 2/8/14/20**：RoPE 前的压缩 latent |
| `indexer_topk.out` | [1, 27, K] | int32 | **仅 index_source 层**：选中的压缩位置，**-1 表示空槽** |
| `attn.out` | [1, 27, 5120] | bf16 | attention 输出（已做逆 RoPE + wo 投影） |
| `hc_post.0` | [1, 27, 4, 5120] | bf16 | attention 结果展回 4 份并混入残差 |
| `hc_mixes.1.0/.1/.2` | 同上 | fp32 | FFN 段的 pre / post / comb |
| `hc_pre.1` | [1, 27, 5120] | bf16 | FFN 的输入 |
| `ffn_norm.out` | [1, 27, 5120] | bf16 | FFN 前 RMSNorm |
| `moe_routing_margin` | [27] | fp32 | **诊断用**：第 k 名与第 k+1 名选择分数之差 |
| `moe_gate.out.0` | [27, 6] | fp32 | 路由权重（已归一化、已乘 route_scale） |
| `moe_gate.out.1` | [27, 6] | int64 | **选中的专家编号——最关键的一项** |
| `moe.in` | [1, 27, 5120] | bf16 | MoE 输入 |
| `moe.out` | [1, 27, 5120] | bf16 | MoE 输出（含 shared expert） |
| `hc_post.1` | [1, 27, 4, 5120] | bf16 | FFN 结果展回 4 份 |
| `layer.out` | [1, 27, 4, 5120] | bf16 | 本层输出残差流 |
| `pre_mix.out` | [1, 27, 4] | fp32 | 交给下一层的 pre 系数 |

### epilogue.safetensors
| 名字 | 形状 | dtype |
|---|---|---|
| `hc_pre.2` | [1, 27, 5120] | bf16 |
| `final_norm.out` | [1, 27, 5120] | bf16 |
| `logits` | [1, 27, 129280] | fp32 |

单次全量 dump 约 **300 MB**（40 层 × ~7 MB + logits 14 MB）。

---

## 4. Dump 格式

- **safetensors**，每个 tag 一个文件，键名如上。
- fp8 / fp4 / e8m0 一律**先转成 fp32 再存**（safetensors 不支持这些 dtype）。
  fp8→fp32 是精确加宽，不损失信息。
- 另存 `manifest.json`，至少包含 `input_ids`、`n_layers_run`、实现版本、commit。
- CPU golden 侧由 `golden.Recorder` 自动产出，NPU 侧照抄键名即可。

---

## 5. 比对流程

```bash
python compare_dumps.py  <golden_dir>  <npu_dir>  [--tolerance default|strict|loose]  [--json out.json]
```

判定按下面的顺序，**不要跳步**：

### 第 1 步 — 前置一致性
`prologue/input_ids` 必须逐位相同；`prologue/embed.out` rel_l2 应 < 1e-6（同一份 bf16 权重的查表）。
不一致说明输入或权重加载就错了，后面不用看。

`engram_hash_ids` 必须**逐位完全相同**。它是纯整数哈希（素数取模 + XOR），没有任何浮点，
不一致就是 hash 实现或 compressed token map 错了。

### 第 2 步 — 逐层路由
对每层比 `moe_gate.out.1`：

- **完全一致** → 进第 3 步。
- **不一致** → 取出不一致的 token，查它们的 `moe_routing_margin`：
  - margin < 1e-3 → 近似平局，记录后放行（但要统计比例，见第 6 节阈值）。
  - margin ≥ 1e-3 → **真 bug**，停在这一层查 gate：`linear(x.float(), weight.float())`
    是否真的用 fp32、`sqrtsoftplus`（`F.softplus(x).sqrt()`）是否实现一致、
    `bias`（`e_score_correction_bias`）是否只参与选择不参与加权。

### 第 3 步 — 逐层数值
按层号从小到大看 `layer.out` 的 rel_l2 / cos。**只有第一个越界的层是有诊断价值的**，
后面的层都在继承上游误差。定位到第一个失败层后，在该层内按下面顺序二分：

```
layer.in → hc_mixes.0.* → hc_pre.0 → attn_norm.out
        → compressor.out → indexer_topk.out → attn.out
        → hc_post.0 → hc_mixes.1.* → hc_pre.1 → ffn_norm.out
        → moe_gate.* → moe.out → hc_post.1 → layer.out
```

第一个越界的中间量就指向具体算子。

### 第 4 步 — 输出
`logits`：比 rel_l2、cos，以及 **top-1 / top-5 token 是否一致**。
top-1 一致是最低要求。

---

## 6. 阈值与理由

阈值不是拍脑袋来的：下限取自 golden 自身的数值敏感度实测（fp32 vs fp64 累加），
NPU 的容差必须**明显宽于**这个不可约噪声，否则测的是浮点抖动不是实现正确性。

| 检查项 | 阈值 | 理由 |
|---|---|---|
| `input_ids`、`engram_hash_ids` | **逐位相同** | 纯整数运算，没有容差可言 |
| `indexer_topk.out` | **逐位相同**（允许 <1% 差异） | 实测 fp32↔fp64 下 100% 稳定；top-512 选择裕度大 |
| `moe_gate.out.1` 路由（**第一个分歧层**） | 不一致 token ≤ **8%**，且这些 token 的 margin 中位数 < 5e-3 | 实测 fp64 扰动下第一个分歧层（layer 1）为 7.4%、翻转 margin 中位 1.2e-3 |
| `moe_gate.out.1` 路由（后续层） | **不设阈值，不作判据** | 实测 layer 2 起翻转/未翻转 margin 完全重叠，已无区分力；请改用强制路由模式（第 7 节） |
| 单层 `layer.out` | rel_l2 ≤ **2e-2**，cos ≥ **0.999** | 激活是 bf16（尾数 8 位，~4e-3），叠加 fp8/fp4 GEMM 累加差异 |
| `attn.out` / `moe.out` | rel_l2 ≤ **3e-2** | 子层输出未经残差平滑，波动比 `layer.out` 大 |
| `hc_mixes.*`（fp32） | rel_l2 ≤ **5e-3**，cos ≥ 0.9999 | 全程 fp32，且 Sinkhorn 20 轮是收缩映射，应该很稳 |
| `logits` | rel_l2 ≤ **3e-2**，**top-1 必须一致** | 与 `layer.out` 同量级；top-1 是业务判据 |

`--tolerance strict`（rel_l2 5e-3 / cos 0.9999）用于**路由已强制对齐后**的纯算子比对。
`--tolerance loose`（5e-2 / 0.995）只用于首次打通、允许路由有分歧时的粗筛。

**通过的定义**：第 1、2 步全过，第 3 步所有层在 `default` 容差内，第 4 步 top-1 一致。

---

## 7. 必做项："强制路由"模式

第 0 节的数据说明：不消掉路由这个离散判决，逐层数值比对在 layer 2 之后就没有信噪比了。
解法是把它从等式里去掉：

> 在 NPU 实现里加一个调试开关，允许**从外部注入专家编号**（直接读 golden dump 的
> `moe_gate.out.1`），跳过自己的 top-k；同时也注入 `moe_gate.out.0` 的路由权重。

这样比对就变成纯算子比对，可以直接上 `--tolerance strict`，一旦超标必定是真 bug。
分两阶段收敛：

- **阶段 A（强制路由）**：注入 golden 的专家编号+权重，逐层比数值。定位所有算子精度问题。
- **阶段 B（自由路由）**：放开 top-k，只比 `moe_gate.out.0/1` 和最终 logits 的 top-1，
  单独验证 gate 本身。

同理，`indexer_topk.out` 也建议支持注入——虽然实测它比 MoE 路由稳得多（fp64 扰动下
100% 一致），但它同样是 top-k 离散判决，长序列下裕度会变小。

这个开关是本协议里**投入产出比最高的一项，请优先做**。

---

## 8. 已知不可比 / 超出范围

- **vision tower（ViT + Aligner）**：本 golden 未跑，纯文本路径。
- **DSpark / MTP 草稿头**（`mtp.*`，3 层）：未跑。参考实现里 `forward_spec` 没有被
  推理循环调用，属于另一个工作项。
- **decode（`start_pos > 0`）**：本 golden 目前只验证了 prefill。滑窗环形缓冲、
  compressor 的跨步状态（`kv_state`/`score_state`）、indexer 的单 query 分支都**没有**
  被覆盖，是明确的缺口（见 REPORT.md 人天估计）。
- **`-1` 索引语义**：`sparse_attn` 里 `topk_idxs == -1` 表示空槽，其 KV 行读作 0、
  score 读作 -inf；整行全 -1 时输出**必须是全 0 而不是 NaN**（running max 用有限的
  -1e30 而非 -inf 兜底）。NPU 实现如果这里返回 NaN，会在滑窗刚开始填充时炸掉。
- **fp8 溢出行为**：torch CPU 的 `to(float8_e4m3fn)` 溢出返回 **NaN 而非饱和**，所以
  必须先 clamp 到 ±448 再转。原 tilelang kernel 同样依赖调用方先 clamp。

---

## 9. 复现 golden

```bash
cd <本目录>
unset LD_PRELOAD
python test_kernel_cpu.py                                   # 6 个 kernel 的单测
python run_golden.py --layers 40 --threads 16 --dump dumps/full40
python compare_dumps.py dumps/full40 <npu_dump_dir>
```

实测：40 层单次 prefill（27 token）**216 秒 / 峰值 RSS 55 GB / 读取 52 GB 权重**。
详见 REPORT.md。
