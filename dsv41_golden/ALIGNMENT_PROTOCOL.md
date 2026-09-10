# DeepSeek-V4.1-Flash — CPU golden ↔ Ascend NPU 逐层对齐协议 v1.0

本文档给昇腾实现方直接使用。照做即可产出可比对的 dump，并用同一个脚本判定通过与否。

CPU golden 的定位：**数值参考**，不是硬件模拟器。它跑的是模型自带的官方参考实现
（`/workspace/models/DeepSeek-V4.1-Flash/inference/`，逐字复制），只把 tilelang 写的 6 个
GPU kernel 换成纯 PyTorch CPU 版。量化语义（哪个值落到哪个 fp8/fp4 码、block scale 怎么算、
clamp 和 amax 下限在哪）与原 kernel **逐条对齐**；归约（GEMM 累加、softmax）改为 fp32 全精度，
**比硬件更准**。

---

## 0. 先读这一节：本模型对齐有**两个**独立难点

历史注记：本节 v1.0 只讲了 MoE 路由，且引用的数字（layer 1 翻转 7.4%、layer 2 翻转 55.6%）
**被交叉验证推翻**。v1.1 修正了数字，并补上了第二个、在路由被钉住之后仍然存在的难点。

### 0.1 难点一：MoE 路由是离散判决（数字已修正）

`Gate` 用 `(scores + bias).topk(k)` 选专家 —— 连续值上的离散判决。当第 k 名与第 k+1 名
分数接近时，任意微小扰动都会翻转选择，而两个不同专家的输出差异是 O(1) 的。

实测（`golden_dumps/smoke3` fp32 累加 vs `golden_dumps/smoke3_fp64` fp64 累加，
一个让结果**严格更准**、单 GEMM 相对误差只有 ~1e-7 的改动）：

| 层 | 按元素逐位比 topk（**旧的、错的算法**） | 其中纯顺序调换 | **真实集合改变（正确）** |
|---|---|---|---|
| layer 0 | 0/27 = 0% | 0 | **0/27 = 0%** |
| layer 1 | 2/27 = 7.4% | **2** | **0/27 = 0%** |
| layer 2 | 15/27 = 55.6% | **9** | **6/27 = 22.2%** |

**为什么旧数字是错的**：`topk` 返回的顺序对结果**没有任何影响**。MoE 对选中的专家求和，
而累加顺序是按**专家编号**循环的（`MoE.forward` 里 `for i in range(n_routed_experts)`），
与 `topk` 的返回顺序无关。所以「专家集合相同、只是顺序调换」必须计为**一致**。
旧算法把 9 个纯顺序调换算成了翻转。

**⇒ 比较专家选择必须按集合比**（`sort(-1)` 之后再比）。`compare_dumps.py` 已按此实现
（`SET_VALUED` + `routing_agreement()`），并把「纯顺序调换」单独计数而不计为失败。

**结论方向不变**：22.2% 的 token 在一个严格更准的扰动下路由到不同专家，照样让逐层
数值比对失去意义。所以第 7 节的强制路由注入仍然是**必做项**。

### 0.2 难点二：一个 bf16 ULP 经因果 attention 扩散（v1.1 新增）

这是**路由被完全钉住之后依然存在**的误差传播路径，而且在深层它是主导项。

实测（fp64 累加 + 注入 fp32 路由方案，`golden_dumps/smoke3_fp64_forced`，
注入已验证逐位精确：`moe_gate.forced.1` 与 golden 完全相同、`forced.0` 差 0）：

| 张量（基线 = `smoke3`，fp32 自由路由） | fp64 **自由**路由 | fp64 **强制**路由 |
|---|---|---|
| layer 0 `layer.out` | 4.28e-4 | 3.97e-4 |
| layer 1 `layer.out` | 1.14e-2 | 7.98e-3 |
| layer 2 `layer.out` | 3.75e-2 | **2.27e-2** |
| `logits` | 5.60e-2 | 3.01e-2 |

**强制路由只把误差减半，没有消掉误差增长。** 注意 layer 2 那个 2.27e-2 ——
它已经**超过了本协议第 6 节自己定的 `layer.out ≤ 2e-2`**，而此时并不存在任何 bug。

机制（逐 token 定位，layer 1、强制路由）：

```
hc_pre.0  (attention 输入) : 27 个 token 里只有 2 个有任何差异
                             token 3 : 5120 个元素里 1358 个差 恰好 1~2 个 bf16 ULP
                             token 10: 1 个元素差 1 ULP（该元素值本身是 0）
attn.out  (attention 输出) : 24/27 个 token 被污染
                             token 0/1/2 保持逐位相同 —— 它们因果上在 token 3 之前
```

链条：

1. fp64 累加让 layer 0 的输出动了 ~1e-7（相对）。
2. **残差流是 bf16（8 位尾数，ULP ≈ 4e-3 相对）**。25/27 个 token 四舍五入后**完全相同** ——
   但 token 3 恰好落在进位边界上，1358 个元素各翻 1~2 个 ULP。
3. **因果 attention 把 token 3 的误差扩散给所有 j ≥ 3 的 token** → 24/27 被污染。
4. 每层重复一次，被污染 token 的比例饱和到 100%，逐 token 幅度再逐层累加。

**⇒ 整张量的 `rel_l2` 增长，主要来自「被污染 token 的比例变大」，不是来自「算子把噪声放大」。**

这解释了一个否则很诡异的现象：layer 1 路由 0 翻转、注入权重逐位相同，`moe.out` 却有
2.33e-2 —— 因为它的输入里早就有一个 token 是脏的。

**⇒ 所以 `rel_l2` 不是好的单一判据。** 必须同时看**逐 token** 统计：
`compare_dumps.py` 现在报 `tokens_bit_exact` 和 `max_token_rel_l2`。
「27 个 token 里 26 个逐位相同、1 个差 1 个 ULP」和「27 个 token 全都差 4e-3」
的 `rel_l2` 可以一样，但前者是舍入、后者是 bug。

**⇒ 并且逐层比对需要第二个开关：注入 `layer.in`（第 7 节）。**

### 0.3 判定顺序（协议的核心）

1. **专家路由不一致时，隐藏态的 rel_l2 没有意义。** 不要用它判定算子对错。
2. 顺序永远是：**先验注入是否生效 → 再比路由 → 路由一致再比数值**。
3. 自由累积模式下，只有**第一个**出现分歧的层有诊断价值。
   **注入 `layer.in` 之后这条限制解除**，每层可独立定位。
4. **强制路由（第 7 节）不是可选项**，没有它 layer 2 之后的数值比对是在测噪声。
5. **强制路由不够**，还要强制 `layer.in`，否则阈值必须逐层放宽（第 6 节）。

（对比：indexer 的 top-512 在本组实验下 100% 稳定，**但见第 5.1 节的适用边界** ——
27 token 时实际 K=13 且「13 选 13」，几乎没有选择余地，这个结论不能外推到长序列。）

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

### 第 1.5 步 — 注入是否真的生效（只在 `force_route != off` 时）
候选侧的 `moe_gate.forced.1` 必须与 golden 的 `moe_gate.out.1` **逐位相同**（按集合，见下）。
代价是每层 27×6 个 int32 的比较，挡掉的是「环境变量拼错 / 方案文件没找到 / 注入被静默跳过」。

**这一步不能省。** 没有它，一个没生效的注入表现为「strict 容差下逐层全面超标」——
和真算子 bug 完全同一个症状，而你会去查算子。

`compare_dumps.py` 自动做这一步（读候选 `manifest.json` 的 `force_route`）。

### 第 2 步 — 逐层路由（**必须按集合比**）
对每层比 `moe_gate.out.1`，**先 `sort(-1)` 再比**：

> `topk` 的返回顺序对结果没有任何影响（MoE 对选中专家求和，累加顺序按专家编号循环）。
> 按元素逐位比会把纯顺序调换算成翻转 —— 第 0.1 节那两个被推翻的数字就是这么来的。

- **集合完全一致** → 进第 3 步。（顺序不同不算问题，`compare_dumps.py` 单独计数。）
- **不一致，且 `force_route` 含 `moe`** → 这是**预期的**，仅供参考，不作判据；
  真正的判据是第 1.5 步。
- **不一致，且自由路由** → 取出不一致的 token，查它们的 `moe_routing_margin`：
  - margin 很小 → 近似平局，记录后放行（比例阈值见第 6 节）。
  - margin 不小却选错 → **真 bug**，停在这一层查 gate：`linear(x.float(), weight.float())`
    是否真的用 fp32、`sqrtsoftplus`（`F.softplus(x).sqrt()`）是否实现一致、
    `bias`（`e_score_correction_bias`）是否只参与选择不参与加权（`ref/model.py:822-823`）。
  - ⚠️ **margin 判据只在最前面一两层有效**：实测 layer 2 起翻转与未翻转 token 的 margin
    分布已经重叠。再往后请改用强制路由模式。

### 第 3 步 — 逐层数值
按层号从小到大看 `layer.out` 的 rel_l2 / cos，**同时看逐 token 的
`tokens_bit_exact` 和 `max_token_rel_l2`**。

> 为什么要看逐 token：第 0.2 节的机制说明整张量 rel_l2 的增长主要来自「被污染 token 的
> 比例变大」。「26/27 个 token 逐位相同、1 个差 1 个 bf16 ULP」和「27 个 token 全都差 4e-3」
> 可以有相同的 rel_l2，但前者是舍入、后者是 bug。

**在自由累积模式下，只有第一个越界的层有诊断价值**，后面的层都在继承上游误差。
**注入 `layer.in` 之后（第 7.2 节）这条限制解除**，每层独立可判。

定位到第一个失败层后，在该层内按下面顺序二分：

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

### 5.1 `indexer_topk.out` 的实际形状与「100% 稳定」的适用边界

```python
topk = min(self.index_topk, end_pos // ratio)          # ref/model.py:577
idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
return torch.where(idxs < compress_lens, idxs + offset, -1).int()
```

`index_topk = 512`，但 **K 是逐层、随序列长度变的**。27 token 标准输入实测
（取自 `golden_dumps/full40/manifest.json`）：

| 层 | `compressor.out` | `indexer_topk.out` |
|---|---|---|
| 2 / 8 / 14（ratio 2，kv_source） | [1, 13, 512] | [1, 27, **13**] |
| 20（ratio 1，kv_source） | [1, 27, 512] | [1, 27, **27**] |
| 24 / 28 / 32 / 36（ratio 1，非 kv_source） | —— | [1, 27, **27**] |

`compress_ratios` 是**逐层**列表（长度 43 = 40 层 + 3 个 MTP），layer 0/1 是 0。

⚠️ **「fp64 扰动下 100% 一致」是在 K=13 且「13 个里选 13 个」的退化条件下测出来的** ——
几乎没有选择余地，所以稳定是必然的，不是裕度大。**这个结论不能外推到长序列。**

两个可以依赖的性质：

- **`indexer_topk.out` 完全确定**，尽管底层 `topk` 在 -inf 平局上不确定：
  被选中的不可达位置无论是哪几个，都统一映射成 `-1`。所以可以要求**逐位相同**。
- **`-1` 补在尾部，且逐行连续**（先按位置升序 `sort`，不可达下标必然 ≥ `compress_lens`，
  排序后落尾部）。**每行尾部长度不同**：prefill 时 `compress_lens` 是 per-row 的 `[seqlen,1]`。
  实测 `full40/layer_02` 每行 `-1` 个数
  `[13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0]`
  → **第 0 行整行是 `-1`**，即第 8 节那条「整行全 -1 必须输出全 0 而非 NaN」
  **在标准输入上就被触发**，不需要构造 case。

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

## 7. 必做项：两个注入开关

完整接口规范见 **`ROUTING_INJECTION.md`**（含签名、寻址方式、fail-safe 要求、环境变量）。
本节只讲为什么要两个、以及各自解决什么。

### 7.1 强制路由（消掉离散判决）

> 在实现里加一个调试开关，允许**从外部注入专家编号和路由权重**（读 golden 的
> `moe_gate.out.1` / `.0`），跳过自己的 top-k。

- **weights 必须和 indices 一起注入。** `norm_topk_prob` 按选中集合重新归一化
  （`ref/model.py:824-826`），只注入 indices 等于换了归一化分母，数值依然不可比。
- **按绝对序列位置寻址，不能按张量行号。** 详见 `ROUTING_INJECTION.md` 第 0 节 ——
  这是唯一一个「在 smoke case 上必然通过、只在真实调度下暴露、且暴露形式伪装成待测对象」
  的坑。
- **注入点在 dispatch 与 global→local expert 映射之前。**
- 必须 dump `moe_gate.forced.0/.1`（实际喂给 expert 的值），供第 1.5 步验证注入生效。

**效果实测**：把 layer 2 的 `layer.out` 从 3.75e-2 降到 2.27e-2。**有效，但不够。**

### 7.2 强制层输入（消掉跨层累积）—— v1.1 新增

第 0.2 节的数据说明：路由钉住之后，主导误差路径变成「一个 bf16 ULP 经因果 attention
扩散」，误差仍然逐层累积，layer 2 就已经越过 2e-2。

> 在每个 block 入口，用 golden dump 的 `layer.in` [1,27,4,5120] 和
> `pre_mix.in` [1,27,4] **覆盖**残差流，然后再跑这一层。

这样每层都拿到 golden 的输入，误差不再跨层累积，**每层的偏差回到「单层噪声」量级**，
`strict` 容差这时才真正可用，而且**任意一层都能独立定位**，不必再遵守
「只有第一个越界的层有诊断价值」。

CPU golden 侧已实现：`run_golden.py --force-layer-in <golden_dump_dir>`
（`GoldenModel.forward(layer_in_dir=...)`）。

**⚠️ 这个方法的边界（必须说明，否则会被过度信任）**：它**不隔离跨层状态** ——
KV cache、layer 2/8/14/20 的 compressor latent、layer 20 的 candidate blocks、
indexer 的选择，这些仍然由本次运行自己前面的层产生。所以是**部分隔离**，不是完全隔离。
但它切断了主导的那条累积路径。

### 7.3 indexer 注入（低优先，接口先留）

`indexer_topk.out` 同样是 top-k 离散判决。本组实验下它 100% 一致，**但见第 5.1 节的
适用边界**：27 token 时实际 K=13 且「13 选 13」，几乎没有选择余地。长序列下 K 逼近 512、
候选远多于 512 时它会变回真正的离散判决。**接口先留好，实现排在 7.1/7.2 之后；
长序列那组上线时必须重测再决定。**

### 7.4 收敛顺序

| 阶段 | 开关 | 容差 | 判据 |
|---|---|---|---|
| **A1** | `moe` + `layer_in` | `strict` | 每层独立比数值。超标即真 bug，直接指向算子 |
| **A2** | `moe` | 见第 6 节逐层放宽表 | 验证跨层累积是否符合预期（不是找 bug） |
| **B** | `off` | `default` | `moe_gate.out.0/1`（按集合）+ logits top-1。单独验 gate |

阶段 A 照常 dump 自己算的 `moe_gate.out.*`，所以一趟同时产出阶段 B 需要的 gate 数据。

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

---

## 10. 单算子对照（给 kernel 线）

第 1~9 节讲整模型逐层对照。kernel 线要的是**单个算子**的 rel_err，方法不同：不需要跑模型，
也不需要 dump。

### 10.1 参考实现在哪

`kernel_cpu.py` 的六个函数就是参考，与原 tilelang kernel **逐条对齐量化语义**
（哪个值落到哪个 fp8/fp4 码、block scale 怎么算、clamp 和 amax 下限在哪），
只把归约换成 fp32/fp64 全精度：

| 函数 | 签名要点 |
|---|---|
| `act_quant(x, block_size=128, scale_fmt=None, scale_dtype=fp32, inplace=False)` | → `(y_e4m3, s)`，s 形状 `[..., N//block_size]` |
| `fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=e8m0)` | → `(y_packed_e2m1_x2, s)` |
| `fp8_gemm(a, a_s, b, b_s, scale_dtype=fp32)` | 累加 dtype 由 `GOLDEN_DOT_DTYPE` 控制 |
| `fp4_gemm(a, a_s, b, b_s, scale_dtype=fp32)` | 同上 |
| `hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps)` | → `(pre, post, comb)`；最后一轴布局 `[0:hc]`=pre、`[hc:2hc]`=post、`[2hc:]`=comb（row-major hc×hc） |
| `sparse_attn(q, kv, attn_sink, topk_idxs, ...)` | `topk_idxs == -1` 是空槽 |

`test_kernel_cpu.py`（~90 条断言，9 个测试，全过）就是**调用范例**，包括每个 kernel 的
边界情形。要新增对照，照它的结构加即可：

```bash
unset LD_PRELOAD          # 必须，jemalloc 在 ~/.bashrc 里，不 unset 会 segfault
python test_kernel_cpu.py
```

### 10.2 用两个参考，不要只用一个

单算子必须同时对两个基准比，两个数字回答不同问题：

| 基准 | 怎么造 | 回答什么 | 量级 |
|---|---|---|---|
| **A. CPU golden kernel** | 同一份量化输入喂 `kernel_cpu.py` | **实现是否正确**（量化语义、scale、布局） | 应当很紧：差异只来自累加顺序 |
| **B. 未量化 fp64 参考** | 用原始 bf16/fp32 输入直接做 `@`，`GOLDEN_DOT_DTYPE=torch.float64` | **量化本身损失多少** | 大得多，且是预期的 |

**只比 B 会把量化误差当成 bug；只比 A 会漏掉「两边用同样错误的量化语义」。**
`test_kernel_cpu.py` 里 `test_fp8_gemm` / `test_fp4_gemm` 就是这么写的：
`< 5e-3` 对 naive 循环（A），`< 0.1` 对未量化 bf16（B）。

### 10.3 喂什么输入

**不要只用随机张量。** 随机数落在激活分布的尾部，会放大 amax 下限、clamp 边界这些分支的
触发概率，也可能反过来根本触发不到。建议三组一起：

1. **真实激活**：从 `golden_dumps/full40` 取对应张量当输入（形状和分布都是真的）。
2. **随机张量**：固定种子，覆盖一般情形。
3. **边界构造**：全零组、单元素超大、正好落在 fp4 码中点、amax 触及下限。
   `test_kernel_cpu.py` 已覆盖一部分。

### 10.4 已知的语义陷阱（会静默给出错结果）

- **fp8 溢出**：torch CPU 的 `to(float8_e4m3fn)` 溢出返回 **NaN 而非饱和**，
  所以必须**先 clamp 到 ±448 再转**。原 tilelang kernel 同样依赖调用方先 clamp。
- **amax 下限，三个不同的值**，抄错会在小激活上分叉：
  - `act_quant`（fp8）：`amax.clamp_min(1e-4)`
  - `fp4_act_quant` + `e8m0` scale（indexer）：`amax` 下限 `6 * 2**-126`，
    scale = `2 ** ceil(log2(amax/6))`
  - `fp4_act_quant` + `e4m3` scale（压缩 KV）：`amax` 下限 `6 * 2**-9`，
    scale = `amax/6` **再舍入到 e4m3** —— 为了让全零组也拿到非零 scale，是训练侧的约定
- **`scale_fmt` 非 None** 时 scale 向上取到 2 的幂（MXFP 风格），与非 None 时不同。
- **`sparse_attn` 的 `-1`**：该 KV 行读作 0、score 读作 -inf；**整行全 -1 时输出必须是
  全 0 而不是 NaN** —— running max 要用有限的 `-1e30` 兜底。
  ⚠️ 这条**在标准输入上就会被触发**（见第 5.1 节：layer 2 第 0 行整行是 -1），
  不是理论边界。
- **RMSNorm 的 eps**：config 是 `1e-20`，golden 全程用它（`ref/model.py:933-934` 等处都
  显式传 `args.norm_eps`；`RMSNorm.__init__` 的默认值 1e-6 从不生效），计算在 fp32 下做。
  实测全部 40 层 × 27 token 的 `mean(x²)` 最小值 = **3.95e-4**（layer 2 attention 输入），
  所以融合算子若把 eps 钳到 1e-6，最坏相对偏差 = `1e-6/(2·3.95e-4)` = **1.3e-3**，
  与 bf16 ULP 同量级，**不可接受**。
  **判据：eps 下限必须 ≤ 8e-10**（由 `eps/(2·3.95e-4) < 1e-6` 反解）。
  适用边界：只测了 dim=5120 的两个 norm；`q_norm` / `kv_norm` / indexer `k_norm` 维度小得多，
  未测。

### 10.5 GEMM 累加精度

`kernel_cpu.GOLDEN_DOT_DTYPE` 默认 `torch.float32`，设成 `torch.float64` 得到更紧的参考
（约慢 8 倍、内存约 2 倍）。`run_golden.py --fp64` 就是这么切的。

⚠️ **fp32 与 fp64 两种参考产出的整模型结果不可混用** —— 第 0 节的实验就是这两者之差，
它翻转了 22% 的路由。单算子对照不受影响（差异 ~1e-7），但一旦进入整模型逐层比对就要统一。
