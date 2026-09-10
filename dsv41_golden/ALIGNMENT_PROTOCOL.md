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
不需要 NPU。**不要设 `ASCEND_RT_VISIBLE_DEVICES`。**

> ⚠️ **v1.0 这里写的两句都不成立，实测**：
> - 「不要 import torch_npu」—— **做不到**，`import torch` 自己就会把它拉进 `sys.modules`。
> - 「不要设 `ASCEND_RT_VISIBLE_DEVICES`」—— **它本来就被容器注入成 `0,1,2,3,4,5,6,7`**，
>   不是我们设或不设的事。
>
> 所以这两条都不是该看的性质。该看的是「**有没有占卡**」，而这一条是可验证的：
> `dsv41_golden/` 下没有任何代码调用 NPU API、在 npu 上分配、或把张量搬过去；
> 而 vendor 代码里会那么做的地方（`ref/generate.py`、`ref/model.py` 的
> `Transformer.__init__` 里 `torch.set_default_device("cuda")`）**从不被调用** ——
> 执行器直接构造 `M.Block`，不走 `M.Transformer`。
> dump 的 `manifest.json` 如实记录了这三项（`torch_npu_loaded_by_torch_import` /
> `ascend_rt_visible_devices` / `uses_npu_device`）。

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
| `hc_mixes.0.0/.1/.2` | [1,27,4] / [1,27,4] / [1,27,4,4] | fp32 | attention 段的 pre / post / comb（Sinkhorn 后）。**⚠️ 见下面的 🔴，别和 `pre_mix.in` 搞混** |
| `hc_pre.0` | [1, 27, 5120] | bf16 | hc 4 份塌缩成 1 份，attention 的输入 |
| `attn_norm.out` | [1, 27, 5120] | bf16 | attention 前 RMSNorm |
| `attn.in` | [1, 27, 5120] | bf16 | 同上（冗余，用于确认 hook 挂对位置） |
| `compressor.out` | [1, 13, 512] | bf16 | **仅 kv_source 层 2/8/14/20**：RoPE 前的压缩 latent |
| `indexer_topk.out` | [1, 27, K] | int32 | **仅 index_source 层**：见下面的 🔴 警告，值**不是**压缩位置编号 |
| `indexer_kv_offset` | [1] | int32 | **仅 index_source 层**：`indexer_topk.out` 的基准偏移，用来归一化 |

> 🔴 **`indexer_topk.out` 的值是「拼接后 `[window ; compressed]` KV 的下标」，不是压缩位置编号，
> 而且偏移量是路径相关的 —— 不归一化就逐位比，会在实现完全正确的情况下报 100% 不一致。**
>
> ```python
> # ref/model.py:775-778
> kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
> compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))  # ← offset
> kv = torch.cat([kv, compress_kv], dim=1)
> # ref/model.py:580
> return torch.where(idxs < compress_lens, idxs + offset, -1).int()
> ```
>
> `offset = kv.size(1)` 是滑窗 KV 张量的长度，而它在两条路径上不同（`ref/model.py:709-719`）：
>
> | 路径 | 滑窗张量 | offset |
> |---|---|---|
> | prefill（`seqlen ≤ window_size`） | `window_kv = kv`，就是本段本身 | **`seqlen`（本组输入 = 27）** |
> | decode | `window_kv = window_kv_cache`，整个环 | **`window_size` = 128** |
>
> 实测：同一个压缩位置，golden 的 prefill 记成 `27..39`，decode 记成 `128..140` ——
> **差 101，但两边都对**。任何滑窗 KV 布局与 golden 不同的实现（paged KV、prefill 也走满
> 128 槽的环）都会整体偏移一个常数。
>
> **⇒ 比对前必须先归一化**：`compressed_pos = where(raw < 0, raw, raw - offset)`
> （`-1` 保持 `-1`）。参考实现 `run_decode_check.py:_unoffset()`。
> **⇒ 实现方也必须 dump 自己的 `indexer_kv_offset`**（concat 时用的那个 offset），
> 否则无法归一化。
> **⇒ `indexer_kv_offset` 本身不作判据** —— 它*应当*在两条路径间不同。
| `attn.o_raw` | [1, 27, 64, 512] | bf16 | **sparse attention 的原始输出，逆 RoPE 之前** |
| `attn.o_unroped` | [1, 27, 64, 512] | bf16 | **逆 RoPE 之后、`wo_a` 之前** |
| `attn.out` | [1, 27, 5120] | bf16 | attention 输出（已做逆 RoPE + wo 投影） |

> 🔴 **为什么 `attn.out` 一个键不够**（`dsv41-model` 提的，理由成立）。
> 从 sparse attention 的输出到 `attn.out` 之间有三步：
> ```
> o  [T,64,512]                       ← attn.o_raw
>   → 逆 GPT-J RoPE（末 64 维，就地修改）  → attn.o_unroped
>   → view [T,8,4096]
>   → einsum("bsgd,grd->bsgr", ·, wo_a)   ← 块对角，8 组各自独立
>   → flatten → wo_b                    → attn.out
> ```
> `attn.out` 对不上时，**四个候选无法区分**：sparse attention 本身 / 逆 RoPE 的旋转约定 /
> 块对角分组是否错位 / `wo_b`。**而其中两个是静默失败**：
> - **逆 RoPE 的约定写错**（GPT-J 交错 vs 半旋转）→ 输出仍是「合理的数」，不报错。
> - **把 `wo_a` 当稠密矩阵乘**（而它是块对角，`ref/model.py:786-788`）→ 每组会看到全部
>   32768 维输入 → 结果同样是「合理的数」。
>
> 有这两个键，`_o_proj` 就能二分到具体一步。
> 成本：每层 2 × 1.7 MB，40 层约 +133 MB（dump 从 271 MB 到约 404 MB）。
>
> 实现上：`apply_rotary_emb(o[..., -rd:], freqs_cis, True)`（`ref/model.py:781`）
> **就地修改**，所以 `o_raw` 必须在它之前 clone；`o_unroped` 是同一块存储在之后读到的值
> （`.view` 只改名不拷贝，einsum 不写回）。

> 🔴 **`attn_sink` 参与 attention，golden 已实现**（v1.0 漏写）。
> `ref/model.py:639` 声明 `attn_sink [n_local_heads] fp32`，`:780` / `:1067` 传入
> `sparse_attn`，`kernel_cpu.py:445-473` 消费。它是 **per-head 的学习 logit，
> 只进 softmax 分母，不贡献 value**：
> ```python
> mx    = scores.amax(-1).clamp_min(-1e30)
> p     = torch.exp(scores - mx)
> denom = p.sum(-1) + torch.exp(sink.view(1,1,h) - mx)   # ← sink 只在这里
> o     = einsum(p, kv_g) / denom
> ```
> checkpoint 里 `layers.#.attn.attn_sink [64]` fp32，40 层 + 3 个 MTP 层全有。
> 实现漏掉它 → 所有行的分母系统性偏小，`attn.out` 每层都分叉。
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

> 🔴 **`hc_mixes.0.0` 不是 `pre_mix.in`，但它们 shape 完全相同（都是 `[1,27,4]` fp32）。**
>
> - `pre_mix.in` = **上一层 FFN 段交下来的** pre 系数。
> - `hc_mixes.0.0` = **本层 attention 段自己算出来的** pre 系数。
>
> shape 一样，所以 dump 时搞混零成本，而症状要到很多层之后才显现。实测两者相差 **O(1)**
> （layer 0~3 的 `max|diff|` 分别是 1.00 / 0.76 / 0.80 / 0.55），不是接近的两个量。
>
> **两条免费自检**（实测在全部 40 层上**逐位相等**，可以直接 `assert`）：
>
> ```
> pre_mix.out[L]  ==  hc_mixes.1.0[L]        逐位相同 —— 交给下一层的就是 FFN 段产出的 pre
> pre_mix.in[L]   ==  pre_mix.out[L-1]       逐位相同 —— 层间链路
> ```
>
> 也就是 `pre_mix.out` 在 dump 里是**冗余**的（等于 `hc_mixes.1.0`），它存在的意义正是让
> 实现方零成本验证「我有没有把 pre 系数接对」。`pre_mix.in[0]` 是
> `make_identity_pre_mix`（`ref/model.py:1159`）产出的单位 pre-mix。

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

### 5.2 🔴 `indexer_topk.out` 需要自己的判据，不能用张量容差

**参考实现和昇腾侧的 q/k 位宽不同**（`dsv41-kernel` 实测）：

| | 官方参考实现 | 昇腾侧实际 |
|---|---|---|
| 压缩 KV | fp4 e2m1，block **16**，scale e4m3 | **MXFP8**：fp8 e4m3 + e8m0，group **64**，rope 段 bf16 |
| Indexer q/k | fp4 e2m1，block **32**，scale e8m0 | **mode 1**：fp8 e4m3 + fp32 per-token-head scale |

**参考 4 bit，昇腾 8 bit —— 量化步长差 4 倍。**

而 index score 是**在量化过的 q/k 上算出来、然后取 top-k**（`ref/model.py:557` → `:578`）——
**离散判决**。步长差异可能**翻转选择**，而翻转不是数值误差，**放宽容差收敛不了**：
你会看到「逐层张量都在容差内，但 `indexer_topk.out` 的索引集合不同」，
而下游一切都建在那个集合上。

（这也给了第 5.1 节那个观察一个机制：fp32↔fp64 扰动下 `indexer_topk.out` 100% 稳定，
**是因为 q/k 已被压到低位宽，量化步长远大于 1e-7 的扰动**。当时记的是现象，现在有原因了。）

#### 判据（三段，`compare_dumps.py:indexer_agreement()`）

```
1. 归一化   两侧各自减掉自己的 indexer_kv_offset（见第 3 节警告）
2. 换了多少  逐 query 算集合对称差 |A △ B|，报换掉的槽数 / 总槽数
3. 有害吗    对每个被换掉的位置，用 indexer_score 算
             「被顶掉的最低分」vs「顶上来的最高分」的间隙
               间隙极小 → 无害（和 MoE 的近似平局同型）
               间隙不小 → 真错，去查量化粒度 / scale 类型
```

**第 3 步是这套判据的全部价值，而它需要一个新的 dump 点。**

| 键 | 形状 | dtype | 含义 |
|---|---|---|---|
| `indexer_score` | [1, 27, n_compressed] | fp32 | **top-k 之前**的原始分数。layer 2 是 `[1,27,13]` |

**没有它，比对只能报「换了 N 个」，说不出「有害吗」** —— 那正是 MoE 路由在
`moe_routing_margin` 出现之前的困境。`compare_dumps.py` 在缺它时会 graceful degrade
并**把 `have_scores: false` 显式写进输出**，而不是静默给一个数字。

> 类比是这套判据成立的理由：`moe_routing_margin` 的价值就是把「平局」和「真 bug」分开。
> 实测例：40 层 decode 检查里 pos 23 / layer 22 有一个 token 路由不同，
> 它的 margin 是 **9.3e-5，全层 27 个里最小、比中位数 2.7e-2 低 290 倍** → 判为近似平局。
> indexer 需要同一个机制，只是量名从 margin 换成 index_score 间隙。

⚠️ **现在还没有实测的间隙分布** —— 需要昇腾侧的 `indexer_score` dump。
所以「4 倍步长会不会真的翻转、翻转有没有害」目前是**先验，不是证据**。
判据先建好，是为了第一次比对就能直接给结论，而不是先花一轮争论怎么判。

---

## 6. 阈值与理由

阈值全部来自实测，不是拍脑袋。**下限取自 golden 自身的数值敏感度**（同一份代码，只把
`kernel_cpu` 里两个 GEMM 的累加 dtype 从 fp32 换成 fp64），NPU 的容差必须**明显宽于**
这个不可约噪声，否则测的是浮点抖动而不是实现正确性。

### 6.0 两条先决认识

**(a) bf16 的相对 ULP 是 `2**-8` = 3.9e-3，这是一条硬地板，与实测无关。**

它是**逐元素的相对**量。所以一个**纯系统性**的差异 —— 比如产生它的 GEMM 换了求和顺序，
把几乎每个元素都推动一个 ULP —— **本身就落在 ~3.9e-3，而实现完全正确**。
⇒ **任何 bf16 张量的阈值必须 ≥ 几个 ULP。**

本协议采用 **`BF16_FLOOR = 3 ULP = 1.17e-2`**，对所有 bf16 张量取
`max(该张量阈值, BF16_FLOOR)`（`compare_dumps.py:tol_for()`）。3 倍的理由：
1 ULP 是「全体元素被系统性推动一格」，再留出输入误差传播和第 10.6 节那条
`sparse_attn` 的 P·V bf16 差异（1.8e-3），约 1.5 倍余量。

> ⚠️ **这一条我自己犯过两次，第二次是在批评了第一次之后。**
>
> 第一次：v1.0 的全局 `strict = 5e-3` 正好压在地板上，拿它比一个**已知无 bug** 的参考跑，
> 40 层里 26 个张量「失败」。
>
> 第二次：v1.1 改成 per-tensor 阈值时，我把 `layer.in` / `hc_pre.*` / `attn_norm.out` /
> `ffn_norm.out` / `moe.in` 这六个 **bf16** 张量设成 `5e-3 = 1.28 ULP` —— 因为我拿的是
> 它们**实测的 fp32↔fp64 地板 1.3e-3**，而忽略了下面还有一层表示地板。
> `dsv41-kernel` 对 `hc_pre` 在 d=5120 的**第一个真实 NPU 实测**是 **6.85e-3 = 1.75 ULP**
> —— 正确，但超了我那个限。
>
> **教训：实测地板是下界之一，表示地板是另一个下界，阈值必须取两者的上包络。**
> fp32 量不受这条约束（5e-3 对 fp32 是 ~42000 ULP，远不算紧）。

**(b) 一个全局阈值管不了所有张量。** fp32 的 `hc_mixes` 实测在 1e-4 量级，bf16 的
`moe.out` 天然在 5e-3 量级 —— 相差 50 倍。同一个数要么放过前者，要么冤枉后者。
`compare_dumps.py` 现在按张量名做**最长前缀匹配**取阈值（`TOLERANCES` + `tol_for()`）。

### 6.1 逐位相同（没有容差可言）

| 检查项 | 理由 |
|---|---|
| `input_ids` | 输入 |
| `engram_hash_ids` | 纯整数哈希（素数取模 + XOR），零浮点 |
| `moe_gate.forced.1`（vs golden 的 `moe_gate.out.1`，按集合） | 注入是否生效，见第 1.5 步 |
| `indexer_topk.out`（**归一化后**） | 见第 5.1 节：不可达位置统一映射成 `-1`，输出完全确定。⚠️ 必须先减掉 `indexer_kv_offset`，见第 3 节警告 |
| `compressor.out` | 实测 4 个 kv_source 层全 40 层偏差 **0.00e+00** |

### 6.2 阶段 A1 —— 强制路由 + 强制 `layer.in`（`--tolerance isolated`）

**实测参考跑**：`golden_dumps/full40`（fp32，自由）vs `golden_dumps/full40_fp64_iso`
（fp64 累加 + 注入 `plan_full40` 路由 + 注入 `full40` 的 `layer.in`）。
390 s / 峰值 55.1 GB / 40 层。路由分歧 **1/1080 行**（仅 layer 28 一个 token）。

40 层的逐层 `rel_l2` 分布。

> # 🔴 读这张表之前必须先读这一句
>
> **这是 golden 自身数值敏感度的「地板」，不是对 NPU 能达到多少的预测。**
> 产生它的实验**只扰动了 `kernel_cpu` 里两个 GEMM 的累加 dtype**，所以计算链上位于
> 第一个被扰动 GEMM **之前**的张量必然是 `0`。
> **`0†` 的意思是「这个扰动碰不到它」，不是「昇腾也应该做到 0」。**
> 拿 `0†` 当 NPU 的验收目标是误用这张表。你的容差必须明显宽于本表。

| 张量 | 中位 | **40 层最大** | 最差层 | 最小逐位相同 token |
|---|---|---|---|---|
| `layer.in` / `pre_mix.in` | 0† | **0†** | — | 27/27 |
| `hc_pre.0` / `attn_norm.out` / `attn.in` | 0† | **0†** | — | 27/27 |
| `compressor.out`（4 层） | 0† | **0†** | — | 13/13 |
| `hc_mixes.0.*` | 0† | 1.3e-6 | 14 | 26/27 |
| `engram.out`（2 层） | 8.6e-6 | 8.6e-6 | 14 | 26/27 |
| `hc_mixes.1.*` | 4.5e-5 | 8.7e-4 | 26 | 1/27 |
| `hc_pre.1` / `ffn_norm.out` / `moe.in` | 4.1e-4 | 1.3e-3 | 26 | 2/27 |
| `hc_post.0` | 3.0e-4 | 3.4e-3 | 26 | 1/27 |
| `moe_gate.out.0` | 1.2e-4 | 4.2e-3 | 1 | — |
| `layer.out` / `hc_post.1` | 1.5e-3 | **5.3e-3** | 26 | 1/27 |
| `attn.out` | 6.1e-4 | **5.4e-3** | 26 | 0/27 |
| **`moe.out`** | 4.8e-3 | **1.64e-2** | 26 | 2/27 |
| `moe_routing_margin` | 2.0e-3 | 1.25e-2 | 26 | — |
| `logits`（epilogue） | — | **1.5e-3** | — | 20/27 |

**† = 该扰动在计算链上到不了这个张量，所以必然为 0。不是 NPU 的目标值。**

**由此推出的阈值（实测最大值 × 2~3 倍余量）：**

| 张量类 | dtype | rel_l2 | = ? ULP | cos | 实测最大 |
|---|---|---|---|---|---|
| `layer.out` / `hc_post.*` / `attn.out` | bf16 | **1.5e-2** | 3.8 | 0.9999 | 5.4e-3 |
| ↳ ⚠️ `attn.out` 另有一个 **1.8e-3 的固定偏差**，见下 | | | | | |
| `moe.out` | bf16 | **4e-2** | 10.2 | 0.9995 | 1.64e-2 |
| `hc_pre.*` / `ffn_norm.out` / `moe.in` / `attn_norm.out` / `layer.in` | bf16 | **1.17e-2**（= `BF16_FLOOR`） | 3.0 | 0.9999 | 4.2e-3 |
| `hc_mixes.*` / `pre_mix.*` / `moe_gate.out.0` | **fp32** | **5e-3** | ~42000 | 0.9999 | 8.7e-4 |
| `logits` | fp32 | **1e-2** + **top-1 必须一致** | — | 0.9999 | 1.5e-3 |
| 其余 bf16 | bf16 | 1.17e-2 | 3.0 | 0.9999 | |
| 其余 fp32 | fp32 | 5e-3 | — | 0.9999 | |

**「= ? ULP」这一列是为了让阈值合理性一眼可查**：bf16 那几行低于 ~2.5 就是定错了。

**`moe.out` 为什么单独放宽到 4e-2**：它是 6 个加权专家 + shared expert 求和后
`type_as(x)` 回 bf16，抵消较多，天然坐在 ~1 个 bf16 ULP 上（中位 4.8e-3 ≈ 3.9e-3）。
而且它未经残差平滑 —— 进了残差之后的 `layer.out` 立刻回到 1.5e-3。

**校准验证**：用 `--tolerance isolated` 比上面那个**已知无 bug** 的参考跑 → **PASS**，
唯一的 informational 是 `moe_routing_margin`（诊断项，不 gate）和 layer 28 那 1 个 token。

> ### 🔴 `attn.out` 上有一个「golden 比被参考对象更准」造成的固定偏差
>
> 原 tilelang `sparse_attn` **把 softmax 概率舍到 bf16 才做 P·V**（分母仍从 fp32 求和），
> 而 CPU golden 全程 fp32 —— 详见第 10.6 节的行号引用。
>
> **实测**：把 golden 的 P 也舍到 bf16，`sparse_attn` 输出 `rel_l2 = 1.80e-3`、
> 逐 (token, head) 最大 `3.27e-3`。
>
> **⇒ 一个忠实照搬原 kernel 的实现，会和 golden 有 ~1.8e-3 的固定差异，这不是 bug。**
> 1.5e-2 的阈值有 8 倍余量，**不用改**；但**不要**因为 `attn.out` 有 1e-3 量级的差就去查算子。
> 要逐位对照时把 `kernel_cpu.SPARSE_ATTN_P_DTYPE` 设成 `torch.bfloat16`。
>
> 这条和第 6.0 节那个 `strict = 5e-3` 的错误是**同一类陷阱的两个方向**，两条都会把人
> 送去查不存在的 bug：
>
> | | 表现 |
> |---|---|
> | 阈值太严（`strict = 5e-3`）| 已知无 bug 的参考跑，40 层里 26 个张量「失败」|
> | **参考太准**（P·V 全 fp32）| **忠实的实现看起来「差了 1e-3」** |

### 6.3 `moe_routing_margin` 不作判据

它是「第 k 名与第 k+1 名选择分数之差」—— 两个相近量之差，相对噪声天然大（实测最大
1.25e-2，无 bug）。它的用途是在**自由路由**下区分「近似平局」和「真 bug」，
不是数值阈值。`compare_dumps.py` 把它列进 `NEVER_GATE`：报告但永不判失败。

### 6.4 阶段 A2 —— 只强制路由，不强制 `layer.in`

**这个模式下 `rel_l2` 不能作为逐层判据**，原因见第 0.2 节：误差跨层累积，
layer 2 就已经到 2.27e-2（无 bug）。实测 3 层：

| | fp64 自由路由 | fp64 强制路由 |
|---|---|---|
| L0 `layer.out` | 4.28e-4 | 3.97e-4 |
| L1 `layer.out` | 1.14e-2 | 7.98e-3 |
| L2 `layer.out` | 3.75e-2 | **2.27e-2** |

**⇒ A2 的用途是「验证跨层累积是否符合预期」，不是找 bug。找 bug 用 A1。**
如果一定要在 A2 下判定，只能用 `--tolerance loose`（5e-2 / 0.995）做首次打通的粗筛，
并且**只看第一个越界的层**。

### 6.5 阶段 B —— 自由路由

| 检查项 | 阈值 | 理由 |
|---|---|---|
| `moe_gate.out.1`（**第一个分歧层**，按集合比） | 见下 | |
| `moe_gate.out.1`（后续层） | **不设阈值，不作判据** | 实测 layer 2 起翻转/未翻转的 margin 分布完全重叠，已无区分力 |
| `logits` | **只看 top-1，rel_l2 不作判据** | 见下面的 🔴 |

> ### 🔴 `logits` 的 rel_l2 只在「batch 形状一致」时有意义
>
> 实测（40 层，**两次纯 prefill，路由已强制对齐，完全没有 decode 参与**）：
>
> ```
> prefill[0:20] 的 logits  vs  prefill[0:27] 的前 20 个位置
>   40 层：rel_l2 = 1.028e-01        ← 10%
>    3 层：rel_l2 = 6.2e-03
> ```
>
> **根因不是「GEMM 的 M 维不同」——那个解释是错的，已被实测推翻。**
>
> 真正的原因是**滑窗索引张量的宽度不同**。`get_window_topk_idxs`（`ref/model.py:410-426`）
> 产出的宽度是 `min(seqlen, window_size)`，而 `window_size = 128 > 27`，所以：
>
> | prefill 长度 | 索引张量宽度 | query 5 的 `-1` 空槽数 | query 19 的 |
> |---|---|---|---|
> | 20 token | **20** | 14 | 0 |
> | 27 token | **27** | 21 | 7 |
>
> 有效的注意力集合完全相同（空槽贡献恰好 0），但 **`sparse_attn` 的归约宽度不同**，
> 于是浮点求和的舍入不同。再经第 0.2 节那条「bf16 ULP 经因果 attention 扩散」放大 40 层。
>
> **决定性证据**：把 `window_size` 调到 **8**（< 20），两边的索引张量宽度都变成 8、
> 空槽数都变成 2，**对照值立刻变成 `0.000e+00` —— 两次 prefill 逐位相同**
> （3 层，其余一字不改）。而此时 GEMM 的 M 维**依然**是 20 vs 27。
>
> **⇒ GEMM 的 M 维不同不产生任何差异；滑窗归约宽度不同才产生。**
>
> **这个修正改变了实用建议**：不是「任何分批方式不同的实现都会差 10%」，而是
> **「滑窗槽位数与 golden 不一致的实现才会差」**。一个滑窗布局和 golden 相同的昇腾实现，
> `logits` 可以对得比 1e-1 好得多。反过来，如果昇腾侧 prefill 也走满 128 槽的环
> （而 golden 走 `seqlen` 宽），**那就会有一个固定的、与 bug 无关的差异**。
>
> 对比：**同一个 40 层跑，注入 `layer.in` 之后 `logits` 只差 1.5e-3**（第 6.2 节）。
> **相差 70 倍。**
>
> **⇒ 如果昇腾侧的分批方式和 golden 不同**（chunked prefill、不同 tile、不同 padding），
> **`logits` 可能差 10% 而实现完全正确。** 所以 v1.0 给的 `logits ≤ 3e-2` 是**低于地板**的，
> 已作废：**自由累积模式下 `logits` 只看 top-1。**
> 需要数值判据时用第 6.2 节的 A1（注入 `layer.in`），那里 `logits` 的阈值 1e-2 是有效的。
>
> ⚠️ `1.028e-01` 是**单点测量**（一个 cut 值、一组文本），标为「已观测」而非「已确认」。
> 但它和 3 层的 6.2e-3、以及注入后的 1.5e-3 一起构成的趋势足够清楚。

⚠️ **v1.0 那条「第一个分歧层不一致 token ≤ 8%」已作废。** 它是从被推翻的 7.4%
（见第 0.1 节，那 2 个 token 其实是纯顺序调换）推出来的。**按集合比时，本组实验下
第一个分歧层是 layer 2，比例 22.2%。**

重推：一个「严格更准」的扰动在第一个分歧层造成 22.2% 的集合改变，所以

> **第一个分歧层的集合不一致比例 ≤ 25%，且不一致 token 的 `moe_routing_margin`
> 中位数 < 5e-3** → 视为数值近似平局，放行并记录。
> 超过 → 停在这一层查 gate（`ref/model.py:809-827`）。

⚠️ 这条阈值的**置信度低于第 6.2 节那张表**：它只有 27 个 token、一组扰动的样本。
它是「不要在这里浪费时间」的软门槛，**不要当验收线用**。真正的验收线是 A1。

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
- **decode（`start_pos > 0`）**：**v1.1 起已覆盖**，见第 11 节。滑窗环形缓冲、
  compressor 跨步状态（`kv_state`/`score_state`）、indexer 单 query 分支、
  engram 哈希 cache 都已验证。
- **`-1` 索引语义**：`sparse_attn` 里 `topk_idxs == -1` 表示空槽，其 KV 行读作 0、
  score 读作 -inf；整行全 -1 时输出**必须是全 0 而不是 NaN**（running max 用有限的
  -1e30 而非 -inf 兜底）。NPU 实现如果这里返回 NaN，会在滑窗刚开始填充时炸掉。

  🔴 **这条 fail-safe 和 `attn_sink` 是同一件事，不是两个独立的补丁。**
  整行全 -1 → scores 全 `-inf` → `mx = clamp_min(-1e30) = -1e30`
  → `p = exp(-inf + 1e30) = 0` → `denom = 0 + exp(sink + 1e30) = inf`
  → `o = 0 / inf = 0`。**是 sink 项让分母非零，从而避免 0/0。**

  所以：若实现单独加了个「空行输出 0」的特判、但 sink 漏了，标准输入 layer 2 第 0 行会
  「看起来对」，而**其他所有行的分母全都偏小** —— 系统性偏差，极难查。
  反之若 `mx` 用了 `-inf` 而不是 `-1e30`，即使 sink 在，也会得到 NaN。
  ⚠️ 实测这条**在标准输入上就被触发**（第 5.1 节：layer 2 第 0 行整行是 -1）。
- **fp8 溢出行为**：torch CPU 的 `to(float8_e4m3fn)` 溢出返回 **NaN 而非饱和**，所以
  必须先 clamp 到 ±448 再转。原 tilelang kernel 同样依赖调用方先 clamp。
### 8.1 `batch > 1`：已定位，是舍入累积而不是 bug；判据是**诊断工具，不是门禁**

同一段 prompt 复制成 batch=2，逐层比 row 0 vs row 1（`compare_batch_rows.py`）：

```
layer  第一个 row0 != row1 的张量     max|diff|     rel_l2   路由是否相同
    0  hc_mixes.0.0                  3.638e-12   7.02e-13      相同     ← 起点
    1  layer.in                      2.384e-07   3.94e-09      相同
    7  layer.in                      1.221e-04   1.24e-06      相同
    9  layer.in                      4.688e-02   3.81e-03      相同     ← 路由还没翻，差异已到 1e-2
   10  layer.in                      4.492e-02   7.33e-03      不同     ← 第一次翻转（token 24, 25）
   39  layer.in                      2.828e+00   7.96e-02      不同
```

**⇒ 第一处分歧是 layer 0 的 `hc_mixes.0.0`，`3.638e-12` —— fp32 舍入量级**
（`hc_mixes` 是 fp32，值 O(1)）。**同一次 batched GEMM 里的两行拿到了不同的舍入**：
batch=2 时 MoE 输入是 `[54, 5120]` 而不是 `[27, 5120]`，BLAS 的分块/归约随之不同。

**⇒ layer 9 那行是关键证据**：**路由仍然相同、差异已经到 4.7e-2** ——
增长发生在路由翻转**之前**，所以路由翻转是**结果不是原因**。
这一行单独就排除了「路由逻辑有 bug」。

#### 🔴 「真 batch bug」有一个可判别的签名，而且已经做成会报错的检查

**不是看最终差多少，是看第一处分歧有多大：**

| 第一处分歧的 `max|diff|` | 判定 |
|---|---|
| **≲ 1e-12**（fp32 首次接触的舍入） | 舍入累积。**不是 bug** |
| **> 1e-6** | **BATCH BUG** —— 某个不该不同的张量**立刻**大幅不同：缓存槽按错的维度索引、该 gather 处做了 broadcast |

`compare_batch_rows.py` 里 `FIRST_DIVERGENCE_BUG_THRESHOLD = 1e-6`，**超了就 exit 1 并指出张量名和层号**。
（1e-6 远高于任何可信的 fp32 首次舍入（~1e-12），又远低于接线错误会产生的量。）

> 这条和 `moe_routing_margin` 同型：**用形状区分两类原因，而不是用阈值判对错。**
> margin 把「平局」和「真错」分开，这个把「舍入累积」和「索引错乱」分开。

#### batch > 1 的判据

| 检查项 | 判据 |
|---|---|
| **batch 内各行之间** | **不能逐位比。** 判「路由一致时，`layer.out` 在第 6.2 节的逐层阈值内」|
| **batch>1 vs batch=1** | **只看 top-1**。rel_l2 不作判据（地板约 **8e-2**，与第 6.5 节那个 1e-1 同量级同机制）|
| **是不是真 bug** | 看第一处分歧的量级（上表），**由 `compare_batch_rows.py` 自动判并返回退出码** |

🔴 **而且这一整节是诊断工具，不是验收门禁。** `batch > 1` 在不同实现之间
**结构性不可逐位比** —— 昇腾的 BLAS 分块方式必然与 CPU golden 不同，所以它的 batch>1
结果与 golden 的 batch>1 结果之间会有**同一量级、与 bug 无关**的差异。
**真正的 batch 验收在 GSM8K 那一级。** 这里的逐层对比用于**出问题时定位**
（哪一层、是不是路由、是不是接线），**不用于放行** ——
别拿它当门禁然后被 8e-2 的地板卡住。

#### 附带确认：注入接口在 batch>1 下不需要新约定

`GoldenModel.forward` 现在支持 batch>1 + 强制路由（**要求各行序列相同**，混合 batch 仍然拒绝，
那需要每行一份方案）。位置映射就是**每行独立的绝对序列位置**（`arange(L).repeat(N)`），
**和 `ROUTING_INJECTION.md` 第 0 节要求的完全一致** ——
一个抽象扩展到新场景时不需要新规则，说明当初选对了（按绝对位置而不是按行号）。

---

## 9. 复现 golden · 完整命令参考

```bash
cd <本目录>
unset LD_PRELOAD          # 必须。jemalloc 在 ~/.bashrc 里，不 unset 会 segfault
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
DU=/workspace/code/dsv4_single_card/vllm_project/golden_dumps
```

### 9.1 kernel 单测（秒级，不加载权重）
```bash
python test_kernel_cpu.py                      # 9 个测试 / ~90 条断言
```

### 9.2 产出基准与强制路由方案
```bash
python run_golden.py --layers 40 --threads 16 --dump $DU/full40_v2       # ~372 s，55 GB
python routing_plan.py $DU/full40_v2 --out $DU/plan_full40_v2            # 271 MB → 58 KB
```

### 9.3 每层独立的参考跑（第 6.2 节噪声地板表的来源）
```bash
python run_golden.py --layers 40 --fp64 --threads 16 \
    --routing-plan $DU/plan_full40_v2 --force-layer-in $DU/full40_v2 \
    --dump $DU/full40_fp64_iso                                          # ~390 s
```

### 9.4 比对 NPU dump
```bash
python compare_dumps.py $DU/full40_v2 <npu_dump> --tolerance isolated   # 阶段 A1
python compare_dumps.py $DU/full40_v2 <npu_dump> --tolerance default    # 阶段 B
```

### 9.5 判据 G-1：prefill↔decode 自洽（第 11 节）
```bash
python run_decode_check.py --cut 20 --layers 40 \
    --routing-plan $DU/plan_full40_v2 --force-layer-in $DU/full40_v2    # 逐层，判数值
python run_decode_check.py --cut 20 --layers 40 \
    --routing-plan $DU/plan_full40_v2                                   # 端到端，判 top-1
python run_decode_check.py --cut 10 --layers 40 \
    --routing-plan $DU/plan_full40_v2                                   # 换 cut 复测对照值
```

### 9.6 环形缓冲回绕（把滑窗调小，让回绕在 27 token 内发生）
```bash
python run_golden.py --layers 3 --threads 16 --window-size 8 --dump $DU/smoke3_win8
python routing_plan.py $DU/smoke3_win8 --out $DU/plan_win8
python run_decode_check.py --cut 20 --layers 3 --window-size 8 \
    --routing-plan $DU/plan_win8 --force-layer-in $DU/smoke3_win8
```
覆盖两条否则走不到的分支：**prefill 里 `seqlen > window` 那条（cutoff 拆分写环）**
和 **decode 的环形回绕**。⚠️ 数值不是真实模型的，但**自洽性仍然成立**（两条路径同一份 config），
所以它是有效的结构性测试。

### 9.7 batch > 1 一致性（逐位判据，无容差）
```bash
python run_golden.py --layers 40 --threads 16 --batch 2
```
把同一段 prompt 复制成 N 个序列，**每一行必须与第 0 行逐位相同**。
不同即 batch 处理 bug（缓存槽按错的维度索引、该 gather 的地方做了 broadcast）。
**这里没有任何东西可以合法地在两行之间不同，所以判据是逐位的。**

### 9.8 engram 参考（都不需要跑模型、不需要卡）
```bash
python engram_hash_ref.py --verify $DU/full40_v2 --decode-from 20   # 整数哈希，9 秒
python engram_lookup_ref.py --out $DU/engram_lookup_ref            # 查表 + 逐位前提验证
```

### 9.9 engram 的 `ElasticBuffer` 位保真（**需要 2 张卡**，见 `ENGRAM_NOTES.md`）
```bash
source /workspace/code/dsv4_single_card/vllm_project/build_artifacts/env_dsv41.sh golden 3
export ASCEND_RT_VISIBLE_DEVICES=2,3 HCCL_NPU_SOCKET_PORT_RANGE=17000-17099
export E_ENG_1_LOCAL_ONLY=1 E_ENG_1_NO_OOB=1        # 远端路径在本机挂，见 ENGRAM_NOTES 3.1c
torchrun --nproc_per_node=2 --tee 3 e_eng_1_bit_fidelity.py
```

### 9.10 容量与耗时测量（**不是**数值参考）
```bash
python run_golden.py --layers 40 --threads 16 --random-tokens 128 --no-engram
```
⚠️ `--no-engram` 会**跳过 layer 1/14 的 engram 查表**，所以结果**不能用于数值对照**。
随机 token 是刻意的：它命中更多专家，是容量估算的最坏情况。

---

## 实测数据（27 token，16 线程）

| 跑法 | 墙钟 | 峰值 RSS |
|---|---|---|
| 40 层 prefill，fp32 累加 | **216~372 s**（随机器负载波动） | 55 GB |
| 40 层，**fp64** 累加 + 双注入 | **390 s** | 55.1 GB |
| 3 层，fp64 + 强制路由 | 25 s | 7.2 GB |
| 40 层 decode 检查（1 次 prefill + 7 步） | ~600 s | 59 GB |

读取 52.4 GB 权重、加载 2610 个专家（65/层，共 384）、gather 1296 行 engram。

🔴 **「峰值 RSS」这个数会被误读**：它几乎全是 mmap 的 page cache（可回收），不是真实内存压力。
实测 `VmRSS 59.3 GB = RssAnon 3.6 GB + RssFile 55.6 GB`，而且峰值 RSS ≈ 已读取的
checkpoint 字节数、逐层单调上升从不下降。**真正随序列长度增长的是匿名内存（3~4 GB）。**
该看的是 `MemAvailable`（`MemoryGuard` 每层查一次，`--floor-gb` 默认 150），不是 RSS。

⚠️ 而且这台机器是共享的：实测出现过容器外进程把 `MemAvailable` 压到 131 GB
（`AnonPages 588 GB` vs 容器内 RSS 合计 59 GB），守卫在第 0 层就拒绝启动。
**约定：等，不降 floor，不碰容器外的东西。**

详见 REPORT.md。

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

### 10.6 与原 tilelang kernel 的语义比对结果

本机**没有 GPU、tilelang 也没装**，所以无法运行原 kernel 做数值对照。做的是**语义逐条比对**：
`/workspace/models/DeepSeek-V4.1-Flash/inference/kernel.py`（原 tilelang，只读）
对 `kernel_cpu.py`。下面每一条都附了行号，可以自己去查。

#### 🔴 P1 —— 已确认不一致，方向是「CPU golden 更准」，会影响验收阈值

**`sparse_attn` 的 P·V 用 bf16 概率，CPU golden 用 fp32。**

```python
# inference/kernel.py:339
acc_s_cast = T.alloc_shared((h, block), BF16)     # ← 概率的 bf16 缓冲
# :374   分母从 fp32 概率求和
T.reduce_sum(acc_s, scores_sum, dim=1)
# :377   概率拷进 bf16
T.copy(acc_s, acc_s_cast)
# :380   用 bf16 概率做 P·V
T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

注意**只有分子受影响，分母（`sum_exp`）两边都从 fp32 概率求和**。

**实测**（不是估算）：把 CPU golden 的 P 也舍到 bf16，`sparse_attn` 输出
`rel_l2 = 1.80e-3`，逐 (token, head) 最大 `3.27e-3`，884736 个元素里 630387 个逐位相同。

**⇒ 一个忠实照搬 GPU kernel 的实现，会和 CPU golden 有 ~1.8e-3 的固定差异，这不是 bug。**
第 6.2 节 `attn.out` 的阈值是 1.5e-2，有 8 倍余量，**所以阈值不用改**。
但**不要**因为看到 `attn.out` 有 1e-3 量级的差就去查算子。

需要逐位对照时，把 `kernel_cpu.SPARSE_ATTN_P_DTYPE` 设成 `torch.bfloat16`，
golden 就复现 GPU kernel 的行为而不是比它更准。

**两个 GEMM 的累加结构不同，但数学等价。** tilelang 每 32 个 K 做一次部分和 →
乘 scale → fp32 二级累加（`kernel.py:266-270` / `550-554`）；CPU 先反量化再一次稠密
fp32 matmul（`kernel_cpu.py:318-322` / `350-354`）。反量化本身**精确无舍入**
（e8m0 是 2 的幂、e4m3 只有 4 位尾数），所以差别纯是累加误差，相对量级 ~1e-6…1e-5,
落到 bf16 输出后基本不可见。

#### ⚠️ P0 —— 无法从源码确定，**需要一台有 GPU 的机器实测**

> ### 先读这条：`fp4_act_quant` 的三个调用点，以及哪条路才相关
>
> ```
> ref/model.py:546   Indexer 的 index key  k      block 32   scale_dtype = e8m0（默认）
> ref/model.py:552   Indexer 的 index query q     block 32   scale_dtype = e8m0（默认）
> ref/model.py:760   MLA._compress_kv 的 latent   block 16   scale_dtype = e4m3   ← 注意 16
> ```
> 官方在 760 行上方自己写了注释：
> `# Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.`
>
> 🔴 **block size 16 是全模型唯一一处**（其他量化点一律 32）。融合算子若硬编码 32，
> 那不是舍入级差异而是**量化粒度**差异：scale 覆盖范围翻倍、amax 取自两倍宽的组、
> 组内小值被多压掉几位。**这条应该排在舍入模式之前查 —— 它是「有没有」而不是「差多少」。**
>
> **三个点全部 `inplace=True`**，即量化完立刻反量化写回：fp4 在激活上只是「有损压缩再还原」，
> **从不产出 packed fp4 张量喂 GEMM**。所以下面的 P0-2（打包顺序）**对激活不适用**，
> 只对权重适用，而权重那条路 `convert.py` 是权威生产者。
>
> **舍入模式仍然适用，而且在 indexer 上比在专家上要紧** —— 因为它喂的是一个离散判决：
> index score 是在 4-bit 量化过的 q/k 上算的（`ref/model.py:557`），然后取 top-k（`:578`）。
> 舍入差异可能**翻转 top-k 选择**，和 MoE 路由同一类问题。
> （顺带这解释了第 5.1 节那个观察：fp32↔fp64 扰动下 `indexer_topk.out` 100% 稳定，
> 很可能是因为 **q/k 已被压到 4 bit，量化步长远大于 1e-7 的扰动**。）
>
> **专家路径不相关**（`dsv41-kernel` 实测）：昇腾 W4A8 MXFP4 路径的激活量化成
> `float8_e4m3fn` 而非 fp4（`w4a8_mxfp4.py:87/285`），且 `npu_dynamic_mx_quant`
> 实测**不接受 fp4 输出**（`aclnnDynamicMxQuantV2` error 561002）。
> 专家路径上 fp4 只存在于权重里，而权重是 checkpoint 里离线量化好的。

1. **`fp4_act_quant` 的 fp32→e2m1 舍入模式。** tilelang 是隐式 cast
   （`kernel.py:171/175` 赋值给 FP4 fragment），舍入模式由 codegen 决定，源码里看不到。
   **CPU 侧已验证是 e2m1 码格上的 round-to-nearest-even**（`kernel_cpu.py:145`，
   7 个中点逐个验过：0.25→code0、0.75→code2、1.25→code2、1.75→code4、2.5→code4、
   3.5→code6、5.0→code6，每次都取偶编码）。
   如果 tilelang 生成的不是 `cvt.rn`，落在中点上的元素最大相对偏差 **33%**（5.0 落 6.0 而非 4.0）。
   **验证方法**：把 `[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]/scale` 过一遍原 kernel，比码值。
2. **fp4 的 nibble 打包顺序。** tilelang 的 sub-byte layout 由它自己推导，源码里看不到
   （`kernel.py:512/537-540`）。
   **但 CPU 侧与权重的生产者 `convert.py:28-31` 逐字一致**：
   ```python
   low = x & 0x0F; high = (x >> 4) & 0x0F
   torch.stack([FP4_TABLE[low], FP4_TABLE[high]], dim=-1).flatten(2)
   ```
   即**低 4 位 = 偶数号元素、高 4 位 = 奇数号元素**（`kernel_cpu.py:162/172-173` 相同）。
   原 kernel 必须和 `convert.py` 一致才能跑通已发布的权重，所以**实际风险很低**；
   但形式上只能标「无法确定」。若真反了是灾难性错误（K 维相邻权重两两互换），
   不是小误差。**验证方法**：一条真实 fp4 权重行 + 已知激活做端到端点乘。
3. **`act_quant` 除法与 clamp 的中间 dtype。** `x_local` 是 bf16、`s_local` 是 fp32
   （`kernel.py:65/67/89`），tilelang 的类型提升规则无法从源码确认。提到 fp32 则与 CPU
   一致；若落回 bf16 则有 double rounding，个别元素差 1 个 e4m3 ulp。

#### ✅ 已确认一致的（可直接引用）

| 语义点 | 依据 |
|---|---|
| **三个 amax 下限** | `act_quant` → `1e-4`（`kernel.py:76`）；`fp4` **e4m3** 分支 → `6*(2**-9)`（`:162`，注释「keep even an all-zero group's scale nonzero」）；`fp4` **e8m0** 分支 → `6*(2**-126)`（`:165`）。与 `kernel_cpu.py:202/247/250` 一一对应，**值和分支条件都对** |
| clamp 边界 | fp8 ±448（`kernel.py:44-45`）、fp4 ±6（`:131`），**都在 cast 之前** |
| scale 算法 | 存的是 **scale = amax/MAX**，不是 1/scale。`scale_fmt` 非 None → `fast_round_scale` 向上取 2 的幂（`:36-37, 77-80`）|
| **e4m3 分支用 `amax / 6`、e8m0 分支用 `amax * (1/6)`** | `kernel.py:163` vs `:166`。1/6 不可精确表示，两者可差 1 ulp，经 `ceil(log2)` 可能翻一个指数档（factor 2）—— 这个易错细节 CPU 镜像正确（`kernel_cpu.py:248/251`）|
| **权重 scale 的两种形状** | `fp8_gemm`：**2D，block×block（实际 32×32）**（`kernel.py:231-232`）；`fp4_gemm`：**1D `[N, K/32]`，N 方向不分块**（`:514-515`，`weight_group_size = 32` 硬编码于 `:505`）。**两个 GEMM 不一样，别抄错** |
| `sparse_attn` running max 初值 | `T.fill(scores_max, -1e30)`（`kernel.py:355`），注释写了理由：`-inf` 会让全 -1 行产生 `exp(-inf - (-inf)) = NaN` |
| `sparse_attn` 空槽三条语义 | 越界槽补 -1（`:360`）、KV 行读 0（`:362`）、score 读 -inf（`:364`）|
| `attn_sink` 只进分母 | `sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])`（`:382-383`），无对应 value |
| `hc_split_sinkhorn` 迭代结构 | 第 1 轮行方向 softmax 后 `+eps`（`kernel.py:437-443`），列归一化每轮都做含第 1 轮（`:446-448`），之后 `iters-1` 轮行归一化再列归一化（`:450-458`）。**eps 位置也一致**：softmax 后加在结果上，其余全部加在分母里 |
| `hc` 的 pre/post/comb 拆分 | `+eps` **只加在 pre**（`:427`）；post 有系数 2 且**无 eps**（`:429`）；comb 是 `[hc,hc]` 行主序（`:431`）|

一个无影响的差异：`FP4_TABLE[8]` 官方 `convert.py:13` 是 `0.0`，`kernel_cpu.py:73` 是 `-0.0`。
`0.0 == -0.0`，参与乘加结果相同。

---

## 11. decode 路径与 prefill↔decode 自洽（判据 G-1）

```bash
DU=/workspace/code/dsv4_single_card/vllm_project/golden_dumps
python run_decode_check.py --cut 20 --layers 40 \
    --routing-plan $DU/plan_full40_v2 --force-layer-in $DU/full40_v2    # 逐层
python run_decode_check.py --cut 20 --layers 40 \
    --routing-plan $DU/plan_full40_v2                                   # 端到端
```

同一段文本喂两遍：一次整段 prefill，一次「短 prefill + 逐 token decode」，然后比。

### 11.1 decode 会碰到 prefill 完全不碰的东西

| 状态 | 在哪 | 谁有 |
|---|---|---|
| `attn.window_kv_cache` [1,128,512] | 滑窗环形缓冲 | **全部 40 层** |
| `attn.compress_kv_cache` [1,256 或 512,512] | 压缩 latent | kv_source 层 2/8/14/20 |
| `attn.compressor.kv_state` [1,2,512] fp32 | 未满组的 kv | ratio>1 的 kv_source 层 2/8/14 |
| `attn.compressor.score_state` [1,2,512] fp32 | 未满组的 gate 分数，**初值 `-inf`** | 同上 |
| `attn.indexer.k_cache` [1,256 或 512,128] | index key | index-key **owner** 层（2/8/14/20；24/28/32/36 复用 `shared_attn.index_k`，自己不持有） |

**这些全部注册为 `persistent=False`**，也就是权重加载器既不填它们、也不检查它们
（`load_block` 遍历的是 `state_dict()`）。而 golden 是**流式**的：建层 → 灌权重 → 跑 → 释放。
prefill 不需要任何东西跨层存活，**decode 需要，而它们就在刚被释放的那个 block 里**。

`golden.LayerStateStore` 负责搬运，并且是**按「在 `named_buffers()` 里但不在 `state_dict()` 里」
自动发现**，不是手写清单 —— 参考实现哪天多一个 cache，会被自动带上，而不是静默地每步从零开始。

⚠️ **`score_state` 初值是 `-inf` 而不是 0**（未填的槽必须拿到零 softmax 权重，0 会给它真实份额）。
所以「让 buffer 保持构造时的值」不是可行的替代方案。

### 11.2 两个模式，因为端到端模式单独回答不了这个问题

**端到端模式**（默认）：比 decode 的 logits 和整段 prefill 的 logits。

它的噪声地板**不小**：20 token 的 prefill 和 27 token prefill 的前 20 个位置**本来就不同** ——
每个 GEMM 的 M 维不同、舍入方式就不同，再叠加第 0.2 节那条「bf16 ULP 经因果 attention 扩散」。
所以这个模式**在同一次运行里测出这个地板**（两段共享的 prefill 前缀）并作为**对照**报出来。

**⇒ 端到端模式的判据是 top-1 一致，不是 rel_l2。** rel_l2 报出来但不 gate，
只标注它是对照的多少倍。

**逐层模式**（`--force-layer-in <golden_dump>`）：每个 block 都拿 golden 在**被解码那个位置**
的 `layer.in`，跨层累积消失，每层独立判定，用第 6.2 节的逐层阈值。**这是能定位 decode bug 的模式。**

**两个模式都强制路由**（`--routing-plan`）。prefill 和 decode 用不同的算术得到同一个量，
差 ~1e-7 —— 而这个量级的差异到 layer 2 会翻转两成 token 的路由。不强制的话，两个模式测的
都是路由分歧而不是 decode 路径。**路由方案按绝对序列位置寻址，所以对 decode 步原样可用** ——
这正是第 0 节那条设计约束的回报。

### 11.3 实测（3 层，路由 + 层输入双注入）

| 项 | 结果 |
|---|---|
| 每个 decode 步比对的张量数 | 71~72 |
| 超出逐层阈值的 | **0** |
| 最差 | `moe.out` @ L2 = 1.70e-2（阈值 4e-2）|
| decode 自己的 gate 选择 vs golden | **81/81 行相同**（`rows_changed: 0`）|
| 端到端对照（两段 prefill 的共享前缀）| 6.2e-3 |
| 端到端 top-1 | 7/7 位置一致 |

**decode 路径的 gate 是对的**：注入进去的和它自己算的是同一组专家。

### 11.4 这个检查暴露的两个「比对方法本身」的错

都不是模型的 bug，是判据的 bug —— 但都会表现为「实现有问题」：

1. **`indexer_topk.out` 的路径相关偏移**（见第 3 节的 🔴 警告）。不归一化就逐位比 →
   在实现完全正确的情况下报 100% 不一致。
2. **忘了按集合比 `moe_gate.out.1`**。我在新写的比对里漏了这一步，立刻报出一个「路由不一致」
   —— 查下去是**纯顺序调换**。第 0.1 节那两个被推翻的数字是同一个原因。
   **凡是比 `moe_gate.out.1` 的地方都要先 `sort(-1)`。**

### 11.5 仍未覆盖

- **多步生成**（decode 到 `start_pos > window_size = 128`）：环形缓冲的**回绕**还没测。
  27 token 的标准输入到不了那里。需要另一组长输入。
- **batch > 1** 的 decode。
- MTP / DSpark 草稿头的 `forward_spec`（本来就在范围外，见第 8 节）。
