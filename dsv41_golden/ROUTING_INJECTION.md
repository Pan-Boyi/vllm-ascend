# 强制路由注入 · 接口规范 v1.0

`ALIGNMENT_PROTOCOL.md` 第 7 节的落地规范。给 `dsv41-model`（vllm-ascend 侧）和
`dsv41-kernel` 直接实现用。

**为什么是必做项，不是可选项**：`Gate` 用 `(scores + bias).topk(6)` 选专家 ——
连续值上的离散判决。实测把 GEMM 累加器 fp32→fp64（严格**更准**、单 GEMM 只动 ~1e-7），
**专家集合**改变的 token 比例是 layer 0: 0% → layer 1: 0% → **layer 2: 22.2%**（6/27）。
两个专家输出差 O(1)，而到 layer 2 翻转 token 的 margin（≤2.1e-3）与未翻转的（≥1.1e-3）
已经重叠 —— margin 阈值失去区分力。**不消掉这个离散判决，layer 2 之后的逐层数值比对
就是在测噪声。**

🔴 **本节 v1.0 写的是 7.4% / 55.6%（以及 margin ≤6.0e-3 / ≥1.4e-3），那些数字是错的，
已修正为上面这组。** 原算法按元素逐位比 `topk` 输出，把纯顺序调换算成翻转；而 `topk` 的
返回顺序对结果没有任何影响（MoE 按**专家编号**循环累加）。layer 1 那 2 个「翻转」全是
顺序调换，layer 2 的 15 个里 9 个是。**⇒ 专家选择必须按集合比**（`sort(-1)` 后再比）。
详见 `ALIGNMENT_PROTOCOL.md` 第 0.1 节。**结论方向不变，注入仍是必做项。**

🔴 **而且它是必要不充分的**：强制路由只把 40 层的误差减半，真正的主力开关是注入
`layer.in`（协议第 0.2 节、第 7 节）。

---

## 0. 🔴 头号设计约束：按「绝对序列位置」寻址，绝不按「张量行号」

这一条排在最前面，因为它是本规范里**唯一一个会在 smoke test 上必然通过、只在真实调度下
暴露、且暴露形式伪装成待测对象**的坑。

golden 是 batch=1 纯 prefill，gate 输入的第 j 行就是第 j 个 token，行号恰好等于位置。
**别的场景都不是**：

| 场景 | 行号 ≠ 位置的原因 |
|---|---|
| chunked prefill | 序列被切成多次 forward，第二段的行 0 是位置 N |
| padding | batch 对齐插入死行 |
| expert dispatch | 为按专家分组而重排 token |
| decode | 每步只有 1 行，位置是 `start_pos` |

如果注入按行号对齐：

1. **在单段、无 padding、batch=1 的 smoke case 上必然通过** —— 因为那时行号确实等于位置。
2. 一上真实调度就错位。
3. **错位的表现形式是「路由不一致」—— 而这恰恰就是我们要测的现象。**
   bug 会完美伪装成待测对象，而且是在已经有几十层 dump 要排查的时候才暴露。

所以注入接口的签名**必须**带位置映射，由实现方回答「本次 gate 调用的第 j 行是哪个绝对位置」。

### 要求的签名

```python
def override_routing(
    layer_id: int,
    positions: LongTensor,   # [n_rows] 第 j 行对应的绝对序列位置
    weights:   Tensor,       # [n_rows, topk] 实现自己算的路由权重
    indices:   Tensor,       # [n_rows, topk] 实现自己算的专家编号
) -> tuple[Tensor, Tensor]:  # 替换后的 (weights, indices)，dtype/device 与入参一致
```

参考实现：`routing_plan.py` 的 `RoutingPlan.override()`。NPU 侧不必用这个类，
但**必须复现它的语义**，特别是下面两条拒绝行为。

### 两条必须 fail-safe 的拒绝行为

两者都是「宁可报错，不可静默放行」。静默放行会把接线 bug 变成看起来很合理的精度 bug。

1. **`positions` 里出现方案未覆盖的位置 → 抛错**，不要退回实现自己的 top-k。
   一个「部分强制」的层产出的不一致，长得和算子 bug 一模一样。
2. **`layer_id` 不在方案里 → 抛错**，同理。

### 注入点的位置

```
gate 算出 (weights, indices)
    → ★ 注入点在这里 ★
    → global expert id → local (EP 分片映射)
    → dispatch / permute
    → expert 计算
```

A5 是 TP=1（硬约束），今天没有 EP 分片，所以专家编号就是全局编号。
但**注入点必须留在 global→local 映射之前** —— 这条今天不生效，半年后引入 EP 会咬人。

---

## 1. 注入数据从哪来

### 推荐：读小方案文件，不要解析整个 dump

```bash
python routing_plan.py /workspace/code/dsv4_single_card/vllm_project/golden_dumps/full40
# → 同目录产出 routing_plan.safetensors (~130 KB) + routing_plan.json
```

完整 dump 是 271 MB / 40 个文件的激活值；注入只需要每层两个张量。
`routing_plan.safetensors` 的键：

| 键 | 形状 | dtype | 含义 |
|---|---|---|---|
| `input_ids` | [T] | int32 | 方案对应的 token，**用来拒绝喂错文本** |
| `positions` | [T] | int32 | 第 t 行的绝对序列位置（prefill 就是 0..T-1，显式写出，不要靠推断） |
| `layer_NN.indices` | [T, 6] | int32 | **全局**专家编号 |
| `layer_NN.weights` | [T, 6] | fp32 | 路由权重，**已归一化、已乘 route_scale** |

`routing_plan.json` 带 `schema_version` / `layers` / `topk` / `expert_id_space` /
`accum_dtype` / `source_dump` / `input_ids`。

> ⚠️ **不同累加精度产出的方案不可互换。** `accum_dtype` 就是为此记录的 ——
> fp32 方案和 fp64 方案的路由本身就有 55% 的差异，这正是我们要消掉的东西。
> 拿错了等于在注入噪声。

### 备选：直接读 dump

`layer_NN.safetensors` 里的 `moe_gate.out.1`（[27,6] int64，专家编号）和
`moe_gate.out.0`（[27,6] fp32，权重）。语义与上表相同。

---

## 2. 注入语义（逐条）

| # | 规则 | 依据 |
|---|---|---|
| 1 | **直接替换 `(weights, indices)`，不再归一化、不再乘 `route_scale`** | `ref/model.py:823-826`：`weights = scores.gather(1, indices)`，然后 `weights /= sum + 1e-20`（注意是 `1e-20` 而不是 `norm_eps`，匹配训练），再 `weights *= route_scale`（=1.5）。dump 的是这一串之后的值 |
| 2 | **weights 和 indices 必须一起注入** | `norm_topk_prob` 按**选中集合**重新归一化。只注入 indices、让实现自己算 weights，等于换了归一化分母，数值依然不可比 |
| 3 | **`bias`（`e_score_correction_bias`）只参与选择、不参与加权** | `ref/model.py:822-823`：`indices = (scores + bias).topk(k)[1]`，但 `weights = scores.gather(...)` 取的是**无 bias** 的 scores |
| 4 | **shared expert 不在注入范围内**，无条件照常算、照常加 | `n_shared_experts = 1`；`MoE.forward` 里 `y += self.shared_experts(x)` 与路由无关。`moe.out` 含 shared expert，`moe_gate.out.*` 只覆盖 6 个 routed expert |
| 5 | **专家编号按全局解释** | golden `world_size=1`，384 专家全量按需读取，没有分片映射要还原 |
| 6 | **40 层全部是 MoE 层，没有 dense 层** | 本模型 config **没有** `first_k_dense_replace` / `n_dense_layers` 字段；`get_moe_config` 对 `layer_id < 40` 一律返回 `(384, 6)`。`golden_dumps/full40` 的 40 个 layer 文件也全部含 `moe_gate.out.1`（两路验证）。**不需要写 dense 分支** |

---

## 3. 🔴 阶段 A 的 gate dump 语义（必须一次定死）

问题：阶段 A（注入生效）时，dump 里的 `moe_gate.out.*` 该记「实现自己算的值」还是
「注入进去的值」？

**定案：`moe_gate.out.*` 记实现自己算的值，另外增加两个键记实际注入的值。**

| 键 | 记什么 | 用途 |
|---|---|---|
| `moe_gate.out.0` / `.1` | **实现自己算的** gate 输出（照常算，只是不用） | 阶段 B（自由路由）的判据。阶段 A 一趟同时把阶段 B 的数据取到，不用跑两遍 |
| `moe_gate.forced.0` / `.1` | **实际注入进 expert 的值** | 见下 |
| `moe_routing_margin` | 实现自己算的 margin | 诊断 |

为什么必须多这两个键 —— 这是对原方案的**唯一实质修改**：

只记「自己算的值」，注入是否真的生效**在 dump 里没有任何痕迹**。
于是「环境变量拼错 / 方案文件没找到 / 注入被静默跳过」这类接线故障，
表现为「逐层数值在 strict 容差下全面超标」—— 和真正的算子 bug 完全同一个症状，
而这时候你会去查算子。

有了 `moe_gate.forced.1`，比对脚本可以**先做一步逐位检查**：
`moe_gate.forced.1` 必须与 golden 的 `moe_gate.out.1` **逐位相同**（27×6 个 int32，
几乎零成本）。不同 → 立刻报「注入未生效」，而不是让你去查 GEMM。

> 记 `moe_gate.forced.*` 时请记**注入函数返回后、dispatch 之前**的值，
> 而不是把读进来的方案原样回写 —— 否则它只证明「文件读到了」，
> 不证明「expert 真的用了这个」。

同时 `manifest.json` 必须写：

```json
{ "force_route": "off" | "moe" | "moe+indexer",
  "routing_plan": "<方案文件绝对路径>",
  "routing_plan_accum_dtype": "torch.float32" }
```

比对脚本按 `force_route` 切换判定顺序：

- `off`（阶段 B）：按 `ALIGNMENT_PROTOCOL.md` 第 5 节原流程，路由是判据。
- `moe`（阶段 A）：**先验 `moe_gate.forced.1` 逐位相同**，再上 `--tolerance strict`
  比数值。此时 `moe_gate.out.1` 的不一致是**预期的、仅供参考**，不作判据 ——
  没有这个标志，比对脚本会自己跟自己比、永远全过或永远全错。

---

## 4. 环境变量

`dsv41-model` 提的三个挡位照用，只改一处（第一个指向方案文件而非整个 dump）：

```
VLLM_DSV41_ROUTING_PLAN = <routing_plan.safetensors 的路径，或它所在目录>
VLLM_DSV41_FORCE_ROUTE  = off | moe | moe+indexer     # 默认 off
VLLM_DSV41_DUMP_DIR     = <NPU dump 输出目录>
```

仍接受 `VLLM_DSV41_GOLDEN_DIR` 指向 dump 目录作为备选（`RoutingPlan.load()`
接受目录或文件路径）。

**启动时必做一次 `check_input_ids`**：把本次请求的 token 与方案的 `input_ids` 比，
不同就抛错。代价是 27 次整数比较，挡掉的是「拿 A 文本的方案跑 B 文本」——
那会表现为灾难性精度回退，极难归因。

---

## 5. indexer 注入

**同意排在 MoE 注入之后**：`moe+indexer` 挡位先占位、先不接。实测 fp32↔fp64 扰动下
`indexer_topk.out` 100% 一致，不是当前风险点。

但记下这个数字，别把「100% 稳定」当成结论外推：

```
topk = min(index_topk, end_pos // ratio)      # ref/model.py:577
```

`index_topk = 512`，但 27 token 的标准输入在 ratio=2 的层上 `end_pos//ratio = 13`
→ **实际 K = 13，不是 512**。也就是说「100% 稳定」是在 **K=13 且要选满 13 个中的 13 个**
（几乎没有选择余地）的条件下测出来的。长序列下 K 逼近 512 而候选数远大于 512 时，
它就变回一个真正的离散判决，裕度会小得多。**长序列那组上线时必须重测，届时再决定是否接。**

各层实际形状（27 token 标准输入，取自 `golden_dumps/full40/manifest.json`）：

| 层 | `compressor.out` | `indexer_topk.out` |
|---|---|---|
| 2 / 8 / 14（ratio 2, kv_source） | [1, 13, 512] | [1, 27, **13**] |
| 20（ratio 1, kv_source） | [1, 27, 512] | [1, 27, **27**] |
| 24 / 28 / 32 / 36（ratio 1, 非 kv_source） | —— | [1, 27, **27**] |

`compress_ratios` 是**逐层**列表（长度 43 = 40 层 + 3 个 MTP），layer 0/1 是 0。

---

## 6. `-1` 空槽语义

```python
# ref/model.py:577-579
topk = min(self.index_topk, end_pos // ratio)
idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
return torch.where(idxs < compress_lens, idxs + offset, -1).int()
```

- **`-1` 补在尾部，且逐行连续。** 先按位置升序 `sort`，不可达位置的下标必然 ≥
  `compress_lens`（最大的那些），排序后落在尾部。**每行的尾部长度不同**
  （prefill 时 `compress_lens` 是 per-row 的 `[seqlen,1]`）。
- **`indexer_topk.out` 是完全确定的**，尽管底层 `topk` 在 -inf 平局上不确定：
  被选中的不可达位置无论是哪几个，都统一映射成 `-1`。所以可以要求**逐位相同**。
- 实测（`golden_dumps/full40/layer_02`）每行 `-1` 个数：
  `[13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0,0]`

  → **第 0 行整行是 `-1`。** 也就是 `ALIGNMENT_PROTOCOL.md` 第 8 节
  「整行全 -1 时输出必须是全 0 而不是 NaN，running max 用有限的 -1e30 而非 -inf 兜底」
  这条**在标准输入上就被触发**，不需要构造 case。第一个 token 在 layer 2 上没有任何
  可达的压缩位置。NPU 实现如果这里返回 NaN，标准 smoke test 第一层稀疏 attention 就炸。

---

## 7. 其余确认项

- **`layer.in` 的 `[1,27,4,5120]` 中 dim=2 的 4** = `hc_mult = 4`，hyper-connection
  的 4 份残差流副本。`embed.out` 是扩展**之前**的 `[1,27,5120]`，扩展方式
  `ref/model.py:1134`：`x.unsqueeze(2).repeat(1, 1, 4, 1)` —— 4 份在扩展点**完全相同**，
  所以副本顺序在这里不可能对不上；分化从第一个 `hc_mixes` 开始。
  初始 `pre_mix` 见 `make_identity_pre_mix`（`ref/model.py:1159`）。
- **`engram_hash_ids` 的 `[1,27,2,24]` 中 dim=2 的 2**：顺序是 **[layer 1, layer 14]**。
  依据 `engram_layer_ids = [1, 14]` 且 `layer_hash_index = layout.layer_ids.index(layer_id)`
  （`ref/model.py:338`）—— 索引 0 → layer 1，索引 1 → layer 14。
- **NPU 侧的三个强制项**（关 chunked prefill、关 aclgraph、`enforce_eager`）合理且必要，
  并且请把 `num_tokens == 27` 这个断言保留 —— 即使注入已经按位置寻址，
  它仍然是「dump 形状是否如预期」的廉价护栏。

---

## 8. 收敛顺序

| 阶段 | 开关 | 容差 | 判据 |
|---|---|---|---|
| **A1** | `moe` + `layer_in` | **`isolated`**（per-tensor，见协议第 6.2 节）| `moe_gate.forced.1` 逐位相同 → 每层独立比数值。超标即真 bug |
| A2 | `moe` | 见协议第 6.4 节 | 验证跨层累积是否符合预期，**不是找 bug** |
| B | `off` | `default` | `moe_gate.out.0/1`（**按集合比**）+ logits **top-1**。单独验 gate 本身 |

阶段 A 一趟同时产出阶段 B 需要的 gate 数据（`moe_gate.out.*` 记的是实现自己算的值），
所以不用跑两遍。

🔴 **本节 v1.0 说阶段 A 用 `strict`（rel_l2 5e-3），那是错的，已撤回。**
`5e-3` 正好压在 bf16 的表示地板上（1 ULP = 3.9e-3），拿它比一个**已知无 bug** 的参考跑，
40 层里 26 个张量「失败」。**现在的阈值按 dtype 取地板**（bf16 至少 3 ULP = 1.17e-2），
见协议第 6.0 / 6.2 节。

🔴 **而且「强制路由 + `strict`」这个组合本身是不够的**：实测强制路由只把误差减半，
`layer.in` 注入才是切断主力路径的那个开关 —— **A1 必须两个开关都开**，见协议第 0.2 节。
