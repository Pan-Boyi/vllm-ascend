# DeepSeek-V4.1-Flash CPU golden — 可行性结论与实测报告

日期：2026-09-10　机器：755 GB RAM / 256 vCPU　模型：`/workspace/models/DeepSeek-V4.1-Flash`（只读）

---

## 1. 一句话结论

**能做，而且已经做出来了。** 40 层完整 prefill 在纯 CPU 上跑通，216 秒，峰值内存 55 GB，
逐层 dump 271 MB。正确性有硬证据：`The capital of France is` → top-1 ` Paris`（logit 19.59，领先第二名 2.7）。

原始可行性判断全部复核无误：`transformers 5.14.1` 没有 `deepseek_v41`、config 无 `auto_map`、
顶层无 `modeling_*.py`（AutoModel / trust_remote_code 确实走不通）；GPU 依赖确实只有
`model.py:12-19` 的 6 个函数；`convert.py` 确实已经是 `device="cpu"`。

---

## 2. 跑通了什么 / 没跑通什么

### 跑通（有实测数据）

| 项目 | 状态 | 证据 |
|---|---|---|
| 6 个 kernel 的 CPU 实现 | ✅ | `test_kernel_cpu.py` 9 组 ~90 条断言全过 |
| 40 层完整 prefill | ✅ | 216 s，logits 全 finite |
| 语义正确性 | ✅ | `The capital of France is` → ` Paris`（19.59，领先 2.7） |
| 滑窗注意力（SWA） | ✅ | 全部 40 层 |
| **`-1` 空槽语义** | ✅ | 27 token < window 128，`get_window_topk_idxs` 真的产出上三角 -1，实跑覆盖 |
| engram n-gram 查表 | ✅ | layer 1 / 14，1296 行从 mmap gather |
| engram 哈希 | ✅ | compressed vocab 实测 99092，与 config 完全一致 |
| 压缩 KV + 两级 indexer | ✅ | layer 2/8/14/20 压缩，layer 20 为 candidate source |
| Hyper-Connections + Sinkhorn | ✅ | 全部 40 层 |
| MoE（384 专家 fp4） | ✅ | 平均 65 专家/层被实际选中并加载 |
| 逐层 dump 机制 | ✅ | 42 个 tag，271 MB |
| 比对工具 | ✅ | 自比对 PASS（strict 容差） |
| 可复现性 | ✅ | 两次独立 40 层运行 top-5 logits 数值完全一致 |

### 没跑通 / 没覆盖（如实列出）

| 项目 | 原因 | 影响 |
|---|---|---|
| ~~**decode（`start_pos > 0`）**~~ | ✅ **v1.1 已完成** | `golden.LayerStateStore` 跨流式层携带 KV 状态；`run_decode_check.py` 两个模式；40 层实测 6/7 个位置各 ~961 个张量全部在阈值内，第 7 个的唯一分歧是该层 margin 最小的 token（9.3e-5，全 27 个里最小）。见 `ALIGNMENT_PROTOCOL.md` 第 11 节 |
| ~~**decode 的环形缓冲回绕**~~ | ✅ **已完成** | `--window-size 8` 让回绕在 27 token 内发生，逐层模式 **PASS**。顺带覆盖了 prefill 里 `seqlen > win` 那条从未走到的分支 |
| ~~**batch > 1**~~ | ⚠️ **已定位，是舍入累积不是 bug；判据已定，但是诊断工具不是门禁** | 见下面的 §2.1 |
| **vision（ViT + Aligner）** | 本次限定纯文本路径 | 多模态未验证 |
| **DSpark / MTP 草稿头**（`mtp.*` 3 层） | 参考实现里 `forward_spec` 没有被推理循环调用 | 投机解码未验证 |
| **与 GPU/tilelang 原 kernel 逐位对照** | 本机无 GPU，`tilelang` 未安装 | 改做了**语义逐条比对**（`ALIGNMENT_PROTOCOL.md` 第 10.6 节，每条附行号）。结果：**已确认一处实质不一致** —— 原 kernel 的 `sparse_attn` 把 softmax 概率舍到 bf16 才做 P·V，golden 全程 fp32，**即 golden 比它要参考的 kernel 更准**，实测固定偏差 `rel_l2 = 1.80e-3`。**两处仍需 GPU 实测**：fp4 的 e2m1 舍入模式、fp4 的 nibble 打包顺序 |
| **与 vllm 基线交叉验证** | 未做 | 少了一个独立的精度锚点。（v1.0 时卡 4/5 的服务不在跑；现在基线线在跑，但这条交叉验证还没安排）|
| `fp4_act_quant` 非 inplace 打包路径 | `model.py` 里三处调用全是 `inplace=True` | 仅单测覆盖，实跑未走到 |

> 关于 vllm：任务说明提到 pid 45035 是精度基线且不能被杀。**本次工作全程没有触碰它**；
> 开始检查时它已经不在了（僵尸进程 + 无监听端口），不是被本次工作影响的。
> 全程内存最低点 558.9 GB，距 150 GB 下限极远。

### 2.1 `batch > 1`：已定位

同一段 prompt 复制成 batch=2，逐层比 row 0 vs row 1：**第一处分歧在 layer 0 的
`hc_mixes.0.0`，`3.638e-12`（fp32 舍入量级）**，然后单调增长到 layer 39 的 `2.828e+00`
（rel_l2 7.96e-2），路由在 layer 10 第一次翻转。

**layer 9 是关键证据：路由仍相同、差异已到 4.7e-2 —— 增长先于翻转，所以翻转是结果不是原因。**

**⇒ 不是 batch bug。** 同一次 batched GEMM 里两行拿到不同舍入（batch=2 时 MoE 输入是
`[54, 5120]` 而非 `[27, 5120]`，BLAS 分块随之不同）。

#### 🔴 顺带测出了一个比 batch 本身更重要的量

**摁住路由**（两行喂同一组专家）重跑：

| | 自由路由 | 强制路由 |
|---|---|---|
| 最终 `layer.out` rel_l2 | 7.96e-2 | **1.83e-2** |
| 第一次路由翻转 | layer 10 | layer 16 |
| 翻转层数 | 26 | 7 |

**剩下的 1.83e-2 里完全没有路由参与。1e-12 → 1.8e-2 是十个数量级的放大，
全部由「bf16 舍入 + 因果 attention 扩散」完成，没有任何离散事件。**

**⇒ 这是协议第 0.2 节那条机制目前最干净的对照实验**（之前是靠逐 token 追踪论证的）。
**⇒ 并且修正了两个注入开关的重要性排序**：`--force-layer-in` 切断的是**主力**路径，
强制路由处理的是**次要**那一项。**强制路由坏了判据变松；`layer.in` 坏了判据完全失效。**

#### 「真 batch bug」的可判别签名（已做成会报错的检查）

不看最终差多少，**看第一处分歧有多大**：≲1e-12 是舍入累积；**>1e-6 是接线错误**
（缓存槽索引错维度、该 gather 处做了 broadcast）。
`compare_batch_rows.py` 超阈值就 **exit 1** 并指出张量名和层号。

⚠️ **这一整节是诊断工具，不是验收门禁。** `batch > 1` 在不同实现之间**结构性不可逐位比**
（昇腾的 BLAS 分块必然与 CPU 不同）。**真正的 batch 验收在 GSM8K 那一级。**

---

## 3. 实测数据

### 3.1 性能与内存

| 场景 | token | 层数 | 耗时 | 峰值 RSS | 读取权重 | 专家/层 |
|---|---|---|---|---|---|---|
| 冒烟 | 27 | 3 | 22.3 s | 7.2 GB | 0.6 GB | 82 |
| **完整** | 27 | 40 | **216 s**（5.4 s/层） | **55 GB** | 52.4 GB | 65 |
| 语义 sanity | 5 | 40 | 125 s（3.1 s/层） | 23.6 GB | 20.5 GB | 20 |
| 扩展性探针 | 128（随机） | 40 | **371 s**（9.3 s/层） | **102 GB** | 99.5 GB | 132 |

耗时几乎完全由**专家权重读盘**主导（每层非专家权重加载只要 0.0–0.1 s）。
磁盘实测：顺序 1.6 GB/s（冷）、24.6 GB/s（page cache）、随机 4K 11389 IOPS。

RSS 随层数线性上涨约 1.3 GB/层（27 token）。这**不是泄漏**，是 glibc malloc 在反复
分配/释放 GB 级张量后的堆碎片——每层结束都 `del` + `gc.collect()`，系统可用内存
始终稳定在 558–570 GB。要压这个数，设 `MALLOC_TRIM_THRESHOLD_` 或显式 `malloc_trim()` 即可。

### 3.2 内存策略（关键设计）

checkpoint 510 GB 里，290 GB 是 MoE 专家、203 GB 是两张 engram 表。全量加载不可行，
所以**权重从不全量驻留**：

1. **逐层流式**：建层 → 从 checkpoint 填充 → 算 → 释放。峰值只有一层。
2. **专家按需**：gate 先跑，只读真正被选中的专家。27 token 时平均 65/384，省掉 83% 的读盘。
3. **engram 表完全不驻留**：一次前向只碰 384006168 行里的 1296 行，直接从 mmap gather，
   一行一次 page fault。**203 GB 的表，实际内存开销约 0。**

层间传递的 `shared_attn.compress_kv` / `index_k` / `topk_idxs` / `candidates` 都是小 cache
张量，生命周期长于产出它的层，所以流式不会打断跨层接线。

安全阀：`MemoryGuard` 在每层边界检查 `MemAvailable`，低于 `--floor-gb`（默认 150）直接抛
`MemoryError` 退出，绝不硬扛。全程未触发。

**不需要跑 `convert.py`**：它 mp=1 会生成一个 510 GB 的单文件，纯属浪费。
直接从 HF checkpoint 流式读更快也更省。

### 3.3 数值敏感度：本模型对齐的头号难点

**只把 GEMM 累加器从 fp32 换成 fp64**（一个让结果严格更准、单 GEMM 相对误差仅 ~1e-7 的改动）：

| 层 | 输出 rel_l2 | 逐元素比 topk（**v1.0 的错数字**）| 其中纯顺序调换 | **真实集合改变** |
|---|---|---|---|---|
| layer 0 | 1.0e-3 | 0 / 27 (0.0%) | 0 | **0.0%** |
| layer 1 | 2.9e-2 | 2 / 27 (7.4%) | **2（全部）** | **0.0%** |
| layer 2 | 1.0e-1 | 15 / 27 (55.6%) | **9** | **6 / 27 = 22.2%** |
| logits | 5.6e-2 | — | | |

> 🔴 **v1.0 报的 7.4% / 55.6% 是错的**（本报告修正）。原算法**按元素逐位比 `topk` 输出**，
> 把「专家集合相同、只是顺序调换」也算成了翻转。而 `topk` 的返回顺序**对结果没有任何影响**：
> MoE 对选中专家求和，累加顺序是按**专家编号**循环的，与 topk 顺序无关。
> **⇒ 比较专家选择必须按集合比**（`sort(-1)` 后再比）。**layer 1 其实一个 token 都没翻。**
>
> 结论方向不变（22.2% 照样让逐层比对失效），但**从 7.4% 推出的那条 ≤8% 阈值已作废**。

机理：`Gate` 用 `(scores + bias).topk(6)` 选专家，是**连续值上的离散判决**。
第 6 名和第 7 名接近时，任意微扰都会翻转选择，而两个不同专家的输出差异是 O(1)。

实测 margin 分布（第 6 名与第 7 名的选择分数之差）：

| 层 | 翻转 token 的 margin | 未翻转 token 的 margin |
|---|---|---|
| layer 1 | max 1.5e-2, 中位 1.2e-3 | min 9.4e-5, 中位 3.7e-3 |
| layer 2 | max 6.0e-3, 中位 2.1e-3 | min 1.4e-3, 中位 6.6e-3 |

**到 layer 2，两者已经完全重叠**——margin 阈值只在最前面一两层能区分「平局」和「真 bug」。

**这直接决定了对齐协议的形状**：昇腾侧必须实现一个「强制路由」调试开关（从 golden dump
注入专家编号+权重），否则 layer 2 之后的逐层数值比对就是在测噪声。
详见 `ALIGNMENT_PROTOCOL.md` 第 0 节和第 7 节。

> 🔴 **但强制路由是必要不充分的**（v1.1 新增）。实测强制路由后 layer 2 的 `layer.out`
> 仍是 **2.27e-2**（vs 自由路由 3.75e-2），**已超过协议自己定的 2e-2，而无任何 bug**。
>
> 逐 token 定位到的机理比「路由滚雪球」更根本：**残差流是 bf16（相对 ULP 3.9e-3）**，
> 一个 ~1e-7 的算术差异对 25/27 个 token 四舍五入后完全相同，但某个 token 恰好落在进位
> 边界 → 翻 1~2 个 ULP → **因果 attention 把它扩散给所有下游 token**（`hc_pre.0` 2/27 脏
> → `attn.out` 24/27 脏）→ 每层重复，污染比例饱和到 100%。
>
> **⇒ 整张量 rel_l2 的增长主要来自「被污染 token 比例变大」，不是算子放大噪声。**
> **⇒ 所以要第二个开关：注入 `layer.in`。** 40 层实测，注入后 `logits` 从 **1.03e-1**
> 降到 **1.5e-3**（70 倍），每层偏差回到 5e-3 量级。

相比之下 `indexer_topk` 在同样扰动下 **100% 稳定** —— **但这个结论不能外推**：
27 token 在 ratio=2 的层上实际 `K = min(512, 13) = 13`，是「13 个里选 13 个」，
几乎没有选择余地；而且 q/k 已被 `fp4_act_quant` 压到 4 bit，量化步长远大于 1e-7 的扰动。
长序列下 K 逼近 512、候选远多于 512 时它会变回真正的离散判决。

### 3.4 扩展性

实测 128 随机 token / 40 层：**371 s，峰值 RSS 102 GB，读取 99.5 GB，平均 132 专家/层**
（单层 4–16 s，随该层命中数 53–204 波动）。对比 27 token 的 216 s / 55 GB / 65 专家：

| token | 耗时 | 峰值 RSS | 读取 | 专家/层 |
|---|---|---|---|---|
| 27 | 216 s | 55 GB | 52 GB | 65 |
| 128 | 371 s | 102 GB | 100 GB | 132 |

两个趋势要注意：

1. **专家命中数随序列长度增长并趋于饱和**（65 → 132，上限 384）。读盘量跟着涨，
   最坏情况全命中约 290 GB/次前向。
2. **峰值 RSS 也跟着涨到 102 GB** —— 🔴 **v1.0 把这归因为 malloc 堆碎片，那是错的。**

   实测分项（`MemoryGuard` 现在分别报告）：
   ```
   VmRSS 33.9 GB = RssAnon 2.2 GB + RssFile 31.7 GB
   ```
   **95% 是 mmap 的 file-backed page cache**，不是堆。三条证据：
   - 峰值 RSS ≈ **已读取的 checkpoint 字节数**（52.4→55.1、99.5→101.6）；
   - 增量 47 GB / 2687 个额外专家 = **17.5 MB/专家**，正好是一个专家三个 fp4 矩阵的大小；
   - RSS **逐层单调上升、从不下降**，而每层权重明明已经释放 —— 堆碎片不会有这个形状。

   **⇒ 真正随序列长度增长的是匿名内存，实测只有 2~4 GB。**「512 token 摸到 200–300 GB
   会逼近安全线」这条外推**在外推错的量**：那个增长是可回收的 page cache，
   而 `MemAvailable` 本来就把它算作可用。
   **⇒ 长序列落地不需要先做内存优化**，§6 原先那 2 人天里的 `malloc_trim` 那部分可以去掉。

   ⚠️ **但不等于「我们很安全」。** 正确的说法是「**我们自己不是风险源，但会被别人的占用挡住**」：
   本次工作后期机器可用内存被**容器外**进程压到 131 GB（`AnonPages 588 GB` vs 容器内
   RSS 合计 59 GB），`MemoryGuard` 在第 0 层边界就拒绝启动（own RSS 3.3 GB），
   四个排队的验证跑全部被挡。**约定：等，不降 floor，不碰容器外的东西。**

原始数据见 `logs_scale128.txt`。（这一项用随机 token 是刻意的：随机 token 会命中更多专家，
是容量估算的最坏情况；正式对齐请用 §2 的固定文本。）

---

## 4. 实现要点与踩到的坑

1. **`float4_e2m1fn_x2` 在 CPU 上是纯存储类型**，没有任何转换 kernel
   （`NotImplementedError: "copy_kernel" not implemented for 'Float4_e2m1fn_x2'`）。
   fp4 的量化取整（含 ties-to-even）、打包（低 nibble 存偶数元素）、解包全部手写位运算。
2. **`to(torch.float8_e4m3fn)` 溢出返回 NaN 而不是饱和**（实测 `500.0 → nan`）。
   必须先 clamp 到 ±448。原 tilelang kernel 同样依赖调用方先 clamp。
3. **`fast_round_scale` 用整数位运算复刻**，不走 `log2/ceil`——浮点 log 在整数边界上
   可能落到错误一侧，而这是决定量化码的离散判决。
4. **找到并修掉一个真 bug**：`dequant_fp8_weight` 里链式 `repeat_interleave(...)[:N]`
   的第二次切片切错了轴。方阵测试形状下会「碰巧正确」，只有非方阵才暴露。
   已修，并补了 `test_dequant_fp8_weight_axes`（N≠K 且都不是 block 整数倍）守住。
5. **忠实保留了一个 kernel 怪癖**：`act_quant(round_scale=False, scale_dtype=e8m0)`
   用**未取整**的 fp32 scale 量化，却存**取整后**的 e8m0 scale，两者对不上。
   `model.py` 永远传 `scale_fmt="ue8m0"` 所以碰不到，但没有偷偷「修正」它，
   而是写了专门的测试固定这个行为。
6. **checkpoint 名字几乎不用映射**：HF checkpoint 的 key 已经等于模块的 `state_dict` key。
   只有两处例外——`wo_a` 需要 fp8+scale 反量化成 bf16（与 `convert.py` 一致），
   专家权重是 `int8` 需要 `.view(torch.float4_e2m1fn_x2)`。
7. **不需要改 `model.py` 的任何 cuda 假设**。11 处 cuda/device 引用里，10 处在
   `world_size > 1` 分支内（默认 1，全部短路），第 11 处在 `if __name__ == "__main__"`
   自测块里。作为模块导入时一处都碰不到。`ref/` 下的 `model.py`/`engram.py`/`vision.py`
   是**逐字副本**，只额外放了一个 `kernel.py` shim 指向 CPU 版。

---

## 5. 全量 golden 可行性结论

**Prefill 的全量 golden 已经可行且已交付**，不需要额外资源：

- 内存：55 GB 峰值（27 token），远低于 150 GB 安全线
- 时间：216 秒/次
- 存储：271 MB/次 dump
- 不需要 GPU/NPU，不需要转换 checkpoint

**Decode 的全量 golden 还不可行**，缺的是跨层缓存持久化（见下）。

---

## 6. 人天估计：做成生产可用的 golden

以下按「一个熟悉这套代码的人」估算，是在当前原型之上的增量。

| 工作项 | 人天 | 说明 |
|---|---|---|
| ~~**decode 路径**~~ | ✅ **已完成** | `LayerStateStore` 按「在 `named_buffers()` 里但不在 `state_dict()` 里」自动发现要携带的 buffer，不是手写清单。两个验证模式（端到端判 top-1、逐层判数值）。⚠️ 剩：环形缓冲回绕、batch>1 |
| ~~**强制路由注入模式**~~ | ✅ **已完成** | `ROUTING_INJECTION.md` + `routing_plan.py`。按**绝对序列位置**寻址（不是张量行号），两条 fail-safe 拒绝行为。端到端验证：注入后 `moe_gate.forced.1` 与 golden 逐位相同。model 线已实现并 10 passed。**并且额外发现「强制路由不充分」，加了第二个开关 `--force-layer-in`** |
| ~~**与 GPU 原 kernel 对照**~~ | ⚠️ **改为语义比对，已完成** | 无 GPU 无 tilelang，改做带行号的语义逐条比对（第 10.6 节）。**剩两条必须有 GPU 才能定**：fp4 的 e2m1 舍入模式（若不是 RNE，中点元素最大偏差 33%）、fp4 nibble 打包顺序（CPU 与 `convert.py` 逐字一致，风险低但若反了是灾难性）。**建议：0.5 人天，前提是能借到一台 GPU 机器** |
| **engram host 查表 / offload** | 3~5 | **首周那两个问题已经答完，而且答案改变了路线** —— 见 `ENGRAM_NOTES.md`。摘要：<br>✅ **G-3 的前提已实测验证**（e8m0 是纯指数、e4m3 三位尾数 → 乘积在 fp32 里精确无舍入，bf16 结果与 fp64 逐位相同）→ **可以要求逐位相同，不需要容差**<br>✅ **`ElasticBuffer` 的 engram 路径字节透明**（埋 2080 个 NaN + 16 个 Inf，全部逐位存活；加源码三层证据共四层一致）→ **fp8 装 bf16 外衣可行，host 203 GB 而非 393 GB**<br>❌ **远端 URMA/RDMA 取行在本机不可用**（kernel 内断言 `Urma drain failed, ret=255`）→ **官方那种分片表形态走不通**<br>❌ **「Triton gather kernel 能否在 A5 编过」这个问题的前提消失了** —— 官方实现里根本没有 Triton，gather 全在闭源 AIV 里<br>⚠️ 三个实现前提（热身集合通信 / `hidden % 128 == 0` 压线 / `eb.destroy()`），每个的症状都误导 |
| **与 vllm 基线交叉验证** | 1.0 | 等 vllm 服务恢复，比 logits / top-1 / 困惑度 |
| **vision 路径** | 1.5 | ViT + Aligner + `image_processor.prepare_vl_inputs`，还要准备固定图片输入 |
| **DSpark / MTP** | 2.0 | `forward_spec` 在参考实现里没有被调用过，等于要自己搭驱动循环 |
| **长序列性能** | 2.0 | 512+ token 时专家命中逼近全量。需要专家权重跨 token 复用/预取，或改成「按专家分组批处理 token」 |
| **CI 化 + 基线固化** | 1.5 | 固定输入的回归基线、阈值配置、失败时自动定位首个分歧层 |
| **文档与交接** | 0.5 | |
| **剩余合计** | **11.5~13.5 人天** | 原估 14，已完成前三项（原估 5.5），新增 engram offload 3~5 |

**v1.0 列的前三项（5.5 人天）已全部完成**，golden 已经从「原型，端到端看起来对」推到
「可以拿去判昇腾实现对错」—— 而且已经在被用：model 线拿它验了上游共享层
（`hc_pre.0` 逐位相同、`hc_mixes.*` 在 1e-7 量级，比验收线低四个数量级），
kernel 线拿它验了 `hc_pre` 的 d=5120（4096/5120/7168 全部工作）。

**做这三项的过程中，最有价值的产出不是代码，是四个被推翻的判据**（详见
`ALIGNMENT_PROTOCOL.md` 第 0.1 / 6.0 / 6.5 节和第 10.6 节）：

| # | 原判据 | 实际 | 危害 |
|---|---|---|---|
| 1 | 路由翻转 layer 1: 7.4% / layer 2: 55.6% | **0% / 22.2%** | 按元素逐位比 topk，把纯顺序调换算成翻转。方向不变，但阈值是从错数字推的 |
| 2 | `strict = 5e-3` 全局阈值 | **压在 bf16 表示地板上**（1 ULP = 3.9e-3）| 已知无 bug 的参考跑，40 层里 26 个张量「失败」→ **假阳性发生器** |
| 3 | `logits ≤ 3e-2` | **两次纯 prefill 的地板就有 1.03e-1** | 根因是**滑窗索引张量宽度不同**（`min(seqlen, window_size)`），不是 GEMM 的 M 维 —— 后者那个解释我先前给错了，已被 `window_size=8` 时对照值降到 `0.000e+00` 推翻。滑窗槽位数与 golden 不一致的实现会必然假失败 |
| 4 | 「golden 是数值参考，比硬件准」→ 以为差异都该往一个方向 | **golden 在 `sparse_attn` 上确实更准，固定偏差 1.8e-3** | 忠实实现看起来「差了 1e-3」，四条线会一起去翻算子 |

**#2 和 #3 是同一类：阈值定在噪声地板以下。#4 是它的镜像：参考比被参考对象准。
两个方向都会把人送去查不存在的 bug，而协议是唯一能挡住这类误判的地方。**

**建议的下一步优先级**：

1. **环形缓冲回绕验证**（0.2 人天）—— 已实现开关，只差跑。当前被机器内存占满挡住。
2. **engram host 查表**（3~5 人天）—— 判据前提已验证，但首周那两个问题需要卡。
3. **借一台 GPU 机器把 fp4 那两条定下来**（0.5 人天）—— 现在它们是「被标出来的未知」，
   比「被当成一致的未知」好，但仍是未知。

---

## 7. 交付物

| 文件 | 说明 |
|---|---|
| `kernel_cpu.py` | 6 个 kernel 的 CPU 实现（含设计契约说明） |
| `test_kernel_cpu.py` | 单测，9 组 ~90 条断言 |
| `ckpt.py` | mmap safetensors 读取器 + 超大张量行 gather |
| `golden.py` | 流式逐层执行器 + dump 机制 |
| `run_golden.py` | 命令行入口 |
| `compare_dumps.py` | golden ↔ NPU dump 比对工具 |
| `ALIGNMENT_PROTOCOL.md` | **给昇腾实现方的对齐协议**（可直接交付） |
| `README.md` | 复现步骤 |
| `ref/` | 参考实现逐字副本，仅新增 `kernel.py` shim 指向 CPU 版 |
| `dumps/full40/` | 40 层完整 dump（271 MB，42 个 tag） |
| `logs_*.txt` | 各次运行的原始日志 |
