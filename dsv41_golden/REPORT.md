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
| **decode（`start_pos > 0`）** | 流式执行逐层释放权重，KV cache 随之销毁，未实现跨调用持久化 | **最大缺口**。滑窗环形缓冲、compressor 的 `kv_state`/`score_state` 跨步状态、indexer 单 query 分支全未验证 |
| **vision（ViT + Aligner）** | 本次限定纯文本路径 | 多模态未验证 |
| **DSpark / MTP 草稿头**（`mtp.*` 3 层） | 参考实现里 `forward_spec` 没有被推理循环调用 | 投机解码未验证 |
| **与 GPU/tilelang 原 kernel 逐位对照** | 本机无 GPU，`tilelang` 未安装（`ModuleNotFoundError: No module named 'tilelang'`） | kernel 只对照了「我对 kernel.py 的阅读 + 独立朴素实现 + 端到端语义」，**没有**对照过真实 GPU 输出 |
| **与 vllm 基线交叉验证** | **卡 4/5 上的 vllm 服务当前没在跑**——只剩 `[vllm] <defunct>` 僵尸进程，`ss -ltnp` 无监听端口 | 少了一个独立的精度锚点 |
| `fp4_act_quant` 非 inplace 打包路径 | `model.py` 里三处调用全是 `inplace=True` | 仅单测覆盖，实跑未走到 |
| batch > 1 | 未测 | — |

> 关于 vllm：任务说明提到 pid 45035 是精度基线且不能被杀。**本次工作全程没有触碰它**；
> 开始检查时它已经不在了（僵尸进程 + 无监听端口），不是被本次工作影响的。
> 全程内存最低点 558.9 GB，距 150 GB 下限极远。

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

| 层 | 输出 rel_l2 | 路由被翻转的 token |
|---|---|---|
| layer 0 | 1.0e-3 | 0 / 27 (0.0%) |
| layer 1 | 2.9e-2 | 2 / 27 (7.4%) |
| layer 2 | 1.0e-1 | 15 / 27 (**55.6%**) |
| logits | 5.6e-2 | — |

机理：`Gate` 用 `(scores + bias).topk(6)` 选专家，是**连续值上的离散判决**。
第 6 名和第 7 名接近时，任意微扰都会翻转选择，而两个不同专家的输出差异是 O(1)。
而且会滚雪球：layer 0 完全一致 → layer 1 7% → layer 2 过半。

实测 margin 分布（第 6 名与第 7 名的选择分数之差）：

| 层 | 翻转 token 的 margin | 未翻转 token 的 margin |
|---|---|---|
| layer 1 | max 1.5e-2, 中位 1.2e-3 | min 9.4e-5, 中位 3.7e-3 |
| layer 2 | max 6.0e-3, 中位 2.1e-3 | min 1.4e-3, 中位 6.6e-3 |

**到 layer 2，两者已经完全重叠**——margin 阈值只在最前面一两层能区分「平局」和「真 bug」。

**这直接决定了对齐协议的形状**：昇腾侧必须实现一个「强制路由」调试开关（从 golden dump
注入专家编号+权重），否则 layer 2 之后的逐层数值比对就是在测噪声。
详见 `ALIGNMENT_PROTOCOL.md` 第 0 节和第 7 节。

相比之下 `indexer_topk`（top-512）在同样扰动下 **100% 稳定**，不是风险点。

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
2. **峰值 RSS 也跟着涨到 102 GB**——这是 malloc 堆碎片累积（不是真实占用），
   但按这个斜率外推，512 token 可能摸到 200–300 GB，**会逼近 150 GB 安全线**。
   长序列 golden 落地前必须先处理这个：显式 `malloc_trim()`、
   设 `MALLOC_TRIM_THRESHOLD_`，或改用 arena 复用。已计入 §6「长序列性能」2 人天。

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
| **decode 路径**（`start_pos > 0`） | 2.5 | 把 `window_kv_cache` / `compress_kv_cache` / `k_cache` / compressor 的 `kv_state`+`score_state` 在流式层之间抽出并回填；再做 prefill↔decode 自洽验证（prefill n-1 步 + decode 1 步 应等于 prefill n 步）。这个自洽检验本身就是很强的正确性证据 |
| **强制路由注入模式** | 1.5 | golden 侧支持从外部读入专家编号/权重；协议第 7 节的必做项，两侧都要改 |
| **与 GPU 原 kernel 对照** | 1.5 | 需要一台有 GPU 的机器装 tilelang 0.1.8，把 6 个 kernel 逐个对拍。目前这一环完全缺失 |
| **与 vllm 基线交叉验证** | 1.0 | 等 vllm 服务恢复，比 logits / top-1 / 困惑度 |
| **vision 路径** | 1.5 | ViT + Aligner + `image_processor.prepare_vl_inputs`，还要准备固定图片输入 |
| **DSpark / MTP** | 2.0 | `forward_spec` 在参考实现里没有被调用过，等于要自己搭驱动循环 |
| **长序列性能** | 2.0 | 512+ token 时专家命中逼近全量。需要专家权重跨 token 复用/预取，或改成「按专家分组批处理 token」 |
| **CI 化 + 基线固化** | 1.5 | 固定输入的回归基线、阈值配置、失败时自动定位首个分歧层 |
| **文档与交接** | 0.5 | |
| **合计** | **14 人天** | 乐观 11，悲观 18 |

**建议优先级**（如果只能做一部分）：

1. **decode 路径**（2.5 人天）——最大的正确性缺口，而且 prefill↔decode 自洽检验能在
   没有 GPU/vllm 的情况下提供独立的正确性证据。
2. **强制路由注入**（1.5 人天）——没有它，逐层比对在 layer 2 之后失去信噪比。
3. **与 GPU 原 kernel 对照**（1.5 人天）——目前 kernel 正确性靠的是「读代码 + 独立朴素
   实现 + 端到端语义」，缺一个直接的权威对照。

前三项 **5.5 人天**，就能把这个 golden 从「原型，端到端看起来对」推到
「可以拿去判昇腾实现对错」。

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
