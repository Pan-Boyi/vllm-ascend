# DeepSeek-V4.1-Flash CPU 数值 golden

纯 CPU 跑通模型自带的官方参考实现，产出逐层激活 dump，用于校验昇腾 NPU 实现的精度。

- **能不能做**：能，已跑通。40 层完整 prefill 216 秒；**decode 路径也已跑通并验证**。
- **对不对**：`The capital of France is` → top-1 ` Paris`（logit 19.59，领先第二名 2.7）。
- **可复现**：两次独立的 40 层跑，940 个张量逐位相同。

### 三份给别人用的文档

| 文档 | 给谁 | 内容 |
|---|---|---|
| [ALIGNMENT_PROTOCOL.md](ALIGNMENT_PROTOCOL.md) | **model 线 + kernel 线** | 逐层对齐协议：喂什么、dump 哪些点、阈值多少。**先看第 0.4 节的判据清单**（16 条，标了门禁/诊断、跑什么、失败了看哪里）。第 10 节是单算子对照（kernel 线专用），第 11 节是 decode |
| [ROUTING_INJECTION.md](ROUTING_INJECTION.md) | **model 线** | 强制路由注入的接口规范 |
| [ENGRAM_NOTES.md](ENGRAM_NOTES.md) | **协调方 + model 线** | engram 的技术底账：G-3 判据的前提验证、两份官方实现的哈希对比、`ElasticBuffer` 为什么用不了、三个待拍板的选项 |
| [REPORT.md](REPORT.md) | 协调方 | 结论、实测数据、人天估计 |

### 🔴 先读这三条，它们决定了别的东西怎么读

1. **MoE 路由是离散判决。** 一个「严格更准」的扰动（GEMM 累加 fp32→fp64，单 GEMM 差 ~1e-7）
   在 layer 2 就让 **22.2%** 的 token 路由到不同专家。路由不一致时逐层 rel_l2 没有意义。
   ⇒ 必须用强制路由注入。（协议第 0.1 节。注意：早期流传的 7.4% / 55.6% 两个数字**是错的**，
   把纯顺序调换算成了翻转 —— 比较专家选择**必须按集合比**。）
2. **强制路由还不够。** 残差流是 bf16（相对 ULP 3.9e-3），一个 ~1e-7 的差异偶尔把某个 token
   推过进位边界，**因果 attention 再把它扩散给所有下游 token**。所以 rel_l2 的增长主要来自
   「被污染 token 比例变大」，不是算子放大噪声。⇒ 还要注入 `layer.in`。（协议第 0.2 节）
3. **阈值必须逐张量给。** bf16 的 ULP 是 3.9e-3，所以对 bf16 张量设 5e-3 以下的阈值是在测舍入。
   （协议第 6 节，附 40 层实测噪声地板表）

## 文件

| 文件 | 作用 |
|---|---|
| `kernel_cpu.py` | 6 个 tilelang kernel 的纯 PyTorch CPU 实现 |
| `test_kernel_cpu.py` | 上述 6 个函数的单测（9 组，~90 条断言，全部对独立写的朴素实现） |
| `ckpt.py` | mmap 版 safetensors 读取器，支持超大张量的**行 gather**（engram 表用） |
| `golden.py` | 流式逐层执行器 + dump 机制 + `LayerStateStore`（decode 跨步状态）|
| `run_golden.py` | prefill 入口 |
| `routing_plan.py` | 从 dump 蒸出强制路由方案（271 MB → 58 KB），并提供参考消费实现 |
| `run_decode_check.py` | **判据 G-1**：prefill↔decode 自洽验证，两个模式 |
| `engram_hash_ref.py` | **独立**的 n-gram 哈希参考，9 秒，不加载权重 |
| `engram_lookup_ref.py` | **判据 G-3**：engram 查表参考(可 import)+ 逐位相同前提的实测验证 + `--verify` 判据 |
| `compare_batch_rows.py` | batch>1 的首个分歧定位,把「真 batch bug」的特征做成可执行判据 |
| `check_docs_interfaces.py` | **文档自检**：文档承诺的每个 flag / 脚本 / 函数 / dump 目录是否真的存在（`--root` 指定扫描树，零覆盖显式失败）|
| `run_logged.sh` | `source` 它拿到 `run` —— 正确取退出码 + 全量落盘变成默认路径，不再靠「记得别用 head/tail」。`bash run_logged.sh --self-test` 有 6 个对照 |
| `compare_dumps.py` | golden ↔ NPU dump 比对工具（per-tensor 阈值、集合比路由、逐 token 统计）|
| `ref/` | `/workspace/models/.../inference/` 的**逐字副本**，只把 `kernel.py` 换成 CPU 版 |

`/workspace/models/DeepSeek-V4.1-Flash` 全程只读，没有任何写入。

## 复现

```bash
unset LD_PRELOAD      # 必须！jemalloc 在 ~/.bashrc 里，不 unset 会 segfault
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd <本目录>
DU=/workspace/code/dsv4_single_card/vllm_project/golden_dumps

# 1) kernel 单测（秒级）
python test_kernel_cpu.py

# 2) 3 层冒烟（~20s，峰值 7 GB）——覆盖 SWA / engram / 压缩KV+indexer
python run_golden.py --layers 3 --threads 16 --dump $DU/smoke3

# 3) 40 层完整 prefill（~216s，峰值 55 GB）
python run_golden.py --layers 40 --threads 16 --dump $DU/full40_v2

# 4) 语义 sanity（不 dump，~125s）
python run_golden.py --layers 40 --threads 16 --prompt "The capital of France is"

# 5) 蒸出强制路由方案（58 KB）
python routing_plan.py $DU/full40_v2 --out $DU/plan_full40_v2

# 6) 每层独立的参考跑（噪声地板表的来源；fp64 累加，~390s）
python run_golden.py --layers 40 --fp64 --threads 16 \
    --routing-plan $DU/plan_full40_v2 --force-layer-in $DU/full40_v2 \
    --dump $DU/full40_fp64_iso

# 7) 与 NPU dump 比对
python compare_dumps.py $DU/full40_v2 <npu_dump_dir> --tolerance isolated   # 阶段 A1
python compare_dumps.py $DU/full40_v2 <npu_dump_dir> --tolerance default    # 阶段 B

# 8) 判据 G-1：prefill↔decode 自洽
python run_decode_check.py --cut 20 --layers 40 \
    --routing-plan $DU/plan_full40_v2 --force-layer-in $DU/full40_v2   # 逐层
python run_decode_check.py --cut 20 --layers 40 --routing-plan $DU/plan_full40_v2  # 端到端

# 9) 环形缓冲回绕（调小滑窗，让回绕在 27 token 内发生）
python run_decode_check.py --cut 20 --layers 3 --window-size 8 \
    --routing-plan $DU/plan_win8 --force-layer-in $DU/smoke3_win8

# 10) engram 参考（都不需要跑模型）
python engram_hash_ref.py --verify $DU/full40_v2 --decode-from 20
python engram_lookup_ref.py --out $DU/engram_lookup_ref
```

产出的资产（**不在 git 里**，dump 太大）都在
`DU=/workspace/code/dsv4_single_card/vllm_project/golden_dumps`：

| 目录 | 内容 |
|---|---|
| `full40_v2` | **基准**，fp32 累加、自由路由、40 层。`full40` 是它的旧版，少一个 `indexer_kv_offset` 键，其余 940 个张量逐位相同 |
| `plan_full40_v2` | 强制路由方案，58 KB |
| `full40_fp64_iso` | 每层独立参考跑，阈值表的来源 |
| `engram_hash_ref` / `engram_lookup_ref` | engram 的两份独立参考 |
| `smoke3*` | 3 层小跑，第 0 节那组敏感度实验 |

常用参数：`--fp64`（GEMM 用 float64 累加，作数值敏感度基线）、
`--floor-gb N`（可用内存低于 N GB 时主动中止，默认 150）、
`--threads N`（默认 16；本机 CPU 是共享资源，别开满）。

## 注意

- **不需要 NPU**。不要设 `ASCEND_RT_VISIBLE_DEVICES`，不要 `import torch_npu`。
- 内存有硬保护：每层边界检查 `MemAvailable`，低于 `--floor-gb` 直接抛 `MemoryError` 退出，
  不会硬扛到把别的进程挤死。
- 权重从不全量驻留：逐层加载→算→释放；专家按 gate 实际选中的按需读；
  engram 的 203 GB 表**完全不驻留**，只从 mmap 里 gather 用到的那几千行。
- **「峰值 RSS 55 GB」这个数会被误读。** 它几乎全是 mmap 的 page cache（可回收），
  不是真实内存压力：实测 `VmRSS 33.9 GB = RssAnon 2.2 + RssFile 31.7`，
  而且峰值 RSS ≈ 已读取的 checkpoint 字节数（52.4→55.1、99.5→101.6），
  逐层单调上升从不下降 —— 每层权重明明已经释放。
  **真正随序列长度增长的是匿名内存。** 该看的是 `MemAvailable`（`MemoryGuard` 已经在看），
  它全程没低于 522 GB。所以 `--floor-gb` 是正确的保护，RSS 不是。
