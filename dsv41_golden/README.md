# DeepSeek-V4.1-Flash CPU 数值 golden

纯 CPU 跑通模型自带的官方参考实现，产出逐层激活 dump，用于校验昇腾 NPU 实现的精度。

- **能不能做**：能，已跑通。40 层完整 prefill，216 秒，峰值 55 GB 内存。
- **对不对**：`The capital of France is` → top-1 ` Paris`（logit 19.59，领先第二名 2.7）。
- 结论、实测数据、人天估计见 [REPORT.md](REPORT.md)
- 给昇腾侧的对齐协议见 [ALIGNMENT_PROTOCOL.md](ALIGNMENT_PROTOCOL.md)

## 文件

| 文件 | 作用 |
|---|---|
| `kernel_cpu.py` | 6 个 tilelang kernel 的纯 PyTorch CPU 实现 |
| `test_kernel_cpu.py` | 上述 6 个函数的单测（9 组，~90 条断言，全部对独立写的朴素实现） |
| `ckpt.py` | mmap 版 safetensors 读取器，支持超大张量的**行 gather**（engram 表用） |
| `golden.py` | 流式逐层执行器 + dump 机制 |
| `run_golden.py` | 命令行入口 |
| `compare_dumps.py` | golden ↔ NPU dump 比对工具 |
| `ref/` | `/workspace/models/.../inference/` 的**逐字副本**，只把 `kernel.py` 换成 CPU 版 |

`/workspace/models/DeepSeek-V4.1-Flash` 全程只读，没有任何写入。

## 复现

```bash
unset LD_PRELOAD      # 必须！jemalloc 在 ~/.bashrc 里，不 unset 会 segfault
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
cd <本目录>

# 1) kernel 单测（秒级）
python test_kernel_cpu.py

# 2) 3 层冒烟（~20s，峰值 7 GB）——覆盖 SWA / engram / 压缩KV+indexer
python run_golden.py --layers 3 --threads 16 --dump dumps/smoke3

# 3) 40 层完整 prefill（~216s，峰值 55 GB）
python run_golden.py --layers 40 --threads 16 --dump dumps/full40

# 4) 语义 sanity（不 dump，~125s）
python run_golden.py --layers 40 --threads 16 --prompt "The capital of France is"

# 5) 与 NPU dump 比对
python compare_dumps.py dumps/full40 <npu_dump_dir>
```

常用参数：`--fp64`（GEMM 用 float64 累加，作数值敏感度基线）、
`--floor-gb N`（可用内存低于 N GB 时主动中止，默认 150）、
`--threads N`（默认 16；本机 CPU 是共享资源，别开满）。

## 注意

- **不需要 NPU**。不要设 `ASCEND_RT_VISIBLE_DEVICES`，不要 `import torch_npu`。
- 内存有硬保护：每层边界检查 `MemAvailable`，低于 `--floor-gb` 直接抛 `MemoryError` 退出，
  不会硬扛到把别的进程挤死。
- 权重从不全量驻留：逐层加载→算→释放；专家按 gate 实际选中的按需读；
  engram 的 203 GB 表**完全不驻留**，只从 mmap 里 gather 用到的那几千行。
