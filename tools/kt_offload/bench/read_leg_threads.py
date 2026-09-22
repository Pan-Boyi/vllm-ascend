"""P3-a: 并行化 mmap(safetensors) -> staging 的拷贝，量线程数曲线。

P2 测到：单线程这条腿只有 17.7-21.8 GiB/s，比 H2D(52.6) 还慢 2.4-3 倍 ⇒ 它才是当前瓶颈。
参考实现用 KT_MXFP4_COPY_THREADS=32 并行同一段（kt_stream_prefill.py:204-235）。
这里只量【主机侧】曲线（0 卡）：pinned 分配需要设备上下文，卡上那一发另做。
阳性对照：线程数从 1 加到 N 若【完全不涨】，说明测的不是拷贝而是别的（页错误/盘）。
"""
import os, sys, time, json, statistics as st
from concurrent.futures import ThreadPoolExecutor
import torch
from safetensors import safe_open

CKPT = "/workspace/models/DeepSeek-V4.1-Flash"
LAYER, NEXP = 20, 64
idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
handles = {}
def h(fn):
    if fn not in handles:
        handles[fn] = safe_open(os.path.join(CKPT, fn), framework="pt")
    return handles[fn]

keys = []
for e in range(NEXP):
    for p in ("w1", "w3", "w2"):
        keys.append(f"layers.{LAYER}.ffn.experts.{e}.{p}.weight")
srcs = [h(idx[k]).get_tensor(k).contiguous().view(torch.uint8) for k in keys]   # mmap 视图
nbytes = sum(t.numel() for t in srcs)
dst = [torch.empty_like(t) for t in srcs]
print(f"[cfg] layer={LAYER} experts={NEXP} tensors={len(srcs)} bytes={nbytes/2**30:.3f} GiB", flush=True)
print(f"[cfg] nproc_reports={os.popen('nproc').read().strip()} cpus_online_getconf={os.sysconf('SC_NPROCESSORS_ONLN')}", flush=True)

def run(nthreads):
    def work(i):
        dst[i].copy_(srcs[i])
    if nthreads == 1:
        t = time.perf_counter()
        for i in range(len(srcs)):
            work(i)
        return time.perf_counter() - t
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        t = time.perf_counter()
        list(ex.map(work, range(len(srcs))))
        return time.perf_counter() - t

rows = []
for n in (1, 2, 4, 8, 16, 32, 64, 96):
    ts = [run(n) for _ in range(5)][1:]          # 丢第一轮（页缓存预热）
    med = st.median(ts)
    rows.append((n, med, nbytes / med / 2**30))
    print(f"  threads {n:3d}: median {med*1e3:7.1f} ms  => {nbytes/med/2**30:6.1f} GiB/s "
          f"({nbytes/med/1e9:6.1f} GB/s)", flush=True)
best = max(rows, key=lambda r: r[2])
one = rows[0][2]
print(f"\n最快 {best[0]} 线程 = {best[2]:.1f} GiB/s，相对单线程 {best[2]/one:.2f}×")
print("阳性对照: " + ("✅ 线程数确实带来加速" if best[2] > one * 1.5 else "❌ 加不动 —— 测的可能不是拷贝"))
