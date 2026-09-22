"""P1/T1.1-T1.2: 实测 kt 侧每专家每层的真实字节数，判 191 vs 269 GiB。

方法：同一层加载两个不同的专家数，取 RSS 差 / 专家数差 —— 差值抵消固定开销。
判据：
  ① 线性（阳性对照）：三个点 (n1,n2,n3) 两两求斜率必须一致（否则测的不是权重）；
  ② 斜率必须能区分两个候选：17.93 MiB（形状推导）vs 12.76 MiB（kt 注释字面值）；
  ③ numa_maps 落点：两个 node 都要有（kt 是严格 BIND，落错就是 OOM 不是溢出）。
不占卡；只加载少量专家（<1 GiB）。
"""
import os, sys, json, subprocess

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["KT_FORCE_SYNC_SUBMIT"] = "1"
os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"

import torch
for _n in ("zeros", "empty", "full"):
    _o = getattr(torch, _n)
    def _nopin(*a, __o=_o, **kw):
        kw.pop("pin_memory", None); return __o(*a, **kw)
    setattr(torch, _n, _nopin)

import kt_kernel.experts_base as _eb
_eb._ensure_ascend_callback_worker = lambda: None
from kt_kernel.experts import KTMoEWrapper
assert "torch_npu" not in sys.modules, "torch_npu got imported; refusing to run"

CKPT = "/workspace/models/DeepSeek-V4.1-Flash"
HIDDEN, INTER, N_EXP, TOPK, LAYER = 5120, 2304, 384, 6, 3
MIB = 2 ** 20

def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("VmRSS not found")

def mem_avail_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError

def numa_bytes():
    """逐 node 的匿名页字节，从 /proc/self/numa_maps 聚合。"""
    tot = {}
    pagesz = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/numa_maps") as f:
        for line in f:
            for tok in line.split():
                if tok.startswith("N") and "=" in tok and tok[1:].split("=")[0].isdigit():
                    n, c = tok[1:].split("=")
                    tot[int(n)] = tot.get(int(n), 0) + int(c) * pagesz
    return tot

if mem_avail_kb() < 150e6:
    raise SystemExit("MemAvailable < 150 GB, refusing to start")

points = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [8, 24, 48]
print(f"[cfg] layer={LAYER} hidden={HIDDEN} inter={INTER} points={points}", flush=True)
print(f"[cfg] compact_mxfp4_scales default = {os.environ.get('KT_MXFP4_COMPACT_SCALES','1 (default on)')}", flush=True)

res = []
for n_cpu in points:
    mask = torch.ones(N_EXP, dtype=torch.bool)   # True = 在 NPU 上，kt 跳过
    mask[:n_cpu] = False                          # False = 下放到 CPU，kt 要为它建 BufferB
    r0, nm0 = rss_kb(), numa_bytes()
    wr = KTMoEWrapper(
        layer_idx=LAYER, num_experts=N_EXP, num_experts_per_tok=TOPK, hidden_size=HIDDEN,
        moe_intermediate_size=INTER, gpu_experts_mask=mask,
        cpuinfer_threads=8, threadpool_count=2, weight_path=CKPT,
        chunked_prefill_size=64, method="MXFP4", numa_nodes=[0, 1], swiglu_limit=10.0,
    )
    wr.load_weights(torch.arange(N_EXP, dtype=torch.int64))
    r1, nm1 = rss_kb(), numa_bytes()
    d = (r1 - r0) * 1024
    res.append((n_cpu, d, {k: nm1.get(k, 0) - nm0.get(k, 0) for k in set(nm0) | set(nm1)}))
    print(f"[load] n_cpu={n_cpu:3d}  ΔRSS={d/MIB:9.1f} MiB  ({d/n_cpu/MIB:6.3f} MiB/专家)  "
          f"Δnuma={ {k: round(v/MIB,1) for k,v in res[-1][2].items()} }", flush=True)
    del wr
    import gc; gc.collect()

print("\n=== 斜率（相邻两点求差，抵消固定开销）===")
slopes = []
for (n1, d1, _), (n2, d2, _) in zip(res, res[1:]):
    s = (d2 - d1) / (n2 - n1) / MIB
    slopes.append(s)
    print(f"  n={n1}→{n2}: {s:.3f} MiB/专家/层")

cand = {"形状推导 3*H*I*0.53125": 3 * HIDDEN * INTER * 0.53125 / MIB,
        "kt 注释字面值 (DSV4 维度)": 12.76,
        "非 compact (0.625 B/elem)": 3 * HIDDEN * INTER * 0.625 / MIB}
print("\n=== 候选 ===")
for k, v in cand.items():
    print(f"  {k:32s} = {v:7.3f} MiB/专家/层")
print(json.dumps({"points": [(n, d) for n, d, _ in res], "slopes_mib": slopes}, indent=1))
