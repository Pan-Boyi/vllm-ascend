#!/usr/bin/env python3
"""DSV4.1 MoE offload: pure-CPU timing of the kt-kernel MXFP4 expert kernel.

Answers "what is the CPU-side floor for single-card bs=1?" by measuring one
layer's ``submit + sync`` wall clock as a function of ``k_cpu`` (how many of a
token's top-6 experts live on the CPU) and of ``T`` (tokens per forward).

Never opens an NPU: ``ASCEND_RT_VISIBLE_DEVICES=""`` and
``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` are set before torch is imported, and the
run aborts if ``torch_npu`` ends up in ``sys.modules``.  Takes no card lease
because it touches no card.

Cache-reuse guard: the point of the measurement is DRAM streaming, so every
repetition draws a *fresh* random expert subset and rotates over all loaded
layers.  With ``--n-cpu 284`` and 3 layers the resident weight set is ~14.6 GiB
against 512 MiB of L3, so a repetition essentially never re-reads a cached
expert.  The reported implied bandwidth is the check bit for this (see
``--report`` output: implied GB/s far above the socket ceiling would mean the
timer is measuring cache hits, not the real load).

Usage (one process per *pool* configuration -- CPUInfer is a process-level
singleton, so cpuinfer_threads / threadpool_count / numa_nodes / KT_CPU_BIND
cannot be changed in-process):

  env -u LD_PRELOAD PYTHONPATH=<kt_site> python3 kt_cpu_bench.py \
      --layers 1,3,5 --n-cpu 284 --threads 64 --threadpool 2 --numa 0,1 \
      --kcpu 0,1,2,3,4,5,6 --tokens 1,4,32 --reps 60 --out run.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

# ---- NPU lockout: must happen before torch is imported ----------------------
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["KT_FORCE_SYNC_SUBMIT"] = "1"
os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402

# kt allocates its pinned buffers with pin_memory=True; a CPU-only torch build
# has no pinned allocator.  Pinning is meaningless without D2H copies.
for _name in ("zeros", "empty", "full"):
    _orig = getattr(torch, _name)

    def _nopin(*a, __orig=_orig, **kw):
        kw.pop("pin_memory", None)
        return __orig(*a, **kw)

    setattr(torch, _name, _nopin)

import kt_kernel  # noqa: E402
import kt_kernel.experts_base as _eb  # noqa: E402

_eb._ensure_ascend_callback_worker = lambda: None
from kt_kernel.experts import KTMoEWrapper  # noqa: E402
from kt_kernel.utils.amx import _select_mxfp4_backend  # noqa: E402

assert "torch_npu" not in sys.modules, "torch_npu got imported; refusing to run"

CKPT = "/workspace/models/DeepSeek-V4.1-Flash"
HIDDEN, INTER, N_EXP, TOPK = 5120, 2304, 384, 6
LIMIT, ROUTE_SCALE = 10.0, 1.5

# One expert, one layer, compact E8M0 scales: w1 + w3 + w2, each n*k/2 nibble
# bytes + n*k/32 scale bytes.  This is the byte count the kernel streams per
# routed CPU expert, and the denominator of the implied-bandwidth check bit.
BYTES_PER_EXPERT_LAYER = 3 * (HIDDEN * INTER // 2 + HIDDEN * INTER // 32)  # 18_800_640

MEM_FLOOR_KB = 100 * 1000 * 1000  # self-stop below ~100 GB MemAvailable
MEM_START_KB = 150 * 1000 * 1000  # refuse to start below ~150 GB


def mem_available_kb() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable not found in /proc/meminfo")


def machine_load() -> dict:
    with open("/proc/loadavg") as f:
        la = f.read().split()
    try:
        top = subprocess.run(
            ["ps", "-eo", "pid,comm,pcpu,rss", "--sort=-pcpu"],
            capture_output=True, text=True, timeout=20,
        )
        busiest = list(top.stdout.splitlines()[1:11])
    except Exception as exc:  # noqa: BLE001
        busiest = [f"<ps failed: {exc}>"]
    def _sh(*cmd) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    online = _sh("getconf", "_NPROCESSORS_ONLN")
    return {
        "loadavg": la[:3],
        # nproc returns min(affinity, OMP_NUM_THREADS) and OMP_NUM_THREADS=1 in
        # this container: it reports 1 on a 256-core machine. Record the true
        # count and the misleading one together, or a later reader will take
        # "1 CPU" for a finding.
        "cpus_online_getconf": int(online) if online.isdigit() else None,
        "nproc_reports": int(_sh("nproc") or 0) or None,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "mem_available_kb": mem_available_kb(),
        "top10_by_cpu": busiest,
        "wallclock": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def check_mem(stage: str, floor_kb: int) -> None:
    kb = mem_available_kb()
    if kb < floor_kb:
        raise SystemExit(
            f"[gate] {stage}: MemAvailable {kb / 1e6:.1f} GB < floor {floor_kb / 1e6:.1f} GB -- stopping"
        )


def _loadavg_now() -> str:
    with open("/proc/loadavg") as f:
        return f.read().split()[0]


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def make_batch(rng, T, k_cpu, cpu_pool, gpu_pool):
    """Routing with exactly ``k_cpu`` CPU experts per token; the rest are GPU
    experts, which kt skips (gpu_experts_mask)."""
    ids = np.empty((T, TOPK), dtype=np.int64)
    for t in range(T):
        c = rng.choice(cpu_pool, k_cpu, replace=False) if k_cpu else np.empty(0, np.int64)
        g = rng.choice(gpu_pool, TOPK - k_cpu, replace=False) if k_cpu < TOPK else np.empty(0, np.int64)
        row = np.concatenate([c, g]).astype(np.int64)
        rng.shuffle(row)
        ids[t] = row
    logits = torch.from_numpy(rng.standard_normal((T, TOPK)).astype(np.float32))
    s = torch.nn.functional.softplus(logits).sqrt()
    w = (s / (s.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE).contiguous()
    return torch.from_numpy(ids), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="1,3,5")
    ap.add_argument("--n-cpu", type=int, default=284, help="CPU-resident experts per layer")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--threadpool", type=int, default=2)
    ap.add_argument("--numa", default="0,1", help="explicit NUMA node list, len == --threadpool")
    ap.add_argument("--kcpu", default="0,1,2,3,4,5,6")
    ap.add_argument("--tokens", default="1")
    ap.add_argument("--reps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--dump-out", default="", help="npy path: dump the T=1,k=4 output for a fixed "
                                                   "seed (bit-exactness A/B across processes)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    layers = [int(v) for v in args.layers.split(",")]
    numa = [int(v) for v in args.numa.split(",")] if args.numa else None
    kcpus = [int(v) for v in args.kcpu.split(",")]
    tokens = [int(v) for v in args.tokens.split(",")]

    check_mem("startup", MEM_START_KB)
    load_before = machine_load()

    cfg = dict(
        tag=args.tag, layers=layers, n_cpu=args.n_cpu, threads=args.threads,
        threadpool=args.threadpool, numa=numa, kcpu=kcpus, tokens=tokens,
        reps=args.reps, warmup=args.warmup, seed=args.seed,
        kt_version=kt_kernel.__version__, kt_file=kt_kernel.__file__,
        cpu_variant=kt_kernel.__cpu_variant__,
        backend=_select_mxfp4_backend().__name__,
        env={k: os.environ.get(k, "<unset>") for k in
             ("KT_CPU_BIND", "KT_MXFP4_COMPACT_SCALES", "KT_MOE_PHASE_TIMING",
              "OMP_NUM_THREADS", "KT_FORCE_SYNC_SUBMIT")},
        bytes_per_expert_layer=BYTES_PER_EXPERT_LAYER,
        host=platform.node(),
    )
    print(json.dumps({"config": cfg}, indent=1), flush=True)

    rng = np.random.default_rng(args.seed)
    # CPU-resident set: a contiguous prefix keeps it reproducible across
    # processes (the sweep compares configurations, not expert subsets).
    cpu_pool = np.arange(args.n_cpu, dtype=np.int64)
    gpu_pool = np.arange(args.n_cpu, N_EXP, dtype=np.int64)
    if len(gpu_pool) < TOPK:
        raise SystemExit(f"--n-cpu {args.n_cpu} leaves only {len(gpu_pool)} GPU experts; need >= {TOPK}")
    mask = torch.ones(N_EXP, dtype=torch.bool)   # True = on GPU, kt skips it
    mask[: args.n_cpu] = False

    wrappers = {}
    t_load0 = time.time()
    for L in layers:
        check_mem(f"before loading layer {L}", MEM_FLOOR_KB)
        t0 = time.time()
        wr = KTMoEWrapper(
            layer_idx=L, num_experts=N_EXP, num_experts_per_tok=TOPK, hidden_size=HIDDEN,
            moe_intermediate_size=INTER, gpu_experts_mask=mask,
            cpuinfer_threads=args.threads, threadpool_count=args.threadpool,
            weight_path=CKPT, chunked_prefill_size=max(64, max(tokens)), method="MXFP4",
            numa_nodes=numa, swiglu_limit=LIMIT,
        )
        wr.load_weights(torch.arange(N_EXP, dtype=torch.int64))
        wrappers[L] = wr
        print(f"[load] layer {L}: {args.n_cpu} CPU experts in {time.time() - t0:.1f}s, "
              f"MemAvailable {mem_available_kb() / 1e6:.1f} GB", flush=True)
    load_secs = time.time() - t_load0
    resident_gib = len(layers) * args.n_cpu * BYTES_PER_EXPERT_LAYER / 2**30
    print(f"[load] total {load_secs:.1f}s; nominal resident weights "
          f"{resident_gib:.2f} GiB over {len(layers)} layers", flush=True)

    results = []
    for T in tokens:
        for k in kcpus:
            # Pre-build every repetition's inputs so the timed region contains
            # only kt work.  Fresh expert draw per rep + layer rotation = no L3
            # reuse between reps.
            n_all = args.warmup + args.reps
            batches = []
            for i in range(n_all):
                x = (torch.from_numpy(rng.standard_normal((T, HIDDEN)).astype(np.float32))).to(torch.bfloat16)
                ids, w = make_batch(rng, T, k, cpu_pool, gpu_pool)
                batches.append((layers[i % len(layers)], x, ids, w))

            # How many *distinct* CPU experts a call streams: that, not k*T, is
            # the weight traffic (kt gathers a batch's tokens per expert).
            distinct = [int(np.unique(b[2].numpy()[b[2].numpy() < args.n_cpu]).size)
                        for b in batches[args.warmup:]]

            comp, prep = [], []
            for i, (L, x, ids, w) in enumerate(batches):
                wr = wrappers[L]
                ta = time.perf_counter_ns()
                wr.copy_inputs_to_cpu_buffers(x, ids, w)
                tb = time.perf_counter_ns()
                wr.forward_on_pinned_buffers(x, 0)
                wr.cpu_infer.sync(0)
                tc = time.perf_counter_ns()
                if i >= args.warmup:
                    prep.append((tb - ta) / 1e3)
                    comp.append((tc - tb) / 1e3)
            check_mem(f"after T={T} k={k}", MEM_FLOOR_KB)

            med = pct(comp, 50)
            n_distinct = float(np.median(distinct)) if distinct else 0.0
            distinct_bytes = n_distinct * BYTES_PER_EXPERT_LAYER
            row = dict(
                T=T, k_cpu=k, n=len(comp),
                p10=pct(comp, 10), p50=med, p90=pct(comp, 90),
                mean=float(np.mean(comp)), std=float(np.std(comp)),
                prep_p50=pct(prep, 50),
                distinct_experts_p50=n_distinct,
                implied_gbs=(distinct_bytes / (med * 1e-6) / 1e9) if med > 0 and k else 0.0,
                mem_available_kb=mem_available_kb(),
                loadavg=_loadavg_now(),
            )
            results.append(row)
            print(f"T={T:<3d} k_cpu={k}  p50={row['p50']:8.1f} us  "
                  f"[p10 {row['p10']:8.1f} / p90 {row['p90']:8.1f}]  n={row['n']}  "
                  f"prep={row['prep_p50']:6.1f} us  distinct_E={n_distinct:5.1f}  "
                  f"implied BW {row['implied_gbs']:6.1f} GB/s",
                  flush=True)

    dump = None
    if args.dump_out:
        # Deterministic A/B artifact: fixed seed, fixed routing, first layer.
        r2 = np.random.default_rng(12345)
        x = (torch.from_numpy(r2.standard_normal((1, HIDDEN)).astype(np.float32))).to(torch.bfloat16)
        ids, w = make_batch(r2, 1, 4, cpu_pool, gpu_pool)
        wr = wrappers[layers[0]]
        wr.copy_inputs_to_cpu_buffers(x, ids, w)
        wr.forward_on_pinned_buffers(x, 0)
        wr.cpu_infer.sync(0)
        out = wr.copy_forward_output_to_device(x).clone()
        raw = out.contiguous().view(torch.uint8).numpy().tobytes()
        np.save(args.dump_out, out.view(torch.uint8).numpy())
        dump = dict(path=args.dump_out, sha256=hashlib.sha256(raw).hexdigest(),
                    nbytes=len(raw), ids=ids.tolist())
        print(f"[dump] {args.dump_out} sha256={dump['sha256']}", flush=True)

    out_doc = dict(config=cfg, load_secs=load_secs, resident_gib=resident_gib,
                   load_before=load_before, load_after=machine_load(),
                   results=results, dump=dump,
                   torch_npu_imported="torch_npu" in sys.modules)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out_doc, f, indent=1)
        print(f"[out] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
