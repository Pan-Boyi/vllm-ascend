# SPDX-License-Identifier: Apache-2.0
"""P2/H2D: how fast can a chunk of expert weights get from host DDR onto the card?

Why this is the next number. The streaming-prefill design (STREAM_PREFILL_PLAN.md
§5) feeds the MoE of a long-prefill layer from a device-side slot that is filled
from host memory, and the dynamic-hot-expert design writes the resident weights
from the same trip. Both share this one leg, and it has NO reading at all: the
only transfer number we have is the DEVICE-INTERNAL torch_npu.copy_memory_
(1.751 GiB in 2.6 ms = 669 GiB/s, E2w C2). A device-internal copy says nothing
about PCIe/HCCS H2D, so the whole design currently rests on an unmeasured leg.

What is measured, per chunk size (experts per chunk):
  R1 pinned host -> device, blocking      (the production path)
  R2 pageable host -> device, blocking    (POSITIVE CONTROL, see below)
  R3 pinned host -> device, non_blocking  (submit cost vs completion cost)
  R4 mmap(safetensors) -> pinned host     (the read leg that must precede R1;
                                           page-cache-warm here, cold is R-E10)

🔴 Positive control (R2). A copy that is really crossing the host/device boundary
gets measurably slower from PAGEABLE memory, because the driver has to bounce it
through an internal pinned staging buffer. If pageable is NOT slower, the number
on screen is not an H2D bandwidth -- it is a cache, a lazy allocation, or a
container artefact -- and the run is reported FAIL rather than as a reading.
This is the same discipline E2' had to learn the hard way: a measurement whose
value would look identical whether or not the thing under test happened has zero
discriminating power, however plausible the number is.

🔴 Interleaving. This machine drifts 2.5x on its own (PLAN §4.15), so the arms
are NOT run one after another: every round runs every arm, and the spread across
rounds of the SAME arm is reported as the noise floor next to the difference
between arms. A difference smaller than that floor is not a difference.

Units are printed twice (GB/s decimal and GiB/s binary) because the CPU-side
curve this has to be compared against (428 GB/s, PLAN §4.3) is decimal; a silent
unit switch is a 7.4% error, which is the same size as some of the effects here.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e_common as E  # noqa: E402
import oncard_common as C  # noqa: E402

NAMES = ("w13", "w13_scale", "w2", "w2_scale")
# 两个总体（本发实测）：H2D pinned 无停顿底 52.6 GiB/s，设备内 D2D 656-662 GiB/s
# （E2w C2 当时读到 669）。差 12.5 倍 ⇒ 判据阈值放在两者【中间】。
# 🔴 第一版把阈值贴着 D2D 写成 669，变异臂 M1（源在设备上）实测 656 落在线下 ⇒ 判据对它
#    零区分力、当场变绿。这就是"护栏不能和被判对象共享误判特征"。
# 阈值不是为通过而调：60..600 之间任何取值都给出同一判决，余量两边各 3-4 倍。
D2D_GIBS = 660.0
H2D_MAX_GIBS = 200.0


def pcts(xs):
    s = sorted(xs)
    n = len(s)
    return (st.median(s), s[max(0, int(0.10 * (n - 1)))], s[min(n - 1, int(0.90 * (n - 1)))])


def rate(nbytes, secs):
    return nbytes / secs / 1e9, nbytes / secs / 2**30      # GB/s (decimal), GiB/s (binary)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--chunks", default="32,64", help="experts per chunk")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--arms", default="R1,R2,R3",
                    help="which arms to run; isolating an arm tests whether the arms "
                         "perturb each other (the 2026-09-22 three-level spread)")
    ap.add_argument("--settle-ms", type=float, default=0.0,
                    help="idle gap after each arm, to let the driver quiesce")
    ap.add_argument("--mem-floor-gib", type=float, default=100.0)
    ap.add_argument("--mem-budget-gib", type=float, default=12.0)
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()

    chunks = [int(x) for x in args.chunks.split(",")]
    mg = E.MemGuard(args.mem_floor_gib, args.mem_budget_gib)
    import npu_moe as N

    rt = N.init_runtime()
    dev = rt.device
    v = C.Verdict("P2 H2D bandwidth for streaming-prefill chunks")
    v.note(E.BUILD_NOTE)
    but = E.build_under_test(args.out)
    v.note(f"build under test: so={but.get('so_primary_md5')}")

    # The real thread count, next to the lying one (PLAN §4.15 #2: nproc is 1 here).
    try:
        nproc = int(os.popen("nproc").read().strip())
    except Exception:
        nproc = -1
    online = os.sysconf("SC_NPROCESSORS_ONLN")
    v.note(f"nproc_reports={nproc} cpus_online_getconf={online} "
           f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"[cfg] layer={args.layer} chunks={chunks} rounds={args.rounds} device={dev}", flush=True)

    ck = C.Ckpt()
    facts: dict = {"build_under_test": but, "layer": args.layer, "rounds": args.rounds,
                   "nproc_reports": nproc, "cpus_online_getconf": online, "arms": {}}

    biggest = max(chunks)
    ids = list(range(biggest))
    t0 = time.time()
    raw = N.load_raw_layer(ck, args.layer, ids)          # pageable host uint8, from mmap
    facts["read_all_secs"] = time.time() - t0
    mg.check("after reading the experts from the checkpoint")

    hbm0 = torch.npu.memory_allocated()
    for cexp in chunks:
        src_page = tuple(t[:cexp].contiguous() for t in raw)
        nbytes = sum(t.numel() for t in src_page)
        src_pin = tuple(torch.empty_like(t, pin_memory=True) for t in src_page)
        for a, b in zip(src_pin, src_page):
            a.copy_(b)
        dst = tuple(torch.empty(t.shape, dtype=t.dtype, device=dev) for t in src_page)
        torch.npu.synchronize()
        print(f"\n[chunk] {cexp} experts = {nbytes / 2**30:.3f} GiB ({nbytes / 1e9:.3f} GB) "
              f"per H2D; device dst allocated {torch.npu.memory_allocated() / 2**30:.2f} GiB", flush=True)
        mg.check(f"after allocating chunk {cexp}")

        def one(src, non_blocking):
            torch.npu.synchronize()
            t = time.perf_counter()
            for d, s in zip(dst, src):
                d.copy_(s, non_blocking=non_blocking)
            submit = time.perf_counter() - t
            torch.npu.synchronize()
            return time.perf_counter() - t, submit

        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        arms = {k: v for k, v in (("R1_pinned_blocking", (src_pin, False)),
                                  ("R2_pageable_blocking", (src_page, False)),
                                  ("R3_pinned_nonblocking", (src_pin, True)))
                if k.split("_")[0] in want}
        if not arms:
            raise SystemExit(f"--arms {args.arms} selected nothing")
        acc = {k: {"wall": [], "submit": []} for k in arms}
        for r in range(args.warmup + args.rounds):          # INTERLEAVED: every round, every arm
            row = []
            for name, (src, nb) in arms.items():
                w, sm = one(src, nb)
                row.append(f"{name.split('_')[0]}={w * 1e3:7.2f}")
                if args.settle_ms:
                    time.sleep(args.settle_ms / 1e3)
                if r >= args.warmup:
                    acc[name]["wall"].append(w)
                    acc[name]["submit"].append(sm)
            print(f"    round {r:3d}{' (warmup)' if r < args.warmup else ''}: " + "  ".join(row), flush=True)
        # R4: the read leg that must precede R1 (page-cache warm).
        r4 = []
        for r in range(args.warmup + args.rounds):
            t = time.perf_counter()
            for a, b in zip(src_pin, src_page):
                a.copy_(b)
            if r >= args.warmup:
                r4.append(time.perf_counter() - t)
        acc["R4_mmap_to_pinned"] = {"wall": r4, "submit": r4}

        res = {}
        for name, d in acc.items():
            med, p10, p90 = pcts(d["wall"])
            gbs, gibs = rate(nbytes, med)
            res[name] = {"series_ms": [round(x * 1e3, 3) for x in d["wall"]],
                         "median_ms": med * 1e3, "p10_ms": p10 * 1e3, "p90_ms": p90 * 1e3,
                         "GB_s": gbs, "GiB_s": gibs,
                         "submit_median_ms": st.median(d["submit"]) * 1e3,
                         "spread_pct": (p90 - p10) / med * 100.0}
            print(f"  {name:24s} median {med * 1e3:7.2f} ms  p10 {p10 * 1e3:7.2f}  p90 {p90 * 1e3:7.2f}  "
                  f"=> {gbs:7.1f} GB/s ({gibs:6.1f} GiB/s)  submit {st.median(d['submit']) * 1e3:6.2f} ms",
                  flush=True)
        facts["arms"][str(cexp)] = {"bytes": nbytes, **res}

        # ---- positive control ------------------------------------------------
        if "R2_pageable_blocking" not in res or "R1_pinned_blocking" not in res:
            v.note(f"chunk {cexp}: pinned-vs-pageable positive control NOT COVERED this run "
                   f"(arms={args.arms}) -- absence of the check is not a pass")
            facts["arms"][str(cexp)]["positive_control"] = "not covered"
            del src_pin, src_page, dst
            torch.npu.empty_cache()
            continue
        # 🔴 判据在第一次红之后改过（2026-09-22），理由与原判据的读数都留在这里：
        #   原判据 = 比两臂【中位】，地板 = 同臂 p10..p90 散布。它在 chunk 32 上红了
        #   （差 138.7% < 地板 153.8%）。查下来散布不是随机噪声，而是一个【与字节数无关】
        #   的外部周期性停顿（+34..40 ms，0.070 GiB 与 0.560 GiB 上同样大小，见 §16），
        #   它按"占用通道的时长"概率性地命中两个臂 ⇒ 中位与 p90 都被它污染，
        #   而【p10 是无停顿底】。所以区分 pinned/pageable 要比两臂的 p10。
        #   常数不是为了通过而调：pinned/pageable 的 p10 差 5 倍以上，取 5% 或 50% 都同样判。
        #   两个判据的读数都打印，红的那个不隐藏。
        pin_med, page_med = res["R1_pinned_blocking"]["median_ms"], res["R2_pageable_blocking"]["median_ms"]
        floor = max(res["R1_pinned_blocking"]["spread_pct"], res["R2_pageable_blocking"]["spread_pct"])
        gain_med = (page_med - pin_med) / pin_med * 100.0
        v.note(f"chunk {cexp}: [原判据，仅记录] 中位 pageable {page_med:.2f} vs pinned {pin_med:.2f} "
               f"= {gain_med:+.1f}%, 同臂散布地板 {floor:.1f}% -> "
               f"{'过' if gain_med > floor else '红（被外部停顿污染，见 §16）'}")
        pin_ms, page_ms = res["R1_pinned_blocking"]["p10_ms"], res["R2_pageable_blocking"]["p10_ms"]
        gain = (page_ms - pin_ms) / pin_ms * 100.0
        v.check(gain > 5.0,
                f"chunk {cexp}: PAGEABLE 的无停顿底必须明显慢于 PINNED 的无停顿底 "
                f"(p10 pageable {page_ms:.2f} ms vs pinned {pin_ms:.2f} ms = {gain:+.1f}%, 要求 >5%)")
        r1_gibs = nbytes / (res["R1_pinned_blocking"]["p10_ms"] / 1e3) / 2**30
        v.check(r1_gibs < H2D_MAX_GIBS,
                f"chunk {cexp}: H2D 必须明显慢于设备内拷贝（本发 D2D 实测 ≈{D2D_GIBS:.0f} GiB/s，阈值 {H2D_MAX_GIBS:.0f}）—— "
                f"不慢就说明这份数据【根本没离开主机】(实测 {r1_gibs:.1f} GiB/s)")

        # ---- 变异臂：让上面那条必检【必须】变红 --------------------------------
        # M1 源张量放在设备上（伪装成 H2D 的 D2D）；M2 干脆不拷贝。
        # 两条都走【同一个】判据函数，证明它真的能抓到"没离开主机"这件事。
        for mname, msrc in (("M1 源在设备上（其实是 D2D）", tuple(d.clone() for d in dst)),
                            ("M2 根本不拷贝（空循环）", None)):
            ts = []
            for _ in range(8):
                torch.npu.synchronize()
                t = time.perf_counter()
                if msrc is not None:
                    for d, sr in zip(dst, msrc):
                        d.copy_(sr, non_blocking=False)
                torch.npu.synchronize()
                ts.append(time.perf_counter() - t)
            mg_gibs = nbytes / min(ts) / 2**30
            v.must_be_red(not (mg_gibs < H2D_MAX_GIBS),
                          f"chunk {cexp}: {mname} -> {mg_gibs:.0f} GiB/s，必检应判其【没离开主机】")
            del msrc
        clean = res["R1_pinned_blocking"]["bytes_gib"] = nbytes / 2**30
        p10s = res["R1_pinned_blocking"]["p10_ms"] / 1e3
        v.note(f"chunk {cexp}: 无停顿底 H2D = {nbytes / p10s / 2**30:.1f} GiB/s "
               f"({nbytes / p10s / 1e9:.1f} GB/s)；中位含停顿 {nbytes / (res['R1_pinned_blocking']['median_ms'] / 1e3) / 2**30:.1f} GiB/s")
        del src_pin, src_page, dst
        torch.npu.empty_cache()

    facts["hbm_baseline_gib"] = hbm0 / 2**30
    with open(os.path.join(args.out, "h2d_result.json"), "w") as f:
        json.dump(facts, f, indent=1)
    return v.finish()


if __name__ == "__main__":
    raise SystemExit(main())
