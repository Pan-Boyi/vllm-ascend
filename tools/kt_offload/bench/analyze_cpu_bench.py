#!/usr/bin/env python3
"""Turn kt_cpu_bench.py runs into the T(k) curve, the bs=1 extrapolation and
the check bits.  Pure arithmetic on the measured JSON -- no new measurement.

  python3 analyze_cpu_bench.py <outdir>
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

N_LAYERS = 40          # DSV4.1-Flash routed-MoE layers (PLAN §1)
N_EXP, TOPK = 384, 6
BYTES_PER_EXPERT_LAYER = 18_800_640
TPOT_MULTICARD_MS = 26.84  # 4-card graph-mode bs1 TPOT, build_artifacts/m1_run/gsm8k_flight4_final_101009/
# sglang bw.c on this same host (docs_cpu_optimization.md §4.1 / §5.2):
BW_1CCD, BW_SOCKET, BW_DUAL = 76.0, 246.0, 495.0  # GB/s


def load(outdir):
    runs = {}
    for p in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        with open(p) as f:
            runs[os.path.basename(p)[:-5]] = json.load(f)
    return runs


def curve_from(runs, tags, T=1):
    """Median-of-passes T(k) in microseconds."""
    acc = {}
    for tag in tags:
        for r in runs[tag]["results"]:
            if r["T"] == T:
                acc.setdefault(r["k_cpu"], []).append(r["p50"])
    return {k: sorted(v)[len(v) // 2] for k, v in sorted(acc.items())}, acc


def lsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b  # intercept, slope


def hypergeom(K, k):
    """P(k of the top-6 drawn without replacement land on the CPU), K = #CPU experts."""
    return (math.comb(K, k) * math.comb(N_EXP - K, TOPK - k)) / math.comb(N_EXP, TOPK)


def binom(pcpu, k):
    return math.comb(TOPK, k) * pcpu**k * (1 - pcpu) ** (TOPK - k)


def per_token_ms(Tk, probs):
    per_layer_us = sum(probs[k] * Tk[k] for k in range(TOPK + 1))
    return N_LAYERS * per_layer_us / 1000.0, per_layer_us


def main():
    outdir = sys.argv[1]
    runs = load(outdir)
    curve_tags = [t for t in runs if t.startswith("curve_")]
    Tk, raw = curve_from(runs, curve_tags, T=1)

    print(f"# T(k) at T=1  (median over {len(curve_tags)} passes: {sorted(curve_tags)})")
    print("k_cpu   p50 us    passes            marginal us   implied marginal GB/s")
    prev = None
    for k in sorted(Tk):
        marg = "" if prev is None else f"{Tk[k] - prev:10.1f}"
        bw = "" if prev is None else f"{BYTES_PER_EXPERT_LAYER / ((Tk[k] - prev) * 1e-6) / 1e9:15.1f}"
        print(f"{k:5d} {Tk[k]:9.1f}   {['%.1f' % v for v in raw[k]]}{marg}{bw}")
        prev = Tk[k]

    for lo in (1, 2):
        ks = [k for k in Tk if k >= lo]
        a, b = lsq(ks, [Tk[k] for k in ks])
        bw_slope = BYTES_PER_EXPERT_LAYER / (b * 1e-6) / 1e9
        print(f"\nfit (k>={lo}):  T(k) = {a:.1f} + {b:.2f}*k us     T(0) = {Tk[0]:.1f} us")
        print(f"  slope bandwidth = {BYTES_PER_EXPERT_LAYER / 1e6:.2f} MB / {b:.2f} us = {bw_slope:.0f} GB/s"
              f"   ({100 * bw_slope / BW_DUAL:.0f}% of the {BW_DUAL:.0f} GB/s dual-socket ceiling,"
              f" {bw_slope / BW_1CCD:.1f}x a single CCD)")
    print("  k=1 sits above the k>=2 line: gate/up has nth*k*2 = ceil(2304/256)*k*2 = 18 tasks at"
          " k=1 vs 48 workers per NUMA pool, so k=1 is parallelism-limited, not bandwidth-limited.")

    print(f"\n# bs=1 extrapolation: per token = {N_LAYERS} layers x E[T(k_cpu)]")
    print("N_res  N_cpu  model         p_cpu  E[k]   E[T]/layer us   CPU ms/token")
    rows = []
    for N_res in (100, 150, 200, 250):
        K = N_EXP - N_res
        for name, probs in (
            ("hypergeom", {k: hypergeom(K, k) for k in range(TOPK + 1)}),
            ("binomial ", {k: binom(K / N_EXP, k) for k in range(TOPK + 1)}),
        ):
            ms, us = per_token_ms(Tk, probs)
            print(f"{N_res:5d} {K:6d}  {name}  {K / N_EXP:6.3f} {6 * K / N_EXP:5.2f}"
                  f"   {us:13.1f}   {ms:12.2f}")
            rows.append((N_res, name.strip(), ms))

    print("\n# parameterised by resident-set hit rate p (k_cpu ~ Binomial(6, 1-p))")
    print("  'uniform'  p = N_res/384 (no skew)")
    print("  'skewed'   p = min(1, 2.1 x N_res/384).  2.1x = sglang DSV4 measured dynamic")
    print("             residency 26-31% hit at 32/256 = 12.5% coverage")
    print("             (sglang_design_doc/dsv4_flash_single_card_design.md:169,")
    print("             roadmap 'static prefix 32 covers ~13%, dynamic ~3x that').")
    print("             ASSUMPTION, not measured on DSV4.1; the factor cannot hold")
    print("             above p=1/2.1=48% coverage and is capped here.")
    print("N_res  uniform p  CPU ms/tok | skewed p  CPU ms/tok")
    for N_res in (100, 150, 200):
        pu = N_res / N_EXP
        ps = min(1.0, 2.1 * pu)
        mu, _ = per_token_ms(Tk, {k: binom(1 - pu, k) for k in range(TOPK + 1)})
        msk, _ = per_token_ms(Tk, {k: binom(1 - ps, k) for k in range(TOPK + 1)})
        cap = " (capped)" if 2.1 * pu > 1 else ""
        print(f"{N_res:5d}   {pu:8.3f} {mu:11.2f} | {ps:8.3f} {msk:11.2f}{cap}")

    print(f"\n# reverse solve: N_resident needed for a CPU-time target (uniform, hypergeom)")
    print("target ms/token   min N_resident   (K = 384 - N_res on CPU)")
    for target in (26.84, 20.0, 15.0, 10.0, 5.0, 2.0):
        need = None
        for N_res in range(0, N_EXP + 1):
            ms, _ = per_token_ms(Tk, {k: hypergeom(N_EXP - N_res, k) for k in range(TOPK + 1)})
            if ms <= target:
                need = N_res
                break
        print(f"{target:13.2f}   {need if need is not None else 'impossible':>14}"
              f"   {'' if need is None else N_EXP - need}")

    # The reverse solve above is only meaningful inside the HBM gate: every
    # resident expert costs HBM.  tools/kt_offload/kt_offload_gate.py:80-82
    # 10.4 + N_res * 0.7004 * KT_DEVICE_FACTOR <= HBM*GPUUTIL - KV_reserve
    gate_max = int((128 * 0.9 - 25 - 10.4) / 0.7004)
    print(f"\n# HBM gate (kt_offload_gate.py defaults: 128 GiB x 0.9 - 25 KV - 10.4 other,"
          f" 0.7004 GiB/expert, KT_DEVICE_FACTOR=1): N_resident <= {gate_max}")
    ms_gate, _ = per_token_ms(Tk, {k: hypergeom(N_EXP - gate_max, k) for k in range(TOPK + 1)})
    print(f"  at N_resident = {gate_max}: CPU {ms_gate:.2f} ms/token (uniform),"
          f" host experts {(N_EXP - gate_max) * 0.7004:.1f} GiB")
    print("  => targets needing N_resident above that are not reachable on one card"
          " without changing KV reserve / GPUUTIL / KT_DEVICE_FACTOR.")

    floor_ms = N_LAYERS * Tk[0] / 1000.0
    print(f"\n# floors: k_cpu=0 on every layer still costs {N_LAYERS} x {Tk[0]:.1f} us "
          f"= {floor_ms:.2f} ms/token (do_numa_job round trip)")
    ms_all_cpu, _ = per_token_ms(Tk, {k: (1.0 if k == TOPK else 0.0) for k in range(TOPK + 1)})
    print(f"# ceiling: every top-6 on the CPU (N_res=0) = {N_LAYERS} x {Tk[6]:.1f} us "
          f"= {ms_all_cpu:.2f} ms/token")

    print(f"\n# conditional comparison with the {TPOT_MULTICARD_MS} ms multi-card bs1 TPOT")
    print("  (single-card NPU time is NOT measured -- these are conditions, not predictions)")
    print("N_res  CPU ms/tok  fully exposed: NPU budget left   fully hidden: NPU must be >= ")
    for N_res in (100, 150, 200):
        ms, _ = per_token_ms(Tk, {k: hypergeom(N_EXP - N_res, k) for k in range(TOPK + 1)})
        print(f"{N_res:5d} {ms:11.2f} {TPOT_MULTICARD_MS - ms:26.2f} ms {ms:26.2f} ms")

    # Check bit 1: shrink the working set until the implied bandwidth becomes
    # physically impossible.  A cache-resident measurement reads *faster than
    # DRAM can deliver*; that is what makes this a discriminating check.
    cache_tags = [t for t in runs if t.startswith("cache_")]
    if cache_tags:
        print(f"\n# check bit: working-set sweep (same pool config, k_cpu=6)")
        print("tag              resident GiB   T(6) us   implied GB/s   verdict")
        for t in sorted(cache_tags, key=lambda t: runs[t]["resident_gib"]):
            r = runs[t]
            row = [x for x in r["results"] if x["T"] == 1 and x["k_cpu"] == 6]
            if not row:
                continue
            g = row[0]["implied_gbs"]
            verdict = ("IMPOSSIBLE -> cache hits" if g > BW_DUAL else
                       "above 1 socket" if g > BW_SOCKET else "plausible DRAM")
            print(f"{t:16s} {r['resident_gib']:12.2f} {row[0]['p50']:9.1f} {g:14.1f}   {verdict}")
        print(f"  hardware (sglang bw.c, same host): 1 CCD {BW_1CCD:.0f} / socket {BW_SOCKET:.0f} /"
              f" dual {BW_DUAL:.0f} GB/s (docs_cpu_optimization.md §4.1, §5.2)")

    # Check bit 2: KT_MXFP4_COMPACT_SCALES must not move a single bit.
    dumps = {t: runs[t]["dump"] for t in runs if runs[t].get("dump")}
    if dumps:
        print("\n# check bit: output bytes per run (KT_MXFP4_COMPACT_SCALES A/B)")
        for t in sorted(dumps):
            print(f"  {t:18s} compact={runs[t]['config']['env']['KT_MXFP4_COMPACT_SCALES']:>7s} "
                  f"sha256={dumps[t]['sha256'][:16]} routed_ids={dumps[t]['ids'][0]}")
        print("  (ctl_diffroute uses a different CPU pool -> different experts -> must differ;"
              " that is what gives the sha discriminating power)")

    # Coverage
    print("\n# coverage")
    for t in sorted(runs):
        c = runs[t]["config"]
        n = sum(r["n"] for r in runs[t]["results"])
        print(f"  {t:18s} layers={c['layers']} n_cpu={c['n_cpu']} th={c['threads']} "
              f"tp={c['threadpool']} numa={c['numa']} bind={c['env']['KT_CPU_BIND']} "
              f"compact={c['env']['KT_MXFP4_COMPACT_SCALES']} timed_calls={n} "
              f"resident={runs[t]['resident_gib']:.2f} GiB "
              f"load={runs[t]['load_before']['loadavg'][0]}")

    # T scaling
    print("\n# T scaling (per-token cost inside one forward), k_cpu=4")
    for tag in sorted(curve_tags):
        for r in runs[tag]["results"]:
            if r["k_cpu"] == 4:
                print(f"  {tag}  T={r['T']:<3d} p50={r['p50']:9.1f} us  "
                      f"per-token {r['p50'] / r['T']:8.1f} us  distinct_E={r['distinct_experts_p50']:.0f}  "
                      f"implied {r['implied_gbs']:.0f} GB/s")


if __name__ == "__main__":
    main()
