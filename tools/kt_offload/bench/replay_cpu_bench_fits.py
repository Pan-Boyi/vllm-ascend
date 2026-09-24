#!/usr/bin/env python3
"""Re-fit every cpu_bench arm from its raw json, so MOE_OFFLOAD_PLAN.md §4.3 / §4.3.1
can be checked without trusting ANALYSIS.txt.

Why this exists (2026-09-24 断言回溯, PLAN §4.35 R-05/R-06):
  - §4.3 quoted ONE curve -- `T(k) = 66.8 + 43.95k`, 428 GB/s -- as if it were a
    property of the machine.  It is a property of ONE arm: 3 layers, a 14.92 GiB
    working set, loadavg ~20.  The same session also measured the *production*
    working set (40 layers, 284 CPU experts, 198.91 GiB) in
    `cpu_bench/20260920_111157_prod/`, and that arm is ~18% steeper.
  - ANALYSIS.txt labels its per-k column `p50 (median over 2 passes)` but with an
    even number of passes `sorted(v)[len(v)//2]` returns the LARGER pass.  This
    script prints each arm separately so nobody has to reason about that.

This script is read-only: it parses the archived json and re-does the least
squares.  It measures nothing and touches no card.

Usage:
  python3 tools/kt_offload/bench/replay_cpu_bench_fits.py [BENCH_ROOT]

BENCH_ROOT defaults to the shared artifact dir; every arm that is missing is
reported rather than silently skipped (a silently skipped arm is exactly how the
production-working-set run went unnoticed for four days).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 3 * (5120*2304/2 + 5120*2304/32) -- MXFP4 nibble + 1-byte E8M0 scale per 32.
# Identical to kt_cpu_bench.py:BYTES_PER_EXPERT_LAYER and to each json's
# config.bytes_per_expert_layer.  NOTE: this is DERIVED from the model shape,
# not measured; 0.7004 GiB/expert is the same derivation x384 experts x40 layers.
BYTES_PER_EXPERT_LAYER = 18_800_640
N_LAYERS = 40

DEFAULT_ROOT = Path(
    "/workspace/code/dsv4_single_card/vllm_project/build_artifacts/moe_offload_ref/cpu_bench"
)

# (relative path, what the arm is for)
ARMS = [
    ("20260920_022930/curve_a.json", "§4.3 主曲线 pass A"),
    ("20260920_022930/curve_b.json", "§4.3 主曲线 pass B"),
    ("20260920_111157_prod/ctl_3layer_today.json", "同日 3 层对照臂"),
    ("20260920_111157_prod/prod_a.json", "生产工作集"),
    ("20260920_111157_prod/prod_b.json", "生产工作集 (机器更忙)"),
]

# E[k_cpu] measured on-card: p2_prof_0922_131258/kt_phase_window.txt:17
K_ONCARD = 4.54


def lsq(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Return (intercept, slope).  Plain OLS, same as analyze_cpu_bench.py:lsq."""
    n = len(xs)
    if n < 2:
        raise ValueError("need >= 2 points")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den
    return my - slope * mx, slope


def r_squared(xs: list[float], ys: list[float], b: float, a: float) -> float:
    my = sum(ys) / len(ys)
    ss_res = sum((y - (b + a * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def load_arm(path: Path) -> dict:
    d = json.loads(path.read_text())
    cfg = d["config"]
    pts = {r["k_cpu"]: r["p50"] for r in d["results"] if r["T"] == 1}
    return {
        "tag": cfg["tag"],
        "layers": len(cfg["layers"]),
        "n_cpu": cfg["n_cpu"],
        "reps": cfg["reps"],
        "threads": cfg["threads"],
        "tp": cfg["threadpool"],
        "numa": cfg["numa"],
        "bytes": cfg["bytes_per_expert_layer"],
        "gib": d.get("resident_gib", float("nan")),
        "load": d.get("load_before", {}).get("loadavg", ["?"])[0],
        "pts": pts,
    }


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    rows, missing = [], []
    for rel, note in ARMS:
        p = root / rel
        if not p.is_file():
            missing.append(rel)
            continue
        a = load_arm(p)
        a["note"] = note
        rows.append(a)

    if not rows:
        print(f"no arm found under {root}", file=sys.stderr)
        return 2

    hdr = (
        f"{'arm':<18}{'note':<22}{'层':>3}{'GiB':>8}{'load':>7}"
        f"{'T(0)':>7}{'截距':>7}{'斜率':>7}{'GB/s':>6}{'R2':>7}"
        f"{'T(4.54)':>9}{'ms/tok':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for a in rows:
        if a["bytes"] != BYTES_PER_EXPERT_LAYER:
            print(f"  !! {a['tag']}: bytes_per_expert_layer={a['bytes']} != {BYTES_PER_EXPERT_LAYER}")
        ks = sorted(k for k in a["pts"] if k >= 2)  # k>=2: k<=1 is parallelism-limited
        if len(ks) < 2:
            print(f"{a['tag']:<18}{a['note']:<22}  (k>=2 点不足, 有 {sorted(a['pts'])})")
            continue
        ys = [a["pts"][k] for k in ks]
        b, slope = lsq([float(k) for k in ks], ys)
        r2 = r_squared([float(k) for k in ks], ys, b, slope)
        gbs = BYTES_PER_EXPERT_LAYER / (slope * 1e-6) / 1e9
        t = b + slope * K_ONCARD
        print(
            f"{a['tag']:<18}{a['note']:<22}{a['layers']:>3}{a['gib']:>8.2f}{a['load']:>7}"
            f"{a['pts'].get(0, float('nan')):>7.1f}{b:>7.1f}{slope:>7.2f}{gbs:>6.0f}{r2:>7.4f}"
            f"{t:>9.1f}{t * N_LAYERS / 1000:>8.2f}"
        )

    print()
    print("读法（§4.35 R-05/R-06/R-07）:")
    print("  * 斜率/GB/s 是 k>=2 的拟合产物, 不是带宽直测; 它随工作集变 (14.92 GiB ~428 -> 生产 ~363).")
    print("  * T(0) 那一列才是直测地板; 与生产同配的是 prod_a/prod_b 的 22.7 / 23.5 us.")
    print("  * 丢掉截距只留斜率 (即 '字节 / 428 GB/s') 不对应任何测量 -- 见 §4.3 R-07.")
    print(f"    对照: 4.54 x 斜率 x 40 层, 各臂 = "
          f"{', '.join(f'{K_ONCARD * lsq([float(k) for k in sorted(x for x in a["pts"] if x >= 2)], [a["pts"][k] for k in sorted(x for x in a["pts"] if x >= 2)])[1] * N_LAYERS / 1000:.2f}' for a in rows)} ms/token")
    print("  * 卡上直测是 8.88 (profiler 关) / 9.59 (开) ms/token -- 台架每一条 *完整拟合* 都高于它 (10.38~12.16).")
    print("    丢截距的那一行有几个臂碰巧落在 8.88 附近; 那是巧合, 不是验证 (§4.35 R-07).")
    if missing:
        print()
        print("缺失的臂 (没有静默跳过):")
        for m in missing:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
