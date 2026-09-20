#!/usr/bin/env python3
"""What the CPU-offload transport costs, by the numbers (M3 measurement item).

Two halves, deliberately separate:

  budget   how many bytes each layer moves per step, and what that is worth at
           a given link bandwidth. Pure arithmetic from the model shape.
  measure  what a profile actually spent on those copies and on the host-func
           node: aggregates the Memcpy / host rows of a
           ``kernel_details.csv`` and prints the gap table (union of busy
           intervals, gaps over a threshold), so "transport" and "callback"
           are attributed separately instead of both hiding in one number.

The point of running both: the budget says the bytes are nearly free, so if
the profile still shows milliseconds there, the cost is per-operation latency
(count x fixed cost), not bandwidth -- and those are fixed by different
changes. Keep this out of the side-stream change: overlap hides cost, it does
not remove it.

Usage:
  transport_budget.py budget  [--tokens 1 --layers 40 --hidden 5120 --topk 6 --gbps 20]
  transport_budget.py measure <kernel_details.csv> [--steps N] [--gap-us 100]
"""

from __future__ import annotations

import argparse
import csv
import sys

# One resident-offload layer moves, per step:
#   D2H  hidden states  [T, hidden] bf16        (kt's pinned input)
#        expert ids     [T, topk]   int64       (kt casts to long before the copy)
#        weights        [T, topk]   fp32
#   H2D  CPU output     [T, hidden] bf16
BF16, FP32, INT64 = 2, 4, 8


def layer_bytes(tokens: int, hidden: int, topk: int) -> dict[str, int]:
    d2h = tokens * hidden * BF16 + tokens * topk * INT64 + tokens * topk * FP32
    h2d = tokens * hidden * BF16
    return {"d2h": d2h, "h2d": h2d, "total": d2h + h2d}


def budget(tokens: int, layers: int, hidden: int, topk: int, gbps: float) -> dict[str, float]:
    per_layer = layer_bytes(tokens, hidden, topk)
    total = per_layer["total"] * layers
    # GB/s as 1e9 B/s (link vendors' unit), result in microseconds
    us = total / (gbps * 1e9) * 1e6
    return {
        "bytes_per_layer": per_layer["total"],
        "d2h_per_layer": per_layer["d2h"],
        "h2d_per_layer": per_layer["h2d"],
        "bytes_per_step": total,
        "us_per_step_at_bandwidth": us,
        "ops_per_step": layers * 3,  # D2H, host-func, H2D
    }


def fixed_cost_per_op(observed_us: float, layers: int, ops_per_layer: int = 3) -> float:
    """If the bytes are free, what per-operation cost would explain the number.

    🔴 Arithmetic only, and the model it belongs to is retired (2026-09-20):
    dividing an aggregate by an operation count produced 62.5 us at 120
    operations and 37.5 us at 200, while the profile measures 1.168 us of
    device time per memcpy. A figure derived this way is not a price and
    cannot arbitrate between operation counts. Kept because the division is
    still the right way to *state* an assumption -- never as a measurement.
    """
    return observed_us / (layers * ops_per_layer)


def _rows(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _col(row: dict, *names: str):
    for n in names:
        for k in row:
            if k.strip().lower() == n:
                return row[k]
    return None


def measure(path: str, steps: int, gap_us: float) -> dict:
    """Attribute a decode profile: copies, host-func, and the gap table.

    Busy time is the union of intervals, never the sum of durations: parallel
    streams would otherwise be counted twice.
    """
    intervals, by_kind = [], {}
    for row in _rows(path):
        name = (_col(row, "name", "kernel name", "op name") or "").strip()
        start = _col(row, "start time(us)", "start(us)", "start")
        dur = _col(row, "duration(us)", "dur(us)", "duration")
        if not name or start is None or dur is None:
            continue
        try:
            s, d = float(str(start).replace(",", "")), float(str(dur).replace(",", ""))
        except ValueError:
            continue
        intervals.append((s, s + d))
        low = name.lower()
        kind = (
            "d2h"
            if "memcpy" in low and ("dtoh" in low or "d2h" in low)
            else "h2d"
            if "memcpy" in low and ("htod" in low or "h2d" in low)
            else "memcpy_other"
            if "memcpy" in low
            else "host_func"
            if "launchcallback" in low or "hostfunc" in low or "host_func" in low
            else "compute"
        )
        agg = by_kind.setdefault(kind, {"count": 0, "us": 0.0})
        agg["count"] += 1
        agg["us"] += d

    busy, span, gaps = _union(intervals)
    big = [g for g in gaps if g >= gap_us]
    out = {
        "rows": len(intervals),
        "span_us": span,
        "busy_us_union": busy,
        "busy_share": busy / span if span else float("nan"),
        "idle_us": span - busy,
        "gaps_over_threshold": len(big),
        "gap_us_over_threshold": sum(big),
        "gap_threshold_us": gap_us,
        "by_kind": by_kind,
    }
    if steps:
        out["steps"] = steps
        for key in ("busy_us_union", "idle_us", "gap_us_over_threshold"):
            out[f"{key}_per_step"] = out[key] / steps
        out["gaps_over_threshold_per_step"] = len(big) / steps
    return out


def _union(intervals: list[tuple[float, float]]) -> tuple[float, float, list[float]]:
    """(busy, span, gaps) from possibly overlapping intervals."""
    if not intervals:
        return 0.0, 0.0, []
    ordered = sorted(intervals)
    busy, gaps = 0.0, []
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s > cur_e:
            gaps.append(s - cur_e)
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    span = ordered[-1][1] if ordered[-1][1] > cur_e else cur_e
    span = max(e for _, e in ordered) - ordered[0][0]
    return busy, span, gaps


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--tokens", type=int, default=1)
    b.add_argument("--layers", type=int, default=40)
    b.add_argument("--hidden", type=int, default=5120)
    b.add_argument("--topk", type=int, default=6)
    b.add_argument("--gbps", type=float, default=20.0)
    b.add_argument("--observed-us", type=float, default=None,
                   help="measured transport+callback microseconds per step, to attribute")
    m = sub.add_parser("measure")
    m.add_argument("csv")
    m.add_argument("--steps", type=int, default=0)
    m.add_argument("--gap-us", type=float, default=100.0)
    a = ap.parse_args()

    if a.cmd == "budget":
        r = budget(a.tokens, a.layers, a.hidden, a.topk, a.gbps)
        print(f"per layer: D2H {r['d2h_per_layer']:,} B + H2D {r['h2d_per_layer']:,} B = {r['bytes_per_layer']:,} B")
        print(f"per step ({a.layers} layers, T={a.tokens}): {r['bytes_per_step']:,} B"
              f" = {r['us_per_step_at_bandwidth']:.1f} us at {a.gbps} GB/s")
        print(f"operations per step: {r['ops_per_step']} (D2H + host-func + H2D per layer)")
        if a.observed_us:
            per_op = fixed_cost_per_op(a.observed_us, a.layers)
            print(f"observed {a.observed_us:.0f} us/step vs {r['us_per_step_at_bandwidth']:.1f} us of bytes"
                  f" => {per_op:.1f} us per operation of fixed cost"
                  f" ({100 * r['us_per_step_at_bandwidth'] / a.observed_us:.1f}% is bandwidth)")
        return 0

    r = measure(a.csv, a.steps, a.gap_us)
    print(f"rows {r['rows']}, span {r['span_us']:.0f} us, device busy (union) {r['busy_us_union']:.0f} us"
          f" = {100 * r['busy_share']:.1f}%, idle {r['idle_us']:.0f} us")
    print(f"gaps >= {r['gap_threshold_us']:.0f} us: {r['gaps_over_threshold']} totalling"
          f" {r['gap_us_over_threshold']:.0f} us")
    for kind, agg in sorted(r["by_kind"].items(), key=lambda kv: -kv[1]["us"]):
        print(f"  {kind:14s} count {agg['count']:6d}  sum {agg['us']:10.0f} us")
    if r.get("steps"):
        print(f"per step ({r['steps']} steps): busy {r['busy_us_union_per_step']:.0f} us, "
              f"idle {r['idle_us_per_step']:.0f} us, big gaps {r['gaps_over_threshold_per_step']:.1f} "
              f"totalling {r['gap_us_over_threshold_per_step']:.0f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
