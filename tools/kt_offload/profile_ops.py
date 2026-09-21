#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Count what a run actually did, per step, from a torch-npu profile.

Written because two operation accounts were in circulation -- 120 host-device
operations per step (40 layers x 3) and 200 (40 x 5) -- and a per-operation
price was derived by dividing an aggregate by whichever count was at hand
(7.5 ms / 120 = 62.5 us, or / 200 = 37.5 us). A price obtained that way cannot
arbitrate between the two counts: it was built from one of them.

This reads ``task_time.csv`` (the ascend profiler's per-task table) and reports
counts and device time by task type, per step, so the question is answered by
the run instead of by the code. It deliberately does NOT attribute tasks to
owners: the table has no names (``kernel_name`` is N/A for these types), so a
memcpy issued by the MoE offload and one issued by the attention look the
same. Everything printed is therefore an upper bound for ours, and the report
says so.

Usage:  profile_ops.py <task_time.csv> [--steps N] [--layers 40] [--price-us P]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import sys
from collections import defaultdict

# the types a host-device transport + callback budget is made of
TRANSPORT = ("MEMCPY_ASYNC", "HOSTFUNC_CALLBACK", "NOTIFY_WAIT")


def read(path: str) -> dict:
    n: dict[str, int] = defaultdict(int)
    total: dict[str, float] = defaultdict(float)
    per_stream: dict[tuple[str, str], int] = defaultdict(int)
    rows = 0
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            kind = (row.get("kernel_type") or "").strip()
            if not kind:
                continue
            rows += 1
            n[kind] += 1
            with contextlib.suppress(ValueError):
                total[kind] += float(row.get("task_time(us)") or 0.0)
            per_stream[(kind, (row.get("stream_id") or "?").strip())] += 1
    return {"n": dict(n), "us": dict(total), "per_stream": dict(per_stream), "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_time_csv")
    ap.add_argument("--steps", type=int, default=None, help="default: the MODEL_EXECUTE count")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--price-us", type=float, default=None, help="a claimed per-operation cost, to test")
    ap.add_argument("--against", default=None, help="a second task_time.csv to compare; refused across builds")
    ap.add_argument("--claimed-total-ms", type=float, default=None, help="the aggregate that price came from")
    args = ap.parse_args()

    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("build_fingerprint", os.path.join(here, "build_fingerprint.py"))
    fpm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fpm)
    mine = fpm.load(os.path.dirname(os.path.abspath(args.task_time_csv)))
    print(f"{fpm.short(mine)}   <- the build these numbers were measured on\n")

    d = read(args.task_time_csv)
    if not d["rows"]:
        print("no rows read: wrong file?", file=sys.stderr)
        return 2
    steps = args.steps or d["n"].get("MODEL_EXECUTE", 0)
    if not steps:
        print("no MODEL_EXECUTE rows and no --steps: cannot report per-step numbers", file=sys.stderr)
        return 2

    print(f"rows {d['rows']}, steps {steps} ({'given' if args.steps else 'MODEL_EXECUTE count'}), "
          f"layers {args.layers}\n")
    print(f"{'type':<22}{'n':>10}{'n/step':>10}{'n/step/layer':>14}{'us/op':>10}{'ms/step':>10}")
    for kind in sorted(d["n"], key=lambda k: -d["us"].get(k, 0.0)):
        cnt, us = d["n"][kind], d["us"].get(kind, 0.0)
        print(f"{kind:<22}{cnt:>10}{cnt / steps:>10.1f}{cnt / steps / args.layers:>14.2f}"
              f"{us / cnt:>10.3f}{us / steps / 1000:>10.3f}")

    print("\nper-step transport + callback (device time, ALL owners -- an upper bound for ours):")
    tot_ms = 0.0
    for kind in TRANSPORT:
        if kind in d["n"]:
            ms = d["us"][kind] / steps / 1000
            tot_ms += ms
            print(f"  {kind:<20} {d['n'][kind] / steps:8.1f}/step  {ms:8.3f} ms/step  "
                  f"({d['us'][kind] / d['n'][kind]:.3f} us each)")
    print(f"  {'sum':<20} {'':8}  {tot_ms:8.3f} ms/step")

    if args.price_us:
        for label, count in (("120 (40x3)", 120), ("200 (40x5)", 200)):
            implied = args.price_us * count / 1000
            print(f"\na price of {args.price_us:g} us/op with {label} ops/step implies {implied:.2f} ms/step")
            for kind in ("MEMCPY_ASYNC", "HOSTFUNC_CALLBACK"):
                if kind in d["n"]:
                    meas = d["us"][kind] / steps / 1000
                    print(f"  measured {kind} device time: {meas:.3f} ms/step "
                          f"({implied / meas:.0f}x smaller than the implied figure)"
                          if meas else f"  measured {kind}: 0")
        if args.claimed_total_ms:
            print(f"\nthe aggregate the price came from ({args.claimed_total_ms:g} ms/step) is "
                  f"{args.claimed_total_ms / max(tot_ms, 1e-9):.1f}x the measured transport+callback device time; "
                  "if that aggregate is real it is mostly NOT device time spent in those operations")
    if args.against:
        other = fpm.load(os.path.dirname(os.path.abspath(args.against)))
        try:
            advisories = fpm.require_same_build(mine, other, "a per-type difference between these two profiles")
        except fpm.CrossBuildError as exc:
            print(f"\n⛔ {exc}")
            return 1
        for a in advisories:
            print("  " + a)
        e = read(args.against)
        steps2 = args.steps or e["n"].get("MODEL_EXECUTE", 0)
        print(f"\nper-type difference vs {args.against} ({steps2} steps), same build:")
        print(f"{'type':<22}{'n/step A':>10}{'n/step B':>10}{'ms/step A':>11}{'ms/step B':>11}{'delta ms':>10}")
        for kind in sorted(set(d["n"]) | set(e["n"])):
            na, nb = d["n"].get(kind, 0) / steps, e["n"].get(kind, 0) / max(steps2, 1)
            ma = d["us"].get(kind, 0.0) / steps / 1000
            mb = e["us"].get(kind, 0.0) / max(steps2, 1) / 1000
            if max(ma, mb) >= 0.05:
                print(f"{kind:<22}{na:>10.1f}{nb:>10.1f}{ma:>11.3f}{mb:>11.3f}{mb - ma:>+10.3f}")

    print("\ncaveat: this table cannot say which owner issued a task (no kernel names for these types), "
          "so every count above is all owners together.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
