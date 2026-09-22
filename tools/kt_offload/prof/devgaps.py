#!/usr/bin/env python3
"""Per-stream device idle from a profile's ``task_time.csv``.

Why this exists and why it is not ``kd.py``: ``kd.py`` unions the kernel
intervals of *every* stream, so a gap on the main stream that another stream
happens to fill is not a gap there. The question the side stream exists to
answer -- "does the main stream still stop while the CPU experts run?" -- is a
*per-stream* question, and ``kernel_details.csv`` carries no stream id.
``task_time.csv`` does, and it also carries the non-kernel tasks
(``NOTIFY_WAIT``, ``HOSTFUNC_CALLBACK``, ``MEMCPY_ASYNC``, ``*_EVENT_WAIT``)
that a host-func seam is made of.

Two readings per stream, kept apart on purpose:

  idle        gap between the end of one task and the start of the next on
              that stream. It says nothing about why.
  waiting     the share of the step spent *inside* a task whose whole job is
              to wait (``NOTIFY_WAIT``, ``*EVENT_WAIT``, ``MEM_WAIT_VALUE``).
              On the serial path the CPU experts are paid for here; on the
              forked path this is what should move off the main stream.

``--per-type`` additionally prints, per task type, the median gap *before* a
task of that type (the §4.22 F1 number: ``HOSTFUNC_CALLBACK`` sits behind a
median 30.75 us of nothing on the stream that owns it, while every other type
starts at 0.000).

Steps are delimited by ``MODEL_EXECUTE`` starts when that marker is present,
which is what makes the per-token columns meaningful; without it only totals
are printed.

    devgaps.py task_time.csv [--json OUT] [--gap-us 100] [--per-type] [--layers 40]
"""
import argparse
import bisect
import csv
import json
import statistics as st
from collections import defaultdict

WAIT_TYPES = ("NOTIFY_WAIT", "EVENT_WAIT", "MEM_WAIT_VALUE")


def is_wait(name: str) -> bool:
    return any(w in name for w in WAIT_TYPES)


def pct(v, q):
    if not v:
        return 0.0
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--json")
    ap.add_argument("--gap-us", type=float, default=100.0)
    ap.add_argument("--per-type", action="store_true")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--top", type=int, default=8, help="how many streams to print")
    ap.add_argument("--types", type=int, default=10, help="how many task types per stream to print")
    a = ap.parse_args()

    per_stream = defaultdict(list)  # sid -> [(start, stop, type)]
    with open(a.csv, newline="") as f:
        for r in csv.DictReader(f):
            try:
                s = float((r["task_start(us)"] or "").strip())
                e = float((r["task_stop(us)"] or "").strip())
            except (KeyError, ValueError):
                continue
            per_stream[(r["stream_id"] or "").strip()].append((s, e, (r["kernel_type"] or "").strip()))
    for v in per_stream.values():
        v.sort()

    # steps: MODEL_EXECUTE starts, wherever they live
    starts = sorted(s for v in per_stream.values() for s, _, t in v if t == "MODEL_EXECUTE")
    steps = max(1, len(starts))
    print(f"{a.csv}: {sum(len(v) for v in per_stream.values())} tasks on {len(per_stream)} streams, "
          f"MODEL_EXECUTE steps={len(starts)}"
          + (f", step period median {st.median([b - x for x, b in zip(starts, starts[1:])]):.0f} us"
             if len(starts) > 2 else ""))

    out = {"steps": len(starts), "streams": {}}
    order = sorted(per_stream, key=lambda k: -len(per_stream[k]))[: a.top]
    print(f"\n{'stream':>8s} {'tasks':>7s} {'n/step':>7s} {'span_ms':>8s} {'busy%':>6s} "
          f"{'wait_ms/step':>12s} {'wait%step':>9s} {'idle_ms/step':>12s} "
          f"{'gaps>=G':>8s} {'G_ms/step':>10s} {'G_med_us':>9s}")
    for sid in order:
        v = per_stream[sid]
        span = v[-1][1] - v[0][0]
        busy = 0.0
        gaps = []
        cs, ce = v[0][0], v[0][1]
        for s, e, _t in v[1:]:
            if s > ce:
                gaps.append(s - ce)
                busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += ce - cs
        wait = sum(e - s for s, e, t in v if is_wait(t))
        big = [g for g in gaps if g >= a.gap_us]
        step_period = span / steps
        row = {
            "tasks": len(v), "span_ms": span / 1000, "busy_pct": 100 * busy / span if span else 0,
            "wait_ms_per_step": wait / steps / 1000,
            "wait_pct_of_step": 100 * wait / span if span else 0,
            "idle_ms_per_step": (span - busy) / steps / 1000,
            "big_gaps_per_step": len(big) / steps,
            "big_gap_ms_per_step": sum(big) / steps / 1000,
            "big_gap_med_us": st.median(big) if big else 0.0,
            "big_gap_p90_us": pct(big, 0.9),
            "step_period_us": step_period,
        }
        out["streams"][sid] = row
        print(f"{sid:>8s} {len(v):7d} {len(v)/steps:7.1f} {span/1000:8.1f} {row['busy_pct']:6.1f} "
              f"{row['wait_ms_per_step']:12.2f} {row['wait_pct_of_step']:9.1f} "
              f"{row['idle_ms_per_step']:12.2f} {len(big)/steps:8.1f} "
              f"{row['big_gap_ms_per_step']:10.2f} {row['big_gap_med_us']:9.1f}")

    if a.per_type:
        print(f"\npre-task gap by type (the §4.22 F1 reading), per stream:")
        print(f"{'stream':>8s} {'type':28s} {'n':>7s} {'n/step':>7s} {'dur_med':>8s} {'dur_mean':>9s} "
              f"{'dur_ms/step':>11s} {'pregap_med':>10s} {'pregap_p90':>10s} {'pregap_ms/step':>14s}")
        for sid in order:
            v = per_stream[sid]
            pre = defaultdict(list)
            dur = defaultdict(list)
            for i, (s, e, t) in enumerate(v):
                dur[t].append(e - s)
                if i:
                    pre[t].append(max(0.0, s - v[i - 1][1]))
            for t in sorted(dur, key=lambda x: -len(dur[x]))[: a.types]:
                g = pre.get(t, [])
                print(f"{sid:>8s} {t[:28]:28s} {len(dur[t]):7d} {len(dur[t])/steps:7.1f} "
                      f"{st.median(dur[t]):8.3f} {st.mean(dur[t]):9.3f} {sum(dur[t])/steps/1000:11.3f} "
                      f"{(st.median(g) if g else 0):10.3f} "
                      f"{pct(g, 0.9):10.3f} {sum(g)/steps/1000:14.3f}")
                out.setdefault("per_type", {}).setdefault(sid, {})[t] = {
                    "n": len(dur[t]), "n_per_step": len(dur[t]) / steps,
                    "dur_med_us": st.median(dur[t]), "dur_mean_us": st.mean(dur[t]),
                    "dur_ms_per_step": sum(dur[t]) / steps / 1000,
                    "pregap_med_us": st.median(g) if g else 0.0,
                    "pregap_ms_per_step": sum(g) / steps / 1000,
                }

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
