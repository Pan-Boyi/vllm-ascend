#!/usr/bin/env python3
"""Aggregate an ascend kernel_details.csv for a bs=1 decode window.

device busy is the UNION of kernel intervals (overlap across streams counted
once); the additive sum is printed next to it so the two can be compared.
"""
import csv, sys, statistics as st, json
from collections import defaultdict

path = sys.argv[1]
GAP_US = 100.0
rows = 0
iv = []
by_type_us = defaultdict(float); by_type_n = defaultdict(int)
gmm = []; steps_marker = 0
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        try:
            s = float(r["Start Time(us)"].strip()); d = float(r["Duration(us)"])
        except (KeyError, ValueError):
            continue
        rows += 1
        t = (r.get("Type") or "").strip()
        iv.append((s, s + d))
        by_type_us[t] += d; by_type_n[t] += 1
        if t == "MoeInitRoutingV3":
            steps_marker += 1
        if t == "GroupedMatmul":
            gmm.append(d)
iv.sort()
busy = 0.0; gaps = []
cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce:
        gaps.append((ce, s - ce)); busy += ce - cs; cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
span = iv[-1][1] - iv[0][0]
add = sum(by_type_us.values())
tokens = steps_marker / 40.0
big = [g for _, g in gaps if g >= GAP_US]
print(f"kernels {rows}   span {span/1000:.1f} ms   MoeInitRoutingV3 {steps_marker} -> tokens {tokens:.1f}")
print(f"device busy (union) {busy/1000:.1f} ms = {100*busy/span:.1f}% of span; additive sum {add/1000:.1f} ms "
      f"({100*add/span:.1f}%) -> cross-stream overlap {100*(add-busy)/add:.2f}% of kernel time")
print(f"per token: busy {busy/1000/tokens:.2f} ms, idle {(span-busy)/1000/tokens:.2f} ms, step period {span/1000/tokens:.2f} ms")
print(f"gaps: {len(gaps)} total ({len(gaps)/tokens:.1f}/token), sum {sum(g for _,g in gaps)/1000/tokens:.2f} ms/token")
print(f"gaps >= {GAP_US:.0f} us: {len(big)} ({len(big)/tokens:.1f}/token), sum {sum(big)/1000/tokens:.2f} ms/token, "
      f"median {st.median(big):.0f} us, p90 {sorted(big)[int(0.9*len(big))]:.0f} us")
small = [g for _, g in gaps if g < GAP_US]
print(f"gaps <  {GAP_US:.0f} us: {len(small)} ({len(small)/tokens:.1f}/token), sum {sum(small)/1000/tokens:.2f} ms/token")
print("\ntop 20 kernel types (per token)")
print(f"{'type':32s} {'ms/tok':>8s} {'n/tok':>8s} {'us/op':>8s} {'%busy':>7s}")
for t, us in sorted(by_type_us.items(), key=lambda kv: -kv[1])[:20]:
    print(f"{t:32s} {us/1000/tokens:8.3f} {by_type_n[t]/tokens:8.1f} {us/by_type_n[t]:8.2f} {100*us/busy:7.1f}")
if gmm:
    print(f"\nGroupedMatmul: {len(gmm)/tokens:.1f} calls/token, median {st.median(gmm):.1f} us, "
          f"mean {st.mean(gmm):.1f} us, p90 {sorted(gmm)[int(0.9*len(gmm))]:.1f} us, "
          f"total {sum(gmm)/1000/tokens:.2f} ms/token")
json.dump({"tokens": tokens, "busy_ms_per_token": busy/1000/tokens, "busy_pct": 100*busy/span,
           "step_period_ms": span/1000/tokens, "idle_ms_per_token": (span-busy)/1000/tokens,
           "big_gaps_per_token": len(big)/tokens, "big_gap_ms_per_token": sum(big)/1000/tokens,
           "small_gap_ms_per_token": sum(small)/1000/tokens,
           "top": {t: [by_type_us[t]/1000/tokens, by_type_n[t]/tokens] for t in
                   sorted(by_type_us, key=lambda x: -by_type_us[x])[:25]}},
          open(sys.argv[2], "w"), indent=1) if len(sys.argv) > 2 else None
