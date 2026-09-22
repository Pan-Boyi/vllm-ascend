#!/usr/bin/env python3
"""NOTIFY_WAIT / HOSTFUNC_CALLBACK structure per step, from task_time.csv."""
import csv, sys, statistics as st
from collections import defaultdict
rows = defaultdict(list)
with open(sys.argv[1], newline="") as f:
    for r in csv.DictReader(f):
        k = (r["kernel_type"] or "").strip()
        if k not in ("NOTIFY_WAIT", "HOSTFUNC_CALLBACK", "MODEL_EXECUTE", "MEMCPY_ASYNC"):
            continue
        rows[k].append((float(r["task_start(us)"].strip()), float(r["task_time(us)"]), (r["stream_id"] or "").strip()))
for k in rows: rows[k].sort()
steps = rows["MODEL_EXECUTE"]
print(f"MODEL_EXECUTE {len(steps)} steps, median duration {st.median([d for _,d,_ in steps]):.0f} us")
nw = rows["NOTIFY_WAIT"]
by_stream = defaultdict(list)
for s, d, sid in nw: by_stream[sid].append(d)
print("NOTIFY_WAIT by stream:", {k: (len(v), round(st.median(v),1)) for k, v in by_stream.items()})
d_all = [d for _, d, _ in nw]
print(f"NOTIFY_WAIT n={len(d_all)} median {st.median(d_all):.1f} us mean {st.mean(d_all):.1f} "
      f"p10 {sorted(d_all)[len(d_all)//10]:.1f} p90 {sorted(d_all)[9*len(d_all)//10]:.1f} "
      f"max {max(d_all):.0f}  sum/step {sum(d_all)/len(steps)/1000:.2f} ms")
# per position inside a step: assign each NOTIFY_WAIT to the step whose window contains it
# steps are delimited by successive MODEL_EXECUTE starts (the marker itself is 3 us)
starts = [s for s, _, _ in steps]
pos = defaultdict(list)
import bisect
for s, d, sid in nw:
    i = bisect.bisect_right(starts, s) - 1
    if i >= 0: pos.setdefault(i, []).append(d)
counts = [len(v) for v in pos.values()]
print(f"steps matched {len(pos)}, NOTIFY_WAIT per step: median {st.median(counts)} min {min(counts)} max {max(counts)}")
# layer index = order within the step
by_idx = defaultdict(list)
for i, v in pos.items():
    for j, d in enumerate(v): by_idx[j].append(d)
print("\nidx  n  median_us  p90_us")
for j in sorted(by_idx):
    v = by_idx[j]
    print(f"{j:3d} {len(v):4d} {st.median(v):9.1f} {sorted(v)[int(0.9*len(v))]:7.1f}")
mc = [d for _, d, _ in rows["MEMCPY_ASYNC"]]
print(f"\nMEMCPY_ASYNC n/step {len(mc)/len(steps):.1f} median {st.median(mc):.3f} us sum/step {sum(mc)/len(steps)/1000:.3f} ms")
hf = [d for _, d, _ in rows["HOSTFUNC_CALLBACK"]]
print(f"HOSTFUNC_CALLBACK n/step {len(hf)/len(steps):.1f} median {st.median(hf):.4f} us")
