#!/usr/bin/env python3
"""Parse [kt-phase] records out of a serve.log byte range.

Records can be interleaved with other stdout writers, so we match the whole
record shape and count what we had to drop instead of assuming lines are whole.
"""
import re, sys, statistics as st, json
from collections import defaultdict

PAT = re.compile(
    r"\[kt-phase\] layer (\d+) numa (\d+) activated (\d+) \| q_input ([\d.]+) \| up_gate ([\d.]+) \| "
    r"act ([\d.]+) \| q_down ([\d.]+) \| down ([\d.]+) \| weight ([\d.]+) \| total ([\d.]+) us")

path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, "rb") as f:
    f.seek(start)
    blob = f.read(end - start).decode("utf-8", "replace")
recs = PAT.findall(blob)
raw_markers = blob.count("[kt-phase]")
fields = ["q_input", "up_gate", "act", "q_down", "down", "weight", "total"]
per_layer = defaultdict(lambda: defaultdict(list))
acts = defaultdict(list)
for m in recs:
    layer, numa, activated = int(m[0]), int(m[1]), int(m[2])
    acts[(layer, numa)].append(activated)
    for name, v in zip(fields, m[3:]):
        per_layer[(layer, numa)][name].append(float(v))

def q(xs, p):
    xs = sorted(xs); 
    return xs[min(len(xs)-1, int(p*len(xs)))]

all_tot = [v for k in per_layer for v in per_layer[k]["total"]]
print(f"records parsed: {len(recs)}   markers seen: {raw_markers}   dropped(interleaved): {raw_markers-len(recs)}")
print(f"layers x numa keys: {len(per_layer)}")
print(f"\nper-call total (all layers, both numa): median {st.median(all_tot):.1f} us  p90 {q(all_tot,0.9):.1f}  "
      f"p99 {q(all_tot,0.99):.1f}  mean {st.mean(all_tot):.1f}  n={len(all_tot)}")
print("\nphase medians over all calls (us):")
for name in fields:
    xs = [v for k in per_layer for v in per_layer[k][name]]
    print(f"  {name:8s} median {st.median(xs):6.1f}   p90 {q(xs,0.9):6.1f}   share {100*st.median(xs)/st.median(all_tot):5.1f}%")
# per layer: max over numa of the median (the layer waits for the slower node)
layer_med, layer_p90, layer_act = {}, {}, {}
for layer in sorted({k[0] for k in per_layer}):
    tots = [per_layer[(layer, n)]["total"] for n in (0, 1) if (layer, n) in per_layer]
    # pair up calls across numa by index: the layer's cost is the slower node
    if len(tots) == 2:
        n = min(len(tots[0]), len(tots[1]))
        pair = [max(tots[0][i], tots[1][i]) for i in range(n)]
    else:
        pair = tots[0]
    layer_med[layer] = st.median(pair); layer_p90[layer] = q(pair, 0.9)
    a = [v for n in (0,1) if (layer,n) in acts for v in acts[(layer,n)]]
    layer_act[layer] = st.mean(a)
print(f"\nper-layer kt CPU (slower NUMA of the pair): median-of-layers {st.median(list(layer_med.values())):.1f} us, "
      f"min {min(layer_med.values()):.1f} (L{min(layer_med, key=layer_med.get)}), "
      f"max {max(layer_med.values()):.1f} (L{max(layer_med, key=layer_med.get)})")
print(f"sum over 40 layers of the per-layer medians: {sum(layer_med.values())/1000:.2f} ms/token   "
      f"(p90 sum {sum(layer_p90.values())/1000:.2f} ms)")
print(f"mean activated experts per numa node per call: {st.mean([layer_act[l] for l in layer_act]):.2f}")
print("\nlayer  med_us  p90_us  activated(mean,per-numa)")
for layer in sorted(layer_med):
    print(f"{layer:5d} {layer_med[layer]:7.1f} {layer_p90[layer]:7.1f}   {layer_act[layer]:.2f}")
json.dump({"layer_median_us": layer_med, "layer_p90_us": layer_p90,
           "sum40_median_ms": sum(layer_med.values())/1000,
           "n_records": len(recs), "dropped": raw_markers-len(recs)},
          open(sys.argv[4], "w"), indent=1) if len(sys.argv) > 4 else None
