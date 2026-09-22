#!/usr/bin/env python3
"""Offline A..F decomposition of the per-layer host-func seam.

Reads the ``[kt-seam]`` records a running service writes when
``VLLM_ASCEND_DSV41_KT_SEAM_TIMING`` is on (vllm_ascend/kt_offload/
seam_timing.py), and optionally the two other instruments of §4.18:

  --ktphase FILE   the JSON ktphase.py writes as its 4th argument
                   ({"layer_median_us": {...}}): kt's own CPU compute, D.
  --nw-csv FILE    a profile's task_time.csv: NOTIFY_WAIT per layer position
                   inside a step, the same slicing nw.py does.
  --nw-median-us X one number instead of the csv (e.g. 353.6 from §4.18).

The bookkeeping this prints:

  A  device reaches the host-func node -> the callback starts running
  B  pre        callback entry -> Python pre-checks done
  C  kt_pre + submit + between   task build and handing it to the worker
  D  kt compute (from --ktphase; the seam records cannot see inside sync)
  E  sync - D   worker wakeup + completion (cross-NUMA reduce, write-back)
  F  post       trailing Python -> callback returns, then the notify to device

Only B, C, sync, F are measured directly; the rest are differences:

  E            = sync - D                        (needs --ktphase)
  A + F_device = NOTIFY_WAIT - cb                (needs a NOTIFY_WAIT source)

which is why both must come from the SAME service and the same steps -- see
tools/kt_offload/oncard/RUNBOOK_seam.md. Every difference is printed with the
inputs it was made of, and a gap that does not close is printed as a gap
rather than folded into a neighbour.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict

REC = re.compile(r"\[kt-seam\] (\{.*\})")

SEGMENTS = ("pre", "kt_pre", "submit", "between", "sync", "kt_tail", "post", "kt", "cb", "gap")
ADDITIVE = ("pre", "kt_pre", "submit", "between", "sync", "kt_tail", "post")
ADDITIVE_LUMPED = ("pre", "kt", "post")


def parse_records(text: str) -> tuple[list[dict], int]:
    """Every ``[kt-seam]`` record in a log, and how many markers did not parse.

    Records share stdout with other writers (kt's own phase timing prints from
    C++), so a line can be cut in half; that is counted, never guessed at.
    """
    good, seen = [], 0
    for line in text.splitlines():
        m = REC.search(line)
        if not m:
            seen += line.count("[kt-seam]")
            continue
        seen += 1
        try:
            good.append(json.loads(m.group(1)))
        except ValueError:
            pass
    return good, seen - len(good)


def merge(records: list[dict]) -> dict:
    """Combine the samples into one per-layer view.

    Each record already carries medians over its own interval, so this is a
    median of medians weighted by nothing -- stated plainly because it is not
    the same as the median over all callbacks, and drifts if the intervals
    have very different counts. ``n`` per layer is printed so that is visible.
    """
    per_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    p90_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[int, int] = defaultdict(int)
    split = any(bool(r.get("split")) for r in records)
    for rec in records:
        for layer_s, v in (rec.get("layers") or {}).items():
            layer = int(layer_s)
            counts[layer] += int(v.get("n", 0))
            for seg in SEGMENTS:
                if seg in (v.get("med_us") or {}):
                    per_layer[layer][seg].append(float(v["med_us"][seg]))
                if seg in (v.get("p90_us") or {}):
                    p90_layer[layer][seg].append(float(v["p90_us"][seg]))
    med = {
        layer: {seg: st.median(xs) for seg, xs in segs.items() if xs}
        for layer, segs in per_layer.items()
    }
    p90 = {
        layer: {seg: max(xs) for seg, xs in segs.items() if xs}
        for layer, segs in p90_layer.items()
    }
    return {"split": split, "median_us": med, "p90_us": p90, "n": dict(counts), "samples": len(records)}


def reconcile(med_us: dict[str, float], split: bool, tol_us: float = 2.0) -> dict:
    """B+C+D+E+F (as measured) against the callback's own duration.

    A timestamp taken in the wrong place shows up here and only here: an
    overlap or a hole in the segmentation leaves a residual that is not ~0.
    """
    keys = ADDITIVE if split else ADDITIVE_LUMPED
    parts = sum(float(med_us.get(k, 0.0)) for k in keys)
    cb = float(med_us.get("cb", 0.0))
    return {
        "parts_us": round(parts, 3),
        "cb_us": round(cb, 3),
        "residual_us": round(cb - parts, 3),
        "ok": abs(cb - parts) <= tol_us,
    }


def notify_wait_by_position(csv_path: str) -> dict[int, float]:
    """Per-layer-position NOTIFY_WAIT medians, sliced exactly as nw.py does.

    A step is the window between two MODEL_EXECUTE starts; the n-th
    NOTIFY_WAIT inside it is layer position n. Copied from nw.py on purpose:
    this script must be runnable on a log alone, and the slicing rule is the
    thing being reused, not the file.
    """
    rows = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            k = (r.get("kernel_type") or "").strip()
            if k not in ("NOTIFY_WAIT", "MODEL_EXECUTE"):
                continue
            rows[k].append((float(r["task_start(us)"].strip()), float(r["task_time(us)"])))
    starts = sorted(s for s, _ in rows["MODEL_EXECUTE"])
    pos: dict[int, list[float]] = defaultdict(list)
    per_step: dict[int, list[float]] = defaultdict(list)
    for s, d in sorted(rows["NOTIFY_WAIT"]):
        i = bisect.bisect_right(starts, s) - 1
        if i >= 0:
            per_step[i].append(d)
    for _step, ds in per_step.items():
        for j, d in enumerate(ds):
            pos[j].append(d)
    return {j: st.median(v) for j, v in pos.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="serve.log (or any file holding the [kt-seam] records)")
    ap.add_argument("--ktphase", help="JSON from ktphase.py (4th arg): kt's own per-layer compute D")
    ap.add_argument("--nw-csv", help="profile task_time.csv: NOTIFY_WAIT per layer position")
    ap.add_argument("--nw-median-us", type=float, help="one NOTIFY_WAIT median for every layer")
    ap.add_argument("--layers", type=int, default=0, help="expected layer count (0 = whatever reported)")
    ap.add_argument("--tol-us", type=float, default=2.0, help="reconciliation tolerance (default 2 us)")
    ap.add_argument("--json-out", help="write the whole decomposition here")
    args = ap.parse_args(argv)

    with open(args.log, "rb") as f:
        text = f.read().decode("utf-8", "replace")
    records, dropped = parse_records(text)
    if not records:
        print("no [kt-seam] records found -- was VLLM_ASCEND_DSV41_KT_SEAM_TIMING set?", file=sys.stderr)
        return 2
    m = merge(records)
    split, med, p90 = m["split"], m["median_us"], m["p90_us"]
    layers = sorted(med)
    print(f"[kt-seam] samples {m['samples']}  layers {len(layers)}  dropped(unparsable) {dropped}  split={split}")
    if not split:
        print("  ⚠ split=false: kt's submit/sync were not separable on this run (no CPUInfer probe);")
        print("    only pre / kt / post are measured, so C and E cannot be told apart.")
    if args.layers and len(layers) != args.layers:
        print(f"  ⚠ {len(layers)} layers reported, expected {args.layers}: a layer whose callback never ran")
        print("    is missing here, and a missing layer and a fast layer read the same in a sum.")

    ktphase = {}
    if args.ktphase:
        raw = json.load(open(args.ktphase))
        ktphase = {int(k): float(v) for k, v in (raw.get("layer_median_us") or raw).items()}
    nw = {}
    if args.nw_csv:
        nw = notify_wait_by_position(args.nw_csv)
        print(f"  NOTIFY_WAIT from {args.nw_csv}: {len(nw)} layer positions, "
              f"median-of-layers {st.median(list(nw.values())):.1f} us")
        if len(nw) != len(layers):
            print(f"  ⚠ {len(nw)} NOTIFY_WAIT positions vs {len(layers)} seam layers: the position index is")
            print("    an order within the step, not a layer id -- only aligned when every layer has a node.")

    cols = ADDITIVE if split else ADDITIVE_LUMPED
    head = "layer     n  " + "".join(f"{c:>9s}" for c in cols) + "       cb      gap"
    if ktphase:
        head += "     D_kt   E=sync-D"
    if nw or args.nw_median_us:
        head += "       NW   A+F_dev"
    print("\n" + head)
    out_layers = {}
    bad = []
    for layer in layers:
        mm = med[layer]
        row = f"{layer:5d} {m['n'].get(layer, 0):5d}  " + "".join(f"{mm.get(c, 0.0):9.2f}" for c in cols)
        row += f" {mm.get('cb', 0.0):8.2f} {mm.get('gap', 0.0):8.1f}"
        rec = reconcile(mm, split, args.tol_us)
        entry = {"med_us": mm, "p90_us": p90.get(layer, {}), "n": m["n"].get(layer, 0), "reconcile": rec}
        if not rec["ok"]:
            bad.append((layer, rec))
        d = ktphase.get(layer)
        if ktphase:
            e = (mm.get("sync", 0.0) - d) if (d is not None and split) else None
            row += f" {d if d is not None else float('nan'):8.1f} " + (f"{e:9.1f}" if e is not None else "        -")
            entry["D_kt_us"] = d
            entry["E_us"] = e
        nwv = nw.get(layer, args.nw_median_us)
        if nw or args.nw_median_us:
            af = (nwv - mm.get("cb", 0.0)) if nwv is not None else None
            row += f" {nwv if nwv is not None else float('nan'):8.1f} " + (f"{af:9.1f}" if af is not None else "        -")
            entry["NW_us"] = nwv
            entry["A_plus_F_device_us"] = af
        out_layers[layer] = entry
        print(row)

    n_layers = len(layers)
    print("\nwhole net (sum of the per-layer medians), ms per token:")
    for c in cols + ("cb",):
        print(f"  {c:9s} {sum(med[l].get(c, 0.0) for l in layers)/1000.0:7.3f}")
    if ktphase:
        dsum = sum(ktphase.get(l, 0.0) for l in layers) / 1000.0
        print(f"  {'D_kt':9s} {dsum:7.3f}")
        if split:
            esum = sum(med[l].get("sync", 0.0) - ktphase.get(l, 0.0) for l in layers if l in ktphase) / 1000.0
            print(f"  {'E':9s} {esum:7.3f}   (sync - D_kt: worker wakeup + completion/write-back)")
    if nw or args.nw_median_us:
        nwsum = sum((nw.get(l, args.nw_median_us) or 0.0) for l in layers) / 1000.0
        cbsum = sum(med[l].get("cb", 0.0) for l in layers) / 1000.0
        print(f"  {'NW':9s} {nwsum:7.3f}")
        print(f"  {'A+F_dev':9s} {nwsum - cbsum:7.3f}   (NOTIFY_WAIT - T_cb: dispatch to this thread + notify back)")
        if ktphase:
            seam = nwsum - sum(ktphase.get(l, 0.0) for l in layers) / 1000.0
            print(f"  {'seam':9s} {seam:7.3f}   (NW - D_kt: the §4.18 107.5 us/layer, now broken down above)")

    print()
    if bad:
        print(f"❌ reconciliation failed on {len(bad)}/{n_layers} layers "
              f"(|cb - sum(parts)| > {args.tol_us} us) -- the segmentation is wrong, not the machine:")
        for layer, rec in bad[:8]:
            print(f"   layer {layer}: parts {rec['parts_us']:.2f} vs cb {rec['cb_us']:.2f} "
                  f"(residual {rec['residual_us']:+.2f} us)")
        print("   a timestamp taken in the wrong place is the usual cause; do not read the table above.")
    else:
        print(f"✅ reconciliation ok on all {n_layers} layers (|cb - sum(parts)| <= {args.tol_us} us)")

    if args.json_out:
        json.dump(
            {"split": split, "samples": m["samples"], "dropped": dropped,
             "layers": {str(k): v for k, v in out_layers.items()}},
            open(args.json_out, "w"), indent=1, default=float)
        print(f"wrote {args.json_out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
