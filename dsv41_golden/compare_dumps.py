"""Compare an Ascend NPU activation dump against the CPU golden.

    python compare_dumps.py GOLDEN_DIR CANDIDATE_DIR
    python compare_dumps.py GOLDEN_DIR CANDIDATE_DIR --tolerance strict --json out.json

Both directories hold the per-tag safetensors files written by ``golden.Recorder``
(``prologue``, ``layer_00`` .. ``layer_39``, ``epilogue``) plus ``manifest.json``.
See ALIGNMENT_PROTOCOL.md for what to dump and what the thresholds mean.

The tool reports, per tensor: relative L2 error, max relative error over elements
above a magnitude floor, and cosine similarity -- then names the *first* layer
that fails, which is almost always the one with the bug.  Later layers inherit
upstream error, so only the first failure is diagnostic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file

# name -> (rel_l2 max, cosine min).  Tightened by --tolerance strict, loosened by loose.
TOLERANCES = {
    "loose":   {"rel_l2": 5e-2, "cos": 0.995},
    "default": {"rel_l2": 2e-2, "cos": 0.9990},
    "strict":  {"rel_l2": 5e-3, "cos": 0.9999},
}
# Elements below this fraction of the tensor's RMS are excluded from max-rel-error:
# a near-zero golden value makes relative error meaningless.
REL_FLOOR_FRAC = 1e-2


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a = candidate, b = golden."""
    if a.dtype in (torch.int32, torch.int64, torch.int16, torch.uint8, torch.bool):
        same = int((a == b).sum())
        total = a.numel()
        return {"kind": "int", "exact": same == total, "match_frac": same / max(total, 1),
                "n": total}
    # float64 throughout: a 5120-wide bf16 vector dotted with itself in fp32 lands
    # at cos = 0.999993, which would look like a real (small) divergence at strict
    # tolerance. The metric must be sharper than the thing it measures.
    af, bf = a.double().flatten(), b.double().flatten()
    finite = torch.isfinite(af) & torch.isfinite(bf)
    out = {"kind": "float", "n": af.numel(),
           "nonfinite": int((~finite).sum())}
    af, bf = af[finite], bf[finite]
    if af.numel() == 0:
        out.update(rel_l2=float("nan"), max_rel=float("nan"), cos=float("nan"))
        return out
    diff = af - bf
    bn = bf.norm()
    out["rel_l2"] = float(diff.norm() / bn) if bn > 0 else float(diff.norm())
    rms = float(bf.pow(2).mean().sqrt())
    keep = bf.abs() > max(rms * REL_FLOOR_FRAC, 1e-30)
    out["max_rel"] = float((diff[keep].abs() / bf[keep].abs()).max()) if keep.any() else 0.0
    denom = af.norm() * bf.norm()
    out["cos"] = float((af @ bf) / denom) if denom > 0 else 1.0
    out["golden_rms"] = rms
    return out


def compare_dir(golden: str, cand: str, tol: dict, verbose: bool = True):
    tags = sorted(f[:-len(".safetensors")] for f in os.listdir(golden)
                  if f.endswith(".safetensors"))
    results, failures, missing = {}, [], []

    for tag in tags:
        gpath = os.path.join(golden, tag + ".safetensors")
        cpath = os.path.join(cand, tag + ".safetensors")
        if not os.path.exists(cpath):
            missing.append(tag)
            continue
        g, c = load_file(gpath), load_file(cpath)
        tag_res = {}
        for name, gt in g.items():
            if name not in c:
                missing.append(f"{tag}/{name}")
                continue
            ct = c[name]
            if tuple(ct.shape) != tuple(gt.shape):
                tag_res[name] = {"kind": "shape", "golden": list(gt.shape),
                                 "candidate": list(ct.shape)}
                failures.append((tag, name, "shape mismatch"))
                continue
            m = metrics(ct, gt)
            tag_res[name] = m
            if m["kind"] == "int":
                if not m["exact"]:
                    failures.append((tag, name, f"{100 * (1 - m['match_frac']):.2f}% indices differ"))
            else:
                if m["nonfinite"]:
                    failures.append((tag, name, f"{m['nonfinite']} non-finite values"))
                elif m["rel_l2"] > tol["rel_l2"] or m["cos"] < tol["cos"]:
                    failures.append((tag, name,
                                     f"rel_l2={m['rel_l2']:.3e} cos={m['cos']:.6f}"))
        results[tag] = tag_res

        if verbose:
            worst = max((v for v in tag_res.values() if v.get("kind") == "float"),
                        key=lambda v: v.get("rel_l2", 0), default=None)
            if worst is not None:
                wname = [k for k, v in tag_res.items() if v is worst][0]
                flag = "FAIL" if any(f[0] == tag for f in failures) else "ok  "
                print(f"  {flag} {tag:12s} worst={wname:22s} "
                      f"rel_l2={worst['rel_l2']:.3e} cos={worst['cos']:.6f}")
    return results, failures, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("candidate")
    ap.add_argument("--tolerance", choices=list(TOLERANCES), default="default")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    tol = TOLERANCES[a.tolerance]
    print(f"golden    {a.golden}")
    print(f"candidate {a.candidate}")
    print(f"tolerance {a.tolerance}: rel_l2 <= {tol['rel_l2']:.0e}, cos >= {tol['cos']}")

    for d in (a.golden, a.candidate):
        mp = os.path.join(d, "manifest.json")
        if os.path.exists(mp):
            with open(mp) as f:
                man = json.load(f)
            print(f"  {os.path.basename(d)}: input_ids={man.get('input_ids')} "
                  f"layers={man.get('n_layers_run')}")

    print()
    results, failures, missing = compare_dir(a.golden, a.candidate, tol)

    print()
    if missing:
        print(f"MISSING from candidate ({len(missing)}): {missing[:10]}")
    if failures:
        print(f"{len(failures)} FAILURES. First divergence is the one to debug:")
        for tag, name, why in failures[:15]:
            print(f"  {tag}/{name}: {why}")
        verdict = 1
    else:
        print("PASS: every dumped tensor is within tolerance")
        verdict = 0

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"tolerance": a.tolerance, "results": results,
                       "failures": [list(x) for x in failures], "missing": missing}, f, indent=2)
        print(f"wrote {a.json}")
    sys.exit(verdict)


if __name__ == "__main__":
    main()
