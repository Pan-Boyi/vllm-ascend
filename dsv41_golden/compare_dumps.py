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

# Per-tensor thresholds, because one global number cannot serve both an fp32
# quantity like ``hc_mixes`` and a bf16 one like ``moe.out``: bf16's relative ULP
# is 2**-8 = 3.9e-3, so any threshold below ~5e-3 on a bf16 tensor is measuring
# rounding, not correctness.  Running the old global strict=5e-3 against a
# known-good reference (fp64 accumulator, routing and layer inputs both forced)
# "failed" 26 of 40 layers for exactly that reason.
#
# ``isolated`` is derived from that run -- the measured maximum over 40 layers,
# times 2-3x headroom.  See ALIGNMENT_PROTOCOL.md section 6 for the table and for
# why these are a FLOOR on agreement, not a prediction of what an NPU will hit.
TOLERANCES = {
    "loose":    {"_default": (5e-2, 0.995)},
    "default":  {"_default": (2e-2, 0.9990),
                 "moe.out": (3e-2, 0.9985), "attn.out": (3e-2, 0.9985),
                 "hc_mixes": (5e-3, 0.9999)},
    "strict":   {"_default": (5e-3, 0.9999),
                 "moe.out": (1e-2, 0.9995)},
    "isolated": {"_default": (5e-3, 0.9999),
                 "layer.out": (1.5e-2, 0.9999), "hc_post": (1.5e-2, 0.9999),
                 "attn.out": (1.5e-2, 0.9999),
                 "moe.out": (4e-2, 0.9995),
                 "hc_mixes": (5e-3, 0.9999),
                 "logits": (1e-2, 0.9999)},
}

# Diagnostic-only tensors: reported, never a failure.  ``moe_routing_margin`` is a
# difference of two adjacent selection scores, so its relative noise is large by
# construction (measured up to 1.25e-2 with no bug present); it exists to tell a
# near-tie from a real routing bug, not to be a threshold.
NEVER_GATE = ("moe_routing_margin",)


def tol_for(name: str, tol: dict) -> tuple:
    """Longest-prefix match on the tensor name, falling back to ``_default``."""
    best, best_len = tol["_default"], -1
    for key, val in tol.items():
        if key != "_default" and name.startswith(key) and len(key) > best_len:
            best, best_len = val, len(key)
    return best
# Elements below this fraction of the tensor's RMS are excluded from max-rel-error:
# a near-zero golden value makes relative error meaningless.
REL_FLOOR_FRAC = 1e-2


# Expert selection is a *set*, not a sequence: MoE sums over the chosen experts and
# the golden accumulates them in expert-id order (``for i in range(n_routed_experts)``),
# so the order ``topk`` happens to return them in changes nothing.  Comparing these
# element-wise counts pure order swaps as routing flips -- which is how the
# fp32-vs-fp64 experiment came to be reported as 7.4%/55.6% when the real figures
# are 0%/22.2% (9 of the 15 "flipped" tokens at layer 2 were order swaps only).
SET_VALUED = ("moe_gate.out.1", "moe_gate.forced.1")


def routing_agreement(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Per-token set comparison of two [n_tokens, topk] expert selections."""
    ai, bi = a.to(torch.int64), b.to(torch.int64)
    same_set = (ai.sort(-1).values == bi.sort(-1).values).all(-1)
    same_seq = (ai == bi).all(-1)
    n = int(same_set.numel())
    return {"kind": "routing", "n": n,
            "exact": bool(same_set.all()),
            "tokens_set_differ": int((~same_set).sum()),
            "tokens_order_only": int((same_set & ~same_seq).sum()),
            "match_frac": float(same_set.sum()) / max(n, 1),
            "differing_tokens": (~same_set).nonzero().flatten().tolist()[:16]}


def metrics(a: torch.Tensor, b: torch.Tensor, name: str = "") -> dict:
    """a = candidate, b = golden."""
    if name in SET_VALUED and a.dim() == 2:
        return routing_agreement(a, b)
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

    # Per-token view.  The dominant divergence path in this model is one token
    # crossing a bf16 rounding boundary and causal attention then spreading it to
    # every later token, so "how many tokens are still bit-identical" separates a
    # local arithmetic difference from one that has already been broadcast.  A
    # whole-tensor rel_l2 hides that completely.
    if a.dim() >= 3 and a.shape[0] == 1 and b.shape[0] == 1:
        av = a.double().reshape(a.shape[1], -1)
        bv = b.double().reshape(b.shape[1], -1)
        tok_exact = int((av == bv).all(-1).sum())
        bn_tok = bv.norm(dim=-1)
        rel_tok = (av - bv).norm(dim=-1) / bn_tok.clamp(min=1e-30)
        out["tokens"] = int(av.shape[0])
        out["tokens_bit_exact"] = tok_exact
        out["max_token_rel_l2"] = float(rel_tok.max())
    return out


def compare_dir(golden: str, cand: str, tol: dict, verbose: bool = True,
                force_route: str = "off"):
    """``force_route`` comes from the candidate's manifest (see ROUTING_INJECTION.md).

    With routing forced, the candidate's *own* gate output is expected to disagree
    with the golden -- that disagreement is the thing being designed out, not a
    failure -- so it drops to informational.  What becomes a hard gate instead is
    that the injection actually landed: ``moe_gate.forced.1`` must equal the
    golden's selection.  Without that check a typo'd env var reads as a broad
    numerical failure, i.e. exactly like an operator bug.
    """
    tags = sorted(f[:-len(".safetensors")] for f in os.listdir(golden)
                  if f.endswith(".safetensors"))
    results, failures, missing, notes = {}, [], [], []
    forced = force_route.startswith("moe")

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
            m = metrics(ct, gt, name)
            tag_res[name] = m
            if m["kind"] == "routing":
                if m["exact"]:
                    pass
                elif forced and name == "moe_gate.out.1":
                    # informational: the candidate's free-running gate, which the
                    # injection deliberately overrode
                    notes.append((tag, name,
                                  f"{m['tokens_set_differ']}/{m['n']} tokens route elsewhere "
                                  f"(expected under force_route={force_route})"))
                else:
                    failures.append((tag, name,
                                     f"{m['tokens_set_differ']}/{m['n']} tokens select a "
                                     f"different expert SET "
                                     f"(+{m['tokens_order_only']} order-only, not counted); "
                                     f"first: {m['differing_tokens'][:6]}"))
            elif m["kind"] == "int":
                if not m["exact"]:
                    failures.append((tag, name, f"{100 * (1 - m['match_frac']):.2f}% indices differ"))
            else:
                lim_rel, lim_cos = tol_for(name, tol)
                m["limit_rel_l2"], m["limit_cos"] = lim_rel, lim_cos
                if m["nonfinite"]:
                    failures.append((tag, name, f"{m['nonfinite']} non-finite values"))
                elif name.startswith(NEVER_GATE):
                    if m["rel_l2"] > lim_rel:
                        notes.append((tag, name,
                                      f"rel_l2={m['rel_l2']:.3e} (diagnostic only, not gated)"))
                elif m["rel_l2"] > lim_rel or m["cos"] < lim_cos:
                    failures.append((tag, name,
                                     f"rel_l2={m['rel_l2']:.3e} (limit {lim_rel:.1e}) "
                                     f"cos={m['cos']:.6f} (limit {lim_cos})"))
        # Did the injection actually reach the experts?  Cross-name on purpose: the
        # candidate records what it fed the experts, the golden records what it chose.
        if "moe_gate.forced.1" in c and "moe_gate.out.1" in g:
            r = routing_agreement(c["moe_gate.forced.1"], g["moe_gate.out.1"])
            tag_res["moe_gate.forced.1_vs_golden"] = r
            if not r["exact"]:
                failures.append((tag, "moe_gate.forced.1",
                                 f"INJECTION DID NOT LAND: {r['tokens_set_differ']}/{r['n']} "
                                 "tokens went to the experts with a different expert set than "
                                 "the golden chose. Check the plan path and FORCE_ROUTE before "
                                 "looking at any operator."))
        elif forced and "moe_gate.out.1" in g and "moe_gate.forced.1" not in c:
            notes.append((tag, "moe_gate.forced.1",
                          "absent: cannot verify the injection landed (see ROUTING_INJECTION.md s3)"))

        results[tag] = tag_res

        if verbose:
            # Rank by rel_l2 relative to that tensor's own limit, not by raw
            # rel_l2: moe.out is always the largest number and would otherwise
            # always be named "worst" even when it is well inside tolerance.
            worst = max((v for v in tag_res.values() if v.get("kind") == "float"),
                        key=lambda v: v.get("rel_l2", 0) / max(v.get("limit_rel_l2", 1), 1e-30),
                        default=None)
            if worst is not None:
                wname = [k for k, v in tag_res.items() if v is worst][0]
                flag = "FAIL" if any(f[0] == tag for f in failures) else "ok  "
                tokinfo = ""
                if "tokens_bit_exact" in worst:
                    tokinfo = (f" tok_exact={worst['tokens_bit_exact']}/{worst['tokens']}"
                               f" max_tok={worst['max_token_rel_l2']:.2e}")
                print(f"  {flag} {tag:12s} worst={wname:22s} "
                      f"rel_l2={worst['rel_l2']:.3e} cos={worst['cos']:.6f}{tokinfo}")
    return results, failures, missing, notes


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
    dflt = tol["_default"]
    extra = ", ".join(f"{k}<={v[0]:.1e}" for k, v in tol.items() if k != "_default")
    print(f"tolerance {a.tolerance}: default rel_l2 <= {dflt[0]:.0e}, cos >= {dflt[1]}"
          + (f"\n          per-tensor: {extra}" if extra else ""))

    force_route = "off"
    for d in (a.golden, a.candidate):
        mp = os.path.join(d, "manifest.json")
        if os.path.exists(mp):
            with open(mp) as f:
                man = json.load(f)
            print(f"  {os.path.basename(d)}: input_ids={man.get('input_ids')} "
                  f"layers={man.get('n_layers_run')} accum={man.get('accum_dtype')} "
                  f"force_route={man.get('force_route', 'off')}")
            if d == a.candidate:
                force_route = man.get("force_route", "off") or "off"

    if force_route != "off":
        print(f"\nforced routing ({force_route}): the candidate's own moe_gate.out.1 is "
              f"informational;\nthe gate is moe_gate.forced.1 == golden's selection.")
    print()
    results, failures, missing, notes = compare_dir(a.golden, a.candidate, tol,
                                                    force_route=force_route)

    print()
    if notes:
        print(f"{len(notes)} informational (not failures):")
        for tag, name, why in notes[:10]:
            print(f"  {tag}/{name}: {why}")
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
            json.dump({"tolerance": a.tolerance, "force_route": force_route,
                       "results": results,
                       "failures": [list(x) for x in failures],
                       "notes": [list(x) for x in notes], "missing": missing}, f, indent=2)
        print(f"wrote {a.json}")
    sys.exit(verdict)


if __name__ == "__main__":
    main()
