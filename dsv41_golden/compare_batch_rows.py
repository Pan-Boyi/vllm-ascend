"""Localise where two identical sequences in one batch stop agreeing.

``run_golden.py --batch N`` replicates the prompt, so every row carries the same
tokens and every row must produce the same output.  At 3 layers they do, bit for
bit; at 40 they do not, and the logits differ by 3.9 on a scale of about 15 --
far too much to be rounding.

This walks the dump layer by layer and names the first tensor where row 0 and
row 1 part company, which distinguishes the two possible causes:

* a real batch-handling bug -- a cache slot indexed on the wrong dimension, a
  broadcast where a gather was needed -- which would show up as a difference in a
  tensor that has no business differing, and often as a *large* difference
  immediately rather than a growing one;
* the MoE discrete verdict again -- with batch 2 the expert GEMM sees 54 rows
  instead of 27, so its rounding differs, and one near-tie token routes elsewhere,
  giving an O(1) divergence downstream.  In that case the first divergence is a
  tiny one in an activation, followed by a routing flip.

The distinguishing evidence is ``moe_gate.out.1``: if the two halves select
different experts, the second story holds, and the check to keep is "same routing
implies same output" rather than plain bit-equality.

    python compare_batch_rows.py DUMP_DIR [--layers 40]
"""

from __future__ import annotations

import argparse
import os

import torch
from safetensors.torch import load_file

# Tensors whose leading axis is batch, versus the flat [batch*seqlen, ...] gate
# tensors that need splitting instead of indexing.
FLAT_KEYS = ("moe_gate.out.0", "moe_gate.out.1", "moe_gate.forced.0",
             "moe_gate.forced.1", "moe_routing_margin")


def split_rows(name: str, t: torch.Tensor, batch: int):
    """Return (row0, row1) for a tensor, or None if it carries no batch axis."""
    if name in FLAT_KEYS:
        if t.shape[0] % batch:
            return None
        n = t.shape[0] // batch
        return t[:n], t[n:2 * n]
    if t.dim() >= 2 and t.shape[0] == batch:
        return t[0], t[1]
    return None                      # e.g. indexer_kv_offset, a scalar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()

    first_diff = None
    routing_diff = []
    print(f"{'layer':>5}  {'first tensor where row 0 != row 1':<24} {'max|diff|':>11} "
          f"{'rel_l2':>10}  {'routing same?':>13}")
    for i in range(a.layers):
        p = os.path.join(a.dump_dir, f"layer_{i:02d}.safetensors")
        if not os.path.exists(p):
            continue
        blob = load_file(p)
        layer_first, layer_first_stats = None, None
        route_same = None
        # Deliberately in execution order, so "first" means first in the layer.
        order = ["layer.in", "pre_mix.in", "hc_mixes.0.0", "hc_mixes.0.1",
                 "hc_mixes.0.2", "hc_pre.0", "attn_norm.out", "compressor.out",
                 "indexer_score", "indexer_topk.out", "attn.o_raw",
                 "attn.o_unroped", "attn.out", "hc_post.0", "hc_mixes.1.0",
                 "hc_mixes.1.1", "hc_mixes.1.2", "hc_pre.1", "ffn_norm.out",
                 "moe.in", "moe_routing_margin", "moe_gate.out.0",
                 "moe_gate.out.1", "moe.out", "hc_post.1", "layer.out"]
        for name in order:
            if name not in blob:
                continue
            pair = split_rows(name, blob[name], a.batch)
            if pair is None:
                continue
            r0, r1 = pair
            if name == "moe_gate.out.1":
                # Compare as sets: top-k order does not affect the result.
                s0 = r0.to(torch.int64).sort(-1).values
                s1 = r1.to(torch.int64).sort(-1).values
                route_same = bool(torch.equal(s0, s1))
                if not route_same:
                    bad = (~(s0 == s1).all(-1)).nonzero().flatten().tolist()
                    routing_diff.append((i, bad))
                continue
            if torch.equal(r0, r1):
                continue
            d = (r0.float() - r1.float()).abs()
            rel = float(d.norm() / max(r0.float().norm().item(), 1e-30))
            if layer_first is None:
                layer_first, layer_first_stats = name, (float(d.max()), rel)
        if layer_first is not None and first_diff is None:
            first_diff = (i, layer_first, layer_first_stats)
        if layer_first is not None or route_same is False:
            mx, rel = layer_first_stats if layer_first_stats else (0.0, 0.0)
            print(f"{i:>5}  {layer_first or '(only routing)':<24} {mx:>11.3e} "
                  f"{rel:>10.3e}  {str(route_same):>13}")

    print()
    if first_diff is None:
        print("row 0 and row 1 are bit-identical on every dumped tensor")
    else:
        i, name, (mx, rel) = first_diff
        print(f"FIRST divergence: layer {i}, {name}, max|diff|={mx:.3e} rel_l2={rel:.3e}")
        print()
        if rel < 1e-2 and mx < 1e-1:
            print("Small and in an activation -> consistent with the MoE story:")
            print("  batch 2 changes the expert GEMM's M, its rounding differs, and a")
            print("  near-tie token later routes elsewhere. Not a batch-handling bug.")
        else:
            print("Large at first appearance -> consistent with a batch-handling bug:")
            print("  a tensor that has no business differing differs immediately.")
    if routing_diff:
        print()
        print(f"layers where the two halves select different experts: "
              f"{[i for i, _ in routing_diff]}")
        i, bad = routing_diff[0]
        print(f"  first at layer {i}, tokens {bad[:8]}")
        print("  => if the first activation divergence is tiny and precedes this,")
        print("     the divergence is the discrete verdict, not the batching.")
    else:
        print()
        print("routing identical between the two halves on every layer")


if __name__ == "__main__":
    main()
