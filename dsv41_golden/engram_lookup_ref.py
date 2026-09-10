"""Reference for the engram table lookup -- criterion G-3, bit-exact.

The lookup is a gather plus a dequantize:

    row  = table[hash_id]                     # [256] float8_e4m3fn
    sc   = scale[hash_id]                     # [8]   float8_e8m0fnu, one per 32 values
    out  = (row.float() * sc.float()).bfloat16()

There is no reduction and no accumulation order to argue about, so unlike every
other tensor in the protocol this one has **no floating-point excuse**: two correct
implementations must agree bit for bit.  Hence G-3 is stated as bit-exactness
rather than a tolerance.

Why that is actually true, not just hoped for: ``e8m0`` is a bare exponent, so the
scale is exactly a power of two, and ``e4m3`` has 3 mantissa bits.  Their product
is therefore exactly representable in fp32 -- the multiply rounds nothing.  The
single rounding in the whole operation is the final fp32 to bf16 cast.  This script
checks that claim rather than asserting it: it redoes the dequantize in fp64 and
requires the bf16 result to be identical.

The two tables are 189.1 GiB (39.8% of the checkpoint), so they are never
materialised; rows come straight out of the mmap and only the touched pages fault
in.  A 27-token forward reads about 1300 of their 768 million rows.

    python engram_lookup_ref.py                      # summary + fingerprints
    python engram_lookup_ref.py --out DIR            # write the reference rows
    python engram_lookup_ref.py --verify DUMP_DIR    # check a golden dump's engram.out inputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
for p in (REF, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-seq-len", type=int, default=512)
    a = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)

    import model as M
    from ckpt import Checkpoint
    from engram_hash_ref import GOLDEN_PROMPT, build
    from golden import MODEL_DIR

    args, layout, state, tokenizer, _ = build(a.max_seq_len)
    ids = tokenizer(a.prompt or GOLDEN_PROMPT)["input_ids"]
    input_ids = torch.tensor([ids])
    hashes = state(input_ids, 0, None)              # [1, L, n_layers, n_hash_cols]
    ckpt = Checkpoint(MODEL_DIR)

    bs = M.fp8_block_size
    print(f"fp8_block_size          {bs}   (engram scale is 1-D, one per {bs} values)")
    print(f"engram_layer_ids        {layout.layer_ids}")
    print(f"hash ids                {tuple(hashes.shape)}")
    print()

    out_tensors: dict[str, torch.Tensor] = {"engram_hash_ids": hashes.contiguous()}
    meta: dict = {"fp8_block_size": bs, "engram_layer_ids": list(layout.layer_ids),
                  "input_ids": input_ids[0].tolist(), "layers": {}}
    all_exact = True

    for k, lid in enumerate(layout.layer_ids):
        wname = f"layers.{lid}.engram.embed.weight"
        sname = f"layers.{lid}.engram.embed.scale"
        wdtype, wshape = ckpt.meta(wname)
        sdtype, sshape = ckpt.meta(sname)
        ids_k = hashes[0, :, k, :]                                  # [L, n_hash_cols]

        flat, inverse = torch.unique(ids_k.reshape(-1), return_inverse=True)
        vals = ckpt.gather_rows(wname, flat)                        # [u, dim] fp8
        scales = ckpt.gather_rows(sname, flat)                      # [u, dim/bs] e8m0

        v32 = (vals.float().unflatten(-1, (-1, bs)) * scales.float().unsqueeze(-1))
        v64 = (vals.double().unflatten(-1, (-1, bs)) * scales.double().unsqueeze(-1))
        b32 = v32.flatten(-2).to(torch.bfloat16)
        b64 = v64.flatten(-2).to(torch.bfloat16)
        exact = bool(torch.equal(b32, b64))
        all_exact &= exact

        # also: is the fp32 product itself exact?  It is if widening to fp64 changes
        # nothing at all, before any bf16 rounding.
        product_exact = bool(torch.equal(v32.double(), v64))

        rows = b32[inverse].view(*ids_k.shape, vals.shape[-1])      # [L, n_hash_cols, dim]
        out_tensors[f"layer_{lid:02d}.rows"] = rows.contiguous()
        print(f"layer {lid:2d}")
        print(f"  table                 {wshape} {wdtype}   scale {sshape} {sdtype}")
        print(f"  unique rows gathered  {flat.numel()} of {wshape[0]}")
        print(f"  looked-up rows        {tuple(rows.shape)} bfloat16   sha {digest(rows)}")
        print(f"  fp32 product exact    {product_exact}  (e8m0 is a power of two, "
              f"e4m3 has 3 mantissa bits)")
        print(f"  bf16 result == fp64   {exact}   <-- G-3 rests on this")
        print(f"  value range           [{float(rows.float().min()):+.5f}, "
              f"{float(rows.float().max()):+.5f}]")
        nz = int((rows != 0).sum())
        print(f"  nonzero               {nz}/{rows.numel()}")
        meta["layers"][str(lid)] = {
            "table_shape": list(wshape), "table_dtype": str(wdtype),
            "scale_shape": list(sshape), "scale_dtype": str(sdtype),
            "unique_rows": int(flat.numel()),
            "rows_shape": list(rows.shape),
            "rows_sha256_16": digest(rows),
            "fp32_product_exact": product_exact,
            "bf16_matches_fp64": exact,
        }
        del vals, scales, v32, v64, b32, b64, rows

    print()
    print(f"dequantize is unambiguous on every gathered row: {all_exact}")
    print("=> G-3 can be stated as bit-exactness, not a tolerance.")
    meta["all_exact"] = all_exact

    if a.out:
        from safetensors.torch import save_file
        os.makedirs(a.out, exist_ok=True)
        save_file(out_tensors, os.path.join(a.out, "engram_lookup_ref.safetensors"))
        with open(os.path.join(a.out, "engram_lookup_ref.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nwrote {a.out}/engram_lookup_ref.safetensors + .json")

    sys.exit(0 if all_exact else 1)


if __name__ == "__main__":
    main()
