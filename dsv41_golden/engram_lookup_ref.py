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
in.  A 27-token forward reads 648 of their 384 million rows per layer.

Read path
---------
``Checkpoint.gather_rows`` slices the mmap as raw ``uint8`` and reinterprets with
``.view(dtype)``, so no value ever passes through a conversion before the
dequantize.  That is a requirement, not an accident: see the signed-zero note
below for what a normalising read path would cost.

🔴 Do NOT dequantize with ``ckpt.dequant_fp8_2d``
-------------------------------------------------
That helper is for the *weights*, whose scale is a 2-D block grid, and it
``repeat_interleave``s along both axes.  The engram scale is **1-D per row**:
shape ``[nrows, 8]``, one exponent per 32 values along ``dim`` only.  Feeding
engram tensors to the 2-D helper silently produces a wrong answer of the right
shape.

🔴 Compare bytes, but modulo signed zero
----------------------------------------
Measured on this checkpoint (see ``--scan``):

* **e4m3 NaN never occurs.** Of the 256 possible bytes, 254 appear in a 600k-row
  stripe of each table -- the two that never appear are exactly ``0x7F`` and
  ``0xFF``, the NaN patterns.  So the usual ``NaN != NaN`` false-alarm hazard is
  **absent here**, and a value comparison is not endangered from that direction.
* **Negative zero does occur**, ~174 bytes per 200k rows, i.e. about one row in
  1150 carries one.  A 27-token forward gathers 648 rows per layer, so it shows
  up in ordinary runs, not just in constructed cases.  ``-0.0`` fp8 times a
  power of two is ``-0.0``, which reaches the bf16 output as ``0x8000``.

Those two facts point opposite ways, and the second is the one that bites:

    an implementation that normalises -0 to +0 is **value-equal and byte-unequal**

so a bare byte comparison raises a false alarm on an implementation that is
numerically perfect.  And ``-0.0`` can never change a downstream result: the
lookup output is only ever added and multiplied, and ``-0.0 + x == +0.0 + x`` and
``-0.0 * x == -0.0`` compare equal for every ``x``.

⇒ **G-3's comparison is byte equality with ±0 mismatches counted and forgiven.**
``compare()`` implements exactly that; it reports the ±0 count so a *systematic*
sign-of-zero difference is still visible, it just is not a failure.

    python engram_lookup_ref.py                        # summary + fingerprints
    python engram_lookup_ref.py --out DIR              # write the reference
    python engram_lookup_ref.py --verify FILE          # check an NPU-produced lookup
    python engram_lookup_ref.py --scan                 # redo the byte-pattern survey
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

# e4m3fn's two NaN encodings and e8m0fnu's one, kept as bytes because that is the
# only form in which they can be looked for without a conversion normalising them.
E4M3_NAN_BYTES = (0x7F, 0xFF)
E8M0_NAN_BYTE = 0xFF


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# the reference itself -- import these three
# ---------------------------------------------------------------------------

def lookup_raw(ck, layer_id: int, hash_ids: torch.Tensor):
    """Gather the raw quantized rows for ``hash_ids``, without dequantizing.

    Returns ``(fp8_rows, scale_rows)`` shaped ``[*hash_ids.shape, dim]`` and
    ``[*hash_ids.shape, dim // block]``, still in their fp8 dtypes -- i.e. still
    the checkpoint's own bytes.

    Dumping these alongside the dequantized result is what separates the two ways
    the lookup can be wrong: **wrong rows** (hash, indexing, or sharding) from
    **wrong dequantize** (scale layout, block size, rounding).  Without them a
    mismatch only says "the output differs".
    """
    wname = f"layers.{layer_id}.engram.embed.weight"
    sname = f"layers.{layer_id}.engram.embed.scale"
    flat, inverse = torch.unique(hash_ids.reshape(-1), return_inverse=True)
    vals = ck.gather_rows(wname, flat)                  # [u, dim]        e4m3
    scales = ck.gather_rows(sname, flat)                # [u, dim/block]  e8m0
    return (vals[inverse].view(*hash_ids.shape, vals.shape[-1]),
            scales[inverse].view(*hash_ids.shape, scales.shape[-1]))


def dequantize(fp8_rows: torch.Tensor, scale_rows: torch.Tensor, block: int = 32,
               check_exact: bool = False) -> torch.Tensor:
    """``[..., dim]`` e4m3 + ``[..., dim/block]`` e8m0 -> ``[..., dim]`` bf16.

    The scale is 1-D: one exponent per ``block`` values along the last axis only.
    With ``check_exact`` the fp32 product is re-derived in fp64 and the two bf16
    results are required to be identical, which is the claim G-3 rests on.
    """
    assert fp8_rows.shape[-1] == scale_rows.shape[-1] * block, (
        f"scale is 1-D per row: expected dim {scale_rows.shape[-1] * block}, "
        f"got {fp8_rows.shape[-1]} -- did you pass a 2-D block scale?"
    )
    v32 = fp8_rows.float().unflatten(-1, (-1, block)) * scale_rows.float().unsqueeze(-1)
    out = v32.flatten(-2).to(torch.bfloat16)
    if check_exact:
        v64 = fp8_rows.double().unflatten(-1, (-1, block)) * scale_rows.double().unsqueeze(-1)
        assert torch.equal(v32.double(), v64), "fp32 product was not exact"
        assert torch.equal(out, v64.flatten(-2).to(torch.bfloat16)), \
            "bf16 result depends on the accumulator width"
    return out


def lookup(ck, layer_id: int, hash_ids: torch.Tensor, block: int = 32) -> torch.Tensor:
    """``hash_ids [B, L, n_hash_cols]`` -> ``[B, L, n_hash_cols, dim]`` bf16.

    Shape-preserving in ``hash_ids``, so it works for a single token, a batch, or
    the flattened row layout an NPU kernel might hand over.
    """
    fp8_rows, scale_rows = lookup_raw(ck, layer_id, hash_ids)
    return dequantize(fp8_rows, scale_rows, block)


def compare(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """Byte equality with ±0 mismatches counted and forgiven -- the G-3 criterion.

    See the module docstring: bytes are the right comparison (no floating-point
    excuse exists here), except that ``-0.0`` vs ``+0.0`` is value-identical,
    unreachable-to-fix from downstream, and does occur in this checkpoint.
    """
    if ref.shape != got.shape:
        return {"pass": False, "reason": f"shape {tuple(got.shape)} != {tuple(ref.shape)}"}
    if ref.dtype != got.dtype:
        return {"pass": False, "reason": f"dtype {got.dtype} != {ref.dtype}"}

    byte_diff = _bytes_differ(ref, got)          # element-wise, dtype-agnostic
    byte_identical = not bool(byte_diff.any())
    if byte_identical:
        return {"pass": True, "byte_identical": True, "signed_zero": 0, "diff": 0,
                "reason": "byte-identical"}

    # An element passes if its BYTES match or its VALUE matches.  One condition
    # covers both hazards: NaN (bytes match, value compare would fail) and signed
    # zero (value matches, byte compare would fail).  Nothing else in bf16 is
    # value-equal with differing bytes, so "value-equal but byte-different" *is*
    # the signed-zero count.
    value_same = ref == got
    forgiven = int((value_same & byte_diff).sum())
    real = int((byte_diff & ~value_same).sum())
    rows = ref.shape[:-1]
    touched = int(byte_diff.view(*rows, -1).any(-1).sum()) if rows else int(byte_diff.sum())
    return {"pass": real == 0, "byte_identical": False, "signed_zero": forgiven,
            "diff": real, "rows_touched": touched,
            "reason": (f"differs only in the sign of zero ({forgiven} elements)"
                       if real == 0 else
                       f"{real} elements differ in value ({touched} rows touched)")}


def _bytes_differ(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise byte inequality for a same-dtype pair."""
    n = a.element_size()
    ab = a.contiguous().view(torch.uint8).view(*a.shape, n)
    bb = b.contiguous().view(torch.uint8).view(*b.shape, n)
    return (ab != bb).any(-1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def byte_scan(ck, rows_per_stripe: int = 200000) -> dict:
    """Survey the actual byte patterns in the tables -- the basis of the ±0 rule."""
    import numpy as np
    out = {}
    for lid in (1, 14):
        for nm, kind in ((f"layers.{lid}.engram.embed.weight", "e4m3"),
                         (f"layers.{lid}.engram.embed.scale", "e8m0")):
            dt, sh = ck.meta(nm)
            N = sh[0]
            idx = torch.cat([torch.arange(0, rows_per_stripe),
                             torch.arange(N // 2, N // 2 + rows_per_stripe),
                             torch.arange(N - rows_per_stripe, N)])
            raw = ck.gather_rows(nm, idx).view(torch.uint8).numpy()
            cnt = np.bincount(raw.reshape(-1), minlength=256)
            rec = {"dtype": str(dt), "shape": list(sh), "bytes_scanned": int(raw.size),
                   "distinct_bytes": int((cnt > 0).sum())}
            if kind == "e4m3":
                rec["nan_bytes"] = int(sum(cnt[b] for b in E4M3_NAN_BYTES))
                rec["pos_zero"] = int(cnt[0x00])
                rec["neg_zero"] = int(cnt[0x80])
            else:
                rec["nan_bytes"] = int(cnt[E8M0_NAN_BYTE])
                nz = np.nonzero(cnt)[0]
                rec["exponents"] = [f"2**{int(x) - 127}" for x in nz]
            out[nm] = rec
            print(f"{nm}\n  {rec}")
            del raw
    return out


def dup_test(ck, layer_ids, block: int) -> bool:
    """Repeated hash ids -- the input shape upstream is known to get wrong quietly.

    A row of 24 hash columns has no reason to hold distinct ids, and in practice it
    does not: the standard prompt already repeats.  Upstream's scale gather returns
    wrong values for repeated indices **without erroring** (maxdiff 32 as measured by
    the kernel line), and 32 is roughly seven times the entire legitimate value range
    of this table, so the fault is enormous and silent at the same time.

    Checked as a property against an independent naive per-row gather, not against
    this module's own dedup path -- ``lookup_raw`` deduplicates with
    ``unique(return_inverse=True)``, so comparing it to itself would prove nothing.
    """
    print("duplicate hash ids -- the shape upstream mishandles silently")
    ids = torch.tensor([[[101, 202, 303, 101, 404, 202, 505, 101,
                          606, 707, 101, 808, 303, 909, 101, 111,
                          222, 202, 333, 444, 555, 666, 777, 888]]], dtype=torch.int64)
    n_uniq = int(torch.unique(ids).numel())
    print(f"  {ids.numel()} columns, {n_uniq} distinct ids, "
          f"{ids.numel() - n_uniq} repeats (one id appears 5 times)")
    ok = True
    for lid in layer_ids:
        fp8, sc = lookup_raw(ck, lid, ids)
        rows = dequantize(fp8, sc, block)
        # independent reference: one gather_rows call per column, no dedup anywhere
        flat = ids.flatten()
        nf = torch.stack([ck.gather_rows(f"layers.{lid}.engram.embed.weight",
                                         torch.tensor([int(i)]))[0] for i in flat])
        ns = torch.stack([ck.gather_rows(f"layers.{lid}.engram.embed.scale",
                                         torch.tensor([int(i)]))[0] for i in flat])
        f_ok = torch.equal(fp8.reshape(-1, fp8.shape[-1]).view(torch.uint8),
                           nf.view(torch.uint8))
        s_ok = torch.equal(sc.reshape(-1, sc.shape[-1]).view(torch.uint8),
                           ns.view(torch.uint8))
        # every occurrence of one id must give byte-identical results
        rb = rows.reshape(-1, rows.shape[-1]).view(torch.uint8)
        fb = fp8.reshape(-1, fp8.shape[-1]).view(torch.uint8)
        sb = sc.reshape(-1, sc.shape[-1]).view(torch.uint8)
        consistent = True
        for uid in torch.unique(flat):
            pos = (flat == uid).nonzero().flatten()
            if len(pos) < 2:
                continue
            for tb in (rb, fb, sb):
                if not all(torch.equal(tb[p], tb[pos[0]]) for p in pos):
                    consistent = False
        rng = float(rows.float().abs().max())
        ok &= f_ok and s_ok and consistent
        print(f"  layer {lid:2d}: fp8 == naive per-row gather {f_ok}   "
              f"scale == naive {s_ok}   repeats byte-consistent {consistent}")
        print(f"            max|value| {rng:.4f} -- upstream's 32 would be "
              f"{32 / max(rng, 1e-9):.0f}x the legitimate range, so a plain range "
              f"check also catches it")
    print()
    print("PASS" if ok else "FAIL")
    if ok:
        print("  The reference deduplicates with unique(return_inverse=True) and the")
        print("  result matches a naive per-row gather byte for byte, so it can serve")
        print("  as the golden for this case.  An NPU implementation that fails it will")
        print("  show up as a SCALE mismatch, not a gather or dequantize one -- see the")
        print("  three-bucket attribution in --verify.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", default=None,
                    help="a safetensors written by --out, or an NPU-produced one with "
                         "the same keys; compared byte-wise modulo signed zero")
    ap.add_argument("--scan", action="store_true", help="redo the byte-pattern survey")
    ap.add_argument("--dup-test", action="store_true",
                    help="check repeated hash ids, the one input shape known to be "
                         "mishandled silently upstream")
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
    hashes = state(input_ids, 0, None)              # [B, L, n_engram_layers, n_hash_cols]
    ckpt = Checkpoint(MODEL_DIR)

    bs = M.fp8_block_size
    print(f"fp8_block_size          {bs}   (engram scale is 1-D, one per {bs} values)")
    print(f"engram_layer_ids        {layout.layer_ids}")
    print(f"hash ids                {tuple(hashes.shape)}   [B, L, n_engram_layers, n_hash_cols]")
    print()

    if a.dup_test:
        ok = dup_test(ckpt, layout.layer_ids, bs)
        ckpt.close()
        sys.exit(0 if ok else 1)

    if a.scan:
        scan = byte_scan(ckpt)
        print()
    else:
        scan = None

    out_tensors: dict[str, torch.Tensor] = {"engram_hash_ids": hashes.contiguous()}
    meta: dict = {"fp8_block_size": bs, "engram_layer_ids": list(layout.layer_ids),
                  "input_ids": input_ids[0].tolist(), "layers": {},
                  "criterion": "byte equality modulo signed zero (see module docstring)"}
    if scan:
        meta["byte_scan"] = scan
    all_exact = True

    for k, lid in enumerate(layout.layer_ids):
        wdtype, wshape = ckpt.meta(f"layers.{lid}.engram.embed.weight")
        sdtype, sshape = ckpt.meta(f"layers.{lid}.engram.embed.scale")
        ids_k = hashes[:, :, k, :]                                  # [B, L, n_hash_cols]

        fp8_rows, scale_rows = lookup_raw(ckpt, lid, ids_k)
        rows = dequantize(fp8_rows, scale_rows, bs, check_exact=False)

        # verify the claim G-3 rests on, and report it rather than asserting it
        v32 = fp8_rows.float().unflatten(-1, (-1, bs)) * scale_rows.float().unsqueeze(-1)
        v64 = fp8_rows.double().unflatten(-1, (-1, bs)) * scale_rows.double().unsqueeze(-1)
        product_exact = bool(torch.equal(v32.double(), v64))
        exact = bool(torch.equal(rows, v64.flatten(-2).to(torch.bfloat16)))
        all_exact &= exact and product_exact

        # the scale is a bare exponent; check torch's fp8 conversion against
        # 2**(byte-127) so the reference does not inherit its semantics
        sb = scale_rows.contiguous().view(torch.uint8).to(torch.int32)
        pow2_exact = bool(torch.equal(scale_rows.float(), torch.pow(2.0, (sb - 127).float())))

        nz_neg = int((fp8_rows.view(torch.uint8) == 0x80).sum())
        nan_b = int(sum((fp8_rows.view(torch.uint8) == b).sum() for b in E4M3_NAN_BYTES))
        out_bf16_negz = int((rows.view(torch.uint16) == 0x8000).sum())

        out_tensors[f"layer_{lid:02d}.rows"] = rows.contiguous()
        out_tensors[f"layer_{lid:02d}.fp8"] = fp8_rows.contiguous()
        out_tensors[f"layer_{lid:02d}.scale"] = scale_rows.contiguous()

        print(f"layer {lid:2d}")
        print(f"  table                 {wshape} {wdtype}   scale {sshape} {sdtype}")
        print(f"  unique rows gathered  {int(torch.unique(ids_k).numel())} of {wshape[0]}")
        print(f"  looked-up rows        {tuple(rows.shape)} bfloat16   sha {digest(rows)}")
        print(f"  raw fp8 rows          {tuple(fp8_rows.shape)}   sha {digest(fp8_rows)}")
        print(f"  raw scale rows        {tuple(scale_rows.shape)}   sha {digest(scale_rows)}")
        print(f"  fp32 product exact    {product_exact}  (e8m0 is a power of two, "
              f"e4m3 has 3 mantissa bits)")
        print(f"  bf16 result == fp64   {exact}   <-- G-3 rests on this")
        print(f"  scale == 2**(b-127)   {pow2_exact}  (independent of torch's fp8 cast)")
        print(f"  e4m3 NaN bytes read   {nan_b}   <-- 0 means value compare is NaN-safe here")
        print(f"  -0 fp8 bytes read     {nz_neg}   -> bf16 -0 in output: {out_bf16_negz}"
              f"   <-- why the criterion forgives ±0")
        print(f"  value range           [{float(rows.float().min()):+.5f}, "
              f"{float(rows.float().max()):+.5f}]")
        print(f"  nonzero               {int((rows != 0).sum())}/{rows.numel()}")
        meta["layers"][str(lid)] = {
            "table_shape": list(wshape), "table_dtype": str(wdtype),
            "scale_shape": list(sshape), "scale_dtype": str(sdtype),
            "unique_rows": int(torch.unique(ids_k).numel()),
            "rows_shape": list(rows.shape),
            "rows_sha256_16": digest(rows),
            "fp8_sha256_16": digest(fp8_rows),
            "scale_sha256_16": digest(scale_rows),
            "fp32_product_exact": product_exact,
            "bf16_matches_fp64": exact,
            "scale_is_pow2_exact": pow2_exact,
            "e4m3_nan_bytes_read": nan_b,
            "neg_zero_fp8_bytes_read": nz_neg,
            "neg_zero_bf16_out": out_bf16_negz,
        }
        del v32, v64

    print()
    print(f"dequantize is unambiguous on every gathered row: {all_exact}")
    print("=> G-3 can be stated as bit-exactness, not a tolerance.")
    meta["all_exact"] = all_exact

    failed = not all_exact
    if a.verify:
        from safetensors.torch import load_file
        got = load_file(a.verify)
        print()
        print(f"verify against {a.verify}")
        missing = [k for k in out_tensors if k not in got]
        if missing:
            print(f"  MISSING keys: {missing}")
            failed = True
        for key in out_tensors:
            if key not in got:
                continue
            r = compare(out_tensors[key], got[key])
            tag = "PASS" if r["pass"] else "FAIL"
            note = r.get("reason") or "equal"
            if r["pass"] and r.get("signed_zero"):
                note += "  -- forgiven, cannot change any downstream value"
            print(f"  {tag:4s} {key:<22} {note}")
            if not r["pass"]:
                failed = True
        # Tell the failure modes apart, which is why .fp8/.scale are dumped.  THREE
        # buckets, not two, because the two quantised inputs do not travel the same
        # road: the tech report keeps the full scale table resident in device local
        # memory and ships only the fp8 rows from host.  So an fp8 mismatch implicates
        # the host fetch, while a scale mismatch cannot -- it never crossed that bus.
        # Merging them would send someone to audit a transfer that did not happen.
        for lid in layout.layer_ids:
            rk = f"layer_{lid:02d}.rows"
            fk = f"layer_{lid:02d}.fp8"
            sk = f"layer_{lid:02d}.scale"
            if rk not in got:
                continue
            rows_ok = compare(out_tensors[rk], got[rk])["pass"]
            fp8_ok = compare(out_tensors[fk], got[fk])["pass"] if fk in got else None
            sc_ok = compare(out_tensors[sk], got[sk])["pass"] if sk in got else None
            if rows_ok:
                continue
            if fp8_ok is False:
                print(f"  layer {lid}: the raw fp8 rows already differ => the HOST "
                      f"GATHER is wrong (hash / index / the host->device fetch). "
                      f"Do not look at the dequantize or the scale.")
            elif sc_ok is False:
                print(f"  layer {lid}: fp8 rows agree, SCALE does not => the "
                      f"device-side scale indexing is wrong. The scale table is "
                      f"resident on device, so this is NOT the host path. "
                      f"⚠️ Duplicate hash ids are the known trap here -- upstream's "
                      f"scale gather returns wrong values for repeated indices "
                      f"without erroring (see --dup-test).")
            elif fp8_ok and sc_ok:
                print(f"  layer {lid}: both quantised inputs agree byte for byte but "
                      f"the output does not => the DEQUANTIZE arithmetic is wrong "
                      f"(block size, scale layout, or the cast).")
            else:
                print(f"  layer {lid}: output differs and .fp8/.scale were not both "
                      f"supplied, so the cause cannot be localised. Dump them.")

    if a.out:
        from safetensors.torch import save_file
        os.makedirs(a.out, exist_ok=True)
        save_file(out_tensors, os.path.join(a.out, "engram_lookup_ref.safetensors"))
        with open(os.path.join(a.out, "engram_lookup_ref.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nwrote {a.out}/engram_lookup_ref.safetensors + .json")

    ckpt.close()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
