# SPDX-License-Identifier: Apache-2.0
"""E2w: which channel can write into RESIDENT weights that are already in the
internal (NZ) format, per expert or wholesale?

Context. E2' (run 20260922_125758_e2p) closed two of three questions for the
streaming-prefill / dynamic-hot-expert design on this build:
  * a per-expert ``npu_format_cast`` is BYTE-IDENTICAL to the same experts inside
    a block cast, and each expert occupies a whole number of contiguous bytes
    (11796480 / 5898240 / 737280 / 368640 B) with no padding across dim 0;
  * but ``dst.data[i:i+k].copy_(chunk_cast)`` is REFUSED:
    "copy_ do not support internal format".
This script asks the remaining question: is there a channel that does work.

🔴 The discipline this script exists to enforce. In E2' the positive control was
built from the very experts it then wrote, so its target already held the
expected bytes and "0 bytes changed" read identically whether the write had
happened or not -- zero discriminating power, and it produced a wrong reading
that only reversed once fixed. So here, for EVERY candidate:
  * the target is pre-filled with a distinguishable pattern (0xA5), never with
    the content that is about to be written;
  * before the write, the target is VERIFIED to differ from the expected bytes,
    and by how many -- a candidate whose baseline already matches is reported as
    having no discriminating power and its result is discarded;
  * the verdict is a byte comparison after the write, never "no exception".
    "It did not raise" is not "it wrote";
  * each candidate carries at least one mutation arm that MUST come out red.

🔴 Address stability is judged separately from writability. A candidate that
writes correctly but hands back a NEW tensor is useless in production: a
captured graph replays the address it recorded, so rebinding ``param.data``
silently keeps feeding the old buffer (this is the known DSV4.1 graph-decode
root cause). Every candidate therefore reports data_ptr before/after, and the
object-swap candidate is included ONLY to separate "cannot write" from
"cannot change address" -- it is marked unusable regardless of its result.

Candidates (cheapest first), all on the real post-load-hook output:
  C1  whole-tensor  dst.data.copy_(src.data)
  C2  whole-tensor  torch_npu.copy_memory_(dst.data, src.data)
  C3  per-expert    byte slices of a uint8 alias of untyped_storage(), copy_
  C4  per-expert    the same byte slices via torch_npu.copy_memory_
  C5  per-expert    dst.data[i:i+1].copy_(...)  -- the channel E2' found refused,
                    repeated here as the in-run negative control so the matrix
                    contains a known-refused row next to the others
  C6  object swap   param.data = src_tensor  (control only; NOT usable)
  C7  round trip    internal -> ND -> per-expert write -> back, then land it in
                    the ORIGINAL storage if any of C1..C4 can; time and the extra
                    HBM of holding a second same-shape tensor are reported
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e_common as E  # noqa: E402
import oncard_common as C  # noqa: E402
from e2p_slice_cast import (ACL_FORMAT_ND, cmp_u8, npu_format, pick_experts,  # noqa: E402
                            raw_alias, storage_nbytes, to_host_nd_u8)

WNAMES = ("w13_weight", "w2_weight")
SNAMES = ("w13_weight_scale", "w2_weight_scale")
ALL = WNAMES + SNAMES
PATTERN = 0xA5


def err(e) -> str:
    return f"{type(e).__name__}: {str(e).splitlines()[0]}"


def gib(n) -> float:
    return n / 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--slots", type=int, default=8, help="slots for the correctness matrix")
    ap.add_argument("--time-slots", type=int, default=100,
                    help="slots for the timing pass (a realistic resident set); 0 to skip")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-floor-gib", type=float, default=100.0)
    ap.add_argument("--mem-budget-gib", type=float, default=10.0)
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()

    mg = E.MemGuard(args.mem_floor_gib, args.mem_budget_gib)
    import numpy as np
    import npu_moe as N
    import torch_npu

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    L, S = args.layer, args.slots
    v = C.Verdict("E2w write channels into resident NZ weights")
    v.note(E.BUILD_NOTE)
    but = E.build_under_test(args.out)
    v.note(f"build under test: so={but.get('so_primary_md5')} trees="
           + ",".join(f"{k}@{(t.get('head') or '?')[:9]}" for k, t in (but.get("trees") or {}).items()))
    rng = np.random.default_rng(args.seed)
    facts: dict = {"build_note": E.BUILD_NOTE, "build_under_test": but, "layer": L, "slots": S,
                   "pattern": PATTERN}
    mg.check("after runtime init")

    # ---- sources -----------------------------------------------------------
    sel = pick_experts(rng, S, 32)
    alt = pick_experts(rng, S, 32)          # the WRONG experts, for the mutation arms
    while alt == sel:
        alt = pick_experts(rng, S, 32)
    need = sorted(set(sel) | set(alt))
    raw = N.load_raw_layer(ck, L, need)
    pos = {e: i for i, e in enumerate(need)}
    raw_sel = tuple(t[[pos[e] for e in sel]] for t in raw)
    raw_alt = tuple(t[[pos[e] for e in alt]] for t in raw)
    mg.check("after loading the source experts")

    ref = N.build_layer(raw_sel, S, dev)     # what the target must end up holding
    src = N.build_layer(raw_sel, S, dev)     # an INDEPENDENT object with the same content
    bad = N.build_layer(raw_alt, S, dev)     # the mutation source
    expected = {n: to_host_nd_u8(getattr(ref, n).data) for n in ALL}
    per_expert = {n: (storage_nbytes(getattr(ref, n).data) or 0) // S for n in ALL}
    fmt = {n: npu_format(getattr(ref, n).data) for n in ALL}
    print(f"\nexperts written {sel}\nmutation experts {alt}")
    print("  formats " + ", ".join(f"{n}={fmt[n]}" for n in ALL))
    print("  bytes/expert " + ", ".join(f"{n}={per_expert[n]}" for n in ALL), flush=True)
    facts["sel"], facts["alt"], facts["formats"], facts["bytes_per_expert"] = sel, alt, fmt, per_expert
    mg.check("after building ref/src/bad")

    def fresh_target():
        """An S-slot layer whose post-load hook has run over a 0xA5 pattern.

        The pattern -- not the content about to be written -- is what gives the
        byte comparison its discriminating power.
        """
        return N.build_layer(tuple(torch.full_like(t, PATTERN) for t in raw_sel), S, dev)

    # ---- the candidates ----------------------------------------------------
    def c1(dst, s):
        for n in ALL:
            getattr(dst, n).data.copy_(getattr(s, n).data)

    def c2(dst, s):
        for n in ALL:
            torch_npu.copy_memory_(getattr(dst, n).data, getattr(s, n).data)

    def _alias_pair(dst, s, n):
        da, sa = raw_alias(getattr(dst, n).data), raw_alias(getattr(s, n).data)
        if da is None or sa is None:
            raise RuntimeError(f"cannot build a uint8 alias of untyped_storage() for {n}")
        return da, sa

    def c3(dst, s, shift=0):
        for n in ALL:
            da, sa = _alias_pair(dst, s, n)
            p = per_expert[n]
            for i in range(S):
                j = (i + shift) % S
                da[i * p:(i + 1) * p].copy_(sa[j * p:(j + 1) * p])

    def c4(dst, s, shift=0):
        for n in ALL:
            da, sa = _alias_pair(dst, s, n)
            p = per_expert[n]
            for i in range(S):
                j = (i + shift) % S
                torch_npu.copy_memory_(da[i * p:(i + 1) * p], sa[j * p:(j + 1) * p])

    def c5(dst, s):
        for n in ALL:
            for i in range(S):
                getattr(dst, n).data[i:i + 1].copy_(getattr(s, n).data[i:i + 1])

    def c6(dst, s):
        for n in ALL:
            getattr(dst, n).data = getattr(s, n).data

    def c7(dst, s):
        """internal -> ND -> per-expert write -> back to internal, then land it in
        the ORIGINAL storage via whichever whole-tensor channel works."""
        for n in ALL:
            d = getattr(dst, n).data
            f = npu_format(d)
            if f == ACL_FORMAT_ND:
                d.copy_(getattr(s, n).data)
                continue
            back = d.transpose(1, 2) if (not d.is_contiguous() and d.dim() == 3) else d
            nd_d = torch_npu.npu_format_cast(back, ACL_FORMAT_ND)
            sb = getattr(s, n).data
            sb = sb.transpose(1, 2) if (not sb.is_contiguous() and sb.dim() == 3) else sb
            nd_s = torch_npu.npu_format_cast(sb, ACL_FORMAT_ND)
            for i in range(S):                      # the per-expert write, in ND
                nd_d[i].copy_(nd_s[i])
            nz = torch_npu.npu_format_cast(nd_d, f)  # back to the internal format
            torch_npu.copy_memory_(back, nz)         # land it in the ORIGINAL storage

    CANDS = [
        ("C1 whole-tensor Tensor.copy_", c1, True),
        ("C2 whole-tensor torch_npu.copy_memory_", c2, True),
        ("C3 per-expert byte slices of untyped_storage alias, copy_", c3, True),
        ("C4 per-expert byte slices of untyped_storage alias, copy_memory_", c4, True),
        ("C5 per-expert dst.data[i:i+1].copy_ (E2' found this refused)", c5, True),
        ("C6 param.data = src_tensor (CONTROL ONLY -- NOT usable in production)", c6, False),
        ("C7 internal -> ND -> per-expert write -> internal -> original storage", c7, True),
    ]

    results = {}
    for name, fn, usable in CANDS:
        print(f"\n===== {name} =====", flush=True)
        dst = fresh_target()
        # discriminating power, measured before anything is written
        base = {n: cmp_u8(to_host_nd_u8(getattr(dst, n).data), expected[n], n) for n in ALL}
        differing = {n: base[n][1] for n in ALL}
        power = all(not base[n][0] for n in ALL)
        v.check(power, f"{name}: target differs from the expected bytes BEFORE the write "
                       f"(discriminating power; {'; '.join(differing[n] for n in WNAMES)})")
        if not power:
            v.note(f"{name}: skipped -- without discriminating power its result means nothing")
            del dst
            continue
        ptr0 = {n: getattr(dst, n).data.data_ptr() for n in ALL}
        hbm0 = torch.npu.memory_allocated()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        raised = None
        try:
            fn(dst, src)
            torch.npu.synchronize()
        except Exception as e:  # noqa: BLE001
            raised = err(e)
        dt = time.perf_counter() - t0
        hbm1 = torch.npu.memory_allocated()
        ptr1 = {n: getattr(dst, n).data.data_ptr() for n in ALL}
        moved = [n for n in ALL if ptr0[n] != ptr1[n]]
        cmps, okall = {}, True
        for n in ALL:
            try:
                ok, det = cmp_u8(to_host_nd_u8(getattr(dst, n).data), expected[n], n)
            except Exception as e:  # noqa: BLE001
                ok, det = False, f"{n}: readback refused ({err(e)})"
            cmps[n], okall = (ok, det), okall and ok
        print(f"  raised={raised}")
        print(f"  bytes: " + "; ".join(cmps[n][1] for n in ALL))
        print(f"  {dt * 1000:.1f} ms, HBM {gib(hbm0):.3f} -> {gib(hbm1):.3f} GiB "
              f"(delta {gib(hbm1 - hbm0):+.3f}), address moved for {moved or 'nothing'}", flush=True)
        v.check(okall and raised is None,
                f"{name}: wrote the expected bytes into all four parameters"
                + (f" -- REFUSED: {raised}" if raised else
                   ("" if okall else " -- no exception, but the bytes are NOT the expected ones "
                                     "('it did not raise' is not 'it wrote')")))
        if okall and raised is None:
            v.check(not moved, f"{name}: the parameter kept its address (data_ptr unchanged) -- "
                               "required, because a captured graph replays the address it recorded"
                    + ("" if not moved else f"; MOVED for {moved} -> unusable under graph capture"))
        results[name] = {"raised": raised, "bytes_ok": okall, "ms": dt * 1000,
                         "hbm_delta_gib": gib(hbm1 - hbm0), "address_moved": moved,
                         "usable_in_production": usable,
                         "detail": {n: cmps[n][1] for n in ALL}}
        # ---- mutation arms -------------------------------------------------
        if okall and raised is None:
            m = fresh_target()
            try:
                fn(m, bad)      # the WRONG experts
                torch.npu.synchronize()
                okm = all(cmp_u8(to_host_nd_u8(getattr(m, n).data), expected[n], n)[0] for n in ALL)
            except Exception as e:  # noqa: BLE001
                okm = False
                v.note(f"{name} mutation 'wrong experts' raised ({err(e)}) -- counts as red")
            v.must_be_red(not okm, f"{name}: writing the WRONG experts")
            results[name]["mut_wrong_experts_red"] = not okm
            del m
            if fn in (c3, c4):
                m2 = fresh_target()
                try:
                    fn(m2, src, shift=1)   # right data, wrong byte offset
                    torch.npu.synchronize()
                    okm2 = all(cmp_u8(to_host_nd_u8(getattr(m2, n).data), expected[n], n)[0] for n in ALL)
                except Exception as e:  # noqa: BLE001
                    okm2 = False
                    v.note(f"{name} mutation 'offset by one expert' raised ({err(e)}) -- counts as red")
                v.must_be_red(not okm2, f"{name}: writing at an offset of one expert")
                results[name]["mut_offset_red"] = not okm2
                del m2
            m3 = fresh_target()   # not writing at all must also be red
            ok3 = all(cmp_u8(to_host_nd_u8(getattr(m3, n).data), expected[n], n)[0] for n in ALL)
            v.must_be_red(not ok3, f"{name}: not writing at all")
            results[name]["mut_no_write_red"] = not ok3
            del m3
        del dst
        torch.npu.empty_cache()
        mg.check(f"after {name.split()[0]}")
    facts["candidates"] = results

    # ---- scope probe: does copy_memory_ honour the view's extent? ----------
    #
    # Why this is not a detail. In the C4 run the loop raised at i=1
    # ("self.storage_offset() == 0 INTERNAL ASSERT FAILED"), so only the single
    # offset-0 slice covering ONE expert had been copied -- yet w13_weight came
    # out identical in ALL 94371840 bytes. Either copy_memory_ writes the whole
    # storage regardless of the view it was handed, or those bytes arrived some
    # other way. The two readings lead to opposite designs: whole-storage means
    # a hot-expert swap must rewrite the entire resident buffer, view-honouring
    # would mean a genuine per-expert write exists. So it gets its own probe
    # rather than an inference from a number that was never meant to answer it.
    print("\n===== scope probe: does torch_npu.copy_memory_ honour the view's extent? =====", flush=True)
    scope = {}
    try:
        d = fresh_target()
        n = "w13_weight"
        before = to_host_nd_u8(getattr(d, n).data)
        da, sa = _alias_pair(d, src, n)
        p = per_expert[n]
        torch_npu.copy_memory_(da[0:p], sa[0:p])   # ONE expert's worth, at offset 0
        torch.npu.synchronize()
        after = to_host_nd_u8(getattr(d, n).data)
        exp = expected[n]
        row0 = bool(torch.equal(after[0], exp[0]))
        others_written = [i for i in range(1, S) if torch.equal(after[i], exp[i])]
        others_unchanged = [i for i in range(1, S) if torch.equal(after[i], before[i])]
        whole = len(others_written) == S - 1
        honours = len(others_unchanged) == S - 1
        scope = {"slice_bytes": p, "storage_bytes": storage_nbytes(getattr(d, n).data),
                 "expert0_written": row0, "other_experts_written": len(others_written),
                 "other_experts_unchanged": len(others_unchanged),
                 "verdict": "whole-storage" if whole else ("honours-extent" if honours else "mixed")}
        print(f"  handed a {p} B view of a {scope['storage_bytes']} B storage: expert0 written={row0}, "
              f"other experts written={len(others_written)}/{S - 1}, unchanged={len(others_unchanged)}/{S - 1}"
              f" -> {scope['verdict']}", flush=True)
        v.check(row0, "scope probe: the expert the view covers was written")
        v.check(whole or honours,
                f"scope probe: the outcome is unambiguous -- {scope['verdict']} "
                f"(written {len(others_written)}, unchanged {len(others_unchanged)} of {S - 1})")
        if whole:
            v.note("copy_memory_ writes the WHOLE storage, not the view it is handed. A per-expert "
                   "write through it is therefore impossible; a hot-expert swap must rewrite the "
                   "entire resident buffer (see the timing pass for what that costs).")
        elif honours:
            v.note("copy_memory_ honours the view's extent, but requires storage_offset()==0, so only "
                   "a prefix starting at expert 0 can be written this way -- not an arbitrary expert.")
        del d
        torch.npu.empty_cache()
    except Exception as e:  # noqa: BLE001
        scope = {"error": err(e)}
        v.note(f"scope probe could not run: {err(e)} -- the whole-storage question stays OPEN")
    facts["copy_memory_scope"] = scope
    mg.check("after the scope probe")

    # ---- timing pass at a realistic resident size --------------------------
    winners = [(n, f) for n, f, u in CANDS
               if u and results.get(n, {}).get("bytes_ok") and not results[n]["raised"]
               and not results[n]["address_moved"]]
    facts["timing"] = {}
    if args.time_slots and winners:
        TS = args.time_slots
        print(f"\n===== timing pass: {TS} slots (a realistic resident set) =====", flush=True)
        ids = list(range(TS))
        rawT = N.load_raw_layer(ck, L, ids)
        mg.check(f"after loading {TS} experts")
        dstT, srcT = N.build_layer(rawT, TS, dev), N.build_layer(rawT, TS, dev)
        nbytes = sum(storage_nbytes(getattr(dstT, n).data) or 0 for n in ALL)
        print(f"  resident weight bytes on card: {gib(nbytes):.3f} GiB", flush=True)
        mg.check(f"after building the {TS}-slot pair")
        saved = (S, per_expert)
        S = TS
        per_expert = {n: (storage_nbytes(getattr(dstT, n).data) or 0) // TS for n in ALL}
        for name, fn in winners:
            ts = []
            for _ in range(args.repeats):
                torch.npu.synchronize()
                t0 = time.perf_counter()
                fn(dstT, srcT)
                torch.npu.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            ok = all(cmp_u8(to_host_nd_u8(getattr(dstT, n).data), to_host_nd_u8(getattr(srcT, n).data), n)[0]
                     for n in ALL)
            best, med = min(ts), sorted(ts)[len(ts) // 2]
            gbs = gib(nbytes) / (best / 1000)
            print(f"  {name}: median {med:.1f} ms, best {best:.1f} ms over {gib(nbytes):.3f} GiB "
                  f"= {gbs:.1f} GiB/s; bytes still correct: {ok}", flush=True)
            v.check(ok, f"timing pass {TS} slots, {name}: the write is still byte-correct at this size")
            facts["timing"][name] = {"slots": TS, "median_ms": med, "best_ms": best,
                                     "all_ms": ts, "bytes_gib": gib(nbytes), "gib_per_s": gbs}
        S, per_expert = saved
        del dstT, srcT, rawT
        torch.npu.empty_cache()
        mg.check("after the timing pass")
    elif not winners:
        v.note("no candidate both wrote correctly and kept its address, so there is nothing to time")

    print(f"\n{N.hbm_line()}", flush=True)
    mg.check("end")
    facts["memguard"] = mg.facts()
    return v.finish(os.path.join(args.out, "e2w_result.json"), facts)


if __name__ == "__main__":
    sys.exit(main())
