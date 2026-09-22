# SPDX-License-Identifier: Apache-2.0
"""E2' (gate 1): can ``npu_format_cast`` be run per expert slice, and is the
result bit-identical to the same experts inside a larger cast? And can the
result be written into an already-cast (NZ + transposed) parameter?

Gate question (streaming prefill / dynamic hot experts): the resident set would
be filled a few experts at a time, so the cast that today runs once over
[384, ...] would have to run over [C, ...] and its output be copied into rows of
an already-cast parameter. Two things can break:
  * the cast is not row-separable (padding/tiling that depends on dim 0), so
    slice i of a C-cast != row sel[i] of the larger cast;
  * the write into an NZ target is silently dropped.

The subject is the production post-load hook itself, reached through
npu_moe.build_layer -> AscendW4A8MXFPDSDynamicFusedMoEMethod
.process_weights_after_loading (w4a8_mxfp4.py:346-367). The cast expression is
NOT re-implemented here: a re-implementation is a description of the production
line, and descriptions drift.

🔴 SCALE IS A STAND-IN, and the report must say so.
  The reference cast is ``--ref-experts`` (default 32) experts, not the 384 of
  a real layer, and the slices are C = 1/4/8, not 64. The reason is a hard
  constraint, not convenience: a co-tenant GSM8K run holds host memory, this
  process may add ~10 GiB at most, and a 384-expert layer is ~7.2 GiB of host
  bytes plus ~15 GiB of HBM.
  Why a 32-expert reference can still answer "is a slice cast bit-identical to
  the same experts inside a block cast": the question is whether the cast's
  output for expert row i depends on anything other than row i's input bytes.
  The test compares a C-cast against a REF-cast of the same experts with
  C != REF, so any dependence on the number of rows, on the row's position in
  the tensor, or on NZ padding computed across dim 0 shows up as differing
  bytes. Two different values of C against the same reference (1/4/8 vs 32)
  make it four independent row-count pairs, and the physical (P) reading
  compares bytes-per-expert of the two casts directly, which is where a dim-0
  dependence would live.
  What this scale does NOT cover, and the report states it as an open gap:
    * a dim-0 dependence that only appears above 32 rows (e.g. a tiling rule
      whose block is larger than 32, or an internal size threshold that picks
      a different kernel for a big tensor). Nothing here can see that.
    * whether the FULL 384-row cast equals the 32-row cast. Not measured.
    * the time and HBM cost of casting 384 experts one at a time.
  So the finding is "row-separable at these sizes", not "row-separable".

Arms
  1 bit-exact   C in --chunks vs the REF-cast, per expert, as uint8 bytes
                (uint8 avoids both bit-exactness traps at once: NaN != NaN and
                -0 == +0 never arise on a byte compare). Three readings, each
                able to fail on its own:
                  H  logical: both sides converted to ND and copied to the host.
                     An internal-format tensor cannot be cloned or copied to the
                     host as it stands on this device, so the ND conversion is
                     not optional -- and it is what makes H blind to padding.
                  D  device: index_select on the cast tensor. A REFUSAL here is a
                     capability statement about index_select, not a wrong answer
                     about row separability, so it is recorded as a note; only an
                     answer that comes back and disagrees fails a check.
                  P  physical: uint8 alias of the untyped storage, with NO format
                     conversion -- the only reading that sees the real layout,
                     including bytes-per-expert.
                Scales (reshape/view/transpose, no kernel) are compared the same way.
  2 write       an S-slot layer filled --write-steps experts at a time with
                dst.<param>.data[i:i+step].copy_(chunk_cast), then the real
                fused expert (quant_method.apply -> npu_moe_init_routing
                quant_mode=3 -> GMM -> unpermute) vs the same S experts built
                the normal way. Criterion: bit-identical (same weights, same
                kernel, same input). If the device REFUSES the slice+copy_, that
                refusal is itself the gate answer: it is reported, and the
                output comparisons are then explicitly not accuracy readings.
                A deliberately coarse not-garbage arm against the torch MXFP4
                reference is printed and checked at a stated loose tau, so
                "both arms agree and both are garbage" cannot pass unnoticed.
  3 positive    index_select(src_nz, 0, idx, out=dst_nz) MUST be a silent no-op:
    control     no exception AND not one byte of dst changed (sglang 935c2aa,
                GPQA 70% -> 41%). If it writes, our model of the failure mode is
                wrong, and the script says INCONSISTENT and fails rather than
                reporting a conclusion that rests on it.
  4 mutations   must be RED: (a) the same bytes compared against the WRONG
                expert row -> the byte criterion; (b) one scale byte flipped and
                (c) the nibbles swapped in the reference -> the numeric criterion.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e_common as E  # noqa: E402
import oncard_common as C  # noqa: E402

# vllm_ascend.utils: ACL_FORMAT_FRACTAL_ND = 2, ACL_FORMAT_FRACTAL_NZ = 29.
#
# 🔴 The production hook asks npu_format_cast for 29, and on this build the
# result reports format 50 (measured 2026-09-22 on card 3, layer 20). So the
# criterion must NOT be "== 29": the number the call is given is not the number
# the tensor comes back with, and a check written against the argument would
# have failed on a perfectly good cast. What actually has to hold is
#   (a) the weights are in SOME internal format, i.e. not ND -- otherwise the
#       cast did nothing and every per-expert comparison would pass trivially;
#   (b) the slice cast and the reference cast are in the SAME format -- a
#       comparative criterion, with no magic constant to go stale.
# The id itself is recorded in the result json either way.
ACL_FORMAT_ND = 2
ACL_FORMAT_NZ_REQUESTED = 29

WNAMES = ("w13_weight", "w2_weight")
SNAMES = ("w13_weight_scale", "w2_weight_scale")


def npu_format(t) -> int | None:
    try:
        import torch_npu

        return int(torch_npu.get_npu_format(t))
    except Exception:
        return None


def storage_nbytes(t) -> int | None:
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:
        return None


def raw_alias(t):
    """uint8 tensor over the whole untyped storage of ``t``, no format conversion.

    The only reading that sees the NZ bytes as the device holds them: ``.cpu()``
    on an NZ tensor converts back to ND on the way out, so padding or tiling
    that differs between a C-cast and a REF-cast is invisible there. Returns
    None if the alias cannot be built -- and then the script says the physical
    layout is UNPROVEN rather than implying it was checked.
    """
    try:
        st = t.untyped_storage()
        out = torch.empty(0, dtype=torch.uint8, device=t.device)
        out.set_(st, 0, (int(st.nbytes()),))
        return out
    except Exception:
        return None


def as_u8(t):
    """uint8 view of ``t`` WITHOUT touching its layout.

    No ``.contiguous()``: on this device that is a clone, and a clone of an
    internal-format tensor raises ("at_npu::native::NPUNativeFunctions::clone
    do not support internal format", card 3, 2026-09-22). Anything that needs
    the bytes on the host must go through to_host_nd_u8 or raw_alias instead.
    """
    t = t.detach()
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def to_host_nd_u8(t):
    """Reading H: the LOGICAL bytes of ``t`` on the host.

    An internal-format tensor cannot be cloned or copied to the host as it
    stands, so it is converted to ND first. That conversion is what makes this
    the logical reading: the NZ tiling is undone on the way out and only the
    values survive, so H cannot see a padding or tiling difference. Reading P
    (raw_alias) is the one that sees the layout. Both arms go through the exact
    same sequence, so the comparison stays valid either way.
    """
    import torch_npu

    x = t.detach()
    fmt = npu_format(x)
    if fmt is not None and fmt != ACL_FORMAT_ND:
        # 🔴 The production hook stores ``cast_output.transpose(1, 2)``, so the
        # parameter is a NON-CONTIGUOUS view of an internal-format tensor, and
        # npu_format_cast refuses exactly that ("does not support non-contiguous
        # tensor with internal format FRACTAL_NZ_C0_16", card 3 2026-09-22).
        # Transposing back recovers the contiguous cast output on the very same
        # storage -- no copy, no reinterpretation. Both arms go through this
        # identical step, so the comparison is unaffected; it only means the
        # host shape is the pre-transpose one.
        if not x.is_contiguous() and x.dim() == 3:
            y = x.transpose(1, 2)
            if y.is_contiguous():
                x = y
        x = torch_npu.npu_format_cast(x, ACL_FORMAT_ND)
    if x.dtype != torch.uint8:
        x = x.view(torch.uint8)
    return x.contiguous().cpu()


def cmp_u8(a, b, what: str):
    """(ok, detail) on uint8 bytes, both moved to the host."""
    a, b = a.detach().cpu(), b.detach().cpu()
    if a.shape != b.shape:
        return False, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    ne = int((a != b).sum())
    if ne == 0:
        return True, f"{what}: {a.numel()} bytes identical"
    first = int(torch.nonzero(a.flatten() != b.flatten())[0])
    return False, f"{what}: {ne}/{a.numel()} bytes differ, first at flat index {first}"


def pick_experts(rng, n: int, hi: int) -> list[int]:
    """``n`` distinct logical ids in [1, hi), always containing hi-1, never a
    contiguous run: the slice's slot i then holds logical id sel[i] != i for
    every i, so a cast that quietly used the slot index instead of the row's
    bytes would be wrong on every row instead of accidentally right on some."""
    if n == 1:
        return [max(1, hi // 3)]
    pool = np.arange(1, hi - 1)  # 0 excluded, hi-1 added below
    if n - 1 > pool.size:
        # n == hi would force the whole range, i.e. slot == logical id, which is
        # the one arrangement this function exists to avoid. Return nothing and
        # let the caller fail by name rather than sampling more than the pool
        # holds (that raised on 2026-09-22 when S was raised to REF).
        return []
    for _ in range(64):
        ids = sorted({int(i) for i in rng.choice(pool, n - 1, replace=False)} | {hi - 1})
        if len(ids) == n and ids != list(range(ids[0], ids[0] + n)):
            return ids
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--ref-experts", type=int, default=32,
                    help="rows in the reference cast (STAND-IN for 384; see the module docstring)")
    ap.add_argument("--chunks", default="1,4,8", help="slice widths C cast on their own")
    ap.add_argument("--write-slots", type=int, default=8)
    ap.add_argument("--write-steps", default="1,4", help="experts written per copy_")
    ap.add_argument("--tokens", default="1,4,32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-floor-gib", type=float, default=100.0)
    ap.add_argument("--mem-budget-gib", type=float, default=10.0)
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()

    mg = E.MemGuard(args.mem_floor_gib, args.mem_budget_gib)
    import npu_moe as N

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    L, REF = args.layer, args.ref_experts
    v = C.Verdict("E2' per-expert npu_format_cast")
    v.note(E.BUILD_NOTE)
    but = E.build_under_test(args.out)
    v.note(f"build under test: so={but.get('so_primary_md5')} trees="
           + ",".join(f"{k}@{(t.get('head') or '?')[:9]}" for k, t in (but.get("trees") or {}).items()))
    STANDIN = (
        f"规模是替身：参考 cast 用 {REF} 个专家（不是真实层的 384），切片 C={args.chunks}（不是 64）。"
        f"原因是硬约束（同机 phase2 GSM8K 占内存，本进程上限 ~{args.mem_budget_gib:.0f} GiB；"
        f"384 专家约 7.2 GiB 主机字节 + ~15 GiB HBM）。它仍能回答「切片 vs 整块是否逐位一致」："
        f"判据问的是 cast 对第 i 行的输出是否只依赖第 i 行的输入字节，而每组比较的两侧行数不同"
        f"（C vs {REF}），任何对行数、行位置或跨 dim 0 的 NZ padding 的依赖都会表现为字节不同；"
        f"P 读数还直接比两次 cast 的 每专家字节数。未覆盖的缺口：只在 >{REF} 行才出现的 dim-0 依赖"
        f"（更大的 tiling block 或按张量大小换 kernel 的阈值）、384 行整层 cast 是否等于 {REF} 行 cast、"
        f"以及逐专家 cast 384 次的时间与 HBM 代价。故结论是「在这些尺寸上行可分」，不是「行可分」。"
    )
    v.note(STANDIN)
    rng = np.random.default_rng(args.seed)
    facts: dict = {"build_note": E.BUILD_NOTE, "build_under_test": but, "layer": L,
                   "scale_standin": STANDIN, "ref_experts": REF, "chunks": args.chunks,
                   "is_standin_for_384": True}
    mg.check("after runtime init")

    # ---- the reference cast (stand-in for the 384-row cast production runs) --
    print(f"\n===== E2' layer {L}: reference cast over {REF} experts (STAND-IN for 384) =====", flush=True)
    ref_ids = list(range(REF))
    raw_ref = N.load_raw_layer(ck, L, ref_ids)
    mg.check(f"after loading {REF} experts")
    full = N.build_layer(raw_ref, REF, dev)
    print(N.hbm_line(), flush=True)
    mg.check("after the reference cast")

    layout = {}
    for name in WNAMES + SNAMES:
        t = getattr(full, name).data
        nb, fmt = storage_nbytes(t), npu_format(t)
        per = (nb / REF) if nb is not None else None
        layout[name] = {"format": fmt, "storage_bytes": nb, "per_expert_bytes": per,
                        "shape": list(t.shape), "dtype": str(t.dtype)}
        print(f"  {name}: shape {tuple(t.shape)} {t.dtype} npu_format={fmt} "
              f"storage {nb} B = {per} B/expert", flush=True)
    facts["layout_ref"] = layout
    facts["format_requested"] = ACL_FORMAT_NZ_REQUESTED
    for name in WNAMES:
        fmt = layout[name]["format"]
        if fmt != ACL_FORMAT_NZ_REQUESTED:
            v.note(f"{name}: the hook asks npu_format_cast for {ACL_FORMAT_NZ_REQUESTED} (NZ) and the "
                   f"tensor comes back reporting format {fmt}. The requested number is therefore NOT a "
                   f"usable criterion; what is checked instead is 'not ND' plus 'both arms the same'.")
        v.check(fmt is not None and fmt != ACL_FORMAT_ND,
                f"{name} of the {REF}-cast is in an internal (non-ND) format: got {fmt} "
                "-- an ND result would mean the cast did nothing and every per-expert compare "
                "would pass for the wrong reason")
    for name in WNAMES + SNAMES:
        nb, per = layout[name]["storage_bytes"], layout[name]["per_expert_bytes"]
        if nb is None:
            v.note(f"{name}: storage size unreadable -- per-expert separability of the PHYSICAL layout UNPROVEN")
            continue
        v.check(per is not None and float(per).is_integer(),
                f"{name} storage {nb} B divides evenly by {REF} experts ({per} B/expert)")

    host_ref, phys_ref = {}, {}
    for name in WNAMES + SNAMES:
        t0 = time.time()
        host_ref[name] = to_host_nd_u8(getattr(full, name).data)
        pa = raw_alias(getattr(full, name).data)
        phys_ref[name] = None
        if pa is not None:
            try:
                phys_ref[name] = pa.cpu()
            except Exception as e:  # noqa: BLE001
                print(f"  [P] {name}: alias built but D2H failed ({type(e).__name__}: {e})", flush=True)
        print(f"  [H] {name} -> host {tuple(host_ref[name].shape)} ({time.time() - t0:.1f}s), "
              f"[P] {'ok ' + str(int(phys_ref[name].numel())) + ' B' if phys_ref[name] is not None else 'UNAVAILABLE'}",
              flush=True)
    mg.check("after host copies of the reference")
    phys_ok = {n: phys_ref[n] is not None for n in phys_ref}
    if not all(phys_ok.values()):
        v.note("physical (P) reading unavailable for " + ",".join(n for n, o in phys_ok.items() if not o)
               + " -- NZ padding differences are invisible in H/D for those")

    rows = []
    for Cn in [int(s) for s in args.chunks.split(",")]:
        sel = pick_experts(rng, Cn, REF)
        if len(sel) != Cn or len(set(sel)) != Cn:
            v.check(False, f"C={Cn}: expert selection came out as {len(sel)} ids ({len(set(sel))} unique)")
            continue
        if Cn > 1 and sel == list(range(sel[0], sel[0] + Cn)):
            v.check(False, f"C={Cn}: selection came out contiguous -- an off-by-one mapping would not show")
            continue
        print(f"\n----- E2' C={Cn}: cast experts {sel} on their own (vs rows of the {REF}-cast) -----", flush=True)
        part = N.build_layer(tuple(t[sel] for t in raw_ref), Cn, dev)
        idx = torch.tensor(sel, dtype=torch.int64, device=dev)
        row = {"C": Cn, "sel": sel, "readings": {}}
        for name in WNAMES + SNAMES:
            pt = getattr(part, name).data
            fmt_p = npu_format(pt)
            if name in WNAMES:
                v.check(fmt_p is not None and fmt_p != ACL_FORMAT_ND,
                        f"C={Cn} {name}: slice cast is in an internal (non-ND) format, got {fmt_p}")
                v.check(fmt_p == layout[name]["format"],
                        f"C={Cn} {name}: slice cast format {fmt_p} == reference cast format "
                        f"{layout[name]['format']} (comparative -- no constant to go stale)")
            # H: logical bytes, both sides through the same ND conversion
            okH, dH = cmp_u8(to_host_nd_u8(pt), host_ref[name][sel], f"C={Cn} {name} [H logical/ND]")
            v.check(okH, dH)
            # D: device-side gather from the cast tensor. A REFUSAL here is a
            # capability statement about index_select, not a wrong answer about
            # row separability, so it is recorded as a note; only an answer that
            # comes back and disagrees is a failed check.
            okD, dD = None, ""
            try:
                dsel = torch.index_select(getattr(full, name).data, 0, idx)
                okD, dD = cmp_u8(to_host_nd_u8(dsel), to_host_nd_u8(pt),
                                 f"C={Cn} {name} [D index_select on the cast tensor]")
                del dsel
                v.check(okD, dD)
            except Exception as e:  # noqa: BLE001
                dD = (f"C={Cn} {name} [D]: index_select on the cast tensor REFUSED: "
                      f"{type(e).__name__}: {e}")
                v.note(dD + " -- a refusal, not a wrong answer; H and P carry the criterion")
            # P1: bytes per expert, from the storage size alone. This needs NO
            # data movement and no op on an internal-format tensor, so it is
            # always available -- and it is the reading that actually answers
            # "does the layout pad across dim 0", which is the physical part of
            # the gate question.
            nb_f, nb_p = layout[name]["storage_bytes"], storage_nbytes(pt)
            okP1, ef, ep = None, None, None
            if nb_f is None or nb_p is None:
                v.note(f"C={Cn} {name} [P1]: storage size unreadable -- physical layout UNPROVEN")
            elif nb_f % REF or nb_p % Cn:
                okP1 = False
                v.check(okP1, f"C={Cn} {name} [P1 bytes/expert]: {nb_p} B / {Cn} or {nb_f} B / {REF} "
                              "is not whole -> the layout is not row-separable")
            else:
                ef, ep = nb_f // REF, nb_p // Cn
                okP1 = ef == ep
                v.check(okP1, f"C={Cn} {name} [P1 bytes/expert]: {ep} B/expert in the C={Cn} cast == "
                              f"{ef} B/expert in the {REF}-cast"
                              + ("" if okP1 else " -> the layout depends on dim 0"))
            # P2: the physical bytes themselves, via a uint8 alias of the
            # storage. Strictly stronger than P1, but it needs a D2H of the
            # alias, which the device may refuse; a refusal leaves P1 standing
            # and is reported as such rather than as a pass.
            okP2, dP2 = None, "not attempted"
            if okP1 and phys_ref[name] is not None and ef is not None:
                ap_ = raw_alias(pt)
                try:
                    ap_ = ap_.cpu() if ap_ is not None else None
                except Exception as e:  # noqa: BLE001
                    ap_, dP2 = None, f"C={Cn} {name} [P2]: D2H of the alias refused ({type(e).__name__}: {e})"
                if ap_ is None:
                    v.note(dP2 + " -- P2 unavailable; P1 (bytes/expert) still holds")
                elif int(ap_.numel()) != Cn * ep:
                    okP2, dP2 = False, (f"C={Cn} {name} [P2]: alias is {int(ap_.numel())} B, expected "
                                        f"{Cn * ep} B")
                    v.check(okP2, dP2)
                else:
                    okP2, dP2 = cmp_u8(ap_.view(Cn, ep), phys_ref[name].view(REF, ef)[sel],
                                       f"C={Cn} {name} [P2 physical bytes, {ep} B/expert]")
                    v.check(okP2, dP2)
            row["readings"][name] = {"format": fmt_p, "H": okH, "D": okD, "P1": okP1, "P2": okP2,
                                     "bytes_per_expert": ep, "bytes_per_expert_ref": ef,
                                     "detail_H": dH, "detail_D": dD, "detail_P2": dP2}
        wrong = [sel[(i + 1) % Cn] for i in range(Cn)] if Cn > 1 else [(sel[0] + 1) % REF]
        okM, dM = cmp_u8(to_host_nd_u8(getattr(part, "w13_weight").data), host_ref["w13_weight"][wrong],
                         "shifted-by-one expert row")
        v.must_be_red(not okM, f"C={Cn} w13_weight compared against the neighbouring expert row ({dM})")
        rows.append(row)
        del part
        torch.npu.empty_cache()
        mg.check(f"after C={Cn}")
    facts["bitexact_rows"] = rows

    # ---- arm 2: the write channel -------------------------------------------
    S = args.write_slots
    wsel = pick_experts(rng, S, REF)
    if len(wsel) != S:
        v.check(False, f"write channel: could not pick {S} distinct experts")
        return v.finish(os.path.join(args.out, "e2p_result.json"), {**facts, "memguard": mg.facts()})
    raw_w = tuple(t[wsel] for t in raw_ref)
    print(f"\n===== E2' write channel: {S} slots, experts {wsel} =====", flush=True)
    ref_lay = N.build_layer(raw_w, S, dev)
    comm = N.comm_method(S)
    mg.check("after the write-channel reference layer")

    def fill_chunked(step: int, mutate=None):
        """A zeroed S-slot layer whose post-load hook has already run, then rows
        overwritten by copy_ from per-chunk casts. Target and source both come
        out of the production hook, so the real cast is on both sides.

        Returns (layer, refusals). A refusal from slice+copy_ into an
        internal-format target IS the gate answer for the write channel, not a
        script bug, so it is collected and reported rather than raised: the
        experiment must still produce its other readings and still release the
        card (memory: 报告不该销毁它所报告的那个东西).
        """
        dst = N.build_layer(tuple(torch.zeros_like(t) for t in raw_w), S, dev)
        refusals = []
        for i in range(0, S, step):
            sl = wsel[i:i + step]
            src = N.build_layer(tuple(t[i:i + step] for t in raw_w), len(sl), dev,
                                mutate=mutate if i == 0 else None)
            for name in WNAMES + SNAMES:
                try:
                    getattr(dst, name).data[i:i + len(sl)].copy_(getattr(src, name).data)
                except Exception as e:  # noqa: BLE001
                    refusals.append(f"{name} rows {i}..{i + len(sl) - 1} (step={step}): "
                                    f"{type(e).__name__}: {e}")
            del src
        torch.npu.synchronize()
        torch.npu.empty_cache()
        return dst, refusals

    wcache: dict = {}

    def wfn(slot):
        e = wsel[int(slot)]
        if e not in wcache:
            ex = ck.expert(L, e)
            wcache.clear()
            wcache[e] = tuple(C.dequant(ex[f"{p}w"], ex[f"{p}s"]) for p in ("w1", "w3", "w2"))
        return wcache[e]

    def wfn_nib(slot):
        ex = ck.expert(L, wsel[int(slot)])
        return tuple(C.dequant(ex[f"{p}w"], ex[f"{p}s"], swap_nibbles=True) for p in ("w1", "w3", "w2"))

    wrows, write_refusals = [], {}
    for step in [int(s) for s in args.write_steps.split(",")]:
        wr_lay, refusals = fill_chunked(step)
        write_refusals[str(step)] = refusals
        mg.check(f"after filling {S} slots {step} at a time")
        print(f"\n-- write channel: {S} slots filled {step} expert(s) per copy_ "
              f"({(S + step - 1) // step} chunks)", flush=True)
        v.check(not refusals,
                f"write step={step}: slice+copy_ into the already-cast target was accepted for every "
                f"row and every parameter" + ("" if not refusals else
                f" -- REFUSED {len(refusals)} time(s), first: {refusals[0]}"))
        if refusals:
            v.note(f"write step={step}: the write channel is REFUSED by the device, which is itself the "
                   "gate answer for this route. The output comparisons below therefore describe a layer "
                   "that was only partly written, and must not be read as accuracy readings.")
        for T in [int(s) for s in args.tokens.split(",")]:
            x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16)
            k = min(C.TOPK, S)
            ids = torch.from_numpy(np.stack([rng.choice(S, k, replace=False) for _ in range(T)]
                                            ).astype(np.int32))
            w = C.route_weights(rng, T)[:, :k].contiguous()
            xd, idd, wd = x.to(dev), ids.to(dev), w.to(dev)
            hit = len({int(i) for i in ids.flatten().tolist()})
            ch = len({int(i) // step for i in ids.flatten().tolist()})
            n_ch = (S + step - 1) // step
            print(f"   T={T}: {ids.numel()} routes over {hit}/{S} slots, {ch}/{n_ch} written chunks touched",
                  flush=True)
            v.check(hit >= 1 and ch >= 1, f"write step={step} T={T} coverage: {hit}/{S} slots, {ch}/{n_ch} chunks")

            y_ref = N.npu_moe(ref_lay, comm, xd, wd, idd).float().cpu()
            y_wr = N.npu_moe(wr_lay, comm, xd, wd, idd).float().cpu()
            same, how = C.bit_identical(y_wr, y_ref)
            d = C.rel_l2(y_wr, y_ref)
            v.check(same, f"write step={step} T={T} chunk-written layer == normally built layer: {how}"
                          + ("" if same else f"; rel_l2 {d:.3e}"))
            refs = C.moe_ref_multi(x, ids.long(), w, wfn, lambda e: True, ("fp32", "mx8", "mx8_acc64"))
            floor = max(C.rel_l2(refs["mx8"], refs["mx8_acc64"]), C.BF16_ULP)
            tau_coarse = max(8 * floor, 0.05)
            dg, nan = C.rel_l2(y_wr, refs["mx8"]), C.nan_inf_count(y_wr)
            print(f"      vs torch MXFP4 ref: rel_l2 {dg:.3e} (mx8 emulation floor {floor:.3e}, "
                  f"COARSE tau {tau_coarse:.3e}); vs fp32 {C.rel_l2(y_wr, refs['fp32']):.3e}; "
                  f"non-finite {nan}", flush=True)
            v.check(nan == 0, f"write step={step} T={T} output has no non-finite value")
            v.check(dg <= tau_coarse, f"write step={step} T={T} COARSE not-garbage vs torch MXFP4 ref "
                                      f"({dg:.3e} <= {tau_coarse:.3e}; coarse by construction -- it separates "
                                      "a real MoE output from zeros/NaN/wrong experts, nothing finer)")
            refs_nib = C.moe_ref_multi(x, ids.long(), w, wfn_nib, lambda e: True, ("mx8",))
            dn = C.rel_l2(y_wr, refs_nib["mx8"])
            v.must_be_red(dn > tau_coarse, f"write step={step} T={T} reference with nibbles swapped "
                                           f"({dn:.3e} vs {tau_coarse:.3e})")
            wrows.append({"step": step, "T": T, "bit_identical": same, "how": how,
                          "rel_l2_vs_ref_layer": d, "rel_l2_vs_torch_mx8": dg, "tau_coarse": tau_coarse,
                          "nonfinite": nan, "slots_hit": hit, "chunks_hit": ch})
        if step == int(args.write_steps.split(",")[0]):
            # mutation on the finest granularity only: one scale byte of the
            # first written chunk, with every row routed to slot 0 so the
            # mutated expert is certainly exercised.
            def flip_scale(name, t):
                if name == "w13_weight_scale":
                    old = int(t[0, 0, 0])
                    t[0, 0, 0] = (old ^ 0x01) & 0xFF
                    print(f"   [mutate] {name}[0,0,0] {old} -> {int(t[0, 0, 0])}", flush=True)

            mut_lay, _mr = fill_chunked(step, mutate=flip_scale)
            T = int(args.tokens.split(",")[-1])
            k = min(C.TOPK, S)
            x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16)
            ids = np.stack([np.concatenate([[0], rng.choice(np.arange(1, S), k - 1, replace=False)])
                            for _ in range(T)])
            ids = torch.from_numpy(ids.astype(np.int32))
            w = C.route_weights(rng, T)[:, :k].contiguous()
            xd, idd, wd = x.to(dev), ids.to(dev), w.to(dev)
            y_ref = N.npu_moe(ref_lay, comm, xd, wd, idd).float().cpu()
            y_mut = N.npu_moe(mut_lay, comm, xd, wd, idd).float().cpu()
            same_m, _ = C.bit_identical(y_mut, y_ref)
            dm = C.rel_l2(y_mut, y_ref)
            v.must_be_red(not same_m, f"one w13 scale byte flipped in the written chunk (rel_l2 {dm:.3e})")
            facts["scale_flip_rel_l2"] = dm
            del mut_lay
        del wr_lay
        torch.npu.empty_cache()
        mg.check(f"after write step={step}")
    facts["write_rows"] = wrows
    facts["write_refusals"] = write_refusals

    # ---- arm 3: the positive control (must reproduce the silent no-op) ------
    #
    # 🔴 The target is built from ZEROS, not from raw_w. An earlier version of
    # this arm built it from the very experts it then index_selects, so the
    # target already held the expected bytes and "0 bytes changed" read exactly
    # the same whether the write had happened or not -- a check bit with no
    # discriminating power, which is worse than no check at all. With a zeroed
    # target, "unchanged" can only mean "not written", and the expected content
    # is verified to differ from the target's initial content before the verdict
    # is allowed to mean anything.
    print("\n===== E2' positive control: index_select(..., out=<already-cast target>) =====", flush=True)
    pc_lay = N.build_layer(tuple(torch.zeros_like(t) for t in raw_w), S, dev)
    pc = {}
    idx = torch.tensor(wsel, dtype=torch.int64, device=dev)
    for name in ("w13_weight", "w13_weight_scale"):
        dst = getattr(pc_lay, name).data
        before = to_host_nd_u8(dst).clone()
        # discriminating power: what a successful write WOULD have put there,
        # obtained without out=. If that equals the target's current content,
        # this control cannot tell a write from a no-op and says so.
        power, power_note = None, ""
        try:
            expect = to_host_nd_u8(torch.index_select(getattr(full, name).data, 0, idx))
            power = (expect.shape != before.shape) or bool((expect != before).any())
            power_note = (f"expected content differs from the zeroed target in "
                          f"{int((expect != before).sum()) if expect.shape == before.shape else -1} bytes")
        except Exception as e:  # noqa: BLE001
            power_note = (f"cannot compute the expected content without out= either "
                          f"({type(e).__name__}: {e})")
        if power is True:
            v.check(True, f"positive control {name}: the control HAS discriminating power -- {power_note}")
        else:
            v.note(f"positive control {name}: discriminating power UNPROVEN -- {power_note}. "
                   "A verdict of 'unchanged' below cannot be read as 'not written'.")
        bp = raw_alias(dst)
        try:
            bp = bp.cpu().clone() if bp is not None else None
        except Exception:
            bp = None   # physical snapshot refused; the logical one still decides
        raised = None
        try:
            torch.index_select(getattr(full, name).data, 0, idx, out=dst)
            torch.npu.synchronize()
        except Exception as e:  # noqa: BLE001
            raised = f"{type(e).__name__}: {e}"
        after = to_host_nd_u8(getattr(pc_lay, name).data)
        apx = raw_alias(getattr(pc_lay, name).data)
        try:
            apx = apx.cpu() if apx is not None else None
        except Exception:
            apx = None
        changed = not torch.equal(before, after)
        nbytes_diff = int((before != after).sum()) if before.shape == after.shape else -1
        changed_phys = None if (bp is None or apx is None) else not torch.equal(bp, apx)
        silent_noop = (raised is None) and (not changed) and (changed_phys is not True)
        print(f"  {name}: raised={raised}  logical bytes changed={changed} ({nbytes_diff} B)  "
              f"physical bytes changed={changed_phys}", flush=True)
        pc[name] = {"raised": raised, "logical_changed": changed, "bytes_differing": nbytes_diff,
                    "physical_changed": changed_phys, "silent_noop_reproduced": silent_noop,
                    "control_has_power": power, "power_note": power_note,
                    "target_was_zeroed": True}
        if silent_noop and power is not True:
            silent_noop = False
            v.check(False, f"positive control {name}: reads as a silent no-op, but the control's "
                           f"discriminating power is unproven ({power_note}), so this reading is "
                           "not evidence of anything. Reported as UNPROVEN, not as reproduced.")
            continue
        if silent_noop:
            v.check(True, f"positive control {name}: silent no-op REPRODUCED (no exception, 0 bytes "
                          "written) -- the failure mode this gate is designed around is real on this build")
        else:
            v.check(False,
                    f"❌ INCONSISTENT: positive control {name} did NOT reproduce the silent no-op "
                    f"(raised={raised}, logical_changed={changed} ({nbytes_diff} B), "
                    f"physical_changed={changed_phys}). Our understanding of the failure mode "
                    "(sglang 935c2aa) does not hold on this build, so any conclusion that assumes "
                    "'out= is silently dropped' has to be re-derived. Reported as INCONSISTENT, "
                    "not as a pass and not as a plain failure of the subject.")
    facts["positive_control"] = pc
    facts["positive_control_reproduced_all"] = all(d["silent_noop_reproduced"] for d in pc.values())
    del pc_lay
    torch.npu.empty_cache()
    print(N.hbm_line(), flush=True)
    mg.check("end")
    facts["memguard"] = mg.facts()
    return v.finish(os.path.join(args.out, "e2p_result.json"), facts)


if __name__ == "__main__":
    sys.exit(main())
