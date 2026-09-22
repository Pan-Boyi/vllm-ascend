# SPDX-License-Identifier: Apache-2.0
"""E5 (gate 3): is "compute the layer in expert blocks and add up" the same as
"compute all the experts at once"?

Gate question (streaming prefill / dynamic hot experts): if the resident set is
smaller than the layer, one forward has to visit the experts in blocks and
accumulate. The per-block routing rewrite is the production transform --
vllm_ascend.kt_offload.placement.split_routing with a per-block LayerPlacement,
i.e. exactly safe_slot_table (in-block logical id -> in-block slot, out-of-block
-> dummy slot 0) and keep_table (out-of-block weight -> 0). Nothing is
re-implemented here.

🔴 SCALE IS A DOWNGRADE, and the report must say so.
  Arm A is ``--experts`` (default 32) experts in one call, NOT the 384 of a real
  layer, and the blocks are 8x4 and 4x8, not 64x6. The reason is a hard
  constraint: a co-tenant GSM8K run holds host memory, this process may add
  ~10 GiB at most, and a 384-expert arm A is ~7.2 GiB of host bytes plus ~15 GiB
  of HBM -- i.e. exactly the whole-layer residency the budget rules out.
  Why 32 experts still answers "does blocking + summing equal one call": the
  mechanism under test is the routing rewrite and the cross-expert sum, and both
  are per-route operations whose correctness does not depend on how many experts
  exist -- a route either lands in its block with its own weight or is zeroed.
  The arms differ in the number of apply() calls (1 vs 4 vs 8) and in the group
  sizes the GMM sees, which is the part that could differ.
  What this scale does NOT cover, stated as an open gap:
    * 384 experts / 6 blocks of 64: not measured. The group_list the GMM gets is
      a different shape there, and a block of 64 with most groups empty is a
      case this run never produces.
    * whether arm A at 384 is even reachable inside the HBM budget, and what one
      forward then costs -- not measured, this run says nothing about time.
    * T is 1/4/32 only; no long-prefill T.
  So the finding is "block accumulation is equivalent at these sizes", not
  "block accumulation is equivalent".

Both criteria have to pass; neither alone is enough (memory:
数值判据会放过离散的语义错误).
  1 numeric  rel_l2(A, B) and the largest deviation in bf16 ULPs, against a
             floor that is MEASURED first, not chosen: the cost of a legal
             change of cross-expert accumulation, which is what
             CROSS_ACC_VARIANT measures on the torch reference (the same bf16
             per-expert outputs summed in fp32 in the opposite order).
             Blocking changes exactly that degree of freedom.
  2 set      (a) per token, the MULTISET of contributing logical experts
             (weight != 0) reconstructed from what each block's kernel actually
             received, mapped back through slot_to_logical -- a multiset, not a
             set, so double counting shows;
             (b) per token, the index set of the output dimensions whose
             magnitude clearly exceeds half the row maximum, A vs B -- a discrete criterion on the kernel output
             itself, so 2 is not merely a re-check of 2a's table arithmetic.

Coverage is reported per block and per T, and a block that never received a
route is named. Zero coverage fails.

Mutation arm (must be RED on criteria 1 and 2a): one block's keep table left
all ones, so its out-of-block routes are also computed -- on the dummy slot,
with their original weight. That is the double count blocking can produce.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e_common as E  # noqa: E402
import oncard_common as C  # noqa: E402

TOPSET_K = 32


def threshold_index_sets(y: torch.Tensor, ref: torch.Tensor, frac: float = 0.5,
                         margin: float = 8 * C.BF16_ULP):
    """Per row: the indices whose |value| CLEARLY exceeds ``frac`` x the row's
    largest |value| in ``ref``; plus, per row, how many entries fell in the
    excluded band around that threshold.

    🔴 Why not "the k largest": a rank boundary is not stable under a legal
    rounding difference. The first version of this criterion took the top-32
    indices, and on card 3 (2026-09-22, T=32, 8x4) it flagged 1 of 17 tokens
    with a symmetric difference of 2 while the two arms differed by ~1 bf16 ULP
    and the expert multiset was identical -- two entries of nearly equal
    magnitude swapping across rank 32, i.e. an artefact of the criterion, not a
    discrete error. At T=1 every row was tied at the boundary and the criterion
    had ZERO coverage, which it correctly failed on rather than passing blind.
    A threshold taken from ONE arm (``ref``) and applied to both, with a
    measured exclusion band, asks the question the criterion was meant to ask.
    The band width is reported so that "nothing was excluded" and "everything
    was excluded" cannot be confused with each other.
    """
    a = y.detach().float().abs()
    thr = ref.detach().float().abs().amax(1, keepdim=True) * frac
    lo, hi = thr * (1.0 - margin), thr * (1.0 + margin)
    return a > hi, (a >= lo) & (a <= hi)


def compare_threshold_sets(y_a, y_b, ref, frac: float = 0.5, margin: float = 8 * C.BF16_ULP):
    """Per token: (sets equal?, |set|, how many indices were excluded).

    The excluded set is the UNION of the two arms' boundary bands, and that is
    the whole point. Excluding only each arm's own band still leaves the set's
    boundary AT the band's upper edge, so an entry sitting just above it in one
    arm and just below it in the other produces a difference of exactly the kind
    the exclusion was meant to remove -- which is what happened on card 3 on
    2026-09-22 (symmetric difference 1..3 while the arms differed by ~1 bf16 ULP
    and the expert multiset was identical). Excluding the union makes any index
    that is near the boundary FOR EITHER ARM not count for either.
    """
    keep_a, band_a = threshold_index_sets(y_a, ref, frac, margin)
    keep_b, band_b = threshold_index_sets(y_b, ref, frac, margin)
    excl = band_a | band_b
    sa, sb = keep_a & ~excl, keep_b & ~excl
    out = []
    for r in range(sa.shape[0]):
        out.append((bool(torch.equal(sa[r], sb[r])), int(sa[r].sum()), int(excl[r].sum()),
                    int((sa[r] ^ sb[r]).sum())))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--experts", type=int, default=32,
                    help="experts in arm A (DOWNGRADE from 384; see the module docstring)")
    ap.add_argument("--tokens", default="1,4,32")
    ap.add_argument("--blocks", default="8,4", help="block widths; 8 -> 8x4, 4 -> 4x8")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-floor-gib", type=float, default=100.0)
    ap.add_argument("--mem-budget-gib", type=float, default=10.0)
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()

    mg = E.MemGuard(args.mem_floor_gib, args.mem_budget_gib)
    import npu_moe as N
    from vllm_ascend.kt_offload.placement import LayerPlacement, split_routing

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    L, NE = args.layer, args.experts
    K = min(C.TOPK, NE)
    v = C.Verdict("E5 in-layer block accumulation")
    v.note(E.BUILD_NOTE)
    but = E.build_under_test(args.out)
    v.note(f"build under test: so={but.get('so_primary_md5')} trees="
           + ",".join(f"{k}@{(t.get('head') or '?')[:9]}" for k, t in (but.get("trees") or {}).items()))
    DOWNGRADE = (
        f"规模是降级：A 臂是 {NE} 个专家一次算（不是 384），分块是 "
        f"{'/'.join(f'{int(b)}x{NE // int(b)}' for b in args.blocks.split(','))}（不是 64x6）。"
        f"原因是硬约束（同机 phase2 GSM8K 占内存，本进程上限 ~{args.mem_budget_gib:.0f} GiB；"
        f"384 专家的 A 臂就是被预算排除掉的那种「整层驻留」：约 7.2 GiB 主机字节 + ~15 GiB HBM）。"
        f"它仍能回答「分块累加是否等于一次全算」：被测机制是路由改写与跨专家求和，两者都是逐路由"
        f"操作，正确性不依赖专家总数——一条路由要么落进它的块并带原权重，要么被置 0；两臂真正不同的是 "
        f"apply() 的调用次数（1 vs {'/'.join(str(NE // int(b)) for b in args.blocks.split(','))}）"
        f"与 GMM 看到的 group 尺寸。未覆盖的缺口：384 专家 / 6 块 x 64 没测（那里 group_list 形状不同，"
        f"且「64 的块里大部分 group 为空」这一情形本次从未产生）；384 的 A 臂在 HBM 预算内是否可达、"
        f"一次 forward 要多久都没测（本次不含任何时间结论）；T 只到 32，没有长 prefill 的 T。"
        f"故结论是「在这些尺寸上分块累加等价」，不是「分块累加等价」。"
    )
    v.note(DOWNGRADE)
    rng = np.random.default_rng(args.seed)
    facts: dict = {"build_note": E.BUILD_NOTE, "build_under_test": but, "layer": L,
                   "scale_downgrade": DOWNGRADE, "experts": NE, "topk": K,
                   "arm_a_form": f"{NE} experts resident, ONE apply call (downgraded from 384)",
                   "is_downgrade_from_384": True}
    mg.check("after runtime init")

    print(f"\n===== E5 layer {L}: arm A = {NE} experts in one call (DOWNGRADE from 384) =====", flush=True)
    raw_all = N.load_raw_layer(ck, L, list(range(NE)))
    mg.check(f"after loading {NE} experts")
    full = N.build_layer(raw_all, NE, dev)
    comm_full = N.comm_method(NE)
    print(N.hbm_line(), flush=True)
    mg.check("after arm A layer")

    widths = [int(s) for s in args.blocks.split(",")]
    plans = {}
    for bw in widths:
        if NE % bw:
            v.check(False, f"block width {bw} does not divide {NE} experts")
            continue
        nb = NE // bw
        blocks = []
        for b in range(nb):
            ids_b = tuple(range(b * bw, (b + 1) * bw))
            pl = LayerPlacement(layer_idx=L, num_experts=NE, num_slots=bw, resident_ids=ids_b)
            sel = list(pl.resident_ids)
            blocks.append({"pl": pl, "slot_tab": pl.safe_slot_table().to(dev),
                           "keep_tab": pl.keep_table().to(dev), "s2l": pl.slot_to_logical(),
                           "layer": N.build_layer(tuple(t[sel] for t in raw_all), bw, dev)})
        plans[bw] = {"nb": nb, "blocks": blocks, "comm": N.comm_method(bw)}
        print(f"  {bw}x{nb}: {nb} block layers built; out-of-block -> dummy slot "
              f"{blocks[0]['pl'].dummy_slot} with keep 0", flush=True)
        mg.check(f"after building {bw}x{nb} block layers")
    if not plans:
        return v.finish(os.path.join(args.out, "e5_result.json"), {**facts, "memguard": mg.facts()})

    cache: dict = {}

    def wfn(e):
        e = int(e)
        if e not in cache:
            ex = ck.expert(L, e)
            cache.clear()
            cache[e] = tuple(C.dequant(ex[f"{p}w"], ex[f"{p}s"]) for p in ("w1", "w3", "w2"))
        return cache[e]

    rows = []
    for T in [int(s) for s in args.tokens.split(",")]:
        x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16)
        ids = torch.from_numpy(np.stack([rng.choice(NE, K, replace=False) for _ in range(T)]).astype(np.int32))
        w = C.route_weights(rng, T)[:, :K].contiguous()
        xd, idd, wd = x.to(dev), ids.to(dev), w.to(dev)

        yA = N.npu_moe(full, comm_full, xd, wd, idd).float().cpu()
        yA2 = N.npu_moe(full, comm_full, xd, wd, idd).float().cpu()
        det, det_how = C.bit_identical(yA2, yA)
        v.check(det, f"T={T} arm A is deterministic across two identical calls: {det_how}")
        v.check(C.nan_inf_count(yA) == 0, f"T={T} arm A output has no non-finite value")

        refs = C.moe_ref_multi(x, ids.long(), w, wfn, lambda e: True,
                              ("bf16", "bf16_acc64", C.CROSS_ACC_VARIANT))
        floor_l2 = max(C.rel_l2(refs["bf16"], refs[C.CROSS_ACC_VARIANT]),
                       C.rel_l2(refs["bf16"], refs["bf16_acc64"]))
        floor_ulp = max(C.ulp_max(refs[C.CROSS_ACC_VARIANT], refs["bf16"]),
                        C.ulp_max(refs["bf16_acc64"], refs["bf16"]))
        tau_l2, tau_ulp = 2 * max(floor_l2, C.BF16_ULP), max(2 * floor_ulp, 4.0)
        print(f"\n-- T={T} floors (measured, not chosen): cross-accumulation rel_l2 {floor_l2:.3e} "
              f"-> tau {tau_l2:.3e}; ulp {floor_ulp:.2f} -> tau {tau_ulp:.2f} "
              f"(bf16 ULP {C.BF16_ULP:.3e})", flush=True)

        setA = [Counter(int(ids[t, k]) for k in range(K) if float(w[t, k]) != 0.0) for t in range(T)]

        for bw in widths:
            if bw not in plans:
                continue
            P = plans[bw]
            nb, comm = P["nb"], P["comm"]

            def run_blocks(keep_override=None, _P=P, _nb=nb, _comm=comm):
                acc = torch.zeros(T, C.HIDDEN, dtype=torch.float32)
                per_block, contrib = [], [Counter() for _ in range(T)]
                for b, blk in enumerate(_P["blocks"]):
                    keep = torch.ones_like(blk["keep_tab"]) if b == keep_override else blk["keep_tab"]
                    r = split_routing(idd, wd, blk["slot_tab"], keep)
                    y = N.npu_moe(blk["layer"], _comm, xd, r.topk_weights, r.topk_ids).float().cpu()
                    acc += y
                    bic, bwc, s2l = r.topk_ids.cpu(), r.topk_weights.cpu(), blk["s2l"]
                    per_block.append(int((bwc != 0).sum()))
                    for t in range(T):
                        for k in range(K):
                            if float(bwc[t, k]) != 0.0:
                                contrib[t][int(s2l[int(bic[t, k])])] += 1
                return acc, per_block, contrib

            yB, per_block, contribB = run_blocks()
            empty = [b for b, n in enumerate(per_block) if n == 0]
            print(f"   {bw}x{nb} T={T} effective routes per block {per_block} "
                  f"(total {sum(per_block)} of {ids.numel()} original); blocks with none: {empty}", flush=True)
            v.check(sum(per_block) == ids.numel(),
                    f"T={T} {bw}x{nb} coverage: {sum(per_block)} effective routes == {ids.numel()} original")
            v.check(len(empty) < nb, f"T={T} {bw}x{nb} at least one block received a route")

            d_l2, d_ulp = C.rel_l2(yB, yA), C.ulp_max(yB, yA)
            ok1 = d_l2 <= tau_l2 and d_ulp <= tau_ulp
            print(f"   {bw}x{nb} T={T} criterion 1: rel_l2 {d_l2:.3e} (tau {tau_l2:.3e}), "
                  f"ulp {d_ulp:.2f} (tau {tau_ulp:.2f})", flush=True)
            v.check(ok1, f"T={T} {bw}x{nb} criterion 1 numeric (rel_l2 {d_l2:.3e}<={tau_l2:.3e}, "
                         f"ulp {d_ulp:.2f}<={tau_ulp:.2f})")
            v.check(C.nan_inf_count(yB) == 0, f"T={T} {bw}x{nb} arm B output has no non-finite value")

            bad = [t for t in range(T) if contribB[t] != setA[t]]
            v.check(not bad, f"T={T} {bw}x{nb} criterion 2a contributing-expert multiset identical for all "
                             f"{T} tokens" + ("" if not bad else f"; differs on {bad[:8]}, e.g. token "
                             f"{bad[0]}: A={dict(setA[bad[0]])} B={dict(contribB[bad[0]])}"))
            cmpB = compare_threshold_sets(yA, yB, yA)
            cmpable = [t for t in range(T) if cmpB[t][1] > 0]
            diff = [t for t in cmpable if not cmpB[t][0]]
            v.check(len(cmpable) == T,
                    f"T={T} {bw}x{nb} criterion 2b coverage: {len(cmpable)}/{T} tokens have a non-empty "
                    f"above-threshold set (sizes {[c[1] for c in cmpB[:8]]}, excluded near the boundary "
                    f"{[c[2] for c in cmpB[:8]]})")
            v.check(not diff, f"T={T} {bw}x{nb} criterion 2b above-threshold output index set identical on "
                              f"{len(cmpable)}/{T} tokens" + ("" if not diff else f"; differs on {diff[:8]}, "
                              f"e.g. token {diff[0]}: symmetric difference {cmpB[diff[0]][3]}"))

            mb = min(1, nb - 1)
            yM, per_m, contribM = run_blocks(keep_override=mb)
            dM_l2, dM_ulp = C.rel_l2(yM, yA), C.ulp_max(yM, yA)
            red1 = not (dM_l2 <= tau_l2 and dM_ulp <= tau_ulp)
            badM = [t for t in range(T) if contribM[t] != setA[t]]
            cmpMl = compare_threshold_sets(yA, yM, yA)
            cmpM = [t for t in range(T) if cmpMl[t][1] > 0]
            diffM = [t for t in cmpM if not cmpMl[t][0]]
            print(f"   {bw}x{nb} T={T} MUTATION block {mb} keep=1: rel_l2 {dM_l2:.3e} ulp {dM_ulp:.2f}; "
                  f"multiset differs on {len(badM)}/{T} tokens; top set on {len(diffM)}/{len(cmpM)}; "
                  f"effective routes {sum(per_m)} vs {ids.numel()}", flush=True)
            v.must_be_red(red1, f"T={T} {bw}x{nb} criterion 1 under 'block {mb} keep table all ones' "
                                f"(rel_l2 {dM_l2:.3e} vs {tau_l2:.3e}, ulp {dM_ulp:.2f} vs {tau_ulp:.2f})")
            v.must_be_red(bool(badM), f"T={T} {bw}x{nb} criterion 2a under 'block {mb} keep table all ones' "
                                      f"({len(badM)}/{T} tokens' multiset changed)")
            v.note(f"T={T} {bw}x{nb} criterion 2b under that mutation: {len(diffM)}/{len(cmpM)} tokens' "
                   f"above-threshold set changed. 2b is the weaker of the two set criteria by construction "
                   "-- a duplicate count that rescales a token's output nearly uniformly need not reorder "
                   "it -- so 2a is the one that carries the discrete guarantee.")

            rows.append({"T": T, "blocks": f"{bw}x{nb}", "rel_l2": d_l2, "ulp": d_ulp,
                         "tau_l2": tau_l2, "tau_ulp": tau_ulp, "floor_l2": floor_l2, "floor_ulp": floor_ulp,
                         "set2a_ok": not bad, "set2b_ok": not diff, "set2b_comparable": len(cmpable),
                         "routes_per_block": per_block, "empty_blocks": empty,
                         "mut_rel_l2": dM_l2, "mut_ulp": dM_ulp, "mut_set2a_tokens": len(badM),
                         "mut_set2b_tokens": len(diffM)})
        print(N.hbm_line(), flush=True)
        mg.check(f"after T={T}")

    facts["rows"] = rows
    facts["memguard"] = mg.facts()
    return v.finish(os.path.join(args.out, "e5_result.json"), facts)


if __name__ == "__main__":
    sys.exit(main())
