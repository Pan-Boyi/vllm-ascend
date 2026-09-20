# SPDX-License-Identifier: Apache-2.0
"""T1: offload ON, one layer, eager, arm by arm (design doc 上卡草案 M1 步骤 2).

Two resident sets on the same layer, each through the real KTLayerOffload
(plan_from_config -> attach_after_init wrapping the post-load hook -> route ->
quant_method.apply -> merge) and the real kt-kernel wrapper:
  prefix    N = --n (default 100): experts 0..N-1 resident
  explicit  --n ids, non-contiguous, logical 0 NOT resident, 383 resident, so slot != id
            everywhere (an identity slot table would be wrong on every route)

Reference layer: the same layer with all 384 experts resident (T0's build).

Arms, per T in --tokens (default 1,4,32):
  A  NPU half (routed_out before merge) vs the full-resident layer fed the ORIGINAL
     ids with the CPU routes' weights set to 0 -> bit-identical (bytes or values).
  B  kt's output (tapped at sync_forward) vs the bf16 torch reference over the CPU
     routes only -> M0b tiers: A (vs fp32) 2*max(floor,ULP) for rel_l2 and per column;
     B (vs bf16 emulation) 2*max(acc floor, ULP/16) for rel_l2, and the largest absolute
     deviation in bf16 ULPs of the output's max (max(2*emulation floor, 4 ULP)).
     Why not a per-column ratio in tier B: at small T a near-cancelling column makes the
     ratio unbounded for a rounding-sized error, and both emulations round identically
     there, so its "floor" reads 0 (2026-09-20 first t1 pass: prefix T=1 8.5e-3 vs 2*ULP,
     T=4 marginal 1.155e-2 vs 1.156e-2). The ratio is still printed as a diagnostic.
  C  merged output vs the full-resident layer on all routes. floor = what fp8
     activations cost on the CPU routes = |npu_full(CPU routes only) - ref_bf16(CPU routes)|,
     both normalised by |npu_full|; tau = max(1.5*floor, BF16_ULP).
  coverage  CPU-route share printed; 0% or 100% fails; >=1 row with >=2 CPU routes
     (the NPU kernel then sees the dummy slot twice in one row) required.
  split  vllm_ascend.kt_offload.placement.check_split on the actual NPU half.
Mutation arms (must be RED):
  slot table rolled by one slot (explicit set) -> A; keep table all ones (CPU routes
  also computed on the NPU dummy slot, double count) -> A and C; reference-side
  nibble swap -> B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oncard_common as C  # noqa: E402


def explicit_ids(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    pool = np.arange(1, C.N_EXP - 1)  # 0 excluded, 383 added below
    ids = sorted(int(i) for i in rng.choice(pool, n - 1, replace=False)) + [C.N_EXP - 1]
    if ids == list(range(ids[0], ids[0] + n)):
        raise AssertionError("explicit set came out contiguous")
    return ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tokens", default="1,4,32")
    ap.add_argument("--sets", default="prefix,explicit")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--numa", default="1", help="kt NUMA node list, e.g. 1 or 0,1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()

    import npu_moe as N
    from vllm_ascend.kt_offload.placement import check_split

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    L = args.layer
    v = C.Verdict("T1 offload-on single layer")
    numa = [int(s) for s in args.numa.split(",")]
    rows = []

    print(f"\n===== T1 layer {L}: reference = all 384 resident =====", flush=True)
    raw_all = N.load_raw_layer(ck, L, list(range(C.N_EXP)))
    full = N.build_layer(raw_all, C.N_EXP, dev)
    comm_full = N.comm_method(C.N_EXP)

    cache = {}

    def wfn(e):
        if e not in cache:
            ex = ck.expert(L, e)
            cache.clear()
            cache[e] = tuple(C.dequant(ex[f"{p}w"], ex[f"{p}s"]) for p in ("w1", "w3", "w2"))
        return cache[e]

    def wfn_nibble(e):
        ex = ck.expert(L, e)
        return tuple(C.dequant(ex[f"{p}w"], ex[f"{p}s"], swap_nibbles=True) for p in ("w1", "w3", "w2"))

    for set_name in args.sets.split(","):
        if set_name == "prefix":
            cfg = N.kt_cfg("prefix", args.n, None, args.threads, numa, graph=False)
        else:
            path = os.path.join(args.out, "explicit_resident.json")
            ids_res = explicit_ids(args.n, args.seed)
            with open(path, "w") as f:
                json.dump({"layers": {str(i): ids_res for i in range(40)}}, f)
            cfg = N.kt_cfg("explicit", 0, path, args.threads, numa, graph=False)
        off, plan = N.make_offload(cfg, L)
        pl = off.placement
        res_ids = list(pl.resident_ids)
        cpu_ids = pl.cpu_ids
        print(f"\n===== T1 layer {L} set={set_name}: {pl.num_resident} resident in {pl.num_slots} slots, "
              f"{len(cpu_ids)} on CPU; first ids {res_ids[:8]} ... =====", flush=True)
        raw_res = tuple(t[res_ids] for t in raw_all)  # slot order = ascending logical id
        t0 = time.time()
        lay = N.build_layer(raw_res, pl.num_slots, dev, offload=off)
        print(f"[build] offload layer + kt load {time.time() - t0:.1f}s; {N.hbm_line()}", flush=True)
        if off.wrapper is None:
            v.check(False, f"{set_name}: kt wrapper was not built (no CPU experts?)")
            continue
        tap = N.KtTap(off.wrapper)
        comm = N.comm_method(pl.num_slots)
        mask = pl.resident_mask()
        keep_dev = mask.to(torch.float32).to(dev)
        rng = np.random.default_rng(args.seed + 1000 + L)
        good_slot = {k: t.clone() for k, t in off._slot_tables.items()}
        good_keep = {k: t.clone() for k, t in off._keep_tables.items()}

        for T in [int(s) for s in args.tokens.split(",")]:
            x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16)
            ids, w = C.make_routing(rng, T, cpu_ids, res_ids)
            cov = C.coverage(ids, mask)
            print(f"-- {set_name} T={T} {cov.line()}", flush=True)
            for p in cov.problems(need_dup=True):
                v.check(False, f"{set_name} T={T} {p}")
            v.check(not cov.problems(need_dup=True), f"{set_name} T={T} coverage")
            xd, idd, wd = x.to(dev), ids.to(dev), w.to(dev)

            t0 = time.time()
            pre, merged, (npu_ids, npu_w) = N.offload_forward(off, lay, comm, xd, wd, idd)
            t_off = time.time() - t0
            kt_out = tap.last.float().cpu() if tap.last is not None else None
            errs = check_split(pl, ids.long(), w, npu_ids.cpu().long(), npu_w.cpu())
            v.check(not errs, f"{set_name} T={T} split contract: {errs or 'ok'}")
            dup_rows = sum(1 for r in npu_ids.cpu().tolist() if len(set(r)) < len(r))
            v.check(dup_rows >= 1, f"{set_name} T={T} NPU half has {dup_rows} rows with a repeated slot id")

            # A: NPU half vs full-resident with CPU weights zeroed (original logical ids)
            y_res_only = N.npu_moe(full, comm_full, xd, wd * keep_dev[idd.long()], idd).cpu()
            same, how = C.bit_identical(pre.cpu(), y_res_only)
            v.check(same, f"{set_name} T={T} arm A (NPU half == full-resident, CPU weights 0): {how}"
                          + ("" if same else f"; rel_l2 {C.rel_l2(pre.cpu(), y_res_only):.3e}"))

            # B: kt vs bf16 torch reference on the CPU routes
            if kt_out is None:
                v.check(False, f"{set_name} T={T} arm B: kt tap saw no output")
            else:
                is_cpu = lambda e: not bool(mask[e])  # noqa: E731
                refs = C.moe_ref_multi(x, ids.long(), w, wfn, is_cpu,
                                       ("fp32", "bf16", "bf16_acc64", C.CROSS_ACC_VARIANT))
                fA, fAc = C.rel_l2(refs["bf16"], refs["fp32"]), C.col_rel_max(refs["bf16"], refs["fp32"])
                # Tier-B floor = the spread between two legal bf16 emulations: the
                # GEMM accumulation inside an expert (acc64) and the cross-expert
                # sum (fp32, reverse order -- the one degree of freedom kt has).
                fB = max(C.rel_l2(refs["bf16"], refs["bf16_acc64"]),
                         C.rel_l2(refs["bf16"], refs[C.CROSS_ACC_VARIANT]))
                tA, tAc = 2 * max(fA, C.BF16_ULP), 2 * max(fAc, C.BF16_ULP)
                tB = 2 * max(fB, C.BF16_ULP / 16)
                # Tier-B per column is measured in ULPs of the output's largest
                # value, not as a per-column ratio: at small T a near-cancelling
                # column makes the ratio unbounded while the absolute error stays
                # at rounding scale, and the emulation-vs-emulation floor for the
                # ratio is 0 there (see C.ulp_max). Floor = the same metric between
                # the two legal emulations; allowance never below 4 ULP.
                fBu = max(C.ulp_max(refs["bf16_acc64"], refs["bf16"]),
                          C.ulp_max(refs[C.CROSS_ACC_VARIANT], refs["bf16"]))
                tBu = max(2 * fBu, 4.0)

                def judge(target_fp32, target_emu):
                    vals = (C.rel_l2(kt_out, target_fp32), C.col_rel_max(kt_out, target_fp32),
                            C.rel_l2(kt_out, target_emu), C.ulp_max(kt_out, target_emu))
                    return vals, vals[0] <= tA and vals[1] <= tAc and vals[2] <= tB and vals[3] <= tBu

                vals, okB = judge(refs["fp32"], refs["bf16"])
                print(f"   arm B floors A {fA:.3e}/{fAc:.3e} -> tau {tA:.3e}/{tAc:.3e}; "
                      f"B l2 {fB:.3e} -> {tB:.3e}, ulp {fBu:.2f} -> {tBu:.2f}; "
                      f"kt vs fp32 {vals[0]:.3e}/{vals[1]:.3e} | vs emu {vals[2]:.3e}/{vals[3]:.2f} ULP "
                      f"(col ratio {C.col_rel_max(kt_out, refs['bf16']):.3e}, diagnostic)")
                v.check(okB, f"{set_name} T={T} arm B (kt vs bf16 reference, M0b tiers)")
                mrefs = C.moe_ref_multi(x, ids.long(), w, wfn_nibble, is_cpu, ("fp32", "bf16"))
                _, okBm = judge(mrefs["fp32"], mrefs["bf16"])
                v.must_be_red(not okBm, f"{set_name} T={T} arm B, reference nibble swap")

            # C: merged vs NPU full-expert
            y_full = N.npu_moe(full, comm_full, xd, wd, idd).float().cpu()
            y_cpu_on_npu = N.npu_moe(full, comm_full, xd, wd * (1 - keep_dev[idd.long()]), idd).float().cpu()
            ref_cpu = refs["bf16"] if kt_out is not None else None
            if ref_cpu is not None:
                floor = C.rel_l2_to(y_cpu_on_npu, ref_cpu, y_full)
                tauC = max(1.5 * floor, C.BF16_ULP)
                dC = C.rel_l2(merged.float().cpu(), y_full)
                sign = float((torch.sign(merged.float().cpu()) == torch.sign(y_full)).float().mean())
                print(f"   arm C floor {floor:.3e} -> tau {tauC:.3e}; merged vs npu_full {dC:.3e}; sign agreement {sign:.4f}")
                v.check(dC <= tauC, f"{set_name} T={T} arm C (merged vs NPU full-expert within 1.5x fp8 floor)")
                rows.append(dict(set=set_name, T=T, cpu_share=cov.cpu_share, armA=how, armC=dC, tauC=tauC,
                                 floorC=floor, armB=vals, t_offload_s=t_off))

            # mutation arms on the production object's tables (by attribute name)
            arms = []
            if set_name == "explicit":
                rolled = {k: ((t.long() + 1) % pl.num_resident).to(t.dtype) for k, t in good_slot.items()}
                arms.append(("slot table rolled by one", rolled, good_keep))
            ones = {k: torch.ones_like(t) for k, t in good_keep.items()}
            arms.append(("keep table all ones (double count)", good_slot, ones))
            for name, st, kt in arms:
                off._slot_tables, off._keep_tables = st, kt
                try:
                    pre_m, merged_m, _ = N.offload_forward(off, lay, comm, xd, wd, idd)
                finally:
                    off._slot_tables, off._keep_tables = good_slot, good_keep
                sameA, _ = C.bit_identical(pre_m.cpu(), y_res_only)
                v.must_be_red(not sameA, f"{set_name} T={T} arm A under '{name}'")
                if ref_cpu is not None and name.startswith("keep"):
                    dCm = C.rel_l2(merged_m.float().cpu(), y_full)
                    v.must_be_red(dCm > tauC, f"{set_name} T={T} arm C under '{name}' ({dCm:.3e} vs {tauC:.3e})")
        del lay, off, tap
        torch.npu.empty_cache()
        print(N.hbm_line(), flush=True)
    return v.finish(os.path.join(args.out, "t1_result.json"), {"rows": rows})


if __name__ == "__main__":
    sys.exit(main())
