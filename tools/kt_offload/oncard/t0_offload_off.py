# SPDX-License-Identifier: Apache-2.0
"""T0: offload OFF. One V4.1 layer, all 384 experts resident (MXFP4 weights, W4A8),
through the real DS W4A8 method + ALLGATHER comm at world_size=1, against torch.

Why first: ALLGATHER + W4A8 (npu_moe_init_routing quant_mode=3) on A5/V4.1 has
never run -- the multi-card baseline uses MC2 (design §g). It runs with offload
off as well, so it is checked on its own before any offload arm.

Per layer L and T in --tokens:
  det     the same input twice -> bit-identical (bytes equal or values equal)
  floor   q    = rel_l2(ref_mx8, ref_bf16)       (what MXFP8 activations alone cost)
          acc  = rel_l2(ref_mx8, ref_mx8_acc64)  (accumulation order)
  gate    rel_l2(npu, ref_bf16) <= tau = 2*max(q, BF16_ULP), and the worst row too
  finite  no NaN/Inf in the NPU output
  report  rel_l2(npu, ref_mx8), rel_l2(npu, ref_mx8_symclamp) (which clamp reading the
          kernel matches), col/row maxima. Not gating: the emulation's fidelity to the
          kernel is what is being measured.
  mutate  nibble swap of one expert (w13+w2) / that expert's whole w2 scale +1 --
          applied to the NPU bytes before the post-load hook; the expert is routed in
          every row; the gate must go RED for both.
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="20", help="comma list from {0,20,39}")
    ap.add_argument("--tokens", default="1,4,32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x-std", type=float, default=1.0)
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()

    import npu_moe as N

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    v = C.Verdict("T0 offload-off single layer")
    rows = []
    for L in [int(s) for s in args.layers.split(",")]:
        print(f"\n===== T0 layer {L}: 384 experts resident, ALLGATHER world_size=1 =====", flush=True)
        raw = N.load_raw_layer(ck, L, list(range(C.N_EXP)))
        comm = N.comm_method(C.N_EXP)
        layer = N.build_layer(raw, C.N_EXP, dev)
        rng = np.random.default_rng(args.seed + L)
        e_mut = 191
        cache = {}

        def wfn(e, _L=L, _cache=cache):
            if e not in _cache:
                ex = ck.expert(_L, e)
                _cache.clear()  # one expert at a time: host memory stays flat
                _cache[e] = tuple(C.dequant(ex[f"{p}w"], ex[f"{p}s"]) for p in ("w1", "w3", "w2"))
            return _cache[e]

        cases = []
        for T in [int(s) for s in args.tokens.split(",")]:
            x = (torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)) * args.x_std).to(torch.bfloat16)
            everyone = list(range(C.N_EXP))
            ids, w = C.make_routing(rng, T, everyone[: C.N_EXP // 2], everyone[C.N_EXP // 2:],
                                    force_expert=e_mut, force_all_rows=True)
            xd, idd, wd = x.to(dev), ids.to(dev), w.to(dev)
            t0 = time.time()
            y1 = N.npu_moe(layer, comm, xd, wd, idd).float().cpu()
            y2 = N.npu_moe(layer, comm, xd, wd, idd).float().cpu()
            t_npu = time.time() - t0
            same, how = C.bit_identical(y1, y2)
            v.check(same, f"L{L} T={T} determinism (two runs): {how}")
            v.check(C.nan_inf_count(y1) == 0, f"L{L} T={T} NPU output finite (non-finite {C.nan_inf_count(y1)})")
            t0 = time.time()
            refs = C.moe_ref_multi(x, ids.long(), w, wfn, lambda e: True,
                                   ("bf16", "mx8", "mx8_acc64", "mx8_symclamp"))
            t_ref = time.time() - t0
            fq = C.rel_l2(refs["mx8"], refs["bf16"])
            facc = C.rel_l2(refs["mx8"], refs["mx8_acc64"])
            fq_row = C.row_rel_max(refs["mx8"], refs["bf16"])
            tau, tau_row = 2 * max(fq, C.BF16_ULP), 2 * max(fq_row, C.BF16_ULP)
            d_bf = C.rel_l2(y1, refs["bf16"])
            d_row = C.row_rel_max(y1, refs["bf16"])
            d_mx = C.rel_l2(y1, refs["mx8"])
            d_sym = C.rel_l2(y1, refs["mx8_symclamp"])
            print(f"L{L} T={T}: floors q={fq:.3e} (row {fq_row:.3e}) acc={facc:.3e} -> tau {tau:.3e}/row {tau_row:.3e}"
                  f" | npu vs bf16 {d_bf:.3e} (row {d_row:.3e}, col {C.col_rel_max(y1, refs['bf16']):.3e})"
                  f" | vs mx8 {d_mx:.3e} | vs mx8_symclamp {d_sym:.3e} | npu {t_npu:.2f}s ref {t_ref:.1f}s", flush=True)
            v.check(d_bf <= tau and d_row <= tau_row, f"L{L} T={T} NPU within 2x the MXFP8-activation floor")
            v.note(f"L{L} T={T} clamp reading: npu closer to {'one-sided gate clamp' if d_mx <= d_sym else 'SYMMETRIC gate clamp'}"
                   f" (mx8 {d_mx:.3e} vs symclamp {d_sym:.3e}); npu {'closer to mx8 than to bf16' if d_mx < d_bf else 'NOT closer to mx8 than to bf16'}")
            cases.append(dict(T=T, x=x, ids=ids, w=w, ref_bf16=refs["bf16"], tau=tau, tau_row=tau_row))
            rows.append(dict(layer=L, T=T, floor_q=fq, floor_acc=facc, tau=tau, npu_vs_bf16=d_bf, npu_row=d_row,
                             npu_vs_mx8=d_mx, npu_vs_mx8_symclamp=d_sym, deterministic=same))
        del layer
        torch.npu.empty_cache()

        # mutation arms on the NPU side: bytes edited before the post-load hook
        def nibble(name, t, e=e_mut):
            if name in ("w13_weight", "w2_weight"):
                b = t[e]
                t[e] = ((b << 4) | (b >> 4)) & 0xFF

        def scale_plus1(name, t, e=e_mut):
            if name == "w2_weight_scale":
                t[e] += 1

        for arm, fn in (("nibble swap, expert %d" % e_mut, nibble), ("w2 scale +1, expert %d" % e_mut, scale_plus1)):
            mlayer = N.build_layer(raw, C.N_EXP, dev, mutate=fn)
            for c in cases:
                ym = N.npu_moe(mlayer, comm, c["x"].to(dev), c["w"].to(dev), c["ids"].to(dev)).float().cpu()
                d, dr = C.rel_l2(ym, c["ref_bf16"]), C.row_rel_max(ym, c["ref_bf16"])
                red = not (d <= c["tau"] and dr <= c["tau_row"])
                print(f"   mutation L{L} T={c['T']} {arm}: rel {d:.3e} row {dr:.3e} (tau {c['tau']:.3e}/{c['tau_row']:.3e})")
                v.must_be_red(red, f"L{L} T={c['T']} {arm}")
            del mlayer
            torch.npu.empty_cache()
        del raw
        print(N.hbm_line(), flush=True)
    return v.finish(os.path.join(args.out, "t0_result.json"), {"rows": rows})


if __name__ == "__main__":
    sys.exit(main())
