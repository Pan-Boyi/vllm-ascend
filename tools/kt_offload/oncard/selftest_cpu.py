# SPDX-License-Identifier: Apache-2.0
"""CPU self-check of the pure-torch half (oncard_common) and of the criteria's two
directions on synthetic data. Opens no device and must not import torch_npu.

Every check prints PASS/FAIL; the last line is the verdict. Run by selftest_cpu.sh.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import oncard_common as C  # noqa: E402

v = C.Verdict("oncard CPU selftest")

# -- no device -----------------------------------------------------------------
v.check("torch_npu" not in sys.modules, "oncard_common imports no torch_npu")

# -- dequant: byte 0x21 = lo nibble 1 (0.5), hi nibble 2 (1.0); scale byte 128 = x2 ---
w = torch.full((1, 16), 0x21, dtype=torch.uint8)
s = torch.full((1, 1), 128, dtype=torch.uint8)
d = C.dequant(w, s)
v.check(d.shape == (1, 32) and float(d[0, 0]) == 1.0 and float(d[0, 1]) == 2.0, "dequant nibble order and E8M0 scale")
v.check(float(C.dequant(w, s, swap_nibbles=True)[0, 0]) == 2.0, "dequant nibble swap arm differs")

# -- MXFP8 emulation -------------------------------------------------------------
p2 = torch.tensor([[1.0, 2.0, 4.0, 0.5] * 8])
v.check(torch.equal(C.mx8_fake_quant(p2), p2), "mx8 keeps powers of two inside one block exactly")
z = torch.zeros(1, 32)
v.check(torch.equal(C.mx8_fake_quant(z), z), "mx8 all-zero block -> zeros (no NaN)")
g = torch.randn(64, 5120)
q = C.mx8_fake_quant(g)
v.check(torch.equal(C.mx8_fake_quant(q), q), "mx8 idempotent")
blk = g.reshape(64, -1, 32)
amax = blk.abs().amax(-1, keepdim=True)
err = ((q.reshape(64, -1, 32) - blk).abs() / amax).max()
v.check(float(err) <= 2.0 ** -3, f"mx8 per-element error <= amax*2^-3 (max {float(err):.3e})")
v.check(0 < C.rel_l2(q, g) < 0.1, f"mx8 rel_l2 in a sane band ({C.rel_l2(q, g):.3e})")
v.check(C.nan_inf_count(C.mx8_fake_quant(g * 1e30)) == 0, "mx8 saturates, no Inf/NaN at large magnitude")

# -- bit_identical: both hazards --------------------------------------------------
a = torch.tensor([0.0, 1.0, float("nan")], dtype=torch.bfloat16)
b = torch.tensor([-0.0, 1.0, float("nan")], dtype=torch.bfloat16)
v.check(C.bit_identical(a, b)[0], "bit_identical: -0 vs +0 and NaN at same place -> identical")
c = torch.tensor([0.0, 1.0078125, float("nan")], dtype=torch.bfloat16)
v.check(not C.bit_identical(a, c)[0], "bit_identical: one ULP apart -> different")
v.check(not C.bit_identical(a, a.float())[0], "bit_identical: dtype mismatch -> different")
v.check(not C.bit_identical(a, torch.tensor([0.0, 1.0, 1.0], dtype=torch.bfloat16))[0], "bit_identical: NaN vs number -> different")

# -- routing + coverage -----------------------------------------------------------
rng = np.random.default_rng(0)
cpu_ids, npu_ids = list(range(100, 384)), list(range(100))
for T in (1, 4, 32):
    ids, wts = C.make_routing(rng, T, cpu_ids, npu_ids, force_expert=191, force_all_rows=True)
    v.check(all(len(set(r)) == 6 for r in ids.tolist()), f"routing T={T}: distinct ids per row")
    v.check(all(191 in r for r in ids.tolist()), f"routing T={T}: forced expert in every row")
    v.check(torch.allclose(wts.sum(1), torch.full((T,), 1.5), atol=1e-5), f"routing T={T}: weights sum to route_scale")
    mask = torch.zeros(384, dtype=torch.bool)
    mask[:100] = True
    cov = C.coverage(ids, mask)
    v.check(not cov.problems(), f"routing T={T}: {cov.line()}")
mask_all = torch.ones(384, dtype=torch.bool)
v.check(any("0% CPU" in p for p in C.coverage(ids, mask_all).problems()), "coverage flags 0% CPU")
v.check(any("100% CPU" in p for p in C.coverage(ids, torch.zeros(384, dtype=torch.bool)).problems()),
        "coverage flags 100% CPU")

# -- the T0 gate in both directions on synthetic experts ---------------------------
# "npu" = an ideal MXFP8-activation implementation: must pass the gate (positive control);
# the same with one expert's w2 doubled (= scale byte +1) must fail it (negative control).
torch.manual_seed(0)
E = 4
experts = {}
for e in range(E):
    wb = torch.randint(0, 256, (C.INTER, C.HIDDEN // 2), dtype=torch.uint8)
    sb = torch.randint(118, 122, (C.INTER, C.HIDDEN // 32), dtype=torch.uint8)
    w3b = torch.randint(0, 256, (C.INTER, C.HIDDEN // 2), dtype=torch.uint8)
    s3b = torch.randint(118, 122, (C.INTER, C.HIDDEN // 32), dtype=torch.uint8)
    w2b = torch.randint(0, 256, (C.HIDDEN, C.INTER // 2), dtype=torch.uint8)
    s2b = torch.randint(118, 122, (C.HIDDEN, C.INTER // 32), dtype=torch.uint8)
    experts[e] = (C.dequant(wb, sb), C.dequant(w3b, s3b), C.dequant(w2b, s2b), s2b, w2b)
T = 4
xs = torch.randn(T, C.HIDDEN).to(torch.bfloat16)
ids = torch.tensor([[0, 1, 2, 3]] * T)
wts = torch.full((T, 4), 1.5 / 4)
wfn = lambda e: experts[e][:3]  # noqa: E731
refs = C.moe_ref_multi(xs, ids, wts, wfn, lambda e: True, ("bf16", "mx8", "mx8_acc64", "bf16_acc64"))
fq, facc = C.rel_l2(refs["mx8"], refs["bf16"]), C.rel_l2(refs["mx8"], refs["mx8_acc64"])
v.check(fq > facc, f"quantisation floor ({fq:.3e}) above the accumulation floor ({facc:.3e})")
tau, tau_row = 2 * max(fq, C.BF16_ULP), 2 * max(C.row_rel_max(refs["mx8"], refs["bf16"]), C.BF16_ULP)
good = refs["mx8_acc64"]
v.check(C.rel_l2(good, refs["bf16"]) <= tau and C.row_rel_max(good, refs["bf16"]) <= tau_row,
        "T0 gate: an ideal MXFP8 implementation passes (positive control)")


def wfn_mut(e):
    w1, w3, _, s2b, w2b = experts[e]
    return (w1, w3, C.dequant(w2b, s2b + 1)) if e == 0 else experts[e][:3]


bad = C.moe_ref_multi(xs, ids, wts, wfn_mut, lambda e: True, ("mx8",))["mx8"]
red = not (C.rel_l2(bad, refs["bf16"]) <= tau and C.row_rel_max(bad, refs["bf16"]) <= tau_row)
v.must_be_red(red, f"T0 gate: one expert's w2 scale +1 in every row ({C.rel_l2(bad, refs['bf16']):.3e} vs tau {tau:.3e})")

# -- tier-B floor: the cross-expert accumulation variant must actually move -----------
#    A floor measured with a variant that cannot move the metric reads as 0 and turns
#    the threshold into "2*ULP", which is not a floor at all (2026-09-20 t1 prefix T=1).
_fl = C.moe_ref_multi(xs, ids, wts, lambda e: experts[e][:3], lambda e: True,
                      ("bf16", "bf16_acc64", C.CROSS_ACC_VARIANT))
v.check(set(_fl) == {"bf16", "bf16_acc64", C.CROSS_ACC_VARIANT}, "moe_ref_multi returns exactly the variants asked for")
_d_acc = C.rel_l2(_fl["bf16"], _fl["bf16_acc64"])
_d_cross = C.rel_l2(_fl["bf16"], _fl[C.CROSS_ACC_VARIANT])
v.check(_fl[C.CROSS_ACC_VARIANT].dtype == _fl["bf16"].dtype and _fl[C.CROSS_ACC_VARIANT].shape == _fl["bf16"].shape,
        "cross-accumulation floor variant has the reference's dtype and shape")
v.check(_d_cross < 1e-2 and _d_acc < 1e-2,
        f"emulation floors stay at rounding scale (cross {_d_cross:.3e}, acc64 {_d_acc:.3e}) -- floors, not bugs")
# ulp_max: the tier-B per-column metric. It must read 0 for an identical tensor,
# count ULPs of the reference's largest value, and stay finite where a per-column
# ratio blows up (one near-zero column with a rounding-sized absolute error).
v.check(C.ulp_max(_fl["bf16"], _fl["bf16"]) == 0.0, "ulp_max(x, x) == 0")
_b = torch.tensor([[1.0, 5e-3]])  # one column 200x smaller, still inside col_rel_max's keep guard
_a = _b.clone()
_a[0, 1] += 3 * C.BF16_ULP  # 3 ULP of the largest value (1.0), landing on the small column
v.check(abs(C.ulp_max(_a, _b) - 3.0) < 1e-6, f"ulp_max counts ULPs of the max ({C.ulp_max(_a, _b):.3f})")
v.check(C.col_rel_max(_a, _b) > 2.0, f"the per-column ratio reads {C.col_rel_max(_a, _b):.3e} for the same "
                                     "3-ULP error -- why tier B stopped using it at small T")
v.check(C.ulp_max(_b * 2, _b) > 100, "ulp_max still separates a wrong tensor by a mile")

# -- §4.31 shared-expert stand-in: does the criterion still see a dropped merge? ------
# npu_moe imports torch only (torch_npu is imported inside the functions that need a
# device), so this half runs on the host. The point of these checks is the GATE
# itself: it has to go RED on a stand-in that inflates the residual stack, because
# that is how the 2026-09-23 runs produced two all-NaN "bit-identical" reference
# steps -- and, at amplitudes just below NaN, would have read a DROPPED MERGE as green.
import npu_moe as NM  # noqa: E402

v.check("torch_npu" not in sys.modules, "npu_moe imports no torch_npu at module level")
_rep = NM.shared_stack_cpu_check_default(layers=8, steps=4, tokens=1, hidden=C.HIDDEN,
                                         inter=C.INTER, verbose=False)
v.check(_rep["max_abs"] < 1e4 and _rep["max_growth"] < 4.0,
        f"fixed stand-in: 8 layers stay bounded (max|h| {_rep['max_abs']:.3e}, "
        f"per-layer growth <= {_rep['max_growth']:.3f})")
v.check(all(not r["same"] for r in _rep["rows"]),
        "fixed stand-in: no-merge differs bit for bit at every layer of every step")
v.check(_rep["min_ulps"] >= NM.MERGE_ULP_MARGIN,
        f"fixed stand-in: weakest merge contribution {_rep['min_ulps']:.1f} bf16 ULP "
        f">= margin {NM.MERGE_ULP_MARGIN} (at step/layer {_rep['min_ulps_at']})")
_g = NM._normalized(NM._shared_bf16_core(C.HIDDEN, C.INTER, 41, torch.device("cpu")))
_x = torch.randn(2, C.HIDDEN).to(torch.bfloat16) * 17.0
v.check(abs(float((NM._rms(_g(_x)) / NM._rms(_x)).mean()) - NM.SHARED_GAIN) < 0.02 * NM.SHARED_GAIN,
        f"fixed stand-in: rms(out)/rms(in) == declared gain {NM.SHARED_GAIN}")
v.check(float(NM._rms(_g(_x * 8.0).float() - 8.0 * _g(_x).float()).max() / NM._rms(8.0 * _g(_x).float()).max()) < 0.05,
        "fixed stand-in: f(8x) == 8 f(x) -- homogeneous, so rms(merge)/rms(h) is layer-invariant")


def _gate_red(fn, **kw) -> bool:
    try:
        NM.shared_stack_cpu_check(fn, layers=8, steps=3, tokens=1, hidden=C.HIDDEN,
                                  verbose=False, **kw)
        return False
    except RuntimeError:
        return True


v.must_be_red(_gate_red(NM.legacy_quadratic_core(C.HIDDEN, C.INTER, 41, torch.device("cpu"))),
              "the pre-2026-09-23 quadratic stand-in (silu(gate)*up, /32 weights, no RMSNorm)")
v.must_be_red(_gate_red(lambda x: (x.float() * 40.0).to(x.dtype)),
              "a merely EXPANDING stand-in (gain 40/layer): amplitude runs away, merge rounds off")
v.must_be_red(_gate_red(_g, merge_rel=2.0 ** -12),
              "a merge contributing 2^-12 of rms(h): below the bf16 ULP margin even though bounded")
v.check(not _gate_red(_g), "the same gate stays GREEN on the fixed stand-in (it is not red for everyone)")

# -- T1: explicit set and the split contract with the real placement code ------------
try:
    sys.path.insert(0, HERE)
    import t1_offload_on as T1

    from vllm_ascend.kt_offload.placement import LayerPlacement, check_split, split_routing

    res = T1.explicit_ids(100, 0)
    v.check(len(res) == 100 and 0 not in res and 383 in res and res == sorted(set(res)),
            "T1 explicit set: 100 ids, 0 excluded, 383 included, ascending")
    pl = LayerPlacement(layer_idx=20, num_experts=384, num_slots=100, resident_ids=tuple(res))
    v.check(all(pl.slot_of(i) != i for i in res[1:]), "T1 explicit set: slot != logical id for resident experts")
    ids, wts = C.make_routing(np.random.default_rng(1), 32, pl.cpu_ids, res)
    st, kt = pl.safe_slot_table(torch.int32), pl.keep_table(torch.float32)
    n_w, n_ids = split_routing(ids, wts, st, kt)
    v.check(not check_split(pl, ids.long(), wts, n_ids.long(), n_w), "split contract holds with the real tables")
    rolled = ((st.long() + 1) % pl.num_resident).to(st.dtype)
    n_w2, n_ids2 = split_routing(ids, wts, rolled, kt)
    v.must_be_red(bool(check_split(pl, ids.long(), wts, n_ids2.long(), n_w2)), "rolled slot table breaks the split")
    n_w3, n_ids3 = split_routing(ids, wts, st, torch.ones_like(kt))
    v.must_be_red(bool(check_split(pl, ids.long(), wts, n_ids3.long(), n_w3)), "keep table all ones breaks the split")
    dup = sum(1 for r in n_ids.tolist() if len(set(r)) < len(r))
    v.check(dup >= 1, f"NPU half has rows with a repeated slot ({dup})")
except ImportError as e:
    v.check(False, f"T1 CPU checks could not import ({e}); run with PYTHONPATH=<tree>:<vllm tree>")

# Importing vllm_ascend (placement above) loads the platform plugin, which imports
# torch_npu. Importing is allowed; initialising a device is not.
_npu = getattr(torch, "npu", None)
_inited = bool(_npu.is_initialized()) if _npu is not None and hasattr(_npu, "is_initialized") else False
v.check(not _inited, f"no NPU device initialised (torch_npu imported: {'torch_npu' in sys.modules}, "
                     f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')!r})")
sys.exit(v.finish())
