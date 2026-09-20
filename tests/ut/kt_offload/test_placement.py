# SPDX-License-Identifier: Apache-2.0
"""kt offload placement and routing split -- CPU only, no NPU, no kt-kernel.

Each contract has an oracle (``check_split`` / explicit asserts) and a
mutation arm that breaks the implementation in one way and must turn the
oracle red. A green oracle is only evidence if its mutation arm is red.
"""

from __future__ import annotations

import json

import pytest
import torch

from vllm_ascend.kt_offload import placement as pl
from vllm_ascend.kt_offload.placement import (
    CPU_SLOT,
    LayerPlacement,
    build_plan,
    check_split,
    explicit_resident_sets,
    frequency_resident_sets,
    frequency_table_report,
    prefix_resident_sets,
    resident_bytes_per_slot,
    split_routing,
)

E, TOPK, L = 384, 6, 40  # DSV4.1


def _freq_table(seed=0, layers=L, experts=E):
    g = torch.Generator().manual_seed(seed)
    # heavy-tailed counts, a different hot set per layer
    return (torch.rand(layers, experts, generator=g) ** 4 * 1000).round()


def _random_topk(tokens, seed=1, experts=E, topk=TOPK, id_dtype=torch.int32, w_dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    # distinct ids per row, like the router and like the profile-run argsort
    ids = torch.argsort(torch.rand(tokens, experts, generator=g), dim=1)[:, :topk].to(id_dtype)
    w = (torch.rand(tokens, topk, generator=g) + 0.05).to(w_dtype) * 1.5
    return ids, w


def _plans():
    freq = _freq_table()
    sets_explicit = [
        sorted(torch.randperm(E, generator=torch.Generator().manual_seed(100 + i))[: 60 + i].tolist()) for i in range(L)
    ]
    return {
        "prefix": build_plan(prefix_resident_sets(L, E, 100), E),
        "frequency": build_plan(frequency_resident_sets(freq, 100), E),
        "explicit_unequal": build_plan(sets_explicit, E),
    }


PLANS = _plans()


# ---------------------------------------------------------------------------
# resident-set construction
# ---------------------------------------------------------------------------


def test_prefix_is_identity_slots():
    p = PLANS["prefix"].layer(7)
    assert p.resident_ids == tuple(range(100))
    assert [p.slot_of(i) for i in range(100)] == list(range(100))
    assert all(p.slot_of(i) == CPU_SLOT for i in range(100, E))


def test_frequency_sets_are_top_n_per_layer_and_differ_across_layers():
    freq = _freq_table()
    sets = frequency_resident_sets(freq, 100)
    assert len(sets) == L and all(len(s) == 100 for s in sets)
    for layer in (0, 17, 39):
        chosen = torch.tensor(sets[layer])
        rest = torch.tensor([i for i in range(E) if i not in set(sets[layer])])
        assert freq[layer, chosen].min() >= freq[layer, rest].max()
    assert len({tuple(s) for s in sets}) > 1, "fixture should give different sets per layer"
    assert any(s != list(range(s[0], s[0] + 100)) for s in sets), "fixture should be non-contiguous"


def test_frequency_tie_break_is_lower_id_first():
    freq = torch.zeros(2, 8)
    freq[0, [1, 3, 5, 7]] = 5.0  # four-way tie for three places
    freq[1] = 1.0  # all tied
    assert frequency_resident_sets(freq, 3) == [[1, 3, 5], [0, 1, 2]]


def test_frequency_rejects_bad_tables():
    with pytest.raises(ValueError):
        frequency_resident_sets(torch.ones(E), 10)
    bad = torch.ones(2, 8)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        frequency_resident_sets(bad, 2)
    with pytest.raises(ValueError):
        frequency_resident_sets(-torch.ones(2, 8), 2)
    with pytest.raises(ValueError):
        frequency_resident_sets(torch.ones(2, 8), 9)


def test_frequency_report_flags_unequal_row_sums():
    freq = torch.full((3, 8), 6.0)  # each row: 8 tokens x top6 = 48
    sets = frequency_resident_sets(freq, 2)
    rep = frequency_table_report(freq, sets, top_k=6)
    assert rep["row_sums_equal"] and rep["row_sums_multiple_of_topk"]
    assert rep["npu_hit_rate"] == pytest.approx([0.25] * 3)
    freq[1, 0] += 6  # one layer saw an extra token: aggregation error
    rep = frequency_table_report(freq, sets, top_k=6)
    assert not rep["row_sums_equal"]


def test_explicit_accepts_dict_and_list_and_validates(tmp_path):
    spec = {"layers": {"0": [5, 1], "1": [383]}}
    assert explicit_resident_sets(spec, 2, E) == [[1, 5], [383]]
    assert explicit_resident_sets([[2], [3, 4]], 2, E) == [[2], [3, 4]]
    with pytest.raises(ValueError):
        explicit_resident_sets({"0": [1]}, 2, E)  # missing layer 1
    with pytest.raises(ValueError):
        explicit_resident_sets([[1, 1], [2]], 2, E)
    with pytest.raises(ValueError):
        explicit_resident_sets([[384], [2]], 2, E)


def test_unequal_layers_pad_to_max_slots_and_empty_layer_is_refused():
    plan = PLANS["explicit_unequal"]
    assert plan.num_slots == 60 + L - 1
    small = plan.layer(0)
    assert small.num_resident == 60 and small.unused_slots == list(range(60, plan.num_slots))
    with pytest.raises(ValueError):
        build_plan([[0, 1], []], E)


def test_plan_from_config_files(tmp_path):
    freq = _freq_table(layers=3, experts=16)
    fpath = tmp_path / "freq.pt"
    torch.save({"counts": freq}, fpath)
    cfg = type(
        "C",
        (),
        dict(
            resident_strategy="frequency",
            num_resident_experts=4,
            frequency_table_path=str(fpath),
            resident_experts_path=None,
        ),
    )
    plan = pl.plan_from_config(cfg, 3, 16, top_k=2)
    assert [list(p.resident_ids) for p in plan.layers] == frequency_resident_sets(freq, 4)
    with pytest.raises(ValueError):
        pl.plan_from_config(cfg, 4, 16, top_k=2)  # wrong layer count
    jpath = tmp_path / "res.json"
    jpath.write_text(json.dumps({"layers": {"0": [3], "1": [0, 15], "2": [7, 8, 9]}}))
    cfg2 = type(
        "C",
        (),
        dict(
            resident_strategy="explicit",
            num_resident_experts=0,
            frequency_table_path=None,
            resident_experts_path=str(jpath),
        ),
    )
    plan2 = pl.plan_from_config(cfg2, 3, 16, top_k=2)
    assert plan2.num_slots == 3 and plan2.layer(1).resident_ids == (0, 15)


# ---------------------------------------------------------------------------
# logical <-> slot round trip
# ---------------------------------------------------------------------------


def _roundtrip_errors(p: LayerPlacement) -> list[str]:
    errs = []
    l2s, s2l = p.logical_to_slot(), p.slot_to_logical()
    for lid in p.resident_ids:
        if int(s2l[l2s[lid]]) != lid:
            errs.append(f"logical {lid} -> slot {int(l2s[lid])} -> {int(s2l[l2s[lid]])}")
    for slot in range(p.num_resident):
        if int(l2s[s2l[slot]]) != slot:
            errs.append(f"slot {slot} does not round-trip")
    if int((l2s != CPU_SLOT).sum()) != p.num_resident:
        errs.append("resident count mismatch")
    if any(int(s2l[s]) != CPU_SLOT for s in p.unused_slots):
        errs.append("unused slot claims an expert")
    if sorted(p.cpu_ids + list(p.resident_ids)) != list(range(p.num_experts)):
        errs.append("resident + cpu != all experts")
    if not torch.equal(p.resident_mask(), l2s != CPU_SLOT):
        errs.append("kt mask disagrees with slot table")
    return errs


@pytest.mark.parametrize("name", list(PLANS))
def test_roundtrip(name):
    checked = 0
    for p in PLANS[name].layers:
        assert _roundtrip_errors(p) == [], (name, p.layer_idx)
        checked += p.num_resident
    assert checked >= L * 60  # coverage: every layer, every resident expert


def test_roundtrip_mutation_is_red():
    # Mutant: slots assigned in frequency-rank order rather than ascending
    # logical order, while slot_to_logical still assumes ascending order.
    p = PLANS["frequency"].layer(3)
    bad = LayerPlacement.__new__(LayerPlacement)
    object.__setattr__(bad, "layer_idx", p.layer_idx)
    object.__setattr__(bad, "num_experts", p.num_experts)
    object.__setattr__(bad, "num_slots", p.num_slots)
    object.__setattr__(bad, "resident_ids", p.resident_ids)
    l2s = [CPU_SLOT] * E
    for slot, lid in enumerate(reversed(p.resident_ids)):
        l2s[lid] = slot
    object.__setattr__(bad, "_l2s", tuple(l2s))
    assert _roundtrip_errors(bad) != []


# ---------------------------------------------------------------------------
# routing split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PLANS))
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("w_dtype", [torch.float32, torch.bfloat16])
def test_split_contract(name, id_dtype, w_dtype):
    routes = 0
    cpu_routes = 0
    for p in PLANS[name].layers[::7]:
        ids, w = _random_topk(257, seed=p.layer_idx, id_dtype=id_dtype, w_dtype=w_dtype)
        nw, nid = split_routing(ids, w, p.safe_slot_table(id_dtype), p.keep_table(w_dtype))
        assert check_split(p, ids, w, nid, nw) == []
        assert nid.shape == ids.shape and nw.shape == w.shape and int(nid.min()) >= 0
        routes += ids.numel()
        cpu_routes += int((~p.resident_mask()[ids.long()]).sum())
    # coverage: both sides of the split were exercised
    assert routes > 0 and 0 < cpu_routes < routes


def test_split_does_not_mutate_inputs():
    p = PLANS["frequency"].layer(0)
    ids, w = _random_topk(33)
    ids0, w0 = ids.clone(), w.clone()
    split_routing(ids, w, p.safe_slot_table(), p.keep_table())
    assert torch.equal(ids, ids0) and torch.equal(w, w0)


def _mut_minus_one(ids, w, p):  # CPU routes as -1 (Ascend routing kernel rejects it)
    l2s = p.logical_to_slot().to(ids.dtype)
    return l2s[ids.long()], w * p.keep_table(w.dtype)[ids.long()]


def _mut_keep_cpu_weight(ids, w, p):  # forget to zero CPU weights -> double count
    return p.safe_slot_table(ids.dtype)[ids.long()], w.clone()


def _mut_identity_slots(ids, w, p):  # assume slot == logical id (only true for prefix)
    res = p.resident_mask()[ids.long()]
    return torch.where(res, ids, torch.zeros_like(ids)), w * p.keep_table(w.dtype)[ids.long()]


def _mut_zero_resident(ids, w, p):  # inverted mask: zero the resident routes
    return p.safe_slot_table(ids.dtype)[ids.long()], w * (1 - p.keep_table(w.dtype)[ids.long()])


def _mut_unused_slot_dummy(ids, w, p):  # dummy target = an unloaded tail slot
    t = p.logical_to_slot()
    t = torch.where(t == CPU_SLOT, torch.full_like(t, p.num_slots - 1), t).to(ids.dtype)
    return t[ids.long()], w * p.keep_table(w.dtype)[ids.long()]


MUTANTS = {
    "minus_one": _mut_minus_one,
    "keep_cpu_weight": _mut_keep_cpu_weight,
    "identity_slots": _mut_identity_slots,
    "zero_resident": _mut_zero_resident,
    "unused_slot_dummy": _mut_unused_slot_dummy,
}


@pytest.mark.parametrize("mutant", list(MUTANTS))
def test_split_mutation_arms_are_red(mutant):
    # frequency for the non-identity slots; explicit_unequal for unused slots
    name = "explicit_unequal" if mutant == "unused_slot_dummy" else "frequency"
    p = PLANS[name].layer(0)
    ids, w = _random_topk(257)
    nid, nw = MUTANTS[mutant](ids, w, p)
    assert check_split(p, ids, w, nid, nw) != [], f"oracle did not catch mutant {mutant}"


def test_identity_slot_mutant_is_invisible_on_prefix():
    # Why frequency placement must be in the suite: on prefix the identity
    # mapping is correct, so a prefix-only test would pass the bug.
    p = PLANS["prefix"].layer(0)
    ids, w = _random_topk(257)
    nid, nw = _mut_identity_slots(ids, w, p)
    assert check_split(p, ids, w, nid, nw) == []


# ---------------------------------------------------------------------------
# numerical semantics: resident (slot-packed) + CPU == all experts
# ---------------------------------------------------------------------------


def _moe_ref(x, ids, w, experts_w):
    # experts_w[e]: [H, H] linear stand-in for one expert; output = sum_k w_k * x @ W_e
    out = torch.zeros_like(x)
    for k in range(ids.shape[1]):
        out += w[:, k : k + 1] * torch.einsum("th,thd->td", x, experts_w[ids[:, k].long()])
    return out


@pytest.mark.parametrize("name", ["frequency", "explicit_unequal"])
def test_split_recombines_to_full_moe(name):
    torch.manual_seed(0)
    p = PLANS[name].layer(5)
    H, T = 16, 64
    experts_w = torch.randn(E, H, H, dtype=torch.float64)
    x = torch.randn(T, H, dtype=torch.float64)
    ids, w = _random_topk(T, seed=9, w_dtype=torch.float64)
    full = _moe_ref(x, ids, w, experts_w)

    # NPU: only resident rows exist, packed by slot; unused slots zero.
    npu_w = torch.zeros(p.num_slots, H, H, dtype=torch.float64)
    for lid in p.resident_ids:
        npu_w[p.slot_of(lid)] = experts_w[lid]
    nw, nid = split_routing(ids, w, p.safe_slot_table(ids.dtype), p.keep_table(w.dtype))
    npu_out = _moe_ref(x, nid, nw, npu_w)
    # CPU (kt semantics): full ids, skip experts in gpu_experts_mask.
    kt_w = torch.where(p.resident_mask()[ids.long()], torch.zeros_like(w), w)
    cpu_out = _moe_ref(x, ids, kt_w, experts_w)

    assert torch.allclose(npu_out + cpu_out, full, rtol=1e-12, atol=1e-12)
    # mutation arm: packing weights in the wrong slot order must break it
    bad_w = torch.zeros_like(npu_w)
    for slot, lid in enumerate(reversed(p.resident_ids)):
        bad_w[slot] = experts_w[lid]
    assert not torch.allclose(_moe_ref(x, nid, nw, bad_w) + cpu_out, full, rtol=1e-6, atol=1e-6)


def test_resident_bytes_per_slot_dsv41():
    assert resident_bytes_per_slot(5120, 2304) == 18_800_640


# ---------------------------------------------------------------------------
# the (weights, ids) contract at the call sites
# ---------------------------------------------------------------------------


def test_split_routing_returns_a_named_pair_weights_first():
    p = PLANS["frequency"].layer(0)
    ids, w = _random_topk(8)
    out = split_routing(ids, w, p.safe_slot_table(ids.dtype), p.keep_table(w.dtype))
    assert out._fields == ("topk_weights", "topk_ids")
    assert out[0] is out.topk_weights and out[1] is out.topk_ids
    assert out.topk_ids.dtype == ids.dtype and out.topk_weights.dtype == w.dtype
    # the order is the one every Ascend MoE call site names: weights, ids
    npu_weights, npu_ids = out
    assert check_split(p, ids, w, npu_ids, npu_weights) == []


@pytest.mark.parametrize(
    "ids_dtype, w_dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.float32), (torch.int32, torch.int32)],
)
def test_routing_dtype_contract_rejects(ids_dtype, w_dtype):
    """A swapped (weights, ids) pair must fail here, not inside the routing kernel."""
    p = PLANS["prefix"].layer(0)
    ids, w = _random_topk(4)
    with pytest.raises(TypeError) as exc:
        split_routing(ids.to(ids_dtype), w.to(w_dtype), p.safe_slot_table(), p.keep_table())
    assert "kt_offload split_routing" in str(exc.value)


def test_routing_dtype_contract_accepts_the_real_dtypes():
    p = PLANS["prefix"].layer(0)
    for ids_dtype in (torch.int32, torch.int64):
        for w_dtype in (torch.float32, torch.bfloat16):
            ids, w = _random_topk(4, id_dtype=ids_dtype, w_dtype=w_dtype)
            pl.assert_routing_dtypes(ids, w, "test")
            split_routing(ids, w, p.safe_slot_table(ids_dtype), p.keep_table(w_dtype))
    with pytest.raises(ValueError):
        pl.assert_routing_dtypes(torch.zeros(4, 6, dtype=torch.int32), torch.zeros(4, 5), "test")


def test_dtype_contract_mutation_is_red():
    """Mutant: coerce instead of raising -- the swapped call then runs and routes by weight value."""
    p = PLANS["prefix"].layer(0)
    ids, w = _random_topk(4)
    coerced_ids = w.to(torch.int32)  # what a silent .to() would produce from the weights
    out = split_routing(coerced_ids, ids.to(torch.float32), p.safe_slot_table(), p.keep_table())
    assert check_split(p, ids, w, out.topk_ids, out.topk_weights) != []
