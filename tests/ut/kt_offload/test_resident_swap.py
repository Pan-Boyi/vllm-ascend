# SPDX-License-Identifier: Apache-2.0
"""Atomic resident-expert swap -- CPU only, no NPU, no kt-kernel.

Oracle: ``check_resident_state``, a *set* criterion judged against kt's mask
(the ``placement.check_split`` choice), plus a weight-row identity so "route
to e lands on e's weights" is checked, not just table shapes. Every mutation
arm below breaks the commit in one way and must turn the oracle red; a green
oracle is only evidence if its arms are red.

Fixture weights encode their logical id in every element (row of expert e in
tensor ``name`` is filled with ``OFFSET[name] + e``), so a row can be
identified after any reordering and two tensors can never be confused.
"""

from __future__ import annotations

import threading

import pytest
import torch

from vllm_ascend.kt_offload import resident_swap as rs
from vllm_ascend.kt_offload.placement import LayerPlacement, check_split, split_routing
from vllm_ascend.kt_offload.resident_swap import (
    ALL_CPU_WEIGHTS,
    STAGES,
    LayerSwapTargets,
    OwnershipConflictError,
    OwnershipLedger,
    ResidentSwapCommitter,
    SwapCorruptedError,
    SwapError,
    ZeroCoverageError,
    check_resident_state,
    kt_mask_of,
)

E, N, SLOTS, TOPK = 384, 100, 100, 6  # DSV4.1 layer; N resident, one num_slots
OFFSET = {"w13_weight": 0, "w2_weight": 10_000, "w13_scale": 20_000, "w2_scale": 30_000}
ROW = {"w13_weight": (4, 3), "w2_weight": (5,), "w13_scale": (2,), "w2_scale": (3,)}


def _weights_for(p: LayerPlacement) -> dict[str, torch.Tensor]:
    out = {}
    for name, shape in ROW.items():
        w = torch.full((p.num_slots, *shape), -1.0, dtype=torch.float32)
        for slot, e in enumerate(p.resident_ids):
            w[slot] = OFFSET[name] + e
        out[name] = w
    return out


def identify_row(name: str, row: torch.Tensor) -> int:
    v = row.reshape(-1)
    if not bool((v == v[0]).all()):
        return -2  # torn row: no single expert
    return int(v[0]) - OFFSET[name]


def _targets(p: LayerPlacement, kt_ids=ALL_CPU_WEIGHTS) -> LayerSwapTargets:
    return LayerSwapTargets(
        layer_idx=p.layer_idx,
        weights={k: torch.nn.Parameter(v, requires_grad=False) for k, v in _weights_for(p).items()},
        slot_tables={dt: p.safe_slot_table(dt) for dt in (torch.int32, torch.int64)},
        keep_tables={dt: p.keep_table(dt) for dt in (torch.float32, torch.bfloat16)},
        device_mask=p.resident_mask(),
        kt_mask=p.resident_mask(),
        kt_cpu_weight_ids=kt_ids,
    )


def _old(layer=3, seed=0) -> LayerPlacement:
    g = torch.Generator().manual_seed(seed)
    ids = sorted(torch.randperm(E, generator=g)[:N].tolist())
    return LayerPlacement(layer_idx=layer, num_experts=E, num_slots=SLOTS, resident_ids=tuple(ids))


def _swap(p: LayerPlacement, k=7, seed=1) -> list[int]:
    """Swap k residents for k CPU experts; the new set reorders many slots."""
    g = torch.Generator().manual_seed(seed)
    res, cpu = list(p.resident_ids), p.cpu_ids
    out_ = [res[i] for i in torch.randperm(len(res), generator=g)[:k].tolist()]
    in_ = [cpu[i] for i in torch.randperm(len(cpu), generator=g)[:k].tolist()]
    return sorted(set(res) - set(out_) | set(in_))


def _new_placement(p: LayerPlacement, ids) -> LayerPlacement:
    return LayerPlacement(layer_idx=p.layer_idx, num_experts=E, num_slots=SLOTS, resident_ids=tuple(sorted(ids)))


def _setup(**kw):
    p = _old()
    tg = _targets(p)
    c = ResidentSwapCommitter({p.layer_idx: p}, {p.layer_idx: tg}, **kw)
    new_ids = _swap(p)
    return p, tg, c, new_ids, _weights_for(_new_placement(p, new_ids))


def _state(tg: LayerSwapTargets) -> dict:
    """Observable state: captured content, live content, addresses."""
    return {
        key: (tg.captured(key).clone(), t.detach().clone(), t.data_ptr(), tg.captured_addr(key))
        for key, t in tg.items()
    }


def _assert_state_equal(a: dict, b: dict):
    assert a.keys() == b.keys()
    for key in a:
        ca, la, pa, xa = a[key]
        cb, lb, pb, xb = b[key]
        assert torch.equal(ca, cb), f"{key}: captured content changed"
        assert torch.equal(la, lb), f"{key}: live content changed"
        assert pa == pb == xa == xb, f"{key}: address changed"


def _red(p_new, tg, ledger=None) -> list[str]:
    return check_resident_state(p_new, tg, identify_row=identify_row, ledger=ledger)


# positive: the commit, judged by the set oracle
def test_fixture_is_discriminating():
    p, tg, c, new_ids, _ = _setup()
    new = _new_placement(p, new_ids)
    assert check_resident_state(p, tg, identify_row=identify_row, ledger=c.ledger) == []
    # the requested set is really different and really reorders slots
    assert set(new_ids) != set(p.resident_ids)
    assert sum(p.slot_of(e) != new.slot_of(e) for e in set(new_ids) & set(p.resident_ids)) > 0
    # the unchanged state judged against the new placement is red
    assert _red(new, tg)


def test_commit_makes_all_five_objects_agree():
    p, tg, c, new_ids, nw = _setup()
    before = _state(tg)
    rep = c.commit(step=10, layer=p.layer_idx, new_resident_ids=new_ids, new_weights=nw)
    new = c.placement(p.layer_idx)
    assert new.resident_ids == tuple(new_ids)
    assert check_resident_state(new, tg, identify_row=identify_row, ledger=c.ledger) == []
    # set criterion from kt's own mask equals the request
    assert set(torch.nonzero(tg.kt_mask).flatten().tolist()) == set(new_ids)
    # nothing was rebound: every address is the captured one
    for key, (_, _, addr, _) in before.items():
        assert tg.live(key).data_ptr() == addr
    # coverage: exactly the symmetric difference moved
    assert rep.ok and rep.moved == 14 == len(set(new_ids) ^ set(p.resident_ids))
    assert set(rep.promoted) == set(new_ids) - set(p.resident_ids)
    assert set(rep.demoted) == set(p.resident_ids) - set(new_ids)
    assert rep.reslotted and rep.stages_done == STAGES
    assert rep.bytes_written == sum(t.numel() * t.element_size() for _, t in tg.items())


def test_commit_order_is_weights_slot_keep_mask_kt_last():
    seen = []
    p, tg, c, new_ids, nw = _setup(on_write=lambda stage, key: seen.append(stage))
    c.commit(1, p.layer_idx, new_ids, nw)
    first = [s for i, s in enumerate(seen) if i == 0 or seen[i - 1] != s]
    assert first == list(STAGES)
    assert seen[-1] == "kt_mask"


def test_routing_after_commit_matches_kt_partition():
    """End-to-end: split_routing through the committed tables vs kt's mask."""
    p, tg, c, new_ids, nw = _setup()
    c.commit(1, p.layer_idx, new_ids, nw)
    new = c.placement(p.layer_idx)
    g = torch.Generator().manual_seed(5)
    ids = torch.argsort(torch.rand(64, E, generator=g), dim=1)[:, :TOPK].to(torch.int32)
    w = torch.rand(64, TOPK, generator=g) + 0.05
    npu = split_routing(ids, w, tg.captured(f"slot_table:{torch.int32}"), tg.captured(f"keep_table:{torch.float32}"))
    assert check_split(new, ids, w, npu.topk_ids, npu.topk_weights) == []
    # and every NPU-effective route hits its own expert's weights
    rows = tg.captured("weights:w2_weight")[npu.topk_ids.long()]
    eff = npu.topk_weights != 0
    assert eff.any()
    got = rows[..., 0][eff] - OFFSET["w2_weight"]
    assert torch.equal(got.long(), ids.long()[eff])


def test_back_to_back_commits_over_steps():
    p, tg, c, new_ids, nw = _setup()
    c.commit(1, p.layer_idx, new_ids, nw)
    back = c.commit(2, p.layer_idx, p.resident_ids, _weights_for(p))
    assert back.ok and back.moved == 14
    assert check_resident_state(p, tg, identify_row=identify_row, ledger=c.ledger) == []


# mutation arms: each MUST be red
ONLY = {
    "only_weights": ("weights",),
    "only_device_mask": ("device_mask",),
    "only_slot_table": ("slot_table",),
    "only_kt_mask": ("kt_mask",),
    "only_keep_table": ("keep_table",),
    "all_but_kt_mask": ("weights", "slot_table", "keep_table", "device_mask"),
    "all_but_weights": ("slot_table", "keep_table", "device_mask", "kt_mask"),
}


@pytest.mark.parametrize("arm", sorted(ONLY))
def test_mutation_partial_write_is_red(arm, monkeypatch):
    real = ResidentSwapCommitter._plan_writes
    keep = ONLY[arm]
    monkeypatch.setattr(
        ResidentSwapCommitter, "_plan_writes", lambda self, *a: [w for w in real(self, *a) if w.stage in keep]
    )
    p, tg, c, new_ids, nw = _setup()
    # verify=False: the external oracle alone must catch it
    c.commit(1, p.layer_idx, new_ids, nw, verify=False)
    new = _new_placement(p, new_ids)
    errs = _red(new, tg, ledger=c.ledger)
    assert errs, f"{arm}: oracle stayed green"
    # and the committer's own post-check refuses it: the plan patch is a
    # PERSISTENT defect, so the roll-forward cannot fix it and the layer must
    # end up poisoned rather than quietly inconsistent
    p2, tg2, c2, ids2, nw2 = _setup()
    with pytest.raises(rs.SwapCorruptedError):
        c2.commit(1, p2.layer_idx, ids2, nw2)
    assert p2.layer_idx in c2._poisoned
    with pytest.raises(rs.SwapCorruptedError):        # and it stays refused afterwards
        c2.commit(2, p2.layer_idx, ids2, nw2)


@pytest.mark.parametrize("which", ["weights", "table"])
def test_mutation_rebind_instead_of_in_place_is_red(which, monkeypatch):
    def rebind(dst, src):
        dst.data = src.detach().clone()

    kw = {}
    if which == "weights":
        kw["weight_writer"] = rebind
    else:
        monkeypatch.setattr(rs, "_table_write", rebind)
    p, tg, c, new_ids, nw = _setup(**kw)
    c.commit(1, p.layer_idx, new_ids, nw, verify=False)
    new = _new_placement(p, new_ids)
    # the live objects hold new content -- a check that read them would pass
    errs = _red(new, tg)
    assert errs and any("rebound" in e for e in errs), errs
    assert tg.rebound()
    # with verify on, a rebinding writer is a code defect, not a transient
    # fault: the roll-forward re-runs the same writer, so the layer is poisoned
    p3, tg3, c3, ids3, nw3 = _setup(**kw)
    with pytest.raises((SwapError, rs.SwapCorruptedError), match="rebound"):
        c3.commit(1, p3.layer_idx, ids3, nw3)
    assert p3.layer_idx in c3._poisoned


def test_mutation_oracle_reading_live_objects_would_miss_rebind(monkeypatch):
    """Why the oracle reads captured aliases: the live object already holds the new bytes."""
    monkeypatch.setattr(rs, "_table_write", lambda dst, src: setattr(dst, "data", src.detach().clone()))
    p, tg, c, new_ids, nw = _setup()
    c.commit(1, p.layer_idx, new_ids, nw, verify=False)
    assert set(torch.nonzero(tg.kt_mask).flatten().tolist()) == set(new_ids)  # live: looks done
    assert set(torch.nonzero(tg.captured("kt_mask")).flatten().tolist()) == set(p.resident_ids)  # C++: old


def test_mutation_swapped_weight_rows_is_red():
    """Tables right, weights in the old order: only the row identity can see it."""
    p, tg, c, new_ids, _ = _setup()
    c.commit(1, p.layer_idx, new_ids, _weights_for(_new_placement(p, new_ids)))
    new = c.placement(p.layer_idx)
    w = tg.captured("weights:w13_weight")
    w[[0, 1]] = w[[1, 0]]
    assert _red(new, tg)
    assert check_resident_state(new, tg) == []  # without row identity: blind to it


# roll-forward: a failure at any write is FINISHED, not undone. Undoing would
# need a copy of the old weights, which ``clone()`` refuses on an internal-format
# tensor; and the only problem a half-written commit creates is INCONSISTENCY
# (new weights, old tables), which computes wrong numbers in silence. So a
# transient fault ends on the REQUESTED set with ``recovered``, a persistent one
# poisons the layer and raises, and no layer is ever left inconsistent.


def _all_write_keys():
    p = _old()
    return [key for key, _ in _targets(p).items()]


@pytest.mark.parametrize("fail_key", _all_write_keys())
def test_roll_forward_after_each_write(fail_key):
    """A transient fault at any single write still lands on the requested set."""

    def boom(stage, key):
        if key == fail_key:
            raise RuntimeError(f"injected at {key}")

    p, tg, c, new_ids, nw = _setup(on_write=boom)
    rep = c.commit(1, p.layer_idx, new_ids, nw)
    assert rep.ok and rep.recovered and "injected" in rep.error
    # the committed state is the FULL commit, not a half one and not the old one
    new_p = c.placement(p.layer_idx)
    assert tuple(new_p.resident_ids) == tuple(sorted(new_ids))
    assert check_resident_state(new_p, tg, identify_row=identify_row, ledger=c.ledger) == []
    assert c.ledger.active is None


@pytest.mark.parametrize("fail_key", _all_write_keys())
def test_a_recovered_commit_reports_its_coverage(fail_key):
    """A commit that only completed by roll-forward must not report zero coverage,
    or a caller applying the coverage rule reads a successful recovery as
    "nothing happened". The expectations come from the targets themselves, not
    from anything the committer reports.
    """

    def boom(stage, key):
        if key == fail_key:
            raise RuntimeError(f"injected at {key}")

    p, tg, c, new_ids, nw = _setup(on_write=boom)
    want_bytes = sum(t.numel() * t.element_size() for _, t in tg.items())

    rep = c.commit(1, p.layer_idx, new_ids, nw)

    assert rep.ok and rep.recovered
    assert rep.stages_done == STAGES
    # The roll-forward rewrites every target, on top of whatever the failed
    # attempt had already moved -- so at least one full pass of bytes.
    assert rep.bytes_written >= want_bytes


def test_a_committed_swap_is_never_silently_the_old_set():
    """The arm that would catch a roll-forward that quietly gave up."""
    p, tg, c, new_ids, nw = _setup()
    rep = c.commit(1, p.layer_idx, new_ids, nw)
    assert rep.ok and not rep.recovered
    assert tuple(c.placement(p.layer_idx).resident_ids) != tuple(p.resident_ids)


@pytest.mark.parametrize("stage", STAGES)
def test_roll_forward_when_the_write_itself_throws(stage, monkeypatch):
    """Failure *inside* a writer (possibly after a partial write) at each stage."""

    def half_then_raise(dst, src):
        flat = dst.view(-1)
        flat[: flat.numel() // 2].copy_(src.reshape(-1)[: flat.numel() // 2])
        raise RuntimeError(f"injected in {stage}")

    kw = {}
    if stage == "weights":
        # only the first call fails -- a transient fault; a channel that stays
        # broken is test_failed_roll_forward_poisons_the_layer
        calls = {"n": 0}

        def first_call_fails(dst, src):
            calls["n"] += 1
            if calls["n"] == 1:
                half_then_raise(dst, src)
            dst.copy_(src)

        kw["weight_writer"] = first_call_fails
    else:
        real_plan = ResidentSwapCommitter._plan_writes
        calls = {"n": 0}

        def first_call_fails(dst, src):
            # transient by construction, like the weights arm
            calls["n"] += 1
            if calls["n"] == 1:
                half_then_raise(dst, src)
            else:
                rs._table_write(dst, src)

        def plan(self, *a):
            return [w._replace(writer=first_call_fails) if w.stage == stage else w for w in real_plan(self, *a)]

        monkeypatch.setattr(ResidentSwapCommitter, "_plan_writes", plan)
    p, tg, c, new_ids, nw = _setup(**kw)
    rep = c.commit(1, p.layer_idx, new_ids, nw)
    assert rep.ok and rep.recovered
    new_p = c.placement(p.layer_idx)
    assert check_resident_state(new_p, tg, identify_row=identify_row, ledger=c.ledger) == []


def test_non_strict_also_rolls_forward():
    """strict only governs whether a REFUSAL raises; recovery is not optional."""
    p, tg, c, new_ids, nw = _setup(strict=False,
                                   on_write=lambda s, k: (_ for _ in ()).throw(RuntimeError("x")))
    rep = c.commit(1, p.layer_idx, new_ids, nw)
    assert rep.ok and rep.recovered and rep.moved == 14
    assert check_resident_state(c.placement(p.layer_idx), tg, identify_row=identify_row,
                                ledger=c.ledger) == []


def test_failed_roll_forward_poisons_the_layer():
    calls = {"n": 0}

    def writer(dst, src):
        calls["n"] += 1
        if calls["n"] >= 2:  # the forward write of the 2nd weight and every restore fail
            raise RuntimeError("device lost")
        dst.copy_(src)

    p, tg, c, new_ids, nw = _setup(weight_writer=writer)
    with pytest.raises(SwapCorruptedError):
        c.commit(1, p.layer_idx, new_ids, nw)
    with pytest.raises(SwapCorruptedError, match="poisoned"):
        c.commit(2, p.layer_idx, new_ids, nw)


# coverage accounting: zero coverage fails, strict raises
def test_zero_coverage_raises_in_strict():
    p, tg, c, _, _ = _setup()
    before = _state(tg)
    with pytest.raises(ZeroCoverageError):
        c.commit(1, p.layer_idx, p.resident_ids, _weights_for(p))
    _assert_state_equal(before, _state(tg))


def test_zero_coverage_is_not_ok_in_non_strict():
    p, tg, c, _, _ = _setup(strict=False)
    rep = c.commit(1, p.layer_idx, p.resident_ids, _weights_for(p))
    assert not rep.ok and rep.moved == 0 and "zero coverage" in rep.error
    step = c.commit_step(2, {p.layer_idx: (p.resident_ids, _weights_for(p))})
    assert not step.ok and step.moved == 0


def test_step_coverage_sums_layers_and_zero_step_raises():
    ps = [_old(layer=i, seed=i) for i in range(3)]
    tgs = {q.layer_idx: _targets(q) for q in ps}
    c = ResidentSwapCommitter({q.layer_idx: q for q in ps}, tgs)
    changes = {}
    for q in ps:
        ids = _swap(q, k=2 + q.layer_idx, seed=10 + q.layer_idx)
        changes[q.layer_idx] = (ids, _weights_for(_new_placement(q, ids)))
    rep = c.commit_step(5, changes)
    assert rep.ok and rep.moved == 2 * (2 + 3 + 4)
    for q in ps:
        assert check_resident_state(c.placement(q.layer_idx), tgs[q.layer_idx], identify_row, c.ledger) == []
    with pytest.raises(ZeroCoverageError):
        c.commit_step(6, {})


def test_refusals_happen_before_any_write():
    p, tg, c, new_ids, nw = _setup()
    before = _state(tg)
    bad = [
        (new_ids[:-1], nw, "resident count"),  # count changes
        (new_ids, {k: v for k, v in nw.items() if k != "w2_scale"}, "new weights"),
        (new_ids, {**nw, "w2_weight": nw["w2_weight"][:, :4].contiguous()}, "w2_weight"),
        (new_ids, {**nw, "w13_weight": tg.weights["w13_weight"]}, "aliases"),
    ]
    for ids, w, msg in bad:
        with pytest.raises(SwapError, match=msg):
            c.commit(1, p.layer_idx, ids, w)
        _assert_state_equal(before, _state(tg))


def test_demoting_an_expert_kt_has_no_cpu_weights_for_is_refused():
    p = _old()
    tg = _targets(p, kt_ids=p.cpu_ids)  # kt loaded the CPU subset only
    c = ResidentSwapCommitter({p.layer_idx: p}, {p.layer_idx: tg})
    new_ids = _swap(p)
    with pytest.raises(SwapError, match="no CPU weights"):
        c.commit(1, p.layer_idx, new_ids, _weights_for(_new_placement(p, new_ids)))
    assert check_resident_state(p, tg, identify_row=identify_row) == []


def test_weight_target_must_be_whole_storage():
    """An offset view must be refused: the copy writes the whole storage.

    Contiguity is deliberately NOT part of the criterion -- the production weight
    is transposed and in the internal NZ format, so it reports non-contiguous
    while owning exactly the storage the GEMM reads. Ownership (offset 0,
    view == storage) is what copy_memory_ needs.
    """
    big = torch.zeros(8, 4)
    with pytest.raises(ValueError, match="own its whole storage"):
        rs.default_weight_write(big[1:2], torch.ones(1, 4))
    assert bool((big == 0).all())


def test_a_transposed_full_storage_target_is_accepted():
    """The shape the card hands us: non-contiguous, whole storage -- modelled
    here rather than assumed, because contiguous stand-ins hide the guard.
    """
    t = torch.zeros(4, 6).t()                    # non-contiguous, still the whole storage
    assert not t.is_contiguous()
    # the criterion that matters holds: offset 0 and the view covers the storage
    assert t.storage_offset() == 0
    assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
    # and on a HOST tensor the separate host guard is what fires, not the
    # ownership one -- host tensors have no internal format, so there a
    # non-contiguous target really is a view worth refusing
    with pytest.raises(ValueError, match="host target must be contiguous"):
        rs.default_weight_write(t, torch.ones_like(t))


def test_kt_mask_of_takes_wrappers_own_tensor_and_refuses_stale_device_copy():
    class W:
        gpu_experts_mask = torch.zeros(E, dtype=torch.bool)
        _gpu_experts_mask_gpu = None

    w = W()
    assert kt_mask_of(w) is w.gpu_experts_mask
    w._gpu_experts_mask_gpu = torch.zeros(E, dtype=torch.bool)
    with pytest.raises(ValueError, match="stale"):
        kt_mask_of(w)


# ownership / concurrency
def test_second_thread_during_commit_raises():
    entered, release = threading.Event(), threading.Event()

    def slow_writer(dst, src):
        entered.set()
        assert release.wait(10)
        dst.copy_(src)

    p, tg, c, new_ids, nw = _setup(weight_writer=slow_writer)
    result = {}

    def first():
        result["rep"] = c.commit(1, p.layer_idx, new_ids, nw)

    t = threading.Thread(target=first)
    t.start()
    assert entered.wait(10)
    err = {}

    def second():
        try:
            c.commit(1, p.layer_idx, p.resident_ids, _weights_for(p))
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    t2 = threading.Thread(target=second)
    t2.start()
    t2.join(10)
    release.set()
    t.join(10)
    assert isinstance(err.get("e"), OwnershipConflictError), err
    assert "overlaps" in str(err["e"])
    assert result["rep"].ok  # the first writer is unaffected
    assert check_resident_state(c.placement(p.layer_idx), tg, identify_row, c.ledger) == []


def test_second_thread_after_commit_raises_and_writes_nothing():
    p, tg, c, new_ids, nw = _setup(strict=False)  # conflicts raise even when not strict
    c.commit(1, p.layer_idx, new_ids, nw)
    before = _state(tg)
    err = {}

    def other():
        try:
            c.commit(2, p.layer_idx, p.resident_ids, _weights_for(p))
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    t = threading.Thread(target=other)
    t.start()
    t.join(10)
    assert isinstance(err.get("e"), OwnershipConflictError) and "second writer" in str(err["e"])
    _assert_state_equal(before, _state(tg))


def test_second_stream_same_thread_raises():
    stream = {"id": "s0"}
    p, tg, c, new_ids, nw = _setup(stream_id=lambda: stream["id"])
    c.commit(1, p.layer_idx, new_ids, nw)
    stream["id"] = "side"
    with pytest.raises(OwnershipConflictError, match="second writer"):
        c.commit(2, p.layer_idx, p.resident_ids, _weights_for(p))


def test_stale_step_and_a_busy_writer_raise():
    """``writer_busy`` is whatever the CALLER's commit point is."""
    flight = {"on": False}
    p, tg, c, new_ids, nw = _setup(writer_busy=lambda: flight["on"])
    c.commit(5, p.layer_idx, new_ids, nw)
    with pytest.raises(OwnershipConflictError, match="stale step"):
        c.commit(4, p.layer_idx, p.resident_ids, _weights_for(p))
    flight["on"] = True
    with pytest.raises(OwnershipConflictError, match="writer is busy"):
        c.commit(6, p.layer_idx, p.resident_ids, _weights_for(p))


def test_ledger_flags_double_and_missing_ownership():
    """The two claim tables are written from different objects; a split shows."""
    p = _old()
    tg = _targets(p)
    led = OwnershipLedger()
    led.record_npu(p.layer_idx, tg)
    led.record_cpu(p.layer_idx, tg)
    assert led.violations(p.layer_idx) == []
    tg.kt_mask[p.resident_ids[0]] = False  # kt now also computes a resident expert
    tg.kt_mask[p.cpu_ids[0]] = True  # and nobody computes this one
    led.record_cpu(p.layer_idx, tg)
    errs = led.violations(p.layer_idx)
    assert any("NPU and CPU" in e for e in errs) and any("neither" in e for e in errs)
    assert led.violations(p.layer_idx + 1)  # unrecorded layer is not a pass


def test_torch_npu_is_imported_only_inside_the_device_write_channel():
    """Module load must work with no NPU; the one device import is in the write channel."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(rs))
    where = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Module))]:
        body = (
            fn.body if isinstance(fn, ast.FunctionDef) else [n for n in fn.body if not isinstance(n, ast.FunctionDef)]
        )
        for node in body:
            for sub in ast.walk(node) if isinstance(fn, ast.FunctionDef) else [node]:
                if isinstance(sub, ast.Import) and any(a.name.startswith("torch_npu") for a in sub.names):
                    where.append(getattr(fn, "name", "<module>"))
    assert where == ["default_weight_write"], where  # positive control: the one we know exists is found
