# SPDX-License-Identifier: Apache-2.0
"""kt_cpu_weight_ids_of: which experts kt holds CPU weights for.

The expectations are written from the kt contract (a full-set load covers every
expert; a subset load covers the non-resident ones), never from the function
under test.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload.placement import LayerPlacement
from vllm_ascend.kt_offload.resident_swap import (
    ALL_CPU_WEIGHTS,
    ResidentSwapCommitter,
    kt_cpu_weight_ids_of,
)

E = 8
RESIDENT = {1, 4, 6}
CPU = sorted(set(range(E)) - RESIDENT)


class _OldKt:
    """kt without the allocation query: only the live mask."""

    num_experts = E

    def __init__(self, resident):
        self.gpu_experts_mask = torch.zeros(E, dtype=torch.bool)
        self.gpu_experts_mask[list(resident)] = True

    def cpu_expert_ids(self):
        return [e for e in range(E) if not bool(self.gpu_experts_mask[e])]


class _NewKt(_OldKt):
    def __init__(self, resident, loaded):
        super().__init__(resident)
        self._loaded = sorted(loaded)

    def cpu_weight_expert_ids(self):
        return list(self._loaded)


def test_full_set_load_reports_all():
    assert kt_cpu_weight_ids_of(_NewKt(RESIDENT, range(E))) == ALL_CPU_WEIGHTS


def test_subset_load_reports_exactly_the_cpu_experts():
    assert kt_cpu_weight_ids_of(_NewKt(RESIDENT, CPU)) == frozenset(CPU)


def test_new_kt_answer_does_not_follow_a_rewritten_mask():
    w = _NewKt(RESIDENT, CPU)
    w.gpu_experts_mask[:] = False  # a swap demoted everything
    assert kt_cpu_weight_ids_of(w) == frozenset(CPU)


def test_old_kt_falls_back_to_the_cpu_set_and_never_claims_all():
    got = kt_cpu_weight_ids_of(_OldKt(RESIDENT))
    assert got == frozenset(CPU)
    assert got != ALL_CPU_WEIGHTS


def test_out_of_range_report_raises():
    with pytest.raises(ValueError, match="outside"):
        kt_cpu_weight_ids_of(_NewKt(RESIDENT, [*CPU, E]))


# live_target_violations: the commit must write what the forward reads
class _Layer:
    """The objects a layer routes and computes through, as layer.py holds them."""

    def __init__(self, num_experts=E, num_slots=4):
        self.weights = {"w13_weight": torch.zeros(num_slots, 4, dtype=torch.uint8)}
        self.slot_tables = {torch.int32: torch.zeros(num_experts, dtype=torch.int32)}
        self.keep_tables = {torch.float32: torch.zeros(num_experts, dtype=torch.float32)}
        self.kt_mask = torch.zeros(num_experts, dtype=torch.bool)

    def live(self):
        return {"weights": self.weights, "slot_tables": self.slot_tables,
                "keep_tables": self.keep_tables, "kt_mask": self.kt_mask}


def _targets_from(layer, live):
    from vllm_ascend.kt_offload.resident_swap import ALL_CPU_WEIGHTS, LayerSwapTargets

    return LayerSwapTargets(
        layer_idx=0, weights=dict(live["weights"]), slot_tables=dict(live["slot_tables"]),
        keep_tables=dict(live["keep_tables"]), device_mask=torch.zeros(E, dtype=torch.bool),
        kt_mask=live["kt_mask"], kt_cpu_weight_ids=ALL_CPU_WEIGHTS)


def test_targets_built_from_the_live_objects_pass():
    from vllm_ascend.kt_offload.resident_swap import live_target_violations

    lay = _Layer()
    assert live_target_violations(_targets_from(lay, lay.live()), **lay.live()) == []


def test_a_freshly_built_slot_table_is_caught():
    """A table rebuilt from the placement: same values, same dtype, different
    storage -- the commit writes it and the forward never reads it.
    """
    from vllm_ascend.kt_offload.resident_swap import live_target_violations

    lay = _Layer()
    detached = dict(lay.live())
    detached["slot_tables"] = {torch.int32: lay.slot_tables[torch.int32].clone()}
    bad = live_target_violations(_targets_from(lay, detached), **lay.live())
    assert len(bad) == 1 and "slot_table" in bad[0] and "leave the routing behind" in bad[0]


def test_a_rebaked_table_detaches_targets_that_were_right_when_built():
    """The reason this is re-checked per commit, not once at construction."""
    from vllm_ascend.kt_offload.resident_swap import live_target_violations

    lay = _Layer()
    tg = _targets_from(lay, lay.live())
    assert live_target_violations(tg, **lay.live()) == []
    lay.keep_tables = {torch.float32: torch.zeros(E, dtype=torch.float32)}   # after_weights_loaded again
    assert any("keep_table" in b for b in live_target_violations(tg, **lay.live()))


def test_a_weight_the_layer_has_and_the_commit_would_not_write_is_caught():
    from vllm_ascend.kt_offload.resident_swap import live_target_violations

    lay = _Layer()
    tg = _targets_from(lay, lay.live())
    lay.weights["w2_weight"] = torch.zeros(4, 4, dtype=torch.uint8)
    bad = live_target_violations(tg, **lay.live())
    assert any("w2_weight" in b and "would not write it" in b for b in bad)


def test_the_committer_refuses_a_detached_target_before_writing():
    """End to end: the guard fires in commit(), before the first byte moves."""
    import pytest as _pytest

    from vllm_ascend.kt_offload.resident_swap import SwapError

    lay = _Layer()
    detached = dict(lay.live())
    detached["kt_mask"] = lay.kt_mask.clone()
    tg = _targets_from(lay, detached)
    p = LayerPlacement(layer_idx=0, num_experts=E, num_slots=4, resident_ids=(0, 1, 2, 3))
    c = ResidentSwapCommitter({0: p}, {0: tg}, strict=True, live_objects=lambda _l: lay.live())
    with _pytest.raises(SwapError, match="not what the forward reads"):
        c.commit(1, 0, (0, 1, 2, 5), {"w13_weight": torch.zeros(4, 4, dtype=torch.uint8)})


def test_the_check_refuses_to_answer_on_meta_tensors():
    """Every ``meta`` tensor reports ``data_ptr() == 0``, so two unrelated ones
    compare equal: a criterion that cannot fail must refuse, not report success.
    """
    import pytest as _pytest

    from vllm_ascend.kt_offload.resident_swap import live_target_violations

    lay = _Layer()
    meta = dict(lay.live())
    meta["slot_tables"] = {torch.int32: torch.zeros(E, dtype=torch.int32, device="meta")}
    tg = _targets_from(lay, {**lay.live(),
                             "slot_tables": {torch.int32: torch.zeros(E, dtype=torch.int32, device="meta")}})
    with _pytest.raises(ValueError, match="cannot judge"):
        live_target_violations(tg, **meta)


# default_post_load must not run kt's loader patch on a chunk / staging module
def test_the_chunk_cast_takes_the_unpatched_hook():
    """kt wraps the layer's post-load hook; the chunk formatter wants only the
    cast. Running the wrapper on a throwaway module would rebind the live layer's
    routing tables. The expectation is what each part is supposed to do -- the
    cast runs, kt's two additions do not -- not what the function happens to call.
    """
    from vllm_ascend.kt_offload.stream_io import default_post_load

    ran = []

    def real_cast(mod):
        ran.append("cast")

    def kt_wrapper(mod):
        ran.append("kt")
        real_cast(mod)
        ran.append("kt-after")

    kt_wrapper._kt_offload_patched = True
    kt_wrapper._kt_offload_original = real_cast

    qm = SimpleNamespace(process_weights_after_loading=kt_wrapper)
    default_post_load(qm, torch.nn.Module())
    assert ran == ["cast"], f"kt's loader additions must not run on a chunk module, got {ran}"


def test_an_unpatched_quant_method_is_used_as_is():
    """The positive control: without the stamp, the method itself is the hook."""
    from vllm_ascend.kt_offload.stream_io import default_post_load

    ran = []
    qm = SimpleNamespace(process_weights_after_loading=lambda mod: ran.append("cast"))
    default_post_load(qm, torch.nn.Module())
    assert ran == ["cast"]
