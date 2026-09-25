# SPDX-License-Identifier: Apache-2.0
"""The swap driver -- CPU only, no NPU, no kt-kernel, no card.

The driver decides *when* the resident set changes and *which* experts move,
then drives ``resident_swap``'s committer; everything it touches is injected.
Fixture weights carry both coordinates (``OFFSET[name] + 1000*L + e``): a slot
left over from the previous layer has the right shape, the right dtype and
plausible numbers, so only a value that names its layer can tell it apart.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import types

import pytest
import torch

from vllm_ascend.kt_offload import resident_driver as rd
from vllm_ascend.kt_offload.placement import LayerPlacement
from vllm_ascend.kt_offload.resident_driver import (
    SLOT_WEIGHT_ATTRS,
    LayerPlan,
    PolicyError,
    TopKHeatPolicy,
    capture_state,
    check_proposal,
    pinned_resident_ids,
    plan_from_heat,
)
from vllm_ascend.kt_offload.resident_swap import (
    ALL_CPU_WEIGHTS,
    OwnershipConflictError,
    ResidentSwapCommitter,
    SwapCorruptedError,
    SwapError,
    ZeroCoverageError,
    check_resident_state,
)

E, K = 32, 8  # experts per layer, resident slots (== num_slots: no unused tail)
LAYERS = (0, 1, 2)
START = tuple(range(K))  # resident at start-up: the prefix strategy
OFFSET = {"w13_weight": 0, "w13_weight_scale": 100_000, "w2_weight": 200_000, "w2_weight_scale": 300_000}
ROW = {"w13_weight": (3, 2), "w13_weight_scale": (2,), "w2_weight": (4,), "w2_weight_scale": (2,)}
LAYER_STRIDE = 1000  # > E, < the gap between two OFFSETs

#: The experts the heat table makes hottest. 0..3 are already resident, 20..23
#: are not, so every layer promotes 4 and demotes 4 -- 8 moved pairs per layer.
HOT = (0, 1, 2, 3, 20, 21, 22, 23)


# Fixtures: a checkpoint, a slot provider, commit targets


def _placement(layer: int, ids=START) -> LayerPlacement:
    return LayerPlacement(layer_idx=layer, num_experts=E, num_slots=K, resident_ids=tuple(sorted(ids)))


def _heat(layer: int, hot=HOT) -> torch.Tensor:
    """Counts for one layer: the hot experts get a lot, everyone else a little.
    Nobody is at zero -- an all-zero row means "never counted" to the driver.
    """
    t = torch.ones(E, dtype=torch.int64)
    for rank, e in enumerate(hot):
        t[e] = 1000 - rank
    return t


def test_slot_attribute_names_match_production():
    from vllm_ascend.kt_offload.stream_chain import SLOT_WEIGHT_ATTRS as PRODUCTION

    assert tuple(SLOT_WEIGHT_ATTRS) == tuple(PRODUCTION)


def test_module_imports_standalone():
    """Import it FIRST in a fresh process, the way an on-card script does."""
    r = subprocess.run([sys.executable, "-c", "import vllm_ascend.kt_offload.resident_driver"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"resident_driver cannot be imported on its own:\n{r.stderr[-2000:]}"


def test_no_torch_npu_at_module_level():
    src = pathlib.Path(rd.__file__).read_text()
    assert "import torch_npu" not in src, "torch_npu must not be imported by this module at all"


# the policy


def test_topk_policy_picks_the_hottest_and_breaks_ties_by_ascending_id():
    p = _placement(0)
    heat = torch.zeros(E, dtype=torch.float64)
    heat[:] = 5  # a complete tie
    chosen = TopKHeatPolicy()(layer=0, placement=p, heat=heat, pinned=frozenset())
    assert chosen == tuple(range(K))
    heat[29] = 9
    chosen = TopKHeatPolicy()(layer=0, placement=p, heat=heat, pinned=frozenset())
    assert chosen == tuple(sorted(list(range(K - 1)) + [29]))


def test_topk_policy_keeps_pinned_experts_and_fills_the_rest():
    p = _placement(0)
    pinned = frozenset({5, 6, 7})
    chosen = TopKHeatPolicy()(layer=0, placement=p, heat=_heat(0).double(), pinned=pinned)
    assert pinned <= set(chosen)
    assert len(chosen) == K
    assert set(chosen) == pinned | {0, 1, 2, 3, 20}


def test_topk_policy_refuses_more_pinned_experts_than_slots():
    p = _placement(0)
    with pytest.raises(PolicyError):
        TopKHeatPolicy()(layer=0, placement=p, heat=_heat(0).double(), pinned=frozenset(range(K + 1)))


# the capture predicate itself


@pytest.mark.parametrize(
    "predicate, expected",
    [
        (lambda: True, True),
        (lambda: False, False),
        (lambda: None, None),
        (lambda: 1, None),  # a truthy non-bool (a Mock, in a test double) is not proof
        ("raises", None),
    ],
)
def test_capture_state_is_three_valued(monkeypatch, predicate, expected):
    def boom():
        raise RuntimeError("no stream")

    fake = types.SimpleNamespace(is_current_stream_capturing=boom if predicate == "raises" else predicate)
    monkeypatch.setattr(torch, "npu", fake, raising=False)
    assert capture_state() is expected


def test_capture_state_without_torch_npu(monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)
    assert capture_state() is None


# plan_from_heat / check_proposal / pinned_resident_ids: fold_swap calls these,
# so they are covered directly here


class _Targets:
    def __init__(self, ids):
        self.kt_cpu_weight_ids = ids


def test_nothing_is_pinned_when_kt_holds_every_expert():
    assert pinned_resident_ids(_Targets(ALL_CPU_WEIGHTS), _placement(0)) == frozenset()


def test_residents_kt_has_no_cpu_weights_for_are_pinned():
    # kt holds CPU weights for 2..E only, so residents 0 and 1 cannot be
    # demoted: kt would read an unallocated BufferB and not raise
    p = _placement(0)                      # resident = START = 0..K-1
    held = tuple(range(2, E))
    assert pinned_resident_ids(_Targets(held), p) == frozenset({0, 1})


def test_a_proposal_that_drops_a_pinned_expert_is_refused_not_narrowed():
    p = _placement(0)
    proposal = tuple(range(1, K + 1))      # drops 0, adds K
    with pytest.raises(PolicyError, match="no CPU weights"):
        check_proposal(0, p, proposal, frozenset({0}))


@pytest.mark.parametrize(
    "ids, why",
    [
        ((0,) * 2 + tuple(range(1, K - 1)), "repeats"),
        (tuple(range(K - 1)) + (E,), "outside"),
        (tuple(range(K - 1)), "resident experts"),
        (tuple(reversed(range(K))), "ascending"),
    ],
)
def test_check_proposal_refuses(ids, why):
    with pytest.raises(PolicyError, match=why):
        check_proposal(0, _placement(0), ids, frozenset())


def test_an_uncounted_layer_plans_nothing():
    """All-zero counts mean "never counted": top-K over zeros would propose
    0..K-1 and move most of the layer for nothing.
    """
    assert plan_from_heat(0, _placement(0), torch.zeros(E, dtype=torch.float64),
                          TopKHeatPolicy(), frozenset()) is None


def test_plan_follows_the_heat():
    p = _placement(0)                      # resident 0..7
    counts = torch.zeros(E, dtype=torch.float64)
    hot = tuple(range(E - K, E))           # 24..31
    for e in hot:
        counts[e] = 10.0
    plan = plan_from_heat(0, p, counts, TopKHeatPolicy(), frozenset())
    assert plan is not None and tuple(plan.new_ids) == hot


def test_a_proposal_that_loses_coverage_is_refused():
    """The check must hold whatever the policy does, so a deliberately bad
    policy is injected."""
    hot = tuple(range(E - K, E))
    p = _placement(0, hot)                 # already resident on the hot set
    counts = torch.zeros(E, dtype=torch.float64)
    for e in hot:
        counts[e] = 10.0
    for e in range(K):
        counts[e] = 1.0
    with pytest.raises(PolicyError, match="less than"):
        plan_from_heat(0, p, counts, lambda **kw: tuple(range(K)), frozenset())
