# SPDX-License-Identifier: Apache-2.0
"""The routing tap, at the one door both MoE paths go through.

🔴 moe_gate.out was first hooked on the `gate` submodule, and that hook can
never fire on this platform: the fused layer does not call the gate as a
module, it reads gate.weight and does the matmul itself
(ops/fused_moe/fused_moe.py:306-312). "It did not fire this time" and "it can
never fire" need different fixes -- wait for the condition, or move the hook --
and only tracing who actually does the work separates them. Measured: 160
reconstruction lines across 40 layers x 4 ranks, zero of them successful.

The routing does go through a router method -- Ascend's routed experts call
the private one, router._select_experts (ops/fused_moe/routed_experts.py:571).
Two peers and I all said otherwise, reading a block that ends in
forward_impl(...); the call is one frame further in.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import moe_router


class _Router:
    """Stands in for a FusedMoERouter. Signature copied from
    vllm/model_executor/layers/fused_moe/router/fused_moe_router.py:49."""

    def __init__(self):
        self.calls = 0

    def _select_experts(self, hidden_states, router_logits,
                        topk_indices_dtype=None, *, input_ids=None):
        """⚠️ The PRIVATE name, copied from the call site that actually runs:
        ops/fused_moe/routed_experts.py:571 calls router._select_experts."""
        self.calls += 1
        n = hidden_states.shape[0]
        return torch.ones(n, 6), torch.arange(n * 6).reshape(n, 6)


@pytest.fixture
def armed(monkeypatch):
    recorded = []
    from vllm_ascend.models.deepseek_v4_1 import align_taps
    monkeypatch.setattr(align_taps, "alignment_active", lambda: True)
    monkeypatch.setattr(align_taps, "current_layer", lambda: 7)
    monkeypatch.setattr(align_taps, "current_positions",
                        lambda: torch.arange(3))
    monkeypatch.setattr(align_taps, "tap",
                        lambda layer, key, tensor, positions: recorded.append(
                            (layer, key, tuple(tensor.shape))))
    moe_router.reset_router_tap_for_test()
    return recorded


def test_the_tap_speaks_before_anything_depends_on_it(armed):
    """🔴 "Prove it speaks" first -- the discipline A-2 was fixed under.

    A tap that is wired but silent and a tap that is not wired produce the
    same dump, so the wiring is exercised here rather than assumed.
    """
    router = moe_router._tap_select_experts(_Router())
    weights, ids = router._select_experts(torch.zeros(3, 8), torch.zeros(3, 4))

    assert [key for _, key, _ in armed] == ["moe_gate.out.0", "moe_gate.out.1"]
    assert [layer for layer, _, _ in armed] == [7, 7]
    assert [shape for _, _, shape in armed] == [(3, 6), (3, 6)]
    # And it returns exactly what the router returned.
    assert weights.shape == (3, 6) and ids.shape == (3, 6)


def test_the_wrapper_counts_itself(armed):
    """⚠️ An absent key has three causes: no alignment run, this path not
    taken, or the hook attached to something never called. The third is what
    bit the gate hook, and an attribute test cannot see it."""
    assert moe_router.router_tap_calls() == 0
    router = moe_router._tap_select_experts(_Router())
    for _ in range(3):
        router._select_experts(torch.zeros(2, 8), torch.zeros(2, 4))
    assert moe_router.router_tap_calls() == 3


def test_the_underlying_router_still_runs_and_its_result_is_untouched():
    """The tap observes; it must not become part of the computation."""
    moe_router.reset_router_tap_for_test()
    inner = _Router()
    router = moe_router._tap_select_experts(inner)
    weights, ids = router._select_experts(torch.zeros(4, 8), torch.zeros(4, 4))
    assert inner.calls == 1
    assert torch.equal(weights, torch.ones(4, 6))
    assert torch.equal(ids, torch.arange(24).reshape(4, 6))


def test_a_tap_failure_does_not_reach_the_model(monkeypatch):
    """It has already cost this line one flight to let one of these
    propagate."""
    from vllm_ascend.models.deepseek_v4_1 import align_taps

    moe_router.reset_router_tap_for_test()
    monkeypatch.setattr(align_taps, "alignment_active", lambda: True)
    monkeypatch.setattr(align_taps, "current_layer", lambda: 0)
    monkeypatch.setattr(align_taps, "current_positions",
                        lambda: torch.arange(2))

    def _explode(*a, **k):
        raise RuntimeError("instrument")

    monkeypatch.setattr(align_taps, "tap", _explode)
    router = moe_router._tap_select_experts(_Router())
    weights, ids = router._select_experts(torch.zeros(2, 8), torch.zeros(2, 4))
    assert weights.shape == (2, 6)


def test_routing_outside_a_layer_records_nothing_and_says_so(monkeypatch):
    """A key under the wrong layer is worse than an absent one."""
    from vllm_ascend.models.deepseek_v4_1 import align_taps

    moe_router.reset_router_tap_for_test()
    warned = []
    monkeypatch.setattr("vllm.logger.logger.warning",
                        lambda fmt, *a: warned.append(fmt % a))
    monkeypatch.setattr(align_taps, "alignment_active", lambda: True)
    monkeypatch.setattr(align_taps, "current_layer", lambda: None)
    recorded = []
    monkeypatch.setattr(align_taps, "tap",
                        lambda *a, **k: recorded.append(a))

    router = moe_router._tap_select_experts(_Router())
    router._select_experts(torch.zeros(2, 8), torch.zeros(2, 4))
    assert recorded == []
    assert len(warned) == 1 and "which layer" in warned[0]


def test_the_arguments_actually_used_are_reported_once(monkeypatch):
    """🔴 A reading, not a check. Half of these have no written-down right
    value; what was missing was the measurement. routed_scaling_factor
    defaults to 1.0 and this model's config says 1.5 -- static reading says
    upstream passes it, and this line is what makes that runtime evidence."""
    said = []
    monkeypatch.setattr("vllm.logger.logger.info",
                        lambda fmt, *a: said.append(fmt % a))
    moe_router.reset_router_tap_for_test()

    moe_router._report_router_arguments(
        {"top_k": 6, "routed_scaling_factor": 1.5, "zero_expert_type": None,
         "e_score_correction_bias": torch.zeros(384)},
        {"top_k", "routed_scaling_factor", "e_score_correction_bias"})
    assert len(said) == 1
    assert "'routed_scaling_factor': 1.5" in said[0]
    assert "zero_expert_type" in said[0]
    # 🔴 The hole this closed. e_score_correction_bias is ACCEPTED by the
    # factory, so it is not "dropped"; it was not on the hand-written list of
    # interesting names, so it was not "seen" either. A reader looking for it
    # found it in neither column and could not tell "not passed" from "passed
    # and not printed" -- and those have opposite meanings.
    assert "e_score_correction_bias" in said[0]
    assert "<tensor (384,)" in said[0], "a tensor is summarised, not dumped"
    # And a behavioural default that really was not passed still says so.
    assert "shared_expert_weight" in said[0]

    moe_router._report_router_arguments({"top_k": 1}, {"top_k"})
    assert len(said) == 1, "once per process, not once per layer"


def test_the_count_is_printed_even_when_it_is_zero(monkeypatch):
    """🔴 The ambiguity that cost four flights.

    The counter existed for three of them and was never printed -- the reader
    was defined and called from nowhere. "The wrapper did not run" and "the
    wrapper ran and the count has no printer" then produce the same output:
    nothing. A line that appears only when the count is non-zero carries
    exactly the information that is missing when it is zero.
    """
    said = []
    monkeypatch.setattr("vllm.logger.logger.info",
                        lambda fmt, *a: said.append(fmt % a))
    moe_router.reset_router_tap_for_test()
    moe_router.report_router_tap_coverage()
    assert len(said) == 1
    assert "entered 0 time(s)" in said[0] and "ZERO" in said[0]

    router = moe_router._tap_select_experts(_Router())
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.alignment_active",
        lambda: False)
    router._select_experts(torch.zeros(2, 8), torch.zeros(2, 4))
    said.clear()
    moe_router.report_router_tap_coverage()
    assert "entered 1 time(s)" in said[0] and "ZERO" not in said[0]


def test_the_flush_path_prints_it():
    """Wired where the recorder rotates, not called by hand -- a reader
    nobody calls is the same as no reader, which is the whole lesson here."""
    import ast
    import inspect
    import textwrap

    from vllm_ascend.models.deepseek_v4_1 import align_taps

    source = textwrap.dedent(inspect.getsource(align_taps._flush_and_rotate))
    called = {node.func.id for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "report_router_tap_coverage" in called


def test_a_swallowed_tap_failure_says_so_once(monkeypatch):
    """A swallowed failure and a hook that never ran produce the same empty
    dump; this tap has already spent four flights on that ambiguity."""
    from vllm_ascend.models.deepseek_v4_1 import align_taps

    warned = []
    monkeypatch.setattr("vllm.logger.logger.warning",
                        lambda fmt, *a: warned.append(fmt % a))
    moe_router.reset_router_tap_for_test()
    monkeypatch.setattr(align_taps, "alignment_active", lambda: True)
    monkeypatch.setattr(align_taps, "current_layer", lambda: 0)
    monkeypatch.setattr(align_taps, "current_positions",
                        lambda: torch.arange(2))

    def _explode(*a, **k):
        raise RuntimeError("instrument")

    monkeypatch.setattr(align_taps, "tap", _explode)
    router = moe_router._tap_select_experts(_Router())
    for _ in range(3):
        router._select_experts(torch.zeros(2, 8), torch.zeros(2, 4))
    assert len(warned) == 1, "once, not once per token"
    assert "swallowed" in warned[0] and "not because the path was not taken" in warned[0]
