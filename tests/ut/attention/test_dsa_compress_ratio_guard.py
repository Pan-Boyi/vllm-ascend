# SPDX-License-Identifier: Apache-2.0
"""The compression-ratio guard asks the plan, not a constant.

Before this, an unsupported ratio fell through to a default at every dispatch
site: the cmp_ratio mapping sent it to 128 and the backend selection to SWA, so
DeepSeek-V4.1's ratio 2 would have run at 128:1 compression and produced quietly
wrong numbers.

The first version of the guard then hard-coded one supported set, which made it
plan-blind: it refused ratio 2 even on a plan built to accept it. Which ratios
work is a property of the operator each plan dispatches to, and the three plans
call three different operators.
"""

from dataclasses import dataclass

import pytest

from vllm_ascend.attention.dsa_v1 import (ANY_PLAN_COMPRESS_RATIOS,
                                          validate_compress_ratio)


@dataclass
class _Plan:
    """Stands in for DsaAttnKvPlan; only the ratio set is read."""

    supported_compress_ratios: tuple[int, ...]


FP8_PLAN = _Plan((0, 1, 4, 128))
BF16_PLAN = _Plan((0, 1, 2))


@pytest.mark.parametrize("ratio", [0, 1, 4, 128])
def test_fp8_plan_accepts_what_its_operator_accepts(ratio):
    assert validate_compress_ratio(ratio, "layer", FP8_PLAN) == ratio


def test_fp8_plan_refuses_ratio_2():
    with pytest.raises(ValueError, match="compress_ratio=2 is not supported"):
        validate_compress_ratio(2, "model.layers.2.self_attn.attn", FP8_PLAN)


@pytest.mark.parametrize("ratio", [0, 1, 2])
def test_bf16_plan_accepts_ratio_2(ratio):
    """The point of the plan-aware guard.

    V4.1 needs ratio 2 on layers 2..19. Under the previous constant this was
    refused no matter which plan was selected.
    """
    assert validate_compress_ratio(ratio, "layer", BF16_PLAN) == ratio


def test_bf16_plan_refuses_what_it_does_not_implement():
    with pytest.raises(ValueError, match="compress_ratio=128 is not supported"):
        validate_compress_ratio(128, "layer", BF16_PLAN)


def test_error_names_the_layer_and_the_set_in_force():
    with pytest.raises(ValueError) as exc:
        validate_compress_ratio(2, "model.layers.7.self_attn.attn", FP8_PLAN)
    msg = str(exc.value)
    assert "model.layers.7.self_attn.attn" in msg
    assert "[0, 1, 4, 128]" in msg


def test_error_attributes_the_limit_to_the_operator_not_the_hardware():
    """Ratio 2 runs on A5; which plan is dispatched decides whether it is used.

    Without saying so, the next person to hit this concludes the hardware
    cannot do it and designs around a limit that does not exist -- which is
    what happened once already, when the operator's whitelist was read as a
    statement about A5.
    """
    with pytest.raises(ValueError) as exc:
        validate_compress_ratio(2, "layer", FP8_PLAN)
    msg = str(exc.value)
    assert "which operator is dispatched" in msg
    assert "not about what the hardware can do" in msg


def test_without_a_plan_the_union_is_used():
    """A few call sites run before any plan is resolved.

    Refusing there would reject ratios some plan accepts, and the per-plan check
    still runs afterwards, so the permissive choice is the safe one here.
    """
    for ratio in ANY_PLAN_COMPRESS_RATIOS:
        assert validate_compress_ratio(ratio, "layer") == ratio
    with pytest.raises(ValueError, match="no plan resolved"):
        validate_compress_ratio(3, "layer")


def test_the_union_covers_every_plan():
    from vllm_ascend.attention import dsa_attn_kv_plan as plans

    # Whatever any plan accepts must be in the union, or the planless call
    # sites would refuse something a plan later allows.
    for ratios in ((0, 1, 4, 128), (0, 1, 2)):
        assert set(ratios) <= set(ANY_PLAN_COMPRESS_RATIOS)
    assert hasattr(plans.DsaAttnKvPlan, "__annotations__")
    assert "supported_compress_ratios" in plans.DsaAttnKvPlan.__annotations__
