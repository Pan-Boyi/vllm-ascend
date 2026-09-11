# SPDX-License-Identifier: Apache-2.0
"""The grouped wo_a layout check.

wo_a is block-diagonal: 8 groups, each mapping its own 4096 input dimensions to
1024. Whether it arrives that way is a property of which quantisation method
served the linear, not of the model -- nothing on Ascend reads the ``is_bmm``
flag upstream sets to mark this weight, and the two places that do reshape it
select it by the prefix ending in "wo_a".

These sizes cannot multiply cleanly in the wrong layout, so a wrong one fails
either way. What this check changes is which failure: a named reshape that did
not run, rather than a shape mismatch reported from inside an operator.

In its own file: the golden-checkpoint tests for the o_proj helpers carry a
module-level skip for when the checkpoint is absent, and nothing here needs it.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.o_proj import (check_grouped_wo_a_layout,
                                                     group_wo_a)

# DeepSeek-V4.1-Flash: 64 heads x 512 head_dim = 32768, split into 8 groups of
# 4096, each projected to 1024.
O_GROUPS, GROUP_IN, O_LORA_RANK = 8, 4096, 1024


def _correct() -> torch.Tensor:
    return torch.empty(O_GROUPS, GROUP_IN, O_LORA_RANK)


def test_the_ascend_layout_is_accepted():
    check_grouped_wo_a_layout(_correct(), O_GROUPS, GROUP_IN, O_LORA_RANK, "w")


def test_a_flat_weight_names_the_load_time_step():
    """The flat form is the checkpoint's own shape, so this is what a layer
    served by a method that does not reshape wo_a actually looks like."""
    flat = torch.empty(O_GROUPS * O_LORA_RANK, GROUP_IN)
    with pytest.raises(ValueError) as exc:
        check_grouped_wo_a_layout(flat, O_GROUPS, GROUP_IN, O_LORA_RANK,
                                  "model.layers.3.self_attn.wo_a")
    msg = str(exc.value)
    assert "model.layers.3.self_attn.wo_a" in msg
    assert "flat checkpoint form" in msg
    # It has to send the reader to the quantisation method, not to the
    # projection -- the projection is where it surfaces, not where it is.
    assert "quantisation method" in msg
    assert "is_bmm" in msg


def test_the_transposed_layout_is_called_out_specifically():
    """The easy mistake, not a rare one.

    group_wo_a, upstream's DeepGEMM path and the ROCm einsum all use
    [groups, rank, in]; only the Ascend operators want the contraction
    dimension in the middle. Mixing the two conventions is one transpose away.
    """
    transposed = torch.empty(O_GROUPS, O_LORA_RANK, GROUP_IN)
    with pytest.raises(ValueError, match="transposed layout"):
        check_grouped_wo_a_layout(transposed, O_GROUPS, GROUP_IN, O_LORA_RANK,
                                  "w")


def test_group_wo_a_produces_the_layout_the_check_rejects():
    """Pins the two conventions against each other.

    If group_wo_a is ever changed to emit the Ascend order, this test goes red
    and says so, rather than the two silently agreeing and one caller getting
    the transpose it did not ask for.
    """
    flat = torch.empty(O_GROUPS * O_LORA_RANK, GROUP_IN)
    grouped = group_wo_a(flat, O_GROUPS, O_LORA_RANK)
    assert tuple(grouped.shape) == (O_GROUPS, O_LORA_RANK, GROUP_IN)
    with pytest.raises(ValueError, match="transposed layout"):
        check_grouped_wo_a_layout(grouped, O_GROUPS, GROUP_IN, O_LORA_RANK, "w")

    # and the transpose of it is what the Ascend operators take.
    check_grouped_wo_a_layout(grouped.transpose(1, 2), O_GROUPS, GROUP_IN,
                              O_LORA_RANK, "w")


def test_an_unrelated_shape_still_reports_both_shapes():
    with pytest.raises(ValueError) as exc:
        check_grouped_wo_a_layout(torch.empty(3, 5, 7), O_GROUPS, GROUP_IN,
                                  O_LORA_RANK, "w")
    msg = str(exc.value)
    assert "(3, 5, 7)" in msg
    assert f"({O_GROUPS}, {GROUP_IN}, {O_LORA_RANK})" in msg


def test_a_wrong_group_count_is_not_waved_through():
    """Right rank, right per-group width, wrong number of groups."""
    with pytest.raises(ValueError):
        check_grouped_wo_a_layout(torch.empty(4, GROUP_IN, O_LORA_RANK),
                                  O_GROUPS, GROUP_IN, O_LORA_RANK, "w")
