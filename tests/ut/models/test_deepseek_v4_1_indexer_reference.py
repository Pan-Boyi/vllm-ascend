# SPDX-License-Identifier: Apache-2.0
"""The reference has to be wrong in different ways than the thing it checks.

It is plain torch against a fused NPU operator, so the shared failure modes are
the transcription ones -- the weight fold, the causal bound, the group-boundary
arithmetic. Those are tested here directly, because a reference that makes the
same mistake as the implementation agrees with it and proves nothing.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.indexer_reference import (
    causal_topk, indexer_logits, pad_value_of, selection_disagreement)


def test_logits_are_the_weighted_sum_over_heads():
    """Written out the slow way, because the einsum is the thing under test."""
    torch.manual_seed(0)
    q = torch.randn(3, 4, 8)
    k = torch.randn(5, 8)
    w = torch.randn(3, 4)
    got = indexer_logits(q, k, w)
    for t in range(3):
        for s in range(5):
            expect = sum(float(w[t, h]) * float(q[t, h] @ k[s]) for h in range(4))
            assert got[t, s].item() == pytest.approx(expect, rel=1e-5)


def test_the_weights_are_applied_not_recomputed():
    """The FP8 path folds the per-token q scale into `weights`.

    A reference that re-applied a scale would disagree with a correct
    implementation and the disagreement would look like an operator bug.
    """
    q = torch.ones(1, 2, 4)
    k = torch.ones(3, 4)
    doubled = indexer_logits(q, k, torch.full((1, 2), 2.0))
    single = indexer_logits(q, k, torch.ones(1, 2))
    assert torch.allclose(doubled, 2 * single)


def test_accumulation_is_float32_whatever_comes_in():
    """bf16 inputs, fp32 arithmetic -- otherwise the reference carries the same
    rounding as the thing it is checking."""
    q = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    k = torch.randn(4, 16, dtype=torch.bfloat16)
    w = torch.randn(2, 3, dtype=torch.bfloat16)
    assert indexer_logits(q, k, w).dtype == torch.float32


@pytest.mark.parametrize("ratio,position,visible", [
    (1, 0, 1), (1, 7, 8),
    (2, 0, 0), (2, 1, 1), (2, 3, 2), (2, 7, 4),
])
def test_how_many_states_a_position_can_see(ratio, position, visible):
    """(position + 1) // ratio -- the same arithmetic the K writer gates on.

    At ratio 2 a token at position 0 sees nothing: its group has not closed, so
    no key has been published for it yet.
    """
    logits = torch.zeros(1, 8)
    logits[0] = torch.arange(8, dtype=torch.float32)
    out = causal_topk(logits, torch.tensor([position]), ratio, topk=8)
    assert int((out[0] != -1).sum()) == visible


def test_a_short_row_is_padded_not_filled_with_the_best_of_nothing():
    """The case the operator's convention has to be read off.

    Two visible states and a topk of 8 leaves six slots. They must be the pad
    value, not repeats of the best state, and not whatever the buffer held.
    """
    # Position 1 at ratio 1 sees two states, 0 and 1. States 2 and 3 score
    # higher and are still invisible -- which is the other half of this test.
    logits = torch.tensor([[3.0, 1.0, 9.0, 9.0]])
    out = causal_topk(logits, torch.tensor([1]), compress_ratio=1, topk=8)
    assert [int(v) for v in out[0, :2].tolist()] == [0, 1]
    assert (out[0, 2:] == -1).all()


def test_the_top_k_is_by_score_not_by_position():
    logits = torch.tensor([[1.0, 5.0, 2.0, 4.0]])
    out = causal_topk(logits, torch.tensor([3]), compress_ratio=1, topk=2)
    assert set(int(v) for v in out[0].tolist()) == {1, 3}


def test_a_reordered_selection_is_not_a_disagreement():
    """A top-k is a set; comparing element-wise reports order as error."""
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    got = torch.tensor([[2, 1, -1]], dtype=torch.int32)
    expected = torch.tensor([[1, 2, -1]], dtype=torch.int32)
    report = selection_disagreement(got, expected, logits)
    assert report["differing_rows"] == []


def test_a_tie_is_reported_as_a_tie_not_as_an_error():
    """Equal logits make the selected set ambiguous. Calling that a mismatch
    would send someone hunting an operator bug that is arithmetic."""
    logits = torch.tensor([[5.0, 5.0, 1.0]])
    report = selection_disagreement(torch.tensor([[0, -1]], dtype=torch.int32),
                                    torch.tensor([[1, -1]], dtype=torch.int32),
                                    logits)
    assert report["differing_rows"] == [0]
    assert report["all_ties"] is True
    assert report["worst_logit_gap"] == 0.0


def test_a_real_ranking_difference_reports_its_size():
    """The case that matters: different sets and different scores."""
    logits = torch.tensor([[9.0, 1.0, 5.0]])
    report = selection_disagreement(torch.tensor([[1, -1]], dtype=torch.int32),
                                    torch.tensor([[0, -1]], dtype=torch.int32),
                                    logits)
    assert report["all_ties"] is False
    assert report["worst_logit_gap"] == pytest.approx(8.0)


def test_the_pad_value_is_read_rather_than_assumed():
    row = torch.tensor([4, 7, -1, -1, -1], dtype=torch.int32)
    assert pad_value_of(row, num_valid=2) == -1
    assert pad_value_of(torch.tensor([4, 7, 0, 0], dtype=torch.int32), 2) == 0


def test_a_non_constant_tail_is_reported_as_no_pad_value():
    """Uninitialised is not padded, and the difference decides whether the
    tail can be read as token ids at all."""
    row = torch.tensor([4, 7, 3, 9], dtype=torch.int32)
    assert pad_value_of(row, num_valid=2) is None
