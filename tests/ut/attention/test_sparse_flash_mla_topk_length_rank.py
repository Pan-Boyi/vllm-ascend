"""`*_topk_length` must carry the leading shape of the table it describes.

Measured 2026-09-13, first request that reached the operator on the c18 arm::

    EZ0011: Parameter ori_topk_length of SparseFlashMla has incorrect
            shape dim 1. It should be 2.

Both SWA builders return the table with a head axis and the count on the very
next line without one::

    per_token_slots = repeat_interleave(slot_ids, ...).unsqueeze(1)  # [T, 1, W]
    per_token_lens  = repeat_interleave(visible_lens, ...)           # [T]

Nothing between those two lines checks that the companion still matches the
table it counts -- which is the same shape as the last few defects in this
family: a value that travels beside a tensor and is free to drift from it.
"""
import math

import pytest
import torch

from vllm_ascend.attention.sparse_flash_mla import _match_topk_length_rank

T, W, H = 5, 512, 4


def _table(rows=T, heads=1, width=W):
    return torch.zeros(rows, heads, width, dtype=torch.int32)


@pytest.mark.parametrize("length_key,table_key", [
    ("ori_topk_length", "ori_sparse_indices"),
    ("cmp_topk_length", "cmp_sparse_indices"),
])
def test_rank1_count_is_reshaped_onto_the_table(length_key, table_key):
    kwargs = {table_key: _table(), length_key: torch.arange(T, dtype=torch.int32)}
    _match_topk_length_rank(kwargs)
    assert tuple(kwargs[length_key].shape) == (T, 1)
    # The values are the point; only the shape may change.
    assert torch.equal(kwargs[length_key].reshape(-1),
                       torch.arange(T, dtype=torch.int32))


def test_already_matching_is_left_untouched():
    lengths = torch.zeros(T, 1, dtype=torch.int32)
    kwargs = {"cmp_sparse_indices": _table(), "cmp_topk_length": lengths}
    _match_topk_length_rank(kwargs)
    assert kwargs["cmp_topk_length"] is lengths


def test_row_count_disagreement_raises_rather_than_reshaping():
    """The third state. Reshaping here would be worse than failing.

    A count whose element total does not match the table can still be forced
    into the right *rank*; what comes out is a plausible tensor describing the
    wrong rows, and the operator would gather at those. This parameter exists
    to stop exactly that.
    """
    kwargs = {"ori_sparse_indices": _table(),
              "ori_topk_length": torch.zeros(T + 3, dtype=torch.int32)}
    with pytest.raises(ValueError) as excinfo:
        _match_topk_length_rank(kwargs)
    # The message has to name both sides: which count, and which table.
    assert "ori_topk_length" in str(excinfo.value)
    assert "ori_sparse_indices" in str(excinfo.value)


@pytest.mark.parametrize("kwargs", [
    {"ori_topk_length": torch.zeros(T, dtype=torch.int32)},   # count, no table
    {"ori_sparse_indices": _table()},                          # table, no count
    {},
])
def test_a_missing_half_is_left_alone(kwargs):
    before = {k: v for k, v in kwargs.items()}
    _match_topk_length_rank(kwargs)
    assert set(kwargs) == set(before)
    for k, v in before.items():
        assert kwargs[k] is v


def test_multi_head_table_is_why_this_reshapes_instead_of_unsqueezing():
    """`unsqueeze(-1)` would pass the single-head case and be wrong here.

    Deriving the shape from the table covers both; adding a fixed axis at the
    two builders only covers the shape those builders happen to emit today.
    """
    kwargs = {"cmp_sparse_indices": _table(heads=H),
              "cmp_topk_length": torch.zeros(T * H, dtype=torch.int32)}
    _match_topk_length_rank(kwargs)
    assert tuple(kwargs["cmp_topk_length"].shape) == (T, H)
    assert math.prod(kwargs["cmp_topk_length"].shape) == T * H
