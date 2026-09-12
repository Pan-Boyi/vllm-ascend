# SPDX-License-Identifier: Apache-2.0
"""Indices and their valid-entry count are one decision, not two.

The operator computes ``oriSparseBlockCount = hasOriTopkLength ?
Min(length, width) : width``. Omitting the length therefore declares every
slot in the row valid -- the ``-1`` padding included -- and the kernel gathers
at those addresses. Measured on the bench with exactly one variable moved: the
same index table crashes at the declared full width and passes at its true
length.

``Min`` reads like a guard and is not one. It clamps a length that exceeds how
wide the table is; nothing downstream knows how many entries were written, so
nothing can clamp to that. A default of "the whole row is valid" is the most
dangerous value available, which is what makes omitting it a defect rather
than an omission.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import (
    build_sparse_flash_mla_kwargs, ori_sparse_selection,
    select_ori_sparse_indices)

NUM_REQUESTS, NUM_TOKENS, NUM_HEADS, HEAD_DIM = 3, 12, 64, 512


class _Req:
    def __init__(self, vision=None, dspark=None, length=None):
        self.vision_swa_indices = vision
        self.dspark_swa_indices = dspark
        self.ori_topk_length = length


def _indices(rows=NUM_TOKENS, width=1152):
    return torch.zeros(rows, 1, width, dtype=torch.int32)


def _descriptor():
    blob = torch.zeros(1024, dtype=torch.int32)
    blob[0] = 28
    return blob


def _common(**overrides):
    kwargs = dict(
        q=torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_DIM),
        num_requests=NUM_REQUESTS,
        swa_kv_cache=torch.zeros(4, 32, 1, HEAD_DIM),
        swa_block_table=torch.zeros(NUM_REQUESTS, 4, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 4, 9, 12], dtype=torch.int32),
        seqused_kv=torch.tensor([4, 5, 3], dtype=torch.int32),
        sinks=torch.zeros(NUM_HEADS, dtype=torch.float32),
        metadata=_descriptor(),
        softmax_scale=0.04419,
        layout_kv="PA_BBND",
        cmp_ratio=1,
        attends_compressed=False,
        ori_win_left=127,
        ori_win_right=0,
    )
    kwargs.update(overrides)
    return kwargs


# ------------------------------------------------------------- the two travel together

def test_indices_without_a_length_are_refused():
    """The exact configuration production was sending."""
    with pytest.raises(ValueError, match="without ori_topk_length"):
        build_sparse_flash_mla_kwargs(
            **_common(ori_sparse_indices=_indices()))


def test_the_refusal_says_why_the_default_is_not_safe():
    """A message that only says "missing argument" invites adding None."""
    with pytest.raises(ValueError, match="most\n?\\s*dangerous one"):
        build_sparse_flash_mla_kwargs(
            **_common(ori_sparse_indices=_indices()))


def test_both_reach_the_call_when_both_are_given():
    lengths = torch.full((NUM_TOKENS,), 125, dtype=torch.int32)
    built = build_sparse_flash_mla_kwargs(
        **_common(ori_sparse_indices=_indices(), ori_topk_length=lengths))
    assert built["ori_sparse_indices"] is not None
    torch.testing.assert_close(built["ori_topk_length"], lengths)


def test_a_length_without_indices_is_not_sent():
    """Nothing to describe, so nothing is declared. The operator would read a
    length against a table it was not given."""
    built = build_sparse_flash_mla_kwargs(
        **_common(ori_topk_length=torch.ones(NUM_TOKENS, dtype=torch.int32)))
    assert "ori_topk_length" not in built
    assert "ori_sparse_indices" not in built


# -------------------------------------------------------------- the selection

def test_the_selection_returns_the_length_beside_its_own_table():
    vision, dspark = _indices(), _indices()
    v_len = torch.full((NUM_TOKENS,), 7, dtype=torch.int32)
    got_idx, got_len = ori_sparse_selection(False, _Req(vision=vision,
                                                        length=v_len))
    assert got_idx is vision
    assert got_len is v_len


def test_the_pair_follows_the_same_precedence_as_the_indices_alone():
    """Otherwise the length could describe the table that was not chosen."""
    vision, dspark = _indices(), _indices(width=8)
    req = _Req(vision=vision, dspark=dspark,
               length=torch.ones(NUM_TOKENS, dtype=torch.int32))
    for attends in (False, True):
        idx, _ = ori_sparse_selection(attends, req)
        assert idx is select_ori_sparse_indices(attends, vision, dspark)


def test_a_request_with_no_indices_yields_no_length():
    idx, length = ori_sparse_selection(False, _Req())
    assert idx is None and length is None


def test_a_metadata_object_predating_the_field_does_not_crash_the_selection():
    """getattr, not attribute access: an older metadata object should reach
    the explicit refusal in the builder rather than an AttributeError several
    frames from anything that mentions sparse indices."""

    class _Old:
        vision_swa_indices = _indices()
        dspark_swa_indices = None

    idx, length = ori_sparse_selection(False, _Old())
    assert idx is not None and length is None


# ----------------------------------------------------- where the length comes from

def test_the_builders_still_return_the_count_this_carries():
    """Derived from the builders' sources.

    Both already computed the per-row valid count and both discarded it at the
    call site. If either stops returning it, this file's whole premise --
    "carry a number that exists" -- becomes "derive one here", which is the
    second source the design exists to avoid.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1

    for fn in (dsa_v1.build_vision_bidirectional_swa_indices,
               dsa_v1.build_dspark_swa_indices):
        source = inspect.getsource(fn)
        returns = [ln for ln in source.splitlines()
                   if ln.strip().startswith("return ")]
        assert returns, fn.__name__
        assert "," in returns[-1], (
            f"{fn.__name__} no longer returns a second value; the valid-entry "
            "count it used to supply now has to come from somewhere else")


def test_the_call_sites_no_longer_discard_it():
    """`x, _ = build_...` is how the count was lost for as long as it existed."""
    import inspect

    from vllm_ascend.attention import dsa_v1

    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder)
    assert "swa_indices, _ = build_" not in source, (
        "a call site is discarding the valid-entry count again")
