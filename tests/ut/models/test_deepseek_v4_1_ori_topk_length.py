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


# ------------------------------------------- what makes the V4.0 change safe

def _vision_table(window=4, max_image=8, page=16, blocks=8, seq=10,
                  spans=None):
    from vllm_ascend.attention.dsa_v1 import \
        build_vision_bidirectional_swa_indices

    block_table = torch.arange(blocks, dtype=torch.int32).repeat(1, 1)
    return build_vision_bidirectional_swa_indices(
        block_table=block_table,
        window_size=window,
        max_image_tokens=max_image,
        block_size=page,
        query_start_loc=torch.tensor([0, seq], dtype=torch.int32),
        seq_lens=torch.tensor([seq], dtype=torch.int32),
        mm_prefix_ranges=spans if spans is not None else {0: []},
        num_tokens=seq,
    )


def _dspark_table(window=4, spec=2, page=16, blocks=8, seq=6):
    from vllm_ascend.attention.dsa_v1 import build_dspark_swa_indices

    block_table = torch.arange(blocks, dtype=torch.int32).repeat(1, 1)
    return build_dspark_swa_indices(
        block_table,
        spec,
        window,
        page,
        torch.tensor([0, seq], dtype=torch.int32),
        torch.tensor([seq], dtype=torch.int32),
        seq,
    )


@pytest.mark.parametrize("build", [_vision_table, _dspark_table],
                         ids=["vision", "dspark"])
def test_the_length_is_exactly_the_count_of_real_entries(build):
    """The invariant the whole rule rests on.

    Not "roughly the valid count" and not "at most the width": the operator
    reads this number as how many leading entries to gather, so it has to be
    the count of entries that are not -1. Slots are non-negative, so the two
    are the same statement.
    """
    indices, lengths = build()
    rows = indices.reshape(indices.shape[0], -1)
    counted = (rows != -1).sum(dim=-1).to(lengths.dtype)
    torch.testing.assert_close(lengths, counted)


@pytest.mark.parametrize("build", [_vision_table, _dspark_table],
                         ids=["vision", "dspark"])
def test_the_old_default_was_wrong_on_every_row_of_these_tables(build):
    """Why changing the shared V4.0 call site is not a gamble.

    The argument for touching a path this line does not exercise was: the new
    behaviour differs from the old only where the old one was already wrong,
    because a table with no padding has count == width and the two agree.

    That argument is weaker than it needs to be here, and the measurement says
    so: neither builder can produce a padding-free row at realistic sizes. The
    vision width is window + max_image_tokens while a row's span is bounded by
    the larger of the two, and the dspark width is rounded up to a multiple of
    128. So on these tables the old default -- "the whole row is valid" -- was
    wrong on *every* row, and the change cannot be a regression on any of
    them. If this ever goes red, a padding-free table has become possible and
    the no-op half of the argument has to carry the weight again.
    """
    indices, lengths = build()
    width = indices.reshape(indices.shape[0], -1).shape[-1]
    assert int(lengths.max()) < width, (
        f"a row now fills all {width} columns; the old full-width default "
        "would have been correct for it, so the reasoning about the shared "
        "V4.0 path needs re-reading rather than this assertion relaxing")


def test_a_padding_free_row_would_have_declared_the_full_width():
    """The no-op half, stated directly rather than left implicit.

    Constructed rather than found in a builder, because the builders cannot
    produce one: this pins what the rule does in that case, which is nothing.
    """
    row = torch.arange(6, dtype=torch.int32).reshape(1, 6)
    count = int((row != -1).sum())
    assert count == row.shape[-1], (
        "with no -1 the true length is the full width, which is exactly what "
        "the operator assumes when no length is passed -- so supplying it "
        "changes nothing for such a table")
