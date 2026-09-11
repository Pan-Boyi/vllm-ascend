# SPDX-License-Identifier: Apache-2.0
"""Every constraint sparse_flash_mla states about itself, checked on CPU.

The twelve rules below were measured on card 7 under CANN 9.2.0-beta.2 as the
operator's own rejection text (build_artifacts/KERNEL_LINE_SNAPSHOT_20260911.md
section 3). On a card they arrive one at a time, each costing a launch and a
round trip; here they arrive together, in 30ms, with no card involved.

Each rule gets two tests where it can: that a violation is caught, and that the
valid form passes. A validator that rejected everything would satisfy the first
half alone.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import (
    HEAD_DIM, validate_sparse_flash_mla_args)

NUM_REQUESTS, NUM_TOKENS, NUM_HEADS = 3, 12, 64


@pytest.fixture
def q():
    return torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_DIM)


def _ori_only(**overrides):
    """A layer attending only its own window stream (compress_ratio 0)."""
    kwargs = dict(
        layout_q="TND",
        layout_kv="PA_BBND",
        cu_seqlens_q=torch.tensor([0, 4, 9, 12], dtype=torch.int32),
        sinks=torch.zeros(NUM_HEADS, dtype=torch.float32),
        cmp_ratio=1,
        ori_win_left=127,
        ori_win_right=0,
        # Window-only layers pass mask mode 4; a layer attending a compressed
        # stream passes 0. See dsa_v1's ori_mask_mode selection.
        ori_mask_mode=4,
        cmp_mask_mode=0,
    )
    kwargs.update(overrides)
    return kwargs


def _with_compressed(**overrides):
    """A layer attending a compressed stream as well (ratio 2, layers 2..19)."""
    kwargs = dict(
        layout_q="TND",
        layout_kv="PA_BBND",
        cu_seqlens_q=torch.tensor([0, 4, 9, 12], dtype=torch.int32),
        sinks=torch.zeros(NUM_HEADS, dtype=torch.float32),
        cmp_ratio=2,
        cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
        seqused_cmp_kv=torch.tensor([2, 3, 2], dtype=torch.int32),
        ori_win_left=-1,
        ori_win_right=-1,
        ori_mask_mode=0,
        cmp_mask_mode=3,
    )
    kwargs.update(overrides)
    return kwargs


def _validate(kwargs, q):
    return validate_sparse_flash_mla_args(kwargs, q, NUM_REQUESTS)


# -- the two configurations that must pass -----------------------------------


def test_the_window_only_configuration_is_accepted(q):
    assert _validate(_ori_only(), q) is not None


def test_the_compressed_configuration_is_accepted(q):
    assert _validate(_with_compressed(), q) is not None


def test_ratio_one_with_a_compressed_stream_needs_a_residual(q):
    """Layers 20..39: ratio 1 meaning 'compressed, pooling disabled'."""
    ok = _with_compressed(cmp_ratio=1,
                          cmp_residual_kv=torch.zeros(NUM_REQUESTS, dtype=torch.int32))
    assert _validate(ok, q) is not None

    with pytest.raises(ValueError, match="additionally requires"):
        _validate(_with_compressed(cmp_ratio=1), q)


# -- one test per stated constraint ------------------------------------------


def test_pa_nd_is_rejected(q):
    """The FP8 plan's layout belongs to a different operator."""
    with pytest.raises(ValueError, match=r"layout_kv must be in"):
        _validate(_ori_only(layout_kv="PA_ND"), q)


def test_tnd_q_requires_a_cumulative_cu_seqlens(q):
    with pytest.raises(ValueError, match="requires cu_seqlens_q"):
        _validate(_ori_only(cu_seqlens_q=None), q)
    # right length, wrong contents: lengths rather than a cumulative sum
    with pytest.raises(ValueError, match="not non-decreasing|cumulative"):
        _validate(_ori_only(cu_seqlens_q=torch.tensor([4, 5, 3, 0])), q)
    # wrong length
    with pytest.raises(ValueError, match="cumulative"):
        _validate(_ori_only(cu_seqlens_q=torch.tensor([0, 4, 9])), q)


def test_cu_seqlens_kv_only_under_tnd(q):
    with pytest.raises(ValueError, match="only when layout_kv is TND"):
        _validate(_ori_only(cu_seqlens_ori_kv=torch.tensor([0, 4, 9, 12])), q)
    with pytest.raises(ValueError, match="only when layout_kv is TND"):
        _validate(_with_compressed(cu_seqlens_cmp_kv=torch.tensor([0, 2, 5, 7])), q)
    # ... and is fine under TND
    assert _validate(_ori_only(layout_kv="TND",
                               cu_seqlens_ori_kv=torch.tensor([0, 4, 9, 12])), q)


def test_sinks_cannot_be_omitted(q):
    with pytest.raises(ValueError, match="sinks must be provided"):
        _validate(_ori_only(sinks=None), q)


def test_q_head_dim_must_be_512(q):
    with pytest.raises(ValueError, match="head_dim must be 512"):
        _validate(_ori_only(), torch.zeros(NUM_TOKENS, NUM_HEADS, 448))


def test_a_compressed_stream_must_disable_both_bounds(q):
    """Half-disabled counts as windowed: one bound left set is still a window."""
    for left, right in ((127, -1), (-1, 0), (127, 0)):
        with pytest.raises(ValueError, match="SWA mode|ori_win_left = ori_win_right"):
            _validate(_with_compressed(ori_win_left=left, ori_win_right=right), q)


def test_the_unmodelled_mask_mode_rule_is_declared_as_such(q):
    """One stated constraint is deliberately not checked.

    "ori-only sparse mode requires both mask modes to be 0" needs a way to tell
    ori-only *sparse* mode from plain sliding-window mode, and nothing measured
    distinguishes them. Leaving it out is a decision; this pins that the
    decision stays visible rather than becoming a silent gap, and that the
    validator does not quietly start guessing at it.
    """
    from vllm_ascend.models.deepseek_v4_1 import sparse_attn_args

    doc = sparse_attn_args.__doc__
    assert "not** modelled" in doc
    assert "ori-only\nsparse mode requires both mask modes to be 0" in doc

    # and it really does accept mask mode 4 on a window-only layer
    assert _validate(_ori_only(ori_mask_mode=4), q) is not None


def test_a_windowed_layer_may_not_also_carry_a_compressed_stream(q):
    with pytest.raises(ValueError, match="SWA mode requires cmp_ratio = 1"):
        _validate(_with_compressed(ori_win_left=127, ori_win_right=0), q)


def test_cmp_ratio_zero_is_never_legal_at_this_boundary(q):
    """0 is a legal configured compress_ratio and an illegal operator
    parameter; conflating the two sends the next reader to the validator
    rather than to the mapping."""
    with pytest.raises(ValueError, match="cmp_ratio must be greater than 0"):
        _validate(_ori_only(cmp_ratio=0), q)


def test_paged_compressed_stream_requires_seqused_cmp_kv(q):
    with pytest.raises(ValueError, match="requires seqused_cmp_kv"):
        _validate(_with_compressed(seqused_cmp_kv=None), q)


def test_cmp_residual_kv_is_one_value_per_request_not_kv_data(q):
    """Despite the name. The operator's own words: 'incorrect shape dim 3.
    It should be 1'."""
    with pytest.raises(ValueError, match="It should be 1"):
        _validate(_with_compressed(cmp_ratio=1,
                                   cmp_residual_kv=torch.zeros(NUM_REQUESTS, 8, 512)), q)
    with pytest.raises(ValueError, match="It should be 1"):
        _validate(_with_compressed(cmp_ratio=1,
                                   cmp_residual_kv=torch.zeros(NUM_REQUESTS + 1)), q)


def test_the_error_quotes_the_operator_rather_than_paraphrasing(q):
    """So a message found in a log can be grepped straight back to here."""
    with pytest.raises(ValueError) as exc:
        _validate(_ori_only(cmp_ratio=0), q)
    assert "sparse_flash_mla would reject this call" in str(exc.value)
    assert "cmp_ratio must be greater than 0" in str(exc.value)
