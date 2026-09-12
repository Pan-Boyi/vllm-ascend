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


# -- assembling a call from metadata -----------------------------------------


def _common(**overrides):
    kwargs = dict(
        q=torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_DIM),
        num_requests=NUM_REQUESTS,
        swa_kv_cache=torch.zeros(4, 32, 1, HEAD_DIM),
        swa_block_table=torch.zeros(NUM_REQUESTS, 4, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 4, 9, 12], dtype=torch.int32),
        seqused_kv=torch.tensor([4, 5, 3], dtype=torch.int32),
        sinks=torch.zeros(NUM_HEADS, dtype=torch.float32),
        metadata=object(),
        softmax_scale=0.04419,
        layout_kv="PA_BBND",
        ori_win_left=127,
        ori_win_right=0,
    )
    kwargs.update(overrides)
    return kwargs


def test_a_window_only_layer_keeps_its_bounds_and_mask_mode_4():
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        build_sparse_flash_mla_kwargs

    built = build_sparse_flash_mla_kwargs(
        **_common(cmp_ratio=1, attends_compressed=False))
    assert built["ori_mask_mode"] == 4
    assert (built["ori_win_left"], built["ori_win_right"]) == (127, 0)
    assert "cmp_kv" not in built and "cmp_mask_mode" not in built


def test_a_compressed_layer_disables_its_bounds():
    """Not a preference -- the operator rejects bounds alongside a compressed
    stream."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        build_sparse_flash_mla_kwargs

    built = build_sparse_flash_mla_kwargs(**_common(
        cmp_ratio=2, attends_compressed=True,
        cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
        cmp_block_table=torch.zeros(NUM_REQUESTS, 2, dtype=torch.int32),
        seqused_cmp_kv=torch.tensor([2, 3, 2], dtype=torch.int32)))
    assert built["ori_mask_mode"] == 0
    assert (built["ori_win_left"], built["ori_win_right"]) == (-1, -1)
    assert built["cmp_mask_mode"] == 3


def test_ratio_one_may_still_attend_a_compressed_stream():
    """V4.1 layers 20..39.

    V4.0 reads ratio <= 1 as "no compressed stream"; V4.1 gives those layers
    ratio 1 to mean "compressed, pooling disabled". Inferring one from the
    other is the mistake this port already made once, so attends_compressed is
    passed in rather than derived.
    """
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        build_sparse_flash_mla_kwargs

    built = build_sparse_flash_mla_kwargs(**_common(
        cmp_ratio=1, attends_compressed=True,
        cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
        cmp_block_table=torch.zeros(NUM_REQUESTS, 2, dtype=torch.int32),
        seqused_cmp_kv=torch.tensor([2, 3, 2], dtype=torch.int32),
        cmp_residual_kv=torch.zeros(NUM_REQUESTS, dtype=torch.int32)))
    assert built["cmp_ratio"] == 1
    assert built["cmp_kv"] is not None
    assert built["cmp_residual_kv"].ndim == 1


def test_claiming_a_compressed_stream_without_one_is_refused():
    """The configuration the operator would happily accept and get wrong.

    mask mode 0 with the bounds disabled, but nothing compressed to attend --
    the layer then sees only its window, unwindowed. Nothing downstream would
    report that.
    """
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        build_sparse_flash_mla_kwargs

    with pytest.raises(ValueError, match="no reason to reject"):
        build_sparse_flash_mla_kwargs(
            **_common(cmp_ratio=2, attends_compressed=True))


def test_every_assembled_call_is_validated():
    """The builder runs the checks itself, so a caller cannot skip them."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        build_sparse_flash_mla_kwargs

    with pytest.raises(ValueError, match="sparse_flash_mla would reject"):
        build_sparse_flash_mla_kwargs(
            **_common(cmp_ratio=0, attends_compressed=False))


def test_a_paged_layout_gets_the_head_axis_back():
    """bind_value drops it; aclnnSparseFlashMla reads the stride vector against
    layout_kv and rejects a three-axis ori_kv under PA_BBND by name."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        kv_with_layout_rank

    cache = torch.zeros(4, 32, 512)
    got = kv_with_layout_rank(cache, "PA_BBND")
    assert got.shape == (4, 32, 1, 512)
    assert got.shape[-2] == 1, "the restored axis is num_kv_heads"


def test_tnd_is_left_at_three_axes():
    """The case an unconditional unsqueeze gets wrong.

    TND describes three axes. A fourth would be accepted -- the operator checks
    the stride vector's length, not what the axes mean -- and the plan would
    run on a tensor whose axes no longer match its layout.
    """
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        kv_with_layout_rank

    cache = torch.zeros(4, 32, 512)
    assert kv_with_layout_rank(cache, "TND") is cache


def test_an_already_correct_rank_is_untouched():
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        kv_with_layout_rank

    cache = torch.zeros(4, 32, 1, 512)
    assert kv_with_layout_rank(cache, "BSND") is cache


def test_an_unknown_layout_is_refused_rather_than_passed_through():
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        kv_with_layout_rank

    with pytest.raises(ValueError, match="no KV rank recorded"):
        kv_with_layout_rank(torch.zeros(4, 32, 512), "PA_ND")


def test_a_rank_that_is_off_by_more_than_the_head_axis_is_refused():
    """Only the one axis bind_value drops is restored. Anything else means the
    cache is not the layout this plan selected, and reshaping it would produce
    the right stride length over the wrong axes."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        kv_with_layout_rank

    with pytest.raises(ValueError, match="not the layout this plan selected"):
        kv_with_layout_rank(torch.zeros(4, 512), "PA_BBND")


def test_every_layout_the_plans_use_has_a_rank():
    """The table is derived from the plans, not from the layouts seen so far.

    A plan selecting a layout with no entry would otherwise reach the operator
    with the cache unchanged.
    """
    import inspect

    from vllm_ascend.attention import dsa_attn_kv_plan
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import (
        LAYOUT_KV_RANK, LAYOUTS_KV)

    source = inspect.getsource(dsa_attn_kv_plan)
    used = {m for m in LAYOUTS_KV if f'"{m}"' in source}
    assert used, "no layout literals found; the plans moved"
    missing = used - set(LAYOUT_KV_RANK)
    assert not missing, f"plans select {sorted(missing)} with no rank recorded"
    # And no invented names: every entry must be a layout the operator accepts.
    unknown = set(LAYOUT_KV_RANK) - set(LAYOUTS_KV)
    assert not unknown, (
        f"{sorted(unknown)} is not in LAYOUTS_KV; a plausible-looking spelling "
        "in this table silently never matches")
