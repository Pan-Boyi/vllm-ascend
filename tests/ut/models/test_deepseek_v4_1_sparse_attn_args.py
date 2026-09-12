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


def _descriptor():
    """A populated 1024-word descriptor.

    Not ``torch.zeros``: an all-zero blob is what a buffer the metadata
    operator never wrote into looks like, and the validator refuses it with
    a query batch present. Contents are not inspected -- and cannot be: a
    well-formed descriptor built for another configuration is exactly the
    failure this port hit, and it passes every check there is.
    """
    blob = torch.zeros(1024, dtype=torch.int32)
    blob[0] = 28  # one core-split boundary
    return blob


def _ori_only(**overrides):
    """A layer attending only its own window stream (compress_ratio 0)."""
    kwargs = dict(
        metadata=_descriptor(),
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
        metadata=_descriptor(),
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


def test_the_mask_mode_rule_records_how_its_signal_was_found(q):
    """This rule was declared unmodelled, and then stopped being unmodelled.

    Leaving it out was a decision: telling ori-only *sparse* mode from plain
    sliding-window mode needed a signal nothing measured provided, and a
    guessed rule would have rejected valid calls and accepted invalid ones with
    equal confidence. The operator then supplied the signal by refusing (4, 0)
    for a call carrying ori_sparse_indices.

    What this pins is that the docstring says which of the two it is, and on
    what basis. A rule that arrives without its provenance is indistinguishable
    from the guess that was deliberately not made.
    """
    from vllm_ascend.models.deepseek_v4_1 import sparse_attn_args

    doc = " ".join(sparse_attn_args.__doc__.split())
    assert "Ori-only sparse mode requires both mask modes to be 0" in doc
    assert "from a measurement rather than a reading" in doc

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


def _descriptor():
    """A populated 1024-word descriptor.

    Not ``torch.zeros``: an all-zero blob is what a buffer the metadata
    operator never wrote into looks like, and the validator refuses it. Not
    ``object()`` either -- that makes the descriptor check skip, which these
    tests would then not be exercising.
    """
    blob = torch.zeros(1024, dtype=torch.int32)
    blob[0] = 28  # one core split boundary; contents are not inspected
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


def _built(**overrides):
    """build_sparse_flash_mla_kwargs over a minimal valid call."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        build_sparse_flash_mla_kwargs

    args = dict(
        q=torch.zeros(12, NUM_HEADS, 512, dtype=torch.bfloat16),
        num_requests=3,
        swa_kv_cache=torch.zeros(2, 32, 1, 512, dtype=torch.bfloat16),
        swa_block_table=torch.zeros((3, 4), dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 4, 9, 12], dtype=torch.int32),
        seqused_kv=torch.ones(3, dtype=torch.int32),
        sinks=torch.zeros(NUM_HEADS, dtype=torch.float32),
        metadata=_descriptor(),
        softmax_scale=0.04,
        cmp_ratio=1,
        layout_kv="PA_BBND",
        attends_compressed=False,
        ori_win_left=127,
        ori_win_right=0,
    )
    args.update(overrides)
    return build_sparse_flash_mla_kwargs(**args)


def test_explicit_indices_select_mode_zero_not_sliding_window():
    """The operator refused (4, 0) on the first real forward.

    "Ori-only sparse mode requires both mask modes to be 0", and what puts it
    in that mode is the explicit index list. This was recorded as unmodelled
    until the operator said which signal distinguishes it.
    """
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import (
        NO_MASK_MODE, build_sparse_flash_mla_kwargs)

    # The length travels with the indices now: without it the operator takes
    # the whole row as valid and gathers at the -1 padding. Supplying it here
    # keeps this test about the mask mode rather than about that rule.
    kwargs = _built(attends_compressed=False,
                    ori_sparse_indices=torch.zeros(3, 8, dtype=torch.int32),
                    ori_topk_length=torch.full((3, ), 8, dtype=torch.int32))
    assert kwargs["ori_mask_mode"] == NO_MASK_MODE
    assert kwargs["ori_win_left"] == -1 and kwargs["ori_win_right"] == -1


def test_window_bounds_without_indices_stay_sliding_window():
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import (
        SLIDING_WINDOW_MASK_MODE, build_sparse_flash_mla_kwargs)

    kwargs = _built(attends_compressed=False, ori_sparse_indices=None,
                    ori_win_left=127, ori_win_right=0)
    assert kwargs["ori_mask_mode"] == SLIDING_WINDOW_MASK_MODE
    assert kwargs["ori_win_left"] == 127


def test_a_compressed_stream_is_mode_zero_too_but_for_another_reason():
    """Two configurations share a value and not a cause; naming the constant
    after either one is how the first version went wrong."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import NO_MASK_MODE

    kwargs = _built(attends_compressed=True, ori_sparse_indices=None,
                    cmp_kv=torch.zeros(2, 32, 1, 512, dtype=torch.bfloat16),
                    cmp_block_table=torch.zeros((3, 4), dtype=torch.int32),
                    seqused_cmp_kv=torch.ones(3, dtype=torch.int32),
                    cmp_ratio=2)
    assert kwargs["ori_mask_mode"] == NO_MASK_MODE
    assert kwargs["ori_win_left"] == -1


def _tables(num_blocks=8, block=128, max_id=3, longest=256):
    return dict(
        ori_kv=torch.zeros(num_blocks, block, 1, 512, dtype=torch.bfloat16),
        ori_block_table=torch.tensor([[0, max_id]], dtype=torch.int32),
        seqused_kv=torch.tensor([longest], dtype=torch.int32))


def test_a_block_id_past_the_cache_is_refused_with_both_numbers():
    """The table and the cache arrive from unrelated objects and nothing pairs
    them. The operator does not range-check, and every core reads through the
    same table, so one wrong table faults all of them in the same place."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        _check_block_table_addresses_this_cache

    with pytest.raises(ValueError) as exc:
        _check_block_table_addresses_this_cache(_tables(num_blocks=4, max_id=9))
    msg = str(exc.value)
    assert "9" in msg and "4 blocks" in msg
    assert "another cache group" in msg


def test_a_length_past_the_capacity_is_refused():
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        _check_block_table_addresses_this_cache

    with pytest.raises(ValueError, match="longest sequence"):
        _check_block_table_addresses_this_cache(
            _tables(num_blocks=2, block=128, max_id=1, longest=257))


def test_a_matching_pair_passes():
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        _check_block_table_addresses_this_cache

    assert _check_block_table_addresses_this_cache(_tables()) is None


def test_an_empty_table_fails_as_unchecked_not_as_a_pass():
    """Zero coverage and a clean run look identical from the outside."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        _check_block_table_addresses_this_cache

    kwargs = _tables()
    kwargs["ori_block_table"] = torch.zeros((0, 2), dtype=torch.int32)
    with pytest.raises(ValueError, match="did not run"):
        _check_block_table_addresses_this_cache(kwargs)


def test_missing_lengths_fail_as_unchecked_too():
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        _check_block_table_addresses_this_cache

    kwargs = _tables()
    del kwargs["seqused_kv"]
    with pytest.raises(ValueError, match="capacity check did"):
        _check_block_table_addresses_this_cache(kwargs)


def test_the_bounds_come_from_the_tensor_not_a_configured_value():
    """A configured num_blocks is true of both groups, which is the confusion
    this exists to catch."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import sparse_attn_args

    body = inspect.getsource(
        sparse_attn_args._check_block_table_addresses_this_cache)
    assert "int(cache.shape[0])" in body and "int(cache.shape[1])" in body
