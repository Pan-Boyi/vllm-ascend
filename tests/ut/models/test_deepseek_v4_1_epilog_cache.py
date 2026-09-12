# SPDX-License-Identifier: Apache-2.0
"""The rank the operator wants, and the width nobody compares.

``indexer_compress_epilog_v2`` reads its cache as
``[num_blocks, tokens_per_block, 1, bytes_per_token]``. It checks the rank and
the third dim itself. It does not check dtype at all, and it takes its element
width from **x**, never comparing it with the cache -- so a width
disagreement is written, not refused.

That is the same shape as the SWA ``block_size`` guard: two quantities that
index each other with nobody comparing them. The guard exists for the shape,
not because a particular defect was foreseen.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.indexer_k_store import (
    INDEX_HEAD_DIM, SCALE_BYTES, SCALE_GROUP, check_epilog_cache,
    with_head_axis)
from vllm_ascend.models.deepseek_v4_1.kv_binding import (HEAD_AXIS,
                                                         _drop_head_axis)

BLOCKS, TOKENS_PER_BLOCK, WIDTH = 4, 32, INDEX_HEAD_DIM + 4


def _cache(blocks=BLOCKS, tokens=TOKENS_PER_BLOCK, heads=1, width=WIDTH):
    return torch.zeros(blocks, tokens, heads, width, dtype=torch.uint8)


def _x(rows=8, width=INDEX_HEAD_DIM, dtype=torch.bfloat16):
    return torch.zeros(rows, width, dtype=dtype)


# ------------------------------------------------------------------ the rank

def test_the_bound_form_is_rank_three_and_is_restored_to_four():
    """The exact round trip: what binding drops, this puts back."""
    bound = _drop_head_axis(_cache())
    assert bound.ndim == 3
    restored = with_head_axis(bound)
    assert restored.ndim == 4
    assert restored.shape == _cache().shape


def test_the_axis_restored_is_the_one_the_operator_constrains():
    """Two independent sources landing on one position.

    Binding drops HEAD_AXIS; the operator requires dim2 == 1. In a rank-4
    tensor those are the same axis, so restoring where it was dropped needs
    no assumption about the axis order -- which is as well, since V4.0's
    warmup dummy (1, 1, 1, D) cannot establish one.
    """
    restored = with_head_axis(_drop_head_axis(_cache()))
    assert restored.shape[2] == 1
    assert restored.ndim + HEAD_AXIS == 2


def test_a_rank_four_cache_is_passed_through_unchanged():
    cache = _cache()
    assert with_head_axis(cache) is cache


def test_a_rank_that_cannot_be_restored_is_refused_not_reshaped():
    with pytest.raises(ValueError, match="inventing an axis order"):
        with_head_axis(torch.zeros(4, 32, dtype=torch.uint8))


def test_view_uint8_does_not_change_the_rank():
    """The premise of the whole diagnosis: the adapter's only transform was a
    dtype view, so the rank at the call is the rank at binding."""
    bf16 = torch.zeros(BLOCKS, TOKENS_PER_BLOCK, WIDTH // 2,
                       dtype=torch.bfloat16)
    assert bf16.view(torch.uint8).ndim == bf16.ndim


# ----------------------------------------------------------------- the guard

def test_the_expected_shape_passes():
    check_epilog_cache(_cache(), _x())


def test_rank_three_is_refused_with_the_operators_own_wording():
    with pytest.raises(ValueError, match="EZ1008"):
        check_epilog_cache(_drop_head_axis(_cache()), _x())


def test_a_third_dim_above_one_is_refused():
    """The operator states this one, and we have not hit it yet."""
    with pytest.raises(ValueError, match="dim2 must be 1"):
        check_epilog_cache(_cache(heads=2), _x())


@pytest.mark.parametrize("width", [INDEX_HEAD_DIM, INDEX_HEAD_DIM + 3,
                                   INDEX_HEAD_DIM + 5, 2 * INDEX_HEAD_DIM])
def test_any_width_but_the_operators_own_is_refused(width):
    """An equality, not a bound.

    The measured pair is 132 against 128, and 132 is also what several wrong
    rules produce -- "x + 4", "round up to 4", "x plus a header". A plausible
    coincidence and a correct contract look identical while nothing checks
    them, so the rule here is the operator's own: cacheD_ = d_ + 4 *
    ceil(d_ / 128), out of its scaleCol_.
    """
    with pytest.raises(ValueError, match="bytes per token"):
        check_epilog_cache(_cache(width=width), _x())


def test_the_rule_follows_d_rather_than_pinning_todays_numbers():
    """A wider x needs a second scale column; 132 must not survive as a
    constant. This is what a literal 132 or 128 would fail to express."""
    wide = 2 * SCALE_GROUP
    check_epilog_cache(_cache(width=wide + 2 * SCALE_BYTES), _x(width=wide))
    with pytest.raises(ValueError, match="bytes per token"):
        check_epilog_cache(_cache(width=wide + SCALE_BYTES), _x(width=wide))


def test_the_expected_width_is_not_written_as_a_literal():
    """Derived from the source: 132 and 128 as bare literals in the formula
    would be a snapshot that goes stale silently when head_dim moves."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import indexer_k_store

    body = inspect.getsource(indexer_k_store.check_epilog_cache)
    formula = [ln for ln in body.splitlines() if "expected =" in ln]
    assert formula, "the width equation is gone"
    assert "132" not in formula[0] and "128" not in formula[0], formula


def test_an_unregistered_dtype_is_refused_because_the_operator_will_not():
    with pytest.raises(ValueError, match="validates no dtype"):
        check_epilog_cache(_cache(), _x(dtype=torch.float32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_every_registered_dtype_is_accepted(dtype):
    """The assertion is the operator's registered set, not this call site's
    narrower choice of bf16. A guard tightened to our current preference
    would refuse a call the operator accepts and read as "the operator
    rejects fp16", which is false."""
    check_epilog_cache(_cache(), _x(dtype=dtype))


def test_overlapping_blocks_are_refused():
    """stride(0) below one block means the next block starts inside this one."""
    cache = _cache()
    narrow = cache.as_strided(cache.shape,
                              (TOKENS_PER_BLOCK * WIDTH // 2,
                               WIDTH, WIDTH, 1))
    with pytest.raises(ValueError, match="blocks overlap"):
        check_epilog_cache(narrow, _x())


def test_the_guard_reads_the_tensor_that_is_passed_to_the_operator():
    """Derived from the source: the guard must see the post-view, post-restore
    tensor, not the one the caller happened to hold. Checking a different
    object is how a correct guard passes at the moment it should fire."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import indexer_k_store

    seam = inspect.getsource(indexer_k_store._checked_cache)
    assert "with_head_axis(k_cache.view(torch.uint8))" in seam
    assert "check_epilog_cache(cache, x)" in seam

    call = inspect.getsource(
        indexer_k_store.indexer_k_norm_rope_store_ascend)
    assert "indexer_compress_cache=_checked_cache(k_cache, x)" in call, (
        "the operator is being handed a cache the guard never saw")
    # The x the guard sees and the x the operator gets must be one object.
    # They were not: the guard read `k` (float32) while the call passed
    # `k.to(torch.bfloat16)`, so the dtype check would have read the wrong
    # tensor and passed at the moment it should fire.
    assert "x = k.to(torch.bfloat16)" in call
    assert "x=x," in call
    assert "x=k.to(" not in call
