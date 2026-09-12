# SPDX-License-Identifier: Apache-2.0
"""Shape, dtype and stride can all be right while the layout is wrong.

The indexer K cache packs a block as *segments*: every token's values first,
then every token's scales. The obvious alternative -- each token owning
``cache_width`` contiguous bytes -- produces tensors of the same shape, the
same dtype and plausible strides, and returns other tokens' numbers.

So the test that matters writes distinguishable marks into known places and
reads them back through the views. A shape assertion cannot tell the two
readings apart; a round trip can.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import (
    SCALE_BYTES, SCALE_GROUP, split_indexer_cache, value_width)

BLOCKS, TOKENS = 3, 8
D = SCALE_GROUP                      # values per token
WIDTH = D + SCALE_BYTES              # one scale column


def _cache():
    return torch.zeros(BLOCKS, TOKENS, WIDTH, dtype=torch.uint8)


# ------------------------------------------------------------- the width rule

def test_the_value_width_is_inverted_from_upstreams_own_formula():
    """attention.py:90 -- head_dim = index_head_dim + index_head_dim//128*4."""
    assert value_width(WIDTH) == D
    assert value_width(2 * SCALE_GROUP + 2 * SCALE_BYTES) == 2 * SCALE_GROUP


def test_a_width_that_is_not_values_plus_scales_is_refused():
    with pytest.raises(ValueError, match="between a value and its scale"):
        value_width(WIDTH + 1)


def test_the_formula_is_not_written_as_todays_numbers():
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import indexer_cache_view

    body = inspect.getsource(indexer_cache_view.value_width)
    assert "132" not in body and "128" not in body, body


# ------------------------------------------------- the layout, by round trip

def test_each_token_reads_back_its_own_values_and_its_own_scale():
    """🔴 The test that separates the two readings.

    Marks are written at the byte offsets the operator's kernel computes:

        value of token t  ->  block*blockStride + t*d
        scale of token t  ->  block*blockStride + tokens*d + t*4

    Under the per-token reading ("cache_width contiguous bytes each") token 1
    would come back holding token 1's tail and token 2's head, and the scale
    would be a slice of someone's values. Both would be well-formed.
    """
    cache = _cache()
    flat = cache.view(-1)
    block_stride = TOKENS * WIDTH

    scales = torch.arange(1, BLOCKS * TOKENS + 1, dtype=torch.float32) * 0.5
    for b in range(BLOCKS):
        for t in range(TOKENS):
            mark = b * TOKENS + t + 1
            flat[b * block_stride + t * D:b * block_stride + (t + 1) * D] = mark
            start = b * block_stride + TOKENS * D + t * SCALE_BYTES
            flat[start:start + SCALE_BYTES] = torch.tensor(
                [scales[b * TOKENS + t]], dtype=torch.float32).view(
                    torch.uint8)

    key, scale = split_indexer_cache(cache)
    assert key.shape == (BLOCKS, TOKENS, 1, D)
    assert scale.shape == (BLOCKS, TOKENS, 1, 1)

    for b in range(BLOCKS):
        for t in range(TOKENS):
            mark = b * TOKENS + t + 1
            assert torch.all(key[b, t, 0] == mark), (
                f"block {b} token {t} read back {key[b, t, 0, :4].tolist()} "
                f"instead of {mark}; the value segment is being addressed as "
                "if each token owned cache_width contiguous bytes")
            assert float(scale[b, t, 0, 0]) == pytest.approx(
                float(scales[b * TOKENS + t]))


def test_the_per_token_reading_would_have_disagreed():
    """The wrong reading, run explicitly, so "this test could not fail" is
    not a thing anyone has to take on trust."""
    cache = _cache()
    flat = cache.view(-1)
    block_stride = TOKENS * WIDTH
    for t in range(TOKENS):
        flat[t * D:(t + 1) * D] = t + 1

    key, _ = split_indexer_cache(cache)
    wrong = cache[:, :, :D]          # "each token owns WIDTH bytes"
    assert not torch.equal(key[0, :, 0, :], wrong[0]), (
        "the segmented and per-token readings agree on this fixture, so the "
        "round trip above is not distinguishing them")
    del block_stride


# ------------------------------------------------------------- identity, not shape

def test_both_views_share_the_caches_storage():
    cache = _cache()
    key, scale = split_indexer_cache(cache)
    ptr = cache.untyped_storage().data_ptr()
    assert key.untyped_storage().data_ptr() == ptr
    assert scale.untyped_storage().data_ptr() == ptr


def test_the_scale_segment_starts_one_whole_value_segment_in():
    """tokens_per_block * d, not d.

    The second is exactly what a per-token reading produces, so this is the
    assertion that tells the two apart at the address level.
    """
    cache = _cache()
    key, scale = split_indexer_cache(cache)
    offset_bytes = scale.storage_offset() * SCALE_BYTES - \
        key.storage_offset() * key.element_size()
    assert offset_bytes == TOKENS * D
    assert offset_bytes != D, "this is the per-token reading's offset"


def test_writing_through_a_view_reaches_the_cache():
    """Views, not copies -- stated as an observable rather than as intent."""
    cache = _cache()
    key, _ = split_indexer_cache(cache)
    key[0, 0, 0, 0] = 7
    assert int(cache[0, 0, 0]) == 7


def test_a_rank_it_cannot_place_the_boundary_in_is_refused():
    with pytest.raises(ValueError, match="cannot place the value/scale"):
        split_indexer_cache(torch.zeros(4, WIDTH, dtype=torch.uint8))


def test_the_unit_head_axis_is_accepted_if_binding_left_it_on():
    cache = torch.zeros(BLOCKS, TOKENS, 1, WIDTH, dtype=torch.uint8)
    key, scale = split_indexer_cache(cache)
    assert key.shape == (BLOCKS, TOKENS, 1, D)
    assert scale.shape == (BLOCKS, TOKENS, 1, 1)
