# SPDX-License-Identifier: Apache-2.0
"""What a V4.1 layer's KV cache must look like before upstream reads it.

Neither of the two shapes already in the tree is right. The Ascend runner binds
a list for the whole V4 family, and upstream's own bind_kv_cache squeezes axis
1. Upstream's V4.1 consumers call .numel() on the value and assert
shape[1:] == (block_size, head_dim), so the list fails loudly and the squeeze
fails quietly -- axis 1 is block_size here, not the head axis, and squeezing a
non-unit dimension does nothing at all.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.kv_binding import (HEAD_AXIS, bind_value,
                                                         is_v41)

BLOCKS, BLOCK, HEADS, HEAD_DIM = 4, 32, 1, 512


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type


@pytest.mark.parametrize("model_type,expected", [
    ("deepseek_v41", True),
    ("deepseek_v41_text", True),
    ("deepseek_v4", False),
    ("deepseek_v4_text", False),
    ("", False),
])
def test_only_v41_changes_shape(model_type, expected):
    """V4.0's consumers read the list, and the indexer indexes into it."""
    assert is_v41(_Config(model_type)) is expected


def test_a_single_tensor_loses_its_unit_head_axis():
    cache = torch.zeros(BLOCKS, BLOCK, HEADS, HEAD_DIM)
    bound = bind_value(cache)
    assert isinstance(bound, torch.Tensor)
    assert bound.shape == (BLOCKS, BLOCK, HEAD_DIM), (
        "upstream asserts shape[1:] == (block_size, head_dim)")
    assert bound.shape[1:] == (BLOCK, HEAD_DIM)


def test_upstreams_own_squeeze_would_not_have_worked():
    """squeeze(1) is what upstream's bind_kv_cache does, commented
    '[B, H=1, N, C] -> [B, N, C]'. Here axis 1 is block_size."""
    cache = torch.zeros(BLOCKS, BLOCK, HEADS, HEAD_DIM)
    assert cache.squeeze(1).shape == cache.shape, (
        "a no-op, because block_size is not 1 -- this is why it fails quietly")
    assert bind_value(cache).shape != cache.shape


def test_the_bound_tensor_answers_numel():
    """engram.py calls .numel() on it directly; a list has no such method."""
    bound = bind_value(torch.zeros(BLOCKS, BLOCK, HEADS, HEAD_DIM))
    assert bound.numel() == BLOCKS * BLOCK * HEAD_DIM
    with pytest.raises(AttributeError):
        [torch.zeros(1)].numel()


def test_a_multi_tensor_cache_keeps_its_slots_addressable():
    """The indexer's cache is two or three tensors read by fixed slot index.

    This used to assert the opposite -- that the list came back wrapped in
    another list -- which is what the implementation did and what made the
    indexer's slot 0 return the whole pair. The test agreed with the code
    because it was written from the code.
    """
    slots = [torch.zeros(BLOCKS, BLOCK, HEADS, 128),
             torch.zeros(BLOCKS, BLOCK, HEADS)]
    bound = bind_value(slots)
    assert len(bound) == 2
    assert bound[0] is slots[0] and bound[1] is slots[1]


def test_a_multi_head_cache_is_handed_over_unreshaped():
    """Not the layout this was written for -- so nothing is dropped from it.

    V4.1's caches all carry num_kv_heads=1. A cache with eight would be a
    different model, and squeezing some other axis to make it fit would give
    the consumer a tensor of the right rank and the wrong contents. It comes
    back whole, and the consumer's own assert on shape says so.
    """
    cache = torch.zeros(BLOCKS, BLOCK, 8, HEAD_DIM)     # 8 kv heads
    bound = bind_value(cache)
    assert bound is cache


def test_the_head_axis_is_the_one_the_backend_allocates():
    """Pins the axis against the backend's own shape function, so a layout
    change there fails here rather than silently squeezing the wrong axis."""
    from vllm_ascend.attention.dsa_v1 import AscendDSABackend

    shape = AscendDSABackend.get_kv_cache_shape(BLOCKS, BLOCK, HEADS, HEAD_DIM)
    assert shape[HEAD_AXIS] == HEADS
    assert len(shape) == 4 and shape == (BLOCKS, BLOCK, HEADS, HEAD_DIM)


def test_the_compressor_ring_binds_to_three_dims_not_a_list():
    """The ring is a V4.1 cache like the others, and its consumer is upstream's.

    ``DeepseekCompressor`` reads ``self.state_cache.kv_cache`` and hands it to
    the compressor kernel expecting [num_blocks, ring_capacity, state_dim]. A
    list there is not a shape error, it is a type error, one frame further in.

    This is the axis-order question in its sharpest form: with num_kv_heads=1,
    (B, N, H, C) and (B, H, N, C) occupy identical bytes with identical
    strides, so no assertion on shape can tell them apart. Only how many
    dimensions survive the squeeze can. Ascend allocates head-at-2, and this
    pins that the ring is not made an exception to it.
    """
    ring = torch.zeros((BLOCKS, 8, 1, 512))
    bound = bind_value(ring)
    assert isinstance(bound, torch.Tensor), (
        "a list reaches the compressor kernel as a list")
    assert bound.shape == (BLOCKS, 8, 512)


def test_a_head_first_ring_is_not_silently_reshaped():
    """If the declared shape ever moves the head axis to 1, do not guess.

    bind_value drops an axis only where the head axis is; it does not hunt for
    whichever axis happens to be 1. A head-first ring therefore comes back
    whole, and the mismatch surfaces at the consumer rather than as a plausible
    tensor with the ring capacity silently removed.
    """
    head_first = torch.zeros((BLOCKS, 1, 8, 512))
    bound = bind_value(head_first)
    assert isinstance(bound, torch.Tensor)
    assert bound.shape == (BLOCKS, 1, 8, 512)


def test_a_tensor_never_comes_back_as_a_list():
    """`'list' object has no attribute 'numel'`, from decode graph capture.

    engram.ensure_cache reads self.swa_cache_module.kv_cache and calls .numel()
    on it. The reshape path can hand over a three-dimensional tensor that has
    no head axis left to drop, and the old "wrap anything unrecognised" default
    turned that into a list -- a type error several frames away, in upstream
    code that never asked for a list. Wrapping was never the conservative
    choice; it was V4.0's convention applied where V4.0's consumers are not.
    """
    for shape in ((BLOCKS, BLOCK, 512), (BLOCKS, BLOCK, HEADS, HEAD_DIM),
                  (BLOCKS, 1, 8, 512), (BLOCKS, 8)):
        bound = bind_value(torch.zeros(shape))
        assert isinstance(bound, torch.Tensor), f"{shape} came back wrapped"
        assert bound.numel() > 0


def test_a_multi_tensor_cache_is_handed_over_at_the_same_depth():
    """The indexer's (values, scales) pair is addressed by fixed slot.

    The earlier version returned [kv_cache] here, wrapping an existing pair in
    another list and moving every slot index down a level -- so cache[0] gave
    the whole pair instead of the values. That half never raised: the indexer
    reads slot 0 and got something, and the SWA failure above it fired first.
    An error that is only reachable after another error is fixed is not a
    latent risk, it is a scheduled one.
    """
    values, scales = torch.zeros(2, 4), torch.ones(2, 4)
    bound = bind_value((values, scales))
    assert len(bound) == 2, "a pair must not come back as a one-element list"
    assert bound[0] is values and bound[1] is scales


def test_a_one_element_list_is_not_several_tensors():
    """_adjust_kv_layout always returns a list, so SWA arrives wrapped.

    The reshape branch builds its shapes as a list because some specs need two
    -- a scale beside the values, a nope/rope split -- and hands back a list of
    one when they do not. Unwrapping here is what makes the consumer's
    kv_cache.numel() work.
    """
    cache = torch.zeros(BLOCKS, BLOCK, 1, HEAD_DIM)
    bound = bind_value([cache])
    assert isinstance(bound, torch.Tensor)
    assert bound.shape == (BLOCKS, BLOCK, HEAD_DIM)


def test_nothing_is_ever_wrapped_one_level_deeper():
    """The invariant behind both halves: this function never adds a level.

    It may remove one (a single-element list, a unit head axis); it must never
    add one, because every consumer addresses what it is given by position or
    by attribute, and an extra level silently answers both.
    """
    for value in (torch.zeros(2, 2), [torch.zeros(2, 2)],
                  [torch.zeros(2, 2), torch.zeros(2, 2)],
                  (torch.zeros(2, 2), torch.zeros(2, 2)), []):
        bound = bind_value(value)
        if isinstance(bound, torch.Tensor):
            continue
        assert len(bound) == len(value), (
            f"{type(value).__name__} of {len(value)} came back as {len(bound)}")
