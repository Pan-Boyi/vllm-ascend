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


def test_a_multi_tensor_cache_keeps_its_list():
    """The indexer's cache is two or three tensors read by fixed slot index;
    flattening it would turn a slot index into an element index."""
    slots = [torch.zeros(BLOCKS, BLOCK, HEADS, 128),
             torch.zeros(BLOCKS, BLOCK, HEADS)]
    assert bind_value(slots) is not slots or isinstance(bind_value(slots), list)
    bound = bind_value(slots)
    assert isinstance(bound, list) and len(bound) == 1 and bound[0] is slots


def test_an_unrecognised_head_axis_is_left_alone():
    """Not the layout this was written for, so it keeps the old behaviour
    rather than reshaping something it does not recognise."""
    cache = torch.zeros(BLOCKS, BLOCK, 8, HEAD_DIM)     # 8 kv heads
    bound = bind_value(cache)
    assert isinstance(bound, list) and bound[0] is cache


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


def test_a_head_first_ring_is_not_silently_flattened():
    """If the declared shape ever moves the head axis to 1, say so here.

    bind_value refuses to guess: an unrecognised layout keeps the old wrapping
    rather than squeezing whichever axis happens to be 1. That makes the
    mismatch visible at the consumer instead of producing a plausible tensor
    with the ring capacity silently removed.
    """
    head_first = torch.zeros((BLOCKS, 1, 8, 512))
    bound = bind_value(head_first)
    assert not isinstance(bound, torch.Tensor)
    assert bound[0].shape == (BLOCKS, 1, 8, 512)
