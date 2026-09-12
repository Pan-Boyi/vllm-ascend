# SPDX-License-Identifier: Apache-2.0
"""Which widening of cos/sin is right, made decidable.

Two widenings give the operator the width it asks for, and nothing downstream
distinguishes them. The first test's job is to show the choice *matters* --
that A and B differ -- because a check written for a choice that turns out not
to matter is a check that can never fail for the reason it was written.

The second pins which one, against the Ascend path that already drives this
operator rather than against a reading of what interleaving ought to mean.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.rope_widen import (cos_sin_for_slice,
                                                         widen_interleaved)

ROPE_DIM = 64
HALF = ROPE_DIM // 2


def _cache(max_pos=8):
    """Upstream's layout: half a rotation of cosines, then its sines."""
    torch.manual_seed(3)
    angles = torch.rand(max_pos, HALF) * 3.0
    return torch.cat([angles.cos(), angles.sin()], dim=-1), angles


def test_the_two_widenings_actually_differ():
    """Otherwise the rest of this file is checking a distinction without one."""
    half = torch.arange(HALF, dtype=torch.float32)
    a = widen_interleaved(half)
    b = half.repeat(2)
    assert a.shape == b.shape == (ROPE_DIM,)
    assert not torch.equal(a, b), (
        "if these agreed, the widening would not be a decision and this "
        "file would be pinning nothing")


def test_channel_i_carries_its_own_pairs_value():
    """Channel i belongs to pair i // 2. That is what makes it interleave."""
    half = torch.arange(HALF, dtype=torch.float32)
    wide = widen_interleaved(half)
    for i in range(ROPE_DIM):
        assert wide[i] == half[i // 2]


def test_it_matches_what_the_working_ascend_path_builds():
    """ops/rope_dsv4.py:218 is freqs.cos().repeat_interleave(2, dim=-1).

    That table is what V4.0 already hands to inplace_partial_rotary_mul, so it
    is the consumer's own convention rather than an interpretation of it. If
    that line changes, this should be revisited -- so it is read, not recalled.
    """
    import inspect

    from vllm_ascend.ops import rope_dsv4

    source = inspect.getsource(rope_dsv4.ComplexExpRotaryEmbedding.__init__)
    assert "repeat_interleave(2, dim=-1)" in source, (
        "the working Ascend rope path no longer widens with repeat_interleave; "
        "the convention this file copies has moved")


def test_the_halves_are_split_where_the_cache_puts_them():
    cache, angles = _cache()
    positions = torch.tensor([0, 3, 7])
    cos, sin = cos_sin_for_slice(cache, positions, ROPE_DIM)
    assert cos.shape == sin.shape == (3, ROPE_DIM)
    for row, pos in enumerate(positions.tolist()):
        for i in range(ROPE_DIM):
            assert cos[row, i] == pytest.approx(float(angles[pos, i // 2].cos()))
            assert sin[row, i] == pytest.approx(float(angles[pos, i // 2].sin()))


def test_a_rotation_built_this_way_matches_a_per_pair_reference():
    """End to end against the arithmetic, not against another widening.

    The reference indexes cos[k] and sin[k] for pair k directly and never
    widens anything, so agreement means the widened form feeds each pair the
    angle that pair is supposed to get.
    """
    cache, angles = _cache()
    positions = torch.tensor([2, 5])
    x = torch.randn(2, ROPE_DIM)
    cos, sin = cos_sin_for_slice(cache, positions, ROPE_DIM)

    pairs = x.view(2, HALF, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    c = angles[positions].cos()
    s = angles[positions].sin()
    expect = torch.stack((even * c - odd * s, odd * c + even * s),
                         dim=-1).view(2, ROPE_DIM)

    wide_pairs = torch.stack((cos.view(2, HALF, 2)[..., 0],
                              sin.view(2, HALF, 2)[..., 0]), dim=-1)
    got = torch.stack((even * wide_pairs[..., 0] - odd * wide_pairs[..., 1],
                       odd * wide_pairs[..., 0] + even * wide_pairs[..., 1]),
                      dim=-1).view(2, ROPE_DIM)
    torch.testing.assert_close(got, expect)


def test_the_wrong_widening_produces_a_different_rotation():
    """The failure this prevents, made visible.

    B is wrong on every other pair -- regular enough to look like a plausible
    model rather than a bug, which is why the criterion is here and not in a
    tolerance downstream.
    """
    cache, angles = _cache()
    positions = torch.tensor([4])
    cos_a, _ = cos_sin_for_slice(cache, positions, ROPE_DIM)
    cos_b = cache[positions][..., :HALF].repeat(1, 2)
    assert not torch.allclose(cos_a, cos_b)
    differing = int((cos_a != cos_b).sum())
    assert differing > ROPE_DIM // 4, (
        f"only {differing} of {ROPE_DIM} channels differ; if the wrong choice "
        "were nearly right it would be far harder to catch downstream")


def test_a_cache_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="pair.*count"):
        cos_sin_for_slice(torch.zeros(8, HALF), torch.tensor([0]), ROPE_DIM)
