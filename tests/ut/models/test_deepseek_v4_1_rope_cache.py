# SPDX-License-Identifier: Apache-2.0
"""The table-driven inverse rotation, cross-checked against the complex one.

Two independently derived implementations of the same thing: the complex form
in test_deepseek_v4_1_rope.py, checked bit-exact against the CPU golden dumps,
and a transliteration of the table-driven kernel that upstream's CUDA and ROCm
output projections both fuse. They are algebraically the same --
``(e + io)(cos - i sin)`` expands to the kernel's
``(e cos + o sin, o cos - e sin)`` -- so agreement checks the transcription
rather than restating it.

What the table-driven form buys: the output projection can be handed the rotary
module the attention layer already built, rather than a second table built from
a second copy of the per-layer parameters, which could drift from it.

Deliberately in its own file: the golden-dump tests carry a module-level skip
for when the dumps are absent, and nothing here needs them. Sharing the file
would have made these disappear on any machine without the dumps, while still
reporting as a pass.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import rope

ROPE_DIM = rope.ROPE_HEAD_DIM
NOPE_DIM = rope.NOPE_HEAD_DIM


def _cache_from_freqs(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Upstream's [max_pos, rope_dim]: cos in the first half, sin in the second."""
    return torch.cat([freqs_cis.real, freqs_cis.imag], dim=-1).contiguous()


def _forward_rope_interleaved(o: torch.Tensor, positions: torch.Tensor,
                              cache: torch.Tensor) -> torch.Tensor:
    """The rotation inverse_rope_from_cos_sin_cache is supposed to undo."""
    half = ROPE_DIM // 2
    cs = cache.index_select(0, positions)
    cos, sin = cs[:, :half].unsqueeze(1), cs[:, half:].unsqueeze(1)
    pairs = o[..., -ROPE_DIM:].float().unflatten(-1, (half, 2))
    even, odd = pairs[..., 0], pairs[..., 1]
    out = o.clone()
    out[..., -ROPE_DIM:] = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    return out


@pytest.fixture
def rope_case():
    torch.manual_seed(0)
    seq, heads = 12, 4
    o = torch.randn(seq, heads, NOPE_DIM + ROPE_DIM, dtype=torch.float32)
    freqs = rope.build_freqs_cis(ROPE_DIM, seq, 160000.0, True)
    return o, freqs, _cache_from_freqs(freqs)


def test_cache_form_agrees_with_the_complex_form(rope_case):
    o, freqs, cache = rope_case
    from_cache = rope.inverse_rope_from_cos_sin_cache(
        o, torch.arange(o.shape[0]), cache)

    # inverse_rope_ is batch-first: it indexes the freqs table by dim 1, so it
    # needs [batch, seq, heads, rope_dim]. That it cannot be handed a flat
    # [tokens, heads, dim] is the other half of why the cache form exists.
    expected = o.clone().unsqueeze(0)
    rope.inverse_rope_(expected[..., -ROPE_DIM:], freqs)

    torch.testing.assert_close(from_cache, expected[0], rtol=1e-6, atol=1e-6)


def test_it_inverts_the_forward_rotation(rope_case):
    """Pins the direction.

    Both directions agree with themselves, so only a round trip distinguishes
    the conjugate from the rotation itself. Getting it backwards rotates by
    2*theta instead of 0: plausible magnitudes, wrong values, no error.
    """
    o, _, cache = rope_case
    positions = torch.arange(o.shape[0])

    rotated = _forward_rope_interleaved(o, positions, cache)
    assert not torch.allclose(rotated, o, atol=1e-3), "forward rotation was a no-op"

    recovered = rope.inverse_rope_from_cos_sin_cache(rotated, positions, cache)
    torch.testing.assert_close(recovered, o, rtol=1e-5, atol=1e-5)


def test_each_row_uses_its_own_position(rope_case):
    """The reason this form exists.

    A golden dump is one sequence at positions 0..n-1, so an implementation
    that ignored `positions` entirely would reproduce it exactly. A real batch
    is not: rows from different sequences sit next to each other, and a decode
    step is a single row at position 5000.
    """
    o, _, cache = rope_case
    shuffled = torch.tensor([3, 1, 0, 2, 7, 5, 4, 6, 11, 9, 8, 10])

    sequential_out = rope.inverse_rope_from_cos_sin_cache(
        o, torch.arange(o.shape[0]), cache)
    shuffled_out = rope.inverse_rope_from_cos_sin_cache(o, shuffled, cache)
    assert not torch.allclose(sequential_out, shuffled_out, atol=1e-4)

    # Row i at position p must match rotating that row alone at p.
    for i, p in enumerate(shuffled.tolist()):
        alone = rope.inverse_rope_from_cos_sin_cache(
            o[i:i + 1], torch.tensor([p]), cache)
        torch.testing.assert_close(shuffled_out[i:i + 1], alone,
                                   rtol=1e-6, atol=1e-6)


def test_the_nope_dimensions_are_untouched(rope_case):
    o, _, cache = rope_case
    out = rope.inverse_rope_from_cos_sin_cache(
        o, torch.arange(o.shape[0]), cache)
    torch.testing.assert_close(out[..., :NOPE_DIM], o[..., :NOPE_DIM])
    assert not torch.allclose(out[..., NOPE_DIM:], o[..., NOPE_DIM:], atol=1e-4)


def test_a_cache_of_the_wrong_width_is_refused(rope_case):
    """Half a cache is the dangerous shape: it would read sin as cos."""
    o, _, cache = rope_case
    with pytest.raises(ValueError, match="cos_sin_cache last dim"):
        rope.inverse_rope_from_cos_sin_cache(
            o, torch.arange(o.shape[0]), cache[:, :ROPE_DIM // 2])


def test_a_position_count_mismatch_is_refused(rope_case):
    o, _, cache = rope_case
    with pytest.raises(ValueError, match="tokens but"):
        rope.inverse_rope_from_cos_sin_cache(
            o, torch.arange(o.shape[0] - 1), cache)
