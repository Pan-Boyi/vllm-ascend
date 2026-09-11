# SPDX-License-Identifier: Apache-2.0
"""Rotary embedding parameters for DeepSeek-V4.1-Flash, and the inverse rotation.

Attention output carries RoPE on its trailing ``qk_rope_head_dim`` dimensions,
which must be rotated back out before the output projection. Two things about
that are easy to get wrong and produce plausible-looking numbers rather than an
error:

**The parameters differ per layer.** Layers whose ``compress_ratios`` entry is 0
use ``rope_theta`` with no YaRN scaling; every other layer uses
``compress_rope_theta`` *with* YaRN. Building one table and using it everywhere
reproduces neither. Verified against the CPU golden dumps: with the right
parameters all three smoke layers come back bit-exact, and with the wrong ones
relative error is 3.4e-1 (layer 0), 3.7e-1 (layer 1) and 4.8e-1 (layer 2).

**The pairing is GPT-J style**, i.e. interleaved ``(x[0::2], x[1::2])`` rather
than split-half. Rotation preserves the norm of each pair, which identifies the
convention without reference to the angles: under interleaved pairing the pair
norms of golden's pre- and post-rotation dumps agree to 3.8e-3 (bf16 rounding);
under split-half they disagree by a factor of 82.
"""

from __future__ import annotations

import math

import torch

#: Only the trailing ``rope_head_dim`` dimensions are rotated; the rest of the
#: head passes through untouched, which makes a useful free self-check.
NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64


def rope_params_for_layer(config, layer_id: int) -> tuple[float, bool]:
    """Return ``(base, use_yarn)`` for one layer's attention RoPE."""
    ratio = config.compress_ratios[layer_id]
    if ratio == 0:
        return float(config.rope_theta), False
    return float(config.compress_rope_theta), True


def build_freqs_cis(
    dim: int,
    seq_len: int,
    base: float,
    use_yarn: bool,
    original_seq_len: int = 65536,
    factor: float = 16.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> torch.Tensor:
    """Complex rotation factors, shaped ``[seq_len, dim // 2]``.

    Mirrors the reference implementation's ``precompute_freqs_cis``. Note there
    is no ``mscale``: the reference does not rescale the attention softmax for
    YaRN, so introducing one here would silently change the logits.
    """
    freqs = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if use_yarn:
        def correction_dim(rotations: float) -> float:
            return (dim * math.log(original_seq_len / (rotations * 2 * math.pi))
                    / (2 * math.log(base)))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low)
                / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    angles = torch.outer(torch.arange(seq_len, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(angles), angles)


def inverse_rope_(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate RoPE back out of ``x``'s trailing dimensions, in place.

    ``x`` is ``[..., seq, heads, rope_dim]`` sliced to the rotated tail. The
    conjugate of the same factors the query was rotated with undoes it.
    """
    out = x
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    conj = freqs_cis.conj()
    conj = (conj.view(1, z.size(1), z.size(-1)) if z.ndim == 3
            else conj.view(1, z.size(1), 1, z.size(-1)))
    return out.copy_(torch.view_as_real(z * conj).flatten(-2))


def inverse_rope_attention_output(
    o: torch.Tensor,
    config,
    layer_id: int,
    seq_len: int | None = None,
) -> torch.Tensor:
    """Undo RoPE on an attention output's trailing dimensions.

    ``o`` is ``[batch, seq, heads, head_dim]``. Returned as a fresh tensor; the
    leading ``NOPE_HEAD_DIM`` dimensions are copied through unchanged.
    """
    if o.shape[-1] != NOPE_HEAD_DIM + ROPE_HEAD_DIM:
        raise ValueError(
            f"expected head_dim {NOPE_HEAD_DIM + ROPE_HEAD_DIM}, got "
            f"{o.shape[-1]}; the nope/rope split is not what this assumes")

    base, use_yarn = rope_params_for_layer(config, layer_id)
    freqs = build_freqs_cis(ROPE_HEAD_DIM, seq_len or o.shape[1], base, use_yarn)
    out = o.clone()
    inverse_rope_(out[..., -ROPE_HEAD_DIM:], freqs.to(o.device))
    return out


#: Why this is torch and not ``inplace_partial_rotary_mul``, which exists on
#: this platform and looks like an exact fit.
#:
#: Its ``partial_slice`` attribute is declared and never read -- it appears
#: twice in the whole operator, both times in a declaration, and the tiling
#: reads only the mode. Passing [448, 512] and [448, 64] do the same thing:
#: nothing, silently. The rotation covers all of ``D``, so handing it a 512-wide
#: head would rotate the 448 NoPE dimensions too.
#:
#: Rotating only the tail would mean passing a 4-D *view* of ``x[..., 448:512]``
#: to an in-place operator, on non-contiguous memory, which nobody has tried.
#: The cos/sin would also need widening from 32 frequencies to 64, because the
#: operator requires their ``D`` to equal ``x``'s.
#:
#: So it is available and it is not a drop-in. This path is bit-exact against
#: golden today; the operator is a performance question for later, with a
#: measurable answer and three untested steps in front of it.
#:
#: ``interleave`` is confirmed to be the GPT-J pairing (the kernel does
#: ``i ^ 1``), and the inverse is the forward rotation with ``sin`` negated --
#: both worth keeping for whoever picks this up.


def inverse_rope_from_cos_sin_cache(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_head_dim: int = ROPE_HEAD_DIM,
) -> torch.Tensor:
    """Undo RoPE using the rotary module's own table, addressed by position.

    This is the form the output projection needs. ``inverse_rope_`` above takes
    the complex factors and assumes positions ``0..seq_len-1``, which is true of
    a golden dump and false of anything batched or decoding: there, each row
    carries its own absolute position and rows from different sequences sit next
    to each other.

    ``o`` is ``[num_tokens, heads, head_dim]``, ``positions`` is ``[num_tokens]``
    and ``cos_sin_cache`` is ``[max_pos, rope_head_dim]`` holding cos in the
    first half of its last dimension and sin in the second -- the layout
    upstream's fused kernel reads, so the table the attention layer already
    built can be used as-is rather than rebuilt here with a second set of
    parameters that could drift from it.

    Returns a fresh tensor; the leading ``head_dim - rope_head_dim`` dimensions
    are passed through unchanged.
    """
    if cos_sin_cache.shape[-1] != rope_head_dim:
        raise ValueError(
            f"cos_sin_cache last dim is {cos_sin_cache.shape[-1]}, expected "
            f"rope_head_dim={rope_head_dim}. It holds cos||sin, each of length "
            "rope_head_dim // 2, so a mismatch here means the halves would be "
            "sliced in the wrong place and sin read as cos.")
    if o.shape[0] != positions.shape[0]:
        raise ValueError(
            f"{o.shape[0]} tokens but {positions.shape[0]} positions; each row "
            "is rotated by its own absolute position, so these cannot differ")

    half = rope_head_dim // 2
    cs = cos_sin_cache.to(torch.float32).index_select(0, positions.to(torch.long))
    cos = cs[:, :half].unsqueeze(1)
    sin = cs[:, half:].unsqueeze(1)

    # Interleaved pairing: dim 2k and 2k+1 rotate together against frequency k.
    pairs = o[..., -rope_head_dim:].to(torch.float32).unflatten(-1, (half, 2))
    even, odd = pairs[..., 0], pairs[..., 1]

    # The conjugate rotation. Forward RoPE is (e*cos - o*sin, o*cos + e*sin);
    # this is that with sin negated, which is what undoes it.
    rotated = torch.stack((even * cos + odd * sin, odd * cos - even * sin),
                          dim=-1).flatten(-2)

    out = o.clone()
    out[..., -rope_head_dim:] = rotated.to(o.dtype)
    return out
