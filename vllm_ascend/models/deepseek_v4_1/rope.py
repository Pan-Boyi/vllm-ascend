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
