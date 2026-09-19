# SPDX-License-Identifier: Apache-2.0
"""Rotary embedding parameters for DeepSeek-V4.1-Flash, and the inverse rotation.

Attention output carries RoPE on its trailing ``qk_rope_head_dim`` dimensions,
which must be rotated back out before the output projection. Two things about
that are easy to get wrong and produce plausible-looking numbers rather than an
error:

**The parameters differ per layer.** Layers whose ``compress_ratios`` entry is 0
use ``rope_theta`` with no YaRN scaling; every other layer uses
``compress_rope_theta`` *with* YaRN. Building one table and using it everywhere
reproduces neither.

**The pairing is GPT-J style**, i.e. interleaved ``(x[0::2], x[1::2])`` rather
than split-half, as in the reference implementation.
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


#: ``inplace_partial_rotary_mul`` does honour ``partial_slice`` on A5: with
#: slice [448, 512] and 64-wide cos/sin it rotates only the tail, leaves the
#: NoPE bytes untouched, and with ``sin`` negated is bit-exact against the
#: torch path below (card A4, CARD_RESULTS.md). Its ``interleave`` mode is the
#: GPT-J pairing, and the inverse is the forward rotation with ``sin`` negated.
#: It takes x as [T, 1, H, D] and cos/sin widened to the slice width
#: (``rope_widen.cos_sin_for_slice``). The operator form is selected by
#: ``VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR`` (default "1" since the
#: 2026-09-19 4-card A/B); "0" selects the torch form.


def _inverse_rope_operator_enabled() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR == "1"


_PARTIAL_ROTARY_OP = None


def _partial_rotary_op():
    """``torch.ops._C_ascend.inplace_partial_rotary_mul``, loading its library.

    On A5 ``enable_custom_op()`` returns False, so nothing imports
    ``vllm_ascend.vllm_ascend_C`` on purpose; the op is registered only if some
    other module happened to import it (today ``device_allocator/camem.py``).
    Without that the flag-on path failed with a bare host error that names
    neither the library nor the flag. Load it here, and say so if it cannot be.
    """
    global _PARTIAL_ROTARY_OP
    if _PARTIAL_ROTARY_OP is not None:
        return _PARTIAL_ROTARY_OP
    if not hasattr(torch.ops._C_ascend, "inplace_partial_rotary_mul"):
        try:
            import vllm_ascend.vllm_ascend_C  # noqa: F401  registers _C_ascend ops
        except ImportError as e:
            raise RuntimeError(
                "VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR=1 needs "
                "torch.ops._C_ascend.inplace_partial_rotary_mul, which is "
                "registered by vllm_ascend.vllm_ascend_C, and that library "
                f"could not be imported ({e}). Build/install vllm_ascend_C or "
                "set the flag to 0.") from e
        if not hasattr(torch.ops._C_ascend, "inplace_partial_rotary_mul"):
            raise RuntimeError(
                "vllm_ascend.vllm_ascend_C was imported but did not register "
                "_C_ascend::inplace_partial_rotary_mul (the built library is "
                "older than this code?). Set "
                "VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR=0 or rebuild.")
    _PARTIAL_ROTARY_OP = torch.ops._C_ascend.inplace_partial_rotary_mul
    return _PARTIAL_ROTARY_OP


def _inverse_rope_operator(o: torch.Tensor, positions: torch.Tensor,
                           cos_sin_cache: torch.Tensor,
                           rope_head_dim: int) -> torch.Tensor:
    """The inverse rotation as one ``inplace_partial_rotary_mul`` call.

    Rotates a copy, so the caller's ``o`` is left as it was, as in the torch
    form.
    """
    from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice

    if o.dim() != 3:
        raise ValueError(
            f"o has shape {tuple(o.shape)}; expected [num_tokens, heads, "
            "head_dim]. The operator takes x 4-D and one axis is added here.")
    head_dim = o.shape[-1]
    cos, sin = cos_sin_for_slice(cos_sin_cache, positions, rope_head_dim)
    out = o.clone(memory_format=torch.contiguous_format)
    _partial_rotary_op()(
        out.unsqueeze(1),                       # [T, 1, H, D]
        cos[:, None, None, :],                  # [T, 1, 1, rope_head_dim]
        (-sin)[:, None, None, :],
        rotary_mode="interleave",
        partial_slice=[head_dim - rope_head_dim, head_dim],
    )
    return out


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

    if _inverse_rope_operator_enabled():
        return _inverse_rope_operator(o, positions, cos_sin_cache, rope_head_dim)

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
