# SPDX-License-Identifier: Apache-2.0
"""Fused RoPE + Q quantisation for the sparse indexer, without Triton.

Upstream dispatches this on a blacklist::

    if has_cutedsl():        ...cutedsl
    elif is_xpu():           ...xpu
    else:                    ...triton

Ascend is neither, so it falls into the Triton branch, and triton-ascend
aborts compiling it -- an MLIR assertion in InterleaveOptimization.cpp about a
deinterleave offset, which is abort() rather than an exception: all four
workers vanished in the same second with exit code None, no traceback, nothing
in dmesg. The kernel was identified from a half-written Triton cache entry.

That makes three of this family, and the rule after the first two was to stop
trying to catch them and take an equivalent route.

This is a transcription of the FP8 branch of upstream's kernel, not a
rederivation of it. Four details would have been wrong if derived, and each is
silent:

* the rotation covers the **trailing** ``2 * half_rot_dim`` of the head, with
  the NoPE part ahead of it;
* the pairing is interleaved -- ``x[2k]`` with ``x[2k+1]``;
* the rotated values make a **round trip through bf16 before the absmax**, so
  the scale is computed from the values as they will be stored, not as they
  were computed;
* the scale is floored at 1e-4 and then **rounded up to a power of two**
  (``exp2(ceil(log2(s)))``).

The weight fold is the other half of the contract: FP8 Q is stored without a
companion scale tensor, so the per-(token, head) scale is folded into the
returned weights, which the downstream logits kernel applies inline. Getting
that wrong scales every logit and raises nothing.
"""

from __future__ import annotations

import torch

#: e4m3fn's largest finite value. Upstream picks 224.0 for e4m3fnuz on gfx942;
#: that dtype does not arise here, and guessing which one applies is exactly
#: the kind of silent factor-of-two this file exists to avoid.
FP8_MAX = 448.0

#: Upstream's floor before the scale is taken, so an all-zero head does not
#: divide by zero.
AMAX_FLOOR = 1e-4


def fused_indexer_q_rope_quant_ascend(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    use_fp4: bool = False,
):
    """Upstream's contract, in torch.

    Returns ``(q_fp8, weights_out)``.
    """
    if use_fp4:
        raise NotImplementedError(
            "the MXFP4 indexer path is not ported. Its contract differs in a "
            "way that is not a detail: the scales live with the Q values in "
            "per-32-element blocks and are NOT folded into the weights, so "
            "reusing the FP8 fold here would scale every logit by a spurious "
            "per-token factor and raise nothing. Port it deliberately or "
            "leave use_fp4 off.")

    assert positions.ndim == 1, positions.shape
    assert index_q.ndim == 3, index_q.shape
    assert index_q_cos_sin_cache.ndim == 2, index_q_cos_sin_cache.shape

    num_tokens, num_heads, head_dim = index_q.shape
    half_rot = index_q_cos_sin_cache.shape[-1] // 2
    rot_dim = 2 * half_rot
    nope_dim = head_dim - rot_dim
    if nope_dim < 0:
        raise ValueError(
            f"head_dim {head_dim} is smaller than the rotation width "
            f"{rot_dim} implied by the cos/sin cache")

    cos_sin = index_q_cos_sin_cache.index_select(
        0, positions.to(torch.long)).to(torch.float32)
    cos = cos_sin[:, :half_rot].unsqueeze(1)      # (T, 1, half)
    sin = cos_sin[:, half_rot:].unsqueeze(1)

    q = index_q.to(torch.float32)
    x_nope = q[..., :nope_dim]
    pairs = q[..., nope_dim:].reshape(num_tokens, num_heads, half_rot, 2)
    x_even, x_odd = pairs[..., 0], pairs[..., 1]

    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    # Before the absmax, not after: the scale must describe the values that
    # will actually be stored.
    r_even = r_even.to(torch.bfloat16).to(torch.float32)
    r_odd = r_odd.to(torch.bfloat16).to(torch.float32)

    amax = torch.maximum(r_even.abs().amax(-1), r_odd.abs().amax(-1))
    if nope_dim > 0:
        amax = torch.maximum(amax, x_nope.abs().amax(-1))

    scale = torch.clamp(amax, min=AMAX_FLOOR) / FP8_MAX
    scale = torch.exp2(torch.ceil(torch.log2(scale)))          # (T, H)
    divisor = scale.unsqueeze(-1)

    q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    if nope_dim > 0:
        q_fp8[..., :nope_dim] = (x_nope / divisor).to(torch.float8_e4m3fn)
    rotated = torch.stack((r_even / divisor, r_odd / divisor), dim=-1)
    q_fp8[..., nope_dim:] = rotated.reshape(
        num_tokens, num_heads, rot_dim).to(torch.float8_e4m3fn)

    weights_out = (index_weights.to(torch.float32) * scale
                   * index_weights_softmax_scale * index_weights_head_scale)
    return q_fp8, weights_out


#: Where upstream's V4.1 attention module holds the name. The call site is
#: ``attention.py:1201`` and it reads this module's own globals, so rebinding
#: the attribute lands whenever it is done -- the same property that made the
#: MoE router hook work and the FusedMoEFactory one not.
_TARGET_MODULE = "vllm.models.deepseek_v4_1.attention"
_TARGET_NAME = "fused_indexer_q_rope_quant"


def apply_indexer_q_patch() -> bool:
    """Rebind the indexer's fused RoPE+quant. Idempotent."""
    import importlib

    module = importlib.import_module(_TARGET_MODULE)
    if not hasattr(module, _TARGET_NAME):
        raise AttributeError(
            f"{_TARGET_MODULE} no longer defines {_TARGET_NAME!r}. Until this "
            "is repointed the indexer compiles a Triton kernel that aborts the "
            "process on this platform, with no traceback to say why.")
    if getattr(module, _TARGET_NAME) is fused_indexer_q_rope_quant_ascend:
        return False
    setattr(module, _TARGET_NAME, fused_indexer_q_rope_quant_ascend)
    return True
