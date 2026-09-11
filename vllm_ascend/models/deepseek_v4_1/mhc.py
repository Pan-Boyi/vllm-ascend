# SPDX-License-Identifier: Apache-2.0
"""Hyper-connections without tilelang.

The V4.1 decoder layer calls ``mhc_pre_delayed_tilelang`` and
``mhc_post_tilelang`` directly in its forward, and tilelang is not installed
here -- the third launch reached the first decoder layer and stopped with
"tilelang is required for mhc but is not installed". They are plain module-level
names in ``vllm.models.deepseek_v4_1.nvidia.model``, so they are rebound there
rather than the layer being forked.

Upstream ships torch equivalents of both in ``kernels/mhc/torch.py``, which is
what makes this cheap. ``mhc_post_torch`` is a drop-in -- same four arguments,
same return. ``mhc_pre_delayed_torch`` is not quite: the tilelang version takes
``norm_weight`` / ``norm_eps`` as well and folds an RMSNorm into the collapsed
layer input it returns, and the decoder layer does pass them. The torch
reference returns that input un-normalised, so the wrapper below applies the
norm afterwards.

This is a correctness-first substitution with a known cost, not a port. The
fused kernel does the mixing, the collapse and the norm in one pass; this is
several. The Ascend operators that would close that -- ``aclnnMhcPreSinkhorn``
and ``aclnnMhcPost`` -- exist and are not bound in this tree, which makes them
the right thing to move to and the wrong thing to wait for.
"""

from __future__ import annotations

import torch
from vllm.model_executor.kernels.mhc.torch import (mhc_post_torch,
                                                   mhc_pre_delayed_torch)

def mhc_post_ascend(x: torch.Tensor, residual: torch.Tensor,
                    post_layer_mix: torch.Tensor,
                    comb_res_mix: torch.Tensor) -> torch.Tensor:
    """mHC post through ``aclnnHcPost``, which this platform does have.

    Same four arguments and return as the tilelang version, so it is rebound
    directly. What differs is the operator's shape contract -- it takes a batch
    axis that V4.1 does not carry, and a ``post`` without the trailing 1 ::

        upstream            operator
        x    (T, D)         (b, s, d)
        res  (T, hc, D)     (b, s, hc, d)
        post (T, hc, 1)     (b, s, hc)      <- trailing 1 dropped
        comb (T, hc, hc)    (b, s, hc, hc)

    The V4.0 Ascend layer does the same unsqueeze; the ``post`` squeeze is
    extra here because V4.1's pre returns that axis and V4.0's does not.

    ``VLLM_ASCEND_DSV41_MHC_POST_TORCH=1`` forces upstream's torch reference
    instead. That exists to A/B the two on a card: the operator is the V4.0
    production path and its arithmetic is the same expression, but "same
    expression" has been wrong before today, and the two have never been
    compared on this model's shapes.
    """
    if envs_dsv41_mhc_post_torch():
        return mhc_post_torch(x, residual, post_layer_mix, comb_res_mix)

    if x.dim() != 2 or residual.dim() != 3:
        raise ValueError(
            f"expected x (tokens, hidden) and residual (tokens, hc, hidden), "
            f"got {tuple(x.shape)} and {tuple(residual.shape)}. The batch axis "
            "is added here, so a tensor that already has one would be given a "
            "second and the operator would read the wrong lengths.")
    if post_layer_mix.shape[-1] != 1:
        raise ValueError(
            f"post has shape {tuple(post_layer_mix.shape)}; the last axis must "
            "be the length-1 one upstream's pre returns, because it is dropped "
            "for the operator. Without the trailing 1 this would squeeze hc "
            "instead when hc happens to be 1.")

    out = torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(0),
        residual.unsqueeze(0),
        post_layer_mix.squeeze(-1).unsqueeze(0),
        comb_res_mix.unsqueeze(0),
    )
    return out.squeeze(0)


def envs_dsv41_mhc_post_torch() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_MHC_POST_TORCH == "1"


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32, cast back at the end.

    Accumulating in the input dtype is the usual way to lose this: hidden_size
    is 5120 and bf16 has an eight-bit significand, so the sum of squares stops
    growing well before the last terms are reached.
    """
    original = x.dtype
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    return (normed * weight.float()).to(original)


def mhc_pre_delayed_ascend(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``mhc_pre_delayed_tilelang``'s contract, on upstream's torch reference.

    The signature is copied from the tilelang version rather than from the
    torch one, because it is the tilelang one the decoder layer calls. The two
    extra arguments are the whole difference: the tilelang kernel folds an
    RMSNorm into the collapsed layer input, and the torch reference does not
    take them at all -- passing them through would be a TypeError, and dropping
    them silently would return an unnormalised input that is the right shape
    and the wrong values.
    """
    post_mix, res_mix, layer_input, next_pre_mix = mhc_pre_delayed_torch(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        pre_mix=pre_mix,
        x=x,
    )

    if norm_weight is not None:
        if norm_weight.shape != (layer_input.shape[-1],):
            raise ValueError(
                f"norm_weight has shape {tuple(norm_weight.shape)} against a "
                f"layer input of width {layer_input.shape[-1]}. The tilelang "
                "kernel asserts this too; without the check a broadcast would "
                "make it a scale rather than a per-channel norm.")
        layer_input = _rms_norm(layer_input, norm_weight, norm_eps)

    return post_mix, res_mix, layer_input, next_pre_mix
