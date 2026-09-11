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

    ``npu_rms_norm_cast(x, gamma, eps) -> (y, y_fp32)`` is an in-repo operator
    that matches this exactly and also hands back the fp32 copy. It is not used
    yet, and the reason it looks unavailable is not the reason it is unused:
    V4.0 calls it behind ``enable_custom_op()``, which returns False on A5, but
    that switch is a blanket "disable all custom ops on this chip" precaution
    rather than a judgement about this operator. The mechanism works --
    ``npu_hc_pre_v2``, also in-repo, runs on A5 after
    ``bootstrap_custom_op_env()``. What is missing is a check that this
    particular operator is numerically right there, which is minutes of work on
    a card and has not been done.


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
    if envs_dsv41_mhc_pre_operator():
        post_mix, res_mix, next_pre_mix = _pre_parts_from_operators(
            residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, x)
        layer_input = _collapse_with(pre_mix, residual)
    else:
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


# -- the operator path -------------------------------------------------------
#
# torch_npu carries the whole delayed chain already; it needed no new operator
# and no new binding. Measured on card 2 by the kernel line, against the
# in-repo npu_hc_pre_v2:
#
#   npu_mhc_pre(...)[0] hIn    bit-identical
#               ...[1] hPost   rel 2.793e-07
#               ...[2] hRes    the comb fragment *before* sinkhorn
#               ...[5] hPre    rel 6.158e-05   <- what "delayed" needs
#   npu_mhc_sinkhorn(hRes)[0]  rel 1.176e-06   == the in-repo comb
#
# So the in-repo operator is npu_mhc_pre + npu_mhc_sinkhorn, and the mhc family
# additionally hands back `pre`, which is the one thing the delayed variant
# needs and the non-delayed one has no reason to expose.


def envs_dsv41_mhc_pre_operator() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR == "1"


def _collapse_with(pre_mix: torch.Tensor | None,
                   residual: torch.Tensor) -> torch.Tensor:
    """Collapse the streams with the *previous* sublayer's pre-mix.

    This is the whole of "delayed", and it is why ``npu_mhc_pre``'s first
    output cannot be used here: that one is collapsed with the pre-mix this
    call just computed, which is the non-delayed variant. The two differ in
    value, not in shape, so substituting one would be silent.
    """
    if pre_mix is None:
        return residual[:, 0]
    return (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)


def _pre_parts_from_operators(residual, fn, hc_scale, hc_base, rms_eps,
                              hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                              sinkhorn_repeat, x):
    """``post``, ``comb`` and the next ``pre``, from the two torch_npu ops."""
    del hc_post_mult_value          # folded into the operator's post
    source = residual.flatten(1) if x is None else x

    # out_flag=1 is not optional: at the default 0 the last three outputs come
    # back empty, and `pre` is one of them. An empty tensor here would look
    # like a shape problem downstream rather than a flag that was left off.
    outputs = torch.ops.npu.npu_mhc_pre(
        source, fn, hc_scale, hc_base,
        norm_eps=rms_eps, hc_eps=hc_pre_eps, out_flag=1)
    if len(outputs) != 6:
        raise RuntimeError(
            f"npu_mhc_pre returned {len(outputs)} tensors, expected 6. The "
            "delayed variant reads [1] post, [2] the pre-sinkhorn comb and "
            "[5] pre; a different arity means the output contract moved.")
    post, comb_fragment, next_pre = outputs[1], outputs[2], outputs[5]
    if next_pre.numel() == 0:
        raise RuntimeError(
            "npu_mhc_pre returned an empty `pre`, which is what out_flag=0 "
            "does. The delayed variant cannot be computed without it.")

    # The fragment already carries scale and bias; adding them again degrades
    # the result to 0.419 against 1.18e-06 for feeding it straight through.
    comb = torch.ops.npu.npu_mhc_sinkhorn(
        comb_fragment, eps=hc_sinkhorn_eps, num_iters=sinkhorn_repeat,
        out_flag=1)[0]

    # Upstream's contract puts a trailing 1 on post; the operator does not.
    if post.dim() == residual.dim() - 1:
        post = post.unsqueeze(-1)
    return post, comb, next_pre
