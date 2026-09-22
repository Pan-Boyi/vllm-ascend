# SPDX-License-Identifier: Apache-2.0
"""Hyper-connections without tilelang.

The V4.1 decoder layer calls ``mhc_pre_delayed_tilelang`` and
``mhc_post_tilelang`` directly, and tilelang is not available here. They are
module-level names in ``vllm.models.deepseek_v4_1.nvidia.model``, so they are
rebound there rather than the layer being forked.

- post runs on the ``aclnnHcPost`` operator (``npu_hc_post``), with the shapes
  adapted to its contract.
- pre uses an operator path built on torch_npu's ``npu_mhc_pre`` /
  ``npu_mhc_sinkhorn`` by default (``VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR``,
  default "1"); "0" selects upstream's torch reference
  (``mhc_pre_delayed_torch``) plus the RMSNorm the tilelang kernel folds in.
  The first layer's broadcast-embedding call stays on torch either way.
"""

from __future__ import annotations

import torch
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch


def mhc_post_ascend(x: torch.Tensor, residual: torch.Tensor,
                    post_layer_mix: torch.Tensor,
                    comb_res_mix: torch.Tensor) -> torch.Tensor:
    """mHC post through ``aclnnHcPost``.

    Same arguments and return as the tilelang version; the operator takes a
    batch axis V4.1 does not carry and a ``post`` without the trailing 1 ::

        upstream            operator
        x    (T, D)         (b, s, d)
        res  (T, hc, D)     (b, s, hc, d)
        post (T, hc, 1)     (b, s, hc)
        comb (T, hc, hc)    (b, s, hc, hc)
    """
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


def _fused_mhc_ops_enabled() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_MHC_FUSED_OPS == "1"


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32, cast back at the end.

    Accumulating in bf16 loses precision over a 5120-wide sum of squares.

    The torch form below is 9 kernels a call and runs once per sublayer --
    measured 702 kernels and **1.27 ms of a 28.8 ms decode step**
    (适配方案 §6.9.15). ``npu_rms_norm`` does the same thing in one call and
    is **already used in this model** for the q/kv norms; it accumulates in
    fp32 internally, which is the property the comment above is about.
    """
    if getattr(x, "is_npu", False) and _fused_mhc_ops_enabled():
        import torch_npu

        return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]
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
    """``mhc_pre_delayed_tilelang``'s contract.

    The signature is the tilelang one, which the decoder layer calls: it also
    takes ``norm_weight`` / ``norm_eps`` and returns the collapsed layer input
    RMS-normalised, which the torch reference does not do.
    """
    # The first layer passes ``x`` (the bare embedding) with ``fn`` already
    # summed over the hc copies, [hc3, D]. npu_mhc_pre needs phi as
    # [hc3, hc*D] against a [T, hc, D] x, so that one call stays on torch.
    if envs_dsv41_mhc_pre_operator() and x is None:
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
# npu_mhc_pre + npu_mhc_sinkhorn reproduce the in-repo npu_hc_pre_v2, and
# npu_mhc_pre additionally returns ``pre``, which the delayed variant needs.


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
    if getattr(residual, "is_npu", False) and _fused_mhc_ops_enabled():
        # The same weighted sum over the stream axis as a single batched
        # matmul: [T,1,S] @ [T,S,D] -> [T,1,D]. The torch form is
        # Cast + Mul + ReduceSum + Cast, measured 309 kernels and 0.85 ms a
        # step, and it is also where a bf16 residual gets widened to fp32 and
        # immediately narrowed again -- the RMSNorm right after widens it
        # back (mhc.py:122). The cube accumulates in fp32, so the widening
        # this replaces was already what the hardware does.
        return torch.bmm(pre_mix.unsqueeze(1).to(residual.dtype),
                         residual).squeeze(1)
    return (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)


def _pre_parts_from_operators(residual, fn, hc_scale, hc_base, rms_eps,
                              hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
                              sinkhorn_repeat, x):
    """``post``, ``comb`` and the next ``pre``, from the two torch_npu ops.

    npu_mhc_pre takes x as [T, n, D] or [B, S, n, D] and refuses 2-D ("Input x
    must be 3D or 4D"), so ``residual`` goes in as it is, [T, hc, D], against
    phi [hc^2 + 2hc, hc*D]. Outputs: post [T, hc], the pre-sinkhorn comb
    [T, hc, hc], pre [T, hc]. The flattened [T, hc*D] form belongs to the torch
    reference only.
    """
    del hc_post_mult_value          # folded into the operator's post
    if x is not None:
        raise ValueError(
            "the operator path takes no x: the first layer's x is the bare "
            "[T, D] embedding with fn summed over the hc copies, which "
            "npu_mhc_pre's [T, n, D] / [n^2+2n, n*D] contract cannot express. "
            "mhc_pre_delayed_ascend keeps that call on torch.")
    if residual.dim() != 3:
        raise ValueError(
            f"residual has shape {tuple(residual.shape)}; expected "
            "(tokens, hc, hidden), the 3-D x npu_mhc_pre takes.")
    tokens, hc = residual.shape[0], residual.shape[1]
    source = residual

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

    for name, got, want in (("post", post, (tokens, hc)),
                            ("comb", comb, (tokens, hc, hc)),
                            ("pre", next_pre, (tokens, hc))):
        if tuple(got.shape) != want:
            raise RuntimeError(
                f"npu_mhc_pre/sinkhorn returned {name} of shape "
                f"{tuple(got.shape)} for x {tuple(residual.shape)}; the "
                f"documented shape is {want}.")

    # Upstream's contract puts a trailing 1 on post; the operator does not.
    return post.unsqueeze(-1), comb, next_pre
