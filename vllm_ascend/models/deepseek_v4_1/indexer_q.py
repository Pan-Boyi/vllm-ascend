# SPDX-License-Identifier: Apache-2.0
"""Fused RoPE + Q quantisation for the sparse indexer, without Triton.

Upstream dispatches this on a blacklist::

    if has_cutedsl():        ...cutedsl
    elif is_xpu():           ...xpu
    else:                    ...triton

Ascend falls into the Triton branch, and triton-ascend aborts compiling it
(an MLIR assertion in InterleaveOptimization.cpp -- ``abort()``, not an
exception), so this takes an equivalent torch route.

This transcribes the **MXFP4** branch of upstream's kernel
(``vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:325-431``). Four
details are silent if wrong:

* the rotation covers the **trailing** ``2 * half_rot_dim`` of the head, with
  the NoPE part ahead of it;
* the pairing is interleaved -- ``x[2k]`` with ``x[2k+1]``;
* the rotated values make a **round trip through bf16** before they are
  quantised, so the block scale is computed from the values as they will be
  stored. Upstream does this in the MXFP4 kernel too, for parity with the fp8
  one; the NoPE half is **not** round-tripped, in either;
* the weights are **not** folded. This is the half of the contract that
  changed with the format, and it changed in the direction that raises
  nothing.

The weight fold, and why this path does not have one: fp8 Q was stored without
a companion scale tensor, so its one per-(token, head) scalar was folded into
the returned weights and the downstream logits kernel applied it inline. MXFP4
scales are **per 32-element block** and travel with the Q values in their own
tensor, so there is no per-token scalar to fold -- and folding the fp8 way
here would scale every logit by a spurious factor, with no shape or dtype to
show for it. ``weights_out`` therefore carries the softmax and head scales
only, exactly as upstream's MXFP4 kernel writes it
(``fused_indexer_q.py:416-431``).
"""

from __future__ import annotations

import torch

import vllm_ascend.envs as envs
from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import (SCALE_GROUP,
                                                                 SCALE_PAIR,
                                                                 VALUES_PER_BYTE
                                                                 )
from vllm_ascend.models.deepseek_v4_1.qkv_prolog import rope_tail_operator
from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice


def _indexer_fused_rope_enabled(sample: torch.Tensor) -> bool:
    """The operator route, when it can run here and is switched on.

    ``is_npu`` rather than a try/except: the CPU unit tests must take the
    torch route, but a genuine operator failure on card has to surface. A
    try/except here would turn "the operator is broken" into "this is slow",
    which is the failure mode dsnext has and we do not copy
    (``dsa_v41.py:879-881``).
    """
    return (getattr(sample, "is_npu", False)
            and envs.VLLM_ASCEND_DSV41_INDEXER_FUSED_ROPE == "1")


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

    Returns ``((q_packed, q_scale), weights_out)``:

    * ``q_packed`` -- ``(T, H, head_dim // 2)`` uint8, two E2M1 values a byte;
    * ``q_scale``  -- ``(T, H, head_dim // 64, 2)`` uint8 E8M0. Upstream's own
      shape is ``(T, H, head_dim // 32)``; the trailing ``(D/64, 2)`` split is
      what ``npu_quant_lightning_indexer_v2`` requires of a mode 3/5 scale
      (``csrc/attention/quant_lightning_indexer_v2/README.md:56-59``). It is
      the same 4 bytes either way, in the same order.

    ``use_fp4`` is upstream's flag and does **not** select the format here.
    It comes from ``dsa_indexer_uses_fp4``, which raises outside Blackwell, so
    on Ascend it is always False -- while the index K cache this Q is matched
    against is MXFP4 unconditionally. Branching on it would leave this path
    permanently unreachable, which is what the removed ``NotImplementedError``
    did. It is accepted and ignored so the signature still matches the name
    this is bound over.
    """
    assert positions.ndim == 1, positions.shape
    assert index_q.ndim == 3, index_q.shape
    assert index_q_cos_sin_cache.ndim == 2, index_q_cos_sin_cache.shape

    num_tokens, num_heads, head_dim = index_q.shape
    if head_dim % (SCALE_GROUP * SCALE_PAIR):
        raise ValueError(
            f"index_q head_dim {head_dim} is not a multiple of "
            f"{SCALE_GROUP * SCALE_PAIR}. MXFP4 needs a whole number of "
            f"{SCALE_GROUP}-element scale blocks, and the operator needs them "
            f"in pairs -- its scale axis is (D/{SCALE_GROUP * SCALE_PAIR}, "
            f"{SCALE_PAIR}).")
    half_rot = index_q_cos_sin_cache.shape[-1] // 2
    rot_dim = 2 * half_rot
    nope_dim = head_dim - rot_dim
    if nope_dim < 0:
        raise ValueError(
            f"head_dim {head_dim} is smaller than the rotation width "
            f"{rot_dim} implied by the cos/sin cache")

    # 🔴 Explicit, because the operator path below would not raise on it.
    # The torch form happens to break: cos (P, 1, half) against pairs
    # (T, H, half) fails to broadcast. The operator takes x as [T,1,H,D] and
    # cos as [P,1,1,D], which broadcasts to a silently wrong rotation. The
    # main-KV route raises here too (rope.py:289-292); this one only did so
    # by accident, and the accident disappears with the operator.
    if positions.shape[0] != num_tokens:
        raise ValueError(
            f"positions has {positions.shape[0]} rows and index_q has "
            f"{num_tokens}. One angle per token is the whole contract; "
            "broadcasting a shorter cos/sin would rotate every token by the "
            "wrong amount and raise nothing.")

    if _indexer_fused_rope_enabled(index_q):
        # ``cos_sin_for_slice`` returns the **widened** (per-channel) form the
        # operator requires, so the half-width cache never reaches it. Passing
        # the half-width form would fail the operator's
        # ``cos.size(-1) == end - start`` check, or worse, pass a reshaped one
        # and rotate by the wrong angles.
        cos, sin = cos_sin_for_slice(index_q_cos_sin_cache, positions, rot_dim)
        # Same call the main KV path has used since e8dea3d36. It clones and
        # rotates only ``[nope_dim, head_dim)``, which is exactly what the
        # torch form below does -- including leaving the NoPE half untouched.
        #
        # ⚠️ **Not bit-identical by construction, and not yet measured on this
        # shape.** The torch form rotates in float32 and rounds once; the
        # operator rotates in the tensor dtype. For a bf16 ``index_q`` the
        # explicit round trip above is provably a no-op
        # (``bf16(fp32(bf16(r))) == bf16(r)``), so the absmax contract in the
        # module docstring survives -- what may differ is the rounding of the
        # rotation itself. The main KV route measured ``torch.equal`` on card
        # (qkv_prolog.py:216-222), but that was [37,8,128]/[37,128]; one shape
        # on one day is not a proof for this one. The switch is what turns a
        # divergence into a flip.
        q_roped = rope_tail_operator(index_q, cos, sin, rot_dim)
    else:
        cos_sin = index_q_cos_sin_cache.index_select(
            0, positions.to(torch.long)).to(torch.float32)
        cos = cos_sin[:, :half_rot].unsqueeze(1)      # (T, 1, half)
        sin = cos_sin[:, half_rot:].unsqueeze(1)

        pairs = index_q[..., nope_dim:].to(torch.float32).reshape(
            num_tokens, num_heads, half_rot, 2)
        x_even, x_odd = pairs[..., 0], pairs[..., 1]

        r_even = x_even * cos - x_odd * sin
        r_odd = x_odd * cos + x_even * sin

        # Upstream's bf16 round trip, in the MXFP4 kernel as in the fp8 one.
        # The NoPE half below is deliberately left alone: neither kernel
        # round-trips it, and doing so here would change bytes that upstream
        # does not.
        #
        # ⚠️ For a bf16 ``index_q`` these two lines are a **no-op** -- storing
        # back into a bf16 tensor rounds identically. They are load-bearing
        # only if ``index_q`` is ever float32 or float16, and nothing here
        # asserts that it is not. A test that feeds bf16 cannot tell whether
        # they are present, so the unit test for this contract uses float32
        # (test_deepseek_v4_1_indexer_q.py).
        r_even = r_even.to(torch.bfloat16).to(torch.float32)
        r_odd = r_odd.to(torch.bfloat16).to(torch.float32)

        rotated = torch.stack((r_even, r_odd), dim=-1).reshape(
            num_tokens, num_heads, rot_dim)
        # clone, not empty_like: the NoPE half has to arrive at the quantiser
        # as index_q had it, and a factory call here would also be a tensor
        # built without an explicit device.
        q_roped = index_q.clone()
        q_roped[..., nope_dim:] = rotated.to(index_q.dtype)

    q_packed, q_scale = _mx_quantise(q_roped)
    q_packed = q_packed.view(torch.uint8).reshape(
        num_tokens, num_heads, head_dim // VALUES_PER_BYTE)
    q_scale = q_scale.contiguous().view(torch.uint8).reshape(
        num_tokens, num_heads, head_dim // (SCALE_GROUP * SCALE_PAIR),
        SCALE_PAIR)

    # No q_scale fold -- see the module docstring.
    weights_out = (index_weights.to(torch.float32)
                   * index_weights_softmax_scale * index_weights_head_scale)
    return (q_packed, q_scale), weights_out


def _mx_quantise(x: torch.Tensor):
    """``npu_dynamic_mx_quant`` to E2M1 values + E8M0 block scales.

    Split out so the argument assembly is testable without the operator, and
    because this is the one line in the file that is a **new route**: the 25
    other ``npu_dynamic_mx_quant`` call sites in this tree all pass
    ``dst_type=torch.float8_e4m3fn``. Nothing here is a flag being flipped.
    """
    import torch_npu

    return torch_npu.npu_dynamic_mx_quant(
        x, dst_type=torch_npu.float4_e2m1fn_x2)


#: Where upstream's V4.1 attention module holds the name. The call site reads
#: that module's globals, so rebinding the attribute reaches it.
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
