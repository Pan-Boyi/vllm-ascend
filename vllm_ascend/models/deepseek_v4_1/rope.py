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

from vllm_ascend.aclnn_availability import require_aclnn_op

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

#: The three names of the same operator. None of them is derivable from
#: another, so all three are READ off csrc rather than transformed:
#:   * the torch schema           csrc/torch_binding.cpp   ops.def("...")
#:   * the aclnn API              csrc/torch_binding.cpp   EXEC_NPU_CMD(...)
#:   * the OpType, which is also the ops-info registry key, verbatim
#:                                .../inplace_partial_rotary_mul_def.cpp
#:                                OP_ADD(InplacePartialRotaryMul)
#: A guessed pair here would make the guard below judge a DIFFERENT operator
#: and still come out green.
_PARTIAL_ROTARY_TORCH_NAME = "inplace_partial_rotary_mul"
_PARTIAL_ROTARY_ACLNN_API = "aclnnInplacePartialRotaryMul"
_PARTIAL_ROTARY_OP_TYPE = "InplacePartialRotaryMul"

#: Operator arguments ``csrc/torch_binding.cpp`` passes to EXEC_NPU_CMD
#: (x, r1, r2, rotary_mode, partial_slice); the workspace/executor/stream tail
#: is appended by the macro and is not part of this count. It is a literal
#: copied out of another repository file, so the R53 suite re-derives it from
#: that EXEC_NPU_CMD at the revision under test -- if somebody edits the call,
#: the test goes red rather than this constant going quietly stale.
_PARTIAL_ROTARY_MUL_ARG_COUNT = 5


def _partial_rotary_op():
    """``torch.ops._C_ascend.inplace_partial_rotary_mul``, if it can actually RUN.

    Two different questions are asked here, in order, because they have
    different answers and different owners.

    **Is the torch schema reachable at all?** On A5 ``enable_custom_op()``
    returns False, so nothing imports ``vllm_ascend.vllm_ascend_C`` on purpose;
    the op is registered only if some other module happened to import it
    (today ``device_allocator/camem.py``). Without that the flag-on path failed
    with a bare host error that names neither the library nor the flag, so the
    import is done here and the two ways it can fail say which is which.

    **Is the operator PATH walkable?** That is a different question, and the
    check that used to stand here could not ask it: it stopped at
    ``hasattr(torch.ops._C_ascend, ...)``. ``csrc/torch_binding.cpp`` registers
    this schema with no condition that has anything to do with the operator:
    the only ``#if`` around it is the ``#ifdef ASCEND_PLATFORM_310P`` /
    ``#else`` that picks which of two whole ``TORCH_LIBRARY_EXPAND`` blocks a
    build compiles, and every A5 build takes the ``#else``. So on any A5
    installation where the C extension imports at all, that ``hasattr`` is
    True whether or not the operator was ever compiled into the operator
    package. A check that passes on every machine is not evidence; it is a
    check that LOOKS like one. Worse, it passed silently in the three shapes
    that matter most here:

    * the aclnn entry points do not resolve at all (the operator was dropped
      from the package's SoC list) -- the call would have died inside
      EXEC_NPU_CMD with a message naming ``libopapi.so``, which is not the
      library at fault;
    * they resolve out of **CANN's own** ``libopapi.so``. This is not
      hypothetical for this operator: CANN 9.2 exports
      ``aclnnInplacePartialRotaryMul`` and its ``GetWorkspaceSize`` itself, at
      the same 5-argument signature, so dlsym succeeds, no TORCH_CHECK fires,
      and a stranger runs in place of the operator card A4 measured;
    * the package answers but registers no kernel for the OpType (aclnn stubs,
      prototype and tiling are built SoC-independently; kernels are not).

    ``require_aclnn_op`` asks all of that and produces a different attribution
    for each. What it cannot judge -- argument ORDER and MEANING of whatever
    resolved, and which SoC this process will run on -- it logs rather than
    folds into the pass.

    Notes on the shape of this function:

    * ``torch_npu`` is deliberately NOT offered as a source. The block above
      has already established that the name is in ``torch.ops._C_ascend``,
      which ``find_torch_op`` tries first, so the level-(1) branch of the probe
      is unreachable from here and no namespace but ``_C_ascend`` is ever
      consulted. Accepting a torch_npu-supplied attribute of the same name
      would be the very failure this change is about: the probe reports that
      case as UNCHECKED and passes.
    * the verdict is memoised in ``_PARTIAL_ROTARY_OP``, exactly as before, so
      the per-token cost of the flag-on path is unchanged: after the first
      call this function is one ``is not None`` test and a return, as it was.
      A refusal is NOT memoised -- only the op is -- so a raise re-checks.
    * requiring a package of ours is safe on A5 only because the search path
      is settled before the first forward, and NOT by the route the comment
      above might suggest: ``enable_custom_op()`` returns False on this SoC
      before it reaches ``bootstrap_custom_op_env`` at all. The prepend to
      ASCEND_CUSTOM_OPP_PATH comes from
      ``AscendPlatform.import_kernels()``, which vLLM calls while importing
      ``vllm._custom_ops``. If that chain is ever broken, GetOpApiFuncAddr
      falls through to CANN's ``libopapi.so`` for every custom operator and
      this guard refuses instead of silently running CANN's -- which is the
      intended direction, but it is a behaviour change and not a no-op.
    """
    global _PARTIAL_ROTARY_OP
    if _PARTIAL_ROTARY_OP is not None:
        return _PARTIAL_ROTARY_OP
    if not hasattr(torch.ops._C_ascend, _PARTIAL_ROTARY_TORCH_NAME):
        try:
            import vllm_ascend.vllm_ascend_C  # noqa: F401  registers _C_ascend ops
        except ImportError as e:
            raise RuntimeError(
                "VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR=1 needs "
                "torch.ops._C_ascend.inplace_partial_rotary_mul, which is "
                "registered by vllm_ascend.vllm_ascend_C, and that library "
                f"could not be imported ({e}). Build/install vllm_ascend_C or "
                "set the flag to 0.") from e
        if not hasattr(torch.ops._C_ascend, _PARTIAL_ROTARY_TORCH_NAME):
            raise RuntimeError(
                "vllm_ascend.vllm_ascend_C was imported but did not register "
                "_C_ascend::inplace_partial_rotary_mul (the built library is "
                "older than this code?). Set "
                "VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR=0 or rebuild.")
    _PARTIAL_ROTARY_OP = require_aclnn_op(
        torch,
        None,  # see the docstring: _C_ascend is already established above
        torch_name=_PARTIAL_ROTARY_TORCH_NAME,
        aclnn_api=_PARTIAL_ROTARY_ACLNN_API,
        op_type=_PARTIAL_ROTARY_OP_TYPE,
        feature=("DSV4.1 inverse RoPE operator "
                 "(VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR=1; "
                 "0 selects the torch form in this same module)"),
        expected_arg_count=_PARTIAL_ROTARY_MUL_ARG_COUNT,
        require_vendor_package=True,
    )
    return _PARTIAL_ROTARY_OP


def _inverse_rope_operator(o: torch.Tensor, positions: torch.Tensor,
                           cos_sin_cache: torch.Tensor,
                           rope_head_dim: int) -> torch.Tensor:
    """The inverse rotation as one ``inplace_partial_rotary_mul`` call.

    Rotates a copy, so the caller's ``o`` is left as it was, as in the torch
    form.
    """
    from vllm_ascend.models.deepseek_v4_1.rope_widen import neg_sin_for_slice

    if o.dim() != 3:
        raise ValueError(
            f"o has shape {tuple(o.shape)}; expected [num_tokens, heads, "
            "head_dim]. The operator takes x 4-D and one axis is added here.")
    head_dim = o.shape[-1]
    # ``neg_sin_for_slice``, not ``cos_sin_for_slice`` + a negation here: the
    # forward RoPE in ``qkv_prolog`` already built this exact pair for this
    # step, and the negation is the same every layer. Both are cached there.
    cos, neg_sin = neg_sin_for_slice(cos_sin_cache, positions, rope_head_dim)
    out = o.clone(memory_format=torch.contiguous_format)
    _partial_rotary_op()(
        out.unsqueeze(1),                       # [T, 1, H, D]
        cos[:, None, None, :],                  # [T, 1, 1, rope_head_dim]
        neg_sin[:, None, None, :],
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
