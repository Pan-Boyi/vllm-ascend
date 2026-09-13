# SPDX-License-Identifier: Apache-2.0
"""Indexer K production for V4.1 kv-source layers, without Triton.

Upstream's ``indexer_k_norm_rope_store`` is one Triton program per token doing
k_norm, RoPE, quantisation and a paged store. It is not dispatched on a
platform blacklist -- there is only the Triton version -- so on Ascend it would
be compiled by triton-ascend, and a per-token paged store is exactly the
pointer-arithmetic shape that has aborted the process three times in this
project. Those aborts are ``abort()``, not exceptions; the rule after the
second was to take an equivalent route rather than try to catch them.

The quantisation and the paged store are not rebuilt here: A5 has a fused
operator for both, ``indexer_compress_epilog_v2``, already used by the V4.0
path in ``vllm_ascend/device/device_op.py``. What this file adds is the part
upstream fuses in and the operator has no notion of -- the group gate, the
norm, and the rotation.

Transcribed, not rederived. Five details are silent if wrong:

* only group-boundary tokens publish a key: ``(position + 1) % ratio == 0``.
  V4.0 has no such gate -- it pools from a compressor state cache -- so this is
  not "the V4.0 K write with a different ratio";
* the latent stands for the *first* token of its group, so the rotation uses
  ``compressed_pos = (position // ratio) * ratio``, not ``position``;
* the rotation covers the **trailing** 64 of 128, interleaved -- the head is
  128 wide and the rope width is 64, and they are different numbers;
* k_norm makes a round trip through bf16 before the rotation, and the rotated
  halves make another one before the store. Neither changes a shape or a dtype;
  both change the bytes that reach the quantiser;
* the operator's skip sentinel is ``slot == -1`` **exactly**, where upstream's
  is ``slot < 0``. Any other negative is read as a real slot and writes outside
  the cache, silently.
"""

from __future__ import annotations

import os

import torch
from vllm.logger import logger

#: The rope width of the V4.1 indexer key, against a head width of 128.
#: Upstream passes these to the kernel as separate constants
#: (``HEAD_SIZE=128, ROPE_HEAD_DIM=64``); reading the head width as the rope
#: width rotates the NoPE half as well and raises nothing.
INDEX_HEAD_DIM = 128
INDEX_ROPE_HEAD_DIM = 64

#: What the fused A5 store operator treats as "skip this token".
SKIP_SLOT = -1

_MXFP4_UNREACHABLE = (
    "MXFP4 indexer keys are not implemented on Ascend, and upstream cannot "
    "reach them either: resolve_indexer_kv_dtype defaults to 'fp8' and the "
    "mxfp4 value is gated to Blackwell datacenter parts "
    "(is_device_capability_family(100)) with a ValueError for everything else. "
    "If this raises, something selected mxfp4 on hardware upstream would have "
    "refused, and the question is why -- not how to add the branch.")


def group_boundary_slots(kv_slot_mapping: torch.Tensor,
                         positions: torch.Tensor,
                         compress_ratio: int) -> torch.Tensor:
    """Slots for the tokens that publish a key; ``-1`` for the rest.

    Upstream gates inside the kernel, on two conditions in this order: the slot
    is negative, or the token is not its group's last. Ascend's store operator
    has no position input, so the gate has to be folded into the slot mapping
    before the call -- which is also where it belongs, since it is the only
    thing that knows the ratio.

    The normalisation to exactly ``-1`` is not tidiness. The operator's kernels
    test ``slot == -1`` and fall through for every other value, so a ``-2`` --
    or anything a future slot builder picks as its own sentinel -- is divided
    by the block size into a negative block index and written. That is memory
    corruption with no error, in a tensor nobody reads until much later.
    """
    if compress_ratio < 1:
        raise ValueError(
            f"compress_ratio {compress_ratio} has no group boundary. Layers "
            "with ratio 0 carry no compressed stream and never reach here; "
            "reaching it means a layer was routed to the wrong K producer.")
    publishes = ((positions + 1) % compress_ratio) == 0
    valid = kv_slot_mapping >= 0
    skip = torch.full_like(kv_slot_mapping, SKIP_SLOT)
    return torch.where(publishes & valid, kv_slot_mapping, skip)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor,
              eps: float) -> torch.Tensor:
    """k_norm in fp32, with upstream's bf16 round trip.

    The round trip is in the reference and in upstream's kernel, and it is not
    decoration: the values that leave here are the ones the absmax downstream
    sees, so keeping fp32 precision produces a different scale from the same
    inputs.
    """
    acc = x.to(torch.float32)
    variance = acc.pow(2).mean(dim=-1, keepdim=True)
    normed = acc * torch.rsqrt(variance + eps) * weight.to(torch.float32)
    return normed.to(torch.bfloat16).to(torch.float32)


def _rope_interleaved_tail(k: torch.Tensor, cos_sin_cache: torch.Tensor,
                           positions: torch.Tensor,
                           rope_head_dim: int) -> torch.Tensor:
    """GPT-J rotation over the trailing ``rope_head_dim``, position-addressed.

    ``cos_sin_cache[p]`` is ``rope_head_dim`` wide and holds the half-width
    cosines followed by the half-width sines, one of each per interleaved pair.
    """
    half = rope_head_dim // 2
    row = cos_sin_cache[positions].to(torch.float32)  # [T, rope_head_dim]
    cos = row[..., :half]
    sin = row[..., half:]

    head_dim = k.shape[-1]
    out = k.clone()
    tail = out[..., head_dim - rope_head_dim:].reshape(*k.shape[:-1], half, 2)
    even = tail[..., 0]
    odd = tail[..., 1]
    new_even = even * cos - odd * sin
    new_odd = odd * cos + even * sin
    # Upstream's second round trip: "bf16 roundtrip for parity with the
    # reference / Q-side kernel numerics."
    #
    # Here it is a no-op, and deliberately kept anyway. Upstream quantises from
    # the fp32 value this produces, so the rounding decides its scale; this
    # route hands bf16 to the store operator and lets it quantise, so the same
    # rounding happens one line later either way. Removing it would be correct
    # today and silently wrong the moment anything downstream takes fp32 -- and
    # it is upstream's line, so it returns on the next re-transcription. Kept
    # with the reason rather than deleted and rediscovered.
    new_even = new_even.to(torch.bfloat16).to(torch.float32)
    new_odd = new_odd.to(torch.bfloat16).to(torch.float32)
    rotated = torch.stack((new_even, new_odd), dim=-1).reshape(
        *k.shape[:-1], rope_head_dim)
    out[..., head_dim - rope_head_dim:] = rotated
    return out


# ⚠️ The three constants below are read out of *another repository's* source
# (the custom operator under csrc/attention/indexer_compress_epilog_v2/). A
# cross-repo constant goes stale exactly like a local one and looks identical
# while it does, so each carries where it was read -- and the width rule is
# written as the operator's formula rather than as the 132 it happens to
# produce today, so a different head_dim moves both sides together.

#: The operator's scale grouping: its tiling computes
#: ``scaleCol_ = ceil(d_ / 128)`` over x's last dimension.
#: Read at ``indexer_compress_epilog_v2_tiling.cpp:49`` (PER_BLOCK).
SCALE_GROUP = 128

#: Bytes per scale column. Read at ``..._single_row.h:45``
#: (``RoundUp<float>(scaleCol) * sizeof(float)``) and cross-checked against
#: ``..._base.h:39`` (``B32_INTERPRE_TO_B8_RATIO = 4``). Spelled from
#: float32's own width so it moves if the scale dtype does.
SCALE_BYTES = torch.finfo(torch.float32).bits // 8

#: What the operator's registration accepts for x. Its compiled binList holds
#: bfloat16 and float16 only. This call site sends bfloat16; the assertion is
#: the registered set rather than our narrower choice, so it never refuses a
#: call the operator would have accepted.
REGISTERED_X_DTYPES = (torch.bfloat16, torch.float16)

#: 🔴 Not enforced here, and on our own roadmap: the tiling budgets UB as
#: ``rowFactor_ * RoundUp(d_, 16) * 2 * DOUBLE_BUFFER`` (tiling:161) -- x's
#: element width hard-coded to 2 bytes -- while the kernel's element type is a
#: compile-time parameter. Those agree today by coincidence, not by
#: constraint, and the condition that breaks it is "someone widens the dtype
#: support", which is what moving to an fp8 KV cache would do.


def check_epilog_cache(cache: torch.Tensor, x: torch.Tensor) -> None:
    """Refuse a cache the operator would mis-address rather than reject.

    The tiling (our own custom op, not CANN's) reads the cache as
    ``[num_blocks, cacheBs_, 1, cacheD_]`` with ``cacheD_`` in bytes, and it
    checks two of those itself -- rank 4, and the third dim being 1. What it
    does **not** check is the pair that matters most:

    * there is no dtype validation anywhere in the tiling;
    * ``d_`` is taken from **x**'s last dimension, never from the cache's, and
      the two are never compared.

    So a width disagreement is not rejected: the operator computes offsets
    from one number and writes into a buffer laid out by the other. That is
    the same shape as the SWA ``block_size`` guard one file over -- two
    quantities that index each other with nobody comparing them -- and the
    reason for a guard here is the same one: not that a particular defect is
    foreseen, but that the disagreement is unobservable.

    ⚠️ Three states, because the exact width relation is *not* established
    here. The operator's ``d_`` counts x's elements while ``cacheD_`` counts
    bytes of a quantised row plus its scale, and which of those the cache's
    last dim holds depends on a layout this code cannot read. A cache row
    narrower than x's row is wrong under every candidate encoding, so that
    raises. Anything else is reported once and passed, saying it was not
    checked -- never silently accepted as if it had been.
    """
    if cache.ndim != 4:
        raise ValueError(
            f"indexer_compress_cache must be rank 4, got {cache.ndim} "
            f"{tuple(cache.shape)} (the tiling reports this as EZ1008 "
            '"only support 4" without saying what it received)')
    if cache.shape[2] != 1:
        raise ValueError(
            f"indexer_compress_cache dim2 must be 1, got {cache.shape[2]} in "
            f"{tuple(cache.shape)}. The tiling reads [num_blocks, "
            "tokens_per_block, 1, bytes_per_token]; a dim2 above 1 means the "
            "axis order is not the one it assumes.")

    tokens_per_block, cache_width = int(cache.shape[1]), int(cache.shape[3])
    row_stride = int(cache.stride(0))
    if row_stride and row_stride < tokens_per_block * cache_width:
        raise ValueError(
            f"indexer_compress_cache block stride {row_stride} is smaller "
            f"than one block ({tokens_per_block} x {cache_width} bytes); "
            "consecutive blocks overlap and the operator would write over "
            "the previous block's tail.")

    # ⚠️ Strength, stated: bf16 is *this call site's* contract, not something
    # read out of the tiling. The operator's own source contains no dtype
    # validation anywhere, and its ``d_`` is an element count -- it must pick
    # a width from an assumed dtype to turn that into bytes. So nothing on the
    # far side would refuse another dtype; this pins what we promise to send.
    if x.dtype not in REGISTERED_X_DTYPES:
        raise ValueError(
            f"x is {x.dtype}; indexer_compress_epilog_v2 registers x as "
            f"{sorted(str(d) for d in REGISTERED_X_DTYPES)} only. The tiling "
            "itself validates no dtype at all -- it turns x's element count "
            "into bytes using an assumed width -- so an unregistered dtype "
            "is not refused on the far side.")

    x_width = int(x.shape[-1])
    expected = x_width + SCALE_BYTES * -(-x_width // SCALE_GROUP)
    if cache_width != expected:
        raise ValueError(
            f"indexer_compress_cache is {cache_width} bytes per token; this x "
            f"needs {expected} = {x_width} quantised values + {SCALE_BYTES} "
            f"bytes x {-(-x_width // SCALE_GROUP)} scale column(s). The "
            "operator takes its width from x alone and never compares the "
            "two, so a disagreement is written at the wrong offset rather "
            "than refused.")



def _log_epilog_args(k_cache: torch.Tensor, x: torch.Tensor,
                     slots: torch.Tensor) -> None:
    """Print what the epilog operator is about to receive.

    The tiling has three failure modes and only the first two announce
    themselves:

        not 4 dims          -> "the dim of indexer_compress_cache only support 4"
        4 dims, dim2 != 1   -> "the third dim ... must be 1"
        4 dims, dim2 == 1,
        wrong axis order    -> nothing.  ``layout`` has exactly one legal value,
                               so no parameter can express "the other order";
                               the tiling computes offsets as though the shape
                               were [blocks, bs, 1, D] and writes to the wrong
                               rows.

    So the third one has to be visible in the numbers, before and after any fix.
    The tiling also takes ``d_`` from ``x`` and ``cacheD_`` from the cache and
    never compares them, which is the same shape as two other seams found the
    same day.
    """
    import math

    cache = with_head_axis(k_cache.view(torch.uint8))
    xw = int(x.shape[-1]) if x.dim() else None
    cw = int(cache.shape[-1]) if cache.dim() else None
    st0 = int(cache.stride(0)) if cache.dim() else None
    bs = int(cache.shape[1]) if cache.dim() >= 2 else None
    prod = (bs * cw) if (bs is not None and cw is not None) else None
    # kernel derived this from the tiling source; keep it as the relation, not
    # as 132/128 -- the relation follows d_, a literal does not.
    want = (xw + 4 * math.ceil(xw / 128)) if xw else None
    logger.error(
        "【IDX-EPILOG】将传入的 cache: ndim=%d shape=%s dtype=%s "
        "(绑定前 k_cache ndim=%d shape=%s dtype=%s) | x: shape=%s dtype=%s | "
        "slot_mapping: shape=%s dtype=%s | layout=2 | dim2==1? %s | "
        "x末维=%s cache末维=%s 等式 d+4*ceil(d/128)=%s %s | "
        "stride(0)=%s cacheBs*cacheD=%s %s",
        cache.dim(), tuple(cache.shape), cache.dtype,
        k_cache.dim(), tuple(k_cache.shape), k_cache.dtype,
        tuple(x.shape), x.dtype, tuple(slots.shape), slots.dtype,
        (cache.shape[2] == 1) if cache.dim() >= 3 else "算不出来",
        xw, cw, want, ("✅相等" if cw == want else "🔴不等"),
        st0, prod,
        ("stride>=乘积" if (st0 is not None and prod is not None
                            and st0 >= prod) else "🔴 stride<乘积"))

def _checked_cache(k_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The cache as the operator will see it, checked against the x beside it."""
    cache = with_head_axis(k_cache.view(torch.uint8))
    check_epilog_cache(cache, x)
    return cache


def with_head_axis(cache: torch.Tensor) -> torch.Tensor:
    """Put back the unit head axis the kv binding dropped.

    ``indexer_compress_epilog_v2`` takes a rank-4 cache. The operator says so
    only as ``EZ1008: the dim of indexer_compress_cache only support 4``,
    without saying what it got -- but V4.0 drives the *same* operator and its
    warmup states the rank outright::

        device_op.py:1150  dummy_cache_shape = (1, 1, 1, hidden_states.shape[-1])
        device_op.py:1152  indexer_compress_epilog_v2(..., layout=2)

    V4.1 arrives at rank 3 instead, because binding drops the unit head axis
    (``kv_binding._drop_head_axis``, and upstream's own ``squeeze(1)``).
    V4.0 never loses it: its cache comes from ``unfold_kvcache`` and goes to
    the operator untouched.

    ⚠️ What V4.0's warmup does *not* establish is the axis **order** -- all
    four of its dims are 1. So this restores the axis that was dropped, at the
    position it was dropped from, and nothing else. Reshaping into an axis
    order that looks right would be a new assumption wearing the same shape,
    and the operator would accept it: rank is all it checks here.

    ``HEAD_AXIS`` is imported rather than written as -2, so the drop and this
    restore cannot come apart.
    """
    from vllm_ascend.models.deepseek_v4_1.kv_binding import HEAD_AXIS

    if cache.ndim == 4:
        return cache
    if cache.ndim != 3:
        raise ValueError(
            f"indexer k cache has rank {cache.ndim} {tuple(cache.shape)}; "
            "indexer_compress_epilog_v2 takes rank 4, and only a rank-3 cache "
            "(the bound form, with its unit head axis dropped) can be "
            "restored to one without inventing an axis order.")
    return cache.unsqueeze(HEAD_AXIS)


def _page_size_of(slot_mapping: torch.Tensor) -> int:
    """The storage page the mapping was built against, found by identity.

    ``format_dsa_slot_mapping`` splits a flat slot with the *builder's own*
    ``storage_block_size``, and every group has its own. So inverting it needs
    that group's number, and there is no way to recover it from the tensor:
    a ``(block, offset)`` pair is valid arithmetic under any page size, and the
    wrong one produces in-range slots belonging to other tokens.

    The identity walk that finds the owning metadata now lives in
    :func:`~vllm_ascend.models.deepseek_v4_1.compressed_slots.owning_req_metadata`,
    because the compressed-stream rebuild needs the same object for its block
    table. Two copies of a walk that must not drift is how the page size ends
    up taken from a sibling group by one caller and not the other.
    """
    from vllm_ascend.models.deepseek_v4_1.compressed_slots import \
        owning_req_metadata

    return int(owning_req_metadata(slot_mapping)[1].storage_block_size)


def flatten_block_offset_slots(slot_mapping: torch.Tensor) -> torch.Tensor:
    """``[T, 2]`` ``(block, offset)`` back to the flat slots it was split from.

    The exact inverse of ``DsaAttnKvPlan.format_dsa_slot_mapping``, including
    its sentinel: that function writes ``-1`` into *both* columns for an
    invalid slot, so a row is invalid iff its block is negative, and stays one
    value rather than becoming ``-block_size - 1``.
    """
    blocks, offsets = slot_mapping[:, 0], slot_mapping[:, 1]
    page = _page_size_of(slot_mapping)
    flat = blocks.to(torch.int64) * page + offsets.to(torch.int64)
    return torch.where(blocks >= 0, flat,
                       torch.full_like(flat, -1)).to(slot_mapping.dtype)


def indexer_k_norm_rope_store_ascend(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    use_fp4_cache: bool,
) -> None:
    """k_norm -> RoPE -> quant -> paged store, the Ascend route.

    Same signature and same ``None`` return as upstream's, so it can be bound
    over the name the V4.1 layer already calls.
    """
    if use_fp4_cache:
        raise NotImplementedError(_MXFP4_UNREACHABLE)

    from vllm_ascend.models.deepseek_v4_1.compressed_slots import (
        check_per_token, layer_owning, per_token_compressed_slots)

    # Kept under its own name: the rebuild below is matched to the metadata
    # object by identity, and flattening makes a new tensor.
    handed = kv_slot_mapping
    num_tokens = handed.shape[0]
    if k_pre.ndim != 2 or k_pre.shape[1] != INDEX_HEAD_DIM:
        raise ValueError(
            f"indexer k_pre must be [num_tokens, {INDEX_HEAD_DIM}], got "
            f"{tuple(k_pre.shape)}")
    if handed.ndim == 2 and handed.shape[-1] == 2:
        pass
    elif handed.ndim != 1:
        raise ValueError(
            "indexer kv_slot_mapping must be flat [num_tokens] or "
            f"[num_tokens, 2] (block, offset); got "
            f"{tuple(handed.shape)}.")
    if num_tokens == 0:
        return

    if compress_ratio > 1:
        # 🔴 The second consumer of the same defect. At a compressing ratio
        # the builder publishes V4.0's DENSE mapping (one row per compressed
        # state) while this writer, like the compressed-stream one, indexes
        # per token -- group_boundary_slots below tests
        # ``(positions[t] + 1) % ratio`` at index t. See compressed_slots.
        #
        # ⚠️ No cache cross-check on the page here, unlike the compressed
        # stream: the indexer's cache reaches this function in two ranks and
        # carries its fp8 scales alongside the values, so ``shape[1]`` is not
        # a second source for the page until that layout is read rather than
        # assumed. One source, said out loud, beats a check on a guess.
        kv_slot_mapping, layers = per_token_compressed_slots(
            handed, positions, compress_ratio)
        num_tokens = kv_slot_mapping.shape[0]
    else:
        # Ratio 1: every token is its own group, so the builder's per-token
        # mapping is already the right convention (dsa_v1.py:1298 fills it
        # exactly in this case); only the (block, offset) form comes off.
        if handed.ndim == 2:
            kv_slot_mapping = flatten_block_offset_slots(handed)
            num_tokens = kv_slot_mapping.shape[0]
        layers = layer_owning(handed)

    slots = group_boundary_slots(kv_slot_mapping, positions[:num_tokens],
                                 compress_ratio)
    check_per_token(slots, positions, compress_ratio, "indexer_k_store",
                    layers, handed=handed)
    k = _rms_norm(k_pre[:num_tokens], rms_norm_weight, rms_norm_eps)
    # A latent stands for the first token of its group.
    compressed_pos = torch.div(positions[:num_tokens], compress_ratio,
                               rounding_mode="floor") * compress_ratio
    k = _rope_interleaved_tail(k, cos_sin_cache, compressed_pos,
                               INDEX_ROPE_HEAD_DIM)

    # Quantisation, value store and scale store, in one operator. The layout
    # matches upstream's fp8 branch byte for byte: whole-block values first,
    # then the per-token fp32 scales.
    # Bound once. The first version checked the cache against `k` while the
    # operator received `k.to(torch.bfloat16)` -- the guard reading a
    # different object than the call, which is the failure it exists to stop.
    # The widths matched, so only the dtype would have differed, and the
    # tiling has no dtype validation to catch what the guard then missed.
    x = k.to(torch.bfloat16)
    # 仪器打【将要被传进去的那个张量】，而且【打在守卫之前】：
    #   · 守卫之前 ⇒ 守卫 raise 时诊断数据仍然存在（守卫失败正是最需要这些数的时候）
    #   · 用同一条推导（with_head_axis(view(uint8))）⇒ 和 _checked_cache 交出的是同一个形状
    #   · 不重复调用 _checked_cache ⇒ 不把守卫跑两遍
    if os.environ.get("DSV41_SFM_CHECK", "") == "1":
        _log_epilog_args(k_cache, x, slots)
    torch.ops._C_ascend.indexer_compress_epilog_v2(
        indexer_compress_cache=_checked_cache(k_cache, x),
        x=x,
        slot_mapping=slots,
        layout=2,
    )


#: Where the V4.1 layer looks the name up. It is imported into the attention
#: module at import time, and the call site reads that module's globals, so
#: rebinding the attribute reaches it whenever this is applied -- the same
#: shape as the MoE router and indexer-q hooks, and unlike FusedMoEFactory.
_TARGET_MODULE = "vllm.models.deepseek_v4_1.attention"
_TARGET_NAME = "indexer_k_norm_rope_store"


def apply_indexer_k_store_patch() -> bool:
    """Point the V4.1 K producer at the Ascend route. Returns True if changed."""
    import importlib

    module = importlib.import_module(_TARGET_MODULE)
    if not hasattr(module, _TARGET_NAME):
        raise AttributeError(
            f"{_TARGET_MODULE} no longer imports {_TARGET_NAME!r}. Until this "
            "is repointed the kv-source layers call upstream's Triton kernel, "
            "which on Ascend is compiled by triton-ascend -- and a per-token "
            "paged store is the shape that aborts the process rather than "
            "raising.")
    if getattr(module, _TARGET_NAME) is indexer_k_norm_rope_store_ascend:
        return False
    setattr(module, _TARGET_NAME, indexer_k_norm_rope_store_ascend)
    return True
