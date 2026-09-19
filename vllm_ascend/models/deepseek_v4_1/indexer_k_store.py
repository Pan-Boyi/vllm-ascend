# SPDX-License-Identifier: Apache-2.0
"""Indexer K production for V4.1 kv-source layers, without Triton.

Upstream's ``indexer_k_norm_rope_store`` is one Triton program per token doing
k_norm, RoPE, quantisation and a paged store. There is only the Triton version,
and on triton-ascend a per-token paged store is a pointer-arithmetic shape that
aborts the compiler (``abort()``, not an exception), so an equivalent route is
taken instead.

The quantisation and the paged store are not rebuilt here: A5 has a fused
operator for both, ``indexer_compress_epilog_v2``, already used by the V4.0
path in ``vllm_ascend/device/device_op.py``. What this file adds is the part
upstream fuses in and the operator has no notion of -- the group gate, the
norm, and the rotation.

Five details are silent if wrong:

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

from functools import lru_cache

import torch

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

    The normalisation to exactly ``-1`` matters: the operator tests
    ``slot == -1`` and writes any other negative value out of bounds.
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

    The round trip is in the reference and in upstream's kernel: these values
    feed the absmax downstream, so fp32 precision would give a different scale.
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
    # Upstream's second bf16 round trip. A no-op here because the result is
    # cast to bf16 before the store anyway; kept for parity with upstream.
    new_even = new_even.to(torch.bfloat16).to(torch.float32)
    new_odd = new_odd.to(torch.bfloat16).to(torch.float32)
    rotated = torch.stack((new_even, new_odd), dim=-1).reshape(
        *k.shape[:-1], rope_head_dim)
    out[..., head_dim - rope_head_dim:] = rotated
    return out


# The constants below mirror the custom operator under
# csrc/attention/indexer_compress_epilog_v2/. The width rule is written as the
# operator's formula so a different head_dim moves both sides together.

#: The operator's scale grouping: its tiling computes
#: ``scaleCol_ = ceil(d_ / 128)`` over x's last dimension.
SCALE_GROUP = 128

#: Bytes per scale column (one float32 per column).
SCALE_BYTES = torch.finfo(torch.float32).bits // 8

#: What the operator's registration accepts for x. This call site sends
#: bfloat16; the check uses the registered set so it never refuses a call the
#: operator would have accepted. Note the tiling budgets UB assuming a 2-byte
#: x element, so widening the dtype support needs a tiling change too.
REGISTERED_X_DTYPES = (torch.bfloat16, torch.float16)


def check_epilog_cache(cache: torch.Tensor, x: torch.Tensor) -> None:
    """Refuse a cache the operator would mis-address rather than reject.

    The tiling reads the cache as ``[num_blocks, cacheBs_, 1, cacheD_]`` with
    ``cacheD_`` in bytes and checks the rank and the unit third dim itself.
    It validates no dtype, and takes ``d_`` from x's last dimension without
    comparing it to the cache's width, so a width disagreement would be
    written at the wrong offset rather than rejected. Every such mismatch
    raises here.
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



def _checked_cache(k_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The cache as the operator will see it, checked against the x beside it."""
    cache = with_head_axis(k_cache.view(torch.uint8))
    check_epilog_cache(cache, x)
    return cache


def with_head_axis(cache: torch.Tensor) -> torch.Tensor:
    """Put back the unit head axis the kv binding dropped.

    ``indexer_compress_epilog_v2`` takes a rank-4 cache (V4.0's warmup passes
    ``(1, 1, 1, D)`` with ``layout=2``), while V4.1's binding drops the unit
    head axis (``kv_binding._drop_head_axis``, like upstream's ``squeeze(1)``).
    Only that axis is restored, at the position it was dropped from; the
    operator checks nothing but rank, so no other reshaping is inferred.

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

    The owning metadata is found by
    :func:`~vllm_ascend.models.deepseek_v4_1.compressed_slots.owning_req_metadata`.
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
        # The builder publishes V4.0's dense mapping here while this writer
        # indexes per token; rebuild it (see compressed_slots). No cache
        # cross-check on the page: this cache's shape[1] is not established
        # as a page size for both ranks it arrives in.
        kv_slot_mapping, layers = per_token_compressed_slots(
            handed, positions, compress_ratio)
        num_tokens = kv_slot_mapping.shape[0]
    else:
        # Ratio 1: the builder's per-token mapping is already right; only the
        # (block, offset) form comes off.
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
    # Bound once so the guard checks the same tensor the operator receives.
    x = k.to(torch.bfloat16)
    if indexer_quant_cache_enabled():
        # Same x, same slots, same packed cache; only the quantiser changes.
        store_with_indexer_quant_cache(k_cache, x, slots)
        return
    torch.ops._C_ascend.indexer_compress_epilog_v2(
        indexer_compress_cache=_checked_cache(k_cache, x),
        x=x,
        slot_mapping=slots,
        layout=2,
    )


# ---------------------------------------------------------------------------
# F9: the same store through CANN's indexer_quant_cache
# (VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE=1).
#
# The call is the one card A6 ran and judged by bytes
# (unit_tests/card_A/a6_indexer_quant_cache.py, CARD_RESULTS.md §A6): the
# packed fp8 cache split into its value and scale views by
# ``split_indexer_cache``, x bf16 [T, 128], the int32 slots from
# ``group_boundary_slots``, quant_mode="fp8", round_scale=True, x_scale=1.0.
# On that call the operator wrote exactly the slots != -1 (357/357, 178/178),
# every scale was a power of two, and the bytes equalled upstream's quantiser
# except for an all-zero row (upstream floors amax at 1e-4: scale 2^-22;
# this operator writes scale 0; the values are 0 either way).
#
# Only the quantiser changes. k_norm, RoPE, the group gate and the slot
# rebuild above are still ours, so the rebind of upstream's Triton K producer
# (apply_indexer_k_store_patch, platform patch #7) is still what keeps the
# Triton kernel from running, flag on or off.
# ---------------------------------------------------------------------------

#: The cache element type A6 ran. The Ascend indexer cache is allocated with
#: it on A5 (``AscendDeepseekV41IndexerCache``, DSV4_COMPRESSED_CACHE). Any
#: other type means a different cache configuration -- the non-compressed
#: profile carries fp16 scales -- and the fp8 call has not been run on it.
QUANT_CACHE_DTYPE = torch.float8_e4m3fn


def indexer_quant_cache_enabled() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE == "1"


@lru_cache
def _indexer_quant_cache_op():
    """``torch.ops.cann_ops_transformer.indexer_quant_cache``, with the
    registration it needs. Resolved once (a failure is not cached).

    The op is registered by importing ``cann_ops_transformer`` (CANN 9.2 python
    package); its first *call* JIT-builds a small C++ binding through
    ``torch.utils.cpp_extension.load`` (torch's extensions dir, file-locked).
    """
    from importlib import import_module

    try:
        import_module("cann_ops_transformer")
        return torch.ops.cann_ops_transformer.indexer_quant_cache
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE=1 needs "
            "cann_ops_transformer.indexer_quant_cache from the CANN 9.2 "
            "toolkit's python package, and it could not be loaded. This is "
            "an environment problem (package missing or not on PYTHONPATH), "
            "not an operator refusal; unset the flag to use "
            "indexer_compress_epilog_v2.") from exc


def quant_cache_views(k_cache: torch.Tensor, x: torch.Tensor):
    """``(key, key_scale)`` views of the packed cache, as A6 passed them.

    Refuses what A6 did not run rather than adapting to it. All checks read
    only shapes, dtypes and strides, so nothing here syncs the device or
    breaks a graph capture.
    """
    from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import \
        split_indexer_cache
    from vllm_ascend.models.deepseek_v4_1.kv_binding import HEAD_AXIS

    if k_cache.dtype != QUANT_CACHE_DTYPE:
        raise ValueError(
            f"indexer k cache is {k_cache.dtype}; the indexer_quant_cache "
            f"path was verified (card A6) on the {QUANT_CACHE_DTYPE} packed "
            "cache the A5 compressed profile allocates. Another element type "
            "is another cache configuration, and quant_mode='fp8' has not "
            "been run on it. Unset VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE.")
    # The unit head axis, if the binding left it on (see with_head_axis).
    packed = k_cache.squeeze(HEAD_AXIS) if (
        k_cache.ndim == 4 and k_cache.shape[HEAD_AXIS] == 1) else k_cache
    if packed.ndim != 3:
        raise ValueError(
            f"indexer k cache is {tuple(k_cache.shape)}; expected the packed "
            "[num_blocks, tokens_per_block, width] (optionally with a unit "
            "head axis) that A6 ran.")
    tokens_per_block, width = int(packed.shape[1]), int(packed.shape[2])
    # split_indexer_cache places block b at b * tokens_per_block * width, and
    # so does the reader. A different block stride would put this writer and
    # the reader in different places.
    if packed.shape[0] > 1 and int(packed.stride(0)) != tokens_per_block * width:
        raise ValueError(
            f"indexer k cache block stride {packed.stride(0)} is not "
            f"{tokens_per_block} x {width}; split_indexer_cache (and the "
            "reader) assume back-to-back blocks, so the operator would write "
            "where nothing reads.")
    if packed.stride(2) != 1 or packed.stride(1) != width:
        raise ValueError(
            f"indexer k cache is not contiguous within a block (strides "
            f"{tuple(packed.stride())}); indexer_quant_cache accepts "
            "non-contiguity only across blocks.")
    key, scale = split_indexer_cache(packed)
    if x.ndim != 2 or x.dtype != torch.bfloat16 or \
            int(x.shape[1]) != int(key.shape[-1]):
        raise ValueError(
            f"x is {x.dtype} {tuple(x.shape)}; the verified call writes bf16 "
            f"[T, {int(key.shape[-1])}] into this cache. The operator only "
            "requires headDim >= d, so a narrower x would be written short "
            "rather than refused.")
    if int(scale.shape[-1]) != 1:
        raise ValueError(
            f"the scale view has {int(scale.shape[-1])} columns; fp8 mode "
            "writes one float32 scale per token.")
    return key, scale


def store_with_indexer_quant_cache(k_cache: torch.Tensor, x: torch.Tensor,
                                   slots: torch.Tensor) -> None:
    """Quantise ``x`` and store it at ``slots`` through indexer_quant_cache.

    ``slots`` is passed exactly as ``group_boundary_slots`` built it: int32,
    ``-1`` = skip, the same sentinel the operator documents. It is not cast:
    the operator takes int32 only, and so does the epilog operator this
    replaces, so a different dtype here is an upstream change to look at, not
    one to convert away.
    """
    if slots.dtype != torch.int32 or slots.ndim != 1 or \
            int(slots.shape[0]) != int(x.shape[0]):
        raise ValueError(
            f"slots are {slots.dtype} {tuple(slots.shape)} for x "
            f"{tuple(x.shape)}; indexer_quant_cache takes int32 [T], one "
            "per row of x.")
    key, scale = quant_cache_views(k_cache, x)
    _indexer_quant_cache_op()(
        cache=key,
        cache_scale=scale,
        x=x,
        slot_mapping=slots,
        quant_mode="fp8",
        round_scale=True,
        x_scale=1.0,
    )


#: Where the V4.1 layer looks the name up. It is imported into the attention
#: module at import time and called through that module's globals, so
#: rebinding the attribute reaches the call site.
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
            "whose per-token paged store aborts the process under "
            "triton-ascend.")
    if getattr(module, _TARGET_NAME) is indexer_k_norm_rope_store_ascend:
        return False
    setattr(module, _TARGET_NAME, indexer_k_norm_rope_store_ascend)
    return True
