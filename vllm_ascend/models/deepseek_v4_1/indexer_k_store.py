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


def _page_size_of(slot_mapping: torch.Tensor) -> int:
    """The storage page the mapping was built against, found by identity.

    ``format_dsa_slot_mapping`` splits a flat slot with the *builder's own*
    ``storage_block_size``, and every group has its own. So inverting it needs
    that group's number, and there is no way to recover it from the tensor:
    a ``(block, offset)`` pair is valid arithmetic under any page size, and the
    wrong one produces in-range slots belonging to other tokens.

    So the page size is taken from the metadata object that *is* carrying this
    tensor -- matched by object identity, not by layer name or by shape. A name
    match would accept a sibling group's metadata, which is exactly the failure
    mode ("addresses a different cache") this has to exclude. No match is a
    refusal, never a default: a guessed page size here writes index keys over
    other tokens' entries with nothing raised.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        # Outside a forward, get_forward_context asserts rather than returning
        # None. Same answer either way -- there is nothing to take the page
        # size from -- so it is reported as that rather than escaping as an
        # AssertionError from a frame that says nothing about slot mappings.
        context = None
    attn_metadata = getattr(context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        raise ValueError(
            "indexer slot mapping is (block, offset) but there is no attention "
            "metadata dict to take the storage page size from.")

    seen = []
    for layer_name, metadata in attn_metadata.items():
        req = getattr(metadata, "req_metadata", None)
        if req is None:
            continue
        page = getattr(req, "storage_block_size", None)
        if page is None:
            continue
        candidates = (getattr(req, "slot_mapping", None),
                      getattr(req, "token_slot_mapping", None))
        if any(c is slot_mapping for c in candidates):
            return int(page)
        seen.append((layer_name, page))

    raise ValueError(
        "the indexer's (block, offset) slot mapping is not the slot mapping of "
        "any attention metadata in this forward, so the storage page size it "
        "was split with is unknown. Inverting it with another group's page "
        f"size writes into other tokens' entries silently. Groups seen: {seen}")


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

    num_tokens = kv_slot_mapping.shape[0]
    if k_pre.ndim != 2 or k_pre.shape[1] != INDEX_HEAD_DIM:
        raise ValueError(
            f"indexer k_pre must be [num_tokens, {INDEX_HEAD_DIM}], got "
            f"{tuple(k_pre.shape)}")
    if kv_slot_mapping.ndim == 2 and kv_slot_mapping.shape[-1] == 2:
        kv_slot_mapping = flatten_block_offset_slots(kv_slot_mapping)
        num_tokens = kv_slot_mapping.shape[0]
    elif kv_slot_mapping.ndim != 1:
        raise ValueError(
            "indexer kv_slot_mapping must be flat [num_tokens] or "
            f"[num_tokens, 2] (block, offset); got "
            f"{tuple(kv_slot_mapping.shape)}.")
    if num_tokens == 0:
        return

    slots = group_boundary_slots(kv_slot_mapping, positions[:num_tokens],
                                 compress_ratio)
    k = _rms_norm(k_pre[:num_tokens], rms_norm_weight, rms_norm_eps)
    # A latent stands for the first token of its group.
    compressed_pos = torch.div(positions[:num_tokens], compress_ratio,
                               rounding_mode="floor") * compress_ratio
    k = _rope_interleaved_tail(k, cos_sin_cache, compressed_pos,
                               INDEX_ROPE_HEAD_DIM)

    # Quantisation, value store and scale store, in one operator. The layout
    # matches upstream's fp8 branch byte for byte: whole-block values first,
    # then the per-token fp32 scales.
    # 算子按【维数】校验这个 cache（tiling 里 "the dim of indexer_compress_cache
    # only support 4"），而它失败时只说"只支持 4"，不说【实际是几】。
    # ⇒ 把实参的形状打出来，否则"这个张量几维"要再烧一发起飞才能知道。
    _viewed = k_cache.view(torch.uint8)
    if os.environ.get("DSV41_SFM_CHECK", "") == "1":
        # tiling 的三档，前两档靠它自己的报错文本区分，第三档【没有任何 host 校验】：
        #   不是 4 维              -> "dim ... only support 4"
        #   4 维但 dim2 != 1       -> "third dim ... must be 1"
        #   4 维且 dim2 == 1 但序错 -> 🔴 不报错，静默按 [blocks,bs,1,D] 算 offset
        # 而 layout 只有一个合法值(2)，没有参数能表达"另一种序"。
        # ⇒ 所以要在【修之前】就能看见第三档，得自己打这四个数。
        # 另：tiling 里 d_ 取自【x】的最后一维、不取 cache 的 ⇒ 两者的宽度【互不校验】。
        _xw = int(k.shape[-1]) if k.dim() else None
        _cw = int(_viewed.shape[-1]) if _viewed.dim() else None
        _st0 = int(_viewed.stride(0)) if _viewed.dim() else None
        _bs = int(_viewed.shape[1]) if _viewed.dim() >= 2 else None
        _prod = (_bs * _cw) if (_bs is not None and _cw is not None) else None
        logger.error(
            "【IDX-EPILOG】indexer_compress_cache: ndim=%d shape=%s dtype=%s "
            "(view 前 ndim=%d shape=%s dtype=%s) | x: shape=%s dtype=%s | "
            "slot_mapping: shape=%s dtype=%s | layout=2 | "
            "🔴 x末维=%s cache末维=%s %s | cache.stride(0)=%s cacheBs*cacheD=%s %s",
            _viewed.dim(), tuple(_viewed.shape), _viewed.dtype,
            k_cache.dim(), tuple(k_cache.shape), k_cache.dtype,
            tuple(k.shape), k.dtype, tuple(slots.shape), slots.dtype,
            _xw, _cw, ("一致" if _xw == _cw else "🔴不一致(无人互校)"),
            _st0, _prod,
            ("stride>=乘积" if (_st0 is not None and _prod is not None
                                and _st0 >= _prod) else "🔴 stride<乘积"))
    torch.ops._C_ascend.indexer_compress_epilog_v2(
        indexer_compress_cache=_viewed,
        x=k.to(torch.bfloat16),
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
