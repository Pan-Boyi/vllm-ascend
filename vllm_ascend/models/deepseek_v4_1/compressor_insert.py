# SPDX-License-Identifier: Apache-2.0
"""The compressor's latent publish, given the slot mapping it can read.

The SWA cache insert takes the ``(block, offset)`` slot mapping (the A5 bf16
default), while upstream's ``rope_quant_insert`` and
``fused_save_compress_norm`` kernels take flat slots and compute
``slot // CACHE_BLOCK`` / ``slot % CACHE_BLOCK`` themselves. Handed the
``[T, 2]`` form, upstream reads ``2T`` tokens and fails its own
``num_tokens <= min(...)`` assertion without naming the numbers.

The page size used for flattening is cross-checked between the metadata and
the cache the kernel will index: a ``(block, offset)`` pair is valid arithmetic
under any page size, so the wrong one silently lands on another token's slot.
"""

from __future__ import annotations

import torch


def rope_quant_insert_ascend(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
    fp8_scale: torch.Tensor | None = None,
) -> None:
    """Upstream's publish, with the slot mapping converted to the flat form.

    Same signature and same ``None`` return, so it can be bound over the name
    the compressor already calls. Everything except the mapping is handed
    through untouched: the kernel choice, the dtype branches and the
    assertions are upstream's and stay upstream's.

    The mapping is where the two ports disagree, and at a compressing ratio it
    is rebuilt rather than reformatted -- see
    :mod:`vllm_ascend.models.deepseek_v4_1.compressed_slots`.
    """
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import \
        rope_quant_insert

    from vllm_ascend.models.deepseek_v4_1.compressed_slots import (
        check_per_token, layer_owning, per_token_compressed_slots)

    if compress_ratio > 1:
        # The builder publishes V4.0's dense mapping (one row per compressed
        # state) while this kernel indexes per token; rebuild it per token.
        flat, layers = per_token_compressed_slots(slot_mapping, positions,
                                                  compress_ratio, kv_cache)
    else:
        # Ratio 1 has no group boundaries, so the builder's per-token mapping
        # is already right; only the (block, offset) form has to come off.
        flat = flatten_slot_mapping(slot_mapping, kv_cache)
        layers = layer_owning(slot_mapping)
    check_per_token(flat, positions, compress_ratio, "rope_quant_insert",
                    layers, handed=slot_mapping)
    rope_quant_insert(latent, positions, cos_sin_cache, kv_cache, flat,
                      compress_ratio, fp8_scale)


def flatten_slot_mapping(slot_mapping: torch.Tensor,
                         cache: torch.Tensor | None = None) -> torch.Tensor:
    """``(block, offset)`` to flat slots, or the mapping unchanged.

    The single conversion used by both compressor ops.
    """
    if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
        return slot_mapping
    return _flatten_for_cache(slot_mapping, cache)


def _flatten_for_cache(slot_mapping: torch.Tensor,
                       cache: torch.Tensor | None) -> torch.Tensor:
    """Flatten ``(block, offset)`` with a page size two sources agree on.

    Without a cache (``fused_save_compress_norm`` may get
    ``state_cache=None``) only the metadata's page size is available.
    """
    from vllm_ascend.models.deepseek_v4_1.indexer_k_store import _page_size_of

    from_metadata = _page_size_of(slot_mapping)
    if cache is None:
        return _flatten(slot_mapping, from_metadata)
    from_cache = int(cache.shape[1])
    if from_metadata != from_cache:
        raise ValueError(
            f"the metadata carrying this slot mapping says its page is "
            f"{from_metadata} states, while the cache the kernel will index "
            f"has {from_cache} per block ({tuple(cache.shape)}). The "
            "kernel derives block and offset from a flat slot using its own "
            "block length, so flattening with the other number puts every "
            "row in a real but wrong place.")

    return _flatten(slot_mapping, from_cache)


def _flatten(slot_mapping: torch.Tensor, page: int) -> torch.Tensor:
    blocks, offsets = slot_mapping[:, 0], slot_mapping[:, 1]
    flat = blocks.to(torch.int64) * page + offsets.to(torch.int64)
    # -1 stays one value. The split writes -1 into *both* columns, so plain
    # arithmetic would give -page - 1: an ordinary negative that reads as a
    # valid-looking slot rather than as the sentinel it came from.
    return torch.where(blocks >= 0, flat,
                       torch.full_like(flat, -1)).to(slot_mapping.dtype)


def fused_save_compress_norm_ascend(
    kv_score: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    compress_ratio: int,
    latent_out: torch.Tensor,
) -> None:
    """The pooling half of the compressor, with a flat slot mapping.

    ``state_cache`` may be ``None``, in which case the page size cannot be
    cross-checked against the cache.
    """
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import \
        fused_save_compress_norm

    flat = flatten_slot_mapping(slot_mapping, state_cache)
    if state_cache is not None and compress_ratio > 1:
        from vllm_ascend.models.deepseek_v4_1.compressed_slots import \
            require_ring_capacity_divides

        require_ring_capacity_divides(int(state_cache.shape[1]),
                                      compress_ratio)
    fused_save_compress_norm(
        kv_score, positions, state_cache, flat,
        query_start_loc, token_to_req_indices, rms_norm_weight, rms_norm_eps,
        compress_ratio, latent_out)


#: The compressor imports these names into its own module at import time and
#: calls them from there, so rebinding those attributes reaches the call sites.
#: A test checks this list against upstream's source.
_TARGET_MODULE = "vllm.models.deepseek_v4_1.compressor"
_REBINDS = (
    ("rope_quant_insert", "rope_quant_insert_ascend"),
    ("fused_save_compress_norm", "fused_save_compress_norm_ascend"),
)


def apply_compressor_insert_patch() -> bool:
    """Point the compressor's slot-mapping consumers at the Ascend route."""
    import importlib

    module = importlib.import_module(_TARGET_MODULE)
    changed = False
    for upstream_name, ascend_name in _REBINDS:
        if not hasattr(module, upstream_name):
            raise AttributeError(
                f"{_TARGET_MODULE} no longer imports {upstream_name!r}. Until "
                "this is repointed the compressor hands upstream a "
                "(block, offset) mapping, which it reads as twice as many "
                "tokens.")
        replacement = globals()[ascend_name]
        if getattr(module, upstream_name) is not replacement:
            setattr(module, upstream_name, replacement)
            changed = True
    return changed


def compressor_insert_patched() -> bool:
    """Whether every slot-mapping consumer goes through this route."""
    import importlib

    module = importlib.import_module(_TARGET_MODULE)
    return all(getattr(module, upstream, None) is globals()[ascend]
               for upstream, ascend in _REBINDS)
