# SPDX-License-Identifier: Apache-2.0
"""The compressor's latent publish, given the slot mapping it can read.

Third consumer of one slot mapping, third expectation of its shape. The SWA
cache insert wants ``(block, offset)`` -- the A5 bf16 default -- the indexer's
epilog wants flat slots and divides by the page itself, and so does upstream's
``rope_quant_insert``: its kernel computes ``slot // CACHE_BLOCK`` and
``slot % CACHE_BLOCK`` from a flat value.

Handed the ``[T, 2]`` form, upstream does not misread it quietly. ``num_tokens
= slot_mapping.numel()`` becomes ``2T`` and its own assertion fires::

    assert num_tokens <= min(latent.shape[0], positions.numel())

That assertion is right and it is what found this. What it does not do is say
which three numbers disagreed, so it arrives as a bare ``AssertionError``
several frames inside upstream.

⚠️ Note what is *not* wrong here: the compressor reads
``k_cache_metadata.slot_mapping`` for its own cache prefix and writes that
cache, exactly as the indexer does. The object is right; only the format is
not -- same as the indexer, and worth stating because "the mapping is wrong"
and "the mapping is in the other form" lead to different fixes.

The page size is cross-checked rather than taken from one place. The metadata
that carries the mapping knows the page it was split with, and the cache the
kernel will index knows its own block length; on this path they must be the
same number. Two sources that must agree beats either alone -- and the failure
they exclude is silent, since a ``(block, offset)`` pair is valid arithmetic
under any page size and the wrong one lands on another token's slot.
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
    """
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import \
        rope_quant_insert

    if slot_mapping.ndim == 2 and slot_mapping.shape[-1] == 2:
        slot_mapping = _flatten_for_cache(slot_mapping, kv_cache)

    rope_quant_insert(latent, positions, cos_sin_cache, kv_cache,
                      slot_mapping, compress_ratio, fp8_scale)


def _flatten_for_cache(slot_mapping: torch.Tensor,
                       kv_cache: torch.Tensor) -> torch.Tensor:
    """Flatten ``(block, offset)`` with a page size two sources agree on."""
    from vllm_ascend.models.deepseek_v4_1.indexer_k_store import _page_size_of

    from_metadata = _page_size_of(slot_mapping)
    from_cache = int(kv_cache.shape[1])
    if from_metadata != from_cache:
        raise ValueError(
            f"the metadata carrying this slot mapping says its page is "
            f"{from_metadata} states, while the cache the kernel will index "
            f"has {from_cache} per block ({tuple(kv_cache.shape)}). The "
            "kernel derives block and offset from a flat slot using its own "
            "block length, so flattening with the other number puts every "
            "row in a real but wrong place.")

    blocks, offsets = slot_mapping[:, 0], slot_mapping[:, 1]
    flat = blocks.to(torch.int64) * from_cache + offsets.to(torch.int64)
    # -1 stays one value. The split writes -1 into *both* columns, so plain
    # arithmetic would give -block_size - 1: an ordinary negative that reads
    # as a valid-looking slot rather than as the sentinel it came from.
    return torch.where(blocks >= 0, flat,
                       torch.full_like(flat, -1)).to(slot_mapping.dtype)


#: The compressor imports the name into its own module at import time and
#: calls it from there, so rebinding that attribute reaches the call site.
_TARGET_MODULE = "vllm.models.deepseek_v4_1.compressor"
_TARGET_NAME = "rope_quant_insert"


def apply_compressor_insert_patch() -> bool:
    """Point the compressor's publish at the Ascend route. True if changed."""
    import importlib

    module = importlib.import_module(_TARGET_MODULE)
    if not hasattr(module, _TARGET_NAME):
        raise AttributeError(
            f"{_TARGET_MODULE} no longer imports {_TARGET_NAME!r}. Until this "
            "is repointed the compressor hands upstream a (block, offset) "
            "mapping, which it reads as twice as many tokens -- caught by its "
            "own assertion, but as a bare AssertionError with none of the "
            "three numbers in it.")
    if getattr(module, _TARGET_NAME) is rope_quant_insert_ascend:
        return False
    setattr(module, _TARGET_NAME, rope_quant_insert_ascend)
    return True


def compressor_insert_patched() -> bool:
    """Whether the compressor's publish currently goes through this route."""
    import importlib

    module = importlib.import_module(_TARGET_MODULE)
    return getattr(module, _TARGET_NAME, None) is rope_quant_insert_ascend
