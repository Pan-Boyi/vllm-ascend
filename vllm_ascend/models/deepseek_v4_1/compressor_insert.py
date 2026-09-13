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

    rope_quant_insert(latent, positions, cos_sin_cache, kv_cache,
                      flatten_slot_mapping(slot_mapping, kv_cache),
                      compress_ratio, fp8_scale)


def flatten_slot_mapping(slot_mapping: torch.Tensor,
                         cache: torch.Tensor | None = None) -> torch.Tensor:
    """``(block, offset)`` to flat slots, or the mapping unchanged.

    The single place this line converts a slot mapping for the compressor
    ops. Both of them want flat slots and there is exactly one way to get
    there; a second implementation is how the third consumer ended up
    unconverted while the second was fixed.
    """
    if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
        return slot_mapping
    return _flatten_for_cache(slot_mapping, cache)


def _flatten_for_cache(slot_mapping: torch.Tensor,
                       cache: torch.Tensor | None) -> torch.Tensor:
    """Flatten ``(block, offset)`` with a page size two sources agree on.

    ⚠️ Three-state on the cross-check. With a cache in hand the page size is
    confirmed against it; without one -- ``fused_save_compress_norm`` may be
    called with ``state_cache=None`` -- only the metadata's number is
    available, and that is reported rather than passed off as checked.
    """
    from vllm_ascend.models.deepseek_v4_1.indexer_k_store import _page_size_of

    from_metadata = _page_size_of(slot_mapping)
    if cache is None:
        _log_unchecked_page(from_metadata)
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


_LOGGED_UNCHECKED_PAGES: set = set()


def _log_unchecked_page(page: int) -> None:
    """Say when the page size had only one source, once per size."""
    if page in _LOGGED_UNCHECKED_PAGES:
        return
    _LOGGED_UNCHECKED_PAGES.add(page)
    from vllm.logger import logger

    logger.info(
        "DSV4.1 compressor: flattening a slot mapping with page size %d taken "
        "from metadata alone -- no cache was passed, so it is NOT "
        "cross-checked. A (block, offset) pair is valid arithmetic under any "
        "page size and the wrong one lands on another token's slot.", page)


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

    The second consumer in this module, found the way the first one was: an
    ``assert num_tokens <= min(...)`` firing because ``slot_mapping.numel()``
    is ``2T`` for a paired mapping. Fixing ``rope_quant_insert`` alone left
    this one, which is what "one mapping, N consumers" looks like when the
    fixes are per-consumer.

    ``state_cache`` may be ``None``, so the page size cannot always be
    cross-checked; ``flatten_slot_mapping`` reports that case rather than
    treating it as checked.
    """
    from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import \
        fused_save_compress_norm

    fused_save_compress_norm(
        kv_score, positions, state_cache,
        flatten_slot_mapping(slot_mapping, state_cache),
        query_start_loc, token_to_req_indices, rms_norm_weight, rms_norm_eps,
        compress_ratio, latent_out)


#: The compressor imports these names into its own module at import time and
#: calls them from there, so rebinding those attributes reaches the call sites.
#:
#: Both of the module's public functions that take a slot mapping are here. A
#: test enumerates them from upstream's source rather than trusting this list,
#: because the list is what was incomplete last time.
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
                "tokens -- caught by its own assertion, but as a bare "
                "AssertionError with none of the three numbers in it.")
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
