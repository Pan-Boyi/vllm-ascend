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

#: Suffix marking a key whose value is a process-local identifier -- a
#: pointer, a handle, an address-derived id. Such a key can never be equal
#: across ranks or across runs, so a bit-exact comparator must skip it. Said
#: in the key rather than in a list of exceptions: the list is the form of
#: "remembering" that gets forgotten, and a baseline that always carries a
#: few known differences stops being usable as a baseline.
PROCESS_LOCAL = ".process_local"


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
        # 🔴 Not the mapping we were handed. At a compressing ratio the
        # builder publishes V4.0's DENSE mapping -- one row per compressed
        # state -- while this kernel indexes per token
        # (fused_compress_quant_cache.py:354-394). Reading the dense one per
        # token skips entry #0 entirely and sends a later token to flat slot
        # 0, another request's row. See compressed_slots for the derivation
        # and the measurements.
        flat, layers = per_token_compressed_slots(slot_mapping, positions,
                                                  compress_ratio, kv_cache)
    else:
        # Ratio 1 has no group boundaries, so the builder's per-token mapping
        # (dsa_v1.py:1298, the branch that does run) is already the right
        # convention; only the (block, offset) form has to come off.
        flat = flatten_slot_mapping(slot_mapping, kv_cache)
        layers = layer_owning(slot_mapping)
    check_per_token(flat, positions, compress_ratio, "rope_quant_insert",
                    layers, handed=slot_mapping)
    _tap_write_side("rope_quant_insert", flat, latent, kv_cache)
    rope_quant_insert(latent, positions, cos_sin_cache, kv_cache, flat,
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

    flat = flatten_slot_mapping(slot_mapping, state_cache)
    if state_cache is not None and compress_ratio > 1:
        from vllm_ascend.models.deepseek_v4_1.compressed_slots import (
            check_ring_mapping, layer_owning)

        # ⚠️ A different criterion from the compressed stream's, deliberately.
        # The ring is written at every token, so `valid ⊆ boundary` would
        # refuse a correct mapping; what it must satisfy is the addressing
        # its reader assumes, `slot % capacity == position % capacity`.
        # Until now that distinction was held by a paragraph of prose, and
        # prose does not go red.
        check_ring_mapping(flat, positions, int(state_cache.shape[1]),
                           "fused_save_compress_norm",
                           layer_owning(slot_mapping),
                           query_start_loc=query_start_loc,
                           compress_ratio=compress_ratio)
    _tap_write_side("fused_save_compress_norm", flat, latent_out, state_cache)
    fused_save_compress_norm(
        kv_score, positions, state_cache, flat,
        query_start_loc, token_to_req_indices, rms_norm_weight, rms_norm_eps,
        compress_ratio, latent_out)


_WRITE_CALLS: dict = {}


def _tap_write_side(who: str, flat_slots, latent, cache) -> None:
    """Where this call is about to write, recorded before it writes.

    🔴 The read side says the compressed stream's entry #0 comes back all
    zeros on every even-numbered prefill while entry #1 is correct. That is
    consistent with two different things -- it was never written, or it was
    written somewhere else -- and only the write side can tell them apart.
    So: the flat slots this call is handed, and the latent it is handed,
    taken **before** the operator runs.

    ⚠️ Tagged ``compressor`` with a call index and the cache's data pointer
    rather than a layer number, because this function does not know its
    layer: it is a free function rebound over the compressor's own import.
    The pointer is what lets a reader attribute a call to a cache; the index
    is what orders calls within a forward. Neither is guessed at.
    """
    from vllm_ascend.models.deepseek_v4_1 import align_taps, read_taps

    if not align_taps.alignment_active() or not read_taps.read_tap_layers():
        return
    try:
        index = _WRITE_CALLS.get(who, 0)
        _WRITE_CALLS[who] = index + 1
        if index == 100:
            from vllm.logger import logger

            logger.warning(
                "DSV4.1 compressor tap: call index reached 100 for %r in one "
                "forward, and the key is formatted %%02d, so indices from "
                "here on collide with earlier ones. The counter resets per "
                "forward, so this means far more compressor calls per forward "
                "than the four-per-kv-source-layer this was sized for.", who)
        # 🔴 Per forward, and under `read.`. Two separate defects, both found
        # by the comparison rather than by this code:
        #
        # * The index used to be a **process-level** counter, so req_000 ended
        #   at 3 and req_001 started at 48 -- the same tensor carried a
        #   different name in two dumps, and a key-by-key comparison saw two
        #   unrelated keys instead of one pair. A name must not carry state
        #   that has nothing to do with what it names.
        # * These are **instrument read-backs**, not model tensors. Under a
        #   bare name they sat in the same namespace as attn.out and moe.out,
        #   so a bit-exact comparison counted them as model differences.
        prefix = f"read.compressor.{who}.{index:02d}"
        align_taps.tap_whole(0, f"{prefix}.slots", flat_slots)
        # ⚠️ `latent` is `torch.empty` (upstream compressor.py:273-278) and is
        # written only at valid group boundaries, so the rows whose slot is
        # negative are **uninitialised memory** -- measured as the constant
        # 5.0097551721019626e+33, which is a recognisable bit pattern and not
        # a value. Recorded whole rather than masked, because masking would
        # change the shape and hide which rows those are; the companion
        # `.slots` says exactly which to drop, and the key's meaning says so
        # too. Reported once so the log says it as well as the manifest.
        _report_uninitialised_rows(who, flat_slots, latent)
        align_taps.tap_whole(0, f"{prefix}.latent", latent)
        # 🔴 Correction. This used to be hashed, with a comment saying the
        # hash let the four-rank same-state comparison report a clean zero.
        # That was wrong: hashing a pointer that differs per process gives a
        # value that differs per process. The measured values (2937847808 vs
        # 2440991232 on two ranks) were the hashed ones.
        #
        # A process-local identifier cannot be made comparable; it can only
        # be taken out of the comparison. Hence the suffix, which says so in
        # the key itself: a comparator filters `.process_local` mechanically
        # and needs no maintained list of exceptions -- a list being the
        # other form of "remembering", which is the thing that gets
        # forgotten. The value still identifies *which* cache within one
        # process, which is all this key was ever for.
        raw = cache.data_ptr() if hasattr(cache, "data_ptr") else 0
        align_taps.tap_whole(
            0, f"{prefix}.cache_id{PROCESS_LOCAL}",
            torch.tensor([(raw * 0x9E3779B1) & 0xFFFFFFFF], dtype=torch.int64))
    except Exception:                                        # noqa: BLE001
        # Instrument. It has already cost one flight to let one of these
        # propagate; it will not cost a second.
        pass


def _report_uninitialised_rows(who: str, flat_slots, latent) -> None:
    """Say how many recorded rows were never written, once per writer.

    The dump would otherwise carry a plausible-looking float where nothing
    was ever stored, and a comparison that does not know to drop those rows
    reports them as model differences -- which is what happened.
    """
    if who in _REPORTED_UNINITIALISED:
        return
    try:
        invalid = int((flat_slots.reshape(-1) < 0).sum())
    except Exception:                                        # noqa: BLE001
        return
    _REPORTED_UNINITIALISED.add(who)
    if not invalid:
        return
    from vllm.logger import logger

    logger.info(
        "DSV4.1 compressor tap: %d of %d rows of %r's latent have a negative "
        "slot and were never written -- the compressor allocates it with "
        "torch.empty and fills only group boundaries. Those rows are "
        "uninitialised memory, not values; drop them using the companion "
        ".slots key before comparing anything.", invalid,
        int(flat_slots.reshape(-1).numel()), who)


_REPORTED_UNINITIALISED: set = set()


def reset_write_taps_for_test() -> None:
    _WRITE_CALLS.clear()
    _REPORTED_UNINITIALISED.clear()


def reset_write_call_index() -> None:
    """Start the per-forward call index over.

    Called from the same place the recorder rotates. The index orders calls
    **within one forward**, which is all it was ever meant to do; letting it
    run process-wide gave the same tensor a different name in every dump.
    """
    _WRITE_CALLS.clear()


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
