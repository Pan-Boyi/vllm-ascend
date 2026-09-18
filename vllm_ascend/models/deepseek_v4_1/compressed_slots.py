# SPDX-License-Identifier: Apache-2.0
"""Per-token slot mappings for V4.1's compressed-stream writers.

``slot_mapping`` on a compressing cache group has two conventions:

``V4.0 (dense)``
    One row per compressed state; what Ascend's ``compressor_metadata``
    publishes and ``dsa_kv_compress_scatter`` consumes.

``V4.1 (per token)``
    One row per token, valid only at group boundaries and ``-1`` elsewhere;
    upstream's ``rope_quant_insert`` kernel indexes slots, positions and
    latent with the same program id.

Both are ``T`` rows of int32 in the same value range, so length cannot tell
them apart. What separates them is which rows hold a valid slot::

    valid    = {t : slot_mapping[t] >= 0}
    boundary = {t : (positions[t] + 1) % ratio == 0}

and a per-token mapping satisfies ``valid`` is a subset of ``boundary``. Containment
rather than equality, because a boundary row may legitimately be ``-1``
(padding). Above ratio 1 no two adjacent rows are boundaries, while the dense
mapping fills rows ``0, 1, 2, ...`` contiguously, so it is always refused.
"""

from __future__ import annotations

from typing import Any

import torch

#: The one value the store operators treat as "skip"; see
#: ``indexer_k_store.group_boundary_slots`` for why nothing else will do.
SKIP_SLOT = -1

#: How many times each ``(consumer, layer, ratio)`` is checked. The check
#: reads index sets back to the host, so it only runs on the first few calls
#: of each site; more than one because the first call is the profile run.
CHECK_LIMIT = 4

_CHECKS: dict[tuple, int] = {}


def reset_for_test() -> None:
    """Forget the per-site check budget (tests share a process)."""
    _CHECKS.clear()


def owning_req_metadata(slot_mapping: torch.Tensor) -> tuple[tuple, Any]:
    """**Every** layer name sharing this exact tensor, and its ``req_metadata``.

    Matched by object identity, never by layer name or shape: every group has
    its own storage page and block table, and taking either from a sibling
    group produces in-range slots belonging to other tokens. Layers sharing a
    KV cache group share one metadata object, so all of them are returned.

    No match is a refusal, never a default.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        # Outside a forward get_forward_context asserts; report it as the
        # missing-metadata case below.
        context = None
    attn_metadata = getattr(context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        raise ValueError(
            "this compressed-stream slot mapping has no attention metadata "
            "dict to take its group's block table and storage page from.")

    seen = []
    names: list = []
    owner = None
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
            names.append(layer_name)
            owner = req
            continue
        seen.append((layer_name, page))
    if owner is not None:
        return tuple(names), owner

    raise ValueError(
        "this slot mapping is not the slot mapping of any attention metadata "
        "in this forward, so the group it belongs to -- and with it the "
        "storage page and block table it must be rebuilt against -- is "
        f"unknown. Groups seen: {seen}")


def layer_owning(slot_mapping: torch.Tensor) -> tuple:
    """Every layer sharing this mapping, or an empty tuple if none is found.

    Used only to label error messages, so it answers instead of raising.
    """
    try:
        return owning_req_metadata(slot_mapping)[0]
    except ValueError:
        return ()


def label(names: tuple) -> str:
    """A message label for a set of layers sharing one mapping, with count."""
    if not names:
        return "0:[unattributed]"
    return f"{len(names)}:[{', '.join(str(n) for n in names)}]"


def per_token_compressed_slots(
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
    cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    """Build the per-token mapping V4.1's writers want, from the block table.

    Upstream's formula verbatim (``compressor_utils.py``)::

        pac  = position // ratio
        slot = block_table[req, pac // page] * page + pac % page

    with ``page`` the *storage* page (upstream passes
    ``cache_config.block_size // ratio``; Ascend calls the same number
    ``storage_block_size``). Non-boundary tokens get ``-1``.

    Rebuilt from the block table rather than permuted from the dense
    mapping, whose multi-request row packing is not visible from here.

    Returns the mapping and the layer names it was built for, since
    ``rope_quant_insert`` (a free function) has no other way to know its layer.
    """
    if compress_ratio < 1:
        raise ValueError(
            f"compress_ratio {compress_ratio} has no group boundary; a layer "
            "with no compressed stream never reaches a compressed-stream "
            "writer, so reaching here means a layer was routed wrongly.")

    layer_names, req = owning_req_metadata(slot_mapping)
    page = int(req.storage_block_size)
    if cache is not None:
        from_cache = int(cache.shape[1])
        if from_cache != page:
            raise ValueError(
                f"the metadata carrying this slot mapping says its storage "
                f"page is {page} rows, while the cache the kernel will index "
                f"has {from_cache} per block ({tuple(cache.shape)}). The "
                "kernel derives block and offset from a flat slot using its "
                "own block length, so building slots with the other number "
                "puts every row in a real but wrong place.")

    block_table = req.block_table
    query_start_loc = req.query_start_loc
    num_tokens = int(slot_mapping.shape[0])
    device = block_table.device
    pos = positions[:num_tokens].to(device=device, dtype=torch.int64)

    token = torch.arange(num_tokens, device=device, dtype=torch.int64)
    # Token -> request without searchsorted/bucketize: one comparison against
    # each request's end offset. num_reqs is small and this runs on every
    # supported device, which matters more here than the asymptotics.
    ends = query_start_loc[1:].to(device=device, dtype=torch.int64)
    req_of_token = (token.unsqueeze(1) >= ends.unsqueeze(0)).sum(dim=1)
    req_of_token = req_of_token.clamp_(max=max(int(block_table.shape[0]) - 1, 0))

    state = torch.div(pos, compress_ratio, rounding_mode="floor")
    block_index = torch.div(state, page, rounding_mode="floor")
    publishes = ((pos + 1) % compress_ratio) == 0
    addressable = (pos >= 0) & (block_index >= 0) & (block_index <
                                                     int(block_table.shape[1]))
    # Clamped only so the gather stays in range for the rows `keep` throws
    # away anyway; a clamped row never survives `addressable`.
    safe_index = block_index.clamp(min=0,
                                   max=max(int(block_table.shape[1]) - 1, 0))
    blocks = block_table.to(torch.int64)[req_of_token, safe_index]

    flat = blocks * page + (state % page)
    # Known limitation: ``blocks >= 0`` cannot tell a padded request row from
    # a real block 0, because the runner fills padded block_table rows with 0
    # (not -1). Under graph replay a padding token can land on such a row and
    # be treated as valid. Nothing here carries a per-token validity signal
    # to fix it; the fix belongs in the runner (fill -1, or pass a mask).
    keep = publishes & addressable & (blocks >= 0)
    slots = torch.where(keep, flat, torch.full_like(flat, SKIP_SLOT))
    return slots.to(slot_mapping.dtype), layer_names


def _index_set(mask: torch.Tensor) -> list:
    return torch.nonzero(mask).flatten().tolist()


def name_the_convention(valid: list, num_tokens: int, ratio: int,
                        num_reqs: int) -> str:
    """Which of the two conventions a mapping's valid-index set looks like."""
    dense_rows = min(num_tokens, num_tokens // max(ratio, 1) + num_reqs)
    leading = 0
    while leading < len(valid) and valid[leading] == leading:
        leading += 1
    if leading >= 2:
        return (
            f"the DENSE (V4.0) convention -- one row per compressed state, "
            f"rows [0, {leading}) valid and adjacent, which no per-token "
            f"mapping can be above ratio 1 (num_compressed_tokens here would "
            f"be {dense_rows} of {num_tokens} rows). Rows past the states "
            f"read as valid too, because the part of the buffer nobody wrote "
            f"is torch.zeros and 0 is a real slot: block 0, offset 0")
    return ("neither convention: the valid rows are not a leading run and "
            "are not the group boundaries either")


def check_per_token(
    slots: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
    consumer: str,
    layer_names: tuple,
    handed: torch.Tensor | None = None,
) -> None:
    """Raise unless every row holding a valid slot is a group boundary.

    See the module docstring for why the criterion is set containment.
    Runs at most :data:`CHECK_LIMIT` times per ``(consumer, layer, ratio)``
    because it reads index sets back to the host.
    """
    if _stream_is_capturing():
        # A host read-back would end the capture; stand down without
        # spending the budget so the eager calls around it are still checked.
        return
    # tuple("layer_02") would split a bare string into characters.
    if isinstance(layer_names, str):
        layer_names = (layer_names, )
    key = (consumer, tuple(layer_names), int(compress_ratio))
    seen = _CHECKS.get(key, 0)
    if seen >= CHECK_LIMIT:
        return
    _CHECKS[key] = seen + 1

    num_tokens = int(slots.shape[0])
    pos = positions[:num_tokens].to(device=slots.device, dtype=torch.int64)
    boundary = _index_set(((pos + 1) % max(compress_ratio, 1)) == 0)
    valid = _index_set(slots >= 0)

    outside = sorted(set(valid) - set(boundary))
    if not outside:
        return

    num_reqs = 0
    if handed is not None:
        try:
            _, req = owning_req_metadata(handed)
            num_reqs = int(req.block_table.shape[0])
        except ValueError:
            num_reqs = 0
    raise ValueError(
        f"DSV4.1 {consumer} on layers {label(layer_names)} has a slot "
        f"mapping with valid "
        f"rows at {_short(outside)} that are not group boundaries. This "
        f"writer indexes the mapping per token and publishes only where "
        f"(position + 1) % {compress_ratio} == 0, which for these "
        f"{num_tokens} positions means rows {_short(boundary)}; the mapping "
        f"has valid rows {_short(valid)}. It is "
        f"{name_the_convention(valid, num_tokens, compress_ratio, num_reqs)}. "
        "Both conventions are T rows of int32 in the same value range, so "
        "length, dtype and block/offset form cannot tell them apart -- only "
        "which rows are valid can. A row valid off a boundary is written "
        "with a latent that was never pooled, into a block belonging to "
        "whoever holds it.")


def _short(values: list, limit: int = 16) -> str:
    if len(values) <= limit:
        return str(values)
    return f"{values[:limit]}...(+{len(values) - limit} more)"


def _stream_is_capturing() -> bool:
    """Whether this stream is capturing a graph right now.

    Same probe as ``engram_embedding._stream_is_capturing``, duplicated to
    keep the engram path out of this module's imports. Answers "no" when the
    predicate is missing or raises, so the check is never silenced by mistake.
    """
    for module in (getattr(torch, "npu", None), getattr(torch, "cuda", None)):
        predicate = getattr(module, "is_current_stream_capturing", None)
        if predicate is None:
            continue
        try:
            # ``is True``: a Mock left on torch.npu is truthy and would
            # silence every caller.
            return predicate() is True
        except Exception:                                    # noqa: BLE001
            return False
    return False


_RING_SHAPES_CHECKED: set = set()


def positions_that_read_across_a_block(capacity: int, ratio: int,
                                       span: int) -> list:
    """Positions where the ring reader would read the wrong block.

    The reader takes its block from ``position // capacity`` and its row from
    ``(position - 1) % capacity``, so the two agree only while ``position``
    and ``position - 1`` sit in the same block, i.e. unless ``capacity``
    divides ``position``. The read happens
    only when the position closes a group, ``(position + 1) % ratio == 0``.
    """
    if capacity < 1 or ratio < 1:
        return []
    return [p for p in range(1, span)
            if (p + 1) % ratio == 0 and p % capacity == 0]


def require_ring_capacity_divides(capacity: int, ratio: int) -> None:
    """Refuse a ring shape whose reader would read across a block.

    A read happens at ``(position + 1) % ratio == 0`` and a block change at
    ``position % capacity == 0``; both hold for some position exactly when
    ``gcd(capacity, ratio) == 1`` (not merely when ``capacity % ratio != 0``:
    ``(6, 4)`` is safe). The reader would then pool a row belonging to a
    different position with nothing raised, so this is checked, not assumed.
    """
    key = (int(capacity), int(ratio))
    if key in _RING_SHAPES_CHECKED:
        return
    _RING_SHAPES_CHECKED.add(key)
    if ratio <= 1:
        return
    bad = positions_that_read_across_a_block(capacity, ratio,
                                             max(capacity, ratio) * 4)
    if not bad:
        return
    raise ValueError(
        f"the compressor state ring has capacity {capacity} with "
        f"compress_ratio {ratio}, and gcd({capacity}, {ratio}) == 1. The "
        f"reader takes its block from (position // {capacity}) and its row "
        f"from "
        f"((position - 1) % {capacity}), so at positions {bad[:8]} it would "
        "read the previous position's row from the *next* block -- a row in "
        "range, belonging to another position, with nothing raised.")
