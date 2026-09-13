# SPDX-License-Identifier: Apache-2.0
"""One field name, two index conventions, and the same-looking code for both.

``slot_mapping`` on a compressing cache group means two different things in
the two halves of this port, and nothing in either half says which:

``V4.0 (dense)``
    One row per **compressed state**. Rows ``[0, num_compressed_tokens)``
    carry the physical slots; that is what ``compressor_metadata`` writes and
    what ``dsa_kv_compress_scatter`` consumes -- it reshapes the compressor's
    output to ``slot_mapping.shape[0]`` rows
    (``dsa_attn_kv_plan.py:209``), so the two counts are the same by
    construction.

``V4.1 (per token)``
    One row per **token**, valid only at group boundaries, ``-1`` elsewhere.
    ``_rope_plain_insert_kernel`` indexes ``cache_slots``, ``positions`` and
    ``latent`` with the same ``tl.program_id(0)``
    (``fused_compress_quant_cache.py:354-394``), and upstream's own producer
    stores one entry per query token with ``PAD_ID = -1``
    (``v1/attention/backends/mla/compressor_utils.py:44-84``).

Ascend publishes the first and V4.1 reads it as the second. Measured on
flight-21/22 at ``T=5, ratio=2, num_reqs=1``: the mapping is
``[10752, 10753, -1, 0, 0]`` -- rows 0,1 the two real states, row 2 the
operator's slack row, rows 3,4 the ``torch.zeros`` nobody ever wrote, because
``dsa_v1.py:1298`` only fills the per-token mapping when ``ratio <= 1``. Read
per token that writes entry #1 and *nothing* for entry #0, and sends ``t=3``
to flat slot 0 -- block 0, offset 0, another request's row -- carrying
``latent[3]``, a non-boundary row the pooling half never wrote. Both were
confirmed bit-exact on the read side.

🔴 **Why the obvious check does not work.** The two conventions have the same
length: V4.1 receives ``self.slot_mapping[:num_actual_tokens]``
(``dsa_v1.py:1779``), so both the right and the wrong mapping are ``T`` rows
of ``(block, offset)`` int32 in the same value range. "How many rows" is a
*scalar*, and two semantics collide in a scalar easily. What separates them is
**which indices hold a valid slot** -- a *set*, the same shape as the thing
being judged:

    valid    = {t : slot_mapping[t] >= 0}
    boundary = {t : (positions[t] + 1) % ratio == 0}

and the criterion is ``valid`` ⊆ ``boundary``. At ``T=5, ratio=2`` the dense
mapping gives ``{0, 1}`` and the per-token one ``{1, 3}``: row 0 is valid and
is not a boundary, so the dense one is refused. Containment rather than
equality because a boundary row may legitimately be ``-1`` (padding, an
unallocated request), while a *non*-boundary row that is valid is never
anything but wrong -- it writes a latent the pooling half never produced into
a block belonging to whoever holds it.

This holds for any ``T`` and any ``num_reqs``: ``boundary`` is a function of
``positions`` alone, and above ratio 1 it never contains two adjacent rows,
while the dense mapping always fills rows ``0, 1, 2, ...`` contiguously up to
``num_compressed_tokens = min(T, T // ratio + num_reqs)``. At ``ratio == 1``
every row is a boundary, containment is free, and there is no defect to catch.

🔴 **The check reports when it passes, too.** A criterion that only speaks on
failure leaves "this path is correct" and "this path never ran" as the same
reading, and the indexer half of this fix has *zero* runtime coverage today
(integration swept all 592 dump keys: nothing indexer-side). So each
``(consumer, layer, ratio)`` announces once on success. Grepping one flight's
log then answers "did both consumers actually run, and on which layers"
without a new tap.
"""

from __future__ import annotations

from typing import Any

import torch

#: The one value the store operators treat as "skip"; see
#: ``indexer_k_store.group_boundary_slots`` for why nothing else will do.
SKIP_SLOT = -1

#: How many times each ``(consumer, layer, ratio)`` is checked. The check
#: reads index sets back to the host, so it cannot run every forward; it runs
#: on the first few calls of each distinct site, which is what proves coverage.
#: Raised deliberately above 1: the first call of a server is the profile run,
#: whose shapes are not a request's.
CHECK_LIMIT = 4

_CHECKS: dict[tuple, int] = {}
_ANNOUNCED: set = set()


def reset_for_test() -> None:
    """Forget the per-site check budget (tests share a process)."""
    _CHECKS.clear()
    _ANNOUNCED.clear()


def owning_req_metadata(slot_mapping: torch.Tensor) -> tuple[str, Any]:
    """The layer name and ``req_metadata`` carrying *this exact tensor*.

    Matched by object identity, never by layer name or shape. A name match
    would accept a sibling group's metadata, and every group has its own
    storage page and its own block table -- taking either from the wrong group
    produces in-range slots belonging to other tokens, with nothing raised.

    No match is a refusal, never a default.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        # Outside a forward, get_forward_context asserts rather than returning
        # None. Same answer either way -- there is nothing to take the block
        # table from -- so it is reported as that rather than escaping as an
        # AssertionError from a frame that says nothing about slot mappings.
        context = None
    attn_metadata = getattr(context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict):
        raise ValueError(
            "this compressed-stream slot mapping has no attention metadata "
            "dict to take its group's block table and storage page from.")

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
            return layer_name, req
        seen.append((layer_name, page))

    raise ValueError(
        "this slot mapping is not the slot mapping of any attention metadata "
        "in this forward, so the group it belongs to -- and with it the "
        "storage page and block table it must be rebuilt against -- is "
        f"unknown. Groups seen: {seen}")


def layer_owning(slot_mapping: torch.Tensor) -> str:
    """The layer name for a mapping, or a word saying it could not be found.

    Used only to label the coverage report. A missing name must not turn a
    working write into a failure, so this one answers instead of raising --
    unlike :func:`owning_req_metadata`, whose answer is arithmetic.
    """
    try:
        return owning_req_metadata(slot_mapping)[0]
    except ValueError:
        return "unattributed"


def per_token_compressed_slots(
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
    cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str]:
    """Build the per-token mapping V4.1's writers want, from the block table.

    Upstream's formula verbatim (``compressor_utils.py:73-78``)::

        pac  = position // ratio
        slot = block_table[req, pac // page] * page + pac % page

    with ``page`` the *storage* page (upstream passes
    ``cache_config.block_size // ratio``; Ascend calls the same number
    ``storage_block_size``). Non-boundary tokens get ``-1``.

    ⚠️ Rebuilt, not rearranged. Turning the dense mapping into a per-token one
    by permutation would need the row packing ``compressor_metadata`` uses for
    multiple requests -- its per-request slack row is visible in
    ``num_compressed_tokens = min(T, T // ratio + num_reqs)`` but its placement
    is not, and that operator's source is not in this tree. The block table is
    a source we can read.

    Returns the mapping and the layer name it was built for, because the
    caller that needs it most -- ``rope_quant_insert``, a free function
    rebound over the compressor's import -- has no other way to know its layer.
    """
    if compress_ratio < 1:
        raise ValueError(
            f"compress_ratio {compress_ratio} has no group boundary; a layer "
            "with no compressed stream never reaches a compressed-stream "
            "writer, so reaching here means a layer was routed wrongly.")

    layer_name, req = owning_req_metadata(slot_mapping)
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
    keep = publishes & addressable & (blocks >= 0)
    slots = torch.where(keep, flat, torch.full_like(flat, SKIP_SLOT))
    return slots.to(slot_mapping.dtype), layer_name


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
            f"be {dense_rows} of {num_tokens} rows). ⚠️ Rows past the states "
            f"read as valid too, because the part of the buffer nobody wrote "
            f"is torch.zeros and 0 is a real slot: block 0, offset 0")
    return ("neither convention: the valid rows are not a leading run and "
            "are not the group boundaries either")


def check_per_token(
    slots: torch.Tensor,
    positions: torch.Tensor,
    compress_ratio: int,
    consumer: str,
    layer_name: str,
    handed: torch.Tensor | None = None,
) -> None:
    """Assert the mapping is per-token, and say so when it is.

    The criterion is containment, not equality: every row holding a valid slot
    must be a group-boundary token. Equality would be wrong in a direction
    that matters -- a boundary token whose request is padding legitimately
    carries ``-1``, and at ratio 1 every token is a boundary while padding
    rows are still ``-1``. Containment is the half that is always true of a
    per-token mapping and always false of a dense one: the dense mapping fills
    rows ``0, 1, 2, ...`` contiguously, and at any ratio above 1 no two
    adjacent rows are both boundaries.

    Both halves are *sets*, which is what separates the two conventions --
    their lengths do not, since both are ``T`` rows.

    Runs at most :data:`CHECK_LIMIT` times per ``(consumer, layer, ratio)``:
    it reads index sets back to the host, so it is an entry check, not a
    per-forward one. The bound is per *site*, so a consumer that never runs
    never announces, which is the point.
    """
    if _stream_is_capturing():
        # Reading index sets back to the host inside a graph capture ends the
        # capture, so the check stands down -- without spending the budget, so
        # it still runs (and still announces) on the eager calls around it. A
        # deployment that captures everything therefore announces nothing,
        # which is the honest reading: this check did not run.
        return
    key = (consumer, layer_name, int(compress_ratio))
    seen = _CHECKS.get(key, 0)
    if seen >= CHECK_LIMIT:
        return
    _CHECKS[key] = seen + 1

    from vllm.logger import logger

    num_tokens = int(slots.shape[0])
    pos = positions[:num_tokens].to(device=slots.device, dtype=torch.int64)
    boundary = _index_set(((pos + 1) % max(compress_ratio, 1)) == 0)
    valid = _index_set(slots >= 0)

    outside = sorted(set(valid) - set(boundary))
    if not outside:
        if key not in _ANNOUNCED:
            _ANNOUNCED.add(key)
            logger.info(
                "DSV4.1 slot-mapping convention OK: consumer=%s layer=%s "
                "ratio=%d T=%d valid=%d/%d boundary rows %s", consumer,
                layer_name, compress_ratio, num_tokens, len(valid),
                len(boundary), _short(boundary))
        return

    num_reqs = 0
    if handed is not None:
        try:
            _, req = owning_req_metadata(handed)
            num_reqs = int(req.block_table.shape[0])
        except ValueError:
            num_reqs = 0
    raise ValueError(
        f"DSV4.1 {consumer} on {layer_name} has a slot mapping with valid "
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

    The same probe as ``engram_embedding._stream_is_capturing`` (that module,
    line 579), written out again rather than imported: pulling the engram path
    into the import graph of every compressed-stream write costs more than six
    lines do. Both answer "no" when the predicate is missing or raises --
    answering "yes" wrongly would silence a check that has never yet run.
    """
    for module in (getattr(torch, "npu", None), getattr(torch, "cuda", None)):
        predicate = getattr(module, "is_current_stream_capturing", None)
        if predicate is None:
            continue
        try:
            return bool(predicate())
        except Exception:                                    # noqa: BLE001
            return False
    return False
