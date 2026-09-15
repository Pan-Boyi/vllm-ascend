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


def owning_req_metadata(slot_mapping: torch.Tensor) -> tuple[tuple, Any]:
    """**Every** layer name sharing this exact tensor, and its ``req_metadata``.

    Matched by object identity, never by layer name or shape. A name match
    would accept a sibling group's metadata, and every group has its own
    storage page and its own block table -- taking either from the wrong group
    produces in-range slots belonging to other tokens, with nothing raised.

    🔴 All the names, not the first one. Layers whose caches share a KV cache
    group share one metadata object, so the first match is the group's
    lowest-numbered member and the rest are invisible. Flight-23 showed
    exactly that: four "convention OK" lines naming layers 2 and 20, and no
    way to tell whether layers 8 and 14 were covered by those lines or had
    never run. A label that reads like a layer and means "first member of a
    set" is the same-name-different-thing failure, in this module's own log.

    Returning the whole tuple costs nothing -- the walk already visits every
    entry -- and turns that question into a reading instead of a guess.

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

    Used only to label the coverage report. A missing name must not turn a
    working write into a failure, so this one answers instead of raising --
    unlike :func:`owning_req_metadata`, whose answer is arithmetic.
    """
    try:
        return owning_req_metadata(slot_mapping)[0]
    except ValueError:
        return ()


def label(names: tuple) -> str:
    """The log's name for a set of layers that share one mapping.

    Says how many, and says so even when there is one -- "layers=1:[x]" and
    "layers=3:[x, y, z]" are then the same kind of reading, and nobody has to
    know whether a bare name meant one layer or a group's first member.
    """
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
    # 🔴 ``blocks >= 0`` 分不出【padding 请求行】和【真的 0 号块】（2026-09-15 核实）：
    #      model_runner_v1.py:3347  slot_mapping[num_tokens:num_tokens_padded].fill_(-1)
    #      model_runner_v1.py:3348  blk_table_tensor[num_reqs:num_reqs_padded].fill_(0)
    #    ⚠️ 那两行上方的注释写的是 "Fill unused with -1" —— 它只描述了【第一行】，
    #      而第二行填的是 0。而 0 是一个**真块号** ⇒ 「没有」被默认值写成了「有」。
    #    ⇒ 图 replay 时，padding token 的 pos 是上一步的残值（:1523/:1530 只写前缀），
    #      req_of_token 会落到 padding 行，取到 block 0，而这里判它有效。
    #
    # ⚫ 但本函数里【没有】可用来修正它的按 token 的有效性信号：
    #      · slot_mapping 的值不能用 —— ratio>1 时压缩组的 slot_mapping 从没被填过
    #        （dsa_v1.py:1298 的分支），本函数只用它的 shape/dtype 是对的
    #      · positions 的 padding 位是残值，不是 -1
    #      · 真实请求数是 host 侧整数，捕获那一刻就被烤死
    #    ⇒ 修法只能在【上游】：要么 block_table 的 padding 行填 -1，要么额外传一个
    #      按 token 的真实掩码。那是 worker 的共享代码（所有模型共用），不在本线地界
    #      ⇒ 只记录，不擅改。
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
    layer_names: tuple,
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
    # ⚠️ A bare string is one name, not eight. ``tuple("layer_02")`` splits
    # into characters and the log then reads "layers=8:[l, a, y, e, r, ...]"
    # -- a plausible-looking count of layers that do not exist. Caught by a
    # test that passed a string; normalised rather than refused, because
    # refusing would turn a label mistake into a failed write.
    if isinstance(layer_names, str):
        layer_names = (layer_names, )
    key = (consumer, tuple(layer_names), int(compress_ratio))
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
                "DSV4.1 slot-mapping convention OK: consumer=%s layers=%s "
                "ratio=%d T=%d valid=%d/%d boundary rows %s. Every layer "
                "listed shares this one mapping, so one line covers all of "
                "them; a layer absent from every line did not run.",
                consumer, label(layer_names), compress_ratio, num_tokens,
                len(valid), len(boundary), _short(boundary))
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
            # 🔴 ``is True``, not ``bool(...)``. A test double left on
            # torch.npu makes the predicate return a Mock, and ``bool(Mock)``
            # is True -- so the probe reports "capturing" and every caller
            # stands down, silently. Measured: eight engram tests and ten
            # slot-mapping tests failed only inside the full suite, for that
            # reason, and had been on the books as unexplained "existing
            # failures". Anything that is not the literal True reads as not
            # capturing, which is the safe direction: saying "yes" wrongly
            # silences a check, saying "no" wrongly costs at worst the CANN
            # error we would have got anyway.
            return predicate() is True
        except Exception:                                    # noqa: BLE001
            return False
    return False


def check_ring_mapping(
    slots: torch.Tensor,
    positions: torch.Tensor,
    capacity: int,
    consumer: str,
    layer_names: tuple,
    query_start_loc: torch.Tensor | None = None,
    compress_ratio: int = 2,
) -> None:
    """The ring mapping's own criterion -- which is *not* the per-token one.

    🔴 Applying :func:`check_per_token` here would refuse a correct mapping.
    The compressor's state ring is written at **every** token, not only at
    group boundaries: measured, ``[344, 345, 346, 347, 348]`` for a five-token
    chunk, five valid rows for five tokens. ``valid ⊆ boundary`` would leave
    three of those five outside and go red on the healthy case.

    What the ring must satisfy instead is the addressing its reader assumes.
    Upstream's kernel reads exactly one row per request program::

        prev = ring + ((position - 1) % STATE_BLOCK) * STATE_ROW_STRIDE

    (``fused_compress_quant_cache.py:152``) and its tail store writes
    ``slot % STATE_BLOCK`` (:184). So the row a position occupies is
    ``position % capacity`` and nothing else: a mapping whose offsets are
    merely *consecutive* rather than *congruent* would put every row in a real
    place the reader will not look at.

    Two criteria, two paths, and neither substitutes for the other -- which is
    the point of writing this one down. The distinction between these
    mappings was held by a paragraph of prose until now, and prose does not
    go red.
    """
    # First, and not skipped by the capture guard or the budget: it is
    # integer arithmetic over two configuration numbers, and what it protects
    # against is silent.
    require_ring_capacity_divides(capacity, compress_ratio)
    if _stream_is_capturing():
        return
    key = ("ring:" + consumer, tuple(layer_names), int(capacity))
    seen = _CHECKS.get(key, 0)
    if seen >= CHECK_LIMIT:
        return
    _CHECKS[key] = seen + 1

    from vllm.logger import logger

    num_tokens = int(slots.shape[0])
    pos = positions[:num_tokens].to(device=slots.device, dtype=torch.int64)
    flat = slots.reshape(num_tokens, -1)[:, 0] if slots.ndim > 1 else slots
    valid = flat >= 0
    congruent = (flat % capacity) == (pos % capacity)
    wrong = _index_set(valid & ~congruent)

    if not wrong:
        if key not in _ANNOUNCED:
            _ANNOUNCED.add(key)
            readers, requests = ring_readers_this_forward(
                query_start_loc, pos)
            logger.info(
                "DSV4.1 ring-mapping addressing OK: consumer=%s layers=%s "
                "capacity=%d T=%d valid=%d/%d, ring READ by %d of %d "
                "requests this forward. Written at every token, not only at "
                "group boundaries -- the per-token criterion does not apply "
                "here and would refuse this. A read count of 0 means the "
                "ring was written and never read, so this forward carries no "
                "evidence about the reader either way.%s", consumer,
                label(layer_names), capacity, num_tokens,
                int(valid.sum()), num_tokens, readers, requests,
                "" if ring_check_is_strict() else
                " REPORTING ONLY -- this criterion has not yet run clean on "
                "the device; VLLM_ASCEND_DSV41_RING_CHECK_STRICT=1 arms it.")
        return

    complaint = (
        f"DSV4.1 {consumer} on layers {label(layer_names)} has a state-ring "
        f"mapping whose rows {_short(wrong)} are not congruent to their "
        f"positions modulo the ring capacity {capacity}. The reader takes "
        f"the row for a position as (position % {capacity}) and nothing "
        "else, so an offset that is merely consecutive lands in a real row "
        "the reader will never look at, and the pooled group then carries "
        "whatever that row held.")
    if not ring_check_is_strict():
        # 🔴 Reports, does not raise -- and this prices "has never executed
        # here", not "might be wrong". Unlike the two per-token checks, which
        # have now run on the card, this one has not, and it fires on *every*
        # forward including each decode step, so its blast radius is an order
        # of magnitude larger. A flight has already been lost to a tap that
        # had never flown. Trigger, not a date: once one flight logs
        # "ring-mapping addressing OK" from the device,
        # VLLM_ASCEND_DSV41_RING_CHECK_STRICT=1 makes it a refusal.
        logger.error("%s ⚠️ NOT raising: this criterion has not yet run clean "
                     "on the device, so it reports only. Set "
                     "VLLM_ASCEND_DSV41_RING_CHECK_STRICT=1 to arm it.",
                     complaint)
        return
    raise ValueError(complaint)


def ring_check_is_strict() -> bool:
    """Whether an incongruent ring mapping refuses the run.

    Default off, with a trigger rather than a date: arm it once a flight has
    logged the OK line from the device.
    """
    import os

    return os.environ.get("VLLM_ASCEND_DSV41_RING_CHECK_STRICT", "0") == "1"


def ring_readers_this_forward(query_start_loc, positions) -> tuple:
    """How many of this forward's requests actually read the state ring.

    🔴 "How far does the reader read" presumes the reader runs, and twice
    today that presumption was wrong: at T=5 the indexer's operator is
    short-circuited before it is called, and the ring's reader is likewise
    gated. Upstream's request program reads the ring only under::

        position = tl.load(positions + start)
        if (position + 1) % 2 == 0:        # the chunk's first token closes a group

    (``fused_compress_quant_cache.py:148-149``) -- so a chunk that starts at
    an even absolute position never touches the ring, and a request's *first*
    chunk starts at position 0 and therefore never does.

    Counted here rather than argued, because the two answers lead to
    different next steps: 0 readers means this forward's evidence about the
    reader is **zero, not negative**, and the way to get evidence is a second
    chunk (chunked prefill, or any decode step at an odd position) -- not a
    longer prompt.
    """
    if query_start_loc is None:
        return -1, -1
    starts = query_start_loc[:-1].to(device=positions.device,
                                     dtype=torch.int64)
    starts = starts[starts < positions.numel()]
    if starts.numel() == 0:
        return 0, 0
    first = positions[starts]
    return int((((first + 1) % 2) == 0).sum()), int(starts.numel())


_RING_SHAPES_CHECKED: set = set()


def positions_that_read_across_a_block(capacity: int, ratio: int,
                                       span: int) -> list:
    """Positions where the ring reader would read the wrong block.

    The reader takes its block from ``position // capacity`` and its row from
    ``(position - 1) % capacity`` (``fused_compress_quant_cache.py:151-152``),
    so the two agree only while ``position`` and ``position - 1`` sit in the
    same block. They do not when ``capacity`` divides ``position``. And the
    read happens only when the position closes a group,
    ``(position + 1) % ratio == 0``.

    Searched rather than reasoned, so the closed form below is *checked*
    against it instead of trusted.
    """
    if capacity < 1 or ratio < 1:
        return []
    return [p for p in range(1, span)
            if (p + 1) % ratio == 0 and p % capacity == 0]


def require_ring_capacity_divides(capacity: int, ratio: int) -> None:
    """The ring's unwritten precondition, written down and made executable.

    🔴 The reader never reads a stale row **today**, and the reason is not in
    the source anywhere. A read happens at ``position ≡ -1 (mod ratio)`` and a
    block change at ``position ≡ 0 (mod capacity)``; a position satisfying
    both exists exactly when ``gcd(capacity, ratio) == 1``. The attainable
    residues of ``k * capacity`` mod ratio are the multiples of
    ``g = gcd(capacity, ratio)``, and ``ratio - 1`` is a multiple of ``g``
    only when ``g == 1`` -- so ``g > 1`` is what makes it impossible, and
    ``gcd(8, 2) = 2`` is the whole reason this is safe.

    ⚠️ Not ``capacity % ratio == 0``. That was the first form written here and
    it is sufficient but not necessary: ``(capacity, ratio) = (6, 4)`` is safe
    and fails it. The test that compares the closed form against the search is
    what found that, which is the reason the guard searches.

    Change either number so that it does not, and the reader crosses into the
    next block and pools a row belonging to a different position. Nothing
    raises: the row is in range and the arithmetic is valid.

    ⚠️ A property that merely *happens to hold* looks exactly like a designed
    guarantee once it is only a comment, and comments do not go red. Raising
    rather than reporting, unlike the congruence criterion above: this one is
    integer arithmetic over configuration, it cannot be tripped by an
    instrument, and if it does fire the run is genuinely unsafe.
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
        "range, belonging to another position, with nothing raised. This "
        "held only because capacity and ratio shared a factor; that is a "
        "property nothing in the source states, which is why it is stated "
        "here.")
