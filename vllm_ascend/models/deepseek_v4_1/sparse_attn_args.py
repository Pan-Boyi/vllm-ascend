# SPDX-License-Identifier: Apache-2.0
"""Assemble and check the arguments to ``sparse_flash_mla``.

Every rule enforced here is one the operator states about itself -- each was
measured on card 7 under CANN 9.2.0-beta.2 by the kernel line and recorded in
``build_artifacts/KERNEL_LINE_SNAPSHOT_20260911.md``, as the operator's own
rejection text rather than a reading of a README. They are re-stated as checks
for one reason: on a card they surface as a rejection from inside the operator,
one constraint at a time, each costing a launch, a queue slot and a round trip.
Here they surface together, on CPU, in a unit test.

This is deliberately a pure function over already-built metadata. What supplies
that metadata is a separate question -- V4.0 builds it inside its attention
backend impl, which V4.1 does not use -- and keeping the two apart means the
rules do not have to be settled twice.

Two of these rules interact in a way worth naming. "SWA mode requires
cmp_ratio=1" and "non-sliding-window modes require ori_win_left =
ori_win_right = -1" together make the windowed and compressed configurations
mutually exclusive at the operator boundary: a layer that attends a compressed
stream must disable its window bounds, and a layer that does not must keep
them.

"Ori-only sparse mode requires both mask modes to be 0" was left unmodelled
here for a while, because telling that mode from plain sliding-window mode
needed a signal nothing measured provided -- both have no compressed stream,
and inventing a rule would have rejected valid calls or accepted invalid ones
with equal confidence. The operator supplied the signal on the first real
forward, by refusing (4, 0) for a call that passed ori_sparse_indices.

So the rule is now modelled, from a measurement rather than a reading: an
explicit index list selects the original stream, which is mode 0, and the
window bounds have nothing left to say. Three configurations, two of them mode
0 for different reasons, and only window-bounds-without-indices is mode 4.

That distinction matters because of the asymmetry in how the two fail: window
bounds alongside a compressed stream are rejected outright, while a wrong mask
mode is not always rejected -- which is the case that yields a finite,
correctly shaped, wrong answer.
"""

from __future__ import annotations

from typing import Any

import torch

#: Accepted KV layouts. PA_ND -- which the FP8 plan uses -- belongs to a
#: different operator and is rejected here.
LAYOUTS_KV = ("TND", "BSND", "PA_BBND")

#: q's head dimension. The operator has no rope_head_dim parameter at all, so
#: the 64 rope dims are part of this 512 rather than declared separately.
HEAD_DIM = 512

#: Sliding-window mask mode, for a layer that attends its own stream by window
#: bounds rather than by an explicit index list.
SLIDING_WINDOW_MASK_MODE = 4
#: The mode required once a compressed stream is present, and also the one
#: required when the original stream is selected by explicit indices -- the
#: operator calls that "ori-only sparse mode" and refuses anything else:
#: "Parameters ori_mask_mode and cmp_mask_mode ... have incorrect values 4, 0.
#: Ori-only sparse mode requires both mask modes to be 0."
NO_MASK_MODE = 0
CMP_MASK_MODE = 3


def select_ori_sparse_indices(attends_compressed: bool, vision_swa_indices,
                             dspark_swa_indices):
    """The explicit original-KV index list this call will use, or ``None``.

    Both the descriptor builder and the call site have to reach the same
    answer, and until now each derived it from its own inputs: the call site
    from the selected tensor, the builder from a separate ``has_indices``
    expression over the two candidates. The two happened to agree, which is
    the worst case -- a structural second source that agrees today is one that
    disagrees on the first day the precedence changes.

    DSpark's indices override vision's, and only when no compressed stream is
    attended. That precedence is the call site's, not an interpretation of it.

    🔴 **The returned indices are in slot space**, not token space: each value
    is ``block_id * storage_block_size + offset_within_block`` (dsa_v1.py, the
    ``slot_ids`` line at the end of both index builders). That fact used to
    exist only in a local variable's name and was lost at the first
    assignment -- after which the tensor is called "indices" everywhere and
    carries no record of which space it is in.

    This is written down because the two spaces are indistinguishable from the
    tensor: both are non-negative int32, and a bound taken from the wrong one
    looks like a working check. A token-space bound (``< seqused``) rejects
    valid slot-space values: a measured call had ``max() = 5663 = 176*32 + 31``
    against ``seqused = 4096``. Any range check on this tensor must take its
    upper bound from *this call's* block table -- ``(block_table.max() + 1) *
    storage_block_size`` -- and must refuse to check, out loud, when the
    caller cannot say which space it is handing over.
    """
    if not attends_compressed and dspark_swa_indices is not None:
        return dspark_swa_indices
    return vision_swa_indices


def is_plain_sliding_window(attends_compressed: bool,
                             ori_sparse_indices) -> bool:
    """Whether this call is sliding-window mode rather than one of the two
    mode-0 configurations.

    The distinguishing signal is the explicit index list, and it was not a
    guess: the module docstring recorded that nothing measured separated
    "ori-only sparse mode" from plain sliding window, and left the constraint
    unmodelled rather than invent a rule. The operator then answered, by
    refusing (4, 0) on the first real forward -- a call that supplied
    ori_sparse_indices with mask mode 4.

    So indices present means the original stream is selected by that list, and
    the window bounds have nothing left to say.
    """
    return not attends_compressed and ori_sparse_indices is None


def _check(condition: bool, operator_message: str, detail: str = "") -> None:
    if condition:
        return
    raise ValueError(
        f"sparse_flash_mla would reject this call: {operator_message}"
        + (f"\n  {detail}" if detail else ""))


def _check_block_table_addresses_this_cache(kwargs: dict[str, Any],
                                            cache_group_key=None,
                                            owner_key=None) -> None:
    """Refuse a block table that does not belong with the cache it is paired with.

    Two separate questions, because the cheap one does not cover the dangerous
    case.

    *Range*: the table's highest block id must be inside the cache. These
    arguments arrive from unrelated places -- the table from the attention
    metadata, the cache from the layer -- and the operator does not range-check
    them. Every core reads its slice through the same table, so one wrong table
    is one fault in the same place on all of them.

    *Identity*: a table from another cache group can be entirely in range --
    when that group has as many blocks or more, every id is valid and simply
    addresses the wrong memory. Range says the numbers fit; only the two
    objects' own account of which group they belong to says they are the same
    allocation.

    Bounds are read from the tensor passed here, never from a configured
    num_blocks: a configured value is true of both groups, which is the
    confusion being checked for.

    ⚠️ ``.max()`` synchronises, once per layer per step. Affordable while the
    priority is finding a fault, not afterwards. TRIGGER: the first performance
    measurement that includes this path.
    """
    if cache_group_key is not None and owner_key is not None \
            and cache_group_key != owner_key:
        raise ValueError(
            f"the attention metadata reports cache group {cache_group_key!r} "
            f"and the cache tensor belongs to {owner_key!r}. Every id in the "
            "block table may still be in range -- a group with as many blocks "
            "or more makes them all valid -- so this is not caught by bounds. "
            "The two objects disagree about which allocation they describe.")

    # seqused_kv, not seqused_ori_kv: this runs inside the builder, and the
    # rename happens later, in the sparse_flash_mla adapter. Reading the
    # renamed key here finds nothing and the capacity check reports itself as
    # not run -- which the existing tests caught before a launch did.
    for table_key, cache_key, used_key in (
            ("ori_block_table", "ori_kv", "seqused_kv"),
            ("cmp_block_table", "cmp_kv", "seqused_cmp_kv")):
        table = kwargs.get(table_key)
        cache = kwargs.get(cache_key)
        if table is None or cache is None:
            continue
        num_blocks, block_size = int(cache.shape[0]), int(cache.shape[1])
        if table.numel() == 0:
            raise ValueError(
                f"{table_key} is empty, so this check did not run. An empty "
                "table is not a batch needing no blocks; it addresses nothing.")
        highest = int(table.max())
        if highest >= num_blocks:
            raise ValueError(
                f"{table_key}'s highest block id is {highest} and {cache_key} "
                f"holds {num_blocks} blocks. These come from different "
                "objects: the table from the attention metadata, the cache "
                "from the layer. An id at or past the end most likely means "
                "the table belongs to another cache group -- V4.1 has several, "
                "with different block counts.")
        used = kwargs.get(used_key)
        if used is None or used.numel() == 0:
            raise ValueError(
                f"{used_key} is missing or empty, so the capacity check did "
                f"not run against {cache_key}. Calling that a pass would claim "
                "coverage this call does not have.")
        capacity = num_blocks * block_size
        longest = int(used.max())
        if longest > capacity:
            raise ValueError(
                f"{used_key}'s longest sequence is {longest} tokens and "
                f"{cache_key} holds {capacity} ({num_blocks} blocks of "
                f"{block_size}). Lengths and cache are sized by different code "
                "paths; a length past the end reads another allocation.")


#: The descriptor blob the ``*_metadata`` operator produces, in int32 words.
#: Same number as ``dsa_v1.DSA_METADATA_BUFFER_SIZE``, restated here rather
#: than imported: this module is the argument contract and importing the
#: builder's constant would make the check pass by construction.
DESCRIPTOR_WORDS = 1024


def _check_descriptor_is_populated(kwargs: dict[str, Any],
                                   q: torch.Tensor) -> None:
    """The coarsest possible check on ``metadata``, and it is worth saying
    exactly how coarse.

    The descriptor is a 1024-word blob carrying the per-core split of query
    tokens. Nothing here can tell whether its *contents* match this call --
    the disagreement that produced an aicore exception was a descriptor built
    for "has a compressed stream" against a call that supplied none, and that
    blob was perfectly well-formed: 83 non-zero words, a legal descriptor for
    a different configuration. ⚠️ So "not all zeros" catches only the case
    where nobody ever ran the metadata operator. A blob that belongs to
    someone else passes every check in this function.

    Three states, deliberately. Zero query tokens means the split cannot be
    populated and silence would be indistinguishable from a pass, so that case
    is reported as *not checked* rather than allowed through quietly.
    """
    metadata = kwargs.get("metadata")
    _check(metadata is not None,
           "metadata must be provided",
           "the operator takes its tiling from the descriptor; without one "
           "the tiling key comes out 0 and every core faults at the same pc")

    numel = getattr(metadata, "numel", None)
    if numel is None:
        # A pointer or an opaque handle. Nothing to inspect, and inventing an
        # interpretation of it would be worse than saying so.
        _log_descriptor_not_checked("metadata is not a tensor")
        return

    _check(numel() == DESCRIPTOR_WORDS,
           f"metadata must be {DESCRIPTOR_WORDS} int32 words",
           f"got {numel()}; a short blob is read past its end by the kernel")

    num_tokens = int(q.shape[0]) if q.ndim >= 1 else 0
    if num_tokens == 0:
        _log_descriptor_not_checked(
            "the batch has no query tokens, so an all-zero split is what an "
            "empty batch legitimately produces")
        return

    _check(bool(metadata.any()),
           "metadata must be a populated descriptor",
           f"all {DESCRIPTOR_WORDS} words are zero while this call has "
           f"{num_tokens} query tokens, so the metadata operator never wrote "
           "into this buffer. Note the converse does not hold: a non-zero "
           "blob may still have been built for another configuration.")


_LOGGED_DESCRIPTOR_SKIPS: set[str] = set()


def _log_descriptor_not_checked(reason: str) -> None:
    """Say that the descriptor check was skipped, once per reason.

    A skipped check and a passed check produce the same absence of output,
    and this port has already spent a day on a negative control that was only
    a dead instrument.
    """
    if reason in _LOGGED_DESCRIPTOR_SKIPS:
        return
    _LOGGED_DESCRIPTOR_SKIPS.add(reason)
    from vllm.logger import logger

    logger.info("DSV4.1 sparse_flash_mla: descriptor contents NOT checked -- %s",
                reason)


def validate_sparse_flash_mla_args(kwargs: dict[str, Any], q: torch.Tensor,
                                   num_requests: int) -> dict[str, Any]:
    """Check one assembled call. Returns the kwargs so it can wrap a build."""
    layout_kv = kwargs.get("layout_kv")
    layout_q = kwargs.get("layout_q")
    cmp_ratio = kwargs.get("cmp_ratio")
    has_cmp_kv = kwargs.get("cmp_kv") is not None

    _check(layout_kv in LAYOUTS_KV,
           f"layout_kv must be in [{', '.join(LAYOUTS_KV)}]",
           f"got {layout_kv!r}; PA_ND belongs to a different operator")

    _check(q.shape[-1] == HEAD_DIM,
           "head_dim must be 512, and q's head dimension must also be 512",
           f"q's last dimension is {q.shape[-1]}; note the operator has no "
           "rope_head_dim parameter, so the rope tail is part of the 512")

    _check(kwargs.get("sinks") is not None, "sinks must be provided",
           "attn_sink is [64] fp32 and cannot be skipped, not even to get "
           "something running")

    if layout_q == "TND":
        cu_seqlens_q = kwargs.get("cu_seqlens_q")
        _check(cu_seqlens_q is not None,
               "layout_q=TND requires cu_seqlens_q, cumulative, length T+1")
        _check(cu_seqlens_q.ndim == 1 and cu_seqlens_q.numel() == num_requests + 1,
               "layout_q=TND requires cu_seqlens_q, cumulative, length T+1",
               f"got shape {tuple(cu_seqlens_q.shape)} for {num_requests} requests")
        _check(bool(torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1])),
               "layout_q=TND requires cu_seqlens_q, cumulative, length T+1",
               "the values are not non-decreasing, so they are lengths rather "
               "than a cumulative sum")

    if layout_kv != "TND":
        for name in ("cu_seqlens_ori_kv", "cu_seqlens_cmp_kv"):
            _check(kwargs.get(name) is None,
                   "cu_seqlens_*_kv are accepted only when layout_kv is TND",
                   f"{name} was supplied with layout_kv={layout_kv!r}; under a "
                   "paged layout the KV lengths travel in seqused_ori_kv")

    _check(cmp_ratio is not None and cmp_ratio > 0,
           "cmp_ratio must be greater than 0",
           f"got {cmp_ratio!r}; a layer with no compressed stream still has to "
           "report a positive ratio, so 0 is never a legal value here even "
           "though 0 is a legal configured compress_ratio")

    windowed = (kwargs.get("ori_win_left"), kwargs.get("ori_win_right")) != (-1, -1)
    if windowed:
        _check(cmp_ratio == 1, "SWA mode requires cmp_ratio = 1",
               f"window bounds are set but cmp_ratio={cmp_ratio}")
        _check(not has_cmp_kv, "SWA mode requires cmp_ratio = 1",
               "window bounds are set alongside a compressed stream; the two "
               "configurations are mutually exclusive at this boundary")
    if has_cmp_kv:
        _check(not windowed,
               "non-sliding-window modes require ori_win_left = ori_win_right = -1",
               f"a compressed stream is present but the window bounds are "
               f"{kwargs.get('ori_win_left')!r}/{kwargs.get('ori_win_right')!r}")

    if has_cmp_kv:
        if layout_kv == "PA_BBND":
            _check(kwargs.get("seqused_cmp_kv") is not None,
                   "has_cmp_kv with layout_kv=PA_BBND requires seqused_cmp_kv")
        if cmp_ratio == 1:
            residual = kwargs.get("cmp_residual_kv")
            _check(residual is not None,
                   "cmp_ratio=1 with a compressed stream additionally requires "
                   "cmp_residual_kv")
            # Despite the name it is not KV data: the operator rejects anything
            # but a 1-D tensor ("incorrect shape dim 3. It should be 1").
            _check(residual.ndim == 1,
                   "cmp_residual_kv incorrect shape dim. It should be 1",
                   f"got {residual.ndim} dims, shape "
                   f"{tuple(residual.shape)}; despite the name this is one "
                   "value per request, not KV data")
            _check(residual.numel() == num_requests,
                   "cmp_residual_kv incorrect shape dim. It should be 1",
                   f"got {residual.numel()} values for {num_requests} requests")

    _check_descriptor_is_populated(kwargs, q)
    _check_block_table_addresses_this_cache(kwargs)
    return kwargs


#: How many axes each KV layout describes. The operator reads ``ori_kv``'s
#: stride vector against this and rejects a length mismatch by name
#: ("Ori_kv strideVec size[3] not match layout_kv expect len[4]"), so the rank
#: is part of the layout, not a property of the cache.
#:
#: The names are the ones LAYOUTS_KV accepts, checked below. Writing the
#: plausible spelling instead is how this table first went in: "PA_BSND" reads
#: like the paged sibling of PA_BBND and is not a layout this operator has.
LAYOUT_KV_RANK = {
    "PA_BBND": 4,   # blocks, block, heads, dim
    "BSND": 4,      # batch, seq, heads, dim
    "TND": 3,       # tokens, heads, dim
}


def kv_with_layout_rank(cache, layout_kv: str):
    """Give the operator ``ori_kv`` at the rank ``layout_kv`` describes.

    The head axis is dropped on the way in and has to come back here. It is
    dropped for a real reason -- upstream's own V4.1 consumers read the cache
    as three dimensions, ``engram.ensure_cache`` calls ``.numel()`` on it and
    the SWA path asserts ``shape[1:] == (block_size, head_dim)`` -- and the
    operator wants four. Both are right; the disagreement is about what a KV
    cache is, and it resolves where the layout is known, which is here.

    Dispatched on the layout rather than unconditionally restored. ``TND`` is
    three axes, so an unconditional ``unsqueeze`` would hand that plan a tensor
    of the right rank with the axes meaning something else -- and the operator
    only checks the stride vector's *length*, so it would be accepted.
    """
    expected = LAYOUT_KV_RANK.get(layout_kv)
    if expected is None:
        raise ValueError(
            f"no KV rank recorded for layout_kv={layout_kv!r}; known layouts "
            f"are {sorted(LAYOUT_KV_RANK)}. Passing the cache through "
            "unchanged would let the operator check the stride length against "
            "a layout nothing here understands.")
    if cache is None or cache.ndim == expected:
        return cache
    if cache.ndim == expected - 1 and expected == 4:
        # The axis bind_value dropped: num_kv_heads, at index -2.
        return cache.unsqueeze(-2)
    raise ValueError(
        f"ori_kv is {cache.ndim}-dimensional ({tuple(cache.shape)}) and "
        f"layout_kv={layout_kv!r} describes {expected} axes. Only the single "
        "dropped head axis is restored here; any other difference means the "
        "cache is not the layout this plan selected.")


def build_sparse_flash_mla_kwargs(
    *,
    q: torch.Tensor,
    num_requests: int,
    swa_kv_cache: torch.Tensor,
    swa_block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    sinks: torch.Tensor,
    metadata: Any,
    softmax_scale: float,
    cmp_ratio: int,
    layout_kv: str,
    attends_compressed: bool,
    ori_win_left: int,
    ori_win_right: int,
    cmp_kv: torch.Tensor | None = None,
    cmp_block_table: torch.Tensor | None = None,
    seqused_cmp_kv: torch.Tensor | None = None,
    cmp_residual_kv: torch.Tensor | None = None,
    # Slot space: block_id * storage_block_size + offset. See
    # select_ori_sparse_indices for why this is stated and not left implicit.
    ori_sparse_indices: torch.Tensor | None = None,
    cmp_sparse_indices: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Assemble one call, mirroring ``DSAImpl._forward_attention``.

    ``attends_compressed`` is passed in rather than inferred from
    ``cmp_ratio``, because the two mean different things for V4.1 and inferring
    one from the other is the mistake this port has already made once: V4.0
    reads ratio <= 1 as "no compressed stream", while V4.1 gives layers 20..39
    ratio 1 to mean "compressed, pooling disabled". A layer that attends a
    compressed stream and reports ratio 1 is a legitimate V4.1 configuration
    and an impossible V4.0 one.

    The window bounds are forced to -1 whenever a compressed stream is present.
    That is not a preference: the operator rejects window bounds alongside a
    compressed stream, and the two configurations are mutually exclusive at
    this boundary.
    """
    kwargs: dict[str, Any] = dict(
        ori_kv=swa_kv_cache,
        ori_block_table=swa_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        sinks=sinks,
        metadata=metadata,
        softmax_scale=softmax_scale,
        cmp_ratio=cmp_ratio,
        # Three configurations, two of which use mode 0 for different reasons:
        # a compressed stream is present, or the original stream is selected by
        # an explicit index list. Only the third -- window bounds, no indices --
        # is sliding-window mode. Window bounds go to -1 whenever the mode is
        # not 4, because "non-sliding-window modes require ori_win_left =
        # ori_win_right = -1".
        ori_mask_mode=(SLIDING_WINDOW_MASK_MODE
                       if is_plain_sliding_window(attends_compressed,
                                                   ori_sparse_indices)
                       else NO_MASK_MODE),
        ori_win_left=(ori_win_left if is_plain_sliding_window(
            attends_compressed, ori_sparse_indices) else -1),
        ori_win_right=(ori_win_right if is_plain_sliding_window(
            attends_compressed, ori_sparse_indices) else -1),
        layout_q="TND",
        layout_kv=layout_kv,
    )

    if ori_sparse_indices is not None:
        kwargs["ori_sparse_indices"] = ori_sparse_indices

    if attends_compressed:
        if cmp_kv is None:
            raise ValueError(
                "attends_compressed is set but no cmp_kv was supplied. The "
                "layer would then be configured for a compressed stream -- "
                "mask mode 0, window bounds disabled -- while attending only "
                "its window, which the operator has no reason to reject.")
        kwargs.update(
            cmp_kv=cmp_kv,
            cmp_block_table=cmp_block_table,
            cmp_mask_mode=CMP_MASK_MODE,
        )
        if seqused_cmp_kv is not None:
            kwargs["seqused_cmp_kv"] = seqused_cmp_kv
        if cmp_residual_kv is not None:
            kwargs["cmp_residual_kv"] = cmp_residual_kv
        if cmp_sparse_indices is not None:
            kwargs["cmp_sparse_indices"] = cmp_sparse_indices

    return validate_sparse_flash_mla_args(kwargs, q, num_requests)
