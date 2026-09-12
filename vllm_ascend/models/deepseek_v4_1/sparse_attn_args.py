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

⚠️ One stated constraint is deliberately **not** modelled here: "ori-only
sparse mode requires both mask modes to be 0". Checking it needs a way to tell
"ori-only sparse mode" -- which uses explicit ori_sparse_indices -- from plain
sliding-window mode, which also has no compressed stream but does pass mask
mode 4. Nothing measured distinguishes them, and inventing a rule to tell them
apart would reject valid calls or accept invalid ones with equal confidence.
It is recorded here rather than silently assumed away; it remains a live
constraint that this function will not catch for you.

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

#: Mask mode for a layer that attends only its own (window) stream, and the
#: mode required once a compressed stream is present.
ORI_ONLY_MASK_MODE = 4
COMPRESSED_MASK_MODE = 0
CMP_MASK_MODE = 3


def _check(condition: bool, operator_message: str, detail: str = "") -> None:
    if condition:
        return
    raise ValueError(
        f"sparse_flash_mla would reject this call: {operator_message}"
        + (f"\n  {detail}" if detail else ""))


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
        ori_mask_mode=COMPRESSED_MASK_MODE if attends_compressed else ORI_ONLY_MASK_MODE,
        ori_win_left=-1 if attends_compressed else ori_win_left,
        ori_win_right=-1 if attends_compressed else ori_win_right,
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
