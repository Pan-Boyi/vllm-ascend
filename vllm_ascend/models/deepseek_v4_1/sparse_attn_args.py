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
#: Causal. The operator lists {0, 3, 4} for ori_mask_mode; 3 is the only one
#: of the three that masks anything without window bounds, which a layer
#: attending a compressed stream at ratio != 1 is not allowed to have.
CAUSAL_MASK_MODE = 3
CMP_MASK_MODE = 3

#: The operator's own mode rules, read out of its host library rather than
#: inferred from the error we happened to hit:
#:
#:   /workspace/code/l00960396/cann92/cann-9.2.0-beta.2/opp/built-in/
#:     op_impl/ai_core/tbe/op_host/lib/linux/x86_64/libophost_transformer.so
#:
#: verbatim, via ``strings``:
#:
#:   "ori_sparse_indices is only supported in SWA mode."
#:   "SWA mode requires cmp_mask_mode=0"
#:   "SWA mode requires cmp_ratio=1"
#:   "Sparse ori_kv mode requires ori_mask_mode=cmp_mask_mode=0"
#:   "Ori_topk_length is required for SWA ori sparse"
#:   "Cmp_topk_length is only supported in CSA mode"
#:   "When cmp_sparse_indices is not nullptr(CSA), cmp_sparse_indices cannot
#:    be empty tensor"
#:   "cmp_residual_k is required when mask_mode is 3 with cmp_ratio not equal
#:    to 1!"
#:
#: 🔴 This resolves a contradiction that looked like two rules about one
#: parameter: one text demanded ``cmp_mask_mode != 0``, another demanded
#: ``cmp_mask_mode == 0``. They are not about one mode. Supplying
#: ``ori_sparse_indices`` *selects* SWA, and SWA's rules then apply to
#: everything else in the call -- so the two texts are each correct within
#: their own mode, and a call that satisfies both entry conditions at once
#: satisfies neither mode's requirements. The contradiction was the finding.
#:
#: 🔴 Behaviour that is **not** in any of those strings, measured on the bench
#: by the kernel line and recorded here because it is the sixth time two
#: operators have shared a parameter name -- and the first time their
#: behaviour is *opposite* rather than merely different:
#:
#:   SparseFlashMla            does NOT read past ``seqused_ori_kv``
#:   QuantLightningIndexerV2   DOES: its range is block-aligned,
#:                             ``ceil(seqused_k / block_size) * block_size``
#:
#: One code path computes the lengths for both. At block_size 128 a five-token
#: request lets the indexer touch ``[0, 128)`` -- five real tokens and 123
#: slots holding whatever the previous owner of that block wrote. Nothing
#: errors: the indices are in range and the values are real numbers belonging
#: to someone else. ``qli_observe`` reports the gap per call.
#:
#: ⚠️ What is **not** established: which mode a call lands in when several
#: optional tensors are present, beyond the two implications quoted above
#: (ori_sparse_indices -> SWA, cmp_sparse_indices -> CSA). The strings name
#: ORI_SPARSE, ORI_CMP_SPARSE, CSA, HCA and SWA; nothing here says how they
#: are chosen, and none of it is guessed at.


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


def ori_sparse_selection(attends_compressed: bool, req) -> tuple:
    """``(indices, topk_length)`` together, because they are one decision.

    Returning them as a pair is the point. The operator reads
    ``oriSparseBlockCount = hasOriTopkLength ? Min(length, width) : width``,
    so a call that supplies indices without a length has declared every slot
    in the row valid -- the ``-1`` padding included -- and the kernel gathers
    at those. Measured on the bench with one variable changed: the same table
    crashes at the declared full width and passes at its true length.

    ``Min`` looks like it guards this and does not. It clamps a length that
    exceeds how wide the table is; nothing downstream knows how many entries
    were actually written, so nothing can clamp to that.

    The length is carried from whoever built the table -- both builders
    already compute it and both used to discard it -- and never recounted
    here. A second count is a second source that can drift from the first.
    """
    indices = select_ori_sparse_indices(
        attends_compressed,
        getattr(req, "vision_swa_indices", None),
        getattr(req, "dspark_swa_indices", None))
    return indices, getattr(req, "ori_topk_length", None)


def is_plain_sliding_window(attends_compressed: bool,
                             ori_sparse_indices,
                             cmp_ratio: int = 2) -> bool:
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

    🔵 ``cmp_ratio`` is what decides whether the window may be sent at all,
    and it splits V4.1's compressed layers in two. The operator's rule is
    "SWA mode requires cmp_ratio=1"; ``_dsa_swa_only_cmp_ratio`` sends 1 for
    every layer whose configured ratio is 0 or 1, so:

        layers 0,1    ratio 0 -> cmp_ratio 1, no compressed stream  mode 4
        layers 20..39 ratio 1 -> cmp_ratio 1                        mode 4
        layers 2..19  ratio 2 -> cmp_ratio 2, window forbidden      not mode 4

    ⚠️ So twenty of the thirty-eight unmasked layers can have the real window
    back, not a bare causal mask. Treating all thirty-eight the same was the
    first shape of this fix and it was "less wrong", not right: it would have
    thrown away ``sliding_window=128`` on the half of them entitled to it.

    ⚠️ This is the **mode-4 predicate**, and only that. What masks the
    original stream on a layer that is *not* mode 4 is a separate question
    with a separate answer -- see ``ori_mask_mode_for``. Folding the two
    together is what left 38 layers with no masking at all.
    """
    if ori_sparse_indices is not None:
        return False
    if not attends_compressed:
        return True
    # A compressed layer may still take the window, but only where the
    # operator allows one: "SWA mode requires cmp_ratio=1".
    return _causal_without_indices() and cmp_ratio == 1


def ori_mask_mode_for(attends_compressed: bool, ori_sparse_indices,
                      cmp_ratio: int = 2) -> int:
    """What masks this layer's original stream.

    Three configurations, and the third one had no answer:

    ===============================  ==============================  =========
    indices present                  the list masks it               0
    no indices, no compressed stream the window masks it             4
    no indices, compressed stream    **nothing masked it**           0 -> 3
    ===============================  ==============================  =========

    🔴 Mode 0 is NO_MASK in the operator's own words ("When mask_mode is 0
    (NO_MASK), attn_mask should not be provided"). V4.1 puts 38 of its 40
    layers in the third row -- layers 2..19 carry ratio 2 and 20..39 carry
    ratio 1, and ``attends_compressed_stream`` is true for all of them -- so
    almost the whole model has been attending its original stream with no
    window, no causality and, since the mm-ranges switch, no index list
    either. Nothing rejects that; it produces fluent text that means nothing.

    That third row was self-consistent while the indices were always present,
    because both SWA index builders write a causal window into the list.
    Turning ``VLLM_ASCEND_DSV41_MM_RANGES_REQUIRE_SPANS`` on removed the list
    -- correctly, it was the wrong path -- and left mode 0 supplying nothing.
    ⇒ The switch fixed one defect and exposed another.

    ⚠️ The answer is **3 (CAUSAL), not 4**. Mode 4 was the first guess and the
    operator rules it out for these layers::

        "SWA mode requires cmp_ratio=1"           (layers 2..19 carry 2)
        "Non-sliding-window modes require ori_win_left=ori_win_right=-1"
        "Ori_mask_mode should be {0, 3, 4} on ..."

    so a ratio-2 layer cannot have window bounds at all, and 3 is the only
    remaining mode that masks anything. It also needs no window bounds, which
    is what the second rule requires of it. This line's own host check already
    encoded the first rule and refused the mode-4 attempt -- the guard was
    right and the hypothesis it refused was mine.

    Behind ``VLLM_ASCEND_DSV41_ORI_WINDOW_WITHOUT_INDICES``, default off, so
    the numerics change is one variable in one flight.
    """
    if is_plain_sliding_window(attends_compressed, ori_sparse_indices,
                               cmp_ratio):
        _count_mask_mode(SLIDING_WINDOW_MASK_MODE, cmp_ratio)
        return SLIDING_WINDOW_MASK_MODE
    if (ori_sparse_indices is None and attends_compressed
            and _causal_without_indices()):
        from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

        announce_once(
            "ori_window_without_indices",
            "ORI_WINDOW_WITHOUT_INDICES=1 sent ori_mask_mode=%d on a layer "
            "attending a compressed stream; without it this call would be "
            "NO_MASK.", CAUSAL_MASK_MODE)
        _count_mask_mode(CAUSAL_MASK_MODE, cmp_ratio)
        return CAUSAL_MASK_MODE
    _count_mask_mode(NO_MASK_MODE, cmp_ratio)
    return NO_MASK_MODE


#: What was actually sent, by (mask mode, cmp_ratio). Runtime evidence, not a
#: guard: an assertion says the code was changed, this says the change reached
#: the operator. Both are needed and neither substitutes for the other -- an
#: experiment whose only observable is "the variable was set" cannot tell
#: "the branch did not run" from "the branch ran and did not help".
_MASK_MODE_COUNTS: dict = {}


def _count_mask_mode(mode: int, cmp_ratio: int) -> None:
    key = (mode, cmp_ratio)
    first = key not in _MASK_MODE_COUNTS
    _MASK_MODE_COUNTS[key] = _MASK_MODE_COUNTS.get(key, 0) + 1
    if first:
        from vllm.logger import logger

        logger.info(
            "DSV4.1 ori_mask_mode: first call with mode=%d at cmp_ratio=%d "
            "(0=NO_MASK, 3=CAUSAL, 4=SLIDING_WINDOW). Histogram so far: %s",
            mode, cmp_ratio, mask_mode_histogram())


def mask_mode_histogram() -> dict:
    """``{"mode=M,cmp_ratio=R": calls}`` -- what the operator was actually
    handed, so a run that changed nothing is distinguishable from a run whose
    switch never reached the branch."""
    return {f"mode={mode},cmp_ratio={ratio}": count
            for (mode, ratio), count in sorted(_MASK_MODE_COUNTS.items())}


def reset_mask_mode_histogram() -> None:
    _MASK_MODE_COUNTS.clear()


def _causal_without_indices() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_ORI_WINDOW_WITHOUT_INDICES == "1"


_DEVICE_PRED = {"evaluated": 0, "skipped_under_capture": 0}


def device_predicate_accounting() -> dict:
    """无条件可读，包括全为零。"""
    return dict(_DEVICE_PRED)


def _device_scalar(t, when_capturing: int) -> int:
    """把设备上的标量取回 host —— 捕获期返回 ``when_capturing`` 并记账。

    🔴 ``int(t)`` 同样是一次 host 读回（107027 / LocalScalarDenseNpu.cpp:23）。
    ⚠️ ``when_capturing`` 由调用方给，且必须是一个【能让下游检查通过】的值 ——
    这是偏宽松的方向，所以要在调用处写清为什么那个值是安全的，并靠
    ``device_predicate_accounting()`` 的 skipped 计数留痕。
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _stream_is_capturing

    if _stream_is_capturing():
        _DEVICE_PRED["skipped_under_capture"] += 1
        return when_capturing
    _DEVICE_PRED["evaluated"] += 1
    return int(t)


def _device_predicate(t) -> bool:
    """把一个【设备上的布尔张量】变成 host 布尔 —— 捕获期【跳过】并视为通过。

    🔴 为什么要跳过：``bool(t)`` 把一个标量从设备读回 host，而捕获期对 stream 的
    同步在任何捕获模式下都非法（实测 107027 /
    ``LocalScalarDenseNpu.cpp:23``；CANN 把这一类叫
    "dynamic control flow on tensor values"）。

    ⚠️ **方向是偏宽松的那一侧**，而偏宽松的错不会被后续工作发现 —— 所以：
      · 只在捕获期跳过。捕获只发生在 warmup，那时喂的是 dummy 数据，
        这些断言本来也说明不了生产的情况。
      · 每次跳过都记账（``device_predicate_accounting()``），
        否则"跳过了"和"检查通过了"读数一样。
      · 真正的生产请求走的是 replay，不在捕获里 ⇒ eager 路径上这些检查一次不少。
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _stream_is_capturing

    if _stream_is_capturing():
        _DEVICE_PRED["skipped_under_capture"] += 1
        return True
    _DEVICE_PRED["evaluated"] += 1
    return bool(t)


def _check(condition: bool, operator_message: str, detail: str = "") -> None:
    if condition:
        return
    raise ValueError(
        f"sparse_flash_mla would reject this call: {operator_message}"
        + (f"\n  {detail}" if detail else ""))


def _cu_seqlens_diagnosis(t) -> str:
    """把【它长什么样】接在报错后面 —— 判据要给诊断,不要只报一个是非。

    🔴 2026-09-15：这一条在图捕获的 warmup 里拦下了一次,而报错只说了
    "not non-decreasing"。于是"是哪一位下降的"只能靠读 model_runner 的
    padding 代码去推 —— 推了三个分支都自洽,一个都不能证实。
    值就在手边,不打出来是白白扔掉唯一的证据。

    ⚠️ 只在【已经要抛异常】的路径上调用:它做 host 读回,捕获期非法。
       而能走到这里就说明我们不在捕获期(捕获期 _device_predicate 直接放行)。
    """
    try:
        v = t.detach().to("cpu").tolist()
    except Exception as exc:                       # 读不回来也要说清是读不回来
        return f"\n  (取不到值: {type(exc).__name__}: {exc})"
    bad = [i for i in range(1, len(v)) if v[i] < v[i - 1]]
    head = v if len(v) <= 24 else v[:12] + ["…"] + v[-12:]
    # 🔴 在 host 拷贝上把同一个谓词【复算一遍】。设备说"不单调"而 host 说
    #    "单调",不是两个读数打架而是一条定位:此刻设备侧的读回本身不可信
    #    (2026-09-15 实测:守卫开火前一秒,两个 tap 刚在 aclnnNonzeroV2 上失败)。
    #    两侧都说"不单调"才是数据真的有问题。
    verdict = ("设备侧=不单调 / host侧=不单调  ⇒ 数据确实有问题"
               if bad else
               "设备侧=不单调 / host侧=**单调**  ⇒ 🔴 两侧矛盾:"
               "此刻设备读回不可信,先查这之前有没有算子失败污染了上下文")
    where = (f"\n  第一处下降在下标 {bad[0]}: {v[bad[0]-1]} -> {v[bad[0]]}"
             f"  (共 {len(bad)} 处下降, 长度 {len(v)})" if bad else
             f"\n  (host 侧扫不到任何下降位, 长度 {len(v)})")
    return f"\n  {verdict}\n  cu_seqlens_q = {head}{where}"


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
        # 捕获期返回 -1：下游判据是 `highest >= num_blocks`，-1 必然通过。
        highest = _device_scalar(table.max(), -1)
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
        # 捕获期返回 -1：下游判据是 `longest > capacity`，-1 必然通过。
        longest = _device_scalar(used.max(), -1)
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

    _check(_device_predicate(metadata.any()),
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
        # ⚠️ 这一条【区分力为零】,留着只是为了在 num_requests 改成别的来源时能响:
        #    唯一的调用点把 num_requests 算作 query_start_loc.numel()-1,
        #    于是 numel()==num_requests+1 是个恒真式。它看起来像长度检查,
        #    而它永远不会拦下任何东西 —— 不随被测量变化的校验位比读不到更糟。
        _check(cu_seqlens_q.ndim == 1 and cu_seqlens_q.numel() == num_requests + 1,
               "layout_q=TND requires cu_seqlens_q, cumulative, length T+1",
               f"got shape {tuple(cu_seqlens_q.shape)} for {num_requests} requests"
               " (⚠️ 恒真式:num_requests 就是从这个 numel 算出来的)")
        # 🔴 先求值、只在【要抛】的时候才造 detail。
        #    上一版把 _cu_seqlens_diagnosis(...) 直接写进 _check 的第三个实参,
        #    而 Python 在调用前会把三个实参**全部**求值 ⇒ 每一次 forward 都做
        #    一次 host 读回。检查通过时它不打印任何东西,所以这个代价在日志里
        #    完全看不见 —— 一个只在"没出事"时才付、且永远不会被注意到的代价。
        _ok_device = _device_predicate(
            torch.all(cu_seqlens_q[1:] >= cu_seqlens_q[:-1]))
        if not _ok_device:
            _check(False,
                   "layout_q=TND requires cu_seqlens_q, cumulative, length T+1",
                   "the values are not non-decreasing, so they are lengths "
                   "rather than a cumulative sum"
                   + _cu_seqlens_diagnosis(cu_seqlens_q))

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
        # 🔴 The operator's rule is about the *ratio*, not about the presence
        # of a compressed stream. This used to check both, and the second
        # check was wider than the text it quoted: it refused
        # cmp_ratio=1 layers that also carry a compressed stream, which is
        # every one of layers 20..39 -- exactly the twenty that are entitled
        # to the window. A guard stricter than the rule it cites reads as
        # rigour and costs the configuration it excluded.
        _check(cmp_ratio == 1, "SWA mode requires cmp_ratio = 1",
               f"window bounds are set but cmp_ratio={cmp_ratio}")
    if has_cmp_kv:
        _check(not windowed or cmp_ratio == 1,
               "non-sliding-window modes require ori_win_left = ori_win_right = -1",
               f"a compressed stream at cmp_ratio={cmp_ratio} cannot be in "
               "sliding-window mode, but the window bounds are "
               f"{kwargs.get('ori_win_left')!r}/"
               f"{kwargs.get('ori_win_right')!r}")

    # 🔴 There is deliberately no "has_cmp_kv requires seqused_cmp_kv" rule
    # here, and the absence is the finding.
    #
    # These kwargs are pre-adapter. The bf16 plan's operator is
    # vllm_ascend.attention.sparse_flash_mla.sparse_flash_mla, which runs
    # after this and derives the compressed lengths itself
    # (seqused_ori_kv // cmp_ratio) when they are absent. So "absent here" is
    # not "absent at the operator", and the rule as first written rejected a
    # call that would have completed -- it killed the first smoke run to get
    # past layer 1.
    #
    # The obvious repair -- allow it when the adapter *could* derive one --
    # turns out to be unreachable: cmp_ratio > 0 is already enforced above and
    # seqused_kv is a required parameter of this function, so the escape
    # clause is always true. A rule that cannot fire is worse than no rule,
    # because it reads as coverage. So the rule is gone and the two conditions
    # it would have depended on are pinned by tests instead, where removing
    # one of them turns red rather than quietly re-opening this.
        residual = kwargs.get("cmp_residual_kv")
        # 🔴 Not "cmp_ratio == 1 requires it". The operator does require it
        # there, but these kwargs are pre-adapter and the adapter fills it in
        # (`seqused_ori_kv % cmp_ratio`, zeros at ratio 1) exactly when a
        # compressed stream is present. Demanding it here refuses a call that
        # would have completed -- which is what stopped the first launch to
        # reach layer 20, the first ratio-1 layer.
        #
        # Fourth time a rule in this function has been the operator's rather
        # than the caller's. The function speaks the caller's vocabulary
        # throughout (seqused_kv, not seqused_ori_kv); a rule about what the
        # operator receives cannot be evaluated one step before the step that
        # produces it. What is still checked is the shape of a residual that
        # *was* supplied -- that part is about this caller's own value.
        if residual is not None:
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

    seqused_cmp_kv = kwargs.get("seqused_cmp_kv")
    if seqused_cmp_kv is not None:
        # One value per request. A cumulative array has num_requests + 1, and
        # that count is the only cheap thing that separates the two: dtype,
        # device and magnitude are identical, and the operator would read a
        # cumulative array as lengths without complaint.
        _check(seqused_cmp_kv.numel() == num_requests,
               "seqused_cmp_kv must hold one length per request",
               f"got {seqused_cmp_kv.numel()} values for {num_requests} "
               f"requests"
               + (" -- that is num_requests + 1, the length of a *cumulative* "
                  "sequence-length array. cu_cmp_seqlen_list is that array; "
                  "this parameter wants seq_lens // cmp_ratio."
                  if seqused_cmp_kv.numel() == num_requests + 1 else ""))
        # ⚠️ And the rank, which the count alone does not pin: a [1, 1]
        # tensor has one element too. Its two siblings here -- cu_seqlens_q
        # and cmp_residual_kv -- have had both checks from the start; this
        # one had only the count, and "two of the three parameters are
        # checked" is the shape that produced the last three defects in this
        # file.
        _check(seqused_cmp_kv.ndim == 1,
               "seqused_cmp_kv must be a 1-D per-request array",
               f"got {seqused_cmp_kv.ndim} dims, shape "
               f"{tuple(seqused_cmp_kv.shape)}; the value comes from the "
               "compressed group's req metadata (seqused_cmp_kv, computed as "
               "seq_lens // cmp_ratio), not from cu_cmp_seqlen_list")

    expected_rank = LAYOUT_KV_RANK.get(layout_kv)
    if expected_rank is not None:
        for name in ("ori_kv", "cmp_kv"):
            cache = kwargs.get(name)
            if cache is None:
                continue
            # Checked here as well as fixed at the call site, because the
            # operator saying so clearly this time is a property of the
            # operator, not a guarantee we hold. The same class of mistake
            # arrived truncated one launch earlier.
            _check(cache.ndim == expected_rank,
                   f"{name} must have {expected_rank} axes under "
                   f"layout_kv={layout_kv}",
                   f"got {cache.ndim} {tuple(cache.shape)}; the unit KV-head "
                   "axis is the one usually missing -- pass the cache through "
                   "kv_with_layout_rank rather than adding it by hand")

    _check_descriptor_is_populated(kwargs, q)
    _check_block_table_addresses_this_cache(kwargs)
    _check_modes_are_not_both_requested(kwargs)
    _report_compressed_arguments(kwargs)

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


_REPORTED_COMPRESSED: set = set()


def _report_compressed_arguments(kwargs: dict[str, Any]) -> None:
    """Say once, per distinct shape, what the compressed stream was handed.

    🔴 Printed from the kwargs the operator receives, not from the variables
    that produced them -- the two have come apart twice on this line already
    (the pre-adapter vocabulary, and a descriptor built from another group).

    ⚠️ Shape, not just presence. ``seqused_cmp_kv`` wants one value per
    request; the cumulative ``cu_cmp_seqlen_list`` is rank 1 as well and one
    element longer, so a rank check passes on either. The rank table can stop
    a wrong *shape* and not a wrong *source*, and "how far the operator
    reads" is exactly the kind of input that changes the output without
    changing anything that looks wrong.
    """
    if not kwargs.get("attends_compressed", "cmp_kv" in kwargs):
        return

    def _shape(key):
        value = kwargs.get(key)
        if value is None:
            return f"{key}=None"
        if isinstance(value, torch.Tensor):
            return f"{key}{tuple(value.shape)}:{value.dtype}"
        return f"{key}={value!r}"

    parts = [_shape(k) for k in ("seqused_cmp_kv", "cmp_residual_kv",
                                 "cmp_sparse_indices", "cmp_topk_length",
                                 "cmp_block_table", "cmp_mask_mode",
                                 "cmp_ratio")]
    signature = " ".join(parts)
    if signature in _REPORTED_COMPRESSED:
        return
    _REPORTED_COMPRESSED.add(signature)
    from vllm.logger import logger

    values = kwargs.get("seqused_cmp_kv")
    logger.info(
        "DSV4.1 compressed args (as passed to the operator): %s | "
        "seqused_cmp_kv values=%s", signature,
        "n/a" if not isinstance(values, torch.Tensor)
        else [int(v) for v in values.reshape(-1)[:8].detach().cpu()])


def reset_compressed_report() -> None:
    _REPORTED_COMPRESSED.clear()


def _check_modes_are_not_both_requested(kwargs: dict[str, Any]) -> None:
    """Refuse a call that asks the operator for two modes at once.

    ``ori_sparse_indices`` is not an extra input to whatever mode the call is
    already in -- it *selects* SWA ("ori_sparse_indices is only supported in
    SWA mode"), and SWA then constrains the rest of the call: cmp_mask_mode
    must be 0 and cmp_ratio must be 1. Sending it alongside a compressed
    stream at mode 3 asks for SWA and for compressed-causal simultaneously,
    and the operator rejects the pair rather than either half::

        EZ0027: Parameters ori_mask_mode and cmp_mask_mode of SparseFlashMla
                have incorrect values 0, 3.
                Reason: Sparse ori_kv mode requires ori_mask_mode=
                cmp_mask_mode=0.

    ⚠️ The likely origin for a text-only request is not this file. Both index
    builders run whenever the checkpoint is a mm-prefix LM, and
    ``mm_req_doc_ranges`` for a text-only request is ``{0: []}`` -- truthy.
    ``mm_ranges_have_spans`` exists to stop that and is **default off**
    (``VLLM_ASCEND_DSV41_MM_RANGES_REQUIRE_SPANS``), so today every text-only
    prefill still builds vision indices, ``select_ori_sparse_indices`` returns
    them, and a ratio-2 layer arrives here holding both. Setting that switch
    to "1" is the single-variable way to test it.

    Raising rather than dropping the indices. Dropping them would produce a
    call the operator accepts, and nothing downstream would say the layer's
    original stream is no longer being selected the way the caller asked --
    a wrong answer that stops looking wrong. The operator refuses this pair
    anyway; all this adds is which of the two decisions to go and change.
    """
    if kwargs.get("ori_sparse_indices") is None:
        return
    cmp_mask_mode = kwargs.get("cmp_mask_mode", NO_MASK_MODE)
    if cmp_mask_mode == NO_MASK_MODE:
        return
    _check(
        False,
        "Sparse ori_kv mode requires ori_mask_mode=cmp_mask_mode=0 (EZ0027); "
        "and separately, ori_sparse_indices is only supported in SWA mode, "
        "which requires cmp_ratio=1",
        f"this call supplies ori_sparse_indices "
        f"{tuple(kwargs['ori_sparse_indices'].shape)} together with "
        f"cmp_mask_mode={cmp_mask_mode} and cmp_ratio="
        f"{kwargs.get('cmp_ratio')}. Those select two different modes. For a "
        "text-only request the indices most likely come from the vision "
        "builder, which runs because mm_req_doc_ranges is {0: []} and "
        "VLLM_ASCEND_DSV41_MM_RANGES_REQUIRE_SPANS is off; setting it to 1 "
        "is the one-variable test.")


def kv_with_layout_rank(cache, layout_kv: str):
    """Give the operator a KV cache at the rank ``layout_kv`` describes.

    ⚠️ Named for ``ori_kv`` when it was written, and for a while only ``ori_kv``
    went through it -- so ``cmp_kv`` reached the operator one axis short and
    was refused ("Cmp_kv strideVec size[3] not match kvLayout expect len[4]").
    Every KV cache argument of this call goes through here; a helper that one
    of two callers uses is a helper that will be missed again.

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
    #: Valid entries per row of ori_sparse_indices. Required whenever indices
    #: are supplied; see ori_sparse_selection for why the default is unsafe.
    ori_topk_length: torch.Tensor | None = None,
    cmp_sparse_indices: torch.Tensor | None = None,
    #: Valid entries per row of cmp_sparse_indices. Required whenever the
    #: compressed indices are supplied, for the reason ori_topk_length is.
    cmp_topk_length: torch.Tensor | None = None,
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
        ori_mask_mode=ori_mask_mode_for(attends_compressed,
                                        ori_sparse_indices, cmp_ratio),
        ori_win_left=(ori_win_left if is_plain_sliding_window(
            attends_compressed, ori_sparse_indices, cmp_ratio) else -1),
        ori_win_right=(ori_win_right if is_plain_sliding_window(
            attends_compressed, ori_sparse_indices, cmp_ratio) else -1),
        layout_q="TND",
        layout_kv=layout_kv,
    )

    if ori_sparse_indices is not None:
        if ori_topk_length is None:
            raise ValueError(
                "ori_sparse_indices was supplied without ori_topk_length. "
                "The operator then takes the whole row as valid -- "
                f"{tuple(ori_sparse_indices.shape)} here -- and gathers at "
                "the -1 padding, which faults every core at one address. "
                "Omitting the length is not a safe default: it is the most "
                "dangerous one, and it is what production was doing.")
        kwargs["ori_sparse_indices"] = ori_sparse_indices
        kwargs["ori_topk_length"] = ori_topk_length

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
            # [T, KV_N, K2] under layout_q=TND. Checked here because the
            # operator's own refusal is truncated -- "has inc[orrect shape"
            # and nothing about what it wanted -- so a rank mistake costs a
            # launch and yields no number. The sibling operators' docs state
            # the rank and K2 = 512 (sparse_attn_sharedkv/README.md:65).
            _check(cmp_sparse_indices.ndim == 3,
                   "cmp_sparse_indices must be [T, KV_N, K2] under layout_q=TND",
                   f"got {cmp_sparse_indices.ndim} dims "
                   f"{tuple(cmp_sparse_indices.shape)}; the KV-head axis is "
                   "the one usually missing, since the buffer upstream keeps "
                   "is [tokens, index_topk]")
            _check(cmp_sparse_indices.shape[0] == q.shape[0],
                   "cmp_sparse_indices must have one row per query token",
                   f"got {cmp_sparse_indices.shape[0]} rows for "
                   f"{q.shape[0]} query tokens")
            # The same pairing the ori side enforces. The selection buffer is
            # -1 padded, and omitting the length declares the padding valid --
            # the kernel then gathers at those slots. Measured single-variable
            # on the ori side: identical table, crashes at the full width,
            # passes at its true length.
            if cmp_topk_length is None:
                raise ValueError(
                    "cmp_sparse_indices was supplied without cmp_topk_length. "
                    "The operator then takes the whole row as valid -- "
                    f"{tuple(cmp_sparse_indices.shape)} here -- and gathers "
                    "at the -1 padding the indexer wrote. Omitting the length "
                    "is not a safe default; it is the most dangerous one.")
            kwargs["cmp_sparse_indices"] = cmp_sparse_indices
            kwargs["cmp_topk_length"] = cmp_topk_length

    return validate_sparse_flash_mla_args(kwargs, q, num_requests)
