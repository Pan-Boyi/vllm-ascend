# SPDX-License-Identifier: Apache-2.0
"""``SparseAttnIndexer.forward_oot`` for DeepSeek-V4.1 on Ascend.

Upstream's base ``forward_oot`` forwards to ``forward_native``, which raises
"only implemented for CUDA, ROCm, XPU and CPU platforms" -- so an unimplemented
OOT indexer and a missing registration produce the identical message. The
registration goes through ``REGISTERED_ASCEND_OPS``, keyed by the in-tree class
name because ``CustomOp.__new__`` substitutes on ``cls.__name__``; that runs in
``NPUWorker.__init__``, before any model is built.

Scope is narrow by V4.1's own doing. ``skip_k_cache_insert`` is the literal
``True`` and ``k`` the literal ``None``: the layer writes the indexer K cache
itself before calling this, so there is no insert half here. What is left is
read the cache, score, take top-k, and write the caller's buffer.

Two things in here are assumptions with assertions rather than verified facts,
and they are marked. Both are answerable only on a card, and both are the first
things to compare against a golden tap.
"""

from __future__ import annotations

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

from vllm_ascend.models.deepseek_v4_1.indexer_select import (
    assert_candidate_mask_is_vacuous, compressed_lengths, select_topk)

#: Upstream's "no token here" sentinel in topk_indices_buffer.
NO_TOKEN = -1

#: Where the written-row count is recorded. It lives **on the buffer tensor**
#: rather than on this module or on the indexer, and that is the point:
#: ``topk_indices_buffer`` is one object shared by every layer, while
#: ``self.indexer`` exists only on index-source layers. A count kept anywhere
#: narrower than the buffer is a count the buffer's other readers cannot see,
#: which is exactly how the last companion value came apart.
_ROWS_WRITTEN_ATTR = "_dsv41_rows_written"


def _current_forward() -> object | None:
    """The object identifying the forward pass now running, if there is one.

    ``set_forward_context`` builds a fresh ``ForwardContext`` per model
    forward and every layer inside that forward sees the same one, so its
    identity is a step token that needs no counter of ours to stay in sync
    with the thing it is counting.

    ``None`` outside a forward -- unit tests, and anything that reaches this
    module off the model path. That is a third state, not a step.
    """
    try:
        return get_forward_context()
    except Exception:
        return None


_WRITERS: dict = {}


def writer_accounting() -> dict:
    """谁写过这个 buffer、各写了多少行。无条件可读，包括为空。"""
    return dict(_WRITERS)


#: 这个 tap 有没有拿到它需要的配置。**不是布尔**：把"没配置所以跳过"
#: 和"配置有、层号算出来了但这层不记读"分成两格，否则 dump 为空时读不出
#: 是仪器没启动还是这层本来就不该有。
_TAP_CONFIG: dict = {"tapped": 0, "no_config": 0, "not_recorded": 0}


def tap_config_accounting() -> dict:
    """indexer 读侧 tap 的覆盖量。无条件可读，包括全零。"""
    return dict(_TAP_CONFIG)


def _note_writer(who: str, rows: int) -> None:
    """记下【是哪个写者】记的那个行数，并在每个写者第一次出现时留一条日志。

    🔴 为什么需要（2026-09-15）：报错只说 "reading 8 rows ... only 5 were written"，
    而这个 buffer 有【两个写者，两把不同的尺】：
        forward_oot           记 hidden_states.shape[0]（padded）
        短上下文 triton 核     记 grid[0]（真实 token 数）
    我把"写侧记了 5"直接推成了"写侧的张量是 5 行"，中间跳过了"是哪个写者记的"
    —— 而那一步的答案决定整条归因。⇒ 让写者身份变成一条产物，不再靠排除法。
    """
    prev = _WRITERS.get(who)
    _WRITERS[who] = rows
    if prev is None:
        from vllm.logger import logger
        logger.info(
            "DSV4.1 indexer selection writer %r marked %d rows (first time). "
            "Two writers use two different rulers: forward_oot marks the "
            "padded token count, the short-context triton kernel marks the "
            "real one. Which of them ran decides how a 'read N wrote M' "
            "mismatch should be read.", who, rows)


def mark_rows_written(buffer, num_rows: int) -> None:
    """Record how many leading rows of ``buffer`` this step actually wrote.

    The forward pass is recorded alongside the count. A bare integer cannot
    tell "4000 rows written just now" from "4000 rows written eight steps
    ago", and those are opposite answers: the first says the rows are this
    step's, the second says every one of them is residue. A check bit that
    does not move with the thing it certifies gives a green light exactly
    where the defect is.

    ⚠️ This keeps the context object alive until the next mark. That is one
    object, replaced every step, and it is deliberate: ``id()`` alone is reused
    after collection, so identity comparison needs the reference.
    """
    setattr(buffer, _ROWS_WRITTEN_ATTR, (_current_forward(), int(num_rows)))


def rows_written(buffer) -> "int | None":
    """Rows written into ``buffer`` this step, or ``None`` if unrecorded.

    ``None`` means nothing has written it since the process started. Upstream
    allocates it with ``torch.empty``, so its contents are then whatever was in
    HBM -- and reading that is one of the two explanations for a forward that
    is not deterministic under greedy. Distinguishing it from a real zero
    matters: one is "no tokens selected", the other is "do not read this".

    When the recorded forward is not the one now running, the count belongs to
    an earlier step. That is reported -- loudly, once -- and the count is still
    returned: a raise here would stop a launch on a hypothesis, and the
    hypothesis is exactly what the launch is supposed to test. The log line is
    the instrument; it cannot block a flight and it says so when the buffer
    being read is last step's.
    """
    record = getattr(buffer, _ROWS_WRITTEN_ATTR, None)
    if record is None:
        return None
    marked, rows = record
    current = _current_forward()
    if marked is not None and current is not None and marked is not current:
        _warn_selection_is_last_steps(rows)
    return rows


_WARNED_STALE_SELECTION = False


def _warn_selection_is_last_steps(rows: int) -> None:
    global _WARNED_STALE_SELECTION
    if _WARNED_STALE_SELECTION:
        return
    _WARNED_STALE_SELECTION = True
    from vllm.logger import logger

    logger.warning(
        "DSV4.1 indexer: topk_indices_buffer is being read in a forward pass "
        "that never wrote it -- the %d recorded rows were written by an "
        "earlier step. Every row read this step is that step's selection. "
        "This is one of the two candidate explanations for greedy decode "
        "being non-deterministic; it is logged rather than raised so the run "
        "continues and the other one stays observable.", rows)


def unpack_indexer_cache(kv_cache):
    """``(values, scales)`` out of whatever the runner bound for this group.

    The V4.1 indexer cache is its own KV cache group with its own spec
    (``AscendDeepseekV41IndexerCache``), so it does not have V4.0's seven-tuple
    shape and the V4.0 unpackers do not apply.

    ⚠️ An earlier version of this said the allocator hands back a pair for a
    spec carrying ``scale_dim=1``. It does not, and nobody should go looking
    for that behaviour: this cache's ``head_dim`` is already the *packed* byte
    width (``attention.py:82-90`` -- ``index_head_dim + index_head_dim//128*4``
    = 132, with upstream's own comment "head_dim already carries the fp8 scale
    padding"), so there is one allocation and the value/scale split lives
    inside it. The pair branch below is kept because an allocator that does
    hand over two tensors needs no splitting, not because this one does.

    The single-tensor case is the real one, and it is split by
    ``indexer_cache_view.split_indexer_cache`` using the byte offsets read out
    of the operator's kernel -- not by slicing at a plausible boundary.
    """
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) == 2:
            return kv_cache[0], kv_cache[1]
        raise ValueError(
            f"the V4.1 indexer cache came back as {len(kv_cache)} tensors; "
            "this expects (values, scales). V4.0's seven-tuple unpackers do "
            "not apply -- the V4.1 indexer is its own cache group with its own "
            "spec. Shapes: "
            f"{[tuple(t.shape) for t in kv_cache if hasattr(t, 'shape')]}")
    if isinstance(kv_cache, torch.Tensor):
        # The packed form, which is what the runner actually allocates here:
        # upstream builds this cache with head_dim already carrying the scale
        # padding (attention.py:82-90, 128 fp8 + 4 fp32 = 132), so the spec
        # reports one tensor and there is no separate scale allocation to
        # unwrap. The two arguments the read operator wants are views into it.
        from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import \
            split_indexer_cache

        return split_indexer_cache(kv_cache)

    raise ValueError(
        f"the V4.1 indexer cache came back as {type(kv_cache).__name__}, "
        "which is neither a (values, scales) pair nor the packed single "
        "tensor this knows how to address. npu_quant_lightning_indexer_v2 "
        "takes key and key_dequant_scale as two arguments and there is "
        "nothing here to derive them from.")


class AscendSparseAttnIndexer(SparseAttnIndexer):
    """The Ascend selection path. Everything else is upstream's."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 🔴 上游的 SparseAttnIndexer.__init__ 自己调了 get_current_vllm_config()，
        # 但只留下了 parallel_config —— 没有 self.vllm_config。读侧 tap 需要
        # hf_text_config 把 prefix 翻成层号，而【forward 时再去要已经太晚】：
        # get_current_vllm_config() 只在建模型的那个 context 里有效，之后拿到的
        # 是一个默认 VllmConfig（不报错，层号会全变成同一个）。
        # 所以在这里抓 —— 上游在同一个函数里调同一个 getter，作用域是对的。
        from vllm.config import get_current_vllm_config
        try:
            self._tap_hf_text_config = (
                get_current_vllm_config().model_config.hf_text_config)
        except Exception as exc:  # 建模型之外构造（单测/探针）时不该炸
            self._tap_hf_text_config = None
            from vllm.logger import logger
            logger.warning(
                "indexer 读侧 tap 取不到 hf_text_config(%s: %s)；"
                "该 tap 会被跳过并计入 tap_config_accounting()['no_config']",
                type(exc).__name__, exc)

    def forward_oot(self, hidden_states, q_quant, k, weights):
        if k is not None:
            raise ValueError(
                "the V4.1 indexer passes k=None and writes its own K cache "
                "(skip_k_cache_insert=True). A non-None k means this is being "
                "called from somewhere that expects the insert half, which "
                "this does not implement -- silently dropping it would leave "
                "the cache holding the previous step's keys.")
        if not self.skip_k_cache_insert:
            raise ValueError(
                "skip_k_cache_insert is False, so the caller expects this to "
                "quantise k and write the indexer K cache. It does not.")

        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            # Profile and dummy runs. Upstream returns the buffer untouched
            # here too; the caller reads the buffer, not the return value.
            #
            # 🔴 The line that used to be here said "during profiling nothing
            # reads it at all". That was wrong, and a traceback proved it:
            # _dummy_run -> _sparse_indexer_and_attn -> forward_mqa reads the
            # buffer on exactly this path. The read is real; what does not
            # follow from it is any conclusion about a real request, because
            # the values are uninitialised either way.
            #
            # The buffer is deliberately left unwritten and unmarked, so a
            # reader can tell "nobody wrote this" from "zero rows selected".
            # forward_mqa checks that distinction rather than assuming it.
            return self.topk_indices_buffer

        metadata = attn_metadata[self.k_cache.prefix]
        req = metadata.req_metadata
        if req is None:
            raise ValueError(
                f"{self.k_cache.prefix}: the attention metadata for this "
                "cache group has no req_metadata, which is where Ascend keeps "
                "seq_lens, query_start_loc and the block table. A batch with "
                "real tokens cannot select top-k without them.")

        num_tokens = hidden_states.shape[0]
        assert_candidate_mask_is_vacuous(
            metadata.max_seq_len, self.compress_ratio,
            self.candidate_blocks.shape[1] if self.candidate_blocks is not None
            else 0,
            self.candidate_block_size, self.k_cache.prefix)

        values, scales = unpack_indexer_cache(self.k_cache.kv_cache)
        q_values, q_scale = (q_quant if isinstance(q_quant, tuple)
                             else (q_quant, None))
        seqused_k, cmp_residual_k = compressed_lengths(req.seq_lens,
                                                       self.compress_ratio)
        key_dequant_scale = scales.squeeze(-2) if scales.ndim > 2 else scales

        # Print-only. QuantLightningIndexerV2 reads a block-aligned range,
        # unlike SparseFlashMla -- so what it may touch is not seqused_k but
        # ceil(seqused_k / block_size) * block_size, and the difference is
        # slots this request never wrote.
        from vllm_ascend.models.deepseek_v4_1.qli_observe import observe

        observe(prefix=self.k_cache.prefix, seqused_k=seqused_k,
                cu_seqlens_q=req.query_start_loc,
                block_table=req.block_table,
                block_size=req.storage_block_size)

        topk_idxs = select_topk(
            query=q_values[:num_tokens],
            q_scale=None if q_scale is None else q_scale[:num_tokens],
            key_cache=values,
            key_dequant_scale=key_dequant_scale,
            weights=weights[:num_tokens],
            index_topk=self.topk_tokens,
            compress_ratio=self.compress_ratio,
            cu_seqlens_q=req.query_start_loc,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            block_table=req.block_table,
            metadata=self._qli_metadata(req, seqused_k, cmp_residual_k,
                                        metadata.max_seq_len,
                                        num_heads_q=q_values.shape[1]),
        )

        # ⚠️ Unverified: upstream's buffer contract is "-1 where there is no
        # token", and what npu_quant_lightning_indexer_v2 writes into the
        # unfilled tail of a short row is not documented and not visible from
        # the schema. The V4.0 path never had to care -- it hands the indices
        # straight to the sparse-attention operator, which has its own
        # convention. Clearing first makes the rows this call does not cover
        # correct; it cannot fix a row the operator pads with something else.
        # First thing to compare against a golden tap.
        # 🔴 The indexer's own output and the lengths behind it. Until now
        # the whole indexer was NOT MEASURED in the dump -- and "not in the
        # table I listed" was being read as "did not change". It is the
        # sparse operator's direct input: identical q and kv with different
        # indices is enough on its own for a different attention output.
        _tap_indexer_reads(self, req, topk_idxs, seqused_k, cmp_residual_k,
                           values)

        self.topk_indices_buffer[:num_tokens] = NO_TOKEN
        flat = topk_idxs.squeeze(1) if topk_idxs.ndim == 3 else topk_idxs
        self.topk_indices_buffer[:num_tokens, :self.topk_tokens] = flat
        mark_rows_written(self.topk_indices_buffer, num_tokens)
        _note_writer("forward_oot", num_tokens)

        return self.topk_indices_buffer

    def _qli_metadata(self, req, seqused_k, cmp_residual_k, max_seq_len: int,
                      *, num_heads_q: int) -> torch.Tensor:
        """The operator's 1024-int descriptor, built here rather than read.

        The shared DSA builder computes an equivalent one and publishes it only
        for compression ratio 4; V4.1 carries 1 and 2. See indexer_select.
        compressed_lengths for why that gate was not widened and what would
        make merging the two worthwhile.

        ⚠️ Unverified: the mode this is built with must equal the mode the call
        passes, and here they are derived from two different places -- the
        cache dtypes at the call, the hardware profile here. They agree on A5;
        a platform where they disagree would build a descriptor for one kernel
        variant and invoke another.
        """
        from vllm_ascend.attention.dsa_attn_kv_plan import (
            dsv4_indexer_qk_dtype, dsv4_indexer_scale_dtype, qli_v2_quant_mode)

        ratio = max(self.compress_ratio, 1)
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            num_heads_q=num_heads_q,
            num_heads_k=1,
            head_dim=self.head_dim,
            topk=self.topk_tokens,
            quant_mode=qli_v2_quant_mode(dsv4_indexer_qk_dtype(),
                                         dsv4_indexer_scale_dtype()),
            cu_seqlens_q=req.query_start_loc,
            seqused_k=seqused_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=seqused_k.shape[0],
            max_seqlen_q=-1,
            max_seqlen_k=max_seq_len // ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=ratio,
            device=str(seqused_k.device),
        )



def _tap_indexer_reads(indexer, req, topk_idxs, seqused_k, cmp_residual_k,
                       key_values) -> None:
    """What the indexer produced, and what it read to produce it.

    Recorded at the point the values exist and before the buffer is
    overwritten, so ``indexer.topk_idxs`` is this call's selection rather
    than whatever the shared buffer holds afterwards.
    """
    from vllm_ascend.models.deepseek_v4_1 import align_taps, read_taps

    if not align_taps.alignment_active():
        return
    hf_text_config = getattr(indexer, "_tap_hf_text_config", None)
    if hf_text_config is None:
        # 🔴 这里【曾经写的是 indexer.vllm_config】—— 那个属性在这个类上
        # 从来不存在（上游只存了 parallel_config），于是 tap 一被启用就是
        # AttributeError。它躲过了 eager 那一发，因为 eager 走短上下文路径、
        # 根本没进 forward_oot：仪器没跑过 ≠ 仪器是好的。
        _TAP_CONFIG["no_config"] += 1
        return
    layer = align_taps.layer_index(indexer.k_cache.prefix, hf_text_config)
    if not read_taps.records_reads(layer):
        _TAP_CONFIG["not_recorded"] += 1
        return
    _TAP_CONFIG["tapped"] += 1

    read_taps.tap_read(layer, "indexer.topk_idxs", topk_idxs)
    read_taps.tap_read(layer, "indexer.seqused_k", seqused_k)
    read_taps.tap_read(layer, "indexer.cmp_residual_k", cmp_residual_k)
    # 🔴 With this cache's own page and this call's own length. The page is
    # not one number: the serve log's group table shows four
    # (SWA 32, two compressed groups at 128 and 256, the compressor ring at
    # 8), and every one of them is correct in its own group. Reusing another
    # cache's page here would take the wrong number of blocks -- silently,
    # since a block id is a block id.
    page = (int(key_values.shape[1])
            if isinstance(key_values, torch.Tensor) and key_values.ndim >= 2
            else None)
    read_taps.tap_cache_read(layer, "indexer.k_cache", key_values,
                             getattr(req, "block_table", None),
                             seqused_k, page)

def register(ops: dict) -> None:
    """Add the Ascend indexer to the OOT registry table.

    Keyed on the in-tree class name: CustomOp.__new__ looks up
    ``op_registry_oot[cls.__name__]`` and instantiates the replacement, which
    is what makes the V4.1 construction site build this without being told.
    """
    ops["SparseAttnIndexer"] = AscendSparseAttnIndexer


class _MarkingKernelLaunch:
    """Upstream's short-context kernel, with the row count recorded after it.

    🔴 ``topk_indices_buffer`` has two writers and only one of them was
    keeping books. ``forward_oot`` marks what it writes; the short-context
    path never reaches it -- ``DeepseekV4Indexer.forward`` fills the buffer
    with this triton kernel and returns ``index_q=None``, so
    ``if self.indexer is not None and index_q is not None`` is false,
    ``indexer_op`` is never called, and the mark is never set.

    The guard reading that mark then reported "never written" about a buffer
    that had just been written -- it was reading its own bookkeeping bit
    rather than the object it judges. Accounting has to follow *every* writer,
    not the writer someone happened to think of.

    The count comes from the launch grid, which is ``(num_tokens,)`` and is
    also exactly the range the kernel writes. Taking it from there rather than
    from a parallel argument keeps it from disagreeing with what actually ran.

    ⚠️ Rows at or past ``num_tokens`` are still unwritten on this path too, so
    the over-read check applies here unchanged -- it was simply unreachable
    while the first check fired.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getitem__(self, grid):
        launch = self._inner[grid]
        rows = grid[0] if isinstance(grid, (tuple, list)) else grid

        def _run(buffer, *args, **kwargs):
            result = launch(buffer, *args, **kwargs)
            mark_rows_written(buffer, int(rows))
            _note_writer("short_context_triton", int(rows))
            return result

        return _run

    def __getattr__(self, name):
        return getattr(self._inner, name)


_SHORT_CONTEXT_MODULE = "vllm.models.deepseek_v4_1.attention"
_SHORT_CONTEXT_NAME = "_fill_short_context_topk_indices"


def apply_short_context_accounting_patch() -> bool:
    """Make the second writer of the selection buffer keep books too."""
    import importlib

    module = importlib.import_module(_SHORT_CONTEXT_MODULE)
    inner = getattr(module, _SHORT_CONTEXT_NAME, None)
    if inner is None:
        raise AttributeError(
            f"{_SHORT_CONTEXT_MODULE} no longer defines "
            f"{_SHORT_CONTEXT_NAME!r}. Until this is repointed the "
            "short-context path writes topk_indices_buffer without recording "
            "how many rows, and the reader cannot tell a written buffer from "
            "an untouched one -- which is uninitialised memory deciding the "
            "output.")
    if isinstance(inner, _MarkingKernelLaunch):
        return False
    setattr(module, _SHORT_CONTEXT_NAME, _MarkingKernelLaunch(inner))
    return True


def short_context_accounting_patched() -> bool:
    import importlib

    module = importlib.import_module(_SHORT_CONTEXT_MODULE)
    return isinstance(getattr(module, _SHORT_CONTEXT_NAME, None),
                      _MarkingKernelLaunch)
