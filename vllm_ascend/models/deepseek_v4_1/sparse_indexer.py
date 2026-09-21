# SPDX-License-Identifier: Apache-2.0
"""``SparseAttnIndexer.forward_oot`` for DeepSeek-V4.1 on Ascend.

Upstream's base ``forward_oot`` falls through to ``forward_native``, which only
supports CUDA/ROCm/XPU/CPU. The replacement is registered through
``REGISTERED_ASCEND_OPS`` under the in-tree class name, because
``CustomOp.__new__`` substitutes on ``cls.__name__``.

V4.1 passes ``skip_k_cache_insert=True`` and ``k=None``: the layer writes the
indexer K cache itself, so this only reads the cache, scores, takes top-k and
writes the caller's buffer.
"""

from __future__ import annotations

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

from vllm_ascend.models.deepseek_v4_1.indexer_select import (
    MXFP4_SCALE_DTYPE, MXFP4_VALUE_DTYPE, assert_candidate_mask_is_vacuous,
    compressed_lengths, paired_e8m0_scale, select_topk)

#: Upstream's "no token here" sentinel in topk_indices_buffer.
NO_TOKEN = -1

#: The written-row count lives on the buffer tensor itself: the buffer is
#: shared by every layer, while ``self.indexer`` exists only on index-source
#: layers.
_ROWS_WRITTEN_ATTR = "_dsv41_rows_written"


def _current_forward() -> object | None:
    """The object identifying the forward pass now running, if there is one.

    ``set_forward_context`` builds a fresh ``ForwardContext`` per model
    forward, so its identity serves as a step token. ``None`` outside a
    forward (unit tests, off-model callers).
    """
    try:
        return get_forward_context()
    except Exception:
        return None


def mark_rows_written(buffer, num_rows: int) -> None:
    """Record how many leading rows of ``buffer`` this step actually wrote.

    The forward pass is recorded alongside the count, since a bare integer
    cannot tell rows written this step from rows left by an earlier one. The
    context object itself is kept (not its ``id()``, which is reused after
    collection).
    """
    setattr(buffer, _ROWS_WRITTEN_ATTR, (_current_forward(), int(num_rows)))


def rows_written(buffer) -> "int | None":
    """Rows written into ``buffer`` this step, or ``None`` if unrecorded.

    ``None`` means nothing has written it since the process started; upstream
    allocates it with ``torch.empty``, so that is "do not read this", which is
    different from zero rows selected.

    When the recorded forward is not the one now running, the count belongs to
    an earlier step; that is warned about once and the count is still returned.
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
        "earlier step, so every row read this step is that step's selection.",
        rows)


def unpack_indexer_cache(kv_cache):
    """``(values, scales)`` out of whatever the runner bound for this group.

    The V4.1 indexer cache is its own KV cache group with its own spec
    (``AscendDeepseekV41IndexerCache``), so it does not have V4.0's seven-tuple
    shape and the V4.0 unpackers do not apply.

    Which form arrives is decided by that spec, not here. Since batch 1 the
    spec carries ``scale_dim != 0``, and a non-zero ``scale_dim`` is what
    switches the runner from binding one packed tensor to binding the planes
    already split (``worker/model_runner_v1.py:5078-5106``) -- three of them
    on a DSV4_COMPRESSED_CACHE part, the third being the whole page again.
    ``indexer_cache_view.planes_from_bound_cache`` takes all three forms and
    is the only place that knows where the value/scale boundary is.
    """
    if isinstance(kv_cache, (tuple, list)) and len(kv_cache) == 2:
        # Kept ahead of the shared resolver so the pair is handed on
        # unexamined, as it always has been.
        return kv_cache[0], kv_cache[1]
    from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import \
        planes_from_bound_cache

    return planes_from_bound_cache(kv_cache)


class AscendSparseAttnIndexer(SparseAttnIndexer):
    """The Ascend selection path. Everything else is upstream's."""

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
            # Profile and dummy runs, as upstream. The buffer is left
            # unwritten and unmarked so forward_mqa (which does read it on
            # this path) can tell "nobody wrote this" from "zero rows".
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
        # 🔴 **Do not squeeze the k_n axis here.** Modes 3 and 5 want the scale
        # to carry *one more* axis than k, not the same number:
        #   doc quant_lightning_indexer.md:132
        #     quant_mode 为 3/5 时，layout_k 为 PA_BBND 时 shape 为
        #     (block_num, block_size, k_n, d/64, 2)          <- 5 axes
        #     其他场景分别为 (block_num, block_size, k_n)     <- 3 axes
        #
        # The squeeze that used to be here was **correct for fp8** (mode 1
        # wants exactly those 3 axes) and became wrong the moment batch 1
        # moved this call to mode 5 -- the pairing adds one axis back, but the
        # squeeze had already taken a different one away, so the rank happened
        # to look plausible at 4.
        # ⚠ Nothing downstream would notice: the bytes and their order are
        # unchanged, so the operator reads the same memory with a different
        # block-to-scale mapping.
        # ⚠ The comment below has been sitting here, correct, above code that
        # contradicted it.
        key_dequant_scale = paired_e8m0_scale(scales)
        if key_dequant_scale.ndim != values.ndim + 1:
            raise ValueError(
                f"the key dequant scale is {tuple(key_dequant_scale.shape)} "
                f"({key_dequant_scale.ndim} axes) for a key of "
                f"{tuple(values.shape)} ({values.ndim} axes). Modes 3 and 5 "
                "take exactly one axis more than the key "
                "(quant_lightning_indexer.md:132: (block_num, block_size, "
                "k_n, d/64, 2)). An axis short is not rejected -- the bytes "
                "are the same and only the block-to-scale mapping changes.")

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

        # Upstream's buffer contract is "-1 where there is no token"; what the
        # operator writes into the unfilled tail of a short row is not
        # documented, so the rows are cleared first.
        self.topk_indices_buffer[:num_tokens] = NO_TOKEN
        flat = topk_idxs.squeeze(1) if topk_idxs.ndim == 3 else topk_idxs
        self.topk_indices_buffer[:num_tokens, :self.topk_tokens] = flat
        mark_rows_written(self.topk_indices_buffer, num_tokens)

        return self.topk_indices_buffer

    def _qli_metadata(self, req, seqused_k, cmp_residual_k, max_seq_len: int,
                      *, num_heads_q: int) -> torch.Tensor:
        """The operator's 1024-int descriptor, built here rather than read.

        The shared DSA builder computes an equivalent one and publishes it only
        for compression ratio 4; V4.1 carries 1 and 2. See indexer_select.
        compressed_lengths for why that gate was not widened and what would
        make merging the two worthwhile.

        🔴 The mode here has to be the mode of the call this describes -- a
        metadata blob that disagrees with its call is not rejected; it shows
        up as an aicore exception with tilingKey 0. It used to come from
        ``dsv4_indexer_qk_dtype`` / ``dsv4_indexer_scale_dtype``, which mirror
        **V4.0's** indexer cache spec (``AscendSFAIndexerCache``, fp8 + fp32)
        and are shared with ``dsa_v1``/``dsa_cp``. Those two agreed with V4.1
        while both caches were fp8; batch 1 moved V4.1's cache to MXFP4 and
        left V4.0's alone, so they no longer describe this call at all. The
        dtypes are passed in from the same tensors the call is assembled from
        instead.
        """
        from vllm_ascend.attention.dsa_attn_kv_plan import qli_v2_quant_mode

        ratio = max(self.compress_ratio, 1)
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            num_heads_q=num_heads_q,
            num_heads_k=1,
            head_dim=self.head_dim,
            topk=self.topk_tokens,
            quant_mode=qli_v2_quant_mode(MXFP4_VALUE_DTYPE,
                                         MXFP4_SCALE_DTYPE),
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



def register(ops: dict) -> None:
    """Add the Ascend indexer to the OOT registry table.

    Keyed on the in-tree class name: CustomOp.__new__ looks up
    ``op_registry_oot[cls.__name__]`` and instantiates the replacement, which
    is what makes the V4.1 construction site build this without being told.
    """
    ops["SparseAttnIndexer"] = AscendSparseAttnIndexer


class _MarkingKernelLaunch:
    """Upstream's short-context kernel, with the row count recorded after it.

    ``topk_indices_buffer`` has two writers. ``forward_oot`` marks what it
    writes; the short-context path fills the buffer with this triton kernel
    and returns ``index_q=None``, so ``indexer_op`` never runs and nothing
    would be marked. The count comes from the launch grid, ``(num_tokens,)``,
    which is exactly the range the kernel writes.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getitem__(self, grid):
        launch = self._inner[grid]
        rows = grid[0] if isinstance(grid, (tuple, list)) else grid

        def _run(buffer, *args, **kwargs):
            result = launch(buffer, *args, **kwargs)
            mark_rows_written(buffer, int(rows))
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
            f"{_SHORT_CONTEXT_NAME!r}; without this patch the short-context "
            "path writes topk_indices_buffer without recording how many rows, "
            "and the reader cannot tell a written buffer from an untouched "
            "one.")
    if isinstance(inner, _MarkingKernelLaunch):
        return False
    setattr(module, _SHORT_CONTEXT_NAME, _MarkingKernelLaunch(inner))
    return True


def short_context_accounting_patched() -> bool:
    import importlib

    module = importlib.import_module(_SHORT_CONTEXT_MODULE)
    return isinstance(getattr(module, _SHORT_CONTEXT_NAME, None),
                      _MarkingKernelLaunch)
