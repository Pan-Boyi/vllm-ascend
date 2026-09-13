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
            # here too; the caller reads the buffer, not the return value, and
            # during profiling nothing reads it at all.
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
        self.topk_indices_buffer[:num_tokens] = NO_TOKEN
        flat = topk_idxs.squeeze(1) if topk_idxs.ndim == 3 else topk_idxs
        self.topk_indices_buffer[:num_tokens, :self.topk_tokens] = flat

        # 🔴 The per-row count of real entries, published beside the buffer.
        #
        # Without it the attention call supplies cmp_sparse_indices and no
        # cmp_topk_length, and the operator then treats the whole row as
        # valid -- padding included -- and gathers at the -1 slots. That is
        # the same mechanism kernel proved single-variable on the *ori* side:
        # the identical table crashes when the declared length is the full
        # width and passes at its true length.
        #
        # Counted here rather than at the consumer because this is where the
        # rows are written and where -1 means what it means. A count taken
        # downstream would be a second source, free to disagree with the table
        # it describes.
        self.topk_lengths_buffer = (flat >= 0).sum(
            dim=-1, dtype=torch.int32) if flat.numel() else torch.zeros(
                num_tokens, dtype=torch.int32, device=flat.device)
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


def register(ops: dict) -> None:
    """Add the Ascend indexer to the OOT registry table.

    Keyed on the in-tree class name: CustomOp.__new__ looks up
    ``op_registry_oot[cls.__name__]`` and instantiates the replacement, which
    is what makes the V4.1 construction site build this without being told.
    """
    ops["SparseAttnIndexer"] = AscendSparseAttnIndexer
