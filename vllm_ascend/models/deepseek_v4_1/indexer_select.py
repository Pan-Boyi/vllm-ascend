# SPDX-License-Identifier: Apache-2.0
"""Top-k selection for the DeepSeek-V4.1 indexer, on Ascend.

Upstream's ``SparseAttnIndexer`` computes the MQA logit matrix, masks it, and
takes a per-row top-k in three separable CUDA steps. Ascend has no separable
version: every indexer operator in ``csrc/torch_binding.cpp`` fuses logits and
top-k into one call and returns only indices (``return_value=1`` adds the top-k
*values*, not the matrix). That single difference decides most of what follows.

What V4.1 hands us is narrow, and narrower than V4.0's:

* ``k`` is the literal ``None`` and ``skip_k_cache_insert`` the literal ``True``
  (``vllm/models/deepseek_v4_1/attention.py:820-825`` and ``:1116``), because
  the layer has already written the indexer K cache itself. So there is no
  cache-insert half to implement here; this reads the cache and selects.
* ``q_quant`` arrives already quantised, with the per-token q scale **folded
  into weights** on the FP8 path -- so the operator's mandatory
  ``query_dequant_scale`` is ones, not a dropped argument. See
  ``_query_dequant_scale``.
* The result travels by mutation of ``topk_indices_buffer``; the return value
  is discarded by the caller.
"""

from __future__ import annotations

import torch

from vllm_ascend.attention.dsa_attn_kv_plan import qli_v2_quant_mode

#: What the V4.1 checkpoint calls the two-level candidate filter's knobs.
#: Read from the config at runtime rather than assumed: in the checkpoint we
#: serve they are set (source layer 20, 2048 blocks of 8), not absent.
CANDIDATE_FIELDS = ("candidate_source_layer_id", "candidate_topk_blocks",
                    "candidate_block_size")


def candidate_mask_is_vacuous(max_seq_len: int, compress_ratio: int,
                              candidate_topk_blocks: int,
                              candidate_block_size: int) -> bool:
    """Whether V4.1's two-level candidate filter can change any top-k here.

    The candidate source reduces its logits to one score per block of
    ``candidate_block_size`` compressed columns and publishes the best
    ``candidate_topk_blocks`` of them; later indexers mask their own logits to
    that set before taking their row top-k. It is a semantic filter, not a
    speed knob -- a masked-out token cannot be selected.

    It is also, at short context, a filter that filters nothing.
    ``select_candidate_blocks`` takes ``min(topk_blocks, nblocks)``
    (``vllm/model_executor/kernels/attention/dsa/candidate_blocks.py:179``), so
    when the row has no more blocks than the budget, every block is published
    and the mask leaves the row alone. The causal bound the same kernel applies
    is already applied by the Ascend operator (``mask_mode=3``).

    Which is why this returns a criterion rather than a policy: with
    ``max_model_len`` 8192 and ratio 1 a row is 1024 blocks against a budget of
    2048, and skipping the filter is exact. At 16385 it stops being exact, and
    nothing about the run would look different.
    """
    if candidate_topk_blocks <= 0 or candidate_block_size <= 0:
        return True
    compressed_len = max_seq_len // max(compress_ratio, 1)
    n_blocks = -(-compressed_len // candidate_block_size)  # ceil
    return n_blocks <= candidate_topk_blocks


def assert_candidate_mask_is_vacuous(max_seq_len: int, compress_ratio: int,
                                     candidate_topk_blocks: int,
                                     candidate_block_size: int,
                                     where: str) -> None:
    """Refuse to select top-k while a candidate mask would have changed it.

    Running anyway is the failure this exists to prevent: no operator errors,
    the shapes are right, the dtype is right, and the selected tokens are drawn
    from the whole context instead of the published blocks. The output of a
    top-k is a set, and a set that is wrong by a few members reads as a small
    numeric difference downstream. Refusing is the only signal available.
    """
    if candidate_mask_is_vacuous(max_seq_len, compress_ratio,
                                 candidate_topk_blocks, candidate_block_size):
        return
    compressed_len = max_seq_len // max(compress_ratio, 1)
    n_blocks = -(-compressed_len // candidate_block_size)
    raise NotImplementedError(
        f"{where}: V4.1's two-level candidate filtering would change this "
        f"selection and Ascend cannot express it. A row here is {n_blocks} "
        f"blocks of {candidate_block_size} (sequence {max_seq_len}, "
        f"compression {compress_ratio}) against a budget of "
        f"{candidate_topk_blocks}, so the source layer publishes a strict "
        "subset and the later layers must mask to it. Every Ascend indexer "
        "operator fuses the logit matrix into the top-k and returns only "
        "indices, so there is no point at which the mask can be applied. "
        f"Below {candidate_topk_blocks * candidate_block_size * max(compress_ratio, 1)} "
        "tokens of context the filter selects every block and skipping it is "
        "exact; this run is past that. Serving needs either a smaller "
        "max_model_len or an operator that exposes the logits.")


def _query_dequant_scale(q_quant, q_scale, num_tokens: int, num_heads: int,
                         device, dtype: torch.dtype) -> torch.Tensor:
    """The scale the operator requires, including when nothing scaled it.

    ``query_dequant_scale`` is a mandatory positional in the operator schema
    (``csrc/torch_binding.cpp:3439-3455``), but V4.1's FP8 path does not have
    one to give: ``fused_indexer_q_rope_quant`` folds the per-token q scale
    into ``weights`` and returns ``index_q_scale = None``, and the Ascend
    replacement in ``indexer_q.py`` folds it the same way. Since the logit is
    linear in that scale -- ``sum_h w_h * q_s * k_s * (q_int . k_int)`` --
    folding into ``w_h`` and passing ones here is exact, not an approximation.

    Passing the operator a scale of ones when the fold did *not* happen would
    be silently wrong, which is why this is a function with a reason rather
    than a ``torch.ones`` at the call site.
    """
    if q_scale is not None:
        return q_scale
    return torch.ones((num_tokens, num_heads), dtype=dtype, device=device)


def select_topk(
    *,
    query: torch.Tensor,
    q_scale: torch.Tensor | None,
    key_cache: torch.Tensor,
    key_dequant_scale: torch.Tensor,
    weights: torch.Tensor,
    index_topk: int,
    compress_ratio: int,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    cmp_residual_k: torch.Tensor | None,
    block_table: torch.Tensor,
    metadata: torch.Tensor,
    op=None,
) -> torch.Tensor:
    """One fused logits+top-k call. Returns int32 indices ``[T, 1, topk]``.

    ``op`` exists so the argument assembly can be tested without a device; it
    defaults to the real operator.

    ``cmp_ratio`` is passed explicitly from the layer. V4.0 compresses by 4
    throughout and the surrounding code has 4 baked in in several places; V4.1
    carries 0, 1 and 2, and 0 means "no compression", for which the operator's
    identity is 1 rather than 0.
    """
    if op is None:
        op = torch.ops._C_ascend.npu_quant_lightning_indexer_v2
    if max(compress_ratio, 1) == 1 and cmp_residual_k is not None:
        # The operator's own cross-parameter rule, checked here rather than
        # trusted to compressed_lengths: this is the last point before the
        # call, and a caller that built the residual itself would otherwise
        # reach the operator's refusal with no idea which of its arguments
        # the ratio made illegal.
        raise ValueError(
            f"cmp_ratio is 1 (layer ratio {compress_ratio}) and "
            f"cmp_residual_k was supplied as {tuple(cmp_residual_k.shape)}. "
            "The operator requires it to be null at this ratio -- "
            "\"Cmp_ratio is 1 or sparse_mode is 0, cmp_residual_k must be "
            "null\" -- and it is refused at tiling, before the kernel runs.")
    num_tokens, num_heads = query.shape[0], query.shape[1]
    topk_idxs, _ = op(
        query=query,
        key=key_cache,
        weights=weights,
        query_dequant_scale=_query_dequant_scale(
            query, q_scale, num_tokens, num_heads, query.device,
            key_dequant_scale.dtype),
        key_dequant_scale=key_dequant_scale,
        topk=index_topk,
        # The mode fixes both the q/k dtype and the scale dtype, and the kernel
        # validates only the former -- a plausible-looking wrong mode runs and
        # returns garbage. Derived from the tensors in hand, never defaulted.
        quant_mode=qli_v2_quant_mode(key_cache.dtype, key_dequant_scale.dtype),
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        metadata=metadata,
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=max(compress_ratio, 1),
        return_value=0,
    )
    return topk_idxs


def compressed_lengths(seq_lens: torch.Tensor,
                       compress_ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(seqused_k, cmp_residual_k)`` for the PA_BBND layout.

    The operator reads the compressed K length and, separately, the residual of
    the original length against ``cmp_ratio``. Both have to be computed from
    the same ratio the call passes, and that ratio has to be the layer's.

    This is a second implementation of something that already exists, and that
    is a debt, not a design. ``AscendDSAMetadataBuilder`` computes exactly these
    two tensors -- correctly, for any ratio -- and then declines to publish them
    behind ``if self.compressor_ratio == 4:``
    (``vllm_ascend/attention/dsa_v1.py``). V4.1 carries 1 and 2 and never 4, so
    nothing reaches us. Widening that gate is the smaller change and the better
    one; it was not taken because there is no cheap way to test V4.0 for the
    regression it could cause, not because two implementations is preferable.

    TRIGGER to merge these into one: a usable V4.0 regression test, or a third
    consumer of the same computation. Whichever comes first.

    That ``== 4`` is itself the shape this codebase keeps producing -- a literal
    standing in for a value that is actually a function of the layer. It reads
    as a fact about the operator and it is a fact about DeepSeek-V4.
    """
    ratio = max(compress_ratio, 1)
    lens = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    seqused_k = torch.div(lens, ratio, rounding_mode="floor").to(torch.int32)
    if ratio == 1:
        # 🔴 The operator refuses a residual at this ratio, in its own words:
        #
        #     Invalid_Argument(EZ0037): Parameter cmp_residual_k of
        #     QuantLightningIndexerV2 is invalid. Reason: Cmp_ratio is 1 or
        #     sparse_mode is 0, cmp_residual_k must be null.
        #
        # and the arithmetic agrees: lens % 1 is zero for every request, so
        # the tensor carries no information. But zero is not null -- an
        # all-zeros int32 tensor is still a tensor, and the check is on
        # presence, not on content. This is where a "harmless" value is worse
        # than no value: it looks like data and is refused as an argument.
        #
        # Reaches every ratio-0 layer too, because the call passes
        # max(ratio, 1) and 0 becomes 1 here as well.
        return seqused_k, None
    return seqused_k, torch.remainder(lens, ratio).to(torch.int32)
