# SPDX-License-Identifier: Apache-2.0
"""Top-k selection for the DeepSeek-V4.1 indexer, on Ascend.

Upstream's ``SparseAttnIndexer`` computes the MQA logit matrix, masks it, and
takes a per-row top-k in three separable CUDA steps. Ascend has no separable
version: every indexer operator in ``csrc/torch_binding.cpp`` fuses logits and
top-k into one call and returns only indices (``return_value=1`` adds the top-k
*values*, not the matrix).

What V4.1 hands us:

* ``k`` is ``None`` and ``skip_k_cache_insert`` is ``True``, because the layer
  has already written the indexer K cache itself; this reads the cache and
  selects.
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
#: Read from the config at runtime: the released checkpoint sets them.
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

    At short context it filters nothing: ``select_candidate_blocks`` takes
    ``min(topk_blocks, nblocks)``, so when a row has no more blocks than the
    budget every block is published and the mask is a no-op. The causal bound
    the same kernel applies is already applied by the Ascend operator
    (``mask_mode=3``).
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

    Running anyway would silently draw the selection from the whole context
    instead of the published blocks.
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



class CandidateGuards:
    """Every V4.1 indexer's candidate-mask guard inputs, for the runner.

    Each indexer calls ``assert_candidate_mask_is_vacuous`` on
    ``metadata.max_seq_len`` in its forward. Under graph capture that value is
    ``max_model_len`` and replay never runs the check again, so a longer
    sequence would silently skip the candidate filter. These are the same
    inputs, read from the same modules, checked by the runner at startup and
    on the host every step.
    """

    def __init__(self, entries):
        self.entries = tuple(entries)

    def check(self, max_seq_len: int, why: str) -> None:
        for where, ratio, topk_blocks, block_size in self.entries:
            assert_candidate_mask_is_vacuous(max_seq_len, ratio, topk_blocks,
                                             block_size, f"{where} [{why}]")


def collect_candidate_guards(model, hf_config) -> "CandidateGuards | None":
    """Guard inputs of every Ascend indexer in a V4.1 model; None otherwise."""
    model_type = getattr(hf_config, "model_type", None)
    if not (isinstance(model_type, str)
            and model_type.startswith("deepseek_v41")):
        return None
    from vllm_ascend.models.deepseek_v4_1.sparse_indexer import (
        AscendSparseAttnIndexer)
    # Same expressions as AscendSparseAttnIndexer.forward_oot passes.
    entries = [(m.k_cache.prefix, m.compress_ratio,
                m.candidate_blocks.shape[1]
                if m.candidate_blocks is not None else 0,
                m.candidate_block_size)
               for m in model.modules()
               if isinstance(m, AscendSparseAttnIndexer)]
    if not entries:
        raise RuntimeError(
            f"{model_type}: found no AscendSparseAttnIndexer in the built "
            "model, so the candidate-mask guard would check nothing. Either "
            "the indexer class changed or it is not the one being built.")
    from vllm.logger import logger
    logger.info(
        "candidate-mask guard: %d indexers, %d read a candidate mask",
        len(entries), sum(1 for e in entries if e[2] > 0))
    return CandidateGuards(entries)

def _query_dequant_scale(q_quant, q_scale, num_tokens: int, num_heads: int,
                         device, dtype: torch.dtype) -> torch.Tensor:
    """The scale the operator requires, including when nothing scaled it.

    ``query_dequant_scale`` is mandatory in the operator schema, but V4.1's
    FP8 path folds the per-token q scale into ``weights`` (upstream and
    ``indexer_q.py`` alike) and passes ``None``. The logit is linear in that
    scale, so passing ones after the fold is exact.
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

    ``cmp_ratio`` comes from the layer: V4.1 carries 0, 1 and 2, and 0 ("no
    compression") is passed to the operator as 1.
    """
    if op is None:
        op = torch.ops._C_ascend.npu_quant_lightning_indexer_v2
    if max(compress_ratio, 1) == 1 and cmp_residual_k is not None:
        # The operator's own cross-parameter rule, checked here to give a
        # clearer error than the operator's tiling refusal.
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
        # The kernel validates only the q/k dtype of the mode, so a wrong
        # mode runs and returns garbage; derive it from the tensors in hand.
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

    ``AscendDSAMetadataBuilder`` computes the same two tensors but publishes
    them only for compression ratio 4 (V4.0); V4.1 carries 1 and 2, so they
    are computed here. The ``sparse_flash_mla`` call reuses this function.
    """
    ratio = max(compress_ratio, 1)
    lens = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
    seqused_k = torch.div(lens, ratio, rounding_mode="floor").to(torch.int32)
    if ratio == 1:
        # The operator requires cmp_residual_k to be null (not zeros) at
        # ratio 1 (EZ0037). Ratio 0 reaches here too, as max(ratio, 1).
        return seqused_k, None
    return seqused_k, torch.remainder(lens, ratio).to(torch.int32)
