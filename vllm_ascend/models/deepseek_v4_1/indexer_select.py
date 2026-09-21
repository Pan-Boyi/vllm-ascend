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
* ``q_quant`` arrives already quantised, as a ``(values, scales)`` pair.
  MXFP4 scales are per 32-element block and travel with the Q values, so --
  unlike the fp8 path this replaced -- there is no per-token scalar folded
  into ``weights``, and the operator's mandatory ``query_dequant_scale`` is
  those scales rather than ones. See ``_query_dequant_scale``.
* both the cache and the Q packer hand over **byte planes** (``uint8``). The
  dtypes that say what the bytes mean are put on here, just before the call,
  because ``qli_v2_quant_mode`` keys on them: there is no ``quant_mode``
  literal on this path, and getting the dtypes right is what makes the mode 5.
* The result travels by mutation of ``topk_indices_buffer``; the return value
  is discarded by the caller.
"""

from __future__ import annotations

import torch

from vllm_ascend.attention.dsa_attn_kv_plan import qli_v2_quant_mode
from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import SCALE_PAIR

#: The dtypes that say what the MXFP4 bytes mean. The cache and the Q packer
#: both hand over ``uint8``, because a byte plane carries no interpretation,
#: and ``qli_v2_quant_mode`` keys on the dtype pair -- so the reinterpretation
#: has to happen **before** the mode is derived, not after. There is no
#: literal ``quant_mode`` anywhere on this path: getting these two right is
#: what makes the mode 5.
#:
#: 🔴 Passing them through is also the only thing that can reach acl's
#: ACL_FLOAT4_E2M1 / ACL_FLOAT8_E8M0 from here. CANN's own binding rewrites
#: the acl dtype for modes 4 and 5 in ``FixQLIV2AclDtypes()``; **our**
#: ``csrc/torch_binding.cpp`` has no equivalent, so whatever torch_npu maps
#: these dtypes to is what the kernel sees. That mapping has not been
#: exercised on a card (o_ds41_qli_vs_ours.md §3.4).
MXFP4_VALUE_DTYPE = torch.float4_e2m1fn_x2
MXFP4_SCALE_DTYPE = torch.float8_e8m0fnu

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
    """The scale the operator requires.

    The fp8 path folded the per-token q scale into ``weights`` and passed
    ``None`` here; the logit is linear in a single per-token scalar, so ones
    were exact after that fold.

    **MXFP4 has no such scalar.** Its scales are per 32-element block and
    travel with the Q values (``indexer_q.fused_indexer_q_rope_quant_ascend``
    returns them), so ones would not be "the fold already applied" -- they
    would be every block exponent thrown away, and the operator would run and
    return garbage. The ones branch is therefore refused rather than kept as a
    default that is no longer exact.
    """
    if q_scale is not None:
        return q_scale
    raise ValueError(
        "the V4.1 indexer got no query dequant scale. On the MXFP4 path the "
        "per-32-element E8M0 block scales are produced alongside the packed "
        "Q values and are not folded into weights, so there is no per-token "
        "scalar that ones could stand in for. A None here means the Q "
        "producer is still the fp8 one (it returned a bare tensor rather than "
        "a (values, scales) pair) while the key cache is MXFP4 -- and every "
        "shape downstream stays valid either way.")


def paired_e8m0_scale(scale: torch.Tensor) -> torch.Tensor:
    """A flat per-token scale row given the trailing ``(D/64, 2)`` axis.

    Modes 3 and 5 give the dequant scales one more dimension than q/k
    (``dsa_attn_kv_plan`` :47-50; the operator's README :56-59). The bytes and
    their order are unchanged -- ``(..., 4)`` becomes ``(..., 2, 2)`` -- which
    is why nothing downstream would notice the axis missing: the operator
    would read the same memory with a different block-to-scale mapping.
    """
    last = int(scale.shape[-1])
    if last % SCALE_PAIR:
        raise ValueError(
            f"the dequant scale row is {last} bytes, which is not a whole "
            f"number of {SCALE_PAIR}-scale pairs. Modes 3 and 5 read it as "
            f"(D/{32 * SCALE_PAIR}, {SCALE_PAIR}).")
    return scale.reshape(*scale.shape[:-1], last // SCALE_PAIR, SCALE_PAIR)


def _as_operator_dtype(t: torch.Tensor, dtype: torch.dtype,
                       what: str) -> torch.Tensor:
    """``t`` reinterpreted as ``dtype``. Bytes are never touched.

    Only a ``uint8`` tensor is reinterpreted, and only into the one dtype this
    path expects for it. Accepting anything else would let an fp8 cache reach
    ``qli_v2_quant_mode`` wearing MXFP4's dtype, which is the one mistake the
    mode table cannot catch -- the kernel validates the q/k dtype against the
    mode but not the scale, so a wrong mode with a plausible dtype runs.
    """
    if t.dtype is dtype:
        return t
    if t.dtype is not torch.uint8:
        raise ValueError(
            f"{what} is {t.dtype}; the MXFP4 path takes the byte plane "
            f"(uint8) and reinterprets it as {dtype} for the operator. "
            "Anything else is a different cache format reaching a mode-5 "
            "call.")
    return t.view(dtype)


#: One line, once per process, naming the mode this read actually used.
#: Grepping a serve log for "fp4" does **not** answer this question: the MoE
#: quantiser logs ``expert_dtype resolved to 'fp4'`` (quant_config.py:75) and
#: has nothing to do with the index K cache. This prefix belongs to this path
#: and to nothing else.
_READ_EVIDENCE_PREFIX = "DSV41-INDEXK-READ"
_read_evidence_logged = False


def _log_read_evidence(quant_mode: int, key_cache: torch.Tensor,
                       key_dequant_scale: torch.Tensor,
                       query: torch.Tensor,
                       q_scale: torch.Tensor) -> None:
    """Report the arguments as they are about to be passed, once.

    ``quant_mode`` is the value being passed, not a recomputation of it: the
    caller derives it once and hands both the log and the operator the same
    object. A check bit that cannot change with the thing it checks is a green
    light that means nothing.

    ``vllm.logger.logger``, not ``init_logger(__name__)``: under
    ``vllm_ascend`` the latter attaches to root at WARNING and an INFO line
    disappears without a trace.
    """
    global _read_evidence_logged
    if _read_evidence_logged:
        return
    _read_evidence_logged = True
    from vllm.logger import logger

    # 🔴 ``q_*`` fields are **appended**, never inserted before the existing
    # ones: verify_mxfp4_batch1.py matches this line by prefix, and moving a
    # field would silently stop matching while the line still looks right.
    #
    # ⚠ Why q is here at all: the 2026-09-20 mxfp4 flight failed with
    # "Parameter q of QuantLightningIndexerV2 has incorrect shape
    # [256, 32, 64] ... head dim of q only supports 128", and this line
    # reported key/scale but **not q** -- so the log could not say whether q
    # had reached the operator as float4 (torch_npu does not widen it) or as
    # uint8 (we forgot to reinterpret it). It took reading the source to tell
    # those apart, and they call for opposite fixes. A check bit that omits
    # the argument the error names is a coverage gap, not a formatting detail.
    logger.info(
        "%s quant_mode=%s key_dtype=%s scale_dtype=%s "
        "q_dtype=%s q_shape=%s q_scale_dtype=%s q_scale_shape=%s",
        _READ_EVIDENCE_PREFIX, quant_mode, key_cache.dtype,
        key_dequant_scale.dtype, query.dtype, tuple(query.shape),
        q_scale.dtype, tuple(q_scale.shape))


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
    from vllm_ascend.models.deepseek_v4_1.call_counter import (
        mark_branch, record)

    # See ``call_counter``: counted at the entry, before the parameter raises
    # below, because "did this Python execute" is the question. The read side
    # is counted **separately** from the write side on purpose -- if the two
    # magnitudes disagree, that is itself the finding: one of them is inside
    # the captured graph and the other is not.
    record("read", query.shape)
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
    q_scale = _query_dequant_scale(query, q_scale, num_tokens, num_heads,
                                   query.device, None)
    # The byte planes, reinterpreted as what they hold. Both the cache
    # (caches.py) and the Q packer (indexer_q.py) hand over uint8; this is
    # where the bytes get their meaning, and it is upstream of the mode.
    query = _as_operator_dtype(query, MXFP4_VALUE_DTYPE, "query")
    key_cache = _as_operator_dtype(key_cache, MXFP4_VALUE_DTYPE, "key cache")
    q_scale = _as_operator_dtype(q_scale, MXFP4_SCALE_DTYPE,
                                 "query dequant scale")
    key_dequant_scale = _as_operator_dtype(key_dequant_scale,
                                           MXFP4_SCALE_DTYPE,
                                           "key dequant scale")
    for name, scale in (("query", q_scale), ("key", key_dequant_scale)):
        if int(scale.shape[-1]) != SCALE_PAIR:
            raise ValueError(
                f"the {name} dequant scale is {tuple(scale.shape)}; modes 3 "
                f"and 5 read it with a trailing (D/{32 * SCALE_PAIR}, "
                f"{SCALE_PAIR}) axis (see paired_e8m0_scale). Without it the "
                "operator reads the same bytes with a different "
                "block-to-scale mapping and nothing is raised.")
    # The kernel validates only the q/k dtype of the mode, so a wrong mode
    # runs and returns garbage; derive it from the tensors in hand. Derived
    # once: the value below is the one the operator gets and the one the
    # evidence line reports.
    quant_mode = qli_v2_quant_mode(key_cache.dtype, key_dequant_scale.dtype)
    _log_read_evidence(quant_mode, key_cache, key_dequant_scale, query,
                       q_scale)
    topk_idxs, _ = op(
        query=query,
        key=key_cache,
        weights=weights,
        query_dequant_scale=q_scale,
        key_dequant_scale=key_dequant_scale,
        topk=index_topk,
        quant_mode=quant_mode,
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
    mark_branch("read", "op")
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
