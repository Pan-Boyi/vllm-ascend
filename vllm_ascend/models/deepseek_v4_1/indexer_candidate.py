# SPDX-License-Identifier: Apache-2.0
"""V4.1's two-level candidate filter on Ascend, through the ``ds41`` operators.

What the filter is: the indexer at ``candidate_source_layer_id`` (20, ratio 1)
reduces its logits to one score per block of ``candidate_block_size``
compressed columns and publishes the best ``candidate_topk_blocks`` of them;
the later ratio-1 indexers (24/28/32/36) mask their own logits to that set
before taking their row top-k. It is a semantic filter, not a speed knob: a
masked-out token cannot be selected, which is why skipping it is only exact
while every block fits in the budget -- ``2048 * 8 * 1 = 16384`` tokens, the
``max_model_len`` ceiling this module exists to lift.

``npu_quant_lightning_indexer_v2`` (the operator ``indexer_select`` calls, and
the only one there was until the ds41 wheels arrived) fuses the logit matrix
into the top-k and returns only indices, so there is no point at which a mask
could be applied. The ``ds41`` pair expresses both halves instead:

* ``ds41.quant_lightning_indexer`` -- the same fused selection, plus two
  **extra outputs** carrying the published blocks;
* ``ds41.quant_sparse_lightning_indexer`` -- the same fused selection, with
  those two as **extra inputs**, scoring only inside them.

🔴 The two are near-mirror images and their contracts differ in ways that do
not fail loudly. Read ``_qli_kwargs`` before changing either call:

* the dequant scales are ``q_descale``/``k_descale`` on the publisher and
  ``descale_q``/``descale_k`` on the consumer. Copying one call over the other
  raises TypeError; passing positionally would silently transpose them.
* ``candidate_topk_blocks`` is a keyword-only int **defaulting to -1**, and at
  -1 the publisher returns empty candidate tensors, does not raise, and its
  ``sparse_indices`` are correct and correctly shaped. Forgetting it degrades
  the filter to "no filter" with every visible signal intact -- which is the
  failure this whole module is meant to prevent. ``_assert_published`` is the
  check that a run actually took the two-level path.
"""

from __future__ import annotations

import torch

#: Both ds41 kernels hard-code the block size (``CANDIDATE_BLOCK_SIZE = 8`` in
#: their DSL source, and they refuse anything else at the host boundary), and
#: the consumer additionally hard-codes a 2048-block capacity. The checkpoint
#: agrees on both; ``assert_candidate_geometry`` is what says so out loud
#: rather than letting a future checkpoint meet a kernel that cannot serve it.
KERNEL_CANDIDATE_BLOCK_SIZE = 8
KERNEL_CANDIDATE_CAPACITY = 2048

#: ``ds41.*`` is not a namespace: the eight operators are eight **overloads**
#: of one ``cann_ops_transformer::ds41`` schema. ``torch.ops`` spells that the
#: same way the schema does, but a registry check must ask for the base name
#: (asking for ``cann_ops_transformer::ds41.quant_lightning_indexer`` returns
#: nothing and looks exactly like "the wheel is not installed").
_DS41_SCHEMA = "cann_ops_transformer::ds41"


def assert_candidate_geometry(candidate_block_size: int,
                              candidate_topk_blocks: int) -> None:
    """Refuse a checkpoint the ds41 kernels cannot express.

    Both numbers are compiled into the kernels, not passed to them in any
    meaningful sense: the block size is a literal in the DSL, and the
    consumer's candidate axis is a fixed 2048. A checkpoint that disagreed
    would be rejected at the operator's host boundary with a message about
    shapes, several frames away from the config field that caused it.
    """
    if candidate_block_size != KERNEL_CANDIDATE_BLOCK_SIZE:
        raise NotImplementedError(
            f"the checkpoint's candidate_block_size is {candidate_block_size}, "
            f"and the ds41 indexer kernels hard-code "
            f"{KERNEL_CANDIDATE_BLOCK_SIZE} (it is a literal in their DSL "
            "source, not a parameter). Nothing on this path can serve that "
            "checkpoint.")
    if candidate_topk_blocks != KERNEL_CANDIDATE_CAPACITY:
        raise NotImplementedError(
            f"the checkpoint's candidate_topk_blocks is "
            f"{candidate_topk_blocks}, and ds41.quant_sparse_lightning_indexer "
            f"takes its candidate axis as exactly {KERNEL_CANDIDATE_CAPACITY} "
            "(it refuses any other width at the host boundary).")


_ds41_import_error: Exception | None = None
_ds41_imported = False


def load_ds41() -> bool:
    """Import the ds41 operators, once, and say whether they are there.

    The import is what registers the schemas; ``torch.ops`` cannot see them
    before it. It is deliberately done at **model construction** rather than
    at the first forward, for two reasons: the binding layer compiles its JIT
    extensions on import (tens of them, some sharing names with ours), and a
    graph capture must not be the thing that first triggers that.

    🔴 The import has a side effect worth stating: it rebuilds
    ``~/.cache/torch_extensions`` unless ``TORCH_EXTENSIONS_DIR`` is set, and
    production loads from that cache. That is a property of the wheel, not of
    this function -- see ``cookbook/ds41_install_runbook.md``.
    """
    global _ds41_imported, _ds41_import_error
    if _ds41_imported:
        return _ds41_import_error is None
    _ds41_imported = True
    try:
        # Registers the eight overloads. The submodule is what the shadowing
        # CANN-bundled copies of this package do not have, so importing the
        # top-level name is not evidence that the ds41 wheels are in front.
        import cann_ops_transformer.ops.ds41  # noqa: F401
    except Exception as exc:  # ImportError, and whatever the builders raise
        _ds41_import_error = exc
        return False
    return True


def ds41_import_error() -> Exception | None:
    """Why ``load_ds41`` said no, for an error message that can be acted on."""
    return _ds41_import_error


def ds41_ops() -> tuple:
    """``(publisher, consumer)``, or a refusal naming what is missing."""
    if not load_ds41():
        raise RuntimeError(
            "the V4.1 two-level candidate filter needs the ds41 indexer "
            "operators, and importing cann_ops_transformer.ops.ds41 failed "
            f"({_ds41_import_error!r}). The three wheels in custom_ops_dsv41 "
            "have to be installed to a directory that is **ahead** of the "
            "CANN-bundled cann_ops_transformer on PYTHONPATH; installing "
            "them into the interpreter is silently shadowed. See "
            "cookbook/ds41_install_runbook.md.")
    ds41 = torch.ops.cann_ops_transformer.ds41
    return ds41.quant_lightning_indexer, ds41.quant_sparse_lightning_indexer


def query_lengths(cu_seqlens_q: torch.Tensor) -> torch.Tensor:
    """``seqused_q`` -- per-request query rows, from the cumulative offsets.

    ⚠ Optional in the operator's Python signature and **not** optional in
    practice: every prebuilt kernel variant was exported with
    ``has_seqused_q=True``, and the presence of each optional tensor is part
    of the key that selects one. Passing None lands outside the prebuilt set,
    where the DSL falls back to compiling a kernel on the spot -- slow, and it
    needs a working compiler on a card that is already serving.
    """
    return (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int32)


def _qli_kwargs(*, query, q_scale, key_cache, key_dequant_scale, weights,
                index_topk, compress_ratio, cu_seqlens_q, seqused_k,
                cmp_residual_k, block_table) -> dict:
    """The arguments both ds41 calls share, under the publisher's spelling.

    🔴 ``metadata`` is **not** among them, and is not an oversight. This
    version of the wheel refuses a non-None ``metadata`` outright ("metadata
    is reserved by the current xlsx constraint and must be None"), so the
    1024-int descriptor ``sparse_indexer._qli_metadata`` builds for the v2
    operator has no receiver here: the ds41 kernels derive the same thing
    from the tensors. The descriptor is still built for the layers that stay
    on v2 (2/8/14), which is why it was not deleted.

    ⚠ The bytes are handed over as ``uint8``, not reinterpreted as
    ``float4_e2m1fn_x2``/``float8_e8m0fnu`` the way ``indexer_select`` does
    for v2. v2 derives its quantisation mode from the dtype pair; ds41 takes
    ``quant_mode=1`` as a literal and **type-checks the storage as uint8**, so
    the reinterpretation v2 requires is exactly what ds41 rejects.
    """
    return dict(
        q=query,
        k=key_cache,
        w=weights,
        q_descale=q_scale,
        k_descale=key_dequant_scale,
        topk=index_topk,
        # 1 is MXFP4 for these kernels. It is not v2's mode 5 renumbered: v2's
        # table keys on the dtype pair and this one does not have a table.
        quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=query_lengths(cu_seqlens_q),
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        metadata=None,
        max_seqlen_q=-1,
        mask_mode=3,
        cmp_ratio=max(compress_ratio, 1),
        layout_q="TND",
        layout_kv="PA_BBND",
        return_value=False,
    )


def _as_consumer_kwargs(shared: dict) -> dict:
    """The publisher's argument names, respelled for the consumer.

    🔴 Two of them differ, and only these two: ``q_descale``/``k_descale``
    become ``descale_q``/``descale_k``. They are the same tensors in the same
    roles; the operators simply disagree on the spelling. Renaming by table
    rather than by hand is what keeps a future edit from swapping them --
    both are uint8 E8M0 and neither shape rejects the other's tensor at the
    boundary, so a swap would run.
    """
    renamed = dict(shared)
    renamed["descale_q"] = renamed.pop("q_descale")
    renamed["descale_k"] = renamed.pop("k_descale")
    return renamed


_published_evidence_logged = False
#: Belongs to this path and to nothing else; ``verify`` scripts match on it.
_PUBLISH_EVIDENCE_PREFIX = "DSV41-CANDIDATE-PUBLISH"


def _assert_published(candidate_indices: torch.Tensor,
                      candidate_lengths: torch.Tensor,
                      candidate_topk_blocks: int, where: str) -> None:
    """Refuse a publisher call that came back with nothing to publish.

    🔴 This is the check for the one failure mode that leaves every other
    signal correct. ``candidate_topk_blocks`` defaults to -1; at -1 the
    operator returns ``(0,)``-shaped candidate tensors, raises nothing, and
    still returns a correct ``sparse_indices`` of the right shape. The
    consumers would then be masking to a buffer nobody filled -- and the
    buffer is ``-1``-padded, so they would quietly select nothing.
    """
    expected = (candidate_topk_blocks, )
    if tuple(candidate_indices.shape[-1:]) != expected or (
            candidate_indices.numel() == 0):
        raise RuntimeError(
            f"{where}: the candidate publisher returned candidate_indices of "
            f"{tuple(candidate_indices.shape)}, not a trailing "
            f"{candidate_topk_blocks}. An empty candidate output is what this "
            "operator returns when candidate_topk_blocks is left at its "
            "default of -1 -- the selection it returns alongside is correct "
            "and correctly shaped, so nothing else here would notice.")
    if candidate_lengths.numel() != candidate_indices.shape[0]:
        raise RuntimeError(
            f"{where}: the publisher returned {candidate_lengths.numel()} "
            f"candidate lengths for {candidate_indices.shape[0]} query rows. "
            "The consumer reads one length per row and would read past the "
            "rows this step wrote.")


def _log_publish_evidence(candidate_indices: torch.Tensor,
                          candidate_lengths: torch.Tensor,
                          query: torch.Tensor, where: str) -> None:
    """Report, once, that the two-level path actually ran and with what.

    ``vllm.logger.logger``, not ``init_logger(__name__)``: under
    ``vllm_ascend`` the latter attaches to root at WARNING and an INFO line
    disappears without a trace (same reason as ``indexer_select``).
    """
    global _published_evidence_logged
    if _published_evidence_logged:
        return
    from vllm_ascend.models.deepseek_v4_1.call_counter import (
        CAPTURING, _capture_state)

    # 🔴 Not during capture, and not "logged anyway without the read": the
    # max below is a device-to-host read, and a D2H inside a graph capture
    # does not merely cost a sync -- it bakes one step's value into the
    # replayed graph. Capture is also the wrong place for this line to be
    # spent: it is once per process, and a warmup forward runs first.
    capture_state = _capture_state()
    if capture_state == CAPTURING:
        return
    _published_evidence_logged = True
    from vllm.logger import logger

    logger.info(
        "%s where=%s capture=%s q_shape=%s cand_shape=%s cand_len_shape=%s "
        "cand_len_max=%s", _PUBLISH_EVIDENCE_PREFIX, where, capture_state,
        tuple(query.shape), tuple(candidate_indices.shape),
        tuple(candidate_lengths.shape),
        # Worth one sync exactly once: "every length is zero" is the other
        # way this path can be live and still select nothing, and no shape
        # would show it.
        int(candidate_lengths.max()) if candidate_lengths.numel() else None)


def publish_candidates(*, query, q_scale, key_cache, key_dequant_scale,
                       weights, index_topk, compress_ratio, cu_seqlens_q,
                       seqused_k, cmp_residual_k, block_table,
                       candidate_topk_blocks: int, candidate_block_size: int,
                       where: str, op=None):
    """Layer 20: select top-k **and** publish this step's candidate blocks.

    Returns ``(topk_indices, candidate_indices, candidate_lengths)``. The
    source layer's own selection is the unfiltered one -- it is what defines
    the candidate set, so there is nothing to filter it against.

    ``op`` exists so the argument assembly can be tested without a device.
    """
    if op is None:
        op, _ = ds41_ops()
    kwargs = _qli_kwargs(query=query, q_scale=q_scale, key_cache=key_cache,
                         key_dequant_scale=key_dequant_scale, weights=weights,
                         index_topk=index_topk,
                         compress_ratio=compress_ratio,
                         cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
                         cmp_residual_k=cmp_residual_k,
                         block_table=block_table)
    topk_idxs, _, candidate_indices, candidate_lengths = op(
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        **kwargs)
    _assert_published(candidate_indices, candidate_lengths,
                      candidate_topk_blocks, where)
    _log_publish_evidence(candidate_indices, candidate_lengths, query, where)
    return topk_idxs, candidate_indices, candidate_lengths


def select_within_candidates(*, query, q_scale, key_cache, key_dequant_scale,
                             weights, index_topk, compress_ratio,
                             cu_seqlens_q, seqused_k, cmp_residual_k,
                             block_table, candidate_indices,
                             candidate_lengths, candidate_block_size: int,
                             op=None):
    """Layers 24/28/32/36: select top-k **inside** the published blocks.

    ``candidate_indices`` is ``[T, 1, 2048]`` int32 and ``candidate_lengths``
    ``[T, 1]`` int32 -- which is also what the publisher returns, so the
    caller's reshape is a no-op on this wheel. It is kept because it is not
    free elsewhere: upstream's Ascend branch bridges a real rank difference
    at ``dsa_v41.py:1060``, and our own destination buffer is 2-D. The
    reshape lives in the caller because that is also where the rows are cut
    to this step's token count.
    """
    if op is None:
        _, op = ds41_ops()
    kwargs = _as_consumer_kwargs(
        _qli_kwargs(query=query, q_scale=q_scale, key_cache=key_cache,
                    key_dequant_scale=key_dequant_scale, weights=weights,
                    index_topk=index_topk, compress_ratio=compress_ratio,
                    cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
                    cmp_residual_k=cmp_residual_k, block_table=block_table))
    topk_idxs, _ = op(candidate_block_indices=candidate_indices,
                      candidate_block_length=candidate_lengths,
                      candidate_block_size=candidate_block_size,
                      output_idx_offset=None,
                      **kwargs)
    return topk_idxs
