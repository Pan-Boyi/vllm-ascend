# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch


@lru_cache
def _get_sparse_flash_mla_ops() -> tuple[Callable, Callable]:
    """Load SparseFlashMla operators without importing vllm_ascend.ops."""
    try:
        import_module("cann_ops_transformer")
        namespace = torch.ops.cann_ops_transformer
        return namespace.sparse_flash_mla, namespace.sparse_flash_mla_metadata
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "DeepSeek-V4 BF16 KV on Ascend A5 requires SparseFlashMla from a matching CANN 9.2 toolkit."
        ) from exc


def _compressed_stream_present(kwargs: dict[str, Any]) -> bool | None:
    """Whether this call carries a compressed stream, or None if unstated.

    The descriptor and the attention call use different vocabularies for the
    same fact -- ``has_cmp_kv`` on one side, the ``cmp_kv`` tensor on the
    other -- so both are read, and neither is inferred from ``cmp_ratio``.
    """
    if "has_cmp_kv" in kwargs:
        return bool(kwargs["has_cmp_kv"])
    if "cmp_kv" in kwargs:
        return kwargs["cmp_kv"] is not None
    return None


def _add_compressed_kv_lengths(kwargs: dict[str, Any]) -> None:
    """Fill in the compressed lengths the operator wants alongside cmp_kv.

    The gate used to be ``cmp_ratio <= 1``, which is V4.0's reading of the
    ratio: there, 1 means "no compressed stream". V4.1 gives layers 20..39
    ratio 1 to mean "compressed, pooling disabled" -- a real compressed
    stream -- so that gate skipped exactly the layers that needed the lengths,
    while the descriptor built from the same 1 demanded them. One value, read
    as opposite meanings on the two sides.

    So the presence of the stream is now taken from a field that states it,
    and the ratio is used only for the arithmetic. When neither field is
    present the caller is V4.0's, and its ``cmp_ratio > 1`` reading is kept
    unchanged rather than reinterpreted.
    """
    cmp_ratio = kwargs.get("cmp_ratio") or 0
    seqused_ori_kv = kwargs.get("seqused_ori_kv")
    present = _compressed_stream_present(kwargs)
    if present is None:
        present = cmp_ratio > 1
    if not present or cmp_ratio < 1 or seqused_ori_kv is None:
        return
    kwargs.setdefault("seqused_cmp_kv", seqused_ori_kv // cmp_ratio)
    kwargs.setdefault("cmp_residual_kv", seqused_ori_kv % cmp_ratio)
    if kwargs.get("max_seqlen_cmp_kv") is None and kwargs.get("max_seqlen_ori_kv") is not None:
        kwargs["max_seqlen_cmp_kv"] = kwargs["max_seqlen_ori_kv"] // cmp_ratio


def sparse_flash_mla_metadata(**kwargs):
    """Adapt existing DSA metadata kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("device", None)
    kwargs.pop("kv_quant_mode", None)
    # This adapter is only selected for the BF16 paged-KV path. SparseFlashMla
    # accepts PA_BBND for this cache; PA_ND belongs to the FP8 quantized op.
    kwargs["layout_kv"] = "PA_BBND"
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    if "max_seqlen_kv" in kwargs:
        kwargs["max_seqlen_ori_kv"] = kwargs.pop("max_seqlen_kv")
    _add_compressed_kv_lengths(kwargs)
    _, metadata_op = _get_sparse_flash_mla_ops()
    return metadata_op(**kwargs)


def sparse_flash_mla(q: torch.Tensor, **kwargs):
    """Adapt existing DSA attention kwargs to SparseFlashMla BF16 KV."""
    kwargs.pop("kv_quant_mode", None)
    kwargs.pop("tile_size", None)
    kwargs.pop("rope_head_dim", None)
    kwargs["layout_kv"] = "PA_BBND"
    if "seqused_kv" in kwargs:
        kwargs["seqused_ori_kv"] = kwargs.pop("seqused_kv")
    _add_compressed_kv_lengths(kwargs)
    attention_op, _ = _get_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)
