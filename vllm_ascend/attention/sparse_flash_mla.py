# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from typing import Any

import torch
from vllm.logger import logger


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


# --- 取证 #48：调用前核对 block table 与 KV cache 的容量关系 -------------------
# 算子【不检查】这两条。descriptor 编的是"哪个核处理哪一段 query"，每个核按记录
# 各取一段；如果 block id 指到 cache 之外，表现就是多核【同一处】越界 + mte 地址类
# 错误 + tilingKey 0 —— 和现场完全一致。
#
# ⚠️ 判据是三态：成立 / 不成立 / 【算不出来】。算不出来【不算通过】——
# 一个构造不出判据却放行的守卫，和没有守卫在结果上无法区分。
_SFM_CHECK = os.environ.get("DSV41_SFM_CHECK", "") == "1"
_SFM_N = 0


def _check_block_table_fits(kwargs: dict[str, Any]) -> None:
    global _SFM_N
    if not _SFM_CHECK:
        return
    _SFM_N += 1
    n = _SFM_N
    if n > 4:          # 前几次调用足够定性，不拖慢整发
        return
    ori_kv = kwargs.get("ori_kv")
    bt = kwargs.get("ori_block_table")
    used = kwargs.get("seqused_ori_kv")
    unknown = []
    if not isinstance(ori_kv, torch.Tensor):
        unknown.append("ori_kv 不是张量")
    if not isinstance(bt, torch.Tensor):
        unknown.append("ori_block_table 不是张量")
    if unknown:
        logger.error("【SFM-CHECK #%d】🔴 判据【算不出来】(不算通过): %s | 实参键=%s",
                     n, "; ".join(unknown), sorted(kwargs.keys()))
        raise RuntimeError(
            f"【SFM-CHECK】判据构造不出来，按【不算通过】处理: {'; '.join(unknown)}; "
            f"实参键={sorted(kwargs.keys())}"
        )
    num_blocks = int(ori_kv.shape[0])
    block_size = int(ori_kv.shape[1]) if ori_kv.dim() >= 2 else None
    bt_max = int(bt.max().item()) if bt.numel() else -1
    used_max = int(used.max().item()) if isinstance(used, torch.Tensor) and used.numel() else None
    cap = num_blocks * block_size if block_size else None
    c1 = "✅" if bt_max < num_blocks else "🔴不成立"
    if used_max is None or cap is None:
        c2 = "🔴算不出来"
    else:
        c2 = "✅" if used_max <= cap else "🔴不成立"
    logger.error(
        "【SFM-CHECK #%d】ori_kv.shape=%s num_blocks=%d block_size=%s | "
        "ori_block_table.shape=%s .max()=%d | seqused_ori_kv.max()=%s 容量=%s | "
        "判据① max<num_blocks: %s | 判据② used_max<=容量: %s",
        n, tuple(ori_kv.shape), num_blocks, block_size,
        tuple(bt.shape), bt_max, used_max, cap, c1, c2,
    )
    # descriptor 与调用的三件并排：descriptor 承诺的 vs 调用实际供的。
    # 算子不校验这一对；不一致不报错，表现为 aicore 异常 + tilingKey 0。
    # ① 键名无条件打一次：「某参数在不在」不该需要再等一轮起飞。
    #    上一轮 cmp_kv 就是因为没打而只能报「没测到」。
    # ② cmp_kv / cmp_block_table 的在与形状 —— 它是判「走的哪条 impl」的一项。
    def _t(x):
        return f"有{tuple(x.shape)}" if isinstance(x, torch.Tensor) else (
            "无" if x is None else f"非张量({type(x).__name__})")
    logger.error(
        "【SFM-KEYS #%d】键=%s | cmp_kv=%s cmp_block_table=%s "
        "ori_kv=%s ori_block_table=%s",
        n, sorted(kwargs.keys()), _t(kwargs.get("cmp_kv")),
        _t(kwargs.get("cmp_block_table")), _t(kwargs.get("ori_kv")),
        _t(kwargs.get("ori_block_table")))

    md = kwargs.get("metadata")
    if isinstance(md, torch.Tensor):
        h = md.detach().to("cpu", torch.int32).contiguous()
        nz = int((h != 0).sum().item())
        logger.error(
            "【SFM-META #%d】metadata: numel=%d 非零=%d data_ptr=0x%x dtype=%s | "
            "调用侧 has_cmp_kv=%s cmp_topk=%s cmp_ratio=%s ori_mask_mode=%s "
            "cmp_mask_mode=%s ori_win=(%s,%s) ori_sparse_indices=%s "
            "cmp_sparse_indices=%s seqused_cmp_kv=%s",
            n, md.numel(), nz, md.data_ptr(), md.dtype,
            kwargs.get("has_cmp_kv"), kwargs.get("cmp_topk"),
            kwargs.get("cmp_ratio"), kwargs.get("ori_mask_mode"),
            kwargs.get("cmp_mask_mode"),
            kwargs.get("ori_win_left"), kwargs.get("ori_win_right"),
            # mask mode 0 要的是 ori 这张表，不是 cmp 那张。两张表不同名不同物，
            # 混了就会得出一个"独立成立"的假违规。
            "有" if kwargs.get("ori_sparse_indices") is not None else "无",
            "有" if kwargs.get("cmp_sparse_indices") is not None else "无",
            "有" if kwargs.get("seqused_cmp_kv") is not None else "无",
        )
        d = os.environ.get("DSV41_SAS_DUMP_DIR", "")
        if d:
            try:
                import numpy as np
                os.makedirs(d, exist_ok=True)
                np.save(os.path.join(d, f"callsite_metadata_{n}.npy"),
                        h.numpy())
            except Exception as e:
                logger.warning("【SFM-META #%d】落盘失败(不致命): %r", n, e)
        if nz == 0:
            # 全零 descriptor 属于"算不出来"，不是"通过"。
            raise RuntimeError(
                f"【SFM-META】metadata 全零（numel={md.numel()}）—— 设备侧任务没有为"
                f" 这个 group 写过它，停在 host 侧")
    else:
        logger.error("【SFM-META #%d】🔴 kwargs 里没有 metadata 张量；键=%s",
                     n, sorted(kwargs.keys()))

    if c1 != "✅" or c2 != "✅":
        raise RuntimeError(
            f"【SFM-CHECK】调用前判据不成立，主动停在 host 侧而不是让它变成 aicore 异常: "
            f"判据①({c1}) ori_block_table.max()={bt_max} vs num_blocks={num_blocks}; "
            f"判据②({c2}) seqused_ori_kv.max()={used_max} vs 容量={cap}"
        )


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
    _check_block_table_fits(kwargs)
    attention_op, _ = _get_sparse_flash_mla_ops()
    return attention_op(q, **kwargs)
