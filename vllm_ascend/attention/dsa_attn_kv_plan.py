# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centralized DeepSeek-V4 attention-KV execution choices."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch_npu

from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla, sparse_flash_mla_metadata
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

_BF16_KV_CACHE_DTYPES = frozenset({"bfloat16", "bf16"})


def _supports_dsv4_compressed_cache() -> bool:
    return get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)


# QuantLightningIndexerV2's ``quant_mode`` selects which quantized kernel
# variant runs, and each variant fixes BOTH the q/k dtype and the dequant
# scale dtype. The kernel rejects a mismatch with
#
#     Invalid_Argument_Tensor_Dtype(EZ0021): Parameters q and k of
#     QuantLightningIndexerV2 have incorrect dtypes ... Reason: The dtype of q
#     and k must match quant_mode.
#
# From the operator's own constraints (csrc/attention/quant_lightning_indexer_v2
# /README.md, "Ascend 950PR/Ascend 950DT"), confirmed by sweeping quant_mode
# against each dtype pair on A5:
#
#     mode  q / k             dequant scale     weights
#     1     float8_e4m3fn     float32           float32
#     2     int8              float16           float16
#     3     float8_e4m3fn     float8_e8m0       float32
#     4     hifloat8          float32           float32
#     5     float4_e2m1       float8_e8m0       float32
#
# Modes 1 and 3 take the SAME q/k dtype and are told apart only by the scale
# dtype, so the q/k dtype alone does not determine the mode. Keying on it
# alone happens to return 1 for today's A5 path (fp8 q/k, fp32 scales), but it
# would keep returning 1 for an e8m0-scaled caller, handing the kernel mode
# 1's contract with mode 3's tensors -- wrong values, no error.
#
# Modes 3 and 5 also give the scales one more dimension than q/k, trailing
# (D/64, 2), with D a multiple of 64. Selecting one of those modes is
# therefore not just a mode change; the caller has to produce scales of that
# shape as well.
#
# A2/A3 support quant_mode 2 only (int8 q/k, float16 scales), which is the
# (int8, float16) entry below.
_QLI_V2_QUANT_MODE: dict[tuple[torch.dtype, torch.dtype], int] = {
    (torch.float8_e4m3fn, torch.float32): 1,
    (torch.int8, torch.float16): 2,
    (torch.float8_e4m3fn, torch.float8_e8m0fnu): 3,
    # Mode 4 is hifloat8, which torch has no dtype for, so it cannot be keyed
    # here. Do NOT add a stand-in dtype for it: any pair listed becomes a
    # silent answer for callers that legitimately pass that dtype.
    (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu): 5,
}


def qli_v2_quant_mode(qk_dtype: torch.dtype, scale_dtype: torch.dtype) -> int:
    """Return the QLI v2 ``quant_mode`` for this (q/k, dequant scale) pair.

    Raises on any pair the operator has no variant for. Falling back to a
    default here would be a silent wrong-value bug: the kernel validates the
    q/k dtype against the mode but not the scale dtype, so a wrong mode with a
    plausible q/k dtype runs and returns garbage.
    """
    try:
        return _QLI_V2_QUANT_MODE[(qk_dtype, scale_dtype)]
    except KeyError:
        supported = ", ".join(f"({q}, {s}) -> {m}" for (q, s), m in sorted(_QLI_V2_QUANT_MODE.items(), key=lambda kv: kv[1]))
        raise ValueError(
            f"QuantLightningIndexerV2 has no quant_mode for q/k dtype {qk_dtype} "
            f"with dequant scale dtype {scale_dtype}. Supported: {supported}. "
            "Modes 3 and 5 additionally require scales shaped (..., D/64, 2)."
        ) from None


def dsv4_indexer_qk_dtype() -> torch.dtype:
    """dtype of the DeepSeek-V4 indexer q/k tensors on this hardware.

    Mirrors ``AscendSFAIndexerCache.get_kv_cache_spec``, which pins the indexer
    cache to fp8_e4m3 exactly when the platform has DSV4_COMPRESSED_CACHE (A5).
    """
    return torch.float8_e4m3fn if _supports_dsv4_compressed_cache() else torch.int8


def dsv4_indexer_scale_dtype() -> torch.dtype:
    """dtype of the DeepSeek-V4 indexer dequant scales on this hardware.

    Mirrors the ``scale_dtype`` that ``AscendSFAIndexerCache.get_kv_cache_spec``
    puts on the cache spec. Used together with ``dsv4_indexer_qk_dtype`` where
    the quant_mode is needed but no tensor is in hand -- the metadata operator
    has to be built with the same mode as the call it describes.
    """
    return torch.float32 if _supports_dsv4_compressed_cache() else torch.float16


def resolve_dsv4_cache_dtype(cache_dtype, model_dtype: str) -> str:
    """Return the KV cache dtype the platform should pin for DeepSeek-V4.

    On A5 the launch request has to stay readable afterwards, because it is the
    only thing that separates an explicit bfloat16 KV request from ``auto``.
    ``auto`` and the model dtype resolve identically everywhere downstream, so
    collapsing every non-bfloat16 request to ``auto`` preserves the upstream
    values while keeping the mode recoverable.
    """
    if not _supports_dsv4_compressed_cache():
        return model_dtype
    return "bfloat16" if str(cache_dtype).lower() in _BF16_KV_CACHE_DTYPES else "auto"


def is_a5_bf16_kv_enabled(vllm_config) -> bool:
    """Return whether BF16 SparseFlashMla KV is enabled on A5.

    Callers must pass the engine ``vllm_config``. Do not look it up from the
    process-global current config: that context is only set during
    ``load_model()`` and a missing lookup would silently pick the FP8 plan.
    """
    if not _supports_dsv4_compressed_cache():
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is None:
        return False
    return str(cache_config.cache_dtype).lower() in _BF16_KV_CACHE_DTYPES


def get_dsv4_attn_kv_dtype(vllm_config) -> torch.dtype:
    """Return the attention KV dtype while preserving non-A5 behavior."""
    return (
        torch.bfloat16
        if not _supports_dsv4_compressed_cache() or is_a5_bf16_kv_enabled(vllm_config)
        else torch.float8_e4m3fn
    )


DSA_COMPRESSOR_SLOT_MAPPING_FLAT = 1
DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET = 2


@dataclass(frozen=True)
class DsaAttnKvPlan:
    """The attention-KV plan only; indexer KV remains independently FP8."""

    uses_sparse_flash_mla: bool
    uses_kv_compress_epilog: bool
    layout_kv: str
    compressor_slot_mapping_format: int
    requires_block_offset_slots: bool
    sparse_attn_op: Callable[..., Any]
    sparse_attn_metadata_op: Callable[..., Any]
    sparse_attn_base_kwargs: dict[str, Any]
    sparse_attn_metadata_kwargs: dict[str, Any]
    include_metadata_device: bool
    applies_sparse_attn_runtime_kwargs: bool

    def get_dsa_sparse_attn_metadata_op(self):
        return self.sparse_attn_metadata_op

    def get_dsa_sparse_attn_metadata_kwargs(self, device) -> dict[str, Any]:
        kwargs = dict(self.sparse_attn_metadata_kwargs)
        if self.include_metadata_device:
            kwargs["device"] = str(device)
        return kwargs

    def get_dsa_sparse_attn_op(self):
        return self.sparse_attn_op

    def get_dsa_sparse_attn_base_kwargs(self) -> dict[str, Any]:
        return dict(self.sparse_attn_base_kwargs)

    def add_dsa_sparse_attn_extra_kwargs(self, extra_kwargs: dict[str, Any], **kwargs_to_add) -> None:
        if self.applies_sparse_attn_runtime_kwargs:
            extra_kwargs.update(kwargs_to_add)

    def get_dsa_compressor_slot_mapping_format(self) -> int:
        return self.compressor_slot_mapping_format

    def format_dsa_slot_mapping(self, slot_mapping: torch.Tensor, block_size: int) -> torch.Tensor:
        if not self.requires_block_offset_slots:
            return slot_mapping
        valid = slot_mapping >= 0
        invalid = torch.full_like(slot_mapping, -1)
        block_idx = torch.where(valid, torch.div(slot_mapping, block_size, rounding_mode="floor"), invalid)
        offset = torch.where(valid, slot_mapping % block_size, invalid)
        return torch.stack([block_idx, offset], dim=-1).to(torch.int32)

    def dsa_kv_compress_scatter(self, cache: torch.Tensor, x: torch.Tensor | None, slot_mapping: torch.Tensor) -> None:
        if x is None:
            return
        if self.uses_sparse_flash_mla:
            if slot_mapping.ndim != 2 or slot_mapping.shape[-1] != 2:
                raise ValueError(f"BF16 DSA slot_mapping must be [num_tokens, 2], got {tuple(slot_mapping.shape)}.")
            # Keep fixed [T, 2] shape under ACLGraph. SparseFlashMla's
            # scatter path receives padded [-1, -1] rows directly; do not
            # introduce a data-dependent Nonzero/gather operation here.
            indices = slot_mapping.to(torch.int64).contiguous()
            updates = x.reshape((slot_mapping.shape[0],) + tuple(cache.shape[2:])).contiguous()
            torch_npu.npu_scatter_nd_update_(cache, indices, updates)
            return
        if not self.uses_kv_compress_epilog:
            torch.ops._C_ascend.npu_scatter_nd_update_sk(cache, slot_mapping, x)
            return
        torch.ops._C_ascend.kv_compress_epilog(
            kv_compress_cache=cache.view(-1, 1, cache.shape[-1]),
            x=x.view(-1, x.shape[-1]),
            slot_mapping=slot_mapping,
            quant_group_size=64,
            quant_mode=2,
            round_scale_flag=True,
            layout=1,
        )


def get_dsa_attn_kv_plan(vllm_config) -> DsaAttnKvPlan:
    """Return the explicit A5 BF16 or upstream-compatible FP8 DSA plan."""
    if not _supports_dsv4_compressed_cache():
        return DsaAttnKvPlan(
            uses_sparse_flash_mla=False,
            uses_kv_compress_epilog=False,
            layout_kv="PA_ND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
            requires_block_offset_slots=True,
            sparse_attn_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv,
            sparse_attn_metadata_op=torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata,
            sparse_attn_base_kwargs={},
            sparse_attn_metadata_kwargs={},
            include_metadata_device=True,
            applies_sparse_attn_runtime_kwargs=True,
        )

    use_bf16 = is_a5_bf16_kv_enabled(vllm_config)
    if use_bf16:
        return DsaAttnKvPlan(
            uses_sparse_flash_mla=True,
            uses_kv_compress_epilog=False,
            layout_kv="PA_BBND",
            compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
            requires_block_offset_slots=True,
            sparse_attn_op=sparse_flash_mla,
            sparse_attn_metadata_op=sparse_flash_mla_metadata,
            sparse_attn_base_kwargs={},
            sparse_attn_metadata_kwargs={},
            include_metadata_device=True,
            applies_sparse_attn_runtime_kwargs=True,
        )
    return DsaAttnKvPlan(
        uses_sparse_flash_mla=False,
        uses_kv_compress_epilog=True,
        layout_kv="PA_ND",
        compressor_slot_mapping_format=DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
        requires_block_offset_slots=False,
        sparse_attn_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv,
        sparse_attn_metadata_op=torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata,
        sparse_attn_base_kwargs={"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": 64},
        sparse_attn_metadata_kwargs={"kv_quant_mode": 1},
        include_metadata_device=False,
        applies_sparse_attn_runtime_kwargs=False,
    )
