# SPDX-License-Identifier: Apache-2.0
"""Ascend sparse-MLA attention for DeepSeek-V4.1-Flash.

Upstream's ``DeepseekV4Attention`` holds the whole layer -- projections,
compressor, indexer, KV insertion, output projection -- and leaves the
platform-specific sparse attention to a subclass. The CUDA subclasses live in
``vllm/models/deepseek_v4_1/nvidia/`` and the ROCm one in ``amd/rocm.py``; this
is the Ascend one.

Only three methods are ``@abstractmethod`` upstream, but that is the floor, not
the real surface: the ROCm subclass overrides eleven, because the base class
also reaches for CUDA C++ extensions (``torch.ops._C.fused_deepseek_v4_*`` in
``_fused_qnorm_rope_kv_insert``) that are not behind any hook. ROCm gets those
for free -- HIP compiles the same csrc and aliases ``torch.cuda`` -- and Ascend
does not. Methods still to be filled in are marked below and raise rather than
silently doing the CUDA thing.
"""

from __future__ import annotations

from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.v1.attention.backend import AttentionBackend

from vllm_ascend.attention.dsa_v1 import (AscendDSAC4Backend,
                                          AscendDSAC128Backend,
                                          AscendDSASWABackend)


class AscendDeepseekV41Attention(DeepseekV4Attention):
    """Sparse-MLA attention on Ascend.

    Not usable for inference yet -- see the ``NotImplementedError``s below.
    Construction is expected to work so that the rest of the model can be
    built and the next gap found.
    """

    # Selected per layer by compress_ratio in the model layer; this is the
    # default the base class records. The real dispatch happens through
    # vllm_ascend.models.layer.attention, which picks C4/C128/SWA.
    backend_cls: ClassVar[type[AttentionBackend]] = AscendDSASWABackend
    swa_backend_cls: ClassVar[type[AttentionBackend] | None] = AscendDSASWABackend

    # False, deliberately: the operator bound today
    # (KvQuantSparseAttnSharedkv) stores 640 B per state -- rope 128 (bf16,
    # unquantised) | nope 448 (fp8_e4m3, per-tile 64) | scale 7 (e8m0) | pad 57,
    # rounded up from 583 to a 128 B boundary. That is not upstream's
    # fp8_ds_mla layout, which packs the same fields into 584 B.
    #
    # CANN's MixedQuantSparseFlashMla in quant_mode=2 *is* byte-identical to
    # fp8_ds_mla; once that operator is bound, this flips to True and the KV
    # budget drops from 1930 to 1790 B/token.
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    # Backends that accept these ratios today. Ratio 2 -- which V4.1 needs on
    # layers 2..19 -- is missing because of the bound operator, not the
    # hardware; see validate_compress_ratio in vllm_ascend.attention.dsa_v1.
    _RATIO_BACKENDS: ClassVar[dict[int, type[AttentionBackend]]] = {
        4: AscendDSAC4Backend,
        128: AscendDSAC128Backend,
    }

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
    ) -> None:
        # aux_stream_list is dropped: it carries torch.cuda.Stream objects for
        # the base class's multi-stream overlap. Passing None makes the base
        # class's four ln_events unused, which is what keeps its
        # torch.cuda.Event allocation harmless here.
        super().__init__(
            vllm_config,
            prefix,
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=None,
            candidate_block_buffer=candidate_block_buffer,
        )

    # -- upstream's abstract surface ---------------------------------------

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        """Q head count the q/output buffers are allocated at.

        No padding is applied. The sharedkv operator validates
        ``sinks.shape[0] == query head num`` and V4.1 has 64 heads, so padding
        here would have to be mirrored into the sink tensor. Revisit if a
        bound operator turns out to require a tile-aligned head count.
        """
        return num_heads

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        raise NotImplementedError(
            "Ascend sparse MLA forward is not wired up yet. It should call "
            "npu_kv_quant_sparse_attn_sharedkv (or MixedQuantSparseFlashMla "
            "once bound), passing attn_sink -- the operator rejects a null "
            "sinks tensor, so it cannot be skipped even to get something "
            "running.")

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Ascend output projection is not wired up yet. It is inverse-RoPE "
            "on the trailing 64 dims, then wo_a as a block-diagonal grouped "
            "GEMM (o_groups=8, each group 4096 -> 1024, giving 8192) and wo_b "
            "as a plain 8192 -> 5120 linear. The quantisation side already "
            "reshapes wo_a into [o_groups, ...] at load time, so this must not "
            "treat it as a dense [8192, 4096] matmul -- that would let every "
            "group see all 32768 input dims.")
