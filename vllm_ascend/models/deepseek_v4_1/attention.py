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
import torch_npu
from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.v1.attention.backend import AttentionBackend

from vllm_ascend.attention.dsa_v1 import (AscendDSAC4Backend,
                                          AscendDSAC128Backend,
                                          AscendDSASWABackend)
from vllm_ascend.models.deepseek_v4_1.o_proj import check_grouped_wo_a_layout
from vllm_ascend.models.deepseek_v4_1.platform_patches import \
    assert_ready_for_ratio_two
from vllm_ascend.models.deepseek_v4_1.rope import \
    inverse_rope_from_cos_sin_cache
from vllm_ascend.utils import oproj_tp_enable

#: MX grouping the A5 quantised batch matmul expects: no grouping on the two
#: batch axes, 32 elements per scale along the contraction axis. Mirrors
#: DSAImpl._forward_o_proj in vllm_ascend/attention/dsa_v1.py, which is the
#: established convention for this weight on this hardware.
MX_GROUP_SIZES = (0, 0, 32)

#: npu_transpose_batchmatmul takes [tokens, groups, in] against
#: [groups, in, rank]; these permutations put the group axis first for the
#: matmul and move it back afterwards.
PERM_X1 = (1, 0, 2)
PERM_X2 = (0, 1, 2)
PERM_Y = (1, 0, 2)


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
        # A ratio-2 layer will have the compressor kernel compiled for it, and
        # that compile aborts the process unless the platform patch has
        # selected away from the joined-pointer path. Checked here, where the
        # ratio is known and a traceback still exists.
        if getattr(self, "compress_ratio", None) == 2 or self._configured_ratio(
                vllm_config, prefix) == 2:
            assert_ready_for_ratio_two(prefix)

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

    @staticmethod
    def _configured_ratio(vllm_config, prefix: str) -> int | None:
        """This layer's compress_ratios entry, read before super() sets it."""
        try:
            from vllm.model_executor.models.utils import extract_layer_index

            ratios = vllm_config.model_config.hf_config.compress_ratios
            return ratios[extract_layer_index(prefix)]
        except Exception:
            return None

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
        """Inverse RoPE, then the block-diagonal wo_a, then wo_b.

        The two halves have very different provenance. The wo_a/wo_b half
        mirrors ``DSAImpl._forward_o_proj`` in ``vllm_ascend/attention/dsa_v1``
        -- same operators, same layout, same permutations -- because V4.0 has
        been running it on this hardware.

        The inverse rotation has no such precedent, and the reason is worth
        recording: V4.0 is MLA, whose attention output lives in the NoPE latent
        space and carries no rotation to undo. V4.1 has ``num_key_value_heads=1``
        with K and V the *same* 512-wide tensor, so the value side carries the
        64 RoPE dimensions and the output inherits them. Skipping this step is
        not a smaller error than getting it wrong.

        Known gap, deliberately: upstream fuses the rotation into the
        quantisation kernel (one Triton launch); this is several plain torch
        ops. Correct first, and it is a self-contained replacement later.
        """
        if oproj_tp_enable():
            raise NotImplementedError(
                "oproj tensor parallelism is not wired into the V4.1 output "
                "projection. dsa_v1's version exchanges activations with "
                "all_to_all and reduce_scatter through address-stable static "
                "buffers; none of that is reproduced here. Refusing rather "
                "than computing each rank's partial result and returning it as "
                "if it were the whole -- DSV4 runs TP=1 on A5, so this should "
                "not be reachable.")

        cos_sin_cache = getattr(self.rotary_emb, "cos_sin_cache", None)
        if cos_sin_cache is None:
            raise AttributeError(
                f"{self.rotary_emb.__class__.__name__} exposes no "
                "cos_sin_cache. The output projection undoes the rotation the "
                "layer's own rotary module applied, so it has to read that "
                "module's table -- rebuilding one here would be a second copy "
                "of the per-layer theta and YaRN parameters, free to drift.")

        o = inverse_rope_from_cos_sin_cache(o, positions, cos_sin_cache,
                                            self.rope_head_dim)

        num_tokens = o.shape[0]
        group_in = o.shape[1] * o.shape[2] // self.n_local_groups
        check_grouped_wo_a_layout(self.wo_a.weight, self.n_local_groups,
                                  group_in, self.o_lora_rank,
                                  getattr(self, "prefix", None) or "wo_a")
        o = o.view(num_tokens, self.n_local_groups, group_in)

        if getattr(self.wo_a, "weight_scale", None) is not None:
            o, act_scale = torch_npu.npu_dynamic_mx_quant(
                o, dst_type=torch.float8_e4m3fn)
            z = torch_npu.npu_transpose_quant_batchmatmul(
                o,
                self.wo_a.weight,
                dtype=torch.bfloat16,
                bias=None,
                group_sizes=MX_GROUP_SIZES,
                x1_scale=act_scale.view(torch.float8_e8m0fnu),
                x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                perm_x1=PERM_X1,
                perm_x2=PERM_X2,
                perm_y=PERM_Y,
            )
        else:
            z = torch_npu.npu_transpose_batchmatmul(
                o,
                self.wo_a.weight,
                bias=None,
                scale=None,
                perm_x1=PERM_X1,
                perm_x2=PERM_X2,
                perm_y=PERM_Y,
                batch_split_factor=1,
            )

        # wo_b goes through its own linear method: its fp8 activation
        # quantisation is not optional here. Dequantising both matrices to bf16
        # and multiplying leaves 1.8e-2 relative error on a single layer,
        # against 3.5e-5 for a real block-scaled fp8 GEMM.
        return self.wo_b(z.reshape(num_tokens, -1))
