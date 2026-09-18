# SPDX-License-Identifier: Apache-2.0
"""Ascend sparse-MLA attention for DeepSeek-V4.1-Flash.

Upstream's ``DeepseekV4Attention`` holds the whole layer -- projections,
compressor, indexer, KV insertion, output projection -- and leaves the
platform-specific sparse attention to a subclass. The CUDA subclasses live in
``vllm/models/deepseek_v4_1/nvidia/`` and the ROCm one in ``amd/rocm.py``; this
is the Ascend one.

Besides the three abstract methods, the base class reaches for CUDA C++
extensions (``torch.ops._C.fused_deepseek_v4_*`` in
``_fused_qnorm_rope_kv_insert``) that are not behind any hook, so those
methods are overridden here as well. Paths not covered here raise rather than
silently doing the CUDA thing.
"""

from __future__ import annotations

from typing import ClassVar

import torch
import torch_npu
from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend
from vllm_ascend.models.deepseek_v4_1.indexer_select import \
    compressed_lengths
from vllm_ascend.models.deepseek_v4_1.o_proj import check_grouped_wo_a_layout
from vllm_ascend.models.deepseek_v4_1.platform_patches import \
    assert_ready_for_ratio_two
from vllm_ascend.models.deepseek_v4_1.rope_groups import register_rope_groups
from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
    kv_with_layout_rank
from vllm_ascend.models.deepseek_v4_1.rope import \
    inverse_rope_from_cos_sin_cache
from vllm_ascend.utils import oproj_tp_enable

#: MX grouping the A5 quantised batch matmul expects: no grouping on the two
#: batch axes, 32 elements per scale along the contraction axis. Mirrors
#: DSAImpl._forward_o_proj in vllm_ascend/attention/dsa_v1.py.
MX_GROUP_SIZES = (0, 0, 32)

#: npu_transpose_batchmatmul takes [tokens, groups, in] against
#: [groups, in, rank]; these permutations put the group axis first for the
#: matmul and move it back afterwards.
PERM_X1 = (1, 0, 2)
PERM_X2 = (0, 1, 2)
PERM_Y = (1, 0, 2)


def _is_real_batch() -> bool:
    """Whether this forward has scheduler metadata behind it.

    The same test ``sparse_indexer.forward_oot`` uses to decide not to write
    the selection buffer, so the reader and the writer agree on what a dummy
    run is.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return False
    return isinstance(getattr(context, "attn_metadata", None), dict)


_WARNED_UNWRITTEN_SELECTION = False


def _warn_unwritten_selection_on_dummy_run() -> None:
    """Warn once per process that a dummy run read the unwritten selection.

    Harmless for the output of a dummy run; on a real batch the same read
    would be a defect, which is why ``_checked_selection`` raises there.
    """
    global _WARNED_UNWRITTEN_SELECTION
    if _WARNED_UNWRITTEN_SELECTION:
        return
    _WARNED_UNWRITTEN_SELECTION = True
    from vllm.logger import logger

    logger.warning(
        "DSV4.1: a dummy/profile forward read the indexer selection buffer "
        "before anything wrote it. Expected on that path: the indexer skips "
        "the write when there is no metadata.")


class AscendDeepseekV41Attention(DeepseekV4Attention):
    """Sparse-MLA attention on Ascend."""

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
    #
    # Flipping it is not a one-line change. Upstream's two fused prologs differ
    # by more than the cache layout: the fp8_ds_mla one
    # (fused_..._rope_quant_insert) also zeroes the padding head slots during
    # Q-side RoPE and returns a padded q it allocated itself, where the bf16 one
    # (fused_..._full_cache_bf16_insert, which is what this False selects)
    # rewrites q in place with no padding at all. Nobody has found where that
    # padding lands on Ascend. So this flips together with quant_mode=2, or the
    # padding heads carry whatever was in them.
    use_fp8_ds_mla_layout: ClassVar[bool] = False

    def get_attn_backend(self) -> type[AttentionBackend]:
        """The backend of this layer's own compressed-KV cache.

        Upstream returns ``backend_cls`` -- here the SWA backend, whose kernel
        block sizes are [32, 64, 128]. A ratio-2 kv-source layer's spec has
        ``block_size`` 256 and shares its KV cache group with the ratio-2 indexer
        caches (AscendDSAC2Backend, [64, 128, 256]). ``select_common_block_size``
        needs a size every backend in the group lists, so 256 fell back to 128 and
        the runner split every allocator block b of the group into kernel blocks
        (2b, 2b+1). The compressed cache view, the planner and the operator all
        work in 128-entry pages at allocator-block stride, so kernel page 2b is
        allocator block 2b's memory -- which the shared pool has handed to some
        other group. Outputs then depend on which block ids a request happens to
        get, and requests corrupt each other.

        Ratio 2 therefore reports the C2 backend, so the group resolves to 256 and
        no split happens. Ratio 1 (block 128) is unaffected.
        """
        if getattr(self, "compress_ratio", 0) == 2:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC2Backend
            return AscendDSAC2Backend
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """The compressed-KV cache, described the way Ascend's kernels read it.

        Without this the layer inherits upstream's, which returns a plain
        ``MLAAttentionSpec``. That is not merely a different class: three places
        in the runner test ``isinstance(spec, AscendMLAAttentionSpec)`` and take
        a non-Ascend branch when it fails -- ``_uses_page_strided_kv_layout``,
        ``_get_attention_kv_cache_dims``, and the page-size formula itself.

        The two conventions also disagree about where the compression ratio
        goes, and it is not a wash. Upstream fixes ``block_size`` at the cache
        config's value and lets ``storage_block_size`` shrink by the ratio;
        Ascend keeps the storage page at what the block-size table says and
        multiplies ``block_size`` up instead. For a ratio-2 layer at cache
        block 128 that is 64 states of storage against 128 -- the same cache
        described with pages half the size. The table is the kernel's contract,
        so the table wins; see the page-size test for the four combinations it
        has to reproduce.

        The ratio-2 spec's ``block_size`` of 256 is also why AscendDSAC2Backend
        lists [64, 128, 256]. Table, spec convention and backend support set are
        three exits from one decision, not three decisions.
        """
        # Only kv-source layers own a compressed-KV cache. Consumers read the
        # source's through the forward context, and ratio-0 layers are pure SWA.
        if not self.is_kv_source:
            return None

        from vllm_ascend.attention.dsa_attn_kv_plan import is_a5_bf16_kv_enabled
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
        from vllm_ascend.device.hardware_profile import (
            HardwareCapability, get_current_hardware_profile)
        from vllm_ascend.models.layer.attention.layer import dsv4_block_sizes

        use_bf16_kv = is_a5_bf16_kv_enabled(vllm_config)
        compressed = get_current_hardware_profile().supports(
            HardwareCapability.DSV4_COMPRESSED_CACHE)

        dtype = self.kv_cache_torch_dtype
        if use_bf16_kv:
            dtype = torch.bfloat16
        elif compressed:
            dtype = torch.float8_e4m3fn
            vllm_config.cache_config.cache_dtype = "float8_e4m3fn"

        # The compressed layout carries the RoPE half and the scale inside the
        # state rather than beside it: 448 NoPE fp8 + 128 RoPE + scale + pad,
        # 640 bytes. Expressed as 128 extra head elements at one byte each, the
        # way V4.0 expresses it, so both models produce the same page from the
        # same table entry.
        cached_head_size = (self.head_dim + 128
                            if compressed and not use_bf16_kv else self.head_dim)

        storage_block_size = dsv4_block_sizes(vllm_config)[
            vllm_config.cache_config.block_size][0][0]
        # max(ratio, 1): a kv source always has at least one token per state.
        # Zero would make block_size 0, which select_common_block_size does not
        # reject -- 0 % n == 0 for every n, so it silently takes the largest.
        ratio = max(self.compress_ratio, 1)
        from vllm_ascend.models.deepseek_v4_1.caches import \
            stated_stream_ratio
        return AscendMLAAttentionSpec(
            # The scheduler counts raw tokens; the kernels use the compressed
            # page that storage_block_size names. block_size is the product.
            block_size=storage_block_size * ratio,
            num_kv_heads=1,
            head_size=cached_head_size,
            dtype=dtype,
            model_version="deepseek_v4",
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
            tokens_per_state=ratio,
            # `ratio` above is the clamped page divisor; this is the layer's
            # configured entry. They differ only at 0, and only kv-source
            # layers reach here, so on this path they always agree -- which is
            # precisely why it has to be passed rather than assumed equal.
            stream_compress_ratio=stated_stream_ratio(self),
        )

    def _fused_qnorm_rope_kv_insert(self, q, kv, positions, attn_metadata):
        """Upstream's fused prolog, decomposed. See deepseek_v4_1.qkv_prolog.

        All three dtype branches of the base method call a
        ``torch.ops._C.fused_deepseek_v4_*`` symbol that does not exist here,
        and it is reached from ``_prepare_and_attn_eager`` on every forward.
        """
        import torch.nn.functional as F

        from vllm_ascend.models.deepseek_v4_1.qkv_prolog import \
            qnorm_rope_kv_insert

        if not isinstance(attn_metadata, dict):
            # Profile run. Upstream pads so the downstream MLA sees the right
            # shape; nothing is written to the cache because nothing is bound.
            if self.n_local_heads < self.padded_heads:
                return F.pad(
                    q, (0, 0, 0, self.padded_heads - self.n_local_heads),
                    value=0.0)
            return q

        swa_metadata = attn_metadata.get(self.swa_cache_layer.prefix)
        if swa_metadata is None:
            raise ValueError(
                f"no attention metadata under {self.swa_cache_layer.prefix!r}; "
                "the SWA cache group is where this layer's slot mapping and "
                "block size come from.")
        swa_kv_cache = self.swa_cache_layer.kv_cache
        if swa_kv_cache.dtype != torch.bfloat16:
            raise NotImplementedError(
                f"the Ascend prolog covers the bf16 SWA cache; this one is "
                f"{swa_kv_cache.dtype}. The fp8_ds_mla (uint8) branch also "
                "zero-fills padded head slots and returns a q the kernel "
                "allocated, and the per-tensor fp8 branch returns a separate "
                "fp8 q -- neither is this function with a different dtype.")

        block_size = swa_metadata.block_size
        if swa_kv_cache.shape[1:] != (block_size, self.head_dim):
            raise ValueError(
                f"the SWA cache is {tuple(swa_kv_cache.shape)} and the "
                f"metadata says block_size {block_size}, head_dim "
                f"{self.head_dim}. These index each other, so a mismatch "
                "writes rows at the wrong offset rather than failing.")
        return qnorm_rope_kv_insert(
            q=q, kv=kv, positions=positions,
            cos_sin_cache=self.rotary_emb.cos_sin_cache,
            swa_kv_cache=swa_kv_cache,
            slot_mapping=swa_metadata.slot_mapping,
            eps=self.eps, rope_head_dim=self.rope_head_dim)

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

        # Kept because three call sites in forward_mqa read it and the base
        # class does not store it. nn.Module.__getattr__ turns a missing
        # attribute into an AttributeError at the first forward rather than at
        # construction, so an unstored config is a runtime failure on a card,
        # several frames inside upstream code.
        self.vllm_config = vllm_config

        # Read by forward_mqa and defined by neither this class nor the base.
        # Upstream's only softmax_scale is DeepseekV4Indexer's -- a different
        # class, for the indexer's own logits -- so the main MLA scale has no
        # owner upstream and has to be stated here.
        #
        # head_dim ** -0.5, matching AscendDSAImpl (dsa_v1.py:1685), which
        # feeds the same build_sparse_flash_mla_kwargs from the same operator
        # family. No YaRN mscale correction: build_deepseek_v4_rope sets
        # mscale and mscale_all_dim to 0 for every V4 layer, so the usual
        # scaled-rope adjustment does not apply.
        self.softmax_scale = self.head_dim**-0.5

        # The compressor metadata builder looks its frequency table up by group
        # name -- dsa_v1 builds f"c{ratio}" for every layer with ratio > 1 --
        # and that registry is populated only by constructing an Ascend
        # ComplexExpRotaryEmbedding. V4.1 builds upstream's rope instead, so
        # nothing ever registered and a ratio-2 layer failed with
        # "RoPE group c2 is not registered". Registered from the parameters of
        # the rope upstream just built, never from a second derivation of them:
        # the base differs per layer (compress_rope_theta 160000 against
        # rope_theta 10000 here) and getting it wrong raises nothing.
        register_rope_groups(
            vllm_config,
            layername=prefix,
            rotary_emb=self.rotary_emb,
            compress_ratio=self.compress_ratio,
            head_size=self.rope_head_dim,
        )

        # The SWA cache reports this layer's kv-cache spec but knows nothing
        # about compression: upstream's DeepseekV4SWACache takes head_dim,
        # window, dtype, prefix, cache_config, backend and block_size, and is
        # constructed inside the base __init__ with a fixed argument list that
        # a subclass cannot extend. So the ratio is attached here, after the
        # base has built it -- the one point where the per-layer ratio and the
        # per-layer cache object are both in scope.
        self.swa_cache_layer.stream_compress_ratio = self.compress_ratio

    # -- upstream's abstract surface ---------------------------------------

    @staticmethod
    def _configured_ratio(vllm_config, prefix: str) -> int | None:
        """This layer's compress_ratios entry, read before super() sets it.

        Reads ``hf_config`` rather than ``hf_text_config``: V4.1-Flash is a
        multimodal checkpoint and ``DeepseekV41Config`` flattens
        ``text_config`` to the top level. ``None`` means the ratio could not
        be read, which callers treat as "not ratio 2".
        """
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

    @staticmethod
    def _unwrap_cache(cache):
        """Strip the single-element list levels the Ascend runner adds.

        This is where the V4.1 kv-cache convention question resolves, and it
        resolves by following what Ascend already does rather than by changing
        either side. Two levels of wrapping exist: the runner's
        `_adjust_kv_layout` returns a list per layer, and the V4-family branch
        of the binding loop wraps that again. `unfold_kvcache` in
        vllm_ascend/ops/dsa strips exactly the length-1 levels and stops at the
        indexer's length-2/3 list, whose slots are addressed by fixed index.

        The alternatives were worse. Routing V4.1 to the runner's other branch
        changes nothing -- vllm_ascend replaces `bind_kv_cache` wholesale at
        import, so neither branch calls a layer's own method. Overriding
        `bind_kv_cache` on the V4.1 layers would mean forking upstream at four
        sites, and upstream's version squeezes dim 1 while the Ascend cache
        shape carries the head dim at index 2 -- on a non-unit dimension
        `squeeze` is a silent no-op, so that fork would quietly do nothing.
        """
        while isinstance(cache, (list, tuple)) and len(cache) == 1:
            cache = cache[0]
        return cache

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Sparse MLA through cann_ops_transformer.sparse_flash_mla.

        The argument rules the operator states about itself are enforced by
        build_sparse_flash_mla_kwargs, on CPU, so they do not have to be
        rediscovered one rejection at a time here.

        What is exercised only on a card is the metadata mapping just below:
        which AscendDSAReqMetadata field supplies each operator argument. Every
        lookup is therefore explicit and named -- a missing field raises here
        with the field named, rather than passing None into the operator.
        """
        from vllm.forward_context import get_forward_context

        from vllm_ascend.attention.dsa_attn_kv_plan import get_dsa_attn_kv_plan
        from vllm_ascend.attention.dsa_v1 import (_dsa_swa_only_cmp_ratio,
                                                  attends_compressed_stream)
        from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import (
            build_sparse_flash_mla_kwargs, ori_sparse_selection)

        attn_metadata = getattr(get_forward_context(), "attn_metadata", None)
        if attn_metadata is None:
            # Warmup / profiling run: no real metadata yet. Upstream's ROCm
            # path does the same. Zeroing rather than leaving the buffer
            # uninitialised keeps a warmup step from producing NaNs that look
            # like a numerical problem.
            output.zero_()
            return

        swa_prefix = self.swa_cache_layer.prefix
        swa_metadata = attn_metadata.get(swa_prefix)
        if swa_metadata is None:
            raise KeyError(
                f"no attention metadata for {swa_prefix!r}. The SWA cache "
                "registers itself in the compilation context at construction, "
                "so an absent entry means its kv-cache spec was not collected "
                "-- check get_kv_cache_spec on the Ascend SWA cache.")
        req = swa_metadata.req_metadata
        if req is None:
            raise ValueError(
                f"{swa_prefix}: metadata carries no req_metadata; the builder "
                "produced an attention metadata object without per-request "
                "fields, so there is nothing to address the KV with.")

        attends_compressed = attends_compressed_stream(self.compress_ratio,
                                                       self.vllm_config)
        plan = get_dsa_attn_kv_plan(self.vllm_config)

        cmp_kv = cmp_block_table = None
        # The group whose descriptor the kernel takes its tiling from. V4.0:
        # `common_attn_metadata = layer_metadata.attention or layer_metadata.swa`
        # -- the compressed group when there is one, the SWA group otherwise.
        # A ratio-0 layer has no compressed group and correctly falls back.
        cmp_req = None
        if attends_compressed:
            # Through the same rank helper as ori_kv: the operator checks the
            # stride-vector length against layout_kv for both caches.
            cmp_kv = kv_with_layout_rank(
                self._unwrap_cache(self._compressed_kv_cache()),
                plan.layout_kv)
            # Not req.block_table: `req` is the SWA group's metadata, and the
            # compressed cache belongs to another group (for a consumer layer,
            # another layer's cache). The other group's block ids are in range
            # and address the wrong blocks. Resolved through the same prefix
            # the cache came from, so table and cache cannot disagree.
            cmp_req = self._compressed_group_metadata(attn_metadata)
            cmp_block_table = cmp_req.block_table

        # Explicit original-KV indices, following dsa_v1's precedence: a vision
        # prefill supplies them so tokens inside an image span see the whole
        # span bidirectionally, and a layer with no compressed stream may carry
        # DSpark's instead. A layer that attends a compressed stream keeps the
        # vision ones and does not take DSpark's.
        ori_sparse_indices, ori_topk_length = ori_sparse_selection(
            attends_compressed, req)

        # Bound once, so the indices and the length describe the same object.
        # Two expressions producing "the selection" is how they come apart.
        cmp_selection = (
            self._checked_selection(q.shape[0])
            if attends_compressed and self.topk_indices_buffer is not None
            else None)

        # Bound once: the residual must use the ratio this call passes.
        cmp_ratio = _dsa_swa_only_cmp_ratio(self.compress_ratio,
                                            self.vllm_config)
        cmp_residual_kv = (
            compressed_lengths(cmp_req.seq_lens, cmp_ratio)[1]
            if attends_compressed and cmp_req is not None else None)

        # sas_metadata is filled asynchronously on the device-metadata stream.
        # Under FULL graph capture this call is what records the cross-stream
        # wait into the graph; without it the operator may read the descriptor
        # while it is still being written. The backend forwards (dsa_v1,
        # dsa_cp) have the same call; this hand-written call site needs its
        # own. It returns immediately when there is no executor.
        from vllm_ascend.worker.device_metadata import (
            DeviceMetadataStage, wait_for_device_metadata)

        sas_metadata = (cmp_req or req).sas_metadata
        if isinstance(sas_metadata, torch.Tensor):
            wait_for_device_metadata(DeviceMetadataStage.ATTENTION,
                                     id(sas_metadata))

        swa_kv = kv_with_layout_rank(
            self._unwrap_cache(self.swa_cache_layer.kv_cache),
            plan.layout_kv)

        kwargs = build_sparse_flash_mla_kwargs(
            q=q,
            num_requests=req.query_start_loc.numel() - 1,
            swa_kv_cache=swa_kv,
            swa_block_table=req.block_table,
            cu_seqlens_q=req.query_start_loc,
            seqused_kv=req.seq_lens,
            sinks=self.attn_sink,
            # The compressed group's descriptor when there is one: the SWA
            # group's is built at ratio 1, a well-formed descriptor with the
            # wrong tiling for a ratio-2 layer, which faults every core.
            metadata=(cmp_req or req).sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=cmp_ratio,
            layout_kv=plan.layout_kv,
            attends_compressed=attends_compressed,
            ori_win_left=req.ori_win_left,
            ori_win_right=req.ori_win_right,
            cmp_kv=cmp_kv,
            cmp_block_table=cmp_block_table,
            # One value per request, from the group that owns the compressed
            # stream -- not the cumulative cu_cmp_seqlen_list, which the
            # operator would also accept and read as lengths.
            seqused_cmp_kv=(cmp_req.seqused_cmp_kv
                            if attends_compressed and cmp_req else None),
            # Derived here rather than read from req.qli_cmp_residual_k, which
            # the shared builder publishes only for ratio 4. None at ratio 1,
            # which is what the operator requires there.
            cmp_residual_kv=cmp_residual_kv,
            ori_sparse_indices=ori_sparse_indices,
            ori_topk_length=ori_topk_length,
            # The indexer's selection; without it the compressed stream is
            # attended in full. Shaped [T, 1, K2] by _checked_selection: the
            # operator takes [T, KV_N, K2] under layout_q=TND and V4.1 has a
            # single KV head.
            cmp_sparse_indices=cmp_selection,
            # Travels with the indices: without a length the operator treats
            # the whole -1 padded row as valid and gathers at the padding.
            cmp_topk_length=(
                self._compressed_topk_length(cmp_selection)
                if cmp_selection is not None else None),
        )

        output.copy_(plan.get_dsa_sparse_attn_op()(q, **kwargs)[0])

    def _checked_selection(self, num_tokens: int) -> torch.Tensor:
        """The indexer's selection for these rows, refusing unwritten ones.

        ``topk_indices_buffer`` is allocated with ``torch.empty`` upstream and
        written only over the rows the indexer handled this step. Rows past
        that hold the previous step's selection (or uninitialised memory):
        plausible non-negative indices that ``_compressed_topk_length`` would
        count as valid. So the check is on provenance -- how many rows were
        written -- not on the values.

        Three states:

        * never written on a dummy/profile run: expected, the indexer skips
          the write when there is no metadata;
        * never written on a real batch: refused;
        * fewer rows written than read: under graph capture ``num_tokens`` is
          padded to the capture size while the indexer wrote only the real
          tokens. The tail is overwritten with ``NO_TOKEN`` (a shape-constant
          device write, safe to capture) so no stale row reaches the operator.
        """
        from vllm_ascend.models.deepseek_v4_1.sparse_indexer import rows_written

        written = rows_written(self.topk_indices_buffer)
        if written is None:
            if not _is_real_batch():
                _warn_unwritten_selection_on_dummy_run()
                return self.topk_indices_buffer[:num_tokens].unsqueeze(1)
            raise ValueError(
                f"{self.prefix}: this layer attends a compressed stream on a "
                "real batch and the indexer selection buffer has never been "
                "written. It is allocated with torch.empty, so reading it "
                "would make the output depend on device memory rather than "
                "on the prompt.")
        if written < num_tokens:
            from vllm_ascend.models.deepseek_v4_1.sparse_indexer import (
                NO_TOKEN, mark_rows_written)
            self.topk_indices_buffer[written:num_tokens] = NO_TOKEN
            mark_rows_written(self.topk_indices_buffer, num_tokens)
        return self.topk_indices_buffer[:num_tokens].unsqueeze(1)

    @staticmethod
    def _compressed_topk_length(selection: torch.Tensor) -> torch.Tensor:
        """Valid entries per row of the selection **being sent**.

        Counted from that exact tensor, which is the one case where counting
        at the consumer is not a second source: it cannot disagree with the
        table it describes, because it *is* that table. The rows are -1 padded
        from the right, so the count of non-negative entries is the number of
        leading valid ones -- which is what the operator's length means.

        Counted here rather than published by the indexer: the buffer is
        shared by every layer, while ``self.indexer`` exists only on index
        source layers.
        """
        return (selection >= 0).sum(dim=-1, dtype=torch.int32)

    def _compressed_group_metadata(self, attn_metadata: dict):
        """The metadata of the group that owns this layer's compressed cache.

        Not the SWA group's. The block table, the compressed lengths and the
        descriptor all belong to whichever group holds the compressed cache --
        for a consumer layer, another layer's (``_compressed_kv_cache``
        follows ``compressed_cache_prefix``). Taking them from the SWA group
        is not a shape error: its block ids are in range and its descriptor is
        well-formed for ratio 1, so the mistake surfaces only as an aicore
        exception. Resolved through the same prefix that selects the cache.
        """
        prefix = (self.prefix if self.is_kv_source
                  else self.compressed_cache_prefix)
        if prefix is None:
            raise ValueError(
                f"{self.prefix}: this layer attends a compressed stream but "
                "has no compressed_cache_prefix, so there is no way to say "
                "which cache group describes it.")
        metadata = attn_metadata.get(prefix)
        if metadata is None:
            raise ValueError(
                f"{self.prefix}: no attention metadata for the compressed "
                f"cache at {prefix!r}. Falling back to this layer's own group "
                "would describe that cache with another group's block ids and "
                "another ratio's descriptor -- both well-formed, both wrong. "
                f"Groups present: {sorted(attn_metadata)[:4]}")
        req = getattr(metadata, "req_metadata", None)
        if req is None or getattr(req, "block_table", None) is None:
            raise ValueError(
                f"{self.prefix}: the metadata for {prefix!r} carries no "
                "req_metadata with a block table, so the compressed cache "
                "cannot be addressed.")
        return req

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Inverse RoPE, then the block-diagonal wo_a, then wo_b.

        The two halves have very different provenance. The wo_a/wo_b half
        mirrors ``DSAImpl._forward_o_proj`` in ``vllm_ascend/attention/dsa_v1``
        -- same operators, same layout, same permutations -- because V4.0 has
        been running it on this hardware.

        The inverse rotation has no V4.0 counterpart: V4.0 is MLA, whose attention output lives in the NoPE latent
        space and carries no rotation to undo. V4.1 has ``num_key_value_heads=1``
        with K and V the *same* 512-wide tensor, so the value side carries the
        64 RoPE dimensions and the output inherits them. Skipping this step is
        not a smaller error than getting it wrong.

        Upstream fuses the rotation into the quantisation kernel; here it is
        several plain torch ops.
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
