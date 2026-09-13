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
#: DSAImpl._forward_o_proj in vllm_ascend/attention/dsa_v1.py, which is the
#: established convention for this weight on this hardware.
MX_GROUP_SIZES = (0, 0, 32)

#: npu_transpose_batchmatmul takes [tokens, groups, in] against
#: [groups, in, rank]; these permutations put the group axis first for the
#: matmul and move it back afterwards.
PERM_X1 = (1, 0, 2)
PERM_X2 = (0, 1, 2)
PERM_Y = (1, 0, 2)


#: One line per distinct state, not per layer per step.
_LOGGED_SELECTION_STATES: set = set()


def _describe_topk_length(topk_length, idx_host):
    """`ori_topk_length` 只有一个正确性判据：逐行等于该行真实有效数。

    不传它 ⇒ 算子按满宽度当「整行有效」，去 -1 padding 的槽 gather。
    传一个属于【别的行】的 length，比不传更糟 —— 它看起来是对的。
    ⇒ 所以这里不报"传了没有"，报的是它和索引表【对不对得上】。
    """
    if topk_length is None:
        return "🔴 未传（算子会按满宽度当整行有效）"
    try:
        import torch
        t = topk_length.detach().to("cpu")
        real = (idx_host != -1).sum(dim=-1).reshape(-1)
        got = t.reshape(-1)
        if got.numel() != real.numel():
            return (f"🔴 长度数不匹配: length {tuple(t.shape)} -> {got.numel()} 行，"
                    f"索引表 {real.numel()} 行")
        bad = int((got != real).sum().item())
        if bad == 0:
            return (f"✅ 逐行相等（{got.numel()} 行）"
                    f" min={int(got.min())} max={int(got.max())}")
        return (f"🔴 {bad}/{got.numel()} 行不等 —— 这个 length 描述的不是这张表"
                f" (length max={int(got.max())} 实际 max={int(real.max())})")
    except Exception as e:  # 取证失败不能顺带打死这一发
        return f"⚠️ 算不出来（不算通过）: {e!r}"

def _log_ori_sparse_indices(prefix, source, idx, num_tokens, caller=None,
                            topk_length=None):
    """Report the original-stream index list by its *content*, not by presence.

    ``is None`` is not the question. These lists are built into a preallocated
    buffer, and a buffer that was never written is neither None nor zero -- it
    holds whatever was there. The operator gathers the original stream through
    it, so a never-written buffer is an out-of-range gather, which is what an
    aicore exception with non-zero MTE error info looks like.

    The tell is the padding: a list that was really built carries -1 in the
    slots it did not fill. No -1 at all, with a large or unpatterned maximum,
    means nothing wrote it.
    """
    import os

    from vllm.logger import logger

    if int(os.environ.get("DSV41_SAS_DUMP_MAX", "0")) <= 0:
        return
    if idx is None:
        logger.info("【ORI-IDX】%s: ori_sparse_indices=None (来源=无) "
                    "ori_topk_length=%s 调用来自 %s",
                    prefix, "无" if topk_length is None else "🔴非空但索引是None",
                    caller)
        return
    try:
        h = idx.detach().to("cpu").contiguous()
        flat = h.reshape(-1)
        n_neg1 = int((flat == -1).sum().item())
        mx = int(flat.max().item()) if flat.numel() else None
        mn = int(flat.min().item()) if flat.numel() else None
        verdict = ("🔴从未写过的buffer嫌疑(无 -1 填充)" if n_neg1 == 0
                   else "有 -1 填充(写过)")
        logger.info(
            "【ORI-IDX】%s: 来源=%s shape=%s dtype=%s | min=%s max=%s "
            "(==-1)=%d %s | 前8=%s | 量程对照: num_tokens=%s 末维=%s | "
            "🔴ori_topk_length: %s | 调用来自 %s",
            prefix, source, tuple(h.shape), h.dtype, mn, mx, n_neg1, verdict,
            flat[:8].tolist(), num_tokens,
            h.shape[-1] if h.dim() else None,
            _describe_topk_length(topk_length, h),
            caller,
        )
    except Exception as e:
        logger.warning("【ORI-IDX】%s: 读不出来(不致命): %r", prefix, e)


def _log_compressed_selection(*, prefix, layer_ratio, attends_compressed,
                              buffer, num_tokens) -> None:
    """Say whether the indexer's selection actually reaches the operator.

    A register-level diff of two launches showed no zero argument slot becoming
    an address after cmp_sparse_indices started being passed -- so it never
    arrived, while the descriptor went on declaring cmp_topk. That is the
    descriptor-versus-call disagreement again, one level down: the code says
    both are gated on the same fact, and at runtime one of them did not fire.

    The layer's ratio is printed beside the gate because the descriptor's
    version of "attends a compressed stream" is computed in the metadata
    builder from the *cache group's* ratio, and this one from the *layer's*.
    Those are two different objects with the same meaning, and nothing so far
    has shown them to agree.

    Four numbers, because three of them cannot separate "no indices at all"
    from "indices whose rows are all -1".
    """
    from vllm.logger import logger

    has_buffer = buffer is not None
    # Deduplicated per layer, not per state. A state key collapses every layer
    # that shares one into a single line, and "the log names one layer" then
    # reads as "one layer took this path" -- which is what it was read as.
    if prefix in _LOGGED_SELECTION_STATES:
        return
    _LOGGED_SELECTION_STATES.add(prefix)

    if not has_buffer:
        logger.info(
            "DSV4.1 %s: layer compress_ratio=%s attends_compressed=%s "
            "topk_indices_buffer=None num_tokens=%s",
            prefix, layer_ratio, attends_compressed, num_tokens)
        return

    rows = buffer[:num_tokens]
    capacity = int(rows.numel())
    pad = int((rows == -1).sum())
    selected = capacity - pad
    # capacity and pad are both reported because "selected == capacity" is not
    # "every slot was chosen": a buffer allocated with torch.empty and never
    # written holds arbitrary values, almost all of them non-negative, and
    # counts as full. Under a causal mask the early tokens cannot fill 512
    # slots, so a *written* buffer always carries padding. No padding at all
    # therefore means unwritten, not saturated -- and those want opposite
    # investigations.
    logger.info(
        "DSV4.1 %s: layer compress_ratio=%s attends_compressed=%s "
        "topk_indices_buffer=present selected=%s pad_minus_one=%s "
        "capacity=%s num_tokens=%s%s",
        prefix, layer_ratio, attends_compressed, selected, pad, capacity,
        num_tokens,
        "  [no padding at all: the buffer reads as never written]"
        if pad == 0 else "")


def _log_batch_shape(prefix, query_start_loc, seq_lens) -> None:
    """The batch's segmentation, printed whole.

    Not summarised as "num_requests", because the question this answers is
    whether the batch is one prompt or a profile run's many dummy segments, and
    a single count invites the reading that produced the last table: a value
    derived from "there is one prompt" when that premise was never measured.
    The segment boundaries are the evidence; the count is a consequence.
    """
    from vllm.logger import logger

    if prefix in _LOGGED_BATCH_SHAPES:
        return
    _LOGGED_BATCH_SHAPES.add(prefix)
    starts = query_start_loc.tolist() if query_start_loc is not None else None
    lens = seq_lens.tolist() if seq_lens is not None else None
    logger.info(
        "DSV4.1 %s batch: num_segments=%s cu_seqlens_q=%s seq_lens=%s",
        prefix,
        (len(starts) - 1) if starts else None,
        starts if starts is None or len(starts) <= 40 else
        f"{starts[:8]} ... {starts[-8:]} (len {len(starts)})",
        lens if lens is None or len(lens) <= 40 else
        f"{lens[:8]} ... {lens[-8:]} (len {len(lens)})")


_LOGGED_BATCH_SHAPES: set = set()


def _is_real_batch() -> bool:
    """Whether this forward has scheduler metadata behind it.

    The same test ``sparse_indexer.forward_oot`` uses to decide not to write
    the selection buffer, so the reader and the writer agree on what a dummy
    run is instead of each having its own idea.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return False
    return isinstance(getattr(context, "attn_metadata", None), dict)


_WARNED_UNWRITTEN_SELECTION = False


def _warn_unwritten_selection_on_dummy_run() -> None:
    """Once per process, not once per layer per step.

    Said at all because the traceback that produced this branch also falsified
    a comment: forward_oot said "during profiling nothing reads it at all",
    and forward_mqa demonstrably does. Reading uninitialised memory on a dummy
    run is harmless for the output, and it is still worth one line that it
    happens.
    """
    global _WARNED_UNWRITTEN_SELECTION
    if _WARNED_UNWRITTEN_SELECTION:
        return
    _WARNED_UNWRITTEN_SELECTION = True
    from vllm.logger import logger

    logger.warning(
        "DSV4.1: a dummy/profile forward read the indexer selection buffer "
        "before anything wrote it. Expected on that path -- the indexer skips "
        "the write when there is no metadata -- and reported once because the "
        "same buffer is uninitialised memory and a real batch reading it "
        "would be a defect.")





def _cache_page_size(cache) -> "int | None":
    """Tokens per block, from the cache's own shape.

    ``[num_blocks, block_size, ...]`` under PA_BBND, so axis 1 is the page.
    Taken from the tensor rather than from the config: the two have already
    disagreed once on this line, and the tensor is the one being indexed.
    """
    if not isinstance(cache, torch.Tensor) or cache.ndim < 2:
        return None
    return int(cache.shape[1])


def _tap_write_side_kv(attn, kv, positions, slot_mapping, cache) -> None:
    """What the original-stream cache is about to be given, before it is.

    ⚠️ Pre-rope. ``qnorm_rope_kv_insert`` applies the rope inside, so this
    tensor is the input to that and the cache row is the output -- comparing
    the two directly is comparing across a rotation, which is exactly the
    mistake that made a cosine of 0.15 look like a finding. The rope is an
    orthogonal transform, so a *content* comparison at equal positions is
    still valid; a positional one is not.

    Recorded whole, because there is no token-position scatter to do here
    that the reader could not do better with the slot mapping alongside.
    """
    from vllm_ascend.models.deepseek_v4_1 import align_taps, read_taps

    if not align_taps.alignment_active():
        return
    try:
        layer = align_taps.layer_index(
            attn.prefix, attn.vllm_config.model_config.hf_text_config)
        if not read_taps.records_reads(layer):
            return
        read_taps.tap_read(layer, "write.ori_kv.latent_pre_rope", kv)
        read_taps.tap_read(layer, "write.ori_kv.positions", positions)
        read_taps.tap_read(layer, "write.ori_kv.slots", slot_mapping)
        # 🔴 Suffixed, not hashed. A raw pointer and a hash of it are both
        # process-local, so neither can be equal across ranks; the only fix
        # is to take the key out of the bit-exact comparison, and the suffix
        # says so in the key itself so a comparator needs no exception list.
        from vllm_ascend.models.deepseek_v4_1.compressor_insert import \
            PROCESS_LOCAL

        read_taps.tap_read(
            layer, f"write.ori_kv.cache_ptr{PROCESS_LOCAL}",
            torch.tensor([cache.data_ptr()], dtype=torch.int64))
    except Exception:                                        # noqa: BLE001
        # Instrument. One of these has already cost a flight.
        pass


def _tap_what_this_layer_read(attn, kwargs: dict, req, cmp_req) -> None:
    """Everything the operator is about to be handed, plus the blocks behind
    the two block tables.

    ⚪ Taken from ``kwargs`` rather than from the variables that produced
    them: what is wanted is what the operator receives, and this line has
    already been bitten twice by the gap between the two (the pre-adapter
    vocabulary, and the descriptor built from another group).

    ⚠️ Only on the layers ``read_taps`` selects, and only while a dump is
    being recorded. The caches behind these tables are gigabytes; gathering
    them for forty layers would produce a dump nobody can compare.
    """
    from vllm_ascend.models.deepseek_v4_1 import align_taps, read_taps

    if not align_taps.alignment_active():
        return
    layer = align_taps.layer_index(attn.prefix,
                                   attn.vllm_config.model_config.hf_text_config)
    if not read_taps.records_reads(layer):
        return

    # The selections and lengths: the indexer's output is the sparse
    # operator's direct input, and until now it was NOT MEASURED rather than
    # measured-and-equal. Identical q and kv with different indices is enough
    # on its own to produce a different output.
    for key in ("ori_sparse_indices", "cmp_sparse_indices", "ori_topk_length",
                "cmp_topk_length", "seqused_ori_kv", "seqused_kv",
                "seqused_cmp_kv", "cmp_residual_kv", "metadata",
                "ori_win_left", "ori_win_right", "ori_mask_mode",
                "cmp_mask_mode", "cmp_ratio"):
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor):
            read_taps.tap_read(layer, key, value)
        elif isinstance(value, int):
            read_taps.tap_read(layer, key, torch.tensor([value]))

    # The two caches, gathered by the block tables this call passes.
    # ⚠️ With the lengths, so the block list is bounded by what the request
    # uses rather than by "non-negative" -- the table is padded with zero and
    # block 0 is somebody else's.
    page = _cache_page_size(kwargs.get("ori_kv"))
    read_taps.tap_cache_read(layer, "ori_kv", kwargs.get("ori_kv"),
                             kwargs.get("ori_block_table"),
                             kwargs.get("seqused_ori_kv")
                             or kwargs.get("seqused_kv"), page)
    read_taps.tap_cache_read(layer, "cmp_kv", kwargs.get("cmp_kv"),
                             kwargs.get("cmp_block_table"),
                             kwargs.get("seqused_cmp_kv"),
                             _cache_page_size(kwargs.get("cmp_kv")))
    # 🔵 The half of golden's sa_in.kv this side keeps as a separate
    # argument. Recorded under its own key with the relation declared, not
    # concatenated into a tensor the operator never received as one.
    read_taps.tap_operator_rows(layer, kwargs.get("cmp_kv"),
                                kwargs.get("cmp_block_table"),
                                kwargs.get("seqused_cmp_kv"))

    # candidate blocks, which only some layers carry; absent is absent, not
    # an empty tensor.
    for source in (cmp_req, req):
        candidates = getattr(source, "candidate_blocks", None)
        if isinstance(candidates, torch.Tensor):
            read_taps.tap_read(layer, "candidate_blocks", candidates)
            break


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

    # A _RATIO_BACKENDS map used to sit here, recording that ratio 2 had no
    # backend. Nothing ever read it, and by the time anyone looked it was false
    # too -- AscendDSAC2Backend exists. Ratio-to-backend now resolves in
    # AscendDeepseekV41IndexerCache.get_attn_backend, from the spec that
    # actually constrains it.

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
        # 🔴 The write side of the original stream. golden searched all 40
        # layers of its own dump -- 275,691 slices -- for anything matching
        # the NoPE 448 dims of our cached rows and found max |cos| = 0.106,
        # while the values are plainly activations (std 0.339 against its
        # 0.306, no zeros, mutually near-orthogonal). So the question is not
        # "where was it written" but "what was written", and only this side
        # has the answer: `kv` here is the tensor that becomes those dims.
        _tap_write_side_kv(self, kv, positions, swa_metadata.slot_mapping,
                           swa_kv_cache)
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

        ⚠️ Reads ``hf_config``, not ``hf_text_config``, and that is currently
        correct but not obviously so: V4.1-Flash is a multimodal checkpoint,
        and the same wrapper has broken three other assumptions written for a
        text-only shape (the ``use_compress`` config read, the ``{0: []}``
        multimodal ranges, the module tree under ``language_model``). This one
        survives because ``DeepseekV41Config`` flattens ``text_config`` to the
        top level -- measured, not assumed: ``hasattr(hf_config,
        "compress_ratios")`` is True and the first entries are [0, 0, 2, 2, 2].

        🔴 The ``except Exception`` is a known silent failure point, tolerated
        for now and not safe. It cannot tell "this model has no ratios" from
        "the read path moved", and both come back as ``None`` -- which every
        caller downstream reads as "no ratio configured", i.e. as a ratio-0
        layer. A ratio-2 layer silently demoted to 0 is the shape of defect
        this port has already spent a night on.

        Tolerated rather than tightened because it sits on the bring-up path
        and tightening it means first establishing what ``None`` should mean
        to each caller. ⇒ Must be narrowed when either of these happens:

          * the read path for ``compress_ratios`` changes at all, or
          * ``DeepseekV41Config`` stops flattening ``text_config``.

        Both are changes someone would make deliberately, so the trigger is
        checkable rather than a reminder to be vigilant.
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
            # Through the same rank helper ori_kv uses. It was written for
            # ori_kv and only ori_kv was routed through it, so cmp_kv reached
            # the operator one axis short -- "Cmp_kv strideVec size[3] not
            # match kvLayout expect len[4]". Third parameter of this call to
            # arrive missing the same head axis; fixing them one at a time is
            # what produced the third.
            cmp_kv = kv_with_layout_rank(
                self._unwrap_cache(self._compressed_kv_cache()),
                plan.layout_kv)
            # 🔴 NOT req.block_table. `req` is the SWA cache group's metadata,
            # and the compressed cache is a different group -- for a consumer
            # layer it is another layer's cache entirely
            # (_compressed_kv_cache follows compressed_cache_prefix). Indexing
            # one cache with the other's block table gives block ids that are
            # in range and point at the wrong blocks: nothing raises, the
            # gather lands on real memory, and it surfaces as an aicore
            # exception with no frame naming the operator.
            #
            # Both the other implementations of this call keep them apart:
            #   dsa_v1:2657/2695   swa_req_metadata.block_table
            #                      vs common_metadata.block_table
            #   flashinfer_sparse:408/364  swa_metadata.block_table
            #                      vs attn_metadata.block_table
            #
            # Resolved through the same prefix the cache came from, so the
            # table and the cache cannot come from different groups.
            cmp_req = self._compressed_group_metadata(attn_metadata)
            cmp_block_table = cmp_req.block_table

        # Explicit original-KV indices, following dsa_v1's precedence: a vision
        # prefill supplies them so tokens inside an image span see the whole
        # span bidirectionally, and a layer with no compressed stream may carry
        # DSpark's instead. A layer that attends a compressed stream keeps the
        # vision ones and does not take DSpark's.
        ori_sparse_indices, ori_topk_length = ori_sparse_selection(
            attends_compressed, req)
        # 来源按【身份】判，不重算那个选择 —— 重算一遍等于给同一个问题写第二个判据，
        # 两个判据会分头漂，而日志说的来源就不再是实际用的那个。
        if ori_sparse_indices is None:
            _ori_idx_src = None
        elif ori_sparse_indices is getattr(req, "dspark_swa_indices", None):
            _ori_idx_src = "dspark_swa_indices"
        elif ori_sparse_indices is getattr(req, "vision_swa_indices", None):
            _ori_idx_src = "vision_swa_indices"
        else:
            _ori_idx_src = "【两个都不是】"
        # 同一层可能被两条 impl 各调一次 —— 现在没有任何数据能排除它。
        _log_ori_sparse_indices(swa_prefix, _ori_idx_src, ori_sparse_indices,
                                num_tokens=q.shape[0],
                                caller=f"{type(self).__module__}.{type(self).__name__}",
                                topk_length=ori_topk_length)

        _log_compressed_selection(
            prefix=swa_prefix,
            layer_ratio=self.compress_ratio,
            attends_compressed=attends_compressed,
            buffer=self.topk_indices_buffer,
            num_tokens=q.shape[0])
        _log_batch_shape(swa_prefix, req.query_start_loc, req.seq_lens)

        # The alignment taps for the three values that exist only here. Behind
        # alignment_active() so the tap functions are not entered at all when
        # no dump directory is set -- the same guarantee the boundary hooks
        # get structurally, stated at each site because these are calls.
        #
        # sa_in.* is the group that splits responsibility the first time a
        # comparison disagrees: all three right and attn.o_raw wrong points at
        # the attention operator; these wrong points upstream of it. Recording
        # them is what makes that a lookup instead of a bisection.
        from vllm_ascend.models.deepseek_v4_1 import align_taps

        if align_taps.alignment_active():
            tap_layer = align_taps.layer_index(
                self.prefix, self.vllm_config.model_config.hf_text_config)
            align_taps.tap(tap_layer, "sa_in.q", q, positions)
            align_taps.tap(tap_layer, "sa_in.kv", kv, positions)
            # No token axis: per-head and scalar. record_whole, or the
            # coverage assertion could never be satisfied for them.
            align_taps.tap_whole(tap_layer, "sa_in.attn_sink", self.attn_sink)
            align_taps.tap_whole(
                tap_layer, "sa_in.softmax_scale",
                torch.tensor([self.softmax_scale], dtype=torch.float32))
            # 🔴 sa_in.topk_idxs is deliberately NOT recorded here yet, and
            # the gap is left visible in the key difference rather than
            # filled. Measured from golden's own layer_00 dump: that key is
            # [1, 1100, 128] holding *token positions* -- row t is
            # max(0, t-127)..t, -1 padded, max value 1099 -- i.e. the causal
            # sliding window, 128 = sliding_window.
            #
            # Three things that are not what we have:
            #   * width 128 is the window, not index_topk (512);
            #   * layers 0/1 have no indexer at all
            #     (index_source_layer_ids = 2,8,14,20,24,28,32,36), so
            #     topk_indices_buffer is not their selection;
            #   * it is token space, while every index this line produces is
            #     slot space (block * storage_block_size + offset).
            #
            # We could synthesise the window list to make the comparison run.
            # That would be a tap on nothing: the operator derives the window
            # internally from mask mode 4, so the recorded tensor would be our
            # reconstruction being compared against golden's, with the thing
            # actually used by the kernel never entering the comparison.
            # Waiting on golden for what this key means on a mask-mode-4
            # layer instead.

        # Bound once, so the indices and the length describe the same object.
        # Two expressions producing "the selection" is how they come apart.
        cmp_selection = (
            self._checked_selection(q.shape[0])
            if attends_compressed and self.topk_indices_buffer is not None
            else None)

        # Bound once for the same reason: the residual has to be computed
        # against the ratio this call passes, not against a ratio computed
        # separately a few lines later.
        cmp_ratio = _dsa_swa_only_cmp_ratio(self.compress_ratio,
                                            self.vllm_config)
        cmp_residual_kv = (
            compressed_lengths(cmp_req.seq_lens, cmp_ratio)[1]
            if attends_compressed and cmp_req is not None else None)

        kwargs = build_sparse_flash_mla_kwargs(
            q=q,
            num_requests=req.query_start_loc.numel() - 1,
            swa_kv_cache=kv_with_layout_rank(
                self._unwrap_cache(self.swa_cache_layer.kv_cache),
                plan.layout_kv),
            swa_block_table=req.block_table,
            cu_seqlens_q=req.query_start_loc,
            seqused_kv=req.seq_lens,
            sinks=self.attn_sink,
            # 🔴 The compressed group's descriptor when there is one. The SWA
            # group's spec reports no compression, so its descriptor is built
            # at ratio 1 -- well-formed, and the wrong tiling for a ratio-2
            # layer. That disagreement does not get rejected; it faults every
            # core at once, which is what this call has been doing.
            metadata=(cmp_req or req).sas_metadata,
            softmax_scale=self.softmax_scale,
            cmp_ratio=cmp_ratio,
            layout_kv=plan.layout_kv,
            attends_compressed=attends_compressed,
            ori_win_left=req.ori_win_left,
            ori_win_right=req.ori_win_right,
            cmp_kv=cmp_kv,
            cmp_block_table=cmp_block_table,
            # 🔴 req.seqused_cmp_kv, not req.cu_cmp_seqlen_list. The latter is
            # cumulative and is what V4.0 sends as the operator's
            # cu_seqlens_cmp_kv under layout TND; this parameter wants one
            # value per request. Sending the cumulative one would have been
            # accepted -- same dtype, same device, plausible values, one
            # element longer -- and read as lengths.
            # Both describe the compressed stream, so both come from the
            # group that owns it -- not from the SWA group, whose ratio is 1.
            seqused_cmp_kv=(cmp_req.seqused_cmp_kv
                            if attends_compressed and cmp_req else None),
            # 🔴 Derived here, not read from req.qli_cmp_residual_k. The
            # shared builder computes this correctly for any ratio and then
            # publishes it behind ``if self.compressor_ratio == 4``
            # (dsa_v1.py) -- V4.1 carries 1 and 2 and never 4, so the field
            # arrives None on exactly the layers that need it, and the
            # operator refuses the call:
            #
            #     "Cmp_residual_kv is required when cmp_mask_mode=3 and
            #      cmp_ratio != 1"
            #     "cmp_residual_k must be provided when mask_mode=3 and
            #      cmp_ratio>1."
            #
            # The value existed one frame away and was not carried, again.
            # ``compressed_lengths`` is the same function this line's own
            # indexer already uses, so this is not a second implementation --
            # it is the third consumer of the one we had to write. Returns
            # None at ratio 1, which is what the operator requires there
            # ("Cmp_ratio is 1 or sparse_mode is 0, cmp_residual_k must be
            # null"), so the two opposite rules are both satisfied by one
            # expression rather than by a branch here.
            cmp_residual_kv=cmp_residual_kv,
            ori_sparse_indices=ori_sparse_indices,
            ori_topk_length=ori_topk_length,
            # The indexer's selection. Without it the compressed stream is
            # attended in full: nothing errors, tokens come out, and they are
            # a V4.1 that does no sparse selection. The descriptor's cmp_topk
            # is set from the same config value, in the builder, because a
            # top-k promised there and not supplied here is the same
            # disagreement that produced the aicore exception.
            # unsqueeze(1): the buffer is [tokens, index_topk] and the
            # operator takes [T, KV_N, K2] under layout_q=TND -- the KV-head
            # axis is missing, and V4.1 has num_key_value_heads = 1 so it is
            # a unit axis. Three sources agree on the rank: the sibling
            # operators' own docs (sparse_attn_sharedkv/README.md:65,
            # kv_quant_sparse_attn_sharedkv/README.md:35), the QLI operator's
            # measured output (T, num_heads_k, topk), and this same call's
            # ori_sparse_indices, which is already 3-D and is accepted.
            #
            # Upstream stores it 2-D because its own consumer reads it that
            # way; the shape the operator wants is not a property of the
            # buffer, so the axis is added here rather than by changing the
            # buffer everyone else reads.
            cmp_sparse_indices=cmp_selection,
            # 🔴 Travels with the indices, same rule as ori_topk_length. The
            # selection buffer is -1 padded; declaring the whole row valid
            # sends the kernel to gather at those slots, which is the
            # mechanism kernel proved single-variable on the ori side. Taken
            # from the indexer that wrote the rows, never recounted here.
            cmp_topk_length=(
                self._compressed_topk_length(cmp_selection)
                if cmp_selection is not None else None),
        )

        # 🔴 Here, immediately before the call, because these are caches and
        # selections: recorded after the forward they answer "what does the
        # cache hold now", which is a different question from "what did this
        # layer read". Two requests with byte-identical q/kv/scale/sink
        # produced different attn.o_raw, so the difference entered through an
        # input that was not in the dump -- and every candidate is here.
        _tap_what_this_layer_read(self, kwargs, req, cmp_req)

        output.copy_(plan.get_dsa_sparse_attn_op()(q, **kwargs)[0])

    def _checked_selection(self, num_tokens: int) -> torch.Tensor:
        """The indexer's selection for these rows, refusing unwritten ones.

        🔴 ``topk_indices_buffer`` is allocated with ``torch.empty`` upstream
        and cleared only over the rows the indexer wrote this step. Reading
        past that returns either the previous step's selection or, on the
        first step, whatever was in HBM -- and both make the forward depend on
        something other than its input, which is what a greedy decode that
        gives three different answers to the same prompt looks like.

        Range and content cannot tell the two apart: a stale row holds
        perfectly plausible non-negative block indices, and the length this
        line derives from it (``(selection >= 0).sum()``) counts them as real.
        So the check is on provenance -- how many rows were written -- not on
        the values.

        Raises rather than clamping. Clamping would silently drop the tokens
        beyond the boundary, which is a wrong answer that no longer looks
        wrong; the whole point of finding this is to see whether it happens.
        """
        from vllm_ascend.models.deepseek_v4_1.sparse_indexer import rows_written

        written = rows_written(self.topk_indices_buffer)
        if written is None:
            # ⚠️ Three states, and the first version had two. On a dummy or
            # profile run the indexer returns the buffer without writing it --
            # this line's own early return in forward_oot -- so "never
            # written" is the expected state there, not a defect. Raising on
            # it fired where the premise does not hold and, worse, stopped the
            # launch before the first real request, so neither arm of the
            # question got answered.
            if not _is_real_batch():
                _warn_unwritten_selection_on_dummy_run()
                return self.topk_indices_buffer[:num_tokens].unsqueeze(1)
            raise ValueError(
                f"{self.prefix}: this layer attends a compressed stream on a "
                "real batch and the indexer selection buffer has never been "
                "written. It is allocated with torch.empty, so every row of "
                "it is whatever was in HBM -- reading it would make the "
                "output depend on the contents of device memory rather than "
                "on the prompt.")
        if num_tokens > written:
            raise ValueError(
                f"{self.prefix}: reading {num_tokens} rows of the indexer "
                f"selection but only {written} were written this step. The "
                f"remaining {num_tokens - written} hold the previous step's "
                "selection -- plausible non-negative indices that "
                "_compressed_topk_length would count as valid, pointing into "
                "blocks that belong to another request.")
        return self.topk_indices_buffer[:num_tokens].unsqueeze(1)

    @staticmethod
    def _compressed_topk_length(selection: torch.Tensor) -> torch.Tensor:
        """Valid entries per row of the selection **being sent**.

        Counted from that exact tensor, which is the one case where counting
        at the consumer is not a second source: it cannot disagree with the
        table it describes, because it *is* that table. The rows are -1 padded
        from the right, so the count of non-negative entries is the number of
        leading valid ones -- which is what the operator's length means.

        🔴 The first version published a count from the indexer instead, on the
        principle that the writer should own it. That principle is right and
        the scope was wrong: ``topk_indices_buffer`` is one buffer shared by
        every layer, while ``self.indexer`` is None on every layer that is not
        an index source. So consumers read shared indices and found no
        length, and the guard fired on a call that was otherwise fine. A value
        and its companion have to live at the same scope or they drift apart
        by construction.
        """
        return (selection >= 0).sum(dim=-1, dtype=torch.int32)

    def _compressed_group_metadata(self, attn_metadata: dict):
        """The metadata of the group that owns this layer's compressed cache.

        🔴 Not the SWA group's. Three of this call's arguments describe the
        compressed stream -- the block table that addresses it, the lengths
        that bound it, and the descriptor the kernel takes its tiling from --
        and all three belong to whichever group holds that cache, which for a
        consumer layer is another layer entirely (``_compressed_kv_cache``
        follows ``compressed_cache_prefix``).

        V4.0 keeps the two apart and this is the shape to copy
        (``dsa_v1.py:2574-2580``)::

            common_attn_metadata = layer_metadata.attention or layer_metadata.swa
            swa_metadata         = layer_metadata.swa
            ...
            ori_block_table = swa_req_metadata.block_table
            metadata        = common_metadata.sas_metadata

        Taking everything from the SWA group is not a shape error and nothing
        downstream can see it: the block ids are in range for both pools, and
        the descriptor is a well-formed descriptor -- for the wrong tiling.
        The SWA group's spec reports no compression, so its descriptor is
        built at ratio 1; handing that to a ratio-2 layer is the
        descriptor/call disagreement that opened this port, and it surfaces
        as an aicore exception with no frame naming the operator.

        Resolved through the same prefix that selects the cache, so cache,
        table, lengths and descriptor cannot come from different groups.
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

        # The two attention-output keys, either side of the one step between
        # them. Recorded here rather than where the operator returns, because
        # "before and after this rotation" is what distinguishes them and only
        # this function sees both.
        #
        # Sliced to n_local_heads: `o` is [tokens, padded_heads, head_dim] and
        # golden's is 64 heads. The padding rows are zeros, so including them
        # would not change any value -- it would change the shape, and a
        # shape mismatch is what the format check is there to catch. Slicing
        # is the same fact stated where it can be checked.
        from vllm_ascend.models.deepseek_v4_1 import align_taps

        if align_taps.alignment_active():
            tap_layer = align_taps.layer_index(
                self.prefix, self.vllm_config.model_config.hf_text_config)
            align_taps.tap(tap_layer, "attn.o_raw",
                           o[:, :self.n_local_heads], positions)

        o = inverse_rope_from_cos_sin_cache(o, positions, cos_sin_cache,
                                            self.rope_head_dim)

        if align_taps.alignment_active():
            align_taps.tap(tap_layer, "attn.o_unroped",
                           o[:, :self.n_local_heads], positions)

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
