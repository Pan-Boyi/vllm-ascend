# SPDX-License-Identifier: Apache-2.0
"""Ascend replacements for the two V4.1 cache layers that upstream hard-codes.

Upstream parameterised one axis of these and not the other. ``DeepseekV4SWACache``
takes its attention backend as a constructor argument, and the V4.1 attention
layer passes ``backend_cls=self.swa_backend_cls`` -- so the backend is already
the Ascend one and needs no patch. What it does *not* parameterise is the
kv-cache spec, which is built inline and returns upstream's class. The
compressor's state cache is the mirror image: its spec is fine, its backend is
a bare ``AttentionBackend`` subclass with no ``get_kv_cache_shape`` at all.

Hence two small subclasses that each override exactly one method, rather than
one replacement for both.
"""

from __future__ import annotations

import torch

from vllm.config import VllmConfig
from vllm.models.deepseek_v4_1.common.engram import ParallelEngramEmbedding
from vllm.models.deepseek_v4_1.attention import DeepseekV4IndexerCache
from vllm.models.deepseek_v4_1.compressor import (CompressorBackend,
                                                  CompressorStateCache)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
    AscendStagedEngramEmbedding


def stated_stream_ratio(cache) -> int | None:
    """This cache's layer stream ratio to report in its spec, or ``None``.

    ``None`` when the switch is off, which leaves the spec byte-identical to
    what it was and the builder on its old fallback chain -- so an A/B
    attributes a launch to this change and nothing else.

    When the switch is on and no ratio was attached, this raises. A silent
    ``None`` there would reproduce exactly the defect the switch claims to
    fix, under a flag that says it is fixed, which is worse than the defect.
    """
    from vllm_ascend import envs

    if envs.VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO != "1":
        return None
    ratio = getattr(cache, "stream_compress_ratio", None)
    if ratio is None:
        # On a cache object -- not on a spec -- ``compress_ratio`` is the
        # layer's configured compress_ratios entry, unclamped, which is the
        # quantity wanted here. The indexer cache carries it; the SWA cache
        # has no such attribute, which is why the attention layer attaches
        # one. Only the *spec* fields of that name are page divisors.
        ratio = getattr(cache, "compress_ratio", None)
    if ratio is None:
        raise ValueError(
            f"{type(cache).__name__} was asked for its compressed-stream "
            "ratio with VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO=1 and carries "
            "none. The V4.1 attention layer attaches it after the base "
            "constructor builds the cache; a cache built anywhere else has "
            "to do the same, because falling back here would report ratio 0 "
            "layers as ratio 1 again.")
    return int(ratio)


class AscendCompressorStateBackend(CompressorBackend):
    """Upstream's compressor backend, plus ``get_kv_cache_shape``.

    Based on ``CompressorBackend`` rather than a DSA backend so the ring keeps
    upstream's ``CompressorMetadataBuilder``; a DSA builder would read
    ``storage_block_size``, which ``CircularBufferSpec`` does not have. The
    Ascend runner's reshape path calls ``get_kv_cache_shape``, which
    ``CompressorBackend`` lacks.

    The shape is ``(B, N, H, C)`` with the head axis at -2, as for the main
    MLA and indexer caches: ``kv_binding.bind_value`` locates the head axis
    there, and the Ascend runner binds V4 caches through it rather than
    through upstream's per-layer ``bind_kv_cache``.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSV41_COMPRESSOR_STATE"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        # Head axis at -2, where bind_value looks for it.
        return num_blocks, block_size, num_kv_heads, head_size


class AscendDeepseekV41CompressorStateCache(CompressorStateCache):
    """The ring buffer, with a backend that can describe its cache shape.

    The spec itself is left alone. Upstream returns a ``CircularBufferSpec`` --
    one block per request, not prefix-cacheable -- and that is a real
    description of a ring, not an approximation of something else. It is also
    an ``AttentionSpec`` subclass, so the Ascend reshape path accepts it.

    Note what is *not* done here: V4.0's Ascend equivalent maps its state cache
    onto ``AscendSlidingWindowMLASpec``, which works there because V4.0's
    upstream state cache genuinely is a sliding window
    (``sliding_window = coff * compress_ratio``). V4.1's has no sliding window
    at all. Re-expressing a per-request ring as a window would mean inventing a
    window length, and a made-up parameter of that kind is invisible in tests.
    """

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AscendCompressorStateBackend


class AscendDeepseekV41SWACache(DeepseekV4SWACache):
    """The sliding-window cache, reporting an Ascend kv-cache spec.

    Only ``get_kv_cache_spec`` changes. ``get_attn_backend`` upstream already
    returns ``self.backend_cls``, which the V4.1 attention layer supplies from
    ``swa_backend_cls`` -- the Ascend SWA backend -- so overriding it would
    replace a correct value with the same one.

    The dtype is left exactly as constructed. V4.1 resolves it once in the
    attention layer (``_resolve_dsv4_kv_cache_dtype``) and hands it in; V4.0's
    Ascend version re-resolves it here and writes back into
    ``vllm_config.cache_config.cache_dtype`` as a side effect, which there is
    no reason to reproduce for a model that already did the resolution.
    """

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        del vllm_config  # every field comes from construction, as upstream
        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            model_version="deepseek_v4",
            stream_compress_ratio=stated_stream_ratio(self),
            # None, following the Ascend V4.0 path, which runs the same SWA
            # backend and does its own page arithmetic -- not upstream's
            # 576/512, which pad for FlashMLA and FlashInfer packing.
            alignment=None,
            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
        )


class AscendEngramEmbedding(AscendStagedEngramEmbedding,
                            ParallelEngramEmbedding):
    """Upstream's engram table, never allocated and never registered.

    The mixin comes first so its ``__init__`` and ``_storage`` win the MRO.
    See ``engram_embedding`` for why the table is not allocated.
    """


class AscendDeepseekV41IndexerCache(DeepseekV4IndexerCache):
    """The indexer's K cache, reported the way the Ascend runner needs it.

    Upstream returns a plain ``MLAAttentionSpec`` and upstream's indexer
    backend; neither reaches the Ascend path. Both are replaced here, following
    the V4.0 Ascend cache, which is the same cache under a model that already
    runs on this hardware.

    The block size is the physical page multiplied by the compression ratio.
    That is not a preference: ``AscendMLAAttentionSpec.__post_init__`` derives
    ``storage_block_size = block_size // tokens_per_state``, so it recovers the
    physical page by dividing the ratio back out. Passing upstream's
    convention -- ``block_size = cache_config.block_size``, unmultiplied --
    would make the derived physical page ``B // ratio``, smaller than the real
    one, and ``real_page_size_bytes`` would then under-allocate.

    The product has to appear verbatim in the backend's supported list.
    Not "be divisible by": ``select_common_block_size`` silently falls back to
    a smaller supported size that divides it, which leaves the planner and the
    operator working from different page sizes with nothing raised. The
    measured case is ratio 2 on the C4 backend -- it passes at cache block
    sizes 64 and 128 and only fails at 32, and 128 is the default.
    """

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        from vllm_ascend.attention.dsa_attn_kv_plan import is_a5_bf16_kv_enabled
        from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
        from vllm_ascend.device.hardware_profile import (
            HardwareCapability, get_current_hardware_profile)
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        compressed = get_current_hardware_profile().supports(
            HardwareCapability.DSV4_COMPRESSED_CACHE)
        if compressed:
            self.dtype = torch.float8_e4m3fn
            if not is_a5_bf16_kv_enabled(vllm_config):
                vllm_config.cache_config.cache_dtype = "float8_e4m3fn"

        storage_block_size = DSV4_BLOCK_SIZES[
            vllm_config.cache_config.block_size][0][0]
        # Ratio 0 means one state per token. A block size of 0 would not
        # fail: select_common_block_size treats 0 as divisible by every
        # supported size and silently picks the largest.
        states_ratio = max(self.compress_ratio, 1)
        return AscendMLAAttentionSpec(
            block_size=storage_block_size * states_ratio,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            model_version="deepseek_v4",
            cache_dtype_str=self.cache_config.cache_dtype,
            scale_dim=1 if self.head_dim == 128 else 0,
            scale_dtype=torch.float if compressed else torch.float16,
            tokens_per_state=states_ratio,
            # states_ratio is the clamped page divisor; this is the ratio the
            # layer was configured with, 0 kept as 0.
            stream_compress_ratio=stated_stream_ratio(self),
        )

    def get_attn_backend(self):
        """Pick the backend whose supported block sizes contain this layer's.

        V4.1's index-source layers carry ratios 1 and 2 only. Ratio 1 needs
        [.., 128], which the SWA backend lists; ratio 2 needs [64, 128, 256],
        which only the C2 backend covers.
        """
        from vllm_ascend.attention.dsa_v1 import (AscendDSAC2Backend,
                                                  AscendDSASWABackend)

        if self.compress_ratio <= 1:
            return AscendDSASWABackend
        if self.compress_ratio == 2:
            # Not C4: it accepts ratio 2 at cache block sizes 64 and 128 and
            # only refuses at 32, leaving planner and operator disagreeing
            # about the page size.
            return AscendDSAC2Backend
        raise NotImplementedError(
            f"indexer compress_ratio {self.compress_ratio} has no Ascend "
            "backend; V4.1's index-source layers carry 1 and 2.")

