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
from vllm.models.deepseek_v4_1.compressor import CompressorStateCache
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.attention.dsa_v1 import AscendDSABackend
from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
    AscendStagedEngramEmbedding


class AscendCompressorStateBackend(AscendDSABackend):
    """Backend for the compressor's ring buffer.

    It exists for one inherited method. Upstream's ``CompressorBackend``
    defines ``get_name`` / ``get_supported_kernel_block_sizes`` /
    ``get_supported_head_sizes`` / ``get_builder_cls`` and no
    ``get_kv_cache_shape`` -- and neither does ``AttentionBackend``, which only
    concrete backends implement. The Ascend runner's generic attention branch
    calls it while reshaping, so on this platform the state cache would fail
    with an AttributeError during KV cache setup, before any forward pass.

    Deliberately no ``get_supported_kernel_block_sizes`` override: the base
    already allows [2, 4, 8, 16, 32, 64, 128], which covers every ring capacity
    the spec can ask for (8 with no speculation, 16 at seven draft tokens).
    Narrowing it here would be asserting something about the kernel that has
    not been measured.
    """

    @staticmethod
    def get_name() -> str:
        return "ASCEND_DSV41_COMPRESSOR_STATE"


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

    This is the class that decides whether the model loads at all. Upstream's
    constructor allocates the whole shard before anything else gets a say, and
    at TP=1 the hash-column sharding degenerates so each rank wants a full
    copy: 188.83 GiB per rank, 755.3 GiB across four. The first real launch
    died exactly there.

    How it died is the part worth keeping: ``rtsMallocHost ... out of memory``.
    The banned pinned-host path was stopped by the host being too small, not
    by any rule of ours. A larger machine would have let it through and then
    run the registration that took an NPU down. So the interception here is at
    construction, and it raises.
    """


class AscendDeepseekV41IndexerCache(DeepseekV4IndexerCache):
    """The indexer's K cache, reported the way the Ascend runner needs it.

    Upstream returns a plain ``MLAAttentionSpec`` and upstream's indexer
    backend; neither reaches the Ascend path. Both are replaced here, following
    the V4.0 Ascend cache, which is the same cache under a model that already
    runs on this hardware.

    The block size is the physical page multiplied by the compression ratio,
    and that product has to appear verbatim in the backend's supported list.
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
        return AscendMLAAttentionSpec(
            block_size=storage_block_size * self.compress_ratio,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            model_version="deepseek_v4",
            cache_dtype_str=self.cache_config.cache_dtype,
            scale_dim=1 if self.head_dim == 128 else 0,
            scale_dtype=torch.float if compressed else torch.float16,
            tokens_per_state=self.compress_ratio,
        )

    def get_attn_backend(self):
        """Pick the backend whose supported block sizes contain this layer's.

        V4.1's index-source layers carry ratios 1 and 2 only. Ratio 1 needs
        [.., 128] and the SWA backend already lists it; ratio 2 needs 256 and
        no existing class covers [64, 128, 256], so it refuses rather than
        borrowing the nearest one.
        """
        from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend

        if self.compress_ratio <= 1:
            return AscendDSASWABackend
        if self.compress_ratio == 2:
            backend = _optional_c2_backend()
            if backend is not None:
                return backend
            raise NotImplementedError(
                "no Ascend backend covers indexer compress_ratio 2, which "
                "needs supported kernel block sizes [64, 128, 256]. Do not "
                "reuse AscendDSAC4Backend: it accepts this layer at cache "
                "block sizes 64 and 128 and only refuses at 32, so the "
                "default configuration would run with the planner and the "
                "operator disagreeing about the page size, silently.")
        raise NotImplementedError(
            f"indexer compress_ratio {self.compress_ratio} has no Ascend "
            "backend; V4.1's index-source layers carry 1 and 2.")


def _optional_c2_backend():
    """The ratio-2 backend, if it has landed yet."""
    try:
        from vllm_ascend.attention.dsa_v1 import AscendDSAC2Backend
    except ImportError:
        return None
    return AscendDSAC2Backend
