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


class AscendCompressorStateBackend(CompressorBackend):
    """Upstream's compressor backend, plus the one method Ascend needs.

    🔴 这个类【曾经】继承 ``AscendDSABackend``，docstring 开头写的是"它只为一个
    继承来的方法而存在"（``get_kv_cache_shape``）。那句话是错的，而且错在一个
    **grep 看不见**的地方：它还继承了 ``get_builder_cls``，于是环组被塞了一个
    ``AscendDSAMetadataBuilder``，而那个 builder 第一件事就是读
    ``kv_cache_spec.storage_block_size`` —— ``CircularBufferSpec`` 没有这个属性。
    2026-09-11 第十六次起飞就死在
    ``AttributeError: 'CircularBufferSpec' object has no attribute 'storage_block_size'``。
    环根本不做 attention：它的读写是 compressor 自己的 Triton kernel，
    上游为它配的是 ``CompressorMetadataBuilder``（算环槽位映射）。
    **为了借一个方法去继承 DSA backend，把那个类的其余部分一起带了过来。**

    ⇒ 正确的基类是上游的 ``CompressorBackend``：它定义了 ``get_name`` /
    ``get_supported_kernel_block_sizes`` / ``get_supported_head_sizes`` /
    ``get_builder_cls``，唯独没有 ``get_kv_cache_shape`` —— ``AttentionBackend``
    也没有，那是只有具体 backend 才实现的方法。而昇腾 runner 的通用 attention
    分支在 reshape 时会调它，所以在本平台上，状态 cache 会在 KV cache 建立阶段、
    任何一次前向之前就以 AttributeError 失败。⇒ 这里只补那一个方法。

    ⚠️ ``get_kv_cache_shape`` 的维序必须是 ``(B, N, H, C)``，头维在 **-2**，
    和主 MLA / indexer K 同一个约定。

    🔴 我一度把它改成 ``(B, H, N, C)``，理由是上游
    ``CompressorStateCache.bind_kv_cache`` 做 ``kv_cache.squeeze(1)``、
    注释写 ``[B, H=1, N, C] -> [B, N, C]``。**那个理由整条不成立**：
    昇腾这条路根本不调用每层的 ``bind_kv_cache`` ——
    ``model_runner_v1.py:4292-4313`` 对 V4 家族是**直接赋**
    ``static_forward_context[name].kv_cache = bind_value(kv_cache)``。
    上游那三处 ``squeeze(1)`` 在本平台是死代码，改声明去迎合它不生效。
    而真正读这个形状的是 ``deepseek_v4_1/kv_binding.py`` 的 ``bind_value``，
    它按 ``HEAD_AXIS = -2`` 找头维：``(B, H, N, C)`` 下 ``shape[-2]`` 是环容量 8，
    不是 1 ⇒ 它认不出这个布局，**退回 list 包装**，而
    ``compressor.py:268`` 直接把它喂给 kernel，期待三维 tensor。
    ⇒ 症状是下一帧的类型错，不是这里的形状错。

    ⚪ 这条是 dsv41-model 用实跑数据纠正的，不是推理：
       ``bind_value((B,H=1,N=8,C)) -> list``；``bind_value((B,N=8,H=1,C)) -> Tensor``。
    ⚪ 值得记住的是**为什么我的判据拦不住**：H=1 时两种维序的字节和 stride
       完全相同，没有任何一条形状断言能区分它们 ——
       能区分的只有"谁在读这个形状、他从哪一轴找头"。
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
        # [B, N, H, C]：头维在 -2，bind_value 就是从那一轴找它的。
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
        # max(ratio, 1): ratio 0 means "no compression", i.e. one state per
        # token, not zero states per page.
        #
        # This is the only interception point, not a tidiness fix. A block size
        # of 0 does not fail: select_common_block_size looks for a supported
        # size that divides the request, and `0 % n == 0` holds for every n, so
        # the first candidate -- the largest -- matches. Measured: 0 resolves
        # to 256 on C2, 512 on C4, 128 on the base backend. The planner then
        # allocates that page. Nothing raises; the only symptom is a page size
        # nobody asked for.
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
                f"AscendDSAC2Backend is not importable from "
                f"vllm_ascend.attention.dsa_v1 in this tree ({_c2_absence()}). "
                "compress_ratio 2 needs supported kernel block sizes "
                "[64, 128, 256] and no other class covers all three. Do not "
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


def _c2_absence() -> str:
    """Why it is missing, in the two words that decide who fixes it.

    "Unsupported" and "not merged into the tree you are running" are the same
    ImportError and want opposite responses. Saying only the first cost a
    launch: the class existed, on a branch, and the message sent everyone
    looking for work that was already done.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "log", "--all", "--oneline", "-1", "-S",
             "class AscendDSAC2Backend", "--", "vllm_ascend/attention/dsa_v1.py"],
            capture_output=True, text=True, timeout=10)
    except Exception:  # git absent, not a checkout, anything
        return "could not check whether it exists elsewhere"
    found = out.stdout.strip()
    if not found:
        return "no commit in this repository ever added it"
    return (f"it exists on another branch -- {found} -- and is not merged "
            "into the tree you are running")
