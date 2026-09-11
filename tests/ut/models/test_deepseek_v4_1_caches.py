# SPDX-License-Identifier: Apache-2.0
"""The two V4.1 cache layers Ascend has to replace, and why each.

Upstream parameterised one axis of each and not the other, in opposite
directions: the SWA cache takes its backend as a constructor argument but
builds its kv-cache spec inline, and the compressor's state cache has a
perfectly good spec but a backend with no get_kv_cache_shape. So there are two
subclasses overriding one method each, rather than one replacement for both.

Both failures land during KV cache setup -- before any forward pass -- which is
why these come before the attention work rather than after it.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.models.deepseek_v4_1 import caches, platform_patches

HEAD_DIM, WINDOW, BLOCK = 512, 2048, 32


def _swa_cache(cache_dtype="auto", dtype=torch.bfloat16):
    """A DeepseekV4SWACache without running __init__ (it registers a prefix)."""
    cache = object.__new__(caches.AscendDeepseekV41SWACache)
    torch.nn.Module.__init__(cache)
    cache.head_dim = HEAD_DIM
    cache.window_size = WINDOW
    cache.block_size = BLOCK
    cache.dtype = dtype
    cache.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    cache.backend_cls = object
    return cache


# -- the SWA cache: spec only ------------------------------------------------


def test_the_spec_is_the_ascend_one():
    """The reason this subclass exists.

    Ascend registers AscendSlidingWindowMLASpec with the sliding-window
    manager and its merge() asserts every spec in a group is that class;
    upstream's inline spec is the plain one.
    """
    spec = _swa_cache().get_kv_cache_spec(None)
    assert isinstance(spec, AscendSlidingWindowMLASpec)
    assert spec.block_size == BLOCK
    assert spec.head_size == HEAD_DIM
    assert spec.sliding_window == WINDOW
    assert spec.num_kv_heads == 1


def test_the_constructed_dtype_is_carried_through_untouched():
    """V4.1 resolves the kv dtype once, in the attention layer, and passes it
    in. V4.0's Ascend cache re-resolves it here and writes back into
    vllm_config.cache_config as a side effect; reproducing that would let a
    second resolution disagree with the first."""
    for dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8):
        cache = _swa_cache(dtype=dtype)
        assert cache.get_kv_cache_spec(None).dtype is dtype
        assert cache.dtype is dtype    # and the call did not mutate the layer


def test_the_fp8_ds_mla_layout_reports_its_packed_row():
    plain = _swa_cache(cache_dtype="auto").get_kv_cache_spec(None)
    packed = _swa_cache(cache_dtype="fp8_ds_mla").get_kv_cache_spec(None)
    # 448B NoPE + 128B RoPE + 8B scales.
    assert packed.state_content_bytes == 584
    assert plain.state_content_bytes is None


def test_the_backend_is_not_overridden():
    """Upstream already returns self.backend_cls, and the V4.1 attention layer
    passes the Ascend backend into it. Overriding would replace a correct value
    with the same one -- and would then be a second place to keep in step."""
    assert "get_attn_backend" not in vars(caches.AscendDeepseekV41SWACache)
    sentinel = type("Sentinel", (), {})
    cache = _swa_cache()
    cache.backend_cls = sentinel
    assert cache.get_attn_backend() is sentinel


# -- the compressor state cache: backend only --------------------------------


def test_upstream_compressor_backend_cannot_describe_its_cache_shape():
    """Pins the premise. If upstream ever adds get_kv_cache_shape to
    CompressorBackend, this subclass stops being necessary and this goes red
    rather than the fix quietly becoming redundant."""
    from vllm.models.deepseek_v4_1.compressor import CompressorBackend

    assert not hasattr(CompressorBackend, "get_kv_cache_shape")


def test_the_ascend_state_backend_can():
    backend = caches.AscendCompressorStateBackend
    assert hasattr(backend, "get_kv_cache_shape")
    assert backend.get_kv_cache_shape(7, 8, 1, 1024) == (7, 8, 1, 1024)
    assert backend.get_name() == "ASCEND_DSV41_COMPRESSOR_STATE"


@pytest.mark.parametrize("num_speculative_tokens,expected_ring", [
    (0, 8), (1, 8), (2, 8), (3, 8), (7, 16),
])
def test_every_ring_capacity_is_a_supported_block_size(num_speculative_tokens,
                                                       expected_ring):
    """The ring capacity upstream computes must be one the backend accepts.

    Reproduces upstream's formula rather than asserting a constant, so a change
    to it shows up here instead of as a wrong cache shape. The base backend
    allows [2, 4, 8, 16, 32, 64, 128], which is why the subclass deliberately
    does not narrow the list.
    """
    rows_per_step = num_speculative_tokens + 2
    ring = max(8, 1 << (rows_per_step - 1).bit_length())
    assert ring == expected_ring
    supported = caches.AscendCompressorStateBackend.get_supported_kernel_block_sizes()
    assert ring in supported


def test_the_spec_is_left_as_a_circular_buffer():
    """Not remapped onto a sliding window.

    V4.0's Ascend state cache does map onto AscendSlidingWindowMLASpec, which
    is honest there because V4.0's upstream state cache really is a sliding
    window. V4.1's has no sliding_window attribute at all, so the same move
    would require inventing a window length -- a made-up parameter that no
    test would catch.
    """
    from vllm.models.deepseek_v4_1.compressor import CompressorStateCache

    assert "get_kv_cache_spec" not in vars(caches.AscendDeepseekV41CompressorStateCache)
    assert "get_attn_backend" in vars(caches.AscendDeepseekV41CompressorStateCache)
    # and upstream's really has no window to copy
    assert not hasattr(CompressorStateCache, "sliding_window")


# -- the rebinds -------------------------------------------------------------


def test_the_construction_sites_see_the_ascend_classes():
    """Both are built inline by bare name inside an __init__, so subclassing
    the owner cannot reach them."""
    import vllm.models.deepseek_v4_1.attention as attention_mod
    import vllm.models.deepseek_v4_1.compressor as compressor_mod

    platform_patches.rebind_cache_classes()
    assert attention_mod.DeepseekV4SWACache is caches.AscendDeepseekV41SWACache
    assert (compressor_mod.CompressorStateCache
            is caches.AscendDeepseekV41CompressorStateCache)


def test_rebinding_is_idempotent():
    platform_patches.rebind_cache_classes()
    assert platform_patches.rebind_cache_classes() == []


def test_a_vanished_name_is_refused_rather_than_skipped(monkeypatch):
    import vllm.models.deepseek_v4_1.attention as attention_mod

    monkeypatch.delattr(attention_mod, "DeepseekV4SWACache")
    with pytest.raises(AttributeError, match="no longer defines"):
        platform_patches.rebind_cache_classes()


def test_apply_all_reports_both_rebinds():
    from vllm_ascend.models import deepseek_v4_1

    assert any("DeepseekV4SWACache" in name
               for name in deepseek_v4_1.APPLIED_PATCHES)
    assert any("CompressorStateCache" in name
               for name in deepseek_v4_1.APPLIED_PATCHES)
