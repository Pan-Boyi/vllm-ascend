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

    platform_patches.rebind_upstream_classes()
    assert attention_mod.DeepseekV4SWACache is caches.AscendDeepseekV41SWACache
    assert (compressor_mod.CompressorStateCache
            is caches.AscendDeepseekV41CompressorStateCache)


def test_rebinding_is_idempotent():
    platform_patches.rebind_upstream_classes()
    assert platform_patches.rebind_upstream_classes() == []


def test_a_vanished_name_is_refused_rather_than_skipped(monkeypatch):
    import vllm.models.deepseek_v4_1.attention as attention_mod

    monkeypatch.delattr(attention_mod, "DeepseekV4SWACache")
    with pytest.raises(AttributeError, match="no longer defines"):
        platform_patches.rebind_upstream_classes()


def test_apply_all_reports_both_rebinds():
    from vllm_ascend.models import deepseek_v4_1

    assert any("DeepseekV4SWACache" in name
               for name in deepseek_v4_1.APPLIED_PATCHES)
    assert any("CompressorStateCache" in name
               for name in deepseek_v4_1.APPLIED_PATCHES)


# -- the indexer's K cache ---------------------------------------------------


def _indexer_cache(compress_ratio, head_dim=128, cache_dtype="auto"):
    cache = object.__new__(caches.AscendDeepseekV41IndexerCache)
    torch.nn.Module.__init__(cache)
    cache.head_dim = head_dim
    cache.compress_ratio = compress_ratio
    cache.dtype = torch.bfloat16
    cache.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
    return cache


class _VllmConfigStub:
    def __init__(self, block_size=128, cache_dtype="auto"):
        self.cache_config = SimpleNamespace(block_size=block_size,
                                            cache_dtype=cache_dtype)


def test_the_indexer_spec_block_size_is_the_page_times_the_ratio():
    """And it must appear verbatim in the backend's supported list.

    Not "divide into": select_common_block_size silently falls back to a
    smaller supported size that divides the request, which leaves the planner
    and the operator using different page sizes with nothing raised.
    """
    from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
    from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

    page = DSV4_BLOCK_SIZES[128][0][0]
    for ratio in (1, 2):
        spec = _indexer_cache(ratio).get_kv_cache_spec(_VllmConfigStub(128))
        assert isinstance(spec, AscendMLAAttentionSpec)
        assert spec.block_size == page * ratio
        assert spec.num_kv_heads == 1
        assert spec.tokens_per_state == ratio


def test_ratio_one_uses_a_backend_that_lists_its_block_size():
    from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend
    from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

    backend = _indexer_cache(1).get_attn_backend()
    assert backend is AscendDSASWABackend
    wanted = DSV4_BLOCK_SIZES[128][0][0] * 1
    assert wanted in backend.get_supported_kernel_block_sizes(), (
        f"{wanted} must appear verbatim, not merely divide something listed")


def test_what_borrowing_c4_for_ratio_two_would_look_like():
    """The shape of the wrong shortcut, pinned independently of the right one.

    Ratio 2 on the C4 backend passes at cache block sizes 64 and 128 and only
    fails at 32 -- and 128 is the default, so the configuration everyone runs
    is the one that would not catch it.

    This used to be checked by asserting that ratio 2 was refused, which stopped
    being true the moment AscendDSAC2Backend was merged: the test was written
    while the backend was missing and pinned that moment rather than the hazard.
    The hazard did not go away with it. Someone deciding that C2 is more trouble
    than it is worth would reintroduce exactly this, and at the default block
    size it would look like it worked.
    """
    from vllm_ascend.attention.dsa_v1 import AscendDSAC4Backend

    supported = AscendDSAC4Backend.get_supported_kernel_block_sizes()
    assert 128 * 2 in supported, "this is why borrowing C4 looks fine"
    assert 32 * 2 not in supported, "and this is the case that breaks it"


def test_ratio_two_routes_to_the_c2_backend():
    """The positive half: ratio 2 goes to the class built for it.

    If C2 stops being importable -- unmerged again, renamed, moved -- this goes
    red here rather than at KV cache setup on a card.
    """
    from vllm_ascend.attention.dsa_v1 import AscendDSAC2Backend

    assert _indexer_cache(2).get_attn_backend() is AscendDSAC2Backend
    # And it covers the whole set the spec can ask for, which is the thing C4
    # did not.
    supported = AscendDSAC2Backend.get_supported_kernel_block_sizes()
    for cache_block_size in (32, 64, 128):
        assert cache_block_size * 2 in supported, (
            f"cache block size {cache_block_size} gives a spec block size of "
            f"{cache_block_size * 2}, which C2 must accept")


def test_the_refusal_still_names_the_shortcut_when_c2_is_gone(monkeypatch):
    """The message is the only thing that survives C2 going missing.

    Kept as a separate test because it is about the text, not the routing: the
    reason not to borrow C4 has to be readable by whoever hits the refusal, and
    that person by definition does not have C2 to look at.
    """
    from vllm_ascend.models.deepseek_v4_1 import caches as mod

    monkeypatch.setattr(mod, "_optional_c2_backend", lambda: None)
    with pytest.raises(NotImplementedError) as exc:
        _indexer_cache(2).get_attn_backend()
    msg = str(exc.value)
    assert "[64, 128, 256]" in msg
    assert "AscendDSAC4Backend" in msg
    assert "only refuses at 32" in msg


def test_the_refusal_says_whether_the_backend_merely_is_unmerged(monkeypatch):
    """Two ImportErrors, opposite responses, one message -- that cost a launch.

    AscendDSAC2Backend had been written and committed on another branch. The
    tree that was launched did not have it, the import failed, and the refusal
    said "no Ascend backend covers ratio 2" -- which reads as "nobody has built
    this" and sent three people looking for work that was already done. The
    message now has to name which of the two it is.
    """
    from vllm_ascend.models.deepseek_v4_1 import caches as mod

    monkeypatch.setattr(mod, "_optional_c2_backend", lambda: None)
    monkeypatch.setattr(mod, "_c2_absence",
                        lambda: "it exists on another branch -- abc123 -- and "
                        "is not merged into the tree you are running")
    with pytest.raises(NotImplementedError) as exc:
        _indexer_cache(2).get_attn_backend()
    msg = str(exc.value)
    assert "not merged into the tree you are running" in msg
    assert "abc123" in msg
    # And the hazard is still spelled out: the reason not to borrow C4 does not
    # depend on why C2 is missing.
    assert "only refuses at 32" in msg


def test_the_absence_check_distinguishes_never_written_from_unmerged():
    """Both arms, against the real repository.

    A probe that cannot tell the two apart would still return a string, and the
    message would still look specific.
    """
    from vllm_ascend.models.deepseek_v4_1.caches import _c2_absence

    said = _c2_absence()
    assert said, "the probe must say something; a blank reads as 'no reason'"
    # Whichever arm this tree is in, it must commit to one of them rather than
    # hedging -- the point of the string is to route the fix.
    assert ("not merged into the tree you are running" in said
            or "no commit in this repository ever added it" in said
            or "could not check" in said)


def test_an_unexpected_ratio_is_named():
    with pytest.raises(NotImplementedError, match="carry 1 and 2"):
        _indexer_cache(4).get_attn_backend()


def test_the_indexer_cache_construction_site_is_rebound():
    import vllm.models.deepseek_v4_1.attention as attention_mod

    platform_patches.rebind_upstream_classes()
    assert (attention_mod.DeepseekV4IndexerCache
            is caches.AscendDeepseekV41IndexerCache)


def test_ratio_zero_is_one_state_per_token_not_zero_states_per_page():
    """max(ratio, 1). Multiplying by 0 gives a block size of 0, which no
    backend can list and which the spec would then divide by."""
    from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

    page = DSV4_BLOCK_SIZES[128][0][0]
    spec = _indexer_cache(0).get_kv_cache_spec(_VllmConfigStub(128))
    assert spec.block_size == page
    assert spec.tokens_per_state == 1
    assert spec.storage_block_size == page, (
        "the spec divides block_size by tokens_per_state to recover the "
        "physical page; both must come back to the page for ratio 0")


def test_the_spec_recovers_the_physical_page_for_every_ratio():
    """Why block_size is multiplied at all: AscendMLAAttentionSpec derives
    storage_block_size = block_size // tokens_per_state. Upstream's
    unmultiplied convention would make the physical page B // ratio."""
    from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

    page = DSV4_BLOCK_SIZES[128][0][0]
    for ratio in (0, 1, 2):
        spec = _indexer_cache(ratio).get_kv_cache_spec(_VllmConfigStub(128))
        assert spec.storage_block_size == page, (
            f"ratio {ratio}: physical page came back as "
            f"{spec.storage_block_size}, not {page}")
