# SPDX-License-Identifier: Apache-2.0
"""Ratio 0 and ratio 1 have to stay apart in the spec.

V4.1 uses both, and they mean opposite things about the compressed stream: 0
is "pure sliding window, no compressed stream", 1 is "compressed, pooling
disabled". Every ratio already on these specs is a divisor of the page size,
so none of them can hold 0 -- ``max(ratio, 1)`` and ``block_size // ratio``
are not tidiness, they are what a page divisor must be. The two meanings were
therefore collapsed on the way into the spec and read back as 1, and a ratio-0
layer's descriptor was built as though it attended a compressed stream while
the call supplied none.

So the check is not "is the arithmetic right" but "are the two values still
distinguishable after a round trip through a spec".
"""

import pytest
import torch

from vllm_ascend.attention.dsa_v1 import attends_compressed_stream
from vllm_ascend.core.kv_cache_interface import (AscendMLAAttentionSpec,
                                                 AscendSlidingWindowMLASpec)


def _swa(stream_ratio, **overrides):
    kwargs = dict(block_size=32, num_kv_heads=1, head_size=512,
                  dtype=torch.bfloat16, sliding_window=128,
                  cache_dtype_str="auto", model_version="deepseek_v4",
                  stream_compress_ratio=stream_ratio)
    kwargs.update(overrides)
    return AscendSlidingWindowMLASpec(**kwargs)


# ------------------------------------------------------- the field's purpose

def test_zero_and_one_produce_different_specs():
    """The whole point. If these compare equal the field does nothing."""
    assert _swa(0) != _swa(1)


def test_the_page_arithmetic_is_untouched_by_the_stream_ratio():
    """The new field must not become a second page divisor.

    compress_ratio stays 1 in both, so storage_block_size stays 32 -- if the
    stream ratio ever reached the page arithmetic, ratio 0 would divide by
    zero and ratio 2 would silently halve a cache that is not compressed.
    """
    for ratio in (0, 1, 2):
        assert _swa(ratio).storage_block_size == 32


def test_zero_is_not_a_legal_page_divisor_which_is_why_this_field_exists():
    """Stated as an executable fact rather than as a comment."""
    with pytest.raises(ZeroDivisionError):
        _swa(0, compress_ratio=0).storage_block_size


# ------------------------------------------------------------- what it feeds

@pytest.mark.parametrize("ratio,expected", [(0, False), (1, True), (2, True)])
def test_the_reader_reaches_the_right_answer_for_v41(ratio, expected,
                                                     monkeypatch):
    """attends_compressed_stream is the consumer this was collapsing.

    V4.1 reads ratio >= 1 as "attends a compressed stream", so 0 and 1 land on
    opposite sides of exactly the boundary the clamp erased.
    """
    class _Cfg:
        kv_source_layer_ids = [2, 8, 14, 20]

    class _VllmConfig:
        class model_config:
            hf_text_config = _Cfg()
            hf_config = _Cfg()

    assert attends_compressed_stream(ratio, _VllmConfig) is expected


def test_a_spec_stating_nothing_leaves_the_reader_on_its_old_chain():
    """Default None, so a V4.0 spec is unchanged and so is an A arm."""
    assert _swa(None).stream_compress_ratio is None
    assert AscendMLAAttentionSpec(
        block_size=128, num_kv_heads=1, head_size=512, dtype=torch.bfloat16,
        cache_dtype_str="auto").stream_compress_ratio is None


# ------------------------------------------------------------------ grouping

def test_a_group_cannot_span_two_streams():
    """merge refuses, which is what keeps them in separate groups.

    The grouping pass merges two buckets only if merge() does not raise, so
    the refusal is the mechanism, not a launch failure. Taking specs[0]'s
    value instead would give one group whose single descriptor is wrong for
    every layer but the first.
    """
    with pytest.raises(ValueError, match="one group cannot serve two streams"):
        AscendSlidingWindowMLASpec.merge([_swa(0), _swa(2)])


def test_an_unstated_ratio_does_not_quietly_join_a_stated_one():
    with pytest.raises(ValueError, match="layers that state none"):
        AscendSlidingWindowMLASpec.merge([_swa(0), _swa(None)])


def test_a_unanimous_group_keeps_its_ratio_through_the_merge():
    """A field dropped by merge is a field that exists until it matters."""
    merged = AscendSlidingWindowMLASpec.merge([_swa(0), _swa(0)])
    assert merged.stream_compress_ratio == 0
    merged_mla = AscendMLAAttentionSpec.merge([
        AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=512,
                               dtype=torch.bfloat16, cache_dtype_str="auto",
                               stream_compress_ratio=2)] * 2)
    assert merged_mla.stream_compress_ratio == 2


# -------------------------------------------------------------- the switch

class _Cache:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def test_the_switch_is_off_by_default(monkeypatch):
    from vllm_ascend.models.deepseek_v4_1.caches import stated_stream_ratio

    monkeypatch.delenv("VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO", raising=False)
    assert stated_stream_ratio(_Cache(stream_compress_ratio=0)) is None


def test_on_it_reports_the_configured_entry_including_zero(monkeypatch):
    from vllm_ascend.models.deepseek_v4_1.caches import stated_stream_ratio

    monkeypatch.setenv("VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO", "1")
    assert stated_stream_ratio(_Cache(stream_compress_ratio=0)) == 0
    assert stated_stream_ratio(_Cache(compress_ratio=0)) == 0
    assert stated_stream_ratio(_Cache(compress_ratio=2)) == 2


def test_on_with_nothing_attached_refuses_rather_than_falling_back(monkeypatch):
    """A silent None here would reproduce the defect under a flag that claims
    to have fixed it."""
    from vllm_ascend.models.deepseek_v4_1.caches import stated_stream_ratio

    monkeypatch.setenv("VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO", "1")
    with pytest.raises(ValueError, match="carries none"):
        stated_stream_ratio(_Cache())


def test_the_attention_layer_attaches_it_to_the_swa_cache():
    """The SWA cache cannot take it as a constructor argument -- upstream
    builds it inside the base __init__ with a fixed list -- so the one place
    where both the ratio and the cache object exist has to do it."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    source = inspect.getsource(AscendDeepseekV41Attention.__init__)
    assert "self.swa_cache_layer.stream_compress_ratio = self.compress_ratio" \
        in source, (
            "nothing attaches the layer's ratio to its SWA cache, so the "
            "cache reports no stream ratio and the switch raises")
