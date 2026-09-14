# SPDX-License-Identifier: Apache-2.0
"""The SWA cache shape guard is load-bearing, and now it is measured.

``attention.py`` refuses when ``swa_kv_cache.shape[1:] != (block_size,
head_dim)``, saying "these index each other, so a mismatch writes rows at the
wrong offset rather than failing". That was a reasoned precaution. It is now
an observation:

    tools/probe_scatter_nd_negative_index.py, card 0, 2026-09-14
      indices [0, 16] into a cache of [8, 16, 32]  ->  wrote (1, 0)

An over-large OFFSET is not bounds-checked -- it aliases into the next block,
because that axis behaves like a flat linear index. (An over-large BLOCK is
ignored, and so is any negative. The two axes are not treated alike.)

⇒ So if ``block_size`` and the cache's second dimension ever disagree, rows
land in a neighbouring block **silently**. The guard is the only thing
between that and a corruption that surfaces as accuracy.

⚠️ Not covered by ``test_deepseek_v4_1_kv_binding.py`` despite that file
containing the same ``shape[1:] == (block_size, head_dim)`` expression: it is
about binding the cache list, a different object. Same words, different
subject.
"""

import pytest
import torch


class _Meta:
    def __init__(self, block_size):
        self.block_size = block_size
        self.slot_mapping = torch.zeros(4, dtype=torch.int32)


class _CacheLayer:
    prefix = "layer.0.swa"

    def __init__(self, cache):
        self.kv_cache = cache


class _Layer:
    """Only what the method touches before the guard. Copied from the call
    order in _fused_qnorm_rope_kv_insert, not invented."""

    def __init__(self, cache, head_dim):
        self.swa_cache_layer = _CacheLayer(cache)
        self.head_dim = head_dim
        self.n_local_heads = 1
        self.padded_heads = 1


def _call(layer, metadata):
    from vllm_ascend.models.deepseek_v4_1.attention import (
        AscendDeepseekV41Attention)

    return AscendDeepseekV41Attention._fused_qnorm_rope_kv_insert(
        layer, q=torch.zeros(4, 1, 8), kv=torch.zeros(4, 8),
        positions=torch.arange(4), attn_metadata=metadata)


def test_the_guard_fires_when_block_size_disagrees():
    """🔴 The mismatch this exists for: 16 declared, 32 in the cache.

    Without the refusal, an offset beyond 16 would write into the next block
    and nothing would report it -- measured, see the module docstring.
    """
    cache = torch.zeros(8, 32, 64, dtype=torch.bfloat16)   # second dim 32
    layer = _Layer(cache, head_dim=64)
    with pytest.raises(ValueError, match="index each other"):
        _call(layer, {_CacheLayer.prefix: _Meta(block_size=16)})


def test_the_guard_fires_when_head_dim_disagrees():
    cache = torch.zeros(8, 16, 64, dtype=torch.bfloat16)
    layer = _Layer(cache, head_dim=128)                    # says 128, is 64
    with pytest.raises(ValueError, match="index each other"):
        _call(layer, {_CacheLayer.prefix: _Meta(block_size=16)})


def test_the_message_names_both_numbers():
    """An error that says "shapes disagree" without saying which two costs a
    round trip to the person reading it at 3am."""
    cache = torch.zeros(8, 32, 64, dtype=torch.bfloat16)
    layer = _Layer(cache, head_dim=64)
    with pytest.raises(ValueError) as exc:
        _call(layer, {_CacheLayer.prefix: _Meta(block_size=16)})
    text = str(exc.value)
    assert "(8, 32, 64)" in text, "the actual cache shape must be named"
    assert "16" in text and "64" in text, "both declared numbers must be named"


def test_a_matching_shape_passes_the_guard():
    """🔵 The negative arm. Without it, a guard that raises unconditionally
    would satisfy every test above."""
    cache = torch.zeros(8, 16, 64, dtype=torch.bfloat16)
    layer = _Layer(cache, head_dim=64)
    try:
        _call(layer, {_CacheLayer.prefix: _Meta(block_size=16)})
    except ValueError as exc:
        if "index each other" in str(exc):
            pytest.fail(f"the guard fired on a matching shape: {exc}")
    except Exception:
        pass          # it goes on to the real prolog; only the guard matters


def test_a_missing_metadata_group_is_named_not_ignored():
    cache = torch.zeros(8, 16, 64, dtype=torch.bfloat16)
    layer = _Layer(cache, head_dim=64)
    with pytest.raises(ValueError, match="no attention metadata under"):
        _call(layer, {"some.other.layer": _Meta(block_size=16)})
