# SPDX-License-Identifier: Apache-2.0
"""The attention-selector hijack, and the upstream contract it depends on.

The hijack rebinds a module-level name in upstream's decoder-layer module. That
is cheap and fails loudly when upstream moves -- but only if something checks
the shape of what it is replacing, which is what the contract test here is for.
"""

import inspect

import pytest

from vllm_ascend.models.deepseek_v4_1 import attn_selector


@pytest.fixture
def upstream():
    import vllm.models.deepseek_v4_1.nvidia.model as module

    original = getattr(module, attn_selector.SELECTOR_NAME)
    yield module
    setattr(module, attn_selector.SELECTOR_NAME, original)


def test_upstream_selector_contract(upstream):
    """Pin what the hijack is replacing.

    If upstream renames the selector, changes its parameters, or stops being
    the thing the decoder layer calls, this goes red at test time rather than
    at model construction time on a machine with the weights loaded.
    """
    selector = getattr(upstream, attn_selector.SELECTOR_NAME)
    params = list(inspect.signature(selector).parameters)
    assert params == ["vllm_config"]

    # The decoder layer must still reach the selector through the module
    # global -- an inlined or method-based lookup would make the rebind a
    # silent no-op.
    source = inspect.getsource(upstream.DeepseekV4DecoderLayer.__init__)
    assert f"{attn_selector.SELECTOR_NAME}(" in source


def test_patch_rebinds_the_selector(upstream):
    attn_selector.apply_attn_selector_patch()
    assert (getattr(upstream, attn_selector.SELECTOR_NAME)
            is attn_selector.select_ascend_attn_cls)


def test_patch_is_idempotent(upstream):
    attn_selector.apply_attn_selector_patch()
    attn_selector.apply_attn_selector_patch()
    assert (getattr(upstream, attn_selector.SELECTOR_NAME)
            is attn_selector.select_ascend_attn_cls)


def test_patch_raises_when_the_name_is_gone(upstream, monkeypatch):
    """A vanished target must not be patched over in silence."""
    monkeypatch.delattr(upstream, attn_selector.SELECTOR_NAME)
    with pytest.raises(AttributeError, match="no longer defines"):
        attn_selector.apply_attn_selector_patch()


def test_selector_returns_the_ascend_attention_class():
    from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    got = attn_selector.select_ascend_attn_cls(vllm_config=None)
    assert got is AscendDeepseekV41Attention
    assert issubclass(got, DeepseekV4Attention)


def test_unimplemented_paths_raise_rather_than_doing_the_cuda_thing():
    """The remaining stub must not be quietly inherited from the CUDA base.

    forward_mqa is abstract upstream, so inheriting it would be a TypeError at
    construction; the point here is that the Ascend class supplies its own and
    that it refuses rather than approximates.

    _o_proj used to be listed here too. It is implemented now, which is why it
    is not -- see test_o_proj_is_no_longer_a_stub below, which holds that
    boundary from the other side so this file still records where it sits.
    """
    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention as cls

    assert cls.forward_mqa is not None
    with pytest.raises(NotImplementedError, match="sparse MLA forward"):
        cls.forward_mqa(object.__new__(cls), None, None, None, None)


def test_o_proj_is_no_longer_a_stub():
    """Records that the output projection crossed over, and how far.

    It is implemented, but only for the single-rank case: the oproj tensor
    parallel path is still refused, and refused explicitly rather than by
    computing one rank's partial result and returning it as the whole. Asserting
    that here means removing the guard without wiring the collectives goes red,
    instead of producing a plausible tensor.
    """
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention as cls

    source = inspect.getsource(cls._o_proj)
    assert "raise NotImplementedError" in source, (
        "the oproj tensor parallel path must still refuse")
    assert "oproj_tp_enable" in source
    # The parts that make it a real implementation rather than a stub.
    assert "inverse_rope_from_cos_sin_cache" in source
    assert "check_grouped_wo_a_layout" in source
    assert "self.wo_b(" in source


def test_padded_q_heads_does_not_pad():
    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention as cls

    # The sharedkv operator validates sinks.shape[0] == query head num, so any
    # padding here would have to be mirrored into the sink tensor.
    assert cls.get_padded_num_q_heads(64) == 64


def test_kv_layout_flag_matches_the_bound_operator():
    """False today: the bound operator packs 640 B/state, not fp8_ds_mla's 584.

    Flip this together with binding MixedQuantSparseFlashMla in quant_mode=2,
    which is byte-identical to fp8_ds_mla -- not before.
    """
    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention as cls

    assert cls.use_fp8_ds_mla_layout is False
