# SPDX-License-Identifier: Apache-2.0
"""Import-time platform adaptations, and the upstream shapes they depend on."""

import inspect

import pytest

from vllm_ascend.models.deepseek_v4_1 import platform_patches


@pytest.fixture
def compress_module():
    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as mod

    original = getattr(mod, platform_patches.JOIN_FLAG)
    yield mod
    setattr(mod, platform_patches.JOIN_FLAG, original)


def test_upstream_still_exposes_the_switch(compress_module):
    """Pin what is being rebound.

    If the flag is renamed or folded into the kernel, this goes red at test
    time. Discovering it at runtime instead means a compiler abort, which takes
    the process with it and cannot be caught.
    """
    assert hasattr(compress_module, platform_patches.JOIN_FLAG)
    assert isinstance(getattr(compress_module, platform_patches.JOIN_FLAG), bool)

    # It has to reach the kernel as a constexpr; a runtime read would mean the
    # joined path could still be compiled.
    source = inspect.getsource(compress_module)
    assert f"JOIN_ROW_PTRS={platform_patches.JOIN_FLAG}" in source.replace(" ", "")


def test_patch_selects_the_separate_loads_path(compress_module):
    setattr(compress_module, platform_patches.JOIN_FLAG, True)
    assert platform_patches.disable_join_row_ptrs() is True
    assert getattr(compress_module, platform_patches.JOIN_FLAG) is False


def test_patch_is_idempotent(compress_module):
    setattr(compress_module, platform_patches.JOIN_FLAG, True)
    platform_patches.disable_join_row_ptrs()
    assert platform_patches.disable_join_row_ptrs() is False
    assert getattr(compress_module, platform_patches.JOIN_FLAG) is False


def test_a_vanished_switch_is_not_patched_over_in_silence(compress_module,
                                                          monkeypatch):
    monkeypatch.delattr(compress_module, platform_patches.JOIN_FLAG)
    with pytest.raises(AttributeError, match="no longer defines"):
        platform_patches.disable_join_row_ptrs()


def test_apply_all_covers_both_adaptations(compress_module):
    import vllm.models.deepseek_v4_1.nvidia.model as model_mod

    from vllm_ascend.models.deepseek_v4_1 import attn_selector

    original_selector = getattr(model_mod, attn_selector.SELECTOR_NAME)
    setattr(compress_module, platform_patches.JOIN_FLAG, True)
    try:
        platform_patches.apply_all()
        assert getattr(compress_module, platform_patches.JOIN_FLAG) is False
        assert (getattr(model_mod, attn_selector.SELECTOR_NAME)
                is attn_selector.select_ascend_attn_cls)
    finally:
        setattr(model_mod, attn_selector.SELECTOR_NAME, original_selector)
