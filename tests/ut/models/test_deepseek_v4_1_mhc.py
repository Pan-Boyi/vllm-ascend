# SPDX-License-Identifier: Apache-2.0
"""Hyper-connections without tilelang.

The decoder layer calls mhc_pre_delayed_tilelang and mhc_post_tilelang
directly in its forward. tilelang is not installed here, and the third launch
reached the first decoder layer before finding that out -- past registration,
past engram, past KV cache setup, past 90 seconds of weight loading. So the
substitution has to be right, and it has to be in place before that forward.
"""

import inspect

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import mhc, platform_patches

TOKENS, HC_MULT, HIDDEN = 5, 4, 64


def _inputs(seed=0):
    torch.manual_seed(seed)
    residual = torch.randn(TOKENS, HC_MULT, HIDDEN, dtype=torch.bfloat16)
    fn = torch.randn(HC_MULT * (HC_MULT + 2), HC_MULT * HIDDEN)
    return dict(
        residual=residual,
        fn=fn,
        hc_scale=torch.rand(3) + 0.5,
        hc_base=torch.randn(HC_MULT * (HC_MULT + 2)),
        rms_eps=1e-20,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        pre_mix=torch.rand(TOKENS, HC_MULT),
    )


def test_post_uses_the_ascend_operator_and_adapts_the_shapes():
    """aclnnHcPost exists here -- it is V4.0's production path, and the
    hc_pre/hc_post pair was extended to hidden 5120 for V4.1.

    Its shape contract differs from upstream's in two ways, and both are
    silent if missed: it carries a batch axis V4.1 does not, and its `post`
    has no trailing 1.
    """
    seen = {}

    class _Ops:
        @staticmethod
        def npu_hc_post(x, residual, post, comb):
            seen.update(x=x.shape, residual=residual.shape,
                        post=post.shape, comb=comb.shape)
            return torch.zeros_like(residual)

    T, HC, D = 5, 4, 8
    real = torch.ops._C_ascend if hasattr(torch.ops, "_C_ascend") else None
    torch.ops._C_ascend = _Ops
    try:
        out = mhc.mhc_post_ascend(
            torch.randn(T, D), torch.randn(T, HC, D),
            torch.randn(T, HC, 1), torch.randn(T, HC, HC))
    finally:
        if real is not None:
            torch.ops._C_ascend = real

    assert seen["x"] == (1, T, D)
    assert seen["residual"] == (1, T, HC, D)
    assert seen["post"] == (1, T, HC), "the trailing 1 must be dropped"
    assert seen["comb"] == (1, T, HC, HC)
    assert out.shape == (T, HC, D), "the batch axis must come back off"


def test_post_falls_back_to_torch_when_asked(monkeypatch):
    """The A/B switch. They compute the same expression, and "the same
    expression" has been wrong here before, so the comparison stays available."""
    from vllm.model_executor.kernels.mhc.torch import mhc_post_torch

    # Patch the accessor rather than the envs module: envs resolves names
    # through __getattr__, so setattr on it installs a real attribute that
    # shadows the lookup and outlives the test.
    monkeypatch.setattr(mhc, "envs_dsv41_mhc_post_torch", lambda: True)

    T, HC, D = 5, 4, 8
    args = (torch.randn(T, D), torch.randn(T, HC, D),
            torch.randn(T, HC, 1), torch.randn(T, HC, HC))
    torch.testing.assert_close(mhc.mhc_post_ascend(*args), mhc_post_torch(*args))


@pytest.mark.parametrize("x_shape,res_shape", [
    ((1, 5, 8), (5, 4, 8)),      # x already batched
    ((5, 8), (1, 5, 4, 8)),      # residual already batched
])
def test_post_refuses_an_already_batched_tensor(x_shape, res_shape):
    with pytest.raises(ValueError, match="batch axis is added here"):
        mhc.mhc_post_ascend(torch.randn(*x_shape), torch.randn(*res_shape),
                            torch.randn(5, 4, 1), torch.randn(5, 4, 4))


def test_post_refuses_a_post_without_the_trailing_one():
    """Without the check this squeezes hc instead, whenever hc is 1."""
    with pytest.raises(ValueError, match="length-1"):
        mhc.mhc_post_ascend(torch.randn(5, 8), torch.randn(5, 4, 8),
                            torch.randn(5, 4), torch.randn(5, 4, 4))


def test_pre_accepts_exactly_what_the_decoder_layer_passes():
    """The call sites pass norm_weight and norm_eps by keyword. The torch
    reference does not take them at all, so a bare rebind would be a
    TypeError in the first forward."""
    params = inspect.signature(mhc.mhc_pre_delayed_ascend).parameters
    for name in ("residual", "fn", "hc_scale", "hc_base", "rms_eps",
                 "hc_pre_eps", "hc_sinkhorn_eps", "hc_post_mult_value",
                 "sinkhorn_repeat", "pre_mix", "x", "norm_weight", "norm_eps"):
        assert name in params, name

    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
    reference = inspect.signature(mhc_pre_delayed_torch).parameters
    assert "norm_weight" not in reference, (
        "if upstream's reference grows these, the wrapper is redundant and "
        "should be replaced by a direct rebind rather than left in place")


def test_only_the_layer_input_differs_from_the_raw_reference():
    """Pins the composition.

    Everything but the collapsed layer input must come through untouched; the
    wrapper's whole job is the norm on that one output. If it altered the
    mixes too, accuracy would move with nothing to point at.
    """
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    args = _inputs()
    weight = torch.randn(HIDDEN, dtype=torch.bfloat16).abs() + 0.5

    raw = mhc_pre_delayed_torch(**args)
    wrapped = mhc.mhc_pre_delayed_ascend(**args, norm_weight=weight, norm_eps=1e-6)

    torch.testing.assert_close(wrapped[0], raw[0])      # post mix
    torch.testing.assert_close(wrapped[1], raw[1])      # residual mix
    torch.testing.assert_close(wrapped[3], raw[3])      # next pre-mix
    assert not torch.allclose(wrapped[2].float(), raw[2].float(), atol=1e-2), (
        "the layer input must be normalised; identical means the norm was "
        "dropped, which returns the right shape and the wrong values")


def test_without_a_norm_weight_it_is_the_reference_exactly():
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    args = _inputs(1)
    raw = mhc_pre_delayed_torch(**args)
    wrapped = mhc.mhc_pre_delayed_ascend(**args)
    for a, b in zip(raw, wrapped):
        torch.testing.assert_close(a, b)


def test_the_norm_is_rms_with_the_weight_applied():
    args = _inputs(2)
    weight = torch.randn(HIDDEN, dtype=torch.bfloat16).abs() + 0.5
    eps = 1e-6

    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
    layer_input = mhc_pre_delayed_torch(**args)[2]

    expected = layer_input.float()
    expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + eps)
    expected = (expected * weight.float()).to(layer_input.dtype)

    got = mhc.mhc_pre_delayed_ascend(**args, norm_weight=weight, norm_eps=eps)[2]
    torch.testing.assert_close(got, expected)


def test_the_norm_accumulates_in_fp32():
    """hidden_size is 5120 and bf16 has an 8-bit significand: a sum of squares
    accumulated in bf16 stops growing long before the last terms."""
    source = inspect.getsource(mhc._rms_norm)
    assert ".float()" in source
    x = torch.full((1, 4096), 0.01, dtype=torch.bfloat16)
    w = torch.ones(4096, dtype=torch.bfloat16)
    got = mhc._rms_norm(x, w, 1e-6).float()
    # every element equal and the norm recovers unit RMS
    assert torch.allclose(got, torch.ones_like(got), atol=2e-2), got[0, :4]


def test_a_mismatched_norm_weight_is_refused():
    args = _inputs(3)
    with pytest.raises(ValueError, match="per-channel norm"):
        mhc.mhc_pre_delayed_ascend(
            **args, norm_weight=torch.ones(HIDDEN // 2, dtype=torch.bfloat16))


# -- the rebinds -------------------------------------------------------------


def test_the_decoder_layer_sees_the_ascend_versions():
    import vllm.models.deepseek_v4_1.nvidia.model as upstream

    platform_patches.rebind_mhc_functions()
    assert upstream.mhc_pre_delayed_tilelang is mhc.mhc_pre_delayed_ascend
    assert upstream.mhc_post_tilelang is mhc.mhc_post_ascend


def test_rebinding_is_idempotent():
    platform_patches.rebind_mhc_functions()
    assert platform_patches.rebind_mhc_functions() == []


def test_a_vanished_name_is_refused_rather_than_skipped(monkeypatch):
    import vllm.models.deepseek_v4_1.nvidia.model as upstream

    monkeypatch.delattr(upstream, "mhc_pre_delayed_tilelang")
    with pytest.raises(AttributeError, match="no longer defines"):
        platform_patches.rebind_mhc_functions()


def test_apply_all_reports_them():
    from vllm_ascend.models import deepseek_v4_1

    assert "mhc_pre_delayed_tilelang" in deepseek_v4_1.APPLIED_PATCHES
    assert "mhc_post_tilelang" in deepseek_v4_1.APPLIED_PATCHES
