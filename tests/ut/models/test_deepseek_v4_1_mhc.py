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


# -- the operator path for delayed pre ---------------------------------------


class _FakeNpuOps:
    """Stands in for torch.ops.npu, returning the six/three-tensor contracts."""

    def __init__(self, tokens, hc, hidden, out_flag_seen=None):
        self.t, self.hc, self.d = tokens, hc, hidden
        self.calls = out_flag_seen if out_flag_seen is not None else []

    def npu_mhc_pre(self, x, phi, alpha, bias, *, gamma=None, norm_eps,
                    hc_eps, out_flag=0):
        self.calls.append(("pre", out_flag))
        empty = torch.empty(0)
        h_in = torch.full((self.t, self.d), 7.0)          # the NON-delayed one
        h_post = torch.full((self.t, self.hc), 2.0)
        h_res = torch.full((self.t, self.hc, self.hc), 3.0)
        h_pre = torch.full((self.t, self.hc), 5.0)
        if not out_flag:
            return h_in, h_post, h_res, empty, empty, empty
        return (h_in, h_post, h_res, torch.ones(self.t),
                torch.ones(self.t, 24), h_pre)

    def npu_mhc_sinkhorn(self, h_res, *, eps, num_iters, out_flag=0):
        self.calls.append(("sinkhorn", num_iters, float(h_res.flatten()[0])))
        return (h_res * 10, torch.empty(0), torch.empty(0))


def _with_fake_ops(fake):
    class _Ops:
        npu = fake
    return _Ops


def test_delayed_uses_the_incoming_pre_mix_not_the_operators_collapsed_output(monkeypatch):
    """The whole of "delayed", and the one substitution that would be silent.

    npu_mhc_pre[0] is collapsed with the pre-mix this call just computed --
    the non-delayed variant. It has the right shape and the wrong values.
    """
    monkeypatch.setattr(mhc, 'envs_dsv41_mhc_pre_operator', lambda: True)
    T, HC, D = 3, 4, 8
    fake = _FakeNpuOps(T, HC, D)
    residual = torch.randn(T, HC, D)
    incoming = torch.rand(T, HC)

    real_ops = torch.ops
    torch.ops = _with_fake_ops(fake)
    try:
        post, comb, layer_input, next_pre = mhc.mhc_pre_delayed_ascend(
            residual, torch.randn(HC * (HC + 2), HC * D), torch.rand(3),
            torch.randn(HC * (HC + 2)), 1e-20, 1e-6, 1e-6, 2.0, 20,
            pre_mix=incoming)
    finally:
        torch.ops = real_ops

    expected = (incoming.unsqueeze(-1) * residual.float()).sum(1).to(residual.dtype)
    torch.testing.assert_close(layer_input, expected)
    assert not torch.allclose(layer_input, torch.full((T, D), 7.0)), (
        "hIn is the non-delayed collapse and must not be used")
    torch.testing.assert_close(next_pre, torch.full((T, HC), 5.0))


def test_out_flag_is_always_one(monkeypatch):
    """At the default 0 the last three outputs come back empty, and `pre` is
    one of them. An empty tensor there reads as a shape problem downstream."""
    monkeypatch.setattr(mhc, 'envs_dsv41_mhc_pre_operator', lambda: True)
    T, HC, D = 3, 4, 8
    fake = _FakeNpuOps(T, HC, D)
    real_ops = torch.ops
    torch.ops = _with_fake_ops(fake)
    try:
        mhc.mhc_pre_delayed_ascend(
            torch.randn(T, HC, D), torch.randn(HC * (HC + 2), HC * D),
            torch.rand(3), torch.randn(HC * (HC + 2)), 1e-20, 1e-6, 1e-6,
            2.0, 20, pre_mix=torch.rand(T, HC))
    finally:
        torch.ops = real_ops

    assert ("pre", 1) in fake.calls


def test_the_sinkhorn_fragment_is_passed_straight_through(monkeypatch):
    """It already carries scale and bias. Adding them again measured 0.419
    against 1.18e-06 for feeding it unchanged."""
    monkeypatch.setattr(mhc, 'envs_dsv41_mhc_pre_operator', lambda: True)
    T, HC, D = 3, 4, 8
    fake = _FakeNpuOps(T, HC, D)
    real_ops = torch.ops
    torch.ops = _with_fake_ops(fake)
    try:
        _, comb, _, _ = mhc.mhc_pre_delayed_ascend(
            torch.randn(T, HC, D), torch.randn(HC * (HC + 2), HC * D),
            torch.rand(3), torch.randn(HC * (HC + 2)), 1e-20, 1e-6, 1e-6,
            2.0, 20, pre_mix=torch.rand(T, HC))
    finally:
        torch.ops = real_ops

    sinkhorn_call = next(c for c in fake.calls if c[0] == "sinkhorn")
    assert sinkhorn_call[2] == 3.0, "the fragment must reach sinkhorn unmodified"
    assert sinkhorn_call[1] == 20, "sinkhorn_repeat must be forwarded"
    torch.testing.assert_close(comb, torch.full((T, HC, HC), 30.0))


def test_an_empty_pre_is_diagnosed_as_the_flag(monkeypatch):
    monkeypatch.setattr(mhc, 'envs_dsv41_mhc_pre_operator', lambda: True)
    T, HC, D = 3, 4, 8

    class _Flag0(_FakeNpuOps):
        def npu_mhc_pre(self, *a, **kw):
            kw["out_flag"] = 0
            return _FakeNpuOps.npu_mhc_pre(self, *a, **kw)

    real_ops = torch.ops
    torch.ops = _with_fake_ops(_Flag0(T, HC, D))
    try:
        with pytest.raises(RuntimeError, match="out_flag=0"):
            mhc.mhc_pre_delayed_ascend(
                torch.randn(T, HC, D), torch.randn(HC * (HC + 2), HC * D),
                torch.rand(3), torch.randn(HC * (HC + 2)), 1e-20, 1e-6, 1e-6,
                2.0, 20, pre_mix=torch.rand(T, HC))
    finally:
        torch.ops = real_ops


def test_the_operator_path_is_off_until_5120_is_compared():
    """Checked at hidden 4096; at 5120 only for producing finite values.
    Defaulting it on would make an unverified numeric path the one that runs."""
    import vllm_ascend.envs as envs

    assert envs.VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR == "0"
    assert not mhc.envs_dsv41_mhc_pre_operator()
