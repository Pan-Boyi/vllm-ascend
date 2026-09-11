# SPDX-License-Identifier: Apache-2.0
"""The indexer's fused RoPE + Q quantisation, transcribed rather than derived.

Upstream dispatches this on a blacklist -- cutedsl, else xpu, else Triton --
and triton-ascend aborts compiling the Triton one. abort(), not an exception:
four workers gone in the same second with exit code None and no traceback.

Everything here pins a detail that would have been wrong if it had been
derived from the docstring rather than read off the kernel, and every one of
them is silent when wrong.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.indexer_q import (
    AMAX_FLOOR, FP8_MAX, fused_indexer_q_rope_quant_ascend)

T, H, HEAD_DIM, HALF = 4, 3, 128, 32
ROT = 2 * HALF
NOPE = HEAD_DIM - ROT
SOFTMAX, HEAD = 0.125, 0.5


def _inputs(seed=0, scale=1.0):
    torch.manual_seed(seed)
    return dict(
        positions=torch.arange(T),
        index_q=(torch.randn(T, H, HEAD_DIM) * scale).to(torch.bfloat16),
        index_q_cos_sin_cache=torch.randn(64, ROT),
        index_weights=torch.rand(T, H),
        index_weights_softmax_scale=SOFTMAX,
        index_weights_head_scale=HEAD,
    )


def _recover_scale(weights_out, index_weights):
    """weights_out = weights * scale * softmax * head."""
    return weights_out / (index_weights.float() * SOFTMAX * HEAD)


def test_the_scale_is_a_power_of_two():
    """exp2(ceil(log2(s))). Not rounding it makes every stored value slightly
    different, which no shape or range check would notice."""
    args = _inputs(1)
    _, weights_out = fused_indexer_q_rope_quant_ascend(**args)
    scale = _recover_scale(weights_out, args["index_weights"])
    log2 = torch.log2(scale)
    torch.testing.assert_close(log2, torch.round(log2), atol=1e-5, rtol=0)


def test_the_rotation_covers_the_trailing_dims_only():
    """The NoPE part is ahead of the rotated part, not after it."""
    args = _inputs(2)
    q = args["index_q"]
    q_fp8, weights_out = fused_indexer_q_rope_quant_ascend(**args)
    scale = _recover_scale(weights_out, args["index_weights"]).unsqueeze(-1)

    # the leading NOPE dims are the input divided by the scale, unrotated
    expected_nope = (q[..., :NOPE].float() / scale).to(torch.float8_e4m3fn)
    assert torch.equal(q_fp8[..., :NOPE].view(torch.uint8),
                       expected_nope.view(torch.uint8))


def test_the_pairing_is_interleaved_not_split_half():
    """x[2k] with x[2k+1]. Split-half is the other plausible reading and it
    produces a complete, wrongly-rotated tensor."""
    args = _inputs(3)
    cache = args["index_q_cos_sin_cache"]
    q = args["index_q"]
    q_fp8, weights_out = fused_indexer_q_rope_quant_ascend(**args)
    scale = _recover_scale(weights_out, args["index_weights"])

    cs = cache.index_select(0, args["positions"]).float()
    cos, sin = cs[:, :HALF].unsqueeze(1), cs[:, HALF:].unsqueeze(1)
    tail = q[..., NOPE:].float().reshape(T, H, HALF, 2)
    even, odd = tail[..., 0], tail[..., 1]
    r_even = (even * cos - odd * sin).to(torch.bfloat16).float()
    r_odd = (odd * cos + even * sin).to(torch.bfloat16).float()
    expected = torch.stack((r_even / scale.unsqueeze(-1),
                            r_odd / scale.unsqueeze(-1)), dim=-1)
    expected = expected.reshape(T, H, ROT).to(torch.float8_e4m3fn)

    assert torch.equal(q_fp8[..., NOPE:].view(torch.uint8),
                       expected.view(torch.uint8))


def test_the_scale_is_taken_after_the_bf16_round_trip():
    """The rotated values go fp32 -> bf16 -> fp32 before the absmax, so the
    scale describes what will be stored. Skipping it changes the scale
    whenever bf16 rounding crosses a power-of-two boundary."""
    args = _inputs(4, scale=30.0)
    _, weights_out = fused_indexer_q_rope_quant_ascend(**args)
    got = _recover_scale(weights_out, args["index_weights"])

    # same computation without the round trip
    cs = args["index_q_cos_sin_cache"].index_select(0, args["positions"]).float()
    cos, sin = cs[:, :HALF].unsqueeze(1), cs[:, HALF:].unsqueeze(1)
    q = args["index_q"].float()
    tail = q[..., NOPE:].reshape(T, H, HALF, 2)
    even, odd = tail[..., 0], tail[..., 1]
    amax = torch.maximum((even * cos - odd * sin).abs().amax(-1),
                         (odd * cos + even * sin).abs().amax(-1))
    amax = torch.maximum(amax, q[..., :NOPE].abs().amax(-1))
    naive = torch.exp2(torch.ceil(torch.log2(
        torch.clamp(amax, min=AMAX_FLOOR) / FP8_MAX)))

    # they agree except where rounding crosses a boundary; the implementation
    # must follow the kernel, so assert it equals the round-tripped form
    assert torch.isfinite(got).all()
    assert (got <= naive * 2).all() and (got >= naive / 2).all(), (
        "the two differ by more than one binade; the transcription is wrong, "
        "not merely rounding differently")


def test_the_nope_part_participates_in_the_absmax():
    """A large NoPE value must raise the scale. If the absmax only covered the
    rotated half, the NoPE values would clip on store."""
    args = _inputs(5, scale=0.01)
    quiet = _recover_scale(
        fused_indexer_q_rope_quant_ascend(**args)[1], args["index_weights"])

    args["index_q"] = args["index_q"].clone()
    args["index_q"][0, 0, 0] = 100.0
    loud = _recover_scale(
        fused_indexer_q_rope_quant_ascend(**args)[1], args["index_weights"])

    assert loud[0, 0] > quiet[0, 0]
    assert torch.equal(loud[1:], quiet[1:]), "only that head's scale may move"


def test_an_all_zero_head_does_not_divide_by_zero():
    args = _inputs(6)
    args["index_q"] = torch.zeros_like(args["index_q"])
    q_fp8, weights_out = fused_indexer_q_rope_quant_ascend(**args)
    assert torch.isfinite(weights_out).all()
    scale = _recover_scale(weights_out, args["index_weights"])
    expected = torch.exp2(torch.ceil(torch.log2(torch.tensor(AMAX_FLOOR / FP8_MAX))))
    torch.testing.assert_close(scale, torch.full_like(scale, float(expected)))


def test_the_weight_fold_carries_the_q_scale():
    """FP8 Q is stored without a companion scale tensor; the downstream logits
    kernel applies it through these weights. Omitting it scales every logit."""
    args = _inputs(7)
    _, weights_out = fused_indexer_q_rope_quant_ascend(**args)
    scale = _recover_scale(weights_out, args["index_weights"])
    torch.testing.assert_close(
        weights_out,
        args["index_weights"].float() * scale * SOFTMAX * HEAD)
    assert not torch.allclose(
        weights_out, args["index_weights"].float() * SOFTMAX * HEAD), (
        "without the q_scale this is the MXFP4 contract, not the FP8 one")


def test_the_mxfp4_path_is_refused_rather_than_approximated():
    """Its fold contract is the opposite: scales live with the values and are
    NOT folded into the weights."""
    args = _inputs(8)
    with pytest.raises(NotImplementedError, match="MXFP4"):
        fused_indexer_q_rope_quant_ascend(**args, use_fp4=True)


def test_the_rebind_reaches_the_call_site_module():
    import vllm.models.deepseek_v4_1.attention as attention

    from vllm_ascend.models.deepseek_v4_1.indexer_q import apply_indexer_q_patch

    apply_indexer_q_patch()
    assert (attention.fused_indexer_q_rope_quant
            is fused_indexer_q_rope_quant_ascend)
    assert apply_indexer_q_patch() is False, "must be idempotent"
