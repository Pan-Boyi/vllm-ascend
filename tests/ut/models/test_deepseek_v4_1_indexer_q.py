# SPDX-License-Identifier: Apache-2.0
"""The indexer's fused RoPE + Q quantisation, transcribed rather than derived.

Upstream dispatches this on a blacklist -- cutedsl, else xpu, else Triton --
and triton-ascend aborts compiling the Triton one. abort(), not an exception:
four workers gone in the same second with exit code None and no traceback.

Everything here pins a detail that would have been wrong if it had been
derived from the docstring rather than read off the kernel, and every one of
them is silent when wrong.

**This file was written against the fp8 branch of that kernel and the
implementation now transcribes the MXFP4 one.** What that cost, per case, is
recorded at each rewrite; the shape of the change is:

* fp8 Q had one per-(token, head) scale, and it was folded into
  ``weights_out`` because the stored values had nowhere to carry it. So the
  scale was *observable from the return value*, and half this file recovered
  it by dividing the weights back out. MXFP4 scales are per 32-element block,
  travel with the values, and are **not** folded -- there is nothing left to
  divide out, and ``AMAX_FLOOR`` / ``FP8_MAX`` have no counterpart in this
  path at all (the E8M0 block scale is the operator's, not ours).
* the scale is therefore computed inside ``npu_dynamic_mx_quant``, which has
  no CPU kernel ("Could not run 'npu::npu_dynamic_mx_quant' with arguments
  from the 'CPU' backend"). Everything this path still decides is either in
  the tensor it hands that operator or in how it reshapes what comes back, so
  that is what the ``quantiser`` fixture captures. A test that needed the
  quantised bytes themselves would need a card and does not belong here.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import indexer_q
from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import (SCALE_GROUP,
                                                                 SCALE_PAIR,
                                                                 VALUES_PER_BYTE
                                                                 )
from vllm_ascend.models.deepseek_v4_1.indexer_q import \
    fused_indexer_q_rope_quant_ascend
from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice

T, H, HEAD_DIM, HALF = 4, 3, 128, 32
ROT = 2 * HALF
NOPE = HEAD_DIM - ROT
SOFTMAX, HEAD = 0.125, 0.5


def _inputs(seed=0, scale=1.0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    return dict(
        positions=torch.arange(T),
        index_q=(torch.randn(T, H, HEAD_DIM) * scale).to(dtype),
        index_q_cos_sin_cache=torch.randn(64, ROT),
        index_weights=torch.rand(T, H),
        index_weights_softmax_scale=SOFTMAX,
        index_weights_head_scale=HEAD,
    )


@pytest.fixture
def quantiser(monkeypatch):
    """Capture what reaches ``npu_dynamic_mx_quant``, since it has no CPU
    kernel. Returns the list of tensors handed over, in call order.

    The stand-in returns zeros of the operator's shapes -- ``(T, H, D/2)``
    packed bytes and ``(T, H, D/32)`` E8M0 scales -- so the reshaping the
    caller does to them is still exercised.
    """
    seen = []

    def _spy(x):
        seen.append(x.clone())
        t, h, d = x.shape
        return (torch.zeros(t, h, d // VALUES_PER_BYTE, dtype=torch.uint8),
                torch.zeros(t, h, d // SCALE_GROUP, dtype=torch.uint8))

    monkeypatch.setattr(indexer_q, "_mx_quantise", _spy)
    return seen


def _expected_q_roped(args):
    """The tensor the quantiser must receive: NoPE untouched, tail rotated
    interleaved, tail round-tripped through bf16."""
    cs = args["index_q_cos_sin_cache"].index_select(
        0, args["positions"]).float()
    cos, sin = cs[:, :HALF].unsqueeze(1), cs[:, HALF:].unsqueeze(1)
    q = args["index_q"]
    tail = q[..., NOPE:].float().reshape(T, H, HALF, 2)
    even, odd = tail[..., 0], tail[..., 1]
    r_even = (even * cos - odd * sin).to(torch.bfloat16).float()
    r_odd = (odd * cos + even * sin).to(torch.bfloat16).float()
    out = q.clone()
    out[..., NOPE:] = torch.stack((r_even, r_odd),
                                  dim=-1).reshape(T, H, ROT).to(q.dtype)
    return out


# -- what the path still decides -----------------------------------------------


def test_the_rotation_covers_the_trailing_dims_only(quantiser):
    """The NoPE part is ahead of the rotated part, not after it.

    Rewritten: the fp8 version recovered the scale from the weights and
    compared stored fp8 bytes. There is no recoverable scale any more, so the
    comparison moved to the tensor handed to the quantiser -- which is where
    this path's choice actually lives. Bit-exact, not a tolerance: the NoPE
    dims must arrive as ``index_q`` had them.
    """
    args = _inputs(2)
    fused_indexer_q_rope_quant_ascend(**args)
    q_roped, = quantiser
    assert torch.equal(q_roped[..., :NOPE], args["index_q"][..., :NOPE])
    assert not torch.equal(q_roped[..., NOPE:], args["index_q"][..., NOPE:]), (
        "the trailing dims must move, or this test proves nothing")


def test_the_pairing_is_interleaved_not_split_half(quantiser):
    """x[2k] with x[2k+1]. Split-half is the other plausible reading and it
    produces a complete, wrongly-rotated tensor."""
    args = _inputs(3)
    fused_indexer_q_rope_quant_ascend(**args)
    q_roped, = quantiser
    assert torch.equal(q_roped, _expected_q_roped(args))

    # and the split-half reading is genuinely different, not a coincidence
    cs = args["index_q_cos_sin_cache"].index_select(
        0, args["positions"]).float()
    cos, sin = cs[:, :HALF].unsqueeze(1), cs[:, HALF:].unsqueeze(1)
    tail = args["index_q"][..., NOPE:].float()
    a, b = tail[..., :HALF], tail[..., HALF:]
    split = torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)
    assert not torch.allclose(q_roped[..., NOPE:].float(), split, atol=1e-2)


def test_the_bf16_round_trip_happens_before_the_quantiser(quantiser):
    """The rotated values go fp32 -> bf16 -> fp32 *before* they are quantised,
    so the block scale describes what will be stored (``indexer_q.py:21-24``).

    This is the fp8-era ``test_the_scale_is_taken_after_the_bf16_round_trip``,
    and it is the one case the format change made *harder* rather than
    pointless: the scale used to be readable out of ``weights_out``, and now it
    is computed inside an operator with no CPU kernel. So the contract is
    pinned one step earlier, on the operand -- which is the only place this
    file can still hold it, and the place the RoPE-operator swap will change.

    ``index_q`` is deliberately **fp32** here. At bf16 the round trip is
    invisible (the store back into a bf16 ``q_roped`` rounds identically), so a
    bf16 input cannot tell an implementation that dropped it from one that
    kept it -- the assertion would pass either way, which is the failure mode
    this file is supposed to catch, not exhibit.
    """
    args = _inputs(4, scale=30.0, dtype=torch.float32)
    fused_indexer_q_rope_quant_ascend(**args)
    q_roped, = quantiser
    assert q_roped.dtype is torch.float32

    tail = q_roped[..., NOPE:]
    assert torch.equal(tail, tail.to(torch.bfloat16).float()), (
        "every rotated value must already be bf16-representable; it is not, "
        "so the round trip was skipped or moved after the quantiser")
    assert torch.equal(q_roped, _expected_q_roped(args))

    # the same rotation without the round trip differs -- otherwise the
    # assertion above is vacuous for these inputs
    cs = args["index_q_cos_sin_cache"].index_select(
        0, args["positions"]).float()
    cos, sin = cs[:, :HALF].unsqueeze(1), cs[:, HALF:].unsqueeze(1)
    pairs = args["index_q"][..., NOPE:].reshape(T, H, HALF, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    raw = torch.stack((even * cos - odd * sin, odd * cos + even * sin),
                      dim=-1).reshape(T, H, ROT)
    assert not torch.equal(tail, raw)


def test_the_nope_half_is_not_round_tripped(quantiser):
    """Neither kernel round-trips the NoPE half (``indexer_q.py:24``), and at
    fp32 that is observable: a value that is not bf16-representable must
    survive intact.

    This replaces the fp8-era ``test_the_nope_part_participates_in_the_absmax``.
    That test is **not** rewritten to pass: its premise is gone. fp8 had one
    scale per (token, head) covering all 128 dims, so a large NoPE value had to
    raise it or the NoPE values would clip. MXFP4 scales one 32-element block
    at a time, so the NoPE blocks and the rotated blocks no longer share a
    scale and cannot influence each other -- asserting they do would assert the
    opposite of the format. What survives of it is the coupling test below.
    """
    args = _inputs(5, dtype=torch.float32)
    # a fp32 value with mantissa bits bf16 cannot hold
    args["index_q"][0, 0, 0] = 1.0000001192092896
    fused_indexer_q_rope_quant_ascend(**args)
    q_roped, = quantiser
    assert q_roped[0, 0, 0].item() == args["index_q"][0, 0, 0].item()
    assert not torch.equal(q_roped[0, 0, :1],
                           q_roped[0, 0, :1].to(torch.bfloat16).float())


def test_a_nope_value_no_longer_moves_the_rotated_bytes(quantiser):
    """MXFP4 block locality, stated as the thing that changed: under fp8 a
    single huge NoPE value rescaled the whole head, so every stored byte in it
    moved. Here it must reach the quantiser in its own 32-element block with
    the rotated tail bit-identical -- if the tail moved, something is still
    computing a head-wide scale somewhere upstream of the operator.
    """
    args = _inputs(6, scale=0.01)
    fused_indexer_q_rope_quant_ascend(**args)
    args["index_q"] = args["index_q"].clone()
    args["index_q"][0, 0, 0] = 100.0
    fused_indexer_q_rope_quant_ascend(**args)
    quiet, loud = quantiser
    assert torch.equal(quiet[..., NOPE:], loud[..., NOPE:])
    assert loud[0, 0, 0].item() == 100.0
    assert torch.equal(quiet[0, 0, 1:NOPE], loud[0, 0, 1:NOPE])


def test_the_weights_carry_no_q_scale(quantiser):
    """The inverse of the fp8-era ``test_the_weight_fold_carries_the_q_scale``.

    fp8 folded its one per-(token, head) scale into these weights because the
    stored values had nowhere to carry it. MXFP4 scales travel with the values,
    so folding anything here multiplies every logit by a spurious factor with
    no shape or dtype to show for it (``indexer_q.py:29-37``). Exact equality,
    and independence from ``index_q``, is what says no fold crept back in.
    """
    args = _inputs(7)
    _, weights_out = fused_indexer_q_rope_quant_ascend(**args)
    torch.testing.assert_close(
        weights_out,
        args["index_weights"].float() * SOFTMAX * HEAD,
        rtol=0, atol=0)

    args["index_q"] = args["index_q"] * 1000.0
    _, again = fused_indexer_q_rope_quant_ascend(**args)
    assert torch.equal(weights_out, again), (
        "weights_out moved with index_q, so something is folding a scale")


def test_the_returned_scale_axis_is_split_into_pairs(quantiser):
    """``npu_dynamic_mx_quant`` hands back ``(T, H, D/32)``; a mode 3/5 scale
    for ``npu_quant_lightning_indexer_v2`` must be ``(T, H, D/64, 2)``
    (``indexer_q.py:64-68``). Same 4 bytes, and passing the operator the
    unsplit form is a shape error at the far end of the graph, not here.
    """
    args = _inputs(8)
    (q_packed, q_scale), _ = fused_indexer_q_rope_quant_ascend(**args)
    assert q_packed.shape == (T, H, HEAD_DIM // VALUES_PER_BYTE)
    assert q_packed.dtype is torch.uint8
    assert q_scale.shape == (T, H, HEAD_DIM // (SCALE_GROUP * SCALE_PAIR),
                             SCALE_PAIR)
    assert q_scale.dtype is torch.uint8


def test_use_fp4_is_accepted_and_ignored(quantiser):
    """Replaces the fp8-era ``test_the_mxfp4_path_is_refused_rather_than_
    approximated``, which asserted a ``NotImplementedError`` that now would be
    a permanently unreachable path: ``use_fp4`` comes from
    ``dsa_indexer_uses_fp4``, which raises outside Blackwell, while the index K
    cache this Q is matched against is MXFP4 unconditionally
    (``indexer_q.py:70-76``). So the flag must not select anything -- both
    calls have to produce the same bytes.
    """
    args = _inputs(9)
    (p0, s0), w0 = fused_indexer_q_rope_quant_ascend(**args)
    (p1, s1), w1 = fused_indexer_q_rope_quant_ascend(**args, use_fp4=True)
    assert torch.equal(p0, p1) and torch.equal(s0, s1)
    assert torch.equal(w0, w1)
    first, second = quantiser
    assert torch.equal(first, second)


def test_a_head_dim_that_is_not_a_whole_number_of_scale_pairs_is_refused():
    """MXFP4 needs whole 32-element blocks and the operator needs them in
    pairs. A head_dim of 96 reshapes without complaint into the packed form and
    only the scale axis is wrong, which surfaces far from here."""
    args = _inputs(10)
    args["index_q"] = args["index_q"][..., :96].contiguous()
    with pytest.raises(ValueError, match="64"):
        fused_indexer_q_rope_quant_ascend(**args)


def test_a_cos_sin_cache_wider_than_the_head_is_refused():
    """nope_dim < 0 would slice to an empty NoPE part and rotate a tail
    narrower than the cache, silently."""
    args = _inputs(11)
    args["index_q_cos_sin_cache"] = torch.randn(64, 2 * HEAD_DIM)
    with pytest.raises(ValueError, match="smaller than the rotation width"):
        fused_indexer_q_rope_quant_ascend(**args)


def test_the_rebind_reaches_the_call_site_module():
    import vllm.models.deepseek_v4_1.attention as attention

    from vllm_ascend.models.deepseek_v4_1.indexer_q import \
        apply_indexer_q_patch

    apply_indexer_q_patch()
    assert (attention.fused_indexer_q_rope_quant
            is fused_indexer_q_rope_quant_ascend)
    assert apply_indexer_q_patch() is False, "must be idempotent"


# -- what this reference is for once the operator path lands ------------------
#
# The plan is to route the indexer at Ascend's own stack, which does its own
# rotation via inplace_partial_rotary_mul with two parameters:
#
#     partial_slice = [nope_dim, head_dim)      rotate only the tail
#     rotary_mode   = "interleave"              pair x[2k] with x[2k+1]
#
# Get either wrong and the output has exactly the right shape and dtype and
# the wrong numbers -- partial_slice covering the whole head silently rotates
# the NoPE part, and "half" pairs x[k] with x[k + d/2]. Neither raises.
#
# So this reference stops being an implementation and becomes the thing those
# two parameters are checked against. The checks below need no card: they pin
# what the parameters *mean*, against an implementation already verified
# against the kernel they replace.


def _rotate_like_operator(x, cos, sin, start, end, mode):
    """What inplace_partial_rotary_mul does, for a given slice and mode.

    ``cos``/``sin`` are **slice-wide** -- ``cos.size(-1) == end - start``, one
    per channel, which is what the operator enforces. The earlier version of
    this helper took them at half that width (one per *pair*) and broadcast
    them over the reshaped pairs; that is upstream's cache layout, not the
    operator's argument, so it quietly checked a rotation nobody performs. The
    widening from one to the other is ``rope_widen.cos_sin_for_slice``, and it
    is not free to choose: repeat_interleave for interleaved pairing, plain
    repeat for half pairing (``rope_widen.py:9-23``).
    """
    assert cos.shape[-1] == end - start, (cos.shape, start, end)
    out = x.clone().float()
    seg = out[..., start:end]
    half = (end - start) // 2
    if mode == "interleave":
        pairs = seg.reshape(*seg.shape[:-1], half, 2)
        c = cos.reshape(*cos.shape[:-1], half, 2)
        s = sin.reshape(*sin.shape[:-1], half, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        rotated = torch.stack(
            (a * c[..., 0] - b * s[..., 0], b * c[..., 1] + a * s[..., 1]),
            dim=-1)
        out[..., start:end] = rotated.reshape(*seg.shape)
    elif mode == "half":
        a, b = seg[..., :half], seg[..., half:]
        ca, cb = cos[..., :half], cos[..., half:]
        sa, sb = sin[..., :half], sin[..., half:]
        out[..., start:end] = torch.cat((a * ca - b * sa, b * cb + a * sb), -1)
    else:
        raise AssertionError(mode)
    return out


def _slice_cos_sin(args, width=ROT):
    """The operator's cos/sin for the tail slice, at the width it wants."""
    cos, sin = cos_sin_for_slice(args["index_q_cos_sin_cache"],
                                 args["positions"], width)
    return cos.unsqueeze(1), sin.unsqueeze(1)


def _reference_rotation(args):
    """The rotation half of the verified implementation, pulled out."""
    cs = args["index_q_cos_sin_cache"].index_select(
        0, args["positions"]).float()
    cos, sin = cs[:, :HALF].unsqueeze(1), cs[:, HALF:].unsqueeze(1)
    q = args["index_q"].float()
    tail = q[..., NOPE:].reshape(T, H, HALF, 2)
    even, odd = tail[..., 0], tail[..., 1]
    out = q.clone()
    out[..., NOPE:] = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin),
        dim=-1).reshape(T, H, ROT)
    return out


def test_partial_slice_and_interleave_reproduce_the_reference():
    """The parameter pair the operator path will pass, checked as semantics."""
    args = _inputs(20)
    cos, sin = _slice_cos_sin(args)
    got = _rotate_like_operator(args["index_q"], cos, sin,
                                start=NOPE, end=HEAD_DIM, mode="interleave")
    torch.testing.assert_close(got, _reference_rotation(args),
                               rtol=1e-5, atol=1e-5)


def test_a_slice_covering_the_whole_head_rotates_the_nope_part():
    """partial_slice = [0, head_dim) is the natural wrong value, and it is
    silent: same shape, same dtype, the NoPE dimensions quietly rotated."""
    args = _inputs(21)
    # a full-width slice needs cos/sin as wide as the head; the cache only has
    # HALF pairs, so the mistake takes the shape of tiling them over 128
    # channels -- which is what makes the NoPE part rotate at all.
    cos, sin = _slice_cos_sin(args)
    wide_cos = cos.repeat(1, 1, HEAD_DIM // ROT)
    wide_sin = sin.repeat(1, 1, HEAD_DIM // ROT)

    wrong = _rotate_like_operator(args["index_q"], wide_cos, wide_sin,
                                  start=0, end=HEAD_DIM, mode="interleave")
    reference = _reference_rotation(args)
    assert not torch.allclose(wrong[..., :NOPE], reference[..., :NOPE],
                              atol=1e-3), (
        "the NoPE part must move, or this test is not demonstrating the hazard")


def test_half_pairing_differs_from_interleave():
    """"half" is the other mode name and pairs x[k] with x[k + d/2]. It
    produces a complete, wrongly-rotated tail.

    Note the widening differs with the mode: half pairing takes the cache's
    half repeated (rope_widen's option B), interleaved takes it
    repeat_interleaved (A). Both are ROT wide, which is exactly why neither
    the operator nor anything downstream can tell them apart.
    """
    args = _inputs(22)
    cos, sin = _slice_cos_sin(args)
    cs = args["index_q_cos_sin_cache"].index_select(
        0, args["positions"]).float()
    half_cos = cs[:, :HALF].repeat(1, 2).unsqueeze(1)
    half_sin = cs[:, HALF:].repeat(1, 2).unsqueeze(1)

    interleaved = _rotate_like_operator(args["index_q"], cos, sin,
                                        NOPE, HEAD_DIM, "interleave")
    half = _rotate_like_operator(args["index_q"], half_cos, half_sin,
                                 NOPE, HEAD_DIM, "half")
    assert interleaved.shape == half.shape
    assert not torch.allclose(interleaved, half, atol=1e-3)
    torch.testing.assert_close(interleaved, _reference_rotation(args),
                               rtol=1e-5, atol=1e-5)


def test_the_cos_sin_width_matches_the_slice_not_the_head():
    """The operator requires cos.size(-1) == end - start: the *slice* width,
    which for this q is ROT and not HEAD_DIM. The cache carries half of that
    (one value per interleaved pair), so the widening is mandatory -- passing
    the cache row straight through is a shape error only when ROT happens to
    differ from HALF, and here it does.
    """
    assert ROT == HEAD_DIM - NOPE
    args = _inputs(23)
    cos, sin = _slice_cos_sin(args)
    assert cos.shape[-1] == ROT == 2 * HALF
    assert sin.shape[-1] == ROT
    with pytest.raises(AssertionError):
        _rotate_like_operator(args["index_q"],
                              cos[..., :HALF], sin[..., :HALF],
                              NOPE, HEAD_DIM, "interleave")
