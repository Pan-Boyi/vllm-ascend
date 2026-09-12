# SPDX-License-Identifier: Apache-2.0
"""Indexer K production: the gate, the rotation, and what reaches the operator.

Every check here is against upstream's Triton kernel read as the specification
(``vllm/models/deepseek_v4_1/common/ops/indexer_k_store.py``), because this is
a transcription. The parts worth testing are the ones that are silent when
wrong: which tokens publish a key, which position the rotation uses, how wide
the rotation is, and what a skipped token's slot looks like by the time the
store operator sees it.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import indexer_k_store as mod
from vllm_ascend.models.deepseek_v4_1.indexer_k_store import (
    INDEX_HEAD_DIM, INDEX_ROPE_HEAD_DIM, SKIP_SLOT, group_boundary_slots,
    indexer_k_norm_rope_store_ascend)


def test_the_rope_is_narrower_than_the_head():
    """Two different numbers, and reading one as the other is silent.

    128 is the head width and 64 the rope width. Rotating all 128 also rotates
    the NoPE half, which changes every logit and raises nothing.
    """
    assert INDEX_HEAD_DIM == 128
    assert INDEX_ROPE_HEAD_DIM == 64
    assert INDEX_ROPE_HEAD_DIM < INDEX_HEAD_DIM


@pytest.mark.parametrize("ratio,expect", [
    (1, [0, 1, 2, 3, 4, 5]),
    (2, [SKIP_SLOT, 1, SKIP_SLOT, 3, SKIP_SLOT, 5]),
])
def test_only_the_last_token_of_a_group_publishes(ratio, expect):
    """(position + 1) % ratio == 0, exactly as the kernel gates."""
    slots = torch.arange(6, dtype=torch.int32)
    positions = torch.arange(6, dtype=torch.int64)
    got = group_boundary_slots(slots, positions, ratio)
    assert got.tolist() == expect


def test_the_gate_reads_positions_not_token_order():
    """Decode batches start mid-sequence; a row index is not a position."""
    slots = torch.tensor([10, 11, 12], dtype=torch.int32)
    positions = torch.tensor([5, 6, 7], dtype=torch.int64)
    # ratio 2: publishes where (p+1) % 2 == 0, i.e. p = 5 and p = 7.
    assert group_boundary_slots(slots, positions, 2).tolist() == [10, -1, 12]


def test_an_already_invalid_slot_stays_skipped():
    slots = torch.tensor([-1, 3], dtype=torch.int32)
    positions = torch.tensor([1, 3], dtype=torch.int64)
    assert group_boundary_slots(slots, positions, 2).tolist() == [-1, 3]


def test_every_skipped_token_is_exactly_minus_one():
    """The operator tests `slot == -1`, not `slot < 0`.

    Upstream's kernel returns on any negative, so a builder that used -2 as its
    own sentinel would be correct there and would be divided by the block size
    into a negative block index here -- a write outside the cache, with no
    error. This is the check that a future slot builder cannot quietly break.
    """
    slots = torch.tensor([-7, -2, -1, 4], dtype=torch.int32)
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    got = group_boundary_slots(slots, positions, 2)
    negatives = got[got < 0]
    assert torch.all(negatives == SKIP_SLOT), got.tolist()


def test_ratio_zero_is_refused_rather_than_dividing_by_it():
    with pytest.raises(ValueError, match="no group boundary"):
        group_boundary_slots(torch.zeros(2, dtype=torch.int32),
                             torch.zeros(2, dtype=torch.int64), 0)


class _RecordingEpilog:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append({k: (v.clone() if isinstance(v, torch.Tensor) else v)
                           for k, v in kwargs.items()})


@pytest.fixture
def epilog(monkeypatch):
    rec = _RecordingEpilog()

    class _Ops:
        indexer_compress_epilog_v2 = rec

    monkeypatch.setattr(torch.ops, "_C_ascend", _Ops, raising=False)
    return rec


def _store(epilog, *, num_tokens=4, ratio=2, slots=None, positions=None,
           cos_sin=None):
    k_pre = torch.randn(num_tokens, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    if slots is None:
        slots = torch.arange(num_tokens, dtype=torch.int32)
    if positions is None:
        positions = torch.arange(num_tokens, dtype=torch.int64)
    if cos_sin is None:
        cos_sin = torch.zeros(64, INDEX_ROPE_HEAD_DIM, dtype=torch.float32)
        cos_sin[:, :INDEX_ROPE_HEAD_DIM // 2] = 1.0  # cos=1, sin=0: identity
    indexer_k_norm_rope_store_ascend(
        k_pre, positions, cos_sin,
        torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16), 1e-6,
        torch.zeros(2, 64, 640, dtype=torch.uint8), slots, ratio, False)
    return k_pre


def test_the_gate_reaches_the_operator_as_the_slot_mapping(epilog):
    _store(epilog, num_tokens=4, ratio=2)
    (call,) = epilog.calls
    assert call["slot_mapping"].tolist() == [-1, 1, -1, 3]
    assert call["layout"] == 2


def test_an_identity_rotation_leaves_the_normed_values(epilog):
    """cos=1, sin=0 makes the rotation a no-op, so what is stored is k_norm.

    Separates two things that would otherwise only be testable together: if the
    rotation is applied to the wrong slice, or to the wrong positions, this
    still passes -- and the next two tests are the ones that catch that.
    """
    k_pre = _store(epilog, num_tokens=2, ratio=1)
    (call,) = epilog.calls
    acc = k_pre.to(torch.float32)
    var = acc.pow(2).mean(-1, keepdim=True)
    expect = (acc * torch.rsqrt(var + 1e-6)).to(torch.bfloat16)
    assert torch.equal(call["x"], expect)


def test_the_rotation_leaves_the_nope_half_alone(epilog):
    """A non-identity rotation must not touch the leading 64."""
    cos_sin = torch.zeros(64, INDEX_ROPE_HEAD_DIM, dtype=torch.float32)
    cos_sin[:, :INDEX_ROPE_HEAD_DIM // 2] = 0.5   # cos
    cos_sin[:, INDEX_ROPE_HEAD_DIM // 2:] = 0.25  # sin
    k_pre = _store(epilog, num_tokens=1, ratio=1, cos_sin=cos_sin)
    (call,) = epilog.calls
    nope = INDEX_HEAD_DIM - INDEX_ROPE_HEAD_DIM
    acc = k_pre.to(torch.float32)
    var = acc.pow(2).mean(-1, keepdim=True)
    normed = (acc * torch.rsqrt(var + 1e-6)).to(torch.bfloat16).to(torch.float32)
    assert torch.equal(call["x"][:, :nope], normed[:, :nope].to(torch.bfloat16))
    assert not torch.equal(call["x"][:, nope:],
                           normed[:, nope:].to(torch.bfloat16))


def test_the_rotation_uses_the_groups_first_position(epilog):
    """compressed_pos = (position // ratio) * ratio.

    A latent stands for the first token of its group, so both tokens of a
    ratio-2 group rotate at the even position. Using `position` instead gives
    every other key a rotation one step ahead -- values that are wrong by an
    angle, not by a shape.
    """
    # Distinct rows so "which row was read" is observable.
    cos_sin = torch.zeros(8, INDEX_ROPE_HEAD_DIM, dtype=torch.float32)
    for p in range(8):
        cos_sin[p, :INDEX_ROPE_HEAD_DIM // 2] = float(p)

    torch.manual_seed(0)
    k_pre = torch.randn(2, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    cache = torch.zeros(2, 64, 640, dtype=torch.uint8)
    # ratio 1 so both tokens publish; positions 2 and 3 are one group at ratio
    # 2 but distinct rows at ratio 1 -- run both and compare.
    for ratio, positions in ((1, [2, 3]), (2, [2, 3])):
        epilog.calls.clear()
        indexer_k_norm_rope_store_ascend(
            k_pre, torch.tensor(positions, dtype=torch.int64), cos_sin,
            torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16), 1e-6,
            cache, torch.tensor([0, 1], dtype=torch.int32), ratio, False)
        if ratio == 1:
            at_two, at_three = epilog.calls[0]["x"][0], epilog.calls[0]["x"][1]
        else:
            # Only position 3 publishes; its group starts at 2, so it must
            # match the row that position 2 produced under ratio 1 -- same
            # angle, different token.
            published = epilog.calls[0]["x"][1]
    assert not torch.equal(at_two, at_three), "rows 2 and 3 must differ"
    # Same k_pre row (index 1), rotated at position 2 rather than 3.
    torch.manual_seed(0)
    ref = torch.randn(2, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    assert torch.equal(k_pre, ref)
    rotated_at_two = mod._rope_interleaved_tail(
        mod._rms_norm(k_pre[1:2], torch.ones(INDEX_HEAD_DIM), 1e-6), cos_sin,
        torch.tensor([2]), INDEX_ROPE_HEAD_DIM).to(torch.bfloat16)
    assert torch.equal(published.unsqueeze(0), rotated_at_two)


def test_mxfp4_says_upstream_cannot_reach_it_either(epilog):
    with pytest.raises(NotImplementedError, match="Blackwell"):
        indexer_k_norm_rope_store_ascend(
            torch.zeros(1, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            torch.zeros(1, dtype=torch.int64),
            torch.zeros(4, INDEX_ROPE_HEAD_DIM),
            torch.ones(INDEX_HEAD_DIM), 1e-6,
            torch.zeros(2, 64, 640, dtype=torch.uint8),
            torch.zeros(1, dtype=torch.int32), 1, True)


def test_a_block_offset_slot_mapping_without_its_page_size_is_refused(epilog):
    """This used to refuse the [T, 2] form outright, and that was right while
    nothing here could know the page size it was split with.

    It is now converted -- but only when the page size can be taken from the
    metadata that is carrying this exact tensor. With no forward context there
    is no such metadata, and the refusal stands rather than a plausible
    constant being substituted: a (block, offset) pair is valid arithmetic
    under any page size, and the wrong one lands on another token's slot
    without leaving the cache.

    The conversion itself is pinned in test_deepseek_v4_1_indexer_slot_form.py.
    """
    with pytest.raises(ValueError, match="attention metadata"):
        indexer_k_norm_rope_store_ascend(
            torch.zeros(2, INDEX_HEAD_DIM, dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(4, INDEX_ROPE_HEAD_DIM),
            torch.ones(INDEX_HEAD_DIM), 1e-6,
            torch.zeros(2, 64, 640, dtype=torch.uint8),
            torch.zeros((2, 2), dtype=torch.int32), 1, False)


def test_an_empty_batch_does_not_call_the_operator(epilog):
    indexer_k_norm_rope_store_ascend(
        torch.zeros(0, INDEX_HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(4, INDEX_ROPE_HEAD_DIM),
        torch.ones(INDEX_HEAD_DIM), 1e-6,
        torch.zeros(2, 64, 640, dtype=torch.uint8),
        torch.zeros(0, dtype=torch.int32), 1, False)
    assert not epilog.calls


def test_the_norm_round_trips_through_bf16_before_the_rotation(epilog):
    """Upstream's k_norm ends `.to(bfloat16).to(float32)`, and it is load-bearing.

    It changes no shape and no dtype, so nothing downstream complains; what it
    changes is the precision the rotation starts from, and therefore the bytes
    that reach the quantiser. A test that stores an unrotated value cannot see
    it -- the final cast to bf16 hides it -- so this one rotates by a non-
    trivial angle and compares against both candidates.
    """
    torch.manual_seed(7)
    k_pre = torch.randn(64, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    half = INDEX_ROPE_HEAD_DIM // 2
    cos_sin = torch.empty(4, INDEX_ROPE_HEAD_DIM, dtype=torch.float32)
    angles = torch.linspace(0.1, 1.4, half)
    cos_sin[:, :half] = torch.cos(angles)
    cos_sin[:, half:] = torch.sin(angles)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.bfloat16).abs() + 0.5

    indexer_k_norm_rope_store_ascend(
        k_pre, torch.zeros(64, dtype=torch.int64), cos_sin, weight, 1e-6,
        torch.zeros(2, 64, 640, dtype=torch.uint8),
        torch.arange(64, dtype=torch.int32), 1, False)
    (call,) = epilog.calls

    def _reference(round_trip: bool):
        acc = k_pre.to(torch.float32)
        var = acc.pow(2).mean(-1, keepdim=True)
        normed = acc * torch.rsqrt(var + 1e-6) * weight.to(torch.float32)
        if round_trip:
            normed = normed.to(torch.bfloat16).to(torch.float32)
        return mod._rope_interleaved_tail(
            normed, cos_sin, torch.zeros(64, dtype=torch.int64),
            INDEX_ROPE_HEAD_DIM).to(torch.bfloat16)

    with_rt, without_rt = _reference(True), _reference(False)
    assert not torch.equal(with_rt, without_rt), (
        "the two candidates must differ, or this test proves nothing")
    assert torch.equal(call["x"], with_rt)


def test_the_post_rotation_round_trip_is_a_no_op_on_this_route(epilog):
    """Stated as a test because the code keeps a line that does nothing.

    Upstream quantises from the fp32 value the rotation produces, so its bf16
    round trip there decides the scale. This route hands bf16 to the store
    operator and lets it quantise, and bf16(fp32(bf16(x))) == bf16(x) -- so the
    rounding happens either way. If that ever stops being true (an fp32 store
    input, a different operator), this goes red and the kept line starts
    mattering again.
    """
    half = INDEX_ROPE_HEAD_DIM // 2
    cos_sin = torch.empty(4, INDEX_ROPE_HEAD_DIM, dtype=torch.float32)
    angles = torch.linspace(0.3, 1.1, half)
    cos_sin[:, :half] = torch.cos(angles)
    cos_sin[:, half:] = torch.sin(angles)
    torch.manual_seed(11)
    k_pre = torch.randn(32, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16)

    indexer_k_norm_rope_store_ascend(
        k_pre, torch.zeros(32, dtype=torch.int64), cos_sin, weight, 1e-6,
        torch.zeros(2, 64, 640, dtype=torch.uint8),
        torch.arange(32, dtype=torch.int32), 1, False)
    (call,) = epilog.calls

    # The same rotation with no intermediate rounding, cast once at the end.
    normed = mod._rms_norm(k_pre, weight, 1e-6)
    nope = INDEX_HEAD_DIM - INDEX_ROPE_HEAD_DIM
    tail = normed[:, nope:].reshape(-1, half, 2)
    cos, sin = cos_sin[0, :half], cos_sin[0, half:]
    even, odd = tail[..., 0], tail[..., 1]
    unrounded = normed.clone()
    unrounded[:, nope:] = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin),
        dim=-1).reshape(-1, INDEX_ROPE_HEAD_DIM)
    assert torch.equal(call["x"], unrounded.to(torch.bfloat16))


def test_the_patch_is_installed_by_apply_all():
    """The hook, not just the function.

    A correct replacement that nothing binds is indistinguishable from no
    replacement at all -- the layer keeps calling upstream's Triton kernel and
    the process aborts with no traceback.
    """
    import vllm.models.deepseek_v4_1.attention as upstream

    from vllm_ascend.models.deepseek_v4_1 import platform_patches

    platform_patches.apply_all()
    assert upstream.indexer_k_norm_rope_store is indexer_k_norm_rope_store_ascend


def test_a_vanished_name_is_refused_rather_than_skipped(monkeypatch):
    import vllm.models.deepseek_v4_1.attention as upstream

    monkeypatch.delattr(upstream, "indexer_k_norm_rope_store")
    with pytest.raises(AttributeError, match="aborts the process"):
        mod.apply_indexer_k_store_patch()
