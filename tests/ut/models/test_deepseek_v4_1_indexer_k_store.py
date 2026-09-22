# SPDX-License-Identifier: Apache-2.0
"""Indexer K production: the gate, the norm, the rotation, and what reaches the
operator.

Every check here is against upstream's Triton kernel read as the specification
(``vllm/models/deepseek_v4_1/common/ops/indexer_k_store.py``), because this is
a transcription. The parts worth testing are the ones that are silent when
wrong: which tokens publish a key, which position the rotation uses, how wide
the rotation is, where the bf16 rounding happens, and what a skipped token's
slot looks like by the time the store operator sees it.

**This file was written against the fp8 row and ``indexer_compress_epilog_v2``;
since batch 1 (ada1cc48e) the row is MXFP4 and the writer is
``indexer_quant_cache``.** Two things went stale with it, and both are fixture
problems rather than findings about the code:

* the fixtures built a 132-byte row from ``indexer_k_store.SCALE_GROUP`` /
  ``SCALE_BYTES`` -- which are the *epilog's* fp8 constants (128 values + one
  fp32 per 128) and are still in that module for the rollback. The MXFP4
  writer splits the row with ``indexer_cache_view``'s constants (two values
  per byte, one E8M0 byte per 32 values: 64 + 4 = 68) and refuses 132, which
  is right: 132 is not a whole number of 17-byte groups, so any split of it
  lands between a value and its scale. The width is now taken from
  ``indexer_cache_view`` and the refusal is pinned below as behaviour;
* the fixtures stubbed ``indexer_compress_epilog_v2``, which the default path
  no longer calls. They now capture the ``indexer_quant_cache`` call instead;
  it has no CPU kernel, so -- as in ``test_deepseek_v4_1_indexer_q.py`` --
  what is pinned is the operand this path hands it, not the quantised bytes.

A third staleness is interface drift: the ratio-2 slot rebuild started reading
``req_metadata.seq_lens`` (069c7d3e9, padded tokens under graph replay) and the
fake metadata did not carry it.

What this file cannot see is anything computed *inside* an NPU operator:
``indexer_quant_cache``'s MXFP4 bytes, ``npu_rms_norm``'s rounding, and
``inplace_partial_rotary_mul``'s fp32 arithmetic. The CPU route is the torch
form in every case (the fused branches key on ``is_npu``), and
``test_the_fused_switches_do_not_move_the_cpu_result`` is what says so.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import indexer_k_store as mod
from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import (
    scale_bytes_per_token, value_bytes_per_token)
from vllm_ascend.models.deepseek_v4_1.indexer_k_store import (
    INDEX_HEAD_DIM, INDEX_ROPE_HEAD_DIM, SKIP_SLOT, group_boundary_slots,
    indexer_k_norm_rope_store_ascend)

NOPE = INDEX_HEAD_DIM - INDEX_ROPE_HEAD_DIM
HALF = INDEX_ROPE_HEAD_DIM // 2
EPS = 1e-6

# The MXFP4 row, from the module that splits it -- not from indexer_k_store's
# SCALE_GROUP / SCALE_BYTES, which describe the epilog's fp8 row and are what
# made every fixture here 132 bytes wide after batch 1.
CACHE_WIDTH = (value_bytes_per_token(INDEX_HEAD_DIM)
               + scale_bytes_per_token(INDEX_HEAD_DIM))

# The fp8 row the epilog writes, for the rollback tests only.
FP8_CACHE_WIDTH = INDEX_HEAD_DIM + mod.SCALE_BYTES * -(-INDEX_HEAD_DIM //
                                                        mod.SCALE_GROUP)

_QUANT_FLAG = "VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE"
_NORM_FLAG = "VLLM_ASCEND_DSV41_INDEXER_FUSED_NORM"
_ROPE_FLAG = "VLLM_ASCEND_DSV41_INDEXER_FUSED_ROPE"


def _bound_cache(blocks=2, tokens_per_block=64, width=CACHE_WIDTH):
    """The cache as binding leaves it: rank 3, unit head axis dropped."""
    return torch.zeros(blocks, tokens_per_block, width, dtype=torch.uint8)


# -- independent references ----------------------------------------------------
#
# Written out rather than calling mod._rms_norm / mod._rope_interleaved_tail,
# so that a test comparing the store's operand with these is comparing two
# implementations and not one with itself. The op order matches the module's
# on purpose: the comparisons are bit-exact, and fp32 is not associative.


def _hand_norm(k_pre, weight, round_trip=True):
    acc = k_pre.to(torch.float32)
    var = acc.pow(2).mean(dim=-1, keepdim=True)
    normed = acc * torch.rsqrt(var + EPS) * weight.to(torch.float32)
    if round_trip:
        normed = normed.to(torch.bfloat16).to(torch.float32)
    return normed


def _hand_rope(k, cos_sin, positions):
    """Interleaved (GPT-J) rotation of the trailing 64, fp32, no rounding."""
    row = cos_sin[positions].to(torch.float32)
    cos, sin = row[..., :HALF], row[..., HALF:]
    out = k.clone()
    tail = out[..., NOPE:].reshape(*k.shape[:-1], HALF, 2)
    even, odd = tail[..., 0], tail[..., 1]
    out[..., NOPE:] = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin),
        dim=-1).reshape(*k.shape[:-1], INDEX_ROPE_HEAD_DIM)
    return out


def _angles(rows, lo=0.1, hi=1.4):
    """A cos/sin cache whose every row is a different non-trivial rotation."""
    cos_sin = torch.empty(rows, INDEX_ROPE_HEAD_DIM, dtype=torch.float32)
    for p in range(rows):
        a = torch.linspace(lo, hi, HALF) * (p + 1)
        cos_sin[p, :HALF] = torch.cos(a)
        cos_sin[p, HALF:] = torch.sin(a)
    return cos_sin


# -- the gate ------------------------------------------------------------------


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


# -- fixtures ------------------------------------------------------------------


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        rec = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                # The cache planes are fp4 / e8m0 views; recording where they
                # point and how they are typed is what matters, and cloning
                # an fp4 view is not something CPU torch promises to do.
                rec[k] = v.clone() if k in ("x", "slot_mapping") else v
            else:
                rec[k] = v
        self.calls.append(rec)


class _Req:
    """The builder's published fields, copied from AscendDSAReqMetadata.

    Source: ``vllm_ascend/attention/dsa_v1.py`` ``AscendDSAReqMetadata`` --
    block_table, seq_lens, query_start_loc, storage_block_size and the two
    slot mappings are published together, and the compressed-stream rebuild
    reads all of them. ``seq_lens`` arrived with 069c7d3e9: it is how the
    rebuild tells a padded request row from a real block 0, and without it
    this fake stopped matching the interface.
    """

    def __init__(self, mapping, page, block_table, query_start_loc, seq_lens):
        self.slot_mapping = mapping
        self.token_slot_mapping = mapping
        self.storage_block_size = page
        self.block_table = block_table
        self.query_start_loc = query_start_loc
        self.seq_lens = seq_lens


class _Meta:
    def __init__(self, req):
        self.req_metadata = req


@pytest.fixture
def owns(monkeypatch):
    """Register a slot mapping as belonging to a group, by identity.

    Needed from the moment the ratio-2 path stopped reformatting the mapping
    it was handed and started rebuilding one from that group's block table.

    ``seq_lens`` is required, not defaulted: a default large enough to keep
    every token live would make the padding drop untestable, and a default
    computed from the row count is wrong whenever positions do not start at 0.
    """
    from vllm_ascend.models.deepseek_v4_1 import compressed_slots
    compressed_slots.reset_for_test()
    holder = {"ctx": None}

    class _Ctx:
        attn_metadata = None

    def _register(mapping, *, seq_lens, page=8, blocks=((5, 6, 7, 8), ),
                  starts=None):
        ctx = _Ctx()
        rows = mapping.shape[0]
        ctx.attn_metadata = {
            "indexer": _Meta(_Req(
                mapping, page,
                torch.tensor(blocks, dtype=torch.int32),
                torch.tensor(starts if starts is not None else [0, rows],
                             dtype=torch.int32),
                torch.tensor(seq_lens, dtype=torch.int32)))
        }
        holder["ctx"] = ctx
        return mapping

    import vllm.forward_context as fc
    monkeypatch.setattr(fc, "get_forward_context",
                        lambda: holder["ctx"] or _Ctx())
    return _register


class _TrippedEpilog:
    """Stands in for torch.ops._C_ascend when the epilog must *not* run."""

    @staticmethod
    def indexer_compress_epilog_v2(**_):
        raise AssertionError(
            "the fp8 epilog writer was called on the MXFP4 path")


@pytest.fixture
def quant_cache(monkeypatch):
    """Capture the ``indexer_quant_cache`` call, which has no CPU kernel.

    The flag is set explicitly rather than left to the default, so an ambient
    ``VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE=0`` cannot silently turn every
    test below into a test of the rollback writer.
    """
    rec = _Recorder()
    monkeypatch.setenv(_QUANT_FLAG, "1")
    monkeypatch.setattr(mod, "_indexer_quant_cache_op", lambda: rec)
    monkeypatch.setattr(torch.ops, "_C_ascend", _TrippedEpilog, raising=False)
    return rec


@pytest.fixture
def epilog(monkeypatch):
    """The fp8 rollback writer, with the MXFP4 writer tripped."""
    rec = _Recorder()

    class _Ops:
        indexer_compress_epilog_v2 = rec

    def _tripped():
        raise AssertionError("indexer_quant_cache was reached with the flag off")

    monkeypatch.setenv(_QUANT_FLAG, "0")
    monkeypatch.setattr(mod, "_indexer_quant_cache_op", _tripped)
    monkeypatch.setattr(torch.ops, "_C_ascend", _Ops, raising=False)
    return rec


def _store(*, num_tokens=4, ratio=2, slots=None, positions=None, cos_sin=None,
           k_pre=None, weight=None, cache=None, use_fp4_cache=False):
    if k_pre is None:
        k_pre = torch.randn(num_tokens, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    if slots is None:
        slots = torch.arange(num_tokens, dtype=torch.int32)
    if positions is None:
        positions = torch.arange(num_tokens, dtype=torch.int64)
    if cos_sin is None:
        cos_sin = torch.zeros(64, INDEX_ROPE_HEAD_DIM, dtype=torch.float32)
        cos_sin[:, :HALF] = 1.0  # cos=1, sin=0: identity
    if weight is None:
        weight = torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16)
    indexer_k_norm_rope_store_ascend(
        k_pre, positions, cos_sin, weight, EPS,
        _bound_cache() if cache is None else cache, slots, ratio,
        use_fp4_cache)
    return k_pre


# -- what reaches the operator -------------------------------------------------


@pytest.mark.parametrize("seq_len,expect", [
    (4, [-1, 40, -1, 41]),
    # Position 3 is at or past the request's length, i.e. a padding token
    # under graph replay. Its block-table row holds a real-looking 0/5, so
    # only seq_lens can drop it -- and it has to arrive as -1, not as 41.
    (3, [-1, 40, -1, -1]),
])
def test_the_gate_reaches_the_operator_as_the_slot_mapping(
        quant_cache, owns, seq_len, expect):
    """Non-boundary tokens reach the operator as -1, boundary ones as the
    destination this group's block table gives.

    At a compressing ratio the destinations are *rebuilt*: the mapping the
    builder publishes is V4.0's dense one, whose valid rows are 0 and 1, not
    1 and 3. Block 5 and page 8 make the rebuilt values readable: position 1
    is compressed state 0 (slot 40) and position 3 is state 1 (slot 41).

    Was: ``call["layout"] == 2`` on the epilog. ``layout`` is an epilog
    argument; that assertion moved to the rollback test below, and this one
    pins what the MXFP4 writer is actually asked for instead.
    """
    slots = owns(torch.arange(4, dtype=torch.int32), seq_lens=[seq_len])
    cache = _bound_cache()
    _store(num_tokens=4, ratio=2, slots=slots, cache=cache)
    (call, ) = quant_cache.calls
    assert call["slot_mapping"].dtype is torch.int32
    assert call["slot_mapping"].tolist() == expect
    assert call["quant_mode"] == "mxfp4"
    assert call["round_scale"] is True and call["x_scale"] == 1.0
    # Both planes reinterpreted (not converted), and both into *this* cache:
    # a plane that is a copy would be written and never read.
    assert call["cache"].dtype is torch.float4_e2m1fn_x2
    assert call["cache_scale"].dtype is torch.float8_e8m0fnu
    for plane in (call["cache"], call["cache_scale"]):
        assert (plane.untyped_storage().data_ptr()
                == cache.untyped_storage().data_ptr())
    assert call["x"].dtype is torch.bfloat16
    assert tuple(call["x"].shape) == (4, INDEX_HEAD_DIM)


def test_the_row_is_the_mxfp4_row_and_the_fp8_row_is_refused(quant_cache):
    """68 bytes, and a 132-byte cache is refused rather than split.

    This is the error every stale fixture in this file used to hit. It is
    pinned as behaviour, not worked around: 132 is not a whole number of
    (16 value + 1 scale)-byte groups, so any split of it would put a scale
    boundary inside a value run and every shape would still look right.
    """
    assert CACHE_WIDTH == 64 + 4
    with pytest.raises(ValueError, match="132 bytes"):
        _store(num_tokens=2, ratio=1, cache=_bound_cache(width=FP8_CACHE_WIDTH))
    assert not quant_cache.calls


def test_the_fp8_rollback_row_still_reaches_the_epilog(epilog):
    """Flag off + fp8 row: the epilog writer, with the layout it was called
    with before batch 1. Kept because ``_refuse_epilog_on_an_mxfp4_row`` keeps
    this branch alive on purpose, and the old ``layout == 2`` assertion lives
    here now."""
    _store(num_tokens=2, ratio=1, cache=_bound_cache(width=FP8_CACHE_WIDTH))
    (call, ) = epilog.calls
    assert call["layout"] == 2
    assert call["slot_mapping"].tolist() == [0, 1]
    assert call["indexer_compress_cache"].ndim == 4


def test_the_epilog_refuses_an_mxfp4_row_and_names_the_flag(epilog):
    """Flag off + MXFP4 row is the half-rollback; it must name the flag, not
    a width, since the width is a symptom one level below the cause."""
    with pytest.raises(RuntimeError, match=_QUANT_FLAG):
        _store(num_tokens=2, ratio=1)
    assert not epilog.calls


# -- the norm and the rotation, through the entry point ------------------------


def test_an_identity_rotation_leaves_the_normed_values(quant_cache):
    """cos=1, sin=0 makes the rotation a no-op, so what is stored is k_norm.

    Separates two things that would otherwise only be testable together: if the
    rotation is applied to the wrong slice, or to the wrong positions, this
    still passes -- and the next two tests are the ones that catch that.
    """
    k_pre = _store(num_tokens=2, ratio=1)
    (call, ) = quant_cache.calls
    expect = _hand_norm(k_pre, torch.ones(INDEX_HEAD_DIM)).to(torch.bfloat16)
    assert torch.equal(call["x"], expect)


def test_the_rotation_leaves_the_nope_half_alone(quant_cache):
    """A non-identity rotation must not touch the leading 64 -- and must
    touch the trailing 64, or this proves nothing."""
    cos_sin = torch.zeros(64, INDEX_ROPE_HEAD_DIM, dtype=torch.float32)
    cos_sin[:, :HALF] = 0.5   # cos
    cos_sin[:, HALF:] = 0.25  # sin
    k_pre = _store(num_tokens=1, ratio=1, cos_sin=cos_sin)
    (call, ) = quant_cache.calls
    normed = _hand_norm(k_pre, torch.ones(INDEX_HEAD_DIM)).to(torch.bfloat16)
    assert torch.equal(call["x"][:, :NOPE], normed[:, :NOPE])
    assert not torch.equal(call["x"][:, NOPE:], normed[:, NOPE:])


def test_the_pairing_is_interleaved_and_the_tail_is_the_trailing_64(
        quant_cache):
    """x[2k] with x[2k+1], over dims 64..127.

    Not in the fp8-era file: the nope-half test above only proves the leading
    64 are untouched, which a split-half rotation of the right slice passes
    too. Pinned bit-exactly against the independent reference, and the
    split-half reading is checked to be genuinely different for these inputs.
    """
    torch.manual_seed(3)
    cos_sin = _angles(4)
    positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    k_pre = _store(num_tokens=4, ratio=1, cos_sin=cos_sin,
                   positions=positions)
    (call, ) = quant_cache.calls
    normed = _hand_norm(k_pre, torch.ones(INDEX_HEAD_DIM))
    assert torch.equal(call["x"],
                       _hand_rope(normed, cos_sin, positions).to(
                           torch.bfloat16))

    row = cos_sin[positions]
    cos, sin = row[:, :HALF], row[:, HALF:]
    a, b = normed[:, NOPE:NOPE + HALF], normed[:, NOPE + HALF:]
    split = normed.clone()
    split[:, NOPE:] = torch.cat((a * cos - b * sin, b * cos + a * sin), -1)
    assert not torch.equal(call["x"], split.to(torch.bfloat16))


def test_the_rotation_uses_the_groups_first_position(quant_cache, owns):
    """compressed_pos = (position // ratio) * ratio.

    A latent stands for the first token of its group, so both tokens of a
    ratio-2 group rotate at the even position. Using `position` instead gives
    every other key a rotation one step ahead -- values that are wrong by an
    angle, not by a shape.
    """
    cos_sin = _angles(8)
    torch.manual_seed(0)
    k_pre = torch.randn(2, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    ones = torch.ones(INDEX_HEAD_DIM, dtype=torch.bfloat16)
    positions = torch.tensor([2, 3], dtype=torch.int64)

    # Ratio 2: positions 2 and 3 are one group, only 3 publishes. They are the
    # last two tokens of a 4-long request, hence seq_lens=[4].
    mapping = owns(torch.tensor([0, 1], dtype=torch.int32), seq_lens=[4])
    indexer_k_norm_rope_store_ascend(k_pre, positions, cos_sin, ones, EPS,
                                     _bound_cache(), mapping, 2, False)
    (call, ) = quant_cache.calls
    assert call["slot_mapping"].tolist()[0] == -1
    published = call["x"][1:2]

    normed = _hand_norm(k_pre[1:2], ones)
    at_two = _hand_rope(normed, cos_sin, torch.tensor([2])).to(torch.bfloat16)
    at_three = _hand_rope(normed, cos_sin, torch.tensor([3])).to(
        torch.bfloat16)
    assert not torch.equal(at_two, at_three), "rows 2 and 3 must differ"
    assert torch.equal(published, at_two)


def test_use_fp4_cache_is_accepted_and_ignored(quant_cache):
    """Replaces ``test_mxfp4_says_upstream_cannot_reach_it_either``.

    That test asserted a ``NotImplementedError`` naming Blackwell for
    ``use_fp4_cache=True``. Its premise is gone, deliberately: on Ascend the
    flag comes from ``dsa_indexer_uses_fp4``, which raises outside Blackwell,
    so it is always False -- while the cache is MXFP4 unconditionally. The
    raise made the MXFP4 store unreachable and was removed (see the entry
    point's docstring). What is true now, and what this pins, is that the
    flag selects nothing: both values write the same operand to the same
    slots with the same quant mode.
    """
    torch.manual_seed(5)
    k_pre = torch.randn(3, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    cos_sin = _angles(4)
    for flag in (False, True):
        _store(num_tokens=3, ratio=1, k_pre=k_pre, cos_sin=cos_sin,
               use_fp4_cache=flag)
    first, second = quant_cache.calls
    for key in ("x", "slot_mapping"):
        assert torch.equal(first[key], second[key]), key
    assert first["quant_mode"] == second["quant_mode"] == "mxfp4"


def test_a_block_offset_slot_mapping_without_its_page_size_is_refused(
        quant_cache):
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
            torch.ones(INDEX_HEAD_DIM), EPS,
            _bound_cache(),
            torch.zeros((2, 2), dtype=torch.int32), 1, False)


def test_an_empty_batch_does_not_call_the_operator(quant_cache):
    indexer_k_norm_rope_store_ascend(
        torch.zeros(0, INDEX_HEAD_DIM, dtype=torch.bfloat16),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(4, INDEX_ROPE_HEAD_DIM),
        torch.ones(INDEX_HEAD_DIM), EPS,
        _bound_cache(),
        torch.zeros(0, dtype=torch.int32), 1, False)
    assert not quant_cache.calls


# -- the two bf16 round trips --------------------------------------------------


def test_the_norm_round_trips_through_bf16_before_the_rotation(
        quant_cache, monkeypatch):
    """Upstream's k_norm ends `.to(bfloat16).to(float32)`, and it is load-bearing.

    It changes no shape and no dtype, so nothing downstream complains; what it
    changes is the precision the rotation starts from, and therefore the bytes
    that reach the quantiser.

    Pinned at the operand of the rotation, which is where the contract lives
    and where the ``npu_rms_norm`` swap lands: *the value fed to the rotation
    is fp32 and already on the bf16 grid*.

    ⚠ The trap from ``test_deepseek_v4_1_indexer_q.py`` (a bf16 input makes a
    round trip unobservable) does **not** apply here, and this test does not
    need an fp32 ``k_pre``: ``k_pre`` is bf16 (upstream asserts it), but the
    norm's output is fp32 arithmetic -- ``rsqrt`` times ``gamma`` leaves the
    bf16 grid for almost every element. The non-vacuity assertion below
    checks exactly that, so if it ever stops being true this test says so
    instead of passing for nothing. (The trap *would* apply on the NoPE half
    at the store: the final cast to bf16 rounds identically either way. That
    is why the end-to-end comparison is on the rotated tail.)
    """
    torch.manual_seed(7)
    k_pre = torch.randn(64, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.bfloat16).abs() + 0.5
    cos_sin = _angles(4)
    positions = torch.arange(64, dtype=torch.int64) % 4

    fed = []
    real_rope = mod._rope_interleaved_tail

    def _spy(k, *args, **kwargs):
        fed.append(k.clone())
        return real_rope(k, *args, **kwargs)

    monkeypatch.setattr(mod, "_rope_interleaved_tail", _spy)
    _store(num_tokens=64, ratio=1, k_pre=k_pre, weight=weight,
           cos_sin=cos_sin, positions=positions)
    (k, ) = fed
    (call, ) = quant_cache.calls

    unrounded = _hand_norm(k_pre, weight, round_trip=False)
    assert not torch.equal(unrounded, unrounded.to(torch.bfloat16).float()), (
        "the fp32 norm must leave the bf16 grid for these inputs, or the "
        "round trip is unobservable and this test proves nothing")

    # The contract, at the rotation's input.
    assert k.dtype is torch.float32, (
        "the rotation must start from fp32; a bf16 hand-off would change the "
        "rotation's arithmetic domain as well")
    assert torch.equal(k, k.to(torch.bfloat16).float()), (
        "every value fed to the rotation must already be bf16-representable; "
        "it is not, so the norm's round trip was dropped or moved")
    # And which grid point: the nearest to the fp32 norm, rounded once.
    assert torch.equal(k, unrounded.to(torch.bfloat16).float())

    # End to end: the round trip moves the stored bytes, on the rotated tail.
    def _candidate(normed):
        return _hand_rope(normed, cos_sin, positions).to(torch.bfloat16)

    with_rt = _candidate(_hand_norm(k_pre, weight, round_trip=True))
    without_rt = _candidate(unrounded)
    assert not torch.equal(with_rt[:, NOPE:], without_rt[:, NOPE:]), (
        "the two candidates must differ on the tail, or this proves nothing")
    assert torch.equal(call["x"], with_rt)


def test_the_post_rotation_round_trip_is_a_no_op_on_this_route(quant_cache):
    """Stated as a test because the code keeps a line that does nothing -- and
    because the fused rotation (``VLLM_ASCEND_DSV41_INDEXER_FUSED_ROPE``) drops
    that line and relies on this being true.

    Upstream quantises from the fp32 value the rotation produces, so its bf16
    round trip there decides the scale. This route hands bf16 to the store
    operator and lets it quantise, and bf16(fp32(bf16(y))) == bf16(y) -- so the
    rounding happens either way. If that ever stops being true (an fp32 store
    input, a different operator), this goes red and the kept line starts
    mattering again.

    The reference below is exactly what the fused route computes *if the
    operator's fp32 rotation equals torch's*: rotate in fp32, no intermediate
    rounding, cast once. That "if" is an operator property and is not visible
    from CPU.
    """
    cos_sin = _angles(4, lo=0.3, hi=1.1)
    torch.manual_seed(11)
    k_pre = torch.randn(32, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.bfloat16).abs() + 0.5
    positions = torch.arange(32, dtype=torch.int64) % 4

    _store(num_tokens=32, ratio=1, k_pre=k_pre, weight=weight,
           cos_sin=cos_sin, positions=positions)
    (call, ) = quant_cache.calls

    normed = _hand_norm(k_pre, weight)
    unrounded = _hand_rope(normed, cos_sin, positions)
    # Non-vacuity: the kept line really does move fp32 values -- the rotated
    # tail is off the bf16 grid -- so the equality below is the final cast
    # absorbing it, not the values already being on the grid.
    #
    # Deliberately *not* asserted: that mod._rope_interleaved_tail's fp32
    # output differs from ``unrounded``. That would pin the kept line itself,
    # and deleting the line -- which is exactly what this test says is safe,
    # and what the fused route does -- would turn it red. Mutation check:
    # with both round-trip lines deleted from the torch form, this test stays
    # green; with them changed to an fp16 round trip, it goes red.
    tail = unrounded[:, NOPE:]
    assert not torch.equal(tail, tail.to(torch.bfloat16).float())

    assert call["x"].dtype is torch.bfloat16
    assert torch.equal(call["x"], unrounded.to(torch.bfloat16))


# -- the fused-operator switches -----------------------------------------------


def _tripwire(name):
    def _raise(*_, **__):
        raise AssertionError(
            f"{name} was called with a CPU tensor; the fused branches key on "
            "is_npu, and a CPU call means that key changed")

    return _raise


def _no_operators(monkeypatch):
    monkeypatch.setattr(mod.torch_npu, "npu_rms_norm",
                        _tripwire("npu_rms_norm"), raising=False)
    monkeypatch.setattr(mod, "rope_tail_operator",
                        _tripwire("rope_tail_operator"))


@pytest.mark.parametrize("norm_flag,rope_flag",
                         [("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")])
@pytest.mark.parametrize("target", ["rms_norm", "rope_tail", "store"])
def test_the_fused_switches_do_not_move_the_cpu_result(
        target, norm_flag, rope_flag, monkeypatch, quant_cache):
    """On CPU every switch setting takes the torch form, bit for bit.

    The fused branches are ``getattr(x, "is_npu", False) and flag == "1"``, so
    a CPU tensor must never reach them. If this diverges, the dispatch key
    changed and this CPU suite has stopped describing what the card runs --
    every other test in the file would then be pinning a route the card does
    not take.

    Each case runs the target twice, under the parametrised setting and under
    both switches off, and requires the two to be ``torch.equal``: that is the
    statement itself. The tripwires make "a CPU tensor reached an NPU
    operator" an error with a name, rather than whatever that operator happens
    to do with a CPU tensor.

    ``rms_norm`` and ``store`` are also compared with the independent
    reference, so the settings cannot agree by being equally wrong.
    ``rope_tail`` is not: its fp32 output includes the post-rotation round
    trip, which is dead at the store (see the test above), and pinning it here
    would make removing a dead line look like a regression.
    """
    _no_operators(monkeypatch)

    torch.manual_seed(13)
    k_pre = torch.randn(8, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.bfloat16).abs() + 0.5
    cos_sin = _angles(4)
    positions = torch.arange(8, dtype=torch.int64) % 4
    normed = _hand_norm(k_pre, weight)

    def _run(norm, rope):
        monkeypatch.setenv(_NORM_FLAG, norm)
        monkeypatch.setenv(_ROPE_FLAG, rope)
        if target == "rms_norm":
            return mod._rms_norm(k_pre, weight, EPS)
        if target == "rope_tail":
            return mod._rope_interleaved_tail(normed, cos_sin, positions,
                                              INDEX_ROPE_HEAD_DIM)
        quant_cache.calls.clear()
        _store(num_tokens=8, ratio=1, k_pre=k_pre, weight=weight,
               cos_sin=cos_sin, positions=positions)
        (call, ) = quant_cache.calls
        return call["x"]

    got = _run(norm_flag, rope_flag)
    baseline = _run("0", "0")
    assert got.dtype == baseline.dtype
    assert torch.equal(got, baseline), (
        f"{target} under {_NORM_FLAG}={norm_flag} {_ROPE_FLAG}={rope_flag} "
        "differs from both switches off on CPU")

    if target == "rms_norm":
        assert torch.equal(got, normed)
    elif target == "store":
        assert torch.equal(
            got, _hand_rope(normed, cos_sin, positions).to(torch.bfloat16))


class _NpuLooking(torch.Tensor):
    """A CPU tensor that answers ``is_npu`` with True.

    The only way to reach the fused branches without a card. Nothing is
    computed by an NPU operator here: the operators are stand-ins, and what is
    tested is the *routing* -- that the switch reaches the operator on an NPU
    tensor, with the arguments the call site promises, and that "0" really
    turns it off. The parity test above cannot see this: it would pass with
    both fused branches deleted.
    """

    @property
    def is_npu(self):
        return True


@pytest.mark.parametrize("flag", ["1", "0"])
def test_on_an_npu_tensor_the_norm_switch_picks_the_route(flag, monkeypatch):
    torch.manual_seed(17)
    k_pre = torch.randn(4, INDEX_HEAD_DIM, dtype=torch.bfloat16)
    weight = torch.randn(INDEX_HEAD_DIM, dtype=torch.bfloat16).abs() + 0.5
    seen = []

    def _fake_rms_norm(x, w, epsilon):
        seen.append((x, w, epsilon))
        # bf16 out, like the operator; the value is the torch form's so the
        # returned contract (fp32, on the grid) can be checked below.
        y = _hand_norm(torch.Tensor(x), torch.Tensor(w)).to(torch.bfloat16)
        return y, torch.zeros(x.shape[0], 1)

    monkeypatch.setenv(_NORM_FLAG, flag)
    monkeypatch.setattr(mod.torch_npu, "npu_rms_norm", _fake_rms_norm,
                        raising=False)
    got = mod._rms_norm(k_pre.as_subclass(_NpuLooking), weight, EPS)

    assert len(seen) == (1 if flag == "1" else 0)
    if seen:
        _, w, epsilon = seen[0]
        assert w is weight and epsilon == EPS
    # Either route returns fp32 values on the bf16 grid -- the operator
    # route's ``.to(float32)`` is what keeps the rotation in fp32.
    assert got.dtype is torch.float32
    assert torch.equal(torch.Tensor(got), _hand_norm(k_pre, weight))


@pytest.mark.parametrize("flag", ["1", "0"])
def test_on_an_npu_tensor_the_rope_switch_picks_the_route(flag, monkeypatch):
    torch.manual_seed(19)
    k = _hand_norm(torch.randn(4, INDEX_HEAD_DIM, dtype=torch.bfloat16),
                   torch.ones(INDEX_HEAD_DIM))
    cos_sin = _angles(4)
    positions = torch.tensor([3, 0, 2, 1], dtype=torch.int64)
    seen = []

    def _fake_operator(x, cos, sin, rope_head_dim):
        seen.append((x, cos, sin, rope_head_dim))
        return x.clone()

    monkeypatch.setenv(_ROPE_FLAG, flag)
    monkeypatch.setattr(mod, "rope_tail_operator", _fake_operator)
    mod._rope_interleaved_tail(k.as_subclass(_NpuLooking), cos_sin, positions,
                               INDEX_ROPE_HEAD_DIM)

    assert len(seen) == (1 if flag == "1" else 0)
    if seen:
        x, cos, sin, rope_head_dim = seen[0]
        # The rope width, not the head width; fp32 in, as the operator's
        # all-DT_FLOAT combination needs.
        assert rope_head_dim == INDEX_ROPE_HEAD_DIM
        assert x.dtype is torch.float32
        # Per-channel cos/sin for *these* positions, widened interleaved:
        # channel i belongs to pair i // 2.
        row = cos_sin[positions]
        assert torch.equal(torch.Tensor(cos),
                           row[:, :HALF].repeat_interleave(2, dim=-1))
        assert torch.equal(torch.Tensor(sin),
                           row[:, HALF:].repeat_interleave(2, dim=-1))


# -- the patch -----------------------------------------------------------------


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
