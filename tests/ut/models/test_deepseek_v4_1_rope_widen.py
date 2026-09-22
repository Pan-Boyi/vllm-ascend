# SPDX-License-Identifier: Apache-2.0
"""Which widening of cos/sin is right, made decidable.

Two widenings give the operator the width it asks for, and nothing downstream
distinguishes them. The first test's job is to show the choice *matters* --
that A and B differ -- because a check written for a choice that turns out not
to matter is a check that can never fail for the reason it was written.

The second pins which one, against the Ascend path that already drives this
operator rather than against a reading of what interleaving ought to mean.
"""

import gc
import weakref

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import rope_widen
from vllm_ascend.models.deepseek_v4_1.rope_widen import (cos_sin_for_slice,
                                                         widen_interleaved)

ROPE_DIM = 64
HALF = ROPE_DIM // 2


def _cache(max_pos=8):
    """Upstream's layout: half a rotation of cosines, then its sines."""
    torch.manual_seed(3)
    angles = torch.rand(max_pos, HALF) * 3.0
    return torch.cat([angles.cos(), angles.sin()], dim=-1), angles


def test_the_two_widenings_actually_differ():
    """Otherwise the rest of this file is checking a distinction without one."""
    half = torch.arange(HALF, dtype=torch.float32)
    a = widen_interleaved(half)
    b = half.repeat(2)
    assert a.shape == b.shape == (ROPE_DIM,)
    assert not torch.equal(a, b), (
        "if these agreed, the widening would not be a decision and this "
        "file would be pinning nothing")


def test_channel_i_carries_its_own_pairs_value():
    """Channel i belongs to pair i // 2. That is what makes it interleave."""
    half = torch.arange(HALF, dtype=torch.float32)
    wide = widen_interleaved(half)
    for i in range(ROPE_DIM):
        assert wide[i] == half[i // 2]


def test_it_matches_what_the_working_ascend_path_builds():
    """ops/rope_dsv4.py:218 is freqs.cos().repeat_interleave(2, dim=-1).

    That table is what V4.0 already hands to inplace_partial_rotary_mul, so it
    is the consumer's own convention rather than an interpretation of it. If
    that line changes, this should be revisited -- so it is read, not recalled.
    """
    import inspect

    from vllm_ascend.ops import rope_dsv4

    source = inspect.getsource(rope_dsv4.ComplexExpRotaryEmbedding.__init__)
    assert "repeat_interleave(2, dim=-1)" in source, (
        "the working Ascend rope path no longer widens with repeat_interleave; "
        "the convention this file copies has moved")


def test_the_halves_are_split_where_the_cache_puts_them():
    cache, angles = _cache()
    positions = torch.tensor([0, 3, 7])
    cos, sin = cos_sin_for_slice(cache, positions, ROPE_DIM)
    assert cos.shape == sin.shape == (3, ROPE_DIM)
    for row, pos in enumerate(positions.tolist()):
        for i in range(ROPE_DIM):
            assert cos[row, i] == pytest.approx(float(angles[pos, i // 2].cos()))
            assert sin[row, i] == pytest.approx(float(angles[pos, i // 2].sin()))


def test_a_rotation_built_this_way_matches_a_per_pair_reference():
    """End to end against the arithmetic, not against another widening.

    The reference indexes cos[k] and sin[k] for pair k directly and never
    widens anything, so agreement means the widened form feeds each pair the
    angle that pair is supposed to get.
    """
    cache, angles = _cache()
    positions = torch.tensor([2, 5])
    x = torch.randn(2, ROPE_DIM)
    cos, sin = cos_sin_for_slice(cache, positions, ROPE_DIM)

    pairs = x.view(2, HALF, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    c = angles[positions].cos()
    s = angles[positions].sin()
    expect = torch.stack((even * c - odd * s, odd * c + even * s),
                         dim=-1).view(2, ROPE_DIM)

    wide_pairs = torch.stack((cos.view(2, HALF, 2)[..., 0],
                              sin.view(2, HALF, 2)[..., 0]), dim=-1)
    got = torch.stack((even * wide_pairs[..., 0] - odd * wide_pairs[..., 1],
                       odd * wide_pairs[..., 0] + even * wide_pairs[..., 1]),
                      dim=-1).view(2, ROPE_DIM)
    torch.testing.assert_close(got, expect)


def test_the_wrong_widening_produces_a_different_rotation():
    """The failure this prevents, made visible.

    B is wrong on every other pair -- regular enough to look like a plausible
    model rather than a bug, which is why the criterion is here and not in a
    tolerance downstream.
    """
    cache, angles = _cache()
    positions = torch.tensor([4])
    cos_a, _ = cos_sin_for_slice(cache, positions, ROPE_DIM)
    cos_b = cache[positions][..., :HALF].repeat(1, 2)
    assert not torch.allclose(cos_a, cos_b)
    differing = int((cos_a != cos_b).sum())
    assert differing > ROPE_DIM // 4, (
        f"only {differing} of {ROPE_DIM} channels differ; if the wrong choice "
        "were nearly right it would be far harder to catch downstream")


def test_a_cache_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="pair.*count"):
        cos_sin_for_slice(torch.zeros(8, HALF), torch.tensor([0]), ROPE_DIM)


# ---------------------------------------------------------------------------
# The per-pass cache.
#
# What it may and may not do, stated once so the assertions below can be read
# against it (the implementation's own statement is at ``rope_widen._CAPACITY``
# and ``rope_widen._lookup``):
#   * outside a forward pass nothing is cached;
#   * within one pass, the same (table, positions) *address, shape, strides,
#     dtype* is computed once -- that is the saving, and it trusts that nobody
#     rewrote those bytes in place during the pass;
#   * a new pass starts from nothing, even if every address is the same;
#   * an address in the table is held, so it cannot be recycled for a
#     different tensor while its entry exists.
# ---------------------------------------------------------------------------


class _Pass:
    """Stands in for vLLM's ForwardContext: a plain, weak-referenceable object
    whose identity is the pass."""


@pytest.fixture(autouse=True)
def _empty_cache():
    rope_widen._CACHE.clear()
    rope_widen._CACHE_PASS[0] = None
    yield
    rope_widen._CACHE.clear()
    rope_widen._CACHE_PASS[0] = None


@pytest.fixture
def in_pass(monkeypatch):
    """``set(p)`` makes ``p`` the running pass; returns a fresh one by default."""
    state = {"pass": _Pass()}
    monkeypatch.setattr(rope_widen, "_forward_token", lambda: state["pass"])

    def set_pass(p=None):
        state["pass"] = _Pass() if p is None else p
        return state["pass"]

    return set_pass


@pytest.fixture
def widenings(monkeypatch):
    """Counts real computations: two widenings (cos, sin) per miss."""
    calls = [0]
    real = rope_widen.widen_interleaved

    def counting(half):
        calls[0] += 1
        return real(half)

    monkeypatch.setattr(rope_widen, "widen_interleaved", counting)
    return calls


def _expected(cache, positions):
    row = cache[positions].to(torch.float32)
    return (row[..., :HALF].repeat_interleave(2, dim=-1),
            row[..., HALF:].repeat_interleave(2, dim=-1))


def _assert_right(cache, positions, got):
    cos, sin = _expected(cache, positions)
    torch.testing.assert_close(got[0], cos, rtol=0, atol=0)
    torch.testing.assert_close(got[1], sin, rtol=0, atol=0)


def _table(seed, max_pos=64):
    g = torch.Generator().manual_seed(seed)
    angles = torch.rand(max_pos, HALF, generator=g) * 3.0
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def test_main_kv_and_indexer_pairs_all_hit_when_interleaved(in_pass,
                                                             widenings):
    """The point of the change: three pairs asked for in turn, every layer,
    and each is computed exactly once per pass.

    Main KV and indexer q share ``positions`` but not the table; indexer k
    shares the main table but not the positions. With one entry each call
    evicted the previous one and this was 3 misses a layer.
    """
    main_table, index_q_table = _table(0), _table(1)
    positions = torch.arange(16, dtype=torch.int64) + 5
    compressed = torch.div(positions, 2, rounding_mode="floor") * 2  # once/step

    first = {}
    for _layer in range(40):
        for name, table, pos in (("main", main_table, positions[:16]),
                                 ("index_q", index_q_table, positions[:16]),
                                 ("index_k", main_table, compressed)):
            got = rope_widen.cos_sin_for_slice(table, pos, ROPE_DIM)
            if name not in first:
                _assert_right(table, pos, got)
                first[name] = got
            assert got[0] is first[name][0] and got[1] is first[name][1], (
                f"{name} was recomputed on a later layer: something evicted it")
            # The inverse path on the main pair, as rope.py does every layer.
            if name == "main":
                c, ns = rope_widen.neg_sin_for_slice(table, pos, ROPE_DIM)
                assert c is first["main"][0]
                torch.testing.assert_close(ns, -first["main"][1])
    assert widenings[0] == 2 * 3, (
        f"{widenings[0]} widenings for 3 pairs over 40 layers; exactly 6 "
        "(cos + sin, once per pair) is what caching per pass means")


def test_the_negation_is_cached_with_its_own_pair(in_pass):
    """neg_sin rides on the entry that made ``cos``; with several entries it
    must still be that entry, and stay cached across other keys' calls."""
    a, b = _table(0), _table(1)
    positions = torch.arange(8)
    _, na = rope_widen.neg_sin_for_slice(a, positions, ROPE_DIM)
    _, nb = rope_widen.neg_sin_for_slice(b, positions, ROPE_DIM)
    _, na2 = rope_widen.neg_sin_for_slice(a, positions, ROPE_DIM)
    assert na2 is na, "the -sin for table a was recomputed after b's call"
    torch.testing.assert_close(na, -_expected(a, positions)[1])
    torch.testing.assert_close(nb, -_expected(b, positions)[1])
    assert not torch.equal(na, nb), "the two tables must not share a -sin"


def test_capacity_is_enforced_and_eviction_is_least_recently_used(
        in_pass, widenings):
    cap = rope_widen._CAPACITY
    table = _table(0)
    base = torch.arange(64)
    # Distinct data_ptr per key: disjoint slices of one live buffer.
    keys = [base[4 * i:4 * i + 4] for i in range(cap + 1)]

    for pos in keys[:cap]:
        rope_widen.cos_sin_for_slice(table, pos, ROPE_DIM)
    assert len(rope_widen._CACHE) == cap
    assert widenings[0] == 2 * cap

    # Touch key 0 so key 1 becomes the least recently used.
    rope_widen.cos_sin_for_slice(table, keys[0], ROPE_DIM)
    assert widenings[0] == 2 * cap, "key 0 should still have been a hit"

    rope_widen.cos_sin_for_slice(table, keys[cap], ROPE_DIM)
    assert len(rope_widen._CACHE) == cap, "the table grew past its bound"

    before = widenings[0]
    rope_widen.cos_sin_for_slice(table, keys[0], ROPE_DIM)
    assert widenings[0] == before, "LRU evicted the recently used key 0"
    rope_widen.cos_sin_for_slice(table, keys[1], ROPE_DIM)
    assert widenings[0] == before + 2, "key 1 (least recently used) survived"
    assert len(rope_widen._CACHE) == cap


def test_a_rewritten_buffer_is_recomputed_on_the_next_pass(in_pass):
    """The runner's case: one persistent ``positions`` buffer, same address,
    new contents every step. This is why the pass is part of the key."""
    table = _table(0)
    buf = torch.arange(8)
    step1 = rope_widen.cos_sin_for_slice(table, buf[:8], ROPE_DIM)
    _assert_right(table, torch.arange(8), step1)

    in_pass()  # next step
    ptr = buf.data_ptr()
    buf.copy_(torch.arange(8) + 30)
    assert buf.data_ptr() == ptr
    step2 = rope_widen.cos_sin_for_slice(table, buf[:8], ROPE_DIM)
    _assert_right(table, torch.arange(8) + 30, step2)


def test_a_new_pass_at_a_dead_passes_address_is_still_a_new_pass(in_pass):
    """Regression: the key used to hold ``id(forward_context)``.

    When a pass's context is freed the next one can be allocated at the same
    address, so the id matches, the persistent buffer's address matches, and
    the old key would serve the previous step's angles. Here the id collision
    is produced for real (CPython reuses the slot of a freed object of the
    same size) rather than assumed.
    """
    table = _table(0)
    buf = torch.arange(8)
    placeholder = _Pass()  # keeps the fixture from holding the pass we free
    second = None
    for _ in range(100):
        first = in_pass()
        rope_widen.cos_sin_for_slice(table, buf, ROPE_DIM)
        old_id = id(first)
        in_pass(placeholder)
        del first  # refcount drops to zero: freed now, no gc needed
        candidate = _Pass()
        if id(candidate) == old_id:
            second = candidate
            break
    if second is None:
        # Skip, not pass: the precondition (an id collision) did not occur,
        # so nothing was tested. The weakref test below covers the mechanism
        # without depending on the allocator.
        pytest.skip("could not provoke an id() collision on this interpreter")

    in_pass(second)
    buf.copy_(torch.arange(8) + 40)
    got = rope_widen.cos_sin_for_slice(table, buf, ROPE_DIM)
    _assert_right(table, torch.arange(8) + 40, got)


def test_the_cache_does_not_keep_a_finished_pass_alive(in_pass):
    table = _table(0)
    p = in_pass()
    rope_widen.cos_sin_for_slice(table, torch.arange(4), ROPE_DIM)
    ref = weakref.ref(p)
    del p
    in_pass()  # the fixture drops its reference too
    gc.collect()
    assert ref() is None, (
        "the cache holds a strong reference to a finished forward pass "
        "(and with it that pass's attention metadata)")


def test_an_address_in_the_table_is_held_so_it_cannot_be_recycled(in_pass):
    """Regression for the within-pass hazard that one entry used to hide.

    The indexer builds ``compressed_pos`` per layer and drops it. If the
    cache kept only its address, the allocator could hand that address to the
    next layer's ``compressed_pos`` -- and at V4.1's ratio-2 -> ratio-1
    boundary that tensor has the same shape and different values. Holding the
    tensor is what makes a key match mean "the same live memory".
    """
    table = _table(0)
    temp = torch.div(torch.arange(16), 2, rounding_mode="floor") * 2
    rope_widen.cos_sin_for_slice(table, temp, ROPE_DIM)
    held = weakref.ref(temp)
    del temp
    gc.collect()
    assert held() is not None, "the entry does not hold its positions tensor"


def test_per_layer_temporaries_across_a_ratio_boundary_get_their_own_angles(
        in_pass):
    """The same hazard end to end, in the call order the indexer will use:
    main KV, then indexer k on a temporary it frees, layer after layer, with
    the compress ratio changing part-way (V4.1-Flash: 2 then 1)."""
    table = _table(0)
    positions = torch.arange(16) + 3
    for ratio in [2] * 4 + [1] * 4:
        rope_widen.cos_sin_for_slice(table, positions[:16], ROPE_DIM)
        compressed = torch.div(positions[:16], ratio,
                               rounding_mode="floor") * ratio
        got = rope_widen.cos_sin_for_slice(table, compressed, ROPE_DIM)
        _assert_right(table, compressed, got)
        del compressed, got
    assert len(rope_widen._CACHE) <= rope_widen._CAPACITY


def test_views_of_one_buffer_hit_and_a_different_stride_does_not(in_pass,
                                                                 widenings):
    """Why the key is an address and not the tensor object: every layer
    passes a new view of the same buffer and must hit. Why it includes the
    strides: two views can start at the same address with the same shape and
    read different elements."""
    table = _table(0)
    base = torch.arange(16)
    rope_widen.cos_sin_for_slice(table, base[:8], ROPE_DIM)
    rope_widen.cos_sin_for_slice(table, base[:8], ROPE_DIM)
    assert widenings[0] == 2, "a fresh view of the same memory missed"

    strided = base[0:16:2]
    assert strided.data_ptr() == base[:8].data_ptr()
    assert strided.shape == base[:8].shape
    got = rope_widen.cos_sin_for_slice(table, strided, ROPE_DIM)
    _assert_right(table, strided, got)


def test_within_a_pass_the_address_is_trusted_that_is_the_contract(in_pass):
    """Pinned so that changing it is a decision, not an accident.

    Within one pass a hit does not re-read the positions: that re-read is the
    work being saved. No caller writes its positions in place mid-pass (the
    runner fills them before the pass; ``compressed_pos`` is a new tensor).
    A caller that starts doing so must not use this function -- this test is
    where that is written down.
    """
    table = _table(0)
    buf = torch.arange(8)
    first = rope_widen.cos_sin_for_slice(table, buf, ROPE_DIM)
    buf.add_(20)
    again = rope_widen.cos_sin_for_slice(table, buf, ROPE_DIM)
    assert again[0] is first[0], "a same-pass, same-address call recomputed"


def test_outside_a_forward_pass_nothing_is_cached(widenings):
    table = _table(0)
    buf = torch.arange(8)
    rope_widen.cos_sin_for_slice(table, buf, ROPE_DIM)
    buf.add_(20)
    got = rope_widen.cos_sin_for_slice(table, buf, ROPE_DIM)
    _assert_right(table, buf, got)
    assert widenings[0] == 4 and len(rope_widen._CACHE) == 0


def test_a_pass_that_cannot_be_weakly_referenced_is_not_cached(
        monkeypatch, widenings):
    """No weakref means no way to tell it from the next pass without keeping
    it alive; the conservative answer is to not cache."""
    token = object()  # object() instances take no weak references
    monkeypatch.setattr(rope_widen, "_forward_token", lambda: token)
    table = _table(0)
    rope_widen.cos_sin_for_slice(table, torch.arange(4), ROPE_DIM)
    rope_widen.cos_sin_for_slice(table, torch.arange(4), ROPE_DIM)
    assert widenings[0] == 4 and len(rope_widen._CACHE) == 0
