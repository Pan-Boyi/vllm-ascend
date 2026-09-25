# SPDX-License-Identifier: Apache-2.0
"""Chunked (per-expert) streaming prefill: the tiling, the tables, the oracles.

Two criteria, never a tolerance: a set-based one (which experts contribute to
which token) and an exactness one (the weights pass through untouched). Each
carries a mutation arm, because a dropped chunk is a small numeric change and a
large semantic one.
"""

from __future__ import annotations

import pytest
import torch

from vllm_ascend.kt_offload.placement import LayerPlacement
from vllm_ascend.kt_offload.stream_prefill import (ChunkTables, assert_eager_only, build_chunk_plan,
                                                   check_chunk_cover, check_chunk_plan, chunk_tables,
                                                   resident_chunk_ids, split_routing_for_chunk)

E, TOPK = 64, 6


def routing(tokens: int, seed: int = 0, num_experts: int = E):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(num_experts, generator=g)[:TOPK] for _ in range(tokens)])
    w = torch.rand(tokens, TOPK, generator=g, dtype=torch.float32) + 0.1
    return ids.to(torch.int32), w


def run_all_chunks(plan, ids, w):
    out_ids, out_w = [], []
    for k in range(plan.num_chunks):
        r = split_routing_for_chunk(ids, w, chunk_tables(plan, k))
        out_ids.append(r.topk_ids)
        out_w.append(r.topk_weights)
    return out_ids, out_w


@pytest.mark.parametrize("chunk", [1, 5, 8, 32, 64, 100])
def test_plan_tiles_every_expert_exactly_once(chunk):
    plan = build_chunk_plan(E, chunk)
    assert check_chunk_plan(plan) == []
    assert sorted(plan.covered) == list(range(E))
    assert plan.num_chunks == -(-E // chunk)          # ceil, last chunk may be short
    assert plan.slot_capacity == min(chunk, E)


def test_plan_over_a_subset_is_judged_against_that_subset():
    """A coverage oracle hard-wired to "all experts" would call this run complete
    while a quarter of the layer never travelled."""
    subset = list(range(10, 40))
    plan = build_chunk_plan(E, 8, expert_ids=subset)
    assert check_chunk_plan(plan, subset) == []
    assert check_chunk_plan(plan) != []                # judged against "all" it must complain


def test_resident_experts_can_be_excluded_from_streaming():
    pl = LayerPlacement(layer_idx=0, num_experts=E, num_slots=8, resident_ids=tuple(range(8)))
    offloaded = resident_chunk_ids(pl)
    assert offloaded == list(range(8, E))
    plan = build_chunk_plan(E, 16, expert_ids=offloaded)
    assert check_chunk_plan(plan, offloaded) == []


@pytest.mark.parametrize("bad", [dict(num_experts=0, chunk_experts=4), dict(num_experts=8, chunk_experts=0)])
def test_degenerate_plans_raise(bad):
    with pytest.raises(ValueError):
        build_chunk_plan(**bad)


def test_repeated_expert_raises():
    """An expert in two chunks would be added twice by the chunk sum."""
    with pytest.raises(ValueError, match="repeat"):
        build_chunk_plan(E, 4, expert_ids=[1, 2, 2, 3])


def test_empty_stream_set_raises():
    with pytest.raises(ValueError, match="zero-coverage"):
        build_chunk_plan(E, 4, expert_ids=[])


@pytest.mark.parametrize("tokens", [1, 7, 64])
@pytest.mark.parametrize("chunk", [4, 16, 64])
def test_every_route_is_claimed_by_exactly_one_chunk(tokens, chunk):
    ids, w = routing(tokens)
    plan = build_chunk_plan(E, chunk)
    cid, cw = run_all_chunks(plan, ids, w)
    assert check_chunk_cover(plan, ids, cid, cw, w) == []


def test_live_weights_are_bit_identical_to_the_original():
    """The chunk must not rescale anything: combine applies the weight once."""
    ids, w = routing(32, seed=3)
    plan = build_chunk_plan(E, 8)
    cid, cw = run_all_chunks(plan, ids, w)
    total = torch.zeros_like(w)
    for c in cw:
        total += c
    assert torch.equal(total, w)          # exact, not allclose


def test_shapes_and_dtypes_are_preserved():
    ids, w = routing(5)
    r = split_routing_for_chunk(ids, w, chunk_tables(build_chunk_plan(E, 8), 0))
    assert r.topk_ids.shape == ids.shape and r.topk_ids.dtype == ids.dtype
    assert r.topk_weights.shape == w.shape and r.topk_weights.dtype == w.dtype
    assert int(r.topk_ids.min()) >= 0     # no -1 ever reaches the routing kernel


def test_out_of_chunk_routes_point_at_a_real_slot():
    """Dummy rows land on slot 0: weights never loaded can be NaN and NaN*0 is
    NaN, so the dummy target must be a slot with real bytes in it."""
    ids, w = routing(16, seed=5)
    plan = build_chunk_plan(E, 8)
    t = chunk_tables(plan, 2)
    r = split_routing_for_chunk(ids, w, t)
    outside = r.topk_weights == 0
    assert outside.any()
    assert torch.all(r.topk_ids[outside] == 0)
    assert int(t.safe_slot_table.max()) < t.num_slots


def test_chunk_index_out_of_range_raises():
    with pytest.raises(IndexError):
        chunk_tables(build_chunk_plan(E, 8), 99)


def test_mut_dropped_chunk_is_red():
    ids, w = routing(32, seed=1)
    plan = build_chunk_plan(E, 8)
    cid, cw = run_all_chunks(plan, ids, w)
    cw[3] = torch.zeros_like(cw[3])                    # that chunk contributes nothing
    assert check_chunk_cover(plan, ids, cid, cw, w) != []


def test_mut_keep_not_zeroed_outside_chunk_is_red():
    """The classic: forget to zero the out-of-chunk weight -> experts counted twice."""
    ids, w = routing(32, seed=2)
    plan = build_chunk_plan(E, 8)
    cid, cw = run_all_chunks(plan, ids, w)
    cw[1] = w.clone()                                  # every route "live" in chunk 1
    assert check_chunk_cover(plan, ids, cid, cw, w) != []


def test_mut_slot_index_off_by_one_is_red():
    ids, w = routing(32, seed=4)
    plan = build_chunk_plan(E, 8)
    cid, cw = run_all_chunks(plan, ids, w)
    cid[2] = (cid[2] + 1) % len(plan.chunks[2])        # maps back to the wrong logical expert
    assert check_chunk_cover(plan, ids, cid, cw, w) != []


def test_mut_rescaled_weight_is_red():
    ids, w = routing(32, seed=6)
    plan = build_chunk_plan(E, 8)
    cid, cw = run_all_chunks(plan, ids, w)
    cw[0] = cw[0] * 1.000001                           # tiny: a tolerance check would pass it
    assert check_chunk_cover(plan, ids, cid, cw, w) != []


def test_mut_plan_missing_an_expert_is_red():
    plan = build_chunk_plan(E, 8)
    holed = type(plan)(num_experts=E, chunk_experts=8, chunks=plan.chunks[:-1])
    assert check_chunk_plan(holed) != []


def test_mut_expert_in_two_chunks_is_red():
    plan = build_chunk_plan(E, 8)
    dup = type(plan)(num_experts=E, chunk_experts=8, chunks=plan.chunks + (plan.chunks[0],))
    assert check_chunk_plan(dup) != []


def test_mut_wrong_number_of_chunk_outputs_is_red():
    ids, w = routing(8, seed=7)
    plan = build_chunk_plan(E, 8)
    cid, cw = run_all_chunks(plan, ids, w)
    assert check_chunk_cover(plan, ids, cid[:-1], cw[:-1], w) != []


def test_streaming_below_the_threshold_is_refused():
    with pytest.raises(ValueError, match="threshold"):
        assert_eager_only(128, [1, 2, 4, 8], min_tokens=512)


def test_streaming_at_a_captured_size_is_refused():
    """A capture list that grows past the threshold must fail loudly, not bake a chunk in."""
    with pytest.raises(ValueError, match="cudagraph_capture_sizes"):
        assert_eager_only(1024, [1, 2, 4, 8, 1024], min_tokens=512)


def test_streaming_above_the_threshold_and_outside_the_capture_list_is_allowed():
    assert_eager_only(4096, [1, 2, 4, 8], min_tokens=512)
    assert_eager_only(512, None, min_tokens=512)


def test_mut_capture_guard_cannot_be_satisfied_by_an_empty_list():
    """An empty list says the caller did not tell us, so the threshold still has
    to hold: the guard must not go quiet when its input is."""
    with pytest.raises(ValueError):
        assert_eager_only(8, [], min_tokens=512)


def test_tables_live_where_asked():
    t = chunk_tables(build_chunk_plan(E, 8), 0, id_dtype=torch.int64, weight_dtype=torch.bfloat16)
    assert isinstance(t, ChunkTables)
    assert t.safe_slot_table.dtype == torch.int64 and t.keep_table.dtype == torch.bfloat16
