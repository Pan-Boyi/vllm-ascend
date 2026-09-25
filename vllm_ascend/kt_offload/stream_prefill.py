# SPDX-License-Identifier: Apache-2.0
"""Streaming prefill: compute a layer's MoE entirely on the NPU, from expert
weights streamed up from host DDR, chunk by chunk.

With only a fraction of the experts resident, a long prefill pays the CPU path
for the rest on every layer, and that cost grows with the token count while the
weight traffic does not (the *distinct* expert count saturates, so past that
point the CPU is compute-bound, not bandwidth-bound). Streaming makes the cost
per-layer instead: move the weights once, let the NPU do the GEMMs. Which is
cheaper depends on the token count, which is why the chunk size and the
threshold are configuration, not constants.

"Chunk" here is NOT a chunk of tokens (that is vLLM's
``--max-num-batched-tokens``) but a chunk of EXPERTS: the device holds one slot
buffer of ``chunk_experts`` experts at a time and the layer is computed as a sum
over chunks. Splitting inside a layer is allowed; deferring any part of a
layer's contribution to a later layer is NOT, so everything here is per-layer,
per-forward.

The sum is exact because for chunk k a route to an expert inside the chunk
becomes that expert's index *within the chunk*, while a route to an expert
outside it becomes the dummy slot with weight 0 and the combine step multiplies
by that weight. Each expert appears in exactly one chunk, so summing the chunk
outputs reproduces the one-shot result.

The tables are the production ones: ``LayerPlacement.safe_slot_table`` /
``keep_table`` already express "route here, or contribute nothing", and a chunk
is the same transform with a different membership set. The on-card verification
of the chunk sum drove the production ``split_routing`` with these tables, so
what was verified on-card is what runs here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch

from vllm_ascend.kt_offload.placement import CPU_SLOT, LayerPlacement, NpuRouting, assert_routing_dtypes, split_routing

__all__ = [
    "ChunkPlan",
    "ChunkTables",
    "build_chunk_plan",
    "chunk_tables",
    "split_routing_for_chunk",
    "check_chunk_plan",
    "check_chunk_cover",
    "assert_eager_only",
]


# ---------------------------------------------------------------------------
# The plan: which experts travel together
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkPlan:
    """A tiling of ``expert_ids`` into chunks of at most ``chunk_experts``.

    ``chunks[k]`` holds logical expert ids; position ``i`` inside a chunk is the
    slot that expert occupies in the streaming buffer while that chunk is on the
    device. The last chunk may be short -- padding it would stream weights no
    route can reach.
    """

    num_experts: int
    chunk_experts: int
    chunks: tuple[tuple[int, ...], ...]

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def covered(self) -> tuple[int, ...]:
        return tuple(e for c in self.chunks for e in c)

    @property
    def slot_capacity(self) -> int:
        """Slots the streaming buffer must hold: the largest chunk, not the config value."""
        return max((len(c) for c in self.chunks), default=0)


def build_chunk_plan(num_experts: int, chunk_experts: int,
                     expert_ids: Sequence[int] | None = None) -> ChunkPlan:
    """Tile ``expert_ids`` (default: all experts, ascending) into chunks.

    ``expert_ids`` lets a caller stream only the *offloaded* experts and keep
    using the resident weights for the rest. Both shapes are legal and the
    checker below judges whichever was asked for: a coverage criterion that
    assumed "all 384" would silently pass a run that streamed a subset.
    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")
    if chunk_experts <= 0:
        raise ValueError(f"chunk_experts must be > 0, got {chunk_experts}")
    ids = list(range(num_experts)) if expert_ids is None else [int(e) for e in expert_ids]
    if not ids:
        raise ValueError("expert_ids is empty: a streaming layer that streams nothing is a "
                         "zero-coverage run, not a fast one")
    bad = [e for e in ids if not 0 <= e < num_experts]
    if bad:
        raise ValueError(f"expert ids out of range [0,{num_experts}): {bad[:8]}")
    if len(set(ids)) != len(ids):
        dup = sorted({e for e in ids if ids.count(e) > 1})
        raise ValueError(f"expert ids repeat {dup[:8]}: an expert in two chunks would be "
                         "added twice by the chunk sum")
    chunks = tuple(tuple(ids[i:i + chunk_experts]) for i in range(0, len(ids), chunk_experts))
    return ChunkPlan(num_experts=num_experts, chunk_experts=chunk_experts, chunks=chunks)


# ---------------------------------------------------------------------------
# The per-chunk routing tables
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkTables:
    """``safe_slot_table`` / ``keep_table`` for one chunk, ready for ``split_routing``.

    ``dummy_slot`` is 0 with keep value 0, exactly as the production placement
    does it for CPU experts: a legal index the Ascend routing kernel accepts,
    whose contribution the combine multiplies away. Slot 0 belonging to the
    chunk's first expert is deliberate -- the dummy rows must land on a slot
    that holds real weights, because unloaded weights can be NaN and NaN * 0 is
    NaN (placement.py argues the same for keeping one resident expert per
    layer).
    """

    chunk_index: int
    expert_ids: tuple[int, ...]
    safe_slot_table: torch.Tensor
    keep_table: torch.Tensor

    @property
    def num_slots(self) -> int:
        return len(self.expert_ids)


def chunk_tables(plan: ChunkPlan, chunk_index: int, *, device: torch.device | str = "cpu",
                 id_dtype: torch.dtype = torch.int32,
                 weight_dtype: torch.dtype = torch.float32) -> ChunkTables:
    """Build the two gather tables for chunk ``chunk_index``."""
    if not 0 <= chunk_index < plan.num_chunks:
        raise IndexError(f"chunk {chunk_index} out of range (plan has {plan.num_chunks})")
    ids = plan.chunks[chunk_index]
    slot = torch.zeros(plan.num_experts, dtype=torch.int64)
    keep = torch.zeros(plan.num_experts, dtype=torch.float32)
    for pos, e in enumerate(ids):
        slot[e] = pos
        keep[e] = 1.0
    return ChunkTables(
        chunk_index=chunk_index,
        expert_ids=ids,
        safe_slot_table=slot.to(device=device, dtype=id_dtype),
        keep_table=keep.to(device=device, dtype=weight_dtype),
    )


def split_routing_for_chunk(topk_ids: torch.Tensor, topk_weights: torch.Tensor,
                            tables: ChunkTables) -> NpuRouting:
    """The chunk's half of a routing decision, via the production ``split_routing``.

    Deliberately a thin wrapper: what was verified on-card is ``split_routing``
    driven by a (slot table, keep table) pair, so the streaming path must call
    that function rather than a copy of it.
    """
    assert_routing_dtypes(topk_ids, topk_weights, "kt_offload split_routing_for_chunk")
    return split_routing(topk_ids, topk_weights, tables.safe_slot_table, tables.keep_table)


# ---------------------------------------------------------------------------
# Oracles. Both are set-based: a numeric closeness check cannot see a route
# that landed on the wrong expert with a plausible weight.
# ---------------------------------------------------------------------------


def check_chunk_plan(plan: ChunkPlan, expert_ids: Sequence[int] | None = None) -> list[str]:
    """Violations of the tiling contract; ``[]`` means it holds."""
    want = sorted(range(plan.num_experts) if expert_ids is None else (int(e) for e in expert_ids))
    got = sorted(plan.covered)
    bad: list[str] = []
    if got != want:
        missing, extra = sorted(set(want) - set(got)), sorted(set(got) - set(want))
        bad.append(f"chunks cover {len(got)} expert slots, expected {len(want)} "
                   f"(missing {missing[:8]}, extra {extra[:8]})")
    if len(set(got)) != len(got):
        bad.append("an expert appears in more than one chunk: the chunk sum would double it")
    over = [k for k, c in enumerate(plan.chunks) if len(c) > plan.chunk_experts]
    if over:
        bad.append(f"chunks {over[:8]} exceed chunk_experts={plan.chunk_experts}")
    if any(len(c) == 0 for c in plan.chunks):
        bad.append("an empty chunk would stream nothing and still cost a pass")
    return bad


def check_chunk_cover(plan: ChunkPlan, topk_ids: torch.Tensor,
                      per_chunk_ids: Iterable[torch.Tensor],
                      per_chunk_weights: Iterable[torch.Tensor],
                      topk_weights: torch.Tensor) -> list[str]:
    """Violations of the per-forward contract; ``[]`` means it holds.

    The criterion used on-card:
      * for every token, the multiset of *logical* experts that contribute with a
        non-zero weight across all chunks equals the original routing's;
      * the weights of the contributing routes are bit-identical to the original
        (a chunk must not rescale anything -- the combine already applies them);
      * every route is claimed by exactly one chunk, so nothing is dropped and
        nothing is counted twice.
    """
    bad: list[str] = []
    ids_l = list(per_chunk_ids)
    w_l = list(per_chunk_weights)
    if len(ids_l) != plan.num_chunks or len(w_l) != plan.num_chunks:
        return [f"got {len(ids_l)} id tensors and {len(w_l)} weight tensors for "
                f"{plan.num_chunks} chunks"]
    membership = torch.full((plan.num_experts,), -1, dtype=torch.int64)
    for k, c in enumerate(plan.chunks):
        for e in c:
            membership[e] = k

    orig = topk_ids.long()
    claimed = torch.zeros_like(orig, dtype=torch.int64)
    for k, (cid, cw) in enumerate(zip(ids_l, w_l)):
        if cid.shape != orig.shape or cw.shape != orig.shape:
            bad.append(f"chunk {k}: shape {tuple(cid.shape)}/{tuple(cw.shape)} != "
                       f"{tuple(orig.shape)} -- the split must preserve shape")
            continue
        live = cw != 0
        claimed += live.long()
        want_live = membership[orig] == k
        if not torch.equal(live, want_live):
            n = int((live != want_live).sum())
            bad.append(f"chunk {k}: {n} route(s) live in the wrong chunk")
        if live.any():
            pos_in_chunk = cid.long()[live]
            chunk_ids = torch.tensor(plan.chunks[k], dtype=torch.int64)
            if int(pos_in_chunk.max()) >= len(chunk_ids) or int(pos_in_chunk.min()) < 0:
                bad.append(f"chunk {k}: slot index outside [0,{len(chunk_ids)})")
            else:
                back = chunk_ids[pos_in_chunk]
                if not torch.equal(back, orig[live]):
                    n = int((back != orig[live]).sum())
                    bad.append(f"chunk {k}: {n} route(s) map back to the wrong logical expert")
            if not torch.equal(cw[live], topk_weights[live]):
                bad.append(f"chunk {k}: live weights differ from the original routing")

    if not torch.equal(claimed, torch.ones_like(claimed)):
        never, twice = int((claimed == 0).sum()), int((claimed > 1).sum())
        bad.append(f"{never} route(s) claimed by no chunk, {twice} by more than one "
                   f"(coverage must be exactly 1 per route)")
    return bad


# ---------------------------------------------------------------------------
# The eager-only precondition
# ---------------------------------------------------------------------------


def assert_eager_only(num_tokens: int, capture_sizes: Sequence[int] | None,
                      min_tokens: int, *, where: str = "stream_prefill") -> None:
    """Refuse to stream inside a captured graph.

    A streaming forward allocates a slot, casts into it and frees it, none of
    which is legal inside a capture (an allocation made during capture comes
    from the graph's private pool, and the cast is a host-side decision the
    replay would never re-take). Safe today only because a forward with >= 512
    tokens is not one of the captured sizes ([1,2,4,8]) -- a *configuration*
    fact, not a law, so it is asserted. A capture list that includes a large
    size must fail loudly here instead of baking one chunk's weights into a
    graph that replays them for every later prefill.
    """
    if num_tokens < min_tokens:
        raise ValueError(f"{where}: refusing to stream a forward of {num_tokens} tokens "
                         f"(threshold {min_tokens}); the caller must take the hybrid path")
    if capture_sizes and num_tokens in set(int(s) for s in capture_sizes):
        raise ValueError(
            f"{where}: {num_tokens} tokens is in cudagraph_capture_sizes "
            f"{sorted(set(int(s) for s in capture_sizes))}. Streaming allocates and casts per "
            "forward, which a graph capture cannot record; raise the streaming threshold above "
            "the largest captured size, or drop that size from the capture list.")


def resident_chunk_ids(placement: LayerPlacement) -> list[int]:
    """The offloaded (non-resident) logical ids of a layer, ascending.

    Streaming these and leaving the resident ones on their existing slots is
    cheaper: resident weights are already on the card. Kept here rather than
    inlined so the chunk plan and the coverage oracle use the same list.
    """
    return [e for e in range(placement.num_experts) if placement.slot_of(e) == CPU_SLOT]
