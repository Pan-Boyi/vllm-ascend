"""Properties of the two-level candidate filter, at the REAL parameters.

``candidate_topk_blocks = 2048`` and ``candidate_block_size = 8``, so level one keeps
2048 blocks of 8 positions = **16384 positions**.  Below that there is nothing to
discard and the filter is exactly a no-op -- which means a 16K-or-shorter context,
GSM8K included, **cannot exercise it at all**.  Everything this file checks would
pass on an implementation whose filter is missing entirely, if it were only ever run
at short context.

``select_candidate_blocks`` is a pure function of (logits, compress_lens,
topk_blocks, block_size), so the path can be driven at 16K+ in milliseconds with no
model run, no weights and no card.  That is the cheap way to cover it **without
changing the parameters being verified** -- shrinking ``candidate_topk_blocks`` to
make the filter bite at short length tests the wiring but no longer tests this
configuration.

The assertions are properties, not a recording of current output: each one is
something an Ascend implementation must reproduce, and three of them are invisible
below 16384 positions.

    python test_candidate_filter.py
"""

from __future__ import annotations

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref"))
from model import select_candidate_blocks                              # noqa: E402

TOPK_BLOCKS = 2048          # config.json: candidate_topk_blocks
BLOCK = 8                   # config.json: candidate_block_size
BINDS_ABOVE = TOPK_BLOCKS * BLOCK      # 16384 -- the filter is a no-op at or below this

fails = 0


def check(name: str, ok: bool, detail: str = ""):
    global fails
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        fails += 1


def scores_for(n_pos: int, reach: int, seed: int = 0) -> torch.Tensor:
    """Index scores for one query: finite up to ``reach``, -inf beyond (unreachable)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(1, 1, n_pos, generator=g) * 10 - 5
    s[..., reach:] = -torch.inf
    return s


def main():
    print(f"candidate_topk_blocks={TOPK_BLOCKS} candidate_block_size={BLOCK} "
          f"=> filter binds only above {BINDS_ABOVE} positions")

    print("\n1. at or below 16384 positions the filter must be an exact no-op")
    for n in (27, 256, BINDS_ABOVE - 1, BINDS_ABOVE):
        s = scores_for(n, n)
        keep = select_candidate_blocks(s, torch.tensor([[n]]), TOPK_BLOCKS, BLOCK)
        check(f"n={n:>5}: every reachable position kept",
              bool(keep.all()), f"kept {int(keep.sum())}/{n}")

    print("\n2. above 16384 it must keep exactly 2048 blocks' worth and no more")
    n = BINDS_ABOVE + 4096
    s = scores_for(n, n)
    keep = select_candidate_blocks(s, torch.tensor([[n]]), TOPK_BLOCKS, BLOCK)
    check(f"n={n}: kept == {BINDS_ABOVE}",
          int(keep.sum()) == BINDS_ABOVE, f"kept {int(keep.sum())}")
    # and the kept set must be whole blocks, never a partial one
    blocks = keep[0, 0].reshape(-1, BLOCK)
    per_block = blocks.sum(-1)
    check("selection is whole blocks only",
          bool(((per_block == 0) | (per_block == BLOCK)).all()),
          f"{int((per_block == BLOCK).sum())} full blocks, "
          f"{int(((per_block > 0) & (per_block < BLOCK)).sum())} partial")

    print("\n3. the block holding the newest position is pinned, even when it scores worst")
    # This is the property most likely to be dropped, and it is invisible below 16384:
    # ref/model.py masks that block to +inf so a full older block cannot outscore it.
    n = BINDS_ABOVE + 4096
    s = scores_for(n, n, seed=1)
    last_block = (n - 1) // BLOCK
    s[..., last_block * BLOCK:] = -4.999          # worst finite score in the tensor
    keep = select_candidate_blocks(s, torch.tensor([[n]]), TOPK_BLOCKS, BLOCK)
    check("newest block kept despite the lowest score",
          bool(keep[0, 0, last_block * BLOCK:].all()),
          f"kept {int(keep[0, 0, last_block * BLOCK:].sum())}/{n - last_block * BLOCK}")

    print("\n4. unreachable positions are never kept, even when blocks are scarce")
    # Fewer reachable blocks than topk_blocks: the leftover picks come back -inf and
    # must be dropped, not kept as if they were real candidates.
    n = BINDS_ABOVE + 4096
    reach = 1000
    s = scores_for(n, reach, seed=2)
    keep = select_candidate_blocks(s, torch.tensor([[reach]]), TOPK_BLOCKS, BLOCK)
    beyond = keep[0, 0, reach:]
    # the block containing position reach-1 is partly unreachable and is pinned, so
    # allow that one block's tail; nothing past it may survive
    tail_end = ((reach - 1) // BLOCK + 1) * BLOCK
    check("nothing kept past the newest block",
          not bool(keep[0, 0, tail_end:].any()),
          f"{int(keep[0, 0, tail_end:].sum())} kept beyond position {tail_end}")
    check("reachable region fully kept when it fits",
          bool(keep[0, 0, :reach].all()),
          f"kept {int(keep[0, 0, :reach].sum())}/{reach}")

    print("\n5. a length that is not a multiple of the block size")
    n = BINDS_ABOVE + 4096 + 3
    s = scores_for(n, n, seed=3)
    keep = select_candidate_blocks(s, torch.tensor([[n]]), TOPK_BLOCKS, BLOCK)
    check("mask width equals the logits width", keep.shape == s.shape,
          f"{tuple(keep.shape)} vs {tuple(s.shape)}")
    check("still whole blocks except the truncated last one",
          int(keep.sum()) in (BINDS_ABOVE, BINDS_ABOVE - BLOCK + (n % BLOCK)),
          f"kept {int(keep.sum())}")

    print("\n6. determinism: the same input twice gives the same mask")
    n = BINDS_ABOVE + 4096
    s = scores_for(n, n, seed=4)
    a = select_candidate_blocks(s, torch.tensor([[n]]), TOPK_BLOCKS, BLOCK)
    b = select_candidate_blocks(s.clone(), torch.tensor([[n]]), TOPK_BLOCKS, BLOCK)
    check("bit-identical across calls", bool(torch.equal(a, b)))

    print()
    if fails:
        print(f"{fails} property/properties FAILED")
    else:
        print("all properties hold on the CPU reference")
        print(f"NOTE: properties 2-5 are vacuous at or below {BINDS_ABOVE} positions, so a "
              f"GSM8K run\n      at <=16K context proves nothing about them.")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
