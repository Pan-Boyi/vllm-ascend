# SPDX-License-Identifier: Apache-2.0
"""Block 0 is genuinely reserved -- checked by running the allocator.

The question this answers, posed as a gate: if the prolog is going to send
invalid rows to a scratch slot, **which line keeps the allocator from handing
that slot out again?** "A true reservation" and "we agreed not to use it"
look identical in code, and only the first survives a long sequence.

The answer is upstream's null block:

  kv_cache_coordinator.py:98          ONE BlockPool, shared by every KV cache
                                      group -- so this covers the SWA group
  block_pool.py:190                   self.null_block = free_block_queue.popleft()
                                      "To represent a placeholder block with
                                       block_id=0"  <- removed from the
                                      allocatable set, not merely labelled
  block_pool.py:724, :747             `if block.ref_cnt == 0 and not
                                      block.is_null` -- it is never put back
                                      on the free queue
  single_type_kv_cache_manager.py:288 "Skip blocks are padded with null
                                      blocks" -- it is the designated padding
                                      target already

Checked by construction rather than by reading: a grep proves the text is
there, not that the object behaves that way.
"""

import pytest


def _pool(num_blocks=8):
    from vllm.v1.core.block_pool import BlockPool

    return BlockPool(num_gpu_blocks=num_blocks, enable_caching=False,
                     hash_block_size=16)


def test_block_zero_is_removed_from_the_allocatable_set():
    pool = _pool()
    assert pool.null_block.block_id == 0
    assert pool.null_block.is_null is True

    # Drain every allocatable block; block 0 must never appear.
    got = pool.get_new_blocks(pool.get_num_free_blocks())
    ids = [b.block_id for b in got]
    assert 0 not in ids, (
        f"the allocator handed out block 0: {ids}. It is supposed to be "
        "popped off the free queue at block_pool.py:190.")
    assert len(set(ids)) == len(ids)


def test_freeing_does_not_return_block_zero_to_the_queue():
    """The half that makes it a reservation rather than a head start.

    Popping it once would be undone by any code path that frees it back. The
    `not block.is_null` guards at :724 and :747 are what make it permanent.
    """
    pool = _pool()
    before = pool.get_num_free_blocks()
    pool.free_blocks([pool.null_block])
    assert pool.get_num_free_blocks() == before, (
        "freeing the null block put it back on the free queue; it is no "
        "longer reserved and a scratch slot built on it would be handed to a "
        "real request")
