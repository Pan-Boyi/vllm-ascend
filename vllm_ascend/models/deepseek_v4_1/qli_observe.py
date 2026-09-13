# SPDX-License-Identifier: Apache-2.0
"""What QuantLightningIndexerV2 was actually handed, and what it may read.

🔴 This operator is the opposite of ``SparseFlashMla`` on the one property
that matters here. Measured on the bench by the kernel line:

    SparseFlashMla              does not read past seqused_ori_kv
    QuantLightningIndexerV2     reads a **block-aligned** range --
                                ceil(seqused_k / block_size) * block_size

With ``block_size = 128`` and a five-token request that is ``[0, 128)``: five
real tokens and 123 slots of whatever else is in that block. If those slots
were written by an earlier request they hold that request's keys, and the
selection comes back referring to them. Nothing errors; the indices are in
range and the values are real numbers belonging to someone else.

⚠️ Same parameter name, opposite behaviour, one code path feeding both. That
is the shape this project has hit repeatedly, and it is why the two lengths
are observed rather than assumed to behave alike.

Print-only. Nothing here changes a tensor, a shape or a call, so a run with
it on must be byte-identical to one without -- which is what makes "the
output matched the baseline" evidence that the instrument did not disturb
the thing it measures.
"""

from __future__ import annotations

import torch

#: How many calls are reported in full before switching to change-only. Eight
#: covers the first two index-source layers across a couple of steps, which is
#: where a wrong length would already be visible.
VERBOSE_CALLS = 8

_CALLS = 0
_LAST: tuple | None = None

#: Block ids this process has already handed to some request, and which
#: request. Used only to say whether a block is being reused -- it holds ids,
#: never contents.
_BLOCK_FIRST_SEEN: dict = {}


def reset_for_test() -> None:
    global _CALLS, _LAST
    _CALLS = 0
    _LAST = None
    _BLOCK_FIRST_SEEN.clear()


def observe(*, prefix: str, seqused_k: torch.Tensor,
            cu_seqlens_q: torch.Tensor, block_table: torch.Tensor,
            block_size: int) -> None:
    """Report this call's lengths and whether its blocks are second-hand."""
    global _CALLS, _LAST

    lengths = _small_list(seqused_k)
    starts = _small_list(cu_seqlens_q)
    blocks = _small_list(block_table.reshape(-1))
    reused = _reused_blocks(prefix, blocks)
    aligned = [_aligned(n, block_size) for n in lengths if n is not None]

    state = (tuple(lengths), tuple(starts), tuple(blocks), tuple(reused))
    _CALLS += 1
    if _CALLS > VERBOSE_CALLS and state == _LAST:
        return
    _LAST = state

    from vllm.logger import logger

    logger.info(
        "DSV4.1 QLI call #%d %s: seqused_k=%s cu_seqlens_q=%s block_size=%d "
        "-> operator may read up to %s (block-aligned). blocks=%s "
        "reused-by-an-earlier-request=%s",
        _CALLS, prefix, lengths, starts, block_size, aligned, blocks,
        reused if reused else "none")


def _aligned(length: int, block_size: int) -> int:
    """The block-aligned range the operator may actually read.

    Reported alongside the length, not instead of it: the gap between the two
    is the number of slots that are read and were never written by this
    request, and that number is the whole point of the line.
    """
    if block_size <= 0:
        return length
    return -(-length // block_size) * block_size


def _reused_blocks(prefix: str, blocks: list) -> list:
    """Which of these blocks some *other* request used first.

    ⚠️ A block being reused is not by itself a defect -- the allocator is
    supposed to recycle them. It matters only together with the line above:
    a block-aligned read past this request's own tokens lands in slots that
    the previous owner wrote.
    """
    reused = []
    for block in blocks:
        if block is None or block < 0:
            continue
        owner = _BLOCK_FIRST_SEEN.setdefault(block, prefix)
        if owner != prefix:
            reused.append(block)
    return reused


def _small_list(tensor: torch.Tensor, limit: int = 8) -> list:
    """A few values off the device, or ``[]`` when there is nothing to read.

    ⚠️ Capped. This runs per call on the model path and a full block table is
    thousands of entries; a debug line that costs a large transfer would be
    the instrument disturbing what it measures, which is the failure mode
    this file is written to avoid.
    """
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        return []
    return [int(v) for v in tensor.reshape(-1)[:limit].detach().cpu()]
