# SPDX-License-Identifier: Apache-2.0
"""Widen a half-width cos/sin row to the width the rotary operator takes.

``inplace_partial_rotary_mul`` wants ``cos`` as wide as the slice it rotates
(``cos.size(-1) == end - start``), while upstream's ``cos_sin_cache[pos]`` holds
``rope_head_dim/2`` cosines followed by the same number of sines -- one of each
per interleaved pair. Getting from one to the other is a widening, and there
are two widenings that produce the right shape:

    A  [c0, c0, c1, c1, ...]      each value to its pair's two channels
    B  [c0, c1, ..., c0, c1, ...] the half repeated

Both are ``rope_head_dim`` wide, so nothing downstream can tell them apart. B
rotates by the wrong angle on every other pair, which is a regular,
plausible-looking error and not a detectable one.

A is correct, and not because it is the more natural reading: it is what the
Ascend path that already drives this operator does.
``ops/rope_dsv4.py:218`` builds its table as
``freqs.cos().repeat_interleave(2, dim=-1)``, and that table is what V4.0 feeds
to the same operator today. The choice is copied from a working consumer rather
than derived from the docstring of the thing being called.
"""

from __future__ import annotations

import torch


def widen_interleaved(half: torch.Tensor) -> torch.Tensor:
    """``[..., n]`` of per-pair values to ``[..., 2n]`` of per-channel values.

    Channel ``i`` belongs to pair ``i // 2``, which is what makes this
    ``repeat_interleave`` and not ``repeat``.
    """
    return half.repeat_interleave(2, dim=-1)


def cos_sin_for_slice(cos_sin_cache: torch.Tensor, positions: torch.Tensor,
                      rope_head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The operator's ``(cos, sin)`` for a batch of positions.

    ``cos_sin_cache`` is upstream's: ``[max_pos, rope_head_dim]``, the cosines
    in the first half and the sines in the second. Returned already widened, so
    the caller never sees the half-width form and cannot widen it twice.
    """
    if cos_sin_cache.shape[-1] != rope_head_dim:
        raise ValueError(
            f"cos_sin_cache is {cos_sin_cache.shape[-1]} wide and the rope "
            f"head dim is {rope_head_dim}. The cache holds half a rotation's "
            "cosines followed by its sines, so these must be equal; a mismatch "
            "means one of them is the *pair* count rather than the channel "
            "count, and the halves would be split in the wrong place.")
    half = rope_head_dim // 2
    row = cos_sin_cache[positions].to(torch.float32)
    return widen_interleaved(row[..., :half]), widen_interleaved(row[..., half:])
