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
    hit = _cached_slice(cos_sin_cache, positions, rope_head_dim)
    if hit is not None:
        return hit[0], hit[1]
    half = rope_head_dim // 2
    row = cos_sin_cache[positions].to(torch.float32)
    cos = widen_interleaved(row[..., :half])
    sin = widen_interleaved(row[..., half:])
    _remember_slice(cos_sin_cache, positions, rope_head_dim, cos, sin)
    return cos, sin


def neg_sin_for_slice(cos_sin_cache: torch.Tensor, positions: torch.Tensor,
                      rope_head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(cos, -sin)`` -- what the **inverse** rotation takes.

    Negating here rather than at the call site is what lets the negation be
    cached with the pair: the inverse path used to compute ``-sin`` fresh on
    every layer, 40 extra ``Neg`` kernels a step for a value that is the same
    every time.
    """
    cos, sin = cos_sin_for_slice(cos_sin_cache, positions, rope_head_dim)
    entry = _CACHE.get("entry")
    if entry is not None and entry[1] is cos:
        if entry[2] is None:
            entry[2] = -sin
        return cos, entry[2]
    return cos, -sin


#: One entry, holding ``[key, cos, neg_sin_or_None, sin]``. Not an LRU: within
#: a forward pass every caller asks for the same ``(cache, positions)``, and
#: across passes the old entry is dead, so anything larger would only keep
#: dead tensors alive.
_CACHE: dict[str, object] = {}


def _forward_token() -> object | None:
    """The object identifying the forward pass now running, or None.

    🔴 This -- not the tensor addresses -- is what makes the cache safe. The
    runner reuses one persistent ``positions`` buffer, so its ``data_ptr()``
    is the **same address with different contents** on the next step; keying
    on addresses alone would serve one step's angles to the next. Outside a
    forward pass (unit tests, profile runs) there is no token and nothing is
    cached, which is the conservative direction.
    """
    try:
        from vllm.forward_context import get_forward_context

        return get_forward_context()
    except Exception:  # noqa: BLE001 -- no context is a valid answer
        return None


def _slice_key(cos_sin_cache: torch.Tensor, positions: torch.Tensor,
               rope_head_dim: int, token: object) -> tuple:
    return (id(token), cos_sin_cache.data_ptr(), positions.data_ptr(),
            tuple(positions.shape), int(positions.dtype.itemsize),
            rope_head_dim)


def _cached_slice(cos_sin_cache, positions, rope_head_dim):
    """The ``(cos, sin)`` computed earlier **in this same forward pass**.

    V4.1 asks for exactly one ``(cache, positions)`` pair per step and then
    rotates with it 80 times: the forward RoPE in ``qkv_prolog`` and the
    inverse in ``rope`` each call this once per layer, on the same table and
    the same positions. Measured 2026-09-22 (§6.9.15): the repeated widening
    was 312 kernels and **0.86 ms of a 28.8 ms decode step**, recomputing an
    identical answer.

    ⚠ Under FULL graph capture this Python runs once and the reused tensor
    becomes a graph-internal value, which is the same saving at replay.
    """
    token = _forward_token()
    if token is None:
        return None
    entry = _CACHE.get("entry")
    if entry is None or entry[0] != _slice_key(cos_sin_cache, positions,
                                               rope_head_dim, token):
        return None
    return entry[1], entry[3]


def _remember_slice(cos_sin_cache, positions, rope_head_dim, cos, sin) -> None:
    token = _forward_token()
    if token is None:
        return
    _CACHE["entry"] = [
        _slice_key(cos_sin_cache, positions, rope_head_dim, token), cos, None,
        sin
    ]
