# SPDX-License-Identifier: Apache-2.0
"""The q-norm / RoPE / SWA-insert prolog, decomposed.

Upstream fuses these steps into one CUDA op per cache dtype, none of which
exist on Ascend; ``_prepare_and_attn_eager`` reaches this on every forward.

The q side is RoPE only. The shared fused operator ends with
``bool apply_q_norm=True``; V4.0 takes the default, but V4.1's attention path
passes ``False`` at every dtype branch, matching the reference implementation
(inference/model.py, ``Attention.forward``: ``wq_b`` then
``apply_rotary_emb``, with no norm in between). The kv side does not norm
either: ``kv`` arrives already normed from ``_split_qkv_and_norm``.

Computed in float32 and rounded once at the end; the fused kernel's internal
rounding points are not visible, so the two may differ by a few ULP.
"""

from __future__ import annotations

import torch

from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice


def rms_norm_weightless(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row RMS norm with no learned weight, in float32.

    Not used on the V4.1 path: this is the fused operator's
    ``apply_q_norm=True`` branch (V4.0's), and V4.1 passes ``False``.
    """
    acc = x.to(torch.float32)
    variance = acc.pow(2).mean(dim=-1, keepdim=True)
    return acc * torch.rsqrt(variance + eps)


def rope_tail(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              rope_head_dim: int) -> torch.Tensor:
    """GPT-J rotation over the trailing ``rope_head_dim``, interleaved.

    ``cos``/``sin`` are per-token and already widened to ``rope_head_dim``;
    they broadcast across any axes between the token axis and the last one, so
    the same function serves ``q`` ``[T, H, D]`` and ``kv`` ``[T, D]``.

    The NoPE part is the *leading* ``head_dim - rope_head_dim``. Rotating the
    whole row instead is a change no shape check sees.
    """
    head_dim = x.shape[-1]
    if rope_head_dim > head_dim:
        raise ValueError(
            f"rope width {rope_head_dim} exceeds the head width {head_dim}")
    nope = head_dim - rope_head_dim
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    # clone: `to` returns the same tensor when the dtype already matches, and
    # the in-place write below must not rotate the caller's kv (upstream keeps
    # kv unchanged; attention reads it only via swa_kv_cache).
    out = x.to(torch.float32).clone()
    tail = out[..., nope:]
    pairs = tail.reshape(*tail.shape[:-1], rope_head_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    # cos/sin are widened per channel and both channels of a pair carry the
    # same value.
    c = cos.reshape(*cos.shape[:-1], rope_head_dim // 2, 2)[..., 0]
    s = sin.reshape(*sin.shape[:-1], rope_head_dim // 2, 2)[..., 0]
    rotated = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1)
    out[..., nope:] = rotated.reshape(*tail.shape)
    return out


def _payload_that_leaves_invalid_rows_untouched(current: torch.Tensor,
                                                rows: torch.Tensor,
                                                valid: torch.Tensor):
    """Valid rows get the new value; invalid rows get their current value.

    Invalid rows are clamped to slot 0 (the reserved null block), and writing
    back what is already there makes them a no-op for any reader of that
    block. This relies on no valid row also targeting slot 0 (a duplicate
    index would make the winning write implementation-defined), and on the
    gather of ``current`` being ordered before the scatter on the same stream.

    Under graph mode ``rows`` is padded to the capture size while the slot
    mapping has the real token count, so row ``i`` goes to slot ``i`` and
    the extra padded rows are dropped.
    """
    n = current.shape[0]
    if rows.shape[0] != n:
        rows = rows[:n]
    return torch.where(valid.unsqueeze(-1), rows.to(current.dtype), current)


def scatter_rows(cache: torch.Tensor, rows: torch.Tensor,
                 slot_mapping: torch.Tensor, block_size: int) -> None:
    """Write one row per token into a paged ``[blocks, block_size, dim]`` cache.

    Two slot-mapping shapes are legal, decided by the cache dtype in the DSA
    metadata builder rather than by the caller::

        FP8_ATTENTION and not a5_bf16_kv   ->  [num_tokens]      flat slots
        otherwise                          ->  [num_tokens, 2]   (block, offset)

    On A5 with a bf16 KV cache the second is the default.

    Both are accepted and dispatched on shape: a ``[T, 2]`` mapping read flat
    is ``2T`` in-range slots belonging to other tokens, with nothing raised.

    ``-1`` means "do not write"; it arrives from padding, not from
    already-cached tokens (cached tokens are not scheduled). Such rows are
    clamped to slot 0 -- the null block 0, which is reserved -- and write back
    the value already there, so they change nothing.
    """
    if slot_mapping.ndim == 2 and slot_mapping.shape[-1] == 2:
        blocks, offsets = slot_mapping[:, 0], slot_mapping[:, 1]
        valid = blocks >= 0
        b_i = torch.where(valid, blocks, blocks.new_zeros(())).to(torch.long)
        o_i = torch.where(valid, offsets, offsets.new_zeros(())).to(torch.long)
        if blocks.numel() == 0:
            return
        payload = _payload_that_leaves_invalid_rows_untouched(
            cache[b_i, o_i], rows, valid)
        cache.index_put_((b_i, o_i), payload)
        return

    if slot_mapping.ndim != 1:
        raise ValueError(
            f"slot_mapping is {tuple(slot_mapping.shape)}; this takes flat "
            "[num_tokens] slots or [num_tokens, 2] (block, offset) pairs.")
    valid = slot_mapping >= 0
    slots = torch.where(valid, slot_mapping, slot_mapping.new_zeros(())).to(torch.long)
    if slots.numel() == 0:
        return
    flat_cache = cache.view(-1, cache.shape[-1])
    payload = _payload_that_leaves_invalid_rows_untouched(
        flat_cache[slots], rows, valid)
    flat_cache.index_copy_(0, slots, payload)


def qnorm_rope_kv_insert(*, q: torch.Tensor, kv: torch.Tensor,
                         positions: torch.Tensor, cos_sin_cache: torch.Tensor,
                         swa_kv_cache: torch.Tensor,
                         slot_mapping: torch.Tensor, eps: float,
                         rope_head_dim: int) -> torch.Tensor:
    """The bf16 branch, decomposed. Returns the rotated ``q``.

    ``q`` is ``[tokens, heads, head_dim]`` and ``kv`` is ``[tokens, head_dim]``;
    both are rotated over the same trailing slice at the same positions, so the
    cos/sin pair is built once.

    ``eps`` is unused and kept for signature parity with the operator, which
    ignores it too when ``apply_q_norm`` is false.
    """
    cos, sin = cos_sin_for_slice(cos_sin_cache, positions, rope_head_dim)
    # No q norm: V4.1 passes apply_q_norm=False (see the module docstring).
    q_out = rope_tail(q, cos, sin, rope_head_dim)
    # kv arrives normed; the op's name says kv_rope, not kv_norm_rope.
    kv_out = rope_tail(kv, cos, sin, rope_head_dim)
    scatter_rows(swa_kv_cache, kv_out, slot_mapping, swa_kv_cache.shape[1])
    return q_out.to(q.dtype)
