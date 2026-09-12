# SPDX-License-Identifier: Apache-2.0
"""The q-norm / RoPE / SWA-insert prolog, decomposed.

Upstream fuses five steps into one CUDA op per cache dtype, and all three of
those symbols are absent on Ascend. Reached from ``_prepare_and_attn_eager`` on
every forward, so this is not a dormant path.

The contract is not inferred from the operator's name. ``deepseek_v4/cpu/
cpu_sparse.py:427`` states it, for the same fused op:

    Q side does per-head weight-free RMSNorm + GPT-J RoPE (zero-filling padded
    head slots); KV side does GPT-J RoPE + UE8M0 FP8 quant + paged insert.

Two things follow that a reading of the name would have got wrong. The q norm
takes no learned weight -- the op is passed ``eps`` and nothing else, and the
layer's own ``q_norm`` is an ``RMSNorm(q_lora_rank)`` applied much earlier, to
``qr``, not to this per-head ``q``. And the kv side does **not** norm: ``kv``
arrives already normed from ``_split_qkv_and_norm``, which is why the op's name
says ``kv_rope`` and not ``kv_norm_rope``.

⚠️ Unverified, and stated rather than assumed:

* **Numerical equivalence with the fused kernel has never been checked by
  anyone.** The decomposition is transcribed from the descriptions above, not
  from a comparison against the op's output.
* **Where bf16 rounding falls.** The fused kernel's internal precision is not
  visible from any of these call sites. This computes in float32 and rounds
  once, at the end. If the kernel rounds between the norm and the rotation --
  as the indexer's K writer does, where it is load-bearing -- the two differ by
  a few ULP in a way no shape or dtype check would show.

Torch first, deliberately. Every step here has a named Ascend operator, and the
operator path is worth taking once these values are known to be right; taking
it first would mean debugging a decomposition and a binding at the same time.
"""

from __future__ import annotations

import torch

from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice


def rms_norm_weightless(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row RMS norm with no learned weight, in float32.

    "Per-head" for ``q`` of ``[tokens, heads, head_dim]`` is the same statement
    as "per-row" over the last axis -- which is why this takes no axis
    argument: there is no other axis it could be.
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

    # clone, not just to(float32): `to` returns the *same* tensor when the
    # dtype already matches, and the in-place write below would then rotate the
    # caller's input. Upstream rewrites q in place but states "kv is unchanged;
    # attention reads kv solely via swa_kv_cache" -- so mutating kv here would
    # be wrong, and silently so, since the rotated values are what gets written
    # to the cache either way.
    out = x.to(torch.float32).clone()
    tail = out[..., nope:]
    pairs = tail.reshape(*tail.shape[:-1], rope_head_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    # cos/sin are widened per channel; one value per pair is enough and both
    # channels of a pair carry the same one, so either index works. Taking [0]
    # rather than slicing keeps the pair axis aligned with `even`/`odd`.
    c = cos.reshape(*cos.shape[:-1], rope_head_dim // 2, 2)[..., 0]
    s = sin.reshape(*sin.shape[:-1], rope_head_dim // 2, 2)[..., 0]
    rotated = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1)
    out[..., nope:] = rotated.reshape(*tail.shape)
    return out


def scatter_rows(cache: torch.Tensor, rows: torch.Tensor,
                 slot_mapping: torch.Tensor, block_size: int) -> None:
    """Write one row per token into a paged ``[blocks, block_size, dim]`` cache.

    ``slot_mapping`` is flat slot indices; ``-1`` means "do not write", which is
    how padded and already-cached tokens arrive. Filtered rather than clamped:
    a clamp would write those rows to slot 0 of block 0, corrupting a real
    token's entry with no error.
    """
    if slot_mapping.ndim != 1:
        raise ValueError(
            f"slot_mapping must be flat [num_tokens]; got "
            f"{tuple(slot_mapping.shape)}. A [num_tokens, 2] mapping is the "
            "(block, offset) form some plans use, which indexes differently.")
    valid = slot_mapping >= 0
    if not bool(valid.all()):
        slots = slot_mapping[valid]
        rows = rows[valid]
    else:
        slots = slot_mapping
    if slots.numel() == 0:
        return
    cache.view(-1, cache.shape[-1]).index_copy_(
        0, slots.to(torch.long), rows.to(cache.dtype))


def qnorm_rope_kv_insert(*, q: torch.Tensor, kv: torch.Tensor,
                         positions: torch.Tensor, cos_sin_cache: torch.Tensor,
                         swa_kv_cache: torch.Tensor,
                         slot_mapping: torch.Tensor, eps: float,
                         rope_head_dim: int) -> torch.Tensor:
    """The bf16 branch, decomposed. Returns the rotated, normed ``q``.

    ``q`` is ``[tokens, heads, head_dim]`` and ``kv`` is ``[tokens, head_dim]``;
    both are rotated over the same trailing slice at the same positions, so the
    cos/sin pair is built once.
    """
    cos, sin = cos_sin_for_slice(cos_sin_cache, positions, rope_head_dim)
    q_out = rope_tail(rms_norm_weightless(q, eps), cos, sin, rope_head_dim)
    # kv arrives normed; the op's name says kv_rope, not kv_norm_rope.
    kv_out = rope_tail(kv, cos, sin, rope_head_dim)
    scatter_rows(swa_kv_cache, kv_out, slot_mapping, swa_kv_cache.shape[1])
    return q_out.to(q.dtype)
