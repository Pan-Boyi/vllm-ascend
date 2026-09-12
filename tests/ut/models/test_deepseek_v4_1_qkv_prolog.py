# SPDX-License-Identifier: Apache-2.0
"""The decomposed prolog, checked against the contract rather than the name.

The contract is stated in deepseek_v4/cpu/cpu_sparse.py:427 for the same fused
op: the Q side does per-head weight-free RMSNorm then GPT-J RoPE; the KV side
does RoPE only, because kv arrives already normed. Each of those three facts is
one the operator's name would have got wrong, so each has a test.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.qkv_prolog import (qnorm_rope_kv_insert,
                                                          rms_norm_weightless,
                                                          rope_tail,
                                                          scatter_rows)

HEAD_DIM = 512
ROPE_DIM = 64
NOPE = HEAD_DIM - ROPE_DIM
EPS = 1e-6


def _cache(max_pos=16):
    torch.manual_seed(11)
    angles = torch.rand(max_pos, ROPE_DIM // 2) * 2.5
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def test_the_norm_carries_no_learned_weight():
    """The op is passed eps and nothing else.

    The layer's own q_norm is RMSNorm(q_lora_rank) applied to qr much earlier,
    not to this per-head q. Multiplying by some weight here would scale every
    query head and raise nothing.
    """
    x = torch.randn(3, 4, HEAD_DIM)
    got = rms_norm_weightless(x, EPS)
    expect = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)
    torch.testing.assert_close(got, expect)
    # A unit-RMS row stays put; a weighted norm could not promise that.
    unit = torch.ones(1, 1, HEAD_DIM)
    torch.testing.assert_close(rms_norm_weightless(unit, 0.0), unit)


def test_the_norm_is_per_head_not_per_token():
    """Normalising across heads would mix queries that must stay independent."""
    x = torch.randn(2, 4, HEAD_DIM)
    x[:, 1] *= 100.0
    got = rms_norm_weightless(x, EPS)
    rms = got.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), rtol=1e-3, atol=1e-3)


def test_the_rotation_leaves_the_nope_part_alone():
    """The NoPE half is the leading head_dim - rope_head_dim.

    Rotating the whole row is a change no shape check sees.
    """
    x = torch.randn(2, HEAD_DIM)
    cos = torch.full((2, ROPE_DIM), 0.5)
    sin = torch.full((2, ROPE_DIM), 0.25)
    out = rope_tail(x, cos, sin, ROPE_DIM)
    torch.testing.assert_close(out[..., :NOPE], x[..., :NOPE].float())
    assert not torch.allclose(out[..., NOPE:], x[..., NOPE:].float())


def test_the_rotation_matches_a_per_pair_reference():
    x = torch.randn(3, HEAD_DIM)
    cache = _cache()
    positions = torch.tensor([1, 5, 9])
    from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice

    cos, sin = cos_sin_for_slice(cache, positions, ROPE_DIM)
    got = rope_tail(x, cos, sin, ROPE_DIM)

    half = ROPE_DIM // 2
    c = cache[positions][:, :half]
    s = cache[positions][:, half:]
    tail = x[:, NOPE:].reshape(3, half, 2).float()
    even, odd = tail[..., 0], tail[..., 1]
    expect = torch.stack((even * c - odd * s, odd * c + even * s),
                         dim=-1).reshape(3, ROPE_DIM)
    torch.testing.assert_close(got[:, NOPE:], expect)


def test_q_and_kv_rotate_at_the_same_positions():
    """One cos/sin pair serves both, broadcast across q's head axis.

    A per-head cos would be a different angle per head, which is not what GPT-J
    RoPE means and is invisible in the shapes.
    """
    cache = _cache()
    positions = torch.tensor([2, 6])
    q = torch.randn(2, 1, HEAD_DIM)
    kv = q[:, 0].clone()
    cache_buf = torch.zeros(4, 8, HEAD_DIM, dtype=torch.bfloat16)
    out_q = qnorm_rope_kv_insert(
        q=q, kv=kv, positions=positions, cos_sin_cache=cache,
        swa_kv_cache=cache_buf, slot_mapping=torch.tensor([0, 1]), eps=EPS,
        rope_head_dim=ROPE_DIM)
    from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice

    cos, sin = cos_sin_for_slice(cache, positions, ROPE_DIM)
    # Same input, one normed and one not: the rotations must agree once the
    # norm is applied to both.
    kv_rotated = rope_tail(kv, cos, sin, ROPE_DIM)
    torch.testing.assert_close(cache_buf.view(-1, HEAD_DIM)[:2].float(),
                               kv_rotated.to(torch.bfloat16).float())
    assert out_q.shape == q.shape


def test_kv_is_not_normed_here():
    """kv arrives normed from _split_qkv_and_norm; norming again is silent.

    The op's name says kv_rope, not kv_norm_rope, and the CPU override spells
    it out. A second norm would rescale every context row.
    """
    cache = _cache()
    kv = torch.randn(2, HEAD_DIM) * 50.0      # far from unit RMS
    cache_buf = torch.zeros(2, 4, HEAD_DIM, dtype=torch.bfloat16)
    qnorm_rope_kv_insert(
        q=torch.randn(2, 1, HEAD_DIM), kv=kv, positions=torch.tensor([0, 1]),
        cos_sin_cache=cache, swa_kv_cache=cache_buf,
        slot_mapping=torch.tensor([0, 1]), eps=EPS, rope_head_dim=ROPE_DIM)
    written = cache_buf.view(-1, HEAD_DIM)[:2].float()
    # The NoPE part passes through untouched, so its scale survives.
    assert written[:, :NOPE].abs().max() > 10.0, (
        "the context rows were rescaled; kv must not be normed here")


def test_a_negative_slot_is_skipped_not_clamped():
    """-1 means "do not write". Clamping would put the row in slot 0."""
    cache = torch.full((2, 4, HEAD_DIM), 7.0)
    rows = torch.ones(3, HEAD_DIM)
    scatter_rows(cache, rows, torch.tensor([-1, 3, -1]), 4)
    flat = cache.view(-1, HEAD_DIM)
    assert torch.all(flat[3] == 1.0)
    assert torch.all(flat[0] == 7.0), "slot 0 was overwritten by a skipped row"


def test_an_all_skipped_batch_writes_nothing():
    cache = torch.full((2, 4, HEAD_DIM), 7.0)
    scatter_rows(cache, torch.ones(2, HEAD_DIM), torch.tensor([-1, -1]), 4)
    assert torch.all(cache == 7.0)


def test_a_block_offset_slot_mapping_is_refused():
    with pytest.raises(ValueError, match=r"\(block, offset\)"):
        scatter_rows(torch.zeros(2, 4, HEAD_DIM), torch.ones(2, HEAD_DIM),
                     torch.zeros((2, 2), dtype=torch.int64), 4)


def test_the_rows_land_where_the_slots_say():
    cache = torch.zeros(3, 4, HEAD_DIM)
    rows = torch.arange(2, dtype=torch.float32).reshape(2, 1).expand(2, HEAD_DIM)
    scatter_rows(cache, rows.contiguous(), torch.tensor([11, 4]), 4)
    assert torch.all(cache[2, 3] == 0.0)     # slot 11 = block 2, offset 3
    assert torch.all(cache[1, 0] == 1.0)     # slot 4  = block 1, offset 0


def test_the_rotation_does_not_mutate_its_input():
    """`x.to(float32)` returns x when x is already float32.

    Without a clone the in-place write rotates the caller's tensor. Upstream
    rewrites q in place but records that "kv is unchanged; attention reads kv
    solely via swa_kv_cache" -- and the corruption is invisible, because the
    rotated values reach the cache either way. This test found that bug.
    """
    x = torch.randn(2, HEAD_DIM)                       # already float32
    before = x.clone()
    rope_tail(x, torch.full((2, ROPE_DIM), 0.5),
              torch.full((2, ROPE_DIM), 0.25), ROPE_DIM)
    torch.testing.assert_close(x, before)


def test_the_prolog_does_not_mutate_kv():
    cache = _cache()
    kv = torch.randn(2, HEAD_DIM)
    before = kv.clone()
    qnorm_rope_kv_insert(
        q=torch.randn(2, 1, HEAD_DIM), kv=kv, positions=torch.tensor([0, 1]),
        cos_sin_cache=cache,
        swa_kv_cache=torch.zeros(2, 4, HEAD_DIM, dtype=torch.bfloat16),
        slot_mapping=torch.tensor([0, 1]), eps=EPS, rope_head_dim=ROPE_DIM)
    torch.testing.assert_close(kv, before)
