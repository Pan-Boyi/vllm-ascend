# SPDX-License-Identifier: Apache-2.0
"""The decomposed prolog, checked against the contract rather than the name.

The contract is stated in deepseek_v4/cpu/cpu_sparse.py:427 for the same fused
op: the Q side does per-head weight-free RMSNorm then GPT-J RoPE; the KV side
does RoPE only, because kv arrives already normed. Each of those three facts is
one the operator's name would have got wrong, so each has a test.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.qkv_prolog import \
    rms_norm_weightless as qkv_prolog_rms
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


def test_the_block_offset_form_writes_where_the_pair_says():
    """On A5 with a bf16 KV cache this is the default form, not an alternative.

    dsa_v1.py:782 picks flat only when FP8_ATTENTION is available *and* bf16 KV
    is off -- the configuration we do not run. This used to be refused, which
    was right while nothing supported it and wrong as a permanent answer.
    """
    cache = torch.zeros(3, 4, HEAD_DIM)
    rows = torch.arange(2, dtype=torch.float32).reshape(2, 1).expand(2, HEAD_DIM)
    pairs = torch.tensor([[2, 3], [1, 0]], dtype=torch.int64)
    scatter_rows(cache, rows.contiguous(), pairs, 4)
    assert torch.all(cache[2, 3] == 0.0)
    assert torch.all(cache[1, 0] == 1.0)
    assert torch.all(cache[0] == 0.0), "nothing may land in an unaddressed block"


def test_the_two_forms_agree_on_the_same_slots():
    """The pair (b, o) and the flat slot b * block_size + o are the same place.

    Written as a test because reading one form as the other raises nothing: a
    [T, 2] mapping read flat is 2T slots with every offset used as a slot, and
    the rows land on real tokens' entries with values still in range.
    """
    block_size = 4
    pairs = torch.tensor([[2, 3], [1, 0], [0, 2]], dtype=torch.int64)
    flat = pairs[:, 0] * block_size + pairs[:, 1]
    rows = torch.randn(3, HEAD_DIM)

    by_pair = torch.zeros(3, block_size, HEAD_DIM)
    by_flat = torch.zeros(3, block_size, HEAD_DIM)
    scatter_rows(by_pair, rows, pairs, block_size)
    scatter_rows(by_flat, rows, flat, block_size)
    torch.testing.assert_close(by_pair, by_flat)


def test_a_negative_block_is_skipped_in_the_pair_form_too():
    cache = torch.full((2, 4, HEAD_DIM), 7.0)
    pairs = torch.tensor([[-1, -1], [1, 2]], dtype=torch.int64)
    scatter_rows(cache, torch.ones(2, HEAD_DIM), pairs, 4)
    assert torch.all(cache[1, 2] == 1.0)
    assert torch.all(cache[0, 0] == 7.0)


def test_an_unrecognised_slot_shape_is_still_refused():
    with pytest.raises(ValueError, match=r"\[num_tokens, 2\]"):
        scatter_rows(torch.zeros(2, 4, HEAD_DIM), torch.ones(2, HEAD_DIM),
                     torch.zeros((2, 3), dtype=torch.int64), 4)


def test_which_form_ran_is_logged_audibly(caplog):
    """Both forms are legal and neither errors when read as the other, so the
    only way to know which one ran is that it says so -- *in the output*.

    The first version of this line used logging.getLogger(__name__), which
    emits nothing once vLLM has configured logging. It ran for two launches and
    left no trace, and "the instrument is silent" looks exactly like "the path
    was not taken". So this asserts on vllm.logger's own logger, which is what
    the other 125 files in the tree use.
    """
    import logging

    from vllm.logger import logger as vllm_logger

    from vllm_ascend.models.deepseek_v4_1 import qkv_prolog

    qkv_prolog._LOGGED_SLOT_FORMS.clear()
    # The handler is attached to vLLM's logger directly: it does not propagate
    # to root, so caplog's default capture sees nothing -- which is the same
    # silence this test exists to rule out.
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    vllm_logger.addHandler(handler)
    try:
        scatter_rows(torch.zeros(2, 4, HEAD_DIM), torch.ones(1, HEAD_DIM),
                     torch.tensor([[1, 2]], dtype=torch.int64), 4)
    finally:
        vllm_logger.removeHandler(handler)
    assert any("(block, offset)" in m for m in records), records


def test_the_path_evidence_logs_do_not_use_a_bare_module_logger():
    """A silent instrument proves nothing, and proves it quietly.

    vLLM configures logging; logging.getLogger(__name__) produces no output
    afterwards. 125 files in this tree import vllm.logger and 11 do not. The
    ones that matter here are the lines whose whole purpose is to show that a
    path executed.
    """
    import ast
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import qkv_prolog

    # Parsed, not grepped: this module's own comment explains why not to use a
    # bare module logger, and a text search matches that explanation. A check
    # that its own prose can fail is a check about spelling, not about calls.
    tree = ast.parse(inspect.getsource(qkv_prolog))
    bare = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "getLogger"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logging"
    ]
    assert not bare, (
        f"{len(bare)} bare logging.getLogger call(s); vLLM configures logging "
        "and these emit nothing afterwards")
    assert "from vllm.logger import logger" in inspect.getsource(qkv_prolog)


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


# ------------------------------- the chain points between attn.in and sa_in.q

def test_the_pre_rope_value_is_offered_to_the_caller():
    """🔴 Three operations sit between q's projection and sa_in.q, and any of
    them can be the wrong one.

        q.wq_b_out   after wq_b, before any norm
        q.pre_rope   after the weight-free per-head RMSNorm   <- this one
        sa_in.q      after the GPT-J rotation

    Without the middle value, "sa_in.q differs" names a three-operation
    segment and costs a bisection per flight. The tap is a caller-supplied
    callable because this module has no layer identity and the caller does.
    """
    seen = {}
    q = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    kv = torch.randn(3, 8, dtype=torch.bfloat16)
    cos_sin = torch.zeros(16, 4, dtype=torch.float32)
    cos_sin[:, :2] = 1.0

    out = qnorm_rope_kv_insert(
        q=q, kv=kv, positions=torch.arange(3),
        cos_sin_cache=cos_sin, swa_kv_cache=torch.zeros(2, 4, 8),
        slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int32),
        eps=1e-6, rope_head_dim=4,
        tap=lambda key, tensor: seen.__setitem__(key, tensor.clone()))

    assert set(seen) == {"rope.q.in", "rope.kv.out"}
    # The normed q before the rotation, and only the slice the rotation
    # touches: the leading dims pass through and are already in sa_in.q.
    normed = rms_norm_weightless(q, 1e-6)
    torch.testing.assert_close(seen["rope.q.in"], normed[..., -4:])
    assert seen["rope.q.in"].shape == (3, 2, 4)
    assert out.shape == (3, 2, 8)


def test_the_rotation_still_happens_on_the_tapped_value():
    """The tap must observe the value the next step consumes, not a copy that
    the next step then ignores -- an instrument that reads a dead branch is
    the failure this module has already had once."""
    q = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    cos_sin = torch.zeros(8, 4, dtype=torch.float32)
    cos_sin[:, :2] = 0.5
    cos_sin[:, 2:] = 0.25
    seen = {}
    out = qnorm_rope_kv_insert(
        q=q, kv=torch.randn(2, 8, dtype=torch.bfloat16),
        positions=torch.arange(2), cos_sin_cache=cos_sin,
        swa_kv_cache=torch.zeros(2, 4, 8),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
        eps=1e-6, rope_head_dim=4,
        tap=lambda key, tensor: seen.__setitem__(key, tensor.clone()))
    nope = 8 - 4
    # The tapped slice is the rotation's input; the rotation changed it.
    assert not torch.allclose(out[..., nope:].to(torch.float32),
                              seen["rope.q.in"])
    # And the leading dims the tap deliberately omits did pass through, which
    # is the reason omitting them loses nothing.
    normed = qkv_prolog_rms(q, 1e-6)
    torch.testing.assert_close(out[..., :nope].to(torch.float32),
                               normed[..., :nope], rtol=2e-2, atol=2e-2)


def test_no_tap_is_the_default_and_changes_nothing():
    q = torch.randn(2, 1, 8, dtype=torch.bfloat16)
    kv = torch.randn(2, 8, dtype=torch.bfloat16)
    cos_sin = torch.zeros(8, 4, dtype=torch.float32)
    cos_sin[:, :2] = 1.0
    args = dict(positions=torch.arange(2), cos_sin_cache=cos_sin,
                slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
                eps=1e-6, rope_head_dim=4)
    a = qnorm_rope_kv_insert(
        q=q, kv=kv, swa_kv_cache=torch.zeros(2, 4, 8), **args)
    b = qnorm_rope_kv_insert(
        q=q, kv=kv, swa_kv_cache=torch.zeros(2, 4, 8),
        tap=lambda key, tensor: None, **args)
    assert torch.equal(a, b)


def test_the_post_rope_kv_is_offered_and_is_what_reaches_the_cache():
    """🔴 The key the kv path did not have.

    sa_in.kv is this function's *argument*, which upstream's contract keeps
    unrotated ("kv is unchanged; attention reads kv solely via
    swa_kv_cache"), so over the rope slice it sits a stage below golden's key
    of the same name -- and a difference there has the exact signature of a
    missing rotation. Measured on flight-24 that it is a stage mismatch and
    not a missing rotation; this key is what makes the slice comparable
    again.
    """
    seen = {}
    kv = torch.randn(3, 8, dtype=torch.bfloat16)
    cache = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    cos_sin = torch.zeros(16, 4, dtype=torch.float32)
    cos_sin[:, :2] = 0.5
    cos_sin[:, 2:] = 0.25
    slots = torch.tensor([0, 1, 2], dtype=torch.int32)

    qnorm_rope_kv_insert(
        q=torch.randn(3, 1, 8, dtype=torch.bfloat16), kv=kv,
        positions=torch.arange(3), cos_sin_cache=cos_sin,
        swa_kv_cache=cache, slot_mapping=slots, eps=1e-6, rope_head_dim=4,
        tap=lambda key, tensor: seen.__setitem__(key, tensor.clone()))

    # It is the rotated kv, not the argument.
    assert not torch.allclose(seen["rope.kv.out"][:, -4:].to(torch.float32),
                              kv[:, -4:].to(torch.float32))
    # And it is what landed in the cache -- the whole point of the key.
    for t in range(3):
        torch.testing.assert_close(
            cache[0, t].to(torch.float32),
            seen["rope.kv.out"][t].to(torch.float32), rtol=1e-2, atol=1e-2)
