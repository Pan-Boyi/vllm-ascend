# SPDX-License-Identifier: Apache-2.0
"""_o_proj's composition, against the fp32 reference.

The pieces were verified separately -- the inverse rotation against the CPU
golden dumps, the block-diagonal projection against the checkpoint. What was
not verified is the wiring between them: the reshape into groups, the three
permutations handed to npu_transpose_batchmatmul, and the flatten before wo_b.
That is where an error would sit, and none of the piecewise tests would move.

The operators themselves cannot run here, so they are replaced by faithful
torch equivalents -- faithful in the sense that they interpret perm_x1 /
perm_x2 / perm_y exactly as the operator documents, rather than being written
to match what _o_proj happens to pass. A fake that ignored the permutations
would make any permutation pass, so the fake asserts they are the ones the
operator needs and the test below checks a wrong permutation is caught.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import attention as attn_mod
from vllm_ascend.models.deepseek_v4_1.attention import AscendDeepseekV41Attention
from vllm_ascend.models.deepseek_v4_1.o_proj import group_wo_a, o_projection
from vllm_ascend.models.deepseek_v4_1.rope import (NOPE_HEAD_DIM, ROPE_HEAD_DIM,
                                                   build_freqs_cis,
                                                   inverse_rope_from_cos_sin_cache)

HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM
O_GROUPS, HEADS_PER_GROUP, O_LORA_RANK, HIDDEN = 2, 2, 8, 6
NUM_HEADS = O_GROUPS * HEADS_PER_GROUP
GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
NUM_TOKENS = 5


def _fake_transpose_batchmatmul(x, w, bias=None, scale=None, perm_x1=None,
                                perm_x2=None, perm_y=None, batch_split_factor=1):
    """npu_transpose_batchmatmul, as documented: permute, bmm, permute back."""
    assert bias is None and scale is None
    y = torch.bmm(x.permute(*perm_x1).float(), w.permute(*perm_x2).float())
    return y.permute(*perm_y)


class _Linear(torch.nn.Module):
    """Stands in for wo_b: a plain [out, in] linear with no bias."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = torch.randn(out_features, in_features, dtype=torch.float32)

    def forward(self, x):
        return x.float() @ self.weight.t()


@pytest.fixture
def layer(monkeypatch):
    torch.manual_seed(7)
    monkeypatch.setattr(attn_mod, "oproj_tp_enable", lambda: False)
    monkeypatch.setattr(attn_mod.torch_npu, "npu_transpose_batchmatmul",
                        _fake_transpose_batchmatmul, raising=False)

    # nn.Module.__setattr__ needs the module dicts, and __init__ is exactly
    # what is being skipped here; initialise the base only.
    self = object.__new__(AscendDeepseekV41Attention)
    torch.nn.Module.__init__(self)
    self.rope_head_dim = ROPE_HEAD_DIM
    self.n_local_groups = O_GROUPS
    self.o_lora_rank = O_LORA_RANK
    self.prefix = "model.layers.7.self_attn"

    freqs = build_freqs_cis(ROPE_HEAD_DIM, 64, 160000.0, True)
    cache = torch.cat([freqs.real, freqs.imag], dim=-1).contiguous()
    self.rotary_emb = SimpleNamespace(cos_sin_cache=cache)

    # [o_groups, group_in, o_lora_rank] -- the Ascend load-time layout. A plain
    # namespace, not an nn.Module: _o_proj reads .weight and .weight_scale, and
    # the absence of weight_scale is what selects the bf16 branch.
    self.wo_a = SimpleNamespace(weight=torch.randn(O_GROUPS, GROUP_IN, O_LORA_RANK))
    self.wo_b = _Linear(HIDDEN, O_GROUPS * O_LORA_RANK)
    return self


def _reference(layer, o, positions):
    """Inverse rotation, then the fp32 block-diagonal reference."""
    rotated = inverse_rope_from_cos_sin_cache(
        o, positions, layer.rotary_emb.cos_sin_cache, ROPE_HEAD_DIM)
    # o_projection wants [o_groups, o_lora_rank, group_in]; the layer holds the
    # transpose of that.
    wo_a_ref = layer.wo_a.weight.transpose(1, 2).contiguous()
    assert wo_a_ref.shape == (O_GROUPS, O_LORA_RANK, GROUP_IN)
    return o_projection(rotated, wo_a_ref, layer.wo_b.weight)


def test_o_proj_matches_the_fp32_reference(layer):
    o = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    positions = torch.tensor([0, 3, 17, 2, 9])

    got = layer._o_proj(o, positions)
    expected = _reference(layer, o, positions)

    assert got.shape == (NUM_TOKENS, HIDDEN)
    torch.testing.assert_close(got.float(), expected.float(), rtol=1e-4, atol=1e-4)


def test_group_wo_a_round_trips_into_the_layer_layout(layer):
    """The two conventions meet here, so tie them together explicitly."""
    flat = torch.randn(O_GROUPS * O_LORA_RANK, GROUP_IN)
    grouped = group_wo_a(flat, O_GROUPS, O_LORA_RANK)
    layer.wo_a.weight = grouped.transpose(1, 2).contiguous()

    o = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    positions = torch.arange(NUM_TOKENS)
    torch.testing.assert_close(layer._o_proj(o, positions).float(),
                               _reference(layer, o, positions).float(),
                               rtol=1e-4, atol=1e-4)


def test_each_group_sees_only_its_own_heads(layer):
    """The property the block-diagonal layout exists for.

    Perturbing a head in group 0 must not move group 1's contribution. A dense
    projection over all 32768 dims would fail this; so would a reshape that
    interleaved heads across groups instead of splitting them into contiguous
    blocks.
    """
    o = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    positions = torch.arange(NUM_TOKENS)

    # Zero wo_b's contribution from group 1's slice, so the output depends only
    # on group 0, then perturb a head belonging to group 1.
    layer.wo_b.weight[:, O_LORA_RANK:] = 0.0
    base = layer._o_proj(o, positions)

    perturbed = o.clone()
    perturbed[:, HEADS_PER_GROUP:, :] += 10.0   # heads owned by group 1
    after = layer._o_proj(perturbed, positions)
    torch.testing.assert_close(base.float(), after.float(), rtol=1e-5, atol=1e-5)

    # and a head in group 0 must move it, or the test above proves nothing.
    moved = o.clone()
    moved[:, :HEADS_PER_GROUP, :] += 10.0
    assert not torch.allclose(base.float(), layer._o_proj(moved, positions).float(),
                              atol=1e-3)


def test_a_wrong_permutation_would_be_caught(layer):
    """Guards the guard: shows the fake is not permutation-blind."""
    o = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    positions = torch.arange(NUM_TOKENS)
    expected = _reference(layer, o, positions)

    monkey = attn_mod.PERM_Y
    try:
        attn_mod.PERM_Y = (0, 1, 2)   # leave the output as [G, T, R]
        wrong = layer._o_proj(o, positions)
        assert not torch.allclose(wrong.float(), expected.float(), atol=1e-3)
    finally:
        attn_mod.PERM_Y = monkey


def test_the_oproj_tp_path_is_still_refused(layer, monkeypatch):
    monkeypatch.setattr(attn_mod, "oproj_tp_enable", lambda: True)
    with pytest.raises(NotImplementedError, match="oproj tensor parallelism"):
        layer._o_proj(torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM),
                      torch.arange(NUM_TOKENS))
