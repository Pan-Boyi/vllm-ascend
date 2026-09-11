# SPDX-License-Identifier: Apache-2.0
"""The block-diagonal output projection, checked against the CPU golden dumps.

Golden records the attention output after the inverse RoPE (``attn.o_unroped``)
and after the projection (``attn.out``), so the projection can be checked on its
own with the checkpoint's real weights.
"""

import glob
import json
import os
import struct

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.o_proj import (block_diagonal_wo_a,
                                                     dequantize_block_scaled,
                                                     group_wo_a, o_projection)

MODEL_DIR = "/workspace/models/DeepSeek-V4.1-Flash"
GOLDEN_DIR = ("/workspace/code/dsv4_single_card/vllm_project/golden_dumps/"
              "smoke3_oproj")
LAYER = 2

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR) or not os.path.isdir(GOLDEN_DIR),
    reason="checkpoint or golden dumps not available")


def _load_tensors(names):
    from safetensors import safe_open
    wanted, found = set(names), {}
    for shard in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            keys = set(json.loads(f.read(header_len))) & wanted
        if keys:
            with safe_open(shard, framework="pt") as sf:
                for k in keys:
                    found[k] = sf.get_tensor(k)
        if len(found) == len(wanted):
            break
    assert len(found) == len(wanted), sorted(wanted - set(found))
    return found


@pytest.fixture(scope="module")
def weights():
    p = f"layers.{LAYER}.attn"
    t = _load_tensors([f"{p}.wo_a.weight", f"{p}.wo_a.scale",
                       f"{p}.wo_b.weight", f"{p}.wo_b.scale"])
    return {
        "wo_a": dequantize_block_scaled(t[f"{p}.wo_a.weight"], t[f"{p}.wo_a.scale"]),
        "wo_b": dequantize_block_scaled(t[f"{p}.wo_b.weight"], t[f"{p}.wo_b.scale"]),
    }


@pytest.fixture(scope="module")
def golden():
    from safetensors.torch import load_file
    return load_file(os.path.join(GOLDEN_DIR, f"layer_{LAYER:02d}.safetensors"))


def _rel_l2(got, ref):
    got, ref = got.float().flatten(), ref.float().flatten()
    return ((got - ref).norm() / ref.norm()).item()


def test_shapes_match_the_config(weights):
    # o_groups = 8, o_lora_rank = 1024, 8 * 4096 = 32768 = 64 heads * 512.
    assert weights["wo_a"].shape == (8192, 4096)
    assert weights["wo_b"].shape == (5120, 8192)
    grouped = group_wo_a(weights["wo_a"], o_groups=8, o_lora_rank=1024)
    assert grouped.shape == (8, 1024, 4096)


def test_reproduces_golden_attn_out(weights, golden):
    """A bf16 reference path, so the bar is bf16's floor, not zero."""
    grouped = group_wo_a(weights["wo_a"], 8, 1024)
    got = o_projection(golden["attn.o_unroped"].squeeze(0), grouped, weights["wo_b"])
    assert _rel_l2(got, golden["attn.out"].squeeze(0)) < 3e-2


def test_a_wrong_group_split_is_measurably_different(weights, golden):
    """Guards the guard: the group split has to matter.

    A dense matmul against the flat wo_a is not the hazard -- the shapes do not
    line up and it fails loudly. The silent mistake is feeding each group the
    wrong slice of the 32768 inputs, which keeps every shape valid. If this
    assertion ever fails, the test above has stopped discriminating.
    """
    o = golden["attn.o_unroped"].squeeze(0).float()
    grouped = group_wo_a(weights["wo_a"], 8, 1024)

    right = block_diagonal_wo_a(o, grouped)
    # same shapes throughout, only the group-to-slice assignment is permuted
    wrong = block_diagonal_wo_a(o, grouped.flip(0))

    assert right.shape == wrong.shape
    assert _rel_l2(wrong, right) > 0.5


def test_dequantisation_multiplier_is_a_power_of_two(weights):
    """ue8m0 stores a bare exponent, so dequantisation introduces no rounding.

    That is what lets the checkpoint's fp8 weights be compared bit-exactly
    elsewhere; a scale format with a mantissa would not.
    """
    from vllm_ascend.models.deepseek_v4_1.o_proj import E8M0_BIAS

    weight = torch.ones((32, 32 * 4), dtype=torch.float32)
    bytes_ = torch.tensor([[E8M0_BIAS - 1, E8M0_BIAS, E8M0_BIAS + 1, E8M0_BIAS]],
                          dtype=torch.uint8)
    # Both spellings must agree: the checkpoint stores float8_e8m0fnu, which
    # torch decodes straight to the multiplier, while a raw uint8 still needs
    # the bias removed. Testing only the uint8 form hid that difference.
    for scale in (bytes_, bytes_.view(torch.float8_e8m0fnu)):
        out = dequantize_block_scaled(weight, scale)
        assert out[0, 0] == 0.5 and out[0, 32] == 1.0 and out[0, 64] == 2.0


def test_group_wo_a_rejects_a_mismatched_shape(weights):
    with pytest.raises(ValueError, match="o_groups \\* o_lora_rank"):
        group_wo_a(weights["wo_a"], o_groups=8, o_lora_rank=999)


def test_dequantize_rejects_a_scale_that_does_not_tile():
    with pytest.raises(ValueError, match="does not tile"):
        dequantize_block_scaled(torch.ones(64, 64), torch.ones(3, 2))
