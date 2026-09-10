# SPDX-License-Identifier: Apache-2.0
"""Inverse RoPE reproduces the CPU golden dumps bit-exactly.

Golden records the attention output both before and after the inverse rotation,
so this checks the whole step against real data rather than against a reading of
the reference implementation.
"""

import json
import os

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.rope import (NOPE_HEAD_DIM,
                                                   ROPE_HEAD_DIM,
                                                   build_freqs_cis,
                                                   inverse_rope_,
                                                   rope_params_for_layer)

GOLDEN_DIR = ("/workspace/code/dsv4_single_card/vllm_project/golden_dumps/"
              "smoke3_oproj")
CONFIG = "/workspace/models/DeepSeek-V4.1-Flash/config.json"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(GOLDEN_DIR) or not os.path.isfile(CONFIG),
    reason="golden dumps or checkpoint config not available")


class _Config:
    """The fields rope_params_for_layer reads, straight from config.json."""

    def __init__(self):
        text = json.load(open(CONFIG))["text_config"]
        self.compress_ratios = text["compress_ratios"]
        self.rope_theta = text["rope_theta"]
        self.compress_rope_theta = text["compress_rope_theta"]


@pytest.fixture(scope="module")
def config():
    return _Config()


def _golden(layer_id):
    from safetensors.torch import load_file
    path = os.path.join(GOLDEN_DIR, f"layer_{layer_id:02d}.safetensors")
    if not os.path.isfile(path):
        pytest.skip(f"no dump for layer {layer_id}")
    return load_file(path)


def _apply(o_raw, base, use_yarn):
    out = o_raw.clone()
    freqs = build_freqs_cis(ROPE_HEAD_DIM, out.shape[1], base, use_yarn)
    inverse_rope_(out[..., -ROPE_HEAD_DIM:], freqs)
    return out


@pytest.mark.parametrize("layer_id", [0, 1, 2])
def test_reproduces_golden_bit_exactly(config, layer_id):
    golden = _golden(layer_id)
    base, use_yarn = rope_params_for_layer(config, layer_id)
    got = _apply(golden["attn.o_raw"], base, use_yarn)
    assert torch.equal(got, golden["attn.o_unroped"])


@pytest.mark.parametrize("layer_id", [0, 1, 2])
def test_wrong_parameters_are_visibly_wrong(config, layer_id):
    """The two parameter sets are not interchangeable.

    Without this, a single global rope table would look like a reasonable
    simplification: it produces finite, plausible numbers everywhere.
    """
    golden = _golden(layer_id)
    right_base, right_yarn = rope_params_for_layer(config, layer_id)
    wrong_base, wrong_yarn = ((config.compress_rope_theta, True)
                              if not right_yarn else (config.rope_theta, False))

    got = _apply(golden["attn.o_raw"], wrong_base, wrong_yarn)
    ref = golden["attn.o_unroped"].float()
    rel = ((got.float() - ref).norm() / ref.norm()).item()
    assert rel > 0.1, f"wrong parameters only shifted things by {rel:.2e}"


def test_layer_parameters_follow_compress_ratios(config):
    # Layers 0 and 1 have ratio 0; 2..39 have ratio 2 or 1; 40..42 are DSpark.
    assert rope_params_for_layer(config, 0) == (float(config.rope_theta), False)
    assert rope_params_for_layer(config, 1) == (float(config.rope_theta), False)
    for layer_id in (2, 19, 20, 39):
        base, use_yarn = rope_params_for_layer(config, layer_id)
        assert (base, use_yarn) == (float(config.compress_rope_theta), True)


@pytest.mark.parametrize("layer_id", [0, 1, 2])
def test_only_the_trailing_dims_are_touched(config, layer_id):
    """A free self-check: the nope half must pass through untouched."""
    golden = _golden(layer_id)
    assert torch.equal(golden["attn.o_raw"][..., :NOPE_HEAD_DIM],
                       golden["attn.o_unroped"][..., :NOPE_HEAD_DIM])
    base, use_yarn = rope_params_for_layer(config, layer_id)
    got = _apply(golden["attn.o_raw"], base, use_yarn)
    assert torch.equal(got[..., :NOPE_HEAD_DIM],
                       golden["attn.o_raw"][..., :NOPE_HEAD_DIM])


def test_pairing_is_interleaved_not_split_half():
    """Rotation preserves each pair's norm, which identifies the convention.

    This needs no angles and no reference implementation: only the correct
    pairing keeps sqrt(a^2 + b^2) invariant between the two dumps.
    """
    golden = _golden(2)
    raw = golden["attn.o_raw"].squeeze(0).float()[..., -ROPE_HEAD_DIM:]
    unroped = golden["attn.o_unroped"].squeeze(0).float()[..., -ROPE_HEAD_DIM:]

    def pair_norms(x, interleaved):
        a, b = ((x[..., 0::2], x[..., 1::2]) if interleaved else
                (x[..., :ROPE_HEAD_DIM // 2], x[..., ROPE_HEAD_DIM // 2:]))
        return (a * a + b * b).sqrt()

    def max_rel(interleaved):
        n_raw, n_unroped = pair_norms(raw, interleaved), pair_norms(unroped, interleaved)
        return ((n_raw - n_unroped).abs()
                / n_unroped.abs().clamp_min(1e-6)).max().item()

    assert max_rel(interleaved=True) < 1e-2
    assert max_rel(interleaved=False) > 1.0


def test_yarn_has_no_mscale(config):
    """The reference does not rescale by YaRN's mscale, so neither does this.

    Adding one would leave every rotation angle correct while scaling the whole
    tensor -- the kind of change that survives a shape check and a finite check.
    """
    freqs = build_freqs_cis(ROPE_HEAD_DIM, 8, config.compress_rope_theta, True)
    assert torch.allclose(freqs.abs(), torch.ones_like(freqs.abs()))
