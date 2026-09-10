# SPDX-License-Identifier: Apache-2.0
"""Upstream's mHC reference reproduces the CPU golden dump on real weights.

DeepSeek-V4.1's hyper-connections keep four residual copies and mix them with
coefficients computed from the stream itself, including a 20-iteration Sinkhorn
normalisation. The plan is to import that implementation from upstream rather
than reimplement it, so this pins the thing that actually matters: that the
imported code produces golden's numbers here, on this hardware, with the
checkpoint's own parameters.

It also pins the call parameters, which are easy to get wrong and fail quietly:

* ``hc_post_mult_value`` is 2.0 -- a constant in upstream's decoder layer, not
  a config field. Passing 1.0 scales every post-mix by half.
* ``rms_eps`` is the model's ``rms_norm_eps`` (1e-20), not a kernel default.
* ``hc_pre_eps`` and ``hc_sinkhorn_eps`` are both ``config.hc_eps`` (1e-6).

Reads safetensors headers plus three small tensors; no NPU required.
"""

import glob
import json
import os
import struct

import pytest
import torch

MODEL_DIR = "/workspace/models/DeepSeek-V4.1-Flash"
GOLDEN_DIR = "/workspace/code/dsv4_single_card/vllm_project/golden_dumps/full40"

# Any layer past the first: layer 0 takes a separate branch that broadcasts the
# embedding into the four copies instead of receiving a residual stream.
LAYER = 5

# From config.json / upstream's DeepseekV4DecoderLayer.
RMS_NORM_EPS = 1e-20
HC_EPS = 1e-6
HC_POST_ALPHA = 2.0
HC_SINKHORN_ITERS = 20

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR) or not os.path.isdir(GOLDEN_DIR),
    reason="checkpoint or golden dumps not available")


def _load_hc_params(layer: int):
    from safetensors import safe_open

    wanted = {
        f"layers.{layer}.hc_attn_fn",
        f"layers.{layer}.hc_attn_scale",
        f"layers.{layer}.hc_attn_base",
    }
    found = {}
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


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
    got, ref = got.float().flatten(), ref.float().flatten()
    return ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()


@pytest.fixture(scope="module")
def mhc_result():
    from safetensors.torch import load_file
    from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch

    params = _load_hc_params(LAYER)
    golden = load_file(os.path.join(GOLDEN_DIR, f"layer_{LAYER:02d}.safetensors"))

    post, comb, layer_input, pre = mhc_pre_delayed_torch(
        residual=golden["layer.in"].squeeze(0).float(),
        fn=params[f"layers.{LAYER}.hc_attn_fn"].float(),
        hc_scale=params[f"layers.{LAYER}.hc_attn_scale"].float(),
        hc_base=params[f"layers.{LAYER}.hc_attn_base"].float(),
        rms_eps=RMS_NORM_EPS,
        hc_pre_eps=HC_EPS,
        hc_sinkhorn_eps=HC_EPS,
        hc_post_mult_value=HC_POST_ALPHA,
        sinkhorn_repeat=HC_SINKHORN_ITERS,
        pre_mix=golden["pre_mix.in"].squeeze(0).float(),
    )
    return golden, {"pre": pre, "post": post, "comb": comb,
                    "layer_input": layer_input}


def test_collapsed_layer_input_is_bit_exact(mhc_result):
    """hc_pre.0 is the four copies collapsed to one -- the attention input."""
    golden, got = mhc_result
    ref = golden["hc_pre.0"].squeeze(0)
    assert torch.equal(got["layer_input"].to(ref.dtype).reshape(ref.shape), ref)


@pytest.mark.parametrize("key,name", [
    ("hc_mixes.0.0", "pre"),
    ("hc_mixes.0.1", "post"),
    ("hc_mixes.0.2", "comb"),
])
def test_mix_coefficients_match_golden(mhc_result, key, name):
    """fp32 throughout, so the bar is round-off, not golden's 5e-3 tolerance."""
    golden, got = mhc_result
    ref = golden[key].squeeze(0)
    assert _rel_l2(got[name].reshape(ref.shape), ref) < 1e-5


def test_hc_mixes_0_0_is_the_produced_pre_not_the_incoming_one(mhc_result):
    """Which ``pre`` the protocol dumps is not obvious from its name.

    ``pre_mix.in`` (the coefficients handed down by the previous sublayer) and
    ``hc_mixes.0.0`` (the ones this sublayer produces) have identical shape, so
    confusing them costs nothing at dump time and produces a mismatch much
    later, in a different layer.
    """
    golden, got = mhc_result
    produced = golden["hc_mixes.0.0"].squeeze(0)
    incoming = golden["pre_mix.in"].squeeze(0)
    assert _rel_l2(got["pre"].reshape(produced.shape), produced) < 1e-5
    assert not torch.allclose(produced.float(), incoming.float())
