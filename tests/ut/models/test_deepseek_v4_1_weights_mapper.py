# SPDX-License-Identifier: Apache-2.0
"""Checkpoint name mapping for DeepSeek-V4.1-Flash.

The exhaustiveness tests read the real checkpoint's safetensors *headers* (a
few KB per shard) -- no weights are loaded and no NPU is needed.
"""

import glob
import json
import os
import struct
from collections import defaultdict

import pytest

from vllm_ascend.models.deepseek_v4_1.weights_mapper import (DSPARK, MAIN,
                                                             VISION,
                                                             UnmappedWeightError,
                                                             map_weight_name)

MODEL_DIR = "/workspace/models/DeepSeek-V4.1-Flash"


def _checkpoint_tensor_names():
    names = []
    for shard in sorted(glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))):
        with open(shard, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        names.extend(k for k in header if k != "__metadata__")
    return names


@pytest.fixture(scope="module")
def names():
    if not os.path.isdir(MODEL_DIR):
        pytest.skip("checkpoint not available")
    got = _checkpoint_tensor_names()
    assert got, "no tensors found in checkpoint"
    return got


# -- exhaustiveness --------------------------------------------------------


def test_every_checkpoint_tensor_maps(names):
    """No tensor may fall through.

    An unmapped tensor leaves a randomly initialised parameter behind: the
    model loads, runs, and is quietly wrong.
    """
    unmapped = []
    for n in names:
        try:
            map_weight_name(n)
        except UnmappedWeightError:
            unmapped.append(n)
    assert not unmapped, f"{len(unmapped)} unmapped, e.g. {unmapped[:5]}"


def test_names_are_unique_within_each_target(names):
    per_target = defaultdict(lambda: defaultdict(list))
    for n in names:
        mapped = map_weight_name(n)
        per_target[mapped.target][mapped.name].append(n)

    for target, by_name in per_target.items():
        collisions = {k: v for k, v in by_name.items() if len(v) > 1}
        assert not collisions, f"{target}: {list(collisions.items())[:3]}"


def test_targets_partition_the_checkpoint(names):
    counts = defaultdict(int)
    for n in names:
        counts[map_weight_name(n).target] += 1
    assert set(counts) == {MAIN, DSPARK, VISION}
    assert sum(counts.values()) == len(names)


def test_unknown_name_raises_rather_than_skipping():
    with pytest.raises(UnmappedWeightError, match="matches no DeepSeek-V4.1"):
        map_weight_name("some.new.tensor.introduced.upstream")


# -- the DSpark trap -------------------------------------------------------


def test_backbone_and_dspark_share_layer_names_but_not_targets(names):
    """mtp.0/1/2 map onto model.layers.0/1/2, same as backbone layers 0/1/2.

    That is correct -- they load into two separate models -- but it is the
    sharp edge behind DSpark's known bug: V4.0 derives the spec layer index
    from any name starting with ``mtp.`` and rewrites it to a hardcoded
    ``model.layers.0.``, collapsing all three blocks onto the first one. The
    overlap is therefore asserted here deliberately, so that ``target`` and
    ``spec_layer`` stay the only things separating them.
    """
    by_target = defaultdict(set)
    for n in names:
        mapped = map_weight_name(n)
        by_target[mapped.target].add(mapped.name)

    overlap = by_target[MAIN] & by_target[DSPARK]
    assert overlap, "expected backbone/DSpark name overlap; mapping changed"
    assert "model.layers.0.hc_attn_base" in overlap


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_each_dspark_block_keeps_its_own_index(idx):
    """mtp.1 and mtp.2 must not collapse onto block 0."""
    mapped = map_weight_name(f"mtp.{idx}.attn.wkv.weight")
    assert mapped.target == DSPARK
    assert mapped.spec_layer == idx
    assert mapped.name == f"model.layers.{idx}.self_attn.wkv.weight"


def test_dspark_blocks_are_a_pipeline_not_three_copies(names):
    """The three DSpark blocks are not interchangeable.

    mtp.0 is the entry: ``main_proj [5120, 15360]`` ingests the concatenated
    hidden states of backbone layers 37/38/39 (15360 = 3 x 5120), normalised by
    ``main_norm``. mtp.2 is the exit: ``markov_head`` emits draft tokens and
    ``confidence_head`` the acceptance signal. mtp.1 carries neither.

    This is why running only block 0 -- the known ``spec_step_idx`` bug -- is
    worse than wasted KV cache: the draft output path lives on block 2 and is
    never reached at all.
    """
    entry, exit_ = set(), set()
    for n in names:
        if ".main_proj." in n or ".main_norm." in n:
            entry.add(n.split(".")[1])
        if ".markov_head." in n or ".confidence_head." in n:
            exit_.add(n.split(".")[1])

    assert entry == {"0"}, f"main_proj/main_norm expected on mtp.0 only: {entry}"
    assert exit_ == {"2"}, f"markov/confidence expected on mtp.2 only: {exit_}"


def test_dspark_entry_and_exit_tensors_map(names):
    for n in names:
        if any(k in n for k in
               (".main_proj.", ".main_norm.", ".markov_head.",
                ".confidence_head.")):
            mapped = map_weight_name(n)
            assert mapped.target == DSPARK
            assert mapped.spec_layer == int(n.split(".")[1])


# -- the rewrites that are easy to get wrong -------------------------------


@pytest.mark.parametrize("ckpt,expected", [
    # attn_norm/ffn_norm must be consumed before the bare attn./ffn. rules.
    ("layers.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
    ("layers.3.ffn_norm.weight",
     "model.layers.3.post_attention_layernorm.weight"),
    # attn_sink sits inside attn., so it becomes self_attn.attn_sink.
    ("layers.3.attn.attn_sink", "model.layers.3.self_attn.attn_sink"),
    # q_norm stays -- only the indexer's q-norm is dropped in V4.1.
    ("layers.3.attn.q_norm.weight", "model.layers.3.self_attn.q_norm.weight"),
    # Nested compressor/indexer trees ride along under self_attn.
    ("layers.2.attn.compressor.wgate.weight",
     "model.layers.2.self_attn.compressor.wgate.weight"),
    ("layers.2.attn.indexer.k_norm.weight",
     "model.layers.2.self_attn.indexer.k_norm.weight"),
    # Expert matrices are renamed for the fused-MoE loader.
    ("layers.5.ffn.experts.17.w1.weight",
     "model.layers.5.mlp.experts.17.gate_proj.weight"),
    ("layers.5.ffn.experts.17.w2.weight",
     "model.layers.5.mlp.experts.17.down_proj.weight"),
    ("layers.5.ffn.experts.17.w3.weight",
     "model.layers.5.mlp.experts.17.up_proj.weight"),
    # Hyper-connection parameters belong to the layer, not a submodule.
    ("layers.7.hc_attn_fn", "model.layers.7.hc_attn_fn"),
    # engram keeps its checkpoint naming.
    ("layers.1.engram.wkv.weight", "model.layers.1.engram.wkv.weight"),
    # Top-level renames.
    ("embed.weight", "model.embed_tokens.weight"),
    ("head.weight", "lm_head.weight"),
    ("norm.weight", "model.norm.weight"),
])
def test_rewrites(ckpt, expected):
    assert map_weight_name(ckpt).name == expected


def test_scale_suffix_becomes_weight_scale():
    assert (map_weight_name("layers.0.attn.wkv.scale").name ==
            "model.layers.0.self_attn.wkv.weight_scale")


def test_hc_scale_is_not_mistaken_for_a_quant_scale():
    """``hc_attn_scale`` is a parameter, not a quantisation scale.

    The ``.scale`` rewrite is anchored at the end for exactly this reason; a
    substring match would rename it to ``hc_attn_weight_scale`` and it would
    then fail to load.
    """
    assert (map_weight_name("layers.0.hc_attn_scale").name ==
            "model.layers.0.hc_attn_scale")


def test_final_norm_is_not_confused_with_layer_norms():
    """Many tensors end in ``norm.weight``; only the bare one is the final norm."""
    assert map_weight_name("norm.weight").name == "model.norm.weight"
    assert (map_weight_name("layers.0.attn.kv_norm.weight").name ==
            "model.layers.0.self_attn.kv_norm.weight")
    assert (map_weight_name("layers.2.attn.compressor.norm.weight").name ==
            "model.layers.2.self_attn.compressor.norm.weight")


def test_vision_tower_is_kept_separate():
    assert map_weight_name("vision.blocks.0.attn.wo.weight").target == VISION
    assert map_weight_name("aligner.w1.weight").target == VISION
