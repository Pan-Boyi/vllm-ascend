# SPDX-License-Identifier: Apache-2.0
"""Tests for the DeepSeek-V4.1 golden-alignment plumbing.

These run against golden's real artifacts when they are present, so they also
serve as a contract check against the CPU-golden line. They are CPU-only and
need no NPU.
"""

import json
import os

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import debug_align as da

GOLDEN_DIR = "/workspace/code/dsv4_single_card/vllm_project/golden_dumps/full40"
PLAN_DIR = "/workspace/code/dsv4_single_card/vllm_project/golden_dumps/plan_full40"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(GOLDEN_DIR) or not os.path.isdir(PLAN_DIR),
    reason="golden dumps not available")

SEQ_LEN = 27
TOPK = 6


@pytest.fixture
def injector(monkeypatch):
    monkeypatch.setenv("VLLM_DSV41_GOLDEN_DIR", GOLDEN_DIR)
    monkeypatch.setenv("VLLM_DSV41_ROUTING_PLAN", PLAN_DIR)
    return da.RouteInjector("moe+layer_in", GOLDEN_DIR, PLAN_DIR)


def _load(path):
    from safetensors.torch import load_file
    return load_file(path)


def _golden_layer(layer_id):
    return _load(os.path.join(GOLDEN_DIR, f"layer_{layer_id:02d}.safetensors"))


def _shuffled_rows_with_padding(seed=0):
    """A batch shaped like the scheduler really hands it over.

    Rows are permuted (expert dispatch reorders tokens) and two padding rows
    are spliced in. Position addressing must survive both.
    """
    torch.manual_seed(seed)
    perm = torch.randperm(SEQ_LEN)
    return torch.cat([perm[:13], torch.tensor([-1, -1]), perm[13:]])


def test_mode_parsing():
    assert da._parse_mode("off") == set()
    assert da._parse_mode("moe") == {"moe"}
    assert da._parse_mode("moe+layer_in") == {"moe", "layer_in"}
    assert da._parse_mode("moe+indexer") == {"moe", "indexer"}
    with pytest.raises(ValueError, match="unknown flag"):
        da._parse_mode("moe+typo")
    # Refining flags without 'moe' would silently leave routing free, which is
    # the one thing the whole facility exists to prevent.
    with pytest.raises(ValueError, match="'moe' is required"):
        da._parse_mode("layer_in")


def test_routing_plan_agrees_with_full_dump():
    """The 58 KB plan and the 271 MB dump must carry identical routing."""
    plan = _load(os.path.join(PLAN_DIR, "routing_plan.safetensors"))
    for layer_id in (0, 2, 20, 39):
        tag = f"layer_{layer_id:02d}"
        gold = _golden_layer(layer_id)
        assert torch.equal(plan[f"{tag}.indices"].to(torch.int64),
                           gold["moe_gate.out.1"].to(torch.int64))
        assert torch.equal(plan[f"{tag}.weights"].to(torch.float32),
                           gold["moe_gate.out.0"].to(torch.float32))


def test_accum_dtype_is_reported(injector):
    # Plans recorded with different accumulator precision are not
    # interchangeable: routing itself differs on ~22% of tokens by layer 2.
    assert injector.accum_dtype() == "torch.float32"


def test_input_ids_guard(injector):
    prologue = _load(os.path.join(GOLDEN_DIR, "prologue.safetensors"))
    injector.check_input_ids(prologue["input_ids"])
    with pytest.raises(AssertionError, match="do not match"):
        injector.check_input_ids(torch.arange(SEQ_LEN))


def test_inject_moe_is_position_addressed(injector):
    row_to_pos = _shuffled_rows_with_padding()
    num_rows = row_to_pos.numel()
    own_ids = torch.full((num_rows, TOPK), 999, dtype=torch.int64)
    own_weights = torch.full((num_rows, TOPK), 7.0)

    ids, weights = injector.inject_moe(2, row_to_pos, own_ids, own_weights)

    gold = _golden_layer(2)["moe_gate.out.1"].to(torch.int64)
    valid = row_to_pos >= 0
    assert torch.equal(ids[valid], gold[row_to_pos[valid]])
    # Padding rows select nothing: ids untouched, weight zeroed so that
    # dispatch stays deterministic.
    assert torch.equal(ids[~valid], own_ids[~valid])
    assert torch.all(weights[~valid] == 0)


def test_row_indexed_addressing_would_differ(injector):
    """Guards the guard: a row-indexed bug must not pass the test above.

    Golden is batch-1 single-chunk prefill, where row index happens to equal
    sequence position. If this assertion ever fails, the fixture stopped
    exercising reordering and the position test became vacuous.
    """
    row_to_pos = _shuffled_rows_with_padding()
    num_rows = row_to_pos.numel()
    gold = _golden_layer(2)["moe_gate.out.1"].to(torch.int64)

    ids, _ = injector.inject_moe(
        2, row_to_pos, torch.zeros((num_rows, TOPK), dtype=torch.int64),
        torch.zeros((num_rows, TOPK)))
    naive = gold[torch.arange(num_rows).clamp(max=SEQ_LEN - 1)]

    valid = row_to_pos >= 0
    assert not torch.equal(ids[valid], naive[valid])


def test_inject_layer_in(injector):
    gold = _golden_layer(2)
    g_in = gold["layer.in"].squeeze(0)
    g_pre = gold["pre_mix.in"].squeeze(0)
    row_to_pos = _shuffled_rows_with_padding()
    num_rows = row_to_pos.numel()

    layer_in, pre_mix = injector.inject_layer_in(
        2, row_to_pos, torch.zeros((num_rows, ) + g_in.shape[1:],
                                   dtype=g_in.dtype),
        torch.zeros((num_rows, ) + g_pre.shape[1:], dtype=g_pre.dtype))

    valid = row_to_pos >= 0
    assert torch.equal(layer_in[valid], g_in[row_to_pos[valid]])
    assert torch.equal(pre_mix[valid], g_pre[row_to_pos[valid]])


def test_recorder_reassembles_out_of_order_chunks(tmp_path):
    """Chunked prefill hands rows over in pieces, not in sequence order."""
    gold = _golden_layer(2)
    ref = gold["attn.out"].squeeze(0)
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)

    for chunk in (torch.arange(5, SEQ_LEN), torch.arange(0, 5)):
        rec.record_layer(2, "attn.out", ref[chunk], chunk)
        rec.record_layer(2, "moe_gate.out.1", gold["moe_gate.out.1"][chunk],
                         chunk)
    rec.flush()

    out = _load(os.path.join(str(tmp_path), "layer_02.safetensors"))
    assert torch.equal(out["attn.out"], gold["attn.out"])
    # moe_gate.* carries no batch axis in the protocol; everything else does.
    assert out["moe_gate.out.1"].shape == gold["moe_gate.out.1"].shape


def test_coverage_assertion_catches_bad_mapping(tmp_path):
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.record_layer(0, "x", torch.zeros(5, 4), torch.tensor([0, 1, 2, 3, 3]))
    with pytest.raises(AssertionError, match="duplicated"):
        rec.flush()


def test_manifest_records_injection_state(tmp_path, injector):
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.add_manifest(force_route=injector.mode,
                     routing_plan=injector.plan_path,
                     routing_plan_accum_dtype=injector.accum_dtype())
    rec.flush()

    manifest = json.load(open(os.path.join(str(tmp_path), "manifest.json")))
    # Without this, a typo'd env var is indistinguishable from a real operator
    # bug: both show up as strict tolerance failing on every layer.
    assert manifest["force_route"] == "moe+layer_in"
    assert manifest["routing_plan_accum_dtype"] == "torch.float32"


# ------------------------------------- the dump says whether it is complete

def _golden(tmp_path, tensors):
    d = tmp_path / "golden"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"tensors": tensors}))
    return str(d)


def test_flush_records_what_it_wrote_in_its_own_manifest(tmp_path):
    """Symmetric with golden's, so our dump can be read by the same code that
    reads golden's -- including as the next run's expected set."""
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.record_whole("layer_00", "attn.sink", torch.zeros(4))
    rec.flush()

    manifest = json.load(open(os.path.join(str(tmp_path), "manifest.json")))
    assert manifest["tensors"] == {"layer_00": ["attn.sink"]}


def test_an_incomplete_dump_says_so_and_keeps_its_bytes(tmp_path, monkeypatch):
    """🔴 The report runs after the tensors are on disk and never raises.

    A dump run costs a launch; throwing it away at the last step because the
    report is bad would destroy the only copy of the thing the report is
    about.
    """
    golden = _golden(tmp_path, {"layer_00": ["attn.sink", "attn.out"]})
    monkeypatch.setenv("VLLM_ASCEND_DSV41_GOLDEN_DIR", golden)

    out = tmp_path / "ours"
    rec = da.AlignRecorder(str(out), seq_len=SEQ_LEN)
    rec.record_whole("layer_00", "attn.sink", torch.zeros(4))
    rec.flush()

    assert os.path.exists(os.path.join(str(out), "layer_00.safetensors")), (
        "the tensors were discarded because the coverage report was bad, "
        "which destroys the only copy of what the report is about")
    coverage = json.load(
        open(os.path.join(str(out), "manifest.json")))["golden_coverage"]
    assert coverage["state"] == "incomplete"
    assert coverage["missing"] == ["layer_00/attn.out"]


def test_a_complete_dump_says_ok(tmp_path, monkeypatch):
    golden = _golden(tmp_path, {"layer_00": ["attn.sink"]})
    monkeypatch.setenv("VLLM_ASCEND_DSV41_GOLDEN_DIR", golden)

    out = tmp_path / "ours"
    rec = da.AlignRecorder(str(out), seq_len=SEQ_LEN)
    rec.record_whole("layer_00", "attn.sink", torch.zeros(4))
    rec.flush()

    coverage = json.load(
        open(os.path.join(str(out), "manifest.json")))["golden_coverage"]
    assert coverage["state"] == "ok"


def test_without_a_golden_dir_the_manifest_says_not_checked(tmp_path,
                                                            monkeypatch):
    """⚠️ Not two empty differences. The reader of this manifest has to be
    able to tell "recorded everything" from "nobody looked"."""
    monkeypatch.delenv("VLLM_ASCEND_DSV41_GOLDEN_DIR", raising=False)

    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.record_whole("layer_00", "attn.sink", torch.zeros(4))
    rec.flush()

    coverage = json.load(
        open(os.path.join(str(tmp_path), "manifest.json")))["golden_coverage"]
    assert coverage["state"] == "not_checked"
    assert "missing" not in coverage


# ------------------------------ the injection scope, written per layer

def test_the_injection_scope_is_per_layer_not_one_sentence(tmp_path):
    """🔴 Layer 0 is the exception the whole structure exists for.

    It receives a 2-D embedding and builds the 4-stream tensor itself, so it
    has no layer.in and no pre_mix at all. Any single sentence about "the
    injected keys" is already false on layer 0 -- which is also why the side
    that produced the tensors writes it: a reader composing it has to guess,
    and its guess is wrong on exactly the layer that differs.
    """
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.record_whole("prologue", "embed.out", torch.zeros(4))
    rec.record_whole("layer_00", "attn.out", torch.zeros(4))
    rec.record_whole("layer_01", "layer.in", torch.zeros(4))
    rec.record_whole("layer_01", "pre_mix.in", torch.zeros(4))
    rec.flush()

    scope = json.load(
        open(os.path.join(str(tmp_path), "manifest.json")))["injection_scope"]
    assert scope["per_layer"]["1"]["injectable_keys"] == ["layer.in",
                                                         "pre_mix.in"]
    assert "layer.in" in scope["per_layer"]["0"]["absent_here"]
    assert "pre_mix.in" in scope["per_layer"]["0"]["absent_here"]
    assert any("embed.out" in k
               for k in scope["per_layer"]["0"]["injectable_keys"])


def test_a_key_that_failed_to_record_is_not_claimed(tmp_path):
    """Derived from what was recorded, never from what was meant to be."""
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.record_whole("layer_02", "layer.in", torch.zeros(4))
    rec.flush()

    scope = json.load(
        open(os.path.join(str(tmp_path), "manifest.json")))["injection_scope"]
    assert scope["per_layer"]["2"]["injectable_keys"] == ["layer.in"]
    assert "pre_mix.in" in scope["per_layer"]["2"]["absent_here"]


def test_the_compressor_layers_are_reported_as_unknown_when_unknown(tmp_path):
    """⚠️ Three states. A frozen 2/8/14/20 would be right for today's config
    and silently wrong for the next one, and 'no compressor latents' is a very
    different claim from 'nobody wrote down which layers produce them'."""
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.record_whole("layer_01", "layer.in", torch.zeros(4))
    rec.flush()

    scope = json.load(
        open(os.path.join(str(tmp_path), "manifest.json")))["injection_scope"]
    text = " ".join(scope["per_layer"]["1"]["not_isolated"])
    assert "was not recorded" in text
    assert "NOT a claim that there are none" in text


def test_the_compressor_layers_are_named_when_supplied(tmp_path):
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.add_manifest(compressor_source_layers=[2, 8, 14, 20],
                     candidate_block_layers=[20])
    rec.record_whole("layer_20", "layer.in", torch.zeros(4))
    rec.flush()

    scope = json.load(
        open(os.path.join(str(tmp_path), "manifest.json")))["injection_scope"]
    text = " ".join(scope["per_layer"]["20"]["not_isolated"])
    assert "2, 8, 14, 20" in text
    assert "this layer's candidate blocks" in text


def test_the_scope_says_isolation_is_partial(tmp_path):
    """The load-bearing half: an error travelling through the KV cache is
    invisible to injection, and that layer reads 'input the same, output the
    same' -- which is what 'no defect here' looks like."""
    rec = da.AlignRecorder(str(tmp_path), seq_len=SEQ_LEN)
    rec.record_whole("layer_03", "layer.in", torch.zeros(4))
    rec.flush()

    scope = json.load(
        open(os.path.join(str(tmp_path), "manifest.json")))["injection_scope"]
    assert "PARTIAL isolation" in scope["note"]
    assert any("KV cache" in item
               for item in scope["per_layer"]["3"]["not_isolated"])


# ------------------------ two dumps in one process, for the strongest compare

def test_one_request_keeps_the_old_layout(tmp_path, monkeypatch):
    """Unchanged when only one dump is asked for, so every existing tool and
    every earlier dump keeps its shape."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_DIR", str(tmp_path))
    monkeypatch.delenv("VLLM_ASCEND_DSV41_DUMP_REQUESTS", raising=False)
    da.reset_for_test()

    rec = da.get_recorder()
    assert rec is not None and rec.dump_dir == str(tmp_path)


def test_two_requests_go_to_numbered_directories(tmp_path, monkeypatch):
    """🔴 The comparison this exists for: same server, same weights, same
    prompt, one request before a different one has run and one after. A
    correct forward makes those byte-identical."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_REQUESTS", "2")
    da.reset_for_test()

    first = da.get_recorder()
    assert first.dump_dir.endswith("req_000")
    first.set_seq_len(SEQ_LEN)
    first.record_whole("layer_00", "attn.out", torch.zeros(4))

    assert da.rotate_recorder() is True
    second = da.get_recorder()
    assert second is not first and second.dump_dir.endswith("req_001")
    second.set_seq_len(SEQ_LEN)
    second.record_whole("layer_00", "attn.out", torch.ones(4))

    assert da.rotate_recorder() is False
    assert os.path.exists(os.path.join(first.dump_dir, "layer_00.safetensors"))
    assert os.path.exists(os.path.join(second.dump_dir,
                                       "layer_00.safetensors"))


def test_a_later_request_does_not_overwrite_the_dumps(tmp_path, monkeypatch):
    """⚠️ Recording stops once the requested dumps are written. A third
    request quietly replacing the second would destroy half the comparison
    and leave two well-formed directories behind."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_REQUESTS", "1")
    da.reset_for_test()

    rec = da.get_recorder()
    rec.set_seq_len(SEQ_LEN)
    rec.record_whole("layer_00", "attn.out", torch.zeros(4))
    assert da.rotate_recorder() is False
    assert da.get_recorder() is None


def test_a_typo_in_the_count_is_refused(tmp_path, monkeypatch):
    """A silent fallback to 1 makes the two-dump comparison read as 'the
    second request was never served'."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_REQUESTS", "two")
    da.reset_for_test()

    with pytest.raises(ValueError, match="is not a number"):
        da.get_recorder()


def test_zero_dumps_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_DIR", str(tmp_path))
    monkeypatch.setenv("VLLM_ASCEND_DSV41_DUMP_REQUESTS", "0")
    da.reset_for_test()

    with pytest.raises(ValueError, match="would record nothing"):
        da.get_recorder()


def test_rotation_without_a_recorder_is_not_an_error(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_DSV41_DUMP_DIR", raising=False)
    da.reset_for_test()
    assert da.rotate_recorder() is False
