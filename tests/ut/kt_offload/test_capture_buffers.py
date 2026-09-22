# SPDX-License-Identifier: Apache-2.0
"""Capture-buffer ownership per layer instance, and the side-stream gate.

The production bug this file was written for (2026-09-22): the "capture
buffers are ready" flag was process-wide (``layer._CAPTURE_BUFFERS_READY``)
while the int64 id buffers it guards are filled per *layer instance*
(``self._ids_i64``). In a real model every layer after the first one therefore
returned early with an empty dict, and the first captured forward on the side
stream raised ``no preallocated int64 id buffer``. On card that surfaced as
ACL 107025 and was written up as a platform limit (§4.19 ②/③).

Note what has zero coverage here and is on-card only: whether an ACL capture
follows the event wait onto the side stream, and 107025 itself. Those are in
tools/kt_offload/oncard/RUNBOOK_sidestream.md.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload import side_stream as ss
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan

H, TOPK, NEXP, LAYERS = 8, 2, 16, 4
SIZES = [1, 2]


class FakeKtBuffer:
    """kt's KExpertsCPUBuffer: class-level, one registration for the process."""

    capture_buffers: dict = {}
    registered: list = []

    @classmethod
    def get_buffer(cls, tensor, top_k):
        cls.capture_buffers[tensor.shape[0]] = object()

    @classmethod
    def reset(cls):
        cls.capture_buffers = {}
        cls.registered = []


class FakeWrapper:
    @classmethod
    def set_capture_batch_sizes(cls, sizes):
        FakeKtBuffer.registered.append(list(sizes))


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    FakeKtBuffer.reset()
    monkeypatch.setattr(kl, "_kt_buffer_class", lambda: FakeKtBuffer)
    monkeypatch.setattr(kl, "_CAPTURE_BUFFERS_READY", None)
    monkeypatch.setattr(kl, "_SIDE_INFLIGHT", {})
    yield


def _layer(layer_idx: int, *, overlap: bool = True):
    plan = build_plan([[0, 3]] * LAYERS, NEXP)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, overlap_side_stream=overlap)
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=H,
                moe_intermediate_size=4,
                num_experts_per_tok=TOPK,
                num_hidden_layers=LAYERS,
            ),
            model="/m",
            enforce_eager=False,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
            cudagraph_capture_sizes=list(SIZES),
        ),
    )
    rt = kl.KTLayerOffload(cfg, plan.layer(layer_idx), layer_idx, vcfg)
    rt.wrapper = FakeWrapper()
    return rt


# -- 1. the bug: every layer instance must get its own buffers --------------


def test_every_layer_instance_gets_its_own_id_buffers():
    """RED before the fix: only the first instance had buffers.

    The guard was ``if key == _CAPTURE_BUFFERS_READY: return`` with the key
    built from capture sizes / hidden / top_k / device -- identical for every
    layer of one model, so layer 1..N-1 returned before ``self._ids_i64`` was
    filled and their captured side-stream forward raised.
    """
    layers = [_layer(i) for i in range(LAYERS)]
    for rt in layers:
        rt._prepare_capture_buffers(torch.device("cpu"))
    for rt in layers:
        assert sorted(rt._ids_i64) == SIZES, f"layer {rt.layer_idx} has {sorted(rt._ids_i64)}"


def test_id_buffers_are_not_shared_between_layers():
    """A6: two layers pointing at one buffer is the mutant, and it must be caught."""
    layers = [_layer(i) for i in range(LAYERS)]
    for rt in layers:
        rt._prepare_capture_buffers(torch.device("cpu"))
    keys = [{id(t) for t in rt._ids_i64.values()} for rt in layers]
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not (keys[i] & keys[j]), f"layers {i}/{j} share an id buffer"
    ok = ss.slot_ownership_ok([(rt.layer_idx, keys[i], False) for i, rt in enumerate(layers)])
    assert ok["ok"], ok
    # mutant: point layer 2 at layer 0's buffers
    bad = ss.slot_ownership_ok([(0, keys[0], False), (2, keys[0], False)])
    assert not bad["ok"] and "share id buffer" in bad["detail"][0]


def test_kt_registration_stays_once_per_process():
    """The kt side really is process-wide; only our per-instance part repeats."""
    for i in range(LAYERS):
        _layer(i)._prepare_capture_buffers(torch.device("cpu"))
    assert FakeKtBuffer.registered == [SIZES], FakeKtBuffer.registered


def test_a_second_model_with_other_capture_sizes_is_still_refused():
    _layer(0)._prepare_capture_buffers(torch.device("cpu"))
    rt = _layer(1)
    rt.capture_sizes = [4]
    with pytest.raises(RuntimeError, match="capture buffers already prepared"):
        rt._prepare_capture_buffers(torch.device("cpu"))


def test_empty_id_buffers_are_reported_by_the_criterion():
    """The exact shape of the bug, as the gate sees it."""
    rep = ss.slot_ownership_ok([(0, {1}, False), (1, set(), False)])
    assert not rep["ok"] and "layer 1 has no preallocated int64 id buffer" in rep["detail"]


# -- 2. A6 in flight: two layers sharing kt's pinned slot -------------------


def test_slot_sharing_layers_in_flight_are_refused():
    assert ss.slot_ownership_ok([(0, {1}, True), (1, {2}, True)])["ok"]  # slots 0 and 1: fine
    bad = ss.slot_ownership_ok([(0, {1}, True), (2, {2}, True)])  # both slot 0
    assert not bad["ok"] and "pinned slot 0" in bad["detail"][0]


# -- 3. the other gate criteria, each with a red arm ------------------------


def test_capture_hostfunc_zero():
    assert ss.capture_hostfunc_ok({0: 3, 1: 3}, {0: 3, 1: 3})["ok"]
    bad = ss.capture_hostfunc_ok({0: 3}, {0: 4})
    assert not bad["ok"] and bad["detail"] == {0: 1}


def test_replay_one_per_layer():
    before = {i: 5 for i in range(LAYERS)}
    after = {i: 8 for i in range(LAYERS)}
    assert ss.replay_hostfunc_ok(before, after, 3, LAYERS)["ok"]
    after[2] = 5  # layer 2's work never made it into the graph
    bad = ss.replay_hostfunc_ok(before, after, 3, LAYERS)
    assert not bad["ok"] and bad["detail"]["wrong"] == {2: 0}


def test_join_record_and_wait_paired():
    side = [("wait", "fork"), ("d2h", None), ("hostfunc", None), ("record", "join")]
    compute = [("record", "fork"), ("wait", "join"), ("h2d", None)]
    assert ss.join_ops_ok(side, compute)["ok"]
    # the on-card 107025 arm (t5 off_evt): recorded, nobody joins
    bad = ss.join_ops_ok(side, [("record", "fork"), ("h2d", None)])
    assert not bad["ok"] and "107025" in bad["detail"][0]
    # no record at all: the H2D may read the pinned output before the callback
    bad2 = ss.join_ops_ok(side[:3], compute)
    assert not bad2["ok"]
    # a wait on an event nobody records: deadlock
    bad3 = ss.join_ops_ok(side, [("wait", "join"), ("h2d", None)])
    assert not bad3["ok"] and "never records" in bad3["detail"][0]


def test_gate_report_is_red_if_any_check_is():
    good = ss.capture_hostfunc_ok({}, {})
    bad = ss.replay_hostfunc_ok({0: 0}, {0: 0}, 1, 1)
    assert ss.gate_report([good])["ok"]
    rep = ss.gate_report([good, bad])
    assert not rep["ok"] and rep["failed"] == ["replay_one_per_layer"]
    assert "replay_one_per_layer" in ss.format_gate(rep)


# -- 4. the switch itself ---------------------------------------------------


def test_config_refuses_the_side_stream_unless_the_env_asks(monkeypatch):
    from vllm_ascend import envs

    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, overlap_side_stream=True)
    vcfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1, enable_eplb=False
        ),
        speculative_config=None,
    )
    monkeypatch.setattr(envs, ss.ENV_NAME, "0", raising=False)
    with pytest.raises(ValueError, match=ss.ENV_NAME):
        cfg._validate_preconditions(vcfg)
    monkeypatch.setattr(envs, ss.ENV_NAME, "1", raising=False)
    cfg._validate_preconditions(vcfg)  # allowed, loudly


def test_the_switch_is_off_by_default():
    assert KTOffloadConfig().overlap_side_stream is False


# -- 5. packed + side stream is a switch that would look on and be off --------


def _preconditions(**kw):
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, **kw)
    vcfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1, enable_eplb=False
        ),
        speculative_config=None,
    )
    return cfg, vcfg


def test_packed_and_side_stream_together_are_refused(monkeypatch):
    """Both on = the packed branch wins in route() and nothing forks.

    The service would then log "SIDE STREAM ENABLED", dump a config saying so,
    and serve the serial packed path: a reading attributed to the side stream
    that the side stream never produced. Mutant arm: delete the raise in
    config._validate_preconditions and this test goes green with both switches
    on, which is the failure being fenced.
    """
    from vllm_ascend import envs

    monkeypatch.setattr(envs, ss.ENV_NAME, "1", raising=False)  # even with the env set
    cfg, vcfg = _preconditions(overlap_side_stream=True, packed_transfer=True)
    with pytest.raises(ValueError, match="cannot both be true"):
        cfg._validate_preconditions(vcfg)
    # ...and each one alone still starts
    cfg, vcfg = _preconditions(overlap_side_stream=True, packed_transfer=False)
    cfg._validate_preconditions(vcfg)
    cfg, vcfg = _preconditions(overlap_side_stream=False, packed_transfer=True)
    cfg._validate_preconditions(vcfg)


def test_the_route_branch_order_this_refusal_is_about_has_not_moved():
    """If packed stops shadowing overlap in route(), revisit the refusal above."""
    import inspect

    from vllm_ascend.kt_offload.layer import KTLayerOffload

    src = inspect.getsource(KTLayerOffload.route)
    packed_at = src.index("if self.packed:")
    overlap_at = src.index("elif self.overlap:")
    assert packed_at < overlap_at, "route() no longer tests packed before overlap"
