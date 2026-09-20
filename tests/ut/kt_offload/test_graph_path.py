# SPDX-License-Identifier: Apache-2.0
"""kt offload captured (graph) path -- CPU only, fake kt, fake capture.

What is checked here: which path each forward takes, what the capture run
records, that nothing on the captured path is gated by a runtime flag, and that
a replay of the recorded host-func node recomputes the CPU experts from the
*current* pinned inputs (the "many inputs, one output" fingerprint of a graph
that did not re-run the node). Whether ACL really replays the node is an
on-card question (docs/moe_offload_design.md, M2 validation).
"""

from __future__ import annotations

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan, check_split

H, TOPK = 8, 2


@pytest.fixture
def our_logs():
    """Capture what *our* logger emits.

    Not ``caplog``: we log through ``vllm.logger``'s logger, and vLLM's
    dictConfig sets ``propagate: False`` on the ``vllm`` logger, so no record
    of ours ever reaches the root handler caplog installs. The failure
    direction is safe (caplog reads empty and the test goes red), but the test
    has to attach to the object it is judging.
    """
    import logging

    records: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage() % () if not record.args else record.getMessage())

    handler = Collect()
    previous = kl.logger.level
    kl.logger.addHandler(handler)
    kl.logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        kl.logger.removeHandler(handler)
        kl.logger.setLevel(previous)



class FakeKTBuffer:
    """Stands in for kt's class-level KExpertsCPUBuffer."""

    capture_bs: list = []
    capture_buffers: dict = {}

    @classmethod
    def reset(cls):
        cls.capture_bs, cls.capture_buffers = [], {}

    @classmethod
    def get_buffer(cls, hs, topk):
        bs = hs.shape[0]
        if bs in cls.capture_buffers:
            return cls.capture_buffers[bs]
        buf = {"in": torch.zeros(bs, H), "out": torch.zeros(bs, H)}
        if bs in cls.capture_bs:
            cls.capture_buffers[bs] = buf
        return buf


class FakeKT:
    """kt wrapper whose CPU 'experts' compute out = 3 * input (on pinned buffers)."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def set_capture_batch_sizes(sizes):
        FakeKTBuffer.capture_bs = list(sizes)

    # eager
    def submit_forward(self, hs, ids, w, stream):
        self.calls.append("submit_forward")

    def sync_forward(self, hs, stream):
        self.calls.append("sync_forward")
        return torch.full_like(hs, 0.5)

    # captured
    def copy_inputs_to_cpu_buffers(self, hs, ids, w):
        self.calls.append("copy_inputs")
        FakeKTBuffer.get_buffer(hs, TOPK)["in"].copy_(hs.float())

    def run_pinned_forward_sync(self, shape_ref, stream):
        buf = FakeKTBuffer.capture_buffers[shape_ref.shape[0]]
        buf["out"].copy_(buf["in"] * 3)

    def copy_forward_output_to_device(self, hs):
        self.calls.append("copy_output")
        return FakeKTBuffer.capture_buffers[hs.shape[0]]["out"].to(hs.dtype)


class Capture:
    """Minimal stand-in for stream capture: flag + recorded host-func nodes."""

    def __init__(self):
        self.on = False
        self.nodes = []
        self.subscribed = []

    def launch(self, stream, fn, payload):
        assert self.on, "host func launched outside a capture"
        self.nodes.append((fn, payload))


@pytest.fixture
def cap(monkeypatch):
    FakeKTBuffer.reset()
    c = Capture()
    monkeypatch.setattr(kl, "_capturing", lambda: c.on)
    monkeypatch.setattr(kl, "_subscribe_stream", lambda stream: c.subscribed.append(stream))
    monkeypatch.setattr(kl, "_launch_host_func", c.launch)
    monkeypatch.setattr(kl, "_kt_buffer_class", lambda: FakeKTBuffer)
    monkeypatch.setattr(kl, "_stream_handle", lambda device: 0)
    monkeypatch.setattr(kl, "_CAPTURE_BUFFERS_READY", None)
    monkeypatch.setattr(kl, "_CAPTURED_PAYLOADS", [])
    monkeypatch.setattr(kl, "HOST_FUNC_CALLS", {})
    fake_npu = SimpleNamespace(current_stream=lambda device=None: SimpleNamespace(npu_stream=1234))
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    return c


def _vcfg(eager=False, sizes=(1, 2, 4, 8)):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=TOPK),
            model="/m",
            enforce_eager=eager,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_AND_PIECEWISE"), cudagraph_capture_sizes=list(sizes)
        ),
    )


def _runtime(monkeypatch, eager=False, sizes=(1, 2, 4, 8), sets=((1, 4, 6),)):
    plan = build_plan([list(s) for s in sets], 16)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1)
    rt = kl.KTLayerOffload(cfg, plan.layer(0), 0, _vcfg(eager, sizes))
    fake = FakeKT()
    monkeypatch.setattr(rt, "_build_wrapper", lambda: fake)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3), requires_grad=False)
    rt.after_weights_loaded(layer)
    return rt, fake


def _inputs(bs, seed):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(bs, H, generator=g).to(torch.bfloat16)
    ids = torch.argsort(torch.rand(bs, 16, generator=g), dim=1)[:, :TOPK].to(torch.int32)
    w = torch.rand(bs, TOPK, generator=g) + 0.1
    return hs, ids, w


def test_graph_mode_detected_and_buffers_prepared(cap, monkeypatch):
    rt, _ = _runtime(monkeypatch)
    assert rt.graph_enabled and rt.capture_sizes == [1, 2, 4, 8]
    assert FakeKTBuffer.capture_bs == [1, 2, 4, 8]
    assert sorted(FakeKTBuffer.capture_buffers) == [1, 2, 4, 8]  # allocated before any capture


def test_eager_config_takes_no_graph_setup(cap, monkeypatch):
    rt, _ = _runtime(monkeypatch, eager=True)
    assert not rt.graph_enabled and FakeKTBuffer.capture_bs == [] and FakeKTBuffer.capture_buffers == {}


def test_capture_sizes_above_kt_limit_are_refused(cap, monkeypatch):
    with pytest.raises(ValueError):
        _runtime(monkeypatch, sizes=(8, 128))  # chunked_prefill_size = 64


def test_non_captured_forward_takes_eager_path(cap, monkeypatch):
    rt, fake = _runtime(monkeypatch)
    hs, ids, w = _inputs(4, 0)
    rt.route(hs, w, ids, in_profile_run=False)
    rt.merge(torch.zeros(4, H, dtype=torch.bfloat16))
    assert fake.calls == ["submit_forward", "sync_forward"] and cap.nodes == []


@pytest.mark.parametrize("in_profile_run", [False, True])
def test_capture_records_one_host_func_and_no_eager_calls(cap, monkeypatch, in_profile_run):
    rt, fake = _runtime(monkeypatch)
    hs, ids, w = _inputs(4, 1)
    cap.on = True
    nw, nid = rt.route(hs, w, ids, in_profile_run=in_profile_run)
    rt.merge(torch.zeros(4, H, dtype=torch.bfloat16))
    cap.on = False
    # the profile flag must not gate anything that is being captured
    assert len(cap.nodes) == 1 and len(kl._CAPTURED_PAYLOADS) == 1
    assert fake.calls == ["copy_inputs", "copy_output"]  # never submit/sync (they sync the device)
    assert cap.subscribed and check_split(rt.placement, ids, w, nid, nw) == []
    fn, payload = cap.nodes[0]
    assert payload[1].device.type == "meta" and tuple(payload[1].shape) == (4, H)
    assert kl._CAPTURED_PAYLOADS[0] is payload  # the node's payload is the one kept alive


def test_capture_of_unprepared_size_is_refused(cap, monkeypatch):
    rt, _ = _runtime(monkeypatch, sizes=(1, 2))
    hs, ids, w = _inputs(4, 2)
    cap.on = True
    with pytest.raises(RuntimeError, match="pre-allocated"):
        rt.route(hs, w, ids, in_profile_run=False)


def test_replay_recomputes_from_current_inputs(cap, monkeypatch):
    """Different inputs through the same recorded node give different CPU outputs."""
    rt, fake = _runtime(monkeypatch)
    hs0, ids, w = _inputs(4, 3)
    cap.on = True
    rt.route(hs0, w, ids, in_profile_run=False)
    captured_out = rt.merge(torch.zeros(4, H, dtype=torch.bfloat16))
    cap.on = False
    fn, payload = cap.nodes[0]

    outs = []
    for seed in (10, 11, 12):
        hs, _, _ = _inputs(4, seed)
        FakeKTBuffer.capture_buffers[4]["in"].copy_(hs.float())  # the captured D2H, replayed
        fn(payload)  # the host-func node, replayed
        outs.append(fake.copy_forward_output_to_device(hs).float())  # the captured H2D, replayed
        assert torch.allclose(outs[-1], hs.float() * 3, atol=0.1)
    assert not torch.equal(outs[0], outs[1]) and not torch.equal(outs[1], outs[2])
    assert kl.HOST_FUNC_CALLS[0] == 3
    # the capture run itself did not execute the node: its CPU part is the stale buffer
    assert torch.all(captured_out == 0)


def test_host_func_errors_are_logged_not_raised(cap, monkeypatch, our_logs):
    rt, fake = _runtime(monkeypatch)
    hs, ids, w = _inputs(2, 4)
    cap.on = True
    rt.route(hs, w, ids, in_profile_run=False)
    cap.on = False
    fn, payload = cap.nodes[0]

    def boom(shape_ref, stream):
        raise RuntimeError("kt failed")

    monkeypatch.setattr(fake, "run_pinned_forward_sync", boom)
    fn(payload)  # must not raise into ACL
    assert any("NOT this step's" in m for m in our_logs), our_logs
    assert kl.HOST_FUNC_CALLS.get(0, 0) == 0


def test_eager_built_layer_refuses_capture(cap, monkeypatch):
    rt, _ = _runtime(monkeypatch, eager=True)
    hs, ids, w = _inputs(2, 5)
    cap.on = True
    with pytest.raises(RuntimeError, match="eager"):
        rt.route(hs, w, ids, in_profile_run=False)


# ---------------------------------------------------------------------------
# mutation arms
# ---------------------------------------------------------------------------


def test_mutation_profile_flag_gates_capture_is_red(cap, monkeypatch):
    """Mutant route: checks the profile flag before the capture branch (the M1 order)."""
    rt, fake = _runtime(monkeypatch)

    def mutant_route(hs, w, ids, *, in_profile_run):
        if in_profile_run and rt.cfg.skip_cpu_in_profile_run:
            return ids, w
        return kl.KTLayerOffload.route(rt, hs, w, ids, in_profile_run=in_profile_run)

    hs, ids, w = _inputs(4, 6)
    cap.on = True
    mutant_route(hs, w, ids, in_profile_run=True)
    assert len(cap.nodes) == 0  # the node the real code records is missing -> every replay lacks CPU experts


def test_mutation_eager_sync_under_capture_is_red(cap, monkeypatch):
    """Mutant: capture runs the eager path; the CPU output is baked as a constant."""
    rt, fake = _runtime(monkeypatch)
    monkeypatch.setattr(kl, "_capturing", lambda: False)  # probe that never sees the capture
    hs, ids, w = _inputs(4, 7)
    cap.on = True
    rt.route(hs, w, ids, in_profile_run=False)
    rt.merge(torch.zeros(4, H, dtype=torch.bfloat16))
    assert cap.nodes == [] and "sync_forward" in fake.calls


def test_mutation_stale_output_fingerprint_is_red(cap, monkeypatch):
    """Mutant replay that skips the host-func node: many inputs, one output."""
    rt, fake = _runtime(monkeypatch)
    hs0, ids, w = _inputs(4, 8)
    cap.on = True
    rt.route(hs0, w, ids, in_profile_run=False)
    cap.on = False
    outs = []
    for seed in (20, 21):
        hs, _, _ = _inputs(4, seed)
        FakeKTBuffer.capture_buffers[4]["in"].copy_(hs.float())
        outs.append(fake.copy_forward_output_to_device(hs).float())  # node not run
    assert torch.equal(outs[0], outs[1])  # the fingerprint the on-card check looks for


# ---------------------------------------------------------------------------
# kt capture probe
# ---------------------------------------------------------------------------


def test_capture_probe_installed_when_kt_has_hook(monkeypatch, our_logs):
    got = []
    kl._install_capture_probe(SimpleNamespace(set_capture_probe=got.append))
    assert got == [kl._capturing]
    kl._install_capture_probe(SimpleNamespace())
    assert any("set_capture_probe" in m for m in our_logs), our_logs


def test_graph_env_for_kt(cap, monkeypatch):
    """_build_wrapper sets KT_EXTERNAL_NPU_REPORT_SUBSCRIBER=1 in graph mode (kt import faked)."""
    monkeypatch.delenv("KT_EXTERNAL_NPU_REPORT_SUBSCRIBER", raising=False)
    monkeypatch.setenv("KT_FORCE_SYNC_SUBMIT", "unset-by-test")  # restored after the test
    built = {}

    class KTMoEWrapper:
        def __init__(self, **kw):
            built.update(kw)
            built["env"] = os.environ.get("KT_EXTERNAL_NPU_REPORT_SUBSCRIBER")

        def load_weights(self, p2l):
            built["p2l"] = p2l

        set_capture_probe = staticmethod(lambda fn: built.setdefault("probe", fn))

    monkeypatch.setitem(sys.modules, "kt_kernel", types.SimpleNamespace(KTMoEWrapper=KTMoEWrapper))
    plan = build_plan([[0, 3]], 16)
    rt = kl.KTLayerOffload(
        KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1), plan.layer(0), 0, _vcfg()
    )
    rt.swiglu_limit = 10.0
    rt._build_wrapper()
    assert built["env"] == "1" and built["method"] == "MXFP4" and built["swiglu_limit"] == 10.0
    assert built["probe"] is kl._capturing and torch.equal(built["p2l"], torch.arange(16))
