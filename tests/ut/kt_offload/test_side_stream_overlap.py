# SPDX-License-Identifier: Apache-2.0
"""Side-stream overlap (cfg.overlap_side_stream) -- CPU only, fake streams.

Two layers of checking, and the second one is the point:

1. The call sequence the real ``KTLayerOffload`` emits (fork -> side wait ->
   D2H/host-func on the side stream -> join -> H2D on the compute stream),
   event lifetime, stale-join clearing, and the refusal when a capture does not
   follow onto the side stream.

2. A tiny stream simulator *executes* that recorded schedule and compares the
   result with the serial (overlap off) one **bit for bit**. Every mutant below
   (join dropped, join recorded too early, events not kept alive, capture not
   propagated) produces a schedule that is FASTER and WRONG -- each test asserts
   both, so the record shows why the on-card criterion is bit-identity and not
   TPOT. sglang's F1 failure was exactly this shape: the capture did not
   propagate to the side stream, the graph got quicker and the CPU experts'
   contribution froze at the capture run's value.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload import side_stream as ss
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan

H, TOPK, NEXP = 8, 2, 16

# ---------------------------------------------------------------------------
# a minimal device: streams of timed ops, events, pinned/device buffers
# ---------------------------------------------------------------------------


class Box:
    """A device or pinned buffer whose content is read when an op executes."""

    def __init__(self, value=None):
        self.value = value


class Op:
    def __init__(self, kind, fn, cost, event=None):
        self.kind, self.fn, self.cost, self.event = kind, fn, cost, event


class Sim:
    def __init__(self):
        self.programs: dict[str, list[Op]] = {}
        self.capturing = False
        self.captured: dict[str, list[Op]] = {}
        self.current = None

    def enqueue(self, stream, op):
        self.programs.setdefault(stream.name, []).append(op)

    def reset_programs(self):
        self.programs = {}
        for attr in ("_idx", "_clock"):
            if hasattr(self, attr):
                delattr(self, attr)

    def run(self, programs) -> float:
        """Execute per-stream op lists from the start, honouring event waits.

        Used for a graph replay: fresh indices and a fresh event table.
        """
        return self._run(programs, {}, {}, {})

    def drain(self) -> float:
        """Host-side device sync: everything enqueued and not yet executed runs.

        Indices, clocks and recorded events persist, so a second drain does not
        re-run what the first one already executed (the host runs ahead between
        drains -- which is exactly the window this file is about).
        """
        if not hasattr(self, "_idx"):
            self._idx, self._recorded, self._clock = {}, {}, {}
        return self._run(self.programs, self._idx, self._recorded, self._clock)

    def drain_stream(self, name: str) -> None:
        """Host wait on one stream: run its ops, and only what they wait on."""
        if not hasattr(self, "_idx"):
            self._idx, self._recorded, self._clock = {}, {}, {}
        if name not in self.programs:
            return
        while True:
            self._run({name: self.programs[name]}, self._idx, self._recorded, self._clock, partial=True)
            if self._idx.get(name, 0) >= len(self.programs[name]):
                return
            # blocked on an event another stream has not recorded yet: let that
            # stream run until it has (the device would do the same).
            blocking = self.programs[name][self._idx[name]]
            progressed = False
            for other, ops in self.programs.items():
                if other == name:
                    continue
                while self._idx.get(other, 0) < len(ops):
                    self._run({other: ops}, self._idx, self._recorded, self._clock, partial=True, one=True)
                    progressed = True
                    if id(blocking.event) in self._recorded:
                        break
                if id(blocking.event) in self._recorded:
                    break
            if not progressed:
                raise AssertionError("deadlock in drain_stream")

    def _run(self, programs, idx, recorded, clock, partial=False, one=False) -> float:
        """Time-ordered execution; an op's effect lands when it completes.

        Completion order across streams is what makes a missing or early join a
        data hazard here, exactly as on the device: the H2D reads the pinned
        buffer at the moment it executes, not when it was enqueued.
        """
        for name in programs:
            idx.setdefault(name, 0)
            clock.setdefault(name, 0.0)
        while True:
            runnable = []
            for name, ops in programs.items():
                if idx[name] >= len(ops):
                    continue
                op = ops[idx[name]]
                if op.kind == "wait" and id(op.event) not in recorded:
                    continue  # blocked until some stream records it
                start = max(clock[name], recorded[id(op.event)]) if op.kind == "wait" else clock[name]
                runnable.append((start + op.cost, name, op))
            if not runnable:
                break
            end, name, op = min(runnable, key=lambda r: (r[0], r[1]))  # noqa: PLW2901
            clock[name] = end
            if op.kind == "record":
                recorded[id(op.event)] = end
            elif op.kind != "wait":
                op.fn()
            idx[name] += 1
            if one:
                break
        if not partial and any(idx[n] < len(ops) for n, ops in programs.items()):
            raise AssertionError("deadlock: an event was never recorded")
        return max(clock.values(), default=0.0)


class FEvent:
    def __init__(self, sim):
        self.sim = sim

    def record(self, stream):
        self.sim.enqueue(stream, Op("record", None, 0.0, self))


class FStream:
    def __init__(self, sim, name):
        self.sim, self.name, self.npu_stream = sim, name, hash(name) & 0xFFFF

    def wait_event(self, ev):
        self.sim.enqueue(self, Op("wait", None, 0.0, ev))

    def synchronize(self):
        """Host waits for THIS stream only (torch.npu.Stream.synchronize).

        Other streams keep their queued work: that is the whole difference
        from kt's torch.npu.synchronize(device), and the window in which the
        host can run ahead of a pending H2D.
        """
        self.sim.host_syncs = getattr(self.sim, "host_syncs", 0) + 1
        self.sim.drain_stream(self.name)


class FakeKT:
    """kt wrapper on the simulator: CPU experts compute out = 3 * input."""

    _layer_has_pending_deferred: dict = {}

    def __init__(self, sim, device_in: Box, pinned_in: Box, pinned_out: Box, out_gpu: Box):
        self.ids_seen: list = []
        self.sim, self.device_in = sim, device_in
        self.pinned_in, self.pinned_out, self.out_gpu = pinned_in, pinned_out, out_gpu
        self.cpu_infer = SimpleNamespace(sync=lambda allow: setattr(self, "host_synced", True))
        self.calls: list[str] = []

    # eager (non-overlap) path, kept so the serial arm can run
    def submit_forward(self, hs, ids, w, stream):
        self.calls.append("submit_forward")
        self.pinned_in.value = self.device_in.value
        self.pinned_out.value = 3 * self.pinned_in.value

    def sync_forward(self, hs, stream):
        self.calls.append("sync_forward")
        self._h2d()
        return self.out_gpu

    # overlap / captured path
    def copy_inputs_to_cpu_buffers(self, hs, ids, w):
        self.calls.append("copy_inputs")
        self.ids_seen.append(ids)
        self.sim.enqueue(
            self.sim.current, Op("d2h", lambda: setattr(self.pinned_in, "value", self.device_in.value), 5.0)
        )

    def forward_on_pinned_buffers(self, hs, stream):
        self.calls.append("forward_on_pinned")
        self.pinned_out.value = 3 * self.pinned_in.value  # host-side submit + worker pool

    def copy_forward_output_to_device(self, hs):
        self.calls.append("copy_output")
        self.sim.enqueue(self.sim.current, Op("h2d", self._h2d, 5.0))
        return self.out_gpu

    def _h2d(self):
        v = self.pinned_out.value
        self.out_gpu.fill_(float("nan") if v is None else v)

    def run_pinned_forward_sync(self, shape_ref, stream):
        self.pinned_out.value = 3 * self.pinned_in.value


@pytest.fixture
def sim(monkeypatch):
    s = Sim()
    compute, side = FStream(s, "compute"), FStream(s, "side")
    s.current = compute
    s.compute, s.side = compute, side
    monkeypatch.setattr(kl, "_capturing", lambda: s.capturing and s.current is not None and _prop(s))
    monkeypatch.setattr(kl, "_subscribe_stream", lambda stream: None)
    monkeypatch.setattr(kl, "_kt_buffer_class", lambda: SimpleNamespace(capture_buffers={1: object()}))
    monkeypatch.setattr(kl, "_stream_handle", lambda device: 0)
    monkeypatch.setattr(kl, "_handle_of", lambda stream: stream.npu_stream)
    monkeypatch.setattr(kl, "_new_stream", lambda device: side)
    monkeypatch.setattr(kl, "_new_event", lambda device: FEvent(s))
    monkeypatch.setattr(kl, "_SIDE_STREAM", None)
    monkeypatch.setattr(kl, "_SIDE_EVENTS", [])
    monkeypatch.setattr(kl, "_CAPTURED_PAYLOADS", [])
    monkeypatch.setattr(kl, "HOST_FUNC_CALLS", {})

    def stream_ctx(stream, device):
        class Ctx:
            def __enter__(_self):
                s.prev, s.current = s.current, stream

            def __exit__(_self, *a):
                s.current = s.prev

        return Ctx()

    monkeypatch.setattr(kl, "_stream_ctx", stream_ctx)

    def host_func(stream, fn, payload):
        s.enqueue(stream, Op("hostfunc", lambda: fn(payload), 40.0))

    monkeypatch.setattr(kl, "_launch_host_func", host_func)

    fake_npu = SimpleNamespace(current_stream=lambda device=None: s.compute)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    s.capture_propagates = True
    return s


def _prop(s):
    """Whether the capture is visible on the stream the code is enqueueing to."""
    return s.current is s.compute or s.capture_propagates


def _runtime(monkeypatch, sim, overlap: bool, graph: bool, boxes):
    plan = build_plan([[0, 3]], NEXP)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, overlap_side_stream=overlap)
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=TOPK),
            model="/m",
            enforce_eager=not graph,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY" if graph else "NONE"),
            cudagraph_capture_sizes=[1] if graph else [],
        ),
    )
    rt = kl.KTLayerOffload(cfg, plan.layer(0), 0, vcfg)
    fake = FakeKT(sim, *boxes)
    monkeypatch.setattr(rt, "_build_wrapper", lambda: fake)
    monkeypatch.setattr(rt, "_prepare_capture_buffers", lambda device: None)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3), requires_grad=False)
    rt.after_weights_loaded(layer)
    rt._ids_i64 = {bs: torch.zeros(bs, TOPK, dtype=torch.int64) for bs in (1, 2, 4, 8)}
    return rt, fake


def _boxes():
    """device_in, pinned_in, pinned_out (host side), out_gpu (a real tensor)."""
    return (Box(1.0), Box(None), Box(None), torch.zeros(1, H))


def _inputs(seed):
    g = torch.Generator().manual_seed(seed)
    hs = torch.randn(1, H, generator=g).to(torch.bfloat16)
    ids = torch.tensor([[0, 9]], dtype=torch.int32)
    w = torch.rand(1, TOPK, generator=g)
    return hs, ids, w


# ---------------------------------------------------------------------------
# 1. the call sequence
# ---------------------------------------------------------------------------


def test_eager_is_never_forked_even_with_overlap_on(monkeypatch, sim):
    """Eager keeps kt's own submit/sync, whatever overlap_side_stream says.

    A fork there buys nothing -- the side stream's first op waits on an event
    recorded on the compute stream after the previous layer's H2D, so waiting
    for the side stream waits for the compute stream anyway -- and it takes kt's
    pinned slots out from under the device sync kt's submit_forward does.
    """
    boxes = _boxes()
    rt, fake = _runtime(monkeypatch, sim, overlap=True, graph=False, boxes=boxes)
    hs, ids, w = _inputs(0)
    rt.route(hs, w, ids, in_profile_run=False)
    assert rt._pending_join is None
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    assert fake.calls == ["submit_forward", "sync_forward"]
    assert "side" not in sim.programs and kl._SIDE_EVENTS == []


def test_a_stale_join_is_cleared_by_an_eager_route(monkeypatch, sim):
    """Capture, then an eager forward: the previous capture's join must not
    survive into that forward's merge (sglang 18c8b21e3 clears it in
    pre-dispatch for every path, not only the forked one)."""
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(9)
    rt.route(hs, w, ids, in_profile_run=False)
    assert rt._pending_join is not None
    sim.capturing = False
    rt.route(hs, w, ids, in_profile_run=False)  # eager, no merge in between
    assert rt._pending_join is None


def test_captured_overlap_records_fork_side_work_and_join(monkeypatch, sim):
    boxes = _boxes()
    rt, fake = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(1)
    rt.route(hs, w, ids, in_profile_run=False)
    assert rt._pending_join is not None
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    assert rt._pending_join is None  # consumed
    assert [o.kind for o in sim.programs["compute"]] == ["record", "wait", "h2d"]
    assert [o.kind for o in sim.programs["side"]] == ["wait", "d2h", "hostfunc", "record"]
    # the event objects the graph's nodes point at are kept for the process
    # (the startup probe's own pair is in there too)
    assert len(kl._SIDE_EVENTS) >= 1 and all(len(p) == 2 for p in kl._SIDE_EVENTS)
    # the probe's payload is in the list too; the layer's is the last one
    assert len(kl._CAPTURED_PAYLOADS) >= 1 and kl._CAPTURED_PAYLOADS[-1][0] is fake
    # kt got int64 ids from a buffer that predates the capture, so its
    # .to(torch.long) is the identity: nothing allocates inside the side-stream
    # context, where an allocation would not belong to the capture stream.
    assert fake.ids_seen and all(t.dtype == torch.int64 for t in fake.ids_seen)
    assert all(t is rt._ids_i64[t.shape[0]] for t in fake.ids_seen)


def test_capture_that_does_not_reach_the_side_stream_is_refused(monkeypatch, sim):
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    sim.capture_propagates = False
    hs, ids, w = _inputs(2)
    with pytest.raises(RuntimeError, match="did not propagate to the side stream"):
        rt.route(hs, w, ids, in_profile_run=False)


def test_stale_join_from_a_previous_capture_is_cleared(monkeypatch, sim):
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(3)
    rt.route(hs, w, ids, in_profile_run=False)
    stale = rt._pending_join
    rt.route(hs, w, ids, in_profile_run=False)  # a second capture, no merge in between
    assert rt._pending_join is not stale and rt._pending_join is not None
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    assert rt._pending_join is None


def test_merge_without_a_join_is_an_error(monkeypatch, sim):
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(4)
    rt.route(hs, w, ids, in_profile_run=False)
    rt._pending_join = None  # what "the join was dropped" looks like from merge
    with pytest.raises(AssertionError, match="no side-stream join"):
        rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))


def test_overlap_off_keeps_the_m2_path(monkeypatch, sim):
    boxes = _boxes()
    rt, fake = _runtime(monkeypatch, sim, overlap=False, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(5)
    rt.route(hs, w, ids, in_profile_run=False)
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    assert "side" not in sim.programs and kl._SIDE_EVENTS == []
    assert [o.kind for o in sim.programs["compute"]] == ["d2h", "hostfunc", "h2d"]


# ---------------------------------------------------------------------------
# 2. execute the recorded schedule: bit-identity vs wall time
# ---------------------------------------------------------------------------


def _capture_schedule(monkeypatch, sim, boxes, *, overlap: bool):
    rt, fake = _runtime(monkeypatch, sim, overlap=overlap, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(7)
    rt.route(hs, w, ids, in_profile_run=False)
    # the resident experts, between route and merge: the work the side stream is
    # supposed to hide behind. (The "~0.62 ms/layer" once written here is
    # retired; the measured per-layer GMM is 44.7 us and the whole-layer window
    # is what t4 item 2 measures. The simulated 30 us below is a fixture, not a
    # claim about the card.)
    sim.enqueue(sim.compute, Op("npu_experts", lambda: None, 30.0))
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    sim.capturing = False
    return {name: list(ops) for name, ops in sim.programs.items()}


def _replay(sim, programs, device_in: Box, out_gpu: torch.Tensor, value: float, routed_out: float = 100.0):
    """One graph replay with a fresh input; returns (merged value, wall time).

    The merge's add is enqueued on the compute stream after the H2D, so the
    merged value is ``routed_out + out_gpu`` read once the schedule has run.
    """
    device_in.value = value
    out_gpu.zero_()
    wall = sim.run(programs)
    return routed_out + float(out_gpu[0, 0]), wall


def _arms(monkeypatch, sim, mutate=None):
    """Serial (overlap off) vs overlap schedules, replayed on the same inputs."""
    serial_boxes = _boxes()
    serial = _capture_schedule(monkeypatch, sim, serial_boxes, overlap=False)
    sim.reset_programs()
    ov_boxes = _boxes()
    overlap = _capture_schedule(monkeypatch, sim, ov_boxes, overlap=True)
    if mutate is not None:
        overlap = mutate(overlap, ov_boxes)
    out_s, out_o, t_s, t_o = [], [], [], []
    for v in (2.0, 5.0, 9.0):
        y, t = _replay(sim, serial, serial_boxes[0], serial_boxes[3], v)
        out_s.append(y)
        t_s.append(t)
        y, t = _replay(sim, overlap, ov_boxes[0], ov_boxes[3], v)
        out_o.append(y)
        t_o.append(t)
    return out_s, out_o, max(t_s), max(t_o)


def _bits_equal(a, b):
    return torch.equal(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64))


def test_correct_overlap_is_bit_identical_and_not_slower(monkeypatch, sim):
    out_s, out_o, t_s, t_o = _arms(monkeypatch, sim)
    assert _bits_equal(out_s, out_o), (out_s, out_o)
    # serial: d2h 5 + host 40 + npu 30 + h2d 5 = 80; overlapped: max(45, 30) + 5 = 50
    assert t_o < t_s, (t_o, t_s)  # the point of the change; never the criterion on its own


def _drop_join(programs, boxes):
    return {n: [o for o in ops if not (n == "compute" and o.kind == "wait")] for n, ops in programs.items()}


def _join_before_host_func(programs, boxes):
    side = programs["side"]
    rec = [o for o in side if o.kind == "record"]
    rest = [o for o in side if o.kind != "record"]
    # the join event recorded right after the D2H, before the host func
    return {**programs, "side": rest[:2] + rec + rest[2:]}


def _events_not_kept(programs, boxes):
    """Events collected: their record/wait nodes degrade into no-ops."""
    out = {}
    for n, ops in programs.items():
        out[n] = [Op("nop", lambda: None, 0.0) if o.kind in ("wait", "record") else o for o in ops]
    return out


def _capture_not_propagated(programs, boxes):
    """The side stream's work never made it into the graph (sglang's F1).

    In that shape the D2H and the host func ran once, eagerly, while the graph
    was being captured: the pinned output keeps that run's value for ever.
    """
    device_in, pinned_in, pinned_out, _ = boxes
    pinned_in.value = device_in.value
    pinned_out.value = 3 * pinned_in.value  # the capture run's CPU output, frozen
    return {n: ([] if n == "side" else [o for o in ops if o.kind != "wait"]) for n, ops in programs.items()}


@pytest.mark.parametrize(
    "name, mutate",
    [
        ("join dropped", _drop_join),
        ("join recorded before the host func", _join_before_host_func),
        ("events not kept alive", _events_not_kept),
        ("capture not propagated to the side stream", _capture_not_propagated),
    ],
)
def test_mutants_are_faster_and_wrong(monkeypatch, sim, name, mutate):
    """Bit-identity catches each mutant; a wall-time criterion passes all of them.

    This is the argument for the on-card criterion: every one of these makes the
    step FASTER, so "TPOT improved" is evidence of nothing.
    """
    out_s, out_o, t_s, t_o = _arms(monkeypatch, sim, mutate)
    assert not _bits_equal(out_s, out_o), f"{name}: bit-identity did not catch it ({out_s} vs {out_o})"
    assert t_o <= t_s, f"{name}: expected the mutant to be no slower (time is not a criterion)"


def test_stale_output_fingerprint_of_the_unpropagated_capture(monkeypatch, sim):
    """The F1 shape specifically: one CPU output reused for every replay."""
    _, out_o, _, _ = _arms(monkeypatch, sim, _capture_not_propagated)
    assert len(set(out_o)) == 1, f"expected one frozen value, got {out_o}"


# ---------------------------------------------------------------------------
# 3. several layers sharing kt's pinned slots, with the host running ahead
#
# What the tests above could not see, and the 2026-09-20 A/B did (38/50 GSM8K
# answers differed with overlap on): kt keeps ONE pair of pinned buffers per
# batch size and hands layer L the slot ``L % 2``. The pinned output of layer L
# is read by an ASYNC H2D on the compute stream, and the CPU task of layer L+2
# writes that same buffer -- from the host, the moment it is submitted. Nothing
# orders those two unless something waits for the device.
#
# In the serial path kt's own submit_forward calls torch.npu.synchronize()
# before submitting, which is what supplied that ordering (its comment only
# claims to protect the input D2H; it protects the output slot too). The first
# side-stream implementation replaced it with side.synchronize(), which waits
# for the side stream only -- so the host ran ahead of the pending H2D.
#
# Everything above this line is single-layer or single-stream-order, i.e. zero
# coverage for slot reuse: with fewer than three layers no two layers share a
# slot, and with the device drained at every step the host can never run ahead.
# ---------------------------------------------------------------------------

DEPTH = 2  # kt's KExpertsCPUBuffer.buffer_depth


class KtBuffers:
    """kt's class-level pinned buffers: one set per batch size, ``L % 2`` slots."""

    def __init__(self):
        self.by_bs: dict[int, dict] = {}

    def get(self, bs: int) -> dict:
        return self.by_bs.setdefault(
            bs, {"in": [None] * DEPTH, "out": [None] * DEPTH, "gpu": [torch.zeros(1) for _ in range(DEPTH)]}
        )


class SlotKT:
    """kt wrapper with the real buffer ownership: slot = layer_idx % 2, shared by bs."""

    _layer_has_pending_deferred: dict = {}

    def __init__(self, sim, buffers: KtBuffers, layer_idx: int, device_in: Box):
        self.sim, self.buffers, self.layer_idx, self.device_in = sim, buffers, layer_idx, device_in
        self.cpu_infer = SimpleNamespace(sync=lambda allow: None)

    def _slot(self, bs):
        return self.buffers.get(bs), self.layer_idx % DEPTH

    # -- serial path (what M2 ships): kt syncs the whole device first
    def submit_forward(self, hs, ids, w, stream):
        self.sim.drain()  # kt's _wait_device -> torch.npu.synchronize(device)
        buf, s = self._slot(hs.shape[0])
        buf["in"][s] = self.device_in.value
        buf["out"][s] = 3 * buf["in"][s]

    def sync_forward(self, hs, stream):
        self.sim.drain()
        return self.copy_forward_output_to_device(hs)

    # -- captured / overlap path
    def copy_inputs_to_cpu_buffers(self, hs, ids, w):
        buf, s = self._slot(hs.shape[0])

        def d2h():
            buf["in"][s] = self.device_in.value

        self.sim.enqueue(self.sim.current, Op("d2h", d2h, 5.0))

    def forward_on_pinned_buffers(self, hs, stream):
        # host-side submit: the CPU experts write the pinned output NOW
        buf, s = self._slot(hs.shape[0])
        buf["out"][s] = 3 * buf["in"][s]

    def run_pinned_forward_sync(self, shape_ref, stream):
        buf, s = self._slot(shape_ref.shape[0])
        buf["out"][s] = 3 * buf["in"][s]

    def copy_forward_output_to_device(self, hs):
        buf, s = self._slot(hs.shape[0])
        gpu = buf["gpu"][s]

        def h2d():
            gpu.fill_(float("nan") if buf["out"][s] is None else buf["out"][s])

        self.sim.enqueue(self.sim.current, Op("h2d", h2d, 5.0))
        return gpu


def _layer_runtime(monkeypatch, sim, buffers, layer_idx, device_in, *, overlap, graph):
    plan = build_plan([[0, 3]] * 8, NEXP)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, overlap_side_stream=overlap)
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=TOPK),
            model="/m",
            enforce_eager=not graph,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY" if graph else "NONE"),
            cudagraph_capture_sizes=[1] if graph else [],
        ),
    )
    rt = kl.KTLayerOffload(cfg, plan.layer(layer_idx), layer_idx, vcfg)
    wrapper = SlotKT(sim, buffers, layer_idx, device_in)
    monkeypatch.setattr(rt, "_build_wrapper", lambda: wrapper)
    monkeypatch.setattr(rt, "_prepare_capture_buffers", lambda device: None)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3), requires_grad=False)
    rt.after_weights_loaded(layer)
    rt._ids_i64 = {bs: torch.zeros(bs, TOPK, dtype=torch.int64) for bs in (1, 2, 4, 8)}
    return rt


def _run_model(monkeypatch, sim, *, overlap, graph, layers=4, steps=3, batch_sizes=(1, 1, 1)):
    """Drive several layers per step the way forward_impl does; return each layer's CPU output."""
    buffers = KtBuffers()
    # one device-side hidden-states buffer per layer (each layer has its own
    # tensor; a single shared Box would be read at execution time and hide the
    # very reordering these tests look for)
    ins = [Box(0.0) for _ in range(layers)]
    rts = [_layer_runtime(monkeypatch, sim, buffers, i, ins[i], overlap=overlap, graph=graph) for i in range(layers)]
    seen: list[float] = []
    for step in range(steps):
        bs = batch_sizes[step % len(batch_sizes)]
        for i, rt in enumerate(rts):
            ins[i].value = 100.0 * (step + 1) + i  # this layer's hidden states, this step
            hs = torch.zeros(bs, H, dtype=torch.bfloat16)
            ids = torch.zeros(bs, TOPK, dtype=torch.int32)
            w = torch.ones(bs, TOPK)
            if graph:
                sim.capturing = True
            rt.route(hs, w, ids, in_profile_run=False)
            sim.enqueue(sim.compute, Op("npu_experts", lambda: None, 30.0))
            rt.merge(torch.zeros(bs, H, dtype=torch.bfloat16))
            sim.capturing = False
            expect = 3 * (100.0 * (step + 1) + i)
            # the merge's add is a compute-stream op after the H2D: it sees the
            # device-side CPU output as of that moment.
            gpu = buffers.get(bs)["gpu"][i % DEPTH]
            sim.enqueue(sim.compute, Op("consume", lambda g=gpu, e=expect: seen.append((float(g[0]), e)), 1.0))
        sim.drain()
        sim.reset_programs()
    return seen


def test_multi_layer_serial_is_correct(monkeypatch, sim):
    """Positive control: overlap off, the same four layers and shared slots."""
    seen = _run_model(monkeypatch, sim, overlap=False, graph=False)
    assert seen and all(got == want for got, want in seen), seen[:6]


def test_multi_layer_overlap_keeps_every_layers_cpu_output(monkeypatch, sim):
    """The regression: with the host running ahead, layer L+2 must not overwrite
    layer L's pinned output before its H2D has read it."""
    seen = _run_model(monkeypatch, sim, overlap=True, graph=False)
    wrong = [(got, want) for got, want in seen if got != want]
    assert not wrong, f"{len(wrong)}/{len(seen)} layers read another layer's CPU output: {wrong[:4]}"


def test_multi_layer_overlap_captured_path_is_safe(monkeypatch, sim):
    """The captured path orders the same reuse through the graph's nodes."""
    seen = _run_model(monkeypatch, sim, overlap=True, graph=True)
    wrong = [(got, want) for got, want in seen if got != want]
    assert not wrong, f"captured overlap corrupted {len(wrong)}/{len(seen)}: {wrong[:4]}"


def test_multi_layer_overlap_under_changing_batch_size(monkeypatch, sim):
    """Batch size alternates (concurrent requests): a new bs means new kt buffers,
    and the old ones must not be written while an H2D still reads them."""
    seen = _run_model(monkeypatch, sim, overlap=True, graph=False, batch_sizes=(1, 2, 1))
    wrong = [(got, want) for got, want in seen if got != want]
    assert not wrong, f"{len(wrong)}/{len(seen)} wrong with a changing batch size: {wrong[:4]}"


# ---------------------------------------------------------------------------
# 4. the switch: refused unless the operator asks for it explicitly
#
# 2026-09-20, in order: (a) the guard read torch.npu.is_current_stream_capturing()
# as if it meant "this stream's work is in the graph" -- it does not; (b) the
# replacement was a start-up probe that captured the same fork/join *pattern*
# with a scalar copy and a trivial host function: it PASSED on the platform
# where every real-layer capture failed with 107025. A stand-in weaker than its
# subject passes where the subject fails. So there is still no probe.
#
# 2026-09-22: the 107025 verdict itself was withdrawn (it was our own empty
# capture buffer, and the one arm that really reproduces 107025 is the one
# whose side-stream record nobody joins -- t5 off_evt vs off_evtw). The switch
# is therefore conditional, not dead: refused unless
# VLLM_ASCEND_DSV41_KT_SIDE_STREAM=1, and then loud.
# ---------------------------------------------------------------------------


def _preconditions_cfg():
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=8, num_resident_experts=100, overlap_side_stream=True)
    vcfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1, enable_eplb=False
        ),
        model_config=SimpleNamespace(enforce_eager=False),
        speculative_config=None,
    )
    return cfg, vcfg


def test_production_refuses_the_switch_without_the_env(monkeypatch):
    from vllm_ascend import envs

    monkeypatch.setattr(envs, ss.ENV_NAME, "0", raising=False)
    cfg, vcfg = _preconditions_cfg()
    with pytest.raises(ValueError, match=ss.ENV_NAME):
        cfg._validate_preconditions(vcfg)
    vcfg.model_config.enforce_eager = True
    with pytest.raises(ValueError, match=ss.ENV_NAME):
        cfg._validate_preconditions(vcfg)  # eager too: it would be a no-op there anyway


def test_the_env_opens_it(monkeypatch):
    from vllm_ascend import envs

    monkeypatch.setattr(envs, ss.ENV_NAME, "1", raising=False)
    cfg, vcfg = _preconditions_cfg()
    cfg._validate_preconditions(vcfg)


def test_a_typo_in_the_env_does_not_open_it(monkeypatch):
    from vllm_ascend import envs

    for value in ("true", "yes", "", "0", "01"):
        monkeypatch.setattr(envs, ss.ENV_NAME, value, raising=False)
        cfg, vcfg = _preconditions_cfg()
        with pytest.raises(ValueError, match=ss.ENV_NAME):
            cfg._validate_preconditions(vcfg)


def test_enabling_it_says_so_in_the_log(monkeypatch, sim):
    """An operator reading serve.log must see which arm is running.

    The vllm logger is captured by hand: it has its own handlers and does not
    propagate to pytest's caplog (see the note at the top of layer.py).
    """
    said = []
    monkeypatch.setattr(kl.logger, "warning", lambda msg, *a: said.append(msg % a if a else msg))
    _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=_boxes())
    assert any("SIDE STREAM ENABLED" in m and "侧流已启用" in m for m in said), said
    # ...and the serial arm stays quiet, so the line means something
    said.clear()
    _runtime(monkeypatch, sim, overlap=False, graph=True, boxes=_boxes())
    assert not [m for m in said if "SIDE STREAM" in m], said


def test_the_object_can_still_be_built_for_bisection(monkeypatch, sim):
    """The on-card harness constructs the config directly to bisect the limit."""
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    assert rt.overlap is True  # no probe, no refusal at this level


# ---------------------------------------------------------------------------
# 5. the two runtime gates the side stream carries (each with its mutant)
# ---------------------------------------------------------------------------


def test_two_layers_sharing_a_kt_slot_cannot_be_in_flight_at_once(monkeypatch, sim):
    """A6 as a check, not a comment: layer L and L+2 share kt's pinned slot.

    In the real order this cannot happen -- L's join is consumed in L's merge,
    which runs before L+2 is reached -- so the mutant has to be built by hand:
    submit L, skip its merge, submit L+2.
    """
    monkeypatch.setattr(kl, "_SIDE_INFLIGHT", {})
    buffers = KtBuffers()
    ins = [Box(1.0) for _ in range(4)]
    rts = [_layer_runtime(monkeypatch, sim, buffers, i, ins[i], overlap=True, graph=True) for i in range(4)]
    sim.capturing = True
    hs = torch.zeros(1, H, dtype=torch.bfloat16)
    ids = torch.zeros(1, TOPK, dtype=torch.int32)
    w = torch.ones(1, TOPK)
    rts[0].route(hs, w, ids, in_profile_run=False)
    rts[1].route(hs, w, ids, in_profile_run=False)  # slot 1: allowed alongside slot 0
    with pytest.raises(AssertionError, match="pinned slot 0"):
        rts[2].route(hs, w, ids, in_profile_run=False)  # slot 0 again, layer 0 not joined


def test_the_normal_layer_order_never_trips_the_slot_guard(monkeypatch, sim):
    """Positive control for the guard above: four layers, three steps, no raise."""
    monkeypatch.setattr(kl, "_SIDE_INFLIGHT", {})
    seen = _run_model(monkeypatch, sim, overlap=True, graph=True)
    assert seen and all(got == want for got, want in seen)
    assert kl._SIDE_INFLIGHT == {}, kl._SIDE_INFLIGHT  # every layer released its slot


def test_a_host_func_that_runs_during_the_capture_is_refused(monkeypatch, sim):
    """The silent non-replay shape: the node was executed, not recorded.

    Nothing about the output can see this at capture time (it is the capture
    run's own value), and it makes every replay faster -- so the counter is
    the only witness.
    """
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(11)
    rt.route(hs, w, ids, in_profile_run=False)
    kl.HOST_FUNC_CALLS[rt.layer_idx] = kl.HOST_FUNC_CALLS.get(rt.layer_idx, 0) + 1
    with pytest.raises(RuntimeError, match="ran during the capture"):
        rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))


def test_the_counter_moving_between_replays_is_fine(monkeypatch, sim):
    """Negative control: outside a capture the counter grows every step."""
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(12)
    rt.route(hs, w, ids, in_profile_run=False)
    sim.capturing = False  # replay, not capture
    kl.HOST_FUNC_CALLS[rt.layer_idx] = kl.HOST_FUNC_CALLS.get(rt.layer_idx, 0) + 1
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
