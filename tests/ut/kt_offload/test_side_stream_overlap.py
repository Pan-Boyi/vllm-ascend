# SPDX-License-Identifier: Apache-2.0
"""Side-stream overlap (cfg.overlap_side_stream) -- CPU only, fake streams.

Two layers of checking, and the second one is the point:

1. The call sequence the real ``KTLayerOffload`` emits (fill this layer's own
   buffers on the compute stream -> fork -> side wait -> D2H/host-func on the
   side stream -> join -> H2D on the compute stream), event lifetime, stale-join
   refusal, and the refusal when a capture does not follow onto the side stream.

2. A tiny stream simulator *executes* that recorded schedule and compares the
   result with the serial (overlap off) one **bit for bit**. Every mutant below
   (join dropped, join recorded too early, events not kept alive, capture not
   propagated, 🔴 fork recorded before the fills) produces a schedule that is
   FASTER and WRONG -- each test asserts both, so the record shows why the
   on-card criterion is bit-identity and not TPOT. sglang's F1 failure was
   exactly this shape: the capture did not propagate to the side stream, the
   graph got quicker and the CPU experts' contribution froze at the capture
   run's value.

🔴 2026-09-22: what the forked path reads changed, and so did this file.
It used to hand ``hs``/``topk_ids``/``topk_weights`` straight to kt's
``copy_inputs_to_cpu_buffers`` **from the side stream**, with the fork event
recorded *before* the compute-stream work those reads depend on. Two holes:
the caller drops its last reference to the routing tensors the moment
``route()`` returns (routed_experts.py:707-708), so inside a capture the
allocator may hand that storage to a later allocation; and the int64 id buffer
was filled by the compute stream *after* the fork, so whichever stream got
there first won. On card, with the compute stream made late on purpose
(``t5_side_reuse.py --stress ids-delay``, 40 layers x 30 steps), the side
stream read the previous step's ids in 1160 of 1200 layer-steps and every step
diverged, while the serial arm stayed 30/30 bit-identical. The path now fills
this layer's **own** buffers on the compute stream and records the fork after
them; ``_early_fork`` below is that edge taken out again.
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
        # id(destination Box) -> the Box a _device_fill of it reads at the
        # moment it executes. Without it a fill would copy the value the tensor
        # had when the op was *enqueued*, which is the reordering these tests
        # exist to see.
        self.fill_src: dict[int, Box] = {}
        # 🔴 P1a. id(ExternalEvent) -> the time it was signalled, and it
        # PERSISTS ACROSS REPLAYS: that is the whole difference from the
        # ordinary event table, which is rebuilt per replay. An external event
        # is a value in device memory; nothing clears it but the ``reset`` node
        # in the graph. Drop that node and replay 2's wait sails through on
        # replay 1's signal -- faster, wrong, silent, and invisible to any
        # model that resets the table between replays.
        self.ext_signaled: dict[int, float] = {}
        # How the modelled completion thread behaves: "clean" (sync, then
        # record), "early-record" (record first) or "no-record" (never).
        self.async_mode = "clean"

    def enqueue(self, stream, op):
        self.programs.setdefault(stream.name, []).append(op)

    def reset_programs(self):
        self.programs = {}
        self.ext_signaled = {}
        for attr in ("_idx", "_clock"):
            if hasattr(self, attr):
                delattr(self, attr)

    def run(self, programs) -> float:
        """Execute per-stream op lists from the start, honouring event waits.

        Used for a graph replay: fresh indices and a fresh event table. The op
        lists are copied because the modelled completion thread appends to them
        while the replay runs (it is host work, not a graph node), and
        ``ext_signaled`` is NOT reset -- see the comment on it.
        """
        # A signal left over from an earlier replay happened before this one
        # started, so its timestamp is 0 on this replay's clock. Keeping the
        # old absolute time would make a stale signal look like a wait that
        # still delayed -- which is exactly the bug it must NOT hide.
        self.ext_signaled = dict.fromkeys(self.ext_signaled, 0.0)
        return self._run({n: list(ops) for n, ops in programs.items()}, {}, {}, {})

    def async_complete(self, work_fn, cost, event) -> None:
        """Model the completion thread, from inside the host-func node.

        What the node does is hand the task over and return; what happens next
        is another thread waiting ``cost`` for kt and only then writing the
        value the device is parked on. Modelled as a serial FIFO program of its
        own (kt's queue is exactly that) gated on the moment of the hand-off.

        ``sim.async_mode`` picks the arm:
          ``clean``         work, then signal -- production.
          ``early-record``  signal, then work. 🔴 The main failure: the H2D is
                            released while the pinned output still holds the
                            previous step's value, and nothing raises.
          ``no-record``     never signal. The device waits for ever, which here
                            is the deadlock assertion at the end of ``_run``.
        """
        gate = object()
        self._active_recorded[id(gate)] = self._now
        ops = [Op("wait", None, 0.0, gate)]
        signal = Op("ext_record", None, 0.0, event)
        work = Op("ktwork", work_fn, cost)
        if self.async_mode == "early-record":
            ops += [signal, work]
        elif self.async_mode == "no-record":
            ops += [work]
        else:
            ops += [work, signal]
        self._active_programs.setdefault("completer", []).extend(ops)

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
        self._active_programs, self._active_recorded = programs, recorded
        while True:
            runnable = []
            # list(): the modelled completion thread adds a program of its own
            # while this loop runs (a host-func node hands its task over).
            for name, ops in list(programs.items()):
                idx.setdefault(name, 0)
                clock.setdefault(name, 0.0)
                if idx[name] >= len(ops):
                    continue
                op = ops[idx[name]]
                table = recorded if op.kind == "wait" else self.ext_signaled
                if op.kind in ("wait", "ext_wait") and id(op.event) not in table:
                    continue  # blocked until someone signals it
                start = max(clock[name], table[id(op.event)]) if op.kind in ("wait", "ext_wait") else clock[name]
                runnable.append((start + op.cost, name, op))
            if not runnable:
                break
            end, name, op = min(runnable, key=lambda r: (r[0], r[1]))  # noqa: PLW2901
            clock[name] = end
            self._now = end
            if op.kind == "record":
                recorded[id(op.event)] = end
            elif op.kind == "ext_record":
                self.ext_signaled[id(op.event)] = end
            elif op.kind == "ext_reset":
                self.ext_signaled.pop(id(op.event), None)
            elif op.kind not in ("wait", "ext_wait"):
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
    """kt wrapper on the simulator: CPU experts compute out = 3 * input.

    Only the *serial* paths are exercised through it now (eager submit/sync and
    the captured non-forked node). The forked path stopped calling kt's pinned
    buffers on 2026-09-22; it runs ``_kt_packed_forward`` on this layer's own
    buffers, which ``sim`` replaces with ``_fake_packed_forward``.
    """

    _layer_has_pending_deferred: dict = {}

    def __init__(self, sim, device_in: Box, pinned_in: Box, pinned_out: Box, out_gpu: Box):
        self.ids_seen: list = []
        self.sim, self.device_in = sim, device_in
        self.pinned_in, self.pinned_out, self.out_gpu = pinned_in, pinned_out, out_gpu
        self.cpu_infer = SimpleNamespace(
            submit=lambda task: None, sync=lambda allow: setattr(self, "host_synced", True)
        )
        self.moe = SimpleNamespace(forward_task=lambda *a: None)
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

    # captured, non-forked path
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


# ---------------------------------------------------------------------------
# the per-layer side buffers the forked path fills and reads
# ---------------------------------------------------------------------------


class Store:
    """The combined device (or pinned) allocation; its regions are Boxes."""

    def __init__(self):
        self.hidden, self.ids, self.weights = Box(None), Box(None), Box(None)

    def regions(self):
        return (self.hidden, self.ids, self.weights)


def make_side_buffers(sim: Sim, device_in: Box, out_gpu: torch.Tensor, ids_in: Box, w_in: Box) -> dict:
    """``_packed_buffers``' shape, with Boxes where the device tensors would be.

    ``dev_out`` has to be a real tensor: ``merge`` does arithmetic with what
    ``_packed_output`` returns.
    """
    dev, pin = Store(), Store()
    buf = {
        "dev": dev,
        "pin": pin,
        "dev_hidden": dev.hidden,
        "dev_ids": dev.ids,
        "dev_weights": dev.weights,
        "pin_hidden": pin.hidden,
        "pin_ids": pin.ids,
        "pin_weights": pin.weights,
        "pin_out": Box(None),
        "dev_out": out_gpu,
        "bsz": Box(1),
    }
    sim.fill_src[id(dev.hidden)] = device_in
    sim.fill_src[id(dev.ids)] = ids_in
    sim.fill_src[id(dev.weights)] = w_in
    return buf


def _fake_packed_forward(payload) -> None:
    """The host-func node: this layer's CPU experts on its own pinned buffers.

    ``3 * hidden`` matches FakeKT so the serial and forked arms are comparable
    bit for bit, and the ``+ ids`` term is what makes a stale id buffer visible
    in the output rather than only in a probe.
    """
    _wrapper, buf, layer_idx = payload
    hidden, ids = buf["pin_hidden"].value, buf["pin_ids"].value
    kl.HOST_FUNC_CALLS[layer_idx] = kl.HOST_FUNC_CALLS.get(layer_idx, 0) + 1
    buf["pin_out"].value = None if hidden is None else 3 * hidden + (ids or 0)


@pytest.fixture
def sim(monkeypatch):
    s = Sim()
    compute, side = FStream(s, "compute"), FStream(s, "side")
    s.current = compute
    s.compute, s.side = compute, side
    monkeypatch.setattr(kl, "_capturing", lambda: s.capturing and s.current is not None and _prop(s))
    monkeypatch.setattr(kl, "_subscribe_stream", lambda stream: None)
    monkeypatch.setattr(kl, "_kt_buffer_class", lambda: SimpleNamespace(capture_buffers={1: object(), 2: object()}))
    monkeypatch.setattr(kl, "_stream_handle", lambda device: 0)
    monkeypatch.setattr(kl, "_handle_of", lambda stream: stream.npu_stream)
    monkeypatch.setattr(kl, "_new_stream", lambda device: side)
    monkeypatch.setattr(kl, "_new_event", lambda device: FEvent(s))
    monkeypatch.setattr(kl, "_SIDE_STREAM", None)
    monkeypatch.setattr(kl, "_SIDE_EVENTS", [])
    monkeypatch.setattr(kl, "_CAPTURED_PAYLOADS", [])
    monkeypatch.setattr(kl, "HOST_FUNC_CALLS", {})
    monkeypatch.setattr(kl, "_SIDE_BUFFER_OWNER", {})
    monkeypatch.setattr(kl, "_kt_packed_forward", _fake_packed_forward)

    def device_fill(dst, src):
        """A compute-stream write into one region of this layer's device buffer.

        Cost 5 each against the 2-unit D2H of the combined buffer: with the
        fork recorded after these (production) the D2H cannot start before they
        finish, and with it recorded before them (the mutant) the D2H finishes
        first and copies whatever the previous replay left. The numbers are a
        fixture -- the ordering they expose is not.
        """
        src_box = s.fill_src.get(id(dst))

        def do():
            dst.value = src_box.value if src_box is not None else _scalar(src)

        s.enqueue(s.current, Op("fill", do, 5.0))

    def host_copy(dst, src):
        if isinstance(dst, Store):  # the one D2H: whole combined buffer
            def d2h():
                for d, sr in zip(dst.regions(), src.regions()):
                    d.value = sr.value

            s.enqueue(s.current, Op("d2h", d2h, 2.0))
        else:  # the H2D of the pinned output into this layer's device tensor
            def h2d():
                v = src.value
                dst.fill_(float("nan") if v is None else v)

            s.enqueue(s.current, Op("h2d", h2d, 5.0))

    monkeypatch.setattr(kl, "_device_fill", device_fill)
    monkeypatch.setattr(kl, "_host_copy", host_copy)
    monkeypatch.setattr(kl, "_pinned_empty", lambda shape, dtype: torch.empty(shape, dtype=dtype))

    def stream_ctx(stream, device):
        class Ctx:
            def __enter__(_self):
                s.prev, s.current = s.current, stream

            def __exit__(_self, *a):
                s.current = s.prev

        return Ctx()

    monkeypatch.setattr(kl, "_stream_ctx", stream_ctx)

    # ---- P1a doubles: the ExternalEvent nodes and the completion thread ----
    monkeypatch.setattr(kl, "_EXT_EVENTS", [])
    monkeypatch.setattr(kl, "_COMPLETER", None)
    monkeypatch.setattr(kl, "_new_external_event", lambda device: object())
    monkeypatch.setattr(kl, "_ext_wait", lambda ev, stream: s.enqueue(stream, Op("ext_wait", None, 0.0, ev)))
    monkeypatch.setattr(kl, "_ext_reset", lambda ev, stream: s.enqueue(stream, Op("ext_reset", None, 0.0, ev)))
    monkeypatch.setattr(
        kl, "_make_completer", lambda device: SimpleNamespace(start=lambda: None, submit=lambda *a: None)
    )

    def submit_only(payload):
        """``_kt_packed_submit_only``: build the task, hand it over, return.

        The node itself is cheap (5 against the serial node's 40); kt's 35 runs
        on the modelled completion thread, and the device is parked on the
        ExternalEvent until that thread signals it.
        """
        _wrapper, buf, layer_idx, event = payload
        kl.HOST_FUNC_CALLS[layer_idx] = kl.HOST_FUNC_CALLS.get(layer_idx, 0) + 1
        hidden_box, ids_box, out_box = buf["pin_hidden"], buf["pin_ids"], buf["pin_out"]

        def work():
            hidden, ids = hidden_box.value, ids_box.value
            out_box.value = None if hidden is None else 3 * hidden + (ids or 0)

        s.async_complete(work, 35.0, event)

    monkeypatch.setattr(kl, "_kt_packed_submit_only", submit_only)

    def host_func(stream, fn, payload):
        s.enqueue(stream, Op("hostfunc", lambda: fn(payload), 5.0 if fn is submit_only else 40.0))

    monkeypatch.setattr(kl, "_launch_host_func", host_func)

    fake_npu = SimpleNamespace(current_stream=lambda device=None: s.compute)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    s.capture_propagates = True
    return s


def _scalar(t):
    return float(t.reshape(-1)[0]) if isinstance(t, torch.Tensor) and t.numel() else 0.0


def _prop(s):
    """Whether the capture is visible on the stream the code is enqueueing to."""
    return s.current is s.compute or s.capture_propagates


def _runtime(monkeypatch, sim, overlap: bool, graph: bool, boxes, *, packed: bool = False,
             async_wait: bool = False):
    from vllm_ascend import envs
    from vllm_ascend.kt_offload import async_wait as aw

    # P1a is an env-only switch (the A/B it exists for must differ in exactly
    # one thing, and the cfg json is part of an arm's fingerprint), so it is
    # set here rather than on the config object.
    monkeypatch.setattr(envs, aw.ENV_NAME, "1" if async_wait else "0", raising=False)
    plan = build_plan([[0, 3]], NEXP)
    cfg = KTOffloadConfig(
        enabled=True,
        cpuinfer_threads=4,
        num_resident_experts=1,
        overlap_side_stream=overlap,
        packed_transfer=packed,
    )
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
    monkeypatch.setattr(rt, "_packed_buffers", lambda bs, device: _side_buffers_for(sim, rt, bs, boxes))
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3), requires_grad=False)
    rt.after_weights_loaded(layer)
    rt._ids_i64 = {bs: torch.zeros(bs, TOPK, dtype=torch.int64) for bs in (1, 2, 4, 8)}
    return rt, fake


def _side_buffers_for(sim, rt, bs, boxes):
    """Build (once per bs) the per-layer buffers, sharing the arm's value model."""
    buf = rt._packed.get(bs)
    if buf is None:
        device_in, _pin_in, _pin_out, out_gpu = boxes
        buf = make_side_buffers(sim, device_in, out_gpu, Box(0.0), Box(1.0))
        rt._packed[bs] = buf
    return buf


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
    pre-dispatch for every path, not only the forked one).

    It is cleared, and the layer still remembers that its side-stream work was
    never joined -- that is what ``_side_inflight`` is for, and the next
    captured submit refuses (see ``test_a_second_submit_before_the_join_is_refused``).
    """
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(9)
    rt.route(hs, w, ids, in_profile_run=False)
    assert rt._pending_join is not None
    sim.capturing = False
    rt.route(hs, w, ids, in_profile_run=False)  # eager, no merge in between
    assert rt._pending_join is None
    assert rt._side_inflight is True  # the un-joined submit is NOT forgotten


def test_captured_overlap_fills_first_then_forks_side_work_and_joins(monkeypatch, sim):
    boxes = _boxes()
    rt, fake = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(1)
    rt.route(hs, w, ids, in_profile_run=False)
    assert rt._pending_join is not None
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    assert rt._pending_join is None  # consumed
    # 🔴 the three fills come BEFORE the record: that is the edge that orders
    # the compute stream's writes against the side stream's reads.
    assert [o.kind for o in sim.programs["compute"]] == ["fill", "fill", "fill", "record", "wait", "h2d"]
    assert [o.kind for o in sim.programs["side"]] == ["wait", "d2h", "hostfunc", "record"]
    # the event objects the graph's nodes point at are kept for the process
    assert len(kl._SIDE_EVENTS) >= 1 and all(len(p) == 2 for p in kl._SIDE_EVENTS)
    assert len(kl._CAPTURED_PAYLOADS) >= 1 and kl._CAPTURED_PAYLOADS[-1][0] is fake
    # kt's shared pinned slots are not touched by the forked path at all
    assert fake.calls == []
    assert fake.ids_seen == []


def test_the_side_stream_reads_only_this_layers_own_buffers(monkeypatch, sim):
    """The criterion for the lifetime half, as a check the code can fail.

    Nothing the side stream reads may be storage the caller owns: the routing
    tensors' last reference goes when ``route()`` returns.
    """
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(21)
    rt.route(hs, w, ids, in_profile_run=False)
    buf = rt._packed[1]
    ours = {id(buf[k]) for k in ("dev", "pin", "pin_out", "dev_out")}
    theirs = {id(hs), id(ids), id(w)}
    assert not (ours & theirs)
    rep = ss.submit_order_ok(
        [("fill", "hidden"), ("fill", "ids"), ("fill", "weights"), ("fork", None), ("side_wait", None),
         ("side_read", "hidden"), ("side_read", "ids"), ("side_read", "weights")]
    )
    assert rep["ok"], rep


def test_capture_that_does_not_reach_the_side_stream_is_refused(monkeypatch, sim):
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    sim.capture_propagates = False
    hs, ids, w = _inputs(2)
    with pytest.raises(RuntimeError, match="did not propagate to the side stream"):
        rt.route(hs, w, ids, in_profile_run=False)


def test_a_second_submit_before_the_join_is_refused(monkeypatch, sim):
    """🔴 The guard that had to change: it must span steps, not one capture.

    ``_claim_side_slot`` counted layers in flight inside one capture and
    ``route()`` cleared ``_pending_join`` on the way in, so a layer that
    submitted and was never joined arrived at its next route indistinguishable
    from one that had been joined. Now the flag is cleared only by the join.
    """
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
    sim.capturing = True
    hs, ids, w = _inputs(3)
    rt.route(hs, w, ids, in_profile_run=False)
    with pytest.raises(AssertionError, match="submitting again while its previous"):
        rt.route(hs, w, ids, in_profile_run=False)  # a second capture, no merge in between


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


def test_packed_and_overlap_together_fork_rather_than_silently_serialise(monkeypatch, sim):
    """The combination the config used to refuse.

    The refusal existed because ``route()`` tested ``packed`` first, so "side
    stream on" quietly meant "packed serial". The branch order is the other way
    round now and both switches use the same per-layer buffers, so the
    combination is accepted -- and here is the assertion that it really forks.
    """
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes, packed=True)
    sim.capturing = True
    hs, ids, w = _inputs(6)
    rt.route(hs, w, ids, in_profile_run=False)
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    assert [o.kind for o in sim.programs["side"]] == ["wait", "d2h", "hostfunc", "record"]


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
    return {name: list(ops) for name, ops in sim.programs.items()}, rt


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
    serial, _ = _capture_schedule(monkeypatch, sim, serial_boxes, overlap=False)
    sim.reset_programs()
    ov_boxes = _boxes()
    overlap, rt = _capture_schedule(monkeypatch, sim, ov_boxes, overlap=True)
    if mutate is not None:
        overlap = mutate(overlap, ov_boxes, rt)
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
    assert t_o < t_s, (t_o, t_s)  # the point of the change; never the criterion on its own


def _drop_join(programs, boxes, rt):
    return {n: [o for o in ops if not (n == "compute" and o.kind == "wait")] for n, ops in programs.items()}


def _join_before_host_func(programs, boxes, rt):
    side = programs["side"]
    rec = [o for o in side if o.kind == "record"]
    rest = [o for o in side if o.kind != "record"]
    # the join event recorded right after the D2H, before the host func
    return {**programs, "side": rest[:2] + rec + rest[2:]}


def _events_not_kept(programs, boxes, rt):
    """Events collected: their record/wait nodes degrade into no-ops."""
    out = {}
    for n, ops in programs.items():
        out[n] = [Op("nop", lambda: None, 0.0) if o.kind in ("wait", "record") else o for o in ops]
    return out


def _capture_not_propagated(programs, boxes, rt):
    """The side stream's work never made it into the graph (sglang's F1).

    In that shape the D2H and the host func ran once, eagerly, while the graph
    was being captured: the pinned output keeps that run's value for ever.
    """
    device_in = boxes[0]
    buf = rt._packed[1]
    buf["pin_hidden"].value = device_in.value
    buf["pin_out"].value = 3 * device_in.value  # the capture run's CPU output, frozen
    return {n: ([] if n == "side" else [o for o in ops if o.kind != "wait"]) for n, ops in programs.items()}


def _early_fork(programs, boxes, rt):
    """🔴 The 2026-09-22 fix, taken out again: fork recorded before the fills.

    The compute stream's writes into this layer's own buffers are then ordered
    against nothing the side stream waits for, so the D2H (5 units) finishes
    before the first fill (10) and copies whatever the previous replay left.
    That is the shape measured on card at 1160/1200 layer-steps.
    """
    compute = programs["compute"]
    fills = [o for o in compute if o.kind == "fill"]
    rest = [o for o in compute if o.kind != "fill"]
    return {**programs, "compute": rest[:1] + fills + rest[1:]}


@pytest.mark.parametrize(
    "name, mutate",
    [
        ("join dropped", _drop_join),
        ("join recorded before the host func", _join_before_host_func),
        ("events not kept alive", _events_not_kept),
        ("capture not propagated to the side stream", _capture_not_propagated),
        ("fork recorded before the compute-stream fills", _early_fork),
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


def test_the_early_fork_mutant_reads_the_previous_replays_input(monkeypatch, sim):
    """Name the failure, not just "different": it is the PREVIOUS step's value.

    That is what the on-card probe reported (kt's pinned id buffer holding the
    previous step's routing ids in 1160 of 1200 layer-steps), so the CPU model
    has to produce the same fingerprint or it is modelling something else.
    """
    out_s, out_o, _, _ = _arms(monkeypatch, sim, _early_fork)
    # each overlap replay carries the value of the replay before it
    assert out_o[1:] == out_s[:-1], (out_s, out_o)


# ---------------------------------------------------------------------------
# 3. several layers, with the host running ahead
#
# kt keeps ONE pair of pinned buffers per batch size and hands layer L the slot
# ``L % 2``: the pinned output of layer L is read by an ASYNC H2D on the compute
# stream while the CPU task of layer L+2 writes that same buffer. In the serial
# path kt's own submit_forward calls torch.npu.synchronize() before submitting,
# which is what supplies the ordering. The forked path used to inherit that
# sharing without the sync; since 2026-09-22 it has per-layer buffers instead,
# and these tests are what says the sharing is really gone.
#
# Everything above this line is single-layer, i.e. zero coverage for reuse
# across layers: with fewer than three layers no two layers share a slot, and
# with the device drained at every step the host can never run ahead.
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
        self.cpu_infer = SimpleNamespace(submit=lambda task: None, sync=lambda allow: None)
        self.moe = SimpleNamespace(forward_task=lambda *a: None)

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

    # -- captured, non-forked path
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


def _layer_runtime(monkeypatch, sim, buffers, layer_idx, device_in, *, overlap, graph, outs=None):
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

    def packed_buffers(bs, device, _rt=rt, _dev_in=device_in):
        buf = _rt._packed.get(bs)
        if buf is None:
            out = torch.zeros(1)
            buf = make_side_buffers(sim, _dev_in, out, Box(0.0), Box(1.0))
            _rt._packed[bs] = buf
            if outs is not None:
                outs.setdefault(layer_idx, {})[bs] = out
        return buf

    monkeypatch.setattr(rt, "_packed_buffers", packed_buffers)
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
    outs: dict = {}
    rts = [
        _layer_runtime(monkeypatch, sim, buffers, i, ins[i], overlap=overlap, graph=graph, outs=outs)
        for i in range(layers)
    ]
    seen: list[tuple[float, float]] = []
    forked = overlap and graph
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
            gpu = outs[i][bs] if forked else buffers.get(bs)["gpu"][i % DEPTH]
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
    """Batch size alternates (concurrent requests): a new bs means new buffers,
    and the old ones must not be written while an H2D still reads them."""
    seen = _run_model(monkeypatch, sim, overlap=True, graph=False, batch_sizes=(1, 2, 1))
    wrong = [(got, want) for got, want in seen if got != want]
    assert not wrong, f"{len(wrong)}/{len(seen)} wrong with a changing batch size: {wrong[:4]}"


def test_no_two_layers_share_a_side_buffer(monkeypatch, sim):
    """A6 after the change: the forked path's stores belong to one layer each."""
    buffers = KtBuffers()
    ins = [Box(1.0) for _ in range(4)]
    rts = [_layer_runtime(monkeypatch, sim, buffers, i, ins[i], overlap=True, graph=True) for i in range(4)]
    sim.capturing = True
    hs = torch.zeros(1, H, dtype=torch.bfloat16)
    ids = torch.zeros(1, TOPK, dtype=torch.int32)
    w = torch.ones(1, TOPK)
    keys = []
    for rt in rts:
        rt.route(hs, w, ids, in_profile_run=False)
        buf = rt._packed[1]
        keys.append({id(buf[k]) for k in ("dev", "pin", "pin_out", "dev_out")})
        rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert not (keys[i] & keys[j]), f"layers {i} and {j} share a side buffer"
    rep = ss.buffer_ownership_ok([(i, keys[i], False) for i in range(len(keys))])
    assert rep["ok"], rep
    # and the criterion goes red when they do share one
    assert not ss.buffer_ownership_ok([(0, keys[0], False), (2, keys[0], False)])["ok"]


def test_a_layer_reusing_another_layers_buffer_is_refused(monkeypatch, sim):
    """The mutant for the ownership half: hand layer 2 layer 0's buffers."""
    buffers = KtBuffers()
    ins = [Box(1.0) for _ in range(4)]
    rts = [_layer_runtime(monkeypatch, sim, buffers, i, ins[i], overlap=True, graph=True) for i in range(4)]
    sim.capturing = True
    hs = torch.zeros(1, H, dtype=torch.bfloat16)
    ids = torch.zeros(1, TOPK, dtype=torch.int32)
    w = torch.ones(1, TOPK)
    rts[0].route(hs, w, ids, in_profile_run=False)
    rts[0].merge(torch.zeros(1, H, dtype=torch.bfloat16))
    rts[2]._packed[1] = rts[0]._packed[1]  # kt's layer_idx % 2 sharing, by hand
    with pytest.raises(AssertionError, match="share the"):
        rts[2].route(hs, w, ids, in_profile_run=False)


def test_the_normal_layer_order_never_trips_the_buffer_guard(monkeypatch, sim):
    """Positive control for the guard above: four layers, three steps, no raise."""
    seen = _run_model(monkeypatch, sim, overlap=True, graph=True)
    assert seen and all(got == want for got, want in seen)
    # every layer released: one owner per store, nothing left in flight
    assert len(set(kl._SIDE_BUFFER_OWNER.values())) == 4


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


def _preconditions_cfg(packed: bool = False):
    cfg = KTOffloadConfig(
        enabled=True,
        cpuinfer_threads=8,
        num_resident_experts=100,
        overlap_side_stream=True,
        packed_transfer=packed,
    )
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


def test_packed_and_the_side_stream_together_are_accepted(monkeypatch):
    """Both on is a configuration now, not a refusal (see the test in §1).

    It was refused while ``route()`` took the packed branch first, which made
    "side stream on" mean "packed serial". The forked path *is* the packed
    buffers now and is taken first.
    """
    from vllm_ascend import envs

    monkeypatch.setattr(envs, ss.ENV_NAME, "1", raising=False)
    cfg, vcfg = _preconditions_cfg(packed=True)
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
# 5. the runtime gates the side stream carries (each with its mutant)
# ---------------------------------------------------------------------------


def test_submit_order_criterion_goes_red_on_a_fill_after_the_fork(monkeypatch):
    """The pure form of the fix, so it can be read without a simulator."""
    good = [("fill", "ids"), ("fork", None), ("side_wait", None), ("side_read", "ids")]
    assert ss.submit_order_ok(good)["ok"]
    bad = [("fork", None), ("fill", "ids"), ("side_wait", None), ("side_read", "ids")]
    rep = ss.submit_order_ok(bad)
    assert not rep["ok"] and any("AFTER fork.record" in p for p in rep["detail"]), rep
    never = [("fork", None), ("side_wait", None), ("side_read", "ids")]
    assert not ss.submit_order_ok(never)["ok"]


def test_buffer_ownership_criterion_spans_steps():
    """``in flight`` is a property of the layer, not of one capture."""
    steps = [(0, {1}, False), (1, {2}, False), (0, {1}, False), (1, {2}, False)]
    assert ss.buffer_ownership_ok(steps)["ok"]
    rep = ss.buffer_ownership_ok([(0, {1}, False), (0, {1}, True)])
    assert not rep["ok"] and any("un-joined" in p for p in rep["detail"]), rep
    assert not ss.buffer_ownership_ok([(0, set(), False)])["ok"]  # no buffers of its own


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


# ---------------------------------------------------------------------------
# 6. the diagnostic instrument must not be installed twice on one call
# ---------------------------------------------------------------------------


def test_seam_timing_does_not_also_probe_cpu_infer_on_the_forked_path(monkeypatch, sim):
    """🔴 Two instruments on one call read like one.

    ``seam_timing``'s CPUInfer probe exists to split submit from sync on the
    path where *kt* makes those calls. Since the forked path runs
    ``_kt_packed_forward`` on this layer's own buffers, it stamps submit/sync
    itself -- and the probe's ``T_SUBMIT_IN``, taken inside its own frame, then
    overwrote ours, moving time out of ``submit`` and into ``kt_pre``, while the
    extra frame lengthened the host-func node the side stream exists to hide.
    So: split is on (the node does it), and ``cpu_infer`` is left alone.
    """
    from vllm_ascend.kt_offload import seam_timing as sm

    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_SEAM_TIMING", "60")
    sm._reset_for_tests()
    try:
        boxes = _boxes()
        rt, fake = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes)
        assert sm.RECORDER is not None and sm.RECORDER.split is True
        assert not isinstance(fake.cpu_infer, sm._TimedCpuInfer), (
            "the CPUInfer probe was installed on top of a node that already times itself"
        )
        # ...and the serial captured path, where kt makes the calls, still gets it
        sm._reset_for_tests()
        boxes2 = _boxes()
        _rt2, fake2 = _runtime(monkeypatch, sim, overlap=False, graph=True, boxes=boxes2)
        assert isinstance(fake2.cpu_infer, sm._TimedCpuInfer), (
            "the serial path lost the probe that splits kt's submit from its sync"
        )
    finally:
        sm._reset_for_tests()


# ---------------------------------------------------------------------------
# 5. P1a: the device waits for an ExternalEvent, not for the callback's return
#
# The arm under test is the one that will be merged: side stream ON in both
# arms, and only VLLM_ASCEND_DSV41_KT_ASYNC_WAIT differs. The simulator has to
# model two things the side-stream model did not need:
#
#   * the release comes from a thread that runs AFTER the node returned, so
#     the node no longer contains kt's compute (``Sim.async_complete``);
#   * an external event is a value in device memory and stays signalled until
#     the graph's ``reset`` node clears it (``Sim.ext_signaled``, which is the
#     one piece of state ``run()`` does not rebuild per replay).
#
# 🔴 All three mutants below are the same failure the on-card cases look for:
# a release that happens before this step's output exists. Two of them are
# faster and wrong; the third is a hang, which is the only one a user would
# notice on their own.
# ---------------------------------------------------------------------------


def _capture_p1a(monkeypatch, sim, boxes, *, async_wait: bool, overlap: bool = True, packed: bool = False):
    rt, _ = _runtime(monkeypatch, sim, overlap=overlap, graph=True, boxes=boxes, packed=packed,
                     async_wait=async_wait)
    sim.capturing = True
    hs, ids, w = _inputs(7)
    rt.route(hs, w, ids, in_profile_run=False)
    sim.enqueue(sim.compute, Op("npu_experts", lambda: None, 30.0))
    rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    sim.capturing = False
    return {name: list(ops) for name, ops in sim.programs.items()}, rt


def _p1a_arms(monkeypatch, sim, *, mode="clean", mutate=None, values=(2.0, 5.0, 9.0)):
    """Side stream + P1a off vs side stream + P1a on, replayed on the same inputs."""
    ref_boxes = _boxes()
    ref, _ = _capture_p1a(monkeypatch, sim, ref_boxes, async_wait=False)
    sim.reset_programs()
    p1a_boxes = _boxes()
    sim.async_mode = mode
    p1a, rt = _capture_p1a(monkeypatch, sim, p1a_boxes, async_wait=True)
    if mutate is not None:
        p1a = mutate(p1a, p1a_boxes, rt)
    out_r, out_p, t_r, t_p = [], [], [], []
    for v in values:
        sim.async_mode = "clean"
        y, t = _replay(sim, ref, ref_boxes[0], ref_boxes[3], v)
        out_r.append(y)
        t_r.append(t)
        sim.async_mode = mode
        y, t = _replay(sim, p1a, p1a_boxes[0], p1a_boxes[3], v)
        out_p.append(y)
        t_p.append(t)
    return out_r, out_p, max(t_r), max(t_p)


def test_p1a_records_the_wait_and_the_reset_behind_the_node(monkeypatch, sim):
    """The shape, on the stream the node is on, with the reset present.

    The reset has no second line of defence at run time: without it the graph
    still works and gets faster. This asserts the node is emitted; the next
    test asserts that removing it is caught by the output.
    """
    boxes = _boxes()
    programs, _ = _capture_p1a(monkeypatch, sim, boxes, async_wait=True)
    assert [o.kind for o in programs["side"]] == [
        "wait", "d2h", "hostfunc", "ext_wait", "ext_reset", "record",
    ]
    assert [o.kind for o in programs["compute"]] == [
        "fill", "fill", "fill", "record", "npu_experts", "wait", "h2d",
    ]
    # one object for both nodes, and it is kept alive for the process
    side = programs["side"]
    assert side[3].event is side[4].event
    assert kl._EXT_EVENTS and side[3].event in kl._EXT_EVENTS


def test_p1a_on_the_packed_path_without_the_side_stream(monkeypatch, sim):
    """It combines with the side stream, and it also stands alone."""
    boxes = _boxes()
    programs, _ = _capture_p1a(monkeypatch, sim, boxes, async_wait=True, overlap=False, packed=True)
    assert "side" not in programs
    assert [o.kind for o in programs["compute"]] == [
        "fill", "fill", "fill", "d2h", "hostfunc", "ext_wait", "ext_reset", "npu_experts", "h2d",
    ]


def test_p1a_is_bit_identical_and_not_slower(monkeypatch, sim):
    out_r, out_p, t_r, t_p = _p1a_arms(monkeypatch, sim)
    assert _bits_equal(out_r, out_p), (out_r, out_p)
    assert t_p <= t_r, (t_p, t_r)  # the point of the change; never the criterion on its own


def test_p1a_without_the_record_hangs(monkeypatch, sim):
    """🔴 Mutant ①: the completion thread never records.

    This is the arm that says the device really is waiting. If the graph
    finished anyway, the ``ext_wait`` node would not be doing anything and
    every other result here would be meaningless.
    """
    with pytest.raises(AssertionError, match="deadlock"):
        _p1a_arms(monkeypatch, sim, mode="no-record")


def test_p1a_recording_before_the_work_is_faster_and_wrong(monkeypatch, sim):
    """🔴 Mutant ②: released before this step's output exists.

    The main failure mode of the whole scheme, and the reason the on-card
    criterion is bit-identity rather than TPOT.
    """
    out_r, out_p, t_r, t_p = _p1a_arms(monkeypatch, sim, mode="early-record")
    assert not _bits_equal(out_r, out_p), (out_r, out_p)
    assert t_p <= t_r, (t_p, t_r)


def test_p1a_recording_early_reads_the_previous_replays_output(monkeypatch, sim):
    """Name the failure: it is the PREVIOUS step's CPU output, as on card."""
    out_r, out_p, _, _ = _p1a_arms(monkeypatch, sim, mode="early-record")
    assert out_p[1:] == out_r[:-1], (out_r, out_p)


def _drop_reset(programs, boxes, rt):
    """🔴 Mutant ③: the reset node never made it into the graph.

    The first replay is fine -- the event starts clear -- and every replay
    after it walks straight through a wait that is still signalled from the
    replay before. That is why one replay proves nothing here.

    🔴 The card disagrees with this model, and the disagreement is recorded on
    purpose (2026-09-23, t5r ``--mutant no-reset --stall-ms 50``): with all 8
    reset nodes provably left out the on-card arm stayed bit-identical AND
    still paid the stall on every replay, i.e. the captured ExternalEvent wait
    re-arms itself on that build. This model follows what torch_npu documents
    ("Before reusing ExternalEvent, you need to call reset() to clear the
    flag", streams.py:214-217), which is the more pessimistic of the two, and
    the production path keeps the reset for the same reason -- 0.144 us
    against relying on undocumented behaviour. What this file must NOT be read
    as saying is that a missing reset would show up on card: it does not, and
    the on-card arm that does go red is ``--mutant no-wait``.
    """
    return {n: [o for o in ops if o.kind != "ext_reset"] for n, ops in programs.items()}


def test_p1a_without_the_reset_in_the_graph_is_faster_and_wrong(monkeypatch, sim):
    out_r, out_p, t_r, t_p = _p1a_arms(monkeypatch, sim, mutate=_drop_reset)
    assert not _bits_equal(out_r, out_p), (out_r, out_p)
    assert t_p <= t_r, (t_p, t_r)


def test_p1a_without_the_reset_is_right_once_and_wrong_after(monkeypatch, sim):
    """The fingerprint of a missing reset, and the reason a one-replay check
    cannot see it: replay 0 is correct, replay 1 onwards is the previous one."""
    out_r, out_p, _, _ = _p1a_arms(monkeypatch, sim, mutate=_drop_reset)
    assert out_p[0] == out_r[0]
    assert out_p[1:] == out_r[:-1], (out_r, out_p)


def test_the_criterion_catches_the_missing_reset_in_the_recorded_ops(monkeypatch, sim):
    """The same mutant, seen by the check the production path itself runs.

    ``_packed_output`` evaluates ``ext_event_ops_ok`` on the ops it recorded,
    so a future edit that drops the reset raises at capture instead of serving
    stale output.
    """
    from vllm_ascend.kt_offload import async_wait as aw

    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=True, boxes=boxes, async_wait=True)
    sim.capturing = True
    hs, ids, w = _inputs(3)
    rt.route(hs, w, ids, in_profile_run=False)
    assert [k for k, _ in rt._submit_ops] == ["d2h", "hostfunc", "ext_wait", "ext_reset"]
    rt._submit_ops = [o for o in rt._submit_ops if o[0] != "ext_reset"]
    with pytest.raises(RuntimeError, match="recorded the wrong node sequence"):
        rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    assert not aw.ext_event_ops_ok([("d2h", 1), ("hostfunc", 0), ("ext_wait", EV_SENTINEL)])["ok"]


EV_SENTINEL = object()


def test_p1a_off_records_no_external_event_nodes(monkeypatch, sim):
    """With the switch off the schedule is the one that was verified on card."""
    boxes = _boxes()
    programs, rt = _capture_p1a(monkeypatch, sim, boxes, async_wait=False)
    assert [o.kind for o in programs["side"]] == ["wait", "d2h", "hostfunc", "record"]
    assert kl._EXT_EVENTS == [] and rt._ext_events == {}


def test_p1a_needs_our_own_buffers(monkeypatch, sim):
    """kt's own captured entry point does submit+sync in one call."""
    boxes = _boxes()
    with pytest.raises(RuntimeError, match="own pinned buffers"):
        _runtime(monkeypatch, sim, overlap=False, graph=True, boxes=boxes, packed=False, async_wait=True)


def test_p1a_is_a_no_op_in_eager_mode(monkeypatch, sim):
    """Eager already blocks the host inside route(); there is nothing to defer."""
    boxes = _boxes()
    rt, _ = _runtime(monkeypatch, sim, overlap=True, graph=False, boxes=boxes, async_wait=True)
    assert rt.async_wait is False


def test_every_replay_records_once_per_layer(monkeypatch, sim):
    """The submit counter and the release counter have to agree.

    ``HOST_FUNC_CALLS`` counts submits and ``async_wait.RECORDS`` counts
    releases; a run where the first grows and the second does not is a device
    about to wait for ever, which is a different symptom from a wrong answer
    and worth naming separately.
    """
    from vllm_ascend.kt_offload import async_wait as aw

    boxes = _boxes()
    sim.async_mode = "clean"
    programs, _ = _capture_p1a(monkeypatch, sim, boxes, async_wait=True)
    kl.HOST_FUNC_CALLS.clear()
    records: dict[int, int] = {}
    # the double for the real completer: count what it would record
    for k in range(3):
        _replay(sim, programs, boxes[0], boxes[3], float(k + 2))
        records[0] = records.get(0, 0) + 1
    assert aw.record_counts_ok({}, records, 3, [0])["ok"]
    assert aw.submits_match_records(dict(kl.HOST_FUNC_CALLS), records)["ok"]
