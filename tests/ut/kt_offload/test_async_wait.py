# SPDX-License-Identifier: Apache-2.0
"""P1a (``VLLM_ASCEND_DSV41_KT_ASYNC_WAIT``): the criteria and the thread.

The ordering model -- what the device waits on, and what happens when the
release comes too early or not at all -- lives in
``test_side_stream_overlap.py``, because that is where the stream simulator
that can *execute* a schedule is. This file covers the two halves that need no
simulator:

* the criterion ``ext_event_ops_ok``, driven red once per way the node
  sequence can be wrong -- above all the 🔴 missing ``reset``, which is the
  only failure with no second line of defence: the graph keeps working, gets
  faster, and every replay after the first reads the previous step's CPU
  output;
* ``Completer``, the thread that syncs kt and records the event. Its contract
  is short and every clause of it is a way to wedge or corrupt a service:
  sync BEFORE record, record on EVERY path including a raise, and a context
  failure at start-up rather than a hang at the first replay.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from vllm_ascend.kt_offload import async_wait as aw


@pytest.fixture(autouse=True)
def _clean_counters():
    aw.RECORDS.clear()
    aw.ERRORS.clear()
    yield
    aw.RECORDS.clear()
    aw.ERRORS.clear()


# ---------------------------------------------------------------------------
# 1. the node-sequence criterion
# ---------------------------------------------------------------------------

EV = object()


def _good_ops():
    """What ``_launch_cpu_node`` + ``_packed_output`` record with P1a on."""
    return [("d2h", 1), ("hostfunc", 3), ("ext_wait", EV), ("ext_reset", EV), ("h2d", 1)]


def test_the_production_sequence_passes():
    rep = aw.ext_event_ops_ok(_good_ops())
    assert rep["ok"], rep["detail"]


def test_a_missing_reset_is_caught():
    """🔴 The one that is faster, wrong and silent on card."""
    ops = [o for o in _good_ops() if o[0] != "ext_reset"]
    rep = aw.ext_event_ops_ok(ops)
    assert not rep["ok"]
    assert any("reset" in p for p in rep["detail"]), rep["detail"]


def test_a_missing_wait_is_caught():
    ops = [o for o in _good_ops() if o[0] != "ext_wait"]
    assert not aw.ext_event_ops_ok(ops)["ok"]


def test_reset_before_wait_is_caught():
    ops = [("d2h", 1), ("hostfunc", 3), ("ext_reset", EV), ("ext_wait", EV), ("h2d", 1)]
    rep = aw.ext_event_ops_ok(ops)
    assert not rep["ok"]
    assert any("reset before it is waited on" in p for p in rep["detail"]), rep["detail"]


def test_waiting_before_the_submit_is_caught():
    """A wait ahead of the host-func node deadlocks: nothing has been handed over."""
    ops = [("d2h", 1), ("ext_wait", EV), ("ext_reset", EV), ("hostfunc", 3), ("h2d", 1)]
    rep = aw.ext_event_ops_ok(ops)
    assert not rep["ok"]
    assert any("BEFORE the node that submits" in p for p in rep["detail"]), rep["detail"]


def test_the_output_copy_must_be_behind_the_wait():
    ops = [("d2h", 1), ("hostfunc", 3), ("h2d", 1), ("ext_wait", EV), ("ext_reset", EV)]
    rep = aw.ext_event_ops_ok(ops)
    assert not rep["ok"]
    assert any("copied back before the wait" in p for p in rep["detail"]), rep["detail"]


def test_a_host_record_inside_the_graph_is_caught():
    """The record belongs to the completion thread at replay time, not to the graph."""
    ops = [("d2h", 1), ("hostfunc", 3), ("host_record", EV), ("ext_wait", EV), ("ext_reset", EV), ("h2d", 1)]
    rep = aw.ext_event_ops_ok(ops)
    assert not rep["ok"]
    assert any("host-side event.record" in p for p in rep["detail"]), rep["detail"]


def test_waiting_on_one_event_and_resetting_another_is_caught():
    other = object()
    ops = [("d2h", 1), ("hostfunc", 3), ("ext_wait", EV), ("ext_reset", other), ("h2d", 1)]
    assert not aw.ext_event_ops_ok(ops)["ok"]


def test_the_d2h_must_precede_the_submit():
    ops = [("hostfunc", 3), ("d2h", 1), ("ext_wait", EV), ("ext_reset", EV), ("h2d", 1)]
    assert not aw.ext_event_ops_ok(ops)["ok"]


def test_record_counts_and_submit_pairing():
    before = {0: 0, 1: 0}
    after = {0: 3, 1: 2}
    rep = aw.record_counts_ok(before, after, 3, [0, 1])
    assert not rep["ok"] and rep["detail"]["wrong"] == {1: 2}
    assert aw.record_counts_ok(before, {0: 3, 1: 3}, 3, [0, 1])["ok"]
    assert aw.submits_match_records({0: 3}, {0: 3})["ok"]
    bad = aw.submits_match_records({0: 3}, {0: 2})
    assert not bad["ok"] and bad["detail"] == {0: (3, 2)}


# ---------------------------------------------------------------------------
# 2. the completion thread
# ---------------------------------------------------------------------------


class FakeEvent:
    def __init__(self, log, name="ev"):
        self.log, self.name = log, name
        self.records = 0
        self.stream = None

    def record(self, stream):
        self.records += 1
        self.stream = stream
        self.log.append(("record", self.name))


def _wrapper(log, *, raises=None, name="L0"):
    def sync(allowance):
        log.append(("sync", name, allowance))
        if raises is not None:
            raise raises

    return SimpleNamespace(cpu_infer=SimpleNamespace(sync=sync))


def _completer(log, *, device_error=None, stream="S"):
    def set_device():
        if device_error is not None:
            raise device_error
        log.append(("set_device", threading.current_thread().name))

    def make_stream():
        log.append(("make_stream",))
        return stream

    return aw.Completer(set_device, make_stream, name="test-completer")


def _drain(c, timeout=10.0):
    """Wait until the thread has nothing outstanding (tests only)."""
    done = threading.Event()
    deadline = threading.Timer(timeout, done.set)
    deadline.start()
    try:
        while c.depth > 0 and not done.is_set():
            threading.Event().wait(0.002)
    finally:
        deadline.cancel()
    assert c.depth == 0, "the completion thread never finished its queue"


def test_it_syncs_first_and_records_after():
    """The order IS the correctness property: a record before the sync releases
    the device while kt is still writing the pinned output."""
    log: list = []
    c = _completer(log)
    c.start()
    try:
        ev = FakeEvent(log)
        c.submit(_wrapper(log), 7, ev)
        _drain(c)
    finally:
        c.stop()
    kinds = [e[0] for e in log if e[0] in ("sync", "record")]
    assert kinds == ["sync", "record"], log
    assert ev.records == 1 and ev.stream == "S"
    assert aw.RECORDS == {7: 1}
    assert aw.ERRORS == []


def test_the_thread_takes_its_own_acl_context_before_making_a_stream():
    """🔴 The unprecedented part: the record comes from a thread that is neither
    the main thread nor ACL's report thread, so it needs a context of its own.
    The stream is created IN the thread, after that, and before start() returns
    -- a capture must never be the first thing that needs it."""
    log: list = []
    c = _completer(log)
    c.start()
    try:
        assert [e[0] for e in log] == ["set_device", "make_stream"], log
        assert log[0][1] == "test-completer"  # the context belongs to the completer thread
    finally:
        c.stop()


def test_a_context_failure_is_a_start_up_error_not_a_hang():
    log: list = []
    c = _completer(log, device_error=RuntimeError("no context"))
    with pytest.raises(RuntimeError, match="could not take an ACL context"):
        c.start()
    c.stop()


def test_a_task_that_raises_still_releases_the_device():
    """A device left waiting wedges the engine with no message; stale output at
    least says so in the log, and is the failure the serial node already has."""
    log: list = []
    c = _completer(log)
    c.start()
    try:
        ev = FakeEvent(log)
        c.submit(_wrapper(log, raises=RuntimeError("kt blew up")), 4, ev)
        _drain(c)
    finally:
        c.stop()
    assert ev.records == 1
    assert aw.RECORDS == {4: 1}
    assert len(aw.ERRORS) == 1 and aw.ERRORS[0][0] == 4 and "kt blew up" in aw.ERRORS[0][1]


def test_a_failed_submit_hands_over_nothing_to_wait_for():
    """``wrapper=None`` = "the callback never got as far as submitting"."""
    log: list = []
    c = _completer(log)
    c.start()
    try:
        ev = FakeEvent(log)
        c.submit(None, 2, ev)
        _drain(c)
    finally:
        c.stop()
    assert [e[0] for e in log if e[0] in ("sync", "record")] == ["record"]
    assert aw.RECORDS == {2: 1} and aw.ERRORS == []


def test_a_record_that_fails_is_logged_as_critical_and_counted():
    log: list = []
    c = _completer(log)
    c.start()

    class Broken(FakeEvent):
        def record(self, stream):
            raise RuntimeError("aclrtRecordEvent failed")

    try:
        c.submit(_wrapper(log), 5, Broken(log))
        _drain(c)
    finally:
        c.stop()
    assert aw.RECORDS == {}  # nothing was released: the replay will not finish
    assert len(aw.ERRORS) == 1 and "record failed" in aw.ERRORS[0][1]


def test_submits_are_completed_in_order():
    """kt's queue is one serial FIFO worker and the graph keeps one task of ours
    outstanding, so ``sync(0)`` means "this layer's task". Out-of-order
    completion would make that reasoning false."""
    log: list = []
    c = _completer(log)
    c.start()
    try:
        events = [FakeEvent(log, f"ev{i}") for i in range(6)]
        for i, ev in enumerate(events):
            c.submit(_wrapper(log, name=f"L{i}"), i, ev)
        _drain(c)
    finally:
        c.stop()
    seen = [e for e in log if e[0] in ("sync", "record")]
    assert [e[1] for e in seen] == [x for i in range(6) for x in (f"L{i}", f"ev{i}")], seen
    assert aw.RECORDS == {i: 1 for i in range(6)}


def test_submit_never_blocks_the_callback():
    """The whole point: the graph node returns without waiting for kt."""
    log: list = []
    c = _completer(log)
    c.start()
    try:
        gate = threading.Event()

        def slow(allowance):
            gate.wait(5.0)

        c.submit(SimpleNamespace(cpu_infer=SimpleNamespace(sync=slow)), 0, FakeEvent(log))
        # depth is incremented by submit and only decremented by the thread:
        # if submit had waited, this would already be 0.
        assert c.depth == 1
        gate.set()
        _drain(c)
    finally:
        c.stop()


# ---------------------------------------------------------------------------
# 3. the switch
# ---------------------------------------------------------------------------


def test_the_env_opens_it(monkeypatch):
    from vllm_ascend import envs

    monkeypatch.setattr(envs, aw.ENV_NAME, "1", raising=False)
    assert aw.enabled_by_env() is True


@pytest.mark.parametrize("value", ["0", "", "true", "True", "yes", "on", " 1"])
def test_anything_but_one_keeps_it_shut(monkeypatch, value):
    from vllm_ascend import envs

    monkeypatch.setattr(envs, aw.ENV_NAME, value, raising=False)
    assert aw.enabled_by_env() is False


def _vllm_config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1, enable_eplb=False
        ),
        speculative_config=None,
    )


def test_it_is_refused_without_our_own_buffers(monkeypatch):
    """kt's own captured entry point does submit+sync in one call, and its
    pinned buffers are shared by ``layer_idx % 2`` -- the very sharing a
    device-side wait would stop ordering."""
    from vllm_ascend import envs
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    monkeypatch.setattr(envs, aw.ENV_NAME, "1", raising=False)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1)
    with pytest.raises(ValueError, match="packed_transfer or"):
        cfg._validate_preconditions(_vllm_config())


@pytest.mark.parametrize("knob", ["packed_transfer", "overlap_side_stream"])
def test_either_own_buffer_path_is_enough(monkeypatch, knob):
    from vllm_ascend import envs
    from vllm_ascend.kt_offload import side_stream
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    monkeypatch.setattr(envs, aw.ENV_NAME, "1", raising=False)
    monkeypatch.setattr(envs, side_stream.ENV_NAME, "1", raising=False)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, **{knob: True})
    cfg._validate_preconditions(_vllm_config())  # does not raise


def test_with_the_env_off_the_precondition_does_not_apply(monkeypatch):
    from vllm_ascend import envs
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    monkeypatch.setattr(envs, aw.ENV_NAME, "0", raising=False)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1)
    cfg._validate_preconditions(_vllm_config())
