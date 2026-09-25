# SPDX-License-Identifier: Apache-2.0
"""Seam timing for the host-func callback -- CPU only, fake kt, real clocks.

The fake wrapper's sleeps are known up front, so each segment is checked
against the sleep it was given rather than against the instrument itself; a
mutation arm moves one timestamp and requires the reconciliation to go red."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload import seam_timing as sm





def _split_parts(row: list[int]) -> int:
    """The segments that must add up to ``cb``, in nanoseconds, for one row."""
    return sum(row[sm.SEGMENTS.index(name)] for name in sm.ADDITIVE)


def _spin(us: float) -> None:
    """Busy-wait: ``sleep`` on a loaded box overshoots by milliseconds, which
    would make the per-segment assertions test the scheduler, not the probe."""
    end = time.perf_counter_ns() + int(us * 1000)
    while time.perf_counter_ns() < end:
        pass


class FakeInfer:
    """kt's CPUInfer: a submit that enqueues and a sync that waits."""

    def __init__(self, submit_us=0.0, sync_us=0.0):
        self.submit_us, self.sync_us = submit_us, sync_us
        self.calls: list[str] = []

    def submit(self, task):
        self.calls.append(f"submit:{task}")
        _spin(self.submit_us)
        return "submitted"

    def sync(self, allow_pending):
        self.calls.append(f"sync:{allow_pending}")
        _spin(self.sync_us)
        return "synced"


class FakeWrapper:
    """Enough of kt for the callback: the pending flags and the two phases.

    ``run_pinned_forward_sync`` reaches the queue through ``self.cpu_infer``,
    the shape the submit/sync probe relies on; the fake keeps it exactly."""

    def __init__(self, *, pre_us=0.0, submit_us=0.0, between_us=0.0, sync_us=0.0, tail_us=0.0):
        self.cpu_infer = FakeInfer(submit_us, sync_us)
        self.pre_us, self.between_us, self.tail_us = pre_us, between_us, tail_us
        self.outputs: list[tuple] = []
        self.pending: dict[int, bool] = {}
        self.layer_idx = 0

    def has_pending_deferred(self, layer_idx):
        return bool(self.pending.get(layer_idx, False))

    def run_pinned_forward_sync(self, shape_ref, stream_handle):
        _spin(self.pre_us)  # _check_qlen_fits_cpp_buffers + get_buffer + task build
        self.cpu_infer.submit("task")
        _spin(self.between_us)  # kt's deferred bookkeeping
        self.cpu_infer.sync(0)
        _spin(self.tail_us)
        self.outputs.append((shape_ref, stream_handle))


@pytest.fixture(autouse=True)
def clean_recorder():
    sm._reset_for_tests()
    kl.HOST_FUNC_CALLS.clear()
    yield
    sm._reset_for_tests()
    kl.HOST_FUNC_CALLS.clear()


def _enable(monkeypatch, num_layers=2, wrapper=None, interval="60"):
    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_SEAM_TIMING", interval)
    return sm.maybe_enable(num_layers, wrapper)


def test_off_records_nothing_and_leaves_the_callback_identical(monkeypatch):
    """The default: no recorder, no probe, and the same call sequence."""
    monkeypatch.delenv("VLLM_ASCEND_DSV41_KT_SEAM_TIMING", raising=False)
    assert sm.maybe_enable(4, FakeWrapper()) is None
    assert sm.RECORDER is None

    w = FakeWrapper()
    kl._kt_host_forward((w, "shape", 77, 1))
    assert w.outputs == [("shape", 77)]  # same arguments, same order
    assert w.cpu_infer.calls == ["submit:task", "sync:0"]
    assert kl.HOST_FUNC_CALLS == {1: 1}
    assert sm.TS == [0] * 8  # not one clock read stored
    assert not isinstance(w.cpu_infer, sm._TimedCpuInfer)


def test_on_does_not_change_what_the_callback_does(monkeypatch):
    """Bit for bit: the fake sees the same arguments and the same calls."""
    off = FakeWrapper()
    kl._kt_host_forward((off, "shape", 77, 1))

    on = FakeWrapper()
    _enable(monkeypatch, 2, on)
    kl._kt_host_forward((on, "shape", 77, 1))

    assert on.outputs == off.outputs
    assert on.cpu_infer._kt_seam_inner.calls == off.cpu_infer.calls
    assert kl.HOST_FUNC_CALLS == {1: 2}


def test_every_segment_lands_where_its_sleep_was(monkeypatch):
    w = FakeWrapper(pre_us=300, submit_us=200, between_us=400, sync_us=1500, tail_us=250)
    rec = _enable(monkeypatch, 2, w)
    assert rec.split, "the CPUInfer probe is what separates submit from sync"
    for _ in range(5):
        kl._kt_host_forward((w, "shape", 7, 1))

    rows, _ = rec.drain([0, 0])
    summary = sm.summarize(rows, rec.split, 1.0)
    med = summary["layers"]["1"]["med_us"]
    for name, want in (("kt_pre", 300), ("submit", 200), ("between", 400), ("sync", 1500), ("kt_tail", 250)):
        assert want * 0.8 <= med[name] <= want * 1.2 + 50, f"{name}={med[name]} us, expected ~{want}"
    assert med["pre"] < 50 and med["post"] < 50  # our own Python, not kt's
    assert 2400 < med["cb"] < 3200
    assert sm.reconcile(med, rec.split)["ok"]


def test_the_gap_is_the_interval_between_two_callbacks_of_one_layer(monkeypatch):
    w = FakeWrapper()
    rec = _enable(monkeypatch, 2, w)
    kl._kt_host_forward((w, "s", 1, 0))
    _spin(2000)
    kl._kt_host_forward((w, "s", 1, 0))
    rows, _ = rec.drain([0, 0])
    gaps = [r[sm.S_GAP] for r in rows[0]]
    assert gaps[0] == 0, "the first callback of a layer has no previous one to subtract"
    assert 1_800_000 < gaps[1] < 6_000_000, gaps  # ns


def test_a_layer_past_the_end_is_dropped_not_written_out_of_bounds(monkeypatch):
    rec = _enable(monkeypatch, 2, FakeWrapper())
    before = list(rec.buf)
    rec.record(9, [1, 2, 0, 0, 0, 0, 3, 0], 4)
    assert rec.buf == before and rec.count == [0, 0]


def test_the_ring_keeps_the_newest_and_says_how_many_it_lost(monkeypatch):
    rec = sm.SeamRecorder(1, capacity=4)
    for i in range(1, 11):
        rec.record(0, [i * 100, i * 100 + 1, 0, 0, 0, 0, i * 100 + 2, 0], i * 100 + 3)
    rows, counts = rec.drain([0])
    assert counts == [10] and len(rows[0]) == 4
    assert rec.dropped[0] == 6, "records that fell out of the ring are counted, not ignored"


def test_without_the_probe_only_the_lumped_kt_segment_is_filled(monkeypatch):
    """A kt build that will not let us wrap CPUInfer: fewer columns, said so."""
    w = FakeWrapper(pre_us=100, sync_us=800)
    monkeypatch.setattr(sm, "install_infer_probe", lambda wrapper: False)
    rec = _enable(monkeypatch, 2, w)
    assert rec.split is False
    for _ in range(3):
        kl._kt_host_forward((w, "s", 1, 1))
    rows, _ = rec.drain([0, 0])
    med = sm.summarize(rows, rec.split, 1.0)["layers"]["1"]["med_us"]
    assert med["submit"] == 0 and med["sync"] == 0, "not measured must read as not measured"
    assert med["kt"] > 800
    assert sm.reconcile(med, rec.split)["ok"], "pre + kt + post must still add up to cb"


def test_the_probe_is_transparent_and_unwraps_for_the_identity_check():
    inner = FakeInfer()
    w = FakeWrapper()
    w.cpu_infer = inner
    assert sm.install_infer_probe(w) is True
    assert sm.unwrap_infer(w.cpu_infer) is inner
    assert w.cpu_infer.submit("t") == "submitted" and w.cpu_infer.sync(0) == "synced"
    assert inner.calls == ["submit:t", "sync:0"]
    assert sm.install_infer_probe(w) is True and sm.unwrap_infer(w.cpu_infer) is inner  # idempotent


def test_two_wrappers_sharing_one_cpuinfer_are_not_read_as_two_queues():
    """The probe is per wrapper; the queue invariant is about the inner object."""
    inner = FakeInfer()
    a, b = FakeWrapper(), FakeWrapper()
    a.cpu_infer = b.cpu_infer = inner
    sm.install_infer_probe(a)
    sm.install_infer_probe(b)
    assert a.cpu_infer is not b.cpu_infer
    kl._reset_wrapper_registry()
    try:
        kl._register_wrapper(a, 0)
        kl._register_wrapper(b, 1)  # must not raise: one CPUInfer behind both
    finally:
        kl._reset_wrapper_registry()


def test_a_raising_callback_records_nothing_and_still_does_not_escape(monkeypatch):
    w = FakeWrapper()
    w.pending = {0: True}  # kt says layer 0 left a deferred task
    rec = _enable(monkeypatch, 3, w)
    kl._kt_host_forward((w, "s", 1, 1))  # layer 1 checks layer 0
    assert rec.count == [0, 0, 0], "a callback that failed has no segments to report"
    assert kl.HOST_FUNC_CALLS == {}








def test_the_sampler_is_a_daemon_and_stops_with_its_process(monkeypatch):
    w = FakeWrapper()
    _enable(monkeypatch, 2, w, interval="0.01")
    t = sm.start(0.01)
    assert t is not None and t.daemon
    assert sm.start(0.01) is t  # one sampler per process
    sm.stop()
    for _ in range(100):
        if not t.is_alive():
            break
        time.sleep(0.01)
    assert not t.is_alive(), "a watcher that outlives its process samples someone else's service"


def test_the_probe_costs_far_less_than_the_seam_it_measures(monkeypatch, capsys):
    """CPU-only number: the instrument must not be the thing being measured.
    The bound is deliberately loose because a shared CI box is noisy; a
    regression that matters would be orders of magnitude, not percent."""
    reps = 2000
    w_off = FakeWrapper()
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        kl._kt_host_forward((w_off, "s", 1, 1))
    off_ns = (time.perf_counter_ns() - t0) / reps

    w_on = FakeWrapper()
    _enable(monkeypatch, 2, w_on)
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        kl._kt_host_forward((w_on, "s", 1, 1))
    on_ns = (time.perf_counter_ns() - t0) / reps

    delta_us = (on_ns - off_ns) / 1000.0
    with capsys.disabled():
        print(f"\n  seam timing overhead: {delta_us:.3f} us/callback "
              f"(off {off_ns/1000:.3f} us, on {on_ns/1000:.3f} us, {reps} reps)")
    assert delta_us < 10.0, f"{delta_us:.2f} us/callback is too much for the seam it measures"


def test_the_realtime_stamps_are_absolute_and_bracket_the_callback(monkeypatch):
    """These two stamps are compared against a profile's ts, so they must be
    epoch-based CLOCK_REALTIME and must straddle the whole callback."""
    w = FakeWrapper(pre_us=50, sync_us=300)
    rec = _enable(monkeypatch, 2, w)
    before = time.clock_gettime_ns(time.CLOCK_REALTIME)
    for _ in range(4):
        kl._kt_host_forward((w, "s", 1, 1))
    after = time.clock_gettime_ns(time.CLOCK_REALTIME)

    rows, _ = rec.drain([0, 0])
    assert len(rows[1]) == 4
    for row in rows[1]:
        entry, exit_ = row[sm.R_ENTER_RT], row[sm.R_EXIT_RT]
        assert before <= entry < exit_ <= after, "not this process's CLOCK_REALTIME"
        # the span covers cb and the record() that follows it, nothing else
        assert 0 <= (exit_ - entry) - row[sm.S_CB] < 50_000, (exit_ - entry, row[sm.S_CB])
    # a different clock from the segments (perf_counter has an arbitrary
    # origin): the two must not be confusable by magnitude
    assert rows[1][0][sm.R_ENTER_RT] > 1_700_000_000_000_000_000




def test_a_callback_with_no_realtime_stamp_is_left_out_not_written_as_zero(monkeypatch):
    rec = sm.SeamRecorder(1, capacity=4)
    rec.record(0, [1000, 1100, 0, 0, 0, 0, 1900, 0], 2000)  # T_ENTER_RT = 0: no stamp
    rows, _ = rec.drain([0])
    assert rows[0][0][sm.R_ENTER_RT] == 0
    assert sm.realtime_rows(rows) == [], "a 0 stamp is not a timestamp"


def test_the_realtime_stamp_costs_a_tenth_of_a_microsecond(capsys):
    """Two extra clock reads per callback, inside a graph node the device is
    blocked on."""
    reps = 20000
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        time.clock_gettime_ns(time.CLOCK_REALTIME)
    rt_ns = (time.perf_counter_ns() - t0) / reps
    t0 = time.perf_counter_ns()
    for _ in range(reps):
        time.perf_counter_ns()
    pc_ns = (time.perf_counter_ns() - t0) / reps
    with capsys.disabled():
        print(f"\n  clock_gettime_ns(CLOCK_REALTIME) {rt_ns:.1f} ns/call, "
              f"perf_counter_ns {pc_ns:.1f} ns/call, {reps} reps")
    assert rt_ns < 1000.0, f"{rt_ns:.0f} ns/call is no longer negligible next to the seam"


def test_a_healthy_split_run_reports_no_unsplit_callback_and_no_residual(monkeypatch):
    w = FakeWrapper(pre_us=50, submit_us=20, between_us=20, sync_us=200, tail_us=20)
    rec = _enable(monkeypatch, 2, w)
    for _ in range(4):
        kl._kt_host_forward((w, "s", 1, 1))
    rows, _ = rec.drain([0, 0])
    layer = sm.summarize(rows, True, 1.0)["layers"]["1"]
    assert layer["n_unsplit"] == 0
    assert layer["resid_us"]["max"] == 0.0, "the segments are one clock's differences: exactly 0"


