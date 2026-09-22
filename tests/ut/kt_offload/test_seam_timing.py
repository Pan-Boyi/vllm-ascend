# SPDX-License-Identifier: Apache-2.0
"""Seam timing for the host-func callback -- CPU only, fake kt, real clocks.

§4.18 left 107.5 us/layer of NOTIFY_WAIT unaccounted for (353.6 measured on
the device minus 246.1 of kt compute). The instrument that is supposed to
split it is only worth anything if three things hold, and each has a test
here:

  * with the switch off the callback behaves bit for bit as before and
    records nothing at all;
  * with it on, each segment is where it claims to be -- driven by a fake
    wrapper whose sleeps are known, every segment must come back with the
    sleep it was given and nothing else;
  * a timestamp taken in the WRONG place is caught. That is the mutation arm:
    the same fake, one stamp moved, and the reconciliation must go red. An
    instrument that cannot fail its own audit is not evidence.

The overhead of the instrument is measured here too (a CPU number, printed by
``-s``): it is subtracted from nothing, but a seam probe that costs as much as
the seam it measures is a different experiment.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import time
from pathlib import Path

import pytest

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload import seam_timing as sm

SEAM_PY = Path(__file__).resolve().parents[3] / "tools" / "kt_offload" / "prof" / "seam.py"


def _load_seam_tool():
    spec = importlib.util.spec_from_file_location("seam_tool", SEAM_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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

    ``run_pinned_forward_sync`` is Python in kt too (experts_base.py:717-772),
    and it reaches the queue through ``self.cpu_infer`` -- which is what makes
    the submit/sync split possible without touching kt's C++. This fake keeps
    that shape exactly, or the probe would be tested against something kt does
    not do.
    """

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


# ---------------------------------------------------------------- off means off


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


# ---------------------------------------------------------------- the segments


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


# ------------------------------------------------------------- the mutation arm


def test_a_timestamp_in_the_wrong_place_is_caught_by_the_reconciliation(monkeypatch):
    """Move one stamp: submit's return recorded after sync instead of before.

    This is the failure the whole instrument has to be able to detect, so it
    is produced on purpose. The segments then overlap (sync is counted twice)
    and the parts no longer add up to the callback.
    """
    w = FakeWrapper(submit_us=200, sync_us=1500)
    rec = _enable(monkeypatch, 2, w)

    for _ in range(3):
        kl._kt_host_forward((w, "s", 1, 1))
    rows, since = rec.drain([0, 0])
    # The control is checked PER CALLBACK, not on the medians. Summing the
    # per-segment medians and comparing against the median of cb compares two
    # statistics that are not equal even on a healthy run (medians do not add),
    # and with three samples on a loaded box it drifts past any fixed
    # tolerance -- an arm whose colour depends on what else the machine was
    # doing proves nothing. Per callback the identity is exact, in integers.
    assert rows[1] and all(_split_parts(row) == row[sm.S_CB] for row in rows[1]), (
        "control arm: every callback's segments must add up to its cb exactly")
    assert all(row[sm.S_SYNC] > 0 for row in rows[1]), "control arm: the probe must have run"

    honest_submit = sm._TimedCpuInfer.submit

    def submit_stamped_too_late(self, task):
        out = honest_submit(self, task)
        # the mutation: "submit returned" is stamped with the PREVIOUS
        # callback's sync-return, i.e. taken after the sync instead of before
        sm.TS[sm.T_SUBMIT_OUT] = sm.TS[sm.T_SYNC_OUT]
        return out

    monkeypatch.setattr(sm._TimedCpuInfer, "submit", submit_stamped_too_late)
    for _ in range(3):
        kl._kt_host_forward((w, "s", 1, 1))
    rows, _ = rec.drain(since)
    if rows:
        # record() kept the row but refused to split it: the middle is zeros
        # while kt and cb stay whole. That is what reaches the offline tool,
        # and both sides must call it broken.
        assert all(row[sm.S_SYNC] == 0 for row in rows[1]), "the mutation was stored as if it were honest"
        med_bad = sm.summarize(rows, rec.split, 1.0)["layers"]["1"]
        assert not sm.reconcile(med_bad["med_us"], rec.split)["ok"], "a misplaced stamp went unnoticed"
        tool = _load_seam_tool()
        assert not tool.check_layer(med_bad["med_us"], med_bad["p90_us"], True)["ok"], (
            "the offline criterion must be red on the same records")
    else:
        # record() refused the record outright (the stamps were not monotonic)
        assert rec.dropped[1] >= 3


def test_the_offline_tool_agrees_with_the_in_process_reconciliation():
    tool = _load_seam_tool()
    good = {"pre": 1.0, "kt_pre": 2.0, "submit": 3.0, "between": 1.0, "sync": 100.0,
            "kt_tail": 1.0, "post": 2.0, "kt": 107.0, "cb": 110.0, "gap": 34000.0}
    assert tool.reconcile(good, True)["ok"]
    bad = dict(good, sync=150.0)  # a segment that overlaps its neighbours
    v = tool.reconcile(bad, True)
    assert not v["ok"] and v["residual_us"] < -40


# ------------------------------------------------------------------ the sampler


def test_the_sampler_line_is_what_the_offline_tool_reads(monkeypatch):
    """End to end: run callbacks, let the sampler log, parse the log."""
    records: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Collect()
    sm.logger.addHandler(handler)
    previous = sm.logger.level
    sm.logger.setLevel(logging.DEBUG)
    try:
        w = FakeWrapper(sync_us=500)
        _enable(monkeypatch, 2, w, interval="0.05")
        for _ in range(6):
            kl._kt_host_forward((w, "s", 1, 1))
        for _ in range(100):
            if any("[kt-seam]" in r for r in records):
                break
            time.sleep(0.02)
    finally:
        sm.stop()
        sm.logger.removeHandler(handler)
        sm.logger.setLevel(previous)

    lines = [r for r in records if "[kt-seam]" in r]
    assert lines, "the sampler produced no record"
    tool = _load_seam_tool()
    parsed, dropped = tool.parse_records("\n".join(lines))
    assert dropped == 0 and parsed
    merged = tool.merge(parsed)
    assert merged["split"] and 1 in merged["median_us"]
    # check_layer, not reconcile: the latter is the median-sum residual, which
    # is not a criterion (see seam.py's docstring) and drifts under load.
    assert tool.check_layer(merged["median_us"][1], merged["p90_us"][1], True)["ok"]
    # and the record really is one line of JSON, not a pretty print
    assert all(len(line.splitlines()) == 1 for line in lines)
    assert json.loads(lines[0].split("[kt-seam] ", 1)[1])["layers"]


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


# --------------------------------------------------------------- what it costs


def test_the_probe_costs_far_less_than_the_seam_it_measures(monkeypatch, capsys):
    """CPU-only number: the instrument must not be the thing being measured.

    The seam under investigation is 107.5 us/layer. The bound here is
    deliberately loose (10 us for a whole callback's worth of stamps, ~10
    segments) because a shared CI box is noisy; a regression that matters
    would be orders of magnitude, not percent.
    """
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
    assert delta_us < 10.0, f"{delta_us:.2f} us/callback is too much next to a 107.5 us seam"


# ----------------------------------------- the CLOCK_REALTIME stamps (§4.22 P0)


def test_the_realtime_stamps_are_absolute_and_bracket_the_callback(monkeypatch):
    """A and F are computed against a profile's ts, so these two stamps have to
    be epoch-based CLOCK_REALTIME and have to straddle the whole callback --
    entry before the first segment, exit after the last store."""
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
        # the realtime span covers cb and the record() call that follows it,
        # and nothing else: that difference is what the offline tool's
        # |A+F - (NW - cb)| < 5 us check sees
        assert 0 <= (exit_ - entry) - row[sm.S_CB] < 50_000, (exit_ - entry, row[sm.S_CB])
    # and it is a *different* clock from the segments: perf_counter is ns since
    # an arbitrary origin, so the two must not be confusable by magnitude
    assert rows[1][0][sm.R_ENTER_RT] > 1_700_000_000_000_000_000


def test_the_realtime_rows_are_what_the_offline_tool_parses(monkeypatch):
    w = FakeWrapper(sync_us=100)
    rec = _enable(monkeypatch, 2, w)
    for _ in range(3):
        kl._kt_host_forward((w, "s", 1, 1))
    rows, _ = rec.drain([0, 0])
    entries = sm.realtime_rows(rows)
    assert [e["layer"] for e in entries] == [1]
    line = sm.format_realtime(entries[0])
    assert len(line.splitlines()) == 1
    tool = _load_seam_tool()
    calls, bad = tool.parse_realtime(line)
    assert bad == 0 and len(calls) == 3
    assert all(layer == 1 and 0 < entry < exit_ and cb > 0 for layer, entry, exit_, cb in calls)
    # a [kt-seam-rt] line must not be counted as a missing [kt-seam] record
    parsed, dropped = tool.parse_records(line)
    assert parsed == [] and dropped == 0


def test_a_callback_with_no_realtime_stamp_is_left_out_not_written_as_zero(monkeypatch):
    rec = sm.SeamRecorder(1, capacity=4)
    rec.record(0, [1000, 1100, 0, 0, 0, 0, 1900, 0], 2000)  # T_ENTER_RT = 0: no stamp
    rows, _ = rec.drain([0])
    assert rows[0][0][sm.R_ENTER_RT] == 0
    assert sm.realtime_rows(rows) == [], "a 0 stamp is not a timestamp"


def test_the_realtime_stamp_costs_a_tenth_of_a_microsecond(capsys):
    """The number the report has to carry: two extra clock reads per callback,
    inside a graph node the device is blocked on."""
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
    assert rt_ns < 1000.0, f"{rt_ns:.0f} ns/call next to a 78 us seam is no longer negligible"


# --------------------------- n_unsplit / resid_us.max, the exact fields (§4.20.1)


def test_a_healthy_split_run_reports_no_unsplit_callback_and_no_residual(monkeypatch):
    w = FakeWrapper(pre_us=50, submit_us=20, between_us=20, sync_us=200, tail_us=20)
    rec = _enable(monkeypatch, 2, w)
    for _ in range(4):
        kl._kt_host_forward((w, "s", 1, 1))
    rows, _ = rec.drain([0, 0])
    layer = sm.summarize(rows, True, 1.0)["layers"]["1"]
    assert layer["n_unsplit"] == 0
    assert layer["resid_us"]["max"] == 0.0, "the segments are one clock's differences: exactly 0"


def test_one_unfilled_callback_among_many_is_counted_and_sized(monkeypatch):
    """What the medians cannot see (§4.20.1): a single callback whose middle
    never got filled. n_unsplit counts it and resid_us.max sizes it."""
    w = FakeWrapper(pre_us=50, submit_us=20, between_us=20, sync_us=200, tail_us=20)
    rec = _enable(monkeypatch, 2, w)
    for _ in range(5):
        kl._kt_host_forward((w, "s", 1, 1))
    honest = sm.summarize(rec.drain([0, 0])[0], True, 1.0)["layers"]["1"]
    assert honest["n_unsplit"] == 0

    # the probe does not run for one callback (its stamps stay from the last
    # one, so record() refuses to split it) -- exactly the mutation arm above
    monkeypatch.setattr(sm._TimedCpuInfer, "submit", lambda self, task: self._kt_seam_inner.submit(task))
    since = list(rec.count)
    kl._kt_host_forward((w, "s", 1, 1))
    for _ in range(4):
        pass
    rows, _ = rec.drain(since)
    layer = sm.summarize(rows, True, 1.0)["layers"]["1"]
    assert layer["n_unsplit"] == 1, rows
    # and the size of what was lost is kt itself, in us
    assert layer["resid_us"]["max"] == pytest.approx(rows[1][0][sm.S_KT] / 1000.0, abs=0.01)
    tool = _load_seam_tool()
    assert not tool.check_layer(layer["med_us"], layer["p90_us"], True, layer)["ok"], (
        "the offline verdict must use the exact field when it is there")
