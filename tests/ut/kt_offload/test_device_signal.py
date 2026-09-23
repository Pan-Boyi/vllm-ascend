# SPDX-License-Identifier: Apache-2.0
"""P1b (``VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL``): the protocol, the thread, the spin.

The *ordering* model -- what the device does, what happens when the release is
early or never comes, and what a counter copy in front of the input copy does
to the output -- lives in ``test_side_stream_overlap.py``, because that is
where the simulator that can execute a schedule is. This file covers the three
halves that need no simulator and could not be tested there:

* the **criteria**, driven red once per way the recorded node sequence can be
  wrong -- above all a host-func node coming back (which would make a reading
  about P1b a reading about P1a) and the two copies swapping places;
* the **polling thread**: expected step numbers, a missed step, a step seen
  twice, and its contract -- run BEFORE record, record on every path including
  a raise, 🔴 and never a record on a timeout;
* the **spin loop itself**, which is real C compiled at start-up: its wait
  semantics, and the measurement that says why it is C at all (a Python
  spinner costs the main thread several times its own speed; this one costs
  it nothing measurable).
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import device_signal as ds
from vllm_ascend.kt_offload import layer as kl


@pytest.fixture(autouse=True)
def _clean_counters():
    ds.STEPS.clear()
    ds.RECORDS.clear()
    ds.ERRORS.clear()
    yield
    ds.STEPS.clear()
    ds.RECORDS.clear()
    ds.ERRORS.clear()


# ---------------------------------------------------------------------------
# 1. the node-sequence criterion
# ---------------------------------------------------------------------------

EV = object()


def _good_ops():
    """What ``route`` + ``_launch_signal_node`` + ``_packed_output`` record."""
    return [
        ("bump", 0),
        ("d2h", 1),
        ("signal", (0, 1)),
        ("ext_wait", EV),
        ("ext_reset", EV),
        ("h2d", 1),
    ]


def test_the_production_sequence_passes():
    rep = ds.signal_ops_ok(_good_ops())
    assert rep["ok"], rep["detail"]
    # and without the bump (every layer but the lowest)
    assert ds.signal_ops_ok(_good_ops()[1:])["ok"]


def test_a_host_func_node_coming_back_is_caught():
    """🔴 The one that makes the whole reading mean something else.

    A host-func node left in the graph IS P1a, and the 35.75 us of idle in
    front of it comes back with it -- while the arm would still be labelled
    P1b. Nothing else in the process would notice.
    """
    rep = ds.signal_ops_ok([*_good_ops(), ("hostfunc", 0)])
    assert not rep["ok"]
    assert any("host-func" in p for p in rep["detail"]), rep["detail"]


def test_the_counter_copy_must_be_behind_the_input_copy():
    """🔴 The failure with no second line of defence: faster, wrong, silent."""
    ops = _good_ops()
    ops[1], ops[2] = ops[2], ops[1]
    rep = ds.signal_ops_ok(ops)
    assert not rep["ok"]
    assert any("BEFORE the input copy" in p for p in rep["detail"]), rep["detail"]


def test_a_missing_signal_is_caught():
    assert not ds.signal_ops_ok([o for o in _good_ops() if o[0] != "signal"])["ok"]


def test_two_signals_in_one_layer_are_caught():
    assert not ds.signal_ops_ok([*_good_ops(), ("signal", (0, 1))])["ok"]


def test_two_increments_in_one_layer_are_caught():
    rep = ds.signal_ops_ok([("bump", 0), *_good_ops()])
    assert not rep["ok"]
    assert any("increment" in p for p in rep["detail"]), rep["detail"]


def test_the_counter_must_be_incremented_before_it_is_copied():
    ops = [("d2h", 1), ("signal", (0, 1)), ("bump", 0), ("ext_wait", EV), ("ext_reset", EV), ("h2d", 1)]
    rep = ds.signal_ops_ok(ops)
    assert not rep["ok"]
    assert any("one step behind" in p for p in rep["detail"]), rep["detail"]


def test_the_wait_and_the_reset_must_be_in_the_graph():
    for kind in ("ext_wait", "ext_reset"):
        assert not ds.signal_ops_ok([o for o in _good_ops() if o[0] != kind])["ok"]
    swapped = [("bump", 0), ("d2h", 1), ("signal", 0), ("ext_reset", EV), ("ext_wait", EV), ("h2d", 1)]
    assert not ds.signal_ops_ok(swapped)["ok"]
    other = object()
    mixed = [("d2h", 1), ("signal", 0), ("ext_wait", EV), ("ext_reset", other), ("h2d", 1)]
    assert not ds.signal_ops_ok(mixed)["ok"]


def test_the_output_copy_must_be_behind_the_wait():
    ops = [("d2h", 1), ("signal", 0), ("h2d", 1), ("ext_wait", EV), ("ext_reset", EV)]
    rep = ds.signal_ops_ok(ops)
    assert not rep["ok"]
    assert any("copied back before the wait" in p for p in rep["detail"]), rep["detail"]


def test_a_host_record_inside_the_graph_is_caught():
    ops = [("d2h", 1), ("signal", 0), ("host_record", EV), ("ext_wait", EV), ("ext_reset", EV), ("h2d", 1)]
    assert not ds.signal_ops_ok(ops)["ok"]


def test_the_wait_must_be_behind_the_signal():
    ops = [("d2h", 1), ("ext_wait", EV), ("ext_reset", EV), ("signal", 0), ("h2d", 1)]
    rep = ds.signal_ops_ok(ops)
    assert not rep["ok"]
    assert any("BEFORE the copy that signals" in p for p in rep["detail"]), rep["detail"]


def test_the_step_and_pairing_criteria():
    assert ds.step_ok(3, 7, 7)["ok"]
    missed = ds.step_ok(3, 9, 7)
    assert not missed["ok"] and missed["detail"]["missed"] == 2
    assert ds.signals_per_replay_ok({0: 1}, {0: 4}, 3, [0])["ok"]
    bad = ds.signals_per_replay_ok({0: 1, 1: 1}, {0: 4, 1: 3}, 3, [0, 1])
    assert not bad["ok"] and bad["detail"]["wrong"] == {1: 2}
    assert ds.records_match_signals({0: 3}, {0: 3})["ok"]
    assert not ds.records_match_signals({0: 3}, {0: 2})["ok"]


# ---------------------------------------------------------------------------
# 2. the polling thread
# ---------------------------------------------------------------------------


class FakeWaiter:
    """A waiter over a dict of words, with the same contract as the C one:
    return the SLOT of the first word that reached its expected step, or -1."""

    def __init__(self, log=None):
        self.words: dict[int, int] = {}
        self.expected: dict[int, int] = {}
        self.log = log if log is not None else []
        self.waits = 0

    def set_expected(self, slot, value):
        self.expected[int(slot)] = int(value)

    def read(self, slot):
        return int(self.words.get(int(slot), 0))

    def wait(self, timeout_ns, hot_ns, sleep_ns):
        self.waits += 1
        deadline = time.monotonic() + min(timeout_ns, 500_000_000) / 1e9
        while time.monotonic() < deadline:
            for slot in sorted(self.expected):
                want = self.expected[slot]
                if want and self.words.get(slot, 0) >= want:
                    return slot
            time.sleep(0.001)
        return -1

    def poke(self, slot, value):
        self.words[int(slot)] = int(value)


class FakeEvent:
    def __init__(self, log, name="ev"):
        self.log, self.name = log, name
        self.records = 0
        self.stream = None

    def record(self, stream):
        self.records += 1
        self.stream = stream
        self.log.append(("record", self.name))


def _poller(log, waiter, *, device_error=None, stream="S"):
    def set_device():
        if device_error is not None:
            raise device_error
        log.append(("set_device", threading.current_thread().name))

    def make_stream():
        log.append(("make_stream",))
        return stream

    return ds.Poller(set_device, make_stream, lambda: waiter, name="test-poller")


def _runner(log, layer=0, *, raises=None):
    def run(bs):
        log.append(("run", layer, bs))
        if raises is not None:
            raise raises

    return run


def _wait_for(pred, what, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.002)
    raise AssertionError(f"timed out waiting for {what}")


def test_it_runs_kt_first_and_records_after():
    """The order IS the correctness property: a record before the run releases
    the device while the pinned output still holds the previous step's."""
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)
    ev = FakeEvent(log)
    p.register(ds.Entry(4, 1, 7, _runner(log, 4), ev))
    p.start()
    try:
        w.poke(7, 1)
        _wait_for(lambda: ev.records == 1, "the record")
    finally:
        p.stop()
    kinds = [e for e in log if e[0] in ("run", "record")]
    assert [k[0] for k in kinds] == ["run", "record"], log
    assert ev.stream == "S"
    assert ds.STEPS == {4: 1} and ds.RECORDS == {4: 1} and ds.ERRORS == []
    assert p.expected_of(4) == 2


def test_the_thread_takes_its_own_acl_context_before_anything_else():
    """The record is an ACL call from a thread that is neither the main one nor
    ACL's report thread, so it needs a context; and the stream is made IN the
    thread, before start() returns, because a capture must never be the first
    thing that needs it."""
    log: list = []
    p = _poller(log, FakeWaiter(log))
    p.start()
    try:
        assert [e[0] for e in log] == ["set_device", "make_stream"], log
        assert log[0][1] == "test-poller"
    finally:
        p.stop()


def test_a_context_failure_is_a_start_up_error_not_a_hang():
    log: list = []
    p = _poller(log, FakeWaiter(log), device_error=RuntimeError("no context"))
    with pytest.raises(RuntimeError, match="could not start"):
        p.start()
    p.stop()


def test_a_layer_that_raises_still_releases_the_device():
    """A device left waiting wedges the engine with no message; stale output at
    least says so in the log, and is the failure the serial node already has."""
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)
    ev = FakeEvent(log)
    p.register(ds.Entry(2, 1, 0, _runner(log, 2, raises=RuntimeError("kt blew up")), ev))
    p.start()
    try:
        w.poke(0, 1)
        _wait_for(lambda: ev.records == 1, "the record after a raise")
    finally:
        p.stop()
    assert ds.STEPS == {2: 1} and ds.RECORDS == {2: 1}
    assert len(ds.ERRORS) == 1 and ds.ERRORS[0][0] == 2 and "kt blew up" in ds.ERRORS[0][1]


def test_each_replay_advances_the_expected_step():
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)
    ev = FakeEvent(log)
    p.register(ds.Entry(0, 1, 0, _runner(log), ev))
    p.start()
    try:
        for step in (1, 2, 3):
            w.poke(0, step)
            _wait_for(lambda s=step: ev.records == s, f"record {step}")
            assert p.expected_of(0) == step + 1
    finally:
        p.stop()
    assert ds.STEPS == {0: 3} and not ds.ERRORS


def test_a_missed_step_is_reported_and_resynchronised():
    """🔴 Mutant ④: the word runs ahead of the step the poller is waiting for.

    It means some replay released its H2D without this thread having run kt for
    it. The layer still runs (the pinned inputs ARE this step's -- the copy in
    front of the counter saw to that), but it must not pass silently, and the
    poller must come back in step rather than trailing for ever.
    """
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)
    ev = FakeEvent(log)
    p.register(ds.Entry(5, 1, 0, _runner(log, 5), ev))
    p.start()
    try:
        w.poke(0, 4)  # three steps went past unseen
        _wait_for(lambda: ev.records == 1, "the record")
        assert p.expected_of(5) == 5, "the poller must resynchronise, not trail"
        w.poke(0, 5)
        _wait_for(lambda: ev.records == 2, "the next record")
    finally:
        p.stop()
    assert len(ds.ERRORS) == 1 and "step mismatch" in ds.ERRORS[0][1], ds.ERRORS
    assert "'missed': 3" in ds.ERRORS[0][1], ds.ERRORS


def test_a_step_that_never_arrives_is_not_released():
    """🔴 The deliberate non-feature.

    A timeout must NOT record: from in here "the service is idle" and "the
    signal is lost" look the same, and a spurious record would release a wait
    nobody armed AND leave the next replay's wait pre-signalled -- turning a
    visible hang into silent stale output on every step after it.
    """
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)
    ev = FakeEvent(log)
    p.register(ds.Entry(1, 1, 0, _runner(log, 1), ev))
    p.TIMEOUT_NS = 20_000_000
    p.start()
    try:
        _wait_for(lambda: w.waits >= 3, "three timed-out waits")
        assert ev.records == 0 and not ds.RECORDS and not ds.STEPS
        assert not ds.ERRORS
    finally:
        p.stop()


def test_a_record_that_fails_is_critical_and_counted():
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)

    class Broken(FakeEvent):
        def record(self, stream):
            raise RuntimeError("acl said no")

    ev = Broken(log)
    p.register(ds.Entry(3, 1, 0, _runner(log, 3), ev))
    p.start()
    try:
        w.poke(0, 1)
        _wait_for(lambda: bool(ds.ERRORS), "the swallowed record error")
    finally:
        p.stop()
    assert ds.STEPS == {3: 1} and ds.RECORDS == {}
    assert "record failed" in ds.ERRORS[0][1]
    assert not ds.records_match_signals(dict(ds.STEPS), dict(ds.RECORDS))["ok"]


def test_the_slot_a_signal_arrives_on_decides_which_layer_runs():
    """🔴 Slots are not positions in the entry list.

    The self-test takes slot 0, so the first registered entry lives at slot 1.
    A poller that indexed its entry list with the waiter's return value would
    run the wrong layer's experts and record the wrong layer's event -- with
    numbers that look entirely plausible.
    """
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)
    evs = {11: FakeEvent(log, "L11"), 12: FakeEvent(log, "L12")}
    p.register(ds.Entry(11, 1, 1, _runner(log, 11), evs[11]))
    p.register(ds.Entry(12, 1, 2, _runner(log, 12), evs[12]))
    p.start()
    try:
        w.poke(2, 1)  # slot 2 = layer 12, which is entry INDEX 1
        _wait_for(lambda: evs[12].records == 1, "layer 12's record")
        assert evs[11].records == 0
        assert [e for e in log if e[0] == "run"] == [("run", 12, 1)]
    finally:
        p.stop()
    assert ds.STEPS == {12: 1}


def test_layers_are_independent_and_registration_is_seen_late():
    """Entries added after the thread started must be picked up (every layer
    registers at weight-load time, which is after the first one started it)."""
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)
    first = FakeEvent(log, "L0")
    p.register(ds.Entry(0, 1, 1, _runner(log, 0), first))
    p.start()
    try:
        late = FakeEvent(log, "L9")
        p.register(ds.Entry(9, 1, 2, _runner(log, 9), late))
        w.poke(2, 1)
        _wait_for(lambda: late.records == 1, "the late entry's record")
        assert first.records == 0
        assert p.expected_of(0) == 1 and p.expected_of(9) == 2
    finally:
        p.stop()


def test_one_layer_with_several_capture_sizes_shares_its_expected_step():
    """A layer has one word per batch size and only one of them moves per
    replay; the step number is the layer's, not the size's."""
    log: list = []
    w = FakeWaiter(log)
    p = _poller(log, w)
    ev1, ev16 = FakeEvent(log, "bs1"), FakeEvent(log, "bs16")
    p.register(ds.Entry(0, 1, 1, _runner(log, 0), ev1))
    p.register(ds.Entry(0, 16, 2, _runner(log, 0), ev16))
    p.start()
    try:
        w.poke(1, 1)  # step 1 ran at bs=1
        _wait_for(lambda: ev1.records == 1, "the bs1 record")
        assert w.expected == {1: 2, 2: 2}, "both of the layer's slots now expect step 2"
        w.poke(2, 2)  # step 2 ran at bs=16; the stale bs1 word (1) must not match
        _wait_for(lambda: ev16.records == 1, "the bs16 record")
        assert ev1.records == 1 and not ds.ERRORS
    finally:
        p.stop()
    assert ds.STEPS == {0: 2}


# ---------------------------------------------------------------------------
# 3. the spin loop (real C, compiled at start-up)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def native():
    try:
        return ds.load_native()
    except RuntimeError as exc:  # no compiler, or every candidate dir is noexec
        pytest.skip(f"the polling helper could not be built here: {exc}")


@pytest.fixture
def words(native):
    import ctypes

    n = 8
    buf = (ctypes.c_uint64 * (n * ds.STRIDE_WORDS))()
    return buf, ds.Waiter(ctypes.addressof(buf), n, lib=native)


def test_the_helper_waits_for_the_expected_step(words):
    _buf, w = words
    w.set_expected(3, 5)
    assert w.wait(20_000_000, 0, 200_000) == -1, "it matched a word that is still behind"
    w.poke_after(3, 5, 0)
    assert w.wait(200_000_000, 10_000_000, 200_000) == 3
    assert w.read(3) == 5


def test_an_unused_slot_never_matches(words):
    """``expected == 0`` is the sentinel. Without it every unused slot would
    match on the first scan, and the poller would spin on a layer that is not
    running."""
    buf, w = words
    buf[2 * ds.STRIDE_WORDS] = 99
    w.set_expected(2, 0)
    assert w.wait(20_000_000, 0, 200_000) == -1


def test_a_word_that_ran_ahead_still_returns_so_python_can_judge(words):
    """The wait condition is ``>=``; the equality is a diagnosis, not a wait.
    A waiter that insisted on ``==`` would hang for ever on a missed step."""
    _buf, w = words
    w.poke_after(1, 9, 0)
    w.set_expected(1, 6)
    assert w.wait(20_000_000, 0, 200_000) == 1
    assert not ds.step_ok(0, w.read(1), 6)["ok"]


def test_the_spin_sees_another_agents_write_within_microseconds(words):
    """How long the device waits between its copy landing and kt starting.

    Both timestamps are taken in C (the write in ``kt_poke_after``, the match
    in the wait itself): a Python statement after ``wait`` would be measuring
    the GIL hand-off, not the loop. For scale, the mechanism this replaces
    costs ~39 us of dispatch plus 35.75 us of device idle in front of the node
    (PLAN §4.22 F1).
    """
    _buf, w = words
    lat = []
    for k in range(1, 61):
        w.set_expected(0, k)
        wrote = []
        t = threading.Thread(target=lambda w_=wrote, k_=k: w_.append(w.poke_after(0, k_, 300_000)))
        t.start()
        assert w.wait(1_000_000_000, 60_000_000, 200_000) == 0
        t.join()
        lat.append((w.seen_ns - wrote[0]) / 1000.0)
    lat.sort()
    median = lat[len(lat) // 2]
    # Loose on purpose: this is a busy shared machine and the number that
    # matters is the order of magnitude. Measured 2026-09-23: median 0.52 us,
    # p90 0.60 us over 300 pokes.
    assert median < 20.0, f"the spin took {median:.1f} us (median) to see the write: {lat[:5]}...{lat[-3:]}"


def test_the_spin_does_not_hold_the_gil():
    """🔴 The reason this is C and not Python.

    A Python thread polling a word executes bytecode, so it holds the GIL
    between checks and the main thread waits for it. The comparison is the
    evidence: the same spin, once through ctypes (GIL released for the whole
    call) and once in Python.

    Measured 2026-09-23 on an idle-ish 256-PU host, pinned to two different
    physical cores: main alone 178 ms, C poller 164 ms, python spinner 1024 ms
    (5.8x), sleeping thread 162 ms.
    """
    import ctypes

    try:
        lib = ds.load_native()
    except RuntimeError as exc:
        pytest.skip(f"the polling helper could not be built here: {exc}")
    n = 4
    buf = (ctypes.c_uint64 * (n * ds.STRIDE_WORDS))()
    w = ds.Waiter(ctypes.addressof(buf), n, lib=lib)

    def work(reps=1_500_000):
        t0 = time.perf_counter()
        x = 0
        for i in range(reps):
            x = (x * 3 + i) & 0xFFFF
        return time.perf_counter() - t0

    def measure(spinner):
        stop = threading.Event()
        th = threading.Thread(target=spinner, args=(stop,))
        th.start()
        time.sleep(0.1)
        try:
            return min(work() for _ in range(5))
        finally:
            stop.set()
            th.join(timeout=10)

    def c_spin(stop):
        w.set_expected(1, 10**9)  # never satisfied
        while not stop.is_set():
            w.wait(50_000_000, 50_000_000, 0)

    def py_spin(stop):
        while not stop.is_set():
            if buf[2 * ds.STRIDE_WORDS] == 12345:
                break

    def asleep(stop):
        while not stop.is_set():
            time.sleep(0.01)

    base = measure(asleep)
    with_c = measure(c_spin)
    with_py = measure(py_spin)
    # The discriminating comparison, not an absolute: on a shared machine the
    # absolute numbers move, the ratio does not.
    # 1.4 rather than the 5.8 measured above: the GIL's own floor for two
    # CPU-bound threads is about 2x (they take turns), the machine is shared,
    # and what has to hold is the DIRECTION -- one of these two spinners takes
    # the interpreter away and the other does not.
    assert with_py > 1.4 * with_c, (
        f"a python spinner cost the main thread {with_py / base:.1f}x and the C one {with_c / base:.1f}x; "
        "if those are the same, this test is not measuring the GIL and the claim in poll_native.c is "
        "unsupported"
    )
    assert with_c < 1.6 * base, f"the C poller cost the main thread {with_c / base:.1f}x (base {base:.3f}s)"


def test_the_built_library_is_keyed_to_the_source(native):
    """An edit to poll_native.c must never be served from a stale .so."""
    import hashlib

    tag = hashlib.sha256(ds._SOURCE.read_text().encode()).hexdigest()[:16]
    found = [d / f"kt_poll_{tag}.so" for d in ds._cache_dirs()]
    assert any(p.exists() for p in found), found
    assert native.kt_poll_abi() == ds._ABI


# ---------------------------------------------------------------------------
# 4. the pieces that live in layer.py
# ---------------------------------------------------------------------------


def test_the_self_test_drives_the_whole_path_and_refuses_a_word_that_never_moves(monkeypatch):
    """The start-up check, on CPU tensors.

    On card it is what turns three silent hangs into start-up errors (no int64
    add, no 8-byte D2H, a DMA write the host cannot see). Here the same code
    runs with ``_host_copy`` doing a real copy, and then with one that does
    nothing -- which is what "the word never moves" looks like.
    """
    monkeypatch.setattr(kl, "_pinned_empty", lambda shape, dtype: torch.empty(shape, dtype=dtype))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    signals = kl._SignalWords(2)
    counter = kl._StepCounter(torch.device("cpu"))
    kl._selftest_signal_path(signals, counter)
    assert int(counter.value.item()) == 0, "the counter must be back at 0 for the first replay"
    assert signals.n == 1, "the self-test takes a slot of its own"

    monkeypatch.setattr(kl, "_host_copy", lambda dst, src: None)  # the copy that never lands
    with pytest.raises(RuntimeError, match="self-test counter copy"):
        kl._selftest_signal_path(kl._SignalWords(2), kl._StepCounter(torch.device("cpu")))


def test_the_signal_words_are_a_cache_line_apart(monkeypatch):
    monkeypatch.setattr(kl, "_pinned_empty", lambda shape, dtype: torch.empty(shape, dtype=dtype))
    signals = kl._SignalWords(3)
    slots = [signals.take() for _ in range(4)]
    assert [s for s, _ in slots] == [0, 1, 2, 3]
    ptrs = [w.data_ptr() for _, w in slots]
    assert all(b - a == 8 * ds.STRIDE_WORDS for a, b in zip(ptrs, ptrs[1:])), ptrs
    assert 8 * ds.STRIDE_WORDS >= 128, "two layers on one cache line is false sharing on every signal"
    with pytest.raises(RuntimeError, match="out of signal slots"):
        signals.take()


def test_the_poller_builds_the_same_kt_task_the_host_func_node_does(monkeypatch):
    """🔴 Anti-drift.

    ``_poll_run`` repeats ``_kt_packed_forward``'s ``forward_task`` call
    literally rather than calling a shared helper, so that instrumenting or
    changing one of them cannot silently change the other's cost. The price of
    that is this test: the two must pass the same arguments, or the polling
    thread is running kt on a different description of the same buffers.
    """
    calls = []
    buf = {
        "bsz": torch.zeros(1, dtype=torch.int32),
        "pin_ids": torch.zeros(1, 8, dtype=torch.int64),
        "pin_weights": torch.zeros(1, 8),
        "pin_hidden": torch.zeros(1, 4, dtype=torch.bfloat16),
        "pin_out": torch.zeros(1, 4, dtype=torch.bfloat16),
    }
    wrapper = SimpleNamespace(
        moe=SimpleNamespace(forward_task=lambda *a: calls.append(a) or "task"),
        cpu_infer=SimpleNamespace(submit=lambda task: calls.append(("submit", task)),
                                  sync=lambda n: calls.append(("sync", n))),
    )
    monkeypatch.setattr(kl, "_assert_no_pending", lambda *a, **k: None)
    monkeypatch.setattr(kl, "report_affinity", SimpleNamespace(apply_in_callback=lambda: None))
    monkeypatch.setattr(kl.seam_timing, "RECORDER", None)
    rt = SimpleNamespace(wrapper=wrapper, layer_idx=0, _packed={1: buf})
    kl.KTLayerOffload._poll_run(rt, 1)
    kl._kt_packed_forward((wrapper, buf, 0))
    task_args = [c for c in calls if not isinstance(c[0], str)]
    assert len(task_args) == 2, calls
    assert task_args[0] == task_args[1], f"the two call sites drifted apart: {task_args}"
    assert [c for c in calls if isinstance(c[0], str)] == [
        ("submit", "task"), ("sync", 0), ("submit", "task"), ("sync", 0),
    ]


# ---------------------------------------------------------------------------
# 5. the switch: preconditions, refused at start-up rather than at replay time
# ---------------------------------------------------------------------------


def _cfg(**kw):
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    base = {"enabled": True, "cpuinfer_threads": 8, "num_resident_experts": 100}
    cfg = KTOffloadConfig(**{**base, **kw})
    vcfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1, enable_eplb=False
        ),
        model_config=SimpleNamespace(enforce_eager=False),
        speculative_config=None,
    )
    return cfg, vcfg


@pytest.fixture
def _env(monkeypatch):
    """Set the switches in the ENVIRONMENT, not as attributes of ``envs``.

    🔴 ``envs`` resolves these through ``__getattr__``, so
    ``monkeypatch.setattr(envs, NAME, ...)`` reads the current value as the
    "old" one and, on undo, writes it back as a REAL attribute -- which shadows
    ``__getattr__`` for the rest of the session and makes every later
    ``monkeypatch.setenv`` of that variable a no-op. That is how this fixture
    silently disabled the seam-timing tests the first time it was written (they
    use ``setenv``). Anything already shadowed is removed here for the duration.
    """
    from vllm_ascend import envs
    from vllm_ascend.kt_offload import async_wait as aw
    from vllm_ascend.kt_offload import side_stream as ss

    for name, value in (
        (ds.ENV_NAME, "1"),
        (aw.ENV_NAME, "0"),
        (ss.ENV_NAME, "1"),
        ("VLLM_ASCEND_DSV41_KT_SEAM_TIMING", None),
    ):
        if name in envs.__dict__:
            monkeypatch.delattr(envs, name)
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return envs


def test_it_needs_the_layers_own_buffers(_env):
    cfg, vcfg = _cfg()
    with pytest.raises(ValueError, match="packed_transfer or"):
        cfg._validate_preconditions(vcfg)


def test_it_is_refused_together_with_p1a(_env, monkeypatch):
    from vllm_ascend.kt_offload import async_wait as aw

    monkeypatch.setenv(aw.ENV_NAME, "1")
    cfg, vcfg = _cfg(packed_transfer=True)
    with pytest.raises(ValueError, match="two ways to start the same"):
        cfg._validate_preconditions(vcfg)


def test_it_is_refused_together_with_the_seam_instrument(_env, monkeypatch):
    """🔴 The instrument measures segments inside a callback that no longer
    runs. A report about nothing is worse than no report (§4.26)."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_SEAM_TIMING", "1")
    cfg, vcfg = _cfg(packed_transfer=True)
    with pytest.raises(ValueError, match="seam"):
        cfg._validate_preconditions(vcfg)


def test_the_accepted_combinations(_env):
    for kw in ({"packed_transfer": True}, {"overlap_side_stream": True},
               {"packed_transfer": True, "overlap_side_stream": True}):
        cfg, vcfg = _cfg(**kw)
        cfg._validate_preconditions(vcfg)


def test_the_sampler_follows_the_counter_that_moves(monkeypatch, _env):
    """With P1b on, HOST_FUNC_CALLS never grows again. A sampler still reading
    it would print nothing at all -- and "every layer ran" and "no layer ran"
    look identical in that output."""
    from vllm_ascend.kt_offload import hostfunc_log

    started = {}
    monkeypatch.setattr(
        hostfunc_log, "start",
        lambda interval, counters, layers, what="host-func": started.update(counters=counters, what=what),
    )
    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG", "30")
    kl._maybe_start_host_func_log(40)
    assert started["counters"] is ds.STEPS and started["what"] == "device-signal"
    monkeypatch.setenv(ds.ENV_NAME, "0")
    started.clear()
    kl._maybe_start_host_func_log(40)
    assert started["counters"] is kl.HOST_FUNC_CALLS and started["what"] == "host-func"
