# SPDX-License-Identifier: Apache-2.0
"""P1b: the graph starts the CPU experts with a memory write, not a callback.

What changes against P1a
------------------------
P1a (``async_wait.py``) took the *waiting* off the ACL callback path: the
host-func node only submits and returns, and the device waits on an
``ExternalEvent`` a completion thread records. Measured on card that is worth
0.53 ms/token (PLAN §4.27) -- roughly the ``F`` leg, "callback returns, device
is told". What it cannot touch is what stays in front of the node:

* ``A``  -- the device reaching the node and the callback being *dispatched*
  (report delivered, GIL taken, argument tuple built);
* 🔴 the **35.75 us of device idle in front of every ``HOSTFUNC_CALLBACK``**
  (§4.22 F1), which is specific to that node type -- ``DAVID_EVENT_WAIT``,
  ``MEM_WAIT_VALUE``, ``MEMCPY_ASYNC`` and ``EVENT_RECORD`` all show a
  preceding gap of 0.000 us.

So P1b takes the host-func node out of the graph altogether. The layer becomes::

    D2H(inputs) -> D2H(8-byte step counter) -> ExternalEvent.wait -> reset -> H2D

and a **resident polling thread** watches the counter word. When it reaches
this step's number the thread runs kt for that layer and records the event.
Nothing in the graph is a host task any more: the two copies and the event
wait are ordinary device nodes, which is exactly the set of node types the
profile shows with no idle gap in front of them.

The ordering that makes it safe
-------------------------------
🔴 The counter's D2H is a **separate copy on the same stream, enqueued after**
the input copy. Same-stream tasks run in order, so "the counter arrived"
implies "the inputs arrived". It must NOT be appended to the input copy as a
tail field: the write order *inside* one DMA is not specified, and a counter
that lands before the payload would start kt on the previous step's routing --
faster, wrong, and silent, which is the failure mode this whole line of work
keeps meeting.

The same rule applies to everything else the polling thread reads (§4.25): it
must all be written before the counter write. Today that is only the pinned
input buffer, and the buffer is per layer, so the only writer is the copy in
front of the counter.

The counter itself is one device scalar, incremented once per replay by the
first offloaded layer of the capture (``StepCounter``). One add per step, not
per layer, and every layer's word must carry the *same* number: a layer whose
word is off by one is a step that was missed or replayed twice, and the poller
says so instead of quietly computing the wrong layer's experts.

Why a C spin
------------
See ``poll_native.c``: a Python thread that spins holds the GIL between checks.
The wait is a single ``ctypes`` call, which releases the GIL for its whole
duration; only the per-layer work (build kt's task, submit, sync, record)
happens in Python, which is exactly what P1a's completion thread already did.

Failure modes and what this module does about them
--------------------------------------------------
* **The event is never recorded** -> the device waits for ever. Everything
  below that owns a task records the event on every exit path, including a kt
  that raised, exactly as ``async_wait.Completer`` does.
* 🔴 **The signal never arrives** (a capture without the counter bump, say) ->
  the poller logs, loudly and repeatedly, and **keeps waiting**. It does NOT
  release the device on a timeout: there is no way to tell "idle service" from
  "lost signal" from inside this thread, and a spurious record would both
  release a wait nobody armed and leave the *next* replay's wait pre-signalled,
  i.e. turn a visible hang into silent stale output on every later step. A hang
  with a message naming the layer and the step is the better failure.
* **A step was missed or seen twice** -> the word does not equal the expected
  step. The poller logs it, counts it, runs this layer anyway (the pinned
  buffer does hold this step's inputs -- it is the bookkeeping that slipped)
  and resynchronises. The on-card cases judge the counter.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import tempfile
import threading
from pathlib import Path

from vllm.logger import logger

ENV_NAME = "VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL"

# Padding between two signal words, in 8-byte units. One 128-byte line each:
# this thread scans every slot while the device DMAs into one of them, and two
# layers sharing a line would mean every write invalidating the line the spin
# is reading for the other.
STRIDE_WORDS = 16

# layer_idx -> signals consumed (one per replay per layer). The P1b counterpart
# of layer.HOST_FUNC_CALLS, which stays at 0 with this switch on because there
# is no host-func node left to count. Kept separate on purpose: a counter that
# silently changes meaning is how an instrument ends up measuring nothing (§4.26).
STEPS: dict[int, int] = {}
# layer_idx -> ExternalEvents recorded. Must equal STEPS; a layer whose signals
# grow while its records do not is a device about to wait for ever.
RECORDS: dict[int, int] = {}
# (layer_idx, what) for everything the poller swallowed so the device could go
# on. Capped like async_wait.ERRORS; the log keeps them all.
ERRORS: list[tuple[int, str]] = []
MAX_ERRORS = 200


def _note_error(layer_idx: int, what: str) -> None:
    if len(ERRORS) < MAX_ERRORS:
        ERRORS.append((int(layer_idx), what))


def enabled_by_env() -> bool:
    """Whether the operator explicitly asked for the device-signal path.

    Read from ``envs`` when asked rather than cached, like
    ``side_stream.enabled_by_env``: the refusal names the variable, so the
    answer has to come from the environment the service was started with.
    """
    from vllm_ascend import envs

    return getattr(envs, ENV_NAME, "0") == "1"


# -- the native waiter -------------------------------------------------------

_SOURCE = Path(__file__).with_name("poll_native.c")
_LIB = None
_LIB_LOCK = threading.Lock()
_ABI = 3


def _cache_dirs() -> list[Path]:
    """Where the built .so may go, best first.

    Several candidates because the answer is a property of the *mount*, not of
    the path: ``/dev/shm`` (which this project exports as ``TMPDIR``) is often
    ``noexec``, and a .so on a noexec mount fails at ``dlopen``, not at
    ``write``. Each candidate is tried end to end -- compile, load, call -- so
    "it is not executable here" is found at start-up.
    """
    from vllm_ascend import envs

    out = []
    override = getattr(envs, "VLLM_ASCEND_DSV41_KT_POLL_CACHE", "")
    if override:
        out.append(Path(override))
    out.append(Path.home() / ".cache" / "vllm_ascend_kt_offload")
    out.append(Path("/tmp") / f"vllm_ascend_kt_offload_{os.getuid()}")  # noqa: S108
    out.append(Path(tempfile.gettempdir()) / f"vllm_ascend_kt_offload_{os.getuid()}")
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def _compile(src: str, out_path: Path, cc: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(f".{os.getpid()}.tmp")
    cmd = [cc, "-O2", "-fPIC", "-shared", "-x", "c", "-o", str(tmp), "-"]
    proc = subprocess.run(cmd, input=src.encode(), capture_output=True, check=False)  # noqa: S603
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{' '.join(cmd)} failed rc={proc.returncode}: {proc.stderr.decode()[:500]}")
    # Atomic publish: two workers starting at once must never dlopen a file
    # another process is still writing.
    tmp.replace(out_path)


def load_native(force: bool = False):
    """Compile (once) and load the spin loop. Returns the ctypes library.

    Built at start-up rather than shipped, because this tree is run from a
    worktree on PYTHONPATH and has no build step of its own. The file name
    carries the source hash, so an edit to ``poll_native.c`` can never be
    served from a stale .so.
    """
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None and not force:
            return _LIB
        src = _SOURCE.read_text()
        tag = hashlib.sha256(src.encode()).hexdigest()[:16]
        cc = os.environ.get("CC", "cc")
        errors = []
        for d in _cache_dirs():
            so = d / f"kt_poll_{tag}.so"
            try:
                if not so.exists():
                    _compile(src, so, cc)
                lib = ctypes.CDLL(str(so))
                lib.kt_poll_abi.restype = ctypes.c_long
                if lib.kt_poll_abi() != _ABI:
                    raise RuntimeError(f"{so} reports ABI {lib.kt_poll_abi()}, this module wants {_ABI}")
                lib.kt_wait_signal.restype = ctypes.c_long
                lib.kt_wait_signal.argtypes = [
                    ctypes.c_void_p,  # words
                    ctypes.c_void_p,  # expected
                    ctypes.c_long,  # n
                    ctypes.c_long,  # stride_words
                    ctypes.c_long,  # timeout_ns
                    ctypes.c_long,  # hot_ns
                    ctypes.c_long,  # sleep_ns
                    ctypes.c_void_p,  # out[2]: scans, ns at the match
                ]
                lib.kt_read_signal.restype = ctypes.c_uint64
                lib.kt_read_signal.argtypes = [ctypes.c_void_p, ctypes.c_long, ctypes.c_long]
                lib.kt_poke_after.restype = ctypes.c_uint64
                lib.kt_poke_after.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_long,
                    ctypes.c_long,
                    ctypes.c_uint64,
                    ctypes.c_long,
                ]
                _LIB = lib
                logger.info("kt_offload device signal: polling helper built and loaded from %s", so)
                return _LIB
            except BaseException as exc:  # noqa: BLE001
                errors.append(f"{d}: {exc!r}")
        raise RuntimeError(
            "kt_offload device signal: could not build or load the polling helper "
            f"({_SOURCE}). A Python thread cannot do this spin without holding the GIL, so there is no "
            f"fallback and the switch refuses to start. Tried: {'; '.join(errors)}"
        )


class Waiter:
    """The GIL-free wait, over a fixed words/expected pair.

    ``expected`` is a mutable byte buffer the caller writes step numbers into,
    so a wait that is already running sees the update on its next scan; the
    poller only ever changes it between waits, but the property is what makes
    the buffer, rather than an argument list, the right shape here.
    """

    def __init__(self, words_ptr: int, n: int, stride_words: int = STRIDE_WORDS, lib=None):
        self._lib = lib if lib is not None else load_native()
        self._words = ctypes.c_void_p(words_ptr)
        self._n = int(n)
        self._stride = int(stride_words)
        self.expected = (ctypes.c_uint64 * self._n)()
        self._out = (ctypes.c_uint64 * 2)()
        self.total_scans = 0
        # CLOCK_MONOTONIC ns at which the last wait saw its word, taken in C.
        # The Python statement after ``wait`` cannot stand in for it: coming
        # back from a foreign call means re-acquiring the GIL first.
        self.seen_ns = 0

    def set_expected(self, slot: int, value: int) -> None:
        self.expected[slot] = int(value)

    def wait(self, timeout_ns: int, hot_ns: int, sleep_ns: int) -> int:
        i = self._lib.kt_wait_signal(
            self._words,
            ctypes.cast(self.expected, ctypes.c_void_p),
            self._n,
            self._stride,
            int(timeout_ns),
            int(hot_ns),
            int(sleep_ns),
            ctypes.cast(self._out, ctypes.c_void_p),
        )
        self.total_scans += int(self._out[0])
        self.seen_ns = int(self._out[1])
        return int(i)

    def read(self, slot: int) -> int:
        return int(self._lib.kt_read_signal(self._words, slot, self._stride))

    def poke_after(self, slot: int, value: int, delay_ns: int) -> int:
        """Tests only: write a slot from C after a delay; returns the write's ns."""
        return int(self._lib.kt_poke_after(self._words, slot, self._stride, int(value), int(delay_ns)))


# -- the criteria (pure, so a test can drive every one of them red) ----------


def signal_ops_ok(ops) -> dict:
    """The shape of one layer's captured CPU-expert section, as a check.

    ``ops`` is ``(kind, what)`` in enqueue order, the same vocabulary
    ``async_wait.ext_event_ops_ok`` uses plus ``("signal", (layer, bs))`` for
    the counter's D2H:

      ``("bump", layer)``      the step counter's increment (the lowest signal
                               layer only; which layer that is, is checked by
                               the caller, which is the only place that knows)
      ``("d2h", bs)``          the input copy that feeds kt
      ``("signal", key)``      the 8-byte counter copy -- the start signal
      ``("ext_wait", ev)``     ``event.wait(stream)``  -- a graph node
      ``("ext_reset", ev)``    ``event.reset(stream)`` -- a graph node
      ``("h2d", bs)``          the output copy back to the device
      ``("hostfunc", layer)``  a host-func node -- 🔴 must NOT be here any more

    🔴 The order ``d2h`` -> ``signal`` is the whole safety argument: same
    stream, so the counter cannot land before the inputs. Reversed, the poller
    starts kt on the previous step's routing and nothing raises.
    """
    problems = []
    kinds = [k for k, _ in ops]
    if "hostfunc" in kinds:
        problems.append(
            f"{kinds.count('hostfunc')} host-func node(s) still in the graph; P1b exists to remove them "
            "(the 35.75 us of device idle in front of one is the target, PLAN §4.22 F1)"
        )
    if kinds.count("signal") != 1:
        problems.append(f"expected exactly one counter copy (the start signal), found {kinds.count('signal')}")
    waits = [w for k, w in ops if k == "ext_wait"]
    resets = [w for k, w in ops if k == "ext_reset"]
    if len(waits) != 1:
        problems.append(f"expected exactly one ExternalEvent wait in the graph, found {len(waits)}")
    if len(resets) != 1:
        problems.append(
            f"expected exactly one ExternalEvent reset in the graph, found {len(resets)}; torch_npu says a "
            "reused ExternalEvent needs one (streams.py:214-217)"
        )
    if waits and resets and waits[0] is not resets[0]:
        problems.append("the graph waits on one event and resets another")
    if any(k == "host_record" for k, _ in ops):
        problems.append(
            "a host-side event.record is among the captured ops; the record is what the polling thread does "
            "at replay time and must not be a graph node (it would release the wait it precedes)"
        )
    if kinds.count("bump") > 1:
        problems.append(
            f"{kinds.count('bump')} step-counter increments in one layer; one per replay is what makes "
            "every layer's word carry the same number"
        )
    i = {k: n for n, (k, _) in enumerate(ops)}
    if "bump" in i and "signal" in i and i["signal"] < i["bump"]:
        problems.append("the counter is copied to the host before it is incremented: the word is one step behind")
    if "signal" in i and "d2h" in i and i["d2h"] > i["signal"]:
        problems.append(
            "🔴 the counter copy is enqueued BEFORE the input copy: 'the counter arrived' would no longer "
            "imply 'the inputs arrived', and the polling thread would run kt on the previous step's routing"
        )
    if "signal" in i and "ext_wait" in i and i["ext_wait"] < i["signal"]:
        problems.append(
            "the graph waits on the event BEFORE the copy that signals the poller; nothing has been handed "
            "over yet, so the first replay would wait for ever"
        )
    if "ext_wait" in i and "ext_reset" in i and i["ext_reset"] < i["ext_wait"]:
        problems.append("the event is reset before it is waited on, which clears the signal it waits for")
    if "h2d" in i and "ext_wait" in i and i["h2d"] < i["ext_wait"]:
        problems.append("the output is copied back before the wait: it would read the previous step's value")
    return {
        "name": "device_signal_ops_in_graph",
        "ok": not problems,
        "detail": problems,
        "why": (
            "the counter copy behind the input copy is what makes 'signal arrived' mean 'inputs arrived', "
            "and the ExternalEvent wait is the only thing holding the H2D back"
        ),
    }


def step_ok(layer_idx: int, got: int, expected: int) -> dict:
    """Is this the step the poller was waiting for?

    ``got < expected`` cannot come out of the waiter (it returns on ``>=``), so
    the only mismatch is a word that ran ahead: replays whose signal this thread
    never consumed. It is reported rather than fixed silently, because "the
    poller and the graph disagree about which step this is" is precisely the
    state in which a layer's kt run could belong to another step.
    """
    missed = int(got) - int(expected)
    return {
        "name": "signal_is_this_step",
        "ok": missed == 0,
        "detail": {"layer": int(layer_idx), "got": int(got), "expected": int(expected), "missed": missed},
        "why": "every replay increments the counter exactly once, so a gap is a step whose signal was lost",
    }


def waiting_state(expected: dict[int, int], words: dict[int, int]) -> dict:
    """Tell "the service is idle" apart from "a step never finished".

    🔴 The polling thread cannot ask anyone whether work is coming, so a wait
    that goes on for ever looks the same from in here whether the service has
    no requests or a signal was lost. These two tables do tell them apart:

    * every layer waiting for the SAME step, and each one's word holding the
      step before it -> the last step completed and nothing new has started.
      That is an idle service, and it is not worth a warning.
    * layers waiting for DIFFERENT steps -> some consumed step N and others are
      still waiting for it, i.e. a replay stopped half way through the model.
      That one is a stalled step, and the layers still waiting are named.

    Returns the classification plus the layers that are behind, so the log line
    can point at them instead of dumping 40 numbers.
    """
    if not expected:
        return {"stalled": False, "step": None, "behind": [], "layers": 0}
    steps = sorted(set(expected.values()))
    ahead = max(steps)
    behind = sorted(layer for layer, want in expected.items() if want < ahead)
    return {
        "stalled": bool(behind),
        "step": ahead,
        "behind": behind,
        "layers": len(expected),
        "words": {layer: words.get(layer) for layer in behind[:6]},
    }


def records_match_signals(steps_delta: dict[int, int], record_delta: dict[int, int]) -> dict:
    """Signals consumed and events recorded pair up layer by layer."""
    bad = {
        i: (steps_delta.get(i, 0), record_delta.get(i, 0))
        for i in set(steps_delta) | set(record_delta)
        if steps_delta.get(i, 0) != record_delta.get(i, 0)
    }
    return {
        "name": "records_match_signals",
        "ok": not bad,
        "detail": bad,
        "why": "each consumed signal ends in exactly one record; a missing one leaves the device waiting",
    }


def signals_per_replay_ok(before: dict[int, int], after: dict[int, int], replays: int, layers) -> dict:
    """Every layer was signalled exactly once per replay."""
    bad = {}
    for i in layers:
        delta = after.get(i, 0) - before.get(i, 0)
        if delta != replays:
            bad[i] = delta
    return {
        "name": "signal_one_per_replay",
        "ok": not bad,
        "detail": {"expected": replays, "wrong": bad},
        "why": "a layer that is not signalled once per replay either hangs or never ran its CPU experts",
    }


# -- the polling thread ------------------------------------------------------


class Entry:
    """One (layer, batch size): its slot in the signal array, work and event."""

    __slots__ = ("bs", "event", "layer_idx", "run", "slot")

    def __init__(self, layer_idx: int, bs: int, slot: int, run, event):
        self.layer_idx = int(layer_idx)
        self.bs = int(bs)
        self.slot = int(slot)
        self.run = run
        self.event = event


class Poller:
    """One thread per process: see the step number, run kt, release the device.

    Injected dependencies, like ``async_wait.Completer``, so the CPU tests can
    drive the whole protocol with no device and no native library:
    ``set_device`` gives the thread its ACL context, ``make_stream`` the stream
    the record goes on (created *in* the thread, after the context, before
    ``start()`` returns), and ``make_waiter`` the object whose ``wait`` blocks.
    """

    # Spin without sleeping for this long after entering a wait, then fall back
    # to short sleeps. A decode step is ~30 ms, so a generating service never
    # leaves the hot window; an idle one drops out of it after the first
    # timeout and stops burning a core.
    HOT_NS = 60_000_000
    SLEEP_NS = 200_000
    TIMEOUT_NS = 200_000_000
    # Consecutive timeouts before the wait goes cold (no spinning at all).
    # 5 x 200 ms = 1 s of nothing happening.
    COLD_AFTER = 5
    # How often the waiting state is looked at, and how long a wait has to last
    # before each kind of it is worth a line. A STALLED step (layers waiting for
    # different steps) is a fault and is named quickly; an idle service is
    # normal and gets one line every few minutes, only so that "the thread is
    # alive" is visible at all.
    # 🔴 The first version logged the idle case after 5 timeouts = 1 s, which
    # put 137 WARNINGs into one on-card logprob arm (2026-09-23): every gap
    # between two requests produced one. A warning that fires in normal
    # operation stops being read.
    LOG_CHECK_EVERY = 25  # 5 s
    STALL_LOG_AFTER = 25  # 5 s
    STALL_LOG_EVERY = 150  # 30 s
    IDLE_LOG_EVERY = 1500  # 300 s

    def __init__(self, set_device, make_stream, make_waiter, *, name: str = "kt-offload-poller"):
        self._set_device = set_device
        self._make_stream = make_stream
        self._make_waiter = make_waiter
        self._name = name
        self._entries: list[Entry] = []
        self._expected: dict[int, int] = {}  # layer_idx -> the step it is waiting for
        self._lock = threading.Lock()
        self._gen = 0
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._waiter = None
        self._startup_error: BaseException | None = None
        self.waits = 0
        self.timeouts = 0

    # -- registration ------------------------------------------------------

    def register(self, entry: Entry) -> None:
        """Add a (layer, bs). Only ever called at weight-load time."""
        with self._lock:
            self._entries.append(entry)
            self._expected.setdefault(entry.layer_idx, 1)
            self._gen += 1

    @property
    def entries(self) -> list[Entry]:
        with self._lock:
            return list(self._entries)

    def expected_of(self, layer_idx: int) -> int:
        with self._lock:
            return self._expected.get(int(layer_idx), 1)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the thread and wait until its context, stream and waiter exist.

        Synchronous for the same reason ``Completer.start`` is: a thread that
        cannot take an ACL context has to be a start-up error, not a hang at
        the first replay.
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                "kt_offload device signal: the polling thread could not start "
                f"({self._startup_error!r}). Without it nothing would ever record the events the graph "
                "waits on, and the first replay would hang instead of failing here."
            ) from self._startup_error

    def stop(self) -> None:
        """Tests only; production keeps the thread for the life of the process."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=30)
        self._thread = None

    # -- the thread --------------------------------------------------------

    def _run(self) -> None:
        try:
            self._set_device()
            self._stream = self._make_stream()
            self._waiter = self._make_waiter()
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        gen = -1
        snapshot: list[Entry] = []
        by_slot: dict[int, Entry] = {}
        idle = 0
        while not self._stop.is_set():
            with self._lock:
                if gen != self._gen:
                    gen, snapshot = self._gen, list(self._entries)
                    # 🔴 Keyed by SLOT, never by position in this list. The
                    # waiter returns an index into the pinned array, and the
                    # two are not the same number: the self-test takes slot 0,
                    # so entry 0 lives at slot 1. Reading one as the other runs
                    # the wrong layer's experts and releases the wrong layer's
                    # H2D -- with entirely plausible-looking numbers.
                    by_slot = {e.slot: e for e in snapshot}
                    for e in snapshot:
                        self._waiter.set_expected(e.slot, self._expected[e.layer_idx])
            if not snapshot:
                self._stop.wait(0.05)
                continue
            self.waits += 1
            hot = 0 if idle >= self.COLD_AFTER else self.HOT_NS
            i = self._waiter.wait(self.TIMEOUT_NS, hot, self.SLEEP_NS)
            if i < 0:
                self.timeouts += 1
                idle += 1
                if idle % self.LOG_CHECK_EVERY == 0:
                    self._log_waiting(snapshot, idle)
                continue
            idle = 0
            entry = by_slot.get(i)
            if entry is None:
                # Only reachable if something set an expected step for a slot
                # that belongs to no entry, i.e. the two tables disagree.
                _note_error(-1, f"signal on unregistered slot {i}")
                logger.critical(
                    "kt_offload device signal: slot %d was signalled but belongs to no layer; the slot "
                    "table and the entry table disagree, so some layer's experts are not being run.",
                    i,
                )
                self._waiter.set_expected(i, 0)
                continue
            self._handle(entry)

    def _log_waiting(self, snapshot: list[Entry], idle: int) -> None:
        """🔴 The only thing standing between a lost signal and a silent hang.

        Deliberately not an auto-release: recording an event nobody is waiting
        on leaves the NEXT replay's wait pre-signalled, i.e. it converts a hang
        into wrong output on every step after it. What this does instead is say
        WHICH of the two silences it is (``waiting_state``) and, for the one
        that is a fault, name the layers that are behind.
        """
        by_layer: dict[int, int] = {}
        for e in snapshot:
            by_layer[e.layer_idx] = max(by_layer.get(e.layer_idx, 0), self._waiter.read(e.slot))
        want = {layer: self.expected_of(layer) for layer in sorted(by_layer)}
        state = waiting_state(want, by_layer)
        seconds = idle * self.TIMEOUT_NS / 1e9
        if state["stalled"]:
            if idle >= self.STALL_LOG_AFTER and idle % self.STALL_LOG_EVERY < self.LOG_CHECK_EVERY:
                logger.warning(
                    "kt_offload device signal: a replay has been stopped half way through the model for "
                    "%.0fs. %d of %d layers are still waiting for step %d (%s; their words hold %s): the "
                    "counter copy of those layers is not reaching the host, and every one of them is parked "
                    "on an ExternalEvent nobody will record. This thread will NOT record one to break the "
                    "deadlock -- that would release the next replay's wait as well, turning a hang into "
                    "silently stale output on every step after it.",
                    seconds,
                    len(state["behind"]),
                    state["layers"],
                    state["step"],
                    state["behind"][:8],
                    state["words"],
                )
        elif idle % self.IDLE_LOG_EVERY < self.LOG_CHECK_EVERY:
            logger.info(
                "kt_offload device signal: idle for %.0fs -- every layer completed step %d and no replay has "
                "started since. The polling thread is alive and cold (it stops spinning after %.1fs).",
                seconds,
                (state["step"] or 1) - 1,
                self.COLD_AFTER * self.TIMEOUT_NS / 1e9,
            )

    def _handle(self, e: Entry) -> None:
        """One layer's turn: check the step, run kt, release the device."""
        got = self._waiter.read(e.slot)
        with self._lock:
            expected = self._expected[e.layer_idx]
        rep = step_ok(e.layer_idx, got, expected)
        if not rep["ok"]:
            _note_error(e.layer_idx, f"step mismatch {rep['detail']}")
            logger.error(
                "kt_offload device signal: layer %d was waiting for step %d and its word holds %d (%d "
                "missed). The pinned inputs ARE this step's -- the copy in front of the counter saw to "
                "that -- so this layer runs anyway and the expected step is resynchronised; but a missed "
                "signal means some replay released its H2D without this thread having run kt for it.",
                e.layer_idx,
                expected,
                got,
                rep["detail"]["missed"],
            )
        try:
            e.run(e.bs)
        except BaseException as exc:  # noqa: BLE001
            _note_error(e.layer_idx, repr(exc))
            logger.error(
                "kt_offload device signal: layer %d's CPU experts raised: %r. The CPU-expert output of this "
                "step is NOT this step's -- the device reads whatever the pinned buffer held. The event is "
                "recorded anyway; a device left waiting would wedge the engine with no message.",
                e.layer_idx,
                exc,
            )
        finally:
            STEPS[e.layer_idx] = STEPS.get(e.layer_idx, 0) + 1
            with self._lock:
                self._expected[e.layer_idx] = int(got) + 1
                for other in self._entries:
                    if other.layer_idx == e.layer_idx:
                        self._waiter.set_expected(other.slot, int(got) + 1)
            try:
                e.event.record(self._stream)
                RECORDS[e.layer_idx] = RECORDS.get(e.layer_idx, 0) + 1
            except BaseException as exc:  # noqa: BLE001
                _note_error(e.layer_idx, f"record failed: {exc!r}")
                logger.critical(
                    "kt_offload device signal: recording layer %d's ExternalEvent FAILED (%r). The graph is "
                    "waiting on it and nothing else will ever signal it: this replay will not finish.",
                    e.layer_idx,
                    exc,
                )
