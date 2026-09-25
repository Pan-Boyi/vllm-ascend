# SPDX-License-Identifier: Apache-2.0
"""Seam timing for the graph-mode host-func path.

Per layer, the device waits measurably longer in NOTIFY_WAIT than kt's own
phase timing (``KT_MOE_PHASE_TIMING=1``) accounts for. This module timestamps
the seams a host thread can see, so that difference stops being one number.

The segments, in the order they happen inside one host-func callback
(``layer._kt_host_forward`` / ``layer._kt_packed_forward``):

  pre       callback entry -> the Python pre-checks are done (_assert_no_pending)
  kt_pre    -> kt's own Python prologue before it enqueues
              (_check_qlen_fits_cpp_buffers, get_buffer, forward_task build)
  submit    CPUInfer.submit(): handing the task to the worker queue
  between   submit returns -> sync is entered (kt's deferred bookkeeping)
  sync      CPUInfer.sync(): worker wakeup + kt compute (kt's own D) + the
            completion path (cross-NUMA reduce, write-back into pinned output)
  kt_tail   kt's Python epilogue after sync
  post      -> callback exit (trailing _assert_no_pending, counter)
  kt        kt_pre..kt_tail as one number; the only one available when the
            CPUInfer probe could not be installed (``split`` is then false)
  cb        callback entry -> callback exit: T_cb
  gap       this callback's entry minus the previous entry *for the same
            layer* -- the host-side view of one device step's period

Not here, because no host clock can see them: the device-side arrival at the
host-func node (A), its dispatch to this thread, and the notify back to the
device (F). They are obtained by subtraction against a profile of the SAME run
-- same service, same steps:

    A + F_device = NOTIFY_WAIT(per layer) - cb(per layer)
    A = entry_realtime - HOSTFUNC_CALLBACK.ts
    F = (NOTIFY_WAIT.ts + NOTIFY_WAIT.dur) - exit_realtime

The per-callback form works because a profile's task timestamps are
CLOCK_REALTIME microseconds sharing a base with this host's clock. The segments
above stay on ``perf_counter_ns`` (monotonic); the two realtime stamps are extra,
taken first thing at entry and last thing in ``record()``, and written out raw --
one ``[kt-seam-rt]`` line per layer per sample -- because the join with the
profile is per callback and a median cannot be joined. The offline tool does the
join and refuses the A/F columns when A + F does not reconcile with
NOTIFY_WAIT - cb.

The callback is inside a graph node, so the device is blocked on it: on the hot
path there is no I/O, no allocation, no dict insert, no locking -- only
``time.perf_counter_ns()`` calls and integer stores into a list sized at
startup. Aggregation and logging happen on a separate sampler thread, the same
shape as hostfunc_log.py.

Off unless ``VLLM_ASCEND_DSV41_KT_SEAM_TIMING`` is set to a sampling interval in
seconds; with it off ``RECORDER`` stays None and the callbacks take the branch
they always took -- not one extra device operation, not one changed byte of
output.
"""

from __future__ import annotations

import atexit
import json
import threading
from time import CLOCK_REALTIME as _CLOCK_REALTIME
from time import clock_gettime_ns as _realtime_ns

# The ``vllm`` logger, not ``logging.getLogger(__name__)``: see the note in
# hostfunc_log.py -- a ``vllm_ascend.*`` logger loses INFO in the worker.
from vllm.logger import logger

# Segment layout of one record. Kept as a flat tuple + index constants so the
# hot path stores into fixed offsets and never touches a dict.
SEGMENTS = (
    "pre",
    "kt_pre",
    "submit",
    "between",
    "sync",
    "kt_tail",
    "post",
    "kt",
    "cb",
    "gap",
)
NSEG = len(SEGMENTS)
(
    S_PRE,
    S_KT_PRE,
    S_SUBMIT,
    S_BETWEEN,
    S_SYNC,
    S_KT_TAIL,
    S_POST,
    S_KT,
    S_CB,
    S_GAP,
) = range(NSEG)

# The segments that must add up to ``cb`` (``kt`` is a roll-up of the middle
# ones and ``gap`` is not part of this callback at all).
ADDITIVE = ("pre", "kt_pre", "submit", "between", "sync", "kt_tail", "post")
# Without the CPUInfer probe only these are filled.
ADDITIVE_LUMPED = ("pre", "kt", "post")

# Timestamp scratch, written by the callback and by the CPUInfer probe that runs
# inside it. One list, reused: allocating a tuple per callback is exactly the
# kind of cost this module exists to measure. Sharing is safe because host-func
# callbacks are delivered on the one ACL report/subscriber thread, serially --
# the property the whole captured path already relies on (one serial FIFO
# worker, one callback at a time). If that stops being true the records
# interleave, which is what ``record()`` drops as non-monotonic timestamps.
#
# ``T_ENTER_RT`` is the one CLOCK_REALTIME stamp in here (ns since the epoch,
# ``time.clock_gettime_ns(time.CLOCK_REALTIME)``); every other slot is
# ``perf_counter_ns``. The two clocks are never subtracted from each other.
TS = [0] * 8
T_ENTER, T_PRE, T_SUBMIT_IN, T_SUBMIT_OUT, T_SYNC_IN, T_SYNC_OUT, T_KT_OUT, T_ENTER_RT = range(8)

# One ring slot: the NSEG segment durations (perf_counter ns), then the
# callback's entry and exit in absolute CLOCK_REALTIME ns (0 = not stamped).
R_ENTER_RT = NSEG
R_EXIT_RT = NSEG + 1
ROW = NSEG + 2

# The one recorder of this process, or None (the feature is off). Read on the
# hot path as a plain module global: no function call, no env lookup.
RECORDER: SeamRecorder | None = None

_MONITOR: threading.Thread | None = None
_STOP = threading.Event()


class SeamRecorder:
    """Per-layer ring buffer of segment durations, in nanoseconds.

    Sized once, at startup: ``num_layers * capacity * ROW`` integers in one flat
    list, so a record is twelve stores into slots that already exist (ten
    segments + the two CLOCK_REALTIME stamps).
    """

    def __init__(self, num_layers: int, capacity: int = 1024):
        self.num_layers = int(num_layers)
        self.capacity = int(capacity)
        self.buf = [0] * (self.num_layers * self.capacity * ROW)
        self.count = [0] * self.num_layers
        self.dropped = [0] * self.num_layers
        self._prev_enter = [0] * self.num_layers
        # True once the CPUInfer probe is in place, i.e. submit/sync are split
        # out of kt's call. False = only ``pre``/``kt``/``post`` are filled.
        self.split = False

    def record(self, layer_idx: int, ts: list, t_exit: int) -> None:
        """Store one callback's segments. Hot path: no allocation, no I/O."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        t0 = ts[T_ENTER]
        t1 = ts[T_PRE]
        t_kt = ts[T_KT_OUT]
        if not (t0 <= t1 <= t_kt <= t_exit):
            self.dropped[layer_idx] += 1
            return
        n = self.count[layer_idx]
        base = (layer_idx * self.capacity + n % self.capacity) * ROW
        buf = self.buf
        buf[base + R_ENTER_RT] = ts[T_ENTER_RT]
        buf[base + S_PRE] = t1 - t0
        buf[base + S_KT] = t_kt - t1
        buf[base + S_POST] = t_exit - t_kt
        buf[base + S_CB] = t_exit - t0
        prev = self._prev_enter[layer_idx]
        buf[base + S_GAP] = t0 - prev if prev else 0
        self._prev_enter[layer_idx] = t0
        s_in, s_out, y_in, y_out = ts[T_SUBMIT_IN], ts[T_SUBMIT_OUT], ts[T_SYNC_IN], ts[T_SYNC_OUT]
        if s_in and t1 <= s_in <= s_out <= y_in <= y_out <= t_kt:
            buf[base + S_KT_PRE] = s_in - t1
            buf[base + S_SUBMIT] = s_out - s_in
            buf[base + S_BETWEEN] = y_in - s_out
            buf[base + S_SYNC] = y_out - y_in
            buf[base + S_KT_TAIL] = t_kt - y_out
        else:
            # No probe (or a callback that did not go through it): the middle is
            # one number. Zero, not a guess -- a zero segment next to a filled
            # ``kt`` reads as "not measured here".
            buf[base + S_KT_PRE] = 0
            buf[base + S_SUBMIT] = 0
            buf[base + S_BETWEEN] = 0
            buf[base + S_SYNC] = 0
            buf[base + S_KT_TAIL] = 0
        # Last clock read of the callback, taken after this function's own
        # stores so that F starts where our Python really ends. Stored before
        # the count moves: the sampler reads only slots below ``count``.
        buf[base + R_EXIT_RT] = _realtime_ns(_CLOCK_REALTIME)
        self.count[layer_idx] = n + 1

    def drain(self, since: list[int]) -> tuple[dict[int, list[list[int]]], list[int]]:
        """Records added since the given per-layer counts, plus the new counts.

        Only whole ring slots are read; if a layer produced more than
        ``capacity`` records in one interval the oldest are gone and the report
        says how many were lost.
        """
        out: dict[int, list[list[int]]] = {}
        now = list(self.count)
        for layer in range(self.num_layers):
            new = now[layer] - since[layer]
            if new <= 0:
                continue
            lost = max(0, new - self.capacity)
            take = min(new, self.capacity)
            rows = []
            for k in range(now[layer] - take, now[layer]):
                base = (layer * self.capacity + k % self.capacity) * ROW
                rows.append(self.buf[base : base + ROW])
            out[layer] = rows
            if lost:
                self.dropped[layer] += lost
        return out, now


def _median(xs: list[int]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(xs[mid]) if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _p90(xs: list[int]) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    return float(xs[min(len(xs) - 1, int(0.9 * len(xs)))])


_MIDDLE_IDX = (S_KT_PRE, S_SUBMIT, S_BETWEEN, S_SYNC, S_KT_TAIL)


def summarize(rows_by_layer: dict[int, list[list[int]]], split: bool, interval_s: float) -> dict:
    """Median/p90 per segment per layer, in microseconds. Pure, hence testable.

    Two per-callback fields ride along, since medians hide one bad callback:

      n_unsplit      callbacks of a ``split`` run whose five middle segments
                     were not filled (the probe did not run for them); always
                     0 when the run is not split -- there is nothing to miss.
      resid_us.max   per callback, cb - sum(additive parts) in us; the value
                     of largest magnitude, sign kept. 0.0 on a healthy run by
                     construction; an unsplit callback in a split run shows up
                     here as its whole ``kt``.
    """
    layers: dict[str, dict] = {}
    add_idx = [SEGMENTS.index(k) for k in (ADDITIVE if split else ADDITIVE_LUMPED)]
    for layer, rows in sorted(rows_by_layer.items()):
        cols = list(zip(*rows))
        med = {name: round(_median(list(cols[i])) / 1000.0, 3) for i, name in enumerate(SEGMENTS)}
        p90 = {name: round(_p90(list(cols[i])) / 1000.0, 3) for i, name in enumerate(SEGMENTS)}
        worst = 0
        n_unsplit = 0
        for r in rows:
            resid = r[S_CB] - sum(r[i] for i in add_idx)
            if abs(resid) > abs(worst):
                worst = resid
            if split and not any(r[i] for i in _MIDDLE_IDX):
                n_unsplit += 1
        layers[str(layer)] = {
            "n": len(rows),
            "med_us": med,
            "p90_us": p90,
            "n_unsplit": n_unsplit,
            "resid_us": {"max": round(worst / 1000.0, 3)},
        }
    keys = ADDITIVE if split else ADDITIVE_LUMPED
    return {
        "interval_s": round(interval_s, 3),
        "split": bool(split),
        "additive": list(keys),
        "layers": layers,
        # Summed over the layers that reported: what one token pays, in ms.
        "sum_med_ms": {
            name: round(sum(v["med_us"][name] for v in layers.values()) / 1000.0, 4)
            for name in SEGMENTS
        },
    }


def reconcile(med_us: dict[str, float], split: bool, tol_us: float = 2.0) -> dict:
    """Do the segments of one layer add up to its callback duration?

    Catches a timestamp taken in the wrong place: a mis-ordered pair either
    overlaps or leaves a hole, and both show up here as a residual that is not
    ~0. Reported, never silently absorbed.
    """
    keys = ADDITIVE if split else ADDITIVE_LUMPED
    total = sum(float(med_us.get(k, 0.0)) for k in keys)
    cb = float(med_us.get("cb", 0.0))
    residual = cb - total
    return {
        "parts_us": round(total, 3),
        "cb_us": round(cb, 3),
        "residual_us": round(residual, 3),
        "ok": abs(residual) <= tol_us,
    }


def format_record(summary: dict) -> str:
    """One structured line per sample, for offline aggregation (prof/seam.py)."""
    return "[kt-seam] " + json.dumps(summary, separators=(",", ":"), sort_keys=True)


def realtime_rows(rows_by_layer: dict[int, list[list[int]]]) -> list[dict]:
    """Per layer: every callback's absolute CLOCK_REALTIME entry/exit, and its cb.

    Units are in the key names: ``*_realtime_ns`` are ns since the epoch
    (``time.clock_gettime_ns(time.CLOCK_REALTIME)``), ``cb_ns`` the perf_counter
    duration the segments add up to; the offline tool compares the two spans,
    which is how a clock mix-up is caught. Callbacks with no realtime stamp (0)
    are left out, never written as 0.
    """
    out = []
    for layer, rows in sorted(rows_by_layer.items()):
        keep = [r for r in rows if r[R_ENTER_RT] and r[R_EXIT_RT]]
        if not keep:
            continue
        out.append(
            {
                "layer": int(layer),
                "entry_realtime_ns": [r[R_ENTER_RT] for r in keep],
                "exit_realtime_ns": [r[R_EXIT_RT] for r in keep],
                "cb_ns": [r[S_CB] for r in keep],
            }
        )
    return out


def format_realtime(entry: dict) -> str:
    """One ``[kt-seam-rt]`` line (one layer of one sample). Not a ``[kt-seam]``
    line: the aggregate parser neither reads nor miscounts it."""
    return "[kt-seam-rt] " + json.dumps(entry, separators=(",", ":"), sort_keys=True)


def install_infer_probe(wrapper) -> bool:
    """Time kt's submit/sync without touching kt's C++ or rebuilding the wheel.

    ``run_pinned_forward_sync`` is Python (kt's ``experts_base.py``): it calls
    ``self.cpu_infer.submit(task)`` then ``self.cpu_infer.sync(allow_pending)``,
    so replacing that attribute with a pass-through that timestamps the two calls
    leaves the task, the pointers, the order and the return values kt's own.

    Returns False if this build does not let us (only ``kt`` is then measured and
    the report says ``split: false``); never raises, a diagnostic that takes the
    service down being worse than a missing column.
    """
    inner = getattr(wrapper, "cpu_infer", None)
    if inner is None or isinstance(inner, _TimedCpuInfer):
        return isinstance(inner, _TimedCpuInfer)
    if not (hasattr(inner, "submit") and hasattr(inner, "sync")):
        return False
    try:
        wrapper.cpu_infer = _TimedCpuInfer(inner)
    except (AttributeError, TypeError) as exc:  # a build that forbids it
        logger.warning("kt_offload seam timing: cannot wrap cpu_infer (%r); submit/sync stay lumped", exc)
        return False
    return True


def unwrap_infer(infer):
    """The real CPUInfer behind a probe, for identity checks (layer._register_wrapper)."""
    return getattr(infer, "_kt_seam_inner", infer)


class _TimedCpuInfer:
    """Pass-through CPUInfer that timestamps submit/sync into ``TS``.

    Every wrapper shares one CPUInfer (layer._register_wrapper enforces it), so
    each wrapper gets its own probe over the *same* inner object -- which is why
    the identity check unwraps instead of comparing the probes.
    """

    def __init__(self, inner):
        self._kt_seam_inner = inner

    def submit(self, task):
        from time import perf_counter_ns

        TS[T_SUBMIT_IN] = perf_counter_ns()
        out = self._kt_seam_inner.submit(task)
        TS[T_SUBMIT_OUT] = perf_counter_ns()
        return out

    def sync(self, allow_pending):
        from time import perf_counter_ns

        TS[T_SYNC_IN] = perf_counter_ns()
        out = self._kt_seam_inner.sync(allow_pending)
        TS[T_SYNC_OUT] = perf_counter_ns()
        return out

    def __getattr__(self, name):
        # Only reached for attributes this class does not define; guard the
        # bootstrap case so a half-built probe cannot recurse.
        if name == "_kt_seam_inner":
            raise AttributeError(name)
        return getattr(self._kt_seam_inner, name)


def maybe_enable(num_layers: int, wrapper=None, split: bool = False) -> SeamRecorder | None:
    """Turn seam timing on if the env var asks; idempotent, never raises."""
    global RECORDER
    from vllm_ascend import envs

    raw = envs.VLLM_ASCEND_DSV41_KT_SEAM_TIMING
    if not raw:
        return None
    try:
        interval = float(raw)
    except ValueError:
        logger.warning("kt_offload: VLLM_ASCEND_DSV41_KT_SEAM_TIMING=%r is not a number of seconds", raw)
        return None
    if interval <= 0:
        return None
    if RECORDER is None:
        RECORDER = SeamRecorder(num_layers)
        logger.info(
            "kt_offload: seam timing on, %d layers, sampling every %.1fs "
            "(VLLM_ASCEND_DSV41_KT_SEAM_TIMING); diagnostic only, no device operation changes",
            num_layers,
            interval,
        )
    if split:
        # The caller times submit/sync itself (the packed path).
        RECORDER.split = True
    if wrapper is not None and install_infer_probe(wrapper):
        RECORDER.split = True
    start(interval)
    return RECORDER


def start(interval_s: float) -> threading.Thread | None:
    """Start the one sampler of this process (idempotent). Returns the thread.

    Daemon, and stops on the process's exit event: a watcher that outlives its
    process goes on sampling somebody else's service, and its output looks
    completely normal (hostfunc_log.py, same reasoning).
    """
    global _MONITOR
    if _MONITOR is not None or interval_s <= 0 or RECORDER is None:
        return _MONITOR
    rec = RECORDER

    def run() -> None:
        since = [0] * rec.num_layers
        while not _STOP.wait(interval_s):
            rows, since = rec.drain(since)
            if not rows:
                continue
            summary = summarize(rows, rec.split, interval_s)
            summary["dropped"] = sum(rec.dropped)
            logger.info("%s", format_record(summary))
            for entry in realtime_rows(rows):
                logger.info("%s", format_realtime(entry))

    _MONITOR = threading.Thread(target=run, name="kt-offload-seam", daemon=True)
    _STOP.clear()
    _MONITOR.start()
    atexit.register(stop)
    return _MONITOR


def stop() -> None:
    """Stop the sampler; safe to call more than once."""
    global _MONITOR
    _STOP.set()
    thread, _MONITOR = _MONITOR, None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)


def _reset_for_tests() -> None:
    """Forget the recorder and the sampler. Tests only; a service never calls it."""
    global RECORDER
    stop()
    RECORDER = None
    for i in range(len(TS)):
        TS[i] = 0
