# SPDX-License-Identifier: Apache-2.0
"""Seam timing for the graph-mode host-func path (the 107.5 us/layer target).

§4.18 measured, on one card, bs1 decode, graph mode, serial: the device waits
a median 353.6 us per layer in NOTIFY_WAIT, while kt's own phase timing
(``KT_MOE_PHASE_TIMING=1``) accounts for 246.1 us of CPU work on the slower
NUMA node. The 107.5 us/layer difference (~4.3 ms/token) is *not* kt compute,
and nothing said where it sits. This module timestamps the seams a host
thread can see, so that difference stops being one number.

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

What is deliberately NOT here: the device-side arrival at the host-func node,
its dispatch to this thread, and the notify back to the device. No host clock
can see those. They are obtained by subtraction against a same-run profile:

    A + F_device = NOTIFY_WAIT(per layer) - cb(per layer)

which is why the runbook (tools/kt_offload/oncard/RUNBOOK_seam.md) insists the
profile and these records come from the same service, same steps.

Cost discipline. The callback is inside a graph node: the device is blocked on
it. So on the hot path there is no I/O, no allocation, no dict insert, no
locking -- only ``time.perf_counter_ns()`` calls and integer stores into a
list that was sized at startup. The aggregation and the logging happen on a
separate sampler thread, the same shape as hostfunc_log.py.

Off unless ``VLLM_ASCEND_DSV41_KT_SEAM_TIMING`` is set to a sampling interval
in seconds. With it off, ``RECORDER`` stays None and the callbacks take the
same branch they always took: not one extra device operation, not one changed
byte of output, on or off.
"""

from __future__ import annotations

import atexit
import json
import threading

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

# Timestamp scratch, written by the callback and by the CPUInfer probe that
# runs inside it. One list, reused: allocating a tuple per callback is exactly
# the kind of cost this module exists to measure.
#
# It is safe to share because host-func callbacks are delivered on the one ACL
# report/subscriber thread, serially -- the same property the whole captured
# path already relies on (one serial FIFO worker, one callback at a time). If
# that ever stops being true the records interleave; ``record()`` drops a
# record whose timestamps are not monotonic, which is what that would look
# like.
TS = [0] * 8
T_ENTER, T_PRE, T_SUBMIT_IN, T_SUBMIT_OUT, T_SYNC_IN, T_SYNC_OUT, T_KT_OUT, _T_SPARE = range(8)

# The one recorder of this process, or None (the feature is off). Read on the
# hot path as a plain module global: no function call, no env lookup.
RECORDER: SeamRecorder | None = None

_MONITOR: threading.Thread | None = None
_STOP = threading.Event()


class SeamRecorder:
    """Per-layer ring buffer of segment durations, in nanoseconds.

    Sized once, at startup: ``num_layers * capacity * NSEG`` integers in one
    flat list. A record is ten integer stores into slots that already exist.
    """

    def __init__(self, num_layers: int, capacity: int = 1024):
        self.num_layers = int(num_layers)
        self.capacity = int(capacity)
        self.buf = [0] * (self.num_layers * self.capacity * NSEG)
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
        base = (layer_idx * self.capacity + n % self.capacity) * NSEG
        buf = self.buf
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
            # No probe (or a callback that did not go through it): the middle
            # is one number. Zero, not a guess -- a zero segment next to a
            # filled ``kt`` reads as "not measured here", a guessed split does
            # not.
            buf[base + S_KT_PRE] = 0
            buf[base + S_SUBMIT] = 0
            buf[base + S_BETWEEN] = 0
            buf[base + S_SYNC] = 0
            buf[base + S_KT_TAIL] = 0
        self.count[layer_idx] = n + 1

    def drain(self, since: list[int]) -> tuple[dict[int, list[list[int]]], list[int]]:
        """Records added since the given per-layer counts, plus the new counts.

        Only whole ring slots are read; if a layer produced more than
        ``capacity`` records in one interval the oldest are gone and the
        report says how many were lost.
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
                base = (layer * self.capacity + k % self.capacity) * NSEG
                rows.append(self.buf[base : base + NSEG])
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


def summarize(rows_by_layer: dict[int, list[list[int]]], split: bool, interval_s: float) -> dict:
    """Median/p90 per segment per layer, in microseconds. Pure, hence testable."""
    layers: dict[str, dict] = {}
    for layer, rows in sorted(rows_by_layer.items()):
        cols = list(zip(*rows))
        med = {name: round(_median(list(cols[i])) / 1000.0, 3) for i, name in enumerate(SEGMENTS)}
        p90 = {name: round(_p90(list(cols[i])) / 1000.0, 3) for i, name in enumerate(SEGMENTS)}
        layers[str(layer)] = {"n": len(rows), "med_us": med, "p90_us": p90}
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

    This is the check that catches a timestamp taken in the wrong place: a
    mis-ordered pair either overlaps or leaves a hole, and both show up here
    as a residual that is not ~0. It is reported, never silently absorbed.
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


def install_infer_probe(wrapper) -> bool:
    """Time kt's submit/sync without touching kt's C++ or rebuilding the wheel.

    ``run_pinned_forward_sync`` is Python (experts_base.py:717-772): it builds
    a task, calls ``self.cpu_infer.submit(task)`` and then
    ``self.cpu_infer.sync(allow_pending)``. Replacing that attribute with a
    pass-through that timestamps the two calls is the whole trick -- the task,
    the pointers, the order and the return values are kt's own.

    Returns False if this build does not let us (then only ``kt`` is measured
    and the report says ``split: false``); never raises: a diagnostic that
    takes the service down is worse than a missing column.
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

    Every wrapper shares one CPUInfer (layer._register_wrapper enforces it),
    so each wrapper gets its own probe over the *same* inner object; that is
    why the identity check unwraps instead of comparing the probes.
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
    process goes on sampling somebody else's service and its output looks
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
