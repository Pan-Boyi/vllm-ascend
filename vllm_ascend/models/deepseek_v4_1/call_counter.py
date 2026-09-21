"""Per-call counters behind the two index-K evidence lines.

**The question this exists to answer**: does the index-K Python run *inside*
the captured graph, or outside it?

A log-once evidence line **cannot** answer it. It fires exactly once in both
worlds, so the two readings are identical -- which is why
``DSV41-INDEXK-WRITE`` appearing at log line 506 while capture starts at 504
proved nothing. A *count* is not identical:

* in the graph  -- Python runs only while capturing  => calls ~= captured shapes
* out of the graph -- Python runs every step         => calls ~= decode steps

Those two are orders of magnitude apart, so the reading cannot be misread.
This is a yes/no question about an **artifact**, which matters because the two
obvious source-level routes are both dead ends: the names in
``splitting_ops`` (``vllm::dsa_forward``, ``vllm::mla_forward``) do **not**
exist as literals in either tree -- they are interpolated at registration --
so grep answers "absent" for a name that is very much in use. (Positive
control for that scan: ``def forward`` matched 1863 times, so the instrument
was alive and the zero was real.)

A second, independent reading comes for free: ``capturing=`` counts the calls
that happened while the stream was capturing, from a probe with no connection
to the counter. Two instruments, reported side by side and **never merged** --
if they disagree, the disagreement is the finding.

What this does **not** cover
----------------------------
* **A summary line that never appears does not mean zero calls.** The
  ``at=exit`` line comes from ``atexit``, which a ``SIGKILL``ed worker never
  runs. The ``at=milestone`` lines are the fallback: they fire at 1, 2, 4, 8
  ... calls, so the last one seen bounds the total within 2x -- ample for an
  orders-of-magnitude question. If **no** line at all appears, that is either
  zero calls or a logger that was never configured, and this module cannot
  tell those two apart.
* **Counts are per process.** Four ranks run in four processes and each emits
  its own lines; that is why ``rank=`` is on **every** line and not only the
  summary. We have already lost a day's main conclusion to four ranks writing
  one artifact where the later writer silently won.
* **``capturing`` answers "Python ran during a capture"**, not "this operator
  ended up in the replayed graph". A block behind a runtime-false condition is
  captured as nothing at all and this probe would still say "capturing".
* ``distinct_shapes`` counts shapes seen **by this site in this process**.

What this **changes** about the environment: log volume only, and boundedly --
at most ~20 milestone lines plus one summary per site per process, because the
milestones are powers of two.
"""
from __future__ import annotations

import atexit
import os
import sys
import threading
from dataclasses import dataclass, field

import torch

#: Our own prefix, deliberately **not** sharing a leading substring with
#: either evidence line.
#:
#: 🔴 ``DSV41-INDEXK-WRITE-COUNT`` was the obvious name and it is a trap: the
#: batch-1 unit tests select evidence lines with
#: ``startswith("DSV41-INDEXK-WRITE")`` and then assert there is **exactly
#: one**, so a counter line under that name turns those tests red while
#: reading like a naming preference. A substring criterion misjudges the line
#: that was just added, every time. ``test_call_counter.py`` holds the
#: tripwire, and it imports the real matchers rather than retyping them.
COUNT_PREFIX = "DSV41-INDEXK-CALLS"

#: The three answers the capture probe can give. ``unaskable`` is a separate
#: answer on purpose: the two existing probes in this package
#: (``engram_embedding`` / ``compressed_slots``) fold it into ``False``, which
#: is right for *gating a path* -- an unavailable predicate must not block a
#: path that works outside capture -- and wrong for *evidence*, where
#: "it said no" and "it could not be asked" are different readings.
CAPTURING = "capturing"
NOT_CAPTURING = "not_capturing"
UNASKABLE = "capture_unaskable"

_LOCK = threading.Lock()


def _capture_state() -> str:
    """Tri-state capture probe. Never raises."""
    for module in (getattr(torch, "npu", None), getattr(torch, "cuda", None)):
        predicate = getattr(module, "is_current_stream_capturing", None)
        if predicate is None:
            continue
        try:
            answer = predicate()
        except Exception:                                    # noqa: BLE001
            return UNASKABLE
        # ``is True`` / ``is False``: a MagicMock left on ``torch.npu`` is
        # truthy and would otherwise read as a confident "capturing".
        if answer is True:
            return CAPTURING
        if answer is False:
            return NOT_CAPTURING
        return UNASKABLE
    return UNASKABLE


def resolve_rank() -> tuple[int, str]:
    """``(rank, source)`` -- the source is part of the reading, not decoration.

    A rank that reads ``0`` because every lookup failed and ``0`` is the
    fallback is **indistinguishable** from a real rank 0, and rank 0 is the
    one we would be least surprised to see. So the fallback is ``-1`` (no
    arithmetic is ever done on it) and the source is printed next to it: a
    check bit that cannot change with the thing it checks is a green light
    that means nothing.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank()), "torch.distributed"
    except Exception:                                        # noqa: BLE001
        pass
    for key in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK"):
        raw = os.environ.get(key)
        if raw is not None and raw.strip().lstrip("-").isdigit():
            return int(raw.strip()), f"env:{key}"
    return -1, "unknown"


@dataclass
class _Site:
    name: str
    calls: int = 0
    states: dict = field(default_factory=dict)
    branches: dict = field(default_factory=dict)
    shapes: set = field(default_factory=set)
    next_milestone: int = 1

    def line(self, at: str) -> str:
        rank, rank_src = resolve_rank()
        branches = ",".join(f"{k}:{v}" for k, v in sorted(
            self.branches.items())) or "none"
        return (
            f"{COUNT_PREFIX} site={self.name} rank={rank} "
            f"rank_src={rank_src} at={at} calls={self.calls} "
            f"distinct_shapes={len(self.shapes)} "
            f"{CAPTURING}={self.states.get(CAPTURING, 0)} "
            f"{NOT_CAPTURING}={self.states.get(NOT_CAPTURING, 0)} "
            f"{UNASKABLE}={self.states.get(UNASKABLE, 0)} "
            f"branches={branches}")


_SITES: dict = {}


def _emit(text: str) -> None:
    """Log the line, and still emit it if logging is already torn down.

    ``atexit`` can run after the logging machinery has been shut down, and a
    summary that disappears exactly at the moment it is produced would be
    indistinguishable from a counter that never ran.
    """
    try:
        from vllm.logger import logger
        logger.info("%s", text)
        return
    except Exception:                                        # noqa: BLE001
        pass
    try:
        print(text, file=sys.stderr, flush=True)
    except Exception:                                        # noqa: BLE001
        pass


def record(site: str, shape=None) -> int:
    """Count one call at ``site``. Returns the new total.

    Called at the **entry** of the instrumented function, before any early
    return: "did this Python execute" is the question, and a call that returns
    early still executed. Which way it went afterwards is
    ``mark_branch``'s job.
    """
    with _LOCK:
        state = _SITES.get(site)
        if state is None:
            state = _SITES[site] = _Site(site)
        state.calls += 1
        key = _capture_state()
        state.states[key] = state.states.get(key, 0) + 1
        if shape is not None:
            state.shapes.add(tuple(int(d) for d in shape))
        due = state.calls >= state.next_milestone
        if due:
            while state.next_milestone <= state.calls:
                state.next_milestone *= 2
            text = state.line("milestone")
        total = state.calls
    if due:
        _emit(text)
    return total


def mark_branch(site: str, branch: str) -> None:
    """Tally which way the call went.

    🔴 Why this is not optional: the writer has **two** store paths (the
    MXFP4 quant-cache operator and the fp8 epilog) plus a zero-token early
    return. Counting inside one of them would report ``calls=0`` when the
    other one ran -- and "the writer never executed" is exactly the wrong
    conclusion to hand a graph-membership question. Accounting has to follow
    every writer.

    ``sum(branches) == calls`` is also a free consistency check: a shortfall
    means calls left by a path nobody tallied, i.e. an exception.
    """
    with _LOCK:
        state = _SITES.get(site)
        if state is None:
            state = _SITES[site] = _Site(site)
        state.branches[branch] = state.branches.get(branch, 0) + 1


def snapshot() -> dict:
    """Current counts, for tests and for a caller that wants to assert."""
    with _LOCK:
        return {
            name: {
                "calls": s.calls,
                "distinct_shapes": len(s.shapes),
                "states": dict(s.states),
                "branches": dict(s.branches),
            }
            for name, s in _SITES.items()
        }


def flush(at: str = "exit") -> list:
    """Emit one summary line per site. Returns the lines emitted."""
    with _LOCK:
        lines = [s.line(at) for _, s in sorted(_SITES.items())]
    for text in lines:
        _emit(text)
    return lines


def reset_for_tests() -> None:
    with _LOCK:
        _SITES.clear()


atexit.register(flush)
