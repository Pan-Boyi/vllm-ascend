# SPDX-License-Identifier: Apache-2.0
"""P1a: the device waits for an ExternalEvent instead of for the callback's return.

What changes
------------
Today the graph node that runs this layer's CPU experts does the whole job
inside the callback::

    D2H -> host-func [ kt_pre | submit | kt compute | sync ] -> H2D

and the device's ``NOTIFY_WAIT`` covers all of it, plus the ACL round trip on
both sides of the callback (``A`` = report delivered + GIL + argument tuple,
``F`` = the callback returning and the device being told). Measured on card
(PLAN §4.20/§4.22): ``A+F`` 78.0 us/layer = 3.83 ms/token, of which ``F`` is
the half this removes, and event waits themselves are almost free
(``DAVID_EVENT_WAIT`` 0.027 us, ``MEM_WAIT_VALUE`` 0.144 us, no idle gap).

With this switch on the graph is::

    D2H -> host-func [ kt_pre | submit ] (returns at once)
        -> ExternalEvent.wait -> ExternalEvent.reset -> H2D

and a **completion thread** calls ``cpu_infer.sync(0)`` and then
``event.record(<its own stream>)``. The device stops waiting the moment kt's
task is done, rather than one more ACL round trip later.

kt is NOT modified
------------------
The split is already ours: ``layer._kt_packed_forward`` builds the task with
``moe.forward_task(...)`` and calls ``cpu_infer.submit`` / ``cpu_infer.sync``
itself, so "submit here, sync there" needs no new kt entry point. Two facts
about kt's queue make the hand-off safe (``kt-kernel/cpu_backend/task_queue.cpp``):

* ``TaskQueue::sync(allow_n_pending)`` is ``cv.wait(lock, pred)`` on
  ``pending <= allow_n``: several threads may wait on it at once, and the
  predicate is a count, not an identity. The queue is one serial FIFO worker,
  and the graph only ever has one of our tasks outstanding (the next layer's
  host-func node sits behind this layer's event wait on the same stream), so
  ``sync(0)`` from the completion thread means "this layer's task is done".
* a task's exception is stored and rethrown by *the next* ``sync``
  (``task_queue.cpp:112-124``), so whoever syncs first gets it. With this
  switch on that is usually the completion thread; it logs it with the same
  wording the callback used, and **records the event anyway** -- see below.

🔴 The failure mode this file is fenced against
-----------------------------------------------
Releasing the device early is *faster and wrong, and nothing raises*: the H2D
copies whatever the pinned output held, which is the previous step's CPU
experts. So:

* the event is **reset inside the graph**, right behind the wait, because
  torch_npu says an ExternalEvent has to be (streams.py:214-217: "Before
  reusing ExternalEvent, you need to call reset() to clear the flag") and
  because it costs 0.144 us. 🔴 On card it turns out not to be load-bearing:
  with every reset node provably left out (t5r ``--mutant no-reset
  --stall-ms 50``, 2026-09-23) the arm stayed bit-identical and still waited on
  every replay, so the captured wait re-arms itself on this build. The reset
  stays anyway -- undocumented behaviour is not something to build on -- but
  nobody should expect a missing one to show up on card. The mutant that does
  go red there is ``--mutant no-wait``; the reset's red arm is the CPU model.
* the record must happen **after** ``sync`` returns, never around it.
* the completion thread must have an ACL context of its own
  (``torch.npu.set_device``). This is the one part with no precedent in this
  tree: ``ExternalEvent`` wait/reset inside a capture and a host-side record
  during replay is what ``mla_v1.py:947-970`` + ``:1744-1747`` already do, but
  there the record comes from the **main** thread between replays.
* if anything at all goes wrong, the event is recorded regardless. A device
  that waits for ever wedges the service with no message; stale output at
  least logs "the CPU-expert output of this step is NOT this step's", which is
  the same wording (and the same severity) the callback already uses.
"""

from __future__ import annotations

import queue
import threading

from vllm.logger import logger

ENV_NAME = "VLLM_ASCEND_DSV41_KT_ASYNC_WAIT"

# layer_idx -> how many times the completion thread has recorded that layer's
# ExternalEvent. The counterpart of layer.HOST_FUNC_CALLS (which counts
# submits): one per replay per layer, and the two must agree. Read by the
# on-card cases; a layer whose submits grow while its records do not is a
# device that is about to wait for ever.
RECORDS: dict[int, int] = {}
# Every failure the completion thread swallowed so the device could proceed,
# as ``(layer_idx, repr(exc))``. Never silently empty: it is also logged.
# Capped, because a kt that fails every layer would otherwise grow this list at
# 40 entries per token for the life of the process; the log keeps them all.
ERRORS: list[tuple[int, str]] = []
MAX_ERRORS = 200


def _note_error(layer_idx: int, what: str) -> None:
    if len(ERRORS) < MAX_ERRORS:
        ERRORS.append((int(layer_idx), what))


def enabled_by_env() -> bool:
    """Whether the operator explicitly asked for the async wait.

    Read from ``envs`` at the moment it is asked, not cached, for the same
    reason ``side_stream.enabled_by_env`` is: the refusal message names the
    variable, so the answer has to come from the environment the service was
    started with.
    """
    from vllm_ascend import envs

    return getattr(envs, ENV_NAME, "0") == "1"


# -- the completion thread ---------------------------------------------------


class Completer:
    """One thread per process: wait for kt, then let the device through.

    Everything it touches is injected so the CPU tests can drive it without a
    device: ``set_device`` gives the thread its ACL context, ``make_stream``
    the stream the record is issued on (created *in the thread*, after the
    context exists, and before any capture can start).
    """

    def __init__(self, set_device, make_stream, *, name: str = "kt-offload-completer"):
        self._set_device = set_device
        self._make_stream = make_stream
        self._name = name
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._startup_error: BaseException | None = None
        # Two counters rather than one depth, because they are written by two
        # threads: ``_submitted`` only by whoever runs the graph node,
        # ``_completed`` only by the thread below. A single ``depth += 1 / -= 1``
        # pair would be a read-modify-write race between them. Only ever read as
        # a diagnostic; the ordering it might look like it guards is supplied by
        # the graph (the next host-func node sits behind this layer's wait).
        self._submitted = 0
        self._completed = 0

    @property
    def depth(self) -> int:
        """Submits handed over and not yet recorded."""
        return self._submitted - self._completed

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the thread and wait until its context and stream exist.

        Synchronous on purpose: a capture must never be the first thing that
        needs this stream, and a thread that failed to get a context has to be
        a start-up error rather than a hang at the first decode step.
        """
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                "kt_offload async wait: the completion thread could not take an ACL context or create its "
                f"stream ({self._startup_error!r}). Without it nothing would ever record the event the "
                "graph waits on, and the first replay would hang instead of failing here."
            ) from self._startup_error

    def stop(self) -> None:
        """Tests only; production keeps the thread for the life of the process."""
        if self._thread is None:
            return
        self._q.put(None)
        self._thread.join(timeout=30)
        self._thread = None

    # -- the hand-off ------------------------------------------------------

    def submit(self, wrapper, layer_idx: int, event) -> None:
        """Hand this layer's outstanding kt task over. Never blocks.

        ``wrapper is None`` means "the submit itself failed, there is nothing
        to wait for -- just let the device through", which is how a callback
        that raised before ``cpu_infer.submit`` avoids wedging the graph.
        """
        self._submitted += 1
        self._q.put((wrapper, int(layer_idx), event))

    # -- the thread --------------------------------------------------------

    def _run(self) -> None:
        try:
            self._set_device()
            self._stream = self._make_stream()
        except BaseException as exc:  # noqa: BLE001
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._q.get()
            if item is None:
                return
            self._complete(*item)

    def _complete(self, wrapper, layer_idx: int, event) -> None:
        try:
            if wrapper is not None:
                # allowance 0 = the queue is empty. kt's FIFO is serial and the
                # graph keeps one task of ours outstanding, so this is exactly
                # "this layer's task has run".
                wrapper.cpu_infer.sync(0)
        except BaseException as exc:  # noqa: BLE001
            _note_error(layer_idx, repr(exc))
            logger.error(
                "kt_offload async wait: layer %d's CPU task raised at sync: %r. The CPU-expert output of "
                "this step is NOT this step's -- the device reads whatever the pinned buffer held. The "
                "event is recorded anyway; a device left waiting would wedge the engine with no message.",
                layer_idx,
                exc,
            )
        finally:
            self._completed += 1
            try:
                event.record(self._stream)
                RECORDS[layer_idx] = RECORDS.get(layer_idx, 0) + 1
            except BaseException as exc:  # noqa: BLE001
                _note_error(layer_idx, f"record failed: {exc!r}")
                logger.critical(
                    "kt_offload async wait: recording layer %d's ExternalEvent FAILED (%r). The graph is "
                    "waiting on it and nothing else will ever signal it: this replay will not finish.",
                    layer_idx,
                    exc,
                )


# -- the criteria (pure, so a test can drive every one of them red) ----------


def ext_event_ops_ok(ops) -> dict:
    """The shape of one layer's captured submit, as a check that can go red.

    ``ops`` is a sequence of ``(kind, what)`` in enqueue order:

      ``("d2h", buf)``        the input copy that feeds kt
      ``("hostfunc", layer)`` the submit-only node
      ``("ext_wait", ev)``    ``event.wait(stream)`` -- recorded INTO the graph
      ``("ext_reset", ev)``   ``event.reset(stream)`` -- recorded INTO the graph
      ``("h2d", buf)``        the output copy back to the device
      ``("host_record", ev)`` a host-side ``event.record`` (must NOT be here)

    🔴 The reset is the one that has no second line of defence. Without it the
    event stays signalled after the first replay, every later replay's wait is
    a no-op, and the H2D copies the pinned output before this step's task has
    written it: faster, wrong, and silent.
    """
    problems = []
    kinds = [k for k, _ in ops]
    if kinds.count("hostfunc") != 1:
        problems.append(f"expected exactly one host-func node, found {kinds.count('hostfunc')}")
    waits = [w for k, w in ops if k == "ext_wait"]
    resets = [w for k, w in ops if k == "ext_reset"]
    if len(waits) != 1:
        problems.append(f"expected exactly one ExternalEvent wait in the graph, found {len(waits)}")
    if len(resets) != 1:
        problems.append(
            f"expected exactly one ExternalEvent reset in the graph, found {len(resets)}; without it the "
            "event stays signalled and every replay after the first sails through the wait"
        )
    if waits and resets and waits[0] is not resets[0]:
        problems.append("the graph waits on one event and resets another")
    if any(k == "host_record" for k, _ in ops):
        problems.append(
            "a host-side event.record is among the captured ops; the record is what the completion thread "
            "does at replay time and must not be a graph node (it would release the wait it precedes)"
        )
    i = {k: n for n, (k, _) in enumerate(ops)}
    if "hostfunc" in i and "d2h" in i and i["d2h"] > i["hostfunc"]:
        problems.append("the inputs are copied to the host AFTER the node that submits them to kt")
    if "hostfunc" in i and "ext_wait" in i and i["ext_wait"] < i["hostfunc"]:
        problems.append(
            "the graph waits on the event BEFORE the node that submits the task; nothing has been handed "
            "to the completion thread yet, so the first replay would wait for ever"
        )
    if "ext_wait" in i and "ext_reset" in i and i["ext_reset"] < i["ext_wait"]:
        problems.append("the event is reset before it is waited on, which clears the signal it waits for")
    if "h2d" in i and "ext_wait" in i and i["h2d"] < i["ext_wait"]:
        problems.append("the output is copied back before the wait: it would read the previous step's value")
    return {
        "name": "ext_event_wait_reset_in_graph",
        "ok": not problems,
        "detail": problems,
        "why": (
            "the ExternalEvent wait is the only thing holding the H2D back, and the reset is the only thing "
            "that makes the NEXT replay wait at all"
        ),
    }


def record_counts_ok(before: dict[int, int], after: dict[int, int], replays: int, layers) -> dict:
    """Every layer's event was recorded exactly once per replay.

    The counterpart of ``side_stream.replay_hostfunc_ok``: that one counts
    submits, this one counts releases. They have to agree, and a run where the
    submits grow and the records do not is a device that is about to wait for
    ever -- which looks like a hang, not like a wrong answer, so it is worth
    naming separately.
    """
    bad = {}
    for i in layers:
        delta = after.get(i, 0) - before.get(i, 0)
        if delta != replays:
            bad[i] = delta
    return {
        "name": "record_one_per_replay",
        "ok": not bad,
        "detail": {"expected": replays, "wrong": bad},
        "why": "a layer whose ExternalEvent is not recorded once per replay either hangs or was never waited on",
    }


def submits_match_records(host_func_delta: dict[int, int], record_delta: dict[int, int]) -> dict:
    """Submits and releases pair up layer by layer."""
    bad = {
        i: (host_func_delta.get(i, 0), record_delta.get(i, 0))
        for i in set(host_func_delta) | set(record_delta)
        if host_func_delta.get(i, 0) != record_delta.get(i, 0)
    }
    return {
        "name": "submits_match_records",
        "ok": not bad,
        "detail": bad,
        "why": "each host-func submit hands exactly one task to the completion thread, which records once",
    }
