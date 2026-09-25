# SPDX-License-Identifier: Apache-2.0
"""Side-stream overlap: the switch, and the criteria that let it be opened.

``overlap_side_stream`` forks kt's CPU-expert work onto a second stream. A real
capture that does so dies at ``capture_end`` with ACL 107025 unless two things
hold: every layer has its own preallocated int64 id buffer
(``layer._prepare_capture_buffers``), and the side stream's last event record is
joined by the compute stream (``_join_side_stream``). ON by default;
``VLLM_ASCEND_DSV41_KT_SIDE_STREAM=0`` or ``overlap_side_stream: false`` turns
it off, and ``kt_offload.switches.resolve`` is the only reader.

Each criterion is a pure function here so the unit tests can drive it red; the
runtime calls the ones it can evaluate in-process:

  ``capture_hostfunc_ok``   nothing ran the host func *during* the capture. If
                            it did, the work was not recorded into the graph and
                            every replay reuses the capture run's CPU output --
                            faster and wrong, and silent.
  ``replay_hostfunc_ok``    every layer's counter grows by exactly one per
                            replay. A layer whose counter does not grow has no
                            CPU-expert work in the graph at all.
  ``join_ops_ok``           the side stream's record is waited on by the compute
                            stream, and the side stream's first op is a wait on
                            a fork the compute stream recorded. The positive
                            control for 107025: drop the wait and the on-card
                            capture fails with it.
  ``slot_ownership_ok``     kt hands layer L the pinned slot ``L % 2``, so L and
                            L+2 share one buffer, and the side stream removes
                            the device-wide sync that used to order them. Kept
                            for the paths that still hand kt its own buffers;
                            the forked path does not.
  ``buffer_ownership_ok``   the same rule as the forked path states it, and
                            **across steps**: every store the side stream
                            touches is one layer's own, and no layer submits
                            again while its previous submit is un-joined.
                            ``slot_ownership_ok`` could not say the second half,
                            because its ``in_flight`` was read within one
                            capture.
  ``submit_order_ok``       everything the side stream reads must be written by
                            the compute stream BEFORE the fork event is
                            recorded. Fork first and fill afterwards leaves the
                            side stream's read ordered against nothing, and on
                            card it then reads the previous step's routing ids.

Zero CPU coverage, on-card only: the ACL capture itself (107025 as a positive
control), the actual TPOT, and whether torch_npu's capture follows the event
wait onto the side stream on the build in use. The code refuses at run time
rather than assuming any of the three (layer._submit_captured_forked).
"""

from __future__ import annotations

# kt's KExpertsCPUBuffer keeps ``buffer_depth`` pinned slots per batch size and
# hands layer L the slot ``L % buffer_depth``. Mirrored here (not imported: kt
# is a host-only dependency and this module is imported on the off path too).
KT_BUFFER_DEPTH = 2

ENV_NAME = "VLLM_ASCEND_DSV41_KT_SIDE_STREAM"


def enabled_by_env() -> bool:
    """Whether the operator explicitly set ``ENV_NAME=1``.

    NOT whether the side stream is on: unset means "default", and the effective
    value comes only from ``kt_offload.switches.resolve``. Nothing on the
    production path calls this; it is for harnesses asking the narrower question.
    """
    from vllm_ascend.kt_offload import switches

    return switches.raw(ENV_NAME) == "1"


def slot_of(layer_idx: int, depth: int = KT_BUFFER_DEPTH) -> int:
    return layer_idx % depth


# -- the criteria -----------------------------------------------------------


def capture_hostfunc_ok(before: dict[int, int], after: dict[int, int]) -> dict:
    """No host-func callback ran while the graph was being captured."""
    ran = {i: after[i] - before.get(i, 0) for i in after if after[i] - before.get(i, 0) != 0}
    return {
        "name": "capture_hostfunc_zero",
        "ok": not ran,
        "detail": ran,
        "why": (
            "a callback that runs during the capture was executed instead of recorded; every replay would "
            "then reuse that one CPU output (faster and wrong)"
        ),
    }


def replay_hostfunc_ok(before: dict[int, int], after: dict[int, int], replays: int, layers: int) -> dict:
    """Every layer's counter grew by exactly ``replays``."""
    bad = {}
    for i in range(layers):
        delta = after.get(i, 0) - before.get(i, 0)
        if delta != replays:
            bad[i] = delta
    return {
        "name": "replay_one_per_layer",
        "ok": not bad,
        "detail": {"expected": replays, "wrong": bad},
        "why": "a layer whose counter does not grow once per replay has no CPU-expert work in the graph",
    }


def join_ops_ok(side_ops, compute_ops) -> dict:
    """fork recorded on compute -> waited on side; join recorded on side -> waited on compute.

    ``*_ops`` are sequences of ``(kind, event_id)`` with kind in
    {"record", "wait", ...}; anything else is ignored. Separates a record with
    nobody waiting (ACL 107025 on card) from one the compute stream joins.
    """
    problems = []
    side_waits = [e for k, e in side_ops if k == "wait"]
    side_records = [e for k, e in side_ops if k == "record"]
    compute_waits = [e for k, e in compute_ops if k == "wait"]
    compute_records = [e for k, e in compute_ops if k == "record"]
    for e in side_records:
        if e not in compute_waits:
            problems.append(f"side recorded event {e} that no compute-stream wait joins (ACL 107025 on card)")
    for e in side_waits:
        if e not in compute_records:
            problems.append(f"side waits on event {e} the compute stream never records (deadlock)")
    if side_records and side_ops[-1][0] != "record":
        problems.append("the side stream's last op is not the join record")
    if not side_records:
        problems.append("the side stream records no join event at all")
    return {
        "name": "join_record_and_wait_paired",
        "ok": not problems,
        "detail": problems,
        "why": "an unjoined record on the side stream is the known 107025 trigger (t5 off_evt vs off_evtw)",
    }


def slot_ownership_ok(layers, depth: int = KT_BUFFER_DEPTH) -> dict:
    """Per-layer id buffers, and no two slot-sharing layers in flight.

    ``layers`` is a sequence of ``(layer_idx, id_buffer_keys, in_flight)``:
    ``id_buffer_keys`` is the set of ``id()`` of that layer's int64 id buffers
    (empty = it never got any), ``in_flight`` whether its side-stream work is
    submitted and not yet joined.
    """
    problems = []
    seen: dict[int, set] = {}
    flight: dict[int, int] = {}
    for idx, keys, in_flight in layers:
        if not keys:
            problems.append(f"layer {idx} has no preallocated int64 id buffer")
        for other, other_keys in seen.items():
            shared = set(keys) & set(other_keys)
            if shared:
                problems.append(f"layers {other} and {idx} share id buffer(s) {sorted(shared)}")
        seen[idx] = set(keys)
        if in_flight:
            slot = slot_of(idx, depth)
            if slot in flight:
                problems.append(
                    f"layers {flight[slot]} and {idx} are both in flight on kt pinned slot {slot} "
                    f"(depth {depth}); the side stream has no device-wide sync to order them"
                )
            flight[slot] = idx
    return {
        "name": "buffer_ownership_a6",
        "ok": not problems,
        "detail": problems,
        "why": "kt hands layer L the pinned slot L%%depth, so L and L+2 share one buffer",
    }


def buffer_ownership_ok(layers) -> dict:
    """Buffer ownership for the per-layer side buffers, and it spans steps.

    ``layers`` is a sequence of ``(layer_idx, buffer_keys, inflight)`` in submit
    order, possibly across several steps:

      ``buffer_keys``  ``id()`` of every store the side stream reads or writes
                       for that layer (empty = none of its own, i.e. still using
                       kt's shared pair).
      ``inflight``     whether that layer's PREVIOUS submit is not yet joined.
                       The cross-step half: the flag is cleared by the join, not
                       by the next route, so a submit that outlives its step
                       arrives here.
    """
    problems = []
    owner: dict = {}
    for idx, keys, inflight in layers:
        if not keys:
            problems.append(f"layer {idx} has no side buffers of its own (it would be sharing kt's slot pair)")
        if inflight:
            problems.append(
                f"layer {idx} submits again while its previous side-stream work is un-joined; its own "
                "buffers would be overwritten while the host-func node still has to read them"
            )
        for key in keys:
            other = owner.setdefault(key, idx)
            if other != idx:
                problems.append(f"layers {other} and {idx} share the side buffer {key}")
    return {
        "name": "buffer_ownership_a6_across_steps",
        "ok": not problems,
        "detail": problems,
        "why": (
            "the side stream may only touch stores this layer owns, and only while its previous submit "
            "has been joined; anything else is a write racing a read with no event between them"
        ),
    }


def submit_order_ok(ops) -> dict:
    """Everything the side stream reads was written before the fork was recorded.

    ``ops`` is one layer's submit as a sequence of ``(kind, what)``:

      ``("fill", buf)``       a compute-stream write into ``buf``
      ``("fork", None)``      ``fork.record(compute)``
      ``("side_wait", None)`` ``side.wait_event(fork)``
      ``("side_read", buf)``  the side stream reading ``buf``

    The fork event is the only thing ordering the compute stream's writes
    against the side stream's reads, so a fill after it buys nothing: the side
    stream is already released and whichever gets there first wins.
    """
    problems = []
    forked = False
    waited = False
    written: set = set()
    late: set = set()
    for kind, what in ops:
        if kind == "fill":
            (late if forked else written).add(what)
        elif kind == "fork":
            if forked:
                problems.append("the fork event is recorded twice in one submit")
            forked = True
        elif kind == "side_wait":
            if not forked:
                problems.append("the side stream waits on a fork event the compute stream has not recorded")
            waited = True
        elif kind == "side_read":
            if not waited:
                problems.append(f"the side stream reads {what} before waiting on the fork event")
            if what in late:
                problems.append(
                    f"the side stream reads {what}, which the compute stream fills AFTER fork.record: "
                    "nothing orders that write against this read"
                )
            elif what not in written:
                problems.append(f"the side stream reads {what}, which the compute stream never filled")
    if not forked:
        problems.append("no fork event was recorded at all")
    return {
        "name": "fills_before_fork",
        "ok": not problems,
        "detail": problems,
        "why": "the fork event is the only edge between the compute stream's writes and the side stream's reads",
    }


def gate_report(checks) -> dict:
    """Combine criteria; ``ok`` only if every one of them is ok."""
    checks = list(checks)
    failed = [c for c in checks if not c["ok"]]
    return {
        "ok": not failed,
        "failed": [c["name"] for c in failed],
        "checks": checks,
    }


def format_gate(report: dict) -> str:
    if report["ok"]:
        return f"kt_offload side-stream gate: all {len(report['checks'])} checks pass"
    lines = ["kt_offload side-stream gate FAILED: " + ", ".join(report["failed"])]
    for c in report["checks"]:
        if not c["ok"]:
            lines.append(f"  {c['name']}: {c['detail']} -- {c['why']}")
    return "\n".join(lines)
