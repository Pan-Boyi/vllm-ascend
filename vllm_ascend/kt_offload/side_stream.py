# SPDX-License-Identifier: Apache-2.0
"""Side-stream overlap: the switch, and the criteria that let it be opened.

Why this module exists
----------------------
``overlap_side_stream`` was refused outright (2026-09-20) because every real
capture that forked onto a second stream died at ``capture_end`` with ACL
107025. On 2026-09-22 that verdict was withdrawn: the on-card case (t5) that
"proved" it was failing on *our own* buffer guard (``no preallocated int64 id
buffer``, see layer._prepare_capture_buffers) and the 107025 was secondary.
With the buffers present, a real three-layer capture on the side stream
captures and replays correctly, and the one arm that still fails with 107025
is the one where the side stream's last event record is never joined by the
compute stream -- which is precisely what production's ``_join_side_stream``
has always done.

So the switch is opened *conditionally*: the config still refuses unless
``VLLM_ASCEND_DSV41_KT_SIDE_STREAM=1`` is set explicitly, and the path carries
the checks below rather than a comment claiming it is safe.

The criteria (k.4's meta-rule: a check that cannot go red is not a check)
-------------------------------------------------------------------------
Each one is a pure function here, so the unit tests can drive it red, and the
runtime calls the ones it can evaluate in-process:

  ``capture_hostfunc_ok``   nothing ran the host func *during* the capture. If
                            it did, the work was not recorded into the graph
                            and every replay reuses the capture run's CPU
                            output -- the silent non-replay failure, which is
                            faster and wrong (sglang F1).
  ``replay_hostfunc_ok``    every layer's counter grows by exactly one per
                            replay. A layer whose counter does not grow has no
                            CPU-expert work in the graph at all.
  ``join_ops_ok``           the side stream's record is waited on by the
                            compute stream, and the side stream's first op is
                            a wait on a fork the compute stream recorded. The
                            *positive control* for 107025: drop the wait and
                            the on-card capture fails with 107025 (t5 off_evt),
                            keep it and it captures (t5 off_evtw).
  ``slot_ownership_ok``     A6. kt hands layer L the pinned slot ``L % 2``, so
                            L and L+2 share one buffer, and the side stream
                            removes the device-wide sync that used to order
                            them. Each layer must own its own int64 id buffer,
                            and no two layers sharing a slot may be in flight
                            at the same time.

Zero CPU coverage, on-card only (tools/kt_offload/oncard/RUNBOOK_sidestream.md):
the ACL capture itself (107025 as a positive control), the actual TPOT, and
whether torch_npu's capture follows the event wait onto the side stream on the
build in use. The code refuses at run time rather than assuming any of the
three (layer._submit_captured_forked).
"""

from __future__ import annotations

# kt's KExpertsCPUBuffer keeps ``buffer_depth`` pinned slots per batch size and
# hands layer L the slot ``L % buffer_depth``. Mirrored here (not imported: kt
# is a host-only dependency and this module is imported on the off path too).
KT_BUFFER_DEPTH = 2

ENV_NAME = "VLLM_ASCEND_DSV41_KT_SIDE_STREAM"


def enabled_by_env() -> bool:
    """Whether the operator explicitly asked for the side stream.

    Read at config-validation time, not cached: the refusal message names this
    variable, so the answer has to come from the environment the service was
    actually started with.
    """
    from vllm_ascend import envs

    return getattr(envs, ENV_NAME, "0") == "1"


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
    {"record", "wait", ...}; anything else is ignored. This is the shape check
    that separates t5's off_evt (record with nobody waiting -> ACL 107025) from
    off_evtw (record + wait -> captures).
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
    """A6: per-layer id buffers, and no two slot-sharing layers in flight.

    ``layers`` is a sequence of ``(layer_idx, id_buffer_keys, in_flight)``:
    ``id_buffer_keys`` is the set of ``id()`` of that layer's int64 id buffers
    (empty = the layer never got any: the 2026-09-22 production bug), and
    ``in_flight`` says whether its side-stream work is submitted and not yet
    joined.
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
