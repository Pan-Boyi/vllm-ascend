# SPDX-License-Identifier: Apache-2.0
"""Per-layer host-func counters, sampled from inside a running service.

Only a host-func node *in the graph* increments ``layer.HOST_FUNC_CALLS``: a
graph replay runs no Python, so if a layer's counter does not grow while the
service decodes, that layer's CPU-expert work is not in the graph at all. That
is the measurement that settles what the 2026-09-20 arms could not -- the
profile arithmetic there was consistent with both "~25-30 of 44 exposure
points lost their work" and "the captured steps lost all 40 and the survivors
are breakable's eager attention".

Off unless ``VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG`` is set to an interval in
seconds. The sampler is a daemon thread that also stops on the stop event at
interpreter exit: a watcher that outlives its process goes on sampling
somebody else's service and its output looks completely normal.
"""

from __future__ import annotations

import atexit
import logging
import threading

logger = logging.getLogger(__name__)

_MONITOR: threading.Thread | None = None
_STOP = threading.Event()


def host_func_report(before: dict[int, int], after: dict[int, int], expected_layers: int) -> dict:
    """What changed between two samples of the counters. Pure, hence testable.

    ``per_layer`` is the delta for every layer the offload knows about, so a
    layer that never ran appears as 0 rather than going missing (an absent key
    and a zero read the same in a log, and only one of them is the finding).
    """
    per_layer = {layer: after.get(layer, 0) - before.get(layer, 0) for layer in range(expected_layers)}
    deltas = sorted(per_layer.values())
    zero = [layer for layer, d in per_layer.items() if d == 0]
    busiest = deltas[-1] if deltas else 0
    return {
        "layers": expected_layers,
        "per_layer": per_layer,
        "min": deltas[0] if deltas else 0,
        "max": busiest,
        "zero_layers": zero,
        "uniform": bool(deltas) and deltas[0] == deltas[-1],
        # the finding: some layers ran and others never did
        "partial": bool(zero) and busiest > 0,
        "total": sum(per_layer.values()),
    }


def format_report(report: dict, interval_s: float) -> str:
    z = report["zero_layers"]
    head = (
        f"kt_offload host-func over {interval_s:.1f}s: {report['total']} calls across {report['layers']} layers, "
        f"per-layer min/max {report['min']}/{report['max']}"
    )
    if report["partial"]:
        return (
            f"{head}; {len(z)} layers never ran: {z[:12]}{'...' if len(z) > 12 else ''}"
            " -- their CPU-expert work is NOT in the graph (a replay runs no Python, so a live layer's"
            " counter must grow every step)"
        )
    if not report["uniform"]:
        return f"{head}; uneven (expected one call per layer per step): {report['per_layer']}"
    return f"{head}; uniform"


def start(interval_s: float, counters: dict[int, int], expected_layers: int) -> threading.Thread | None:
    """Start the one sampler of this process (idempotent). Returns the thread."""
    global _MONITOR
    if _MONITOR is not None or interval_s <= 0:
        return _MONITOR

    def run() -> None:
        before = dict(counters)
        while not _STOP.wait(interval_s):
            after = dict(counters)
            report = host_func_report(before, after, expected_layers)
            if report["total"] or report["partial"]:
                logger.info("%s", format_report(report, interval_s))
            before = after

    _MONITOR = threading.Thread(target=run, name="kt-offload-hostfunc", daemon=True)
    _STOP.clear()
    _MONITOR.start()
    atexit.register(stop)
    logger.info(
        "kt_offload: host-func counter sampling every %.1fs for %d layers "
        "(VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG); diagnostic only",
        interval_s,
        expected_layers,
    )
    return _MONITOR


def stop() -> None:
    """Stop the sampler; safe to call more than once."""
    global _MONITOR
    _STOP.set()
    thread, _MONITOR = _MONITOR, None
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
