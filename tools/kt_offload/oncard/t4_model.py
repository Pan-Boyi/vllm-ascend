# SPDX-License-Identifier: Apache-2.0
"""T4's arithmetic and verdict, with no device in it, so the CPU tests can
exercise every branch (including the ones that must refuse to conclude).

Nothing here measures. Each function takes samples that t4 collected on card
and turns them into the three numbers the decision needs:

  1. node_us   -- what one host-func node costs the compute stream
  2. window_us -- how much NPU work sits between the submit and the wait
  3. ceiling   -- min(cpu_us, window_us) - node_us, the best case per layer

and then says whether form C is worth building. Distributions, never single
points: every input is a list of paired per-round samples.
"""

from __future__ import annotations

import math

# A host-func node costing this much or more eats the whole window: form C is
# void at that point whatever else holds.
NODE_VOID_US = 100.0


def stats(samples) -> dict:
    """median / p10 / p90 / n of one arm. Refuses to summarise too few."""
    xs = sorted(float(x) for x in samples)
    if len(xs) < 20:
        raise ValueError(f"t4 needs >= 20 samples per arm, got {len(xs)}")
    if any(not math.isfinite(x) for x in xs):
        raise ValueError("t4: a non-finite sample; the timer did not read")

    def q(p: float) -> float:
        i = p * (len(xs) - 1)
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)

    return {"median": q(0.5), "p10": q(0.1), "p90": q(0.9), "n": len(xs), "min": xs[0], "max": xs[-1]}


def paired_delta(with_samples, without_samples) -> dict:
    """Per-round difference, not a difference of medians.

    The arms are replayed interleaved (one round = one replay of each), so the
    round index pairs them and a drift that moves both arms together drops out.
    """
    if len(with_samples) != len(without_samples):
        raise ValueError(f"t4: unpaired arms ({len(with_samples)} vs {len(without_samples)})")
    return stats([a - b for a, b in zip(with_samples, without_samples)])


def negative_control(graph_samples, eager_samples, tol: float) -> tuple[bool, str]:
    """The no-node arm must read like the status quo.

    Both arms run the same NPU work on the same input; if the graph arm is much
    cheaper, the graph did not run what we think it ran, and every delta built
    on it is meaningless. Reported as a ratio, with the tolerance named.
    """
    g, e = stats(graph_samples)["median"], stats(eager_samples)["median"]
    if e <= 0:
        return False, "the eager arm read 0 us: the timer is broken, not the graph"
    ratio = g / e
    ok = (1 - tol) <= ratio <= (1 + tol)
    return ok, f"graph no-node median {g:.1f} us vs eager {e:.1f} us (ratio {ratio:.2f}, tol +/-{tol:.0%})"


def instrument_control(fat_delta_us: float, fat_us: float) -> tuple[bool, str]:
    """A node that burns ``fat_us`` must show up. Without this arm a small
    node delta cannot be told from a measurement that sees no node at all."""
    need = 0.6 * fat_us
    ok = fat_delta_us >= need
    return ok, f"deliberately fat node ({fat_us:.0f} us) measured as {fat_delta_us:.1f} us (need >= {need:.0f})"


def cross_check(t4_node_us: float, ab_node_us: float | None, factor: float = 2.0) -> dict:
    """t4's node cost against the A/B-derived per-operation cost.

    Two independent sources. Agreeing is what makes either usable; disagreeing
    is a finding, not a menu -- this never returns a preferred value.
    """
    if ab_node_us is None:
        return {"state": "unavailable", "line": "no A/B-derived per-operation cost given: not cross-checked"}
    if ab_node_us <= 0 or t4_node_us <= 0:
        lo = min(ab_node_us, t4_node_us)
        return {
            "state": "inconclusive",
            "line": f"one source is <= 0 (t4 {t4_node_us:.1f} us, A/B {ab_node_us:.1f} us, min {lo:.1f}): "
            "a ratio would be meaningless",
        }
    ratio = max(t4_node_us, ab_node_us) / min(t4_node_us, ab_node_us)
    agree = ratio <= factor
    return {
        "state": "agree" if agree else "inconsistent",
        "ratio": ratio,
        "line": f"t4 node {t4_node_us:.1f} us vs A/B-derived {ab_node_us:.1f} us: ratio {ratio:.2f} "
        f"({'within' if agree else 'OUTSIDE'} {factor:g}x) -- "
        + ("both usable" if agree else "INCONSISTENT: neither number may be used until this is resolved"),
    }


def ceiling(cpu_us: float, window_us: float, node_us: float, layers: int) -> dict:
    """Per-layer best case and what it is worth per step.

    The wait cannot move before the merge, so the window is this layer's own
    NPU work; the CPU cannot contribute more than it takes; and the node is
    paid whether or not the overlap wins anything.
    """
    overlappable = min(cpu_us, window_us)
    per_layer = overlappable - node_us
    return {
        "cpu_us": cpu_us,
        "window_us": window_us,
        "node_us": node_us,
        "overlappable_us": overlappable,
        "bound_by": "cpu" if cpu_us < window_us else "npu_window",
        "per_layer_us": per_layer,
        "per_step_ms": per_layer * layers / 1000.0,
        "layers": layers,
    }


def verdict(ceil: dict, packed_gain_ms: float | None = None) -> dict:
    """Worth building, or not. One sentence, and the reason it rests on."""
    node_us, per_layer, per_step = ceil["node_us"], ceil["per_layer_us"], ceil["per_step_ms"]
    if node_us >= NODE_VOID_US:
        return {
            "worth_it": False,
            "why": f"one host-func node costs {node_us:.1f} us >= {NODE_VOID_US:.0f} us: it eats the whole "
            f"window ({ceil['window_us']:.1f} us). Form C is void on this measurement alone.",
        }
    if per_layer <= 0:
        return {
            "worth_it": False,
            "why": f"the node ({node_us:.1f} us) costs at least as much as the overlappable part "
            f"({ceil['overlappable_us']:.1f} us, bound by {ceil['bound_by']}): nothing left to win.",
        }
    line = (
        f"best case {per_layer:.1f} us/layer = {per_step:.2f} ms/step "
        f"(overlappable {ceil['overlappable_us']:.1f} us, bound by {ceil['bound_by']}, "
        f"minus {node_us:.1f} us of node)"
    )
    if packed_gain_ms is not None:
        cheaper = "form C" if per_step > packed_gain_ms else "reducing operations (packed)"
        line += f"; measured packed gain {packed_gain_ms:.2f} ms/step, so {cheaper} is the bigger of the two"
        return {"worth_it": per_step > packed_gain_ms, "why": line}
    return {"worth_it": None, "why": line + "; no measured packed gain to compare against yet"}
