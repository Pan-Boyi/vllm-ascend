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

# No absolute threshold here on purpose. "A node costing >= 100 us kills it"
# was a rule of thumb built on a per-operation price the 2026-09-20 profile
# retired; what decides is the node against *this run's measured* window and
# CPU time, which is what ceiling() already computes.


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
    """Worth building, or not. One sentence, and the measurement it rests on.

    Everything is relative to this run's measured numbers: the node against
    the window it has to fit inside, and the result against whatever the other
    route actually bought. No absolute per-operation price appears anywhere.
    """
    node_us, per_layer, per_step = ceil["node_us"], ceil["per_layer_us"], ceil["per_step_ms"]
    if node_us >= ceil["window_us"]:
        return {
            "worth_it": False,
            "why": f"one host-func node costs {node_us:.1f} us and the whole window between submit and wait "
            f"is {ceil['window_us']:.1f} us: the node eats the window it was supposed to hide behind.",
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
        cheaper = "form C" if per_step > packed_gain_ms else "the other route"
        line += f"; measured alternative gain {packed_gain_ms:.2f} ms/step, so {cheaper} is the bigger of the two"
        return {"worth_it": per_step > packed_gain_ms, "why": line}
    return {"worth_it": None, "why": line + "; no measured alternative to compare against yet"}


# ---------------------------------------------------------------------------
# The proxy (t4_proxy.py): per-layer device time -> a predicted ms/token, and
# the calibration against whatever the whole-network A/B already measured.
#
# 🔴 The numbers below are the criterion this line declared BEFORE the first
# proxy flight ran (tools/kt_offload/oncard/PROXY_CRITERIA.md, committed in the
# same change as this code). They are constants here so that "we moved the bar
# after seeing the result" would show up as a diff.
# ---------------------------------------------------------------------------

PROXY_REL_TOL = 0.50        # |pred - meas| / |meas| must be <= this to call a mechanism calibrated
PROXY_MIN_CALIBRATED = 2    # how many mechanisms have to clear it before the proxy may be used


def per_layer_delta(a_samples, b_samples, layers: int) -> dict:
    """Arm A minus arm B, per layer of the measured stack.

    Both arms replay the SAME ``layers``-deep stack, interleaved round by
    round, so the difference is paired first and divided afterwards. Dividing
    two medians instead would throw away the pairing that makes a drift
    common to both arms cancel.
    """
    if layers < 1:
        raise ValueError(f"per_layer_delta: layers must be >= 1, got {layers}")
    d = paired_delta(a_samples, b_samples)
    out = {k: (v / layers if k in ("median", "p10", "p90", "min", "max") else v) for k, v in d.items()}
    out["layers"] = layers
    return out


def predict_ms_per_token(per_layer_us: dict, model_layers: int = 40) -> dict:
    """The whole-network figure the proxy predicts: per-layer us x layers.

    This is an extrapolation and says so: it assumes every layer of the model
    behaves like the layers in the measured stack, and that the mechanism
    changes nothing outside the device timeline.
    """
    out = {k: (v * model_layers / 1000.0 if k in ("median", "p10", "p90", "min", "max") else v)
           for k, v in per_layer_us.items()}
    out["model_layers"] = model_layers
    out["per_layer_us"] = per_layer_us["median"]
    return out


def resolved(delta: dict) -> tuple[bool, str]:
    """Did the proxy resolve this difference at all?

    p10 and p90 of the PAIRED difference must land on the same side of zero as
    the median. If they straddle it, the arms are not separated by this
    instrument in this run and the median is a coin flip with a decimal point.
    """
    m, lo, hi = delta["median"], delta["p10"], delta["p90"]
    if m == 0:
        return False, "the paired difference has a median of exactly 0: nothing to resolve"
    same = (lo > 0 and hi > 0 and m > 0) or (lo < 0 and hi < 0 and m < 0)
    return same, (f"paired difference median {m:.3f}, p10 {lo:.3f}, p90 {hi:.3f}: "
                  + ("resolved (p10 and p90 are on the median's side of zero)"
                     if same else "NOT resolved -- p10/p90 straddle zero"))


def calibrate(pred_ms: float, meas_ms: float | None, rel_tol: float = PROXY_REL_TOL) -> dict:
    """One mechanism's proxy prediction against its whole-network measurement.

    Never returns a corrected or preferred value: a proxy that has to be
    rescaled by a factor nobody measured is not a proxy, it is a fudge.
    """
    if meas_ms is None:
        return {"state": "unmeasured", "line": "no whole-network measurement given: not calibrated"}
    if meas_ms == 0:
        return {"state": "inconclusive",
                "line": "the whole-network measurement is 0 ms/token: a relative error would be meaningless"}
    same_sign = (pred_ms > 0) == (meas_ms > 0) and pred_ms != 0
    rel = abs(pred_ms - meas_ms) / abs(meas_ms)
    if not same_sign:
        state = "sign_disagrees"
    elif rel <= rel_tol:
        state = "agree"
    else:
        state = "sign_only"
    return {
        "state": state,
        "pred_ms": pred_ms,
        "meas_ms": meas_ms,
        "rel_err": rel,
        "bias": "over" if abs(pred_ms) > abs(meas_ms) else "under",
        "line": f"proxy {pred_ms:+.2f} vs measured {meas_ms:+.2f} ms/token: "
                f"relative error {rel:.0%} (tol {rel_tol:.0%}), "
                + {"agree": "CALIBRATED",
                   "sign_only": "same sign, outside tolerance -- ordering only",
                   "sign_disagrees": "OPPOSITE SIGN -- the proxy is wrong about this mechanism"}[state],
    }


def proxy_verdict(calibrations: dict, min_calibrated: int = PROXY_MIN_CALIBRATED) -> dict:
    """Is the proxy usable, and for what.

    ``usable``        every measured mechanism calibrated, and there are at
                      least ``min_calibrated`` of them: the proxy's ms/token
                      may be quoted as an estimate (still not as a result).
    ``ranking_only``  every measured mechanism got the sign right but at least
                      one is outside tolerance: the proxy may order candidates
                      and screen them, and its magnitudes may not be quoted.
    ``unusable``      any mechanism came out with the wrong sign, or fewer than
                      ``min_calibrated`` mechanisms were available to judge on.
    """
    measured = {k: c for k, c in calibrations.items() if c["state"] not in ("unmeasured", "inconclusive")}
    if any(c["state"] == "sign_disagrees" for c in measured.values()):
        bad = sorted(k for k, c in measured.items() if c["state"] == "sign_disagrees")
        return {"state": "unusable", "n_measured": len(measured),
                "why": f"the proxy got the SIGN wrong on {', '.join(bad)}: it does not even order these mechanisms"}
    if len(measured) < min_calibrated:
        return {"state": "unusable", "n_measured": len(measured),
                "why": f"only {len(measured)} mechanism(s) had a whole-network number to calibrate against; "
                       f"the criterion asks for {min_calibrated}. Not enough evidence either way."}
    n_ok = sum(1 for c in measured.values() if c["state"] == "agree")
    if n_ok == len(measured):
        return {"state": "usable", "n_measured": len(measured),
                "why": f"all {len(measured)} calibrated mechanisms are within the declared tolerance; "
                       "the proxy may be quoted as an estimate for screening, never as a result"}
    weak = sorted(k for k, c in measured.items() if c["state"] == "sign_only")
    return {"state": "ranking_only", "n_measured": len(measured),
            "why": f"every mechanism has the right sign but {', '.join(weak)} is outside the tolerance: "
                   "use the proxy to ORDER and SCREEN candidates only, never for a number"}
