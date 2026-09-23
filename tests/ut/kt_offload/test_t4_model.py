# SPDX-License-Identifier: Apache-2.0
"""T4's arithmetic and refusals, on the CPU.

t4 itself needs a card; what it concludes does not. Everything that turns
samples into the three numbers and the verdict lives in
``tools/kt_offload/oncard/t4_model.py`` so the branches that must *refuse*
(too few samples, a dead timer, two sources that disagree) can be exercised
here, before a booked window pays for finding out.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

MODEL = Path(__file__).resolve().parents[3] / "tools" / "kt_offload" / "oncard" / "t4_model.py"
spec = importlib.util.spec_from_file_location("t4_model", MODEL)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


# -- stats ---------------------------------------------------------------


def test_stats_refuses_a_sample_set_too_small_to_read():
    with pytest.raises(ValueError, match=">= 20"):
        M.stats(list(range(19)))
    assert M.stats(list(range(20)))["n"] == 20


def test_stats_refuses_a_dead_timer():
    xs = [1.0] * 19 + [float("nan")]
    with pytest.raises(ValueError, match="non-finite"):
        M.stats(xs)


def test_stats_reports_the_spread_not_just_the_middle():
    s = M.stats(list(range(100, 200)))
    assert s["median"] == pytest.approx(149.5)
    assert s["p10"] < s["median"] < s["p90"]
    assert s["min"] == 100 and s["max"] == 199


# -- pairing -------------------------------------------------------------


def test_paired_delta_recovers_a_constant_offset():
    base = [100 + (i % 7) for i in range(24)]
    withn = [b + 12 for b in base]
    assert M.paired_delta(withn, base)["median"] == pytest.approx(12)


def test_pairing_survives_a_drift_that_moves_both_arms():
    """Why the arms are interleaved and paired by round.

    Measured interleaved, a drift that moves both arms cancels. Measured one
    arm after the other -- the obvious way to write this harness -- the same
    drift lands entirely in the difference, and 12 us of node cost reads as
    hundreds.
    """
    drift = [i * 50.0 for i in range(24)]  # clocks settle, the card warms up
    base = [100.0 + d for d in drift]  # arm without the node, round i
    withn = [b + 12.0 for b in base]  # arm with the node, same round i
    assert M.paired_delta(withn, base)["median"] == pytest.approx(12.0)

    # the same measurement taken sequentially: rounds 0..23 then 24..47
    seq_base = [100.0 + i * 50.0 for i in range(24)]
    seq_with = [112.0 + i * 50.0 for i in range(24, 48)]
    naive = M.stats(seq_with)["median"] - M.stats(seq_base)["median"]
    assert naive > 1000, f"the sequential form should be swamped by the drift, got {naive}"


def test_paired_delta_refuses_unpaired_arms():
    with pytest.raises(ValueError, match="unpaired"):
        M.paired_delta([1.0] * 24, [1.0] * 23)


# -- controls ------------------------------------------------------------


def test_negative_control_passes_when_the_graph_matches_eager():
    xs = [100 + (i % 5) for i in range(24)]
    ok, line = M.negative_control(xs, xs, 0.35)
    assert ok and "ratio 1.00" in line


def test_negative_control_goes_red_when_the_graph_ran_something_else():
    eager = [100.0] * 24
    graph = [5.0] * 24  # a graph that did not contain the MoE
    ok, line = M.negative_control(graph, eager, 0.35)
    assert not ok and "ratio 0.05" in line


def test_negative_control_names_a_dead_eager_arm_instead_of_dividing():
    ok, line = M.negative_control([1.0] * 24, [0.0] * 24, 0.35)
    assert not ok and "timer is broken" in line


def test_instrument_control_is_red_when_a_fat_node_is_invisible():
    assert M.instrument_control(480.0, 500.0)[0]
    ok, line = M.instrument_control(3.0, 500.0)
    assert not ok and "need >= 300" in line


# -- cross-source --------------------------------------------------------


def test_cross_check_says_so_when_there_is_no_second_source():
    out = M.cross_check(40.0, None)
    assert out["state"] == "unavailable" and "not cross-checked" in out["line"]


def test_cross_check_agrees_and_disagrees_without_picking_a_winner():
    assert M.cross_check(40.0, 60.0)["state"] == "agree"
    bad = M.cross_check(5.0, 62.5)
    assert bad["state"] == "inconsistent"
    assert "INCONSISTENT" in bad["line"] and "neither number may be used" in bad["line"]
    # it never hands back a preferred value
    assert not any(k in bad for k in ("use", "value", "best"))


def test_cross_check_refuses_a_ratio_against_zero():
    assert M.cross_check(40.0, 0.0)["state"] == "inconclusive"


# -- the ceiling and the verdict -----------------------------------------


def test_ceiling_is_bound_by_whichever_side_is_shorter():
    cpu_bound = M.ceiling(cpu_us=60.0, window_us=100.0, node_us=5.0, layers=40)
    assert cpu_bound["bound_by"] == "cpu" and cpu_bound["per_layer_us"] == pytest.approx(55.0)
    assert cpu_bound["per_step_ms"] == pytest.approx(2.2)
    npu_bound = M.ceiling(cpu_us=400.0, window_us=100.0, node_us=5.0, layers=40)
    assert npu_bound["bound_by"] == "npu_window" and npu_bound["per_layer_us"] == pytest.approx(95.0)


def test_a_node_as_big_as_the_measured_window_kills_form_c():
    """The rule is relative to this run's measured window -- there is no
    absolute per-operation threshold any more (the price it came from is
    retired), so the same node cost decides differently on different cards."""
    kills = M.verdict(M.ceiling(cpu_us=400.0, window_us=100.0, node_us=100.0, layers=40))
    assert kills["worth_it"] is False and "eats the window" in kills["why"]
    # the identical node against a bigger measured window is not fatal
    survives = M.verdict(M.ceiling(cpu_us=400.0, window_us=900.0, node_us=100.0, layers=40))
    assert survives["worth_it"] is None and "300.0 us/layer" in survives["why"]  # cpu-bound: 400 - 100


def test_the_model_has_no_absolute_price_constant_left():
    assert not hasattr(M, "NODE_VOID_US"), "an absolute threshold came back; it was built on a retired price"
    src = (Path(M.__file__).read_text() if getattr(M, "__file__", None) else MODEL.read_text())
    for retired in ("62.5", "37.5", "0.62 ms"):
        assert retired not in src, f"{retired} is back in the decision model"


def test_a_node_bigger_than_the_cpu_side_kills_it_even_with_a_wide_window():
    """The other way round: plenty of NPU work to hide behind, but the CPU
    finishes sooner than the node costs, so there is nothing to overlap."""
    out = M.verdict(M.ceiling(cpu_us=30.0, window_us=900.0, node_us=40.0, layers=40))
    assert out["worth_it"] is False and "nothing left to win" in out["why"]


def test_the_verdict_compares_against_the_measured_packed_gain_when_there_is_one():
    ceil = M.ceiling(cpu_us=400.0, window_us=100.0, node_us=5.0, layers=40)  # 3.8 ms/step
    assert M.verdict(ceil, packed_gain_ms=1.0)["worth_it"] is True
    loses = M.verdict(ceil, packed_gain_ms=9.0)
    assert loses["worth_it"] is False and "the other route" in loses["why"]


def test_the_verdict_refuses_to_decide_with_nothing_to_compare_against():
    out = M.verdict(M.ceiling(cpu_us=400.0, window_us=100.0, node_us=5.0, layers=40))
    assert out["worth_it"] is None and "no measured alternative" in out["why"]


def test_the_mutation_arm_t4_runs_on_card_flips_the_verdict():
    """The same arm t4 executes with measured data: substituting the fat
    node's cost must turn the answer over. If it does not, the decision rule
    is insensitive to the very number the test was booked to measure."""
    cpu_us, window_us, layers = 400.0, 100.0, 40
    real = M.verdict(M.ceiling(cpu_us, window_us, 5.0, layers), packed_gain_ms=1.0)
    mutated = M.verdict(M.ceiling(cpu_us, window_us, 500.0, layers), packed_gain_ms=1.0)
    assert real["worth_it"] is True and mutated["worth_it"] is False


def test_per_step_scales_with_the_layer_count_and_nothing_else():
    a = M.ceiling(400.0, 100.0, 5.0, 40)["per_step_ms"]
    b = M.ceiling(400.0, 100.0, 5.0, 20)["per_step_ms"]
    assert math.isclose(a, 2 * b)


# -- the proxy (t4_proxy.py): the declared criterion, and its refusals ------
#
# 🔴 These constants are the criterion PROXY_CRITERIA.md declared BEFORE the
# first proxy flight. A test that asserts them is what makes "we moved the bar
# after seeing the number" a visible diff rather than a memory.


def test_the_declared_proxy_criterion_is_the_one_in_the_file():
    assert M.PROXY_REL_TOL == 0.50
    assert M.PROXY_MIN_CALIBRATED == 2


def test_per_layer_delta_pairs_first_and_divides_afterwards():
    # arm A is 80 us slower per round over an 8-layer stack, with a drift on top
    a = [1000.0 + 80.0 + 3.0 * i for i in range(24)]
    b = [1000.0 + 3.0 * i for i in range(24)]
    d = M.per_layer_delta(a, b, 8)
    assert d["median"] == pytest.approx(10.0)
    assert d["layers"] == 8


def test_per_layer_delta_refuses_a_stack_with_no_layers():
    with pytest.raises(ValueError, match="layers must be >= 1"):
        M.per_layer_delta([1.0] * 24, [0.0] * 24, 0)


def test_the_prediction_is_the_per_layer_figure_times_the_model_depth():
    d = M.per_layer_delta([-75.0 * 8] * 24, [0.0] * 24, 8)
    p = M.predict_ms_per_token(d, 40)
    assert p["median"] == pytest.approx(-3.0)
    assert p["per_layer_us"] == pytest.approx(-75.0)
    assert p["model_layers"] == 40


def test_a_difference_that_straddles_zero_is_not_resolved():
    ok, line = M.resolved({"median": 1.0, "p10": -2.0, "p90": 3.0})
    assert not ok and "NOT resolved" in line
    ok, _ = M.resolved({"median": -1.0, "p10": -3.0, "p90": -0.2})
    assert ok


def test_a_difference_of_exactly_zero_is_not_resolved_either():
    ok, line = M.resolved({"median": 0.0, "p10": -1.0, "p90": 1.0})
    assert not ok and "nothing to resolve" in line


def test_calibrate_accepts_a_prediction_inside_the_declared_tolerance():
    c = M.calibrate(-3.9, -3.00)
    assert c["state"] == "agree" and c["rel_err"] == pytest.approx(0.30)
    assert c["bias"] == "over"


def test_calibrate_rejects_a_prediction_outside_it_without_rescaling_anything():
    c = M.calibrate(-0.01, -3.00)
    assert c["state"] == "sign_only"
    # the point of the refusal: it never hands back a corrected number
    assert "factor" not in c and c["pred_ms"] == -0.01


def test_calibrate_calls_the_wrong_sign_what_it_is():
    assert M.calibrate(+3.00, -3.00)["state"] == "sign_disagrees"
    assert M.calibrate(0.0, -3.00)["state"] == "sign_disagrees"


def test_calibrate_says_so_when_there_is_nothing_to_calibrate_against():
    assert M.calibrate(-3.0, None)["state"] == "unmeasured"
    assert M.calibrate(-3.0, 0.0)["state"] == "inconclusive"


def test_one_mechanism_is_never_enough_to_declare_the_proxy_usable():
    v = M.proxy_verdict({"sidestream": M.calibrate(-3.1, -3.00)})
    assert v["state"] == "unusable" and "only 1 mechanism" in v["why"]


def test_two_mechanisms_inside_the_tolerance_make_it_usable():
    v = M.proxy_verdict({"sidestream": M.calibrate(-3.1, -3.00), "p1a": M.calibrate(-0.60, -0.53)})
    assert v["state"] == "usable" and v["n_measured"] == 2


def test_the_right_sign_but_the_wrong_size_buys_ordering_only():
    v = M.proxy_verdict({"sidestream": M.calibrate(-3.1, -3.00), "p1a": M.calibrate(-0.05, -0.53)})
    assert v["state"] == "ranking_only" and "p1a" in v["why"]


def test_a_single_sign_flip_makes_the_whole_proxy_unusable():
    v = M.proxy_verdict({"sidestream": M.calibrate(-3.1, -3.00), "p1a": M.calibrate(+0.60, -0.53)})
    assert v["state"] == "unusable" and "SIGN" in v["why"]


def test_an_unresolved_mechanism_does_not_count_towards_the_criterion():
    cals = {"sidestream": M.calibrate(-3.1, -3.00),
            "p1a": {"state": "inconclusive", "line": "the proxy did not resolve p1a in this run"}}
    assert M.proxy_verdict(cals)["state"] == "unusable"


def test_the_null_mechanism_must_not_calibrate_as_the_side_stream():
    """The mutation arm t4_proxy runs on card, here with the numbers it needs.

    Two captures of one configuration differ by nothing, so whatever the proxy
    reads between them is its own floor. If +-50% were wide enough to accept
    that floor as -3.00 ms/token, the criterion would have no discriminating
    power at all.
    """
    for floor_ms in (0.0, 0.05, -0.2, 0.4):
        assert M.calibrate(floor_ms, -3.00)["state"] != "agree"
    # and the criterion is not vacuous: a floor that really were -3 would pass
    assert M.calibrate(-3.0, -3.00)["state"] == "agree"


def test_a_fat_node_read_as_a_mechanism_comes_out_with_the_wrong_sign():
    # 500 us/layer x 40 layers = +20 ms/token, against a measured -3.00 saving
    assert M.calibrate(500.0 * 40 / 1000.0, -3.00)["state"] == "sign_disagrees"
