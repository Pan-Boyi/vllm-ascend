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


def test_a_node_at_or_above_the_void_threshold_kills_form_c():
    ceil = M.ceiling(cpu_us=400.0, window_us=100.0, node_us=M.NODE_VOID_US, layers=40)
    out = M.verdict(ceil)
    assert out["worth_it"] is False and "void" in out["why"]


def test_a_node_that_eats_the_window_kills_form_c_even_below_the_threshold():
    out = M.verdict(M.ceiling(cpu_us=400.0, window_us=30.0, node_us=40.0, layers=40))
    assert out["worth_it"] is False and "nothing left to win" in out["why"]


def test_the_verdict_compares_against_the_measured_packed_gain_when_there_is_one():
    ceil = M.ceiling(cpu_us=400.0, window_us=100.0, node_us=5.0, layers=40)  # 3.8 ms/step
    assert M.verdict(ceil, packed_gain_ms=1.0)["worth_it"] is True
    loses = M.verdict(ceil, packed_gain_ms=9.0)
    assert loses["worth_it"] is False and "reducing operations (packed)" in loses["why"]


def test_the_verdict_refuses_to_decide_with_nothing_to_compare_against():
    out = M.verdict(M.ceiling(cpu_us=400.0, window_us=100.0, node_us=5.0, layers=40))
    assert out["worth_it"] is None and "no measured packed gain" in out["why"]


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
