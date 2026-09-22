# SPDX-License-Identifier: Apache-2.0
"""t5's scenario matrix, resolved on the CPU before a card is booked.

t5 needs a card; deciding *what* it will run does not. The sweep added on
2026-09-22 for the single-layer bit-identity arm (RUNBOOK_sidestream.md item
1) takes ``--layer 0,20,39 --tokens 1,4,32``, and the two ways that can be
silently wrong -- a window running past the last layer, and two scenarios
sharing one serial reference -- are settled here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ONCARD = Path(__file__).resolve().parents[3] / "tools" / "kt_offload" / "oncard"
sys.path.insert(0, str(ONCARD))
spec = importlib.util.spec_from_file_location("t5_auto_dispatch", ONCARD / "t5_auto_dispatch.py")
T5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T5)


def test_the_runbook_matrix_resolves_to_nine_scenarios():
    sc = T5.plan_scenarios("0,20,39", "1,4,32", 3)
    assert len(sc) == 9
    assert [s["layer_ids"][0] for s in sc[::3]] == [0, 20, 37]
    assert sorted({s["tokens"] for s in sc}) == [1, 4, 32]


def test_a_window_past_the_last_layer_is_shifted_and_says_so():
    """--layer 39 --layers 3 on a 40-layer model = the last three layers.

    Silently dropping it would leave the tail of the model untested while the
    command line still said 39; silently running layers 39/40/41 would ask the
    checkpoint for layers that do not exist.
    """
    (sc,) = T5.plan_scenarios("39", "1", 3)
    assert sc["layer_ids"] == [37, 38, 39] and sc["shifted"] is True
    (clean,) = T5.plan_scenarios("20", "1", 3)
    assert clean["layer_ids"] == [20, 21, 22] and clean["shifted"] is False


def test_every_window_covers_both_kt_pinned_slots():
    """A6: the point of >= 3 layers is that L and L+2 share kt's slot L%2."""
    for sc in T5.plan_scenarios("0,20,39", "1,32", 3):
        assert sc["slots"] == [0, 1], sc


def test_scenarios_are_distinct_so_none_can_reuse_anothers_reference():
    keys = [T5.scenario_key("on", sc) for sc in T5.plan_scenarios("0,20,39", "1,4,32", 3)]
    assert len(set(keys)) == len(keys)
    assert "on@L37_T32" in keys


@pytest.mark.parametrize(
    "layer, tokens, layers, match",
    [
        ("20", "1", 2, "layers must be >= 3"),
        ("99", "1", 3, "past the model"),
        ("20", "0", 3, ">= 1"),
        ("20", "", 3, "empty list"),
        ("x", "1", 3, "not an integer"),
        ("-1", "1", 3, "negative"),
        ("20", "1", 41, "exceeds the model"),
    ],
)
def test_refusals(layer, tokens, layers, match):
    with pytest.raises(ValueError, match=match):
        T5.plan_scenarios(layer, tokens, layers)


def test_duplicates_collapse_rather_than_running_twice():
    assert len(T5.plan_scenarios("20,20", "1,1", 3)) == 1


def test_the_mutant_arm_is_declared_and_explained():
    """The drop-join mutant is what gives the bit-identity criterion teeth."""
    assert "drop-join" in T5.PLAN and "discriminating power" in T5.PLAN
    src = (ONCARD / "t5_auto_dispatch.py").read_text()
    # the mutant records the join on the side stream and skips the compute
    # stream's wait -- if it skipped the record too, it would only be
    # re-running the *_evt arms' question
    assert "join.record(side)" in src
    assert "must NOT come out captured-and-bit-identical" in src
