# SPDX-License-Identifier: Apache-2.0
"""Recording the first decode step, and refusing to record anything else.

The two arms are bit-identical through prefill (923/923 keys, zero real
tensors differing) and already 0.6 logprob apart at decode step 1. So the
divergence is born in one forward, and only that forward is worth a dump.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import decode_step1
from vllm_ascend.models.deepseek_v4_1.debug_align import AlignRecorder


@pytest.fixture(autouse=True)
def _clean():
    decode_step1.reset_for_test()
    yield
    decode_step1.reset_for_test()


def test_step_two_is_not_recorded_over_step_one():
    """🔴 Step 2 also has exactly one query row.

    Without the already-recorded check it would arm again and overwrite step
    1 with a dump carrying the same key names, the same shapes and a
    different forward -- the artifact would be well formed and would be
    describing the wrong step.
    """
    decode_step1.note_prefill("req:7")
    assert decode_step1.is_first_decode_step("req:7", rows=1, seq_len=6)
    decode_step1.note_decode_recorded("req:7")
    assert not decode_step1.is_first_decode_step("req:7", rows=1, seq_len=7)


def test_a_request_whose_prefill_was_not_recorded_is_not_recorded():
    """There is nothing on the other side to compare its decode against, and
    a lone decode dump reads exactly like half of a pair."""
    assert not decode_step1.is_first_decode_step("req:9", rows=1, seq_len=6)


def test_an_unknown_identity_refuses_rather_than_guessing():
    """🔴 "the forward after the prefill" is only the same request while
    nothing interleaves, and when something does the error is silent.

    ⚠️ The identity guard is isolated deliberately. Written the obvious way
    -- ``note_prefill(None)`` then assert refusal -- this passes with the
    guard deleted, because ``note_prefill`` drops None and the *prefill*
    check is what refuses. The construction has to make the None the only
    thing standing between the call and a recording, or it tests the
    neighbour and reports the guard.
    """
    decode_step1._SEEN_PREFILL.add(None)          # only the guard can refuse
    assert not decode_step1.is_first_decode_step(None, rows=1, seq_len=6)

    # And the neighbour it was being confused with still works on its own.
    decode_step1._SEEN_PREFILL.discard(None)
    assert not decode_step1.is_first_decode_step(None, rows=1, seq_len=6)


def test_multi_row_forwards_are_not_decode_step_one():
    decode_step1.note_prefill("req:7")
    assert not decode_step1.is_first_decode_step("req:7", rows=5, seq_len=6)
    assert not decode_step1.is_first_decode_step("req:7", rows=2, seq_len=6)


def test_two_requests_do_not_share_the_decode_slot():
    """Identity, not ordering: req A's prefill must not arm req B's decode."""
    decode_step1.note_prefill("req:1")
    assert not decode_step1.is_first_decode_step("req:2", rows=1, seq_len=6)
    assert decode_step1.is_first_decode_step("req:1", rows=1, seq_len=6)


def _recorder(tmp_path, seq_len, expected):
    rec = AlignRecorder(str(tmp_path), seq_len=seq_len)
    rec.expected_positions = expected
    return rec


def test_one_row_at_its_own_position_is_complete(tmp_path):
    """A decode dump is complete when it holds exactly the position it owns.

    ⚠️ The prefill criterion -- every position written once -- would refuse
    every key here, and "refused" is the artifact a broken row mapping
    produces. Narrowing what is expected must not turn a correct decode dump
    into one that reads like a defect.
    """
    rec = _recorder(tmp_path, seq_len=6, expected=[5])
    rec.record("layer_00", "moe.out", torch.ones(1, 4),
               torch.tensor([5], dtype=torch.int64))
    assert rec._coverage_problems() == []


def test_a_row_at_the_wrong_position_is_still_refused(tmp_path):
    """🔴 The narrowing must not weaken the assertion it narrows.

    A criterion of "some position was written" would accept this, and a row
    scattered to a position the forward does not own is exactly the failure
    the coverage assertion exists for: the tensor looks right and compares
    wrong, and downstream it reads as a model difference rather than as a
    mapping bug.
    """
    rec = _recorder(tmp_path, seq_len=6, expected=[5])
    rec.record("layer_00", "moe.out", torch.ones(1, 4),
               torch.tensor([2], dtype=torch.int64))
    problems = rec._coverage_problems()
    assert problems, "a row outside the expected position must be refused"
    assert "WROTE-OUTSIDE=[2]" in problems[0], problems
    assert "missing=[5]" in problems[0], problems


def test_the_prefill_criterion_is_unchanged(tmp_path):
    """expected_positions=None must behave byte-for-byte as before."""
    rec = _recorder(tmp_path, seq_len=3, expected=None)
    rec.record("layer_00", "x", torch.ones(3, 2),
               torch.tensor([0, 1, 2], dtype=torch.int64))
    assert rec._coverage_problems() == []

    rec2 = _recorder(tmp_path, seq_len=3, expected=None)
    rec2.record("layer_00", "x", torch.ones(2, 2),
                torch.tensor([0, 1], dtype=torch.int64))
    assert rec2._coverage_problems(), "a short prefill must still be refused"


def test_the_coverage_line_is_printed_including_zero(monkeypatch):
    """🔴 Both halves of the routing tap's lesson.

    That counter had no caller for three flights, and when it got one the
    line said "this dump" over a process-cumulative number. So: a reader
    that runs, and two numbers whose labels match their own scopes.
    """
    said = []
    monkeypatch.setattr("vllm.logger.logger.info",
                        lambda fmt, *a: said.append(fmt % a))

    decode_step1.report_decode_step1_coverage()
    assert "armed 0 time(s) this dump" in said[0]
    assert "ZERO this dump" in said[0], "zero must say what zero means"

    decode_step1.note_prefill("req:1")
    decode_step1.note_decode_recorded("req:1")
    decode_step1.report_decode_step1_coverage()
    assert "armed 1 time(s) this dump" in said[1], said[1]
    assert "(1 since this process started" in said[1], said[1]

    decode_step1.report_decode_step1_coverage()
    # 🔴 The running-total mistake, in the shape it would take here.
    assert "armed 0 time(s) this dump" in said[2], said[2]
    assert "(1 since this process started" in said[2], said[2]


def test_the_arming_check_count_is_reported_even_when_nothing_arms():
    """"never asked" and "asked and said no" are different situations, and
    the difference is the whole diagnosis when no decode dump appears."""
    decode_step1.is_first_decode_step("req:1", rows=1, seq_len=6)
    decode_step1.is_first_decode_step("req:1", rows=1, seq_len=6)
    assert decode_step1._ARM_CALLS == 2
