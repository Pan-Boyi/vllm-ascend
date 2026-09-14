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


# --- the experts key names what it actually holds ---------------------------

def test_the_key_is_named_for_what_the_module_actually_returns():
    """🔴 The hardest same-name-different-thing: nothing in align_taps was
    wrong.

    The hook point was right, the key name was right for what it was *meant*
    to be, and the docstring even explained why the key existed. What drifted
    was the description against **what the attached module is on the path
    taken at runtime** -- a fact that lives in another file's other branch
    (fused_moe.py:304 returns early without shared, :320+ returns
    routed+shared, and the condition is ascend_shared_experts is not None).
    golden nearly derived a defect that does not exist from it.
    """
    from vllm_ascend.models.deepseek_v4_1.align_taps import (
        experts_include_shared)

    class _Experts(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.ascend_shared_experts = shared

    assert experts_include_shared(_Experts(torch.nn.Identity())) is True
    assert experts_include_shared(_Experts(None)) is False
    # ⚠️ Read off the live object, so a module that has never heard of the
    # attribute is the mega-path answer, not a crash.
    assert experts_include_shared(torch.nn.Identity()) is False


def test_the_reconstruction_refuses_when_the_parts_would_double_count(
        monkeypatch):
    """🔴 Refused structurally, not by the parts happening to be absent.

    Today it does not fire only because moe.shared.out is absent as well --
    a coincidence of which submodules exist. The next build that exposes
    shared_experts as a module would turn this from an error into a silently
    wrong reading: a residual of about 73% of the norm, large and confident
    and meaningless.
    """
    from vllm_ascend.models.deepseek_v4_1 import align_taps

    said = []
    monkeypatch.setattr("vllm.logger.logger.info",
                        lambda fmt, *a: said.append(fmt % a))
    align_taps._REPORTED_RECONSTRUCTION_NA.clear()
    # Both parts present AND the fused key present: the double-count setup.
    align_taps._MOE_PARTS[3] = {
        "moe.fused_out": torch.ones(2, 4),
        "moe.experts.out": torch.ones(2, 4),
        "moe.shared.out": torch.ones(2, 4),
    }
    align_taps.check_moe_reconstruction(3, torch.ones(2, 4))

    assert any("NOT APPLICABLE" in t for t in said), said
    assert any("twice" in t for t in said), said
    # 🔴 And it must not be mistaken for the missing-parts case: the two
    # readings lead to opposite next steps (go find the tap / stop asking).
    assert not any("could NOT be computed" in t for t in said), said


def test_the_attach_code_actually_uses_the_predicate(monkeypatch):
    """🔴 The predicate being right is not the same as the key being right.

    Written with only the predicate under test, a mutant that computes
    ``includes`` and then always attaches the old name stays green -- the
    answer is correct and nothing acts on it. That is the same shape as a
    counter with no reader: the piece that decides works, and the piece that
    uses it is what nobody drove.
    """
    from vllm_ascend.models.deepseek_v4_1 import align_taps

    keys = []
    monkeypatch.setattr(align_taps, "_make_moe_tap",
                        lambda layer_id, key: keys.append(key) or (
                            lambda *a, **k: None))
    monkeypatch.setattr(align_taps, "_report_moe_configuration",
                        lambda ffn: None)

    class _Experts(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.ascend_shared_experts = shared

    class _Ffn(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.experts = _Experts(shared)

    class _Layer(torch.nn.Module):
        def __init__(self, shared):
            super().__init__()
            self.ffn = _Ffn(shared)

    align_taps._attach_moe_internals(_Layer(torch.nn.Identity()), 0, [])
    assert "moe.fused_out" in keys, keys
    assert "moe.experts.out" not in keys, (
        "the fused path must not publish the routed-only name", keys)

    keys.clear()
    align_taps._attach_moe_internals(_Layer(None), 0, [])
    assert "moe.experts.out" in keys, keys
    assert "moe.fused_out" not in keys, keys
