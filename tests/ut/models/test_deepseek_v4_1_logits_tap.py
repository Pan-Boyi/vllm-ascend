# SPDX-License-Identifier: Apache-2.0
"""The first token's full-vocabulary logits.

Needed because judgement v2's threshold is a KL, and a KL needs the
candidate's mass where *golden* has mass -- which the candidate's own top-k
cannot supply. Approximating it would bias the KL small, toward "pass".
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import logits_tap


class _Recorder:
    """Copied from AlignRecorder: the two methods this tap calls."""

    def __init__(self):
        self.whole = {}
        self.manifest = {}

    def record_whole(self, tag, key, tensor):
        self.whole[(tag, key)] = tensor

    def add_manifest(self, **kw):
        self.manifest.update(kw)


@pytest.fixture
def rec(monkeypatch):
    logits_tap.reset_for_test()
    r = _Recorder()
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align
    monkeypatch.setattr(debug_align, "get_recorder", lambda: r)
    import vllm_ascend.models.deepseek_v4_1.align_taps as align_taps
    monkeypatch.setattr(align_taps, "recording_armed", lambda: True)
    yield r
    logits_tap.reset_for_test()


def test_it_records_golden_s_slice_and_nothing_else(rec):
    """Shape and dtype are the claim, not an implementation detail.

    golden's reference is (1, T, 129280) float32 and the comparison takes
    [0, -1]. 🔴 "logits" has named four different tensors in this port -- pre
    and post gather, per row and per request -- so the key is pinned to one
    of them.
    """
    logits_tap.record_logits(torch.arange(12, dtype=torch.float32).reshape(3, 4))
    stored = rec.whole[("epilogue", logits_tap.KEY)]
    assert stored.dtype == torch.float32
    assert stored.shape == (4, )
    # The LAST row: the next-token logits for the final position.
    assert stored.tolist() == [8.0, 9.0, 10.0, 11.0]


def test_a_three_dimensional_input_is_sliced_like_golden(rec):
    logits_tap.record_logits(torch.arange(24, dtype=torch.float32).reshape(1, 3, 8))
    stored = rec.whole[("epilogue", logits_tap.KEY)]
    assert stored.shape == (8, )
    assert stored.tolist() == list(range(16, 24))


def test_a_bf16_source_is_named_rather_than_silently_upcast(rec):
    """🔴 A silent upcast would make a storage difference look like
    agreement: golden's reference is float32, and comparing a bf16-derived
    tensor against it is a comparison against a coarser quantity."""
    logits_tap.record_logits(torch.ones(1, 4, dtype=torch.bfloat16))
    assert rec.manifest["logits_source_dtype"] == "torch.bfloat16"
    assert rec.whole[("epilogue", logits_tap.KEY)].dtype == torch.float32
    assert rec.manifest["logits_vocab"] == 4


def test_an_unarmed_forward_records_nothing_but_is_counted(monkeypatch):
    """Dummy runs, profile runs and non-step-1 decodes. Counted, not silent:
    "never entered" and "entered and declined" are the whole diagnosis when
    the key is missing."""
    logits_tap.reset_for_test()
    r = _Recorder()
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align
    monkeypatch.setattr(debug_align, "get_recorder", lambda: r)
    import vllm_ascend.models.deepseek_v4_1.align_taps as align_taps
    monkeypatch.setattr(align_taps, "recording_armed", lambda: False)

    logits_tap.record_logits(torch.ones(1, 4))
    assert r.whole == {}
    assert logits_tap._CALLS == 1
    assert logits_tap._SKIPPED_UNARMED == 1


def test_zero_says_that_zero_is_normal_under_dp(monkeypatch, rec):
    """⚠️ Under DP the logits exist only on the serving rank, so zero is the
    CORRECT reading on three ranks in four. Without that sentence a normal
    dump and a broken tap produce the same line."""
    said = []
    monkeypatch.setattr("vllm.logger.logger.info",
                        lambda fmt, *a: said.append(fmt % a))

    logits_tap.report_logits_tap_coverage()
    assert "recorded 0 time(s) this dump" in said[0]
    assert "CORRECT on every rank that did not serve" in said[0]
    assert "entered 0 time(s)" in said[0], (
        "the entered count separates 'not this rank' from 'override not on "
        "the object the runner holds'")

    logits_tap.record_logits(torch.ones(1, 4))
    logits_tap.report_logits_tap_coverage()
    assert "recorded 1 time(s) this dump" in said[1], said[1]

    # 🔴 The running-total mistake, in the shape it would take here.
    logits_tap.report_logits_tap_coverage()
    assert "recorded 0 time(s) this dump" in said[2], said[2]
    assert "(1 since this process started" in said[2], said[2]


def test_the_override_is_on_the_class_the_runner_reaches():
    """🔴 The recurring failure in this port, four times over: the symbol is
    there, the module is there, and the path does not go through it.

    compute_logits is not a module forward, so there is no hook; and the
    runner reaches it through wrappers that delegate to ``original_model``.
    The override therefore has to be on the model class itself, and this
    asserts that it is -- not that some function exists somewhere.
    """
    import inspect

    from vllm_ascend.models.deepseek_v4_1.model import (
        AscendDeepseekV41ForCausalLM)

    assert "compute_logits" in vars(AscendDeepseekV41ForCausalLM), (
        "the override must be defined on this class, not inherited")
    src = inspect.getsource(AscendDeepseekV41ForCausalLM.compute_logits)
    assert "super().compute_logits" in src
    # Recorded after the call, never around it.
    assert src.index("super().compute_logits") < src.index("record_logits(")


def test_the_coverage_reader_is_actually_called_at_flush():
    """🔴 The routing tap's counter existed for three flights with no caller
    anywhere, so "it never ran" and "nobody printed it" were the same
    output. A reader that is defined and never called is not a reader."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import align_taps

    src = inspect.getsource(align_taps._flush_and_rotate)
    assert "report_logits_tap_coverage()" in src


def test_the_key_addresses_as_the_name_golden_declares():
    """🔴 A tap that runs and a key that matches are two different things.

    Keys are addressed ``<tag>/<key>``. Golden declares this one as
    ``epilogue/logits`` -- read out of a real dump's
    ``golden_coverage.missing``, next to ``epilogue/final_norm.out`` and
    ``layer_00/attn_norm.out``.

    The tap originally used ``KEY = "epilogue.logits.last"``, which would
    have been written as ``epilogue/epilogue.logits.last``. It would have
    flown, produced its 517 KB, and left ``epilogue/logits`` reported
    missing -- a dump that is well formed, a tensor that is right, and a
    coverage line saying the key was never recorded.

    ⚠️ This asserts the composed address, not the constant, because the
    defect lived in the composition rather than in either half.
    """
    from vllm_ascend.models.deepseek_v4_1.logits_tap import KEY, TAG

    assert f"{TAG}/{KEY}" == "epilogue/logits", (
        f"this tap writes {TAG}/{KEY}; golden declares epilogue/logits")
