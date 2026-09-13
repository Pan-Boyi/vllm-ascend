# SPDX-License-Identifier: Apache-2.0
"""A switch with no observable artifact cannot be used as an experiment.

Setting a switch and having its branch never run produces the same log as
setting it and having the branch run and not help. The only line mentioning
it either way is vLLM's own "Unknown vLLM environment variable detected",
which proves the variable was *set*, not that the code path was *taken* --
and a hypothesis was nearly reported as refuted on exactly that reading.
"""

import logging

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.switch_log import (announce_once,
                                                         reset_for_test)


@pytest.fixture
def lines():
    from vllm.logger import logger

    captured = []

    class _Collect(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    reset_for_test()
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        reset_for_test()


def test_it_announces_the_first_time(lines):
    assert announce_once("k", "something happened") is True
    assert any("switch in effect" in line and "something happened" in line
               for line in lines)


def test_it_stays_quiet_afterwards(lines):
    """A per-token decision point must not flood the log; the question it
    answers is answered by the first occurrence."""
    announce_once("k", "once")
    assert announce_once("k", "once") is False
    assert len([line for line in lines if "switch in effect" in line]) == 1


def test_different_switches_each_get_a_line(lines):
    announce_once("a", "first")
    announce_once("b", "second")
    assert len([line for line in lines if "switch in effect" in line]) == 2


# ----------------------------------------- each switch announces its branch

def test_the_mm_ranges_gate_announces_when_it_excludes(monkeypatch, lines):
    from vllm_ascend.attention.dsa_v1 import mm_ranges_have_spans

    monkeypatch.setenv("VLLM_ASCEND_DSV41_MM_RANGES_REQUIRE_SPANS", "1")
    assert mm_ranges_have_spans({0: []}) is False
    assert any("MM_RANGES_REQUIRE_SPANS=1 excluded" in line for line in lines)


def test_the_mm_ranges_gate_stays_quiet_when_it_does_not(monkeypatch, lines):
    """Two arms. Announcing on the path that was going to happen anyway says
    nothing about whether the switch changed something."""
    from vllm_ascend.attention.dsa_v1 import mm_ranges_have_spans

    monkeypatch.setenv("VLLM_ASCEND_DSV41_MM_RANGES_REQUIRE_SPANS", "1")
    assert mm_ranges_have_spans({0: [(1, 2)]}) is True
    assert not any("MM_RANGES_REQUIRE_SPANS" in line for line in lines)


def test_the_mask_mode_switch_announces_the_mode_it_sent(monkeypatch, lines):
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        ori_mask_mode_for

    monkeypatch.setenv("VLLM_ASCEND_DSV41_ORI_WINDOW_WITHOUT_INDICES", "1")
    assert ori_mask_mode_for(True, None) == 3
    assert any("ori_mask_mode=3" in line for line in lines)


def test_the_mask_mode_switch_is_quiet_when_off(monkeypatch, lines):
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        ori_mask_mode_for

    monkeypatch.delenv("VLLM_ASCEND_DSV41_ORI_WINDOW_WITHOUT_INDICES",
                       raising=False)
    assert ori_mask_mode_for(True, None) == 0
    assert not any("ORI_WINDOW_WITHOUT_INDICES" in line for line in lines)


def test_the_engram_mode_announces_from_the_copy(monkeypatch, lines):
    """🔴 From the copy, not from the parse. Parsing happens whether or not
    the branch is ever reached, and that is the whole distinction."""
    from tests.ut.models.test_deepseek_v4_1_engram_embedding import (  # noqa
        LAYOUT, _Lookup, _tables)

    monkeypatch.setenv("VLLM_ASCEND_DSV41_ENGRAM_BLOCKING_H2D", "double")
    layer = _Lookup(_tables())
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
    layer.lookup(torch.zeros(2, 24, dtype=torch.int32), out=None)

    assert any("engram staging H2D mode 'double' reached the copy" in line
               for line in lines)


def test_nothing_is_announced_when_the_lookup_never_runs(monkeypatch, lines):
    """The failure this whole module is about: the value is set, the branch
    is not taken, and the log reads the same."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_ENGRAM_BLOCKING_H2D", "double")
    assert not any("reached the copy" in line for line in lines)


def test_the_announcement_changes_no_behaviour(monkeypatch):
    """⚠️ Evidence, not a guard. Deleting these lines must leave the run
    identical and only the reading of it worse."""
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        ori_mask_mode_for

    monkeypatch.setenv("VLLM_ASCEND_DSV41_ORI_WINDOW_WITHOUT_INDICES", "1")
    reset_for_test()
    first = ori_mask_mode_for(True, None)
    second = ori_mask_mode_for(True, None)      # announcement suppressed now
    assert first == second == 3
