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

def test_the_mm_ranges_gate_announces_when_it_excludes(lines):
    from vllm_ascend.attention.dsa_v1 import mm_ranges_have_spans

    assert mm_ranges_have_spans({0: []}) is False
    assert any("takes no vision index path" in line for line in lines)


def test_the_mm_ranges_gate_stays_quiet_when_it_does_not(lines):
    """Two arms. Announcing on the path that was going to happen anyway says
    nothing about whether the gate changed something."""
    from vllm_ascend.attention.dsa_v1 import mm_ranges_have_spans

    assert mm_ranges_have_spans({0: [(1, 2)]}) is True
    assert not any("vision index path" in line for line in lines)


def test_the_announcement_changes_no_behaviour():
    """Evidence, not a guard: suppressing the line must not change the
    answer."""
    from vllm_ascend.attention.dsa_v1 import mm_ranges_have_spans

    reset_for_test()
    first = mm_ranges_have_spans({0: []})
    second = mm_ranges_have_spans({0: []})     # announcement suppressed now
    assert first is second is False
