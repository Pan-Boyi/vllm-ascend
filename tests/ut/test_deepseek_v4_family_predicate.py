# SPDX-License-Identifier: Apache-2.0
"""Which DeepSeek-V4 tests must widen for V4.1, and which must not.

Six sites compared ``model_type`` for exact equality with ``deepseek_v4``.
V4.1 reports ``deepseek_v41``, so each of those branches simply did not run,
and a branch that does not run says nothing.

Five other sites compare ``model_version``, which looks like the same test and
is not: upstream's V4.1 attention spec sets ``model_version="deepseek_v4"``
itself, so those equalities already match and widening them would break the
match. The two are kept apart here on purpose -- "tidying up the rest of them"
is the obvious next move and it would be wrong.
"""

import pathlib
import re

import pytest

from vllm_ascend.utils import is_deepseek_v4_family

ROOT = pathlib.Path(__file__).resolve().parents[2] / "vllm_ascend"


class _Config:
    def __init__(self, model_type=None):
        if model_type is not None:
            self.model_type = model_type


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v41",
                                        "deepseek_v41_text", "deepseek_v4_text"])
def test_the_family_is_recognised(model_type):
    assert is_deepseek_v4_family(_Config(model_type)) is True


@pytest.mark.parametrize("model_type", ["deepseek_v3", "deepseek_v2",
                                        "qwen3", "kimi_k3", "", None])
def test_other_models_are_not(model_type):
    assert is_deepseek_v4_family(_Config(model_type)) is False


def test_a_config_without_model_type_is_not_the_family():
    """Half-built configs must not be taken for a match."""
    assert is_deepseek_v4_family(_Config()) is False
    assert is_deepseek_v4_family(object()) is False


def _sources():
    for path in ROOT.rglob("*.py"):
        yield path, path.read_text()


def test_no_exact_model_type_comparison_survives():
    """The whole point: V4.1 must not be excluded by an equality test."""
    pattern = re.compile(r'model_type[^=!\n]*[=!]= *"deepseek_v4"')
    offenders = [f"{p}:{i}" for p, text in _sources()
                 for i, line in enumerate(text.splitlines(), 1)
                 if pattern.search(line)]
    assert not offenders, offenders


def test_model_version_comparisons_are_left_alone():
    """These are correct as equalities and must stay that way.

    If this reaches zero, someone widened them too -- and V4.1's specs, which
    declare model_version="deepseek_v4", would stop being matched by anything.
    """
    pattern = re.compile(r'model_version[^=!\n]*== *"deepseek_v4"')
    found = [f"{p}:{i}" for p, text in _sources()
             for i, line in enumerate(text.splitlines(), 1)
             if pattern.search(line)]
    assert len(found) >= 4, found
