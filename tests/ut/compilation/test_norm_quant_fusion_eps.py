"""Fusion patterns must be registered for the eps the model actually uses.

eps is traced into each pattern as a literal, so a pattern matches only a graph whose RMSNorm carries
exactly that value. Registering just 1e-5 and 1e-6 silently skipped fusion on DeepSeek-V4.1, whose
rms_norm_eps is 1e-20 -- nothing raised, the norm and quant ops simply stayed separate.
"""

from types import SimpleNamespace

import pytest

from vllm_ascend.compilation.passes.norm_quant_fusion_pass import (
    COMMON_EPSILONS,
    _epsilons_to_register,
)


def _config(rms_norm_eps):
    return SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(rms_norm_eps=rms_norm_eps)))


def test_registers_the_models_own_eps():
    """The regression: DeepSeek-V4.1's 1e-20 was never registered."""
    assert 1e-20 in _epsilons_to_register(_config(1e-20))


def test_models_own_eps_comes_first():
    assert _epsilons_to_register(_config(1e-20))[0] == 1e-20


def test_keeps_the_common_fallbacks():
    for eps in COMMON_EPSILONS:
        assert eps in _epsilons_to_register(_config(1e-20))


@pytest.mark.parametrize("eps", COMMON_EPSILONS)
def test_no_duplicates_when_the_model_uses_a_common_value(eps):
    """DeepSeek-V4 uses 1e-6, which must not be registered twice."""
    registered = _epsilons_to_register(_config(eps))
    assert registered.count(eps) == 1


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(),
        SimpleNamespace(model_config=None),
        SimpleNamespace(model_config=SimpleNamespace(hf_text_config=None)),
        SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace())),
        _config(None),
        _config("not-a-number"),
    ],
)
def test_falls_back_when_the_eps_cannot_be_read(config):
    assert _epsilons_to_register(config) == list(COMMON_EPSILONS)
