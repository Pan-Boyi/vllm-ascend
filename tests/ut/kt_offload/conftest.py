"""Per-test isolation for the two pieces of process-global state in kt_offload:
the four switch env vars ("unset" is a state of its own, not a synonym for "0")
and the prefill window's module state.
"""

from __future__ import annotations

import pytest

from vllm_ascend.kt_offload import prefill_window

_SWITCH_ENVS = (
    "VLLM_ASCEND_DSV41_KT_SIDE_STREAM",
    "VLLM_ASCEND_DSV41_KT_ASYNC_WAIT",
    "VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW",
    "VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL",
)


def _drop_shadows():
    # monkeypatch.setattr(envs, NAME, ...) restores the value as a real module
    # attribute on undo, which then shadows the environment for later tests.
    from vllm_ascend import envs

    for name in _SWITCH_ENVS:
        vars(envs).pop(name, None)


@pytest.fixture(autouse=True)
def _kt_switches_unset(monkeypatch):
    for name in _SWITCH_ENVS:
        monkeypatch.delenv(name, raising=False)
    _drop_shadows()
    yield
    _drop_shadows()


@pytest.fixture(autouse=True)
def _fresh_prefill_window():
    prefill_window.reset()
    yield
    prefill_window.reset()
