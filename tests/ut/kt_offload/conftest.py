# SPDX-License-Identifier: Apache-2.0
"""Start every kt_offload test from "all four switches unset".

Since PLAN §4.33 "unset" is a state of its own (the default: side stream,
P1a and the shared-window move on, P1b off), not a synonym for "0". A value
left in the parent shell -- the on-card runbooks export these -- would
otherwise change what a test that never mentions them is testing.

Also drops any *module attribute* of that name from ``vllm_ascend.envs``:
``monkeypatch.setattr(envs, NAME, ...)`` (used by older tests here) restores
the value ``envs.__getattr__`` returned at the time as a real attribute on
undo, which then shadows the environment for every later test -- with
defaults on, that turns a later ``setenv`` into a silent no-op.
"""

import pytest

_SWITCH_ENVS = (
    "VLLM_ASCEND_DSV41_KT_SIDE_STREAM",
    "VLLM_ASCEND_DSV41_KT_ASYNC_WAIT",
    "VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW",
    "VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL",
)


def _drop_shadows():
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
