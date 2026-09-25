# SPDX-License-Identifier: Apache-2.0
"""Per-layer host-func sampling.

Only a host-func node inside the graph increments the counters, so a layer
whose count does not grow while the service decodes has no CPU-expert work in
the graph.
"""

from __future__ import annotations

import threading

import pytest

from vllm_ascend.kt_offload import hostfunc_log as hf


def test_uniform_growth_is_the_healthy_reading():
    before = {i: 10 for i in range(40)}
    after = {i: 13 for i in range(40)}
    r = hf.host_func_report(before, after, 40)
    assert r["uniform"] and not r["partial"] and r["min"] == r["max"] == 3 and r["total"] == 120
    assert "uniform" in hf.format_report(r, 5.0)


def test_layers_that_never_ran_are_named():
    before = {i: 0 for i in range(40)}
    after = {i: (5 if i < 10 else 0) for i in range(40)}
    r = hf.host_func_report(before, after, 40)
    assert r["partial"] and len(r["zero_layers"]) == 30 and r["max"] == 5
    msg = hf.format_report(r, 5.0)
    assert "30 layers never ran" in msg and "NOT in the graph" in msg


def test_a_missing_key_reads_as_zero_not_as_absent():
    """An absent layer and a zero layer look the same in a log; only one is the finding."""
    r = hf.host_func_report({}, {0: 4}, 3)
    assert r["per_layer"] == {0: 4, 1: 0, 2: 0} and r["partial"]


def test_all_quiet_is_not_reported_as_partial():
    r = hf.host_func_report({i: 1 for i in range(4)}, {i: 1 for i in range(4)}, 4)
    assert not r["partial"] and r["total"] == 0  # nothing decoding: no claim either way


def test_sampler_is_a_daemon_and_stops_with_its_process():
    counters = {0: 0}
    try:
        t = hf.start(0.01, counters, 1)
        assert t is not None and t.daemon, "a watcher that outlives its process samples someone else's service"
        assert hf.start(0.01, counters, 1) is t  # idempotent: one sampler per process
        counters[0] = 5
        for _ in range(50):
            if not t.is_alive():
                break
            threading.Event().wait(0.01)
    finally:
        hf.stop()
    assert not t.is_alive()
    hf.stop()  # safe twice


def test_off_by_default(monkeypatch):
    from vllm_ascend import envs

    monkeypatch.delenv("VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG", raising=False)
    assert envs.VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG == ""
    assert hf.start(0.0, {}, 40) is None


def test_layer_starts_it_only_when_asked(monkeypatch):
    from vllm_ascend.kt_offload import layer as kl

    started = []
    monkeypatch.setattr(hf, "start", lambda interval, counters, layers: started.append((interval, layers)))
    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG", "")
    kl._maybe_start_host_func_log(40)
    assert started == []
    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG", "2.5")
    kl._maybe_start_host_func_log(40)
    assert started == [(2.5, 40)]
    monkeypatch.setenv("VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG", "abc")
    kl._maybe_start_host_func_log(40)  # a bad value warns, it does not raise
    assert len(started) == 1


@pytest.mark.parametrize("interval", [0.0, -1.0])
def test_non_positive_interval_never_starts(interval):
    assert hf.start(interval, {}, 4) is None
