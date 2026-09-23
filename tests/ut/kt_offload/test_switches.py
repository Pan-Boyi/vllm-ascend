# SPDX-License-Identifier: Apache-2.0
"""PLAN §4.33: the side stream, P1a and the shared-window move default ON; P1b OFF.

One resolver (``kt_offload.switches.resolve``) decides every switch; config
validation, the layer runtime and the start-up log all go through it. These
cases pin its table, the start-up refusals that survive the flip, and eager.

CPU only. Environment values are set with ``setenv`` (conftest.py clears all
four before every test, so "unset" really is unset).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload import switches
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan

SIDE = switches.SIDE_STREAM_ENV
P1A = switches.ASYNC_WAIT_ENV
P1B = switches.DEVICE_SIGNAL_ENV
SIW = switches.SHARED_IN_WINDOW_ENV


def _cfg(**kw):
    return KTOffloadConfig(enabled=True, cpuinfer_threads=8, num_resident_experts=100, **kw)


def _vcfg(eager: bool = False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1, enable_eplb=False
        ),
        model_config=SimpleNamespace(enforce_eager=eager),
        speculative_config=None,
    )


def _state(sw):
    return (sw.side_stream.on, sw.async_wait.on, sw.shared_in_window.on, sw.device_signal.on)


# ---------------------------------------------------------------------------
# 1. the table
# ---------------------------------------------------------------------------


def test_all_unset_is_side_on_p1a_on_shared_on_p1b_off():
    cfg = _cfg()
    cfg._validate_preconditions(_vcfg())  # starts
    sw = switches.resolve(cfg)
    assert _state(sw) == (True, True, True, False)
    assert (sw.side_stream.source, sw.async_wait.source, sw.shared_in_window.source, sw.device_signal.source) == (
        "default",
        "auto",
        "auto",
        "default",
    )


def test_the_envs_read_unset_as_empty_not_zero():
    from vllm_ascend import envs

    for name in (SIDE, P1A, P1B, SIW):
        assert getattr(envs, name) == "", name


def test_cfg_overlap_false_and_all_unset_turns_all_three_off_and_starts():
    cfg = _cfg(overlap_side_stream=False)
    cfg._validate_preconditions(_vcfg())  # no raise
    sw = switches.resolve(cfg)
    assert _state(sw) == (False, False, False, False)
    assert sw.side_stream.source == "cfg" and sw.async_wait.source == "auto" and sw.shared_in_window.source == "auto"


def test_packed_alone_keeps_p1a_on_but_not_the_shared_move():
    sw = switches.resolve(_cfg(overlap_side_stream=False, packed_transfer=True))
    assert _state(sw) == (False, True, False, False)
    assert "packed_transfer" in sw.async_wait.why


def test_side_zero_is_the_kill_switch_and_the_autos_follow(monkeypatch):
    monkeypatch.setenv(SIDE, "0")
    cfg = _cfg(overlap_side_stream=True)
    cfg._validate_preconditions(_vcfg())
    assert _state(switches.resolve(cfg)) == (False, False, False, False)


def test_side_one_does_not_override_a_false_cfg(monkeypatch):
    monkeypatch.setenv(SIDE, "1")
    sw = switches.resolve(_cfg(overlap_side_stream=False))
    assert sw.side_stream.on is False and sw.side_stream.source == "cfg"
    assert "does not override" in sw.side_stream.why


def test_p1a_zero_turns_only_p1a_off(monkeypatch):
    monkeypatch.setenv(P1A, "0")
    sw = switches.resolve(_cfg())
    assert _state(sw) == (True, False, True, False) and sw.async_wait.source == "env"


def test_shared_zero_turns_only_the_shared_move_off(monkeypatch):
    monkeypatch.setenv(SIW, "0")
    sw = switches.resolve(_cfg())
    assert _state(sw) == (True, True, False, False) and sw.shared_in_window.source == "env"


def test_all_explicit_zero_is_the_serial_arm(monkeypatch):
    """What run_arm.sh's A arm passes: every switch off, and it starts."""
    for name in (SIDE, P1A, P1B, SIW):
        monkeypatch.setenv(name, "0")
    for cfg in (_cfg(), _cfg(overlap_side_stream=False)):
        cfg._validate_preconditions(_vcfg())
        assert _state(switches.resolve(cfg)) == (False, False, False, False)


def test_the_explicit_accepted_arm_resolves_like_the_default(monkeypatch):
    """The D arm of the second acceptance (cfg overlap true, SIDE=1 P1A=1
    P1B=0 SIW=1) and "nothing set" must be the same configuration -- that is
    what makes the on-card smoke comparable with D4 bit for bit."""
    default = _state(switches.resolve(_cfg()))
    for name, value in ((SIDE, "1"), (P1A, "1"), (P1B, "0"), (SIW, "1")):
        monkeypatch.setenv(name, value)
    assert _state(switches.resolve(_cfg(overlap_side_stream=True, packed_transfer=False))) == default


# ---------------------------------------------------------------------------
# 2. refusals that survive the flip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", [P1A, P1B])
def test_explicit_one_without_own_buffers_still_refuses(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match=f"{name}=1 needs kt_offload_config.packed_transfer or the side stream"):
        _cfg(overlap_side_stream=False)._validate_preconditions(_vcfg())


def test_explicit_p1a_with_the_side_stream_killed_by_env_refuses(monkeypatch):
    monkeypatch.setenv(SIDE, "0")
    monkeypatch.setenv(P1A, "1")
    with pytest.raises(ValueError, match=f"side stream off\\(env: {SIDE}=0\\)"):
        _cfg()._validate_preconditions(_vcfg())


def test_p1b_on_with_p1a_unset_makes_p1a_step_aside(monkeypatch):
    monkeypatch.setenv(P1B, "1")
    cfg = _cfg()
    cfg._validate_preconditions(_vcfg())  # no raise: P1a was never asked for
    sw = switches.resolve(cfg)
    assert _state(sw) == (True, False, True, True)
    assert sw.async_wait.source == "auto" and P1B in sw.async_wait.why


def test_p1b_on_with_p1a_explicit_zero(monkeypatch):
    monkeypatch.setenv(P1B, "1")
    monkeypatch.setenv(P1A, "0")
    _cfg()._validate_preconditions(_vcfg())
    assert _state(switches.resolve(_cfg())) == (True, False, True, True)


def test_both_explicit_one_refuses_and_says_how_to_fix_it(monkeypatch):
    monkeypatch.setenv(P1B, "1")
    monkeypatch.setenv(P1A, "1")
    with pytest.raises(ValueError) as e:
        _cfg()._validate_preconditions(_vcfg())
    msg = str(e.value)
    assert f"{P1B}=1 and {P1A}=1 are both set explicitly" in msg
    assert f"unset {P1A}" in msg and f"unset {P1B}" in msg


def test_the_layer_resolver_never_raises_on_explicit_values(monkeypatch):
    """The on-card harnesses (t4/t5) build layers without config validation
    and set both to 1 to flip per arm; validate=False honours both."""
    monkeypatch.setenv(P1B, "1")
    monkeypatch.setenv(P1A, "1")
    sw = switches.resolve(_cfg(overlap_side_stream=False), validate=False)
    assert sw.async_wait.on and sw.device_signal.on


@pytest.mark.parametrize("value", ["true", "on", "01", " 1", "no"])
@pytest.mark.parametrize("name", [SIDE, P1A, P1B, SIW])
def test_an_unreadable_value_refuses(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        _cfg()._validate_preconditions(_vcfg())


def test_shared_one_without_the_side_stream_warns_but_starts(monkeypatch):
    from vllm.logger import logger

    said = []
    monkeypatch.setattr(logger, "warning", lambda msg, *a: said.append(msg % a if a else msg))
    monkeypatch.setenv(SIW, "1")
    _cfg(overlap_side_stream=False)._validate_preconditions(_vcfg())
    assert any(f"{SIW}=1 without the side stream" in m for m in said), said
    # unset follows the side stream, so it never lands there
    said.clear()
    monkeypatch.delenv(SIW)
    _cfg(overlap_side_stream=False)._validate_preconditions(_vcfg())
    assert not [m for m in said if SIW in m], said


# ---------------------------------------------------------------------------
# 3. the start-up line
# ---------------------------------------------------------------------------


def _startup_line(monkeypatch, cfg, eager=False):
    from vllm.logger import logger

    said = []
    monkeypatch.setattr(logger, "info", lambda msg, *a: said.append(msg % a if a else msg))
    cfg._validate_preconditions(_vcfg(eager))
    lines = [m for m in said if m.startswith("kt_offload switches")]
    assert len(lines) == 1, said
    return lines[0]


def test_startup_logs_every_effective_value_with_its_source(monkeypatch):
    line = _startup_line(monkeypatch, _cfg())
    assert "side_stream=on(default)" in line
    assert "async_wait=on(auto: side stream on)" in line
    assert "shared_in_window=on(auto: follows the side stream)" in line
    assert "device_signal=off(default)" in line
    assert "enforce_eager" not in line


def test_startup_line_names_env_and_cfg_sources(monkeypatch):
    monkeypatch.setenv(P1A, "0")
    line = _startup_line(monkeypatch, _cfg(overlap_side_stream=False))
    assert "side_stream=off(cfg: overlap_side_stream=false)" in line
    assert f"async_wait=off(env: {P1A}=0)" in line
    assert "shared_in_window=off(auto" in line


def test_startup_line_says_eager_runs_the_original_path(monkeypatch):
    assert "enforce_eager" in _startup_line(monkeypatch, _cfg(), eager=True)


def test_nothing_is_logged_when_offload_is_off(monkeypatch):
    from vllm.logger import logger

    said = []
    monkeypatch.setattr(logger, "info", lambda msg, *a: said.append(msg))
    KTOffloadConfig()._validate_preconditions(_vcfg())
    assert said == []


# ---------------------------------------------------------------------------
# 4. the layer runtime: same resolver, and eager keeps its original path
# ---------------------------------------------------------------------------

H, TOPK = 8, 2


def _layer_vcfg(graph: bool):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=TOPK),
            model="/m",
            enforce_eager=not graph,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY" if graph else "NONE"),
            cudagraph_capture_sizes=[1] if graph else [],
        ),
    )


def _layer(cfg, graph: bool):
    plan = build_plan([[0, 3]], 16)
    return kl.KTLayerOffload(cfg, plan.layer(0), 0, _layer_vcfg(graph))


def test_graph_layer_defaults_match_the_resolver():
    rt = _layer(KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1), graph=True)
    assert rt.graph_enabled
    assert (rt.overlap, rt.async_wait, rt.shared_in_window, rt.device_signal) == (True, True, True, False)
    assert rt.switches == switches.resolve(rt.cfg)


def test_eager_layer_turns_the_automatic_values_off():
    rt = _layer(KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1), graph=False)
    assert not rt.graph_enabled
    # the side stream keeps the config's value (eager never forks whatever it
    # says); the automatic P1a and shared-window move have nothing to act on
    assert (rt.overlap, rt.async_wait, rt.shared_in_window, rt.device_signal) == (True, False, False, False)


def test_eager_layer_keeps_explicit_ones_as_before_4_33(monkeypatch):
    monkeypatch.setenv(P1A, "1")
    monkeypatch.setenv(SIW, "1")
    rt = _layer(KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1), graph=False)
    assert rt.async_wait and rt.shared_in_window  # P1a then drops itself in _enable_async_wait


class _FakeKT:
    def __init__(self):
        self.calls = []

    def submit_forward(self, hs, ids, w, stream):
        self.calls.append("submit_forward")

    def sync_forward(self, hs, stream):
        self.calls.append("sync_forward")
        return torch.full_like(hs, 0.5)


def _eager_run(monkeypatch, cfg):
    monkeypatch.setattr(kl, "_stream_handle", lambda device: 0)

    def no_side_stream(device):
        raise AssertionError("eager created a side stream")

    monkeypatch.setattr(kl, "_side_stream", no_side_stream)
    said = []
    monkeypatch.setattr(kl.logger, "warning", lambda msg, *a: said.append(msg % a if a else msg))
    rt = _layer(cfg, graph=False)
    fake = _FakeKT()
    monkeypatch.setattr(rt, "_build_wrapper", lambda: fake)
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(rt.num_slots, 3), requires_grad=False)
    rt.after_weights_loaded(layer)
    g = torch.Generator().manual_seed(0)
    hs = torch.randn(1, H, generator=g).to(torch.bfloat16)
    ids = torch.tensor([[0, 9]], dtype=torch.int32)
    w = torch.rand(1, TOPK, generator=g)
    rt.route(hs, w, ids, in_profile_run=False)
    out = rt.merge(torch.zeros(1, H, dtype=torch.bfloat16))
    return rt, fake, out, said


def test_eager_with_every_default_runs_the_serial_path_bit_for_bit(monkeypatch):
    """The §4.33 flip must not touch eager: same kt calls, same output, no
    side stream, and none of the per-layer ENABLED lines."""
    default_cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1)
    serial_cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, overlap_side_stream=False)
    rt_d, fake_d, out_d, said_d = _eager_run(monkeypatch, default_cfg)
    rt_s, fake_s, out_s, _ = _eager_run(monkeypatch, serial_cfg)
    assert fake_d.calls == fake_s.calls == ["submit_forward", "sync_forward"]
    assert torch.equal(out_d, out_s)
    assert rt_d._ext_events == {} and rt_d._pending_join is None
    assert not [m for m in said_d if "ENABLED" in m or "SHARED IN WINDOW" in m], said_d
