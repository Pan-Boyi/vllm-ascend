# SPDX-License-Identifier: Apache-2.0
"""``VLLM_ASCEND_DSV41_EXPERT_COUNT_EVERY`` -- how often the counter dumps. CPU only.

Counts are cumulative and one process writes one file, so per-request numbers
come from diffing two dumps; that only holds if the period really is what the
environment says and the period in force is written into the dump.
"""

from __future__ import annotations

import pathlib
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.kt_offload import expert_counter as ec

COUNTER_SRC = pathlib.Path(ec.__file__)
ENV = "VLLM_ASCEND_DSV41_EXPERT_COUNT_EVERY"


def _install_npu(monkeypatch):
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False
    )


def _arm(mod, monkeypatch, tmp_path, every):
    """The real record()/dump() path, with only the runtime context faked."""
    _install_npu(monkeypatch)
    monkeypatch.setattr(mod, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "ENABLED", True)
    monkeypatch.setattr(mod, "DUMP_EVERY", every)
    monkeypatch.setattr(mod, "_S", None)
    monkeypatch.setattr(mod, "_CONFIG_ENFORCE_EAGER", None)
    monkeypatch.setattr(mod, "_batch_split", lambda: (0, 4))
    monkeypatch.setattr(mod, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(mod, "_rows_are_dp_gathered", lambda: (False, None))
    monkeypatch.setattr(mod.atexit, "register", lambda fn: None)
    return mod


def _layer(i):
    return SimpleNamespace(layer_name=f"model.layers.{i}.ffn.experts")


def _ids(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.argsort(torch.rand(rows, 384, generator=g), dim=1)[:, :6].to(torch.int32)


def _forward(mod, seed, layers=(0,)):
    """One forward: record() on each of ``layers``. Layer 0 is the denominator."""
    for i in layers:
        mod.record(_layer(i), _ids(4, seed=seed), False)


def _dumps(tmp_path):
    return sorted(tmp_path.glob("rank*_pid*.pt"))


def test_default_is_still_200(monkeypatch):
    """The knob must not have moved the behaviour of anyone who does not set it."""
    monkeypatch.delenv(ENV, raising=False)
    assert envs.VLLM_ASCEND_DSV41_EXPERT_COUNT_EVERY == "200"
    assert ec._dump_every() == 200


@pytest.mark.parametrize("raw,want", [("1", 1), ("7", 7), ("25", 25), ("200", 200), (" 3 ", 3)])
def test_a_valid_value_takes_effect(monkeypatch, raw, want):
    monkeypatch.setenv(ENV, raw)
    assert ec._dump_every() == want


@pytest.mark.parametrize("raw", ["0", "-3", "abc", "", "1.5", "1e3"])
def test_an_invalid_value_keeps_200_and_says_so(monkeypatch, raw):
    """Fail loud, not raise: this module is imported whether or not the counter
    was asked for. The warning is read off the module's own ``logger``, which does
    not propagate to pytest's root handler, so caplog would see nothing.
    """
    said = []
    monkeypatch.setattr(ec.logger, "warning", lambda msg, *a: said.append(msg % a if a else msg))
    monkeypatch.setenv(ENV, raw)
    assert ec._dump_every() == 200
    assert any(ENV in m for m in said), (
        "an ignored value must be reported -- silently keeping 200 makes a run "
        f"that nobody can tell apart from one that got the period it asked for; said={said}"
    )


@pytest.mark.parametrize("every", [1, 2, 5])
def test_dumps_land_exactly_every_n_counted_forwards(monkeypatch, tmp_path, every):
    mod = _arm(ec, monkeypatch, tmp_path, every)
    for n in range(1, 3 * every + 1):
        _forward(mod, seed=n)
        state = torch.load(_dumps(tmp_path)[0], weights_only=False) if _dumps(tmp_path) else None
        landed = None if state is None else state["steps"]
        # after n forwards the newest dump is the last multiple of `every` <= n
        want = (n // every) * every
        assert landed == (want or None), f"every={every} n={n}: newest dump holds {landed}, want {want}"


def test_the_dump_records_the_period_in_force(monkeypatch, tmp_path):
    """A slice must not be readable at the wrong granularity by a later reader."""
    mod = _arm(ec, monkeypatch, tmp_path, 1)
    _forward(mod, seed=1)
    assert torch.load(_dumps(tmp_path)[0], weights_only=False)["dump_every"] == 1


def test_two_consecutive_dumps_differ_by_exactly_one_forward(monkeypatch, tmp_path):
    """At every=1 the difference between two consecutive dumps is one forward's
    counts and nothing else -- what makes a per-request slice a slice."""
    mod = _arm(ec, monkeypatch, tmp_path, 1)
    _forward(mod, seed=1)
    before = torch.load(_dumps(tmp_path)[0], weights_only=False)
    a = before["counts"]["prefill"][0].clone()
    _forward(mod, seed=2)
    after = torch.load(_dumps(tmp_path)[0], weights_only=False)
    delta = after["counts"]["prefill"][0] - a
    assert torch.equal(delta, torch.bincount(_ids(4, seed=2).reshape(-1).long(), minlength=384))
    assert after["steps"] - before["steps"] == 1
    assert int(delta.sum()) == (after["tokens"]["prefill"][0] - before["tokens"]["prefill"][0]) * after["topk"]


def _mutant(tag, old, new):
    src = COUNTER_SRC.read_text()
    assert src.count(old) == 1, f"mutant {tag}: anchor not found exactly once: {old!r}"
    mod = types.ModuleType(f"vllm_ascend.kt_offload.expert_counter__mutant_{tag}")
    mod.__file__ = str(COUNTER_SRC)
    exec(compile(src.replace(old, new), f"{COUNTER_SRC} [mutant {tag}]", "exec"), mod.__dict__)
    return mod


def test_a_build_that_ignores_the_env_is_caught(monkeypatch, tmp_path):
    """Positive control: a counter never wired to the environment must turn the
    checks above red."""
    mut = _mutant("hardcoded_200", "\n    return n\n\n\nDUMP_EVERY", "\n    return 200\n\n\nDUMP_EVERY")
    monkeypatch.setenv(ENV, "1")
    assert mut._dump_every() == 200, "the mutant is not mutated -- this control proves nothing"
    assert ec._dump_every() == 1, "the real module must disagree with the mutant"

    # the mutant's period is still 200, so every=1 gets no dump at forward 1
    mut = _arm(mut, monkeypatch, tmp_path, mut._dump_every())
    _forward(mut, seed=1)
    assert not _dumps(tmp_path), "mutant dumped at forward 1 -- the arm is not testing the period"
