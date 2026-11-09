# SPDX-License-Identifier: Apache-2.0
"""``zerocopy_w13`` must reach kt and the placement, not just the config object.

kt reads ``KT_ZEROCOPY_WEIGHTS`` / ``KT_ZEROCOPY_SCOPE`` from the environment, so
these tests assert the wiring is observable: a switch that is accepted and does
nothing is the failure mode that survives every green test.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from vllm_ascend.kt_offload.layer import KT_ZEROCOPY_ENV, apply_zerocopy_env


def _cfg(**kw):
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    base = dict(enabled=True, cpuinfer_threads=8, num_resident_experts=100, overlap_side_stream=False)
    base.update(kw)
    return KTOffloadConfig(**base)


@pytest.fixture(autouse=True)
def _clean_zerocopy_env(monkeypatch):
    """Start every test from "none of the three exported"."""
    for name in KT_ZEROCOPY_ENV:
        monkeypatch.delenv(name, raising=False)
    yield


def test_zerocopy_on_sets_the_three_variables_kt_reads():
    assert apply_zerocopy_env(_cfg(zerocopy_w13=True)) == KT_ZEROCOPY_ENV
    assert os.environ["KT_ZEROCOPY_WEIGHTS"] == "1"
    assert os.environ["KT_ZEROCOPY_SCOPE"] == "w13"
    # Deliberate: placement has exactly one owner. numa_placement migrates and
    # faults these pages in by file offset before kt reads them; kt doing its own
    # pin/migrate on top means neither arm can be attributed.
    assert os.environ["KT_ZEROCOPY_MLOCK"] == "0"


def test_zerocopy_off_sets_nothing_at_all():
    """A leftover exported KT_ZEROCOPY_WEIGHTS would make the *next* run zero-copy."""
    assert apply_zerocopy_env(_cfg(zerocopy_w13=False)) == {}
    for name in KT_ZEROCOPY_ENV:
        assert name not in os.environ


def test_scope_is_w13_and_not_the_full_set():
    """KT_ZEROCOPY_WEIGHTS without a SCOPE does *whole-model* zero-copy, which
    costs kt compute -- the opposite of why it is on."""
    assert KT_ZEROCOPY_ENV["KT_ZEROCOPY_SCOPE"] == "w13"


@pytest.mark.parametrize(
    "name, exported, switch, match",
    [
        ("KT_ZEROCOPY_WEIGHTS", "0", True, "disagrees"),
        # The dangerous direction: kt would zero-copy while numa_placement never ran,
        # and the run would be filed as the copy-path arm.
        ("KT_ZEROCOPY_WEIGHTS", "1", False, "disagrees"),
        ("KT_ZEROCOPY_SCOPE", "all", True, "KT_ZEROCOPY_SCOPE"),
        # MLOCK=1 on top of our 0 is two owners of the placement.
        ("KT_ZEROCOPY_MLOCK", "1", True, "KT_ZEROCOPY_MLOCK"),
    ],
)
def test_exported_variable_disagreeing_with_the_config_raises(monkeypatch, name, exported, switch, match):
    monkeypatch.setenv(name, exported)
    with pytest.raises(RuntimeError, match=match):
        apply_zerocopy_env(_cfg(zerocopy_w13=switch))


def test_an_agreeing_export_is_accepted(monkeypatch):
    """Only a *disagreement* raises; the same values are not a conflict."""
    for name, value in KT_ZEROCOPY_ENV.items():
        monkeypatch.setenv(name, value)
    assert apply_zerocopy_env(_cfg(zerocopy_w13=True)) == KT_ZEROCOPY_ENV


@pytest.mark.parametrize("off", ["", "0"])
def test_off_spellings_agree_with_the_switch_being_off(monkeypatch, off):
    """kt parses "" and "0" as off on both sides, so neither is a conflict."""
    monkeypatch.setenv("KT_ZEROCOPY_WEIGHTS", off)
    assert apply_zerocopy_env(_cfg(zerocopy_w13=False)) == {}


def test_the_conflict_message_names_the_variable_and_both_values(monkeypatch):
    """An operator has to be able to fix it without reading this file."""
    monkeypatch.setenv("KT_ZEROCOPY_SCOPE", "all")
    with pytest.raises(RuntimeError) as exc:
        apply_zerocopy_env(_cfg(zerocopy_w13=True))
    msg = str(exc.value)
    assert "KT_ZEROCOPY_SCOPE" in msg and "'all'" in msg and "'w13'" in msg


def test_build_wrapper_applies_the_env_before_importing_kt(monkeypatch):
    """kt reads these at *load time* inside the extension; setting them after
    ``from kt_kernel import KTMoEWrapper`` is ignored without a word."""
    import inspect

    import vllm_ascend.kt_offload.layer as layer_mod

    src = inspect.getsource(layer_mod.KTLayerOffload._build_wrapper)
    i_env = src.index("apply_zerocopy_env")
    i_import = src.index("from kt_kernel import")
    assert i_env < i_import, "apply_zerocopy_env must run before the kt import"


def test_setup_placement_is_a_no_op_when_the_switch_is_off(monkeypatch):
    """Off must allocate nothing and read no file -- asserted by making every
    expensive call explode rather than by the absence of an exception."""
    from vllm_ascend.kt_offload import wiring

    def boom(*a, **k):
        raise AssertionError("setup_placement touched the checkpoint with the switch off")

    monkeypatch.setattr(wiring, "WeightSource", boom)
    monkeypatch.setattr(wiring, "place_weights", boom)
    monkeypatch.setattr(wiring, "derive_fraction", boom)

    class _Ascend:
        kt_offload_config = _cfg(zerocopy_w13=False)

    assert wiring.setup_placement(object(), ascend_config=_Ascend()) is None


def test_setup_placement_runs_before_load_model_in_the_worker():
    """kt's load_weights runs *inside* model_runner.load_model(): a page placed
    before that read is placed once, after it every page must be migrated again."""
    from pathlib import Path

    worker = (Path(__file__).resolve().parents[3] / "vllm_ascend" / "worker" / "worker.py").read_text()
    assert "setup_placement" in worker, "nothing calls setup_placement"
    i_place = worker.index("setup_placement(self.vllm_config)")
    i_load = worker.index("self.model_runner.load_model()")
    i_stream = worker.index("setup_streaming(self.vllm_config)")
    assert i_place < i_load < i_stream, (
        "setup_placement must come before model_runner.load_model(), and setup_streaming "
        "after it (it needs the layers to exist)"
    )


def test_setup_placement_uses_the_configured_fraction_when_pinned(monkeypatch):
    """`derived` and `configured` are different claims and the log says which."""
    from vllm_ascend.kt_offload import wiring

    seen = {}

    def fake_place(source, layers, experts, **kw):
        seen.update(kw)
        seen["layers"], seen["experts"] = list(layers), list(experts)
        return _FakePlan(), _FakeReport()

    monkeypatch.setattr(wiring, "place_weights", fake_place)
    monkeypatch.setattr(wiring, "WeightSource", lambda *a, **k: _FakeSource())
    monkeypatch.setattr(
        wiring, "derive_fraction", lambda *a, **k: pytest.fail("a pinned fraction must not be re-derived")
    )

    cfg = _cfg(
        zerocopy_w13=True,
        threadpool_count=2,
        numa_nodes=[0, 1],
        zerocopy_node_fraction=[0.3, 0.7],
        zerocopy_streaming_node=1,
    )
    cfg._refuse_without_zerocopy_support = lambda: None

    class _Ascend:
        kt_offload_config = cfg

    plan, _rep = wiring.setup_placement(_FakeVllmConfig(), ascend_config=_Ascend())
    assert seen["fraction"] == [0.3, 0.7]
    assert seen["tp"] == 2 and seen["part_nodes"] == [0, 1]
    assert seen["streaming_node"] == 1
    assert plan.planned_pages > 0


def test_setup_placement_raises_when_the_plan_covers_nothing(monkeypatch):
    """Switch on and the placement reached no bytes: the copy path wearing a
    zero-copy label, and it must not start."""
    from vllm_ascend.kt_offload import wiring

    empty = _FakePlan()
    empty.planned_pages = 0
    monkeypatch.setattr(wiring, "place_weights", lambda *a, **k: (empty, _FakeReport()))
    monkeypatch.setattr(wiring, "WeightSource", lambda *a, **k: _FakeSource())

    cfg = _cfg(zerocopy_w13=True, threadpool_count=2, zerocopy_node_fraction=[0.5, 0.5], zerocopy_streaming_node=0)
    cfg._refuse_without_zerocopy_support = lambda: None

    class _Ascend:
        kt_offload_config = cfg

    with pytest.raises(RuntimeError, match="planned 0 pages"):
        wiring.setup_placement(_FakeVllmConfig(), ascend_config=_Ascend())


@pytest.mark.parametrize("layers, experts", [(None, None), ([1], [3, 17, 383])])
def test_setup_placement_derives_from_the_exact_checkpoint_set(monkeypatch, layers, experts):
    """Exercise the real derivation signature, including the per-NUMA need.

    The default placement covers all 384 experts, including the 100 residents;
    a caller-supplied subset must size the same set that place_weights receives.
    Fractions are independent local shares, not a distribution summing to one.
    """
    from vllm_ascend.kt_offload import numa_placement, wiring

    layer_ids = list(range(2)) if layers is None else layers
    expert_ids = list(range(384)) if experts is None else experts
    projection_bytes = 2304 * 5120 // 2
    need_per_part = len(layer_ids) * len(expert_ids) * 2 * projection_bytes // 2
    source = _SizedFakeSource(projection_bytes)
    seen = {}

    monkeypatch.setattr(wiring, "WeightSource", lambda *a, **k: source)
    # Leave wiring.derive_fraction real: a permissive mock hid the missing
    # need_bytes_per_part argument that prevented the default path from starting.
    monkeypatch.setattr(
        numa_placement,
        "read_node_memory",
        lambda nodes: {
            2: numa_placement.NodeMemory(2, need_per_part // 4, 1024, 1024),
            5: numa_placement.NodeMemory(5, 2 * need_per_part, 1024, 1024),
        },
    )

    def fake_place(src, placed_layers, placed_experts, **kw):
        assert src is source
        seen.update(kw)
        assert list(placed_layers) == layer_ids
        assert list(placed_experts) == expert_ids
        return _FakePlan(), _FakeReport()

    monkeypatch.setattr(wiring, "place_weights", fake_place)
    cfg = _cfg(
        zerocopy_w13=True,
        numa_nodes=[2, 5],
        zerocopy_reserve_gib=0,
        zerocopy_streaming_node=5,
    )
    cfg._refuse_without_zerocopy_support = lambda: None
    wiring.setup_placement(
        _FakeVllmConfig(), ascend_config=SimpleNamespace(kt_offload_config=cfg), layers=layers, experts=experts
    )
    assert seen["fraction"] == [0.25, 1.0]
    assert seen["fraction_source"] == "derived"
    assert seen["tp"] == 2 and seen["part_nodes"] == [2, 5]
    assert source.located == {
        (layer, expert, proj, "weight") for layer in layer_ids for expert in expert_ids for proj in ("w1", "w3")
    }
    assert source.closed


def test_setup_placement_closes_checkpoint_if_derivation_refuses(monkeypatch):
    from vllm_ascend.kt_offload import numa_placement, wiring

    source = _SizedFakeSource(2304 * 5120 // 2)
    monkeypatch.setattr(wiring, "WeightSource", lambda *a, **k: source)
    monkeypatch.setattr(
        numa_placement,
        "read_node_memory",
        lambda nodes: {n: numa_placement.NodeMemory(n, 0, 0, 0) for n in nodes},
    )
    monkeypatch.setattr(wiring, "place_weights", lambda *a, **k: pytest.fail("no memory was granted"))
    cfg = _cfg(zerocopy_w13=True, zerocopy_streaming_node=0)
    cfg._refuse_without_zerocopy_support = lambda: None
    with pytest.raises(numa_placement.PlacementError, match="no node has memory"):
        wiring.setup_placement(_FakeVllmConfig(), ascend_config=SimpleNamespace(kt_offload_config=cfg))
    assert source.closed


class _SizedFakeSource:
    num_experts = 384

    def __init__(self, projection_bytes):
        self.projection_bytes = projection_bytes
        self.located = set()
        self.closed = False

    def locate(self, layer, expert, proj, kind):
        self.located.add((layer, expert, proj, kind))
        return SimpleNamespace(nbytes=self.projection_bytes)

    def close(self):
        self.closed = True


class _FakeSource:
    num_experts = 4

    def close(self):
        pass


class _FakePlan:
    planned_pages = 100

    def planned_pages_by_node(self):
        return {0: 50, 1: 50}


class _FakeReport:
    needed_move = 10
    faulted_in = 10

    def __str__(self):
        return "fake report"


class _FakeVllmConfig:
    class model_config:
        model = "/models/DeepSeek-V4.1-Flash"  # label only; nothing reads this path

        class hf_text_config:
            num_hidden_layers = 2
