# SPDX-License-Identifier: Apache-2.0
"""kt offload: config, the feature-off path, and the layer runtime with a fake kt.

CPU only. The kt wrapper is replaced by a recorder; nothing here imports
kt_kernel or touches an NPU.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan, check_split, frequency_resident_sets

ROUTED_EXPERTS = pathlib.Path(__file__).resolve().parents[3] / "vllm_ascend/ops/fused_moe/routed_experts.py"


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_defaults_off():
    cfg = KTOffloadConfig()
    assert cfg.enabled is False
    assert kl.offload_enabled(SimpleNamespace(kt_offload_config=cfg)) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_resident_experts=100),  # cpuinfer_threads missing
        dict(cpuinfer_threads=8),  # prefix without N
        dict(cpuinfer_threads=8, resident_strategy="frequency", num_resident_experts=100),  # no table
        dict(cpuinfer_threads=8, resident_strategy="explicit"),  # no list
        dict(cpuinfer_threads=8, num_resident_experts=100, threadpool_count=2, numa_nodes=[0]),
        dict(cpuinfer_threads=8, num_resident_experts=100, resident_strategy="random"),
        dict(cpuinfer_threads=8, num_resident_experts=100, overlap_side_stream=True),  # reserved for M3
    ],
)
def test_config_rejects_incomplete(kwargs):
    with pytest.raises(ValueError):
        KTOffloadConfig(enabled=True, **kwargs)


def test_config_rejects_unknown_key():
    with pytest.raises(ValueError):
        KTOffloadConfig(enabled=False, num_gpu_experts=3)


def _vllm_config(tp=1, dp=1, pp=1, eager=True, eplb=False, spec=None):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp, data_parallel_size=dp, pipeline_parallel_size=pp, enable_eplb=eplb
        ),
        model_config=SimpleNamespace(enforce_eager=eager),
        speculative_config=spec,
    )


def test_preconditions():
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=8, num_resident_experts=100)
    cfg._validate_preconditions(_vllm_config())
    cfg._validate_preconditions(_vllm_config(eager=False))  # graph mode allowed since M2
    for bad in (dict(dp=4), dict(tp=2), dict(eplb=True), dict(spec=object())):
        with pytest.raises(ValueError):
            cfg._validate_preconditions(_vllm_config(**bad))
    KTOffloadConfig()._validate_preconditions(_vllm_config(dp=4, eager=False))  # off: no checks


def test_ascend_config_accepts_section():
    from vllm_ascend.ascend_config import AscendConfig

    assert "kt_offload_config" in AscendConfig.__dataclass_fields__


# ---------------------------------------------------------------------------
# feature off: nothing is created, nothing is mutated
# ---------------------------------------------------------------------------


def _fake_init(self, layer_name, params_dtype, moe_config, quant_config, expert_map_manager, **kw):
    pass


@pytest.mark.parametrize(
    "ascend_config",
    [
        SimpleNamespace(kt_offload_config=KTOffloadConfig()),
        SimpleNamespace(),  # section missing
        MagicMock(),  # tests elsewhere mock the whole ascend config
    ],
    ids=["default", "missing", "magicmock"],
)
def test_off_path_creates_nothing(ascend_config):
    moe_config = SimpleNamespace(num_local_experts=384)
    args = ("model.layers.3.ffn.experts", None, moe_config, None, None)
    out = kl.create_layer_offload(ascend_config, _fake_init, args, {})
    assert out is None
    assert moe_config.num_local_experts == 384


def test_off_path_mutation_is_red():
    # Mutant: truthiness instead of `is True` -- a MagicMock config enables offload.
    def enabled_mutant(ascend_config):
        cfg = getattr(ascend_config, "kt_offload_config", None)
        return bool(cfg is not None and getattr(cfg, "enabled", False))

    assert kl.offload_enabled(MagicMock()) is False
    assert enabled_mutant(MagicMock()) is True


def _guarded_offload_calls(tree: ast.AST) -> tuple[list[int], list[int]]:
    """Line numbers of kt_offload method calls, split into guarded / unguarded.

    Guarded = inside an ``if <x>.kt_offload is not None`` / ``if kt_offload is not None``.
    """
    guarded, unguarded = [], []

    def is_guard(test):
        return (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.IsNot)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
            and "kt_offload" in ast.unparse(test.left)
        )

    def visit(node, inside):
        if isinstance(node, ast.If) and is_guard(node.test):
            for child in node.body:
                visit(child, True)
            for child in node.orelse:
                visit(child, inside)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            target = ast.unparse(node.func.value)
            if target.endswith("kt_offload") or target.endswith("kt_offload.placement"):
                (guarded if inside else unguarded).append(node.lineno)
        for child in ast.iter_child_nodes(node):
            visit(child, inside)

    visit(tree, False)
    return guarded, unguarded


def test_every_hook_in_routed_experts_is_guarded():
    guarded, unguarded = _guarded_offload_calls(ast.parse(ROUTED_EXPERTS.read_text()))
    assert unguarded == [], f"unguarded kt_offload calls at lines {unguarded}"
    # coverage: attach_after_init, slot_of, route, merge (+ num_slots is an attribute)
    assert len(guarded) == 4, guarded
    src = ROUTED_EXPERTS.read_text()
    assert src.count("create_layer_offload(") == 1
    assert "kt_offload = None" in src  # class default for __new__-built instances


def test_guard_check_mutation_is_red():
    src = "def f(self):\n    kt_offload = self.kt_offload\n    kt_offload.merge(1)\n"
    guarded, unguarded = _guarded_offload_calls(ast.parse(src))
    assert unguarded == [3] and guarded == []


# ---------------------------------------------------------------------------
# layer runtime with a fake kt wrapper
# ---------------------------------------------------------------------------


class FakeKT:
    def __init__(self, cpu_value=0.25):
        self.submits = []
        self.syncs = 0
        self.cpu_value = cpu_value

    def submit_forward(self, hs, ids, w, stream):
        self.submits.append((hs, ids, w))

    def sync_forward(self, hs, stream):
        self.syncs += 1
        return torch.full_like(hs, self.cpu_value)


def _runtime(monkeypatch, resident_sets, num_experts=16, fake=None):
    monkeypatch.setattr(kl, "_stream_handle", lambda device: 0)
    plan = build_plan(resident_sets, num_experts)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1)
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=8, moe_intermediate_size=4, num_experts_per_tok=2),
            model="/m",
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
    )
    rt = kl.KTLayerOffload(cfg, plan.layer(0), 0, vcfg)
    monkeypatch.setattr(rt, "_build_wrapper", lambda: fake or FakeKT())
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3), requires_grad=False)
    rt.after_weights_loaded(layer)
    return rt, layer


def test_route_submits_original_and_returns_split(monkeypatch):
    rt, _ = _runtime(monkeypatch, [[2, 5, 11]])
    hs = torch.randn(4, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[2, 3], [5, 11], [0, 1], [11, 15]], dtype=torch.int32)
    w = torch.rand(4, 2) + 0.1
    nw, nid = rt.route(hs, w, ids, in_profile_run=False)
    ((s_hs, s_ids, s_w),) = rt.wrapper.submits
    assert s_ids is ids and s_w is w and s_hs.data_ptr() == hs.data_ptr()  # original routing to kt
    assert check_split(rt.placement, ids, w, nid, nw) == []
    out = rt.merge(torch.zeros(4, 8, dtype=torch.bfloat16))
    assert rt.wrapper.syncs == 1 and out.dtype == torch.bfloat16
    assert torch.all(out == 0.25)
    assert rt.merge(out) is out  # nothing pending: merge is a no-op


def test_profile_run_skips_cpu(monkeypatch):
    rt, _ = _runtime(monkeypatch, [[0, 1]])
    hs = torch.randn(3, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 9], [4, 1], [7, 8]], dtype=torch.int32)
    w = torch.rand(3, 2)
    nw, nid = rt.route(hs, w, ids, in_profile_run=True)
    assert rt.wrapper.submits == []
    assert check_split(rt.placement, ids, w, nid, nw) == []  # NPU half unchanged in shape/legality
    routed = torch.randn(3, 8, dtype=torch.bfloat16)
    assert rt.merge(routed) is routed


def test_route_rejects_non_bf16(monkeypatch):
    rt, _ = _runtime(monkeypatch, [[0]])
    with pytest.raises(TypeError):
        rt.route(torch.randn(2, 8), torch.rand(2, 2), torch.zeros(2, 2, dtype=torch.int32), in_profile_run=False)


def test_all_resident_layer_needs_no_kt(monkeypatch):
    built = []
    rt, _ = _runtime(monkeypatch, [list(range(16))])
    # _runtime monkeypatched _build_wrapper; all-resident must not call it
    assert rt.wrapper is None and built == []
    hs = torch.randn(2, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    w = torch.rand(2, 2)
    nw, nid = rt.route(hs, w, ids, in_profile_run=False)
    assert torch.equal(nid, ids) and torch.equal(nw, w)


def test_zero_unused_slots(monkeypatch):
    rt, layer = _runtime(monkeypatch, [[0, 1]] + [], num_experts=16)
    # single layer plan has no unused slots; build a two-layer plan instead
    plan = build_plan([[0, 1], [3, 4, 5, 6]], 16)
    rt.placement = plan.layer(0)
    m = torch.nn.Module()
    m.w13_weight = torch.nn.Parameter(torch.full((4, 3), 7.0), requires_grad=False)
    m.bias_like = torch.nn.Parameter(torch.full((5,), 7.0), requires_grad=False)
    rt.zero_unused_slots(m)
    assert torch.all(m.w13_weight[:2] == 7) and torch.all(m.w13_weight[2:] == 0)
    assert torch.all(m.bias_like == 7)


def test_layer_index_from_name():
    assert kl._layer_index("model.layers.12.ffn.experts") == 12
    assert kl._layer_index("language_model.model.layers.0.ffn.experts") == 0
    with pytest.raises(ValueError):
        kl._layer_index("mtp.0.ffn.experts")


def test_bind_by_name_matches_upstream_signature():
    named = kl.bind_routed_experts_args(_fake_init, ("n", torch.bfloat16, "cfg", None, "emm"), {"x": 1})
    assert named["layer_name"] == "n" and named["moe_config"] == "cfg"


def test_frequency_plan_through_runtime(monkeypatch):
    g = torch.Generator().manual_seed(3)
    freq = torch.rand(2, 16, generator=g)
    sets = frequency_resident_sets(freq, 5)
    rt, _ = _runtime(monkeypatch, sets)
    hs = torch.randn(9, 8, dtype=torch.bfloat16)
    ids = torch.argsort(torch.rand(9, 16, generator=g), dim=1)[:, :2].to(torch.int64)
    w = torch.rand(9, 2, generator=g)
    nw, nid = rt.route(hs, w, ids, in_profile_run=False)
    assert check_split(rt.placement, ids, w, nid, nw) == []


def test_attach_wraps_post_load_and_requires_swiglu_limit(monkeypatch):
    rt, _ = _runtime(monkeypatch, [[0, 3], [1, 2, 4]])
    calls = []

    class QM:
        def process_weights_after_loading(self, layer):
            calls.append(("orig", float(layer.w13_weight[-1].sum())))

    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.full((3, 2), 5.0), requires_grad=False)
    layer.quant_method = QM()
    layer.log2phy = None
    layer.moe_config = SimpleNamespace(swiglu_limit=None)
    with pytest.raises(ValueError):
        rt.attach_after_init(layer)
    layer.moe_config = SimpleNamespace(swiglu_limit=10.0)
    rt.attach_after_init(layer)
    assert rt.swiglu_limit == 10.0 and layer.local_num_experts == 3
    layer.quant_method.process_weights_after_loading(layer)
    # the unused tail slot (layer 0 has 2 residents, 3 slots) was zeroed before the original ran
    assert calls == [("orig", 0.0)]
    assert rt.wrapper is not None and set(rt._slot_tables) == {torch.int32, torch.int64}
    layer.log2phy = torch.arange(5)
    with pytest.raises(ValueError):
        rt.attach_after_init(layer)


def test_route_rejects_swapped_ids_and_weights(monkeypatch):
    rt, _ = _runtime(monkeypatch, [[0, 1]])
    hs = torch.randn(2, 8, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 9], [4, 1]], dtype=torch.int32)
    w = torch.rand(2, 2)
    with pytest.raises(TypeError, match="topk_ids must be int32/int64"):
        rt.route(hs, ids, w, in_profile_run=False)  # swapped, as the on-card scaffold did
    with pytest.raises(TypeError, match="no topk_weights table baked"):
        rt.route(hs, w.to(torch.float64), ids, in_profile_run=False)  # no table baked for f64


def test_production_call_site_binds_route_by_field():
    """forward_impl must read .topk_weights/.topk_ids, never unpack route() positionally."""
    tree = ast.parse(ROUTED_EXPERTS.read_text())
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "route"
    ]
    assert len(calls) == 1
    src = ROUTED_EXPERTS.read_text()
    assert "npu_routing.topk_weights" in src and "npu_routing.topk_ids" in src
    # no `a, b = <something>.route(...)` anywhere
    bad = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Tuple)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "route"
    ]
    assert bad == [], f"route() unpacked positionally at {bad}"


def test_call_site_check_mutation_is_red():
    src = "def f(self):\n    topk_weights, topk_ids = self.kt_offload.route(a, b, c)\n"
    tree = ast.parse(src)
    bad = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Tuple)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "route"
    ]
    assert bad == [2]
