# SPDX-License-Identifier: Apache-2.0
"""The last metre: ``setup_streaming`` really builds and installs the streaming path.

What is judged is not what the streaming path *computes* but whether the objects
the forward will use are the objects the assembly produced, and whether the
assembly happens where it must. Expectations never come from the object under
test: the per-expert byte count is computed from the synthetic checkpoint's
safetensors headers, and the resident sets and shapes are this file's constants.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from safetensors.torch import save_file

import vllm_ascend
from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload import stream_io as sio
from vllm_ascend.kt_offload import weight_source as ws
from vllm_ascend.kt_offload import wiring
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan
from vllm_ascend.kt_offload.resident_driver import SLOT_WEIGHT_ATTRS
from vllm_ascend.kt_offload.resident_swap import live_target_violations
from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDSDynamicFusedMoEMethod

E, SLOTS, CHUNK, TOPK = 12, 4, 3, 2
NUM_LAYERS, MOE_LAYERS = 4, (1, 2)      # not 0: a layer mix-up must be able to show
HIDDEN, INTER, GROUP = 32, 16, 8
RESIDENT = (0, 1, 2, 3)

#: per-expert records of the synthetic checkpoint, in the checkpoint's own terms
SYN_SHAPES = {
    ("w1", "weight"): (INTER, HIDDEN // 2),
    ("w3", "weight"): (INTER, HIDDEN // 2),
    ("w2", "weight"): (HIDDEN, INTER // 2),
    ("w1", "scale"): (INTER, HIDDEN // GROUP),
    ("w3", "scale"): (INTER, HIDDEN // GROUP),
    ("w2", "scale"): (HIDDEN, INTER // GROUP),
}
#: what one expert costs, from the CHECKPOINT side. The provider derives the
#: same number from the quant method's declared shapes; they must agree, and
#: they are computed from different things.
CKPT_BYTES_PER_EXPERT = sum(n * k for n, k in SYN_SHAPES.values())

#: the layer's own parameters, as the quant method declares them for SLOTS rows
PARAM_SHAPES = {
    "w13_weight": (SLOTS, 2 * INTER, HIDDEN // 2),
    "w2_weight": (SLOTS, HIDDEN, INTER // 2),
    "w13_weight_scale": (SLOTS, 2 * INTER, HIDDEN // GROUP),
    "w2_weight_scale": (SLOTS, HIDDEN, INTER // GROUP),
}


@pytest.fixture(scope="module")
def syn_ckpt(tmp_path_factory):
    """A real safetensors checkpoint, small. Only its index is read here."""
    d = tmp_path_factory.mktemp("setup_streaming_ckpt")
    fn = "model-00001-of-00001.safetensors"
    tensors, wmap = {}, {}
    for layer in range(NUM_LAYERS):
        for e in range(E):
            for proj in ws.PROJS:
                for kind in ws.KINDS:
                    key = ws.expert_key(layer, e, proj, kind)
                    n, k = SYN_SHAPES[(proj, kind)]
                    raw = torch.full((n, k), (e + 121) % 8 + ws.E8M0_LO if kind == "scale" else e + 1,
                                     dtype=torch.uint8)
                    tensors[key] = raw.view(torch.float8_e8m0fnu) if kind == "scale" else raw.view(torch.int8)
                    wmap[key] = fn
    save_file(tensors, str(d / fn))
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    return str(d)


class RecordingMethod(AscendW4A8MXFPDSDynamicFusedMoEMethod):
    """The production shape declarations, with the cast replaced by a recorder.

    ``get_weight`` / ``get_dynamic_quant_param`` are the real ones, so the
    provider's slot is sized the way production sizes it;
    ``process_weights_after_loading`` is the one thing a host cannot run
    (``npu_format_cast``), so it records instead.
    """

    def __init__(self):  # noqa: D107 - the real __init__ wants a full quant config
        self.group_size = GROUP
        self.dynamic_eplb = False
        self.post_load_modules: list[torch.nn.Module] = []

    def process_weights_after_loading(self, module) -> None:
        self.post_load_modules.append(module)


def _copy_memory_(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``torch_npu.copy_memory_``: whole storage to whole storage, view ignored."""
    d = torch.empty(0, dtype=torch.uint8).set_(dst.untyped_storage())
    s = torch.empty(0, dtype=torch.uint8).set_(src.untyped_storage())
    if d.numel() != s.numel():
        raise RuntimeError(f"copy_memory_: {s.numel()} B into {d.numel()} B")
    d.copy_(s)


class FakeKt:
    """kt, as ``resident_swap`` reads it: its OWN pinned mask, and a full-set load."""

    def __init__(self, num_experts: int, resident):
        self.num_experts = num_experts
        self.gpu_experts_mask = torch.zeros(num_experts, dtype=torch.bool)
        self.gpu_experts_mask[list(resident)] = True
        self.loaded = list(range(num_experts))

    def cpu_weight_expert_ids(self):
        return list(self.loaded)

    def cpu_expert_ids(self):
        return [e for e in range(self.num_experts) if not bool(self.gpu_experts_mask[e])]


#: bound before the fixture replaces ``sio.DeviceSeam``, so ``Seam.make`` still
#: builds the production dataclass rather than calling itself.
_REAL_DEVICE_SEAM = sio.DeviceSeam


class Seam:
    """Every device-touching allocation, counted. The instrument for "off is off"."""

    def __init__(self):
        self.pinned: list[int] = []
        self.device: list[tuple] = []
        self.post_load: list[tuple] = []

    def alloc_pinned(self, nbytes: int) -> torch.Tensor:
        self.pinned.append(int(nbytes))
        return torch.zeros(int(nbytes), dtype=torch.uint8)

    def alloc_device(self, shape, dtype, device) -> torch.Tensor:
        self.device.append((tuple(int(s) for s in shape), dtype, device))
        return torch.zeros(tuple(int(s) for s in shape), dtype=dtype)

    def h2d(self, dst, src, *, what):
        dst.view(-1)[: src.numel()].copy_(src.view(-1))
        return None

    def wait(self, handle) -> None:
        return None

    def run_post_load(self, quant_method, module) -> None:
        self.post_load.append((quant_method, module))
        sio.default_post_load(quant_method, module)

    def make(self):
        return _REAL_DEVICE_SEAM(alloc_pinned=self.alloc_pinned, alloc_device=self.alloc_device,
                                 h2d=self.h2d, wait=self.wait, post_load=self.run_post_load)


def _vcfg(*, profiled: bool = False, capture_sizes=(), eager: bool = True, model: str = "/served-model"):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=HIDDEN, moe_intermediate_size=INTER,
                                           num_experts_per_tok=TOPK, num_hidden_layers=NUM_LAYERS),
            model=model, enforce_eager=eager),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE" if eager else "FULL_DECODE_ONLY"),
            cudagraph_capture_sizes=list(capture_sizes)),
        cache_config=SimpleNamespace(num_gpu_blocks=1234 if profiled else None),
    )


def make_module() -> torch.nn.Module:
    mod = torch.nn.Module()
    for name, shape in PARAM_SHAPES.items():
        mod.register_parameter(name, torch.nn.Parameter(torch.zeros(shape, dtype=torch.uint8),
                                                        requires_grad=False))
    mod.quant_method = RecordingMethod()
    mod.moe_config = SimpleNamespace(swiglu_limit=10.0)
    mod.log2phy = None
    return mod


class World:
    """Two MoE layers, loaded exactly the way production loads them."""

    def __init__(self, monkeypatch, *, ckpt: str | None, stream: bool = True, dynamic: bool = True,
                 profiled: bool = False, capture_sizes=(), eager: bool = True, min_tokens: int = 512,
                 model: str = "/served-model"):
        kl._reset_offload_registry()
        monkeypatch.setattr(kl, "_capturing", lambda: False)
        monkeypatch.setattr(kl.KTLayerOffload, "_build_wrapper",
                            lambda rt: FakeKt(E, rt.placement.resident_ids))
        monkeypatch.setattr(torch_npu, "copy_memory_", _copy_memory_, raising=False)
        # kt's capture buffers need the real kt_kernel; the capture SIZES --
        # what the start-up refusal reads -- are still derived by production
        # code from compilation_config.
        monkeypatch.setattr(kl.KTLayerOffload, "_prepare_capture_buffers", lambda rt, device: None)
        self.seam = Seam()
        monkeypatch.setattr(sio, "DeviceSeam", self.seam.make)
        # overlap_side_stream defaults to on, and this World builds real
        # KTLayerOffloads on the host, where after_weights_loaded would create a
        # torch_npu Stream and raise. Off explicitly: what is under test here is
        # the streaming-prefill wiring, not the side stream.
        self.cfg = KTOffloadConfig(
            enabled=True, cpuinfer_threads=1, num_resident_experts=SLOTS, weight_path=ckpt,
            overlap_side_stream=False,
            stream_prefill=stream, dynamic_resident=dynamic, stream_prefill_chunk_experts=CHUNK,
            stream_prefill_min_tokens=min_tokens)
        self.ascend_config = SimpleNamespace(kt_offload_config=self.cfg)
        self.vllm_config = _vcfg(profiled=profiled, capture_sizes=capture_sizes, eager=eager, model=model)
        plan = build_plan([list(RESIDENT)] * NUM_LAYERS, E)
        self.rt: dict[int, kl.KTLayerOffload] = {}
        self.mod: dict[int, torch.nn.Module] = {}
        for i in MOE_LAYERS:
            rt = kl.KTLayerOffload(self.cfg, plan.layer(i), i, self.vllm_config)
            mod = make_module()
            rt.attach_after_init(mod)               # kt patches the post-load hook here
            mod.quant_method.process_weights_after_loading(mod)
            self.rt[i], self.mod[i] = rt, mod

    def setup(self, vllm_config=None, **kw):
        return wiring.setup_streaming(vllm_config or self.vllm_config, ascend_config=self.ascend_config, **kw)

    def wiring_state(self) -> dict:
        return {i: (rt._stream_provider, rt._stream_slot_provider, rt._resident_fold)
                for i, rt in self.rt.items()}


@pytest.fixture(autouse=True)
def _clean_registry():
    yield
    kl._reset_offload_registry()


def test_the_switch_off_allocates_nothing_and_touches_no_layer(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt, stream=False, dynamic=False)
    before = w.wiring_state()
    assert w.setup() is None
    assert w.seam.pinned == [] and w.seam.device == []
    assert w.wiring_state() == before
    for rt in w.rt.values():
        assert rt._stream_provider is None
        assert rt._stream_slot_provider is None
        assert rt._resident_fold is None
        assert rt._steps is None


def test_the_feature_off_wholesale_allocates_nothing(monkeypatch, syn_ckpt):
    """``enabled: false`` with the streaming switches set: still nothing."""
    w = World(monkeypatch, ckpt=syn_ckpt)
    w.ascend_config.kt_offload_config = KTOffloadConfig(
        enabled=False, overlap_side_stream=False, stream_prefill=True, dynamic_resident=True)
    assert w.setup() is None
    assert w.seam.pinned == [] and w.seam.device == []
    assert all(rt._stream_provider is None for rt in w.rt.values())


def test_positive_control_the_same_instrument_moves_when_the_switch_is_on(monkeypatch, syn_ckpt):
    """Without this, "zero allocations" above would also pass on a dead seam."""
    w = World(monkeypatch, ckpt=syn_ckpt, stream=True, dynamic=True)
    assert w.setup() is not None
    assert w.seam.pinned == [CHUNK * CKPT_BYTES_PER_EXPERT] * sio.DEFAULT_HOST_BUFFERS
    assert len(w.seam.device) > 0
    assert all(rt._stream_provider is not None for rt in w.rt.values())


def test_the_provider_is_built_from_the_live_layers(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt)
    got = w.setup()
    prov = got.provider
    assert got.layers == tuple(MOE_LAYERS)
    # checkpoint: cfg.weight_path when it is set (layer.py KTLayerOffload.__init__)
    assert prov.source.ckpt_dir == syn_ckpt
    assert prov.source.num_experts == E
    # chunk size: the tunable, not a constant
    assert prov.chunk_experts == CHUNK
    # hidden/intermediate: judged through the byte count, whose expectation
    # comes from the checkpoint's safetensors headers -- not from the provider.
    header_bytes = ws.WeightSource(syn_ckpt, num_experts=E).plan_chunk(MOE_LAYERS[0], (0,)).nbytes
    assert header_bytes == CKPT_BYTES_PER_EXPERT
    assert prov.bytes_per_expert == CKPT_BYTES_PER_EXPERT
    assert prov.staging_bytes == CHUNK * CKPT_BYTES_PER_EXPERT
    # the slot really is chunk-sized, one buffer per parameter
    assert {n: s for n, (s, _) in prov.shapes.items()} == {
        n: (CHUNK, *shape[1:]) for n, shape in PARAM_SHAPES.items()}


def test_the_checkpoint_defaults_to_the_served_model(monkeypatch, syn_ckpt):
    """``weight_path: null`` -> ``model_config.model``, the same fallback the layer took."""
    w = World(monkeypatch, ckpt=None, model=syn_ckpt)
    assert w.setup().provider.source.ckpt_dir == syn_ckpt
    assert all(rt.weight_path == syn_ckpt for rt in w.rt.values())


def test_layers_that_disagree_on_a_model_wide_number_are_refused(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt)
    w.rt[MOE_LAYERS[1]].hidden_size = HIDDEN * 2
    with pytest.raises(RuntimeError, match="disagree on hidden_size"):
        w.setup()
    assert w.seam.pinned == [] and w.seam.device == []


def test_the_assembled_targets_are_the_live_objects(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt)
    got = w.setup()
    assert got.fold is not None
    for i, rt in w.rt.items():
        tg = got.fold.committer.targets(i)
        assert live_target_violations(tg, **rt.live_swap_objects()) == []
        for dt in (torch.int32, torch.int64):
            assert tg.slot_tables[dt] is rt._slot_tables[dt]
        for dt in (torch.float32, torch.bfloat16):
            assert tg.keep_tables[dt] is rt._keep_tables[dt]
        for name in SLOT_WEIGHT_ATTRS:
            assert tg.weights[name] is getattr(w.mod[i], name)
        assert tg.kt_mask is rt.wrapper.gpu_experts_mask


@pytest.mark.parametrize("key", ["slot_tables", "keep_tables", "weights", "kt_mask"])
def test_mut_a_table_built_afresh_makes_the_assembly_refuse(monkeypatch, syn_ckpt, key):
    """The assembly picks up a fresh copy instead of the live one: same values,
    different storage, so the commit would move the weights and leave the
    routing pointing at the old slots. ``setup_streaming`` must not return."""
    honest = kl.KTLayerOffload.live_swap_objects

    def fresh(self):
        live = dict(honest(self))
        p = self.placement
        if key == "slot_tables":
            live[key] = {dt: p.safe_slot_table(dt) for dt in (torch.int32, torch.int64)}
        elif key == "keep_tables":
            live[key] = {dt: p.keep_table(dt) for dt in (torch.float32, torch.bfloat16)}
        elif key == "kt_mask":
            live[key] = p.resident_mask()
        else:
            live[key] = {n: t.detach().clone() for n, t in live[key].items()}
        return live

    w = World(monkeypatch, ckpt=syn_ckpt)
    monkeypatch.setattr(kl.KTLayerOffload, "live_swap_objects", fresh)
    with pytest.raises(RuntimeError, match="not the objects the forward reads"):
        w.setup()
    assert all(rt._resident_fold is None for rt in w.rt.values())


def test_one_provider_object_serves_the_stream_and_the_fold(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt)
    got = w.setup()
    for rt in w.rt.values():
        assert rt._stream_provider is got.provider
        assert rt._stream_slot_provider.__self__ is got.provider
        assert rt._stream_slot_provider == got.provider.slot_for
        assert rt._resident_fold is got.fold
        assert rt._steps is not None
    assert got.fold.provider is got.provider
    assert len({id(rt._stream_provider) for rt in w.rt.values()}) == 1
    # One slot and one staging for the whole MODEL: four device buffers each,
    # not four per layer. The count is the observable that separates the two.
    assert len(w.seam.device) == 2 * len(SLOT_WEIGHT_ATTRS)
    assert len(w.seam.pinned) == sio.DEFAULT_HOST_BUFFERS
    # one step counter, shared, so all layers of one forward commit at one step
    assert len({id(rt._steps) for rt in w.rt.values()}) == 1


def test_stream_only_installs_the_provider_and_no_fold(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt, dynamic=False)
    got = w.setup()
    assert got.fold is None
    for rt in w.rt.values():
        assert rt._stream_provider is got.provider
        assert rt._resident_fold is None
    # the staging is the dynamic path's, so only the slot was allocated
    assert len(w.seam.device) == len(SLOT_WEIGHT_ATTRS)


def test_the_provider_gets_the_layers_own_patched_quant_method(monkeypatch, syn_ckpt):
    """The wiring hands over the PATCHED method: ``default_post_load`` must
    reach the hook kt wrapped, and the wiring must not call
    ``process_weights_after_loading`` itself, which would rebind the live tables
    of the layer in flight and reload its CPU half."""
    w = World(monkeypatch, ckpt=syn_ckpt)
    i = MOE_LAYERS[0]
    qm = w.mod[i].quant_method
    got = w.setup()
    assert got.provider.quant_method is qm
    hook = qm.process_weights_after_loading
    assert getattr(hook, "_kt_offload_patched", False) is True
    assert getattr(hook, "_kt_offload_original", None) is not None
    # the wiring itself ran the hook exactly as many times as the load did (once)
    assert qm.post_load_modules == [w.mod[i]]

    rt, tables, keeps = w.rt[i], w.rt[i]._slot_tables, w.rt[i]._keep_tables
    throwaway = make_module()
    sio.default_post_load(got.provider.quant_method, throwaway)
    assert qm.post_load_modules == [w.mod[i], throwaway]     # the ORIGINAL hook ran...
    assert rt._slot_tables is tables and rt._keep_tables is keeps   # ...and kt's additions did not
    assert rt._module is w.mod[i]


def test_a_layer_that_never_saw_the_wiring_refuses_to_stream(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt, min_tokens=4)
    rt, mod = w.rt[MOE_LAYERS[0]], w.mod[MOE_LAYERS[0]]
    hs = torch.zeros(8, HIDDEN)
    tw = torch.ones(8, TOPK)
    ids = torch.zeros(8, TOPK, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="no slot provider is installed"):
        rt.maybe_stream_prefill(mod, hs, tw, ids, in_profile_run=False)
    # the stream wired by hand but not the fold: the other half must be loud too
    rt._stream_slot_provider = lambda chunk_index, expert_ids: None
    with pytest.raises(RuntimeError, match="no resident fold is installed"):
        rt.maybe_stream_prefill(mod, hs, tw, ids, in_profile_run=False)


def test_setup_with_no_registered_layer_refuses(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt)
    kl._reset_offload_registry()
    with pytest.raises(RuntimeError, match="no MoE layer has registered"):
        w.setup()
    assert w.seam.pinned == [] and w.seam.device == []


def test_mut_a_half_installed_model_does_not_leave_setup(monkeypatch, syn_ckpt):
    """One layer wired, the other not -- the failure would otherwise surface on
    whichever request reaches the unwired layer first."""
    w = World(monkeypatch, ckpt=syn_ckpt, dynamic=False)
    honest = kl.install_stream_provider

    def only_the_first(provider, *, layers=None):
        return honest(provider, layers=list(layers)[:1])

    monkeypatch.setattr(wiring, "install_stream_provider", only_the_first)
    with pytest.raises(RuntimeError, match="did not take the streaming provider"):
        w.setup()


def test_mut_a_second_provider_for_the_fold_is_refused(monkeypatch, syn_ckpt):
    """The fold must read the chunks out of the object that served them."""
    w = World(monkeypatch, ckpt=syn_ckpt)
    honest = kl.install_resident_fold

    def fold_with_another_provider(provider, **kw):
        fold = honest(provider, **kw)
        for rt in w.rt.values():
            rt._stream_provider = SimpleNamespace(slot_for=provider.slot_for, set_layer=provider.set_layer)
        return fold

    monkeypatch.setattr(wiring, "install_resident_fold", fold_with_another_provider)
    with pytest.raises(RuntimeError, match="did not take the streaming provider"):
        w.setup()


def test_dynamic_resident_without_stream_prefill_is_refused(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt, stream=False, dynamic=True)
    with pytest.raises(RuntimeError, match="dynamic_resident without stream_prefill"):
        w.setup()
    assert w.seam.pinned == [] and w.seam.device == []


def test_setup_refuses_once_the_kv_cache_has_been_sized(monkeypatch, syn_ckpt):
    w = World(monkeypatch, ckpt=syn_ckpt, profiled=True)
    with pytest.raises(RuntimeError, match="KV cache has already been sized"):
        w.setup()
    assert w.seam.pinned == [] and w.seam.device == []


def test_setup_is_accepted_before_the_profile(monkeypatch, syn_ckpt):
    """Positive control for the gate above: the same call, profile not yet run."""
    w = World(monkeypatch, ckpt=syn_ckpt, profiled=False)
    assert w.setup() is not None
    assert w.seam.device != []


def test_the_layers_own_cache_config_is_checked_too(monkeypatch, syn_ckpt):
    """A clean config handed in does not excuse layers that have already been profiled.

    Stream-only on purpose: with the fold on, ``install_resident_fold``'s own
    gate would also catch this, and then the arm would not be judging the gate
    in ``setup_streaming``.
    """
    w = World(monkeypatch, ckpt=syn_ckpt, profiled=True, dynamic=False)
    with pytest.raises(RuntimeError, match="KV cache has already been sized"):
        w.setup(vllm_config=_vcfg(profiled=False))
    assert w.seam.pinned == [] and w.seam.device == []


def test_a_captured_size_at_or_above_the_threshold_is_refused_at_startup(monkeypatch, syn_ckpt):
    """``assert_eager_only`` would otherwise raise mid-serve on the first such forward."""
    w = World(monkeypatch, ckpt=syn_ckpt, eager=False, capture_sizes=(1, 2, 4, 8, 1024), min_tokens=512)
    with pytest.raises(RuntimeError, match="at or above stream_prefill_min_tokens"):
        w.setup()
    assert w.seam.pinned == [] and w.seam.device == []


def test_a_capture_list_below_the_threshold_is_accepted(monkeypatch, syn_ckpt):
    """Positive control: the production capture list ([1,2,4,8]) must pass."""
    w = World(monkeypatch, ckpt=syn_ckpt, eager=False, capture_sizes=(1, 2, 4, 8), min_tokens=512)
    assert w.setup() is not None


WORKER_PY = Path(vllm_ascend.__file__).parent / "worker" / "worker.py"


def _method(tree: ast.Module, cls: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef) and sub.name == name:
                    return sub
    raise AssertionError(f"{cls}.{name} not found in {WORKER_PY}")


def _calls(node: ast.AST) -> list[tuple[int, str]]:
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            f = sub.func
            out.append((sub.lineno, f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")))
    return sorted(out)


def test_the_call_site_is_after_the_model_load_and_before_any_profiling():
    """``NPUWorker.load_model`` is the one method that runs after every MoE
    layer's ``process_weights_after_loading`` and before anything that profiles;
    the landing point is read out of the source, not asserted in prose."""
    tree = ast.parse(WORKER_PY.read_text())
    load = _method(tree, "NPUWorker", "load_model")
    names = [n for _, n in _calls(load)]
    assert "setup_streaming" in names, names
    assert "profile_run" not in names and "determine_available_memory" not in names, names
    first_setup = min(ln for ln, n in _calls(load) if n == "setup_streaming")
    last_load = max(ln for ln, n in _calls(load) if n == "load_model")
    assert last_load < first_setup, "setup_streaming must follow model_runner.load_model()"
    # ...and the profile really is somewhere else, later.
    profile = _method(tree, "NPUWorker", "determine_available_memory")
    assert "profile_run" in [n for _, n in _calls(profile)]
    assert "setup_streaming" not in [n for _, n in _calls(profile)]


def test_the_call_site_reaches_both_model_runners():
    """Independent of ``use_v2_model_runner``: the worker calls whichever it built."""
    src = WORKER_PY.read_text()
    assert "self.model_runner.load_model()" in src
    tree = ast.parse(src)
    init_device = _method(tree, "NPUWorker", "init_device")
    built = {n for _, n in _calls(init_device)} & {"NPUModelRunner", "NPUModelRunnerV2"}
    assert built == {"NPUModelRunner", "NPUModelRunnerV2"}, built


def test_a_chunk_size_that_does_not_divide_the_experts_is_refused(monkeypatch, syn_ckpt):
    """A non-divisor leaves a short final chunk, which stream_io flags as
    card-unknown, so production must not reach it. The expectation is arithmetic
    on the layer's own expert count, not anything the wiring reports."""
    w = World(monkeypatch, ckpt=syn_ckpt)
    w.cfg.stream_prefill_chunk_experts = 5          # 12 % 5 == 2
    with pytest.raises(ValueError, match="does not divide"):
        w.setup()
    assert w.seam.pinned == [] and w.seam.device == [], "it refused before allocating anything"


def test_a_chunk_size_that_divides_is_accepted(monkeypatch, syn_ckpt):
    """Positive control: the same call with a divisor goes through."""
    w = World(monkeypatch, ckpt=syn_ckpt)
    w.cfg.stream_prefill_chunk_experts = 6          # 12 % 6 == 0
    assert w.setup() is not None
    assert w.seam.device != []


class _Scheme:
    """Stands in for AscendW4A8MXFPDynamicFusedMoEMethod: declares the shapes."""

    def get_weight(self, experts, inter, hidden, dtype):
        import torch
        return {"w13_weight": torch.empty(experts, 2 * inter, hidden // 2, dtype=torch.uint8),
                "w2_weight": torch.empty(experts, hidden, inter // 2, dtype=torch.uint8)}

    def get_dynamic_quant_param(self, experts, inter, hidden, dtype):
        import torch
        return {"w13_weight_scale": torch.empty(experts, 2 * inter, hidden // 32, dtype=torch.uint8),
                "w2_weight_scale": torch.empty(experts, hidden, inter // 32, dtype=torch.uint8)}


class _Wrapper:
    """Stands in for AscendFusedMoEMethod: holds the scheme, owns the post-load hook."""

    def __init__(self, scheme):
        self.quant_method = scheme

    def process_weights_after_loading(self, module):       # what kt patches
        pass


def test_shapes_come_from_the_scheme_inside_the_wrapper():
    from vllm_ascend.kt_offload.stream_io import shape_declaring_method
    scheme = _Scheme()
    assert shape_declaring_method(scheme) is scheme                  # flat: itself
    assert shape_declaring_method(_Wrapper(scheme)) is scheme        # wrapped: unwrapped once


def test_an_object_with_no_shape_source_anywhere_is_refused():
    """Loud, and names both objects -- a silent fallback would hand the GEMM a
    buffer built from the wrong declaration and fail much later."""
    from vllm_ascend.kt_offload.stream_io import StreamIOError, shape_declaring_method

    class _Neither:
        pass

    with pytest.raises(StreamIOError, match="carries "):
        shape_declaring_method(_Neither())
    with pytest.raises(StreamIOError, match="carries "):
        shape_declaring_method(_Wrapper(_Neither()))


def test_chunk_param_shapes_accepts_the_production_shape_of_quant_method():
    """The wrapper must produce the same shapes as the flat scheme."""
    from vllm_ascend.kt_offload.stream_io import chunk_param_shapes
    flat = chunk_param_shapes(_Scheme(), 8, 256, 128)
    wrapped = chunk_param_shapes(_Wrapper(_Scheme()), 8, 256, 128)
    assert flat == wrapped and flat, (flat, wrapped)


def test_all_covers_every_entry_point_the_worker_imports():
    """``__all__`` must name every symbol ``NPUWorker.load_model`` imports. The
    expectation is read out of worker.py's source rather than restated here: a
    list copied from the object under test cannot catch that object being
    wrong."""
    import pathlib
    import re
    from vllm_ascend.kt_offload import wiring

    worker_src = (pathlib.Path(wiring.__file__).parents[1] / "worker" / "worker.py").read_text()
    imported = set()
    for m in re.finditer(r"from vllm_ascend\.kt_offload\.wiring import ([^\n(]+)", worker_src):
        imported |= {n.strip() for n in m.group(1).split(",") if n.strip()}
    assert imported, "worker.py imports nothing from wiring -- the scan is broken, not the code"
    missing = sorted(imported - set(wiring.__all__))
    assert not missing, f"wiring.__all__ does not name {missing}, which worker.py imports"
