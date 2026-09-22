# SPDX-License-Identifier: Apache-2.0
"""Load-time code must name the device of every tensor it creates.

The dimension under test: a CPU unit test runs with the default factory device
set to the CPU, while the model loader runs with it set to the NPU (vLLM wraps
weight loading in ``with torch.device(device)``). A factory call that omits
``device=`` therefore lands on a *different* device in production than in this
suite, and no CPU test can see the difference -- the tensor is on the CPU
either way, for the wrong reason. Two of these tensors are handed straight to
kt, which requires host memory: ``gpu_experts_mask`` and
``physical_to_logical_map_cpu``.

Three arms, and the last two are what give the first its meaning:

1. real code, loader's default device -> every load-time tensor is where it
   was meant to be;
2. mutant (the explicit ``device="cpu"`` stripped from the source), CPU
   default -> **green**, which is why this dimension had zero coverage: no
   existing test could have failed on it;
3. the same mutant, loader's default device -> **red**.

A fourth check scans the package's source: every ``torch`` factory call must
name a device, so the next call site someone adds is covered before it is
written.

``meta`` stands in for the NPU default: it needs no accelerator, so this file
runs in both test modes, and nothing in the load-time code ever asks for it
(asserted below).

This is not the ``_pinned_empty()`` hook from the round before. That hook
removed a test-mode *symptom* -- pinned allocation needs a registered
accelerator, so one mode was red and the other green -- and said nothing about
which device a tensor lands on when nobody names one.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import expert_counter as real_counter
from vllm_ascend.kt_offload import layer as real_layer
from vllm_ascend.kt_offload import placement as real_placement
from vllm_ascend.kt_offload.config import KTOffloadConfig

H, TOPK, NEXP = 8, 2, 16
POISON = "meta"  # a non-CPU default device that needs no accelerator
EXPLICIT_DEVICE = re.compile(r",\s*device=\"cpu\"")


@contextlib.contextmanager
def default_device(kind: str):
    """What the loader does around weight loading: change the default device."""
    with torch.device(kind):
        yield


def load_mutant(qualname: str):
    """Re-exec a module with every explicit ``device="cpu"`` removed.

    A source-level mutant, not a patched ``torch.zeros``: the default-device
    machinery dispatches on the *identity* of the factory functions (and caches
    that set), so a wrapper around ``torch.zeros`` silently stops receiving the
    default device and would make every arm below read "cpu" for the wrong
    reason. Registered in ``sys.modules`` under its own mutant name, which
    ``dataclasses`` needs to resolve the module's annotations and which no
    import in the suite can reach.
    """
    real = importlib.import_module(qualname)
    path = real.__file__
    mutated, n = EXPLICIT_DEVICE.subn("", Path(path).read_text())
    assert n > 0, f"{qualname} has no explicit device= to strip; the mutant would be a no-op"
    name = f"{qualname}__mutant_no_explicit_device"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(mutated, f"{path} [mutant: no explicit device]", "exec"), mod.__dict__)
    mod._mutations = n
    return mod


@pytest.fixture(scope="module")
def mutants():
    return SimpleNamespace(
        placement=load_mutant("vllm_ascend.kt_offload.placement"),
        layer=load_mutant("vllm_ascend.kt_offload.layer"),
        counter=load_mutant("vllm_ascend.kt_offload.expert_counter"),
    )


@pytest.fixture
def real():
    return SimpleNamespace(placement=real_placement, layer=real_layer, counter=real_counter)


def failure(fn) -> BaseException | None:
    """Run a check and hand back what made it red, or None if it passed.

    A check is red on the mutant either by asserting, or because the mutant
    cannot build or use the tensor at all (reading or copying a meta tensor
    raises). The arm below also requires the reason to *name* the poison
    device, so a mutant that is red for an unrelated reason does not read as
    coverage.
    """
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the reason is the assertion
        return exc
    return None


# -- the subjects: one function per load-time surface --------------------


def check_placement_tables(mods, _monkeypatch, _tmp) -> None:
    """The tables the routing split and kt's resident mask are built from."""
    pl = mods.placement
    p = pl.build_plan([[1, 3, 5]], NEXP).layer(0)
    tables = {
        "resident_mask": p.resident_mask(),  # kt copies this into pinned memory
        "logical_to_slot": p.logical_to_slot(),
        "slot_to_logical": p.slot_to_logical(),
        "safe_slot_table": p.safe_slot_table(torch.int32),
        "keep_table": p.keep_table(torch.float32),
    }
    for name, t in tables.items():
        assert t.device.type == "cpu", f"{name} landed on {t.device}: it took the loader's default device"
    # the values too: a table built on the wrong device cannot answer these
    assert tables["resident_mask"].tolist() == [i in (1, 3, 5) for i in range(NEXP)]
    assert tables["slot_to_logical"].tolist()[: p.num_resident] == [1, 3, 5]


def check_frequency_table(mods, _monkeypatch, tmp) -> None:
    """The resident-set frequency table, read while the config is resolved."""
    path = str(Path(tmp) / "freq.json")
    Path(path).write_text(str([[float(j) for j in range(NEXP)] for _ in range(2)]).replace("'", ""))
    freq = mods.placement.load_frequency_table(path)
    assert freq.device.type == "cpu", f"the frequency table landed on {freq.device}"
    assert freq.shape == (2, NEXP) and float(freq[1, 1]) == 1.0


class FakeWrapper:
    """kt's wrapper; records the two arguments that must be host tensors."""

    _layer_has_pending_deferred: dict = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.moe = SimpleNamespace(forward_task=lambda *a: a)
        self.cpu_infer = SimpleNamespace(submit=lambda task: None, sync=lambda allow: None)
        self.p2l = None

    def load_weights(self, physical_to_logical_map_cpu):
        # kt's parameter name says what it needs: a host map
        self.p2l = physical_to_logical_map_cpu


def _layer_runtime(mods, monkeypatch, packed: bool):
    lm, pl = mods.layer, mods.placement
    plan = pl.build_plan([[0, 3]], NEXP)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, packed_transfer=packed)
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=TOPK),
            model="/m",
            enforce_eager=True,  # no capture: this file is about load time
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"), cudagraph_capture_sizes=[]),
    )
    made: list[FakeWrapper] = []

    def factory(**kw):
        made.append(FakeWrapper(**kw))
        return made[-1]

    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=factory))
    monkeypatch.setattr(lm, "_install_capture_probe", lambda cls: None)
    # each arm stands in for a fresh process: the wrapper registry (A5) is
    # process-level by design, and a module-scoped mutant keeps earlier arms'
    # wrappers alive
    lm._reset_wrapper_registry()
    # Pinned allocation needs a registered accelerator, so the CPU suite cannot
    # call the real one. The stand-in names its device exactly as the subject
    # does -- and that the subject does is asserted separately, on its source.
    monkeypatch.setattr(lm, "_pinned_empty", lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cpu"))
    rt = lm.KTLayerOffload(cfg, plan.layer(0), 0, vcfg)
    mod = torch.nn.Module()
    # stands in for the loader's NPU module: load-time tensors must follow the
    # *module*, never the process default
    mod.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3, device="cpu"), requires_grad=False)
    return rt, mod, made


def check_kt_arguments(mods, monkeypatch, _tmp, packed: bool = False) -> None:
    rt, mod, made = _layer_runtime(mods, monkeypatch, packed)
    rt.after_weights_loaded(mod)
    assert made, "kt was never constructed"
    mask = made[0].kwargs["gpu_experts_mask"]
    assert mask.device.type == "cpu", f"kt's gpu_experts_mask landed on {mask.device}"
    assert int(mask.sum()) == rt.placement.num_resident
    p2l = made[0].p2l
    assert p2l is not None, "kt never got its physical->logical map"
    assert p2l.device.type == "cpu", f"kt's physical->logical map landed on {p2l.device}"
    assert p2l.tolist() == list(range(NEXP))
    for dt, t in rt._slot_tables.items():
        assert t.device.type == "cpu", f"slot table {dt} landed on {t.device}"
    for dt, t in rt._keep_tables.items():
        assert t.device.type == "cpu", f"keep table {dt} landed on {t.device}"
    if packed:
        buf = rt._packed_buffers(1, mods.layer.KTLayerOffload._device_of(mod))
        for key in ("dev", "dev_out", "dev_hidden", "dev_ids", "dev_weights", "pin", "pin_out", "bsz"):
            assert buf[key].device.type == "cpu", f"packed {key} landed on {buf[key].device}"
        assert int(buf["bsz"][0]) == 1, "the packed batch-size cell is unreadable"


def check_kt_arguments_packed(mods, monkeypatch, tmp) -> None:
    check_kt_arguments(mods, monkeypatch, tmp, packed=True)


def check_counter_dump(mods, monkeypatch, tmp) -> None:
    """The routing counter's per-bucket token totals, written at exit."""
    ec = mods.counter
    monkeypatch.setattr(ec, "OUT_DIR", str(tmp))
    monkeypatch.setattr(ec, "ENABLED", True)
    monkeypatch.setattr(ec, "_S", None)
    monkeypatch.setattr(ec, "_capturing", lambda: False)
    monkeypatch.setattr(ec, "_batch_split", lambda: (1, 2))
    monkeypatch.setattr(ec, "_dp_info", lambda: (0, 1))
    monkeypatch.setattr(ec, "_rows_are_dp_gathered", lambda: (False, None))
    monkeypatch.setattr(ec.atexit, "register", lambda fn: None)
    ids = torch.zeros(2, ec.TOPK, dtype=torch.int32, device="cpu")
    ec.record(SimpleNamespace(layer_name="model.layers.0.ffn.experts"), ids, False)
    path = ec.dump()
    out = torch.load(path, weights_only=True)
    for bucket, t in out["tokens"].items():
        assert t.device.type == "cpu", f"tokens[{bucket}] landed on {t.device}"
    assert int(out["tokens"]["decode"][0]) == 1 and int(out["tokens"]["prefill"][0]) == 1


SUBJECTS = {
    "placement_tables": check_placement_tables,
    "frequency_table": check_frequency_table,
    "kt_arguments": check_kt_arguments,
    "kt_arguments_packed": check_kt_arguments_packed,
    "counter_dump": check_counter_dump,
}


# -- arm 1: the real code, under the loader's default device -------------


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_load_time_tensors_are_host_under_the_loader_default(real, monkeypatch, tmp_path, name):
    with default_device(POISON):
        assert torch.zeros(2).device.type == POISON, "the loader-like default device is not in effect"
        SUBJECTS[name](real, monkeypatch, str(tmp_path))


# -- arm 2/3: the mutant, under each default device ----------------------

# Which module's explicit ``device=`` each subject depends on: one module is
# mutated per arm, so a red arm is attributable to that module's source.
MUTATED_MODULE = {
    "placement_tables": "placement",
    "frequency_table": "placement",
    "kt_arguments": "layer",
    "kt_arguments_packed": "layer",
    "counter_dump": "counter",
}


def _mods(mutants, mutated: str) -> SimpleNamespace:
    modules = {"placement": real_placement, "layer": real_layer, "counter": real_counter}
    modules[mutated] = getattr(mutants, mutated)
    return SimpleNamespace(**modules)


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_mutant_is_invisible_under_a_cpu_default(mutants, monkeypatch, tmp_path, name):
    """What every other test in this suite runs with: on a CPU default the
    explicit device changes nothing, so none of them covers this dimension."""
    assert torch.zeros(2).device.type == "cpu", "this arm needs the default device on the CPU"
    mods = _mods(mutants, MUTATED_MODULE[name])
    SUBJECTS[name](mods, monkeypatch, str(tmp_path))  # green, and that is the point


@pytest.mark.parametrize("name", sorted(SUBJECTS))
def test_mutant_is_red_under_the_loader_default(mutants, monkeypatch, tmp_path, name):
    subject, mutated = SUBJECTS[name], MUTATED_MODULE[name]
    mods = _mods(mutants, mutated)
    with default_device(POISON):
        assert torch.zeros(2).device.type == POISON, "the loader-like default device is not in effect"
        exc = failure(lambda: subject(mods, monkeypatch, str(tmp_path)))
    assert exc is not None, (
        f"{name}: {mutated}.py fell back to the default device and every check still passed"
    )
    assert POISON in str(exc), f"{name}: red for a reason that does not name the default device: {exc!r}"


# -- controls on the instrument ------------------------------------------


def test_the_mutant_removed_something_from_every_module(mutants):
    counts = {m: getattr(mutants, m)._mutations for m in ("placement", "layer", "counter")}
    assert all(n > 0 for n in counts.values()), counts
    # separate module objects: the real modules are untouched, and their
    # source still names its devices
    for name, realmod in (("placement", real_placement), ("layer", real_layer), ("counter", real_counter)):
        assert getattr(mutants, name) is not realmod
    assert EXPLICIT_DEVICE.search(Path(real_placement.__file__).read_text()), "the real source lost its explicit device"


def test_the_poison_device_is_one_the_load_time_code_never_asks_for():
    """Positive control on the instrument: if the load-time code asked for
    ``meta`` itself, the arms above would compare a device against itself.
    Matched as a device literal, not as a substring -- "attn metadata" contains
    the word."""
    literal = re.compile(rf"device=[\"']{POISON}[\"']")
    for mod in (real_placement, real_counter):
        assert not literal.search(Path(mod.__file__).read_text()), f"{mod.__name__} asks for {POISON}"
    # layer.py does ask for meta, but only for shape references inside forward
    load_time = Path(real_layer.__file__).read_text().split("    def route(")[0]
    assert not literal.search(load_time), "layer.py's load-time code asks for the poison device"


def test_pinned_allocation_names_its_device():
    """The one load-time allocator the CPU suite replaces with a stand-in, so
    its device naming is checked on the subject itself."""
    import inspect

    src = inspect.getsource(real_layer._pinned_empty)
    assert 'device="cpu"' in src and "pin_memory=True" in src, src


# -- the same rule, as a scan over the source ----------------------------

FACTORIES = frozenset(
    {"zeros", "ones", "empty", "full", "tensor", "arange", "rand", "randn", "eye", "linspace", "as_tensor"}
)
# ``*_like`` and ``from_numpy`` are exempt: the first takes the device from its
# argument, the second is host memory by construction.
SCANNED = ("placement", "layer", "expert_counter", "config", "hostfunc_log")


def scan_factory_calls(src: str) -> tuple[int, list[str]]:
    """(calls seen, calls that let the default device decide)."""
    import ast

    seen, bare = 0, []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in FACTORIES or not ast.unparse(node.func.value).endswith("torch"):
            continue
        seen += 1
        if not any(kw.arg == "device" for kw in node.keywords):
            bare.append(f"line {node.lineno}: {ast.unparse(node)[:80]}")
    return seen, bare


def test_no_tensor_in_this_package_takes_the_default_device():
    """The arms above cover the call sites that exist today; this covers the
    next one someone adds. Reports its own coverage: a module the scan cannot
    see would otherwise read exactly like a clean one."""
    total = 0
    for name in SCANNED:
        mod = importlib.import_module(f"vllm_ascend.kt_offload.{name}")
        seen, bare = scan_factory_calls(Path(mod.__file__).read_text())
        assert not bare, f"{name}.py lets the default device decide:\n  " + "\n  ".join(bare)
        total += seen
    # 15 until 2026-09-22, when the forked path dropped its ``torch.empty(...,
    # device="meta")`` shape reference: it hands kt a buffer dict now, not a
    # shape. The floor is a "the scan still finds the file" check, so it tracks
    # the real count rather than being left above it.
    assert total >= 14, f"the scan only saw {total} factory calls; it is looking in the wrong place"


def test_the_scan_can_go_red():
    """Positive control: the scan must flag an omitted device, and must not
    flag the two exempt spellings."""
    seen, bare = scan_factory_calls("import torch\nx = torch.zeros(3, dtype=torch.int64)\n")
    assert seen == 1 and len(bare) == 1
    seen, bare = scan_factory_calls(
        "import torch\na = torch.zeros_like(x)\nb = torch.from_numpy(y)\nc = torch.empty(2, device='cpu')\n"
    )
    assert seen == 1 and not bare
