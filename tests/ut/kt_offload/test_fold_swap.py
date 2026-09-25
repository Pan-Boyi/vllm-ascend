# SPDX-License-Identifier: Apache-2.0
"""The folded swap-in, judged against an INDEPENDENT static build: the expected
weights are checkpoint bytes read with ``safe_open`` into a production
``explicit`` placement, and the chosen set is this file's ``HOT`` -- neither
comes from the fold. Nothing under test is stubbed; only what exists on the card
alone is modelled (``Platform``, and ``test_stream_chain``'s kernel fakes).
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import textwrap
import threading
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from safetensors import safe_open
from safetensors.torch import save_file
from torch.overrides import TorchFunctionMode

# The scheduler outputs are built by the module that judges ``observe``, not a
# second copy of them.
from tests.ut.kt_offload.test_prefill_window import new_req, prefill_steps, sched

# The chain's production objects and its kernel fakes are reused, not rewritten.
from tests.ut.kt_offload.test_stream_chain import (  # noqa: F401  (ascend_runtime is an autouse fixture)
    _fake_grouped_matmul,
    _fake_init_routing_v2,
    _fake_swiglu_group_quant,
    _fake_unpermute,
    ascend_runtime,
    moe_config,
    quant_method,
)
from vllm_ascend.kt_offload import fold_swap, prefill_window
from vllm_ascend.kt_offload import fold_swap as fold_swap_module
from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload import weight_source as ws
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.fold_swap import (
    FoldError,
    FoldIncompleteError,
    ResidentFold,
    fold_weight_write,
)
from vllm_ascend.kt_offload.placement import plan_from_config
from vllm_ascend.kt_offload.prefill_window import PrefillWindow
from vllm_ascend.kt_offload.resident_driver import PolicyError
from vllm_ascend.kt_offload.resident_swap import (
    ALL_CPU_WEIGHTS,
    LayerSwapTargets,
    OwnershipConflictError,
    WriterStamp,
    ZeroCoverageError,
    check_resident_state,
)
from vllm_ascend.kt_offload.stream_chain import SLOT_WEIGHT_ATTRS, run_streamed_layer
from vllm_ascend.kt_offload.stream_io import DeviceSeam, StreamSlotProvider, default_post_load
from vllm_ascend.kt_offload.stream_prefill import build_chunk_plan
from vllm_ascend.kt_offload.weight_source import WeightSource

HIDDEN, INTER, GROUP = 128, 128, 32        # k = 4 blocks: the 4-D scale transpose is really non-contiguous
LAYERS, E, SLOTS, CHUNK, TOPK = 2, 12, 4, 3, 2
TOKENS, MIN_TOKENS, CAPTURE = 64, 16, [1, 2, 4, 8]
LAYER = 1                                   # not 0: a layer mix-up must be able to show

#: The resident sets the scenarios start from, and the set the routing makes hot.
#: HOT keeps expert 1 (resident, but its slot moves 1 -> 0), promotes 5, 7, 10,
#: demotes 0, 2, 3, and spans all four chunks [0-2] [3-5] [6-8] [9-11].
OLD = {0: (0, 4, 8, 11), 1: (0, 1, 2, 3)}
HOT = {0: (2, 4, 6, 9), 1: (1, 5, 7, 10)}

WEIGHT_PIECES = {
    "w13_weight": (("w1", "weight"), ("w3", "weight")),
    "w13_weight_scale": (("w1", "scale"), ("w3", "scale")),
    "w2_weight": (("w2", "weight"),),
    "w2_weight_scale": (("w2", "scale"),),
}
SYN_SHAPES = {
    ("w1", "weight"): (INTER, HIDDEN // 2),
    ("w3", "weight"): (INTER, HIDDEN // 2),
    ("w2", "weight"): (HIDDEN, INTER // 2),
    ("w1", "scale"): (INTER, HIDDEN // GROUP),
    ("w3", "scale"): (INTER, HIDDEN // GROUP),
    ("w2", "scale"): (HIDDEN, INTER // GROUP),
}


_DEVICE_GETTER = torch._C.TensorBase.__dict__["device"]
_WRITE_OK_ON_NZ = {"requires_grad_", "set_", "share_memory_"}


def _sid(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr()


def raw(t: torch.Tensor) -> torch.Tensor:
    """The storage's bytes, read with the platform model OFF: the oracle's privileged view."""
    with torch._C.DisableTorchFunction():
        return torch.empty(0, dtype=torch.uint8).set_(t.untyped_storage()).clone()


class Platform(TorchFunctionMode):
    """Device + internal-format (NZ) model, keyed on STORAGE so it survives ``.data =``
    and views. Follows the production layout: the cast returns a fresh tensor with a
    per-expert byte permutation, NZ storages refuse ``clone``/``equal`` and every
    in-place write, and ``copy_memory_`` overwrites the WHOLE destination storage.
    """

    FAKE_DEVICE = torch.device("meta")

    def __init__(self):
        super().__init__()
        self.device: set[int] = set()
        self.nz: set[int] = set()
        self._keep: list[torch.Tensor] = []     # registered storages must not be freed and reused
        self.casts: list[tuple] = []
        self.copies: list[int] = []
        self.copy_memory_hook = None

    # -- registration -----------------------------------------------------
    def on_device(self, t: torch.Tensor, nz: bool = False) -> torch.Tensor:
        with torch._C.DisableTorchFunction():
            self.device.add(_sid(t))
            if nz:
                self.nz.add(_sid(t))
        self._keep.append(t)
        return t

    def is_nz(self, t) -> bool:
        return isinstance(t, torch.Tensor) and _sid(t) in self.nz

    def is_dev(self, t) -> bool:
        return isinstance(t, torch.Tensor) and _sid(t) in self.device

    # -- the refusals ------------------------------------------------------
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = getattr(func, "__name__", "")
        if name == "__get__" and getattr(func, "__self__", None) is _DEVICE_GETTER:
            if self.is_dev(args[0]):
                return self.FAKE_DEVICE
            return func(*args, **kwargs)
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        if name == "clone" and tensors and self.is_nz(tensors[0]):
            raise RuntimeError("clone: do not support internal format")
        if name == "equal" and any(self.is_nz(t) for t in tensors):
            raise RuntimeError("equal: do not support internal format")          # modelled pessimistically
        if name == "__setitem__" and tensors and self.is_nz(tensors[0]):
            raise RuntimeError("setitem: internal format destination refused")
        if (name.endswith("_") and not name.startswith("__") and name not in _WRITE_OK_ON_NZ
                and tensors and self.is_nz(tensors[0])):
            raise RuntimeError(f"{name}: internal format destination refused")
        return func(*args, **kwargs)

    # -- the two card-only ops ---------------------------------------------
    def npu_format_cast(self, x, fmt, customize_dtype=None, input_dtype=None):
        if not self.is_dev(x):
            raise RuntimeError("npu_format_cast: host tensor")
        if self.is_nz(x):
            raise RuntimeError("npu_format_cast: already internal format")
        if not x.is_contiguous():
            raise RuntimeError("npu_format_cast: does not support non-contiguous input")
        self.casts.append((tuple(x.shape), int(fmt)))
        with torch._C.DisableTorchFunction():
            # Per-expert byte permutation: NZ bytes are NOT the ND bytes, and
            # each expert is cast independently.
            y = x.flip(-1).roll(1, dims=-2).contiguous()
        return self.on_device(y, nz=True)

    def copy_memory_(self, dst, src):
        if self.copy_memory_hook is not None:
            if self.copy_memory_hook(dst, src) == "skip":
                return
        if not self.is_dev(dst):
            raise RuntimeError("copy_memory_: host destination")
        if dst.storage_offset() != 0:
            raise RuntimeError("copy_memory_: storage_offset() == 0 assertion")
        with torch._C.DisableTorchFunction():
            d = torch.empty(0, dtype=torch.uint8).set_(dst.untyped_storage())
            s = torch.empty(0, dtype=torch.uint8).set_(src.untyped_storage())
            if d.numel() != s.numel():
                raise RuntimeError(f"copy_memory_: {s.numel()} B into {d.numel()} B")
            d.copy_(s)                    # the WHOLE storage, whatever view was passed
        self.copies.append(_sid(dst))


@pytest.fixture
def plat(monkeypatch):
    p = Platform()
    monkeypatch.setattr(torch_npu, "npu_format_cast", p.npu_format_cast, raising=False)
    monkeypatch.setattr(torch_npu, "copy_memory_", p.copy_memory_, raising=False)
    nt = SimpleNamespace(record=lambda *a, **k: None)
    monkeypatch.setattr(torch_npu, "npu_moe_init_routing_v2", _fake_init_routing_v2(nt), raising=False)
    monkeypatch.setattr(torch_npu, "npu_grouped_matmul", _fake_grouped_matmul(nt), raising=False)
    monkeypatch.setattr(torch_npu, "npu_moe_token_unpermute", _fake_unpermute(nt), raising=False)
    monkeypatch.setattr(torch.ops, "_C_ascend",
                        SimpleNamespace(npu_swiglu_group_quant=_fake_swiglu_group_quant(nt)), raising=False)
    with p:
        yield p


def test_platform_model_is_the_production_layout(plat, ckpt, tmp_path):
    """The model must refuse what the card refuses, or every arm below is weaker than it looks."""
    res = static_build(ckpt, LAYER, explicit_placement(tmp_path, LAYER, OLD[LAYER]), plat)
    w, s = res.w13_weight, res.w13_weight_scale
    assert w.device.type != "cpu" and s.device.type != "cpu"
    assert not w.is_contiguous() and not s.is_contiguous()      # transposed NZ / transposed 4-D view
    assert s.dim() == 4
    with pytest.raises(RuntimeError, match="internal format"):
        w.clone()
    with pytest.raises(RuntimeError, match="internal format"):
        w.data.copy_(w.data)
    with pytest.raises(RuntimeError, match="storage_offset"):
        torch_npu.copy_memory_(w.data[1:], w.data[:1])
    # copy_memory_ ignores the view: a 1-expert view rewrites every expert
    src = plat.on_device(torch.full_like(raw(w), 7))
    torch_npu.copy_memory_(w.data[:1], src)
    assert bool((raw(w) == 7).all())


def _bytes(layer, expert, proj, kind) -> torch.Tensor:
    g = torch.Generator().manual_seed(1_000 * layer + 10 * expert + 3 * ws.PROJS.index(proj) + ws.KINDS.index(kind))
    shape = SYN_SHAPES[(proj, kind)]
    if kind == "scale":
        return torch.randint(ws.E8M0_LO, ws.E8M0_HI + 1, shape, generator=g).to(torch.uint8)
    return torch.randint(1, 256, shape, generator=g).to(torch.uint8)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("fold_ckpt")
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    shards = {n: {} for n in names}
    wmap = {}
    for layer in range(LAYERS):
        for e in range(E):
            for proj in ws.PROJS:
                for kind in ws.KINDS:
                    k = ws.expert_key(layer, e, proj, kind)
                    fn = names[(layer + e) % 2]
                    b = _bytes(layer, e, proj, kind)
                    shards[fn][k] = b.view(torch.float8_e8m0fnu) if kind == "scale" else b.view(torch.int8)
                    wmap[k] = fn
    for fn, tensors in shards.items():
        save_file(tensors, str(d / fn))
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    return str(d)


def oracle_record(ckpt_dir: str, layer: int, expert: int, name: str) -> torch.Tensor:
    """``safe_open`` bytes of one expert's record in ``name`` (w13 = w1 then w3)."""
    with open(pathlib.Path(ckpt_dir) / "model.safetensors.index.json") as f:
        wmap = json.load(f)["weight_map"]
    out = []
    for proj, kind in WEIGHT_PIECES[name]:
        key = ws.expert_key(layer, expert, proj, kind)
        with safe_open(str(pathlib.Path(ckpt_dir) / wmap[key]), framework="pt") as h:
            out.append(h.get_tensor(key).contiguous().view(-1).view(torch.uint8).clone())
    return torch.cat(out)


def explicit_placement(tmp_path, layer: int, ids):
    """The production start-up path for ``resident_strategy="explicit"``."""
    sets = {str(i): list(OLD[i]) for i in range(LAYERS)}
    sets[str(layer)] = sorted(int(e) for e in ids)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"explicit_{layer}_{'_'.join(map(str, sets[str(layer)]))}.json"
    path.write_text(json.dumps({"layers": sets}))
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=1, resident_strategy="explicit",
                          resident_experts_path=str(path))
    return plan_from_config(cfg, LAYERS, E, TOPK).layer(layer)


def per_expert_shapes():
    qm = quant_method()
    shapes = dict(qm.get_weight(1, INTER, HIDDEN, torch.bfloat16))
    shapes.update(qm.get_dynamic_quant_param(1, INTER, HIDDEN, torch.bfloat16))
    return {n: tuple(shapes[n].shape[1:]) for n in SLOT_WEIGHT_ATTRS}


def static_build(ckpt_dir, layer, placement, plat) -> torch.nn.Module:
    """A resident layer as a server STARTED with ``placement`` would hold it: bytes
    from ``safe_open`` at ``slot_of(e)``, then the production post-load hook."""
    mod = torch.nn.Module()
    for name, shape in per_expert_shapes().items():
        rows = torch.zeros((placement.num_slots, *shape), dtype=torch.uint8)
        for e in placement.resident_ids:
            rows[placement.slot_of(e)] = oracle_record(ckpt_dir, layer, e, name).view(shape)
        mod.register_parameter(name, torch.nn.Parameter(plat.on_device(rows), requires_grad=False))
    quant_method().process_weights_after_loading(mod)
    return mod


def weight_violations(resident, expected) -> list[str]:
    bad = []
    for name in SLOT_WEIGHT_ATTRS:
        got, want = getattr(resident, name), getattr(expected, name)
        if tuple(got.shape) != tuple(want.shape) or got.stride() != want.stride():
            bad.append(f"{name}: layout {tuple(got.shape)}/{got.stride()} != {tuple(want.shape)}/{want.stride()}")
        rg, rw = raw(got), raw(want)
        if not torch.equal(rg, rw):
            bad.append(f"{name}: {int((rg != rw).sum())} of {rw.numel()} bytes differ from the static build")
    return bad


def table_violations(targets, placement, ledger) -> list[str]:
    bad = check_resident_state(placement, targets, ledger=ledger)
    for dt, t in targets.slot_tables.items():
        if not torch.equal(t, placement.safe_slot_table(dt)):
            bad.append(f"slot_table[{dt}] != the static explicit table")
    for dt, t in targets.keep_tables.items():
        if not torch.equal(t, placement.keep_table(dt)):
            bad.append(f"keep_table[{dt}] != the static explicit table")
    for key in ("device_mask", "kt_mask"):
        if not torch.equal(getattr(targets, key), placement.resident_mask()):
            bad.append(f"{key} != the static explicit mask")
    return bad


def routing(layer: int = LAYER, hot=None):
    """Routing whose unique top-SLOTS is ``hot``: 16 routes each, every other expert 8."""
    hot = tuple(HOT[layer] if hot is None else hot)
    cold = [e for e in range(E) if e not in hot]
    ids = torch.tensor([[hot[i % len(hot)], cold[i % len(cold)]] for i in range(TOKENS)], dtype=torch.int32)
    g = torch.Generator().manual_seed(layer)
    w = torch.rand(TOKENS, TOPK, generator=g, dtype=torch.float32) + 0.1
    x = torch.randn(TOKENS, HIDDEN, generator=g).to(torch.bfloat16)
    return x, ids, w


class Seam:
    def __init__(self, plat):
        self.plat = plat
        self.device_allocs = 0

    def alloc_device(self, shape, dtype, device):
        self.device_allocs += 1
        return self.plat.on_device(torch.zeros(tuple(shape), dtype=dtype))

    def seam(self):
        return DeviceSeam(alloc_pinned=lambda n: torch.zeros(int(n), dtype=torch.uint8),
                          alloc_device=self.alloc_device,
                          h2d=lambda dst, src, what: dst.copy_(src), wait=lambda h: None,
                          post_load=default_post_load)          # the PRODUCTION hook


def make_targets(placement, resident) -> LayerSwapTargets:
    return LayerSwapTargets(
        layer_idx=placement.layer_idx,
        weights={n: getattr(resident, n) for n in SLOT_WEIGHT_ATTRS},
        slot_tables={dt: placement.safe_slot_table(dt) for dt in (torch.int32, torch.int64)},
        keep_tables={dt: placement.keep_table(dt) for dt in (torch.float32, torch.bfloat16)},
        device_mask=placement.resident_mask(),
        kt_mask=placement.resident_mask(),
        kt_cpu_weight_ids=ALL_CPU_WEIGHTS,
    )


#: "no window given" is a real value here, so the default cannot be ``None``.
AUTO = object()


class World:
    """Resident layers built from OLD, a real provider on the synthetic checkpoint, a fold."""

    def __init__(self, ckpt_dir, plat, tmp_path, layers=(LAYER,), capture=False, **fold_kw):
        self.ckpt, self.plat, self.tmp = ckpt_dir, plat, tmp_path
        # What the runner publishes: by default a prompt that fits in ONE prefill
        # forward, fresh (a new request) per ``stream()`` call unless given a window.
        self._seq = 0
        self._req = 0
        self.hooks_given = []
        self.window = self.one_chunk_window()
        fold_kw.setdefault("window_source", lambda: self.window)
        # The same threshold the layer's streaming gate uses in this file
        # (``_runtime``): the fold's commit point is decided against it, so a World
        # whose fold believed another number would model a server that cannot exist.
        fold_kw.setdefault("stream_min_tokens", MIN_TOKENS)
        self.seam = Seam(plat)
        self.placement = {i: explicit_placement(tmp_path, i, OLD[i]) for i in layers}
        self.resident = {i: static_build(ckpt_dir, i, self.placement[i], plat) for i in layers}
        for i in layers:
            self.resident[i].quant_method = quant_method()
            self.resident[i].moe_config = moe_config()
        self.targets = {i: make_targets(self.placement[i], self.resident[i]) for i in layers}
        self.provider = StreamSlotProvider(
            WeightSource(ckpt_dir, num_experts=E, copy_threads=1), quant_method(),
            hidden_size=HIDDEN, intermediate_size=INTER, chunk_experts=CHUNK,
            device="cpu", seam=self.seam.seam(), copy_threads=1)
        self.fold = ResidentFold(self.placement, self.targets, provider=self.provider,
                                 step_source=lambda: 1,
                                 capture_predicate=(lambda: True) if capture else (lambda: False), **fold_kw)

    def one_chunk_window(self, key=None, tokens=TOKENS) -> PrefillWindow:
        """A whole prompt in one prefill forward: first chunk and last chunk at once."""
        self._seq += 1
        if key is None:
            self._req += 1
            key = f"req{self._req}"
        return PrefillWindow(key=key, first=True, last=True, computed=0, scheduled=tokens,
                             prompt_len=tokens, seq=self._seq)

    def chunked_windows(self, n, key="chunked", tokens=TOKENS) -> list[PrefillWindow]:
        """One prompt cut into ``n`` prefill forwards, built from the definition
        (``first`` = nothing computed yet, ``last`` = this chunk finishes it)."""
        out = []
        for i in range(n):
            self._seq += 1
            out.append(PrefillWindow(key=key, first=i == 0, last=i == n - 1, computed=i * tokens,
                                     scheduled=tokens, prompt_len=n * tokens, seq=self._seq))
        return out

    def stream(self, layer=LAYER, *, expert_ids=None, hook_wrap=None, fold=True, hot=None, window=AUTO,
               ids=None):
        """One streamed MoE layer, through the real chain, with or without the fold.

        ``window`` defaults to a fresh one-chunk prefill (a new request each call);
        pass ``None`` for a step the runner could not attribute to a request.
        """
        self.window = self.one_chunk_window() if window is AUTO else window
        x, routed, w = routing(layer, hot)
        if ids is not None:
            # A routing built by the caller, with activations and weights sized to it.
            g = torch.Generator().manual_seed(layer + 977)
            n = int(ids.shape[0])
            w = torch.rand(n, TOPK, generator=g, dtype=torch.float32) + 0.1
            x = torch.randn(n, HIDDEN, generator=g).to(torch.bfloat16)
        else:
            ids = routed
        plan = build_chunk_plan(E, CHUNK, expert_ids=expert_ids)
        expected = list(range(E)) if expert_ids is None else list(expert_ids)
        self.provider.set_layer(layer)

        def run(hook):
            # ``None`` is "stream exactly as with the switch off", which is what
            # every prefill chunk but the last must get.
            self.hooks_given.append(hook)
            if hook is not None and hook_wrap is not None:
                hook = hook_wrap(hook)
            kw = {} if hook is None else {"after_chunk": hook}
            return run_streamed_layer(quant_method=quant_method(), plan=plan, slot_for=self.provider.slot_for,
                                      hidden_states=x, topk_ids=ids, topk_weights=w, moe_config=moe_config(),
                                      top_k=TOPK, capture_sizes=CAPTURE, min_tokens=MIN_TOKENS,
                                      resident_layer=self.resident[layer], expected_expert_ids=expected,
                                      strict=True, **kw)

        if not fold:
            return run(None), None
        return self.fold.fold_streamed_layer(layer, ids, run)

    def expected(self, layer=LAYER, ids=None):
        p = explicit_placement(self.tmp, layer, HOT[layer] if ids is None else ids)
        return p, static_build(self.ckpt, layer, p, self.plat)

    def verdict(self, layer=LAYER, ids=None) -> list[str]:
        """Weights AND tables AND placement, against the static explicit build."""
        p, exp = self.expected(layer, ids)
        bad = weight_violations(self.resident[layer], exp)
        bad += table_violations(self.targets[layer], p, self.fold.committer.ledger)
        if self.fold.committer.placement(layer) != p:
            bad.append(f"committer placement {self.fold.committer.placement(layer).resident_ids} != {p.resident_ids}")
        return bad

    def snapshot(self, layer=LAYER):
        return ({n: raw(getattr(self.resident[layer], n)) for n in SLOT_WEIGHT_ATTRS},
                {k: t.detach().clone() for k, t in self.targets[layer].items() if not k.startswith("weights")})

    def unchanged(self, snap, layer=LAYER) -> bool:
        w, t = self.snapshot(layer)
        return all(torch.equal(w[n], snap[0][n]) for n in w) and all(torch.equal(t[k], snap[1][k]) for k in t)


@pytest.fixture
def world(ckpt, plat, tmp_path):
    return World(ckpt, plat, tmp_path)


def test_scenario_is_discriminating(world):
    old, hot = set(OLD[LAYER]), set(HOT[LAYER])
    assert hot & old and hot - old, "the chosen set must MIX kept residents and promoted offloaded experts"
    chunks = build_chunk_plan(E, CHUNK).chunks
    assert len({k for k, c in enumerate(chunks) for e in c if e in hot}) >= 2, "HOT must span several chunks"
    new_p, _ = world.expected()
    assert any(world.placement[LAYER].slot_of(e) != new_p.slot_of(e) for e in hot & old), \
        "a kept expert must change slot, or 'written in chunk order' could pass"
    # the start state is far from the expectation in every parameter
    bad = world.verdict()
    assert {n for n in SLOT_WEIGHT_ATTRS if any(b.startswith(n + ":") for b in bad)} == set(SLOT_WEIGHT_ATTRS)
    # the routing's top-SLOTS is HOT, by counts derived here (not by the fold's policy)
    _, ids, _ = routing()
    counts = torch.bincount(ids.reshape(-1).long(), minlength=E)
    assert set(torch.topk(counts, SLOTS).indices.tolist()) == hot
    assert int(counts[list(hot)].min()) > int(counts[[e for e in range(E) if e not in hot]].max())


#: A routing split so that NEITHER half votes for HOT and the WHOLE prompt does:
#: half A makes 1 and 5 hot, half B 7 and 10, the rest spread thinly; summed,
#: HOT[1] is the unique top-SLOTS.
def split_routing(tokens: int = 32):
    hot = tuple(HOT[LAYER])
    halves = []
    for lead in (hot[:2], hot[2:]):
        rest = [e for e in range(E) if e not in lead]
        halves.append(torch.tensor([[lead[i % len(lead)], rest[i % len(rest)]] for i in range(tokens)],
                                   dtype=torch.int32))
    return halves


def _counts(ids) -> torch.Tensor:
    return torch.bincount(ids.reshape(-1).long(), minlength=E)


def test_the_split_routing_is_discriminating():
    """Neither chunk alone chooses HOT; the two together choose it uniquely --
    otherwise "accumulated" and "decided by the last chunk" give the same answer."""
    a, b = split_routing()
    hot = set(HOT[LAYER])
    for half in (a, b):
        assert set(torch.topk(_counts(half), SLOTS).indices.tolist()) != hot
    total = _counts(a) + _counts(b)
    assert set(torch.topk(total, SLOTS).indices.tolist()) == hot
    assert int(total[list(hot)].min()) > int(total[[e for e in range(E) if e not in hot]].max())
    # the chunks are the SAME size, so nothing about their length says which is last
    assert a.shape == b.shape


def _prefill(world, ids_per_chunk, key="A", layer=LAYER):
    """Run one request's prefill: one streamed forward per chunk, in order."""
    windows = world.chunked_windows(len(ids_per_chunk), key=key, tokens=int(ids_per_chunk[0].shape[0]))
    out = []
    for ids, window in zip(ids_per_chunk, windows):
        out.append(world.stream(layer, ids=ids, window=window))
    return out


def test_heat_accumulates_over_the_prefill_chunks(world):
    """After K chunks the running heat is the sum of the K bincounts, counted here
    rather than read off the fold."""
    halves = split_routing()
    windows = world.chunked_windows(len(halves), key="A", tokens=int(halves[0].shape[0]))
    want = torch.zeros(E, dtype=torch.float64)
    for ids, window in zip(halves, windows):
        world.stream(LAYER, ids=ids, window=window)
        want += _counts(ids).to(torch.float64)
        got = world.fold.accumulated_heat(LAYER)
        assert got is not None and torch.equal(got, want), (got, want)
    assert world.fold.heat_request == "A"


def test_a_chunked_prefill_commits_once(world):
    """K chunks, ONE commit -- and it is the last chunk that carries it."""
    halves = split_routing()
    reports = [rep for _, rep in _prefill(world, halves)]
    cov = world.fold.coverage
    committed = [i for i, r in enumerate(reports) if r is not None]
    assert cov.layers_committed == 1, (
        f"one prefill of {len(halves)} chunks committed {cov.layers_committed} time(s), on chunk(s) "
        f"{committed}; each commit is a whole-layer cast plus a whole-tensor write")
    assert committed == [len(halves) - 1], "the commit must be on the LAST chunk"
    assert cov.layers_folded == 1
    assert cov.skipped == {"not_last_chunk": len(halves) - 1}
    # the denominator still adds up
    assert cov.decisions_seen == cov.layers_folded + sum(cov.skipped.values())
    assert cov.violations(expected_staged=[(LAYER, e) for e in HOT[LAYER]]) == []


def test_mut_deciding_on_every_chunk_commits_K_times(world, monkeypatch):
    """Variant arm for the one-commit criterion: take the "last chunk only" rule away."""
    monkeypatch.setattr(ResidentFold, "_decides_now", lambda self, window: True)
    reports = [rep for _, rep in _prefill(world, split_routing())]
    committed = world.fold.coverage.layers_committed
    assert committed == 2 and all(r is not None for r in reports), \
        f"the arm must expose the repeated commits; got {committed}"
    assert world.fold.coverage.skipped == {}


def test_a_chunk_that_is_not_the_last_changes_nothing(world):
    """Zero side effects on a non-final chunk, said with observables: staging bytes,
    staging generation, resident weights, routing tables and the hook handed on."""
    before = world.snapshot()
    staging_before = {n: raw(t) for n, t in world.fold.staging.items()}
    gen_before = world.fold._staging_generation
    # after_chunk must not be reachable at all: if the fold called it, this raises.
    world.fold.after_chunk = lambda *a, **k: pytest.fail("after_chunk ran on a non-final prefill chunk")

    halves = split_routing()
    windows = world.chunked_windows(2, key="A", tokens=int(halves[0].shape[0]))
    out, rep = world.stream(LAYER, ids=halves[0], window=windows[0])

    assert rep is None
    assert world.hooks_given == [None], "the streamed pass must be the unfolded one, hook and all"
    assert out.coverage.chunks_run == len(build_chunk_plan(E, CHUNK).chunks)   # it did stream
    assert world.unchanged(before), "a non-final chunk moved a resident byte or a table entry"
    assert all(torch.equal(raw(t), staging_before[n]) for n, t in world.fold.staging.items())
    assert world.fold._staging_generation == gen_before
    assert world.fold.coverage.skipped == {"not_last_chunk": 1} and world.fold.coverage.layers_committed == 0


def test_the_chunked_decision_is_the_decision_of_one_big_forward(ckpt, plat, tmp_path):
    """K chunks accumulated == the same tokens in a single forward. Both arms are
    judged against the same independent static build, so neither is the other's oracle."""
    halves = split_routing()
    chunked = World(ckpt, plat, tmp_path / "chunked")
    _prefill(chunked, halves)
    assert chunked.verdict() == []

    whole = World(ckpt, plat, tmp_path / "whole")
    _, rep = whole.stream(LAYER, ids=torch.cat(halves, dim=0))
    assert rep is not None and rep.swap.ok
    assert whole.verdict() == []

    # and said again on the two plans themselves
    assert tuple(rep.plan.new_ids) == tuple(sorted(HOT[LAYER]))


def test_mut_heat_of_the_last_chunk_only_is_red(ckpt, plat, tmp_path, monkeypatch):
    """Variant arm for the equivalence: overwrite the running heat instead of adding
    to it, which is the reading the last chunk alone would give."""
    monkeypatch.setattr(ResidentFold, "_accumulate", lambda self, layer, counts: counts)
    w = World(ckpt, plat, tmp_path)
    _, rep = _prefill(w, split_routing())[-1]
    assert rep is not None
    assert tuple(rep.plan.new_ids) != tuple(sorted(HOT[LAYER])), \
        "the arm must change the decision, or the criterion cannot see the accumulation"
    assert w.verdict() != []


def test_a_new_request_starts_the_heat_over(world):
    """Request B's decision must not carry request A's tokens. A is cut short
    (preempted), so only "this is a different request" can clear its row."""
    a_half, b_half = split_routing()
    # A: one chunk of a prompt that would have had two. No commit, heat left over.
    world.stream(LAYER, ids=a_half, window=world.chunked_windows(2, key="A")[0])
    assert world.fold.coverage.layers_committed == 0

    # B: a whole prompt in one forward, whose own routing is the second half.
    _, rep = world.stream(LAYER, ids=b_half,
                          window=PrefillWindow(key="B", first=True, last=True, computed=0,
                                               scheduled=int(b_half.shape[0]),
                                               prompt_len=int(b_half.shape[0]), seq=99))
    assert rep is not None
    want = set(torch.topk(_counts(b_half), SLOTS).indices.tolist())
    assert set(rep.plan.new_ids) == want, "B was decided from B's tokens alone"
    assert torch.equal(world.fold.accumulated_heat(LAYER), _counts(b_half).to(torch.float64))
    assert world.fold.heat_request == "B"


def test_mut_never_starting_the_heat_over_is_red(world, monkeypatch):
    """Variant arm: take the reset away and B is decided by A's tokens as well."""
    clean = set(torch.topk(_counts(split_routing()[1]), SLOTS).indices.tolist())
    monkeypatch.setattr(ResidentFold, "_note_window", lambda self, window: None)
    a_half, b_half = split_routing()
    world.stream(LAYER, ids=a_half, window=world.chunked_windows(2, key="A")[0])
    _, rep = world.stream(LAYER, ids=b_half,
                          window=PrefillWindow(key="B", first=True, last=True, computed=0,
                                               scheduled=int(b_half.shape[0]),
                                               prompt_len=int(b_half.shape[0]), seq=99))
    assert rep is not None and set(rep.plan.new_ids) != clean, \
        "the arm must let A's heat decide B, or the reset is not what makes B clean"
    assert set(rep.plan.new_ids) == set(HOT[LAYER])       # i.e. A+B, which is not B


def test_a_step_with_no_window_folds_nothing(world, monkeypatch):
    """No attributable prefill this step: stream as usual, fold nothing, say so once."""
    said = []
    monkeypatch.setattr(fold_swap_module.logger, "warning",
                        lambda fmt, *a: said.append(fmt % a if a else fmt))
    before = world.snapshot()
    staging_before = {n: raw(t) for n, t in world.fold.staging.items()}

    out, rep = world.stream(window=None)
    world.stream(window=None)                      # a second one must not keep shouting

    assert rep is None and world.hooks_given == [None, None]
    assert out.coverage.chunks_run == len(build_chunk_plan(E, CHUNK).chunks)
    assert world.fold.coverage.skipped == {"no_prefill_window": 2}
    assert world.fold.coverage.layers_committed == 0
    assert world.unchanged(before)
    assert all(torch.equal(raw(t), staging_before[n]) for n, t in world.fold.staging.items())
    assert len(said) == 1 and "no commit point" in said[0], said


def test_two_layers_accumulate_separately(ckpt, plat, tmp_path):
    """Each layer's running heat is its own routing, not the last layer's."""
    w = World(ckpt, plat, tmp_path, layers=(0, 1))
    halves = split_routing()
    windows = w.chunked_windows(2, key="A", tokens=int(halves[0].shape[0]))
    for window in windows:                       # both layers in each "forward"
        for layer in (0, 1):
            w.stream(layer, ids=halves[0] if layer == 0 else halves[1], window=window)
    want0 = 2 * _counts(halves[0]).to(torch.float64)
    want1 = 2 * _counts(halves[1]).to(torch.float64)
    assert torch.equal(w.fold.accumulated_heat(0), want0)
    assert torch.equal(w.fold.accumulated_heat(1), want1)
    assert not torch.equal(want0, want1), "the two layers must be distinguishable"


def test_the_same_window_twice_on_one_layer_is_refused(world):
    """A step whose window was never refreshed would double-count, and could commit twice."""
    window = world.chunked_windows(2, key="A")[0]
    world.stream(LAYER, ids=split_routing()[0], window=window)
    with pytest.raises(FoldError, match="already been decided under prefill window"):
        world.stream(LAYER, ids=split_routing()[0], window=window)


def test_the_runner_hook_is_not_optional(ckpt, plat, tmp_path):
    """A fold built with the production window source refuses if nothing publishes:
    falling back to "decide on every chunk" would look, in every counter, like a
    fold that works."""
    w = World(ckpt, plat, tmp_path, window_source=prefill_window.require)
    with pytest.raises(RuntimeError, match="nothing has published a prefill window"):
        w.stream()


def test_folded_resident_equals_the_static_explicit_build(world, plat):
    casts_before = len(plat.casts)
    _, rep = world.stream()
    assert rep is not None and rep.swap.ok and not rep.swap.recovered
    # one cast of the whole staging per layer, plus one per chunk for the GEMMs;
    # counted before the oracle, which casts for its own static build
    n_chunks = len(build_chunk_plan(E, CHUNK).chunks)
    whole = [c for c in plat.casts[casts_before:] if c[0][0] == SLOTS]
    assert len(plat.casts) - casts_before == 2 * n_chunks + 2 and len(whole) == 2
    assert world.verdict() == []
    # the committer could not read the NZ weights back, and says so instead of passing
    assert rep.weights_verified is False


def test_two_layers_share_one_staging_and_both_land(ckpt, plat, tmp_path):
    w = World(ckpt, plat, tmp_path, layers=(0, 1))
    allocs = w.seam.device_allocs
    w.stream(0)
    w.stream(1)
    assert w.seam.device_allocs == allocs, "the staging is allocated once, not per layer"
    assert w.verdict(0) == [] and w.verdict(1) == []


def test_a_parked_folded_layer_refuses_to_commit_other_layers_bytes(ckpt, plat, tmp_path):
    """The cast slot's SCALES are views of the staging, and the staging is shared, so
    committing a parked layer after another has staged would land the other layer's
    bytes with every guard green (nothing in the commit path can compare NZ). The
    arm asserts the REFUSAL, not the corruption.
    """
    w = World(ckpt, plat, tmp_path, layers=(0, 1))
    x, ids, weights = routing(1, None)
    plan = build_chunk_plan(E, CHUNK)
    w.provider.set_layer(1)
    assert w.fold.begin(1, ids) is not None
    run_streamed_layer(quant_method=quant_method(), plan=plan, slot_for=w.provider.slot_for,
                       hidden_states=x, topk_ids=ids, topk_weights=weights, moe_config=moe_config(),
                       top_k=TOPK, capture_sizes=CAPTURE, min_tokens=MIN_TOKENS,
                       resident_layer=w.resident[1], expected_expert_ids=list(range(E)),
                       strict=True, after_chunk=w.fold.after_chunk)
    parked = w.fold.finish()

    # ... and now layer 0 stages into the same buffers.
    x0, ids0, _ = routing(0, None)
    w.provider.set_layer(0)
    assert w.fold.begin(0, ids0) is not None
    chunk0 = tuple(build_chunk_plan(E, CHUNK).chunks[0])
    w.provider.slot_for(0, chunk0)          # the provider now holds layer 0's bytes
    w.fold.after_chunk(0, chunk0)

    with pytest.raises(FoldError, match="staging has been written"):
        w.fold.commit(parked)
    # The layer the parked slot belongs to was left exactly as it was.
    assert w.verdict(1, ids=OLD[1]) == []


def test_a_fold_that_cannot_move_anything_says_so(ckpt, plat, tmp_path, monkeypatch):
    """A kt subset load pins every resident expert, so the policy has zero free slots
    and the fold is a no-op. Reported as plain "no_change" it is indistinguishable
    from "the heat says nothing should move", so it needs its own reason.
    """
    w = World(ckpt, plat, tmp_path)
    # kt holds CPU weights only for the OFFLOADED experts: the subset load.
    tg = w.fold.committer.targets(LAYER)
    tg.kt_cpu_weight_ids = frozenset(set(range(E)) - set(OLD[LAYER]))

    said = []
    monkeypatch.setattr(fold_swap_module.logger, "warning",
                        lambda fmt, *a: said.append(fmt % a if a else fmt))

    _, rep = w.stream()
    _, rep2 = w.stream()          # a second forward must not keep shouting

    assert rep is None and rep2 is None
    # Its own reason, not folded into the benign "the heat said no change".
    assert w.fold.coverage.skipped == {"all_resident_pinned": 2}
    assert len(said) == 1 and "cannot move anything" in said[0], said
    assert w.verdict(LAYER, ids=OLD[LAYER]) == []              # and it changed nothing


def test_the_folded_output_is_the_unfolded_output(ckpt, plat, tmp_path):
    """The fold reads the chunks; it must not change what the chain computes."""
    out_off, _ = World(ckpt, plat, tmp_path / "a").stream(fold=False)
    out_on, rep = World(ckpt, plat, tmp_path / "b").stream()
    assert rep is not None
    assert torch.equal(out_on.output, out_off.output)


REVERSED = list(range(E))[::-1]      # chunks [11,10,9] [8,7,6] [5,4,3] [2,1,0]


def test_folded_resident_is_right_whatever_the_chunk_order(world):
    """Rows go to NEW-SLOT order, not arrival order: streamed back to front, still exact."""
    _, rep = world.stream(expert_ids=REVERSED)
    assert rep is not None and rep.staged == SLOTS
    assert world.verdict() == []


def test_mut_staging_rows_in_arrival_order_is_red(world, monkeypatch):
    """Rows written in arrival order, not new-slot order. Streamed ascending the two
    coincide, so this arm streams back to front, where the mistake shows."""
    real = fold_swap._stage_rows

    def arrival(staging, rows, src, cols):
        n = int(staging._fold_next) if hasattr(staging, "_fold_next") else 0
        moved = real(staging, torch.arange(n, n + rows.numel()), src, cols)
        staging._fold_next = n + rows.numel()
        return moved          # the same rows, the wrong ones: the byte count cannot tell

    monkeypatch.setattr(fold_swap, "_stage_rows", arrival)
    world.stream(expert_ids=REVERSED)
    assert any("differ" in b for b in world.verdict())


def test_mut_a_staging_write_that_does_nothing_is_red(world, monkeypatch):
    """A no-op copy that still REPORTS the bytes a real one would have moved:
    ``bytes_staged`` reconciles to the byte, the coverage gives full marks and the
    commit reports ok. Only the bit-for-bit criterion sees it.
    """
    monkeypatch.setattr(fold_swap, "_stage_rows",
                        lambda staging, rows, src, cols: int(rows.numel()) * fold_swap._row_bytes(staging))
    _, rep = world.stream()
    assert rep.swap.ok and rep.bytes_staged == SLOTS * expert_bytes(world.ckpt)
    assert world.fold.coverage.check({(LAYER, e) for e in HOT[LAYER]}) == []
    assert any("differ" in b for b in world.verdict()), "only the byte oracle can see this one"


def test_mut_uncast_staging_written_to_the_resident_is_red(world, monkeypatch):
    """ND bytes where NZ bytes belong: same size, plausible, wrong."""
    real = world.provider.format_nd

    def no_cast(tensors, n):
        slot = real(tensors, n)
        nd = {name: world.plat.on_device(t.detach().contiguous())
              for name, t in tensors.items()}
        return type(slot)(w13_weight=nd["w13_weight"].transpose(1, 2), w13_weight_scale=slot.w13_weight_scale,
                          w2_weight=nd["w2_weight"].transpose(1, 2), w2_weight_scale=slot.w2_weight_scale)

    monkeypatch.setattr(world.provider, "format_nd", no_cast)
    world.stream()
    assert any("w13_weight: " in b and "differ" in b for b in world.verdict())


def test_mut_provider_bound_to_another_layer_is_red(ckpt, plat, tmp_path):
    """The chunk bytes of layer 0 folded into layer 1."""
    w = World(ckpt, plat, tmp_path, layers=(0, 1))
    x, ids, wts = routing(LAYER)
    plan = build_chunk_plan(E, CHUNK)
    w.provider.set_layer(0)            # the mistake

    def run(hook):
        return run_streamed_layer(quant_method=quant_method(), plan=plan, slot_for=w.provider.slot_for,
                                  hidden_states=x, topk_ids=ids, topk_weights=wts, moe_config=moe_config(),
                                  top_k=TOPK, capture_sizes=CAPTURE, min_tokens=MIN_TOKENS,
                                  expected_expert_ids=list(range(E)), after_chunk=hook)

    with pytest.raises(FoldError, match="another layer"):
        w.fold.fold_streamed_layer(LAYER, ids, run)
    assert w.verdict(LAYER, OLD[LAYER]) == []          # nothing landed


def test_mut_staging_cleared_between_chunks_is_red(world):
    def clear_after_chunk_1(hook):
        def h(k, ids):
            hook(k, ids)
            if k == 1:
                for t in world.fold.staging.values():
                    t.zero_()
        return h

    world.stream(hook_wrap=clear_after_chunk_1)
    bad = world.verdict()
    assert any("differ" in b for b in bad), "rows staged in chunks 0-1 were wiped and nobody noticed"


def test_mut_staging_reallocated_between_chunks_is_red(world):
    def realloc_after_chunk_1(hook):
        def h(k, ids):
            hook(k, ids)
            if k == 1:
                for name, t in list(world.fold._staging.items()):
                    world.fold._staging[name] = world.seam.alloc_device(tuple(t.shape), t.dtype, "cpu")
        return h

    with pytest.raises(FoldError, match="reallocated"):
        world.stream(hook_wrap=realloc_after_chunk_1)
    assert world.fold.layer_in_flight() is False


def test_mut_staging_reallocated_without_the_address_guard_is_still_red(world, monkeypatch):
    """Blinding ``_staging_moved`` removes every address guard, leaving only the
    static-explicit byte oracle -- which must still catch it."""
    monkeypatch.setattr(ResidentFold, "_staging_moved", lambda self, expected: [])

    def realloc_after_chunk_1(hook):
        def h(k, ids):
            hook(k, ids)
            if k == 1:
                for name, t in list(world.fold._staging.items()):
                    world.fold._staging[name] = world.seam.alloc_device(tuple(t.shape), t.dtype, "cpu")
        return h

    world.stream(hook_wrap=realloc_after_chunk_1)
    assert any("differ" in b for b in world.verdict())


def test_coverage_is_the_chosen_set(world):
    _, rep = world.stream()
    hot, old = set(HOT[LAYER]), set(OLD[LAYER])
    assert rep.staged == len(hot) == SLOTS
    assert world.fold.coverage.check({(LAYER, e) for e in hot}) == []
    assert world.fold.coverage.pairs_moved == {(LAYER, e) for e in hot ^ old}
    assert set(rep.swap.promoted) == hot - old and set(rep.swap.demoted) == old - hot


def test_zero_coverage_is_a_failure(world):
    """Routing whose hot set IS the resident set: nothing folds, and the run must not pass."""
    _, rep = world.stream(hot=OLD[LAYER])
    assert rep is None and world.fold.coverage.skipped == {"no_change": 1}
    with pytest.raises(ZeroCoverageError):
        world.fold.coverage.check({(LAYER, e) for e in HOT[LAYER]})
    with pytest.raises(ValueError, match="must not come from the fold"):
        world.fold.coverage.check()


def test_mut_coverage_short_of_the_chosen_set_is_red(world, monkeypatch):
    """A fold that stages one expert fewer than chosen, judged by the chosen set, not by itself."""
    real = ResidentFold.after_chunk

    def drop_last(self, k, ids):
        real(self, k, ids)
        if self._active and self._active.staged and k == 3:
            self._active.staged.pop()

    monkeypatch.setattr(ResidentFold, "after_chunk", drop_last)
    world.stream()
    with pytest.raises(ZeroCoverageError, match="missing"):
        world.fold.coverage.check({(LAYER, e) for e in HOT[LAYER]})


# ``bytes_staged`` is a DENOMINATOR check: the copies say how much moved, not the
# bookkeeping. It cannot see wrong CONTENT -- see
# ``test_mut_a_staging_write_that_does_nothing_is_red``, green here on purpose.


def expert_bytes(ckpt_dir: str, layer: int = LAYER, expert: int = 0) -> int:
    """Bytes one expert occupies across the four parameters, read from the CHECKPOINT
    with ``safe_open`` -- not from the fold, the provider or the staging."""
    return sum(int(oracle_record(ckpt_dir, layer, expert, n).numel()) for n in SLOT_WEIGHT_ATTRS)


def test_bytes_staged_is_the_chosen_set_times_one_experts_bytes(world):
    _, rep = world.stream()
    want = SLOTS * expert_bytes(world.ckpt)
    assert rep.bytes_staged == want and world.fold.coverage.bytes_staged == want
    assert f"{want} B copied" in world.fold.coverage.summary()
    # and one expert really is that many bytes in the staging the fold allocated
    assert world.fold._expert_bytes == expert_bytes(world.ckpt)


def test_mut_counting_bytes_the_copy_did_not_move_is_red(world, monkeypatch):
    """The count decoupled from the copy that actually ran: the copy happens exactly
    as before, but the number it reports is no longer its own."""
    real = fold_swap._stage_rows
    monkeypatch.setattr(fold_swap, "_stage_rows", lambda *a: (real(*a), 0)[1])
    before = world.snapshot()
    with pytest.raises(FoldError, match="have come apart"):
        world.stream()
    assert world.unchanged(before), "nothing may be committed once the accounting is broken"


def test_mut_a_chunk_whose_copy_moved_fewer_rows_is_red(world, monkeypatch):
    """Half the rows copied, all of them marked filled: the byte count is the only witness."""
    real = fold_swap._stage_rows

    def first_half(staging, rows, src, cols):
        real(staging, rows, src, cols)
        return int(rows.numel()) * fold_swap._row_bytes(staging) // 2

    monkeypatch.setattr(fold_swap, "_stage_rows", first_half)
    with pytest.raises(FoldError, match=r"moved \d+ B"):
        world.stream()


def test_chosen_expert_not_streamed_raises(world):
    """Stream only the offloaded experts: kept resident 1 never comes by."""
    offloaded = [e for e in range(E) if e not in OLD[LAYER]]
    before = world.snapshot()
    with pytest.raises(FoldIncompleteError, match=r"\[1\]"):
        world.stream(expert_ids=offloaded)
    assert world.unchanged(before)
    assert world.fold.committer.placement(LAYER) == world.placement[LAYER]
    assert world.fold.coverage.skipped == {"incomplete": 1} and not world.fold.layer_in_flight()


def test_a_chunk_that_never_reached_the_hook_raises(world):
    before = world.snapshot()

    def skip_chunk_2(hook):
        return lambda k, ids: None if k == 2 else hook(k, ids)

    with pytest.raises(FoldIncompleteError, match=r"\[7\]"):
        world.stream(hook_wrap=skip_chunk_2)
    assert world.unchanged(before)


def denominator_violations(fold) -> list[str]:
    """``decisions_seen == layers_folded + sum(skipped)``: if a decision can be
    counted and then vanish, no other coverage reading says what happened."""
    return [b for b in fold.coverage.violations() if b.startswith("denominator")]


def _bad_policy(*, layer, placement, heat, pinned):
    """A proposal the driver's own ``check_proposal`` refuses: one expert too few."""
    return sorted(HOT[layer])[:-1]


def _begin_paths(world):
    """Every way ``begin`` can raise AFTER it has counted the decision."""
    _, ids, _ = routing()

    def unknown_layer():
        world.fold.begin(LAYERS + 7, ids)

    def id_out_of_range():
        world.fold.begin(LAYER, torch.full((4, TOPK), E + 1, dtype=torch.int32))

    def same_window_twice():
        world.window = world.one_chunk_window()
        _, cold, _ = routing(LAYER, hot=OLD[LAYER])
        world.fold.begin(LAYER, cold)          # counted, skipped "no_change", _active stays None
        world.fold.begin(LAYER, cold)          # the runner published no new window

    return {"unknown_layer": unknown_layer, "id_out_of_range": id_out_of_range,
            "same_window_twice": same_window_twice}


@pytest.mark.parametrize("path", ["unknown_layer", "id_out_of_range", "same_window_twice"])
def test_the_denominator_survives_every_raising_path_of_begin(world, path):
    with pytest.raises(FoldError):
        _begin_paths(world)[path]()
    assert denominator_violations(world.fold) == [], world.fold.coverage.summary()
    assert any(k.startswith("raised_") for k in world.fold.coverage.skipped), world.fold.coverage.skipped


def test_the_denominator_survives_a_policy_refusal(ckpt, plat, tmp_path):
    w = World(ckpt, plat, tmp_path, policy=_bad_policy)
    with pytest.raises(PolicyError):
        w.stream()
    assert denominator_violations(w.fold) == []
    assert w.fold.coverage.skipped == {"raised_PolicyError": 1}


def label_violations(fold, want: dict[str, int]) -> list[str]:
    """The skip labels must be exactly ``want``: a failure filed under another reason is a lie."""
    got = dict(fold.coverage.skipped)
    return [] if got == want else [f"skip labels {got} != {want}"]


def _failing_cast(world):
    """Break the ONE cast ``finish`` does -- a per-chunk cast that raises aborts the
    streamed layer, which is a different and separately labelled path."""
    real = world.provider.format_nd

    def boom(tensors, n):
        if tensors is world.fold._staging:
            raise RuntimeError("npu_format_cast: out of memory")
        return real(tensors, n)

    world.provider.format_nd = boom


def test_a_failure_that_is_not_a_missing_expert_is_not_called_incomplete(world):
    """``incomplete`` means "a chosen expert never came by". A cast that blew up is not that."""
    _failing_cast(world)
    before = world.snapshot()
    with pytest.raises(RuntimeError, match="out of memory"):
        world.stream()
    assert label_violations(world.fold, {"raised_RuntimeError": 1}) == []
    assert denominator_violations(world.fold) == [] and world.unchanged(before)


def test_a_missing_expert_is_still_called_incomplete(world):
    """The other half of the same criterion: the label must not drift the other way either."""
    offloaded = [e for e in range(E) if e not in OLD[LAYER]]
    with pytest.raises(FoldIncompleteError):
        world.stream(expert_ids=offloaded)
    assert label_violations(world.fold, {"incomplete": 1}) == []
    assert denominator_violations(world.fold) == []


def test_mut_counting_a_decision_and_recording_no_skip_is_red(world, monkeypatch):
    """``begin`` counted the decision, then raised, and recorded no skip."""
    real = ResidentFold._skip
    monkeypatch.setattr(ResidentFold, "_skip",
                        lambda self, why: None if why.startswith("raised_") else real(self, why))
    with pytest.raises(FoldError):
        _begin_paths(world)["unknown_layer"]()
    assert any("denominator" in b for b in denominator_violations(world.fold))


def test_mut_filing_every_failure_under_incomplete_is_red(world, monkeypatch):
    """The other half: ``finish``'s ``finally`` says "incomplete" whatever failed. The
    denominator stays whole here -- it was never what tells the two failures apart."""
    real = ResidentFold._skip
    monkeypatch.setattr(ResidentFold, "_skip",
                        lambda self, why: real(self, "incomplete" if why.startswith("raised_") else why))
    _failing_cast(world)
    with pytest.raises(RuntimeError):
        world.stream()
    assert denominator_violations(world.fold) == []
    assert label_violations(world.fold, {"raised_RuntimeError": 1}) != [], \
        "a cast failure filed as 'a chosen expert never came by' must be visible"


def _vcfg():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=HIDDEN, moe_intermediate_size=INTER,
                                           num_experts_per_tok=TOPK, num_hidden_layers=LAYERS),
            model="/m", enforce_eager=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"),
                                           cudagraph_capture_sizes=CAPTURE),
    )


def _runtime(world, monkeypatch, *, dynamic: bool, install: bool = True, wire: bool = False):
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=SLOTS, stream_prefill=True,
                          stream_prefill_min_tokens=MIN_TOKENS, stream_prefill_chunk_experts=CHUNK,
                          dynamic_resident=dynamic)
    rt = kl.KTLayerOffload(cfg, world.placement[LAYER], LAYER, _vcfg())
    rt.capture_sizes = CAPTURE
    rt._stream_slot_provider = world.provider.slot_for
    if wire:
        # What install_resident_fold installs: the provider OBJECT as well, so the
        # layer binds it per forward (set_layer).
        rt._stream_provider = world.provider
    if install:
        rt._resident_fold = world.fold
    monkeypatch.setattr(kl, "_capturing", lambda: False)
    world.provider.set_layer(LAYER)
    return rt


def _forward(world, rt):
    x, ids, w = routing()
    return rt.maybe_stream_prefill(world.resident[LAYER], x, w, ids, in_profile_run=False)


def test_switch_off_is_bit_identical_even_with_a_fold_installed(ckpt, plat, tmp_path, monkeypatch):
    base = World(ckpt, plat, tmp_path / "base")
    out_base = _forward(base, _runtime(base, monkeypatch, dynamic=False, install=False))

    w = World(ckpt, plat, tmp_path / "off")
    before = w.snapshot()
    rt = _runtime(w, monkeypatch, dynamic=False)
    out = _forward(w, rt)
    assert torch.equal(out, out_base)
    assert w.unchanged(before), "switch off moved a resident byte or a table entry"
    assert w.fold.coverage.decisions_seen == 0 and rt.fold_report is None
    assert rt.placement is w.placement[LAYER]


def test_switch_off_is_bit_identical_with_the_provider_wired_too(ckpt, plat, tmp_path, monkeypatch):
    """The same comparison with the provider OBJECT installed: binding it per forward
    must not change a byte, or ``dynamic_resident=false`` stops being the old path."""
    base = World(ckpt, plat, tmp_path / "base")
    out_base = _forward(base, _runtime(base, monkeypatch, dynamic=False, install=False))

    w = World(ckpt, plat, tmp_path / "wired")
    before = w.snapshot()
    rt = _runtime(w, monkeypatch, dynamic=False, wire=True)
    out = _forward(w, rt)
    assert torch.equal(out, out_base)
    assert w.unchanged(before)
    assert w.fold.coverage.decisions_seen == 0 and rt.fold_report is None
    assert w.provider.layer == LAYER          # the layer did the binding itself


def test_switch_off_criterion_sees_a_swap_when_there_is_one(ckpt, plat, tmp_path, monkeypatch):
    """Positive control: the same comparison, switch ON, must see the change."""
    w = World(ckpt, plat, tmp_path)
    before = w.snapshot()
    rt = _runtime(w, monkeypatch, dynamic=True)
    _forward(w, rt)
    assert not w.unchanged(before)
    assert rt.fold_report is not None and rt.fold_report.swap.ok
    assert rt.placement.resident_ids == tuple(sorted(HOT[LAYER]))
    assert w.verdict() == []


def test_switch_on_without_a_fold_refuses(world, monkeypatch):
    rt = _runtime(world, monkeypatch, dynamic=True, install=False)
    with pytest.raises(RuntimeError, match="no resident fold is installed"):
        _forward(world, rt)


def test_run_streamed_layer_without_the_hook_is_unchanged(world):
    """``after_chunk=None`` is the loop as it was: same output as not passing it."""
    x, ids, w = routing()
    plan = build_chunk_plan(E, CHUNK)
    kw = dict(quant_method=quant_method(), plan=plan, slot_for=world.provider.slot_for, hidden_states=x,
              topk_ids=ids, topk_weights=w, moe_config=moe_config(), top_k=TOPK, capture_sizes=CAPTURE,
              min_tokens=MIN_TOKENS, expected_expert_ids=list(range(E)))
    world.provider.set_layer(LAYER)
    a = run_streamed_layer(**kw)
    b = run_streamed_layer(**kw, after_chunk=None)
    assert torch.equal(a.output, b.output)


def _fake_kt(monkeypatch, *, fullset: bool):
    """Make ``from kt_kernel.utils import amx`` answer a known way -- otherwise the
    refusal is "kt_kernel is not importable", which is not the one under test."""
    import sys
    import types
    amx = types.ModuleType("kt_kernel.utils.amx")
    amx._fullset_load_requested = lambda: fullset
    utils = types.ModuleType("kt_kernel.utils")
    utils.amx = amx
    pkg = types.ModuleType("kt_kernel")
    pkg.utils = utils
    monkeypatch.setitem(sys.modules, "kt_kernel", pkg)
    monkeypatch.setitem(sys.modules, "kt_kernel.utils", utils)
    monkeypatch.setitem(sys.modules, "kt_kernel.utils.amx", amx)


def test_config_dynamic_resident_needs_stream_prefill_and_is_still_refused():
    vc = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1,
                                                         pipeline_parallel_size=1, enable_eplb=False),
                         scheduler_config=SimpleNamespace(max_num_seqs=1),
                         speculative_config=None)
    with pytest.raises(ValueError, match="needs stream_prefill"):
        KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=4,
                        dynamic_resident=True)._validate_preconditions(vc)
    assert KTOffloadConfig().dynamic_resident is False


def test_dynamic_resident_is_refused_unless_kt_holds_every_expert(monkeypatch):
    """The question is asked of the kt this process imported, not of the build: on a
    kt with no ``KT_FULLSET_LOAD`` at all, exporting it is silently ignored, the
    fold commits nothing, and the run reads as "the heat said no change".
    """
    vc = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1,
                                                         pipeline_parallel_size=1, enable_eplb=False),
                         scheduler_config=SimpleNamespace(max_num_seqs=1),
                         speculative_config=None)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=4,
                          overlap_side_stream=False, stream_prefill=True, dynamic_resident=True)
    _fake_kt(monkeypatch, fullset=False)
    with pytest.raises(ValueError, match="KT_FULLSET_LOAD=1"):
        cfg._validate_preconditions(vc)
    # positive control: the identical config passes once kt reports the full set
    _fake_kt(monkeypatch, fullset=True)
    cfg._validate_preconditions(vc)


def _dyn_cfg(**kw):
    return KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=4, stream_prefill=True,
                           dynamic_resident=True, **kw)


def _vc(max_num_seqs):
    return SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1,
                                                           pipeline_parallel_size=1, enable_eplb=False),
                           scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
                           speculative_config=None)


@pytest.mark.parametrize("seqs", [2, 8, 256])
def test_dynamic_resident_is_refused_above_one_sequence(seqs):
    """The bs=1 gate: the refusal must say WHY, and the why is not 'unsafe'."""
    with pytest.raises(ValueError, match=f"requires max_num_seqs=1; this server is configured for {seqs}") as e:
        _dyn_cfg()._validate_preconditions(_vc(seqs))
    text = str(e.value)
    assert "Nothing is computed incorrectly at a larger batch" in text
    assert "unsafe" not in text.lower()


def test_one_sequence_passes_the_same_gate(monkeypatch):
    """Positive control: the identical config at max_num_seqs=1 gets past this check.
    It is still refused afterwards, but by another entry that says nothing about
    the batch size."""
    cfg = _dyn_cfg()
    cfg._refuse_batched_dynamic_resident(_vc(1))          # the gate itself: no exception
    _fake_kt(monkeypatch, fullset=False)
    with pytest.raises(ValueError) as e:
        cfg._validate_preconditions(_vc(1))
    assert "KT_FULLSET_LOAD" in str(e.value)
    assert "max_num_seqs" not in str(e.value)


def test_the_gate_is_only_about_dynamic_resident():
    """stream_prefill on its own has no batch-size limit; the gate must not invent one."""
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=4,
                          overlap_side_stream=False, stream_prefill=True)
    cfg._refuse_batched_dynamic_resident(_vc(8))          # no exception
    cfg._validate_preconditions(_vc(8))                   # and nothing else refuses it either
    # Discrimination: the same batch size WITH dynamic_resident must still be refused,
    # and for the batch reason -- otherwise a deleted gate would pass this too.
    dyn = KTOffloadConfig(enabled=True, cpuinfer_threads=1, num_resident_experts=4,
                          overlap_side_stream=False, stream_prefill=True, dynamic_resident=True)
    with pytest.raises(ValueError, match="max_num_seqs"):
        dyn._validate_preconditions(_vc(8))


def test_the_gate_refuses_a_config_it_cannot_read():
    """No scheduler_config = no answer. Assuming bs=1 would be the silent failure."""
    vc = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1, data_parallel_size=1,
                                                         pipeline_parallel_size=1, enable_eplb=False),
                         speculative_config=None)
    with pytest.raises(ValueError, match="has no scheduler_config"):
        _dyn_cfg()._refuse_batched_dynamic_resident(vc)


@pytest.mark.parametrize("cap, why", [(True, "captured"), (None, "capture_state_unknown")])
def test_no_decision_unless_provably_eager(ckpt, plat, tmp_path, cap, why):
    w = World(ckpt, plat, tmp_path)
    w.fold.capture_predicate = lambda: cap
    before = w.snapshot()
    staging_before = {n: raw(t) for n, t in w.fold.staging.items()}
    out, rep = w.stream()
    assert rep is None and w.fold.coverage.skipped == {why: 1}
    assert out.coverage.chunks_run == len(build_chunk_plan(E, CHUNK).chunks)   # streamed, just not folded
    assert w.unchanged(before)
    assert all(torch.equal(raw(t), staging_before[n]) for n, t in w.fold.staging.items())


def test_commit_while_the_layer_is_in_flight_raises(world):
    """The committer's ``writer_busy`` guard is live: what it refuses is a commit
    while THIS layer is still folding, not "a forward is in flight"."""
    seen = {}

    def commit_mid_layer(hook):
        def h(k, ids):
            hook(k, ids)
            if k == 1:
                p = world.fold.committer.placement(LAYER)
                with pytest.raises(OwnershipConflictError, match="writer is busy"):
                    world.fold.committer.commit(1, LAYER, HOT[LAYER], {n: None for n in SLOT_WEIGHT_ATTRS})
                seen["refused"] = p
        return h

    world.stream(hook_wrap=commit_mid_layer)
    assert "refused" in seen
    assert world.verdict() == []


def test_the_guard_is_false_at_the_commit_point_and_is_not_about_forwards(world):
    """At the moment the fold commits the predicate is False -- and a forward IS
    running, because the commit point is the layer boundary INSIDE the forward. All
    it buys is that the commit cannot be issued from the layer still filling the
    staging; a second writer is the ownership ledger's job, asserted separately below.
    """
    seen = {}
    real = world.fold.committer.commit

    def watch(step, layer, ids, weights, **kw):
        seen["guard_at_commit"] = world.fold.committer.writer_busy()
        seen["layer_in_flight"] = world.fold.layer_in_flight()
        return real(step, layer, ids, weights, **kw)

    world.fold.committer.commit = watch
    _, rep = world.stream()
    assert rep.swap.ok
    assert seen == {"guard_at_commit": False, "layer_in_flight": False}
    assert world.fold.committer.writer_busy == world.fold.layer_in_flight
    # What DOES stand between the swap and a second writer: the ledger's identity.
    led, me = world.fold.committer.ledger, threading.get_ident()
    assert led.active is None
    led.claim(WriterStamp(9, LAYER, me, None))
    with pytest.raises(OwnershipConflictError, match="overlaps commit in progress"):
        led.claim(WriterStamp(9, LAYER, me, None))
    led.release(WriterStamp(9, LAYER, me, None))
    with pytest.raises(OwnershipConflictError, match="second writer"):
        led.claim(WriterStamp(9, LAYER, me + 1, None))      # another thread


def _fail_first_weight_write(plat, after_failure: str):
    """copy_memory_ half-writes and raises on its first call; afterwards it works, or is silently dead."""
    state = {"n": 0}

    def hook(dst, src):
        state["n"] += 1
        if state["n"] == 1:
            with torch._C.DisableTorchFunction():
                d = torch.empty(0, dtype=torch.uint8).set_(dst.untyped_storage())
                s = torch.empty(0, dtype=torch.uint8).set_(src.untyped_storage())
                d[: d.numel() // 2].copy_(s[: s.numel() // 2])
            raise RuntimeError("injected: copy_memory_ failed half way")
        return "skip" if after_failure == "dead" else None

    return hook, state


def test_recovery_lands_tables_and_weights(world, plat):
    plat.copy_memory_hook, state = _fail_first_weight_write(plat, "works")
    _, rep = world.stream()
    assert rep.swap.ok and rep.swap.recovered and "injected" in rep.swap.error
    assert state["n"] > len(SLOT_WEIGHT_ATTRS)          # the roll-forward really rewrote
    assert world.verdict() == [], "recovery must land BOTH the tables and the weights"


def test_mut_recovery_judged_on_tables_alone_misses_torn_weights(world, plat):
    """The roll-forward's weight channel goes silently dead after the fault: the
    committer cannot read NZ weights back, so it reports committed-and-recovered
    with perfect tables. A tables-only recovery check is GREEN here; ours is red."""
    plat.copy_memory_hook, _ = _fail_first_weight_write(plat, "dead")
    _, rep = world.stream()
    assert rep.swap.ok and rep.swap.recovered and rep.weights_verified is False
    p, exp = world.expected()
    assert table_violations(world.targets[LAYER], p, world.fold.committer.ledger) == []   # tables-only: green
    assert weight_violations(world.resident[LAYER], exp) != []                           # weights: red
    assert world.verdict() != []


# The only write into a resident parameter is a whole-tensor ``copy_memory_``,
# enforced by a WHITELIST: the rules are stated over the SHAPE of a write, because
# a blacklist of names is trivially bypassed (``dst.data[:] = src``,
# ``copy_memory_`` bound to another name, ``getattr(dst, "copy_")(src)``, an
# unlisted in-place method, ``setattr(dst, "data", src)`` -- all of which compile
# and land bytes).
#
#   * a method call whose name ends in ``_`` (Torch's in-place convention) is a
#     write, unless the (method, enclosing function) pair is declared below;
#   * a call with an ``out=`` keyword writes into that argument;
#   * ``x.data = ...`` rebinds, ``x.data[...] = ...`` writes through it;
#   * any other subscript assignment is a write unless its base is one of the
#     plain Python containers this module keeps;
#   * an in-place method referenced WITHOUT calling it is an alias about to be
#     called under another name;
#   * ``getattr`` / ``setattr`` hide the name from every rule above, so both are
#     refused outside the one attribute read this module is allowed.
#
# A write issued from another module or inside a C extension is invisible to the
# scan and does NOT rest on it: at run time the platform refuses every in-place
# write into an NZ tensor, and the committer re-reads every target's
# ``data_ptr()`` after every write.

#: (method, enclosing function) the module may call in place, and why it is safe.
_INPLACE_OK = {
    ("index_copy_", "_stage_rows"),          # the ND staging write; ND on both sides, never a resident parameter
    ("copy_memory_", "fold_weight_write"),   # the one device channel into a resident parameter
    ("add_", "_accumulate"),                 # the host float64 heat row, not a device tensor
}
#: Bases whose subscript assignment is a plain Python container store, not a tensor write.
_SUBSCRIPT_OK = {"self._staging", "self._heat", "self._decided_at", "self.coverage.skipped", "a.filled"}
#: The single dynamic attribute access allowed: the provider capability probe, a read.
_GETATTR_OK = {("__init__", "getattr(provider, need, None)")}
#: Writes whose name does not end in ``_`` (the ``_`` rule covers every other one).
_NON_UNDERSCORE_WRITES = {"narrow_copy"}


def _assign_violations(node, target) -> list[str]:
    ln = f"line {node.lineno}"
    if isinstance(target, ast.Attribute) and target.attr == "data":
        return [f"{ln}: '.data =' rebinds instead of writing in place"]
    if isinstance(target, ast.Subscript):
        base = ast.unparse(target.value)
        if base.endswith(".data"):
            return [f"{ln}: '{base}[...] =' is a subscript write through .data"]
        if base not in _SUBSCRIPT_OK:
            return [f"{ln}: subscript assignment to {base!r} is not one of the declared containers "
                    f"{sorted(_SUBSCRIPT_OK)}; on a tensor it would write the storage in place"]
    if isinstance(node, ast.AugAssign) and isinstance(target, ast.Name):
        return [f"{ln}: '{ast.unparse(target)} {type(node.op).__name__}=' is an in-place operator on a bare "
                "name, which on a tensor writes its storage"]
    return []


def _call_violations(node, owner) -> list[str]:
    ln, bad = f"line {node.lineno}", []
    if any(kw.arg == "out" for kw in node.keywords):
        bad.append(f"{ln}: out= writes into the tensor it is handed: {ast.unparse(node)[:80]}")
    if isinstance(node.func, ast.Name) and node.func.id in ("getattr", "setattr"):
        if node.func.id == "setattr" or (owner.get(id(node)), ast.unparse(node)) not in _GETATTR_OK:
            bad.append(f"{ln}: {node.func.id} hides the attribute name from this scan: {ast.unparse(node)[:80]}")
    if isinstance(node.func, ast.Attribute):
        attr = node.func.attr
        in_place = (attr.endswith("_") and not attr.startswith("_")) or attr in _NON_UNDERSCORE_WRITES
        if in_place and (attr, owner.get(id(node))) not in _INPLACE_OK:
            bad.append(f"{ln}: .{attr}() is an in-place write, and ({attr}, {owner.get(id(node))}) is not one "
                       "of the declared channels")
    return bad


def write_channel_violations(src: str) -> list[str]:
    """Scan ``fold_swap.py`` for every write it can see and allow only the declared
    ones; see the note above for what it does and does not cover."""
    tree = ast.parse(src)
    bad: list[str] = []
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    owner = {}
    for f in funcs.values():
        for node in ast.walk(f):
            owner.setdefault(id(node), f.name)
    called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}

    cm_calls = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                bad += _assign_violations(node, t)
        if isinstance(node, ast.Attribute) and id(node) not in called:
            if node.attr.endswith("_") and not node.attr.startswith("_"):
                bad.append(f"line {node.lineno}: '{ast.unparse(node)}' binds an in-place write under another "
                           "name instead of calling it here")
        if isinstance(node, ast.Call):
            bad += _call_violations(node, owner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "index_copy_":
                recv = node.func.value
                if owner.get(id(node)) != "_stage_rows" or not (isinstance(recv, ast.Name) and recv.id == "staging"):
                    bad.append(f"line {node.lineno}: index_copy_ outside the ND staging write")
            if attr == "copy_memory_":
                cm_calls.append(node)
    if len(cm_calls) != 1:
        bad.append(f"copy_memory_ is called {len(cm_calls)} times; exactly one channel is allowed")
    for node in cm_calls:
        if owner.get(id(node)) != "fold_weight_write":
            bad.append(f"line {node.lineno}: copy_memory_ outside fold_weight_write")
        want = [("dst", "data"), ("src", "data")]
        got = [(a.value.id, a.attr) if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) else None
               for a in node.args]
        if got != want or node.keywords:
            bad.append(f"line {node.lineno}: copy_memory_ must take the whole parameters (dst.data, src.data), "
                       f"got {ast.unparse(node)}")
    fw = funcs.get("fold_weight_write")
    if fw is None:
        bad.append("fold_weight_write is gone")
    else:
        first_calls = [ast.unparse(s.value.func) for s in fw.body if isinstance(s, ast.Expr)
                       and isinstance(s.value, ast.Call)]
        if not first_calls or first_calls[0] != "_assert_owns_storage":
            bad.append("fold_weight_write must check whole-storage ownership before writing")
    return bad


FOLD_SRC = pathlib.Path(fold_swap.__file__).read_text()
CM_LINE = "    torch_npu.copy_memory_(dst.data, src.data)\n"


def test_write_channel_scan_is_green_on_the_module():
    assert CM_LINE in FOLD_SRC
    assert write_channel_violations(FOLD_SRC) == []


def test_the_committer_writes_through_that_channel(world):
    assert world.fold.committer.weight_writer is fold_weight_write


@pytest.mark.parametrize("mutant, why", [
    ("    dst.copy_(src)\n", ".copy_() is an in-place write"),
    ("    torch_npu.copy_memory_(dst[1:].data, src.data)\n", "whole parameters"),
    ("    dst.data = src\n", ".data ="),
    ("    torch_npu.copy_memory_(dst.data.narrow(0, 1, 1), src.data)\n", "whole parameters"),
    # Each must be caught for its OWN reason, not by the "copy_memory_ is called
    # 0 times" rule that any replacement trips.
    ("    dst.data[:] = src\n", "subscript write through .data"),
    ("    dst[:] = src\n", "subscript assignment to 'dst'"),
    ("    _cm = torch_npu.copy_memory_\n    _cm(dst.data, src.data)\n", "under another name"),
    ("    getattr(dst, 'copy_')(src)\n", "getattr hides the attribute name"),
    ("    setattr(dst, 'data', src)\n", "setattr hides the attribute name"),
    ("    dst.index_add_(0, torch.arange(dst.shape[0]), src)\n", ".index_add_() is an in-place write"),
    ("    dst.masked_scatter_(torch.ones_like(dst, dtype=torch.bool), src)\n", ".masked_scatter_()"),
    ("    torch.add(src, 0, out=dst)\n", "out= writes into the tensor"),
])
def test_mut_write_channel_scan_goes_red(mutant, why):
    src = FOLD_SRC.replace(CM_LINE, mutant)
    assert src != FOLD_SRC
    compile(src, "fold_swap_mutant.py", "exec")            # syntactically valid mutant
    bad = write_channel_violations(src)
    assert any(why in b for b in bad), bad


def test_fold_weight_write_refuses_views(plat):
    dst = plat.on_device(torch.zeros(4, 6, dtype=torch.uint8), nz=True)
    with pytest.raises(ValueError, match="whole storage"):
        fold_weight_write(dst[1:], plat.on_device(torch.ones(3, 6, dtype=torch.uint8)))
    with pytest.raises(ValueError, match="whole storage"):
        fold_weight_write(dst, plat.on_device(torch.ones(8, 6, dtype=torch.uint8))[:4])


def test_module_imports_standalone():
    import subprocess
    import sys

    r = subprocess.run([sys.executable, "-c", "import vllm_ascend.kt_offload.fold_swap"],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


def test_the_fold_reuses_the_drivers_policy_and_checks():
    """One policy and one set of refusals, not a second copy."""
    src = inspect.getsource(fold_swap)
    assert "plan_from_heat(" in src and "TopKHeatPolicy()" in src
    assert "def check_proposal" not in textwrap.dedent(src)


# The short tail: the last forward of a prompt is often below
# ``stream_prefill_min_tokens`` and never reaches the fold at all. Everything
# below therefore goes through ``rt.maybe_stream_prefill``, the production entry,
# so the gate is the real one and "this forward streamed" is its own answer
# rather than this file's model of it.

#: Tokens per full prefill chunk here -- ``split_routing``'s halves.
BUDGET = 32
#: A tail in (0, MIN_TOKENS): the broken case. And one at exactly MIN_TOKENS,
#: which is the boundary the gate (``<``) lets through.
SHORT_TAIL, EXACT_TAIL = 8, MIN_TOKENS


def hot_only_routing(n: int, layer: int = LAYER):
    """``n`` tokens routed only to HOT, so a tail chunk cannot change which experts
    are hot -- leaving exactly one variable below: WHICH forward commits."""
    hot = tuple(HOT[layer])
    return torch.tensor([[hot[i % len(hot)], hot[(i + 1) % len(hot)]] for i in range(n)], dtype=torch.int32)


def test_the_tail_routing_cannot_change_the_answer():
    """The tail routing is neutral, so only the commit point is under test."""
    a, b = split_routing(BUDGET)
    total = _counts(a) + _counts(b)
    tail = _counts(hot_only_routing(EXACT_TAIL))
    cold = [e for e in range(E) if e not in HOT[LAYER]]
    assert set(torch.topk(total, SLOTS).indices.tolist()) == set(HOT[LAYER])
    assert set(torch.topk(total + tail, SLOTS).indices.tolist()) == set(HOT[LAYER])
    assert int(tail[cold].max()) == 0 and int(tail[list(HOT[LAYER])].min()) > 0


def _windows(prompt_len: int, budget: int = BUDGET, key: str = "R", cached: int = 0) -> list[PrefillWindow]:
    """The windows the runner publishes for one prompt, built from the definition
    (``first`` = nothing computed yet, ``last`` = this forward finishes it).
    ``cached`` is a prefix the cache already covered, so no window is ``first``."""
    out, computed = [], cached
    while computed < prompt_len:
        n = min(budget, prompt_len - computed)
        out.append(PrefillWindow(key=key, first=computed == 0, last=computed + n >= prompt_len,
                                 computed=computed, scheduled=n, prompt_len=prompt_len, seq=len(out) + 1))
        computed += n
    return out


def _forward_ids(world, rt, ids):
    """One forward through the production entry; ``None`` back = it took the hybrid
    path. The only helper here that lets the ``stream_prefill_min_tokens`` gate
    speak -- ``world.stream`` calls the fold directly and would stream 4 tokens."""
    n = int(ids.shape[0])
    g = torch.Generator().manual_seed(n + 977)
    w = torch.rand(n, TOPK, generator=g, dtype=torch.float32) + 0.1
    x = torch.randn(n, HIDDEN, generator=g).to(torch.bfloat16)
    return rt.maybe_stream_prefill(world.resident[LAYER], x, w, ids, in_profile_run=False)


def _prefill_through_the_gate(world, rt, prompt_len: int, key: str = "R", cached: int = 0):
    """A whole prompt, one forward per window. Returns the windows and, per forward,
    ``(it streamed, commits it carried)`` -- the first from ``maybe_stream_prefill``
    itself, the second from the coverage counter."""
    windows = _windows(prompt_len, key=key, cached=cached)
    halves = list(split_routing(BUDGET))
    assert [w.scheduled for w in windows[:2]] == [BUDGET, BUDGET]
    ids_per_forward = halves + [hot_only_routing(w.scheduled) for w in windows[2:]]
    got = []
    for ids, window in zip(ids_per_forward, windows):
        before = world.fold.coverage.layers_committed
        world.window = window
        streamed = _forward_ids(world, rt, ids) is not None
        got.append((streamed, world.fold.coverage.layers_committed - before))
    return windows, got


def test_a_prompt_with_a_short_tail_commits_on_the_last_forward_that_streams(ckpt, plat, tmp_path, monkeypatch):
    """A tail in (0, min_tokens) must still commit, exactly once, on chunk 2 of 3 --
    the last forward that streams. The expected set is HOT, which neither half of
    the prompt chooses alone, so the heat used was everything accumulated up to
    the deciding forward.
    """
    w = World(ckpt, plat, tmp_path)
    rt = _runtime(w, monkeypatch, dynamic=True, wire=True)
    windows, got = _prefill_through_the_gate(w, rt, 2 * BUDGET + SHORT_TAIL)

    assert [x.scheduled for x in windows] == [BUDGET, BUDGET, SHORT_TAIL]
    assert [x.last for x in windows] == [False, False, True]
    assert [x.remaining for x in windows] == [BUDGET + SHORT_TAIL, SHORT_TAIL, 0]
    assert [streamed for streamed, _ in got] == [True, True, False], \
        f"the {SHORT_TAIL}-token tail is below stream_prefill_min_tokens={MIN_TOKENS}: it must go hybrid"
    assert [commits for _, commits in got] == [0, 1, 0], \
        f"one commit, on the last forward that streams; got {[c for _, c in got]}"
    cov = w.fold.coverage
    assert cov.layers_committed == 1 and cov.skipped == {"not_last_chunk": 1}
    assert cov.decisions_seen == cov.layers_folded + sum(cov.skipped.values())
    assert cov.prefills_without_commit == 0

    # the accumulation: everything up to the deciding forward, and that is what decided
    a, b = split_routing(BUDGET)
    assert torch.equal(w.fold.accumulated_heat(LAYER), (_counts(a) + _counts(b)).to(torch.float64))
    assert tuple(rt.fold_report.plan.new_ids) == tuple(sorted(HOT[LAYER]))
    # ... and the bytes really landed, judged against the independent static build
    assert w.verdict() == []
    assert cov.violations(expected_staged=[(LAYER, e) for e in HOT[LAYER]]) == []


def test_mut_committing_only_on_the_window_marked_last_never_commits(ckpt, plat, tmp_path, monkeypatch):
    """Variant arm: ``_decides_now`` back to ``window.last``, which gives zero
    commits, a whole request's heat thrown away, and every counter self-consistent."""
    monkeypatch.setattr(ResidentFold, "_decides_now", lambda self, window: bool(window.last))
    w = World(ckpt, plat, tmp_path)
    rt = _runtime(w, monkeypatch, dynamic=True, wire=True)
    before = w.snapshot()
    _, got = _prefill_through_the_gate(w, rt, 2 * BUDGET + SHORT_TAIL)

    cov = w.fold.coverage
    assert cov.layers_committed == 0, \
        f"the arm must expose the missed commit; it committed {cov.layers_committed} time(s)"
    assert [commits for _, commits in got] == [0, 0, 0]
    assert cov.skipped == {"not_last_chunk": 2} and cov.decisions_seen == 2
    # and the self-consistency that made it invisible: the denominator is fine
    assert cov.decisions_seen == cov.layers_folded + sum(cov.skipped.values())
    assert w.unchanged(before), "nothing moved, which is the bug"
    assert w.verdict() != []


@pytest.mark.parametrize("tail", [0, EXACT_TAIL])
def test_a_tail_that_streams_keeps_the_commit_on_the_last_chunk(ckpt, plat, tmp_path, monkeypatch, tail):
    """Positive controls: r == 0 and r == min_tokens exactly. Both forwards DO stream
    (the gate is ``<``), so the commit must stay on the true last chunk; the
    r == min_tokens arm is the boundary -- one token fewer is the case above."""
    w = World(ckpt, plat, tmp_path)
    rt = _runtime(w, monkeypatch, dynamic=True, wire=True)
    windows, got = _prefill_through_the_gate(w, rt, 2 * BUDGET + tail)

    n = 2 + bool(tail)
    assert len(windows) == n and windows[-1].last and windows[-1].remaining == 0
    assert [streamed for streamed, _ in got] == [True] * n, "every forward here is at or above the gate"
    assert [commits for _, commits in got] == [0] * (n - 1) + [1], \
        f"the commit must stay on the last chunk; it landed on {[i for i, (_, c) in enumerate(got) if c]}"
    assert w.fold.coverage.layers_committed == 1
    assert w.fold.coverage.skipped == {"not_last_chunk": n - 1}
    assert w.verdict() == []


def test_a_prefix_cached_prompt_with_a_short_tail_commits_too(ckpt, plat, tmp_path, monkeypatch):
    """The prompt starts part way in, so no window is ``first``: the rule is about
    what the prefill has LEFT and reads neither ``computed`` nor ``first``."""
    cached = 1000
    w = World(ckpt, plat, tmp_path)
    rt = _runtime(w, monkeypatch, dynamic=True, wire=True)
    windows, got = _prefill_through_the_gate(w, rt, cached + 2 * BUDGET + SHORT_TAIL, cached=cached)

    assert [x.first for x in windows] == [False, False, False], "a cached prefix is not a first chunk"
    assert [x.computed for x in windows] == [cached, cached + BUDGET, cached + 2 * BUDGET]
    assert [x.remaining for x in windows] == [BUDGET + SHORT_TAIL, SHORT_TAIL, 0]
    assert [streamed for streamed, _ in got] == [True, True, False]
    assert [commits for _, commits in got] == [0, 1, 0]
    assert w.verdict() == []


def _capture_warnings(monkeypatch) -> list[str]:
    """By replacing ``logger.warning``, never with caplog: vllm's logger does not
    propagate to pytest's root handler, so a caplog assertion would see nothing and
    pass just as happily against a build that says nothing at all."""
    said: list[str] = []
    monkeypatch.setattr(fold_swap_module.logger, "warning", lambda fmt, *a: said.append(fmt % a if a else fmt))
    return said


def test_a_prefill_that_never_reached_the_fold_is_loud_once(ckpt, plat, tmp_path, monkeypatch):
    """A prompt shorter than the gate: nothing in the fold runs, so nothing in the
    fold can count it. The whole chain is production here, because the point is
    that the check fires from the per-step hook, not from a forward that did not
    happen.
    """
    said = _capture_warnings(monkeypatch)
    prefill_window.arm()
    w = World(ckpt, plat, tmp_path, window_source=prefill_window.require)
    prefill_window.on_prefill_end(w.fold.note_prefill_end)
    rt = _runtime(w, monkeypatch, dynamic=True, wire=True)

    # A HEALTHY prefill first, then the two silent ones: the other order would pass
    # just as well against a fold that counted commits cumulatively rather than per
    # request, and B's commit must not answer for A's prefill.
    prefill_window.observe(sched(new=[new_req("B", TOKENS)], scheduled={"B": TOKENS}))
    x, ids, weights = routing()
    assert rt.maybe_stream_prefill(w.resident[LAYER], x, weights, ids, in_profile_run=False) is not None
    prefill_window.observe(sched(cached=[("B", TOKENS)], scheduled={"B": 1}))
    assert w.fold.coverage.layers_committed == 1
    assert said == [] and w.fold.coverage.prefills_without_commit == 0, \
        f"a prefill that DID commit must be neither counted nor shouted about: {said}"

    short = MIN_TOKENS - 1
    for req, expect in (("A", 1), ("C", 2)):
        prefill_window.observe(sched(new=[new_req(req, short)], scheduled={req: short}))
        assert _forward_ids(w, rt, hot_only_routing(short)) is None, "below the gate: the hybrid path"
        prefill_window.observe(sched(cached=[(req, short)], scheduled={req: 1}))    # first decode step
        assert w.fold.coverage.prefills_without_commit == expect, \
            f"{req}'s prefill committed nothing and B's commit must not answer for it"

    assert w.fold.coverage.decisions_seen == 1, "neither A nor C was ever put to the fold"
    assert len(said) == 1, f"warn once per reason, not per request: {said}"
    assert "request A" in said[0] and "stream_prefill_min_tokens" in said[0], said[0]
    assert w.fold.coverage.violations(expected_staged=[(LAYER, e) for e in HOT[LAYER]]), \
        "and it is a violation, not just a log line"


def test_a_prefill_that_folded_and_never_committed_is_loud_too(ckpt, plat, tmp_path, monkeypatch):
    """The other half: decisions were taken, nothing was committed (the request is
    preempted half way through its prompt). Same counter, a different message --
    "never asked" and "asked and moved nothing" are different bugs."""
    said = _capture_warnings(monkeypatch)
    prefill_window.arm()
    w = World(ckpt, plat, tmp_path, window_source=prefill_window.require)
    prefill_window.on_prefill_end(w.fold.note_prefill_end)
    rt = _runtime(w, monkeypatch, dynamic=True, wire=True)

    prefill_window.observe(prefill_steps("A", prompt_len=4 * BUDGET, budget=BUDGET)[0][0])
    assert _forward_ids(w, rt, split_routing(BUDGET)[0]) is not None, "it did stream"
    assert w.fold.coverage.decisions_seen == 1 and w.fold.coverage.layers_committed == 0
    prefill_window.observe(sched(new=[new_req("B", TOKENS)], scheduled={"B": TOKENS}))   # A is gone

    assert w.fold.coverage.prefills_without_commit == 1
    assert len(said) == 1 and "not one commit" in said[0], said
    assert "request A" in said[0]
