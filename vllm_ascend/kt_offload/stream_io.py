# SPDX-License-Identifier: Apache-2.0
"""The slot provider: "chunk k needs experts [e...]" -> a ``ChunkSlot``.

Between ``stream_prefill.py`` (which experts travel together) and
``stream_chain.py`` (compute) sits the I/O: host gather into a *pinned* staging
buffer (``weight_source.WeightSource``), H2D (four copies, one per weight
group), then the production post-load hook, whose outputs are the slot's four
tensors. The cast is never re-derived here: ``npu_format_cast``'s arguments, the
transposes and the scale reshape live only in
``AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading``, which
this module calls on a chunk-sized module, and the device buffer *shapes* come
from the same method (``get_weight`` / ``get_dynamic_quant_param``).

The **ND device buffers** (the H2D destination) are allocated ONCE and reused by
every chunk of every layer; plain ND, so ``Tensor.copy_`` works -- the
restriction that refuses ordinary writes is on *internal-format* destinations.
The **NZ output of the cast is per chunk**, freed when the caller drops the
previous slot: ``npu_format_cast`` returns a new tensor, no API casts into an
existing buffer, and reusing one NZ buffer would mean cast-into-a-temp plus
``torch_npu.copy_memory_(dst.data, src.data)`` -- the only write channel this
build accepts for internal format, since ``Tensor.copy_``, offset views and
``index_select(out=)`` are refused and ``copy_memory_`` ignores the view range
and writes the whole storage. The temp is allocated either way: no HBM saved,
one D2D copy added, and the stable address it would buy only matters for graph
replay, which streaming excludes (``stream_prefill.assert_eager_only``). The
**host** side is the ping-pong: two pinned buffers, so chunk k+1's gather
overlaps chunk k's H2D; copies are submitted non-blocking and a pinned buffer is
not refilled until the event of the H2D that read it has fired.

Two traps only the card can see: the post-hook tensors are NOT reported as the
format they were cast to (the hook is *given* 29 and ``get_npu_format`` returns
**50**, ``FRACTAL_NZ_C0_16``, so a criterion that pins 29 goes red on correct
code); and a **short last chunk** hands the hook ``buffer[:n]`` -- contiguous, at
storage offset 0, but still a view, and this build refuses offset views on
internal-format destinations (the cast reads ND and writes a fresh tensor, so it
is expected to be fine, but only the card can say so).

``tests/ut/kt_offload/test_stream_io.py`` pins the boundary. Stale bytes -- chunk
k served chunk k-1's weights -- stays plausible on-card, so it gets two
independent detectors here: a per-buffer stamp and an optional byte probe.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.kt_offload.stream_chain import SLOT_WEIGHT_ATTRS, ChunkSlot
from vllm_ascend.kt_offload.weight_source import (
    DEFAULT_COPY_THREADS,
    GROUPS,
    ChunkLayout,
    GatherResult,
    WeightSource,
    ZeroCoverageError,
)

__all__ = [
    "PARAM_FOR_GROUP",
    "DEFAULT_HOST_BUFFERS",
    "StreamIOError",
    "DeviceSeam",
    "SlotCoverage",
    "StreamSlotProvider",
    "DEFAULT_SLOT_COPY_THREADS",
    "chunk_param_shapes",
    "layout_signature",
    "default_alloc_pinned",
    "default_alloc_device",
    "default_h2d",
    "default_wait",
    "default_post_load",
]

#: ``weight_source`` group name -> the production parameter it fills. Group names
#: are the checkpoint side (w13 = w1 then w3 per expert, a plain byte concat);
#: parameter names are the ones ``process_weights_after_loading`` rewrites
#: (``SLOT_WEIGHT_ATTRS``). Explicit rather than derived, so a rename on either
#: side fails a check instead of silently filling the wrong buffer.
PARAM_FOR_GROUP: dict[str, str] = {
    "w13_weight": "w13_weight",
    "w13_scale": "w13_weight_scale",
    "w2_weight": "w2_weight",
    "w2_scale": "w2_weight_scale",
}

#: Host-side pinned buffers. Two = one being filled while one is being copied
#: out; the device side stays at one slot.
DEFAULT_HOST_BUFFERS = 2

#: mmap -> pinned staging threads, from ``weight_source.DEFAULT_COPY_THREADS``.
#: Measured on this host: it is the peak of the read leg and it balances the H2D
#: floor. The reference implementation's default is ``KT_MXFP4_COPY_THREADS=32``,
#: tuned on a 40-core single-NUMA box; here 32 is *slower* and more is worse
#: still. Do not "restore" 32 without re-running that bench on this machine.
DEFAULT_SLOT_COPY_THREADS = DEFAULT_COPY_THREADS


class StreamIOError(RuntimeError):
    """The slot handed back would not be the chunk's weights (stale, unformatted, mis-sized)."""


# ---------------------------------------------------------------------------
# The four device-touching steps, behind one injectable seam
# ---------------------------------------------------------------------------


def default_alloc_pinned(nbytes: int) -> torch.Tensor:
    """Host staging: page-locked uint8.

    A pageable source is several times slower on this box, and only a pinned
    source can be copied ``non_blocking``.
    """
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def default_alloc_device(shape: Sequence[int], dtype: torch.dtype, device: Any) -> torch.Tensor:
    """The reused ND slot. Allocated once per provider; see the module docstring."""
    return torch.empty(tuple(shape), dtype=dtype, device=device)


def default_h2d(dst: torch.Tensor, src: torch.Tensor, *, what: str) -> Any:
    """Submit one group's copy and hand back something to wait on.

    ``non_blocking=True`` returns long before the bytes have moved, which is what
    lets the copy overlap the next gather; the returned event is what keeps
    refilling the pinned source safe. Off-card there is no stream, so the handle
    is ``None``.
    """
    dst.copy_(src, non_blocking=True)
    if dst.device.type == "npu":
        import torch_npu  # noqa: F401  registers torch.npu; imported lazily

        event = torch.npu.Event()
        event.record()
        return event
    return None


def default_wait(handle: Any) -> None:
    if handle is not None:
        handle.synchronize()


def default_post_load(quant_method: Any, module: torch.nn.Module) -> None:
    """Run the PRODUCTION post-load hook on the chunk module.

    ``AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading`` owns
    the cast arguments (``npu_format_cast(..., 29,
    customize_dtype=float8_e4m3fn, input_dtype=float4_e2m1fn_x2)``), the weight
    transpose and the scale reshape/transpose.

    kt PATCHES that method on the layer's quant_method
    (``layer.attach_after_init``), wrapping it in ``zero_unused_slots`` and
    ``after_weights_loaded``. Those two belong to the layer's own weights and
    must NEVER run on a chunk or on the folded swap's staging:
    ``after_weights_loaded`` rebinds the live routing tables of the layer in
    flight to a throwaway module and reloads its entire CPU half (the wrapper
    registry refuses the second wrapper only after that damage). So this takes
    the hook kt wrapped, not kt's wrapper; unpatched methods (every other
    backend) carry no stamp and are used as they are.
    """
    hook = quant_method.process_weights_after_loading
    hook = getattr(hook, "_kt_offload_original", hook)
    hook(module)


@dataclass(frozen=True)
class DeviceSeam:
    """The four device-touching steps, injectable so the boundary can be modelled.

    Defaults are the real thing; the unit tests replace all four with recorders,
    since a host cannot pin memory, copy to a device or hold internal format.
    """

    alloc_pinned: Callable[[int], torch.Tensor] = default_alloc_pinned
    alloc_device: Callable[..., torch.Tensor] = default_alloc_device
    h2d: Callable[..., Any] = default_h2d
    wait: Callable[[Any], None] = default_wait
    post_load: Callable[[Any, torch.nn.Module], None] = default_post_load


# ---------------------------------------------------------------------------
# Shapes: asked of the production method, never written out here
# ---------------------------------------------------------------------------


def moe_scheme(quant_method: Any, *required: str) -> Any:
    """The ``AscendMoEScheme`` behind a layer's ``quant_method``, checked.

    Production hands the layer's ``quant_method``, which is
    ``AscendFusedMoEMethod`` -- a WRAPPER; what the streaming chain needs
    (``get_weight``, ``quant_type``, ``act_quant_type``) is on the scheme it
    holds (``method_adapters.py``: ``self.quant_method = scheme``; e.g.
    ``AscendW4A8MXFPDynamicFusedMoEMethod`` in ``w4a8_mxfp4.py``). The post-load
    hook is the other way round -- on the WRAPPER, and it is the wrapper kt
    patches (``default_post_load`` unwraps that patch) -- so **the two objects
    must not be collapsed into one**.

    ``required`` names the attributes the caller is about to use, checked here
    at once. Unwrapping is one explicit level, not a ``getattr`` fallback chain:
    a fallback would also accept an object carrying an inner ``quant_method``
    for an unrelated reason, and a wrong shape or quant type fails much later,
    as a buffer the GEMM quietly rejects.
    """
    if not required:
        raise ValueError("moe_scheme needs the attribute names the caller is about to use; "
                         "without them it cannot tell a scheme from a wrapper")
    if all(hasattr(quant_method, a) for a in required):
        return quant_method
    inner = getattr(quant_method, "quant_method", None)
    if inner is not None and all(hasattr(inner, a) for a in required):
        return inner
    missing_outer = [a for a in required if not hasattr(quant_method, a)]
    missing_inner = ("n/a" if inner is None
                     else [a for a in required if not hasattr(inner, a)])
    raise StreamIOError(
        f"neither {type(quant_method).__name__} (missing {missing_outer}) nor its inner quant_method "
        f"({type(inner).__name__ if inner is not None else 'absent'}, missing {missing_inner}) carries "
        f"{list(required)}; the streaming chain reads these off the scheme the model loader used, and "
        "there is no second source for them")


def shape_declaring_method(quant_method: Any) -> Any:
    """The object that declares a layer's parameter shapes: the scheme inside the
    layer's ``quant_method``, not the wrapper it is. See ``moe_scheme``.
    """
    return moe_scheme(quant_method, "get_weight", "get_dynamic_quant_param")


def chunk_param_shapes(quant_method: Any, chunk_experts: int, intermediate_size: int, hidden_size: int,
                       params_dtype: torch.dtype = torch.bfloat16) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``name -> (shape, dtype)`` for a chunk-sized layer, from the quant method itself.

    ``get_weight`` / ``get_dynamic_quant_param`` are what the model loader asks
    for the real layer; asking them for ``chunk_experts`` rows is the only
    difference. The empty tensors they return are never kept.
    """
    if chunk_experts <= 0:
        raise ValueError(f"chunk_experts must be > 0, got {chunk_experts}")
    decl = shape_declaring_method(quant_method)
    shapes = dict(decl.get_weight(chunk_experts, intermediate_size, hidden_size, params_dtype))
    shapes.update(decl.get_dynamic_quant_param(chunk_experts, intermediate_size, hidden_size, params_dtype))
    missing = [n for n in SLOT_WEIGHT_ATTRS if n not in shapes]
    if missing:
        raise StreamIOError(f"{type(decl).__name__} does not declare {missing}; the streaming slot "
                            f"is built from the parameters the post-load hook rewrites ({list(SLOT_WEIGHT_ATTRS)})")
    out: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
    for name in SLOT_WEIGHT_ATTRS:
        t = shapes[name]
        if t.dtype != torch.uint8:
            # The DS method declares the E8M0 scales uint8 so the load is a byte
            # move: into a typed E8M0 buffer the copy becomes a value conversion
            # that flattens every exponent, with no error -- the same trap
            # weight_source refuses to walk into.
            raise StreamIOError(f"{name} is declared {t.dtype}, not uint8: the streaming path moves BYTES, "
                                "and a non-uint8 destination turns the copy into a value conversion")
        if t.dim() != 3 or int(t.shape[0]) != chunk_experts:
            raise StreamIOError(f"{name} shape {tuple(t.shape)} is not [chunk_experts={chunk_experts}, n, k]")
        out[name] = (tuple(int(s) for s in t.shape), t.dtype)
    return out


def _check_group_map() -> None:
    """The two name spaces must still line up; a rename on either side fails here."""
    group_names = {g for g, _ in GROUPS}
    if set(PARAM_FOR_GROUP) != group_names:
        raise StreamIOError(f"weight_source groups {sorted(group_names)} do not match the staging map "
                            f"{sorted(PARAM_FOR_GROUP)}")
    if set(PARAM_FOR_GROUP.values()) != set(SLOT_WEIGHT_ATTRS):
        raise StreamIOError(f"staging map fills {sorted(PARAM_FOR_GROUP.values())} but the chunk slot is "
                            f"{sorted(SLOT_WEIGHT_ATTRS)}")


# ---------------------------------------------------------------------------
# The chunk-sized module the production hook runs on
# ---------------------------------------------------------------------------


class _ChunkModule(torch.nn.Module):
    """Carries exactly the four weights ``process_weights_after_loading`` rewrites.

    The parameters *alias* the provider's persistent ND buffers rather than being
    them: the hook assigns ``layer.w13_weight.data = <cast output>``, so were the
    attribute the buffer itself, the reused ND slot would be repointed at the
    per-chunk NZ tensor and every later chunk would cast whatever the last cast
    left. A ``Parameter`` shares the storage but is its own tensor object.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor]):
        super().__init__()
        for name in SLOT_WEIGHT_ATTRS:
            self.register_parameter(name, torch.nn.Parameter(tensors[name], requires_grad=False))


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


@dataclass
class SlotCoverage:
    """What the provider actually moved. Zero is a failure, never a pass.

    A path that gathered and served nothing produces the same output as
    "streaming is off" (the reference implementation falls back to hybrid
    silently in several places); this accounting is what tells them apart.
    """

    chunks_served: int = 0
    gathers: int = 0
    bytes_gathered: int = 0
    bytes_to_device: int = 0
    h2d_calls: int = 0
    pinned_buffers: int = 0
    pinned_bytes: int = 0
    expert_slots_served: int = 0          # counts multiplicity across layers/chunks
    experts_produced: set[tuple[int, int]] = field(default_factory=set)   # (layer, expert)
    layers_served: set[int] = field(default_factory=set)

    def summary(self) -> str:
        return (f"stream_io coverage: chunks={self.chunks_served} experts={len(self.experts_produced)} "
                f"layers={len(self.layers_served)} gathered={self.bytes_gathered} B "
                f"to_device={self.bytes_to_device} B in {self.h2d_calls} copies "
                f"pinned={self.pinned_buffers}x{self.pinned_bytes} B")

    def violations(self, expected: Iterable[tuple[int, int]] | None = None,
                   *, bytes_per_expert: int | None = None) -> list[str]:
        bad: list[str] = []
        if self.chunks_served == 0:
            bad.append("no chunk was served: zero coverage is a failure, not a pass")
        if self.bytes_gathered == 0:
            bad.append("no byte was gathered from the checkpoint")
        if self.bytes_to_device == 0:
            bad.append("no byte was copied to the device: the slot would hold whatever was there before")
        if not self.experts_produced:
            bad.append("no expert was produced")
        if bytes_per_expert is not None:
            want = self.expert_slots_served * bytes_per_expert
            if self.bytes_to_device != want:
                bad.append(f"{self.bytes_to_device} B reached the device but {self.expert_slots_served} expert "
                           f"slot(s) x {bytes_per_expert} B = {want} B were served")
        if expected is not None:
            want_set = {(int(a), int(b)) for a, b in expected}
            if self.experts_produced != want_set:
                missing = sorted(want_set - self.experts_produced)
                extra = sorted(self.experts_produced - want_set)
                bad.append(f"produced {len(self.experts_produced)} of the expected {len(want_set)} "
                           f"(layer, expert) pairs (missing {missing[:8]}, extra {extra[:8]})")
        return bad

    def check(self, expected: Iterable[tuple[int, int]] | Sequence[int] | None = None, *, layer: int | None = None,
              bytes_per_expert: int | None = None, strict: bool = True) -> list[str]:
        """Return the violations; with ``strict`` (the default) raise on any.

        ``expected`` is mandatory under ``strict`` and must come from OUTSIDE
        this object -- an expectation derived from what it checks cannot fail.
        Pass e.g. ``stream_prefill.resident_chunk_ids(placement)``.
        """
        if strict and expected is None:
            raise ValueError(
                "SlotCoverage.check(strict=True) needs the expected experts, and they must not come from "
                "the provider: an expectation read off the object under test cannot fail.")
        bad = self.violations(_normalise_expected(expected, layer), bytes_per_expert=bytes_per_expert)
        if bad and strict:
            raise ZeroCoverageError(f"{self.summary()} -- " + "; ".join(bad))
        return bad


def _normalise_expected(expected: Any, layer: int | None) -> list[tuple[int, int]] | None:
    if expected is None:
        return None
    items = list(expected)
    if not items:
        raise ValueError("expected experts is empty: 'nothing was supposed to stream' is not a criterion")
    if all(isinstance(e, (tuple, list)) and len(e) == 2 for e in items):
        return [(int(a), int(b)) for a, b in items]
    if layer is None:
        raise ValueError("expected experts given as plain ids needs layer= so the pairs are unambiguous")
    return [(int(layer), int(e)) for e in items]


# ---------------------------------------------------------------------------
# The provider
# ---------------------------------------------------------------------------


@dataclass
class _HostBuffer:
    """One pinned staging buffer and what it currently holds."""

    buffer: torch.Tensor
    key: tuple | None = None            # (layer, chunk_index, experts) it holds
    pending: tuple | None = None        # key a background gather is filling in
    future: Future | None = None
    h2d_handle: Any = None              # event of the last H2D that READ this buffer
    gather: GatherResult | None = None


class StreamSlotProvider:
    """Turns ``(chunk_index, expert_ids)`` into a ``ChunkSlot`` on the device.

    Use it as ``run_streamed_layer(slot_for=provider.slot_for, ...)`` after
    ``set_layer(layer_idx)``: one layer at a time, because that is the shape of
    the callback and because deferring any part of a layer across layers is
    forbidden. The device and host buffers outlive the layer and are reused by
    every layer.

    **Construct it before the KV-cache profile run**: every byte of the slot is
    allocated in ``__init__``, and a slot taken afterwards comes out of memory
    vLLM has already promised to the KV cache.

    ``probe_bytes`` (default 0 = off) reads a few bytes back from each device
    buffer after the copy and compares them with the staging buffer. It costs a
    device->host sync per chunk, so it is a diagnostic, not the steady state --
    but it is the only detector that can see a chunk served with the *previous*
    chunk's bytes. Head AND tail are probed: bytes two chunks happen to share
    have no discriminating power.
    """

    def __init__(self, source: WeightSource | str, quant_method: Any, *, hidden_size: int, intermediate_size: int,
                 chunk_experts: int, device: Any = "npu", params_dtype: torch.dtype = torch.bfloat16,
                 seam: DeviceSeam | None = None, host_buffers: int = DEFAULT_HOST_BUFFERS,
                 copy_threads: int = DEFAULT_SLOT_COPY_THREADS, layer: int | None = None,
                 probe_bytes: int = 0, strict: bool = True):
        _check_group_map()
        if host_buffers < 1:
            raise ValueError(f"host_buffers must be >= 1, got {host_buffers}")
        self.source = (WeightSource(source, copy_threads=copy_threads) if isinstance(source, str) else source)
        self.owns_source = isinstance(source, str)
        self.quant_method = quant_method
        self.chunk_experts = int(chunk_experts)
        self.device = device
        self.seam = seam or DeviceSeam()
        self.strict = strict
        self.probe_bytes = int(probe_bytes)
        self.coverage = SlotCoverage()
        self._layer = layer
        self._lock = threading.Lock()

        self.shapes = chunk_param_shapes(quant_method, self.chunk_experts, intermediate_size, hidden_size,
                                         params_dtype)
        #: bytes one expert occupies across the four parameters -- the number the
        #: H2D accounting is checked against.
        self.bytes_per_expert = sum(_prod(shape[1:]) for shape, _ in self.shapes.values())
        self.staging_bytes = self.bytes_per_expert * self.chunk_experts

        # One device slot, allocated once and reused by every chunk.
        self._device: dict[str, torch.Tensor] = {}
        for name, (shape, dtype) in self.shapes.items():
            buf = self.seam.alloc_device(shape, dtype, device)
            if tuple(buf.shape) != shape or buf.dtype != dtype:
                raise StreamIOError(f"device buffer {name} came back {buf.dtype}{tuple(buf.shape)}, "
                                    f"asked for {dtype}{shape}")
            self._device[name] = buf
        self._device_key: tuple | None = None

        # Host-side ping-pong, pinned.
        self._host: list[_HostBuffer] = []
        for _ in range(host_buffers):
            buf = self.seam.alloc_pinned(self.staging_bytes)
            if buf.dtype != torch.uint8 or buf.device.type != "cpu" or not buf.is_contiguous():
                raise StreamIOError(f"staging buffer must be contiguous CPU uint8, got {buf.dtype} "
                                    f"on {buf.device}")
            if buf.numel() < self.staging_bytes:
                raise StreamIOError(f"staging buffer holds {buf.numel()} B, a chunk needs {self.staging_bytes} B")
            self._host.append(_HostBuffer(buf))
            self.coverage.pinned_buffers += 1
            self.coverage.pinned_bytes += int(buf.numel())
        self._prefetch_pool: ThreadPoolExecutor | None = None
        #: wall clock per stage of slot_for, and how many chunks it covers
        self._stage_s: dict[str, float] = {"acquire": 0.0, "upload": 0.0, "format": 0.0}
        self._stage_n = 0
        self._record_bytes_checked = False
        self._last_host: _HostBuffer | None = None

    # -- layer binding ----------------------------------------------------
    @property
    def layer(self) -> int:
        if self._layer is None:
            raise StreamIOError("no layer bound: call set_layer(idx) before asking for a chunk "
                                "(the slot_for callback carries the chunk index, not the layer)")
        return self._layer

    def set_layer(self, layer: int) -> StreamSlotProvider:
        """Bind the provider to a layer. Drops any prefetch aimed at the old one."""
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ValueError(f"layer must be a non-negative int, got {layer!r}")
        self._drain_prefetch()
        self._layer = layer
        return self

    # -- the callback -----------------------------------------------------
    def slot_for(self, chunk_index: int, expert_ids: Sequence[int]) -> ChunkSlot:
        """Gather -> H2D -> production post-load hook -> ``ChunkSlot``."""
        ids = tuple(int(e) for e in expert_ids)
        if not ids:
            raise ZeroCoverageError(f"chunk {chunk_index}: no expert to stream")
        if len(ids) > self.chunk_experts:
            raise StreamIOError(f"chunk {chunk_index} has {len(ids)} experts but the slot holds "
                                f"{self.chunk_experts}; the device buffer is allocated once and not grown")
        if len(set(ids)) != len(ids):
            raise ValueError(f"chunk {chunk_index}: duplicate expert ids {ids}")
        key = (self.layer, int(chunk_index), ids)

        # Per-stage wall clock. These three are the whole of slot_for, and
        # ``_acquire`` includes waiting for a prefetch -- which is what tells
        # overlap apart from serialisation.
        _t0 = time.perf_counter()
        host = self._acquire(key)
        _t1 = time.perf_counter()
        views = host.gather.layout.views(host.buffer)
        self._check_record_bytes(host.gather.layout, len(ids))
        self._upload(host, views, len(ids), key)
        _t2 = time.perf_counter()
        if self.probe_bytes:
            self._probe(views, len(ids), key)
        slot = self._format(len(ids))
        _t3 = time.perf_counter()
        self._stage_s["acquire"] += _t1 - _t0
        self._stage_s["upload"] += _t2 - _t1
        self._stage_s["format"] += _t3 - _t2
        self._stage_n += 1
        # Every 120 chunks (10 layers at the default 12 chunks/layer): enough
        # lines to see the split drift between layers, few enough not to be the
        # cost itself.
        if self._stage_n >= 120:
            logger.info("kt_offload stream I/O: %s", self.stage_report())

        self.coverage.chunks_served += 1
        self.coverage.expert_slots_served += len(ids)
        self.coverage.layers_served.add(key[0])
        self.coverage.experts_produced.update((key[0], e) for e in ids)
        return slot

    def stage_report(self, *, reset: bool = True) -> str:
        """Where the streamed bytes' wall clock went, and the implied bandwidth.

        Compare that bandwidth against the two legs' known rates to see which is
        off. ``acquire`` covers waiting for the prefetch, so an acquire far above
        the gather's own cost means the prefetch is not running ahead.
        """
        n = self._stage_n
        if not n:
            return "no chunk served"
        gib = (self.staging_bytes * n) / (1 << 30)
        total = sum(self._stage_s.values())
        parts = []
        for name in ("acquire", "upload", "format"):
            t = self._stage_s[name]
            parts.append(f"{name} {t:.2f}s ({100 * t / total:.0f}%, {gib / t:.1f} GiB/s)" if t > 0
                         else f"{name} 0")
        out = (f"{n} chunk(s), {gib:.1f} GiB staged, {total:.2f}s total "
               f"({1000 * total / n:.1f} ms/chunk): " + ", ".join(parts))
        if reset:
            self._stage_s = {"acquire": 0.0, "upload": 0.0, "format": 0.0}
            self._stage_n = 0
        return out

    # -- 1. host gather ---------------------------------------------------
    def prefetch(self, chunk_index: int, expert_ids: Sequence[int]) -> bool:
        """Start filling a pinned buffer for a chunk in the background.

        The ping-pong: host side only, no device buffer doubled. Returns whether
        a gather was started; already-held or already-pending chunks start
        nothing.
        """
        ids = tuple(int(e) for e in expert_ids)
        if not ids:
            raise ZeroCoverageError(f"chunk {chunk_index}: nothing to prefetch")
        key = (self.layer, int(chunk_index), ids)
        with self._lock:
            if any(h.key == key or h.pending == key for h in self._host):
                return False
            # Never refill the buffer the current chunk is still copying out of:
            # _fill would block on its H2D event and the prefetch would serialise
            # with the very copy it is supposed to overlap.
            target = None
            for h in self._host:
                if h.future is None and h is not self._last_host:
                    target = h
                    break
            if target is None:
                return False
            target.pending = key
            if self._prefetch_pool is None:
                self._prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kt_slot_prefetch")
            target.future = self._prefetch_pool.submit(self._fill, target, key)
            return True

    def _fill(self, host: _HostBuffer, key: tuple) -> None:
        # The pinned buffer must not be overwritten while the copy that reads it
        # is still in flight; non_blocking H2D is exactly the case where "the
        # call returned" does not mean "the bytes were read".
        if host.h2d_handle is not None:
            self.seam.wait(host.h2d_handle)
            host.h2d_handle = None
        host.key = None
        host.gather = None
        result = self.source.gather(key[0], key[2], host.buffer)
        host.gather = result
        host.key = key
        with self._lock:
            self.coverage.gathers += 1
            self.coverage.bytes_gathered += result.coverage.produced_bytes

    def _acquire(self, key: tuple) -> _HostBuffer:
        host = self._acquire_buffer(key)
        self._last_host = host
        return host

    def _acquire_buffer(self, key: tuple) -> _HostBuffer:
        with self._lock:
            pending = next((h for h in self._host if h.pending == key), None)
        if pending is not None:
            # .result() re-raises the worker's exception instead of handing back a
            # half-filled buffer.
            pending.future.result()
            with self._lock:
                pending.future, pending.pending = None, None
            if pending.key != key or pending.gather is None:
                raise StreamIOError(f"prefetch for {key} finished without filling its buffer")
            return pending
        with self._lock:
            held = next((h for h in self._host if h.key == key and h.future is None), None)
        if held is not None:
            return held
        with self._lock:
            target = next((h for h in self._host if h.future is None), None)
        if target is None:
            target = self._host[key[1] % len(self._host)]
            target.future.result()
            with self._lock:
                target.future, target.pending = None, None
        self._fill(target, key)
        return target

    def _drain_prefetch(self) -> None:
        for h in self._host:
            if h.future is not None:
                try:
                    h.future.result()
                finally:
                    h.future, h.pending = None, None

    def _check_record_bytes(self, layout: ChunkLayout, num_experts: int) -> None:
        """The checkpoint's per-expert record must be what the parameter holds.

        Checked once, being a property of the checkpoint and the model dims. A
        mismatch means the provider was built for different dimensions than the
        weights on disk, which would otherwise surface as a size error deep
        inside a copy, or not at all.
        """
        if self._record_bytes_checked:
            return
        for group, param in PARAM_FOR_GROUP.items():
            if group not in layout.groups:
                raise StreamIOError(f"the gather layout has no group {group!r}")
            record = layout.groups[group][1]
            want = _prod(self.shapes[param][0][1:])
            if record != want:
                raise StreamIOError(
                    f"checkpoint gives {record} B per expert for {group}, but {param} holds {want} B per "
                    f"expert ({self.shapes[param][0]}): the provider's hidden/intermediate sizes do not "
                    "match the checkpoint")
        if layout.nbytes != self.bytes_per_expert * num_experts:
            raise StreamIOError(f"chunk layout is {layout.nbytes} B, parameters need "
                                f"{self.bytes_per_expert * num_experts} B")
        self._record_bytes_checked = True

    # -- 2. H2D -----------------------------------------------------------
    def _device_view(self, name: str, num_experts: int) -> torch.Tensor:
        buf = self._device[name]
        return buf if num_experts == int(buf.shape[0]) else buf[:num_experts]

    def _upload(self, host: _HostBuffer, views: Mapping[str, torch.Tensor], num_experts: int, key: tuple) -> None:
        self._device_key = None   # stale until every group has been copied
        handle = None
        # In staging order (``weight_source.GROUPS``), so the pinned buffer is
        # read front to back rather than jumped around.
        for group in (g for g, _ in GROUPS):
            param = PARAM_FOR_GROUP[group]
            dst = self._device_view(param, num_experts).reshape(-1)
            src = views[group].reshape(-1)
            if dst.numel() != src.numel():
                raise StreamIOError(f"{param}: device view holds {dst.numel()} B, staging group {group} "
                                    f"has {src.numel()} B")
            handle = self.seam.h2d(dst, src, what=param)
            self.coverage.h2d_calls += 1
            self.coverage.bytes_to_device += int(src.numel())
        # The pinned source stays off-limits until this copy has drained.
        host.h2d_handle = handle
        self._device_key = key

    def _probe(self, views: Mapping[str, torch.Tensor], num_experts: int, key: tuple) -> None:
        for group, param in PARAM_FOR_GROUP.items():
            dev = self._device_view(param, num_experts).reshape(-1)
            src = views[group].reshape(-1)
            n = min(self.probe_bytes, src.numel())
            if n == 0:
                continue
            for where, window in (("head", slice(0, n)), ("tail", slice(src.numel() - n, src.numel()))):
                got = dev[window].to("cpu")
                if not torch.equal(got, src[window]):
                    raise StreamIOError(
                        f"chunk {key[1]} layer {key[0]}: the device buffer for {param} does not hold the "
                        f"bytes that were staged for it ({where} {n} B differ). The copy was issued but the "
                        "slot would have been computed with some other chunk's weights.")

    # -- 3. format (the production hook) ----------------------------------
    def _format(self, num_experts: int) -> ChunkSlot:
        if self._device_key is None:
            raise StreamIOError("the device buffers were not filled for this chunk: refusing to format and "
                                "hand back whatever the previous chunk left there")
        return self.format_nd({name: self._device_view(name, num_experts) for name in SLOT_WEIGHT_ATTRS},
                              num_experts)

    def format_nd(self, tensors: Mapping[str, torch.Tensor], num_experts: int) -> ChunkSlot:
        """Run the production post-load hook on ND device tensors and hand back the slot.

        The one place the cast happens, for a chunk AND for the folded swap's
        whole-layer staging (``fold_swap.py``). The ND tensors are not modified;
        the hook rebinds the module's parameters -- the weights to a fresh
        ``npu_format_cast`` output, the scales to a 4-D transposed VIEW of the ND
        bytes, so the caller must keep the ND scale storage alive and untouched
        while it uses the slot.
        """
        module = _ChunkModule(tensors)
        before = {name: layout_signature(getattr(module, name).data) for name in SLOT_WEIGHT_ATTRS}
        self.seam.post_load(self.quant_method, module)
        after = {name: getattr(module, name).data for name in SLOT_WEIGHT_ATTRS}
        unchanged = [n for n in SLOT_WEIGHT_ATTRS if layout_signature(after[n]) == before[n]]
        if unchanged:
            # All four are rewritten by the production hook: the weights get a
            # new ``npu_format_cast`` allocation (address) then a transpose
            # (stride), the scales a reshape to 4-D and a transpose (shape and
            # stride). Same address, shape AND stride means untouched, i.e. raw
            # ND bytes reaching the GEMM -- at best aclnnGroupedMatmulV4
            # refusing the call (161002), at worst numbers. Identity (``is``) is
            # NOT the test: ``param.data`` hands back a fresh tensor object on
            # every access, so it would pass for a hook that did nothing.
            raise StreamIOError(f"the post-load hook left {unchanged} exactly as they were: the chunk would "
                                "be handed to the GEMM in the checkpoint's ND layout, not the layout "
                                "process_weights_after_loading produces")
        slot = ChunkSlot(**{name: after[name] for name in SLOT_WEIGHT_ATTRS})
        bad = slot.violations(expect_slots=num_experts)
        if bad:
            raise StreamIOError(f"the formatted chunk is not a usable slot: {'; '.join(bad)}")
        return slot

    def served_nd(self, chunk_index: int, expert_ids: Sequence[int]) -> dict[str, torch.Tensor]:
        """The ND device bytes of the chunk just served, in the chunk's slot order.

        For the folded swap (``fold_swap.py``), which copies the chosen experts
        into its whole-layer staging between chunks. They are the H2D
        destination, still in the checkpoint's ND layout (the cast writes a fresh
        tensor and leaves them alone), valid only until the next ``slot_for``
        overwrites them -- hence the key check: another chunk, or this index on
        another layer, is right shape, plausible bytes, wrong expert.
        """
        ids = tuple(int(e) for e in expert_ids)
        key = (self.layer, int(chunk_index), ids)
        if self._device_key != key:
            raise StreamIOError(f"the device buffers hold {self._device_key}, not {key}: the chunk asked for "
                                "was not the last one served, and its bytes are gone")
        return {name: self._device_view(name, len(ids)) for name in SLOT_WEIGHT_ATTRS}

    # -- coverage / teardown ----------------------------------------------
    def check_coverage(self, expected: Iterable[Any] | None = None, *, layer: int | None = None,
                       strict: bool | None = None) -> list[str]:
        """Coverage verdict; ``expected`` must come from outside (see ``SlotCoverage.check``)."""
        return self.coverage.check(expected, layer=layer, bytes_per_expert=self.bytes_per_expert,
                                   strict=self.strict if strict is None else strict)

    def close(self) -> None:
        self._drain_prefetch()
        if self._prefetch_pool is not None:
            self._prefetch_pool.shutdown(wait=True)
            self._prefetch_pool = None
        for h in self._host:
            if h.h2d_handle is not None:
                self.seam.wait(h.h2d_handle)
                h.h2d_handle = None
        if self.owns_source:
            self.source.close()


def layout_signature(t: torch.Tensor) -> tuple:
    """Address + shape + stride: what a cast/transpose changes and a no-op does not.

    ``stride()`` is wrapped because internal-format tensors regularly refuse
    ordinary metadata calls on this build; a refusal is itself a difference from
    the ND input, so ``None`` keeps the comparison meaningful instead of raising.
    """
    try:
        stride = tuple(t.stride())
    except Exception:  # noqa: BLE001 - see the docstring
        stride = None
    return (t.data_ptr(), tuple(t.shape), stride)


def _prod(dims: Sequence[int]) -> int:
    n = 1
    for d in dims:
        n *= int(d)
    return n
