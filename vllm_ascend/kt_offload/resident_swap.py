# SPDX-License-Identifier: Apache-2.0
"""Atomic commit of a hot-expert swap-in, one layer at a time.

Pure torch, no torch_npu import at module load, no kt-kernel. Every object the
commit writes is *injected* (``LayerSwapTargets``), so the unit test drives the
whole protocol on CPU tensors.

A commit changes which logical experts are resident in one layer. Five objects
encode that and all five must move together:

==================  ========================================  =====================
stage               object                                     read by
==================  ========================================  =====================
``weights``         resident params ``[num_slots, ...]``       GMM (captured graph)
``slot_table``      ``safe_slot_table`` per id dtype            ``split_routing``
``keep_table``      ``keep_table`` per weight dtype             ``split_routing``
``device_mask``     ``[E]`` bool on the device                  device-side consumers
``kt_mask``         kt's pinned ``gpu_experts_mask``            kt C++, live, by raw pointer
==================  ========================================  =====================

Stage order is fixed: weights -> slot table -> keep table -> masks, the kt mask
last. Until the keep table flips, no route reaches a slot whose weights changed;
until kt's mask flips, kt keeps computing exactly the experts the NPU still
zeroes. A failure after any prefix of the stages is repaired by the roll-forward
below. Deferring the mask writes to the last layer would instead leave "new
weights, old mask" on an error in between: wrong numbers, no exception.

Every write is in place. A captured graph replays the *address* it recorded:
``param.data = new_tensor`` writes fine and is then never read again. Every
target's address is recorded when ``LayerSwapTargets`` is built and checked
after every write.

WHEN a commit may happen is the caller's to decide, and the two callers differ:
``resident_driver`` commits at a step boundary with no forward running,
``fold_swap`` at a LAYER boundary inside the forward. So ``writer_busy`` below
is not "is a forward running" but "is the caller's own critical section still
open". A second thread or stream, or two steps' Python overlapping under async
scheduling, is refused by ``OwnershipLedger`` instead: one writer identity
(thread, stream) for its lifetime, one commit at a time, raising on a second.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import torch
from vllm.logger import logger

from vllm_ascend.kt_offload.placement import LayerPlacement
from vllm_ascend.kt_offload.weight_source import ZeroCoverageError

__all__ = [
    "STAGES",
    "ALL_CPU_WEIGHTS",
    "LayerSwapTargets",
    "OwnershipLedger",
    "OwnershipConflictError",
    "ResidentSwapCommitter",
    "StepReport",
    "SwapCorruptedError",
    "SwapError",
    "SwapReport",
    "WriterStamp",
    "ZeroCoverageError",
    "check_resident_state",
    "default_weight_write",
    "kt_cpu_weight_ids_of",
    "live_target_violations",
    "kt_mask_of",
]

STAGES = ("weights", "slot_table", "keep_table", "device_mask", "kt_mask")

#: ``kt_cpu_weight_ids`` value meaning "kt loaded every expert's CPU weights".
ALL_CPU_WEIGHTS = "all"


class SwapError(RuntimeError):
    """A commit failed and was rolled back; the layer is exactly as before."""


class SwapCorruptedError(RuntimeError):
    """Rollback itself failed: the layer is in an unknown state and is poisoned."""


class OwnershipConflictError(RuntimeError):
    """A second writer (thread / stream / overlapping commit / stale step) appeared."""


# ---------------------------------------------------------------------------
# Write channels
# ---------------------------------------------------------------------------


def _assert_whole_storage(dst: torch.Tensor, what: str) -> None:
    # torch_npu.copy_memory_ ignores the view it is given and copies the whole
    # storage (a 1-expert view of an 8-expert storage rewrote all 8), so a view
    # here would silently overwrite its neighbours. Contiguity is deliberately
    # NOT required: the production weight is transposed and in the internal NZ
    # format, so it reports non-contiguous while owning exactly the storage the
    # GEMM reads, and demanding contiguity rejected every real target. What must
    # hold is storage OWNERSHIP: offset 0 and the view covering the whole
    # storage.
    nbytes = dst.numel() * dst.element_size()
    if dst.storage_offset() != 0 or dst.untyped_storage().nbytes() != nbytes:
        raise ValueError(
            f"{what}: weight target must own its whole storage (offset={dst.storage_offset()}, "
            f"view bytes={nbytes}, storage bytes={dst.untyped_storage().nbytes()})"
        )
    if dst.device.type == "cpu" and not dst.is_contiguous():
        raise ValueError(f"{what}: host target must be contiguous")


def default_weight_write(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Whole-tensor write into a resident parameter, in place.

    On the NPU the only channel measured to work on the internal (NZ) format is
    ``torch_npu.copy_memory_`` (``Tensor.copy_`` refuses internal formats, an
    offset view trips ``storage_offset() == 0``). The CPU branch exists for the
    unit test only.
    """
    _assert_whole_storage(dst, "default_weight_write")
    if dst.device.type == "cpu":
        dst.copy_(src)
        return
    import torch_npu  # device-only dependency

    torch_npu.copy_memory_(dst.data, src.data)


def _table_write(dst: torch.Tensor, src: torch.Tensor) -> None:
    # Routing tables and masks are plain ND tensors: copy_ is valid for them
    # (the refusal is about internal formats only).
    dst.copy_(src)


def _clone(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone()


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def kt_mask_of(wrapper) -> torch.Tensor:
    """The mask kt's C++ actually reads, taken from the wrapper.

    kt copies the mask it is *given* into its own pinned tensor and hands C++
    that tensor's ``data_ptr()``. Writing the tensor ``layer.py`` passed to the
    constructor is a no-op for kt, so the target must be
    ``wrapper.gpu_experts_mask``.
    """
    mask = wrapper.gpu_experts_mask
    # kt keeps a lazily built device copy of the mask. Nothing builds it today,
    # but once built it would be a sixth copy this commit does not write, i.e.
    # stale after the first swap.
    if getattr(wrapper, "_gpu_experts_mask_gpu", None) is not None:
        raise ValueError("kt wrapper holds a device copy of gpu_experts_mask that a swap would leave stale")
    return mask


def live_target_violations(
    targets: LayerSwapTargets,
    *,
    weights: Mapping[str, torch.Tensor],
    slot_tables: Mapping[torch.dtype, torch.Tensor],
    keep_tables: Mapping[torch.dtype, torch.Tensor],
    kt_mask: torch.Tensor,
) -> list[str]:
    """Targets that are NOT the objects the layer actually routes and computes through.

    A commit writes ``LayerSwapTargets``; the forward reads whatever ``layer.py``
    baked into ``_slot_tables`` / ``_keep_tables`` and whatever kt was handed.
    Build the targets from a *fresh* ``placement.safe_slot_table(dt)`` -- what
    the obvious code, and the unit-test fixture, does -- and the commit writes
    tensors nobody reads: the weights move, the routing does not, decode sends
    every token to the old slot. No error, plausible magnitudes, wrong answer,
    and ``check_resident_state`` stays green because it reads the targets' own
    captured tensors.

    Identity is by ``data_ptr()``, not ``is``: ``param.data`` hands back a fresh
    Python object on every access while naming the same storage, so ``is`` would
    fail on the correct wiring. Shape and dtype are compared too -- a
    same-address tensor of another dtype is a different consumer.

    ``device_mask`` is NOT checked: nothing on the device reads a resident mask
    in this build (routing goes through the two tables, kt reads its own pinned
    copy), so there is no object to compare it against. A future device-side
    consumer must be added here at the same time it is added to the forward, or
    it inherits exactly the failure above.
    """
    bad: list[str] = []
    blind: list[str] = []

    def same(got: torch.Tensor | None, want: torch.Tensor | None, what: str) -> None:
        if got is None or want is None:
            bad.append(f"{what}: targets have {'a tensor' if got is not None else 'nothing'}, the layer has "
                       f"{'a tensor' if want is not None else 'nothing'}")
            return
        if got.data_ptr() == 0 or want.data_ptr() == 0:
            # An address of 0 is not an address: every ``meta`` tensor reports
            # 0, so two UNRELATED meta tensors compare equal and this function
            # would report "no violations" for exactly the mis-wiring it exists
            # to catch. Refuse rather than answer a question this comparison
            # cannot answer on these objects.
            blind.append(what)
            return
        if got.data_ptr() != want.data_ptr():
            bad.append(f"{what}: the commit would write 0x{got.data_ptr():x}, the forward reads "
                       f"0x{want.data_ptr():x} -- the swap would move the weights and leave the routing behind")
        elif got.dtype != want.dtype or tuple(got.shape) != tuple(want.shape):
            bad.append(f"{what}: same storage but {got.dtype}{tuple(got.shape)} vs {want.dtype}{tuple(want.shape)}")

    for name, t in targets.weights.items():
        same(t, weights.get(name), f"weights[{name}]")
    for extra in sorted(set(weights) - set(targets.weights)):
        bad.append(f"weights[{extra}]: the layer has it, the commit would not write it")
    for label, got_map, want_map in (("slot_table", targets.slot_tables, slot_tables),
                                     ("keep_table", targets.keep_tables, keep_tables)):
        for dt in sorted(set(got_map) | set(want_map), key=str):
            same(got_map.get(dt), want_map.get(dt), f"{label}[{dt}]")
    same(targets.kt_mask, kt_mask, "kt_mask")
    if blind:
        raise ValueError(
            f"live_target_violations cannot judge {sorted(blind)}: their data_ptr() is 0, which every 'meta' "
            "tensor reports, so identity here would be true for any two tensors. Returning 'no violations' "
            "would be a green light from a check that cannot go red -- run this against the real objects.")
    return bad


def kt_cpu_weight_ids_of(wrapper) -> frozenset[int] | str:
    """``LayerSwapTargets.kt_cpu_weight_ids`` read from what kt actually loaded.

    kt with ``KT_FULLSET_LOAD`` answers ``cpu_weight_expert_ids()`` from a
    snapshot taken right after its C++ load, so it does not drift when a swap
    rewrites the mask. An older kt has no such query; there the answer is the CPU
    set at call time, ``cpu_expert_ids()``, which is what a subset load allocated
    -- and if that kt did load the full set, under-reporting only makes demotion
    refuse (fail-closed), never read an unallocated BufferB. Call it before the
    first swap: after one, an old kt's ``cpu_expert_ids()`` describes the new
    mask, not the allocation.
    """
    query = getattr(wrapper, "cpu_weight_expert_ids", None)
    if query is None:
        return frozenset(int(i) for i in wrapper.cpu_expert_ids())
    ids = frozenset(int(i) for i in query())
    num_experts = int(wrapper.num_experts)
    if not ids <= frozenset(range(num_experts)):
        raise ValueError(f"kt reports CPU weights for {sorted(ids - frozenset(range(num_experts)))} "
                         f"outside [0, {num_experts})")
    return ALL_CPU_WEIGHTS if len(ids) == num_experts else ids


@dataclass
class LayerSwapTargets:
    """The five objects of one layer a commit writes, plus the addresses captured.

    ``kt_cpu_weight_ids`` is required and has no default: which experts kt holds
    CPU weights for. kt loads only the CPU experts' weights whenever it is built
    with a non-empty resident mask and its loader supports subsets, which is what
    ``layer.py`` does today. Demoting an expert kt has no weights for makes kt
    read an unallocated BufferB: wrong numbers, not an error. Pass
    ``ALL_CPU_WEIGHTS`` only when kt was loaded with the full set.
    """

    layer_idx: int
    weights: dict[str, torch.Tensor]
    slot_tables: dict[torch.dtype, torch.Tensor]
    keep_tables: dict[torch.dtype, torch.Tensor]
    device_mask: torch.Tensor
    kt_mask: torch.Tensor
    kt_cpu_weight_ids: frozenset[int] | str
    # key -> alias sharing the storage recorded at construction: what a graph
    # captured then would replay. Checks read through these, not the live
    # attributes, so a rebind shows up as "old content".
    _captured: dict[str, torch.Tensor] = field(init=False, repr=False)
    _addrs: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self):
        if not self.weights or not self.slot_tables or not self.keep_tables:
            raise ValueError(f"layer {self.layer_idx}: weights, slot_tables and keep_tables must be non-empty")
        if self.kt_mask.dtype != torch.bool or self.kt_mask.device.type != "cpu":
            raise ValueError(f"layer {self.layer_idx}: kt mask must be a host bool tensor (C++ reads uint8_t*)")
        if self.device_mask.dtype != torch.bool:
            raise ValueError(f"layer {self.layer_idx}: device mask must be bool")
        for key, t in self.items():
            # is_contiguous() is NOT the criterion, and requiring it rejected
            # the production weights on card: the post-load hook transposes them
            # and casts to the internal (NZ) format, so a real ``w13_weight``
            # reports non-contiguous while being exactly the tensor the GEMM and
            # the graph read. What matters for ``torch_npu.copy_memory_`` is
            # storage ownership from offset 0 (see ``_assert_whole_storage``).
            if t.storage_offset() != 0:
                raise ValueError(
                    f"layer {self.layer_idx}: target {key} has storage_offset "
                    f"{t.storage_offset()}; copy_memory_ refuses offset views and writes whole "
                    "storages, so an offset target would be written past its own end")
            if t.device.type == "cpu" and not t.is_contiguous():
                # Host tensors have no internal format, so there contiguity is
                # still a meaningful (and cheap) way to catch a view.
                raise ValueError(f"layer {self.layer_idx}: host target {key} is not contiguous")
        if self.kt_cpu_weight_ids != ALL_CPU_WEIGHTS:
            self.kt_cpu_weight_ids = frozenset(int(i) for i in self.kt_cpu_weight_ids)
        self._captured = {key: t.detach().view_as(t) for key, t in self.items()}
        self._addrs = {key: t.data_ptr() for key, t in self.items()}

    def items(self) -> Iterable[tuple[str, torch.Tensor]]:
        """(key, live tensor) in commit order."""
        for name, t in self.weights.items():
            yield f"weights:{name}", t
        for dt, t in self.slot_tables.items():
            yield f"slot_table:{dt}", t
        for dt, t in self.keep_tables.items():
            yield f"keep_table:{dt}", t
        yield "device_mask", self.device_mask
        yield "kt_mask", self.kt_mask

    def live(self, key: str) -> torch.Tensor:
        return dict(self.items())[key]

    def captured(self, key: str) -> torch.Tensor:
        return self._captured[key]

    def captured_addr(self, key: str) -> int:
        return self._addrs[key]

    def rebound(self) -> list[str]:
        """Keys whose live object no longer sits at the captured address."""
        return [key for key, t in self.items() if t.data_ptr() != self._addrs[key]]

    def restore_binding(self, key: str) -> None:
        """Point a rebound target back at the captured storage (rollback only)."""
        self.live(key).data = self._captured[key]


# ---------------------------------------------------------------------------
# Ownership (W4)
# ---------------------------------------------------------------------------


class WriterStamp(NamedTuple):
    step: int
    layer: int
    thread: int
    stream: object


class OwnershipLedger:
    """Who owns each (layer, expert), and who may write.

    Two claim tables per layer, each written by a *different* stage and read back
    from the object that stage wrote: ``npu`` from the captured keep tables and
    device mask, ``cpu`` from kt's captured mask. A (layer, expert) claimed by
    both sides is computed twice; by neither, not at all. Keeping the two writers
    separate is the point: a single "owner" array written by one piece of code
    would agree with itself however wrong it was.

    The writer bit admits one identity (thread, stream) for the ledger's lifetime
    and one commit at a time. A second thread or stream issuing a commit means
    the caller's premise about when a swap is safe is already false, so it
    raises rather than waits. This -- not ``writer_busy`` -- is what stands
    between the swap and a genuinely concurrent writer.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active: WriterStamp | None = None
        self._owner: tuple[int, object] | None = None
        self._last_step: int | None = None
        self.npu: dict[int, torch.Tensor] = {}
        self.cpu: dict[int, torch.Tensor] = {}

    @property
    def active(self) -> WriterStamp | None:
        return self._active

    def claim(self, stamp: WriterStamp) -> None:
        with self._lock:
            if self._active is not None:
                raise OwnershipConflictError(f"commit {stamp} overlaps commit in progress {self._active}")
            ident = (stamp.thread, stamp.stream)
            if self._owner is not None and ident != self._owner:
                raise OwnershipConflictError(
                    f"second writer {stamp}: this ledger belongs to (thread, stream)={self._owner}"
                )
            if self._last_step is not None and stamp.step < self._last_step:
                # A commit for an older step after a newer one is a writer
                # that did not see the step boundary move.
                raise OwnershipConflictError(f"stale step {stamp.step} after step {self._last_step}")
            self._owner = ident
            self._last_step = stamp.step
            self._active = stamp

    def release(self, stamp: WriterStamp) -> None:
        with self._lock:
            if self._active != stamp:
                raise OwnershipConflictError(f"release by {stamp}, but the active commit is {self._active}")
            self._active = None

    def record_npu(self, layer: int, targets: LayerSwapTargets) -> None:
        keeps = [targets.captured(f"keep_table:{dt}") != 0 for dt in targets.keep_tables]
        claim = targets.captured("device_mask").to("cpu").clone()
        for k in keeps:
            claim &= k.to("cpu")
        self.npu[layer] = claim

    def record_cpu(self, layer: int, targets: LayerSwapTargets) -> None:
        self.cpu[layer] = ~targets.captured("kt_mask").clone()

    def snapshot(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return (self.npu.get(layer), self.cpu.get(layer))

    def restore(self, layer: int, snap: tuple[torch.Tensor | None, torch.Tensor | None]) -> None:
        for table, value in zip((self.npu, self.cpu), snap):
            if value is None:
                table.pop(layer, None)
            else:
                table[layer] = value

    def violations(self, layer: int) -> list[str]:
        if layer not in self.npu or layer not in self.cpu:
            return [f"layer {layer}: ownership not recorded by both writers"]
        npu, cpu = self.npu[layer], self.cpu[layer]
        errs = []
        both, neither = npu & cpu, ~npu & ~cpu
        if both.any():
            errs.append(f"layer {layer}: {int(both.sum())} experts owned by NPU and CPU: {_ids(both)}")
        if neither.any():
            errs.append(f"layer {layer}: {int(neither.sum())} experts owned by neither side: {_ids(neither)}")
        return errs


def _ids(mask: torch.Tensor, limit: int = 8) -> list[int]:
    return torch.nonzero(mask, as_tuple=False).flatten()[:limit].tolist()


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


def check_resident_state(
    placement: LayerPlacement,
    targets: LayerSwapTargets,
    identify_row: Callable[[str, torch.Tensor], int] | None = None,
    ledger: OwnershipLedger | None = None,
) -> list[str]:
    """Violations of "one layer's five objects agree"; [] means they do. Test oracle.

    The reference set is **kt's mask**, read through the captured alias (what C++
    reads), not the placement and not the NPU tables -- the same choice as
    ``placement.check_split``, so an error shared by the NPU-side objects still
    shows. Every other object is judged as a *set* against it:

    * ``placement`` resident set == kt set;
    * device mask set == kt set; each keep table's nonzero set == kt set, values in {0, 1};
    * each slot table maps the kt set bijectively onto ``0..n-1`` and every
      other expert to ``dummy_slot``; all slot-table dtypes agree;
    * with ``identify_row``: for every e in the kt set, the weight row at
      ``slot_table[e]`` identifies as e, for every weight tensor (a route to
      e reaches e's weights);
    * every target still sits at its captured address;
    * with ``ledger``: each expert owned by exactly one side.
    """
    layer = placement.layer_idx
    errs: list[str] = []
    rebound = targets.rebound()
    if rebound:
        errs.append(f"layer {layer}: rebound (graph replays the old buffer): {rebound}")

    kt = targets.captured("kt_mask").to("cpu")
    n_exp = placement.num_experts
    if kt.numel() != n_exp:
        return errs + [f"layer {layer}: kt mask has {kt.numel()} entries, expected {n_exp}"]
    kt_set = set(_ids(kt, limit=n_exp))
    if not kt_set:
        errs.append(f"layer {layer}: kt mask has no resident expert")
    if set(placement.resident_ids) != kt_set:
        errs.append(
            f"layer {layer}: placement set != kt set "
            f"(+{sorted(set(placement.resident_ids) - kt_set)[:8]} -{sorted(kt_set - set(placement.resident_ids))[:8]})"
        )

    dev = targets.captured("device_mask").to("cpu")
    if set(_ids(dev, limit=n_exp)) != kt_set:
        errs.append(f"layer {layer}: device mask set != kt set")

    for dt in targets.keep_tables:
        keep = targets.captured(f"keep_table:{dt}").to("cpu").float()
        if not bool(((keep == 0) | (keep == 1)).all()):
            errs.append(f"layer {layer}: keep_table[{dt}] has values other than 0/1")
        if set(_ids(keep != 0, limit=n_exp)) != kt_set:
            errs.append(f"layer {layer}: keep_table[{dt}] set != kt set")

    ref_slots: torch.Tensor | None = None
    n = len(kt_set)
    kt_sorted = sorted(kt_set)
    for dt in targets.slot_tables:
        slots = targets.captured(f"slot_table:{dt}").to("cpu").long()
        if ref_slots is None:
            ref_slots = slots
        elif not torch.equal(slots, ref_slots):
            errs.append(f"layer {layer}: slot tables disagree across dtypes ({dt})")
        res_slots = sorted(int(slots[e]) for e in kt_sorted)
        if res_slots != list(range(n)):
            errs.append(f"layer {layer}: slot_table[{dt}] does not map the kt set onto 0..{n - 1}")
        others = ~kt
        if others.any() and bool((slots[others] != placement.dummy_slot).any()):
            errs.append(f"layer {layer}: slot_table[{dt}] sends a CPU expert somewhere other than the dummy slot")

    if identify_row is not None and ref_slots is not None:
        for name in targets.weights:
            w = targets.captured(f"weights:{name}")
            wrong = [e for e in kt_sorted if identify_row(name, w[int(ref_slots[e])]) != e]
            if wrong:
                errs.append(f"layer {layer}: weights[{name}]: {len(wrong)} routed experts land on another expert's row")

    if ledger is not None:
        errs.extend(ledger.violations(layer))
    return errs


# ---------------------------------------------------------------------------
# Committer
# ---------------------------------------------------------------------------


@dataclass
class SwapReport:
    """What one layer's commit moved, counted from the committed sets."""

    step: int
    layer: int
    ok: bool
    promoted: tuple[int, ...] = ()  # CPU -> NPU
    demoted: tuple[int, ...] = ()  # NPU -> CPU
    reslotted: tuple[int, ...] = ()  # stayed resident, slot changed
    bytes_written: int = 0
    stages_done: tuple[str, ...] = ()
    error: str | None = None
    # None = verify was off; True/False = the weight read-back ran / could not.
    # Separate from ``ok`` on purpose: "the write was accepted" and "the bytes
    # were read back and matched" are different claims, and on this build the
    # second may be unavailable (see the comment at the call site).
    weights_verified: bool | None = None
    weights_verify_error: str | None = None
    # True when the commit only completed because the roll-forward path ran.
    # ``ok`` stays True (the layer IS on the requested set); this says the happy
    # path did not hold -- a swap that needs recovery every time is a failing
    # card, not a working feature.
    recovered: bool = False

    @property
    def moved(self) -> int:
        """(layer, expert) pairs whose owner changed: the coverage of this commit."""
        return len(self.promoted) + len(self.demoted)


@dataclass
class StepReport:
    step: int
    layers: list[SwapReport]

    @property
    def moved(self) -> int:
        return sum(r.moved for r in self.layers if r.ok)

    @property
    def ok(self) -> bool:
        return bool(self.layers) and all(r.ok for r in self.layers) and self.moved > 0


class _Write(NamedTuple):
    stage: str
    key: str
    src: torch.Tensor
    writer: Callable[[torch.Tensor, torch.Tensor], None]


class ResidentSwapCommitter:
    """Commits resident-set changes layer by layer, all-or-nothing per layer.

    ``weight_writer`` is the device channel for the resident parameters
    (default: ``default_weight_write``); ``weight_refetch(layer, ids)``
    re-materialises the NEW set's weights when the weight write itself failed
    (``_roll_forward``; there is no snapshot). ``stream_id`` names the current
    stream (production: the NPU stream handle). ``writer_busy()`` is the CALLER's
    "my critical section is still open, do not write yet" -- the driver feeds it
    "a forward is in flight", the fold "this layer is still being folded"; the
    name comes from the driver's meaning, NOT the fold's (a forward always IS in
    flight when the fold commits). It is not a concurrency guard; the ledger is.
    ``on_write(stage, key)`` is called after every single write (tests use it to
    record order and inject faults).

    ``strict=True`` raises on any failure. ``strict=False`` rolls back, logs a
    warning and returns ``ok=False``; ownership conflicts and a failed rollback
    raise in both modes -- they mean the step-boundary premise is false or the
    layer is lost.
    """

    def __init__(
        self,
        placements: Mapping[int, LayerPlacement],
        targets: Mapping[int, LayerSwapTargets],
        *,
        strict: bool = True,
        weight_writer: Callable[[torch.Tensor, torch.Tensor], None] = default_weight_write,
        weight_refetch: Callable[[int, tuple[int, ...]], Mapping[str, torch.Tensor]] | None = None,
        stream_id: Callable[[], object] = lambda: None,
        writer_busy: Callable[[], bool] = lambda: False,
        on_write: Callable[[str, str], None] | None = None,
        ledger: OwnershipLedger | None = None,
        live_objects: Callable[[int], Mapping[str, Any]] | None = None,
    ):
        if set(placements) != set(targets):
            raise ValueError(f"placements cover layers {sorted(placements)}, targets cover {sorted(targets)}")
        self._placements = dict(placements)
        self._targets = dict(targets)
        self.strict = strict
        self.weight_writer = weight_writer
        self.weight_refetch = weight_refetch
        self.stream_id = stream_id
        self.writer_busy = writer_busy
        self.on_write = on_write
        # Returns {weights, slot_tables, keep_tables, kt_mask} as the LAYER
        # holds them right now. Checked before every commit, not once at build: a
        # rebake (``after_weights_loaded`` running again) silently detaches the
        # targets, and the symptom is a wrong answer, not an error.
        self.live_objects = live_objects
        self.ledger = ledger or OwnershipLedger()
        self._poisoned: dict[int, str] = {}
        for layer in self._targets:
            self._check_matches(self._placements[layer], self._targets[layer])
            self.ledger.record_npu(layer, self._targets[layer])
            self.ledger.record_cpu(layer, self._targets[layer])

    # -- queries ----------------------------------------------------------
    def placement(self, layer: int) -> LayerPlacement:
        return self._placements[layer]

    def targets(self, layer: int) -> LayerSwapTargets:
        return self._targets[layer]

    # -- commit -----------------------------------------------------------
    def commit_step(
        self,
        step: int,
        changes: Mapping[int, tuple[Iterable[int], Mapping[str, torch.Tensor]]],
        *,
        verify: bool = True,
    ) -> StepReport:
        """Commit every layer of one step. Zero total coverage is a failure."""
        reports = [
            self.commit(step, layer, ids, weights, verify=verify) for layer, (ids, weights) in sorted(changes.items())
        ]
        rep = StepReport(step, reports)
        if rep.moved == 0:
            msg = f"step {step}: zero coverage ({len(reports)} layers requested, 0 (layer, expert) moved)"
            if self.strict:
                raise ZeroCoverageError(msg)
            logger.warning("kt_offload resident swap: %s", msg)
        return rep

    def commit(
        self,
        step: int,
        layer: int,
        new_resident_ids: Iterable[int],
        new_weights: Mapping[str, torch.Tensor],
        *,
        verify: bool = True,
    ) -> SwapReport:
        """Make ``new_resident_ids`` the resident set of ``layer``.

        ``new_weights[name]`` is the full ``[num_slots, ...]`` content of that
        parameter in the *new* slot order (ascending logical id): the write is
        whole-tensor because the only NZ write channel writes whole storages.
        """
        if layer in self._poisoned:
            raise SwapCorruptedError(
                f"layer {layer} is poisoned by an earlier failed roll-forward: {self._poisoned[layer]}"
            )
        if self.writer_busy():
            raise OwnershipConflictError(
                f"step {step} layer {layer}: swap requested while the caller's writer is busy (the driver: a "
                "forward is in flight; the fold: this layer is still folding)")
        if self.live_objects is not None:
            detached = live_target_violations(self._targets[layer], **self.live_objects(layer))
            if detached:
                raise SwapError(f"layer {layer}: the commit targets are not what the forward reads -- "
                                + "; ".join(detached))
        stamp = WriterStamp(int(step), int(layer), threading.get_ident(), self.stream_id())
        self.ledger.claim(stamp)
        try:
            return self._commit_claimed(stamp, new_resident_ids, new_weights, verify)
        finally:
            self.ledger.release(stamp)

    def _commit_claimed(self, stamp, new_resident_ids, new_weights, verify) -> SwapReport:
        layer, step = stamp.layer, stamp.step
        old = self._placements[layer]
        tg = self._targets[layer]
        report = SwapReport(step=step, layer=layer, ok=False)
        # Everything that can be refused is refused here, before the first byte
        # moves: a rollback is the last line, not the plan.
        try:
            new = self._validate(old, tg, new_resident_ids, new_weights)
        except (ValueError, ZeroCoverageError) as e:
            return self._fail(report, e)
        old_set, new_set = set(old.resident_ids), set(new.resident_ids)
        report.promoted = tuple(sorted(new_set - old_set))
        report.demoted = tuple(sorted(old_set - new_set))
        report.reslotted = tuple(sorted(e for e in old_set & new_set if old.slot_of(e) != new.slot_of(e)))

        writes = self._plan_writes(new, tg, new_weights)
        ledger_snap = self.ledger.snapshot(layer)
        done: list[str] = []
        failed_at = -1
        try:
            for i, w in enumerate(writes):
                failed_at = i
                dst = tg.live(w.key)
                w.writer(dst, w.src)
                report.bytes_written += dst.numel() * dst.element_size()
                if verify and dst.data_ptr() != tg.captured_addr(w.key):
                    raise SwapError(f"layer {layer}: {w.key} was rebound instead of written in place")
                if self.on_write is not None:
                    self.on_write(w.stage, w.key)
                if not done or done[-1] != w.stage:
                    done.append(w.stage)
            self.ledger.record_npu(layer, tg)
            self.ledger.record_cpu(layer, tg)
            if verify:
                errs = self._verify(new, tg, new_weights, report)
                # The tables are checked as sets, but the committer cannot
                # identify a weight row, so read back what the graph will read: a
                # skipped or misdirected weight write leaves every table
                # consistent and the numbers wrong. Costs one device compare of
                # the resident weights per swapped layer.
                if errs:
                    raise SwapError(f"layer {layer}: post-commit check failed: {errs}")
        except Exception as e:  # noqa: BLE001 -- every failure mid-commit is rolled FORWARD
            # Roll forward, and if that lands the layer on the requested set the
            # commit SUCCEEDED: the model now has the set the caller asked for,
            # and reporting failure would make the next decision be taken against
            # a false picture. That a recovery was needed is its own field.
            # NOTE: stages_done is NOT reset from ``done`` here -- the partial
            # prefix ``done`` holds is superseded by what the roll-forward
            # rewrote, and it sets the field itself.
            self._roll_forward(layer, tg, new, writes, failed_at, new_weights, ledger_snap, e, report)
            self._placements[layer] = new
            report.ok = True
            report.recovered = True
            report.error = f"recovered by roll-forward after {type(e).__name__}: {e}"
            logger.warning("kt_offload resident swap: step %d layer %d completed by roll-forward "
                           "after %s", step, layer, e)
            return report
        self._placements[layer] = new
        report.stages_done = tuple(done)
        report.ok = True
        return report

    def _fail(self, report: SwapReport, err: Exception) -> SwapReport:
        report.ok = False
        report.error = f"{type(err).__name__}: {err}"
        if isinstance(err, OwnershipConflictError):
            raise err
        if self.strict:
            if isinstance(err, (SwapError, ZeroCoverageError)):
                raise err
            raise SwapError(f"step {report.step} layer {report.layer}: {report.error}") from err
        logger.warning(
            "kt_offload resident swap: step %d layer %d NOT committed (rolled back): %s",
            report.step,
            report.layer,
            report.error,
        )
        return report

    def _verify(self, new, tg: LayerSwapTargets, new_weights, report: SwapReport) -> list[str]:
        """The post-commit checks. Used by the commit AND by the roll-forward.

        One function on purpose: ``check_resident_state`` alone judges the tables
        and masks and cannot see weight CONTENT, so a roll-forward that ran only
        it would "recover" a layer into new tables and OLD weights and report it
        as committed. A recovery path that verifies less than the path it
        recovers turns a caught failure into a silent one.
        """
        errs = check_resident_state(new, tg, ledger=self.ledger)
        # The weight read-back may be IMPOSSIBLE, not just expensive: this build
        # refuses several ops on internal-format tensors (clone outright), so
        # torch.equal may raise here. A check that cannot run must be visible --
        # recorded as "not verified", never swallowed into a pass.
        for name, src in new_weights.items():
            try:
                same = torch.equal(tg.captured(f"weights:{name}"), src)
            except Exception as exc:  # noqa: BLE001
                report.weights_verified = False
                report.weights_verify_error = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                break
            if not same:
                errs.append(f"weights[{name}] captured content != requested")
        else:
            report.weights_verified = True
        return errs

    def _roll_forward(self, layer, tg: LayerSwapTargets, new, writes, failed_at: int,
                      new_weights, ledger_snap, cause: Exception, report: SwapReport) -> None:
        """Finish the commit instead of undoing it.

        Why forward and not back:

        * A swap changes the LONG-TERM resident set; nothing ever reverts it. The
          only reason to touch a failed commit is that a half-written layer is
          INCONSISTENT (new weights, old tables) and computes wrong numbers
          silently.
        * Undoing needs the old bytes back, and this build has no way to keep
          them: ``clone()`` is refused on an internal-format tensor, so a
          rollback path only ever looks real on the CPU tests, which clone
          contiguous ND stand-ins. Dropping the snapshot also removes a per-layer
          device copy of the resident weights that was never in the budget.
        * Going forward needs nothing: the new weights are already on the card,
          and the tables and masks can simply be written again. If the WEIGHT
          write itself failed, the source still exists on the host, so
          ``weight_refetch`` re-materialises it.

        Safety rests on the window being unobservable: the caller's own commit
        point is closed (``writer_busy``) and the ledger refuses a second writer,
        both checked before the first byte moves. What "unobservable" means is
        the caller's to state: a step boundary for the driver, a layer boundary
        inside the forward for the fold.
        """
        try:
            # Re-run EVERY write, not just the ones after the failure: when the
            # post-commit check is what failed, nobody knows which write is the
            # bad one, and the writes are idempotent by construction
            # (whole-tensor). ``failed_at`` only decides whether the weight
            # source has to be re-materialised.
            todo = list(writes)
            if failed_at >= 0 and writes[failed_at].stage == "weights" and self.weight_refetch is not None:
                fresh = self.weight_refetch(layer, tuple(new.resident_ids))
                todo = [w._replace(src=fresh[w.key.split(":", 1)[1]]) if w.stage == "weights" else w
                        for w in todo]
            for w in todo:
                dst = tg.live(w.key)
                w.writer(dst, w.src)
                # Account for what the recovery moved: without this a commit
                # that only completed BECAUSE of the roll-forward reports
                # bytes_written=0 / stages_done=(), and a coverage criterion
                # reads a successful recovery as "nothing happened".
                report.bytes_written += dst.numel() * dst.element_size()
            # The stages the roll-forward actually rewrote, in order, deduped --
            # derived from the writes it performed, never a constant, which would
            # keep reporting full coverage for a loop that wrote nothing.
            report.stages_done = tuple(dict.fromkeys(w.stage for w in todo))
            self.ledger.record_npu(layer, tg)
            self.ledger.record_cpu(layer, tg)
            errs = self._verify(new, tg, new_weights, report)
            if errs:
                raise SwapError(f"roll-forward left the layer inconsistent: {errs}")
        except Exception as e:  # noqa: BLE001
            self._poisoned[layer] = f"roll-forward failed ({e!r}) after {cause!r}"
            raise SwapCorruptedError(f"layer {layer}: {self._poisoned[layer]}") from e

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _check_matches(p: LayerPlacement, tg: LayerSwapTargets) -> None:
        if p.layer_idx != tg.layer_idx:
            raise ValueError(f"placement layer {p.layer_idx} != targets layer {tg.layer_idx}")
        if tg.kt_mask.numel() != p.num_experts or tg.device_mask.numel() != p.num_experts:
            raise ValueError(f"layer {p.layer_idx}: masks must have {p.num_experts} entries")
        for dt, t in {**tg.slot_tables, **tg.keep_tables}.items():
            if t.numel() != p.num_experts or t.dtype != dt:
                raise ValueError(f"layer {p.layer_idx}: table keyed {dt} is {t.dtype}[{t.numel()}]")
        for name, t in tg.weights.items():
            if t.dim() < 1 or t.shape[0] != p.num_slots:
                raise ValueError(f"layer {p.layer_idx}: weights[{name}] dim 0 is not num_slots={p.num_slots}")

    def _validate(self, old: LayerPlacement, tg, new_resident_ids, new_weights) -> LayerPlacement:
        new = LayerPlacement(
            layer_idx=old.layer_idx,
            num_experts=old.num_experts,
            num_slots=old.num_slots,
            resident_ids=tuple(sorted(int(i) for i in new_resident_ids)),
        )
        if new.num_resident != old.num_resident:
            # The count is fixed -- a swap changes *which* experts are resident,
            # not *how many*: kt caches num_gpu_experts at construction and the
            # unused tail slots were zeroed for this count.
            raise ValueError(f"layer {old.layer_idx}: resident count {old.num_resident} -> {new.num_resident}")
        moved = set(new.resident_ids) ^ set(old.resident_ids)
        if not moved:
            raise ZeroCoverageError(f"layer {old.layer_idx}: zero coverage, requested set equals the resident set")
        if tg.kt_cpu_weight_ids != ALL_CPU_WEIGHTS:
            no_bufb = sorted((set(old.resident_ids) - set(new.resident_ids)) - tg.kt_cpu_weight_ids)
            if no_bufb:
                raise ValueError(
                    f"layer {old.layer_idx}: demoting experts {no_bufb[:8]} kt holds no CPU weights for "
                    "(kt was loaded with a subset; kt needs the full set)"
                )
        if set(new_weights) != set(tg.weights):
            raise ValueError(
                f"layer {old.layer_idx}: new weights {sorted(new_weights)} != targets {sorted(tg.weights)}"
            )
        for name, src in new_weights.items():
            dst = tg.weights[name]
            if src.shape != dst.shape or src.dtype != dst.dtype:
                raise ValueError(
                    f"layer {old.layer_idx}: weights[{name}] {src.dtype}{tuple(src.shape)} "
                    f"!= target {dst.dtype}{tuple(dst.shape)}"
                )
            if src.untyped_storage().data_ptr() == dst.untyped_storage().data_ptr():
                raise ValueError(f"layer {old.layer_idx}: weights[{name}] source aliases its target")
        return new

    def _plan_writes(self, new: LayerPlacement, tg: LayerSwapTargets, new_weights) -> list[_Write]:
        """The writes of one commit, in STAGES order. The one place order is decided."""
        out: list[_Write] = []
        for name, src in new_weights.items():
            out.append(_Write("weights", f"weights:{name}", src, self.weight_writer))
        for dt, t in tg.slot_tables.items():
            out.append(_Write("slot_table", f"slot_table:{dt}", new.safe_slot_table(dt).to(t.device), _table_write))
        for dt, t in tg.keep_tables.items():
            out.append(_Write("keep_table", f"keep_table:{dt}", new.keep_table(dt).to(t.device), _table_write))
        out.append(_Write("device_mask", "device_mask", new.resident_mask().to(tg.device_mask.device), _table_write))
        out.append(_Write("kt_mask", "kt_mask", new.resident_mask(), _table_write))
        return out
