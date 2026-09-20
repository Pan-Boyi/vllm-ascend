# SPDX-License-Identifier: Apache-2.0
"""Resident-expert placement and the routing split. Pure torch, no NPU.

Three id spaces appear here:

* **logical id** ``0..num_experts-1`` -- the global expert number the router
  emits and the checkpoint uses (``layers.L.ffn.experts.<id>``). kt-kernel is
  indexed by logical id too (``gpu_experts_mask[id]``).
* **slot** ``0..num_slots-1`` -- the row of the NPU weight tensors
  (``w13_weight[slot]``). Resident experts are packed in ascending logical
  order, so for the prefix placement slot == logical id.
* ``CPU_SLOT`` (-1) -- "not on the NPU". It never reaches an NPU kernel: the
  Ascend routing kernel has no -1 sentinel, so a CPU route is sent to a real
  resident slot (``dummy_slot``) with weight 0 instead.

``num_slots`` is one number for the whole model because the ALLGATHER token
dispatcher is a process-wide singleton sized by the *last* layer's
``num_local_experts`` (moe_comm_method.py setup_moe_comm_method). Layers with
fewer residents leave their tail slots unused; nothing routes to them.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

logger = logging.getLogger(__name__)

CPU_SLOT = -1


# ---------------------------------------------------------------------------
# Per-layer placement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerPlacement:
    """Which logical experts of one layer live in which NPU slot."""

    layer_idx: int
    num_experts: int
    num_slots: int
    resident_ids: tuple[int, ...]
    _l2s: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        ids = tuple(int(i) for i in self.resident_ids)
        if list(ids) != sorted(set(ids)):
            raise ValueError(
                f"layer {self.layer_idx}: resident ids must be unique and ascending, got {list(ids)[:16]}..."
            )
        if ids and (ids[0] < 0 or ids[-1] >= self.num_experts):
            raise ValueError(f"layer {self.layer_idx}: resident id out of range [0, {self.num_experts}).")
        if len(ids) > self.num_slots:
            raise ValueError(f"layer {self.layer_idx}: {len(ids)} resident experts > {self.num_slots} slots.")
        if not ids:
            # The CPU routes need a real resident slot as their zero-weight
            # target; an unused slot holds unloaded bytes whose product with 0
            # can still be NaN.
            raise ValueError(f"layer {self.layer_idx}: at least one resident expert is required.")
        l2s = [CPU_SLOT] * self.num_experts
        for slot, lid in enumerate(ids):
            l2s[lid] = slot
        object.__setattr__(self, "resident_ids", ids)
        object.__setattr__(self, "_l2s", tuple(l2s))

    # -- scalar lookups (weight loading) --------------------------------
    def slot_of(self, logical_id: int) -> int:
        """NPU slot of ``logical_id``, or CPU_SLOT. Used by the weight loader."""
        return self._l2s[int(logical_id)]

    @property
    def num_resident(self) -> int:
        return len(self.resident_ids)

    @property
    def dummy_slot(self) -> int:
        """Zero-weight target for CPU routes: a slot with loaded weights."""
        return 0

    @property
    def unused_slots(self) -> list[int]:
        return list(range(self.num_resident, self.num_slots))

    @property
    def cpu_ids(self) -> list[int]:
        return [i for i, s in enumerate(self._l2s) if s == CPU_SLOT]

    # -- tables ---------------------------------------------------------------
    def resident_mask(self) -> torch.Tensor:
        """``[num_experts]`` bool, True = on the NPU. This is kt's gpu_experts_mask."""
        m = torch.zeros(self.num_experts, dtype=torch.bool)
        m[list(self.resident_ids)] = True
        return m

    def logical_to_slot(self) -> torch.Tensor:
        return torch.tensor(self._l2s, dtype=torch.int64)

    def slot_to_logical(self) -> torch.Tensor:
        out = torch.full((self.num_slots,), CPU_SLOT, dtype=torch.int64)
        out[: self.num_resident] = torch.tensor(self.resident_ids, dtype=torch.int64)
        return out

    def safe_slot_table(self, dtype: torch.dtype = torch.int32) -> torch.Tensor:
        """``[num_experts]``: resident -> its slot, CPU -> dummy_slot."""
        t = self.logical_to_slot()
        t = torch.where(t == CPU_SLOT, torch.full_like(t, self.dummy_slot), t)
        return t.to(dtype)

    def keep_table(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """``[num_experts]``: 1 for resident, 0 for CPU experts."""
        return self.resident_mask().to(dtype)


# ---------------------------------------------------------------------------
# Routing split (the only per-forward op this module contributes)
# ---------------------------------------------------------------------------


class NpuRouting(NamedTuple):
    """The NPU half of a routing decision.

    Field order is ``(topk_weights, topk_ids)``: the order every Ascend MoE
    call site names them in (``_select_experts``, ``quant_method.apply``), so
    positional unpacking at a call site cannot silently swap them.
    """

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor


ID_DTYPES = (torch.int32, torch.int64)


def assert_routing_dtypes(topk_ids: torch.Tensor, topk_weights: torch.Tensor, where: str) -> None:
    """Refuse routing tensors the Ascend MoE kernels cannot take, at the call site.

    The kernel's own complaint arrives far away and names neither the caller
    nor the argument that is wrong: a float ``topk_ids`` surfaces inside
    ``npu_moe_init_routing_v2`` as ``aclnnMoeInitRoutingV3 ... EZ1001: Tensor
    expertIdx not implemented for DT_FLOAT`` (seen on card 2026-09-20, from a
    caller that had swapped weights and ids). Never coerce with ``.to()``: the
    swap would then run and produce silently wrong routing.
    """
    if topk_ids.dtype not in ID_DTYPES:
        raise TypeError(
            f"{where}: topk_ids must be int32/int64 (the Ascend routing kernel's expertIdx takes DT_INT32), "
            f"got {topk_ids.dtype} with shape {tuple(topk_ids.shape)}. If this is float, the caller has "
            "probably passed topk_weights where topk_ids belong -- the argument order is (ids, weights) here "
            "and the return order is NpuRouting(topk_weights, topk_ids)."
        )
    if not topk_weights.is_floating_point():
        raise TypeError(
            f"{where}: topk_weights must be a floating dtype, got {topk_weights.dtype} with shape "
            f"{tuple(topk_weights.shape)} (weights/ids swapped?)."
        )
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"{where}: topk_ids {tuple(topk_ids.shape)} and topk_weights {tuple(topk_weights.shape)} "
            "must have the same shape."
        )


def split_routing(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    safe_slot_table: torch.Tensor,
    keep_table: torch.Tensor,
) -> NpuRouting:
    """NPU half of a routing decision.

    ``topk_ids`` are *logical* ids (before any log2phy). Resident routes become
    their slot with the weight unchanged; CPU routes become ``dummy_slot`` with
    weight 0. Shapes and dtypes are preserved; no -1 is produced. Two gathers
    and a multiply; both tables live on the compute device.
    """
    assert_routing_dtypes(topk_ids, topk_weights, "kt_offload split_routing")
    npu_ids = safe_slot_table[topk_ids.long()].to(topk_ids.dtype)
    npu_weights = topk_weights * keep_table[topk_ids.long()].to(topk_weights.dtype)
    return NpuRouting(topk_weights=npu_weights, topk_ids=npu_ids)


def check_split(
    placement: LayerPlacement,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    npu_ids: torch.Tensor,
    npu_weights: torch.Tensor,
) -> list[str]:
    """Violations of the split contract; [] means it holds. Test oracle.

    The partition is checked against what kt computes (routes whose logical
    id is *not* in kt's gpu_experts_mask), not against the NPU tables, so an
    error shared by both tables would still show.
    """
    errs: list[str] = []
    if npu_ids.shape != topk_ids.shape or npu_weights.shape != topk_weights.shape:
        errs.append(
            f"shape changed: ids {tuple(topk_ids.shape)}->{tuple(npu_ids.shape)}, "
            f"weights {tuple(topk_weights.shape)}->{tuple(npu_weights.shape)}"
        )
        return errs
    if npu_ids.dtype != topk_ids.dtype or npu_weights.dtype != topk_weights.dtype:
        errs.append(f"dtype changed: {topk_ids.dtype}->{npu_ids.dtype}, {topk_weights.dtype}->{npu_weights.dtype}")
    n_routes = topk_ids.numel()
    if n_routes == 0:
        errs.append("zero routes checked")
        return errs
    lo, hi = int(npu_ids.min()), int(npu_ids.max())
    if lo < 0 or hi >= placement.num_resident:
        errs.append(f"NPU id outside the loaded slots [0,{placement.num_resident}): min={lo} max={hi}")

    kt_cpu = ~placement.resident_mask()[topk_ids.long()]  # what kt will compute
    l2s = placement.logical_to_slot()[topk_ids.long()]
    res = ~kt_cpu
    if res.any():
        if not torch.equal(npu_ids[res].long(), l2s[res]):
            errs.append(f"{int((npu_ids[res].long() != l2s[res]).sum())} resident routes not mapped to their slot")
        if not torch.equal(npu_weights[res], topk_weights[res]):
            errs.append("resident route weight altered")
    if kt_cpu.any() and bool((npu_weights[kt_cpu] != 0).any()):
        errs.append(f"{int((npu_weights[kt_cpu] != 0).sum())} CPU routes keep a nonzero NPU weight (double count)")
    # NPU-effective routes and kt routes must partition every route.
    npu_eff = npu_weights != 0
    zero_w = topk_weights == 0
    overlap = npu_eff & kt_cpu
    missing = ~npu_eff & ~kt_cpu & ~zero_w
    if overlap.any():
        errs.append(f"{int(overlap.sum())} routes computed on both sides")
    if missing.any():
        errs.append(f"{int(missing.sum())} routes computed on neither side")
    # Recombination: NPU weight + kt weight == original, exactly.
    recombined = npu_weights + torch.where(kt_cpu, topk_weights, torch.zeros_like(topk_weights))
    if not torch.equal(recombined, topk_weights):
        errs.append("npu_weight + cpu_weight != original weight")
    return errs


# ---------------------------------------------------------------------------
# Building the whole-model plan
# ---------------------------------------------------------------------------


def prefix_resident_sets(num_layers: int, num_experts: int, n: int) -> list[list[int]]:
    n = _check_n(n, num_experts)
    return [list(range(n)) for _ in range(num_layers)]


def frequency_resident_sets(freq: torch.Tensor, n: int) -> list[list[int]]:
    """Per layer, the ``n`` most frequent experts; ties -> lower id first."""
    if freq.dim() != 2:
        raise ValueError(f"frequency table must be [num_layers, num_experts], got {tuple(freq.shape)}")
    f = freq.detach().to("cpu", torch.float64)
    if not torch.isfinite(f).all() or (f < 0).any():
        raise ValueError("frequency table has negative or non-finite counts")
    n = _check_n(n, f.shape[1])
    out = []
    for row in f:
        # stable sort on -count keeps ascending id order among equal counts
        order = torch.sort(-row, stable=True).indices[:n]
        out.append(sorted(int(i) for i in order))
    return out


def explicit_resident_sets(spec, num_layers: int, num_experts: int) -> list[list[int]]:
    """``spec``: list of per-layer id lists, or {"layers": {"<L>": [...]}} / {"<L>": [...]}."""
    if isinstance(spec, Mapping):
        spec = spec.get("layers", spec)
        missing = [layer for layer in range(num_layers) if str(layer) not in spec and layer not in spec]
        if missing:
            raise ValueError(f"explicit resident set misses layers {missing[:10]}")
        rows = [spec[str(layer)] if str(layer) in spec else spec[layer] for layer in range(num_layers)]
    elif isinstance(spec, Sequence):
        if len(spec) != num_layers:
            raise ValueError(f"explicit resident set lists {len(spec)} layers, model has {num_layers}")
        rows = list(spec)
    else:
        raise TypeError(f"unsupported explicit resident set: {type(spec).__name__}")
    out = []
    for layer, row in enumerate(rows):
        ids = [int(i) for i in row]
        if len(set(ids)) != len(ids):
            raise ValueError(f"layer {layer}: duplicate resident ids")
        if any(i < 0 or i >= num_experts for i in ids):
            raise ValueError(f"layer {layer}: resident id out of range [0, {num_experts})")
        out.append(sorted(ids))
    return out


def load_frequency_table(path: str) -> torch.Tensor:
    """Read a ``[num_layers, num_experts]`` count table (.pt/.pth/.npy/.json)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pt", ".pth"):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(data, Mapping):
            for key in ("counts", "activation_freq", "freq", "data"):
                if key in data:
                    data = data[key]
                    break
            # tools/kt_offload/expert_count_report.py writes counts={decode, prefill, all}
            if isinstance(data, Mapping) and "all" in data:
                data = data["all"]
    elif ext == ".npy":
        import numpy as np

        data = torch.from_numpy(np.load(path))
    elif ext == ".json":
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, Mapping):
            data = data.get("counts", data.get("freq"))
        data = torch.tensor(data)
    else:
        raise ValueError(f"unsupported frequency table format: {path}")
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"{path}: expected a tensor of counts, got {type(data).__name__}")
    return data


def frequency_table_report(freq: torch.Tensor, resident_sets: list[list[int]], top_k: int) -> dict:
    """Numbers to log with a frequency placement, so a bad table is visible.

    * ``row_sums`` should be identical across layers (every layer routes the
      same tokens) and a multiple of ``top_k``; unequal sums mean the table
      was aggregated wrongly (e.g. a DP rank missing for some layers).
    * ``npu_hit_rate[L]`` = share of the table's routes the resident set of
      layer L covers -- the NPU share the placement predicts for that workload.
    """
    f = freq.detach().to("cpu", torch.float64)
    sums = f.sum(dim=1)
    hits = [
        float(f[layer, ids].sum() / sums[layer]) if sums[layer] > 0 else float("nan")
        for layer, ids in enumerate(resident_sets)
    ]
    return {
        "row_sums_equal": bool(torch.all(sums == sums[0])),
        "row_sum_min": float(sums.min()),
        "row_sum_max": float(sums.max()),
        "row_sums_multiple_of_topk": bool(torch.all(torch.remainder(sums, top_k) == 0)),
        "empty_layers": [i for i, s in enumerate(sums.tolist()) if s == 0],
        "npu_hit_rate_min": min(hits),
        "npu_hit_rate_mean": sum(hits) / len(hits),
        "npu_hit_rate": hits,
    }


@dataclass(frozen=True)
class OffloadPlan:
    num_layers: int
    num_experts: int
    num_slots: int
    layers: tuple[LayerPlacement, ...]
    report: dict = field(default_factory=dict, compare=False)

    def layer(self, layer_idx: int) -> LayerPlacement:
        return self.layers[layer_idx]


def build_plan(
    resident_sets: list[list[int]],
    num_experts: int,
    report: dict | None = None,
) -> OffloadPlan:
    num_layers = len(resident_sets)
    num_slots = max(len(s) for s in resident_sets)
    layers = tuple(
        LayerPlacement(layer_idx=i, num_experts=num_experts, num_slots=num_slots, resident_ids=tuple(s))
        for i, s in enumerate(resident_sets)
    )
    return OffloadPlan(num_layers, num_experts, num_slots, layers, report or {})


def plan_from_config(cfg, num_layers: int, num_experts: int, top_k: int) -> OffloadPlan:
    """Resolve ``KTOffloadConfig`` into a plan. Reads the table/list files."""
    report: dict = {"strategy": cfg.resident_strategy}
    if cfg.resident_strategy == "prefix":
        sets = prefix_resident_sets(num_layers, num_experts, cfg.num_resident_experts)
    elif cfg.resident_strategy == "frequency":
        freq = load_frequency_table(cfg.frequency_table_path)
        if tuple(freq.shape) != (num_layers, num_experts):
            raise ValueError(
                f"{cfg.frequency_table_path}: shape {tuple(freq.shape)}, expected [{num_layers}, {num_experts}] "
                "(main-model layers x global expert ids)."
            )
        sets = frequency_resident_sets(freq, cfg.num_resident_experts)
        report.update(frequency_table_report(freq, sets, top_k))
        if not report["row_sums_equal"]:
            logger.warning(
                "kt_offload: frequency table rows have unequal sums (%s..%s); check how it was aggregated.",
                report["row_sum_min"],
                report["row_sum_max"],
            )
    elif cfg.resident_strategy == "explicit":
        with open(cfg.resident_experts_path) as f:
            spec = json.load(f)
        sets = explicit_resident_sets(spec, num_layers, num_experts)
    else:
        raise ValueError(f"unknown resident_strategy {cfg.resident_strategy!r}")
    report["resident_per_layer_min"] = min(len(s) for s in sets)
    report["resident_per_layer_max"] = max(len(s) for s in sets)
    return build_plan(sets, num_experts, report)


def resident_bytes_per_slot(hidden_size: int, intermediate_size: int, group_size: int = 32) -> int:
    """NPU bytes one resident expert costs in one layer (MXFP4 + E8M0 scales).

    w13 [2I, H/2] + w2 [H, I/2] uint8 nibble pairs, scales [2I, H/g] + [H, I/g].
    DSV4.1 (H=5120, I=2304): 18,800,640 B = 17.93 MiB. Excludes any padding
    the NZ format cast may add (not measured).
    """
    w = 2 * intermediate_size * hidden_size // 2 + hidden_size * intermediate_size // 2
    s = 2 * intermediate_size * hidden_size // group_size + hidden_size * intermediate_size // group_size
    return w + s


def _check_n(n: int, num_experts: int) -> int:
    n = int(n)
    if n <= 0 or n > num_experts:
        raise ValueError(f"num_resident_experts must be in [1, {num_experts}], got {n}")
    return n
