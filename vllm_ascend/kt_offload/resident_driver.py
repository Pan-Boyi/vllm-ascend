# SPDX-License-Identifier: Apache-2.0
"""The swap driver: when to change the resident set, which experts, and who commits it.

``resident_swap.py`` changes one layer atomically; this module calls it, from two
call sites that must stay apart:

``decide(step)``  -- inside a forward (the prefill forward that holds the routing
    counts). Reads the heat, asks the policy for each layer's new resident set,
    checks the proposal and parks it; writes nothing. Refuses unless the current
    stream is **provably** not being captured (``capture_state``): a captured
    decision is taken once and re-asserted by every replay that never ran it.
``commit(decision)`` -- at the **step boundary**, no forward in flight. The only
    place bytes move: pulls each layer's new weights and hands them to
    ``ResidentSwapCommitter``.

The decision needs a live forward (that is where the counts and the capture state
are readable); the commit needs the absence of one (a resident parameter must not
be rewritten under a GEMM that is reading it). ``maybe_swap()`` wraps both for a
caller already at a step boundary.

The weights are fetched with ``StreamSlotProvider.slot_for`` -- the same loader
the streaming prefill uses, in the same post-load layout -- so a promoted
expert's bytes cross the PCIe link twice. That provider owns **one** device slot
and every chunk overwrites it, so the driver must pull and commit one layer at a
time (layer 1's pull would destroy layer 0's weights before the commit read them)
and cannot call ``ResidentSwapCommitter.commit_step``, which takes every layer's
weights at once, i.e. assumes 40 live slots; the step-level zero-coverage rule
``commit_step`` enforces is reproduced here instead.

The committer is all-or-nothing **per layer**, not per step: if layer 7 fails
after layers 0..6 committed, those six stay in the new state. A failed layer is
repaired by rolling **forward**, not back -- ``clone()`` is refused outright on an
internal (NZ) tensor, so a device-side weight snapshot never existed and there is
no "undo". The driver therefore:

* **never rolls back and never repairs a layer itself**: repair inside a layer is
  the committer's (rewrite the stages that did not land; re-pull through
  ``weight_refetch``), and the driver only supplies that callback because it
  holds the provider.
* **never assumes which state a failed layer is in**: a repaired layer is *new*,
  one whose repair also failed is unserviceable and raises, and both are read
  back from ``committer.placement(layer)`` and reported in ``placements_after``.
* **stops at the first unrepaired layer and halts** (``self.halted``) -- every
  later ``decide`` / ``commit`` raises until a human clears it -- because that
  layer is already unserviceable and what fails is almost always shared machinery
  (the host pull, the one device write channel), so continuing turns one fault
  into forty.
* **is safe to stop half way**, because layers are independent (``placement.py``
  and kt's mask are both per layer): a mixed step is self-consistent, merely a
  placement nobody chose, which is why it must not be decided on top of.
* does **not** halt when **zero** layers committed and no layer was left
  unserviceable -- that step was atomic by arithmetic. A ``SwapCorruptedError``
  halts regardless of the count and raises in both strict modes: the damage is to
  a layer, not to the step.

Every commit reports how many (layer, expert) pairs moved, per layer and per
step; a step that commits and moves nothing raises ``ZeroCoverageError`` under
``strict``. A *decision* that proposes nothing is allowed (the resident set can
already be the top-K): counted in ``skipped["no_change"]``, never reaching the
committer -- but a whole run that never moved anything still fails
``check_coverage``. The per-step check is a cross-check, not a self-report: the
pairs the decision *asked* to move against the pairs the committer *reports*
moving, and the experts the **provider** says it produced against the ids the
driver asked for -- the independent detector, since a provider pointed at the
wrong layer produces numbers that look perfect.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend.kt_offload.placement import LayerPlacement
from vllm_ascend.kt_offload.resident_swap import (
    ALL_CPU_WEIGHTS,
    OwnershipConflictError,
    ResidentSwapCommitter,
    StepReport,
    SwapCorruptedError,
    SwapReport,
    ZeroCoverageError,
)

__all__ = [
    "SLOT_WEIGHT_ATTRS",
    "DriverError",
    "PolicyError",
    "LayerPlan",
    "TopKHeatPolicy",
    "capture_state",
    "check_proposal",
    "pinned_resident_ids",
    "plan_from_heat",
]

#: The four parameters a slot carries, in the names ``ChunkSlot`` exposes.
#: Spelled out rather than imported: ``stream_chain`` pulls in ``vllm_ascend.ops``
#: and this module must stay importable on its own. The test cross-checks it
#: against ``stream_chain.SLOT_WEIGHT_ATTRS``, so a rename there fails loudly
#: instead of silently filling nothing.
SLOT_WEIGHT_ATTRS = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")



class DriverError(RuntimeError):
    """The driver refuses: a proposal, a slot or a precondition is not usable."""


class PolicyError(DriverError):
    """The policy returned something the driver will not act on."""






# ---------------------------------------------------------------------------
# Capture state
# ---------------------------------------------------------------------------


def capture_state() -> bool | None:
    """Is the *current stream* being captured? ``True`` / ``False`` / ``None`` = cannot tell.

    The predicate is ``layer.py``'s ``_capturing``
    (``torch.npu.is_current_stream_capturing() is True``), **copied, not
    imported**: importing the model package applies the V4.1 platform patches and
    importing ``layer.py`` drags in the whole kt runtime. It reads the current
    stream, i.e. the one whose work this decision would describe, not "a capture
    is happening somewhere".

    Unlike ``layer.py`` this is three-valued: "not captured" has to be *positive*.
    No ``torch.npu``, no predicate, a predicate that raises or one that returns a
    non-bool all give ``None`` and the driver then does not decide -- so a plain
    CPU process decides nothing unless it injects a predicate.
    """
    predicate = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    if predicate is None:
        return None
    try:
        result = predicate()
    except Exception:  # noqa: BLE001 - any failure means "cannot tell"
        return None
    if result is True:
        return True
    if result is False:
        return False
    return None


# ---------------------------------------------------------------------------
# Heat
# ---------------------------------------------------------------------------




def _as_counts(heat: Any, layer: int, num_experts: int) -> torch.Tensor:
    t = torch.as_tensor(heat, dtype=torch.float64).detach().to("cpu").reshape(-1)
    if t.numel() != num_experts:
        raise DriverError(f"layer {layer}: heat has {t.numel()} entries, the layer has {num_experts} experts")
    if not bool(torch.isfinite(t).all()):
        raise DriverError(f"layer {layer}: heat contains NaN/inf")
    if bool((t < 0).any()):
        raise DriverError(f"layer {layer}: heat contains negative counts")
    return t


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TopKHeatPolicy:
    """Keep the ``num_resident`` hottest experts; ties go to the lower logical id.

    Two things this policy is *not allowed* to do, both enforced by the driver
    afterwards rather than trusted here:

    * change **how many** experts are resident. ``num_slots`` is one number for
      the whole model (the ALLGATHER dispatcher is a process-wide singleton and
      takes ``num_local_experts`` from the last layer built) and kt caches
      ``num_gpu_experts`` when it is constructed, so 99 or 101 ids would be a
      different model;
    * demote an expert kt holds no CPU weights for -- that reads an unallocated
      BufferB and produces wrong numbers without an error. Those ids arrive as
      ``pinned`` and are kept unconditionally.

    The returned order is ascending logical id, which *is* the slot order
    (``LayerPlacement`` assigns slot ``i`` to ``resident_ids[i]``), so the
    weights pulled in that order land in the right slots.

    ``max_moves`` bounds the churn per layer per step: only the hottest
    ``max_moves`` promotions happen, paying for themselves by demoting the
    coldest residents. Each costs one expert's bytes over the H2D link inside the
    step-boundary gap, so the bound is how long that gap may be. ``None`` =
    unbounded.
    """

    max_moves: int | None = None

    def __call__(
        self,
        *,
        layer: int,
        placement: LayerPlacement,
        heat: torch.Tensor,
        pinned: frozenset[int],
    ) -> tuple[int, ...]:
        k = placement.num_resident
        keep = sorted(int(e) for e in pinned)
        free = k - len(keep)
        if free < 0:
            raise PolicyError(
                f"layer {layer}: {len(keep)} experts must stay resident but only {k} slots exist"
            )
        counts = heat.tolist()
        pinned_set = set(keep)
        candidates = sorted(
            (e for e in range(placement.num_experts) if e not in pinned_set),
            key=lambda e: (-counts[e], e),
        )
        chosen = set(keep) | set(candidates[:free])
        if self.max_moves is not None:
            chosen = self._bounded(set(placement.resident_ids), chosen, counts, self.max_moves)
        return tuple(sorted(chosen))

    @staticmethod
    def _bounded(old: set[int], chosen: set[int], counts: Sequence[float], max_moves: int) -> set[int]:
        if max_moves < 0:
            raise PolicyError(f"max_moves must be >= 0, got {max_moves}")
        incoming = sorted(chosen - old, key=lambda e: (-counts[e], e))[:max_moves]
        if len(incoming) == len(chosen - old):
            return chosen
        outgoing = sorted(old - chosen, key=lambda e: (counts[e], e))[: len(incoming)]
        return (old - set(outgoing)) | set(incoming)


# ---------------------------------------------------------------------------
# Decision / reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerPlan:
    """One layer's proposed new resident set, with what it would cost and buy."""

    layer: int
    new_ids: tuple[int, ...]
    promote: tuple[int, ...]
    demote: tuple[int, ...]
    share_before: float
    share_after: float

    @property
    def moves(self) -> frozenset[tuple[int, int]]:
        """(layer, expert) pairs this plan changes the owner of."""
        return frozenset((self.layer, e) for e in self.promote + self.demote)








# ---------------------------------------------------------------------------
# One layer's plan from a heat row (shared by the driver and the folded swap)
# ---------------------------------------------------------------------------


def pinned_resident_ids(targets: Any, p: LayerPlacement) -> frozenset[int]:
    """Resident experts that must NOT be demoted, read off the commit targets.

    kt allocates a CPU-side BufferB only for the experts it was told to load
    (``amx.py``). Demoting an expert outside that set makes kt's C++ read an
    unallocated buffer: **wrong numbers, not an error**. Until the full-set load
    lands those ids are pinned here, and a policy that ignores the pin is refused
    rather than quietly narrowed (a dropped demotion would leave the count wrong
    and the promotion unpaid for).
    """
    ids = targets.kt_cpu_weight_ids
    if ids == ALL_CPU_WEIGHTS:
        return frozenset()
    return frozenset(p.resident_ids) - frozenset(ids)


def check_proposal(layer: int, p: LayerPlacement, new_ids: tuple[int, ...], pinned: frozenset[int]) -> None:
    """Refuse a policy proposal the commit could not honour, or should not."""
    if len(set(new_ids)) != len(new_ids):
        raise PolicyError(f"layer {layer}: proposal repeats an expert: {new_ids[:16]}")
    if any(e < 0 or e >= p.num_experts for e in new_ids):
        raise PolicyError(f"layer {layer}: proposal has an id outside [0, {p.num_experts})")
    if len(new_ids) != p.num_resident:
        raise PolicyError(
            f"layer {layer}: the policy proposed {len(new_ids)} resident experts, the layer has "
            f"{p.num_resident}. num_slots is one number for the whole model and kt cached "
            "num_gpu_experts when it was built; a swap changes WHICH, never HOW MANY."
        )
    if list(new_ids) != sorted(new_ids):
        raise PolicyError(
            f"layer {layer}: the proposal is not in ascending logical id order, which is the slot order "
            "the weights are pulled in (LayerPlacement gives slot i to resident_ids[i])"
        )
    lost = sorted(pinned - set(new_ids))
    if lost:
        raise PolicyError(
            f"layer {layer}: the proposal demotes {lost[:8]}, which kt holds no CPU weights for. kt would "
            "read an unallocated BufferB and return wrong numbers without an error. "
            "Refusing rather than narrowing the proposal."
        )


def plan_from_heat(
    layer: int,
    p: LayerPlacement,
    counts: torch.Tensor,
    policy: Callable[..., Sequence[int]],
    pinned: frozenset[int],
    *,
    check: Callable[[int, LayerPlacement, tuple[int, ...], frozenset[int]], None] = check_proposal,
) -> LayerPlan | None:
    """Ask ``policy`` for ``layer``'s next resident set and check it. ``None`` = no counts.

    ``counts`` is a float64 host row of length ``num_experts`` (``_as_counts``).
    A plan with no promotion and no demotion is returned as is; whether that is
    worth anything is the caller's call.
    """
    total = float(counts.sum())
    if total <= 0:
        # An all-zero row is not "every expert is equally cold", it is "this
        # layer was never counted". Top-K over it would propose experts
        # 0..K-1 and move most of the layer for nothing.
        return None
    new_ids = tuple(int(e) for e in policy(layer=layer, placement=p, heat=counts, pinned=pinned))
    check(layer, p, new_ids, pinned)
    old = set(p.resident_ids)
    new = set(new_ids)
    share_before = float(counts[sorted(old)].sum()) / total
    share_after = float(counts[sorted(new)].sum()) / total
    if share_after < share_before - 1e-12:
        raise PolicyError(
            f"layer {layer}: the proposal covers {share_after:.4f} of the counted activations, less than "
            f"the set it replaces ({share_before:.4f}). A swap that reduces coverage of the traffic it "
            "was chosen from is a policy with an inverted comparator, not a decision."
        )
    return LayerPlan(
        layer=layer,
        new_ids=new_ids,
        promote=tuple(sorted(new - old)),
        demote=tuple(sorted(old - new)),
        share_before=share_before,
        share_after=share_after,
    )


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------


