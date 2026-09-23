# SPDX-License-Identifier: Apache-2.0
"""Shared-expert-in-the-window: the switch, and the criteria that make it checkable.

Why this module exists
----------------------
With ``overlap_side_stream`` on, one MoE layer's main stream runs (offline
trace reconciliation, PLAN §4.30/§4.31):

  fork -> 13 resident-expert ops inside the window (~77 us) -> DAVID_EVENT_WAIT
  (the join, ~294 us of *idle* waiting for the CPU experts) -> H2D + fp32 add
  (kt merge) -> shared-expert MLP (38.25 us/layer) -> Add_BF16(shared + routed).

The shared expert reads the MoE layer's input hidden states. It depends on
neither ``routed_out`` nor the CPU experts' output: it sits behind the join
only because of call order -- ``merge()`` is called inside
``routed_experts.forward_impl`` and the shared MLP only after that function
returns (``fused_moe._forward_impl``). Moving the merge to *after* the shared
MLP puts those 38.25 us/layer into the idle window, ~1.5 ms/token at 40 layers,
and the window (77 + 38 = 115 us) stays far below the side-stream arm (~370 us),
so nothing new becomes critical path.

🔴 What can go wrong, and why a comment is not enough
-----------------------------------------------------
The merge is the *only* place the CPU half of the experts is added. Deferring
it splits "who calls it" from "who computed it", and both halves of that split
fail silently:

  **a merge that never happens** -- the CPU experts' contribution is dropped.
  Nothing raises: the tensor has the right shape and plausible values, and on
  this model the resident half alone still produces fluent text. That is the
  §4.23 failure mode (faster and wrong) with an even weaker signature.

  **a merge that happens twice** -- today the second call is a no-op because
  ``_pending`` was consumed, so it would not double-add; but the code that
  produced it is wrong, and the next change to ``merge()`` turns it into one.

  **something reading ``routed_out`` before the merge** -- the upstream runner
  adds ``shared_out + fused_output`` (vllm moe_runner.py). If the merge landed
  after that add, the CPU half would be added to a tensor nobody uses.

So the accounting is a runtime assertion (``KTLayerOffload`` counts merge calls
per route and checks the count at the next route) plus the pure predicates
below, which the unit tests drive red and the on-card case reuses.
"""

from __future__ import annotations

ENV_NAME = "VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW"


def enabled_by_env() -> bool:
    """Whether the operator explicitly set ``ENV_NAME=1``.

    NOT whether the shared-MLP move is on: since PLAN §4.33 unset means "default", and
    the effective value (cfg, env and the automatic defaults together) comes
    only from ``kt_offload.switches.resolve``. Nothing on the production path
    calls this any more; it is kept for harnesses that ask the narrower
    question.
    """
    from vllm_ascend.kt_offload import switches

    return switches.raw(ENV_NAME) == "1"


# -- the criteria -----------------------------------------------------------


def merge_balance_ok(merges: int, *, layer_idx: int = -1) -> dict:
    """Exactly one merge call per route, whoever makes it.

    ``merges`` is how many times ``merge()`` was called since this layer's last
    ``route()``. Zero means the CPU half of this layer was dropped; more than
    one means the call site is duplicated (harmless today only because the
    second call finds nothing pending).
    """
    return {
        "name": "one_merge_per_route",
        "ok": merges == 1,
        "detail": {"layer": layer_idx, "merges": merges},
        "why": (
            "a route with no merge drops this layer's CPU experts silently (right shape, plausible values, "
            "nothing raised); a route with two merges means the call site is duplicated"
        ),
    }


def window_ops_ok(ops) -> dict:
    """The deferred order, as a sequence of ``(kind, what)`` for one layer.

    Kinds:

      ``("fork", None)``    the side stream is released (kt starts on the CPU)
      ``("window", what)``  main-stream work inside the window
      ``("shared", what)``  a shared-expert op on the main stream
      ``("join", None)``    the main stream waits for the side stream
      ``("merge", None)``   ``kt_offload.merge(routed_out)``
      ``("read", what)``    anything else reading ``routed_out``
                            (the runner's ``shared_out + fused_output``)

    What it checks -- each one is a way the change fails:

      * exactly one merge (see ``merge_balance_ok``; repeated here so a
        recorded sequence alone is enough to judge);
      * every shared op is before the merge -- that is the whole point, and a
        sequence where they are not is the un-moved order;
      * when a fork is present, every shared op is inside ``fork..join``:
        after the join the window is over and nothing is hidden;
      * no ``read`` of ``routed_out`` before the merge -- the CPU half must be
        in the tensor before anyone consumes it.
    """
    problems = []
    forked = joined = merged = False
    merges = 0
    for kind, what in ops:
        if kind == "fork":
            if forked:
                problems.append("two forks in one layer")
            forked = True
        elif kind == "join":
            if not forked:
                problems.append("a join with no fork before it")
            joined = True
        elif kind == "merge":
            merges += 1
            if forked and not joined:
                problems.append("merge without the join: the H2D would read the pinned output un-filled")
            merged = True
        elif kind == "shared":
            if merged:
                problems.append(f"shared-expert op {what!r} runs AFTER the merge: it is still outside the window")
            elif forked and joined:
                problems.append(f"shared-expert op {what!r} runs after the join: the window is already over")
        elif kind == "read":
            if not merged:
                problems.append(f"{what!r} reads routed_out before the merge: the CPU half is not in it yet")
    if merges != 1:
        problems.append(f"{merges} merge(s) in this layer, expected exactly 1")
    if not any(k == "shared" for k, _ in ops):
        problems.append("no shared-expert op in the sequence at all")
    return {
        "name": "shared_inside_window",
        "ok": not problems,
        "detail": problems,
        "why": (
            "the shared MLP only buys anything while the main stream is idle waiting for the CPU experts, "
            "and the merge must still precede every reader of routed_out"
        ),
    }
