# SPDX-License-Identifier: Apache-2.0
"""Shared-expert-in-the-window: the switch, and the criteria that make it checkable.

With ``overlap_side_stream`` on, a MoE layer's main stream forks the side
stream, runs the resident experts, then waits at the join for the CPU experts
before the kt merge and the shared-expert MLP. The shared expert depends on
neither ``routed_out`` nor the CPU experts -- it sits behind the join only
because of call order, so moving the merge to *after* the shared MLP puts the
shared MLP into the idle window. That window stays far shorter than the
side-stream arm, so nothing new becomes critical path.

Both halves fail silently, hence a runtime assertion rather than a comment:

  * a merge that never happens drops the CPU experts' contribution. Nothing
    raises -- right shape, plausible values, still-fluent text.
  * a merge that happens twice is a no-op today only because ``_pending`` was
    consumed; the next change to ``merge()`` turns it into a double-add.
  * anything reading ``routed_out`` before the merge (the upstream runner adds
    ``shared_out + fused_output``) sees the resident half only.

``KTLayerOffload`` counts merge calls per route and checks at the next route;
the pure predicates below are what the unit tests drive.
"""

from __future__ import annotations

ENV_NAME = "VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW"


def enabled_by_env() -> bool:
    """Whether the operator explicitly set ``ENV_NAME=1``.

    NOT whether the shared-MLP move is on: unset means "default", and the
    effective value comes only from ``kt_offload.switches.resolve``. Nothing on
    the production path calls this; it is for harnesses asking that question.
    """
    from vllm_ascend.kt_offload import switches

    return switches.raw(ENV_NAME) == "1"


# -- the criteria -----------------------------------------------------------


def merge_balance_ok(merges: int, *, layer_idx: int = -1) -> dict:
    """Exactly one merge call per route, whoever makes it.

    ``merges`` is how many times ``merge()`` was called since this layer's last
    ``route()``. Zero means this layer's CPU half was dropped; more than one
    means the call site is duplicated.
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

    Requires exactly one merge, every shared op inside ``fork..join`` and before
    the merge, and no read of ``routed_out`` before the merge.
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
