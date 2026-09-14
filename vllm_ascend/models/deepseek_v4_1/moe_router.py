# SPDX-License-Identifier: Apache-2.0
"""Make the MoE router Ascend's, at a hijack point a from-import cannot outrun.

vllm_ascend already replaces ``FusedMoEFactory``, and that replacement builds
the Ascend router. It does not reach V4.1, and moving the platform patch
earlier did not fix it, because the problem is not when the patch runs -- it is
*where the name is read*::

    deepseek_v4/nvidia/model.py:33   from ...layer import FusedMoEFactory
    deepseek_v4/nvidia/model.py:995  self.experts = FusedMoEFactory(...)

The call looks the name up in *V4.0's* globals, where the from-import froze it
at import time. Patching the attribute on ``layer`` cannot reach that, whenever
it happens, and vLLM imports that module while resolving the architecture --
before anything of ours runs.

``create_fused_moe_router`` has the same from-import shape and none of the
problem, because its only call site is inside the module that imports it::

    layer.py:31   from .router.router_factory import create_fused_moe_router
    layer.py:285  router = create_fused_moe_router(...)   <- the one caller

That lookup happens in ``layer``'s own globals, at call time. So rebinding it
works no matter when we get there, as long as it is before a model is built.
The distinction worth keeping: a from-import snapshots the *object*, but the
name in the importing module is still resolved on every call -- what matters is
whose namespace the call site reads, not how the name got there.

The call site forwards ``hash_indices_table``, ``bias_vl`` and
``image_sentinel_lo``, so the hash routing V4.1 needs is reachable from here.
"""

from __future__ import annotations

import inspect
from typing import Any

#: Upstream's factory takes this and Ascend's does not. Dropping it is only
#: safe while it is at its default; see below.
_UPSTREAM_ONLY = ("zero_expert_type",)


def _upstream_defaults() -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.router.router_factory import \
        create_fused_moe_router

    return {name: p.default
            for name, p in inspect.signature(create_fused_moe_router).parameters.items()}


def create_ascend_router_for_upstream(**kwargs):
    """Upstream's ``create_fused_moe_router`` signature, Ascend's routers.

    Arguments Ascend's factory does not take are dropped, but only when they
    are still at upstream's default -- a caller that set one is asking for
    behaviour this cannot provide, and dropping it silently would route with
    the wrong rule and no error.
    """
    from vllm_ascend.ops.fused_moe.router.router_factory import \
        create_ascend_fused_moe_router

    accepted = set(inspect.signature(create_ascend_fused_moe_router).parameters)
    defaults = _upstream_defaults()

    ignored = {}
    for name in kwargs:
        if name in accepted:
            continue
        value = kwargs[name]
        if name in defaults and value == defaults[name]:
            continue
        ignored[name] = value

    if ignored:
        raise ValueError(
            "the Ascend MoE router factory does not take "
            f"{sorted(ignored)}, and the caller set them to {ignored}. "
            "Passing them on is impossible and dropping them would route by a "
            "different rule than the caller asked for, with no error. The "
            "Ascend factory needs extending before this configuration can run.")

    router = create_ascend_fused_moe_router(
        **{k: v for k, v in kwargs.items() if k in accepted})
    _report_router_arguments(kwargs, accepted)
    return _tap_select_experts(router)


#: Arguments whose **default is a behaviour**. Each is a value the caller must
#: set for this model and gets wrong by omission, silently. Two have already
#: cost a flight in this family -- ``apply_q_norm`` (default True, V4.1 needs
#: False) and ``swiglu_limit`` (default 0.0 = no clamp) -- so these are
#: reported with what actually arrived rather than checked against a guess.
#:
#: ⚠️ ``routed_scaling_factor`` defaults to 1.0 and this model's config says
#: 1.5. Static reading says upstream passes it; this line is the runtime
#: evidence, which static reading is not.
_BEHAVIOURAL_DEFAULTS = (
    "top_k", "global_num_experts", "scoring_func", "renormalize",
    "routed_scaling_factor", "num_fused_shared_experts",
    "shared_expert_weight", "use_grouped_topk", "num_expert_group",
    "topk_group",
)

_REPORTED_ARGUMENTS = False


def _report_router_arguments(kwargs: dict, accepted: set) -> None:
    """The values this router was actually built with, once.

    🔴 Not a check. A check needs a known-right value to compare against, and
    for half of these nobody has written one down; what was missing was the
    *reading*. Reported at the factory's exit because that is the one point
    where the caller's intent and the object that will act on it meet.
    """
    global _REPORTED_ARGUMENTS
    if _REPORTED_ARGUMENTS:
        return
    _REPORTED_ARGUMENTS = True
    from vllm.logger import logger

    # 🔴 Every argument, not a hand-picked list. The first version printed
    # only _BEHAVIOURAL_DEFAULTS, and a reader went looking for
    # e_score_correction_bias in the output: it was in neither column --
    # not "dropped", because the Ascend factory accepts it, and not "seen",
    # because it was not on my list. An instrument that shows a subset chosen
    # in advance cannot answer a question nobody had when the list was
    # written, and this line exists precisely for those questions.
    #
    # Tensors are summarised rather than dumped: what a reader needs from one
    # here is whether it arrived at all and what shape it is.
    def _describe(value):
        shape = getattr(value, "shape", None)
        if shape is not None:
            return f"<tensor {tuple(shape)} {getattr(value, 'dtype', '?')}>"
        return value

    passed = {name: _describe(value) for name, value in sorted(kwargs.items())}
    not_passed = sorted(name for name in _BEHAVIOURAL_DEFAULTS
                        if name not in kwargs)
    dropped = sorted(set(kwargs) - accepted)
    logger.info(
        "DSV4.1 MoE router built with ALL arguments as passed: %s. Of the "
        "ones whose default is a behaviour, these were NOT passed and so took "
        "the default: %s. Arguments upstream passed that the Ascend factory "
        "does not take (all at upstream's default, or this would have "
        "raised): %s.", passed, not_passed or "none", dropped or "none")


def _tap_select_experts(router):
    """Record the routing decision at the one door both MoE paths use.

    🔴 Why here and not on the gate module. ``moe_gate.out`` was first hooked
    on ``ffn.gate``, and that hook can never fire on this platform: the fused
    layer does not call the gate as a module at all, it reads ``gate.weight``
    and does the matmul itself (ops/fused_moe/fused_moe.py:306-312). "It did
    not fire this time" and "it can never fire" need different fixes -- wait
    for the condition, or move the hook -- and only tracing who actually does
    the work separates them.

    🔴 And not on ``select_experts`` either, which was the second wrong
    answer. That is the *public* method; Ascend's routed experts call the
    private one directly -- ``self.router._select_experts(...)``
    (ops/fused_moe/routed_experts.py:571). ``select_experts`` calls
    ``_select_experts`` too, so wrapping the private one covers both callers
    and the public method needs no wrapper of its own.

    ⚠️ Two peers and I all said "this path goes through no router method at
    all", reading a block that ends in ``routed_experts.forward_impl(...)``.
    The routing does go through a router method; the call is one frame
    further in. The conclusion was one step past the evidence, and the
    correction is a line number, not an opinion.

    ⚠️ The wrapper also counts its own calls. An absent key has three causes
    -- no alignment run, this path not taken, the hook attached to something
    that is never called -- and the third is what bit the gate hook. An
    attribute test cannot see it; an accounting bit can.
    """
    import torch

    original = router._select_experts

    def _wrapped(*args, **kwargs):
        weights, ids = original(*args, **kwargs)
        global _ROUTER_CALLS
        _ROUTER_CALLS += 1
        try:
            from vllm_ascend.models.deepseek_v4_1 import align_taps

            if align_taps.alignment_active():
                layer = align_taps.current_layer()
                if layer is not None:
                    # golden's own key names, so the two dumps need no
                    # mapping: .0 is the normalised top-k weights and .1 the
                    # expert ids.
                    align_taps.tap(layer, "moe_gate.out.0", weights,
                                   align_taps.current_positions())
                    align_taps.tap(layer, "moe_gate.out.1", ids,
                                   align_taps.current_positions())
                else:
                    _report_no_layer()
        except Exception:                                    # noqa: BLE001
            # Instrument. It has already cost this line one flight to let one
            # of these propagate.
            pass
        return weights, ids

    router._select_experts = _wrapped
    return router


_ROUTER_CALLS = 0
_REPORTED_NO_LAYER = False


def router_tap_calls() -> int:
    """How many times the routing tap has run. Zero is a reading, not a gap."""
    return _ROUTER_CALLS


def reset_router_tap_for_test() -> None:
    global _ROUTER_CALLS, _REPORTED_ARGUMENTS, _REPORTED_NO_LAYER
    _ROUTER_CALLS = 0
    _REPORTED_ARGUMENTS = False
    _REPORTED_NO_LAYER = False


def _report_no_layer() -> None:
    global _REPORTED_NO_LAYER
    if _REPORTED_NO_LAYER:
        return
    _REPORTED_NO_LAYER = True
    from vllm.logger import logger

    logger.warning(
        "DSV4.1 MoE router tap ran outside any decoder layer's forward, so it "
        "cannot say which layer the routing belongs to and recorded nothing. "
        "A key under the wrong layer would be worse than an absent one.")


def apply_moe_router_patch() -> bool:
    """Rebind ``create_fused_moe_router`` in ``layer``. Idempotent.

    Returns True if it changed anything.
    """
    import vllm.model_executor.layers.fused_moe.layer as layer

    name = "create_fused_moe_router"
    if not hasattr(layer, name):
        raise AttributeError(
            f"{layer.__name__} no longer defines {name!r}. Until this is "
            "repointed the MoE routes through upstream's router, which calls "
            "topk_softplus_sqrt in vLLM's CUDA _moe_C extension -- absent "
            "here, so it fails in the first forward through the MoE with the "
            "Ascend equivalents present and unused.")
    if getattr(layer, name) is create_ascend_router_for_upstream:
        return False
    setattr(layer, name, create_ascend_router_for_upstream)
    return True
