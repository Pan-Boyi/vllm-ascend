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

    return create_ascend_fused_moe_router(
        **{k: v for k, v in kwargs.items() if k in accepted})


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
