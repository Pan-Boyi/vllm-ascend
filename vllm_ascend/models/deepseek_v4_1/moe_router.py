# SPDX-License-Identifier: Apache-2.0
"""Route DeepSeek-V4.1's MoE through Ascend's router.

Replacing ``FusedMoEFactory`` does not reach V4.x: ``deepseek_v4/nvidia/model.py``
from-imports it, freezing the name into that module's globals before anything
here runs. ``create_fused_moe_router`` is different: its only call site is in
``fused_moe/layer.py``, the module that imports it, so the name is looked up in
``layer``'s globals at call time and rebinding it there works whenever it
happens before a model is built.

The call site forwards ``hash_indices_table``, ``bias_vl`` and
``image_sentinel_lo``, so the hash routing V4.1 needs is reachable from here.
"""

from __future__ import annotations

import inspect
from typing import Any


def _upstream_defaults() -> dict[str, Any]:
    from vllm.model_executor.layers.fused_moe.router.router_factory import \
        create_fused_moe_router

    return {name: p.default
            for name, p in inspect.signature(create_fused_moe_router).parameters.items()}


def create_ascend_router_for_upstream(**kwargs):
    """Upstream's ``create_fused_moe_router`` signature, Ascend's routers.

    Arguments Ascend's factory does not take are dropped only while they are
    at upstream's default; a caller that set one is asking for a routing rule
    this cannot provide.
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
            "Dropping them would route by a different rule than the caller "
            "asked for; the Ascend factory needs extending to support them.")

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
            f"{layer.__name__} no longer defines {name!r}. Without this hook the "
            "MoE routes through upstream's router, which calls "
            "topk_softplus_sqrt from vLLM's CUDA _moe_C extension and fails in "
            "the first MoE forward.")
    if getattr(layer, name) is create_ascend_router_for_upstream:
        return False
    setattr(layer, name, create_ascend_router_for_upstream)
    return True
