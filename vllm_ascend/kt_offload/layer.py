# SPDX-License-Identifier: Apache-2.0
"""Per-layer kt offload runtime and the hooks AscendRoutedExperts calls.

Hook contract (routed_experts.py). Every call site is guarded by
``self.kt_offload is not None``; with the feature off ``kt_offload`` is None
and no function here runs on the forward path.

  __init__       ``create_layer_offload(...)`` before ``RoutedExperts.__init__``
                 (shrinks ``moe_config.num_local_experts`` to ``num_slots`` so the
                 quant method allocates only the resident rows), then
                 ``attach_after_init(layer)``.
  weight_loader  ``_map_global_expert_id_to_local_expert_id`` -> ``slot_of``;
                 CPU experts map to -1, which the upstream loader treats as
                 "not local" and skips (vllm routed_experts.py weight_loader).
  post-load      quant_method.process_weights_after_loading is wrapped: unused
                 slots are zeroed, then the original runs, then the routing
                 tables go to the device and the kt wrapper loads its experts.
  forward_impl   ``route(...)`` right after expert selection (logical ids),
                 ``merge(...)`` right after finalize.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from typing import Any

import torch

from vllm_ascend.kt_offload.placement import LayerPlacement, OffloadPlan, plan_from_config, split_routing

logger = logging.getLogger(__name__)

# One plan per process: placement is a whole-model decision made once.
_PLAN: OffloadPlan | None = None
_PLAN_KEY: tuple | None = None


def _layer_index(layer_name: str) -> int:
    """Checkpoint layer number of ``...layers.<N>.ffn.experts``."""
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", layer_name + ".")
    if m is None:
        raise ValueError(f"kt_offload: cannot find 'layers.<N>.' in MoE layer name {layer_name!r}")
    return int(m.group(1))


def get_plan(cfg, num_layers: int, num_experts: int, top_k: int) -> OffloadPlan:
    global _PLAN, _PLAN_KEY
    key = (id(cfg), num_layers, num_experts, top_k)
    if _PLAN is None or key != _PLAN_KEY:
        _PLAN = plan_from_config(cfg, num_layers, num_experts, top_k)
        _PLAN_KEY = key
        rep = {k: v for k, v in _PLAN.report.items() if k != "npu_hit_rate"}
        logger.info(
            "kt_offload plan: %d layers x %d experts, %d NPU slots/layer; %s",
            num_layers,
            num_experts,
            _PLAN.num_slots,
            rep,
        )
    return _PLAN


def bind_routed_experts_args(init_fn, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Name the arguments AscendRoutedExperts forwards to RoutedExperts.__init__."""
    bound = inspect.signature(init_fn).bind_partial(None, *args, **kwargs)
    return dict(bound.arguments)


def offload_enabled(ascend_config) -> bool:
    """True only for an explicit ``enabled: true`` (a mock or missing section is off)."""
    cfg = getattr(ascend_config, "kt_offload_config", None)
    return cfg is not None and getattr(cfg, "enabled", False) is True


def create_layer_offload(ascend_config, init_fn, args: tuple, kwargs: dict) -> KTLayerOffload | None:
    """Build the layer's offload state, or None when the feature is off.

    Called before ``RoutedExperts.__init__``; mutates ``moe_config`` so the
    weights are allocated for ``num_slots`` experts only. With the feature off
    it returns before touching anything.
    """
    if not offload_enabled(ascend_config):
        return None
    from vllm.config import get_current_vllm_config

    cfg = ascend_config.kt_offload_config

    named = bind_routed_experts_args(init_fn, args, kwargs)
    layer_name: str = named["layer_name"]
    moe_config = named["moe_config"]
    vllm_config = get_current_vllm_config()
    hf = vllm_config.model_config.hf_text_config
    num_layers = int(hf.num_hidden_layers)
    num_experts = int(moe_config.num_logical_experts)
    top_k = int(moe_config.experts_per_token)

    layer_idx = _layer_index(layer_name)
    if layer_idx >= num_layers:
        raise ValueError(
            f"kt_offload: MoE layer {layer_name!r} is past the {num_layers} main layers (MTP/draft); "
            "offload of draft experts is not implemented."
        )
    if moe_config.num_experts != num_experts:
        raise ValueError(
            "kt_offload: redundant/shared-fused experts are not supported "
            f"(num_experts={moe_config.num_experts}, logical={num_experts})."
        )
    if moe_config.ep_size != 1 or moe_config.tp_size != 1:
        raise ValueError(f"kt_offload: needs ep_size=tp_size=1, got {moe_config.ep_size}/{moe_config.tp_size}")
    if getattr(ascend_config, "mix_placement", False):
        raise ValueError("kt_offload: mix_placement appends shared experts to the routed ids; not supported.")
    if ascend_config.eplb_config.dynamic_eplb:
        raise ValueError("kt_offload: dynamic EPLB is not supported.")

    plan = get_plan(cfg, num_layers, num_experts, top_k)
    placement = plan.layer(layer_idx)
    moe_config.num_local_experts = plan.num_slots
    return KTLayerOffload(cfg, placement, layer_idx, vllm_config)


class KTLayerOffload:
    """CPU half of one MoE layer: kt wrapper + NPU routing tables."""

    def __init__(self, cfg, placement: LayerPlacement, layer_idx: int, vllm_config):
        self.cfg = cfg
        self.placement = placement
        self.layer_idx = layer_idx
        hf = vllm_config.model_config.hf_text_config
        self.hidden_size = int(hf.hidden_size)
        self.intermediate_size = int(hf.moe_intermediate_size)
        self.top_k = int(hf.num_experts_per_tok)
        self.weight_path = cfg.weight_path or vllm_config.model_config.model
        self.chunked_prefill_size = cfg.chunked_prefill_size or int(vllm_config.scheduler_config.max_num_batched_tokens)
        self.swiglu_limit = 0.0
        self.wrapper = None
        self._slot_tables: dict[torch.dtype, torch.Tensor] = {}
        self._keep_tables: dict[torch.dtype, torch.Tensor] = {}
        self._pending: torch.Tensor | None = None

    @property
    def num_slots(self) -> int:
        return self.placement.num_slots

    # -- construction-time ------------------------------------------------
    def attach_after_init(self, layer) -> None:
        """Finish wiring once RoutedExperts.__init__ has run."""
        layer.local_num_experts = self.num_slots
        if getattr(layer, "log2phy", None) is not None:
            raise ValueError("kt_offload: a log2phy map is active; offload owns the placement.")
        # kt's clamp is off at 0.0; it must be handed the model's value
        # explicitly (DSV4.1: 10.0), or the CPU experts silently skip the clamp
        # the NPU experts apply (npu_swiglu_group_quant clamp_value).
        limit = getattr(layer.moe_config, "swiglu_limit", None)
        if limit is None:
            raise ValueError(
                "kt_offload: moe_config.swiglu_limit is unset; refusing to run the CPU experts "
                "without the clamp the NPU experts apply."
            )
        self.swiglu_limit = float(limit)

        quant_method = layer.quant_method
        original = quant_method.process_weights_after_loading

        def process_weights_after_loading(mod, _orig=original, _self=self):
            _self.zero_unused_slots(mod)
            _orig(mod)
            _self.after_weights_loaded(mod)

        # Instance attribute: the model loader calls
        # module.quant_method.process_weights_after_loading(module).
        quant_method.process_weights_after_loading = process_weights_after_loading

    def zero_unused_slots(self, layer) -> None:
        """Unused tail slots were never loaded; make their bytes defined."""
        unused = self.placement.unused_slots
        if not unused:
            return
        for name, param in layer.named_parameters(recurse=False):
            if param.dim() >= 1 and param.shape[0] == self.num_slots:
                param.data[unused[0] :].zero_()

    def after_weights_loaded(self, layer) -> None:
        """Pre-bake device tables and load the kt experts (before any capture)."""
        device = self._device_of(layer)
        p = self.placement
        self._slot_tables = {dt: p.safe_slot_table(dt).to(device) for dt in (torch.int32, torch.int64)}
        self._keep_tables = {dt: p.keep_table(dt).to(device) for dt in (torch.float32, torch.bfloat16)}
        if not p.cpu_ids:
            return  # every expert resident: nothing for kt to do
        self.wrapper = self._build_wrapper()

    @staticmethod
    def _device_of(layer) -> torch.device:
        for param in layer.parameters(recurse=False):
            return param.device
        raise RuntimeError("kt_offload: MoE layer has no parameters")

    def _build_wrapper(self):
        if self.cfg.force_sync_submit:
            os.environ["KT_FORCE_SYNC_SUBMIT"] = "1"
        from kt_kernel import KTMoEWrapper  # host-only dependency, imported when enabled

        wrapper = KTMoEWrapper(
            layer_idx=self.layer_idx,
            num_experts=self.placement.num_experts,
            num_experts_per_tok=self.top_k,
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
            gpu_experts_mask=self.placement.resident_mask(),
            cpuinfer_threads=self.cfg.cpuinfer_threads,
            threadpool_count=self.cfg.threadpool_count,
            weight_path=self.weight_path,
            chunked_prefill_size=self.chunked_prefill_size,
            method=self.cfg.method,
            numa_nodes=self.cfg.numa_nodes,
            swiglu_limit=self.swiglu_limit,
        )
        # kt indexes experts by logical id; with EPLB off physical == logical.
        wrapper.load_weights(torch.arange(self.placement.num_experts, dtype=torch.int64))
        return wrapper

    # -- forward ------------------------------------------------------------
    def route(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        in_profile_run: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Submit the CPU half with the original routing; return the NPU half.

        ``topk_ids`` must be logical ids (log2phy is refused at init), and
        this must run after the profile-run randomisation so both halves see
        the same ids.
        """
        self._pending = None
        if (
            self.wrapper is not None
            and hidden_states.shape[0] > 0
            and not (in_profile_run and self.cfg.skip_cpu_in_profile_run)
        ):
            if hidden_states.dtype != torch.bfloat16:
                raise TypeError(f"kt_offload: kt takes bf16 activations, got {hidden_states.dtype}")
            hs = hidden_states.contiguous()
            self.wrapper.submit_forward(hs, topk_ids, topk_weights, _stream_handle(hs.device))
            self._pending = hs
        return split_routing(
            topk_ids,
            topk_weights,
            self._slot_tables[topk_ids.dtype],
            self._keep_tables[topk_weights.dtype],
        )

    def merge(self, routed_out: torch.Tensor) -> torch.Tensor:
        """Add the CPU half at the routed-output point (already scaled weights)."""
        hs, self._pending = self._pending, None
        if hs is None:
            return routed_out
        if routed_out.shape != hs.shape:
            raise AssertionError(f"kt_offload: routed_out {tuple(routed_out.shape)} vs kt input {tuple(hs.shape)}")
        cpu_out = self.wrapper.sync_forward(hs, _stream_handle(hs.device))
        # One rounding to the activation dtype instead of two.
        return (routed_out.float() + cpu_out.float()).to(routed_out.dtype)


def _stream_handle(device: torch.device) -> int:
    stream = torch.npu.current_stream(device)
    return int(stream.npu_stream)
