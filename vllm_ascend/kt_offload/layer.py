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

Two execution paths, chosen per forward by whether the current stream is
being captured (the same predicate the engram host gather uses):

  eager    kt ``submit_forward`` / ``sync_forward`` (host blocks, device syncs).
  captured D2H into kt's pinned buffers, a ``_launch_host_func`` node that runs
           ``run_pinned_forward_sync`` at every replay, H2D back. Nothing on
           this path may depend on a runtime condition: whatever the capture
           run skips is absent from every replay.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from typing import Any

import torch

from vllm_ascend.kt_offload.placement import (
    LayerPlacement,
    NpuRouting,
    OffloadPlan,
    assert_routing_dtypes,
    plan_from_config,
    split_routing,
)

logger = logging.getLogger(__name__)

# One plan per process: placement is a whole-model decision made once.
_PLAN: OffloadPlan | None = None
_PLAN_KEY: tuple | None = None

# Captured graphs hold host-func nodes whose payloads are read at every
# replay; they must live for the whole process.
_CAPTURED_PAYLOADS: list = []
# layer_idx -> number of times its host-func node has run (replays + the
# breakable wrapper's post-capture replay). Read by the on-card checks.
HOST_FUNC_CALLS: dict[int, int] = {}
# kt's pinned/device buffers for the capture sizes are process-wide
# (KExpertsCPUBuffer is class-level); they are registered and allocated once.
_CAPTURE_BUFFERS_READY: tuple | None = None
# One side stream for the whole process, created outside any capture.
_SIDE_STREAM = None
# A captured graph keeps record/wait nodes pointing at these event objects, so
# every pair stays alive for the life of the process (sglang lost a capture to
# collected events).
_SIDE_EVENTS: list = []


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
        self._ids_i64: dict[int, torch.Tensor] = {}
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
        self._pending_captured = False
        self._pending_join: tuple | None = None
        self.overlap = bool(getattr(cfg, "overlap_side_stream", False))
        # Graph mode: capture sizes are known from the config before any
        # capture; kt needs them registered and their buffers allocated first.
        mc, cc = vllm_config.model_config, getattr(vllm_config, "compilation_config", None)
        mode = getattr(cc, "cudagraph_mode", None)
        self.graph_enabled = (
            not getattr(mc, "enforce_eager", True) and mode is not None and getattr(mode, "name", str(mode)) != "NONE"
        )
        sizes = getattr(cc, "cudagraph_capture_sizes", None) or []
        self.capture_sizes = sorted(set(sizes)) if self.graph_enabled else []

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
        if self.overlap:
            # Created here, never inside a capture, and subscribed for host-func
            # reports before any capture can reach it.
            side = _side_stream(device)
            if self.graph_enabled:
                _subscribe_stream(side)
            if not self._overlap_supported():
                raise RuntimeError(
                    "kt_offload: overlap_side_stream needs kt's copy_inputs_to_cpu_buffers / "
                    "run_pinned_forward_sync / copy_forward_output_to_device; "
                    f"this kt build ({type(self.wrapper).__name__}) does not have them."
                )
        if self.graph_enabled:
            self._prepare_capture_buffers(device)

    def _overlap_supported(self) -> bool:
        """The captured overlap path needs only the graph-mode kt entry points."""
        w = self.wrapper
        return all(
            hasattr(w, name)
            for name in ("copy_inputs_to_cpu_buffers", "run_pinned_forward_sync", "copy_forward_output_to_device")
        )

    def _prepare_capture_buffers(self, device: torch.device) -> None:
        """Register every capture size with kt and allocate its buffers now.

        Only registered sizes are kept by KExpertsCPUBuffer, and allocating
        pinned host memory or a new device buffer inside a capture is not an
        option. Done once per process (the buffers are class-level in kt).
        """
        global _CAPTURE_BUFFERS_READY
        key = (tuple(self.capture_sizes), self.hidden_size, self.top_k, str(device))
        if key == _CAPTURE_BUFFERS_READY:
            return
        if _CAPTURE_BUFFERS_READY is not None:
            raise RuntimeError(f"kt_offload: capture buffers already prepared for {_CAPTURE_BUFFERS_READY}, now {key}")
        too_big = [bs for bs in self.capture_sizes if bs > self.chunked_prefill_size]
        if too_big:
            raise ValueError(
                f"kt_offload: capture sizes {too_big} exceed chunked_prefill_size {self.chunked_prefill_size}"
            )
        kt_buffer = _kt_buffer_class()
        type(self.wrapper).set_capture_batch_sizes(list(self.capture_sizes))
        for bs in self.capture_sizes:
            kt_buffer.get_buffer(torch.empty((bs, self.hidden_size), dtype=torch.bfloat16, device=device), self.top_k)
        # kt casts the ids to int64 before its D2H. Under capture that cast
        # would allocate on whichever stream we are enqueueing to; on the side
        # stream the allocation is not the capture stream's, so whether it comes
        # from the graph's private pool is not ours to assume. Hand kt ids that
        # are already int64, from a buffer allocated here, before any capture:
        # kt's .to(torch.long) is then the identity and allocates nothing.
        self._ids_i64 = {
            bs: torch.empty((bs, self.top_k), dtype=torch.int64, device=device) for bs in self.capture_sizes
        }
        missing = [bs for bs in self.capture_sizes if bs not in kt_buffer.capture_buffers]
        if missing:
            raise RuntimeError(f"kt_offload: kt did not keep capture buffers for sizes {missing}")
        _CAPTURE_BUFFERS_READY = key
        logger.info(
            "kt_offload: kt capture buffers ready for %d sizes (max %d)",
            len(self.capture_sizes),
            max(self.capture_sizes, default=0),
        )

    @staticmethod
    def _device_of(layer) -> torch.device:
        for param in layer.parameters(recurse=False):
            return param.device
        raise RuntimeError("kt_offload: MoE layer has no parameters")

    def _build_wrapper(self):
        if self.cfg.force_sync_submit:
            os.environ["KT_FORCE_SYNC_SUBMIT"] = "1"
        if self.graph_enabled:
            # We subscribe the capture stream ourselves (_subscribe_stream);
            # kt must not start its own poller or subscribe it a second time.
            os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"
        from kt_kernel import KTMoEWrapper  # host-only dependency, imported when enabled

        _install_capture_probe(KTMoEWrapper)

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
    ) -> NpuRouting:
        """Submit the CPU half with the original routing; return the NPU half.

        ``topk_ids`` must be logical ids (log2phy is refused at init), and
        this must run after the profile-run randomisation so both halves see
        the same ids. The result is a named pair, so a call site cannot bind
        the two tensors the wrong way round.
        """
        assert_routing_dtypes(topk_ids, topk_weights, "kt_offload route")
        for name, dtype, baked in (
            ("topk_ids", topk_ids.dtype, self._slot_tables),
            ("topk_weights", topk_weights.dtype, self._keep_tables),
        ):
            if baked and dtype not in baked:
                raise TypeError(
                    f"kt_offload route: no {name} table baked for {dtype}; the placement tables were built "
                    f"for {sorted(str(d) for d in baked)} in process_weights_after_loading."
                )
        self._pending = None
        self._pending_captured = False
        # Clear any join left by a previous capture on every route, not only in
        # the captured branch: a forward that captured and then runs eager would
        # otherwise carry that stale cross-stream dependency (sglang 18c8b21e3).
        self._pending_join = None
        if self.wrapper is not None and hidden_states.shape[0] > 0:
            if hidden_states.dtype != torch.bfloat16:
                raise TypeError(f"kt_offload: kt takes bf16 activations, got {hidden_states.dtype}")
            if _capturing():
                # No runtime condition here (not even the profile flag): a
                # branch the capture run skips is missing from every replay.
                hs = hidden_states.contiguous()
                if self.overlap:
                    self._submit_captured_forked(hs, topk_ids, topk_weights)
                else:
                    self._submit_captured(hs, topk_ids, topk_weights)
                self._pending, self._pending_captured = hs, True
            elif not (in_profile_run and self.cfg.skip_cpu_in_profile_run):
                # Eager is never forked, whatever overlap_side_stream says (the
                # reference implementation does the same, and see the note on
                # _submit_eager_forked for why a fork buys nothing here).
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
        if self._pending_captured:
            self._pending_captured = False
            # The side stream's host-func node must have returned before the
            # H2D reads the pinned output: join first, at every replay.
            self._join_side_stream(hs.device)
            # H2D from the pinned output the host-func node filled; stream
            # order puts it after that node at every replay.
            cpu_out = self.wrapper.copy_forward_output_to_device(hs)
        else:
            cpu_out = self.wrapper.sync_forward(hs, _stream_handle(hs.device))
        # One rounding to the activation dtype instead of two.
        return (routed_out.float() + cpu_out.float()).to(routed_out.dtype)

    def _submit_captured(self, hs: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
        """Record D2H + host-func for this layer into the graph being captured."""
        if not self.graph_enabled:
            raise RuntimeError("kt_offload: a capture reached a layer built for eager mode (enforce_eager on?)")
        bs = hs.shape[0]
        if bs not in _kt_buffer_class().capture_buffers:
            raise RuntimeError(
                f"kt_offload: capturing {bs} tokens but kt has no pre-allocated buffers for that size "
                f"(prepared: {self.capture_sizes}). Allocating inside the capture is not possible."
            )
        stream = torch.npu.current_stream(hs.device)
        _subscribe_stream(stream)
        self.wrapper.copy_inputs_to_cpu_buffers(hs, topk_ids, topk_weights)
        # kt reads only the shape of this tensor at replay; a meta tensor keeps
        # no graph-pool memory alive.
        shape_ref = torch.empty(tuple(hs.shape), dtype=hs.dtype, device="meta")
        payload = (self.wrapper, shape_ref, int(stream.npu_stream), self.layer_idx)
        _CAPTURED_PAYLOADS.append(payload)
        _launch_host_func(stream, _kt_host_forward, payload)

    # -- side-stream overlap (cfg.overlap_side_stream) ----------------------

    # Why there is no eager fork (2026-09-20). A first version forked the eager
    # path too, replacing kt's device-wide torch.npu.synchronize() with
    # side.synchronize(). It buys nothing: the side stream's first op is a wait
    # on an event recorded on the compute stream *after* the previous layer's
    # H2D, so waiting for the side stream transitively waits for the compute
    # stream to reach that point anyway (tests/ut/kt_offload/
    # test_side_stream_overlap.py models it). It also puts kt's pinned-slot
    # reuse (slot = layer_idx % 2) and its temp-buffer reallocation on a bs
    # change outside the protection of that device sync. The reference
    # implementation forks only in the captured path; so do we.

    def _submit_captured_forked(self, hs: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
        """Capture the D2H + host-func on the side stream, to be joined in merge()."""
        if not self.graph_enabled:
            raise RuntimeError("kt_offload: a capture reached a layer built for eager mode (enforce_eager on?)")
        bs = hs.shape[0]
        if bs not in _kt_buffer_class().capture_buffers:
            raise RuntimeError(
                f"kt_offload: capturing {bs} tokens but kt has no pre-allocated buffers for that size "
                f"(prepared: {self.capture_sizes}). Allocating inside the capture is not possible."
            )
        device = hs.device
        compute, side = torch.npu.current_stream(device), _side_stream(device)
        # A forward that captured and then runs eager would otherwise leave the
        # previous capture's join here for this forward to wait on.
        self._pending_join = None
        fork, join = _new_event(device), _new_event(device)
        _SIDE_EVENTS.append((fork, join))  # the graph's nodes reference these
        fork.record(compute)
        side.wait_event(fork)
        shape_ref = torch.empty(tuple(hs.shape), dtype=hs.dtype, device="meta")
        payload = (self.wrapper, shape_ref, _handle_of(side), self.layer_idx)
        _CAPTURED_PAYLOADS.append(payload)
        ids_for_kt = self._ids_for_kt(topk_ids)
        with _stream_ctx(side, device):
            # The capture follows the event wait onto the side stream. If it did
            # not, everything below would run once, at capture time, and every
            # replay would reuse that one CPU output: faster and wrong, which no
            # timing check can see. Refuse instead.
            if not _capturing():
                raise RuntimeError(
                    "kt_offload: the graph capture did not propagate to the side stream, so the CPU-expert "
                    "D2H and host-func node would not be recorded (replays would reuse the capture run's "
                    "output). Run with overlap_side_stream=false, or with a torch_npu whose capture follows "
                    "an event wait onto another stream."
                )
            self.wrapper.copy_inputs_to_cpu_buffers(hs, ids_for_kt, topk_weights)
            _launch_host_func(side, _kt_host_forward, payload)
        self._pending_join = (join, side, compute)

    def _ids_for_kt(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """int64 ids in a buffer that predates the capture (see _prepare_capture_buffers).

        The copy is enqueued on the caller's stream (the compute stream), not
        inside the side-stream context, so nothing allocates or writes there.
        """
        if topk_ids.dtype == torch.int64:
            return topk_ids
        buf = self._ids_i64.get(topk_ids.shape[0])
        if buf is None:
            raise RuntimeError(
                f"kt_offload: no preallocated int64 id buffer for {topk_ids.shape[0]} tokens "
                f"(prepared: {sorted(self._ids_i64)}); a cast inside the capture would allocate."
            )
        buf.copy_(topk_ids)
        return buf

    def _join_side_stream(self, device) -> None:
        """Make the compute stream wait for this layer's side-stream work."""
        pending, self._pending_join = self._pending_join, None
        if pending is None:
            if self.overlap:
                raise AssertionError(
                    "kt_offload: merge() found no side-stream join to record although overlap is on; "
                    "the H2D would read the pinned output before the host func filled it."
                )
            return
        join, side, compute = pending
        join.record(side)
        compute.wait_event(join)


def _kt_host_forward(payload) -> None:
    """Graph-node callback: run this layer's CPU experts on the pinned buffers.

    Exceptions must not escape (nothing on the ACL side receives them); the
    device would then read the previous step's CPU output. Log loudly.
    """
    wrapper, shape_ref, stream_handle, layer_idx = payload
    try:
        wrapper.run_pinned_forward_sync(shape_ref, stream_handle)
        HOST_FUNC_CALLS[layer_idx] = HOST_FUNC_CALLS.get(layer_idx, 0) + 1
    except BaseException as exc:  # noqa: BLE001
        logger.error(
            "kt_offload host-func for layer %d raised inside a graph node: %r. The CPU-expert output "
            "of this step is NOT this step's -- the device reads whatever the pinned buffer held.",
            layer_idx,
            exc,
        )


# -- platform indirections (monkeypatched by the CPU unit tests) -------------


def _capturing() -> bool:
    """Is the current stream being captured?

    Same predicate as the engram host gather
    (models/deepseek_v4_1/engram_embedding.py _stream_is_capturing). Copied,
    not imported: importing that package applies the V4.1 platform patches.
    ``is True`` so a test double returning a Mock does not read as capturing.
    """
    predicate = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    if predicate is None:
        return False
    return predicate() is True


_REPORT_SUBSCRIBED: set[int] = set()


def _subscribe_stream(stream) -> None:
    """Subscribe the capture stream for host-func reports; swallow only 107011.

    torch_npu >= 2.10 already subscribes a capture stream inside NPUGraph, and
    engram may have subscribed it too; a repeat subscribe raises 107011, which
    means "already subscribed". Anything else propagates.
    """
    import torch_npu

    key = int(stream.npu_stream)
    if key in _REPORT_SUBSCRIBED:
        return
    try:
        torch_npu.npu._subscribe_report(stream)
    except RuntimeError as exc:
        if "107011" not in str(exc):
            raise
    _REPORT_SUBSCRIBED.add(key)


def _launch_host_func(stream, fn, payload) -> None:
    import torch_npu

    torch_npu.npu._launch_host_func(stream, fn, payload)


def _side_stream(device):
    """The one side stream of this process, created outside any capture."""
    global _SIDE_STREAM
    if _SIDE_STREAM is None:
        _SIDE_STREAM = _new_stream(device)
    return _SIDE_STREAM


def _new_stream(device):
    return torch.npu.Stream(device=device)


def _new_event(device):
    del device
    return torch.npu.Event()


def _stream_ctx(stream, device):
    del device
    return torch.npu.stream(stream)


def _handle_of(stream) -> int:
    return int(stream.npu_stream)


def _kt_buffer_class():
    from kt_kernel.experts_base import KExpertsCPUBuffer

    return KExpertsCPUBuffer


def _install_capture_probe(kt_wrapper_cls) -> None:
    """Point kt's capture check at our predicate instead of sglang's flag.

    Our captured path never calls kt's _wait_device, so this is a second line
    of defence; a kt build without the hook only gets a warning.
    """
    setter = getattr(kt_wrapper_cls, "set_capture_probe", None)
    if setter is None:
        logger.warning(
            "kt_offload: this kt-kernel has no set_capture_probe (needs ktransformers 292be81+); "
            "kt's own device sync falls back to torch.npu.is_current_stream_capturing() alone."
        )
        return
    setter(_capturing)


def _stream_handle(device: torch.device) -> int:
    stream = torch.npu.current_stream(device)
    return int(stream.npu_stream)
