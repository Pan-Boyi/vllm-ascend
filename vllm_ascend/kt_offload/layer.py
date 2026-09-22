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
import os
import re
import weakref
from time import perf_counter_ns
from typing import Any

import torch

# The ``vllm`` logger, not ``logging.getLogger(__name__)``: in the worker
# process only ``configure_ascend_file_logging()`` runs (worker.py), which adds
# a file handler to the ``vllm_ascend`` logger but never sets its level, so that
# logger's effective level is the root's WARNING. A ``vllm_ascend.*`` name
# therefore loses INFO at the level gate, and its WARNING/ERROR reach only
# ~/ascend/log/vllm_ascend/*.log, never serve.log. ``init_logger(__name__)``
# would not change that: it is ``getLogger`` plus patched methods
# (vllm/logger.py:204-214). 132 files in this tree use the import below.
from vllm.logger import logger

from vllm_ascend.kt_offload import seam_timing
from vllm_ascend.kt_offload.placement import (
    LayerPlacement,
    NpuRouting,
    OffloadPlan,
    assert_routing_dtypes,
    host_ops_per_layer,
    packed_layout,
    plan_from_config,
    split_routing,
)

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
        # Only the diagnostic sampler needs this; a test double need not carry it.
        self.num_layers = int(getattr(vllm_config.model_config.hf_text_config, "num_hidden_layers", 0)) or (
            placement.layer_idx + 1
        )
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
        self.packed = bool(getattr(cfg, "packed_transfer", False))
        self._packed: dict[int, dict] = {}
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
        _maybe_start_host_func_log(self.num_layers)
        device = self._device_of(layer)
        p = self.placement
        self._slot_tables = {dt: p.safe_slot_table(dt).to(device) for dt in (torch.int32, torch.int64)}
        self._keep_tables = {dt: p.keep_table(dt).to(device) for dt in (torch.float32, torch.bfloat16)}
        if not p.cpu_ids:
            return  # every expert resident: nothing for kt to do
        self.wrapper = self._build_wrapper()
        # Seam timing, if the env var asks for it. The CPUInfer probe is what
        # splits kt's submit from its sync without touching kt's C++; on the
        # packed path we make those two calls ourselves, so no probe is needed
        # (and wrapping there would only add a frame to the thing measured).
        seam_timing.maybe_enable(self.num_layers, None if self.packed else self.wrapper, split=self.packed)
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
        if self.packed:
            if not self._packed_supported():
                raise RuntimeError(
                    "kt_offload: packed_transfer needs kt's moe.forward_task and cpu_infer.submit/sync; "
                    f"this kt build ({type(self.wrapper).__name__}) does not expose them."
                )
            # Graph mode: every capture size up front (nothing may allocate
            # inside a capture). Eager: lazily, on first use.
            for bs in self.capture_sizes:
                self._packed_buffers(bs, device)
        if self.graph_enabled:
            self._prepare_capture_buffers(device)

    # -- packed transfer (cfg.packed_transfer) ------------------------------

    def _packed_supported(self) -> bool:
        moe = getattr(self.wrapper, "moe", None)
        infer = getattr(self.wrapper, "cpu_infer", None)
        return all(
            (
                moe is not None and hasattr(moe, "forward_task"),
                infer is not None and hasattr(infer, "submit") and hasattr(infer, "sync"),
            )
        )

    def _packed_buffers(self, bs: int, device) -> dict:
        """One combined input buffer per batch size, and our own output pair.

        Own buffers on purpose: kt hands layer L the pinned slot ``L % 2``, so
        two layers share one buffer and the ordering between one layer's output
        H2D and the other's CPU write is something the caller has to maintain.
        Per-layer buffers remove that coupling, and one combined input region
        turns kt's three input copies into one (placement.host_ops_per_layer).
        """
        buf = self._packed.get(bs)
        if buf is not None:
            return buf
        if bs > self.chunked_prefill_size:
            # kt's own wrappers check this (experts_base._check_qlen_fits_cpp_buffers);
            # we call the C++ task directly, so the check is ours. The C++ output
            # pointer advances qlen * hidden_size with no bound of its own: past
            # chunked_prefill_size it writes outside the allocation and the heap
            # damage surfaces later, somewhere else.
            raise ValueError(
                f"kt_offload packed_transfer: {bs} tokens exceeds chunked_prefill_size "
                f"{self.chunked_prefill_size}; the CPU MoE would write past its output buffer."
            )
        lay = packed_layout(bs, self.hidden_size, self.top_k)
        dev = torch.empty(lay["nbytes"], dtype=torch.uint8, device=device)
        pin = _pinned_empty((lay["nbytes"],), torch.uint8)

        def view(store, region, dtype, shape):
            off, nbytes = lay[region]
            return store[off : off + nbytes].view(dtype).view(*shape)

        buf = {
            "layout": lay,
            "dev": dev,
            "pin": pin,
            "dev_hidden": view(dev, "hidden", torch.bfloat16, (bs, self.hidden_size)),
            "dev_ids": view(dev, "ids", torch.int64, (bs, self.top_k)),
            "dev_weights": view(dev, "weights", torch.float32, (bs, self.top_k)),
            "pin_hidden": view(pin, "hidden", torch.bfloat16, (bs, self.hidden_size)),
            "pin_ids": view(pin, "ids", torch.int64, (bs, self.top_k)),
            "pin_weights": view(pin, "weights", torch.float32, (bs, self.top_k)),
            "pin_out": _pinned_empty((bs, self.hidden_size), torch.bfloat16),
            "dev_out": torch.empty((bs, self.hidden_size), dtype=torch.bfloat16, device=device),
            "bsz": _pinned_empty((1,), torch.int32).fill_(bs),
        }
        for name, dtype, shape in (
            ("pin_hidden", torch.bfloat16, (bs, self.hidden_size)),
            ("pin_ids", torch.int64, (bs, self.top_k)),
            ("pin_weights", torch.float32, (bs, self.top_k)),
            ("pin_out", torch.bfloat16, (bs, self.hidden_size)),
        ):
            t = buf[name]
            # The C++ task takes raw pointers: a non-contiguous view or the
            # wrong dtype is read as if it were the layout it expects.
            if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
                raise AssertionError(
                    f"kt_offload packed buffer {name}: {t.dtype}{tuple(t.shape)} "
                    f"contiguous={t.is_contiguous()}, want {dtype}{shape} contiguous"
                )
        self._packed[bs] = buf
        return buf

    def _packed_submit(self, hs: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> dict:
        """Fill the combined buffer on the device, then one host-device copy."""
        buf = self._packed.get(hs.shape[0])
        if buf is None:
            if _capturing():
                raise RuntimeError(
                    f"kt_offload: capturing {hs.shape[0]} tokens with packed_transfer but no buffer was "
                    f"prepared for that size (prepared: {sorted(self._packed)}); allocating inside a capture "
                    "is not possible."
                )
            buf = self._packed_buffers(hs.shape[0], hs.device)
        _device_fill(buf["dev_hidden"], hs)  # device-side fills: ordinary device work
        _device_fill(buf["dev_ids"], topk_ids)
        _device_fill(buf["dev_weights"], topk_weights)
        _host_copy(buf["pin"], buf["dev"])  # the one host-device copy
        return buf

    def _packed_payload(self, bs: int) -> tuple:
        buf = self._packed[bs]
        return (self.wrapper, buf, self.layer_idx)

    def host_ops(self) -> dict[str, int]:
        """Host-device operations this layer costs per step (for the accounting)."""
        return host_ops_per_layer(self.packed)

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
        # device="cpu" is explicit: this runs while the model loads, where the
        # default device is the NPU, and kt's argument is a *host* map
        # (physical_to_logical_map_cpu).
        wrapper.load_weights(torch.arange(self.placement.num_experts, dtype=torch.int64, device="cpu"))
        _refuse_cross_layer(wrapper, self.layer_idx)  # A1: deferral cannot be produced
        _register_wrapper(wrapper, self.layer_idx)  # A5/A9: one wrapper per index, one CPUInfer
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
            # A3: what kt would READ as incremental for this submit. The knob
            # (A1) says deferral cannot be produced; this says none was seen.
            _assert_no_pending(self.wrapper, self.layer_idx, self.layer_idx - 1, "before this layer's submit")
            if _capturing():
                # No runtime condition here (not even the profile flag): a
                # branch the capture run skips is missing from every replay.
                hs = hidden_states.contiguous()
                if self.packed:
                    self._packed_submit(hs, topk_ids, topk_weights)
                    stream = torch.npu.current_stream(hs.device)
                    _subscribe_stream(stream)
                    payload = self._packed_payload(hs.shape[0])
                    _CAPTURED_PAYLOADS.append(payload)
                    _launch_host_func(stream, _kt_packed_forward, payload)
                    self._pending, self._pending_captured = hs, True
                elif self.overlap:
                    self._submit_captured_forked(hs, topk_ids, topk_weights)
                else:
                    self._submit_captured(hs, topk_ids, topk_weights)
                self._pending, self._pending_captured = hs, True
            elif not (in_profile_run and self.cfg.skip_cpu_in_profile_run):
                # Eager is never forked, whatever overlap_side_stream says (the
                # reference implementation does the same, and see the note on
                # _submit_eager_forked for why a fork buys nothing here).
                hs = hidden_states.contiguous()
                if self.packed:
                    self._packed_submit(hs, topk_ids, topk_weights)
                    # Wait for our own copy only: kt's submit_forward calls
                    # torch.npu.synchronize(device), which waits for every
                    # stream on the device, twice per layer (again in
                    # sync_forward). One event on this stream is enough, and
                    # our per-layer buffers remove the slot-reuse reason for
                    # the wider wait.
                    _wait_for_copy(hs.device)
                    _kt_packed_forward(self._packed_payload(hs.shape[0]))
                else:
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
            cpu_out = self._packed_output(hs) if self.packed else self.wrapper.copy_forward_output_to_device(hs)
        elif self.packed:
            cpu_out = self._packed_output(hs)
        else:
            cpu_out = self.wrapper.sync_forward(hs, _stream_handle(hs.device))
        # A4: after this layer drained, its own flag must be clear. A stale
        # True survives an exception on kt's pre-submit checks and then makes
        # the NEXT drain wait for one task fewer -- an early return, not a
        # raise, so nothing else would notice.
        _assert_no_pending(self.wrapper, self.layer_idx, self.layer_idx, "after this layer drained")
        # One rounding to the activation dtype instead of two.
        return (routed_out.float() + cpu_out.float()).to(routed_out.dtype)

    def _packed_output(self, hs: torch.Tensor) -> torch.Tensor:
        """H2D of our own pinned output; ordered after the host-func node."""
        buf = self._packed[hs.shape[0]]
        _host_copy(buf["dev_out"], buf["pin_out"])
        return buf["dev_out"]

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


# There is no start-up probe any more. The one written on 2026-09-20 captured
# the same fork/join *pattern* with a scalar copy and a trivial host function,
# and it PASSED on the platform where every real-layer capture failed with
# 107025 -- a stand-in weaker than its subject passes where the subject fails,
# which is the second shape of "the guard reads something else" (the first was
# reading torch.npu.is_current_stream_capturing() as if it meant "this stream's
# work is in the graph"). The switch is refused in config instead; a probe may
# come back only if it can reproduce both known failures (107025 and the silent
# no-replay) -- that meta-criterion is the price of having one at all.


def _kt_packed_forward(payload) -> None:
    """Run this layer's CPU experts straight on our combined pinned buffer.

    kt's own entry points read its three pinned allocations; here the same C++
    task is handed our three pointers into one buffer, so the step paid one
    host-device copy instead of three. Exceptions must not escape a graph node.

    Four properties of that C++ task this relies on (kt, 2026-09-20):
      * the arguments are positional and ``qlen`` is a *pointer* to an int32
        batch size, not the number (ext_bindings.cpp ForwardBindings::Args);
      * every buffer must stay alive and unchanged until the task has run --
        ours live in ``self._packed`` for the process, the D2H before this node
        and the H2D after it are on the same stream, and each layer has its own
        buffer, so kt's ``layer_idx % 2`` slot sharing does not apply to us;
      * ``incremental=False`` overwrites the output (True would read it back
        and accumulate, which is not what the merge wants);
      * kt's queue is one serial FIFO worker, and it re-raises a task's
        exception only at the next sync -- so the submit and the sync stay in
        the same node here, and a layer that submits without syncing would hide
        its own errors for ever.
    """
    wrapper, buf, layer_idx = payload
    # Seam timing, same segments as _kt_host_forward. Here the submit and the
    # sync are ours, so the split needs no CPUInfer probe: ``kt_pre`` is the
    # forward_task build, ``kt_tail`` is empty by construction.
    rec = seam_timing.RECORDER
    ts = seam_timing.TS
    if rec is not None:
        ts[seam_timing.T_SUBMIT_IN] = 0
        ts[seam_timing.T_ENTER] = perf_counter_ns()
    try:
        # A3/A4 per replay: in graph mode route()/merge() run once, at capture,
        # so this node is the only place a per-step check exists. Raising here
        # is caught below and logged -- the graph has nowhere to return to.
        _assert_no_pending(wrapper, layer_idx, layer_idx - 1, "before this node's submit")
        if rec is not None:
            ts[seam_timing.T_PRE] = perf_counter_ns()
        task = wrapper.moe.forward_task(
            buf["bsz"].data_ptr(),
            buf["pin_ids"].shape[-1],
            buf["pin_ids"].data_ptr(),
            buf["pin_weights"].data_ptr(),
            buf["pin_hidden"].data_ptr(),
            buf["pin_out"].data_ptr(),
            False,  # incremental: no deferred experts
        )
        if rec is None:
            wrapper.cpu_infer.submit(task)
            wrapper.cpu_infer.sync(0)
        else:
            ts[seam_timing.T_SUBMIT_IN] = perf_counter_ns()
            wrapper.cpu_infer.submit(task)
            ts[seam_timing.T_SUBMIT_OUT] = ts[seam_timing.T_SYNC_IN] = perf_counter_ns()
            wrapper.cpu_infer.sync(0)
            ts[seam_timing.T_SYNC_OUT] = ts[seam_timing.T_KT_OUT] = perf_counter_ns()
        _assert_no_pending(wrapper, layer_idx, layer_idx, "after this node drained")
        HOST_FUNC_CALLS[layer_idx] = HOST_FUNC_CALLS.get(layer_idx, 0) + 1
        if rec is not None:
            rec.record(layer_idx, ts, perf_counter_ns())
    except BaseException as exc:  # noqa: BLE001
        logger.error(
            "kt_offload packed host-func for layer %d raised: %r. The CPU-expert output of this step is "
            "NOT this step's -- the device reads whatever the pinned buffer held.",
            layer_idx,
            exc,
        )


def _kt_host_forward(payload) -> None:
    """Graph-node callback: run this layer's CPU experts on the pinned buffers.

    Exceptions must not escape (nothing on the ACL side receives them); the
    device would then read the previous step's CPU output. Log loudly.
    """
    wrapper, shape_ref, stream_handle, layer_idx = payload
    # Seam timing (VLLM_ASCEND_DSV41_KT_SEAM_TIMING): off -> RECORDER is None
    # and this costs one global read and one branch per segment boundary. On,
    # it reads clocks and stores integers; it changes no argument, no order
    # and no device operation. See seam_timing.py for what each segment is.
    rec = seam_timing.RECORDER
    ts = seam_timing.TS
    if rec is not None:
        ts[seam_timing.T_SUBMIT_IN] = 0
        ts[seam_timing.T_ENTER] = perf_counter_ns()
    try:
        _assert_no_pending(wrapper, layer_idx, layer_idx - 1, "before this node's submit")
        if rec is not None:
            ts[seam_timing.T_PRE] = perf_counter_ns()
        wrapper.run_pinned_forward_sync(shape_ref, stream_handle)
        if rec is not None:
            ts[seam_timing.T_KT_OUT] = perf_counter_ns()
        _assert_no_pending(wrapper, layer_idx, layer_idx, "after this node drained")
        HOST_FUNC_CALLS[layer_idx] = HOST_FUNC_CALLS.get(layer_idx, 0) + 1
        if rec is not None:
            # Taken before the store, so the bookkeeping this module costs is
            # not counted inside the callback it measures.
            rec.record(layer_idx, ts, perf_counter_ns())
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




# -- the no-cross-layer contract, read off kt's own bookkeeping --------------
#
# kt keeps one process-level dict, ``BaseMoEWrapper._layer_has_pending_deferred``,
# keyed by bare layer_idx (experts_base.py:411). Three facts follow, and the
# checks below read the dict through kt's public accessor rather than the
# switch that is supposed to keep it empty:
#   * a submit reads the PREVIOUS layer's flag and turns it into
#     ``incremental=True`` (:667/:744/:819) -- a silent accumulation across a
#     layer boundary, which this deployment forbids;
#   * a layer's own flag sets its drain allowance to 1 (:771/:942), i.e. the
#     sync waits for one task FEWER. That failure returns early instead of
#     raising, and the output read afterwards is half written;
#   * the flag is written False only after a successful immediate submit
#     (:697/:755/:844), while the checks before it (`_check_qlen_fits_cpp_buffers`
#     at :593/:650/:728, `_wait_device` at :817) can raise first and leave a
#     stale True behind.
# A1 (max_deferred_experts_per_token == 0) says deferral cannot be *produced*;
# these say nothing was *read*. Both are kept: they fail in different places.
_PENDING_API_WARNED = False
# weak refs: a wrapper that has been dropped no longer writes kt's flags, and
# a test (or a rebuilt model) may legitimately construct a fresh one for the
# same index. What must never happen is two LIVE wrappers on one index.
_WRAPPER_REGISTRY: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()


def _pending_reader(wrapper):
    """kt's public accessor, or None on a build that predates it."""
    global _PENDING_API_WARNED
    reader = getattr(wrapper, "has_pending_deferred", None)
    if reader is None and not _PENDING_API_WARNED:
        _PENDING_API_WARNED = True
        logger.warning(
            "kt_offload: this kt build has no has_pending_deferred(); the no-cross-layer contract is "
            "checked only by the configuration knob (A1), not by what kt actually read. Coverage of "
            "the A3/A4 checks on this build is zero."
        )
    return reader


def _assert_no_pending(wrapper, layer_idx: int, which: int, when: str) -> None:
    """Refuse to continue if kt's flag for layer ``which`` is set."""
    reader = _pending_reader(wrapper)
    if reader is None:
        return
    if not reader(which):
        return
    own = which == layer_idx
    raise RuntimeError(
        f"kt_offload: layer {layer_idx}: kt reports a pending deferred task for layer {which} {when}. "
        + (
            "Its own flag makes this layer's drain allowance 1 (experts_base.py:771/:942), so the sync "
            "returns one task early and the output read after it is half written."
            if own
            else "The next submit would read it as incremental=True (experts_base.py:667/:744/:819) and "
            "accumulate the previous layer's CPU experts into this layer's output."
        )
        + " Crossing a layer boundary is forbidden for this deployment; a stale True is also what an "
        "exception on kt's pre-submit checks leaves behind."
    )


def _reset_wrapper_registry() -> None:
    """Forget every registered wrapper. For tests that stand in for a fresh
    process; a service never calls this, and calling it would throw away the
    only record that two wrappers share an index."""
    _WRAPPER_REGISTRY.clear()


def _register_wrapper(wrapper, layer_idx: int) -> None:
    """A5/A9: the bookkeeping is process-level, so our wrappers must not share
    a layer index, and they must share one CPUInfer.

    kt's dict is keyed by the bare layer index: a second wrapper built for the
    same index would write flags this one then reads as "the previous layer
    left something". And two CPUInfer instances would mean two queues, so
    "the queue is serial" -- the assumption every ordering argument here rests
    on -- would quietly stop being true.
    """
    previous = _WRAPPER_REGISTRY.get(layer_idx)
    if previous is not None and previous is not wrapper:
        raise RuntimeError(
            f"kt_offload: a second live kt wrapper was built for layer_idx {layer_idx}. kt's deferred "
            "bookkeeping is a process-level dict keyed by that index (experts_base.py:411), so the two "
            "would read each other's flags."
        )
    _WRAPPER_REGISTRY[layer_idx] = wrapper
    # Unwrap the seam-timing probe: every wrapper gets its own pass-through
    # over the *same* CPUInfer, so comparing the probes would read "two
    # queues" where there is one. The invariant is about the queue.
    infer = seam_timing.unwrap_infer(getattr(wrapper, "cpu_infer", None))
    for other_idx, other in _WRAPPER_REGISTRY.items():
        other_infer = seam_timing.unwrap_infer(getattr(other, "cpu_infer", None))
        if other is wrapper or other_infer is None or infer is None:
            continue
        if other_infer is not infer:
            raise RuntimeError(
                f"kt_offload: layer {layer_idx} and layer {other_idx} have different CPUInfer objects "
                f"({infer!r} vs {other_infer!r}). Two queues means the serial-FIFO ordering every "
                "cross-layer argument here rests on no longer holds."
            )

def _refuse_cross_layer(wrapper, layer_idx: int) -> None:
    """Hard constraint (user, 2026-09-20): nothing may cross a layer boundary.

    Deferring part of a token's CPU experts to a later layer or step is
    forbidden -- not deferred, forbidden -- so kt's deferral must be off. It
    defaults to 0 and we never ask for it, but a default is a policy and this
    is meant to be enforcement: read it off the object that would do the
    deferring, at the moment it exists, and refuse rather than discover it in
    an output.

    The other half of the same constraint, ``incremental=False`` on every
    forward_task call, is asserted where that call is made (and in
    tests/ut/kt_offload/test_packed_transfer.py).
    """
    seen = {}
    for attr in ("max_deferred_experts_per_token", "num_deferred_experts_per_token", "deferred_experts"):
        value = getattr(wrapper, attr, None)
        if value is None:
            continue
        seen[attr] = value
        try:
            nonzero = int(value) != 0
        except (TypeError, ValueError):
            nonzero = bool(value)
        if nonzero:
            raise RuntimeError(
                f"kt_offload: layer {layer_idx}: kt reports {attr}={value!r}, which defers part of a token's "
                "CPU experts past this layer. Crossing a layer boundary is forbidden for this deployment "
                "(design doc, hard constraint): the CPU contribution must be merged in the same layer that "
                "routed it."
            )
    if not seen:
        logger.warning(
            "kt_offload: layer %d: this kt build exposes no deferral attribute to check; the no-cross-layer "
            "constraint is unverified on this build (looked for max_deferred_experts_per_token and friends)",
            layer_idx,
        )

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


def _maybe_start_host_func_log(num_layers: int) -> None:
    """Start the per-layer host-func sampler if the env var asks for it."""
    from vllm_ascend import envs
    from vllm_ascend.kt_offload import hostfunc_log

    raw = envs.VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG
    if not raw:
        return
    try:
        interval = float(raw)
    except ValueError:
        logger.warning("kt_offload: VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=%r is not a number of seconds", raw)
        return
    hostfunc_log.start(interval, HOST_FUNC_CALLS, num_layers)


def _pinned_empty(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Pinned host memory: the only kind an async host-device copy may use."""
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _device_fill(dst: torch.Tensor, src: torch.Tensor) -> None:
    """A device-side copy into the combined buffer (not a host-device operation)."""
    dst.copy_(src)


def _host_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """One host-device copy: the operation whose count we are reducing."""
    dst.copy_(src, non_blocking=True)


def _wait_for_copy(device) -> None:
    """Host waits for the work already enqueued on this stream, nothing wider."""
    event = _new_event(device)
    event.record(torch.npu.current_stream(device))
    event.synchronize()


def _stream_handle(device: torch.device) -> int:
    stream = torch.npu.current_stream(device)
    return int(stream.npu_stream)
