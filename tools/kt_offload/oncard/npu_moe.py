# SPDX-License-Identifier: Apache-2.0
"""NPU half of the on-card single-layer checks.

What is real and what is a stand-in (README "范围"):
  real   AscendW4A8MXFPDSDynamicFusedMoEMethod (process_weights_after_loading, apply,
         apply_gmm1_act_quant, apply_gmm2), AllGatherCommImpl.fused_experts
         (TokenDispatcherWithAllGather -> npu_moe_init_routing quant_mode=3 -> GMM ->
         npu_moe_token_unpermute), KTLayerOffload (attach_after_init wrapping of the
         post-load hook, route, merge, the captured path), kt-kernel.
  shim   the layer module (an nn.Module carrying the parameters and the attributes
         apply() reads) and the moe_config namespace. AscendRoutedExperts' own
         constructor/forward_impl wiring is not exercised here: forward_impl is
         reproduced call for call (route -> apply -> finalize(no-op at DP=1) -> merge).
         Its hook placement is covered by the CPU UT (AST guards) and by the
         whole-network smoke (design doc, 上卡草案 步骤 3/4).
  skipped prepare()/finalize(): PrepareAndFinalizeWithAllGather is a no-op at DP=1
         (design §g); calling it would need the DP group, i.e. a distributed init.
         No torch.distributed / HCCL is initialised in this process.

Import order matters: bootstrap the custom-op path and load vllm_ascend_C before
the device is touched (serving gets them from NPUPlatform.import_kernels and
device_allocator/camem.py).
"""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oncard_common as C  # noqa: E402

_RUNTIME = None


def init_runtime():
    """Custom-op env, vllm_ascend_C, ascend config, device 0. Idempotent."""
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    t0 = time.time()
    from vllm_ascend.utils import bootstrap_custom_op_env

    bootstrap_custom_op_env()
    import torch_npu  # noqa: F401

    try:
        import vllm_ascend.vllm_ascend_C  # noqa: F401  registers torch.ops._C_ascend.*
    except ImportError as e:
        raise RuntimeError(f"vllm_ascend.vllm_ascend_C could not be imported ({e}); "
                           "npu_swiglu_group_quant lives there") from e
    if not hasattr(torch.ops._C_ascend, "npu_swiglu_group_quant"):
        raise RuntimeError("torch.ops._C_ascend.npu_swiglu_group_quant is not registered")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_ascend.ascend_config import init_ascend_config

    vcfg = VllmConfig()
    ctx = set_current_vllm_config(vcfg)
    ctx.__enter__()  # held for the process
    init_ascend_config(vcfg)
    torch.npu.set_device(0)
    dev = torch.device("npu", 0)
    _RUNTIME = SimpleNamespace(vllm_config=vcfg, device=dev)
    print(f"[runtime] ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
          f"device={dev} ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH', '')[:120]} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return _RUNTIME


def moe_config(num_local_experts: int):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return SimpleNamespace(
        experts_per_token=C.TOPK,
        num_experts=C.N_EXP,
        num_logical_experts=C.N_EXP,
        num_local_experts=num_local_experts,
        activation=MoEActivation.SILU,
        activation_situ_beta=None,
        activation_situ_linear_beta=None,
        swiglu_limit=C.LIMIT,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
        ep_size=1,
        tp_size=1,
        is_sequence_parallel=False,
    )


def quant_method():
    """The real DS W4A8 MXFP method. Its __init__ reads only group_size from the
    served quant config and dynamic_eplb from the ascend config; both are set to
    what the served V4.1 config gives (32, False)."""
    from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import AscendW4A8MXFPDSDynamicFusedMoEMethod

    qm = AscendW4A8MXFPDSDynamicFusedMoEMethod.__new__(AscendW4A8MXFPDSDynamicFusedMoEMethod)
    qm.group_size = C.GROUP
    qm.dynamic_eplb = False
    return qm


class ShimLayer(torch.nn.Module):
    """Parameters + the attributes AscendW4A8MXFPDynamicFusedMoEMethod.apply reads."""

    def __init__(self, num_slots: int, qm, cfg, device):
        super().__init__()
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation

        # Shapes/dtypes from the method itself (its host empties are never touched);
        # the parameters are allocated on the device and zeroed.
        shapes = qm.get_weight(num_slots, C.INTER, C.HIDDEN, torch.bfloat16)
        shapes.update(qm.get_dynamic_quant_param(num_slots, C.INTER, C.HIDDEN, torch.bfloat16))
        for name, t in shapes.items():
            p = torch.zeros(t.shape, dtype=t.dtype, device=device)
            self.register_parameter(name, torch.nn.Parameter(p, requires_grad=False))
        self.quant_method = qm
        self.moe_config = cfg
        self.local_num_experts = num_slots
        self.log2phy = None
        self.ascend_expert_map = None
        self.global_redundant_expert_num = 0
        self.ascend_mc2_mask = None
        self.apply_router_weight_on_input = False
        self.ascend_pertoken_scale = None
        self.activation = MoEActivation.SILU


def load_raw_layer(ck: C.Ckpt, layer: int, ids: list[int]):
    """Host uint8 tensors for the experts ``ids`` in slot order: (w13, w13s, w2, w2s)."""
    t0 = time.time()
    rows = [[], [], [], []]
    for e in ids:
        ex = ck.expert(layer, e)
        bad = C.expert_shapes_ok(ex)
        if bad:
            raise ValueError(f"layer {layer} expert {e}: {bad}")
        for i, t in enumerate(C.pack_npu_rows(ex)):
            rows[i].append(t)
    out = tuple(torch.stack(r) for r in rows)
    gib = sum(t.numel() for t in out) / 2**30
    print(f"[load] layer {layer}: {len(ids)} experts, {gib:.2f} GiB host bytes ({time.time() - t0:.1f}s)", flush=True)
    return out


def build_layer(raw, num_slots: int, device, mutate=None, offload=None):
    """Upload ``raw`` (rows for slots 0..len-1) into a fresh shim layer and run the
    quant method's post-load hook, exactly as the model loader would.

    ``mutate(name, dev_tensor)`` may edit the uploaded uint8 bytes before the
    post-load hook (mutation arms). ``offload``: a KTLayerOffload; its
    attach_after_init wraps the hook (zero unused slots -> original -> tables + kt).
    """
    qm = quant_method()
    layer = ShimLayer(num_slots, qm, moe_config(num_slots), device)
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    for name, t in zip(names, raw):
        p = getattr(layer, name)
        if p.dtype != torch.uint8 or tuple(p.shape[1:]) != tuple(t.shape[1:]) or t.shape[0] > p.shape[0]:
            raise ValueError(f"{name}: param {p.dtype}{tuple(p.shape)} vs checkpoint rows {t.dtype}{tuple(t.shape)}")
        p.data[: t.shape[0]].copy_(t)
    if mutate is not None:
        for name in names:
            mutate(name, getattr(layer, name).data)
    if offload is not None:
        offload.attach_after_init(layer)
    t0 = time.time()
    layer.quant_method.process_weights_after_loading(layer)
    torch.npu.synchronize()
    print(f"[build] {num_slots} slots on {device}, post-load hook {time.time() - t0:.1f}s, "
          f"HBM allocated {torch.npu.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return layer


def comm_method(num_local_experts: int):
    from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl

    return AllGatherCommImpl(moe_config(num_local_experts))


def npu_moe(layer, comm, x, topk_weights, topk_ids):
    """quant_method.apply under a forward context whose moe_comm_method is ``comm``."""
    from vllm.forward_context import override_forward_context

    ctx = SimpleNamespace(moe_comm_method=comm, in_profile_run=False, capturing=False,
                          moe_comm_type=None, additional_kwargs={})
    with override_forward_context(ctx):
        res = layer.quant_method.apply(layer=layer, x=x, topk_weights=topk_weights, topk_ids=topk_ids,
                                       shared_experts=None, shared_experts_input=None)
    return res.routed_out


def kt_cfg(strategy: str, n: int, explicit_path: str | None, threads: int, numa: list[int], graph: bool):
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    return KTOffloadConfig(
        enabled=True, weight_path=C.CKPT, method="MXFP4", resident_strategy=strategy,
        num_resident_experts=n if strategy != "explicit" else 0,
        resident_experts_path=explicit_path, cpuinfer_threads=threads,
        threadpool_count=len(numa), numa_nodes=numa, chunked_prefill_size=64,
        skip_cpu_in_profile_run=True, force_sync_submit=True,
    )


def fake_vllm_config(capture_sizes: list[int] | None):
    """The fields KTLayerOffload reads (hf_text_config, model, scheduler, compilation)."""
    graph = bool(capture_sizes)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=C.HIDDEN, moe_intermediate_size=C.INTER,
                                           num_experts_per_tok=C.TOPK, num_hidden_layers=40),
            model=C.CKPT,
            enforce_eager=not graph,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY" if graph else "NONE"),
            cudagraph_capture_sizes=list(capture_sizes or []),
        ),
    )


def make_offload(cfg, layer_idx: int, capture_sizes=None):
    """Real plan + real KTLayerOffload for one layer (as create_layer_offload builds it)."""
    from vllm_ascend.kt_offload.layer import KTLayerOffload
    from vllm_ascend.kt_offload.placement import plan_from_config

    plan = plan_from_config(cfg, 40, C.N_EXP, C.TOPK)
    placement = plan.layer(layer_idx)
    return KTLayerOffload(cfg, placement, layer_idx, fake_vllm_config(capture_sizes)), plan


class KtTap:
    """Records what kt handed back (sync_forward / copy_forward_output_to_device return a
    reused device buffer, so it is cloned at once). Installed by name on the instance."""

    def __init__(self, wrapper):
        self.wrapper, self.last = wrapper, None
        orig = wrapper.sync_forward

        def sync_forward(hidden_states, cuda_stream, _orig=orig, _tap=self):
            out = _orig(hidden_states, cuda_stream)
            _tap.last = out.detach().clone()
            return out

        wrapper.sync_forward = sync_forward


def offload_forward(off, layer, comm, x, topk_weights, topk_ids):
    """forward_impl, call for call: route -> apply -> (finalize no-op) -> merge.

    Returns (npu_part_before_merge, merged)."""
    npu_routing = off.route(x, topk_weights, topk_ids, in_profile_run=False)
    w2, ids2 = npu_routing.topk_weights, npu_routing.topk_ids
    pre = npu_moe(layer, comm, x, w2, ids2)
    pre_copy = pre.detach().clone()
    merged = off.merge(pre)
    return pre_copy, merged, (ids2, w2)


def hbm_line() -> str:
    return (f"HBM allocated {torch.npu.memory_allocated() / 2**30:.2f} GiB, "
            f"peak {torch.npu.max_memory_allocated() / 2**30:.2f} GiB")
