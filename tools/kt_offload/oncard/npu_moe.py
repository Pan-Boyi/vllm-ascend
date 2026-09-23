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


def kt_cfg(strategy: str, n: int, explicit_path: str | None, threads: int, numa: list[int], graph: bool,
           overlap: bool = False, packed: bool = False):
    from vllm_ascend.kt_offload.config import KTOffloadConfig

    return KTOffloadConfig(
        enabled=True, weight_path=C.CKPT, method="MXFP4", resident_strategy=strategy,
        overlap_side_stream=overlap, packed_transfer=packed,
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


# ---------------------------------------------------------------------------
# Shared-expert MLP stand-in (§4.31)
# ---------------------------------------------------------------------------
#
# What it is for: t5r's layers call quant_method.apply with shared_experts=None,
# so the production order "routed -> kt merge -> shared MLP" has no shared MLP
# in it at all and the switch that moves the merge behind it cannot be
# exercised. This builds a block with the SHAPES the served model has (5120 ->
# 2x2304 -> 5120, n_shared_experts=1) out of the quantized matmul operators the
# build has, so the two orders can be compared bit for bit.
#
# 🔴 What it is NOT: the served shared expert. The weights are random, and
# which operator chain runs is decided by probing this build (the served path
# is DynamicMxQuant -> QBMM -> SwigluGroupQuant -> QBMM; the W8A8 chain below is
# the same family at the same shapes, and the bf16 one is not quantized at all).
# The correctness question -- "does moving the merge behind this block change
# any bit?" -- does not depend on which chain runs, because both arms run the
# same one. Any timing read off it does, so the chain is printed and recorded.


BF16_EPS = 2.0 ** -8        # bf16 keeps 8 significand bits: one ULP is 2^-8 relative
SHARED_GAIN = 0.5           # DECLARED: rms(shared(x)) == SHARED_GAIN * rms(x), per token
MERGE_ULP_MARGIN = 8.0      # DECLARED: a merge contribution below this many bf16 ULP of the
                            # layer output cannot be seen bit for bit -> the criterion is blind


def _rms(x: torch.Tensor) -> torch.Tensor:
    """Per-token RMS in float32 (capture-safe: no host sync, no data-dependent shape)."""
    xf = x.float()
    return xf.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)


def _shared_weights(hidden: int, inter: int, seed: int, device, dtype):
    """Random weights at 1/sqrt(fan_in), i.e. a block whose output RMS is O(input RMS).

    🔴 The old /32 scale is what made the stand-in explode: see _normalized below.
    """
    g = torch.Generator().manual_seed(seed)
    if dtype == torch.int8:
        gu = torch.randint(-8, 8, (hidden, 2 * inter), generator=g, dtype=torch.int8)
        dn = torch.randint(-8, 8, (inter, hidden), generator=g, dtype=torch.int8)
    else:
        gu = (torch.randn((hidden, 2 * inter), generator=g) / hidden ** 0.5).to(dtype)
        dn = (torch.randn((inter, hidden), generator=g) / inter ** 0.5).to(dtype)
    return gu.to(device), dn.to(device)


def _normalized(core, gain: float = SHARED_GAIN):
    """Wrap a raw MLP core into a HOMOGENEOUS, NON-EXPANDING stand-in.

    🔴 Why this exists (2026-09-23, three t5r runs blocked before any arm ran):
    a raw silu(x@gu[:i]) * (x@gu[i:]) core is a QUADRATIC form -- its gain is
    O(|x|), not O(1). t5r stacks L layers as a residual chain and every layer
    calls the SAME stand-in, so the amplitude is squared once per layer:
    18.8 -> 6.4e2 -> 4.0e5 -> 3.1e11 -> 9.4e22 -> NaN from layer 5 on. Two
    all-NaN reference steps are "bit-identical" (bit_identical is NaN-aware) and
    the discriminating-pair self-check fired. Shrinking --layers does NOT fix
    it: at 1e5..1e22 the merge's O(|h|) contribution falls below one bf16 ULP of
    the layer output, so a DROPPED MERGE would read bit-identical = green. The
    stand-in has to be non-expanding for the criterion to have any power at all.

    Fix, and it is also what production does: the served shared expert reads the
    post-attention-norm hidden, so normalize the input to RMS 1, run the core,
    and put the caller's scale back with a declared gain:

        shared(x) = gain * rms(x) * core(x/rms(x)) / rms(core(x/rms(x)))

    Consequences relied on by the criterion: (a) rms(shared(x)) == gain*rms(x)
    exactly, so the residual stack grows by at most (1 + gain + merge share) per
    layer instead of squaring; (b) shared(c*x) == c*shared(x) for c>0, so every
    relative quantity -- above all rms(merge)/rms(h) -- is the same at layer 7 as
    at layer 0, which is what keeps the merge above the bf16 ULP floor; (c) the
    guarantee comes from the wrapper, not from the core, so it holds for the
    W8A8 chain and the bf16 chain alike.
    """

    def run(x: torch.Tensor) -> torch.Tensor:
        r = _rms(x)
        z = (x.float() / r).to(x.dtype)   # post-RMSNorm hidden, as production feeds it
        y = core(z).float()
        return (y * (gain * r / _rms(y))).to(x.dtype)

    return run


def _shared_w8a8_core(hidden: int, inter: int, seed: int, device, clamp_limit: float):
    """The W8A8 chain of vllm_ascend/ops/fused_moe/shared_experts.py, call for call."""
    import torch_npu

    gu, dn = _shared_weights(hidden, inter, seed, device, torch.int8)
    gu_scale = torch.full((2 * inter,), 0.01, dtype=torch.float32, device=device)
    gu_scale_fp32 = gu_scale.clone()
    dn_scale = torch.full((hidden,), 0.01, dtype=torch.float32, device=device)

    def run(x: torch.Tensor) -> torch.Tensor:
        qx, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        h = torch_npu.npu_quant_matmul(qx, gu, gu_scale, pertoken_scale=None, bias=None,
                                       output_dtype=torch.int32)
        q2, swiglu_scale = torch.ops._C_ascend.npu_dequant_swiglu_quant(
            x=h, weight_scale=gu_scale_fp32, activation_scale=pertoken_scale, bias=None,
            quant_scale=None, quant_offset=None, group_index=None, activate_left=True,
            quant_mode=1, swiglu_mode=1, clamp_limit=clamp_limit,
        )
        return torch_npu.npu_quant_matmul(q2, dn, dn_scale, pertoken_scale=swiglu_scale, bias=None,
                                          output_dtype=x.dtype)

    return run


def _mxfp_linear_weight(out_features: int, in_features: int, seed: int, device, group: int = None):
    """A W4A8_MXFP linear's weight/scale, prepared exactly as the linear method's
    process_weights_after_loading does (w4a8_mxfp4.py: format_cast -> transpose,
    and the scale's (n, k//2, 2) -> transpose(-3,-2) relayout)."""
    import torch_npu

    group = C.GROUP if group is None else group
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(0, 256, (out_features, in_features // 2), generator=g, dtype=torch.uint8).to(device)
    # e8m0 exponent byte: 127 is 2^0. One byte per group of ``group`` inputs.
    sc = torch.full((out_features, in_features // group), 127, dtype=torch.uint8, device=device)
    w = torch_npu.npu_format_cast(w, 29, customize_dtype=torch.float8_e4m3fn,
                                  input_dtype=torch_npu.float4_e2m1fn_x2)
    w = w.transpose(-1, -2)
    n, k = sc.shape
    sc = sc.reshape(n, k // 2, 2).transpose(-3, -2)
    return w, sc


def _shared_mxfp8_core(hidden: int, inter: int, seed: int, device, clamp_value: float):
    """🔴 The chain production actually runs for the DSV4.1 shared expert:
    shared_experts.py's QuantType.W4A8MXFP branch --
      npu_dynamic_mx_quant(x, e4m3) -> gate_up_proj (W4A8_MXFP linear: npu_quant_matmul
      with e8m0 scales and x2_dtype=float4_e2m1fn_x2) -> npu_swiglu_group_quant(
      dst_type=e4m3, quant_mode=2, clamp_value=swiglu_limit) -> down_proj (same linear).
    Weights are random, the operator sequence and the parameter shapes are the served ones.
    """
    import torch_npu

    gu_w, gu_s = _mxfp_linear_weight(2 * inter, hidden, seed, device)
    dn_w, dn_s = _mxfp_linear_weight(hidden, inter, seed + 1, device)

    def _mm(qx, scale, w, ws, out_dtype):
        return torch_npu.npu_quant_matmul(
            qx, w, ws,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=None,
            output_dtype=out_dtype,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[0, 0, C.GROUP],
        )

    def run(x: torch.Tensor) -> torch.Tensor:
        qx, sx = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        h = _mm(qx, sx, gu_w, gu_s, torch.bfloat16)
        q2, s2, _ = torch.ops._C_ascend.npu_swiglu_group_quant(
            h, topk_weight=None, group_index=None, dst_type=torch.float8_e4m3fn,
            quant_mode=2, clamp_value=clamp_value,
        )
        return _mm(q2, s2, dn_w, dn_s, x.dtype)

    return run


def _shared_bf16_core(hidden: int, inter: int, seed: int, device):
    """Same shapes, no quantization. Timing off it means less."""
    gu, dn = _shared_weights(hidden, inter, seed, device, torch.bfloat16)

    def run(x: torch.Tensor) -> torch.Tensor:
        h = x @ gu
        gate, up = h[:, :inter], h[:, inter:]
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype) @ dn

    return run


def legacy_quadratic_core(hidden: int, inter: int, seed: int, device):
    """The stand-in as it was before 2026-09-23: /32 weights, no normalization.

    Kept ONLY so the CPU self-check can be shown to go red on it (selftest_cpu).
    Never wire this into a run.
    """
    g = torch.Generator().manual_seed(seed)
    gu = (torch.randn((hidden, 2 * inter), generator=g) / 32).to(torch.bfloat16).to(device)
    dn = (torch.randn((inter, hidden), generator=g) / 32).to(torch.bfloat16).to(device)

    def run(x: torch.Tensor) -> torch.Tensor:
        h = x @ gu
        gate, up = h[:, :inter], h[:, inter:]
        return (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype) @ dn

    return run


# W8A8 clamp_limit: this build rejected 0.0 outright --
#   [OPS_NN][GetAttr] OpName:[DequantSwigluQuant] Parameter clamp_limit ... has
#   incorrect value 0.000000. Reason: clamp_limit should be positive finite.
# -> aclnnDequantSwigluQuantV2 Invalid_Argument(EZ0026).
# The tiling that runs is CANN 9.1.0's built-in (libophost_nn.so), which demands
# clamp_limit > 0; the repo's own csrc only demands >= 0 and its proto header
# still documents "0.0 disables the clamp", so the repo's contract and the
# installed operator disagree. 7.0 is the value the tree itself falls back to
# (vllm_ascend/lora/quant_moe.py: ``limit=swiglu_limit or 7.0``) and it is what
# moe_config passes as swiglu_limit here. Candidates are tried in order and the
# one that runs is printed: a clamp changes the numbers, which is fine for
# correctness (both arms run the SAME fn) but has to be on the record.
W8A8_CLAMP_CANDIDATES = (C.LIMIT if (C.LIMIT or 0) > 0 else 7.0, 7.0, 1.0)


def _w8a8_builders():
    for cl in W8A8_CLAMP_CANDIDATES:
        yield (f"w8a8(clamp_limit={cl})",
               lambda h, i, s, d, _cl=cl: _shared_w8a8_core(h, i, s, d, _cl))


MXFP8_CLAMP_CANDIDATES = (C.LIMIT if (C.LIMIT or 0) > 0 else 7.0, 7.0)


def _chain_builders(kind: str):
    """(name, core_builder) pairs. The served chain first, then the fallbacks.

    ``auto`` never reaches W8A8: production's DSV4.1 shared expert is the MXFP8
    branch, and W8A8 needs an operator this build rejects at 0.0 clamp anyway.
    ``--shared-chain w8a8`` still selects it explicitly.
    """
    out = []
    if kind in ("auto", "mxfp8"):
        out.extend((f"mxfp8(clamp_value={cv})",
                    lambda h, i, s, d, _cv=cv: _shared_mxfp8_core(h, i, s, d, _cv))
                   for cv in MXFP8_CLAMP_CANDIDATES)
    if kind == "w8a8":
        out.extend(_w8a8_builders())
    if kind in ("auto", "bf16"):
        out.append(("bf16", _shared_bf16_core))
    return out


def _probe_chain(fn, hidden: int, device, sync: bool = True) -> tuple[float, float]:
    """Refuse a chain that is not finite, not the right shape, not homogeneous or
    not at the declared gain. These are the three properties the criterion rests on."""
    g = torch.Generator().manual_seed(1234)
    x = (torch.randn((2, hidden), generator=g) * 3.0).to(torch.bfloat16).to(device)
    out = fn(x)
    out2 = fn(x * 4.0)
    if sync:
        torch.npu.synchronize()
    if tuple(out.shape) != (2, hidden):
        raise ValueError(f"shape {tuple(out.shape)}, want (2, {hidden})")
    if not torch.isfinite(out.float()).all():
        raise ValueError("non-finite output")
    gain = float((_rms(out) / _rms(x)).mean())
    if not (0.5 * SHARED_GAIN <= gain <= 2.0 * SHARED_GAIN):
        raise ValueError(f"gain rms(out)/rms(in) = {gain:.3f}, want ~{SHARED_GAIN} (expanding stand-in)")
    hom = float((_rms(out2.float() - 4.0 * out.float()) / _rms(4.0 * out.float())).max())
    if not hom < 0.05:
        raise ValueError(f"not homogeneous: rel|f(4x)-4f(x)| = {hom:.3e}")
    return gain, hom


def build_shared_mlp(device, hidden: int = None, inter: int = None, seed: int = 41, kind: str = "auto"):
    """Probe the build for a shared-MLP chain and return (fn, kind).

    The probe runs each candidate once on a 2-token input OUTSIDE any capture and
    keeps the first that is finite, right-shaped, homogeneous and at the declared
    gain. A chain that raises (wrong signature on this build, operator not
    installed, an attribute this build rejects) or fails a property is skipped and
    the reason printed -- silently falling through to bf16 would turn "the
    quantized chain is missing" into a timing result nobody can read, and silently
    keeping an expanding chain would turn the whole run into a false green.
    """
    hidden = C.HIDDEN if hidden is None else hidden
    inter = C.INTER if inter is None else inter
    tried = []
    for name, build in _chain_builders(kind):
        try:
            fn = _normalized(build(hidden, inter, seed, device))
            gain, hom = _probe_chain(fn, hidden, device)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{name}: {type(e).__name__}: {str(e)[:160]}")
            print(f"[shared-mlp] chain {name!r} unusable on this build -- {tried[-1]}", flush=True)
            continue
        what = ("THE SERVED OP MIX (mx-quant -> mxfp matmul -> swiglu_group_quant -> mxfp matmul)"
                if name.startswith("mxfp8") else
                "quantized matmuls, but NOT the served family (production's shared expert is MXFP8)"
                if name.startswith("w8a8") else
                "BF16 MATMULS, NOT the served op mix -- correctness only")
        print(f"[shared-mlp] chain {name!r}: {hidden} -> 2x{inter} -> {hidden}, {what}; "
              f"RMSNorm-in + gain {SHARED_GAIN} (measured {gain:.3f}, homogeneity {hom:.2e})", flush=True)
        return fn, name
    raise RuntimeError(f"no shared-MLP chain works on this build: {tried}")


# ---------------------------------------------------------------------------
# CPU proof that the criterion can still see a dropped merge (§4.31)
# ---------------------------------------------------------------------------
#
# Runs on the HOST, opens no device, and is a HARD gate at t5r start-up: a
# stand-in that fails it makes every "bit-identical" in the run meaningless.
# What it models is exactly the arm's arithmetic --
#     h <- (h + merged) + shared(h),   merged = routed(h) [+ cpu(h)]
# -- with routed/cpu replaced by low-rank random linear maps at DECLARED
# relative magnitudes, and it runs the whole thing twice, with and without the
# cpu (= merge) term. The no-merge run has to differ BIT FOR BIT at EVERY layer
# and by at least MERGE_ULP_MARGIN bf16 ULP, at every step.


def _lowrank(hidden: int, rank: int, seed: int, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    a = (torch.randn((hidden, rank), generator=g) / hidden ** 0.5).to(dtype)
    b = (torch.randn((rank, hidden), generator=g) / rank ** 0.5).to(dtype)

    def run(x: torch.Tensor, rel: float) -> torch.Tensor:
        y = (x @ a) @ b
        return (y.float() * (rel * _rms(x) / _rms(y))).to(x.dtype)

    return run


def shared_stack_cpu_check(shared_fn, *, layers: int, steps: int, tokens: int, hidden: int,
                           merge_rel: float = 0.25, routed_rel: float = 0.5, seed: int = 41,
                           amp_cap: float = 1.0e4, growth_cap: float = 4.0,
                           ulp_margin: float = MERGE_ULP_MARGIN, verbose: bool = True) -> dict:
    """Raise unless the stand-in leaves the no-merge mutant visible at every layer.

    Returns a report dict; raises RuntimeError naming the first failing (step,
    layer) otherwise. Pure host torch -- no torch_npu, no device.
    """
    maps = [(_lowrank(hidden, 64, seed + 11 * l), _lowrank(hidden, 64, seed + 7919 + 11 * l))
            for l in range(layers)]
    g = torch.Generator().manual_seed(seed + 5)
    fails, rows, finals = [], [], []
    worst = None
    for s in range(steps):
        x = torch.randn((tokens, hidden), generator=g).to(torch.bfloat16)
        h_on, h_off = x, x
        for l in range(layers):
            routed_map, cpu_map = maps[l]
            for tag, h in (("merge", h_on), ("no-merge", h_off)):
                if not torch.isfinite(h.float()).all():
                    fails.append(f"step {s} layer {l} [{tag}]: non-finite h")
                amp = float(h.float().abs().max())
                if amp > amp_cap:
                    fails.append(f"step {s} layer {l} [{tag}]: max|h| {amp:.3e} > cap {amp_cap:.1e}")
            if fails:
                break
            nxt = []
            for h, with_merge in ((h_on, True), (h_off, False)):
                pre = routed_map(h, routed_rel)
                merged = pre + cpu_map(h, merge_rel) if with_merge else pre
                sh = shared_fn(h)
                nxt.append((h + merged) + sh)
            h_on_next, h_off_next = nxt
            growth = float(_rms(h_on_next).mean() / _rms(h_on).mean())
            same, how = C.bit_identical(h_on_next, h_off_next)
            rel = float(_rms(h_on_next.float() - h_off_next.float()).mean() / _rms(h_on_next).mean())
            ulps = rel / BF16_EPS
            rows.append({"step": s, "layer": l, "growth": growth, "ulps": ulps, "same": same})
            if worst is None or ulps < worst["ulps"]:
                worst = rows[-1]
            if same:
                fails.append(f"step {s} layer {l}: no-merge is BIT-IDENTICAL to merge ({how})")
            if not (ulps >= ulp_margin):
                fails.append(f"step {s} layer {l}: no-merge differs by {ulps:.2f} bf16 ULP "
                             f"< margin {ulp_margin}")
            if not (1.0 / growth_cap <= growth <= growth_cap):
                fails.append(f"step {s} layer {l}: per-layer growth {growth:.3f} outside "
                             f"[{1.0 / growth_cap:.3f}, {growth_cap:.1f}]")
            h_on, h_off = h_on_next, h_off_next
            if fails:
                break
        finals.append(h_on)
        if fails:
            break
    if not fails and len(finals) > 1:
        for k in range(1, len(finals)):
            if C.bit_identical(finals[k], finals[k - 1])[0]:
                fails.append(f"steps {k - 1} and {k} produced identical outputs; the inputs do not "
                             f"discriminate")
    amps = [float(h.float().abs().max()) for h in finals]
    rep = {"layers": layers, "steps": steps, "tokens": tokens, "hidden": hidden,
           "merge_rel": merge_rel, "routed_rel": routed_rel, "ulp_margin": ulp_margin,
           "max_abs": max(amps) if amps else None,
           "max_growth": max((r["growth"] for r in rows), default=None),
           "min_ulps": worst["ulps"] if worst else None,
           "min_ulps_at": (worst["step"], worst["layer"]) if worst else None,
           "rows": rows, "fails": fails}
    if verbose:
        head = (f"[cpu-check] stand-in discriminating power: {layers} layers x {steps} steps x "
                f"{tokens} tok, declared merge share {merge_rel} of rms(h), routed {routed_rel}")
        print(head, flush=True)
        if rows:
            print(f"[cpu-check] max|h| {rep['max_abs']:.3e} (cap {amp_cap:.0e}), per-layer growth "
                  f"<= {rep['max_growth']:.3f} (cap {growth_cap}), no-merge visible at every layer: "
                  f"min {rep['min_ulps']:.1f} bf16 ULP at step/layer {rep['min_ulps_at']} "
                  f"(margin {ulp_margin})", flush=True)
    if fails:
        raise RuntimeError("shared-MLP stand-in has NO discriminating power -- refusing to run: "
                           + "; ".join(fails[:4]))
    return rep


def shared_stack_cpu_check_default(layers: int, steps: int, tokens: int, hidden: int, inter: int,
                                   seed: int = 41, **kw) -> dict:
    """The gate t5r runs: the same wrapper and the same core arithmetic, on the host.

    The bf16 core stands in for whichever chain the card picks, because the
    amplitude guarantee comes from _normalized, which both chains share.
    """
    fn = _normalized(_shared_bf16_core(hidden, inter, seed, torch.device("cpu")))
    return shared_stack_cpu_check(fn, layers=layers, steps=steps, tokens=tokens, hidden=hidden,
                                  seed=seed, **kw)
