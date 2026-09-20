#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition

env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # Build only the vllm_ascend_C extension and keep vllm_ascend/_cann_ops_custom
    # untouched (COMPILE_CUSTOM_KERNELS=0 builds nothing at all).
    "SKIP_ACLNN_BUILD": lambda: bool(int(os.getenv("SKIP_ACLNN_BUILD", "0"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    # NOTE (R44): this is the ONLY key in this file that appears anywhere in the
    # gating of the operators listed in the block below, and even here it is a
    # co-requirement, not an enabler: ascend_config.py:151-155 only ASSERTS that
    # it is set once additional_config.eplb_config.dynamic_eplb is already true.
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # ----------------------------------------------------------------------
    # Operators that are BUILT INTO csrc BUT NOT INSTALLED in the A5 operator
    # package -- i.e. one flipped switch away from failing. Inventory, evidence
    # and the per-operator verdicts:
    #     build_artifacts/official_code_review/o_csrc_vs_installed_ops.md
    #
    # WHEN THEY FAIL: at the operator's FIRST CALL, not at import and not at
    # startup. Every torch schema in csrc/torch_binding.cpp is registered
    # unconditionally, so the extension imports and the service comes up
    # exactly as it does on a machine where these operators exist. "The server
    # started" is not evidence that any of them is installed.
    #
    # NONE OF THEM IS AN ENVIRONMENT VARIABLE, so none of them can be given a
    # comment next to its own entry here -- scanned by read point, not by
    # scanning this file (they are additional_config fields, a vLLM config, or
    # not user-facing at all). Where each one's switch actually lives:
    #   FusedSparseAttentionOverlap   additional_config.sparse_kv_offload_config
    #                                 .use_fused_overlap  (ascend_config.py:1327,
    #                                 default False; parent .enabled :1322)
    #   StoreKVBlock                  additional_config.enable_sparse_li_c8
    #                                 (ascend_config.py:450, default False) AND a
    #                                 PD kv_producer node (derived at :705)
    #   DispatchFFNCombine(+BF16/W4A8) additional_config.enable_fused_mc2 == 1
    #                                 (ascend_config.py:418, default 0). The BF16
    #                                 and W4A8 variants have no Python switch at
    #                                 all: csrc/mc2/dispatch_ffn_combine/
    #                                 dispatch_ffn_combine_torch_adpt.h:38-73
    #                                 picks one by weight dtype.
    #   CopyAndExpandEagleInputs      vLLM's speculative_config.parallel_drafting
    #                                 (vllm/config/speculative.py:166, default
    #                                 False) via needs_extra_input_slots; nothing
    #                                 in vllm-ascend gates it.
    #   TransposeKvCacheByBlock       additional_config
    #                                 .enable_transpose_kv_cache_by_block
    #                                 (ascend_config.py:427, DEFAULT TRUE) AND
    #                                 enable_custom_op() AND a Mooncake P2P KV
    #                                 transfer deployment. On A5 it is held off
    #                                 only by enable_custom_op() -- see below.
    #                                 R55: both fused call sites now refuse
    #                                 through custom_op_guards
    #                                 .require_transpose_kv_cache_by_block, so
    #                                 flipping the switch produces an
    #                                 attributed refusal instead of an
    #                                 EXEC_NPU_CMD failure.
    #   GroupedMatmulSwigluQuantWeightNzTensorList
    #                                 eplb_config.dynamic_eplb
    #                                 (ascend_config.py:116, default False) AND
    #                                 ascend_fusion_config.fusion_ops_gmmswigluquant
    #                                 (:102, default True) AND a W8A8-dynamic
    #                                 checkpoint AND enable_custom_op().
    #   AddRmsNormBias                no user switch: enable_custom_op() alone
    #                                 (utils.py:419) for ops/layernorm.py:73,99;
    #                                 the fusion-pass patterns additionally need
    #                                 additional_config.ascend_compilation_config
    #                                 .fuse_norm_quant (ascend_config.py:70,
    #                                 default True).
    #                                 R55: the two layernorm.py call sites now
    #                                 refuse through custom_op_guards
    #                                 .require_add_rms_norm_bias. The THREE
    #                                 norm_quant_fusion_pass.py pattern
    #                                 declarations are NOT guarded -- see that
    #                                 module's docstring for why, and the R55
    #                                 suite scans for them on every run.
    #
    # enable_custom_op() is FALSE ON A5 (hardware_profile.py's A5 capability set
    # has no RUNTIME_CUSTOM_OPS), which is what holds the last three off today --
    # a hardware probe, not a decision anybody made about these operators. The
    # defaults above are therefore misleading read on their own: two of them say
    # True.
    #
    # R55 addressed exactly those two (TransposeKvCacheByBlock, AddRmsNormBias):
    # their call paths now RAISE with an attribution instead of being rescued by
    # enable_custom_op(). The criterion there is whether a kernel is installed,
    # NOT whether enable_custom_op() is true -- the flag is the accident, not the
    # decision. The other five in this list are still held off by their own
    # switches only.
    # ----------------------------------------------------------------------
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
    # DeepSeek-V4.1 mHC pre: "1" (default) uses npu_mhc_pre + npu_mhc_sinkhorn
    # instead of upstream's torch reference; "0" selects the torch reference.
    # On by default since the 2026-09-19 4-card A/B (together with the inverse
    # RoPE operator: TPOT -25..-38%, 12/12 cells separated).
    "VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR": lambda: os.getenv("VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR", "1"),
    # DeepSeek-V4.1: "1" carries each layer's configured compress ratio (0
    # included) to the metadata builder as its own spec field. Off: the
    # builder's compressor_ratio also sets tokens_per_state, so redirecting it
    # changes the page size too; enabling this requires splitting the two.
    "VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO": lambda: os.getenv("VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO", "0"),
    # DeepSeek-V4.1 sliding-window cache block size, in tokens. "0" (default)
    # keeps whatever the attention layer passed -- upstream hard-codes 32 at
    # vllm/models/deepseek_v4_1/attention.py:456. A positive value replaces it
    # and must appear verbatim in the SWA backend's supported kernel block
    # sizes. The SWA page is 32 tokens * 512 * 2 B = 32 KiB, which the planner
    # pads up to the 128 KiB full-MLA page (4.00x), and each of the 40
    # single-layer SWA groups then charges a whole 529 KiB pool block per
    # admitted block. 128 makes the page exact and cuts the admitted blocks
    # per request by 4. Ascend's own V4.0 path already runs SWA at
    # DSV4_BLOCK_SIZES[128][0][1] = 128 on the same backend.
    "VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE": lambda: int(os.getenv("VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE") or 0),
    # DeepSeek-V4.1 engram: "1" runs the n-gram hash, the id D2H, the host
    # gather and the H2D in the model runner before every forward, outside any
    # captured graph; the graph then only dequantises a persistent staging
    # buffer. "0" keeps the host gather inside the forward, recorded as a
    # host-function node under graph capture.
    "VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH": lambda: os.getenv("VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH", "0"),
    # DeepSeek-V4.1: "1" (default) undoes RoPE before the output projection with
    # one inplace_partial_rotary_mul over the rope slice (sin negated) instead
    # of the torch ops; "0" selects the torch ops. A5 honours partial_slice
    # bit-exactly (card A4). On by default since the 2026-09-19 4-card A/B
    # (together with the mHC pre operator).
    "VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR": lambda: os.getenv("VLLM_ASCEND_DSV41_INVERSE_ROPE_OPERATOR", "1"),
    # Forward RoPE (q and kv) through the same inplace_partial_rotary_mul the
    # inverse already uses. "1" (default) = operator, "0" = the torch
    # decomposition. ⚠ The two are not bit-identical: the operator rotates in
    # the tensor dtype, the torch form in float32 with one rounding at the
    # end. Kept switchable so an accuracy question can be answered by flipping
    # it rather than by reverting. Measured saving: ~1.5 ms of a 28.8 ms
    # bs=1 decode step (适配方案 §6.9.15).
    # mHC 的两处 torch 分解换融合件：RMSNorm 走 npu_rms_norm（模型里 q/kv norm
    # 已在用），stream collapse 走一次 bmm。"1"（默认）= 融合，"0" = 原 torch 式。
    # ⚠ 与 RoPE 那条不同，这两处【不保证逐位相同】：归约顺序不同，差几个 ULP。
    # 合计约 2.1 ms / 28.8 ms 一步（适配方案 §6.9.15）。
    "VLLM_ASCEND_DSV41_MHC_FUSED_OPS": lambda: os.getenv("VLLM_ASCEND_DSV41_MHC_FUSED_OPS", "1"),
    "VLLM_ASCEND_DSV41_FORWARD_ROPE_OPERATOR": lambda: os.getenv("VLLM_ASCEND_DSV41_FORWARD_ROPE_OPERATOR", "1"),
    # 融合 [T] -> [T, 2] 的 slot mapping：8 个算子 + 1 次 copy_ 压成 1 次下发，
    # 每步 42 组。设 0 退回 torch 分解（两条路逐位相同），用于二分回归。
    "VLLM_ASCEND_DSV41_FUSED_SLOT_MAPPING": lambda: os.getenv("VLLM_ASCEND_DSV41_FUSED_SLOT_MAPPING", "1"),
    # indexer 的 q RoPE / k_norm / k 尾部旋转换成已在树里跑着的算子（第 2 轮序 4）。
    # 🔴 k 侧的 rms_norm 是【非逐位】的：手写版中途那次 bf16 round trip 是喂给下游
    #    absmax 的（indexer_k_store.py:82-86），而 MXFP4 的块 scale 是
    #    exp2(ceil(log2(...)))，1 ulp 差跨过 2 的幂边界就整块跳一个 binade。
    #    设 0 退回手写版，用于把精度问题变成一次翻转而不是一次回退。
    "VLLM_ASCEND_DSV41_INDEXER_FUSED_ROPE": lambda: os.getenv("VLLM_ASCEND_DSV41_INDEXER_FUSED_ROPE", "1"),
    "VLLM_ASCEND_DSV41_INDEXER_FUSED_NORM": lambda: os.getenv("VLLM_ASCEND_DSV41_INDEXER_FUSED_NORM", "1"),
    # 视觉 MoE router 走融合算子（调两次 + where 实现按行选表），替掉
    # 702 个 torch 小算子 / 1.292 ms 每步。设 0 退回 torch 分解。
    # 两条路已在卡上对拍：topk_ids 逐位相同、权重差在 fp32 eps 量级。
    "VLLM_ASCEND_DSV41_FUSED_VISION_ROUTER": lambda: os.getenv("VLLM_ASCEND_DSV41_FUSED_VISION_ROUTER", "1"),
    # DeepSeek-V4.1 indexer K write: quantise and store through CANN's
    # indexer_quant_cache instead of indexer_compress_epilog_v2.
    # k_norm/RoPE/slot gating are unchanged either way.
    #
    # 🔴 Default "1" since the MXFP4 index K cache (batch 1). This is no
    # longer a performance choice: the cache row is 68 bytes (64 packed E2M1
    # values + 4 E8M0 scales) and indexer_quant_cache with
    # quant_mode="mxfp4" is the only writer that produces it.
    # indexer_compress_epilog_v2 writes the fp8 row (128 values + one fp32
    # scale = 132 bytes) and refuses this cache rather than filling it
    # (indexer_k_store._refuse_epilog_on_an_mxfp4_row).
    #
    # ⚠ So "0" is not a rollback on its own -- it has to move together with
    # the cache spec in caches.py, in one change. Half of it leaves two page
    # sizes in one KV cache group, and MLA specs cannot be page-padded
    # (vllm/v1/core/kv_cache_utils.py:1392-1397 raises).
    #
    # The earlier default was "0" pending a GSM8K/TPOT comparison on the card
    # (CARD_TEST_TABLE 戊); that comparison was of fp8-vs-fp8 quantisers and
    # no longer decides this flag.
    "VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE": lambda: os.getenv("VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE", "1"),
    # DeepSeek-V4.1 graph mode without VLLM_USE_BREAKABLE_CUDAGRAPH=1: "1"
    # allows it, anything else refuses to start. Default "0". Not a
    # performance switch -- it is the deliberate opt-out for the preflight in
    # platform.py:_require_validated_dsv41_graph_mode, which exists because
    # forgetting VLLM_USE_BREAKABLE_CUDAGRAPH=1 is otherwise silent: the run
    # takes the piecewise torch.compile + plain ACLGraph path that no DSV4.1
    # accuracy or soak number covers. Set it only to run that path on purpose.
    "VLLM_ASCEND_DSV41_ALLOW_UNVALIDATED_GRAPH_MODE": lambda: os.getenv(
        "VLLM_ASCEND_DSV41_ALLOW_UNVALIDATED_GRAPH_MODE", "0"),
    # DeepSeek-V4.1 MoE offload (M0c): output directory for the per-layer
    # routed-expert frequency counter (vllm_ascend/kt_offload/expert_counter.py).
    # Empty (default) = off. Diagnostic only; requires --enforce-eager. Read
    # once at import. Not sensitive.
    "VLLM_ASCEND_DSV41_EXPERT_COUNT": lambda: os.getenv("VLLM_ASCEND_DSV41_EXPERT_COUNT", ""),
    # DeepSeek-V4.1 MoE offload: sampling interval in seconds for the per-layer
    # host-func counters (vllm_ascend/kt_offload/hostfunc_log.py). Only a
    # host-func node inside the graph increments them, so a layer whose count
    # does not grow while decoding has no CPU-expert work in the graph. "" or
    # "0" (default) = off. Diagnostic; read once at import. Not sensitive.
    "VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG": lambda: os.getenv("VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG", ""),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
