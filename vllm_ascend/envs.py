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
    # DeepSeek-V4.1 indexer K write: "1" quantises and stores through CANN's
    # indexer_quant_cache (fp8 mode, round_scale=True: power-of-two scales,
    # byte-equal to upstream's quantiser, card A6) instead of
    # indexer_compress_epilog_v2 (scale absmax/448, not a power of two; N5).
    # k_norm/RoPE/slot gating are unchanged either way. Off until GSM8K and
    # TPOT are compared on the card (CARD_TEST_TABLE 戊).
    "VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE": lambda: os.getenv("VLLM_ASCEND_DSV41_INDEXER_QUANT_CACHE", "0"),
    # DeepSeek-V4.1 graph mode without VLLM_USE_BREAKABLE_CUDAGRAPH=1: "1"
    # allows it, anything else refuses to start. Default "0". Not a
    # performance switch -- it is the deliberate opt-out for the preflight in
    # platform.py:_require_validated_dsv41_graph_mode, which exists because
    # forgetting VLLM_USE_BREAKABLE_CUDAGRAPH=1 is otherwise silent: the run
    # takes the piecewise torch.compile + plain ACLGraph path that no DSV4.1
    # accuracy or soak number covers. Set it only to run that path on purpose.
    "VLLM_ASCEND_DSV41_ALLOW_UNVALIDATED_GRAPH_MODE": lambda: os.getenv(
        "VLLM_ASCEND_DSV41_ALLOW_UNVALIDATED_GRAPH_MODE", "0"),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
