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
    # Build only the vllm_ascend_C extension and leave vllm_ascend/_cann_ops_custom as it is. Setting
    # COMPILE_CUSTOM_KERNELS=0 does not achieve this: it also clears ext_modules, so nothing is built
    # at all. Useful when a change touches only the torch bindings, since rebuilding the aclnn package
    # wipes and repopulates 132 MB of installed operators for no reason.
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
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
    # DeepSeek-V4.1 layer-by-layer alignment against the CPU golden reference.
    # All four are debug-only and off by default; they cost nothing when unset.
    # Directory to write NPU dumps into, named per the alignment protocol.
    "VLLM_ASCEND_DSV41_DUMP_DIR": lambda: os.getenv("VLLM_ASCEND_DSV41_DUMP_DIR", ""),
    # Golden's full dump directory (golden_dumps/full40). Required for
    # 'layer_in' injection; also the fallback routing source.
    "VLLM_ASCEND_DSV41_GOLDEN_DIR": lambda: os.getenv("VLLM_ASCEND_DSV41_GOLDEN_DIR", ""),
    # Golden's compact routing plan (golden_dumps/plan_full40), 58 KB versus
    # 271 MB for the full dump; preferred source for routing injection.
    "VLLM_ASCEND_DSV41_ROUTING_PLAN": lambda: os.getenv("VLLM_ASCEND_DSV41_ROUTING_PLAN", ""),
    # Which decisions to take from golden instead of computing them:
    # "off" (default) | "moe" | "moe+layer_in" | "moe+indexer"; flags join with "+".
    "VLLM_ASCEND_DSV41_FORCE_ROUTE": lambda: os.getenv("VLLM_ASCEND_DSV41_FORCE_ROUTE", "off"),
    # DeepSeek-V4.1 mHC post: "1" uses upstream's torch reference instead of
    # aclnnHcPost. Default "0" (the operator, which is also V4.0's production
    # path). Exists to A/B the two on a card -- they compute the same
    # expression, and "the same expression" has been wrong here before.
    "VLLM_ASCEND_DSV41_MHC_POST_TORCH": lambda: os.getenv("VLLM_ASCEND_DSV41_MHC_POST_TORCH", "0"),
    # DeepSeek-V4.1 mHC pre: "1" uses npu_mhc_pre + npu_mhc_sinkhorn instead of
    # upstream's torch reference. Default "0" until the operator chain is
    # compared numerically at hidden 5120 -- it has been checked at 4096, and
    # at 5120 only for producing finite values.
    "VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR": lambda: os.getenv("VLLM_ASCEND_DSV41_MHC_PRE_OPERATOR", "0"),
    # DeepSeek-V4.1 engram dequantise: "1" takes a torch route instead of the
    # Triton kernel. A bisection instrument, not a production fallback -- a
    # triton-ascend compiler assertion aborts the process rather than raising,
    # so the only cheap way to attribute one is to remove a kernel and look.
    "VLLM_ASCEND_DSV41_ENGRAM_DEQUANT_TORCH": lambda: os.getenv("VLLM_ASCEND_DSV41_ENGRAM_DEQUANT_TORCH", "0"),
    # DeepSeek-V4.1 vision SWA gate: "1" requires an actual image span before
    # the bidirectional-vision index path runs. Default "0" keeps the current
    # behaviour, which tests the *container*: a text-only request on a
    # multimodal checkpoint gets ``mm_req_doc_ranges = {0: []}`` -- a truthy
    # dict of empty span lists -- so the vision branch fires, the call is put
    # in ori-only sparse mode (mask 0, no window), and a 1152-wide index list
    # is handed to an operator that would otherwise run plain sliding window.
    # Separate from the spec switch below on purpose: two mechanisms enabled
    # together cannot be attributed by a single launch.
    "VLLM_ASCEND_DSV41_MM_RANGES_REQUIRE_SPANS": lambda: os.getenv("VLLM_ASCEND_DSV41_MM_RANGES_REQUIRE_SPANS", "0"),
    # Whether an absent index list, on its own, selects sliding-window mode --
    # even on a layer that also attends a compressed stream. Default "0" keeps
    # today's behaviour, which sends ori_mask_mode=0 (NO_MASK) there.
    "VLLM_ASCEND_DSV41_ORI_WINDOW_WITHOUT_INDICES": lambda: os.getenv("VLLM_ASCEND_DSV41_ORI_WINDOW_WITHOUT_INDICES", "0"),
    # How the engram staging H2D is made safe against its host source being
    # rewritten: "0"/"" (today), "block", "sync", "double". See
    # engram_embedding.H2D_MODES -- they are separate rungs on purpose.
    "VLLM_ASCEND_DSV41_ENGRAM_BLOCKING_H2D": lambda: os.getenv("VLLM_ASCEND_DSV41_ENGRAM_BLOCKING_H2D", "0"),
    # Per-layer output fingerprints in the log: "all", or a comma list of
    # layer indices. Empty (the default) registers no hooks at all.
    "VLLM_ASCEND_DSV41_TRACE_LAYERS": lambda: os.getenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", ""),
    # DeepSeek-V4.1 kv-cache spec: "1" makes the layer's configured
    # compress_ratios entry -- 0 included -- reach the metadata builder as its
    # own spec field. Default "0" keeps today's behaviour, where ratio 0 is
    # clamped to 1 for the page arithmetic and read back as "this layer
    # attends a compressed stream". Neither tokens_per_state nor
    # AscendSlidingWindowMLASpec.compress_ratio can carry it: both divide the
    # page size, and 0 is not a legal divisor.
    #
    # 🔴 MEASURED, and it is why this is still off: turning it on moves a
    # variable that has two jobs. The builder's ``compressor_ratio`` names the
    # stream *and* feeds ``tokens_per_state`` (dsa_v1.py:1168/1681), which
    # ``AscendDSAMetadata.block_size`` multiplies the storage page by
    # (dsa_v1.py:596). Redirecting the reader therefore moved the page size
    # too: a ratio-2 layer reported block_size 32 * 2 = 64 against a real
    # (55737, 32, 512) cache, and the SWA insert guard refused it. Reviving
    # this means splitting that variable in two -- a page ratio on the old
    # chain and a stream ratio on this field -- not re-enabling it as written.
    #
    # ⚪ The grouping change predicted here did not happen: measured, kv-cache
    # groups are already one layer each (max group id 42), so no merge ever
    # sees two streams and the refusal in _merged_stream_compress_ratio is
    # unreachable in this configuration.
    "VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO": lambda: os.getenv("VLLM_ASCEND_DSV41_SPEC_STREAM_RATIO", "0"),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
