# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Refusals for csrc operators whose OWN switch is ``True`` by default.

Why these two, and why here
---------------------------
R44 inventoried the operators that ``csrc`` builds but the A5 operator package
does not install. Two of them are reached by a switch that DEFAULTS TO TRUE:

* ``TransposeKvCacheByBlock`` --
  ``additional_config.enable_transpose_kv_cache_by_block`` (``ascend_config.py``)
  defaults to ``True``; the fused path additionally needs ``enable_custom_op()``
  and a Mooncake P2P KV-transfer deployment.
* ``AddRmsNormBias`` -- no user switch at all; ``enable_custom_op()`` alone
  selects it in ``ops/layernorm.py``. (The norm-quant fusion pass registers
  three patterns that MENTION the same operator; those are deliberately not
  covered here -- see "What this module does NOT cover" below.)

Neither operator is in the A5 package: its ``aclnn`` symbols are absent from
``libcust_opapi.so`` and its OpType is absent from the ops-info registry. What
keeps both paths off today is ``enable_custom_op()`` returning False, and that
is FALSE FOR A REASON THAT HAS NOTHING TO DO WITH THESE OPERATORS: the A5
hardware profile's capability set does not contain ``RUNTIME_CUSTOM_OPS``
(``vllm_ascend/device/hardware_profile.py``), so ``enable_custom_op`` short
circuits for EVERY custom operator at once. A capability-set edit, a new device
type, a ``_CUSTOM_OP_ENABLED`` override or simply different hardware turns both
paths on again, with nothing in between.

So today they are RESCUED, not REFUSED. This module refuses.

What the criterion is -- and what it deliberately is NOT
-------------------------------------------------------
Each guard below asks ``vllm_ascend.aclnn_availability.require_aclnn_op``
whether the operator can actually RUN: the torch schema, the aclnn entry
points, WHICH library they came out of, and whether the package that supplied
them registers a kernel for the OpType.

It does NOT ask ``enable_custom_op()``. That flag is the accident this change
exists to stop relying on; a guard built on it would restate the rescue as
though it were a decision, and would go green on exactly the machine where the
operator is missing but the flag is on. ``enable_custom_op()`` is a question
about hardware POLICY; "is a kernel installed" is a question about the
ARTIFACT, and only the second one can fail in the direction that matters.

Nothing here reads a device, initialises acl, or allocates a tensor.

Cost
----
``require_aclnn_op`` is memoised inside the probe on the operator name plus the
environment variables its answer depends on, so a repeated call is dict lookups
plus four ``os.environ.get`` -- measured at ~5 us warm on this machine. Both
call paths are unreachable on A5 today (``enable_custom_op()`` is False), so the
measured cost of this change on the delivered configuration is zero.
TODO(trigger: ``enable_custom_op()`` becomes true on the target hardware, i.e.
the A5 profile gains ``RUNTIME_CUSTOM_OPS`` or a new profile is added that has
it): re-measure ``AscendRMSNorm.forward_oot``, which would then call
``require_add_rms_norm_bias`` once per norm per layer per step, and memoise on
``aclnn_availability.environment_fingerprint()`` if it costs more than 1%.

What this module does NOT cover
-------------------------------
* ``vllm_ascend/compilation/passes/norm_quant_fusion_pass.py`` names
  ``npu_add_rms_norm_bias`` in three fusion PATTERNS. Those are pattern
  declarations handed to ``pm.register_replacement`` / ``nge.register_replacement``,
  not a model calling the operator, and whether tracing them ever reaches the
  aclnn layer (as opposed to the registered meta kernel) is not something this
  change established. They are left alone, they are named here so the omission
  is visible, and the R55 test suite scans for them on every run as a live
  positive control: if that scan ever reports zero, the scanner broke.
* The SoC the process will actually run on. No device is touched here, so
  "a kernel for this OpType is registered in the package" is as far as the
  criterion goes; the probe logs that limit rather than dropping it.
"""

import torch
import torch_npu

from vllm_ascend.aclnn_availability import require_aclnn_op

# Operator-argument counts, read off the EXEC_NPU_CMD in csrc -- the tokens
# after the aclnn API name, before the workspace/executor/stream tail that
# EXEC_NPU_CMD appends itself. They are literals copied out of another
# repository file, so the R55 suite re-derives both from the csrc of the
# revision under test; if somebody edits an EXEC_NPU_CMD the test goes red
# rather than these constants going quietly stale.
#
#   csrc/moe/add_rms_norm_bias/add_rms_norm_bias_torch_adpt.h
#       EXEC_NPU_CMD(aclnnAddRmsNormBias, x1, x2, gamma, beta, epsilon, y, rstd, x)
_ADD_RMS_NORM_BIAS_ARG_COUNT = 8
#   csrc/torch_binding.cpp
#       EXEC_NPU_CMD(aclnnTransposeKvCacheByBlock, kCache, vCache, blockIDs,
#                    blockSize, headNum, headDim, splitNum, layerNum)
_TRANSPOSE_KV_CACHE_BY_BLOCK_ARG_COUNT = 8

# torch name -> (aclnn API, OpType). Read off the call sites, not derived: the
# aclnn API is the first token of the operator's EXEC_NPU_CMD and the OpType is
# the argument of its OP_ADD (and, verbatim, its ops-info registry key). No
# rule turns one of these names into the other, so a guessed pair would make a
# guard judge a DIFFERENT operator and still look green.
_ADD_RMS_NORM_BIAS = ("npu_add_rms_norm_bias", "aclnnAddRmsNormBias", "AddRmsNormBias")
_TRANSPOSE_KV_CACHE_BY_BLOCK = (
    "transpose_kv_cache_by_block",
    "aclnnTransposeKvCacheByBlock",
    "TransposeKvCacheByBlock",
)


def require_add_rms_norm_bias():
    """Refuse the fused add+RMSNorm+bias path unless OUR operator is installed.

    Returns the vetted torch op, so that the check and the call are the same
    expression and cannot drift apart.

    Reached from ``AscendRMSNorm.forward_oot`` / ``AscendGemmaRMSNorm.forward_oot``
    when a residual is present and ``enable_custom_op()`` is true. There is no
    user-facing switch: on a machine whose hardware profile has
    ``RUNTIME_CUSTOM_OPS``, this is simply the path.

    Raises RuntimeError naming the responsible party -- the vllm-ascend build,
    the operator-package build, or whoever configured the package search path
    -- rather than letting the call reach EXEC_NPU_CMD, whose own message names
    ``libopapi.so`` and therefore sends the reader to the wrong person.
    """
    return require_aclnn_op(
        torch,
        torch_npu,
        torch_name=_ADD_RMS_NORM_BIAS[0],
        aclnn_api=_ADD_RMS_NORM_BIAS[1],
        op_type=_ADD_RMS_NORM_BIAS[2],
        feature=(
            "fused add+RMSNorm+bias (npu_add_rms_norm_bias). This path has no user "
            "switch: enable_custom_op() alone selects it, so nothing below it was "
            "a decision about this operator"
        ),
        expected_arg_count=_ADD_RMS_NORM_BIAS_ARG_COUNT,
        require_vendor_package=True,
    )


def require_transpose_kv_cache_by_block():
    """Refuse the fused Mooncake KV-cache reformat unless the kernel is installed.

    Returns the vetted torch op, for the same reason as above.

    Reached from ``reformat_kv_cache_with_fused_op`` in both Mooncake P2P
    connectors. Its selector, ``additional_config.enable_transpose_kv_cache_by_block``,
    DEFAULTS TO TRUE; the non-fused ``reformat_kv_cache`` is what runs when the
    selector is off. This guard does not fall back to it: the deployment asked
    for the fused operator and the operator is not there, and silently running
    something else is how a missing kernel becomes a performance mystery instead
    of an error.
    """
    return require_aclnn_op(
        torch,
        torch_npu,
        torch_name=_TRANSPOSE_KV_CACHE_BY_BLOCK[0],
        aclnn_api=_TRANSPOSE_KV_CACHE_BY_BLOCK[1],
        op_type=_TRANSPOSE_KV_CACHE_BY_BLOCK[2],
        feature=(
            "fused Mooncake KV-cache block transpose (transpose_kv_cache_by_block), "
            "selected by additional_config.enable_transpose_kv_cache_by_block, which "
            "DEFAULTS TO TRUE"
        ),
        expected_arg_count=_TRANSPOSE_KV_CACHE_BY_BLOCK_ARG_COUNT,
        require_vendor_package=True,
    )
