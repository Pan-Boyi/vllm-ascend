# SPDX-License-Identifier: Apache-2.0
"""Registry entry point for DeepSeek-V4.1-Flash on Ascend.

Subclasses upstream's model instead of forking it; the platform differences
(attention class selection, construction-site rebinds and one compiler flag)
are applied when this package is imported, so there is no path that reaches
the model without them. A drifted rebind fails loudly, a drifted fork would
not.
"""

from __future__ import annotations

from vllm.models.deepseek_v4_1.nvidia.vl_model import DeepseekV41ForCausalLM

from vllm_ascend.models.deepseek_v4_1.platform_patches import (
    CLASS_REBINDS, join_row_ptrs_disabled)


class AscendDeepseekV41ForCausalLM(DeepseekV41ForCausalLM):
    """DeepSeek-V4.1-Flash on Ascend.

    The constructor checks that the platform patches are in force; otherwise
    the model would build out of CUDA-selected parts and fail much later.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        _assert_patches_applied()
        super().__init__(vllm_config=vllm_config, prefix=prefix)


def _assert_patches_applied() -> None:
    """Refuse to build out of upstream's CUDA-selected parts.

    Reads the live bindings rather than the list ``apply_all`` returned: an
    upstream module re-imported afterwards would restore its original names
    while that list still reports success.
    """
    import importlib

    from vllm_ascend.models.deepseek_v4_1 import caches

    wrong = []
    for module_name, upstream_name, ascend_name in CLASS_REBINDS:
        module = importlib.import_module(module_name)
        bound = getattr(module, upstream_name, None)
        if bound is not getattr(caches, ascend_name):
            wrong.append(
                f"{module_name}.{upstream_name} is "
                f"{getattr(bound, '__name__', bound)!r}, expected {ascend_name}")

    if wrong or not join_row_ptrs_disabled():
        raise RuntimeError(
            "DeepSeek-V4.1 platform patches are not in force:\n  "
            + "\n  ".join(wrong or ["(rebinds are fine)"])
            + f"\n  join_row_ptrs_disabled={join_row_ptrs_disabled()}\n"
            "Importing vllm_ascend.models.deepseek_v4_1 applies them, so this "
            "means something re-imported an upstream module afterwards and "
            "restored its original names.")
