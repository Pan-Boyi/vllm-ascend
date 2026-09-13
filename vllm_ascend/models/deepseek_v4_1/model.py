# SPDX-License-Identifier: Apache-2.0
"""Registry entry point for DeepSeek-V4.1-Flash on Ascend.

Deliberately thin. V4.0's Ascend port forks upstream's model into a 1352-line
file; this one subclasses upstream's and lets the platform differences be
handled where they actually are -- the attention class selector, three
construction-site rebinds, and one compiler flag, all applied when this
package is imported.

That choice is about which failure mode is preferred, not about effort. A
rebind that drifts fails loudly: the name disappears, ``apply_all`` raises and
says which one. A forked model file that drifts produces numbers that are
quietly different from upstream's, which is the failure this port has spent
the most time chasing.

The architecture string in the checkpoint is ``DeepseekV41ForCausalLM``, which
upstream maps to its *multimodal* wrapper -- a vision tower and aligner built
unconditionally, with the language model underneath. That is upstream's
structure and it is inherited as is.

Importing this module runs the package ``__init__``, which applies the
platform patches. That ordering is the whole reason the patches live in the
package rather than in the model registry: there is no path that reaches the
model without them.
"""

from __future__ import annotations

from vllm.models.deepseek_v4_1.nvidia.vl_model import DeepseekV41ForCausalLM

from vllm_ascend.models.deepseek_v4_1.platform_patches import (
    CLASS_REBINDS, join_row_ptrs_disabled)


class AscendDeepseekV41ForCausalLM(DeepseekV41ForCausalLM):
    """DeepSeek-V4.1-Flash on Ascend.

    Upstream's class, reached after the platform patches have been applied.
    The constructor checks that they were, because the alternative is a model
    that builds successfully out of CUDA-selected parts and fails much later,
    somewhere that says nothing about why.
    """

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        _assert_patches_applied()
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # 🔴 The alignment taps' only production call site. Without it the
        # hooks are never registered, and the symptom is not "no hooks" -- the
        # six inline taps in attention.py still run, reach record() before any
        # set_seq_len(), and raise, so the dump comes back empty and the cause
        # is two inferences away. That happened twice.
        #
        # Here rather than in the runner because this is the one place that
        # sees the built model and belongs to this line. It returns an empty
        # list and registers nothing when no dump directory is set, so the
        # off case is unchanged by construction.
        from vllm_ascend.models.deepseek_v4_1.align_taps import \
            install_alignment_taps

        self._alignment_tap_handles = install_alignment_taps(self)


def _assert_patches_applied() -> None:
    """Refuse to build out of upstream's CUDA-selected parts.

    Each of these is silent when missing, in its own way: an unpatched
    selector builds the CUDA attention class, an unpatched engram table asks
    for 188.83 GiB of device memory, and an unpatched compiler flag aborts the
    process rather than raising when the first ratio-2 kernel is compiled.

    This reads the bindings themselves rather than the list ``apply_all``
    returned. The distinction is the whole point: that list records what was
    true once, at import, and the failure being guarded against is precisely
    something re-importing an upstream module afterwards and restoring its
    original names -- which leaves the list saying everything is fine.
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
