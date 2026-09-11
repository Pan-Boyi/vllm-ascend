# SPDX-License-Identifier: Apache-2.0
"""The platform patch must be in place before any upstream model is imported.

V4.0's model module does, at import time::

    from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory

A from-import binds the name into that module's namespace then and there.
vllm_ascend's platform patch replaces the *module attribute*, which does
nothing for a name already bound somewhere else. So whichever happens first
decides, permanently, and nothing reports the loss.

This package's apply_all() reaches into upstream's V4.1 model module to rebind
the attention selector and the mHC functions, and that transitively imports
V4.0's. Importing it before the platform patch pinned the upstream factory, so
the Ascend MoE router was never installed and the first launch to reach the
MoE died in topk_softplus_sqrt -- a CUDA entry point -- with the Ascend
equivalents sitting there unused.

Run in subprocesses because import state is global: a test that checked this
in-process would be measuring whatever an earlier test had already imported.
"""

import subprocess
import sys
import textwrap

import pytest

PREAMBLE = "import vllm_ascend.ops\n"


def _run(body: str) -> str:
    # dedent the body *then* prepend: dedent on the concatenation finds no
    # common prefix, because the preamble has none, and silently does nothing.
    script = PREAMBLE + textwrap.dedent(body)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout.strip().splitlines()[-1]


def test_importing_this_package_leaves_the_factory_patched():
    """The property that was broken. Importing the package is the whole of
    what the model registry does, so this is the real path."""
    got = _run("""
        import vllm_ascend.models.deepseek_v4_1  # noqa: F401
        import vllm.models.deepseek_v4.nvidia.model as v40
        print(v40.FusedMoEFactory.__name__)
    """)
    assert got == "_ascend_FusedMoE", (
        f"V4.0's local binding is {got!r}; the Ascend MoE router will not be "
        "installed and the routing will reach a CUDA-only entry point")


def test_the_ordering_is_what_decides_it():
    """Both arms, so the test cannot pass because the patch happens to be
    unnecessary. Importing first and patching afterwards must still lose."""
    patched_first = _run("""
        from vllm_ascend.utils import adapt_patch
        adapt_patch(is_global_patch=True)
        import vllm.models.deepseek_v4.nvidia.model as v40
        print(v40.FusedMoEFactory.__name__)
    """)
    imported_first = _run("""
        import vllm.models.deepseek_v4.nvidia.model as v40
        from vllm_ascend.utils import adapt_patch
        adapt_patch(is_global_patch=True)
        print(v40.FusedMoEFactory.__name__)
    """)
    assert patched_first == "_ascend_FusedMoE"
    assert imported_first == "FusedMoEFactory", (
        "if this stops being true the from-import is no longer the mechanism "
        "and this test is measuring something else")


def test_apply_all_does_it_rather_than_relying_on_the_caller():
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import platform_patches

    source = inspect.getsource(platform_patches.apply_all)
    assert "_apply_platform_patches_first()" in source
    # ... and before the rebinds, which are what do the importing
    assert (source.index("_apply_platform_patches_first()")
            < source.index("apply_attn_selector_patch()"))
