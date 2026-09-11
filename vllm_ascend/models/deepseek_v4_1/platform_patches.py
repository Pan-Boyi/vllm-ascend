# SPDX-License-Identifier: Apache-2.0
"""Import-time platform adaptations for DeepSeek-V4.1 on Ascend.

Everything here rebinds a name upstream already exposes for platform
differences, rather than forking the code around it. Applied at import, before
any forward pass, because one of them guards a compiler crash that no
``try``/``except`` can catch.
"""

from __future__ import annotations

from typing import Any

#: Upstream's switch for reading a ratio-2 request's two pooled rows as one
#: tile. Its comment already records that ROCm's Triton cannot legalize
#: ``tt.join`` on pointers; Ascend's cannot either.
JOIN_FLAG = "_JOIN_ROW_PTRS"


def _compress_module() -> Any:
    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as mod

    return mod


def disable_join_row_ptrs() -> bool:
    """Take the separate-loads path in the compressor kernel. Idempotent.

    On the joined path Ascend's compiler does not raise -- it aborts, in
    ``OffsetAnalysis.cpp`` while parsing a pointer type, taking the process and
    a core file with it. So this cannot be a runtime fallback: the flag is read
    as a ``tl.constexpr`` when the kernel is first compiled, and by then there
    is nothing left to recover into.

    The two branches are arithmetically the same -- ``tl.join`` plus a two-line
    softmax against two loads plus an explicit online softmax -- so this changes
    scheduling, not semantics. Upstream reaches the joined path here only
    because its guard tests for ROCm specifically rather than for the
    limitation.

    Returns True if the flag was changed.
    """
    mod = _compress_module()
    if not hasattr(mod, JOIN_FLAG):
        raise AttributeError(
            f"upstream no longer defines {JOIN_FLAG!r} in "
            "fused_compress_quant_cache. The joined-pointer path may have been "
            "removed, or renamed -- but if it still exists and nothing selects "
            "away from it, compiling a ratio-2 compressor here aborts the "
            "process. Re-read the module before assuming this patch is "
            "obsolete.")
    if getattr(mod, JOIN_FLAG) is False:
        return False
    setattr(mod, JOIN_FLAG, False)
    return True


def join_row_ptrs_disabled() -> bool:
    """Whether the compressor will take the separate-loads path."""
    return getattr(_compress_module(), JOIN_FLAG) is False


def assert_ready_for_ratio_two(where: str) -> None:
    """Refuse to reach a ratio-2 compressor compile with the joined path live.

    Without this the failure surfaces inside the compiler: it aborts the
    process in OffsetAnalysis.cpp and leaves a core file, at whatever moment
    the first ratio-2 kernel happens to be compiled -- which is well after
    everything looks configured. Checking here moves it somewhere readable.
    """
    if not join_row_ptrs_disabled():
        raise RuntimeError(
            f"{where}: compressor ratio 2 is about to be compiled while "
            f"{JOIN_FLAG} is still true. Ascend's compiler does not raise on "
            "that path -- it aborts the process. apply_all() has not run, or "
            "something re-imported the module afterwards.")


def apply_all() -> list[str]:
    """Apply every adaptation and report which. Safe to call more than once.

    Returns the names applied, and refuses to return quietly having done
    nothing: a patch set that silently becomes a no-op looks exactly like one
    that worked, and these two guard a compiler abort and a CUDA-only class
    selection respectively.
    """
    from vllm_ascend.models.deepseek_v4_1.attn_selector import \
        apply_attn_selector_patch

    applied = []
    disable_join_row_ptrs()
    if join_row_ptrs_disabled():
        applied.append(JOIN_FLAG)
    apply_attn_selector_patch()
    applied.append("_select_dsv4_attn_cls")

    if not applied:
        raise RuntimeError(
            "apply_all() applied nothing. Either upstream changed shape and "
            "every patch became a no-op, or this is being called against a "
            "module set it does not recognise.")
    return applied
