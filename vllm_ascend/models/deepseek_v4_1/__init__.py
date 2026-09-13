# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on Ascend.

Importing this package applies the platform adaptations, because one of them
selects away from a compiler path that aborts the process rather than raising,
and the flag it sets is read when a kernel is first compiled -- by which point
there is nothing to recover into.

They live here rather than in the model registry so that their scope is "V4.1
was actually loaded". Registered globally they would also reach V4.0, whose
compressor has no need of them.
"""

from vllm_ascend.models.deepseek_v4_1.platform_patches import apply_all

APPLIED_PATCHES = apply_all()

# 🔴 打一行，因为「没有证据」和「没发生」在日志里长得一样。
#    apply_all() 本身是 fail-safe 的（一个都没打到会 raise，上游改名会 raise），
#    所以这一行不是守卫 —— 它是【证据】。承重的那一条是 _JOIN_ROW_PTRS：
#    上游默认 `not is_rocm()` ⇒ 在昇腾上默认是 True，而 True 那条路
#    编译器【不 raise、直接 abort】，带走进程和一个 core。
#    ⇒ 任何一发的日志里都该能当场看到它被关掉了，不必事后去推。
_logger = None
try:
    from vllm.logger import logger as _logger
except Exception:  # 日志不可用不该影响 import
    pass
if _logger is not None:
    from vllm_ascend.models.deepseek_v4_1.platform_patches import (
        JOIN_FLAG, join_row_ptrs_disabled)
    _logger.info(
        "DSV4.1 平台补丁已应用 %d 项: %s | 承重项 %s 已关闭=%s",
        len(APPLIED_PATCHES), ", ".join(APPLIED_PATCHES),
        JOIN_FLAG, join_row_ptrs_disabled())

__all__ = ["APPLIED_PATCHES"]
