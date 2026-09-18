# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on Ascend.

Importing this package applies the platform adaptations. One of them turns off
a compiler path that aborts the process, and its flag is read at first kernel
compile, so it must be set before the model is built. They live here rather
than in the model registry so they only apply when V4.1 is loaded.
"""

from vllm_ascend.models.deepseek_v4_1.platform_patches import apply_all

APPLIED_PATCHES = apply_all()

__all__ = ["APPLIED_PATCHES"]
