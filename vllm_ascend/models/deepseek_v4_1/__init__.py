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

__all__ = ["APPLIED_PATCHES"]
