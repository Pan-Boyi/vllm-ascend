"""Shim: the reference implementation imports ``kernel``; give it the CPU one.

inference/kernel.py is tilelang/CUDA-only.  Nothing else in the reference tree is
modified -- these are verbatim copies of the files from
/workspace/models/DeepSeek-V4.1-Flash/inference/ (which stays read-only).
"""

from kernel_cpu import *          # noqa: F401,F403
from kernel_cpu import (          # noqa: F401  -- explicit, for `from kernel import ...`
    act_quant,
    fp4_act_quant,
    fp4_gemm,
    fp8_gemm,
    hc_split_sinkhorn,
    sparse_attn,
)
