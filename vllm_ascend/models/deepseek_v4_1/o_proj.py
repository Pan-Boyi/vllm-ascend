# SPDX-License-Identifier: Apache-2.0
"""Output projection for DeepSeek-V4.1-Flash attention.

After the inverse RoPE, attention output goes through two projections:

    o        [T, 64, 512]            64 heads x head_dim
      view   [T, 8, 4096]            o_groups = 8, each seeing 8 heads
      wo_a   [8, 1024, 4096]         block diagonal -- each group has its own
                                     4096 -> 1024 matrix (o_lora_rank = 1024)
      flat   [T, 8192]
      wo_b   [5120, 8192]            an ordinary linear
    result   [T, 5120]

``wo_a`` is stored as one ``[8192, 4096]`` tensor. A dense matmul against it is
*not* the hazard -- the shapes do not line up, so that mistake fails loudly. The
silent one is the split: reshaping ``[T, 32768]`` into ``[T, 8, 4096]`` decides
which input slice each group sees, and getting that ordering wrong keeps every
shape valid while mixing the wrong 4096 dimensions into each group.

The quantisation side already reshapes ``wo_a`` into ``[o_groups, ...]`` during
``process_weights_after_loading``, so a wired-up implementation receives it
grouped; the reshape here is for tests and for any path that gets the flat
tensor.
"""

from __future__ import annotations

import torch

#: Exponent bias for the ue8m0 block scales the checkpoint stores.
E8M0_BIAS = 127
#: Quantisation block shape for non-expert weights (config weight_block_size).
QUANT_BLOCK = 32


def dequantize_block_scaled(weight: torch.Tensor,
                            scale: torch.Tensor,
                            block: int = QUANT_BLOCK) -> torch.Tensor:
    """Undo fp8 + ue8m0 block-scale quantisation into fp32.

    ``scale`` holds one exponent per ``block x block`` tile of ``weight``; the
    stored byte is the exponent plus a bias, so the multiplier is a power of two
    and the multiply itself is exact.
    """
    rows, cols = weight.shape
    if scale.shape != (rows // block, cols // block):
        raise ValueError(
            f"scale {tuple(scale.shape)} does not tile weight {tuple(weight.shape)} "
            f"at block {block}")

    # torch decodes float8_e8m0fnu to the multiplier itself (byte 127 -> 1.0),
    # so the bias is already gone. Subtracting it again yields 2**(1-127),
    # i.e. zero in fp32 -- silently turning the whole matrix into zeros.
    if scale.dtype == torch.float8_e8m0fnu:
        factors = scale.to(torch.float32)
    else:
        factors = torch.pow(2.0, scale.to(torch.float32) - E8M0_BIAS)
    factors = factors.repeat_interleave(block, 0).repeat_interleave(block, 1)
    return weight.to(torch.float32) * factors


def group_wo_a(wo_a: torch.Tensor, o_groups: int, o_lora_rank: int) -> torch.Tensor:
    """Reshape the stored ``[o_groups * o_lora_rank, group_in]`` into groups."""
    if wo_a.shape[0] != o_groups * o_lora_rank:
        raise ValueError(
            f"wo_a has {wo_a.shape[0]} rows, expected o_groups * o_lora_rank = "
            f"{o_groups} * {o_lora_rank} = {o_groups * o_lora_rank}")
    return wo_a.view(o_groups, o_lora_rank, wo_a.shape[1])


def block_diagonal_wo_a(o: torch.Tensor, wo_a_grouped: torch.Tensor) -> torch.Tensor:
    """Apply the per-group projection.

    ``o`` is ``[..., tokens, heads, head_dim]`` or already grouped as
    ``[..., tokens, o_groups, group_in]``; ``wo_a_grouped`` is
    ``[o_groups, o_lora_rank, group_in]``. Returns ``[..., tokens, o_groups,
    o_lora_rank]``.

    Each group multiplies only its own slice. Contrast a dense matmul against
    the flat ``[8192, 4096]`` form, which mixes every group's input.
    """
    o_groups, o_lora_rank, group_in = wo_a_grouped.shape
    grouped = o.reshape(*o.shape[:-2], o_groups, group_in)
    return torch.einsum("...tgd,grd->...tgr", grouped.float(),
                        wo_a_grouped.float())


def o_projection(o: torch.Tensor,
                 wo_a_grouped: torch.Tensor,
                 wo_b: torch.Tensor) -> torch.Tensor:
    """Full projection: block-diagonal wo_a, then the dense wo_b.

    Reference path in fp32. A wired-up implementation hands wo_b to the
    quantisation method instead -- the fp8 activation quantisation there is not
    optional: dequantising both matrices to bf16 and multiplying leaves a
    relative error of 1.8e-2 on a single layer, versus 3.5e-5 when wo_b runs as
    a real block-scaled fp8 GEMM.
    """
    z = block_diagonal_wo_a(o, wo_a_grouped)
    return z.flatten(-2) @ wo_b.float().t()


#: Layout the Ascend load-time reshapes leave ``wo_a.weight`` in. Note this is
#: the transpose of ``group_wo_a`` above and of upstream's DeepGEMM layout
#: ``[o_groups, o_lora_rank, group_in]``: ``npu_transpose_batchmatmul`` wants
#: the contraction dimension in the middle. Both sites that build it agree --
#: ``vllm_ascend/ops/linear.py`` (bf16, at weight_loader time) and
#: ``vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py`` (mxfp8, at
#: process_weights_after_loading time).
ASCEND_GROUPED_WO_A_LAYOUT = "[o_groups, group_in, o_lora_rank]"


def check_grouped_wo_a_layout(weight: torch.Tensor, o_groups: int,
                              group_in: int, o_lora_rank: int,
                              where: str) -> None:
    """Refuse a ``wo_a`` that was never reshaped into groups.

    Not because the wrong layout would go unnoticed -- for these particular
    sizes (group_in 4096, o_lora_rank 1024) none of the wrong shapes multiply
    cleanly, so something would fail either way. The point is *what* fails:
    without this, the report is a shape mismatch from inside
    ``npu_transpose_batchmatmul``, naming two tensors and no reason, at the
    first forward pass. Nothing in it suggests that a step at weight-load time
    was skipped, which is where the fix actually is.

    That step is easy to skip. Nothing on Ascend honours
    ``is_bmm``/``bmm_batch_size``, which is how upstream marks this weight; the
    grouping is instead reconstructed from ``prefix.endswith("wo_a")`` in two
    independent places, and only for some quantisation schemes -- a plain
    block-fp8 linear method leaves the weight 2-D. So "the reshape happened" is
    a property of which method served the layer, not of the model.

    The transposed case has its own branch because it is the easy mistake
    rather than a rare one: ``group_wo_a`` above, upstream's DeepGEMM path and
    the ROCm einsum all use ``[o_groups, o_lora_rank, group_in]``, and only the
    Ascend operators want the contraction dimension in the middle.
    """
    expected = (o_groups, group_in, o_lora_rank)
    if tuple(weight.shape) == expected:
        return

    detail = (f"{where}: wo_a has shape {tuple(weight.shape)}, expected "
              f"{expected} -- {ASCEND_GROUPED_WO_A_LAYOUT}.")
    if weight.ndim == 2 and weight.shape[0] == o_groups * o_lora_rank:
        raise ValueError(
            detail + " The weight is still in its flat checkpoint form, so "
            "the load-time grouping did not run for this layer. Look at which "
            "quantisation method served this linear rather than at the "
            "projection: only some of them reshape wo_a, and they select it by "
            "the prefix ending in 'wo_a' -- nothing on Ascend reads the "
            "is_bmm flag upstream sets for exactly this purpose.")
    if tuple(weight.shape) == (o_groups, o_lora_rank, group_in):
        raise ValueError(
            detail + " This is the transposed layout -- upstream's DeepGEMM "
            "and group_wo_a() order, not the one npu_transpose_batchmatmul "
            "takes. Transpose the last two dimensions.")
    raise ValueError(detail)
