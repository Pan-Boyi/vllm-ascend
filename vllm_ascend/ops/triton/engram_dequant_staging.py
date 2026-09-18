"""Dequantise DeepSeek-V4.1 engram rows gathered on the host.

The host gathers the table rows a step needs into ``staging_weight [rows, dim]``
(raw fp8 e4m3 bytes) and ``staging_scale [rows, dim // quant_block]`` (raw ue8m0
bytes); row r of both is the same table row, and this kernel writes it to output
row r in bf16. All row arithmetic lives on the host side (``EngramLayout``).
"""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def engram_dequant_staging(
    staging_weight,  # [rows, DIM]      uint8, raw fp8 e4m3 bytes
    staging_scale,  # [rows, N_SCALES] uint8, raw ue8m0 bytes
    out,  # [rows, DIM]      bf16
    num_rows,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    """One row per loop iteration over a persistent grid (faster than 2-D
    row blocks on A5)."""
    n_scales: tl.constexpr = DIM // QUANT_BLOCK
    cols = tl.arange(0, DIM)
    scale_off = tl.arange(0, n_scales)
    for row in tl.range(tl.program_id(0), num_rows, GRID):
        raw = tl.load(staging_weight + row * DIM + cols)
        values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        # Load the scales contiguously and broadcast them; indexing with
        # cols // QUANT_BLOCK reads wrong values outside 2-D blocks.
        scale = tl.load(staging_scale + row * n_scales + scale_off)
        # ue8m0 is a power of two, so its byte is the fp32 exponent field.
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scaled = tl.reshape(values, (n_scales, QUANT_BLOCK)) * scale[:, None]
        tl.store(out + row * DIM + cols, tl.reshape(scaled, (DIM,)).to(tl.bfloat16))


def launch(staging_weight, staging_scale, out, dim: int, quant_block: int, grid: int = 56) -> None:
    """``grid`` defaults to the A5 vector core count."""
    rows = staging_weight.shape[0]
    if staging_scale.shape[0] != rows or out.shape[0] != rows:
        raise ValueError(
            f"row counts disagree: weight {staging_weight.shape[0]}, "
            f"scale {staging_scale.shape[0]}, out {out.shape[0]}; row r of each "
            "must be the same gathered row"
        )
    if staging_weight.shape[1] != dim or staging_scale.shape[1] != dim // quant_block:
        raise ValueError(
            f"staging shapes {tuple(staging_weight.shape)} / {tuple(staging_scale.shape)} do not "
            f"match dim={dim}, quant_block={quant_block} (expected [rows, {dim}] and "
            f"[rows, {dim // quant_block}])"
        )
    engram_dequant_staging[(min(grid, max(rows, 1)),)](
        staging_weight,
        staging_scale,
        out,
        rows,
        DIM=dim,
        QUANT_BLOCK=quant_block,
        GRID=min(grid, max(rows, 1)),
    )
