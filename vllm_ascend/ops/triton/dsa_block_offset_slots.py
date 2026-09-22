# SPDX-License-Identifier: Apache-2.0
"""One kernel for DSA's ``[T] -> [T, 2]`` (block, offset) slot mapping.

The torch spelling in ``dsa_attn_kv_plan.format_dsa_slot_mapping`` is eight
operators (``ge``, ``full_like``, ``div``, ``where``, ``%``, ``where``,
``stack``, ``to``), and the builder then copies the result into its resident
buffer -- nine dispatches. It runs once per KV-cache group, 42 times per
decode step.

🔴 **Why fusing this is worth anything at all**: the work is trivially cheap
and the cost is almost entirely launch overhead. Measured 2026-09-22 on the
eager stream (stream 61, bs=1/4k/FULL aclgraph): ~1503 kernels/step for
1.612 ms, i.e. ~1.07 us per kernel for what is a handful of elementwise ops
on a [T] vector with T <= 8. Cutting nine dispatches to one is the entire
point; the arithmetic was never the problem.

⚠️ **This is not the "memoise across groups" idea, which is dead.** Measured
the same day: of the 42 calls per step, 41 have distinct
``slot_mapping.data_ptr()`` -- each group carries its own slot mapping, so
there is nothing to reuse. Worse, only 2 distinct ``(ratio, block_size)``
pairs exist across those 42 calls, so a structural cache key would hand 40
groups another group's physical slots. See the report's 3.3.

``out`` writes straight into the builder's resident ``[max_tokens, 2]``
buffer, which is what removes the ninth dispatch. That buffer's address is
baked into the FULL graph, so writing in place is required, not merely
convenient (see commit 48d3202a6, where a per-step allocation was read at
its captured address).
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["num_tokens", "block_size"])
def _dsa_block_offset_slots_kernel(
    slot_mapping_ptr,  # [num_tokens], int64
    out_ptr,  # [num_tokens, 2], int32
    num_tokens,
    block_size,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_id in range(pid, n_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_tokens
        # ``other=-1`` keeps out-of-range lanes on the invalid branch, so a
        # masked lane can never produce a real slot.
        slot = tl.load(slot_mapping_ptr + offsets, mask=mask, other=-1)
        valid = slot >= 0
        # Only the valid branch is kept, so floor- vs trunc-division on the
        # negative sentinel cannot change the result. For slot >= 0 the two
        # agree, which is what makes this bit-identical to the torch path.
        block_idx = tl.where(valid, slot // block_size, -1).to(tl.int32)
        offset = tl.where(valid, slot % block_size, -1).to(tl.int32)
        # Interleaved store: row i is (out[2i], out[2i + 1]).
        tl.store(out_ptr + offsets * 2, block_idx, mask=mask)
        tl.store(out_ptr + offsets * 2 + 1, offset, mask=mask)


def dsa_block_offset_slots(
    slot_mapping: torch.Tensor,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``[T]`` flat slots -> ``[T, 2]`` int32 (block, offset), ``-1`` preserved.

    ``out``, when given, must be a contiguous ``[T, 2]`` int32 tensor and is
    written in place; this is how the caller avoids a separate copy into its
    resident buffer.
    """
    if slot_mapping.ndim != 1:
        raise ValueError(
            f"flat slot mapping must be [num_tokens], got {tuple(slot_mapping.shape)}"
        )
    num_tokens = slot_mapping.shape[0]
    if out is None:
        out = torch.empty(
            (num_tokens, 2), dtype=torch.int32, device=slot_mapping.device
        )
    else:
        # A non-contiguous or mis-shaped destination would be written with the
        # interleaved stride above and produce plausible-looking wrong slots,
        # so it is refused rather than reinterpreted.
        if tuple(out.shape) != (num_tokens, 2):
            raise ValueError(
                f"out must be [{num_tokens}, 2], got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32:
            raise ValueError(f"out must be int32, got {out.dtype}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    if num_tokens == 0:
        return out

    # One program per tile, not the persistent-grid form the other triton ops
    # in this tree use. num_tokens is at most max_num_batched_tokens (4096
    # here) and 1 in bs=1 decode, so n_blocks is 1..4 -- fewer tiles than
    # cores either way, and a persistent grid would only add a dependency on
    # ``get_vectorcore_num()``, which asserts unless
    # ``init_device_properties_triton()`` has run. This is called from the
    # metadata builder, and buying nothing in exchange for an initialisation
    # order requirement is a bad trade.
    BLOCK_SIZE = 1024
    n_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    _dsa_block_offset_slots_kernel[(n_blocks,)](
        slot_mapping,
        out,
        num_tokens,
        block_size,
        n_blocks,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
