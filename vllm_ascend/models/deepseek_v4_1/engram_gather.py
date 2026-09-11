# SPDX-License-Identifier: Apache-2.0
"""Gather engram rows out of a host-resident table, on the AIV cores.

The two engram tables are ~94 GiB each and never enter HBM. They live in one
shared mapping on the host that every rank maps, registered with CANN so the
device can address it, and a Triton (AIV) kernel reads the rows it needs
directly. There is no collective and no data movement besides the gather --
each rank sees the whole table.

Two details are load-bearing and quiet when wrong:

**The address is passed as an integer, not as a tensor.** CANN's own code does
the same (``elastic_buffer.cpp:895-897``). A host-mapped device pointer put in
a tensor argument fails in Triton's launcher, which queries pointer attributes
and gets ``ACL_ERROR_RT_DRV_INTERNAL_ERROR`` -- before the kernel runs, and with
an error that reads like the hardware refusing the access.

**The scale is one exponent per 32 values along the row, not a 2-D block grid.**
Non-expert weights use 32x32 tiles and engram does not; feeding a row of engram
scales to a 2-D dequantiser produces a correctly shaped, entirely wrong answer.

The kernel below carries upstream's semantics (vocab range, head sharding,
zeros for padded heads, tail masking) on an access shape that a host mapping
tolerates. Upstream's own shape does not: its 2-D block load faults there, and
its ``cols // QUANT_BLOCK`` scale gather reads the wrong values while staying
correct on HBM -- so HBM-only verification cannot see it. Measured on A5 the
per-row shape is also about twice as fast on HBM, so there is one kernel rather
than one per address space.

Registration is not idempotent: re-registering the same range returns 507910.
A table whose ``.data`` is replaced must be unregistered before it is
registered again, which is the opposite failure direction from CUDA, where a
stale view is silently reused.

Dequantisation is exact by construction: ue8m0 is a bare exponent, so the
multiplier is a power of two and the fp32 product carries no rounding. The
exponent is turned into a float by writing the bits rather than by ``exp2``,
which keeps that exactness independent of any transcendental implementation.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

#: ``aclrtHostRegisterV2`` flag: map the host range into device address space.
ACL_HOST_REG_MAPPED = 0x2
#: One scale per this many values along a row. Engram's scale is 1-D -- one
#: exponent per block along the row -- unlike the 32x32 grid used for weights.
SCALE_BLOCK = 32


@triton.jit
def engram_lookup_hostptr(
    weight_addr,          # 1-element int64 tensor: device address of the fp8 table
    scales_addr,          # 1-element int64 tensor: device address of the ue8m0 scales
    ids,
    out,
    vocab_start,
    vocab_end,
    num_rows,
    ids_stride_t,
    ids_stride_h,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    GRID: tl.constexpr,
):
    """Gather fp8 rows, apply their ue8m0 block scales, write bf16.

    Only this rank's heads are read; padded heads write zeros for all-gather.
    `weight_addr`/`scales_addr` hold device addresses of host memory registered with
    aclrtHostRegisterV2; HBM addresses work unchanged.
    """
    n_scales: tl.constexpr = DIM // QUANT_BLOCK
    weight = tl.load(weight_addr).to(tl.pointer_type(tl.uint8))
    scales = tl.load(scales_addr).to(tl.pointer_type(tl.uint8))
    cols = tl.arange(0, DIM)
    scale_off = tl.arange(0, n_scales)
    for row in tl.range(tl.program_id(0), num_rows, GRID):
        head = HEAD_START + row % LOCAL_HEADS
        token = (row // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head * ids_stride_h,
            mask=head < TOTAL_HEADS,
            other=-1,
        ).to(tl.int64)
        owned = (head < TOTAL_HEADS) & (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        raw = tl.load(weight + local * DIM + cols, mask=owned, other=0)
        values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        # Contiguous, one scale per QUANT_BLOCK, then broadcast across the block. Indexing these
        # with cols // QUANT_BLOCK instead reads the same values from HBM and the wrong ones from a
        # host mapping.
        scale = tl.load(scales + local * n_scales + scale_off, mask=owned, other=0)
        # ue8m0 is a power of two, so its byte *is* the fp32 exponent field.
        scale = (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        scaled = tl.reshape(values, (n_scales, QUANT_BLOCK)) * scale[:, None]
        tl.store(out + row * DIM + cols, tl.reshape(scaled, (DIM,)).to(tl.bfloat16))


@dataclass
class HostTable:
    """A host mapping registered with CANN, addressable from the device."""

    host_addr: int
    device_addr: int
    num_bytes: int

    def __post_init__(self):
        if not self.device_addr:
            raise RuntimeError("registration produced a null device address")


def _acl():
    lib = ctypes.CDLL("libascendcl.so")
    lib.aclrtHostRegisterV2.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                                        ctypes.c_uint32]
    lib.aclrtHostRegisterV2.restype = ctypes.c_int
    # Three parameters. The trailing flag is not optional: calling it with two
    # leaves whatever is in the register as the flag and it returns
    # ACL_ERROR_INVALID_PARAM, which reads as "this memory cannot be mapped".
    lib.aclrtHostGetDevicePointer.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_void_p),
                                              ctypes.c_uint32]
    lib.aclrtHostGetDevicePointer.restype = ctypes.c_int
    lib.aclrtHostUnregister.argtypes = [ctypes.c_void_p]
    lib.aclrtHostUnregister.restype = ctypes.c_int
    return lib


def register_host_table(host_addr: int, num_bytes: int) -> HostTable:
    """Map an already-locked host range into the device address space."""
    lib = _acl()
    rc = lib.aclrtHostRegisterV2(ctypes.c_void_p(host_addr), num_bytes,
                                 ACL_HOST_REG_MAPPED)
    if rc != 0:
        raise RuntimeError(f"aclrtHostRegisterV2 failed: rc={rc}")
    dev = ctypes.c_void_p()
    rc = lib.aclrtHostGetDevicePointer(ctypes.c_void_p(host_addr),
                                       ctypes.byref(dev), 0)
    if rc != 0:
        lib.aclrtHostUnregister(ctypes.c_void_p(host_addr))
        raise RuntimeError(f"aclrtHostGetDevicePointer failed: rc={rc}")
    return HostTable(host_addr, dev.value or 0, num_bytes)


def unregister_host_table(table: HostTable) -> None:
    rc = _acl().aclrtHostUnregister(ctypes.c_void_p(table.host_addr))
    if rc != 0:
        raise RuntimeError(f"aclrtHostUnregister failed: rc={rc}")


def gather(table: HostTable,
           scale_table: HostTable,
           hash_ids: torch.Tensor,
           dim: int,
           num_rows: int,
           block: int = SCALE_BLOCK,
           grid: int | None = None) -> torch.Tensor:
    """Look up ``hash_ids`` and dequantise, returning ``[*shape, dim]`` bf16.

    ``hash_ids`` must already be bounds-checked; see :func:`check_in_range`.
    ``grid`` defaults to the device's parallel-unit count. Upstream halves it
    when gathering in the background, which costs about 2x here: a vector core
    handed back is not one a cube core can use.
    """
    flat = hash_ids.reshape(-1).to(torch.int64)
    if flat.device.type != "npu":
        flat = flat.npu()
    rows = flat.numel()
    out = torch.empty((rows, dim), dtype=torch.bfloat16, device="npu")

    # The address travels as data inside an ordinary device tensor. Passing it
    # as a pointer argument is rejected by Triton's launcher, which only admits
    # DEVICE and HOST_NUMA locations -- a registered host mapping reports HOST.
    addr = torch.tensor([table.device_addr], dtype=torch.int64, device="npu")
    s_addr = torch.tensor([scale_table.device_addr], dtype=torch.int64,
                          device="npu")
    ids_2d = flat.view(rows, 1)
    if grid is None:
        grid = torch.npu.get_device_properties(0).multi_processor_count

    engram_lookup_hostptr[(grid, )](
        addr, s_addr, ids_2d, out, 0, num_rows, rows,
        ids_2d.stride(0), ids_2d.stride(1),
        HEAD_START=0, LOCAL_HEADS=1, TOTAL_HEADS=1,
        DIM=dim, QUANT_BLOCK=block, GRID=grid)
    return out.view(*hash_ids.shape, dim)


def check_in_range(hash_ids: torch.Tensor, num_rows: int) -> None:
    """Refuse out-of-range ids rather than letting the kernel read past the table.

    Nothing downstream catches this: the gather reads whatever bytes follow the
    mapping -- another table's rows, or a fault -- and both look like plausible
    embeddings. The reference implementation's own Python path has no range
    check either, so this is the only place it can happen.
    """
    if hash_ids.numel() == 0:
        return
    lo = int(hash_ids.min())
    hi = int(hash_ids.max())
    if lo < 0 or hi >= num_rows:
        raise IndexError(
            f"engram hash id out of range: [{lo}, {hi}] outside [0, {num_rows}). "
            "A stale compressed-token map or a vocab-size mismatch rehashes the "
            "whole table, and every id it produces looks reasonable.")


class AscendParallelEngramEmbedding:
    """Ascend override surface for upstream's ``ParallelEngramEmbedding``.

    Upstream's ``lookup`` reaches for two things this platform supplies
    differently, and nothing else, so the mixin replaces exactly those:

    ``_storage`` -- upstream hands the kernel tensors that alias host memory
        through UVA. There is no equivalent here: a tensor built over a
        registered host mapping reports its location as HOST, and Triton's
        launcher admits only DEVICE and HOST_NUMA. The addresses travel as
        data in one-element int64 tensors instead.

    ``lookup`` -- the kernel differs (see this module's docstring) and so does
        the grid. Upstream halves it when gathering in the background to leave
        SMs for the main stream; on A5 that costs about 2x, because the cores
        being handed back are vector cores and the main stream's work is on the
        cube cores. Whether the overlap it buys is worth that has to be
        measured end to end; until then the grid is not halved.

    Mixed in ahead of upstream's class so these win the MRO:

        class AscendEngramEmbedding(AscendParallelEngramEmbedding,
                                    ParallelEngramEmbedding):
            pass
    """

    _host_table: HostTable | None = None
    _host_scale_table: HostTable | None = None

    def bind_host_tables(self, table: HostTable, scale_table: HostTable) -> None:
        """Adopt registered mappings for the weight and scale tables.

        Re-binding unregisters first: registration is not idempotent here and a
        repeat returns 507910. Upstream's equivalent guard exists for the
        opposite hazard -- on CUDA a stale view is reused silently.
        """
        self.release_host_tables()
        self._host_table = table
        self._host_scale_table = scale_table

    def release_host_tables(self) -> None:
        for attr in ("_host_table", "_host_scale_table"):
            existing = getattr(self, attr, None)
            if existing is not None:
                unregister_host_table(existing)
                setattr(self, attr, None)

    def _storage(self):
        if self._host_table is None or self._host_scale_table is None:
            raise RuntimeError(
                "engram host tables are not bound; call bind_host_tables() "
                "after registering the mappings. Upstream reaches for a UVA "
                "view here, which this platform has no equivalent for.")
        dev = torch.npu.current_device()
        return (torch.tensor([self._host_table.device_addr], dtype=torch.int64,
                             device=f"npu:{dev}"),
                torch.tensor([self._host_scale_table.device_addr],
                             dtype=torch.int64, device=f"npu:{dev}"))

    def lookup(self, indices: torch.Tensor, out: torch.Tensor,
               background: bool = False) -> None:
        rows = indices.shape[0] * self.part_n_hash_cols
        if not rows:
            return
        check_in_range(indices, self.num_embeddings)
        weight_addr, scale_addr = self._storage()
        grid = min(rows, self._num_sms)
        engram_lookup_hostptr[(grid, )](
            weight_addr, scale_addr, indices, out,
            self.vocab_start_idx, self.vocab_end_idx, rows,
            indices.stride(0), indices.stride(1),
            HEAD_START=self.head_start,
            LOCAL_HEADS=self.part_n_hash_cols,
            TOTAL_HEADS=self.n_hash_cols,
            DIM=self.dim,
            QUANT_BLOCK=self.block_size,
            GRID=grid)
