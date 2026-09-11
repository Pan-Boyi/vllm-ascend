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
#: Exponent bias of ue8m0, and the fp32 exponent field offset. Both are
#: tl.constexpr because a @triton.jit body cannot read a plain module global.
E8M0_BIAS = 127
FP32_EXPONENT_SHIFT = tl.constexpr(23)
#: One scale per this many values along a row.
SCALE_BLOCK = 32


@triton.jit
def _gather_engram_rows(
    table_addr_ptr,      # int64 scalar tensor: device address of the fp8 table
    scale_addr_ptr,      # int64 scalar tensor: device address of the e8m0 table
    hash_ids_ptr,        # [n] int64
    out_ptr,             # [n, DIM] bf16
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.load(hash_ids_ptr + pid).to(tl.int64)

    table = tl.load(table_addr_ptr).to(tl.pointer_type(tl.uint8))
    scales = tl.load(scale_addr_ptr).to(tl.pointer_type(tl.uint8))

    off = tl.arange(0, DIM)
    raw = tl.load(table + row * DIM + off)
    values = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    # ue8m0 byte -> 2**(byte - 127), written as an fp32 bit pattern so the
    # multiplier is exactly a power of two with no transcendental involved.
    n_scales: tl.constexpr = DIM // BLOCK
    s_off = tl.arange(0, n_scales)
    s_raw = tl.load(scales + row * n_scales + s_off).to(tl.uint32)
    s_val = (s_raw << FP32_EXPONENT_SHIFT).to(tl.float32, bitcast=True)

    # broadcast each scale across its BLOCK values
    scaled = tl.reshape(values, (n_scales, BLOCK)) * s_val[:, None]
    tl.store(out_ptr + pid * DIM + off,
             tl.reshape(scaled, (DIM, )).to(tl.bfloat16))


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
           block: int = SCALE_BLOCK) -> torch.Tensor:
    """Look up ``hash_ids`` and dequantise, returning ``[*shape, dim]`` bf16.

    ``hash_ids`` must already be bounds-checked; see :func:`check_in_range`.
    """
    flat = hash_ids.reshape(-1).to(torch.int64)
    if flat.device.type != "npu":
        flat = flat.npu()
    out = torch.empty((flat.numel(), dim), dtype=torch.bfloat16, device="npu")

    addr = torch.tensor([table.device_addr], dtype=torch.int64, device="npu")
    s_addr = torch.tensor([scale_table.device_addr], dtype=torch.int64,
                          device="npu")
    _gather_engram_rows[(flat.numel(), )](addr, s_addr, flat, out,
                                          DIM=dim, BLOCK=block)
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
