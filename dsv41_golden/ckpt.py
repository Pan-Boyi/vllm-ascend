"""mmap-backed reader for the HF DeepSeek-V4.1-Flash checkpoint.

Why not ``safetensors.safe_open``: it can only slice along leading dimensions, and
the engram tables need an arbitrary *row gather* out of a 384-million-row, 101 GB
tensor.  mmap gives that for the cost of one page fault per row, and it also lets
every other weight be read lazily and freed the moment a layer is done with it.

The checkpoint directory is opened read-only and never written.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Dict, List

import numpy as np
import torch

# safetensors dtype tag -> (torch dtype, bytes per element)
_DTYPES = {
    "F64": (torch.float64, 8),
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "F8_E4M3": (torch.float8_e4m3fn, 1),
    "F8_E5M2": (torch.float8_e5m2, 1),
    "F8_E8M0": (torch.float8_e8m0fnu, 1),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
    "BOOL": (torch.bool, 1),
}


class _Shard:
    """One .safetensors file, header parsed, body mmapped."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            header_len = int.from_bytes(f.read(8), "little")
            self.header = json.loads(f.read(header_len))
        self.data_start = 8 + header_len
        self._mm = None

    @property
    def mm(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self._mm

    def entry(self, name: str) -> dict:
        return self.header[name]

    def raw(self, name: str) -> np.ndarray:
        """The tensor's bytes as a 1-D read-only uint8 view into the mmap."""
        e = self.header[name]
        lo, hi = e["data_offsets"]
        return self.mm[self.data_start + lo: self.data_start + hi]

    def close(self):
        if self._mm is not None:
            del self._mm
            self._mm = None


class Checkpoint:
    """Name -> tensor over a sharded safetensors checkpoint.

    ``get`` returns an owned copy (safe to keep, free, or mutate).  ``gather_rows``
    pulls a scattered set of rows out of a tensor without materializing it.
    """

    def __init__(self, path: str):
        self.path = path
        index_path = os.path.join(path, "model.safetensors.index.json")
        with open(index_path) as f:
            self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        self._shards: Dict[str, _Shard] = {}

    # -- internals ---------------------------------------------------------
    def _shard(self, name: str) -> _Shard:
        fn = self.weight_map[name]
        if fn not in self._shards:
            self._shards[fn] = _Shard(os.path.join(self.path, fn))
        return self._shards[fn]

    # -- public ------------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def names(self) -> List[str]:
        return list(self.weight_map)

    def meta(self, name: str):
        e = self._shard(name).entry(name)
        return _DTYPES[e["dtype"]][0], tuple(e["shape"])

    def get(self, name: str) -> torch.Tensor:
        """Read a whole tensor into freshly allocated memory."""
        sh = self._shard(name)
        e = sh.entry(name)
        dtype, _ = _DTYPES[e["dtype"]]
        buf = sh.raw(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")             # read-only buffer
            t = torch.frombuffer(memoryview(buf), dtype=torch.uint8)
        return t.clone().view(dtype).view(*e["shape"])

    def nbytes(self, name: str) -> int:
        e = self._shard(name).entry(name)
        lo, hi = e["data_offsets"]
        return hi - lo

    def gather_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Rows of a 2-D tensor, read straight out of the mmap.

        ``rows`` is any int tensor; the result has shape ``(*rows.shape, ncols)``.
        Only the touched pages are faulted in, so this works on a tensor far
        larger than RAM -- which is the whole point for the engram tables
        (384 M rows, ~98 GB each).
        """
        sh = self._shard(name)
        e = sh.entry(name)
        dtype, esize = _DTYPES[e["dtype"]]
        shape = e["shape"]
        assert len(shape) == 2, f"gather_rows needs a 2-D tensor, {name} is {shape}"
        nrows, ncols = shape
        row_bytes = ncols * esize

        flat = rows.reshape(-1).to(torch.int64)
        assert int(flat.min()) >= 0 and int(flat.max()) < nrows, (
            f"{name}: row index out of range [0, {nrows})"
        )

        lo, _ = e["data_offsets"]
        base = sh.data_start + lo
        table = sh.mm[base: base + nrows * row_bytes].reshape(nrows, row_bytes)

        picked = np.ascontiguousarray(table[flat.numpy()])          # the actual page faults
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = torch.frombuffer(memoryview(picked), dtype=torch.uint8)
        return t.clone().view(dtype).view(*rows.shape, ncols)

    def close(self):
        for s in self._shards.values():
            s.close()
        self._shards.clear()


def dequant_fp8_2d(weight: torch.Tensor, scale: torch.Tensor, block: int = 32) -> torch.Tensor:
    """[N, K] e4m3 + [ceil(N/b), ceil(K/b)] e8m0 -> bf16, matching convert.py's wo_a path."""
    N, K = weight.shape
    s = scale.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return (weight.float() * s[:N, :K]).bfloat16()
