# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only checkpoint-backed Engram rows; no full-table allocation or copy.

PyTorch safetensors uses a private CPU file mapping. Its tensor API has no
read-only dtype, so this backend never exposes a writing loader operation.
The original checkpoint must remain unchanged for the lifetime of the model.
Only requested rows are read into the existing bounded RAM staging buffers.
"""

import hashlib
import json
import os
import stat
from contextlib import suppress
from pathlib import Path

import torch
from safetensors import safe_open

from .host_staging import RAM_CHUNK_ROWS, RamEngramBuffer

MAX_SAFETENSORS_HEADER_BYTES = 100_000_000
_DTYPES = {
    torch.bfloat16: ("BF16",),
    torch.float8_e4m3fn: ("F8_E4M3", "F8_E4M3FN"),
    # This is a byte view of native E8M0, never a numerical uint8 conversion.
    torch.uint8: ("F8_E8M0",),
}


def _stat_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors header key: {key}")
        result[key] = value
    return result


class _CheckpointOwner:
    """Keep the readonly descriptor and safetensors mapping alive with views."""

    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.fd = None
        self.reader = None
        self.context = None
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
            info = os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Engram mmap checkpoint must be a regular file")
            self.identity = _stat_identity(info)
            self.validate()
        except Exception:
            self.close()
            raise

    def validate(self):
        if self.fd is None or (
            _stat_identity(os.fstat(self.fd)) != self.identity or _stat_identity(self.path.stat()) != self.identity
        ):
            raise RuntimeError("Engram mmap checkpoint identity changed")

    def close(self):
        context = self.context
        self.context = self.reader = None
        try:
            if context is not None:
                context.__exit__(None, None, None)
        finally:
            if self.fd is not None:
                os.close(self.fd)
                self.fd = None

    def __del__(self):
        with suppress(Exception):
            self.close()


class MmapEngramBuffer(RamEngramBuffer):
    """Shape-only Parameter plus immutable, file-backed CPU chunk views.

    The inherited gather only reads chunks. Writing methods are forbidden, even
    before loading: readiness means the exact indexed checkpoint was bound, not
    that an empty meta Parameter was written. Escaped chunk views retain their
    descriptor owner independently of buffer.close().
    """

    def __init__(self, shape, dtype, *, chunk_rows=RAM_CHUNK_ROWS):
        self.shape = torch.Size(shape)
        if dtype not in _DTYPES:
            raise ValueError("Engram mmap supports only native BF16, FP8 codes or E8M0 byte views")
        if len(self.shape) != 2 or any(size <= 0 for size in self.shape) or chunk_rows <= 0:
            raise ValueError("Engram mmap needs a nonempty row table and positive chunk_rows")
        self.dtype, self.chunk_rows = dtype, chunk_rows
        self.row_bytes = self.shape[1] * torch.empty((), dtype=dtype, device="meta").element_size()
        self.tensor = torch.empty(self.shape, dtype=dtype, device="meta")
        self.chunks = []
        self._loaded_ranges = []
        self._closing = False
        self._owner = self._view = None
        self.source_identity = None
        self._index_identity = None

    @property
    def loaded(self):
        if self._closing or self._owner is None or self._loaded_ranges != [(0, self.shape[0])]:
            return False
        try:
            self._validate_source()
        except (OSError, RuntimeError):
            return False
        return True

    def _validate_source(self):
        if self._owner is None:
            raise RuntimeError("Engram mmap checkpoint has not been loaded")
        self._owner.validate()
        if self._index_identity is not None:
            path, identity, _ = self._index_identity
            if _stat_identity(path.stat()) != identity:
                raise RuntimeError("Engram mmap checkpoint index identity changed")

    def begin_load(self):
        self._require_open()
        self._loaded_ranges = []
        self.chunks.clear()
        self._view = self._owner = None
        self.source_identity = self._index_identity = None

    def fill_(self, value):
        raise RuntimeError("Engram mmap checkpoint storage is read-only")

    def zero_(self):
        raise RuntimeError("Engram mmap checkpoint storage is read-only")

    def copy_rows(self, start, source):
        raise RuntimeError("Engram mmap checkpoint storage is read-only")

    @staticmethod
    def _check_file_mapping(tensor, owner, data_offset):
        """Reject a backend that silently materialized a CPU tensor copy."""
        first = tensor.data_ptr()
        end = first + tensor.numel() * tensor.element_size()
        cursor = first
        device, inode = owner.identity[:2]
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            lo, hi = (int(value, 16) for value in fields[0].split("-"))
            if hi <= cursor or lo > cursor:
                continue
            major, minor = (int(value, 16) for value in fields[3].split(":"))
            if (
                fields[1][0] != "r"
                or fields[1][-1] != "p"
                or int(fields[4]) != inode
                or (major, minor) != (os.major(device), os.minor(device))
                or int(fields[2], 16) + cursor - lo != data_offset + cursor - first
            ):
                raise RuntimeError("Engram tensor is not a private mapping of the expected checkpoint bytes")
            cursor = min(hi, end)
            if cursor == end:
                return
        raise RuntimeError("Engram tensor has no complete checkpoint file mapping")

    def bind_checkpoint(self, path, key, *, source_shape, row_start=0, index_reference=None):
        self.begin_load()
        owner = None
        try:
            owner = _CheckpointOwner(path)
            raw_length = os.pread(owner.fd, 8, 0)
            header_size = int.from_bytes(raw_length, "little")
            if len(raw_length) != 8 or not 0 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("Invalid Engram safetensors header size")
            if 8 + header_size > owner.identity[2]:
                raise ValueError("Engram safetensors header exceeds its file")
            raw_header = os.pread(owner.fd, header_size, 8)
            if len(raw_header) != header_size:
                raise ValueError("Truncated Engram safetensors header")
            header = json.loads(raw_header, object_pairs_hook=_unique_keys)
            info = header[key]
            offsets = info["data_offsets"]
            if (
                info["dtype"] not in _DTYPES[self.dtype]
                or tuple(info["shape"]) != tuple(source_shape)
                or len(source_shape) != 2
                or source_shape[1] != self.shape[1]
                or not isinstance(row_start, int)
                or row_start < 0
                or row_start + self.shape[0] > source_shape[0]
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] < offsets[1] <= owner.identity[2] - 8 - header_size
                or offsets[1] - offsets[0] != source_shape[0] * self.row_bytes
            ):
                raise ValueError("Engram mmap checkpoint dtype, shape or byte range mismatch")
            if index_reference is not None:
                index_path, index_sha = index_reference
                index_path = Path(index_path).resolve(strict=True)
                before = _stat_identity(index_path.stat())
                if hashlib.sha256(index_path.read_bytes()).hexdigest() != index_sha:
                    raise RuntimeError("Engram mmap checkpoint index bytes changed")
                if _stat_identity(index_path.stat()) != before:
                    raise RuntimeError("Engram mmap checkpoint index identity changed")
                self._index_identity = (index_path, before, index_sha)
            # Reopen the held inode, not the mutable checkpoint pathname.
            owner.context = safe_open(f"/proc/self/fd/{owner.fd}", framework="pt", device="cpu")
            owner.reader = owner.context.__enter__()
            view = owner.reader.get_tensor(key)
            if self.dtype == torch.uint8:
                if view.dtype != torch.float8_e8m0fnu:
                    raise ValueError("Engram mmap scale tensor must retain native E8M0 bits")
                view = view.view(torch.uint8)
            if view.device.type != "cpu" or view.dtype != self.dtype or not view.is_contiguous():
                raise ValueError("Engram mmap reader did not return the exact contiguous CPU dtype")
            if tuple(view.shape) != tuple(source_shape):
                raise ValueError("Engram mmap reader shape differs from its header")
            self._check_file_mapping(view, owner, 8 + header_size + offsets[0])
            view = view[row_start : row_start + self.shape[0]]
            view._engram_checkpoint_owner = owner
            chunks = [view[start : start + self.chunk_rows] for start in range(0, self.shape[0], self.chunk_rows)]
            for chunk in chunks:
                chunk._engram_checkpoint_owner = owner
            owner.validate()
            self._owner, self._view, self.chunks = owner, view, chunks
            self.source_identity = {
                "path": str(owner.path),
                "stat": list(owner.identity),
                "key": key,
                "source_shape": list(source_shape),
                "source_dtype": info["dtype"],
                "data_offsets": offsets,
                "header_sha256": hashlib.sha256(raw_header).hexdigest(),
                "row_start": row_start,
                "rows": self.shape[0],
                "open_mode": "O_RDONLY",
                "mapping": "PRIVATE_FILE_MAPPING_READ_ONLY_LOADER_API",
                "index": (
                    {
                        "path": str(self._index_identity[0]),
                        "stat": list(self._index_identity[1]),
                        "sha256": self._index_identity[2],
                    }
                    if self._index_identity is not None
                    else None
                ),
            }
            self._validate_source()
            self._loaded_ranges = [(0, self.shape[0])]
        except Exception:
            self.begin_load()
            if owner is not None:
                owner.close()
            raise

    def gather_rows(self, row_ids, output):
        self._require_open()
        if not self.loaded:
            raise RuntimeError("Engram mmap checkpoint is unloaded or its source identity changed")
        super().gather_rows(row_ids, output)
        self._validate_source()

    def close(self):
        if not self._closing:
            self.begin_load()
            self._closing = True
            self.tensor = None
