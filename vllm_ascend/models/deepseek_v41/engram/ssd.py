# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verified, read-only SSD tables behind the mainline row-staging interface."""

import mmap
import os
from pathlib import Path
from weakref import finalize

import numpy as np
import torch

from . import ssd_storage
from .host_staging import RamEngramBuffer


def ssd_directory(path):
    if not isinstance(path, str) or not path or not Path(path).is_absolute():
        raise ValueError("Engram SSD requires an explicit absolute engram_ssd_path")
    directory = Path(path).resolve(strict=True)
    ssd_storage.validate_filesystem(directory, "ssd")
    return directory


class SsdEngramBuffer(RamEngramBuffer):
    """Map verified native MXFP8/E8M0 bytes without allocating a CPU table.

    Only gather_rows exposes data, through the caller's bounded raw staging
    buffer. The read-only NumPy view is never wrapped in a writable Torch tensor.
    Linux page cache participates; this is not a hard page-cache budget.
    """

    def __init__(self, shape, dtype):
        self.shape = torch.Size(shape)
        if dtype not in (torch.float8_e4m3fn, torch.uint8) or len(self.shape) != 2 or any(n <= 0 for n in self.shape):
            raise ValueError("Engram SSD requires native MXFP8 codes or E8M0 byte scales and a nonempty row table")
        self.dtype = dtype
        self.tensor = torch.empty(self.shape, dtype=dtype, device="meta")
        self.chunks = []
        self._loaded_ranges = []
        self._closing = False
        self._mapping = self._array = self._fd = None
        self._fd_finalizer = None
        self._path = self._manifest = self._name = None
        self.source_identity = None

    @property
    def loaded(self):
        if self.closed or self._fd is None:
            return False
        try:
            self._validate_source()
        except (OSError, ValueError, RuntimeError):
            return False
        return True

    def _validate_source(self):
        if self._fd is None:
            raise RuntimeError("Engram SSD storage is unloaded")
        ssd_storage.validate_table(self._manifest, self._name, os.fstat(self._fd), *self.shape)
        ssd_storage.validate_table(self._manifest, self._name, self._path.stat(follow_symlinks=False), *self.shape)

    def begin_load(self):
        self._require_open()
        self._array = None
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._fd is not None:
            if self._fd_finalizer is not None:
                self._fd_finalizer()
                self._fd_finalizer = None
            else:
                os.close(self._fd)
            self._fd = None
        self._path = self._manifest = self._name = self.source_identity = None

    def bind_table(self, stage_dir, checkpoint_dir, name):
        self.begin_load()
        try:
            directory = ssd_directory(stage_dir)
            manifest = ssd_storage.load_manifest(directory, checkpoint_dir)
            entry = manifest["tables"][name]
            expected = "F8_E4M3" if self.dtype == torch.float8_e4m3fn else "F8_E8M0"
            if entry["dtype"] != expected:
                raise ValueError("Engram SSD storage dtype differs from the native checkpoint")
            self._path, self._manifest, self._name = directory / entry["file"], manifest, name
            self._fd = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            self._fd_finalizer = finalize(self, os.close, self._fd)
            self._validate_source()
            self._mapping = mmap.mmap(self._fd, entry["nbytes"], flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
            self._mapping.madvise(mmap.MADV_RANDOM)
            self._array = np.frombuffer(self._mapping, dtype=np.uint8).reshape(self.shape)
            self._validate_source()
            self.source_identity = {"path": str(self._path), "table": entry, "checkpoint": manifest["checkpoint"]}
        except Exception:
            self.begin_load()
            raise

    def fill_(self, value):
        raise RuntimeError("Engram SSD storage is read-only")

    def zero_(self):
        raise RuntimeError("Engram SSD storage is read-only")

    def copy_rows(self, start, source):
        raise RuntimeError("Engram SSD storage is read-only")

    def gather_rows(self, row_ids, output):
        self._require_open()
        if not self.loaded:
            raise RuntimeError("Engram SSD storage is unloaded or its file identity changed")
        if (
            row_ids.device.type != "cpu"
            or row_ids.dtype != torch.int64
            or row_ids.ndim != 1
            or output.device.type != "cpu"
            or output.dtype != torch.uint8
            or output.shape != (row_ids.numel(), self.shape[1])
            or not output.is_contiguous()
        ):
            raise ValueError("Engram SSD gather needs flat CPU int64 IDs and contiguous raw-byte output")
        # Validate before touching the file: NumPy's default negative indexing
        # must never turn an invalid Engram ID into a valid last row.
        if row_ids.numel() and bool(((row_ids < 0) | (row_ids >= self.shape[0])).any()):
            raise ValueError("Engram SSD gather row is out of range")
        # mode='clip' avoids NumPy's mandatory temporary output for mode='raise'.
        # All indices were explicitly checked above. Duplicate order is kept.
        np.take(self._array, row_ids.numpy(), axis=0, out=output.numpy(), mode="clip")
        self._validate_source()

    def close(self):
        if not self.closed:
            self.begin_load()
            self._closing = True
            self.tensor = None
