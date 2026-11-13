# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RAM-only Engram tables; copy just the requested raw rows to device staging.

This follows the host-row staging mechanism in 59ce1e18, while the existing
mainline device kernel owns FP8/E8M0 decoding. Full tables use ordinary CPU
memory, never CANN host registration or device allocations.
"""

import torch

RAM_CHUNK_ROWS = 1 << 22


class RamEngramBuffer:
    """Ordinary CPU chunks behind a shape-only checkpoint Parameter."""

    def __init__(self, shape, dtype, *, chunk_rows=RAM_CHUNK_ROWS):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        if len(self.shape) != 2 or any(size <= 0 for size in self.shape) or chunk_rows <= 0:
            raise ValueError("RAM Engram needs a nonempty row table and positive chunk_rows")
        self.chunk_rows = chunk_rows
        self.row_bytes = self.shape[1] * torch.empty((), dtype=dtype, device="cpu").element_size()
        self.tensor = None
        self.chunks: list[torch.Tensor] = []
        self._loaded_ranges: list[tuple[int, int]] = []
        self._closing = False
        try:
            for start in range(0, self.shape[0], chunk_rows):
                self.chunks.append(
                    torch.empty((min(chunk_rows, self.shape[0] - start), self.shape[1]), dtype=dtype, device="cpu")
                )
            self.tensor = torch.empty(self.shape, dtype=dtype, device="meta")
        except Exception:
            self.close()
            raise

    @property
    def closed(self):
        return self._closing

    def _require_open(self):
        if self._closing:
            raise RuntimeError("RAM Engram storage is closed")

    def cpu_tensors(self):
        return tuple(self.chunks)

    @property
    def loaded(self):
        return not self._closing and self._loaded_ranges == [(0, self.shape[0])]

    def begin_load(self):
        self._require_open()
        self._loaded_ranges = []

    def fill_(self, value):
        self.begin_load()
        for tensor in self.chunks:
            tensor.fill_(value)
        return self

    def zero_(self):
        return self.fill_(0)

    def copy_rows(self, start: int, source: torch.Tensor):
        """Write exact logical CPU rows, including slices crossing chunk edges."""
        self._require_open()
        if (
            not isinstance(start, int)
            or source.ndim != 2
            or source.device.type != "cpu"
            or source.dtype != self.dtype
            or source.shape[1] != self.shape[1]
            or start < 0
            or start + source.shape[0] > self.shape[0]
        ):
            raise ValueError("RAM Engram copy needs in-range CPU rows with the exact shape and dtype")
        stop = start + source.shape[0]
        if stop == start:
            return
        retained = []
        for lo, hi in self._loaded_ranges:
            if hi <= start or lo >= stop:
                retained.append((lo, hi))
            else:
                if lo < start:
                    retained.append((lo, start))
                if hi > stop:
                    retained.append((stop, hi))
        self._loaded_ranges = retained
        offset = start
        while offset < stop:
            index, local = divmod(offset, self.chunk_rows)
            count = min(stop - offset, self.chunks[index].shape[0] - local)
            target = self.chunks[index][local : local + count]
            values = source[offset - start : offset - start + count]
            if self.dtype == torch.float8_e4m3fn:
                target.view(torch.uint8).copy_(values.view(torch.uint8))
            else:
                target.copy_(values)
            offset += count
        merged = []
        for lo, hi in sorted([*self._loaded_ranges, (start, stop)]):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
            else:
                merged.append((lo, hi))
        self._loaded_ranges = merged

    def gather_rows(self, row_ids: torch.Tensor, output: torch.Tensor):
        """Gather raw bytes, preserving duplicate IDs and their original order."""
        self._require_open()
        raw_dtype = torch.uint8 if self.dtype == torch.float8_e4m3fn else self.dtype
        if (
            row_ids.device.type != "cpu"
            or row_ids.dtype != torch.int64
            or row_ids.ndim != 1
            or output.device.type != "cpu"
            or output.dtype != raw_dtype
            or output.shape != (row_ids.numel(), self.shape[1])
        ):
            raise ValueError("RAM Engram gather needs flat CPU int64 IDs and matching raw output")
        if row_ids.numel() == 0:
            return
        if bool(((row_ids < 0) | (row_ids >= self.shape[0])).any()):
            raise ValueError("RAM Engram gather row is out of range")
        chunk_ids = row_ids // self.chunk_rows
        for index in chunk_ids.unique().tolist():
            positions = torch.nonzero(chunk_ids == index, as_tuple=False).flatten()
            local = row_ids[positions] - index * self.chunk_rows
            source = self.chunks[index]
            if self.dtype == torch.float8_e4m3fn:
                source = source.view(torch.uint8)
            output.index_copy_(0, positions, torch.index_select(source, 0, local))

    def close(self):
        self._closing = True
        self._loaded_ranges = []
        self.chunks.clear()
        self.tensor = None


class RamEngramStaging:
    """Bounded raw-row buffers, used synchronously on one eager device stream."""

    def __init__(self, max_tokens, heads, width, dtype, scale_dtype, device):
        if max_tokens <= 0 or heads <= 0 or width <= 0 or width % 32:
            raise ValueError("RAM Engram staging needs positive capacity/heads and group-32 width")
        expected_scale = (
            None if dtype == torch.bfloat16 else torch.uint8 if dtype == torch.float8_e4m3fn else torch.float32
        )
        if dtype not in (torch.bfloat16, torch.int8, torch.float8_e4m3fn) or scale_dtype != expected_scale:
            raise ValueError("RAM Engram staging needs BF16, INT8/FP32 or native FP8/E8M0 storage")
        self.max_tokens = max_tokens
        self.heads = heads
        self.width = width
        self.dtype = dtype
        self.scale_dtype = scale_dtype
        self.device = torch.device(device)
        self.capacity_rows = max_tokens * heads
        self._closed = False
        self._busy = False
        self._stream = None
        self._assert_eager(bind_stream=False)
        raw_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else dtype
        self.host_weight = torch.empty((self.capacity_rows, width), dtype=raw_dtype, device="cpu")
        self.host_scale = (
            torch.empty((self.capacity_rows, width // 32), dtype=scale_dtype, device="cpu")
            if scale_dtype is not None
            else None
        )
        self.device_weight = torch.empty_like(self.host_weight, device=self.device)
        self.device_scale = (
            torch.empty_like(self.host_scale, device=self.device) if self.host_scale is not None else None
        )
        # Finish initialization before a first lookup on a different stream.
        self.stage_ids = (
            torch.arange(self.capacity_rows, dtype=torch.int64, device="cpu")
            .reshape(max_tokens, heads)
            .to(self.device, non_blocking=False)
        )

    def device_tensors(self):
        return tuple(t for t in (self.device_weight, self.device_scale, self.stage_ids) if t is not None)

    def _assert_eager(self, *, bind_stream=True):
        if self.device.type == "npu":
            capturing = torch.npu.is_current_stream_capturing()
            if not isinstance(capturing, bool) or capturing:
                raise RuntimeError("RAM Engram staging requires an uncaptured eager stream")
            if not bind_stream:
                return
            stream = torch.npu.current_stream()
            if self._stream is None:
                self._stream = stream
            elif self._stream != stream:
                # Only our previous stream can still be reading these buffers.
                # A failed wait keeps that owner and prevents any overwrite.
                self._stream.synchronize()
                self._stream = stream

    def lookup(self, codes, scales, indices, output, *, head_start=0, vocab_start=0, vocab_end=None):
        if self._closed or self._busy:
            raise RuntimeError("RAM Engram staging is closed or already in use")
        self._busy = True
        try:
            return self._lookup(
                codes, scales, indices, output, head_start=head_start, vocab_start=vocab_start, vocab_end=vocab_end
            )
        finally:
            self._busy = False

    def _lookup(self, codes, scales, indices, output, *, head_start, vocab_start, vocab_end):
        self._assert_eager()
        tokens = indices.shape[0] if indices.ndim == 2 else 0
        rows = tokens * self.heads
        vocab_end = vocab_start + codes.shape[0] if vocab_end is None else vocab_end
        pad_heads = output.numel() // (tokens * self.width) if tokens else self.heads
        if (
            indices.ndim != 2
            or indices.dtype not in (torch.int32, torch.int64)
            or indices.device != self.device
            or head_start < 0
            or indices.shape[1] < head_start + self.heads
            or tokens > self.max_tokens
            or codes.dtype != self.dtype
            or codes.shape[1] != self.width
            or vocab_end - vocab_start != codes.shape[0]
            or output.dtype != torch.bfloat16
            or output.device != self.device
            or pad_heads < self.heads
            or output.numel() != tokens * pad_heads * self.width
            or output.shape[-1] != self.width
            or not output.is_contiguous()
        ):
            raise ValueError("RAM Engram staging metadata, row capacity or output mismatch")
        if (scales is None) != (self.scale_dtype is None) or (
            scales is not None
            and (scales.dtype != self.scale_dtype or scales.shape != (codes.shape[0], self.width // 32))
        ):
            raise ValueError("RAM Engram codes/scales do not describe the same logical rows")
        if rows == 0:
            return output
        if not codes.loaded or (scales is not None and not scales.loaded):
            raise RuntimeError("RAM Engram lookup requires all checkpoint rows loaded")
        # This synchronizes the IDs; full tables never leave ordinary RAM.
        selected = (
            indices[:, head_start : head_start + self.heads]
            .detach()
            .to(device="cpu", dtype=torch.int64, non_blocking=False)
        )
        owned = (selected >= vocab_start) & (selected < vocab_end)
        local = torch.where(owned, selected - vocab_start, 0).reshape(-1)
        codes.gather_rows(local, self.host_weight[:rows])
        invalid = ~owned.reshape(-1)
        self.host_weight[:rows][invalid] = 0
        if scales is not None:
            scales.gather_rows(local, self.host_scale[:rows])
            # Invalid FP8 rows use zero codes and a finite unit scale.
            self.host_scale[:rows][invalid] = 127 if self.scale_dtype == torch.uint8 else 0
        self.device_weight[:rows].copy_(self.host_weight[:rows], non_blocking=False)
        if scales is not None:
            self.device_scale[:rows].copy_(self.host_scale[:rows], non_blocking=False)
        from .npu import gather_dequantize_engram_int8

        weight = self.device_weight[:rows].view(self.dtype)
        if pad_heads > self.heads:
            output.view(tokens, pad_heads, self.width)[:, self.heads :].zero_()
        gather_dequantize_engram_int8(
            weight,
            self.device_scale[:rows] if scales is not None else None,
            self.stage_ids[:tokens],
            self.width,
            local_heads=self.heads,
            pad_heads=pad_heads,
            output=output.view(-1, self.width),
            vocab_start=0,
            vocab_end=rows,
        )
        return output

    def close(self):
        if self._closed:
            return
        if self._busy:
            raise RuntimeError("RAM Engram staging is in use")
        if self.device.type == "npu" and self._stream is not None:
            self._stream.synchronize()
        self._closed = True
        self.host_weight = self.host_scale = None
        self.device_weight = self.device_scale = self.stage_ids = None
