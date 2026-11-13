# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real CPU storage with a fake CANN ABI and independent device aliases."""

import ctypes
import errno
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import npu


class HostLibrary:
    def __init__(self, payload_bytes):
        self.storage = ctypes.create_string_buffer(payload_bytes + 2 * 65536)
        self.original = ctypes.addressof(self.storage) + 17
        self.payload = (self.original + 65535) // 65536 * 65536
        self.register_calls = []
        self.pointer_calls = []
        self.unregister_calls = []
        self.free_calls = []
        self.registered = {}
        self.aliases = {}
        self.fail_register_at = None
        self.fail_pointer_at = None
        self.null_pointer_at = None
        self.fail_unregister_once = set()
        self.raise_unregister_once = set()
        self.fail_free_once = False
        self.events = []

    def aclrtMallocHost(self, output, size, flags):
        self.allocation_size = size
        assert flags == 0
        output._obj.value = self.original
        return 0

    def aclrtHostRegisterV2(self, pointer, size, flags):
        index = len(self.register_calls)
        self.register_calls.append((pointer.value, size, flags))
        assert self.original <= pointer.value < pointer.value + size <= self.original + self.allocation_size
        if index == self.fail_register_at:
            return 7
        self.registered[pointer.value] = size
        return 0

    def aclrtHostGetDevicePointer(self, pointer, output, flags):
        index = len(self.pointer_calls)
        self.pointer_calls.append(pointer.value)
        assert pointer.value in self.registered and flags == 0
        # Adjacent host chunks intentionally map to unrelated device addresses.
        alias = [0x900000000, 0x400000000, 0xB00000000][index]
        self.aliases[alias] = pointer.value
        output._obj.value = None if index == self.null_pointer_at else alias
        return 7 if index == self.fail_pointer_at else 0

    def aclrtHostUnregister(self, pointer):
        self.events.append("unregister")
        self.unregister_calls.append(pointer.value)
        assert pointer.value in self.registered, "successful chunks must never be unregistered twice"
        if pointer.value in self.raise_unregister_once:
            self.raise_unregister_once.remove(pointer.value)
            raise OSError(errno.EBUSY, "synthetic CANN unregister failure")
        if pointer.value in self.fail_unregister_once:
            self.fail_unregister_once.remove(pointer.value)
            return 7
        del self.registered[pointer.value]
        return 0

    def aclrtFreeHost(self, pointer):
        self.events.append("free")
        self.free_calls.append(pointer.value)
        assert pointer.value == self.original and not self.registered
        if self.fail_free_once:
            self.fail_free_once = False
            return 7
        return 0


@pytest.fixture
def host_chunks(monkeypatch):
    def create(rows=10, width=7, dtype=torch.uint8):
        row_bytes = width * torch.empty((), dtype=dtype).element_size()
        lib = HostLibrary(rows * row_bytes)
        monkeypatch.setattr(npu, "CHUNK_ROWS", 4)
        monkeypatch.setattr(npu, "_host_library", lambda: lib)
        monkeypatch.setattr(npu.DeviceOperator, "host_register_flags", lambda: 2)
        return lib, (rows, width), dtype

    return create


@pytest.mark.parametrize("rows", [1, 3, 4, 5, 8, 9, 10])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_chunks_publish_independent_aliases_and_exact_nonpage_tail(host_chunks, rows, dtype):
    lib, shape, dtype = host_chunks(rows=rows, dtype=dtype)
    buffer = npu.HostUvaBuffer(shape, dtype, "cpu")
    row_bytes = 7 * torch.empty((), dtype=dtype).element_size()
    assert buffer.tensor.device.type == "cpu" and buffer.tensor.data_ptr() == lib.payload
    assert lib.allocation_size == rows * row_bytes + 65535
    assert lib.register_calls == [
        (lib.payload + start * row_bytes, min(4, rows - start) * row_bytes, 2) for start in range(0, rows, 4)
    ]
    assert buffer.ptrs.tolist() == list(lib.aliases)
    buffer.tensor.copy_(torch.arange(rows * 7).reshape(shape).to(dtype))
    # Resolve boundary rows through each alias independently, as the unchanged
    # gather kernel does. This reads the actual ctypes-backed CPU tensor bytes.
    for row in sorted({0, rows - 1, *[r for r in (3, 4, 7, 8) if r < rows]}):
        chunk, offset = divmod(row, 4)
        host = lib.aliases[buffer.ptrs[chunk].item()] + offset * row_bytes
        actual = (ctypes.c_char * row_bytes).from_address(host)
        torch.testing.assert_close(torch.frombuffer(actual, dtype=dtype), buffer.tensor[row], rtol=0, atol=0)
    buffer.close()
    buffer.close()
    assert len(lib.unregister_calls) == len(lib.register_calls)
    assert lib.free_calls == [lib.original]


def test_production_chunk_geometry_uses_one_gib_codes_and_32_mib_scales():
    assert npu.CHUNK_ROWS == 1 << 22
    assert npu.CHUNK_ROWS * 256 * torch.empty((), dtype=torch.float8_e4m3fn).element_size() == 1 << 30
    assert npu.CHUNK_ROWS * 8 * torch.empty((), dtype=torch.float8_e8m0fnu).element_size() == 32 << 20
    assert (npu.CHUNK_ROWS * 8) % 65536 == 0


@pytest.mark.parametrize("failure", ["register", "device_pointer", "null_device_pointer", "publication"])
def test_partial_initialization_rolls_back_only_owned_registrations(monkeypatch, host_chunks, failure):
    lib, shape, dtype = host_chunks()
    if failure == "register":
        lib.fail_register_at = 1
    elif failure == "device_pointer":
        lib.fail_pointer_at = 1
    elif failure == "null_device_pointer":
        lib.null_pointer_at = 1
    else:
        monkeypatch.setattr(npu.torch, "tensor", Mock(side_effect=ValueError("pointer table publication failed")))
    with pytest.raises((RuntimeError, ValueError)):
        npu.HostUvaBuffer(shape, dtype, "cpu")
    owned_count = 1 if failure == "register" else 2 if failure != "publication" else 3
    assert lib.unregister_calls == [call[0] for call in reversed(lib.register_calls[:owned_count])]
    assert not lib.registered and lib.free_calls == [lib.original]


def test_cpu_tensor_creation_failure_releases_the_single_allocation(monkeypatch, host_chunks):
    lib, shape, dtype = host_chunks()
    monkeypatch.setattr(npu.torch, "frombuffer", Mock(side_effect=ValueError("CPU view creation failed")))
    with pytest.raises(ValueError, match="CPU view creation failed"):
        npu.HostUvaBuffer(shape, dtype, "cpu")
    assert lib.register_calls == lib.unregister_calls == []
    assert lib.free_calls == [lib.original]


@pytest.mark.parametrize("failure", ["rc", "exception"])
def test_partial_close_keeps_allocation_and_retries_only_remaining_chunks(monkeypatch, host_chunks, failure):
    lib, shape, dtype = host_chunks()
    buffer = npu.HostUvaBuffer(shape, dtype, "cpu")
    # Only the device publication/synchronization boundary is mocked. The host
    # payload above remains an actual CPU tensor backed by the ctypes allocation.
    buffer.ptrs = SimpleNamespace(device=SimpleNamespace(type="npu"))
    synchronize = Mock(side_effect=lambda: lib.events.append("synchronize"))
    monkeypatch.setattr(npu.torch.npu, "synchronize", synchronize)
    remaining = lib.register_calls[1][0]
    getattr(lib, "fail_unregister_once" if failure == "rc" else "raise_unregister_once").add(remaining)
    with pytest.raises(RuntimeError if failure == "rc" else OSError) as caught:
        buffer.close()
    if failure == "exception":
        assert caught.value.errno == errno.EBUSY
    assert lib.events[0] == "synchronize" and synchronize.call_count == 1
    assert list(lib.registered) == [remaining] and lib.free_calls == []
    assert buffer.tensor is not None and buffer.ptrs is None
    with pytest.raises(RuntimeError, match="closing or closed"):
        npu.gather_dequantize_host_uva(buffer, None, torch.tensor([[0]]))
    before = lib.unregister_calls[:]
    buffer.close()
    buffer.close()
    assert lib.unregister_calls == before + [remaining]
    assert lib.free_calls == [lib.original] and synchronize.call_count == 1


def test_failed_synchronize_cannot_unregister_or_free(monkeypatch, host_chunks):
    lib, shape, dtype = host_chunks()
    buffer = npu.HostUvaBuffer(shape, dtype, "cpu")
    buffer.ptrs = SimpleNamespace(device=SimpleNamespace(type="npu"))
    synchronize = Mock(side_effect=[RuntimeError("work still in flight"), None])
    monkeypatch.setattr(npu.torch.npu, "synchronize", synchronize)
    with pytest.raises(RuntimeError, match="work still in flight"):
        buffer.close()
    assert lib.unregister_calls == lib.free_calls == [] and buffer.ptrs is not None
    with pytest.raises(RuntimeError, match="closing or closed"):
        npu.gather_dequantize_host_uva(buffer, None, torch.tensor([[0]]))
    buffer.close()
    buffer.close()
    assert synchronize.call_count == 2 and lib.free_calls == [lib.original]


def test_free_failure_retry_never_unregisters_chunks_twice(host_chunks):
    lib, shape, dtype = host_chunks()
    buffer = npu.HostUvaBuffer(shape, dtype, "cpu")
    lib.fail_free_once = True
    with pytest.raises(RuntimeError, match="aclrtFreeHost failed"):
        buffer.close()
    before = lib.unregister_calls[:]
    assert not lib.registered and buffer.tensor is not None
    buffer.close()
    assert lib.unregister_calls == before and lib.free_calls == [lib.original, lib.original]


@pytest.mark.parametrize("failure", ["unregister", "free"])
def test_failed_constructor_rollback_retains_owner_for_explicit_retry(host_chunks, failure):
    lib, shape, dtype = host_chunks()
    lib.fail_pointer_at = 1
    if failure == "unregister":
        lib.fail_unregister_once.add(lib.payload)
    else:
        lib.fail_free_once = True
    with pytest.raises(RuntimeError, match="aclrtHostGetDevicePointer failed") as caught:
        npu.HostUvaBuffer(shape, dtype, "cpu")
    buffer = caught.value.buffer
    assert buffer.tensor.device.type == "cpu" and buffer._allocation_pointer.value == lib.original
    assert list(lib.registered) == ([lib.payload] if failure == "unregister" else [])
    assert lib.free_calls == ([] if failure == "unregister" else [lib.original])
    assert ("aclrtHostUnregister failed" if failure == "unregister" else "aclrtFreeHost failed") in str(
        caught.value.__cause__
    )
    before = lib.unregister_calls[:]
    buffer.close()
    assert lib.unregister_calls == before + ([lib.payload] if failure == "unregister" else [])
    assert lib.free_calls == ([lib.original] if failure == "unregister" else [lib.original, lib.original])
