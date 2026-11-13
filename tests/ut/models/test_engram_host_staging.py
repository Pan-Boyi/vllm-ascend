# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real ordinary CPU RAM and raw row staging, without an NPU."""

import json
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram import npu
from vllm_ascend.models.deepseek_v41.engram.host_staging import RamEngramBuffer, RamEngramStaging


def source_rows(rows, width, dtype):
    bits = torch.arange(rows * width, device="cpu").reshape(rows, width).to(torch.uint8)
    if dtype == torch.float8_e4m3fn:
        return bits.view(dtype)
    if dtype == torch.int8:
        return bits.view(torch.int8)
    return bits.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8, torch.float8_e4m3fn, torch.uint8, torch.float32])
def test_real_cpu_storage_crosses_chunks_preserving_bits(monkeypatch, dtype):
    monkeypatch.setattr(npu, "_host_library", Mock(side_effect=AssertionError("RAM must not call CANN")))
    values = source_rows(11, 64, dtype)
    with torch.device("meta"):
        buffer = RamEngramBuffer((11, 64), dtype, chunk_rows=4)
    assert buffer.tensor.is_meta and buffer.tensor.shape == (11, 64)
    assert [t.shape[0] for t in buffer.cpu_tensors()] == [4, 4, 3]
    assert all(t.device.type == "cpu" and not t.is_meta and not t.is_pinned() for t in buffer.cpu_tensors())
    buffer.zero_()
    assert not buffer.loaded
    buffer.copy_rows(0, values[:7])
    assert not buffer.loaded
    buffer.copy_rows(7, values[7:])
    assert buffer.loaded
    ids = torch.tensor([10, 0, 3, 4, 7, 8, 4], dtype=torch.int64)
    raw_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else dtype
    actual = torch.empty((len(ids), 64), dtype=raw_dtype)
    buffer.gather_rows(ids, actual)
    expected = values.view(torch.uint8) if dtype == torch.float8_e4m3fn else values
    torch.testing.assert_close(actual, expected[ids], rtol=0, atol=0)
    buffer.copy_rows(5, values[:0])
    assert buffer.loaded
    buffer.close()
    buffer.close()
    assert buffer.closed and not buffer.loaded and not buffer.cpu_tensors() and buffer.tensor is None
    with pytest.raises(RuntimeError, match="closed"):
        buffer.copy_rows(0, values[:1])
    npu._host_library.assert_not_called()


def test_failed_overwrite_invalidates_coverage(monkeypatch):
    buffer = RamEngramBuffer((11, 64), torch.uint8, chunk_rows=4)
    values = source_rows(11, 64, torch.uint8)
    buffer.copy_rows(0, values)
    assert buffer.loaded
    original = torch.Tensor.copy_
    calls = []

    def fail_second(target, source, *args, **kwargs):
        calls.append(target)
        if len(calls) == 2:
            raise OSError("copy failed")
        return original(target, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", fail_second)
    with pytest.raises(OSError, match="copy failed"):
        buffer.copy_rows(0, values)
    assert not buffer.loaded
    buffer.close()


def test_partial_cpu_allocation_is_released(monkeypatch):
    original = torch.empty
    allocated = []

    def fail_second(*args, **kwargs):
        if kwargs.get("device") == "cpu" and args and tuple(args[0]) == (4, 64):
            if allocated:
                raise MemoryError("RAM allocation failed")
            tensor = original(*args, **kwargs)
            allocated.append(weakref.ref(tensor))
            return tensor
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", fail_second)
    with pytest.raises(MemoryError, match="RAM allocation failed"):
        RamEngramBuffer((11, 64), torch.uint8, chunk_rows=4)
    assert allocated[0]() is None


@pytest.mark.parametrize("bad", ["negative", "past_end", "wrong_dtype", "wrong_width", "meta"])
def test_copy_rejects_invalid_source(bad):
    buffer = RamEngramBuffer((11, 64), torch.uint8, chunk_rows=4)
    source = torch.zeros((2, 64), dtype=torch.uint8)
    start = 0
    if bad == "negative":
        start = -1
    elif bad == "past_end":
        start = 10
    elif bad == "wrong_dtype":
        source = source.bfloat16()
    elif bad == "wrong_width":
        source = torch.zeros((2, 63), dtype=torch.uint8)
    else:
        source = source.to("meta")
    with pytest.raises(ValueError, match="in-range CPU rows"):
        buffer.copy_rows(start, source)
    assert not buffer.loaded
    buffer.close()


@pytest.mark.parametrize("ids", [[-1], [11]])
def test_raw_gather_rejects_invalid_ids_instead_of_wrapping(ids):
    buffer = RamEngramBuffer((11, 64), torch.uint8, chunk_rows=4)
    with pytest.raises(ValueError, match="out of range"):
        buffer.gather_rows(torch.tensor(ids), torch.empty((len(ids), 64), dtype=torch.uint8))
    buffer.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8, torch.float8_e4m3fn])
def test_raw_rows_only_route_to_existing_mainline_decoder(monkeypatch, dtype):
    values = source_rows(11, 64, dtype)
    codes = RamEngramBuffer(values.shape, dtype, chunk_rows=4)
    codes.copy_rows(0, values)
    scales = None
    scale_dtype = None
    if dtype != torch.bfloat16:
        scale_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else torch.float32
        scale_values = torch.tensor([0, 255, 127, 128, 1, 254, 2, 126, 129, 253, 3] * 2, dtype=scale_dtype).reshape(
            11, 2
        )
        scales = RamEngramBuffer(scale_values.shape, scale_dtype, chunk_rows=4)
        scales.copy_rows(0, scale_values)
    with torch.device("meta"):
        staging = RamEngramStaging(3, 3, 64, dtype, scale_dtype, "cpu")
    assert all(t.device.type == "cpu" for t in staging.device_tensors())
    addresses = [t.data_ptr() for t in staging.device_tensors()]
    calls = []

    def decoder(weight, scale, ids, width, **kwargs):
        # This spy checks raw transfer and dispatch, not NPU numerical results.
        calls.append((weight.clone(), scale.clone() if scale is not None else None, ids.clone(), kwargs))
        assert weight.dtype == dtype and width == 64
        assert kwargs["local_heads"] == kwargs["pad_heads"] == 3
        assert kwargs["vocab_start"] == 0 and kwargs["vocab_end"] == ids.numel()
        kwargs["output"].fill_(17)
        return kwargs["output"]

    monkeypatch.setattr(npu, "gather_dequantize_engram_int8", decoder)
    monkeypatch.setattr(npu, "_host_library", Mock(side_effect=AssertionError("RAM must not register tables")))
    ids = torch.tensor([[999, 10, 13, 20, 999], [999, 19, -1, 21, 999]])
    output = torch.full((2, 3, 64), float("nan"), dtype=torch.bfloat16)
    assert staging.lookup(codes, scales, ids, output, head_start=1, vocab_start=10, vocab_end=21) is output
    selected = ids[:, 1:4]
    owned = (selected >= 10) & (selected < 21)
    local = torch.where(owned, selected - 10, 0).reshape(-1)
    expected = values.view(torch.uint8)[local] if dtype == torch.float8_e4m3fn else values[local].clone()
    expected[~owned.reshape(-1)] = 0
    actual = calls[0][0].view(torch.uint8) if dtype == torch.float8_e4m3fn else calls[0][0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if scales is not None:
        expected_scale = scale_values[local].clone()
        expected_scale[~owned.reshape(-1)] = 127 if scale_dtype == torch.uint8 else 0
        torch.testing.assert_close(calls[0][1], expected_scale, rtol=0, atol=0)
    assert torch.equal(calls[0][2], torch.arange(6).reshape(2, 3))
    assert (output == 17).all()
    # A smaller next request must publish current rows, with stable allocations.
    staging.lookup(codes, scales, torch.tensor([[10, 11, 12]]), output[:1], vocab_start=10, vocab_end=21)
    assert calls[1][0].shape == (3, 64) and calls[1][2].numel() == 3
    assert [t.data_ptr() for t in staging.device_tensors()] == addresses
    staging.close()
    staging.close()
    assert not staging.device_tensors()
    with pytest.raises(RuntimeError, match="closed"):
        staging.lookup(codes, scales, ids, output)
    codes.close()
    if scales is not None:
        scales.close()
    npu._host_library.assert_not_called()


@pytest.mark.parametrize("failure", ["unloaded", "capacity", "busy"])
def test_staging_refuses_unready_or_unbounded_calls(failure):
    codes = RamEngramBuffer((11, 64), torch.bfloat16, chunk_rows=4)
    staging = RamEngramStaging(2, 3, 64, torch.bfloat16, None, "cpu")
    tokens = 3 if failure == "capacity" else 1
    if failure != "unloaded":
        codes.copy_rows(0, torch.zeros((11, 64), dtype=torch.bfloat16))
    if failure == "busy":
        staging._busy = True
    with pytest.raises(ValueError if failure == "capacity" else RuntimeError):
        staging.lookup(
            codes, None, torch.zeros((tokens, 3), dtype=torch.int64), torch.empty((tokens, 3, 64), dtype=torch.bfloat16)
        )
    staging._busy = False
    staging.close()
    codes.close()


def test_capture_is_explicitly_rejected(monkeypatch):
    staging = RamEngramStaging(2, 3, 64, torch.bfloat16, None, "cpu")
    original_device = staging.device
    # Only the native capture-query boundary is mocked; storage remains real CPU.
    staging.device = SimpleNamespace(type="npu")
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True, raising=False)
    with pytest.raises(RuntimeError, match="uncaptured eager"):
        staging._assert_eager()
    staging.device = original_device
    staging.close()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.int8, torch.float8_e4m3fn])
def test_checkpoint_callback_streams_into_ram_without_meta_writes(tmp_path, monkeypatch, dtype):
    key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    values = source_rows(17, 64, dtype)
    tensors = {key: values}
    scale_values = None
    if dtype == torch.float8_e4m3fn:
        scale_values = torch.arange(34).reshape(17, 2).to(torch.uint8)
        tensors[scale_key] = scale_values.view(torch.float8_e8m0fnu)
    elif dtype == torch.int8:
        scale_values = torch.full((17, 2), 0.25)
        tensors[scale_key] = scale_values
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._shared_group = None
    table.vocab_start_idx, table.vocab_end_idx = 3, 14
    table.block_size, table.dim = 32, 64
    table._codes_uva = RamEngramBuffer((11, 64), dtype, chunk_rows=4)
    table.weight = torch.nn.Parameter(table._codes_uva.tensor, requires_grad=False)
    table._scales_uva = RamEngramBuffer((11, 2), scale_values.dtype, chunk_rows=4) if scale_values is not None else None
    table.weight_scale_inv = (
        torch.nn.Parameter(table._scales_uva.tensor, requires_grad=False) if table._scales_uva is not None else None
    )
    table.bind_checkpoint(tmp_path, key)
    monkeypatch.setattr(
        embedding_mod,
        "quantize_engram_rows",
        Mock(side_effect=AssertionError("checkpoint bits must not be requantized")),
    )
    table.load_checkpoint(tmp_path, key, chunk_rows=7)
    assert table.weight.is_meta and table._codes_uva.loaded
    expected = values[3:14].view(torch.uint8) if dtype == torch.float8_e4m3fn else values[3:14]
    actual = torch.cat(
        [t.view(torch.uint8) if dtype == torch.float8_e4m3fn else t for t in table._codes_uva.cpu_tensors()]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if table._scales_uva is not None:
        torch.testing.assert_close(torch.cat(table._scales_uva.cpu_tensors()), scale_values[3:14], rtol=0, atol=0)
        assert table._scales_uva.loaded
    table._weight_loader(table.weight, None)
    assert table._codes_uva.loaded
    table.close_host_offload()
    assert not table._codes_uva and table.weight.numel() == 0


def test_padded_heads_are_zeroed_before_mainline_dispatch(monkeypatch):
    codes = RamEngramBuffer((7, 64), torch.bfloat16, chunk_rows=4)
    codes.copy_rows(0, torch.ones((7, 64), dtype=torch.bfloat16))
    staging = RamEngramStaging(2, 3, 64, torch.bfloat16, None, "cpu")
    output = torch.full((1, 5, 64), float("nan"), dtype=torch.bfloat16)

    def decoder(weight, scale, ids, width, **kwargs):
        assert kwargs["local_heads"] == 3 and kwargs["pad_heads"] == 5
        shaped = kwargs["output"].view(1, 5, 64)
        assert not shaped[:, 3:].count_nonzero()
        shaped[:, :3].fill_(9)

    monkeypatch.setattr(npu, "gather_dequantize_engram_int8", decoder)
    staging.lookup(codes, None, torch.tensor([[0, 1, 2]]), output)
    assert (output[:, :3] == 9).all() and not output[:, 3:].count_nonzero()
    staging.close()
    codes.close()


def test_partial_scale_checkpoint_keeps_pair_unready(tmp_path, monkeypatch):
    key, scale_key = "layers.1.engram.embed.weight", "layers.1.engram.embed.scale"
    tensors = {key: torch.zeros((17, 64), dtype=torch.int8), scale_key: torch.ones((17, 2))}
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}})
    )
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table._shared_group = None
    table.vocab_start_idx, table.vocab_end_idx = 3, 14
    table.block_size, table.dim = 32, 64
    table._codes_uva = RamEngramBuffer((11, 64), torch.int8, chunk_rows=4)
    table._scales_uva = RamEngramBuffer((11, 2), torch.float32, chunk_rows=4)
    table.weight = torch.nn.Parameter(table._codes_uva.tensor, requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(table._scales_uva.tensor, requires_grad=False)
    original = table._scales_uva.copy_rows

    def fail_after_prefix(start, source):
        if start:
            raise OSError("scale checkpoint read failed")
        original(start, source)

    monkeypatch.setattr(table._scales_uva, "copy_rows", fail_after_prefix)
    with pytest.raises(OSError, match="scale checkpoint"):
        table.load_checkpoint(tmp_path, key, chunk_rows=7)
    assert table._codes_uva.loaded and not table._scales_uva.loaded
    table.close_host_offload()


def test_stream_owner_binds_at_lookup_and_waits_before_switch(monkeypatch):
    staging = RamEngramStaging(2, 3, 64, torch.bfloat16, None, "cpu")
    device = staging.device
    staging.device = SimpleNamespace(type="npu")
    first = SimpleNamespace(name="first", synchronize=Mock())
    second = SimpleNamespace(name="second", synchronize=Mock())
    current = Mock(side_effect=[first, second])
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False, raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", current, raising=False)
    staging._assert_eager(bind_stream=False)
    assert staging._stream is None and not current.called
    staging._assert_eager()
    assert staging._stream is first and not first.synchronize.called
    staging._assert_eager()
    first.synchronize.assert_called_once_with()
    assert staging._stream is second
    staging.close()
    second.synchronize.assert_called_once_with()
    staging.device = device


def test_failed_stream_wait_keeps_owner_and_storage_for_retry(monkeypatch):
    staging = RamEngramStaging(2, 3, 64, torch.bfloat16, None, "cpu")
    staging.device = SimpleNamespace(type="npu")
    first = SimpleNamespace(name="first", synchronize=Mock(side_effect=[OSError("stream wait failed"), None]))
    second = SimpleNamespace(name="second", synchronize=Mock(side_effect=[OSError("close wait failed"), None]))
    staging._stream = first
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False, raising=False)
    monkeypatch.setattr(torch.npu, "current_stream", lambda: second, raising=False)
    addresses = [tensor.data_ptr() for tensor in staging.device_tensors()]
    with pytest.raises(OSError, match="stream wait failed"):
        staging._assert_eager()
    assert staging._stream is first and not staging._closed
    assert [tensor.data_ptr() for tensor in staging.device_tensors()] == addresses
    staging._assert_eager()
    assert staging._stream is second
    with pytest.raises(OSError, match="close wait failed"):
        staging.close()
    assert not staging._closed and [tensor.data_ptr() for tensor in staging.device_tensors()] == addresses
    staging.close()
    assert staging._closed and not staging.device_tensors()


def test_unknown_capture_state_is_rejected(monkeypatch):
    staging = RamEngramStaging(2, 3, 64, torch.bfloat16, None, "cpu")
    device = staging.device
    staging.device = SimpleNamespace(type="npu")
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: None, raising=False)
    with pytest.raises(RuntimeError, match="uncaptured eager"):
        staging._assert_eager()
    assert staging._stream is None
    staging.device = device
    staging.close()
