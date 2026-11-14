# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real safetensors CPU mappings, immutable bytes and bounded row gathering."""

import gc
import hashlib
import json
import os
import sys
import weakref
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.models.deepseek_v41.engram import checkpoint_mmap as mmap_mod
from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram.checkpoint_mmap import MmapEngramBuffer

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Checkpoint mmap verifies its own Linux file mapping")

KEY = "layers.1.engram.embed.weight"
SCALE_KEY = "layers.1.engram.embed.scale"


def checkpoint(tmp_path, dtype=torch.float8_e4m3fn):
    raw = torch.arange(17 * 64, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(17, 64)
    values = raw.view(dtype) if dtype == torch.float8_e4m3fn else raw.to(dtype)
    scales = torch.arange(34, dtype=torch.uint8).reshape(17, 2)
    tensors = {KEY: values}
    if dtype == torch.float8_e4m3fn:
        tensors[SCALE_KEY] = scales.view(torch.float8_e8m0fnu)
    path = tmp_path / "model.safetensors"
    save_file(tensors, path)
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {key: path.name for key in tensors}}))
    return path, index, values, scales


def bind(buffer, path, index, key=KEY):
    buffer.bind_checkpoint(
        path,
        key,
        source_shape=(17, buffer.shape[1]),
        row_start=3,
        index_reference=(index, hashlib.sha256(index.read_bytes()).hexdigest()),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_mmap_cross_chunk_duplicate_rows_preserve_bits_without_full_cpu_allocation(tmp_path, monkeypatch, dtype):
    path, index, values, _ = checkpoint(tmp_path, dtype)
    original = path.read_bytes()
    real_empty = torch.empty

    def no_full_cpu_table(*args, **kwargs):
        assert kwargs.get("device") == "meta", "mmap construction must not allocate a table in CPU RAM"
        return real_empty(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch, "empty", no_full_cpu_table)
        buffer = MmapEngramBuffer((11, 64), dtype, chunk_rows=4)
    assert buffer.tensor.is_meta and not buffer.loaded and not buffer.cpu_tensors()
    output_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else dtype
    ids = torch.tensor([10, 0, 3, 4, 7, 8, 4], dtype=torch.int64)
    output = torch.empty((len(ids), 64), dtype=output_dtype)
    with pytest.raises(RuntimeError, match="unloaded"):
        buffer.gather_rows(ids, output)
    bind(buffer, path, index)
    assert buffer.loaded and buffer.tensor.is_meta
    assert [chunk.shape[0] for chunk in buffer.cpu_tensors()] == [4, 4, 3]
    assert all(t.device.type == "cpu" and t.is_contiguous() and not t.is_meta for t in buffer.cpu_tensors())
    assert buffer.source_identity["open_mode"] == "O_RDONLY"
    assert buffer.source_identity["source_dtype"] in ("BF16", "F8_E4M3", "F8_E4M3FN")
    buffer.gather_rows(ids, output)
    expected = values[3:14]
    expected = expected.view(torch.uint8) if dtype == torch.float8_e4m3fn else expected
    torch.testing.assert_close(output, expected[ids], rtol=0, atol=0)
    for write in (lambda: buffer.zero_(), lambda: buffer.fill_(1), lambda: buffer.copy_rows(0, values[:1])):
        with pytest.raises(RuntimeError, match="read-only"):
            write()
    assert buffer.loaded and path.read_bytes() == original
    buffer.close()
    buffer.close()
    assert buffer.closed and not buffer.loaded and not buffer.cpu_tensors() and buffer.tensor is None
    with pytest.raises(RuntimeError, match="closed"):
        buffer.gather_rows(ids, output)
    assert path.read_bytes() == original


def test_e8m0_is_byte_reinterpreted_and_keeps_special_exponents(tmp_path):
    path, index, _, _ = checkpoint(tmp_path)
    # A separate real fixture contains exponent zero, NaN and endpoint bytes.
    raw = torch.tensor([0, 1, 126, 127, 128, 254, 255, 3] * 4 + [0, 255], dtype=torch.uint8).reshape(17, 2)
    save_file({SCALE_KEY: raw.view(torch.float8_e8m0fnu)}, path)
    index.write_text(json.dumps({"weight_map": {SCALE_KEY: path.name}}))
    buffer = MmapEngramBuffer((11, 2), torch.uint8, chunk_rows=4)
    bind(buffer, path, index, SCALE_KEY)
    ids = torch.tensor([0, 1, 2, 3, 8, 10, 1])
    output = torch.empty((len(ids), 2), dtype=torch.uint8)
    buffer.gather_rows(ids, output)
    torch.testing.assert_close(output, raw[3:14][ids], rtol=0, atol=0)
    buffer.close()


def test_escaped_chunk_keeps_mapping_and_readonly_fd_until_view_release(tmp_path):
    path, index, values, _ = checkpoint(tmp_path, torch.bfloat16)
    buffer = MmapEngramBuffer((11, 64), torch.bfloat16, chunk_rows=4)
    bind(buffer, path, index)
    chunk = buffer.cpu_tensors()[0]
    owner = weakref.ref(chunk._engram_checkpoint_owner)
    fd = owner().fd
    buffer.close()
    assert owner() is not None and os.fstat(fd).st_size == path.stat().st_size
    torch.testing.assert_close(chunk, values[3:7], rtol=0, atol=0)
    del chunk
    gc.collect()
    assert owner() is None
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("mutation", ["replace", "truncate", "rewrite", "index_replace"])
def test_source_mutation_invalidates_mapping_before_gather(tmp_path, mutation):
    path, index, _, _ = checkpoint(tmp_path)
    buffer = MmapEngramBuffer((11, 64), torch.float8_e4m3fn, chunk_rows=4)
    bind(buffer, path, index)
    if mutation in ("replace", "index_replace"):
        target = index if mutation == "index_replace" else path
        replacement = tmp_path / "replacement"
        replacement.write_bytes(target.read_bytes())
        replacement.replace(target)
    elif mutation == "truncate":
        with path.open("r+b") as file:
            file.truncate(path.stat().st_size - 1)
    else:
        before = path.stat()
        with path.open("r+b") as file:
            file.seek(-1, 2)
            file.write(b"\x80")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert path.stat().st_ctime_ns != before.st_ctime_ns
    assert not buffer.loaded
    output = torch.empty((1, 64), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="source identity changed"):
        buffer.gather_rows(torch.tensor([0]), output)
    buffer.close()


@pytest.mark.parametrize("dtype", [torch.int8, torch.float32, torch.float8_e8m0fnu])
def test_converted_or_unsupported_storage_is_rejected(dtype):
    with pytest.raises(ValueError, match="native BF16, FP8"):
        MmapEngramBuffer((11, 64), dtype)


@pytest.mark.parametrize("failure", ["shape", "dtype", "range", "private_copy"])
def test_bad_binding_remains_unloaded_and_releases_owner(tmp_path, monkeypatch, failure):
    path, index, _, _ = checkpoint(tmp_path)
    buffer = MmapEngramBuffer((11, 64), torch.float8_e4m3fn, chunk_rows=4)
    kwargs = {"source_shape": (17, 64), "row_start": 3}
    if failure == "shape":
        kwargs["source_shape"] = (18, 64)
    elif failure == "dtype":
        buffer.dtype = torch.bfloat16
    elif failure == "range":
        kwargs["row_start"] = 10
    else:
        original_open = mmap_mod.safe_open

        class CopyingReader:
            def __init__(self, *args, **kwargs):
                self.context = original_open(*args, **kwargs)

            def __enter__(self):
                self.reader = self.context.__enter__()
                return self

            def get_tensor(self, key):
                return self.reader.get_tensor(key).clone()

            def __exit__(self, *args):
                return self.context.__exit__(*args)

        monkeypatch.setattr(mmap_mod, "safe_open", CopyingReader)
    with pytest.raises((ValueError, RuntimeError)):
        buffer.bind_checkpoint(path, KEY, **kwargs)
    assert not buffer.loaded and not buffer.cpu_tensors() and buffer._owner is None
    buffer.close()


def mapped_table(dtype=torch.float8_e4m3fn):
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table.host_backend = "mmap"
    table.num_embeddings = 17
    table._shared_group = table._ram_staging = None
    table.vocab_start_idx, table.vocab_end_idx = 3, 14
    table._codes_uva = MmapEngramBuffer((11, 64), dtype, chunk_rows=4)
    table._scales_uva = MmapEngramBuffer((11, 2), torch.uint8, chunk_rows=4) if dtype == torch.float8_e4m3fn else None
    table.weight = torch.nn.Parameter(table._codes_uva.tensor, requires_grad=False)
    table.weight_scale_inv = (
        torch.nn.Parameter(table._scales_uva.tensor, requires_grad=False) if table._scales_uva is not None else None
    )
    return table


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_real_indexed_loader_binds_pair_without_copy_or_zero(tmp_path, monkeypatch, dtype):
    path, _, values, scales = checkpoint(tmp_path, dtype)
    original = path.read_bytes()
    table = mapped_table(dtype)
    table.bind_checkpoint(tmp_path, KEY)
    monkeypatch.setattr(table, "_copy_storage_rows", Mock(side_effect=AssertionError("must not copy mapped rows")))
    monkeypatch.setattr(embedding_mod, "quantize_engram_rows", Mock(side_effect=AssertionError("native bits required")))
    table._weight_loader(table.weight, None)
    assert table.weight.is_meta and table._codes_uva.loaded
    output_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else dtype
    output = torch.empty((2, 64), dtype=output_dtype)
    table._codes_uva.gather_rows(torch.tensor([10, 0]), output)
    expected = values.view(torch.uint8) if dtype == torch.float8_e4m3fn else values
    torch.testing.assert_close(output, expected[[13, 3]], rtol=0, atol=0)
    if table._scales_uva is not None:
        assert table._scales_uva.loaded
        result = torch.empty((2, 2), dtype=torch.uint8)
        table._scales_uva.gather_rows(torch.tensor([10, 0]), result)
        torch.testing.assert_close(result, scales[[13, 3]], rtol=0, atol=0)
    assert path.read_bytes() == original
    table.close_host_offload()


def test_scale_failure_unloads_both_mapped_tables_and_cannot_be_read(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    table = mapped_table()
    monkeypatch.setattr(table._scales_uva, "bind_checkpoint", Mock(side_effect=OSError("scale map failed")))
    with pytest.raises(OSError, match="scale map failed"):
        table.load_checkpoint(tmp_path, KEY)
    assert not table._codes_uva.loaded and not table._scales_uva.loaded
    assert not table._codes_uva.cpu_tensors() and not table._scales_uva.cpu_tensors()
    table.close_host_offload()


def test_index_cannot_map_a_shard_outside_model_root(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    checkpoint(tmp_path)
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {KEY: "../model.safetensors"}}))
    table = mapped_table()
    with pytest.raises(ValueError, match="within the original checkpoint"):
        table.load_checkpoint(model, KEY)
    assert not table._codes_uva.loaded and not table._scales_uva.loaded
    table.close_host_offload()
