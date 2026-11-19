# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real SSD fixture mappings, native row bits, identity and loader rollback."""

import gc
import json
import os
import struct
import sys
import weakref
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod
from vllm_ascend.models.deepseek_v41.engram import ssd_storage
from vllm_ascend.models.deepseek_v41.engram.ssd import SsdEngramBuffer, ssd_directory

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="SSD mappings require Linux disk filesystem checks")


@pytest.fixture
def prepared(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    index, payloads = {}, {}
    for layer, rows in ((1, 17), (14, 19)):
        header, data = {}, bytearray()
        shard = f"layer-{layer}.safetensors"
        for kind, width, dtype in (("weight", 256, "F8_E4M3"), ("scale", 8, "F8_E8M0")):
            name = f"layers.{layer}.engram.embed.{kind}"
            payload = np.arange(rows * width, dtype=np.int64).astype(np.uint8).reshape(rows, width)
            start = len(data)
            data.extend(payload.tobytes())
            header[name] = {"shape": [rows, width], "dtype": dtype, "data_offsets": [start, len(data)]}
            index[name] = shard
            payloads[name] = payload
        raw = json.dumps(header).encode()
        (checkpoint / shard).write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    (checkpoint / ssd_storage.INDEX_FILE).write_text(json.dumps({"weight_map": index}))
    stage = tmp_path / "ssd"
    ssd_storage.prepare_tables(checkpoint, stage, chunk_bytes=317)
    return checkpoint, stage, payloads


@pytest.mark.parametrize("kind,width,dtype", [("weight", 256, torch.float8_e4m3fn), ("scale", 8, torch.uint8)])
def test_native_raw_rows_tail_duplicates_empty_and_no_full_cpu_table(prepared, monkeypatch, kind, width, dtype):
    checkpoint, stage, payloads = prepared
    name = f"layers.1.engram.embed.{kind}"
    real_empty = torch.empty

    def meta_only(*args, **kwargs):
        assert kwargs.get("device") == "meta", "SSD must not allocate or pin a complete CPU table"
        return real_empty(*args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(torch, "empty", meta_only)
        buffer = SsdEngramBuffer((17, width), dtype)
    assert buffer.tensor.is_meta and not buffer.loaded and buffer.cpu_tensors() == ()
    buffer.bind_table(str(stage), checkpoint, name)
    assert buffer.loaded and not buffer._array.flags.writeable
    for ids in ([16, 0, 1, 16, 7], []):
        row_ids = torch.tensor(ids, dtype=torch.int64)
        out = torch.empty((len(ids), width), dtype=torch.uint8)
        buffer.gather_rows(row_ids, out)
        np.testing.assert_array_equal(out.numpy(), payloads[name][ids])
    for write in (lambda: buffer.zero_(), lambda: buffer.fill_(0), lambda: buffer.copy_rows(0, out)):
        with pytest.raises(RuntimeError, match="read-only"):
            write()
    fd = buffer._fd
    buffer.close()
    buffer.close()
    assert not buffer.loaded and buffer.closed and buffer.tensor is None
    with pytest.raises(OSError):
        os.fstat(fd)
    with pytest.raises(RuntimeError, match="closed"):
        buffer.gather_rows(torch.tensor([0]), torch.empty((1, width), dtype=torch.uint8))


@pytest.mark.parametrize("ids", [[-1], [17], [0, 18]])
def test_invalid_ids_never_wrap_or_clip_to_a_real_row(prepared, ids):
    checkpoint, stage, _ = prepared
    buffer = SsdEngramBuffer((17, 256), torch.float8_e4m3fn)
    buffer.bind_table(str(stage), checkpoint, "layers.1.engram.embed.weight")
    out = torch.full((len(ids), 256), 37, dtype=torch.uint8)
    with pytest.raises(ValueError, match="out of range"):
        buffer.gather_rows(torch.tensor(ids), out)
    assert bool((out == 37).all())
    buffer.close()


def test_discarded_buffer_releases_its_original_file_descriptor(prepared):
    checkpoint, stage, _ = prepared
    buffer = SsdEngramBuffer((17, 256), torch.float8_e4m3fn)
    buffer.bind_table(str(stage), checkpoint, "layers.1.engram.embed.weight")
    fd, ref = buffer._fd, weakref.ref(buffer)
    del buffer
    gc.collect()
    assert ref() is None
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("mutation", ["replace", "truncate", "writable", "symlink"])
def test_changed_table_invalidates_readiness_before_access(prepared, mutation):
    checkpoint, stage, _ = prepared
    name = "layers.1.engram.embed.weight"
    path = stage / ssd_storage.staged_file_name(name)
    buffer = SsdEngramBuffer((17, 256), torch.float8_e4m3fn)
    buffer.bind_table(str(stage), checkpoint, name)
    if mutation in ("replace", "symlink"):
        copy = stage / "copy"
        copy.write_bytes(path.read_bytes())
        copy.chmod(0o444)
        if mutation == "replace":
            copy.replace(path)
        else:
            path.unlink()
            path.symlink_to(copy)
    else:
        path.chmod(0o600)
        if mutation == "truncate":
            with path.open("r+b") as handle:
                handle.truncate(1)
    assert not buffer.loaded
    with pytest.raises(RuntimeError, match="identity changed"):
        buffer.gather_rows(torch.tensor([0]), torch.empty((1, 256), dtype=torch.uint8))
    buffer.close()


def test_mainline_loader_binds_codes_and_scales_and_rolls_back_together(prepared, monkeypatch):
    checkpoint, stage, payloads = prepared
    table = object.__new__(embedding_mod.AscendParallelEngramEmbedding)
    torch.nn.Module.__init__(table)
    table.host_backend, table.ssd_path = "ssd", str(stage)
    table._shared_group = table._ram_staging = None
    table._codes_uva = SsdEngramBuffer((17, 256), torch.float8_e4m3fn)
    table._scales_uva = SsdEngramBuffer((17, 8), torch.uint8)
    table.weight = torch.nn.Parameter(table._codes_uva.tensor, requires_grad=False)
    table.weight_scale_inv = torch.nn.Parameter(table._scales_uva.tensor, requires_grad=False)
    key = "layers.1.engram.embed.weight"
    table.bind_checkpoint(checkpoint, key)
    table._weight_loader(table.weight, None)
    assert table._codes_uva.loaded and table._scales_uva.loaded
    out = torch.empty((2, 256), dtype=torch.uint8)
    table._codes_uva.gather_rows(torch.tensor([16, 0]), out)
    np.testing.assert_array_equal(out.numpy(), payloads[key][[16, 0]])
    monkeypatch.setattr(table._scales_uva, "bind_table", Mock(side_effect=OSError("scale binding failed")))
    with pytest.raises(OSError, match="scale binding failed"):
        table._weight_loader(table.weight, None)
    assert not table._codes_uva.loaded and not table._scales_uva.loaded
    table.close_host_offload()


@pytest.mark.parametrize("path", [None, "", "relative"])
def test_ssd_path_requires_an_explicit_absolute_directory(path):
    with pytest.raises(ValueError, match="explicit absolute"):
        ssd_directory(path)


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float32])
def test_ssd_refuses_checkpoint_conversion(dtype):
    with pytest.raises(ValueError, match="native MXFP8"):
        SsdEngramBuffer((17, 256), dtype)
