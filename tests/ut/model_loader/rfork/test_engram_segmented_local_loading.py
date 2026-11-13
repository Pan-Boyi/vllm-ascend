# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RFork readiness must describe real CPU chunks behind meta table parameters."""

import gc
import weakref

import pytest
import torch

from .rfork_test_support import _stub


class SegmentedCpuStorage:
    """Small protocol fixture: metadata is meta; payloads are independent tensors."""

    def __init__(self, shape, dtype):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.tensor = torch.empty(shape, dtype=dtype, device="meta")
        self.tensors = [torch.zeros((rows, *shape[1:]), dtype=dtype) for rows in (3, 3, 1)]
        self.loaded = False
        self._closing = False
        self.close_count = 0

    def cpu_tensors(self):
        return tuple(self.tensors)

    def close(self):
        if self._closing:
            return
        self._closing = True
        self.loaded = False
        self.close_count += 1
        self.tensors.clear()


@pytest.fixture
def segmented_table_factory(monkeypatch):
    class SegmentedEngramTable(torch.nn.Module):
        def __init__(self, quantized=True):
            super().__init__()
            self.cpu_offload = True
            self._checkpoint_path = "segmented-checkpoint"
            self._checkpoint_key = "layers.1.engram.embed.weight"
            self.load_count = 0
            self.fail_after_codes = False
            self.return_with_unloaded_scales = False
            code_dtype = torch.int8 if quantized else torch.bfloat16
            self._codes_uva = SegmentedCpuStorage((7, 8), code_dtype)
            self._scales_uva = SegmentedCpuStorage((7, 2), torch.float32) if quantized else None
            self.weight = torch.nn.Parameter(self._codes_uva.tensor, requires_grad=False)
            self.weight_scale_inv = (
                torch.nn.Parameter(self._scales_uva.tensor, requires_grad=False) if quantized else None
            )
            self.expected_codes = (torch.arange(56).reshape(7, 8) - 28).to(code_dtype)
            self.expected_scales = torch.arange(1, 15, dtype=torch.float32).reshape(7, 2) if quantized else None

        @staticmethod
        def _write(storage, source):
            offset = 0
            for tensor in storage.cpu_tensors():
                stop = offset + tensor.shape[0]
                tensor.copy_(source[offset:stop])
                offset = stop
            assert offset == source.shape[0]
            storage.loaded = True

        def load_checkpoint(self, path, key):
            assert (path, key) == (self._checkpoint_path, self._checkpoint_key)
            self.load_count += 1
            self._write(self._codes_uva, self.expected_codes)
            if self.fail_after_codes:
                raise OSError("synthetic scale checkpoint read failure")
            if self._scales_uva is not None:
                self._write(self._scales_uva, self.expected_scales)
                if self.return_with_unloaded_scales:
                    self._scales_uva.loaded = False

        def close_host_offload(self):
            for attribute, name in (("_codes_uva", "weight"), ("_scales_uva", "weight_scale_inv")):
                storage = getattr(self, attribute)
                if storage is None:
                    continue
                storage.close()
                setattr(self, attribute, None)
                parameter = getattr(self, name)
                setattr(
                    self,
                    name,
                    torch.nn.Parameter(torch.empty(0, dtype=parameter.dtype, device="meta"), requires_grad=False),
                )

    _stub(
        monkeypatch,
        "vllm_ascend.models.deepseek_v41.engram.embedding",
        AscendParallelEngramEmbedding=SegmentedEngramTable,
    )
    return SegmentedEngramTable


def _chunks(table):
    return tuple(
        tensor
        for storage in (table._codes_uva, table._scales_uva)
        if storage is not None
        for tensor in storage.cpu_tensors()
    )


def _metadata_ids(table):
    return {id(tensor) for tensor in (table.weight, table.weight_scale_inv) if tensor is not None}


def _assert_payload(table):
    for storage, expected in (
        (table._codes_uva, table.expected_codes),
        (table._scales_uva, table.expected_scales),
    ):
        if storage is None:
            continue
        offset = 0
        for tensor in storage.cpu_tensors():
            assert tensor.device.type == "cpu" and not tensor.is_meta
            stop = offset + tensor.shape[0]
            torch.testing.assert_close(tensor, expected[offset:stop], rtol=0, atol=0)
            offset = stop
        assert offset == expected.shape[0]


def test_unloaded_engram_meta_is_rejected_but_unrelated_meta_is_skipped(
    local_weights, tensor_runtime, segmented_table_factory, monkeypatch
):
    table = segmented_table_factory()
    model = torch.nn.Module()
    model.table = table
    model.unrelated_meta = torch.nn.Parameter(torch.empty(2, device="meta"))
    model.device_weight = torch.nn.Parameter(torch.ones(4))
    monkeypatch.setattr(
        tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda tensor: tensor is model.device_weight
    )

    assert local_weights.get_local_tensor_ids(model) == set()
    assert local_weights.get_unready_meta_tensor_ids(model) == _metadata_ids(table)
    assert set(tensor_runtime.tensor_layout.find_non_npu_state_tensors(model)) == {
        "table.weight",
        "table.weight_scale_inv",
    }
    assert table.load_count == 0


@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "int8-scales"])
@pytest.mark.parametrize("entrypoint", ["rfork", "default"])
def test_complete_chunks_are_real_loaded_and_only_real_chunk_ids_are_ready(
    local_weights, segmented_table_factory, monkeypatch, quantized, entrypoint
):
    table = segmented_table_factory(quantized)
    original_chunks = _chunks(table)
    original_ptrs = [tensor.data_ptr() for tensor in original_chunks]
    assert len(set(original_ptrs)) == len(original_ptrs)

    def no_payload_concat(*args, **kwargs):
        raise AssertionError("readiness must not concatenate or materialize the full host table")

    with monkeypatch.context() as context:
        context.setattr(torch, "cat", no_payload_concat)
        if entrypoint == "rfork":
            local_weights.load_local_weights(table)
        else:
            table.load_checkpoint(table._checkpoint_path, table._checkpoint_key)
            assert local_weights.get_local_tensor_ids(table) == set()
            local_weights.record_loaded_local_weights(table)

        assert table.weight.is_meta
        assert table.weight_scale_inv is None or table.weight_scale_inv.is_meta
        ready = {id(tensor) for tensor in original_chunks}
        assert local_weights.get_local_tensor_ids(table) == ready
        assert ready.isdisjoint(_metadata_ids(table))
        assert local_weights.get_unready_meta_tensor_ids(table) == set()
        assert [tensor.data_ptr() for tensor in _chunks(table)] == original_ptrs
        local_weights.load_local_weights(table)
        assert table.load_count == 1
    _assert_payload(table)


def test_loaded_chunks_clear_the_meta_guard_and_stay_out_of_transfer_manifest(
    local_weights, tensor_runtime, segmented_table_factory, monkeypatch
):
    model = torch.nn.Module()
    model.table = segmented_table_factory()
    model.device_weight = torch.nn.Parameter(torch.ones(4))
    monkeypatch.setattr(
        tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda tensor: tensor is model.device_weight
    )
    local_weights.load_local_weights(model)

    assert tensor_runtime.tensor_layout.find_non_npu_state_tensors(model) == []
    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, False)
    assert [name for name, _ in collected] == ["device_weight"]
    assert collected[0][1] is model.device_weight
    assert {id(tensor) for _, tensor in collected}.isdisjoint({id(tensor) for tensor in _chunks(model.table)})


def test_default_record_cannot_promote_unloaded_meta_parameters(local_weights, segmented_table_factory):
    table = segmented_table_factory()
    with pytest.raises(RuntimeError, match="Engram host table"):
        local_weights.record_loaded_local_weights(table)
    assert local_weights.get_local_tensor_ids(table) == set()
    assert local_weights.get_unready_meta_tensor_ids(table) == _metadata_ids(table)
    assert not any(tensor.count_nonzero().item() for tensor in _chunks(table))


def test_checkpoint_exception_retains_no_ready_state_and_retry_loads_all_chunks(local_weights, segmented_table_factory):
    table = segmented_table_factory()
    table.fail_after_codes = True
    with pytest.raises(OSError, match="scale checkpoint read failure"):
        local_weights.load_local_weights(table)
    assert table._codes_uva.loaded is True and table._scales_uva.loaded is False
    assert local_weights.get_local_tensor_ids(table) == set()
    assert local_weights.get_unready_meta_tensor_ids(table) == _metadata_ids(table)
    assert all(not tensor.count_nonzero().item() for tensor in table._scales_uva.cpu_tensors())

    table.fail_after_codes = False
    local_weights.load_local_weights(table)
    _assert_payload(table)
    assert local_weights.get_local_tensor_ids(table) == {id(tensor) for tensor in _chunks(table)}
    assert table.load_count == 2


def test_loader_return_without_both_loaded_flags_is_rejected(local_weights, segmented_table_factory):
    table = segmented_table_factory()
    table.return_with_unloaded_scales = True
    with pytest.raises(RuntimeError, match="Engram host table"):
        local_weights.load_local_weights(table)
    assert local_weights.get_local_tensor_ids(table) == set()
    assert local_weights.get_unready_meta_tensor_ids(table) == _metadata_ids(table)


@pytest.mark.parametrize("storage_name", ["_codes_uva", "_scales_uva"])
def test_any_unloaded_or_closed_storage_invalidates_the_whole_pair(
    local_weights, segmented_table_factory, storage_name
):
    table = segmented_table_factory()
    local_weights.load_local_weights(table)
    storage = getattr(table, storage_name)
    storage.loaded = False
    assert local_weights.get_local_tensor_ids(table) == set()
    assert local_weights.get_unready_meta_tensor_ids(table) == _metadata_ids(table)
    storage.close()
    assert local_weights.get_local_tensor_ids(table) == set()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_tail",
        "wrong_dtype",
        "wrong_width",
        "meta_leaf",
        "noncontiguous_leaf",
        "duplicate_leaf",
        "wrong_logical_shape",
    ],
)
def test_loaded_flag_does_not_override_real_chunk_geometry(local_weights, segmented_table_factory, mutation):
    table = segmented_table_factory()
    table.load_checkpoint(table._checkpoint_path, table._checkpoint_key)
    storage = table._codes_uva
    if mutation == "missing_tail":
        storage.tensors.pop()
    elif mutation == "wrong_dtype":
        storage.tensors[1] = storage.tensors[1].to(torch.float32)
    elif mutation == "wrong_width":
        storage.tensors[1] = torch.empty((3, 7), dtype=storage.dtype)
    elif mutation == "meta_leaf":
        storage.tensors[1] = torch.empty((3, 8), dtype=storage.dtype, device="meta")
    elif mutation == "noncontiguous_leaf":
        storage.tensors[1] = torch.empty((3, 16), dtype=storage.dtype)[:, ::2]
    elif mutation == "duplicate_leaf":
        storage.tensors[1] = storage.tensors[0]
    else:
        storage.tensor = torch.empty((8, 8), dtype=storage.dtype, device="meta")
        storage.shape = storage.tensor.shape
    assert storage.loaded is True
    with pytest.raises(RuntimeError, match="Engram host table"):
        local_weights.record_loaded_local_weights(table)
    assert local_weights.get_local_tensor_ids(table) == set()


@pytest.mark.parametrize("mutation", ["meta_parameter", "leaf_identity", "leaf_storage", "leaf_order", "checkpoint"])
def test_replacing_ready_metadata_or_real_storage_invalidates_record(local_weights, segmented_table_factory, mutation):
    table = segmented_table_factory()
    local_weights.load_local_weights(table)
    old_storage = None
    if mutation == "meta_parameter":
        table.weight = torch.nn.Parameter(torch.empty_like(table.weight), requires_grad=False)
    elif mutation == "leaf_identity":
        old = table._codes_uva.tensors[0]
        replacement = old.view_as(old)
        assert replacement is not old and replacement.data_ptr() == old.data_ptr()
        table._codes_uva.tensors[0] = replacement
    elif mutation == "leaf_storage":
        tensor = table._codes_uva.tensors[0]
        old_storage = tensor.detach()
        tensor.data = tensor.clone()
        assert tensor.data_ptr() != old_storage.data_ptr()
    elif mutation == "leaf_order":
        table._codes_uva.tensors.reverse()
    else:
        table._checkpoint_key = "layers.14.engram.embed.weight"
    assert local_weights.get_local_tensor_ids(table) == set()
    assert local_weights.get_unready_meta_tensor_ids(table) == _metadata_ids(table)


def test_draft_sharing_target_storage_neither_reloads_nor_releases_it(local_weights, segmented_table_factory):
    table = segmented_table_factory()
    target, draft = torch.nn.Module(), torch.nn.Module()
    target.table = draft.table = table
    local_weights.load_local_weights(target)
    chunks = _chunks(table)
    storages = (table._codes_uva, table._scales_uva)

    local_weights.load_local_weights(draft)
    local_weights.release_local_weights(draft)
    assert table.load_count == 1 and all(storage.close_count == 0 for storage in storages)
    assert local_weights.get_local_tensor_ids(draft) == {id(tensor) for tensor in chunks}
    _assert_payload(table)

    local_weights.release_local_weights(target)
    assert all(storage.close_count == 1 for storage in storages)
    assert local_weights.get_local_tensor_ids(target) == set()


def test_readiness_state_does_not_retain_model_metadata_or_real_chunks(local_weights, segmented_table_factory):
    def make_state_and_refs():
        model = torch.nn.Module()
        model.table = segmented_table_factory()
        local_weights.load_local_weights(model)
        table = model.table
        refs = [weakref.ref(value) for value in (model, table, table.weight, table.weight_scale_inv, *_chunks(table))]
        return table._rfork_engram_state, refs

    state, refs = make_state_and_refs()
    gc.collect()
    assert state.owner() is None
    assert all(ref() is None for ref in refs)
