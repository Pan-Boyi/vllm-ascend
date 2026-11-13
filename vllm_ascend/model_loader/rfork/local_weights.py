# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep DeepSeek Engram host loading and readiness inside RFork."""

import sys
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class _EngramLocalState:
    owner: weakref.ReferenceType[nn.Module]
    checkpoint: tuple[Any, str]
    tensors: tuple[tuple[weakref.ReferenceType[torch.Tensor], int], ...] = ()
    parameters: tuple[weakref.ReferenceType[torch.Tensor], ...] = ()


def _engram_tables(model: nn.Module | None) -> Iterator[Any]:
    # Avoid importing NPU model dependencies for unrelated architectures.
    module = sys.modules.get("vllm_ascend.models.deepseek_v41.engram.embedding")
    table_cls = getattr(module, "AscendParallelEngramEmbedding", None)
    if table_cls is None or not isinstance(model, nn.Module):
        return
    for table in model.modules():
        if isinstance(table, table_cls) and table.cpu_offload:
            yield table


def _table_parameters(table: Any) -> tuple[torch.Tensor, ...]:
    return tuple(tensor for tensor in (table.weight, table.weight_scale_inv) if tensor is not None)


def _table_tensors(table: Any) -> tuple[torch.Tensor, ...]:
    """Enumerate physical host storage without materializing a logical table."""
    tensors = []
    for parameter, buffer in (
        (table.weight, getattr(table, "_codes_uva", None)),
        (table.weight_scale_inv, getattr(table, "_scales_uva", None)),
    ):
        if parameter is None:
            continue
        cpu_tensors = getattr(buffer, "cpu_tensors", None)
        if callable(cpu_tensors):
            tensors.extend(cpu_tensors())
        else:
            # Shared/legacy tables retain their ordinary contiguous Parameter.
            tensors.append(parameter)
    return tuple(tensors)


def _storage_is_loaded(table: Any) -> bool:
    for parameter, buffer in (
        (table.weight, getattr(table, "_codes_uva", None)),
        (table.weight_scale_inv, getattr(table, "_scales_uva", None)),
    ):
        if parameter is None:
            continue
        cpu_tensors = getattr(buffer, "cpu_tensors", None)
        if not callable(cpu_tensors):
            if (
                parameter.is_meta
                or parameter.device.type != "cpu"
                or parameter.numel() <= 0
                or not parameter.data_ptr()
            ):
                return False
            continue
        # A returned fallback loader or a shape-only meta Parameter cannot prove
        # that every chunk was written. The buffer owns complete-row coverage.
        if (
            buffer.loaded is not True
            or tuple(buffer.shape) != tuple(parameter.shape)
            or buffer.dtype != parameter.dtype
        ):
            return False
        chunks = tuple(cpu_tensors())
        if not chunks:
            return False
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.is_meta
            or tensor.device.type != "cpu"
            or tensor.dtype != parameter.dtype
            or tensor.ndim != parameter.ndim
            or tuple(tensor.shape[1:]) != tuple(parameter.shape[1:])
            or tensor.numel() <= 0
            or not tensor.is_contiguous()
            or not tensor.data_ptr()
            for tensor in chunks
        ):
            return False
        if sum(tensor.shape[0] for tensor in chunks) != parameter.shape[0]:
            return False
    return True


def _ready_tensors(table: Any) -> tuple[torch.Tensor, ...]:
    state = getattr(table, "_rfork_engram_state", None)
    if not isinstance(state, _EngramLocalState):
        return ()
    if state.checkpoint != (table._checkpoint_path, table._checkpoint_key):
        return ()
    parameters = _table_parameters(table)
    if len(parameters) != len(state.parameters) or any(
        parameter is not loaded_ref() for parameter, loaded_ref in zip(parameters, state.parameters)
    ):
        return ()
    try:
        if not _storage_is_loaded(table):
            return ()
        tensors = _table_tensors(table)
    except (AttributeError, RuntimeError, TypeError):
        # A closing/replaced buffer must lose its exemption, including meta
        # parameters, even if its storage accessor now refuses inspection.
        return ()
    if len(tensors) != len(state.tensors):
        return ()
    for tensor, (loaded_ref, data_ptr) in zip(tensors, state.tensors):
        if tensor is not loaded_ref() or tensor.device.type != "cpu" or tensor.data_ptr() != data_ptr:
            return ()
    return tensors


def _new_state(model: nn.Module, table: Any) -> _EngramLocalState:
    state = _EngramLocalState(weakref.ref(model), (table._checkpoint_path, table._checkpoint_key))
    table._rfork_engram_state = state
    return state


def _record_tensors(state: _EngramLocalState, table: Any) -> None:
    # Weak refs must not pin replaced tensors or released CANN allocations.
    if not _storage_is_loaded(table):
        raise RuntimeError("Engram host table loading did not complete for all physical chunks")
    tensors = _table_tensors(table)
    if not tensors or len({id(tensor) for tensor in tensors}) != len(tensors):
        raise RuntimeError("Engram host table does not expose complete distinct physical storage")
    state.tensors = tuple((weakref.ref(tensor), tensor.data_ptr()) for tensor in tensors)
    state.parameters = tuple(weakref.ref(parameter) for parameter in _table_parameters(table))


def load_local_weights(model: nn.Module) -> None:
    """Load known host tables before transfer registration or draft reuse."""
    for table in _engram_tables(model):
        if _ready_tensors(table):
            continue
        state = _new_state(model, table)
        table.load_checkpoint(*state.checkpoint)
        # A partial load must never exempt the weight/scale pair from the guard.
        _record_tensors(state, table)


def record_loaded_local_weights(model: nn.Module | None) -> None:
    """Record tables only after the fallback loader returns a complete model."""
    if model is None:
        return
    for table in _engram_tables(model):
        if not _ready_tensors(table):
            _record_tensors(_new_state(model, table), table)


def get_local_tensor_ids(model: nn.Module) -> set[int]:
    """Exempt only still-live tensors whose local loading completed."""
    return {id(tensor) for table in _engram_tables(model) for tensor in _ready_tensors(table)}


def get_unready_meta_tensor_ids(model: nn.Module) -> set[int]:
    """Do not let unloaded segmented host tables bypass RFork as meta tensors."""
    return {
        id(parameter)
        for table in _engram_tables(model)
        if not _ready_tensors(table)
        for parameter in _table_parameters(table)
        if parameter.is_meta and parameter.numel() > 0
    }


def release_local_weights(model: nn.Module) -> None:
    """Release discarded host tables after RFork unregisters transfer memory."""
    for table in _engram_tables(model):
        state = getattr(table, "_rfork_engram_state", None)
        if isinstance(state, _EngramLocalState):
            # A draft may share the target table; never release target storage.
            if state.owner() is not model:
                continue
            state.tensors = ()
            state.parameters = ()
        table.close_host_offload()
