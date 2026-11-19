# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v41 import model as model_mod
from vllm_ascend.models.deepseek_v41.engram import embedding as embedding_mod


def make_ascend_config(**kwargs):
    return AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False), **kwargs)


def make_vllm_config(*, cpu_offload=True, eager=True, graph_mode=CUDAGraphMode.NONE):
    return SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=cpu_offload),
        model_config=SimpleNamespace(enforce_eager=eager),
        compilation_config=SimpleNamespace(cudagraph_mode=graph_mode),
    )


def test_host_backend_defaults_to_uva():
    assert make_ascend_config().engram_host_backend == "uva"


def test_adding_ssd_path_preserves_the_existing_positional_token_capacity(monkeypatch):
    monkeypatch.setattr(embedding_mod, "get_tp_group", lambda: SimpleNamespace(cpu_group=None))
    monkeypatch.setattr(embedding_mod, "in_the_same_node_as", lambda group: [True])
    monkeypatch.setattr(embedding_mod, "get_engram_dp_group", lambda: None)
    monkeypatch.setattr(embedding_mod, "get_engram_dp_size", lambda: 1)
    monkeypatch.setattr(embedding_mod, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(embedding_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        embedding_mod.AscendParallelEngramEmbedding,
        "_allocate_weights",
        lambda self: (
            torch.empty((17, 32), dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty((17, 1), dtype=torch.uint8, device="meta"),
        ),
    )
    table = embedding_mod.AscendParallelEngramEmbedding(
        17, 32, (17,), 0, 32, True, False, torch.float8_e4m3fn, "ram", 9
    )
    assert table.max_tokens == 9 and table.ssd_path is None


@pytest.mark.parametrize("backend", ["uva", "ram", "mmap", "ssd"])
def test_host_backend_accepts_explicit_storage_choice(backend):
    assert make_ascend_config(engram_host_backend=backend).engram_host_backend == backend


@pytest.mark.parametrize("backend", ["unknown", "", None, True, 1])
def test_host_backend_rejects_unknown_storage_choice(backend):
    with pytest.raises(ValueError, match="engram_host_backend"):
        make_ascend_config(engram_host_backend=backend)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY])
def test_ssd_uses_the_same_external_staging_graph_contract(monkeypatch, tmp_path, mode):
    config = make_ascend_config(
        engram_host_backend="ssd", engram_ssd_path=str(tmp_path), multistream_engram_overlap=False
    )
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    validate = Mock(return_value=tmp_path)
    monkeypatch.setattr(model_mod, "ssd_directory", validate)
    assert (
        model_mod._engram_host_backend_for_runtime(make_vllm_config(eager=False, graph_mode=mode), has_engram=True)
        == "ssd"
    )
    validate.assert_called_once_with(str(tmp_path))


def test_ssd_path_validation_precedes_allocation(monkeypatch):
    config = make_ascend_config(engram_host_backend="ssd", multistream_engram_overlap=False)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    with pytest.raises(ValueError, match="explicit absolute"):
        model_mod._engram_host_backend_for_runtime(make_vllm_config(), has_engram=True)


def test_ssd_rejects_rfork_before_it_can_materialize_a_meta_table(monkeypatch, tmp_path):
    config = make_ascend_config(
        engram_host_backend="ssd", engram_ssd_path=str(tmp_path), multistream_engram_overlap=False
    )
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    monkeypatch.setattr(model_mod, "ssd_directory", Mock(return_value=tmp_path))
    runtime = make_vllm_config()
    runtime.load_config = SimpleNamespace(load_format="rfork")
    with pytest.raises(ValueError, match="standard checkpoint loader"):
        model_mod._engram_host_backend_for_runtime(runtime, has_engram=True)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY])
@pytest.mark.parametrize("eager", [False, True])
@pytest.mark.parametrize("backend", ["ram", "mmap"])
def test_ram_backend_accepts_eager_or_graph_external_full(monkeypatch, mode, eager, backend):
    config = make_ascend_config(engram_host_backend=backend, multistream_engram_overlap=False)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    vllm = make_vllm_config(eager=eager, graph_mode=mode)
    assert model_mod._engram_host_backend_for_runtime(vllm, has_engram=True) == backend


def test_ram_backend_rejects_a5_default_overlap_until_explicitly_disabled(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
    )
    config = make_ascend_config(engram_host_backend="ram")
    assert config.multistream_engram_overlap is True
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    with pytest.raises(ValueError, match="requires multistream_engram_overlap=false"):
        model_mod._engram_host_backend_for_runtime(make_vllm_config(), has_engram=True)


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL, CUDAGraphMode.FULL_DECODE_ONLY])
def test_ram_backend_rejects_overlap_before_allocation(monkeypatch, mode):
    config = make_ascend_config(engram_host_backend="ram", multistream_engram_overlap=True)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    with pytest.raises(ValueError, match="requires multistream_engram_overlap=false"):
        model_mod._engram_host_backend_for_runtime(make_vllm_config(eager=False, graph_mode=mode), has_engram=True)


@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE, None, "FULL", object()])
@pytest.mark.parametrize("backend", ["ram", "mmap"])
def test_ram_backend_rejects_piecewise_and_unknown_graph_modes(monkeypatch, mode, backend):
    config = make_ascend_config(engram_host_backend=backend, multistream_engram_overlap=False)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    with pytest.raises(ValueError, match="supports only NONE, FULL or FULL_DECODE_ONLY"):
        model_mod._engram_host_backend_for_runtime(make_vllm_config(eager=False, graph_mode=mode), has_engram=True)


def test_ram_backend_rejects_missing_compilation_config(monkeypatch):
    config = make_ascend_config(engram_host_backend="ram", multistream_engram_overlap=False)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    vllm = make_vllm_config(eager=False)
    del vllm.compilation_config
    with pytest.raises(ValueError, match="supports only NONE, FULL or FULL_DECODE_ONLY"):
        model_mod._engram_host_backend_for_runtime(vllm, has_engram=True)


def test_uva_backend_preserves_graph_and_overlap_settings(monkeypatch):
    config = make_ascend_config(engram_host_backend="uva", multistream_engram_overlap=True)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    assert model_mod._engram_host_backend_for_runtime(make_vllm_config(eager=False), has_engram=True) == "uva"


@pytest.mark.parametrize("has_engram,cpu_offload", [(False, True), (False, False), (True, False)])
def test_irrelevant_ram_setting_does_not_restrict_other_execution(monkeypatch, has_engram, cpu_offload):
    getter = Mock(side_effect=AssertionError("irrelevant Engram host configuration was read"))
    monkeypatch.setattr(model_mod, "get_ascend_config", getter)
    config = make_vllm_config(cpu_offload=cpu_offload, eager=False)
    assert model_mod._engram_host_backend_for_runtime(config, has_engram=has_engram) == "uva"
    getter.assert_not_called()


def test_absent_engram_config_does_not_read_host_configuration(monkeypatch):
    getter = Mock(side_effect=AssertionError("absent Engram host configuration was read"))
    monkeypatch.setattr(model_mod, "get_ascend_config", getter)
    config = make_vllm_config(eager=False)
    config.engram_config = None
    assert model_mod._engram_host_backend_for_runtime(config, has_engram=True) == "uva"
    getter.assert_not_called()
