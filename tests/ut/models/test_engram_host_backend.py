# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.models.deepseek_v41 import model as model_mod


def make_ascend_config(**kwargs):
    return AscendConfig(sparse_kv_offload_config=SimpleNamespace(enabled=False), **kwargs)


def make_vllm_config(*, cpu_offload=True, eager=True):
    return SimpleNamespace(
        engram_config=SimpleNamespace(cpu_offload=cpu_offload),
        model_config=SimpleNamespace(enforce_eager=eager),
    )


def test_host_backend_defaults_to_uva():
    assert make_ascend_config().engram_host_backend == "uva"


@pytest.mark.parametrize("backend", ["uva", "ram"])
def test_host_backend_accepts_explicit_storage_choice(backend):
    assert make_ascend_config(engram_host_backend=backend).engram_host_backend == backend


@pytest.mark.parametrize("backend", ["ssd", "", None, True, 1])
def test_host_backend_rejects_unknown_storage_choice(backend):
    with pytest.raises(ValueError, match="engram_host_backend"):
        make_ascend_config(engram_host_backend=backend)


def test_ram_backend_requires_eager_without_overlap(monkeypatch):
    config = make_ascend_config(engram_host_backend="ram", multistream_engram_overlap=False)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    assert model_mod._engram_host_backend_for_runtime(make_vllm_config(), has_engram=True) == "ram"


def test_ram_backend_rejects_a5_default_overlap_until_explicitly_disabled(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.ascend_config.get_current_hardware_profile", lambda: get_hardware_profile(AscendDeviceType.A5)
    )
    config = make_ascend_config(engram_host_backend="ram")
    assert config.multistream_engram_overlap is True
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    with pytest.raises(ValueError, match="requires multistream_engram_overlap=false"):
        model_mod._engram_host_backend_for_runtime(make_vllm_config(), has_engram=True)


@pytest.mark.parametrize(
    "eager,overlap,message",
    [
        (False, False, "requires --enforce-eager"),
        (True, True, "requires multistream_engram_overlap=false"),
        (False, True, "requires --enforce-eager"),
    ],
)
def test_ram_backend_rejects_unsupported_execution_before_allocation(monkeypatch, eager, overlap, message):
    config = make_ascend_config(engram_host_backend="ram", multistream_engram_overlap=overlap)
    monkeypatch.setattr(model_mod, "get_ascend_config", lambda: config)
    with pytest.raises(ValueError, match=message):
        model_mod._engram_host_backend_for_runtime(make_vllm_config(eager=eager), has_engram=True)


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
