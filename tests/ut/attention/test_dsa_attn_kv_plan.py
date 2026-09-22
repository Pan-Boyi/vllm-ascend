# SPDX-License-Identifier: Apache-2.0
from contextlib import ExitStack
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.attention.dsa_attn_kv_plan import (
    DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET,
    DSA_COMPRESSOR_SLOT_MAPPING_FLAT,
    get_dsa_attn_kv_plan,
    get_dsv4_attn_kv_dtype,
    is_a5_bf16_kv_enabled,
    resolve_dsv4_cache_dtype,
)
from vllm_ascend.attention.sparse_flash_mla import sparse_flash_mla
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.utils import AscendDeviceType

_DSA_C_ASCEND_OPS = (
    "npu_sparse_attn_sharedkv",
    "npu_sparse_attn_sharedkv_metadata",
    "npu_kv_quant_sparse_attn_sharedkv",
    "npu_kv_quant_sparse_attn_sharedkv_metadata",
    "kv_compress_epilog",
    "npu_scatter_nd_update_sk",
)


@pytest.fixture(autouse=True)
def _stub_dsa_c_ascend_ops():
    # CPU images do not register these custom ops on torch.ops._C_ascend.
    with ExitStack() as stack:
        for name in _DSA_C_ASCEND_OPS:
            stack.enter_context(mock.patch.object(torch.ops._C_ascend, name, create=True, new=MagicMock()))
        yield


def _config(use_bf16: bool):
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="bfloat16" if use_bf16 else "auto"))


def _cache_config(cache_dtype: str = "auto"):
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))


def _on(device_type):
    return mock.patch(
        "vllm_ascend.attention.dsa_attn_kv_plan.get_current_hardware_profile",
        return_value=get_hardware_profile(device_type),
    )


def test_get_dsa_attn_kv_plan_requires_vllm_config():
    with pytest.raises(TypeError):
        get_dsa_attn_kv_plan()


def test_a5_fp8_plan_uses_flat_shared_kv():
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(False))
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_FLAT
        assert plan.get_dsa_sparse_attn_metadata_kwargs("npu:0") == {"kv_quant_mode": 1}


def test_a5_bf16_plan_uses_sparse_flash_mla():
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(True))
        assert plan.get_dsa_sparse_attn_op() is sparse_flash_mla
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
        torch.testing.assert_close(
            plan.format_dsa_slot_mapping(torch.tensor([5, -1], dtype=torch.int32), 128),
            torch.tensor([[0, 5], [-1, -1]], dtype=torch.int32),
        )


def test_non_a5_plan_preserves_shared_kv_runtime_kwargs():
    with _on(AscendDeviceType.A3):
        plan = get_dsa_attn_kv_plan(_config(True))
        assert plan.get_dsa_compressor_slot_mapping_format() == DSA_COMPRESSOR_SLOT_MAPPING_BLOCK_OFFSET
        kwargs: dict[str, Any] = {}
        plan.add_dsa_sparse_attn_extra_kwargs(kwargs, cu_seqlens_ori_kv=torch.tensor([0, 1]))
        assert "cu_seqlens_ori_kv" in kwargs


def test_scatter_skips_none_updates():
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(False))
        cache = torch.zeros(2, 1, 4)
        with mock.patch.object(torch.ops._C_ascend, "kv_compress_epilog") as epilog:
            plan.dsa_kv_compress_scatter(cache, None, torch.tensor([0], dtype=torch.int32))
            epilog.assert_not_called()


def test_is_a5_bf16_kv_enabled_requires_vllm_config():
    with _on(AscendDeviceType.A5), pytest.raises(TypeError):
        is_a5_bf16_kv_enabled()


def test_only_explicit_bfloat16_selects_bf16_kv_on_a5():
    with _on(AscendDeviceType.A5):
        assert is_a5_bf16_kv_enabled(_cache_config("bfloat16"))
        assert not is_a5_bf16_kv_enabled(_cache_config())
        assert not is_a5_bf16_kv_enabled(_cache_config("fp8"))
        # The A5 spec path rewrites cache_dtype once FP8 KV is chosen.
        assert not is_a5_bf16_kv_enabled(_cache_config("float8_e4m3fn"))


def test_a5_bf16_kv_is_disabled_on_non_a5():
    with _on(AscendDeviceType.A3):
        assert not is_a5_bf16_kv_enabled(_cache_config("bfloat16"))


@pytest.mark.parametrize(
    ("device_type", "cache_dtype", "expected_dtype"),
    [
        (AscendDeviceType.A3, "auto", torch.bfloat16),
        (AscendDeviceType.A5, "bfloat16", torch.bfloat16),
        (AscendDeviceType.A5, "auto", torch.float8_e4m3fn),
    ],
)
def test_dsv4_attn_kv_dtype_preserves_device_modes(device_type, cache_dtype, expected_dtype):
    with _on(device_type):
        assert get_dsv4_attn_kv_dtype(_cache_config(cache_dtype)) == expected_dtype


def test_non_a5_pins_cache_dtype_to_the_model_dtype():
    with _on(AscendDeviceType.A3):
        for launch in ("auto", "bfloat16", "fp8"):
            assert resolve_dsv4_cache_dtype(launch, "bfloat16") == "bfloat16"


def test_a5_collapses_non_bfloat16_requests_to_auto():
    # "auto" resolves to the model dtype everywhere downstream, so it carries
    # the FP8 mode without changing any value upstream would have computed.
    with _on(AscendDeviceType.A5):
        assert resolve_dsv4_cache_dtype("bfloat16", "bfloat16") == "bfloat16"
        assert resolve_dsv4_cache_dtype("auto", "bfloat16") == "auto"
        assert resolve_dsv4_cache_dtype("fp8", "bfloat16") == "auto"


def test_a5_mode_survives_the_spec_path_rewrite():
    with _on(AscendDeviceType.A5):
        for launch in ("auto", "fp8"):
            pinned = resolve_dsv4_cache_dtype(launch, "bfloat16")
            assert not is_a5_bf16_kv_enabled(_cache_config(pinned))
            # layer.get_kv_cache_spec pins FP8 once it has picked the mode.
            assert not is_a5_bf16_kv_enabled(_cache_config("float8_e4m3fn"))

        pinned = resolve_dsv4_cache_dtype("bfloat16", "bfloat16")
        assert is_a5_bf16_kv_enabled(_cache_config(pinned))

# ---------------------------------------------------------------------------
# format_dsa_slot_mapping: the fused [T] -> [T, 2] path
#
# 🔴 What these can and cannot prove. The fused spelling is a triton kernel
# with no CPU implementation, so everything below exercises the **torch
# fallback** and the ``out=`` contract. That the kernel agrees with torch
# bit-for-bit is a card-only claim and is verified separately -- see the
# perf report's 3.3. Writing a CPU test that "passes" for the fused path
# would only prove the dispatch skipped it.
# ---------------------------------------------------------------------------


def _bf16_plan():
    return get_dsa_attn_kv_plan(_config(True))


def test_a_negative_slot_becomes_a_negative_pair_on_both_columns():
    # Not (-1, offset): the scatter operators skip a row only when the block
    # index is negative, and the compressor path reads the second column too.
    with _on(AscendDeviceType.A5):
        out = _bf16_plan().format_dsa_slot_mapping(
            torch.tensor([-1, 0, 129], dtype=torch.int64), 128
        )
        torch.testing.assert_close(
            out, torch.tensor([[-1, -1], [0, 0], [1, 1]], dtype=torch.int32)
        )


def test_out_is_written_in_place_and_returned():
    # The builder passes a view of its resident buffer, whose address the FULL
    # graph has baked in. Returning a fresh tensor instead of filling that
    # view is the failure mode this pins (48d3202a6).
    with _on(AscendDeviceType.A5):
        dest = torch.full((3, 2), 7, dtype=torch.int32)
        got = _bf16_plan().format_dsa_slot_mapping(
            torch.tensor([5, -1, 256], dtype=torch.int64), 128, out=dest
        )
        assert got is dest
        torch.testing.assert_close(
            dest, torch.tensor([[0, 5], [-1, -1], [2, 0]], dtype=torch.int32)
        )


def test_out_and_no_out_agree():
    with _on(AscendDeviceType.A5):
        plan = _bf16_plan()
        slots = torch.tensor([0, 127, 128, -1, 4095], dtype=torch.int64)
        dest = torch.empty((5, 2), dtype=torch.int32)
        plan.format_dsa_slot_mapping(slots, 128, out=dest)
        torch.testing.assert_close(dest, plan.format_dsa_slot_mapping(slots, 128))


@pytest.mark.parametrize(
    "dest, why",
    [
        (torch.empty((2, 2), dtype=torch.int32), "too few rows"),
        (torch.empty((3, 3), dtype=torch.int32), "not a (block, offset) pair"),
        (torch.empty((3, 2), dtype=torch.int64), "wrong dtype"),
    ],
)
def test_a_mismatched_out_is_refused_rather_than_reinterpreted(dest, why):
    # The fused kernel stores at ``offsets * 2`` and ``offsets * 2 + 1``, so a
    # mis-shaped destination would be filled with plausible-looking wrong
    # slots. Both paths must refuse it, or the CPU tests stop describing the
    # card. ``why`` is only here to name the case in the report.
    with _on(AscendDeviceType.A5):
        with pytest.raises((ValueError, RuntimeError)):
            _bf16_plan().format_dsa_slot_mapping(
                torch.tensor([5, -1, 0], dtype=torch.int64), 128, out=dest
            )


def test_the_flat_plan_still_honours_out():
    # requires_block_offset_slots=False keeps flat slots, and that builder's
    # resident buffer is 1-D (dsa_v1.py:1009). Refusing ``out`` here would
    # only break the plan we do not run in production -- the worst place for
    # a defect to sit, because nothing in our own runs would surface it.
    with _on(AscendDeviceType.A5):
        plan = get_dsa_attn_kv_plan(_config(False))
        assert not plan.requires_block_offset_slots
        dest = torch.zeros(3, dtype=torch.int32)
        got = plan.format_dsa_slot_mapping(
            torch.tensor([5, -1, 9], dtype=torch.int64), 128, out=dest
        )
        assert got is dest
        torch.testing.assert_close(dest, torch.tensor([5, -1, 9], dtype=torch.int32))


def test_an_empty_mapping_is_not_an_error():
    # Dummy runs reach the builder with zero tokens.
    with _on(AscendDeviceType.A5):
        out = _bf16_plan().format_dsa_slot_mapping(
            torch.empty(0, dtype=torch.int64), 128
        )
        assert tuple(out.shape) == (0, 2) and out.dtype == torch.int32


def test_the_fused_switch_does_not_change_the_cpu_result(monkeypatch):
    # The env switch exists to bisect a regression without a rebuild. On CPU
    # both settings must take the torch path, so the switch is a no-op here;
    # if this ever diverges, the dispatch is keying off something other than
    # is_npu and the CPU suite has stopped describing the card.
    with _on(AscendDeviceType.A5):
        plan = _bf16_plan()
        slots = torch.tensor([0, 130, -1], dtype=torch.int64)
        expected = plan.format_dsa_slot_mapping(slots, 128)
        for setting in ("0", "1"):
            monkeypatch.setenv("VLLM_ASCEND_DSV41_FUSED_SLOT_MAPPING", setting)
            torch.testing.assert_close(plan.format_dsa_slot_mapping(slots, 128), expected)
