# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU backend

from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import _swiglu_group_index
from vllm_ascend.utils import bootstrap_custom_op_env


@pytest.fixture(scope="module", autouse=True)
def load_custom_ops():
    if not torch.npu.is_available() or "950" not in torch.npu.get_device_name(0):
        pytest.skip("MXFP group quantization requires Ascend 950")
    bootstrap_custom_op_env()
    importlib.import_module("vllm_ascend.vllm_ascend_C")


def quantize(x, group_index):
    return torch.ops._C_ascend.npu_swiglu_group_quant(
        x,
        topk_weight=None,
        group_index=group_index,
        dst_type=torch.float8_e4m3fn,
        quant_mode=2,
        clamp_value=0.0,
    )[:2]


def check_reference(x, quantized, scale, valid_rows):
    gate, up = x[:valid_rows].cpu().float().chunk(2, dim=-1)
    reference = torch.nn.functional.silu(gate) * up
    exponents = scale[:valid_rows].view(torch.uint8).cpu().reshape(valid_rows, -1).float() - 127
    reconstructed = quantized[:valid_rows].cpu().float() * torch.exp2(exponents).repeat_interleave(32, dim=-1)
    assert torch.isfinite(reconstructed).all()
    # E4M3 quantization and the BF16 activation intermediate both round values.
    assert torch.linalg.vector_norm(reconstructed - reference) / torch.linalg.vector_norm(reference) < 0.05


@pytest.mark.parametrize("experts", [100, 384])
@pytest.mark.parametrize("group_list_type", [0, 1])
@torch.inference_mode()
def test_mxfp_total_group_index_uses_product_helper(experts, group_list_type):
    rows, width = 768, 2304
    counts = torch.zeros(experts, dtype=torch.int64)
    counts[0], counts[experts // 2], counts[-1] = 555, 100, 113
    groups = (counts if group_list_type == 1 else counts.cumsum(0)).to("npu")
    original_groups = groups.clone()
    group_index = _swiglu_group_index(groups, group_list_type)
    torch.testing.assert_close(groups, original_groups, rtol=0, atol=0)
    assert group_index.dtype == torch.int64 and group_index.device == groups.device
    assert group_index.shape == (32,)
    torch.testing.assert_close(group_index.cpu(), torch.tensor([rows] + [0] * 31), rtol=0, atol=0)

    # Fixed BF16 input at the exact failing profile geometry. All rows are live,
    # so the no-index control is valid here; product routing keeps its total.
    x = ((torch.arange(rows * 2 * width).reshape(rows, 2 * width) % 97 - 48) / 16).to(
        device="npu", dtype=torch.bfloat16
    )
    expected, expected_scale = quantize(x, None)
    actual, actual_scale = quantize(x, group_index)
    torch.npu.synchronize()
    torch.testing.assert_close(actual.view(torch.uint8).cpu(), expected.view(torch.uint8).cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        actual_scale.view(torch.uint8).cpu(), expected_scale.view(torch.uint8).cpu(), rtol=0, atol=0
    )
    check_reference(x, actual, actual_scale, rows)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("width", [128, 256, 512, 640, 768])
@pytest.mark.parametrize("valid_rows", [7, 440])
@torch.inference_mode()
def test_mxfp_tail_repeatability(dtype, width, valid_rows):
    x = torch.randn(5120, 2 * width, device="npu", dtype=dtype)
    group_index = torch.tensor([valid_rows], device="npu", dtype=torch.int64)
    expected = None
    for _ in range(4):
        quantized, scale = quantize(x, group_index)
        current = quantized[:valid_rows].view(torch.uint8).cpu().clone()
        if expected is not None:
            torch.testing.assert_close(current, expected, rtol=0, atol=0)
        expected = current
        check_reference(x, quantized, scale, valid_rows)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_mxfp_tail_graph_replay(dtype):
    valid_rows = 440
    x = torch.randn(5120, 1280, device="npu", dtype=dtype)
    group_index = torch.tensor([valid_rows], device="npu", dtype=torch.int64)
    for _ in range(3):
        quantize(x, group_index)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        quantized, scale = quantize(x, group_index)
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        graph.replay()
        expected = quantized[:valid_rows].view(torch.uint8).cpu().clone()
        check_reference(x, quantized, scale, valid_rows)
        graph.replay()
        torch.testing.assert_close(quantized[:valid_rows].view(torch.uint8).cpu(), expected, rtol=0, atol=0)
