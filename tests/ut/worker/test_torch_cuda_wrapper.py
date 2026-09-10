# SPDX-License-Identifier: Apache-2.0
"""``_torch_cuda_wrapper`` must leave every CUDA entry point backed by NPU.

The wrapper redirects ``torch.cuda.*`` onto ``torch.npu.*`` so that upstream
model code written for CUDA runs here. Two failure shapes are worth pinning:

* An entry point nobody redirects still raises on NPU -- ``torch.cuda``'s own
  implementations are stubs ("Torch not compiled with CUDA enabled").
* An entry point restored to *itself* in ``finally`` silently keeps whatever
  the ``except`` branch installed, which is a placeholder that does nothing.
  That one is invisible: no error, no crash, wrong behaviour.
"""

import pytest
import torch

from vllm_ascend.worker.model_runner_v1 import _torch_cuda_wrapper

# Everything the wrapper promises to back with a real NPU implementation.
REDIRECTED = [
    "Event",
    "Stream",
    "default_stream",
    "current_stream",
    "stream",
    "synchronize",
    "mem_get_info",
    "is_current_stream_capturing",
    "get_device_properties",
]


def test_every_redirect_points_at_the_npu_implementation():
    with _torch_cuda_wrapper():
        for name in REDIRECTED:
            assert getattr(torch.cuda, name) is getattr(torch.npu, name), name
        assert torch.cuda.CUDAGraph is torch.npu.NPUGraph


def test_redirects_survive_the_finally_block():
    """Guards against restoring an entry point to itself.

    ``torch.cuda.Stream = torch.cuda.Stream`` in ``finally`` looks like the
    lines around it but restores nothing, so a failed init left the placeholder
    installed.
    """
    with _torch_cuda_wrapper():
        pass
    for name in REDIRECTED:
        assert getattr(torch.cuda, name) is getattr(torch.npu, name), name


def test_placeholders_do_not_outlive_a_failed_init():
    with pytest.raises(RuntimeError, match="NPUModelRunner init failed"):
        with _torch_cuda_wrapper():
            raise ValueError("simulated init failure")
    for name in REDIRECTED:
        assert getattr(torch.cuda, name) is getattr(torch.npu, name), name


def test_device_properties_reports_a_usable_parallel_unit_count():
    """The reason get_device_properties is redirected at all.

    Upstream sizes a persistent Triton grid from ``multi_processor_count``.
    NPU has no SMs, but ``_NPUDeviceProperties`` reports the vector core count
    under that name, which is a meaningful basis for a grid-stride loop.
    """
    with _torch_cuda_wrapper():
        props = torch.cuda.get_device_properties(0)
    assert props.multi_processor_count > 0
    assert props.vector_core_num > 0
