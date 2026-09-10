# SPDX-License-Identifier: Apache-2.0
"""``_torch_cuda_wrapper`` must leave every CUDA entry point backed by NPU.

The wrapper redirects ``torch.cuda.*`` onto ``torch.npu.*`` so that upstream
model code written for CUDA runs here. Two failure shapes are worth pinning:

* An entry point nobody redirects still raises on NPU -- ``torch.cuda``'s own
  implementations are stubs ("Torch not compiled with CUDA enabled").
* An entry point restored to *itself* in ``finally`` silently keeps whatever
  the ``except`` branch installed, which is a placeholder that does nothing.
  That one is invisible: no error, no crash, wrong behaviour.

Each case runs in its own interpreter. The thing under test is process-global
state, and anything else in the suite may touch ``torch.cuda`` at any point, so
asserting on it in-process makes the result depend on collection order --
green when this file runs alone, red when it runs after ``tests/ut/models``.
Snapshotting around each test fixes "we leak onto others" but cannot fix
"others leak onto us"; only a fresh interpreter can.
"""

import subprocess
import sys
import textwrap

import pytest

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

_PREAMBLE = """
import torch, torch_npu
# vllm_ascend.ops first: importing model_runner_v1 into a fresh interpreter
# trips a cycle through device_op -> ops -> fused_moe -> device_op, and a
# subprocess is always a fresh interpreter.
import vllm_ascend.ops  # noqa: F401
from vllm_ascend.worker.model_runner_v1 import _torch_cuda_wrapper

REDIRECTED = {redirected!r}

def check_all():
    for name in REDIRECTED:
        got, want = getattr(torch.cuda, name), getattr(torch.npu, name)
        assert got is want, f"{{name}}: {{got!r}} is not {{want!r}}"
"""


def _run(body: str) -> subprocess.CompletedProcess:
    script = _PREAMBLE.format(redirected=REDIRECTED) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=600)


def _assert_ok(result: subprocess.CompletedProcess) -> None:
    if result.returncode != 0:
        pytest.fail(f"subprocess failed ({result.returncode}):\n"
                    f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def test_every_redirect_points_at_the_npu_implementation():
    _assert_ok(_run("""
        with _torch_cuda_wrapper():
            check_all()
            assert torch.cuda.CUDAGraph is torch.npu.NPUGraph
    """))


def test_redirects_survive_the_finally_block():
    """Guards against restoring an entry point to itself.

    ``torch.cuda.Stream = torch.cuda.Stream`` in ``finally`` looks like the
    lines around it but restores nothing, so a failed init left the placeholder
    installed.
    """
    _assert_ok(_run("""
        with _torch_cuda_wrapper():
            pass
        check_all()
    """))


def test_placeholders_do_not_outlive_a_failed_init():
    _assert_ok(_run("""
        try:
            with _torch_cuda_wrapper():
                raise ValueError("simulated init failure")
        except RuntimeError as e:
            assert "NPUModelRunner init failed" in str(e), e
        else:
            raise AssertionError("the wrapper swallowed the failure")
        check_all()
    """))


def test_device_properties_reports_a_usable_parallel_unit_count():
    """The reason get_device_properties is redirected at all.

    Upstream sizes a persistent Triton grid from ``multi_processor_count``.
    NPU has no SMs, but ``_NPUDeviceProperties`` reports the vector core count
    under that name, which is a meaningful basis for a grid-stride loop.
    """
    _assert_ok(_run("""
        with _torch_cuda_wrapper():
            props = torch.cuda.get_device_properties(0)
        assert props.multi_processor_count > 0, props
        assert props.vector_core_num > 0, props
    """))
