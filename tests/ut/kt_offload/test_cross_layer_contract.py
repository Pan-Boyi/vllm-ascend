# SPDX-License-Identifier: Apache-2.0
"""Nothing may cross a layer boundary, checked on kt's own bookkeeping.

kt keeps one process-level dict of pending-deferred flags keyed by the bare
layer index; reading it through kt's public ``has_pending_deferred`` is the
only check that says what kt *actually saw*. A stale True makes kt's own drain
allowance 1, so the sync waits for one task fewer and returns early -- no
exception, just an output that is half written.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl


class Flagged:
    """A kt wrapper whose pending flags the test drives, with kt's API shape."""

    def __init__(self, layer_idx: int, pending: dict[int, bool] | None = None, infer=None):
        self.layer_idx = layer_idx
        self.max_deferred_experts_per_token = 0
        self.flags = pending if pending is not None else {}
        self.cpu_infer = infer if infer is not None else SimpleNamespace(submit=lambda t: None, sync=lambda a: None)
        self.moe = SimpleNamespace(forward_task=lambda *a: ("task", *a))
        self.calls = []

    def has_pending_deferred(self, layer_idx=None):
        idx = self.layer_idx if layer_idx is None else layer_idx
        return bool(self.flags.get(idx, False))

    def run_pinned_forward_sync(self, shape_ref, stream):
        self.calls.append("run_pinned_forward_sync")


@pytest.fixture(autouse=True)
def fresh_registry():
    kl._reset_wrapper_registry()
    yield
    kl._reset_wrapper_registry()


def test_a3_refuses_when_the_previous_layer_left_something():
    w = Flagged(5, {4: True})
    with pytest.raises(RuntimeError, match="incremental=True"):
        kl._assert_no_pending(w, 5, 4, "before this layer's submit")


def test_a4_names_the_early_return_not_just_the_flag():
    """The own-layer case must name its consequence: the drain waits for one task fewer."""
    w = Flagged(5, {5: True})
    with pytest.raises(RuntimeError) as exc:
        kl._assert_no_pending(w, 5, 5, "after this layer drained")
    assert "allowance 1" in str(exc.value) and "half written" in str(exc.value)


def test_the_checks_pass_on_a_clean_wrapper():
    w = Flagged(5, {})
    kl._assert_no_pending(w, 5, 4, "before")
    kl._assert_no_pending(w, 5, 5, "after")


def test_layer_zero_has_no_previous_layer_to_read():
    w = Flagged(0, {})
    kl._assert_no_pending(w, 0, -1, "before")  # kt returns False for an unknown key


def test_a_build_without_the_accessor_reports_zero_coverage(monkeypatch):
    """Absence of the accessor is not evidence that nothing is pending."""
    said = []
    monkeypatch.setattr(kl, "_PENDING_API_WARNED", False)
    monkeypatch.setattr(kl.logger, "warning", lambda *a, **k: said.append(a[0]))
    kl._assert_no_pending(SimpleNamespace(), 3, 2, "before")  # no raise
    assert any("Coverage of" in m and "zero" in m for m in said), said


def test_a3_and_a4_sit_on_the_eager_path():
    import ast
    import inspect

    for fn, want in ((kl.KTLayerOffload.route, "layer_idx - 1"), (kl.KTLayerOffload.merge, "self.layer_idx,")):
        src = inspect.getsource(fn)
        assert "_assert_no_pending(" in src, f"{fn.__name__} does not check kt's flags"
        assert want in src, f"{fn.__name__} checks the wrong layer index"
    # and on the two graph-node callbacks: the only per-replay place
    for node in (kl._kt_packed_forward, kl._kt_host_forward):
        calls = [
            n
            for n in ast.walk(ast.parse(inspect.getsource(node)))
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_assert_no_pending"
        ]
        assert len(calls) == 2, f"{node.__name__} has {len(calls)} checks, want one before and one after"


def test_the_graph_node_logs_the_refusal_instead_of_raising_into_acl(monkeypatch):
    """A raise inside a host-func node has nowhere to go; it must be logged instead."""
    logged = []
    monkeypatch.setattr(kl.logger, "error", lambda *a, **k: logged.append(a[0] % a[1:] if len(a) > 1 else a[0]))
    before = dict(kl.HOST_FUNC_CALLS)
    w = Flagged(3, {2: True})
    buf = {
        "bsz": torch.zeros(1, dtype=torch.int32),
        "pin_ids": torch.zeros(1, 2, dtype=torch.int64),
        "pin_weights": torch.zeros(1, 2),
        "pin_hidden": torch.zeros(1, 4, dtype=torch.bfloat16),
        "pin_out": torch.zeros(1, 4, dtype=torch.bfloat16),
    }
    kl._kt_packed_forward((w, buf, 3))
    assert any("raised" in m for m in logged), logged
    assert kl.HOST_FUNC_CALLS.get(3, 0) == before.get(3, 0), "a refused step was counted as having run"


def test_a5_refuses_two_live_wrappers_on_one_layer_index():
    first = Flagged(7)
    kl._register_wrapper(first, 7)
    second = Flagged(7)
    with pytest.raises(RuntimeError, match="process-level dict keyed by that index"):
        kl._register_wrapper(second, 7)
    assert first is not None  # keep it alive: the refusal is about two LIVE wrappers


def test_a5_allows_the_same_wrapper_to_register_twice():
    w = Flagged(7)
    kl._register_wrapper(w, 7)
    kl._register_wrapper(w, 7)


def test_a5_forgets_a_wrapper_that_is_gone():
    """A dropped wrapper writes no flags; refusing then would be a false red."""
    import gc

    kl._register_wrapper(Flagged(9), 9)
    gc.collect()
    kl._register_wrapper(Flagged(9), 9)  # must not raise


def test_a9_refuses_two_cpuinfer_objects():
    shared = SimpleNamespace(submit=lambda t: None, sync=lambda a: None)
    a = Flagged(1, infer=shared)
    b = Flagged(2, infer=SimpleNamespace(submit=lambda t: None, sync=lambda a: None))
    kl._register_wrapper(a, 1)
    with pytest.raises(RuntimeError, match="different CPUInfer"):
        kl._register_wrapper(b, 2)


def test_a9_accepts_the_shared_queue():
    shared = SimpleNamespace(submit=lambda t: None, sync=lambda a: None)
    for idx in (1, 2, 3):
        kl._register_wrapper(Flagged(idx, infer=shared), idx)


def test_the_registry_is_wired_into_the_build_path():
    import inspect

    src = inspect.getsource(kl.KTLayerOffload._build_wrapper)
    assert "_register_wrapper(" in src and "_refuse_cross_layer(" in src, src
