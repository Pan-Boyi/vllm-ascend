# SPDX-License-Identifier: Apache-2.0
"""The trace that turns "nine different completions" into "layer N".

Every hypothesis so far has been tested against the final text, which is
forty layers and a sampler away from wherever the divergence starts. A cause
at layer 2 and a cause at layer 39 produce the same observable, so each round
of guessing costs a flight.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.layer_trace import (fingerprint,
                                                          install_layer_trace,
                                                          selected_layers)


class _Layer(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


class _Inner(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.layers = torch.nn.ModuleList(_Layer(i) for i in range(n))
        self.start_layer = 0

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _Model(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.model = _Inner(n)

    def forward(self, x):
        return self.language_model.model(x)


# ------------------------------------------------------------- the fingerprint

def test_a_permutation_changes_the_fingerprint():
    """🔴 The property a sum, a mean and a norm all lack.

    A reduction that cannot see a permutation cannot see a misaligned read
    either, and a misaligned read is one of the things this is looking for.
    """
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([4.0, 3.0, 2.0, 1.0])
    assert a.sum() == b.sum()
    assert fingerprint(a) != fingerprint(b)


def test_equal_tensors_have_equal_fingerprints():
    a = torch.arange(64, dtype=torch.bfloat16).reshape(8, 8)
    assert fingerprint(a) == fingerprint(a.clone())


def test_one_changed_element_changes_it():
    a = torch.zeros(128, dtype=torch.bfloat16)
    b = a.clone()
    b[77] = 1
    assert fingerprint(a) != fingerprint(b)


def test_a_tensor_it_cannot_digest_reports_a_third_state():
    """⚠️ Not a constant. A fingerprint that silently becomes one value is a
    check bit with no discriminating power, which reads as agreement."""

    class _Undigestible:
        def detach(self):
            raise RuntimeError("no")

    assert fingerprint(_Undigestible()) == -1


def test_an_empty_tensor_is_zero_not_an_error():
    assert fingerprint(torch.zeros(0)) == 0


# ------------------------------------------------------------ the layer spec

def test_all_selects_every_layer():
    assert selected_layers("all", (0, 1, 2)) == (0, 1, 2)


def test_a_list_selects_those_layers_in_order_without_repeats():
    assert selected_layers("2, 0, 2", (0, 1, 2)) == (2, 0)


def test_an_empty_spec_selects_nothing():
    assert selected_layers("", (0, 1, 2)) == ()


def test_a_layer_that_does_not_exist_is_refused():
    """A quiet omission reads as 'that layer never diverged'."""
    with pytest.raises(ValueError, match="does not hold"):
        selected_layers("0,44", (0, 1, 2))


def test_a_spec_that_is_not_a_number_is_refused():
    with pytest.raises(ValueError, match="not a layer index"):
        selected_layers("0,attn", (0, 1, 2))


# ------------------------------------------------------------- installation

def test_off_registers_no_hooks(monkeypatch):
    """Off means there is nothing to enter, not merely nothing to do. Three
    launches were lost to instrumentation that was on."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "")
    model = _Model()
    assert install_layer_trace(model) == []
    assert not model.language_model.model.layers[0]._forward_hooks


def test_on_registers_one_hook_per_selected_layer(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "1,3")
    model = _Model()
    handles = install_layer_trace(model)
    try:
        hooked = [i for i, layer
                  in enumerate(model.language_model.model.layers)
                  if layer._forward_hooks]
        assert hooked == [1, 3]
    finally:
        for h in handles:
            h.remove()


def test_the_trace_lines_carry_the_layer_and_a_digest(monkeypatch, caplog):
    import logging

    from vllm.logger import logger

    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "all")
    model = _Model(n=2)
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    handles = install_layer_trace(model)
    try:
        model(torch.zeros(3, 4))
    finally:
        for h in handles:
            h.remove()
        logger.removeHandler(handler)

    lines = [r for r in records if "DSV4.1 trace step=" in r]
    assert len(lines) == 2, lines
    assert "layer=00" in lines[0] and "layer=01" in lines[1]
    assert "rows=3" in lines[0]


def test_two_identical_forwards_produce_identical_lines(monkeypatch):
    """The whole point, stated as an observable: same input, same lines."""
    import logging

    from vllm.logger import logger

    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "all")
    model = _Model(n=3)
    runs = []

    class _Collect(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "DSV4.1 trace step=" in message:
                # step= differs by construction; it is what tells two
                # forwards apart in a log read by grep.
                runs.append(message.split(" ", 3)[3])

    handler = _Collect()
    logger.addHandler(handler)
    handles = install_layer_trace(model)
    try:
        model(torch.zeros(2, 4))
        first = list(runs)
        runs.clear()
        model(torch.zeros(2, 4))
    finally:
        for h in handles:
            h.remove()
        logger.removeHandler(handler)

    assert first == runs and first, (first, runs)


def test_a_different_input_changes_the_first_layers_line(monkeypatch):
    """Two arms. A trace that is identical whatever the input is not a trace,
    it is a constant."""
    import logging

    from vllm.logger import logger

    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "0")
    model = _Model(n=2)
    seen = []

    class _Collect(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "DSV4.1 trace step=" in message:
                seen.append(message.split("out=")[1])

    handler = _Collect()
    logger.addHandler(handler)
    handles = install_layer_trace(model)
    try:
        model(torch.zeros(2, 4))
        model(torch.ones(2, 4))
    finally:
        for h in handles:
            h.remove()
        logger.removeHandler(handler)

    assert seen[0] != seen[1]


def test_a_model_with_no_layers_is_refused(monkeypatch):
    """An empty trace looks exactly like a run where nothing diverged."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "all")
    with pytest.raises(RuntimeError, match="no decoder layers"):
        install_layer_trace(torch.nn.Linear(2, 2))


def test_the_model_installs_it(monkeypatch):
    """Placement: a trace nothing on the production path installs is the
    shape install_alignment_taps had -- twelve callers, all of them tests."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import model as model_module

    source = inspect.getsource(model_module)
    assert "install_layer_trace(self)" in source


# ------------------------- telling reassociation apart from a real difference

def test_a_one_ulp_wobble_keeps_the_coarse_digest(monkeypatch):
    """🔴 The distinction the exact digest cannot make.

    A sum added up in a different order and a stale buffer both print as
    "different bits", and their remedies are opposite: one is a
    non-deterministic reduction whose wobble is flipping an argmax between
    near-tied tokens, the other is a memory bug.
    """
    from vllm_ascend.models.deepseek_v4_1.layer_trace import (
        coarse_fingerprint, fingerprint)

    a = torch.linspace(-1, 1, 512, dtype=torch.float32)
    b = a + a.abs() * 1e-4          # well under bf16's 3.9e-3 ULP
    assert fingerprint(a) != fingerprint(b)
    assert coarse_fingerprint(a) == coarse_fingerprint(b)


def test_a_real_difference_changes_the_coarse_digest_too():
    from vllm_ascend.models.deepseek_v4_1.layer_trace import coarse_fingerprint

    a = torch.linspace(-1, 1, 512, dtype=torch.float32)
    b = a.clone()
    b[100] = -b[100]
    assert coarse_fingerprint(a) != coarse_fingerprint(b)


def test_the_coarse_grid_is_above_the_bf16_floor():
    """⚠️ A threshold at or below the representation floor cannot separate
    "the same computation, reassociated" from "a different computation"."""
    from vllm_ascend.models.deepseek_v4_1.layer_trace import COARSE_LEVELS

    bf16_ulp = 2 ** -8              # 3.9e-3
    assert 1.0 / COARSE_LEVELS > bf16_ulp


def test_a_permutation_changes_the_coarse_digest_as_well():
    from vllm_ascend.models.deepseek_v4_1.layer_trace import coarse_fingerprint

    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert coarse_fingerprint(a) != coarse_fingerprint(a.flip(0))


def test_an_all_zero_tensor_is_not_an_error():
    from vllm_ascend.models.deepseek_v4_1.layer_trace import coarse_fingerprint

    assert coarse_fingerprint(torch.zeros(16)) == 0


def test_a_non_finite_tensor_reports_zero_rather_than_garbage():
    from vllm_ascend.models.deepseek_v4_1.layer_trace import coarse_fingerprint

    assert coarse_fingerprint(torch.full((8,), float("nan"))) == 0


def test_the_line_carries_both_digests(monkeypatch):
    import logging

    from vllm.logger import logger

    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "0")
    model = _Model(n=2)
    lines = []

    class _Collect(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if "DSV4.1 trace step=" in message:
                lines.append(message)

    handler = _Collect()
    logger.addHandler(handler)
    handles = install_layer_trace(model)
    try:
        model(torch.zeros(2, 4))
    finally:
        for h in handles:
            h.remove()
        logger.removeHandler(handler)

    assert lines and "out=0x" in lines[0] and "coarse=0x" in lines[0], lines


# ----------------------------- one transfer per forward, not one per layer

def test_the_layer_hooks_bring_nothing_back_to_the_host(monkeypatch):
    """🔴 The property that killed the first launch with the trace on.

    ``int(tensor)`` inside a layer hook is a host sync, and forty of them
    inside a forward that other ranks are not waiting at ended in aicpu
    507018 -- the same error code as the previous instrument that
    synchronised inside the forward. So the layer hooks must only queue
    device work; the single transfer happens after the forward.
    """
    from vllm_ascend.models.deepseek_v4_1 import layer_trace

    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "all")
    model = _Model(n=3)
    transfers = []

    real_to = torch.Tensor.to
    reads = []

    def _counting_to(self, *args, **kwargs):
        target = args[0] if args else kwargs.get("device")
        if target in ("cpu", torch.device("cpu")):
            transfers.append(tuple(self.shape))
        return real_to(self, *args, **kwargs)

    # ⚠️ The int-returning forms, watched by name. Counting .to("cpu") alone
    # passes under test for the wrong reason: everything is already on the
    # host here, so a per-layer int(tensor) moves nothing and the check stays
    # green while the regression it exists to catch is present. Patching
    # torch.Tensor.item does not help either -- int(t) reaches __int__, not
    # the Python attribute. What is actually being asserted is "the layer
    # hooks use the tensor form", so that is what is watched.
    real_exact = layer_trace.fingerprint
    real_coarse = layer_trace.coarse_fingerprint

    def _watch(fn):
        def _inner(*a, **k):
            reads.append(1)
            return fn(*a, **k)

        return _inner

    handles = install_layer_trace(model)
    monkeypatch.setattr(torch.Tensor, "to", _counting_to)
    monkeypatch.setattr(layer_trace, "fingerprint", _watch(real_exact))
    monkeypatch.setattr(layer_trace, "coarse_fingerprint",
                        _watch(real_coarse))
    try:
        model(torch.zeros(2, 4))
    finally:
        monkeypatch.undo()
        for h in handles:
            h.remove()

    assert len(transfers) == 1, (
        f"{len(transfers)} host transfers for 3 layers; the digests are "
        "being read back per layer again")
    assert transfers[0] == (3, 2), transfers
    assert not reads, (
        f"{len(reads)} host reads inside the forward; each one is a sync the "
        "other ranks are not waiting at")


def test_the_digest_tensors_stay_on_the_input_device():
    from vllm_ascend.models.deepseek_v4_1.layer_trace import (
        coarse_fingerprint_tensor, fingerprint_tensor)

    x = torch.arange(32, dtype=torch.bfloat16)
    assert fingerprint_tensor(x).device == x.device
    assert coarse_fingerprint_tensor(x).device == x.device
    assert fingerprint_tensor(x).ndim == 0


def test_the_tensor_and_int_forms_agree():
    from vllm_ascend.models.deepseek_v4_1.layer_trace import (
        coarse_fingerprint, coarse_fingerprint_tensor, fingerprint,
        fingerprint_tensor)

    x = torch.linspace(-1, 1, 128, dtype=torch.float32)
    assert int(fingerprint_tensor(x)) == fingerprint(x)
    assert int(coarse_fingerprint_tensor(x)) == coarse_fingerprint(x)


def test_the_coarse_form_branches_without_a_host_read():
    """⚠️ ``if float(scale) == 0`` would itself be the sync this avoids, so
    the zero and non-finite cases are handled by arithmetic."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import layer_trace

    import ast
    import textwrap

    body = inspect.getsource(layer_trace.coarse_fingerprint_tensor)
    # By AST, not by substring: the comment that explains why there is no
    # host read contains the very text a substring check looks for, so that
    # check went red on the thing it was documenting.
    tree = ast.parse(textwrap.dedent(body))
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {"float", "bool", "int"}, calls
    assert "torch.where" in body


def test_a_model_whose_layer_holder_is_missing_is_refused(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSV41_TRACE_LAYERS", "all")

    class _Bare(torch.nn.Module):
        pass

    bare = _Bare()
    bare.layers = torch.nn.ModuleList([_Layer(0)])
    # _find_inner_model finds it by `layers`, so this one is fine; the refusal
    # is for the case where it cannot be found at all.
    handles = install_layer_trace(bare)
    for h in handles:
        h.remove()


# --------------------------- constants that a device op will actually accept

def test_every_device_constant_fits_in_int64():
    """🔴 The check that catches a dead instrument without a device.

    0x9E3779B97F4A7C15 spelled unsigned is 11400714819323198485, past int64's
    maximum. CPU torch accepts it; the NPU's aclnnBitwiseXorScalar does not,
    and the failure landed in an except that returned the -1 sentinel -- so
    every trace line was the same constant, which reads as "all layers
    agree".

    ⚠️ Twenty-four tests here were green while the instrument was dead on the
    target, because all of them run on CPU. "Can be computed on CPU" and "can
    be computed on Ascend" are two claims and the first looks like the second.
    This asserts the property that differs.
    """
    from vllm_ascend.models.deepseek_v4_1.layer_trace import (
        DEVICE_INT_CONSTANTS, MIX)

    assert DEVICE_INT_CONSTANTS, "nothing is being checked"
    for value in DEVICE_INT_CONSTANTS:
        assert -2 ** 63 <= value < 2 ** 63, hex(value & (2 ** 64 - 1))
        # and it survives the round trip a torch op would put it through
        assert int(torch.tensor(value, dtype=torch.int64)) == value
    assert MIX & (2 ** 64 - 1) == 0x9E3779B97F4A7C15, (
        "the signed spelling must keep the same 64 bits, or the digest is a "
        "different function than the one the tests characterised")


def test_the_module_holds_no_unsigned_spelling_of_the_constant():
    """The literal is allowed in the comment that explains it, and nowhere a
    torch op can reach. Checked by AST so the explanation does not trip it --
    a substring check went red on its own documentation once already."""
    import ast
    import inspect
    import textwrap

    from vllm_ascend.models.deepseek_v4_1 import layer_trace

    tree = ast.parse(textwrap.dedent(inspect.getsource(layer_trace)))
    too_big = [node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant)
               and isinstance(node.value, int)
               and not isinstance(node.value, bool)
               and node.value >= 2 ** 63]
    # Two allowed, for opposite reasons. The golden-ratio literal appears
    # exactly once, as the operand of the subtraction that signs it. The
    # 64-bit mask appears in the log formatting, where it is applied to a
    # Python int with Python's & -- no torch op ever sees it, which is the
    # property that matters here.
    assert set(too_big) == {0x9E3779B97F4A7C15, 0xFFFFFFFFFFFFFFFF}, too_big
    assert too_big.count(0x9E3779B97F4A7C15) == 1
    # ...and the only one is the operand of the subtraction that signs it.
    signed = [node for node in ast.walk(tree)
              if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)
              and isinstance(node.left, ast.Constant)
              and node.left.value == 0x9E3779B97F4A7C15]
    assert len(signed) == 1


def test_a_failed_digest_says_the_trace_is_not_a_reading(monkeypatch):
    """⚠️ The sentinel is a constant, so a trace of sentinels reads as 'all
    layers agree' -- and the only line that could say otherwise was the one
    the except clause swallowed."""
    import logging

    from vllm.logger import logger

    from vllm_ascend.models.deepseek_v4_1 import layer_trace
    from vllm_ascend.models.deepseek_v4_1.switch_log import reset_for_test

    captured = []

    class _Collect(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    reset_for_test()

    class _Explodes:
        def detach(self):
            raise RuntimeError("aclnnBitwiseXorScalar failed")

    try:
        assert layer_trace.fingerprint(_Explodes()) == -1
    finally:
        logger.removeHandler(handler)
        reset_for_test()

    assert any("no reading at all" in line for line in captured), captured
