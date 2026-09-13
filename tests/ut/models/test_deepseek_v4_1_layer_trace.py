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
