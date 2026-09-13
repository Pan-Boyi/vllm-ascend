# SPDX-License-Identifier: Apache-2.0
"""An instrument that can kill the run is not an instrument.

tap_cache_read raised aclnnIndex 561103 on a 4-D NPU cache and the flight
came back 0/4 with an empty dump -- a server start and a warm-up spent on
the observer rather than on the thing observed. Everything in read_taps is
read-only and optional by construction, so nothing in it has any business
propagating.
"""

import logging

import pytest
import torch

from vllm.logger import logger
from vllm_ascend.models.deepseek_v4_1 import read_taps


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # ⚠️ switch_log too. announce_once is per process, so without this the
    # second test's line is suppressed by the first test's -- and the
    # assertion "it said so exactly once" then fails for a reason that has
    # nothing to do with the code under test. Passed alone, failed in the
    # file: state leaking between arms, which is the thing two-armed tests
    # exist to avoid.
    from vllm_ascend.models.deepseek_v4_1 import switch_log

    monkeypatch.setenv("VLLM_ASCEND_DSV41_TAP_READ_LAYERS", "2")
    read_taps.reset_for_test()
    switch_log.reset_for_test()
    yield
    read_taps.reset_for_test()
    switch_log.reset_for_test()


@pytest.fixture
def lines():
    captured = []

    class _Collect(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    try:
        yield captured
    finally:
        logger.removeHandler(handler)


# ---------------------------------------------- the gather, and its operator

def test_index_select_is_equivalent_to_advanced_indexing():
    """🔴 Why the substitution is safe.

    cache[ids] on a 4-D tensor goes through aclnnIndex, the high-dimensional
    index kernel, which failed on the NPU. index_select is the dedicated
    dim-0 gather and is exactly equivalent for one 1-D index on the first
    axis -- pinned here rather than asserted in prose.
    """
    cache = torch.arange(6 * 32 * 4, dtype=torch.float32).reshape(6, 32, 1, 4)
    ids = torch.tensor([3, 1], dtype=torch.int32)
    assert torch.equal(cache[ids.to(torch.int64)],
                       torch.index_select(cache, 0, ids))


def test_the_gather_does_not_use_advanced_indexing():
    """Pinned as source: a behaviour test cannot tell the two apart on CPU,
    and CPU is exactly where the failing one passed."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(read_taps.tap_cache_read)))
    subscripts = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript)
                  and isinstance(n.slice, ast.Name)]
    assert not subscripts, (
        f"{[ast.unparse(n) for n in subscripts]} indexes with a tensor; on a "
        "4-D NPU cache that reaches aclnnIndex, which is what took the run "
        "down")


def test_only_the_selected_blocks_are_taken():
    recorded = []
    read_taps.tap_read = _capture(read_taps.tap_read, recorded)
    try:
        cache = torch.arange(6 * 2 * 4, dtype=torch.float32).reshape(6, 2, 4)
        table = torch.tensor([[4, 4, 1]], dtype=torch.int32)
        read_taps.tap_cache_read(2, "ori_kv", cache, table)
    finally:
        read_taps.tap_read = read_taps.tap_read.__wrapped_original__
    keys = {k for _, k, _ in recorded}
    assert keys == {"ori_kv.block_ids", "ori_kv.block_digest",
                    "ori_kv.block_digest_coarse"}
    ids = [t for _, k, t in recorded if k == "ori_kv.block_ids"][0]
    assert ids.numel() == 2, "de-duplicated: 4 appears twice"
    digest = [t for _, k, t in recorded if k == "ori_kv.block_digest"][0]
    assert digest.numel() == 2, "one digest per block kept"


def _capture(original, sink):
    def _wrapper(layer_id, key, tensor):
        sink.append((layer_id, key, tensor))

    _wrapper.__wrapped_original__ = original
    return _wrapper


# ------------------------------------ the instrument cannot kill the run

def test_a_failing_tap_does_not_propagate(lines, monkeypatch):
    """🔴 The property the flight was missing."""
    def _explode(*a, **k):
        raise RuntimeError("call aclnnIndex failed, error code is 561103")

    monkeypatch.setattr(read_taps, "_block_ids", _explode)
    read_taps.tap_cache_read(2, "ori_kv", torch.zeros(4, 2),
                             torch.zeros(1, 2, dtype=torch.int32))
    assert any("has disabled itself" in line for line in lines), lines


def test_it_disables_itself_rather_than_failing_every_call(lines, monkeypatch):
    """⚠️ Throwing every call would fill the log and keep paying the cost;
    going quiet without a word would leave a dump missing keys for no stated
    reason, which reads as "this layer had nothing to read"."""
    calls = []

    def _explode(*a, **k):
        calls.append(1)
        raise RuntimeError("boom")

    monkeypatch.setattr(read_taps, "_block_ids", _explode)
    for _ in range(5):
        read_taps.tap_cache_read(2, "ori_kv", torch.zeros(4, 2),
                                 torch.zeros(1, 2, dtype=torch.int32))
    assert calls == [1], "it kept retrying after the first failure"
    assert len([ln for ln in lines if "has disabled itself" in ln]) == 1


def test_a_cache_failure_does_not_take_the_value_taps_with_it(monkeypatch):
    """🔴 The correction this flight forced.

    The flags were shared, tap_cache_read failed on a cache the device holds
    in an internal format, and it took tap_read down with it -- losing the
    selections, the lengths and the indexer's topk_idxs, which are small
    tensors already passed as arguments and never touch the gather. Those
    were exactly the keys the investigation needed.
    """
    recorded = []

    def _explode(*a, **k):
        raise RuntimeError("call aclnnIndexSelect failed, 561103")

    monkeypatch.setattr(read_taps, "_block_ids", _explode)
    read_taps.tap_cache_read(2, "ori_kv", torch.zeros(4, 2),
                             torch.zeros(1, 2, dtype=torch.int32))

    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda *a: recorded.append("tapped"))
    read_taps.tap_read(2, "seqused_cmp_kv", torch.zeros(3))
    assert recorded == ["tapped"], (
        "tap_read was disabled by a failure in a part it does not depend on")


def test_block_contents_failing_still_leaves_the_block_ids(monkeypatch):
    """⚠️ Which blocks and what is in them are two findings, and only the
    second one can fail this way."""
    recorded = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda layer, key, tensor: recorded.append(key))

    def _explode(*a, **k):
        raise RuntimeError("call aclnnIndexSelect failed, 561103")

    monkeypatch.setattr(read_taps, "_tap_block_contents", _explode)
    with pytest.raises(RuntimeError):
        read_taps._tap_block_contents(2, "ori_kv", None, None)
    read_taps.reset_for_test()

    monkeypatch.setattr(read_taps, "_tap_block_contents",
                        read_taps._never_breaks_the_run(_explode))
    read_taps.tap_cache_read(2, "ori_kv", torch.zeros(4, 2),
                             torch.tensor([[0, 1]], dtype=torch.int32))
    assert "read.ori_kv.block_ids" in recorded


def test_the_contents_are_digested_by_basic_indexing():
    """🔴 Two gather operators have failed on this tensor with the same code:
    aclnnIndex and aclnnIndexSelect. The same error from two different
    kernels points away from "which gather" and at the tensor. cache[int(i)]
    is basic indexing -- a view, not a kernel."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(read_taps._tap_block_contents)))
    calls = {ast.unparse(n.func) for n in ast.walk(tree)
             if isinstance(n, ast.Call)}
    assert not any("index_select" in c for c in calls), calls
    subscripts = [ast.unparse(n) for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript)]
    assert any("int(index)" in sub for sub in subscripts), subscripts


def test_the_digest_distinguishes_two_blocks():
    """A digest that cannot tell two blocks apart answers nothing."""
    recorded = {}
    import vllm_ascend.models.deepseek_v4_1.align_taps as at

    original = at.tap_whole
    at.tap_whole = lambda layer, key, tensor: recorded.__setitem__(key, tensor)
    try:
        cache = torch.zeros(3, 8)
        cache[1] = 1.0
        read_taps._tap_block_contents(
            2, "ori_kv", cache, torch.tensor([0, 1], dtype=torch.int32))
    finally:
        at.tap_whole = original
    digest = recorded["read.ori_kv.block_digest"]
    assert digest.numel() == 2 and digest[0] != digest[1]


def test_a_healthy_tap_is_not_disabled(monkeypatch):
    """Two arms: the guard must not swallow a working instrument."""
    recorded = []
    monkeypatch.setattr(read_taps, "_declare_moment",
                        lambda: recorded.append(1))
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda *a: recorded.append("tapped"))
    read_taps.tap_read(2, "seqused_cmp_kv", torch.zeros(3))
    assert "tapped" in recorded


def test_a_layer_that_is_not_selected_records_nothing(monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DSV41_TAP_READ_LAYERS", "2")
    recorded = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda *a: recorded.append("tapped"))
    read_taps.tap_read(3, "seqused_cmp_kv", torch.zeros(3))
    assert recorded == []
