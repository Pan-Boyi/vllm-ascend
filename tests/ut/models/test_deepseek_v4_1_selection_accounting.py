# SPDX-License-Identifier: Apache-2.0
"""One buffer, two writers, and a count that only followed one of them.

``topk_indices_buffer`` is written from two places on the V4.1 route:

* the indexer proper -- ``AscendSparseAttnIndexer.forward_oot``, which marked
  what it wrote from the first version;
* the **short-context path**, which fills the buffer with a triton kernel and
  returns ``index_q=None``, so ``if self.indexer is not None and index_q is not
  None`` is false and the indexer -- with its bookkeeping -- never runs.

The guard downstream then said "never written" about a buffer that had just
been written, and stopped a launch. It was reading its own bookkeeping bit
rather than the object it judges, and the bit had one writer while the object
had two.

⚠️ The two paths are chosen **per ratio group**, not per request: with
index_topk=512 the ratio-2 layers take the short circuit below 1026 tokens and
the ratio-1 layers below 513, so a single request can run some layers down each
path. Any statement about "the layers" that does not say which group it means
spans both.

So the tests here are of two kinds: that the second writer now keeps books,
and that a *third* writer appearing upstream turns this file red rather than
costing a flight.
"""

import ast
import pathlib

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.sparse_indexer import (
    _MarkingKernelLaunch, _SHORT_CONTEXT_MODULE, _SHORT_CONTEXT_NAME,
    apply_short_context_accounting_patch, mark_rows_written, rows_written,
    short_context_accounting_patched)


def _upstream_attention_source() -> str:
    import importlib

    module = importlib.import_module(_SHORT_CONTEXT_MODULE)
    return pathlib.Path(module.__file__).read_text()


# ------------------------------------------------------- who writes the buffer

#: Enclosing functions upstream that hand ``topk_indices_buffer`` to something
#: that can write it. Not a list of what we believe -- the test below derives
#: the set from upstream's source and compares, so a third writer is a failure
#: here instead of a wrong answer on a card.
KNOWN_WRITERS = {
    # the short-context triton kernel; wrapped by this line's rebind
    "forward",
    # SparseAttnIndexer(..., self.topk_indices_buffer, ...) -- the indexer
    # proper, whose forward_oot we replace and which marks its own rows
    "__init__",
}


def _functions_passing_the_buffer(source: str) -> set[str]:
    """Every function that hands the buffer to a call, by AST not by grep.

    Grep cannot tell ``self.topk_indices_buffer`` in an argument list from the
    same text in a comment or a type annotation, and the annotation-only
    mentions outnumber the real ones here.
    """
    tree = ast.parse(source)
    holders: set[str] = set()

    def mentions_buffer(node: ast.AST) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and \
                    sub.attr == "topk_indices_buffer":
                return True
            if isinstance(sub, ast.Name) and sub.id == "topk_indices_buffer":
                return True
        return False

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and any(
                    mentions_buffer(a) for a in node.args) or \
               isinstance(node, ast.Call) and any(
                   mentions_buffer(k.value) for k in node.keywords):
                holders.add(func.name)
            if isinstance(node, ast.Assign) and any(
                    mentions_buffer(t) for t in node.targets) and \
                    isinstance(node.targets[0], ast.Subscript):
                holders.add(func.name)
    return holders


def test_upstream_has_exactly_the_writers_this_line_accounts_for():
    """🔴 The test that makes a third writer cost a test run, not a flight."""
    found = _functions_passing_the_buffer(_upstream_attention_source())
    assert found, (
        "no function in upstream's V4.1 attention passes topk_indices_buffer "
        "to anything -- the scan matched nothing, which looks identical to "
        "'every writer is accounted for'. Repoint it before trusting it.")
    assert found == KNOWN_WRITERS, (
        f"upstream's writers of topk_indices_buffer are {sorted(found)}, and "
        f"this line accounts for {sorted(KNOWN_WRITERS)}. A writer that keeps "
        "no books makes the downstream provenance guard report the opposite "
        "of the truth -- either it raises on a written buffer, or it passes on "
        "rows nobody wrote.")


def test_the_scan_would_notice_a_new_writer():
    """The scan, run against a source that has one more writer."""
    extra = _upstream_attention_source() + (
        "\n\ndef _a_third_writer(self):\n"
        "    _some_kernel[(4,)](self.topk_indices_buffer)\n")
    assert "_a_third_writer" in _functions_passing_the_buffer(extra)


def test_the_scan_does_not_count_annotations_or_comments():
    source = (
        "def only_annotated(topk_indices_buffer: torch.Tensor | None = None):\n"
        "    # topk_indices_buffer is mentioned here too\n"
        "    self.topk_indices_buffer = topk_indices_buffer\n"
        "    return None\n")
    assert _functions_passing_the_buffer(source) == set()


# ---------------------------------------------------- the second writer's books

class _FakeJITFunction:
    """Stands in for the triton kernel: subscript with a grid, then call."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(buffer, *args, **kwargs):
            self.calls.append((grid, buffer, args, kwargs))
            buffer[:grid[0]] = 5
            return "launched"

        return launch


def test_the_wrapper_records_the_rows_the_grid_covers():
    inner = _FakeJITFunction()
    wrapped = _MarkingKernelLaunch(inner)
    buffer = torch.full((8, 4), -1)

    assert rows_written(buffer) is None
    result = wrapped[(3,)](buffer, "positions", TOP_K=4)
    assert result == "launched"
    assert rows_written(buffer) == 3, (
        "the wrapper did not record what the launch wrote, which is the whole "
        "of what it is for")
    assert inner.calls[0][0] == (3,)
    assert inner.calls[0][2] == ("positions", )
    assert inner.calls[0][3] == {"TOP_K": 4}


def test_the_count_comes_from_the_grid_not_from_a_parallel_argument():
    """⚠️ The grid *is* the written range: the kernel's row index is
    ``tl.program_id(0)``. Taking the count from anywhere else is a second
    source that can disagree with what ran."""
    inner = _FakeJITFunction()
    buffer = torch.zeros(16, 4)
    _MarkingKernelLaunch(inner)[(11,)](buffer, None, TOP_K=999)
    assert rows_written(buffer) == 11


def test_rows_at_or_past_the_grid_are_still_unwritten():
    """The over-read check applies to this path unchanged; it was simply
    unreachable while the first check fired on every short prompt."""
    inner = _FakeJITFunction()
    buffer = torch.zeros(16, 4)
    _MarkingKernelLaunch(inner)[(11,)](buffer, None)
    written = rows_written(buffer)
    assert written is not None and written < 16, (
        "a full-height count would tell the reader every row is real, and the "
        "kernel's grid says otherwise")


def test_an_integer_grid_is_accepted():
    inner = _FakeJITFunction()

    class _IntGrid(_FakeJITFunction):
        def __getitem__(self, grid):
            def launch(buffer, *a, **k):
                return grid

            return launch

    buffer = torch.zeros(4, 2)
    _MarkingKernelLaunch(_IntGrid())[6](buffer)
    assert rows_written(buffer) == 6


def test_attributes_of_the_wrapped_kernel_stay_reachable():
    inner = _FakeJITFunction()
    inner.cache_key = "abc"
    assert _MarkingKernelLaunch(inner).cache_key == "abc"


# ------------------------------------------------------------------ the rebind

def test_the_patch_binds_the_wrapper_over_upstreams_name():
    import importlib

    module = importlib.import_module(_SHORT_CONTEXT_MODULE)
    original = getattr(module, _SHORT_CONTEXT_NAME)
    try:
        apply_short_context_accounting_patch()
        assert short_context_accounting_patched()
        assert isinstance(getattr(module, _SHORT_CONTEXT_NAME),
                          _MarkingKernelLaunch)
    finally:
        setattr(module, _SHORT_CONTEXT_NAME, original)


def test_applying_it_twice_does_not_wrap_the_wrapper():
    """⚠️ Importing the package applies the patch, so this has to unwrap
    first. Without that the first call here returns False for the reason the
    test is checking for, and the assertion passes while proving nothing."""
    import importlib

    module = importlib.import_module(_SHORT_CONTEXT_MODULE)
    current = getattr(module, _SHORT_CONTEXT_NAME)
    inner = current._inner if isinstance(current,
                                         _MarkingKernelLaunch) else current
    try:
        setattr(module, _SHORT_CONTEXT_NAME, inner)
        assert apply_short_context_accounting_patch() is True
        assert apply_short_context_accounting_patch() is False
        wrapped = getattr(module, _SHORT_CONTEXT_NAME)
        assert not isinstance(wrapped._inner, _MarkingKernelLaunch)
    finally:
        setattr(module, _SHORT_CONTEXT_NAME, current)


def test_importing_the_package_has_already_applied_it():
    """Production call site, stated as an observable rather than as a grep."""
    import importlib

    importlib.import_module("vllm_ascend.models.deepseek_v4_1")
    assert short_context_accounting_patched()


def test_a_renamed_kernel_raises_rather_than_patching_nothing():
    import importlib

    module = importlib.import_module(_SHORT_CONTEXT_MODULE)
    original = getattr(module, _SHORT_CONTEXT_NAME)
    try:
        delattr(module, _SHORT_CONTEXT_NAME)
        with pytest.raises(AttributeError, match="no longer defines"):
            apply_short_context_accounting_patch()
    finally:
        setattr(module, _SHORT_CONTEXT_NAME, original)


# --------------------------------------------------- the count is step-scoped

class _Ctx:
    """Stands in for a ForwardContext: identity is all that is read."""


@pytest.fixture
def warnings_from_vllm():
    """Records from vllm's logger, which does not propagate to pytest's.

    ⚠️ The stock log fixture came back empty while the warning was plainly in
    the captured stdout -- an empty record list reads exactly like "nothing
    was logged", which is the opposite of what happened.
    """
    import logging

    from vllm.logger import logger

    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def test_a_count_from_this_forward_is_returned(monkeypatch):
    from vllm_ascend.models.deepseek_v4_1 import sparse_indexer

    ctx = _Ctx()
    monkeypatch.setattr(sparse_indexer, "_current_forward", lambda: ctx)
    buffer = torch.zeros(4, 2)
    mark_rows_written(buffer, 4)
    assert rows_written(buffer) == 4


def test_a_count_from_an_earlier_forward_is_reported(monkeypatch,
                                                    warnings_from_vllm):
    """🔴 The discriminating case. A bare integer reads the same whether the
    rows were written a microsecond ago or eight steps ago, and those are
    opposite answers about whether the buffer holds this step's selection."""
    from vllm_ascend.models.deepseek_v4_1 import sparse_indexer

    sparse_indexer._WARNED_STALE_SELECTION = False
    first, second = _Ctx(), _Ctx()
    monkeypatch.setattr(sparse_indexer, "_current_forward", lambda: first)
    buffer = torch.zeros(4, 2)
    mark_rows_written(buffer, 4)

    monkeypatch.setattr(sparse_indexer, "_current_forward", lambda: second)
    assert rows_written(buffer) == 4, (
        "the count is still returned on purpose: raising here would stop "
        "the launch on the hypothesis the launch is meant to test")
    assert any("never wrote it" in m for m in warnings_from_vllm), (
        "a stale count went by without a word, which is the state this was "
        "added to make visible")


def test_the_report_does_not_fire_when_the_forward_is_the_same(
        monkeypatch, warnings_from_vllm):
    """Two arms, separated: the warning has to be able to stay quiet."""
    from vllm_ascend.models.deepseek_v4_1 import sparse_indexer

    sparse_indexer._WARNED_STALE_SELECTION = False
    ctx = _Ctx()
    monkeypatch.setattr(sparse_indexer, "_current_forward", lambda: ctx)
    buffer = torch.zeros(4, 2)
    mark_rows_written(buffer, 4)
    rows_written(buffer)
    assert not any("never wrote it" in m for m in warnings_from_vllm)


def test_no_forward_context_is_a_third_state_not_a_mismatch(
        monkeypatch, warnings_from_vllm):
    """Outside a forward -- these tests, and anything off the model path --
    there is no step to compare against. That is neither "same step" nor
    "different step", and it must not be reported as either."""
    from vllm_ascend.models.deepseek_v4_1 import sparse_indexer

    sparse_indexer._WARNED_STALE_SELECTION = False
    monkeypatch.setattr(sparse_indexer, "_current_forward", lambda: None)
    buffer = torch.zeros(4, 2)
    mark_rows_written(buffer, 2)
    assert rows_written(buffer) == 2
    assert not any("never wrote it" in m for m in warnings_from_vllm)


def test_current_forward_returns_none_outside_a_forward():
    from vllm_ascend.models.deepseek_v4_1.sparse_indexer import \
        _current_forward

    assert _current_forward() is None


def test_an_unwritten_buffer_is_still_distinguishable_from_zero_rows():
    """The reason the count is three-state at all: "nothing wrote this" and
    "this step selected nothing" need different answers."""
    buffer = torch.zeros(4, 2)
    assert rows_written(buffer) is None
    mark_rows_written(buffer, 0)
    assert rows_written(buffer) == 0
