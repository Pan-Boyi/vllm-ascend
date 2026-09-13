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
                    "ori_kv.block_digest_coarse", "ori_kv.rows",
                    "ori_kv.block0_row0"}
    ids = [t for _, k, t in recorded if k == "ori_kv.block_ids"][0]
    assert ids.numel() == 2, "de-duplicated: 4 appears twice"
    digest = [t for _, k, t in recorded if k == "ori_kv.block_digest"][0]
    assert digest.numel() == 2, "one digest per block kept"
    # The one deliberate exception, and it must stay under its own key: the
    # bounded keys above must not grow a block 0 the request does not own,
    # because "which blocks this request read" is what they answer.
    assert 0 not in ids.reshape(-1).tolist()


def test_block_zero_row_zero_is_recorded_whoever_owns_it():
    """🔴 Deliberately outside the block-table bound.

    The write being watched is a write into somebody else's row -- read per
    token, the dense slot mapping sends tokens past num_compressed_tokens to
    flat slot 0 -- so the bound that makes every other key correct is exactly
    what hides it. It was visible once, by accident, while the bound was
    missing; this reopens that window on purpose, one row wide.
    """
    recorded = []
    read_taps.tap_read = _capture(read_taps.tap_read, recorded)
    try:
        cache = torch.arange(6 * 2 * 4, dtype=torch.float32).reshape(6, 2, 4)
        table = torch.tensor([[4, 5]], dtype=torch.int32)
        read_taps.tap_cache_read(2, "cmp_kv", cache, table)
    finally:
        read_taps.tap_read = read_taps.tap_read.__wrapped_original__
    row = [t for _, k, t in recorded if k == "cmp_kv.block0_row0"][0]
    assert row.shape == (1, 4)
    torch.testing.assert_close(row[0], cache[0][0])


def test_the_canary_is_the_row_not_a_verdict():
    """"Is it zero" is the wrong question -- what settles it is whether the
    row still equals latent[3], and that comparison needs both tensors."""
    recorded = []
    read_taps.tap_read = _capture(read_taps.tap_read, recorded)
    try:
        cache = torch.zeros(3, 2, 4)
        read_taps.tap_cache_read(2, "cmp_kv", cache,
                                 torch.tensor([[1]], dtype=torch.int32))
    finally:
        read_taps.tap_read = read_taps.tap_read.__wrapped_original__
    rows = [t for _, k, t in recorded if k == "cmp_kv.block0_row0"]
    assert len(rows) == 1, "an all-zero row is still recorded, not skipped"


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


# ------------------- the diagnosis that decides between the two causes

def test_the_cache_is_described_before_anything_can_fail(lines):
    """🔴 Printed unconditionally, not only on failure.

    The traceback's own last lines say "Convert tensor to contiguous failed"
    after an AsStrided, and two candidate causes fit: a non-contiguous input
    the conversion cannot handle, or an index that is out of range. Exactly
    these numbers decide it, and one of the two does not raise -- so waiting
    for a failure to print them answers only half the question.
    """
    cache = torch.zeros(6, 2, 4)
    read_taps._tap_block_contents(2, "ori_kv", cache,
                                  torch.tensor([1, 4], dtype=torch.int32))
    line = [ln for ln in lines if "DSV4.1 read tap" in ln]
    assert line, lines
    for field in ("shape=", "stride=", "contiguous=", "storage_offset=",
                  "ids=", "in_range="):
        assert field in line[0], (field, line[0])


def test_the_description_says_when_an_index_is_out_of_range(lines):
    """"The output element is out of input range" is the operator's own
    wording; this makes the same judgement before the call, where it can
    name the id."""
    read_taps._tap_block_contents(2, "ori_kv", torch.zeros(3, 2),
                                  torch.tensor([0, 9], dtype=torch.int32))
    line = [ln for ln in lines if "DSV4.1 read tap" in ln][0]
    assert "in_range=False" in line and "max_id=9" in line, line


def test_a_view_that_cannot_be_landed_is_reported_as_that_step(lines,
                                                               monkeypatch):
    """⚠️ Its own step, not the caller's "tap failed". A failure to land
    means the tensor cannot be read on this device; a failure in the caller
    means the addressing was wrong. Different remedies."""
    class _Unlandable(torch.Tensor):
        pass

    view = torch.zeros(2, 4)
    monkeypatch.setattr(
        torch.Tensor, "clone",
        lambda self, *a, **k: (_ for _ in ()).throw(
            RuntimeError("Convert tensor to contiguous failed")))
    assert read_taps._landed(view, 3) is None
    assert any("could not materialise block 3" in ln for ln in lines), lines
    assert any("not the indexing" in ln for ln in lines), lines
    del _Unlandable


def test_the_block_ids_survive_a_landing_failure(lines, monkeypatch):
    recorded = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda layer, key, tensor: recorded.append(key))
    monkeypatch.setattr(read_taps, "_landed", lambda view, index: None)
    read_taps.tap_cache_read(2, "ori_kv", torch.zeros(4, 2),
                             torch.tensor([[0, 1]], dtype=torch.int32))
    assert "read.ori_kv.block_ids" in recorded
    assert not any("block_digest" in key for key in recorded)


# ------------------- the write side of the original stream

def test_the_original_stream_write_is_tapped_before_the_insert():
    """🔴 golden searched all 40 layers of its own dump -- 275,691 slices --
    for anything matching the NoPE 448 dims of our cached rows and found max
    |cos| = 0.106, while the values are plainly activations. So the question
    is not "where was it written" but "what was written", and only the write
    side answers it."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    body = inspect.getsource(
        AscendDeepseekV41Attention._fused_qnorm_rope_kv_insert)
    assert body.index("_tap_write_side_kv") < body.index(
        "return qnorm_rope_kv_insert"), (
        "the tap must run before the insert; afterwards the question has "
        "changed from 'what was handed over' to 'what is in the cache now'")


def test_the_pre_rope_latent_says_it_is_pre_rope():
    """⚠️ The cached row is this after rope. Comparing the two directly
    compares across a rotation -- the mistake that made a cosine of 0.15
    look like a finding."""
    from vllm_ascend.models.deepseek_v4_1.align_taps import KEY_SEMANTICS

    text = KEY_SEMANTICS["read.write.ori_kv.latent_pre_rope"]
    assert "before** rope" in text or "before rope" in text
    assert "orthogonal" in text, (
        "the entry has to say why a content comparison survives the rotation "
        "and a positional one does not -- otherwise the next reader repeats "
        "the same wrong comparison")


# ------------- the padding zero that read as a block of this request's

def test_the_block_list_is_bounded_by_the_sequence_not_by_non_negative():
    """🔴 Block tables are padded with **zero**, not -1 (dsa_v1.py:1570
    fills unused rows with 0), so filtering on non-negative keeps the
    padding -- and block 0 is a real block belonging to somebody else. That
    is what produced ori=[5, 0] and cmp=[84, 85, 0] in the dumps: a trailing
    zero that reads as "this request also touched block 0" and would have had
    its contents digested and attributed here.
    """
    table = torch.tensor([[5, 0, 0, 0]], dtype=torch.int32)
    ids = read_taps._block_ids(table, torch.tensor([5]), block_size=32)
    assert ids.tolist() == [5]


def test_a_long_sequence_keeps_every_block_it_needs():
    """Two arms: the bound must not truncate a request that really spans
    several blocks."""
    table = torch.tensor([[5, 9, 11, 0]], dtype=torch.int32)
    ids = read_taps._block_ids(table, torch.tensor([70]), block_size=32)
    assert ids.tolist() == [5, 9, 11]


def test_an_unbounded_list_says_it_is_unbounded(lines):
    """⚠️ Better than no list, but the reader must know which of the two
    they are holding."""
    table = torch.tensor([[5, 0, 0]], dtype=torch.int32)
    ids = read_taps._block_ids(table, None, None)
    assert ids.tolist() == [5, 0]
    assert any("upper bound, not as this request's blocks" in line
               for line in lines), lines


def test_the_page_size_comes_from_the_cache_not_the_config():
    """The two have disagreed once on this line already, and the tensor is
    the one being indexed."""
    from vllm_ascend.models.deepseek_v4_1.attention import _cache_page_size

    assert _cache_page_size(torch.zeros(64, 32, 1, 512)) == 32
    assert _cache_page_size(None) is None
    assert _cache_page_size(torch.zeros(8)) is None


def test_zero_is_kept_when_it_is_genuinely_the_first_block():
    """A request really allocated block 0 is not padding. The bound keeps it
    because it falls inside the needed count, which is the difference the
    non-negative filter could not express."""
    table = torch.tensor([[0, 7, 0]], dtype=torch.int32)
    ids = read_taps._block_ids(table, torch.tensor([40]), block_size=32)
    assert ids.tolist() == [0, 7]


def test_two_caches_with_different_pages_are_bounded_by_their_own(lines):
    """🔴 There is no such thing as "the" block size.

    The serve log's group table has four: SWA 32, two compressed groups at
    128 and 256, the compressor ring at 8 -- and each is correct in its own
    group. A fixture with one cache cannot express the difference, which is
    exactly the shape of bench that cannot show the failure it is meant to
    guard: reusing one page for another cache takes the wrong number of
    blocks, silently, because a block id is a block id.
    """
    recorded = {}
    import vllm_ascend.models.deepseek_v4_1.align_taps as at

    original = at.tap_whole
    at.tap_whole = lambda layer, key, tensor: recorded.__setitem__(key, tensor)
    try:
        # 70 tokens at page 32 needs 3 blocks; at page 128 it needs 1.
        table = torch.tensor([[5, 9, 11, 0, 0]], dtype=torch.int32)
        read_taps.tap_cache_read(2, "ori_kv", torch.zeros(64, 32, 1, 8),
                                 table, torch.tensor([70]), 32)
        read_taps.tap_cache_read(2, "cmp_kv", torch.zeros(64, 128, 1, 8),
                                 table, torch.tensor([70]), 128)
    finally:
        at.tap_whole = original

    assert recorded["read.ori_kv.block_ids"].tolist() == [5, 9, 11]
    assert recorded["read.cmp_kv.block_ids"].tolist() == [5], (
        "the compressed cache's 128-token page needs one block for 70 "
        "tokens; taking three is what reusing the SWA page does")


def test_the_indexer_tap_passes_its_own_page_and_length():
    """Pinned as source: the indexer call had neither, so it fell into the
    unbounded branch and its block list included the padding zeros."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.sparse_indexer import \
        _tap_indexer_reads

    body = inspect.getsource(_tap_indexer_reads)
    assert "key_values.shape[1]" in body
    assert "seqused_k, page" in body
