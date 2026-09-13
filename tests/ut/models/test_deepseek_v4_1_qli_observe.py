# SPDX-License-Identifier: Apache-2.0
"""The one parameter that has not been observed, on the operator that reads
past it.

SparseFlashMla does not read past seqused_ori_kv -- measured. QuantLightning
IndexerV2 does: its range is block-aligned, ceil(seqused_k / block_size) *
block_size. With block_size 128 and a five-token request that is [0, 128):
five real tokens and 123 slots of whatever else is in that block.

Same parameter name, opposite behaviour, one code path feeding both.
"""

import logging

import pytest
import torch

from vllm.logger import logger
from vllm_ascend.models.deepseek_v4_1 import qli_observe


@pytest.fixture
def lines():
    captured = []

    class _Collect(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    qli_observe.reset_for_test()
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        qli_observe.reset_for_test()


def _observe(prefix="layers.2", lengths=(5, ), blocks=(0, 1), block_size=128):
    qli_observe.observe(
        prefix=prefix,
        seqused_k=torch.tensor(lengths, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, lengths[0]], dtype=torch.int32),
        block_table=torch.tensor(blocks, dtype=torch.int32),
        block_size=block_size)


def test_the_line_reports_the_block_aligned_range_not_only_the_length(lines):
    """🔴 The gap between the two numbers is the count of slots that are read
    and were never written by this request. That gap is the whole point."""
    _observe(lengths=(5, ), block_size=128)
    assert any("seqused_k=[5]" in line and "up to [128]" in line
               for line in lines), lines


def test_an_exact_multiple_has_no_gap(lines):
    """Two arms: the line must be able to say "nothing extra is read"."""
    _observe(lengths=(256, ), block_size=128)
    assert any("seqused_k=[256]" in line and "up to [256]" in line
               for line in lines), lines


def test_a_block_another_request_used_first_is_named(lines):
    """⚠️ Reuse is not itself a defect -- the allocator is supposed to recycle.
    It matters only together with the aligned range above."""
    _observe(prefix="layers.2", blocks=(7, 8))
    _observe(prefix="layers.8", blocks=(8, 9))
    assert any("reused-by-an-earlier-request=[8]" in line
               for line in lines), lines


def test_a_first_use_is_not_called_reuse(lines):
    _observe(prefix="layers.2", blocks=(3, 4))
    assert any("reused-by-an-earlier-request=none" in line
               for line in lines), lines


def test_the_same_prefix_reusing_its_own_block_is_not_reuse(lines):
    """The question is whether *another* request wrote it. A layer seeing its
    own block again is the normal case and would drown the real signal."""
    _observe(prefix="layers.2", blocks=(5, ))
    lines.clear()
    _observe(prefix="layers.2", blocks=(5, ))
    assert all("reused-by-an-earlier-request=[5]" not in line
               for line in lines), lines


def test_the_first_calls_are_all_reported(lines):
    for _ in range(qli_observe.VERBOSE_CALLS):
        _observe()
    assert len([line for line in lines if "DSV4.1 QLI call" in line]) == \
        qli_observe.VERBOSE_CALLS


def test_after_that_only_changes_are_reported(lines):
    for _ in range(qli_observe.VERBOSE_CALLS + 5):
        _observe()
    reported = [line for line in lines if "DSV4.1 QLI call" in line]
    assert len(reported) == qli_observe.VERBOSE_CALLS, reported

    lines.clear()
    _observe(lengths=(6, ))
    assert len([line for line in lines if "DSV4.1 QLI call" in line]) == 1, (
        "a changed length after the verbose window went unreported, which is "
        "the one thing the change-only mode must not do")


def test_the_block_table_read_is_capped(lines):
    """⚠️ A full block table is thousands of entries. A debug line that cost a
    large transfer per call would be the instrument disturbing what it
    measures."""
    qli_observe.observe(
        prefix="layers.2",
        seqused_k=torch.tensor([5], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
        block_table=torch.arange(4096, dtype=torch.int32),
        block_size=128)
    line = [ln for ln in lines if "DSV4.1 QLI call" in ln][0]
    assert "blocks=[0, 1, 2, 3, 4, 5, 6, 7]" in line, line


def test_empty_tensors_do_not_raise(lines):
    qli_observe.observe(
        prefix="layers.2",
        seqused_k=torch.zeros(0, dtype=torch.int32),
        cu_seqlens_q=torch.zeros(0, dtype=torch.int32),
        block_table=torch.zeros(0, dtype=torch.int32),
        block_size=128)
    assert any("DSV4.1 QLI call" in line for line in lines)


def test_the_indexer_calls_it_before_select_topk():
    """Placement: observed where the values are handed over, not where they
    are computed. "What was sent" and "what was meant to be sent" are
    different questions and only one of them is evidence."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.sparse_indexer import \
        AscendSparseAttnIndexer

    body = inspect.getsource(AscendSparseAttnIndexer.forward_oot)
    assert body.index("observe(") < body.index("topk_idxs = select_topk")


def test_it_changes_nothing(lines):
    """Print-only, stated as an observable: the tensors it is given come back
    untouched. The "output matched the baseline" check on the flight depends
    on this being true."""
    seqused = torch.tensor([5], dtype=torch.int32)
    blocks = torch.tensor([1, 2], dtype=torch.int32)
    before = (seqused.clone(), blocks.clone())
    qli_observe.observe(prefix="p", seqused_k=seqused,
                        cu_seqlens_q=torch.tensor([0, 5], dtype=torch.int32),
                        block_table=blocks, block_size=128)
    assert torch.equal(seqused, before[0]) and torch.equal(blocks, before[1])
