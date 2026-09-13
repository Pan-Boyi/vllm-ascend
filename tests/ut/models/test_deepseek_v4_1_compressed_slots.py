# SPDX-License-Identifier: Apache-2.0
"""The two slot-mapping conventions, and the criterion that separates them.

The defect these tests pin: Ascend's builder publishes V4.0's *dense* mapping
(one row per compressed state) for a compressing cache group, and both of
V4.1's writers index it *per token*. Read per token, the dense mapping writes
the second compressed entry, never the first, and sends a later token to a
slot belonging to whoever holds block 0.

Every constant in :func:`test_the_rebuild_reproduces_the_measured_mapping`
came off a flight, not off this file.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import compressed_slots


class _Req:
    """Copied from AscendDSAReqMetadata (vllm_ascend/attention/dsa_v1.py:1793).

    Not simplified: the rebuild reads block_table, query_start_loc and
    storage_block_size, and a double missing any of them would pass a rebuild
    that never ran.
    """

    def __init__(self, mapping, page, blocks, starts):
        self.slot_mapping = mapping
        self.token_slot_mapping = mapping
        self.storage_block_size = page
        self.block_table = torch.tensor(blocks, dtype=torch.int32)
        self.query_start_loc = torch.tensor(starts, dtype=torch.int32)


class _Meta:
    def __init__(self, req):
        self.req_metadata = req


class _Ctx:
    def __init__(self, attn_metadata=None):
        self.attn_metadata = attn_metadata


@pytest.fixture
def ctx(monkeypatch):
    compressed_slots.reset_for_test()
    holder = {"ctx": _Ctx()}

    def _install(attn_metadata):
        holder["ctx"] = _Ctx(attn_metadata)

    import vllm.forward_context as fc
    monkeypatch.setattr(fc, "get_forward_context", lambda: holder["ctx"])
    return _install


def _owned(ctx, mapping, page=128, blocks=((84, 85), ), starts=None, name="c2"):
    starts = [0, mapping.shape[0]] if starts is None else starts
    ctx({name: _Meta(_Req(mapping, page, blocks, starts))})
    return mapping


def _dense(num_tokens, ratio, num_reqs, blocks, page):
    """What the builder actually publishes, in the shape it publishes it.

    Rows [0, num_compressed_tokens) hold the states densely; the rest of the
    T-row buffer is the torch.zeros it was allocated with (dsa_v1.py:1135),
    because the per-token fill only runs at ratio <= 1 (dsa_v1.py:1298).
    """
    rows = min(num_tokens, num_tokens // ratio + num_reqs)
    flat = torch.zeros(num_tokens, dtype=torch.int32)
    states = num_tokens // ratio
    for state in range(min(states, rows)):
        flat[state] = blocks[state // page] * page + state % page
    if states < rows:
        flat[states] = -1
    return flat


# ------------------------------------------------- the rebuild is upstream's

def test_the_rebuild_reproduces_the_measured_mapping(ctx):
    """Flight-22, layer 2, T=5, ratio 2, one request, block 84, page 128.

    Upstream would produce [-1, 10752, -1, 10753, -1]:
    position 1 closes state 0 (84 * 128 + 0) and position 3 closes state 1.
    What the builder published instead was [10752, 10753, -1, 0, 0].
    """
    handed = _owned(ctx, torch.zeros(5, dtype=torch.int32))
    slots, layer = compressed_slots.per_token_compressed_slots(
        handed, torch.arange(5), 2)
    assert slots.tolist() == [-1, 10752, -1, 10753, -1]
    assert layer == "c2"


def test_a_second_page_of_states_walks_the_block_table(ctx):
    """State index, not token index, chooses the block -- they differ by the
    ratio, and using the token index would change blocks twice as fast."""
    page = 2
    handed = _owned(ctx, torch.zeros(8, dtype=torch.int32), page=page,
                    blocks=((7, 9), ))
    slots, _ = compressed_slots.per_token_compressed_slots(
        handed, torch.arange(8), 2)
    # states 0,1 -> block 7 offsets 0,1; states 2,3 -> block 9 offsets 0,1
    assert slots.tolist() == [-1, 14, -1, 15, -1, 18, -1, 19]


def test_each_request_uses_its_own_row_of_the_block_table(ctx):
    """Two requests packed into one forward. Taking request 0's blocks for
    every token is in-range arithmetic that lands on another request's rows."""
    handed = _owned(ctx, torch.zeros(5, dtype=torch.int32), page=128,
                    blocks=((84, ), (40, )), starts=[0, 3, 5])
    positions = torch.tensor([0, 1, 2, 0, 1])
    slots, _ = compressed_slots.per_token_compressed_slots(handed, positions, 2)
    assert slots.tolist() == [-1, 84 * 128, -1, -1, 40 * 128]


def test_ratio_zero_is_refused_rather_than_dividing_by_it(ctx):
    handed = _owned(ctx, torch.zeros(2, dtype=torch.int32))
    with pytest.raises(ValueError, match="no group boundary"):
        compressed_slots.per_token_compressed_slots(handed, torch.arange(2), 0)


def test_the_page_needs_two_sources_to_agree(ctx):
    handed = _owned(ctx, torch.zeros(2, dtype=torch.int32), page=128)
    cache = torch.zeros(4, 32, 512, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="real but wrong place"):
        compressed_slots.per_token_compressed_slots(handed, torch.arange(2), 2,
                                                    cache)


# ------------------------------------------------------------- the criterion

def test_the_dense_mapping_is_refused_and_named(ctx):
    """The whole point. What the builder publishes must not pass."""
    dense = _dense(5, ratio=2, num_reqs=1, blocks=[84], page=128)
    assert dense.tolist() == [10752, 10753, -1, 0, 0]
    with pytest.raises(ValueError) as excinfo:
        compressed_slots.check_per_token(dense, torch.arange(5), 2,
                                         "rope_quant_insert", "c2")
    message = str(excinfo.value)
    assert "DENSE (V4.0) convention" in message
    # Which rows are the problem, not just that something is. Row 0 holds
    # entry #0's destination at a non-boundary; row 4 is a never-written zero
    # that reads as block 0 offset 0 -- the stray write, in the message.
    assert "[0, 4]" in message
    assert "block 0, offset 0" in message


def test_the_rebuilt_mapping_passes(ctx):
    handed = _owned(ctx, torch.zeros(5, dtype=torch.int32))
    slots, layer = compressed_slots.per_token_compressed_slots(
        handed, torch.arange(5), 2)
    compressed_slots.check_per_token(slots, torch.arange(5), 2,
                                     "rope_quant_insert", layer)


@pytest.mark.parametrize("num_tokens,num_reqs", [(5, 1), (8, 1), (8, 2),
                                                 (17, 3), (64, 7)])
def test_length_cannot_tell_the_two_apart_but_the_valid_rows_can(
        ctx, num_tokens, num_reqs):
    """🔴 The negative control for the criterion we did *not* use.

    Both mappings are T rows of int32 in the same value range, so a check on
    "how many rows" passes both -- at every batch shape. The dense mapping's
    valid rows move with num_compressed_tokens = min(T, T//ratio + num_reqs),
    while the boundary rows depend only on positions, so the *set* separates
    them wherever the scalar does not.
    """
    starts = [0]
    per_req = num_tokens // num_reqs
    for r in range(num_reqs - 1):
        starts.append(per_req * (r + 1))
    starts.append(num_tokens)
    positions = torch.cat([
        torch.arange(starts[i + 1] - starts[i]) for i in range(num_reqs)
    ])
    handed = _owned(ctx, torch.zeros(num_tokens, dtype=torch.int32),
                    page=128, blocks=tuple((80 + r, ) for r in range(num_reqs)),
                    starts=starts)
    rebuilt, layer = compressed_slots.per_token_compressed_slots(
        handed, positions, 2)
    dense = _dense(num_tokens, 2, num_reqs, [84], 128)

    assert rebuilt.shape == dense.shape, "a length check would pass both"

    compressed_slots.reset_for_test()
    compressed_slots.check_per_token(rebuilt, positions, 2, "w", layer)
    compressed_slots.reset_for_test()
    with pytest.raises(ValueError, match="not group boundaries"):
        compressed_slots.check_per_token(dense, positions, 2, "w", layer)


def test_containment_not_equality_so_padding_is_not_an_error(ctx):
    """A boundary row may legitimately be -1; a non-boundary row may not be
    valid. Equality would turn ordinary padding into a refused flight."""
    slots = torch.tensor([-1, 10752, -1, -1, -1], dtype=torch.int32)
    compressed_slots.check_per_token(slots, torch.arange(5), 2, "w", "c2")


def test_ratio_one_has_no_boundary_to_violate(ctx):
    slots = torch.tensor([0, -1, 2, 3], dtype=torch.int32)
    compressed_slots.check_per_token(slots, torch.arange(4), 1, "w", "c1")


# ------------------------------------- the criterion is also the instrument

def _announcements(monkeypatch):
    lines = []

    class _Logger:
        @staticmethod
        def info(fmt, *args):
            lines.append(fmt % args)

    import vllm.logger
    monkeypatch.setattr(vllm.logger, "logger", _Logger)
    return lines


def test_passing_is_announced_once_per_site(ctx, monkeypatch):
    """🔴 Silence has to mean something.

    "No exception" is true both when a writer's mapping is right and when the
    writer never ran at all -- and the indexer half of this fix has no
    runtime coverage to distinguish them. So the check speaks when it passes,
    once per (consumer, layer, ratio), and a flight's log answers "did both
    writers run, and on which layers" without a new tap.
    """
    lines = _announcements(monkeypatch)
    slots = torch.tensor([-1, 10752, -1, 10753, -1], dtype=torch.int32)
    for _ in range(3):
        compressed_slots.check_per_token(slots, torch.arange(5), 2,
                                         "rope_quant_insert", "layer_02")
    assert len(lines) == 1
    assert "consumer=rope_quant_insert" in lines[0]
    assert "layer=layer_02" in lines[0]
    assert "ratio=2" in lines[0]

    compressed_slots.check_per_token(slots, torch.arange(5), 2,
                                     "indexer_k_store", "layer_02")
    assert len(lines) == 2, "a second consumer is a second site"
    assert "consumer=indexer_k_store" in lines[1]


def test_a_writer_that_never_runs_announces_nothing(ctx, monkeypatch):
    """The half of the reading that makes the other half worth having."""
    lines = _announcements(monkeypatch)
    compressed_slots.check_per_token(
        torch.tensor([-1, 0], dtype=torch.int32), torch.arange(2), 2,
        "rope_quant_insert", "layer_02")
    assert len(lines) == 1
    assert not [line for line in lines if "indexer_k_store" in line]


def test_the_budget_is_per_site_not_global(ctx, monkeypatch):
    """A busy consumer must not spend the budget of a quiet one."""
    _announcements(monkeypatch)
    slots = torch.tensor([-1, 0], dtype=torch.int32)
    for _ in range(compressed_slots.CHECK_LIMIT + 4):
        compressed_slots.check_per_token(slots, torch.arange(2), 2, "busy",
                                         "layer_02")
    dense = torch.tensor([0, -1], dtype=torch.int32)
    with pytest.raises(ValueError):
        compressed_slots.check_per_token(dense, torch.arange(2), 2, "quiet",
                                         "layer_02")


def test_a_spent_budget_stops_checking(ctx, monkeypatch):
    """Stated rather than discovered: past the budget the check is silent on
    failure too, because it reads index sets back to the host and cannot run
    every forward. The budget is what bounds that cost."""
    _announcements(monkeypatch)
    dense = torch.tensor([0, -1], dtype=torch.int32)
    for _ in range(compressed_slots.CHECK_LIMIT):
        with pytest.raises(ValueError):
            compressed_slots.check_per_token(dense, torch.arange(2), 2, "w",
                                             "layer_02")
    compressed_slots.check_per_token(dense, torch.arange(2), 2, "w", "layer_02")


# ------------------------------------------------------------- the identity

def test_a_sibling_groups_block_table_is_not_borrowed(ctx):
    """Every group has its own page and its own blocks, and a name or shape
    match would accept the wrong one with in-range arithmetic."""
    mine = torch.zeros(4, dtype=torch.int32)
    ctx({"swa": _Meta(_Req(torch.zeros(4, dtype=torch.int32), 32, ((1, ), ),
                           [0, 4]))})
    with pytest.raises(ValueError, match="not the slot mapping of"):
        compressed_slots.per_token_compressed_slots(mine, torch.arange(4), 2)


def test_no_metadata_at_all_is_refused_not_defaulted(ctx):
    ctx(None)
    with pytest.raises(ValueError, match="no attention metadata"):
        compressed_slots.per_token_compressed_slots(
            torch.zeros(2, dtype=torch.int32), torch.arange(2), 2)


def test_the_label_lookup_answers_rather_than_raises(ctx):
    """layer_owning only labels a log line; it must not turn a working write
    into a failed one."""
    ctx(None)
    assert compressed_slots.layer_owning(
        torch.zeros(2, dtype=torch.int32)) == "unattributed"


# ------------------------------------------------------- under graph capture

def test_the_check_stands_down_under_capture_without_spending_its_budget(
        ctx, monkeypatch):
    """Reading index sets to the host would end a capture.

    And standing down must not consume the budget: otherwise a captured
    warmup would use it up and the eager calls that follow -- the ones that
    can actually be checked -- would be silently skipped.
    """
    lines = _announcements(monkeypatch)
    dense = torch.tensor([0, -1], dtype=torch.int32)
    monkeypatch.setattr(compressed_slots, "_stream_is_capturing", lambda: True)
    for _ in range(compressed_slots.CHECK_LIMIT + 2):
        compressed_slots.check_per_token(dense, torch.arange(2), 2, "w", "l")
    assert lines == []

    monkeypatch.setattr(compressed_slots, "_stream_is_capturing", lambda: False)
    with pytest.raises(ValueError, match="not group boundaries"):
        compressed_slots.check_per_token(dense, torch.arange(2), 2, "w", "l")
