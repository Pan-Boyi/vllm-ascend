# SPDX-License-Identifier: Apache-2.0
"""The staging convention, and the two things that can silently break it.

The CPU writes rows in an order the device recomputes from its program id. If
those disagree, both sides still handle well-formed bytes -- they are merely
paired with the wrong token -- so nothing raises. That is the failure this
module is defined once to prevent, and these are its guards.
"""

import json
import os
from types import SimpleNamespace

import pytest

from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout
from vllm_ascend.models.deepseek_v4_1.engram_staging import (DEFAULT_STAGE_DIR,
                                                             staged_file_name)

CHECKPOINT = "/workspace/models/DeepSeek-V4.1-Flash"


def _real_config():
    with open(os.path.join(CHECKPOINT, "config.json")) as handle:
        return SimpleNamespace(**json.load(handle)["text_config"])


@pytest.fixture
def layout():
    return EngramLayout(n_hash_cols=24, dim=256, quant_block=32,
                        layer_ids=(1, 14),
                        num_embeddings=(384006168, 384016682))


def test_derived_shapes(layout):
    assert layout.n_scales == 8
    assert layout.row_bytes == 264


def test_row_index_is_token_major(layout):
    """Token-major, not head-major. Both orderings are dense and total, which
    is exactly why swapping them produces no error."""
    assert layout.row_index(0, 0) == 0
    assert layout.row_index(0, 23) == 23
    assert layout.row_index(1, 0) == 24
    assert layout.row_index(5, 7) == 127
    # every (token, col) pair maps to a distinct row, with no gaps
    seen = {layout.row_index(t, c) for t in range(4) for c in range(24)}
    assert seen == set(range(96))


def test_staging_sizes_match_the_measured_figures(layout):
    # 24 rows x 264 B per token per layer; both engram layers per step.
    assert layout.staging_bytes(1) * 2 == 12_672
    assert layout.staging_bytes(256) * 2 == 3_244_032
    assert layout.staging_bytes(8192) * 2 == 103_809_024
    assert layout.staging_shapes(256) == ((6144, 256), (6144, 8))


# -- guard 1: the TP=1 degeneracy this numbering relies on --------------------


def test_undivided_hash_columns_are_accepted(layout):
    layout.assert_head_sharding_is_degenerate(24, 24, 0)


@pytest.mark.parametrize("local,total,start", [
    (12, 24, 0),     # sharded across two ranks
    (12, 24, 12),    # ... and this is the second one
    (24, 24, 24),    # nonzero start with full width
    (8, 8, 0),       # a different column count entirely
])
def test_any_sharding_is_refused(layout, local, total, start):
    """Relying on a degenerate case is fine; relying on it silently is not.

    Under sharding row_index would keep returning valid row numbers for the
    wrong rows, which is indistinguishable from working.
    """
    with pytest.raises(ValueError, match="undivided hash columns"):
        layout.assert_head_sharding_is_degenerate(local, total, start)


# -- guard 2: config shapes against the files themselves ----------------------


@pytest.mark.skipif(not os.path.isfile(os.path.join(CHECKPOINT, "config.json")),
                    reason="checkpoint config not available")
def test_from_config_on_the_real_checkpoint():
    got = EngramLayout.from_config(_real_config())
    assert got.n_hash_cols == 24, "(max_ngram_size - 1) * n_heads = 3 * 8"
    assert got.dim == 256
    assert got.n_scales == 8
    assert got.layer_ids == (1, 14)


@pytest.mark.skipif(
    not os.path.isdir(DEFAULT_STAGE_DIR)
    or not os.path.isfile(os.path.join(CHECKPOINT, "config.json")),
    reason="engram tables are not staged on this machine")
def test_the_real_tables_agree_with_the_real_config():
    """The same fact by two routes: config shapes vs actual file sizes."""
    EngramLayout.from_config(_real_config()).validate_against_files(
        DEFAULT_STAGE_DIR, staged_file_name)


@pytest.mark.skipif(not os.path.isdir(DEFAULT_STAGE_DIR),
                    reason="engram tables are not staged on this machine")
def test_the_attention_head_dim_is_caught(layout):
    """head_dim is 256 for engram and 512 for attention in this model.

    Reading the wrong one makes every gathered row straddle two real rows,
    with no error anywhere -- so the check has to be against the bytes, and
    the message has to name the confusion.
    """
    wrong = EngramLayout(n_hash_cols=24, dim=512, quant_block=32,
                         layer_ids=(1,), num_embeddings=(384006168,))
    with pytest.raises(ValueError) as exc:
        wrong.validate_against_files(DEFAULT_STAGE_DIR, staged_file_name)
    msg = str(exc.value)
    assert "512" in msg and "256" in msg
    assert "attention" in msg


def test_a_missing_table_is_reported_not_skipped(tmp_path, layout):
    with pytest.raises(ValueError, match="missing at"):
        layout.validate_against_files(str(tmp_path), staged_file_name)


# -- the CPU gather, and the ordering hazard it introduces --------------------


def _synthetic_tables(layout, rows=512):
    """Tables whose contents encode their own row number, so a mispaired row
    is identifiable rather than merely different."""
    import numpy as np
    w = np.zeros((rows, layout.dim), dtype=np.uint8)
    s = np.zeros((rows, layout.n_scales), dtype=np.uint8)
    w[:, 0] = np.arange(rows) % 256
    w[:, 1] = (np.arange(rows) // 256) % 256
    s[:, 0] = np.arange(rows) % 251
    return w, s


def _empty_staging(layout, num_tokens):
    import numpy as np
    (wshape, sshape) = layout.staging_shapes(num_tokens)
    return (np.zeros(wshape, dtype=np.uint8), np.zeros(sshape, dtype=np.uint8))


def test_each_staging_row_holds_the_table_row_that_token_asked_for(layout):
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    num_tokens, rows = 7, 512
    wt, st = _synthetic_tables(layout, rows)
    rng = np.random.default_rng(3)
    # hash_ids[token][col] -> table row
    hash_ids = rng.integers(0, rows, size=(num_tokens, layout.n_hash_cols))

    row_ids = np.empty(num_tokens * layout.n_hash_cols, dtype=np.int64)
    for t in range(num_tokens):
        for c in range(layout.n_hash_cols):
            row_ids[layout.row_index(t, c)] = hash_ids[t, c]

    ow, os_ = _empty_staging(layout, num_tokens)
    gather_into_staging(wt, st, row_ids, ow, os_)

    for t in range(num_tokens):
        for c in range(layout.n_hash_cols):
            r = layout.row_index(t, c)
            want = hash_ids[t, c]
            assert np.array_equal(ow[r], wt[want]), (t, c)
            assert np.array_equal(os_[r], st[want]), (t, c)


def test_a_head_major_ordering_is_wrong_without_being_invalid(layout):
    """The failure this layout module exists to prevent.

    Head-major and token-major are both dense, total, in-range orderings. Swap
    them and every staging row still holds a real table row -- just the wrong
    one for that position. Nothing raises; the only symptom is accuracy.
    """
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    num_tokens, rows = 7, 512
    wt, st = _synthetic_tables(layout, rows)
    rng = np.random.default_rng(3)
    hash_ids = rng.integers(0, rows, size=(num_tokens, layout.n_hash_cols))

    def build(head_major):
        ids = np.empty(num_tokens * layout.n_hash_cols, dtype=np.int64)
        for t in range(num_tokens):
            for c in range(layout.n_hash_cols):
                r = (c * num_tokens + t) if head_major else layout.row_index(t, c)
                ids[r] = hash_ids[t, c]
        ow, os_ = _empty_staging(layout, num_tokens)
        gather_into_staging(wt, st, ids, ow, os_)
        return ow

    correct, swapped = build(False), build(True)
    assert not np.array_equal(correct, swapped), "the orderings must differ"
    # ... and every row of the wrong one is still a genuine table row
    for r in range(swapped.shape[0]):
        assert np.any(np.all(wt == swapped[r], axis=1)), (
            "a mispaired row is still well-formed data, which is exactly why "
            "nothing downstream can notice")


def test_threading_does_not_change_the_result(layout):
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    rows = 4096
    num_tokens = 2048                      # 49152 rows, over the threshold
    wt, st = _synthetic_tables(layout, rows)
    row_ids = np.random.default_rng(0).integers(
        0, rows, size=num_tokens * layout.n_hash_cols, dtype=np.int64)

    a_w, a_s = _empty_staging(layout, num_tokens)
    b_w, b_s = _empty_staging(layout, num_tokens)
    gather_into_staging(wt, st, row_ids, a_w, a_s, threads=1)
    gather_into_staging(wt, st, row_ids, b_w, b_s, threads=16)
    assert np.array_equal(a_w, b_w) and np.array_equal(a_s, b_s)


def test_a_staging_buffer_of_the_wrong_length_is_refused(layout):
    import numpy as np
    from vllm_ascend.models.deepseek_v4_1.engram_layout import \
        gather_into_staging

    wt, st = _synthetic_tables(layout, 256)
    ow, os_ = _empty_staging(layout, 4)
    row_ids = np.zeros(layout.n_hash_cols * 5, dtype=np.int64)   # 5 tokens
    with pytest.raises(ValueError, match="index each other positionally"):
        gather_into_staging(wt, st, row_ids, ow, os_)
