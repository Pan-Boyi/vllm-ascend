# SPDX-License-Identifier: Apache-2.0
"""Two consumers of one slot mapping want two different shapes.

The SWA cache insert takes ``(block, offset)`` -- the A5 bf16 default -- and
``indexer_compress_epilog_v2`` takes flat slots and divides by the page size
itself. The indexer's mapping is its *own* group's, addressed in its own cache
(upstream's ``_produce_k`` reads ``attn_metadata[self.k_cache.prefix]`` and
writes ``self.k_cache.kv_cache``), so this is a format mismatch and not a
wrong-object one. Which is exactly why the conversion has to be careful: a
``(block, offset)`` pair is valid arithmetic under *any* page size, and the
wrong one produces in-range slots belonging to other tokens.

So the page size is matched by object identity against the metadata carrying
the tensor, and a failure to match is a refusal rather than a default.
"""

import pytest
import torch

from vllm_ascend.attention.dsa_attn_kv_plan import DsaAttnKvPlan
from vllm_ascend.models.deepseek_v4_1 import indexer_k_store

PAGE = 32


class _Req:
    def __init__(self, mapping, page=PAGE):
        self.slot_mapping = mapping
        self.token_slot_mapping = mapping
        self.storage_block_size = page


class _Meta:
    def __init__(self, req):
        self.req_metadata = req


class _Ctx:
    def __init__(self, attn_metadata):
        self.attn_metadata = attn_metadata


@pytest.fixture
def ctx(monkeypatch):
    """Install a forward context whose metadata dict the test controls."""
    holder = {}

    def _install(attn_metadata):
        holder["ctx"] = _Ctx(attn_metadata)

    import vllm.forward_context as fc
    monkeypatch.setattr(fc, "get_forward_context",
                        lambda: holder.get("ctx", _Ctx(None)))
    return _install


def _bound_cache():
    """Rank 3, and the width the operator's own rule asks for.

    Not an arbitrary shape: the store now cross-checks the cache against x,
    so a fixture with a made-up width stops exercising the path it is here to
    exercise and starts exercising the guard.
    """
    from vllm_ascend.models.deepseek_v4_1.indexer_k_store import (
        INDEX_HEAD_DIM, SCALE_BYTES, SCALE_GROUP)

    width = INDEX_HEAD_DIM + SCALE_BYTES * -(-INDEX_HEAD_DIM // SCALE_GROUP)
    return torch.zeros(4, 32, width, dtype=torch.uint8)


def _split(flat, page=PAGE):
    """Produce the (block, offset) form the same way the plan does."""
    valid = flat >= 0
    invalid = torch.full_like(flat, -1)
    block = torch.where(valid, torch.div(flat, page, rounding_mode="floor"),
                        invalid)
    offset = torch.where(valid, flat % page, invalid)
    return torch.stack([block, offset], dim=-1).to(torch.int32)


# ------------------------------------------------------------- the inversion

def test_it_is_the_exact_inverse_of_the_plans_split(ctx):
    flat = torch.tensor([0, 31, 32, 4095, 5663], dtype=torch.int32)
    pairs = _split(flat)
    ctx({"indexer": _Meta(_Req(pairs))})
    torch.testing.assert_close(
        indexer_k_store.flatten_block_offset_slots(pairs), flat)


def test_the_sentinel_stays_one_value_and_does_not_become_arithmetic(ctx):
    """The plan writes -1 into *both* columns, so -1 * page + -1 would be
    -33 -- an ordinary negative that the store kernel's `slot == -1` test does
    not catch, and that its divide turns into a negative block it writes."""
    flat = torch.tensor([-1, 64, -1], dtype=torch.int32)
    pairs = _split(flat)
    assert pairs[0].tolist() == [-1, -1]
    ctx({"indexer": _Meta(_Req(pairs))})
    got = indexer_k_store.flatten_block_offset_slots(pairs)
    assert got.tolist() == [-1, 64, -1]


def test_the_split_this_inverts_is_still_the_plans_own(ctx):
    """Derived from the plan's source rather than reimplemented twice.

    If format_dsa_slot_mapping changes its arithmetic or its sentinel, the
    inverse here stops being one and nothing else would say so.
    """
    plan = object.__new__(DsaAttnKvPlan)
    object.__setattr__(plan, "requires_block_offset_slots", True)
    flat = torch.tensor([-1, 0, 7, 1000], dtype=torch.int32)
    pairs = plan.format_dsa_slot_mapping(flat, PAGE)
    ctx({"indexer": _Meta(_Req(pairs))})
    torch.testing.assert_close(
        indexer_k_store.flatten_block_offset_slots(pairs), flat)


# --------------------------------------------------------- the page size

def test_the_page_size_comes_from_the_metadata_holding_this_tensor(ctx):
    """Two groups, two page sizes, and the right one chosen by identity."""
    flat = torch.tensor([300], dtype=torch.int32)
    mine = _split(flat, page=PAGE)
    other = _split(flat, page=128)
    ctx({"swa": _Meta(_Req(other, page=128)),
         "indexer": _Meta(_Req(mine, page=PAGE))})
    torch.testing.assert_close(
        indexer_k_store.flatten_block_offset_slots(mine), flat)


def test_a_sibling_groups_page_size_is_not_borrowed(ctx):
    """The failure this guards. A name or shape match would accept it, and
    the resulting slots stay in range and belong to other tokens."""
    pairs = _split(torch.tensor([300], dtype=torch.int32))
    ctx({"swa": _Meta(_Req(torch.zeros(1, 2, dtype=torch.int32), page=128))})
    with pytest.raises(ValueError, match="not the slot mapping of"):
        indexer_k_store.flatten_block_offset_slots(pairs)


def test_no_metadata_at_all_is_refused_not_defaulted(ctx):
    ctx(None)
    pairs = _split(torch.tensor([300], dtype=torch.int32))
    with pytest.raises(ValueError, match="no attention metadata"):
        indexer_k_store.flatten_block_offset_slots(pairs)


def test_a_wrong_page_size_would_have_stayed_in_range(ctx):
    """Why identity and not a plausible constant: the wrong answer is not
    out of bounds, it is another token's slot."""
    flat = torch.tensor([300], dtype=torch.int32)
    pairs = _split(flat, page=PAGE)
    blocks, offsets = pairs[:, 0], pairs[:, 1]
    wrong = int(blocks[0]) * 128 + int(offsets[0])
    assert wrong != int(flat[0])
    assert 0 <= wrong < 55738 * 32, (
        "if the wrong page size gave an out-of-range slot, the hardware would "
        "catch this and the identity match would be belt and braces")


# ------------------------------------------------------- the store entry point

def test_the_store_accepts_the_block_offset_form_now(ctx, monkeypatch):
    """End to end through the entry point upstream calls."""
    flat = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    pairs = _split(flat)
    ctx({"indexer": _Meta(_Req(pairs))})

    seen = {}

    def _fake_epilog(*, indexer_compress_cache, x, slot_mapping, layout):
        seen["slots"] = slot_mapping.clone()

    monkeypatch.setattr(
        indexer_k_store.torch.ops._C_ascend, "indexer_compress_epilog_v2",
        _fake_epilog, raising=False)

    indexer_k_store.indexer_k_norm_rope_store_ascend(
        k_pre=torch.zeros(4, indexer_k_store.INDEX_HEAD_DIM),
        positions=torch.tensor([0, 1, 2, 3]),
        cos_sin_cache=torch.zeros(8, indexer_k_store.INDEX_ROPE_HEAD_DIM),
        rms_norm_weight=torch.ones(indexer_k_store.INDEX_HEAD_DIM),
        rms_norm_eps=1e-6,
        k_cache=_bound_cache(),
        kv_slot_mapping=pairs,
        compress_ratio=1,
        use_fp4_cache=False,
    )
    assert seen["slots"].ndim == 1
    assert seen["slots"].tolist() == [0, 1, 2, 3]


def test_a_three_column_mapping_is_still_refused(ctx):
    """Widening the accepted shapes must not widen it to anything 2-D."""
    with pytest.raises(ValueError, match="must be flat"):
        indexer_k_store.indexer_k_norm_rope_store_ascend(
            k_pre=torch.zeros(2, indexer_k_store.INDEX_HEAD_DIM),
            positions=torch.tensor([0, 1]),
            cos_sin_cache=torch.zeros(8, indexer_k_store.INDEX_ROPE_HEAD_DIM),
            rms_norm_weight=torch.ones(indexer_k_store.INDEX_HEAD_DIM),
            rms_norm_eps=1e-6,
            k_cache=_bound_cache(),
            kv_slot_mapping=torch.zeros(2, 3, dtype=torch.int32),
            compress_ratio=1,
            use_fp4_cache=False,
        )
