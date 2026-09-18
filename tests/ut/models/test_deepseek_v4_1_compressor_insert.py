# SPDX-License-Identifier: Apache-2.0
"""The third consumer of one slot mapping, and the page size two sources agree on.

Upstream's ``rope_quant_insert`` takes ``num_tokens = slot_mapping.numel()``
and its kernel derives block and offset from a flat slot. Handed the A5 bf16
``[T, 2]`` form it sees ``2T`` tokens and its own assertion fires -- correctly,
and with none of the three numbers in the message.

The conversion is not the interesting part. The page size is: a
``(block, offset)`` pair is valid arithmetic under any page size, and the
wrong one produces a slot that is in range and belongs to another token. Here
two independent sources must agree -- the metadata that carries the mapping,
and the cache the kernel will index -- and a disagreement is refused.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import compressor_insert

PAGE = 32
CACHE_DIM = 512


class _Req:
    """The fields the builder publishes, copied from AscendDSAReqMetadata.

    block_table and query_start_loc are here because the compressed-stream
    rebuild reads them (the builder publishes all four together). A
    double without them passes a rebuild that never runs, which is how the
    previous double let the dense mapping through.
    """

    def __init__(self, mapping, page=PAGE, block_table=None,
                 query_start_loc=None):
        self.slot_mapping = mapping
        self.token_slot_mapping = mapping
        self.storage_block_size = page
        rows = mapping.shape[0]
        self.block_table = (torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
                            if block_table is None else block_table)
        self.query_start_loc = (
            torch.tensor([0, rows], dtype=torch.int32)
            if query_start_loc is None else query_start_loc)


class _Meta:
    def __init__(self, req):
        self.req_metadata = req


class _Ctx:
    def __init__(self, attn_metadata):
        self.attn_metadata = attn_metadata


@pytest.fixture
def ctx(monkeypatch):
    holder = {}

    def _install(attn_metadata):
        holder["ctx"] = _Ctx(attn_metadata)

    import vllm.forward_context as fc
    monkeypatch.setattr(fc, "get_forward_context",
                        lambda: holder.get("ctx", _Ctx(None)))
    return _install


def _split(flat, page=PAGE):
    valid = flat >= 0
    invalid = torch.full_like(flat, -1)
    block = torch.where(valid, torch.div(flat, page, rounding_mode="floor"),
                        invalid)
    offset = torch.where(valid, flat % page, invalid)
    return torch.stack([block, offset], dim=-1).to(torch.int32)


def _cache(page=PAGE, blocks=8):
    return torch.zeros(blocks, page, CACHE_DIM, dtype=torch.bfloat16)


@pytest.fixture
def upstream(monkeypatch):
    """Capture what upstream would have been called with."""
    seen = {}

    def _fake(latent, positions, cos_sin_cache, kv_cache, slot_mapping,
              compress_ratio, fp8_scale=None):
        seen["slot_mapping"] = slot_mapping.clone()
        seen["compress_ratio"] = compress_ratio
        seen["fp8_scale"] = fp8_scale

    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as m
    monkeypatch.setattr(m, "rope_quant_insert", _fake)
    return seen


def _call(slot_mapping, kv_cache=None, ratio=2):
    compressor_insert.rope_quant_insert_ascend(
        torch.zeros(4, 512, dtype=torch.bfloat16),
        torch.arange(4),
        torch.zeros(8, 64),
        kv_cache if kv_cache is not None else _cache(),
        slot_mapping,
        ratio,
    )


# ------------------------------------------------------- the shape it now takes

def test_the_paired_form_reaches_upstream_flat(ctx, upstream):
    flat = torch.tensor([0, 31, 32, 255], dtype=torch.int32)
    pairs = _split(flat)
    ctx({"compressor": _Meta(_Req(pairs))})
    _call(pairs, ratio=1)
    torch.testing.assert_close(upstream["slot_mapping"], flat)


def test_the_token_count_upstream_derives_is_now_right(ctx, upstream):
    """The assertion that fired: num_tokens = slot_mapping.numel().

    With the paired form that is 2T, which is exactly why upstream refused.
    """
    pairs = _split(torch.arange(4, dtype=torch.int32))
    ctx({"compressor": _Meta(_Req(pairs))})
    assert pairs.numel() == 8
    _call(pairs, ratio=1)
    assert upstream["slot_mapping"].numel() == 4


def test_an_already_flat_mapping_is_passed_through(ctx, upstream):
    flat = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    ctx({"compressor": _Meta(_Req(flat))})
    _call(flat, ratio=1)
    torch.testing.assert_close(upstream["slot_mapping"], flat)


def test_everything_else_is_handed_through_untouched(ctx, upstream):
    """Only the mapping changes. The kernel choice, the dtype branches and
    every assertion stay upstream's."""
    pairs = _split(torch.tensor([0, 1], dtype=torch.int32))
    ctx({"compressor": _Meta(_Req(pairs))})
    _call(pairs, ratio=1)
    assert upstream["compress_ratio"] == 1
    assert upstream["fp8_scale"] is None


def test_the_sentinel_survives_the_flattening(ctx, upstream):
    flat = torch.tensor([-1, 64, -1], dtype=torch.int32)
    pairs = _split(flat)
    assert pairs[0].tolist() == [-1, -1]
    ctx({"compressor": _Meta(_Req(pairs))})
    _call(pairs, ratio=1)
    assert upstream["slot_mapping"].tolist() == [-1, 64, -1]


# ------------------------------------------------------------- the page size

def test_the_two_sources_must_agree(ctx, upstream):
    """A cache whose block length differs from the metadata's page is
    refused, because only one of them can be the number the kernel divides by
    and picking either silently lands rows on other tokens."""
    pairs = _split(torch.tensor([300], dtype=torch.int32))
    ctx({"compressor": _Meta(_Req(pairs, page=PAGE))})
    with pytest.raises(ValueError, match="real but wrong place"):
        _call(pairs, kv_cache=_cache(page=128))


def test_agreement_is_what_lets_it_through(ctx, upstream):
    pairs = _split(torch.tensor([300], dtype=torch.int32), page=128)
    ctx({"compressor": _Meta(_Req(pairs, page=128))})
    _call(pairs, kv_cache=_cache(page=128), ratio=1)
    assert upstream["slot_mapping"].tolist() == [300]


def test_a_wrong_page_size_would_have_stayed_in_range(ctx):
    """Why this is checked rather than assumed: the wrong answer is not out
    of bounds, it is another token's slot.

    Sized against the pool this actually runs on -- 55738 blocks, measured --
    rather than against the toy cache above, because "in range" is a claim
    about the real allocation. A toy pool would make the wrong slot land
    outside it and suggest the hardware would catch this, which it would not.
    """
    real_blocks, real_page = 55738, 128
    pairs = _split(torch.tensor([300], dtype=torch.int32), page=PAGE)
    wrong = int(pairs[0, 0]) * real_page + int(pairs[0, 1])
    assert wrong != 300
    assert 0 <= wrong < real_blocks * real_page, (
        "the wrong page size gives an out-of-range slot at this size, so the "
        "argument for checking rather than assuming needs different numbers")


# ----------------------------------------------------------------- the rebind

def test_a_missing_upstream_name_is_refused_loudly(monkeypatch):
    """If upstream stops importing one of them, the patch must fail rather
    than silently leave that consumer on the paired mapping."""
    import importlib

    module = importlib.import_module(compressor_insert._TARGET_MODULE)
    monkeypatch.delattr(module, compressor_insert._REBINDS[0][0])
    with pytest.raises(AttributeError, match="no longer imports"):
        compressor_insert.apply_compressor_insert_patch()


def test_apply_all_includes_it():
    """A patch nobody calls is a patch that is not applied."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import platform_patches

    source = inspect.getsource(platform_patches.apply_all)
    assert "apply_compressor_insert_patch" in source


# ------------------------------- every consumer, enumerated from upstream

def test_every_upstream_function_taking_a_slot_mapping_is_rebound():
    """Enumerated from upstream's source, not from the list in our module.

    The list is what was incomplete last time: rope_quant_insert was fixed and
    fused_save_compress_norm was not, so the paired mapping reached it and its
    own assert fired -- one launch later, at layer 19 instead of layer 2.

    A third function growing a slot_mapping parameter turns this red instead of
    failing at runtime.
    """
    import inspect

    from vllm.models.deepseek_v4_1.common.ops import fused_compress_quant_cache

    takes_mapping = {
        name for name, fn in vars(fused_compress_quant_cache).items()
        if not name.startswith("_") and inspect.isfunction(fn)
        and "slot_mapping" in inspect.signature(fn).parameters
    }
    assert takes_mapping, (
        "no upstream function takes a slot_mapping any more; this check just "
        "became vacuous")

    rebound = {upstream for upstream, _ in compressor_insert._REBINDS}
    assert takes_mapping <= rebound, (
        f"these upstream functions take a slot mapping and nothing converts "
        f"it for them: {sorted(takes_mapping - rebound)}")


def test_both_rebinds_apply_together():
    import importlib

    module = importlib.import_module(compressor_insert._TARGET_MODULE)
    originals = {name: getattr(module, name)
                 for name, _ in compressor_insert._REBINDS}
    try:
        from vllm.models.deepseek_v4_1.common.ops import \
            fused_compress_quant_cache as ops
        for name, _ in compressor_insert._REBINDS:
            setattr(module, name, getattr(ops, name))
        assert not compressor_insert.compressor_insert_patched()

        assert compressor_insert.apply_compressor_insert_patch() is True
        assert compressor_insert.compressor_insert_patched()
        assert compressor_insert.apply_compressor_insert_patch() is False
    finally:
        for name, original in originals.items():
            setattr(module, name, original)


def test_the_pooling_half_gets_a_flat_mapping(ctx, monkeypatch):
    seen = {}

    def _fake(kv_score, positions, state_cache, slot_mapping, *rest):
        seen["slot_mapping"] = slot_mapping.clone()

    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as m
    monkeypatch.setattr(m, "fused_save_compress_norm", _fake)

    # A ring mapping the reader could use: row == position % 32, block 3.
    flat = torch.tensor([96, 97, 98], dtype=torch.int32)
    pairs = _split(flat)
    ctx({"compressor": _Meta(_Req(pairs))})
    compressor_insert.fused_save_compress_norm_ascend(
        torch.zeros(3), torch.arange(3), _cache(), pairs,
        None, None, torch.ones(4), 1e-6, 2, torch.zeros(3, 4))
    torch.testing.assert_close(seen["slot_mapping"], flat)


def test_a_missing_state_cache_flattens_with_the_metadata_page(
        ctx, monkeypatch):
    """state_cache may be None; then the metadata's page is the only source."""
    seen = {}

    def _fake(kv_score, positions, state_cache, slot_mapping, *rest):
        seen["slot_mapping"] = slot_mapping.clone()

    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as m
    monkeypatch.setattr(m, "fused_save_compress_norm", _fake)

    flat = torch.tensor([64], dtype=torch.int32)
    pairs = _split(flat)
    ctx({"compressor": _Meta(_Req(pairs))})
    compressor_insert.fused_save_compress_norm_ascend(
        torch.zeros(1), torch.arange(1), None, pairs,
        None, None, torch.ones(4), 1e-6, 2, torch.zeros(1, 4))
    torch.testing.assert_close(seen["slot_mapping"], flat)


# ------------------------------------ the mapping the compressing path builds

def test_a_compressing_ratio_hands_upstream_a_per_token_mapping(ctx, upstream):
    """The defect, end to end.

    The builder publishes V4.0's dense mapping; upstream's kernel indexes it
    per token. Reverting this path to flattening turns this red: upstream
    would receive [10752, 10753, -1, 0, 0], which writes entry #1, never
    entry #0, and sends position 3 to flat slot 0.
    """
    dense = torch.tensor([10752, 10753, -1, 0, 0], dtype=torch.int32)
    req = _Req(dense, page=128,
               block_table=torch.tensor([[84, 85]], dtype=torch.int32),
               query_start_loc=torch.tensor([0, 5], dtype=torch.int32))
    ctx({"c2": _Meta(req)})
    compressor_insert.rope_quant_insert_ascend(
        torch.zeros(5, 512, dtype=torch.bfloat16),
        torch.arange(5),
        torch.zeros(8, 64),
        _cache(page=128),
        dense,
        2,
    )
    assert upstream["slot_mapping"].tolist() == [-1, 10752, -1, 10753, -1]


# ------------------------------------- the ring's capacity precondition

def test_a_ring_capacity_the_reader_would_read_across_is_refused(
        ctx, monkeypatch):
    """gcd(capacity, ratio) == 1 makes the ring reader read the previous
    position's row from the next block; it is refused before the operator."""
    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as m
    monkeypatch.setattr(m, "fused_save_compress_norm", lambda *a: None)

    from vllm_ascend.models.deepseek_v4_1 import compressed_slots
    compressed_slots._RING_SHAPES_CHECKED.clear()

    page = 3
    pairs = _split(torch.tensor([0, 1, 2], dtype=torch.int32), page=page)
    ctx({"compressor": _Meta(_Req(pairs, page=page))})
    with pytest.raises(ValueError, match="read the previous position's row"):
        compressor_insert.fused_save_compress_norm_ascend(
            torch.zeros(3), torch.arange(3), _cache(page=page), pairs,
            None, None, torch.ones(4), 1e-6, 2, torch.zeros(3, 4))
