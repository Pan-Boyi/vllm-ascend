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
    _call(pairs)
    torch.testing.assert_close(upstream["slot_mapping"], flat)


def test_the_token_count_upstream_derives_is_now_right(ctx, upstream):
    """The assertion that fired: num_tokens = slot_mapping.numel().

    With the paired form that is 2T, which is exactly why upstream refused.
    """
    pairs = _split(torch.arange(4, dtype=torch.int32))
    ctx({"compressor": _Meta(_Req(pairs))})
    assert pairs.numel() == 8
    _call(pairs)
    assert upstream["slot_mapping"].numel() == 4


def test_an_already_flat_mapping_is_passed_through(ctx, upstream):
    flat = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    ctx({"compressor": _Meta(_Req(flat))})
    _call(flat)
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
    _call(pairs)
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
    _call(pairs, kv_cache=_cache(page=128))
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
    costing a flight.
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

    flat = torch.tensor([0, 31, 32], dtype=torch.int32)
    pairs = _split(flat)
    ctx({"compressor": _Meta(_Req(pairs))})
    compressor_insert.fused_save_compress_norm_ascend(
        torch.zeros(3), torch.arange(3), _cache(), pairs,
        None, None, torch.ones(4), 1e-6, 2, torch.zeros(3, 4))
    torch.testing.assert_close(seen["slot_mapping"], flat)


def test_a_missing_state_cache_reports_that_the_page_was_not_cross_checked(
        ctx, monkeypatch):
    """state_cache may be None, and then only one source for the page size
    exists. Reported, not passed off as checked."""
    import logging

    from vllm.logger import logger as vllm_logger

    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as m
    monkeypatch.setattr(m, "fused_save_compress_norm", lambda *a: None)
    compressor_insert._LOGGED_UNCHECKED_PAGES.clear()

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    pairs = _split(torch.tensor([64], dtype=torch.int32))
    ctx({"compressor": _Meta(_Req(pairs))})
    handler = Capture()
    vllm_logger.addHandler(handler)
    try:
        compressor_insert.fused_save_compress_norm_ascend(
            torch.zeros(1), torch.arange(1), None, pairs,
            None, None, torch.ones(4), 1e-6, 2, torch.zeros(1, 4))
    finally:
        vllm_logger.removeHandler(handler)

    assert any("NOT" in m and "cross-checked" in m for m in records), records


# ---------------------------- the write side, for the read side to pair with

def test_the_write_side_is_recorded_before_the_operator_runs(monkeypatch):
    """🔴 The read side says the compressed stream's entry #0 comes back all
    zeros on every even-numbered prefill while entry #1 is correct. That is
    consistent with "never written" and with "written elsewhere", and only
    the write side separates them -- taken before the call, because after it
    the question has changed."""
    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    order = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.alignment_active",
        lambda: True)
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.read_taps.read_tap_layers",
        lambda: (2, ))
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda layer, key, tensor: order.append(key))
    ci.reset_write_taps_for_test()

    ci._tap_write_side("rope_quant_insert", torch.tensor([5, 6]),
                       torch.zeros(2, 4), torch.zeros(4, 4))
    assert order == ["compressor.rope_quant_insert.00.slots",
                     "compressor.rope_quant_insert.00.latent",
                     "compressor.rope_quant_insert.00.cache_ptr"]


def test_calls_are_numbered_so_a_forward_can_be_ordered(monkeypatch):
    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    keys = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.alignment_active",
        lambda: True)
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.read_taps.read_tap_layers",
        lambda: (2, ))
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda layer, key, tensor: keys.append(key))
    ci.reset_write_taps_for_test()

    for _ in range(2):
        ci._tap_write_side("rope_quant_insert", torch.tensor([5]),
                           torch.zeros(1, 4), torch.zeros(4, 4))
    assert any(".01.slots" in k for k in keys), keys


def test_it_is_silent_when_nothing_is_being_recorded(monkeypatch):
    """Two arms: an instrument that runs when nobody asked is a cost with no
    reader."""
    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    keys = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.alignment_active",
        lambda: False)
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda layer, key, tensor: keys.append(key))
    ci.reset_write_taps_for_test()
    ci._tap_write_side("rope_quant_insert", torch.tensor([5]),
                       torch.zeros(1, 4), torch.zeros(4, 4))
    assert keys == []


def test_a_failing_write_tap_does_not_reach_the_operator(monkeypatch):
    """⚠️ It has already cost one flight to let an instrument propagate."""
    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    def _explode(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.alignment_active",
        lambda: True)
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.read_taps.read_tap_layers",
        lambda: (2, ))
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole", _explode)
    ci.reset_write_taps_for_test()
    ci._tap_write_side("rope_quant_insert", torch.tensor([5]),
                       torch.zeros(1, 4), torch.zeros(4, 4))


def test_the_slots_recorded_are_the_flattened_ones(monkeypatch):
    """The slots the operator is handed, not the (block, offset) pairs the
    caller had -- those are two different values and only one of them is
    what the write lands on."""
    import inspect

    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    body = inspect.getsource(ci.rope_quant_insert_ascend)
    assert body.index("flat = flatten_slot_mapping") < \
        body.index("_tap_write_side")
    assert "_tap_write_side(\"rope_quant_insert\", flat" in body
