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
    rebuild reads them (dsa_v1.py:1795-1800 publishes all four together). A
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

    # A ring mapping the reader could actually use: row == position % 32,
    # here block 3. The arbitrary [0, 31, 32] this used to carry is refused
    # now, and rightly -- the reader takes the row for a position as
    # (position % capacity), so consecutive-but-incongruent offsets land
    # where nothing will look. The subject here is the flattening, so the
    # data has to be valid in every other respect.
    flat = torch.tensor([96, 97, 98], dtype=torch.int32)
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
        "vllm_ascend.models.deepseek_v4_1.align_taps.recording_armed",
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
    assert order == ["read.compressor.rope_quant_insert.00.slots",
                     "read.compressor.rope_quant_insert.00.latent",
                     "read.compressor.rope_quant_insert.00.cache_id.process_local"]


def test_calls_are_numbered_so_a_forward_can_be_ordered(monkeypatch):
    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    keys = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.recording_armed",
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
        "vllm_ascend.models.deepseek_v4_1.align_taps.recording_armed",
        lambda: True)
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.read_taps.read_tap_layers",
        lambda: (2, ))
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole", _explode)
    ci.reset_write_taps_for_test()
    ci._tap_write_side("rope_quant_insert", torch.tensor([5]),
                       torch.zeros(1, 4), torch.zeros(4, 4))


def test_the_slots_recorded_are_the_ones_the_operator_gets():
    """The slots the operator is handed, not the mapping the caller had --
    those are two different values and only one is what the write lands on.

    Walked as an AST rather than matched as text: there are now two branches
    that produce the mapping, and a substring check would pass while one of
    them handed something else to the tap or to upstream.
    """
    import ast
    import inspect
    import textwrap

    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(ci.rope_quant_insert_ascend)))
    calls = {
        node.func.id: node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_tap_write_side" in calls and "rope_quant_insert" in calls

    def _mapping_argument(call, index):
        arg = call.args[index]
        assert isinstance(arg, ast.Name), ast.dump(arg)
        return arg.id

    recorded = _mapping_argument(calls["_tap_write_side"], 1)
    passed = _mapping_argument(calls["rope_quant_insert"], 4)
    assert recorded == passed == "flat"

    assigned = {
        target.id
        for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    } | {
        element.id
        for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Tuple)
        for element in target.elts if isinstance(element, ast.Name)
    }
    assert "flat" in assigned


# ------------------------------------ the mapping the compressing path builds

def test_a_compressing_ratio_hands_upstream_a_per_token_mapping(ctx, upstream):
    """🔴 The defect, end to end.

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


# ---------------------------------- process-local keys leave the comparison

def test_a_process_local_key_says_so_in_its_own_name():
    """🔴 Corrected, not tidied.

    These were hashed, with a comment claiming the hash let the four-rank
    same-state comparison report a clean zero. Hashing a pointer that differs
    per process gives a value that differs per process, so the comparison
    carried ten known differences every run and relied on the reader
    remembering to subtract them.

    A process-local identifier cannot be made comparable; it can only leave
    the comparison. The suffix says that in the key, so a comparator filters
    it mechanically -- an exceptions list is the other form of remembering,
    and it is the one that gets forgotten.
    """
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import attention, compressor_insert

    assert compressor_insert.PROCESS_LOCAL == ".process_local"
    for source in (inspect.getsource(compressor_insert._tap_write_side),
                   inspect.getsource(attention._tap_write_side_kv)):
        for line in source.splitlines():
            if "data_ptr()" in line or "cache_id" in line or "cache_ptr" in line:
                continue
        assert "PROCESS_LOCAL" in source, source

    # And nothing may publish a raw pointer under an unsuffixed key.
    for module in (compressor_insert, attention):
        text = inspect.getsource(module)
        assert '"write.ori_kv.cache_ptr"' not in text
        assert '.cache_id"' not in text


def test_the_ring_mapping_has_its_own_criterion(ctx, monkeypatch):
    """🔴 Not the per-token one, and the difference matters in both
    directions.

    The ring is written at every token, so `valid ⊆ boundary` would refuse a
    correct ring mapping. What it must satisfy is the addressing its reader
    assumes: the row for a position is (position % capacity) and nothing
    else, so an offset that is merely consecutive lands in a real row the
    reader will never look at.
    """
    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as m
    monkeypatch.setattr(m, "fused_save_compress_norm", lambda *a: None)

    from vllm_ascend.models.deepseek_v4_1 import compressed_slots
    compressed_slots.reset_for_test()
    monkeypatch.setattr(compressed_slots, "_stream_is_capturing", lambda: False)

    # Congruent: accepted, and every token valid -- which the per-token
    # criterion would have rejected as three non-boundary rows.
    pairs = _split(torch.tensor([96, 97, 98], dtype=torch.int32))
    ctx({"compressor": _Meta(_Req(pairs))})
    compressor_insert.fused_save_compress_norm_ascend(
        torch.zeros(3), torch.arange(3), _cache(), pairs,
        None, None, torch.ones(4), 1e-6, 2, torch.zeros(3, 4))

    # Shifted by one: still in range, still a real row, and caught.
    compressed_slots.reset_for_test()
    shifted = _split(torch.tensor([97, 98, 99], dtype=torch.int32))
    ctx({"compressor": _Meta(_Req(shifted))})
    monkeypatch.setenv("VLLM_ASCEND_DSV41_RING_CHECK_STRICT", "1")
    with pytest.raises(ValueError, match="not congruent to their positions"):
        compressor_insert.fused_save_compress_norm_ascend(
            torch.zeros(3), torch.arange(3), _cache(), shifted,
            None, None, torch.ones(4), 1e-6, 2, torch.zeros(3, 4))


def test_the_ring_criterion_reports_without_raising_until_it_is_armed(
        ctx, monkeypatch, caplog):
    """🔴 Priced for "has never executed here", not for "might be wrong".

    It fires on every forward including each decode step, and it has not run
    on the device; a flight has already been lost to a tap that had never
    flown. So a violation is loud and the run continues, until one flight
    logs the OK line from the card and the switch arms it.
    """
    import vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache as m
    monkeypatch.setattr(m, "fused_save_compress_norm", lambda *a: None)
    monkeypatch.delenv("VLLM_ASCEND_DSV41_RING_CHECK_STRICT", raising=False)

    from vllm_ascend.models.deepseek_v4_1 import compressed_slots
    compressed_slots.reset_for_test()
    monkeypatch.setattr(compressed_slots, "_stream_is_capturing", lambda: False)
    assert compressed_slots.ring_check_is_strict() is False

    shifted = _split(torch.tensor([97, 98, 99], dtype=torch.int32))
    ctx({"compressor": _Meta(_Req(shifted))})
    seen = []
    monkeypatch.setattr("vllm.logger.logger.error",
                        lambda fmt, *a: seen.append(fmt % a))
    # No raise, and the message says both what is wrong and that it is not
    # being enforced -- a silent downgrade would be worse than no criterion.
    compressor_insert.fused_save_compress_norm_ascend(
        torch.zeros(3), torch.arange(3), _cache(), shifted,
        None, None, torch.ones(4), 1e-6, 2, torch.zeros(3, 4))
    assert len(seen) == 1
    assert "not congruent to their positions" in seen[0]
    assert "NOT raising" in seen[0]
    assert "VLLM_ASCEND_DSV41_RING_CHECK_STRICT=1" in seen[0]


def test_the_ring_reader_count_is_reported(ctx, monkeypatch):
    """🔴 "How far does the reader read" presumes the reader runs.

    Upstream reads the ring only when a chunk's first token closes a group
    (fused_compress_quant_cache.py:148), so a request's first chunk -- which
    starts at position 0 -- never reads it. Counted rather than argued: 0
    readers means a forward carries zero evidence about the reader, which is
    not the same as evidence that it is fine.
    """
    from vllm_ascend.models.deepseek_v4_1 import compressed_slots

    positions = torch.tensor([0, 1, 2, 3, 4, 7, 8], dtype=torch.int64)
    qsl = torch.tensor([0, 5, 7], dtype=torch.int32)
    readers, requests = compressed_slots.ring_readers_this_forward(qsl,
                                                                   positions)
    # Request 0 starts at position 0 (even -> no read); request 1 starts at
    # position 7 (odd -> reads row 6).
    assert (readers, requests) == (1, 2)

    only_first_chunks = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
    assert compressed_slots.ring_readers_this_forward(
        torch.tensor([0, 5], dtype=torch.int32), only_first_chunks) == (0, 1)


# ------------------------- the call index is per forward, and under read.

def test_the_call_index_restarts_each_forward(monkeypatch):
    """🔴 It used to be a process-level counter, and it is part of a key name.

    Measured on flight-24: req_000 ended at index 3 and req_001 started at 48
    -- the same tensor carried a different name in two dumps, and a key-by-key
    comparison saw two unrelated keys instead of one pair. A name must not
    carry state that has nothing to do with what it names.
    """
    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    keys = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.recording_armed",
        lambda: True)
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.read_taps.read_tap_layers",
        lambda: (2, ))
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda layer, key, tensor: keys.append(key))
    ci.reset_write_taps_for_test()

    for _ in range(3):
        ci._tap_write_side("rope_quant_insert", torch.tensor([5]),
                           torch.zeros(1, 4), torch.zeros(4, 4))
    assert [k for k in keys if k.endswith(".slots")] == [
        "read.compressor.rope_quant_insert.00.slots",
        "read.compressor.rope_quant_insert.01.slots",
        "read.compressor.rope_quant_insert.02.slots"]

    # The next forward starts over, so the second request's keys match the
    # first's instead of continuing from wherever the process happened to be.
    keys.clear()
    ci.reset_write_call_index()
    ci._tap_write_side("rope_quant_insert", torch.tensor([5]),
                       torch.zeros(1, 4), torch.zeros(4, 4))
    assert keys[0] == "read.compressor.rope_quant_insert.00.slots"


def test_the_rotate_path_resets_the_index():
    """Wired to the same place the recorder rotates, not called by hand --
    a reset nobody calls is the same as no reset."""
    import ast
    import inspect
    import textwrap

    from vllm_ascend.models.deepseek_v4_1 import align_taps

    source = textwrap.dedent(inspect.getsource(align_taps._flush_and_rotate))
    called = {node.func.id for node in ast.walk(ast.parse(source))
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "reset_write_call_index" in called


def test_rows_that_were_never_written_are_reported(monkeypatch, caplog):
    """🔴 The latent is torch.empty and only group boundaries are written, so
    rows with a negative slot are uninitialised memory -- measured as the
    constant 5.0097551721019626e+33, which looks like a value and is not.

    Recorded whole rather than masked (the companion .slots says which to
    drop, and masking would hide which rows they were), but said out loud.
    """
    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    said = []
    monkeypatch.setattr("vllm.logger.logger.info",
                        lambda fmt, *a: said.append(fmt % a))
    ci.reset_write_taps_for_test()
    ci._report_uninitialised_rows(
        "rope_quant_insert", torch.tensor([10752, -1, 10753, -1, -1]),
        torch.zeros(5, 4))
    assert len(said) == 1 and "3 of 5 rows" in said[0]
    assert "uninitialised memory, not values" in said[0]

    # Once per writer: a per-call line would bury the flight log.
    ci._report_uninitialised_rows(
        "rope_quant_insert", torch.tensor([-1]), torch.zeros(1, 4))
    assert len(said) == 1

    # And nothing to say when every row was written.
    ci.reset_write_taps_for_test()
    said.clear()
    ci._report_uninitialised_rows(
        "rope_quant_insert", torch.tensor([1, 2]), torch.zeros(2, 4))
    assert said == []


def test_the_index_does_not_advance_on_an_unarmed_forward(monkeypatch):
    """🔴 The property that was missing, and the reason the fix did not take.

    The tap counted under alignment_active() -- the env switch, true on every
    decode step -- and reset under _ARMED, which is per forward. So 11 decode
    steps x 4 layers advanced the index while the reset never fired for them,
    and the same tensor arrived in two dumps under two names: 00..03 for one
    request and 44..47 for the next.

    ⚠️ The existing tests could not see this. They patched the gate the code
    was using, so they were green whichever gate that was. A test that names
    the condition it is standing in for cannot tell you the condition is the
    wrong one; only one that exercises the *other* state can.
    """
    import vllm_ascend.models.deepseek_v4_1.compressor_insert as ci

    keys = []
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.read_taps.read_tap_layers",
        lambda: (2, ))
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.tap_whole",
        lambda layer, key, tensor: keys.append(key))
    ci.reset_write_taps_for_test()

    # Unarmed: a decode step. Nothing recorded, and nothing counted.
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.recording_armed",
        lambda: False)
    for _ in range(44):
        ci._tap_write_side("rope_quant_insert", torch.tensor([5]),
                           torch.zeros(1, 4), torch.zeros(4, 4))
    assert keys == []

    # Armed: the next recorded forward starts at 00, not at 44.
    monkeypatch.setattr(
        "vllm_ascend.models.deepseek_v4_1.align_taps.recording_armed",
        lambda: True)
    ci._tap_write_side("rope_quant_insert", torch.tensor([5]),
                       torch.zeros(1, 4), torch.zeros(4, 4))
    assert keys[0] == "read.compressor.rope_quant_insert.00.slots"
