# SPDX-License-Identifier: Apache-2.0
"""Indices and their valid-entry count are one decision, not two.

The operator computes ``oriSparseBlockCount = hasOriTopkLength ?
Min(length, width) : width``. Omitting the length therefore declares every
slot in the row valid -- the ``-1`` padding included -- and the kernel gathers
at those addresses. Measured on the bench with exactly one variable moved: the
same index table crashes at the declared full width and passes at its true
length.

``Min`` reads like a guard and is not one. It clamps a length that exceeds how
wide the table is; nothing downstream knows how many entries were written, so
nothing can clamp to that. A default of "the whole row is valid" is the most
dangerous value available, which is what makes omitting it a defect rather
than an omission.
"""

import pathlib

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import (
    build_sparse_flash_mla_kwargs, ori_sparse_selection,
    select_ori_sparse_indices)

NUM_REQUESTS, NUM_TOKENS, NUM_HEADS, HEAD_DIM = 3, 12, 64, 512


class _Req:
    def __init__(self, vision=None, dspark=None, length=None):
        self.vision_swa_indices = vision
        self.dspark_swa_indices = dspark
        self.ori_topk_length = length


def _indices(rows=NUM_TOKENS, width=1152):
    return torch.zeros(rows, 1, width, dtype=torch.int32)


def _descriptor():
    blob = torch.zeros(1024, dtype=torch.int32)
    blob[0] = 28
    return blob


def _common(**overrides):
    kwargs = dict(
        q=torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_DIM),
        num_requests=NUM_REQUESTS,
        swa_kv_cache=torch.zeros(4, 32, 1, HEAD_DIM),
        swa_block_table=torch.zeros(NUM_REQUESTS, 4, dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 4, 9, 12], dtype=torch.int32),
        seqused_kv=torch.tensor([4, 5, 3], dtype=torch.int32),
        sinks=torch.zeros(NUM_HEADS, dtype=torch.float32),
        metadata=_descriptor(),
        softmax_scale=0.04419,
        layout_kv="PA_BBND",
        cmp_ratio=1,
        attends_compressed=False,
        ori_win_left=127,
        ori_win_right=0,
    )
    kwargs.update(overrides)
    return kwargs


# ------------------------------------------------------------- the two travel together

def test_indices_without_a_length_are_refused():
    """The exact configuration production was sending."""
    with pytest.raises(ValueError, match="without ori_topk_length"):
        build_sparse_flash_mla_kwargs(
            **_common(ori_sparse_indices=_indices()))


def test_the_refusal_says_why_the_default_is_not_safe():
    """A message that only says "missing argument" invites adding None."""
    with pytest.raises(ValueError, match="most\n?\\s*dangerous one"):
        build_sparse_flash_mla_kwargs(
            **_common(ori_sparse_indices=_indices()))


def test_both_reach_the_call_when_both_are_given():
    lengths = torch.full((NUM_TOKENS,), 125, dtype=torch.int32)
    built = build_sparse_flash_mla_kwargs(
        **_common(ori_sparse_indices=_indices(), ori_topk_length=lengths))
    assert built["ori_sparse_indices"] is not None
    torch.testing.assert_close(built["ori_topk_length"], lengths)


def test_a_length_without_indices_is_not_sent():
    """Nothing to describe, so nothing is declared. The operator would read a
    length against a table it was not given."""
    built = build_sparse_flash_mla_kwargs(
        **_common(ori_topk_length=torch.ones(NUM_TOKENS, dtype=torch.int32)))
    assert "ori_topk_length" not in built
    assert "ori_sparse_indices" not in built


# -------------------------------------------------------------- the selection

def test_the_selection_returns_the_length_beside_its_own_table():
    vision, dspark = _indices(), _indices()
    v_len = torch.full((NUM_TOKENS,), 7, dtype=torch.int32)
    got_idx, got_len = ori_sparse_selection(False, _Req(vision=vision,
                                                        length=v_len))
    assert got_idx is vision
    assert got_len is v_len


def test_the_pair_follows_the_same_precedence_as_the_indices_alone():
    """Otherwise the length could describe the table that was not chosen."""
    vision, dspark = _indices(), _indices(width=8)
    req = _Req(vision=vision, dspark=dspark,
               length=torch.ones(NUM_TOKENS, dtype=torch.int32))
    for attends in (False, True):
        idx, _ = ori_sparse_selection(attends, req)
        assert idx is select_ori_sparse_indices(attends, vision, dspark)


def test_a_request_with_no_indices_yields_no_length():
    idx, length = ori_sparse_selection(False, _Req())
    assert idx is None and length is None


def test_a_metadata_object_predating_the_field_does_not_crash_the_selection():
    """getattr, not attribute access: an older metadata object should reach
    the explicit refusal in the builder rather than an AttributeError several
    frames from anything that mentions sparse indices."""

    class _Old:
        vision_swa_indices = _indices()
        dspark_swa_indices = None

    idx, length = ori_sparse_selection(False, _Old())
    assert idx is not None and length is None


# ----------------------------------------------------- where the length comes from

def test_the_builders_still_return_the_count_this_carries():
    """Derived from the builders' sources.

    Both already computed the per-row valid count and both discarded it at the
    call site. If either stops returning it, this file's whole premise --
    "carry a number that exists" -- becomes "derive one here", which is the
    second source the design exists to avoid.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1

    for fn in (dsa_v1.build_vision_bidirectional_swa_indices,
               dsa_v1.build_dspark_swa_indices):
        source = inspect.getsource(fn)
        returns = [ln for ln in source.splitlines()
                   if ln.strip().startswith("return ")]
        assert returns, fn.__name__
        assert "," in returns[-1], (
            f"{fn.__name__} no longer returns a second value; the valid-entry "
            "count it used to supply now has to come from somewhere else")


#: Builders whose second return value is the per-row valid-entry count.
_COUNT_RETURNING_BUILDERS = ("build_dspark_swa_indices",
                             "build_vision_bidirectional_swa_indices")


def _discarding_call_sites():
    """Every place in vllm_ascend that throws away one of those counts.

    Enumerated from the AST across the whole package, not read out of one
    class. The first version of this check called ``inspect.getsource`` on
    ``AscendDSAMetadataBuilder`` and asserted the pattern was absent from it.
    It passed -- and a fourth call site in
    ``attention/context_parallel/dsa_cp.py`` was discarding the same value
    from the same function with the same spelling, in a different class in a
    different file. The check was not wrong about what it looked at; its scope
    was narrower than the defect's and it said nothing about that.

    Returns ``(sites, files_scanned, calls_seen)`` so the caller can fail on
    zero coverage. A scan that parses nothing reports no violations, which is
    indistinguishable from a clean tree.
    """
    import ast

    root = pathlib.Path(__file__).resolve().parents[3] / "vllm_ascend"
    sites, files_scanned, calls_seen = [], 0, 0
    for path in root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        files_scanned += 1
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id in _COUNT_RETURNING_BUILDERS):
                continue
            calls_seen += 1
            for target in node.targets:
                if not isinstance(target, ast.Tuple):
                    continue
                for i, element in enumerate(target.elts):
                    if isinstance(element, ast.Name) and element.id == "_":
                        sites.append(
                            f"{path.relative_to(root)}:{node.lineno} "
                            f"{call.func.id} return[{i}]")
    return sites, files_scanned, calls_seen


def test_no_call_site_discards_the_valid_entry_count():
    """`x, _ = build_...` is how the count was lost for as long as it existed.

    ⚠️ This still only catches the tuple-unpack spelling. `res = build_(...)`
    followed by `res[0]` discards the same value and is invisible here. The
    gap is written down rather than papered over -- but the scope is now the
    package, which is the half that actually went wrong.
    """
    sites, files_scanned, calls_seen = _discarding_call_sites()
    assert files_scanned > 100 and calls_seen >= 4, (
        f"coverage collapsed: {files_scanned} files parsed, {calls_seen} "
        "calls to the count-returning builders found. A scan that sees "
        "nothing reports no violations and looks exactly like a clean tree.")
    assert not sites, (
        "these call sites discard the per-row valid-entry count, which the "
        "operator needs or it declares the whole row valid: " + ", ".join(sites))


# ------------------------------------------- what makes the V4.0 change safe

def _vision_table(window=4, max_image=8, page=16, blocks=8, seq=10,
                  spans=None):
    from vllm_ascend.attention.dsa_v1 import \
        build_vision_bidirectional_swa_indices

    block_table = torch.arange(blocks, dtype=torch.int32).repeat(1, 1)
    return build_vision_bidirectional_swa_indices(
        block_table=block_table,
        window_size=window,
        max_image_tokens=max_image,
        block_size=page,
        query_start_loc=torch.tensor([0, seq], dtype=torch.int32),
        seq_lens=torch.tensor([seq], dtype=torch.int32),
        mm_prefix_ranges=spans if spans is not None else {0: []},
        num_tokens=seq,
    )


def _dspark_table(window=4, spec=2, page=16, blocks=8, seq=6):
    from vllm_ascend.attention.dsa_v1 import build_dspark_swa_indices

    block_table = torch.arange(blocks, dtype=torch.int32).repeat(1, 1)
    return build_dspark_swa_indices(
        block_table,
        spec,
        window,
        page,
        torch.tensor([0, seq], dtype=torch.int32),
        torch.tensor([seq], dtype=torch.int32),
        seq,
    )


@pytest.mark.parametrize("build", [_vision_table, _dspark_table],
                         ids=["vision", "dspark"])
def test_the_length_is_exactly_the_count_of_real_entries(build):
    """The invariant the whole rule rests on.

    Not "roughly the valid count" and not "at most the width": the operator
    reads this number as how many leading entries to gather, so it has to be
    the count of entries that are not -1. Slots are non-negative, so the two
    are the same statement.
    """
    indices, lengths = build()
    rows = indices.reshape(indices.shape[0], -1)
    counted = (rows != -1).sum(dim=-1).to(lengths.dtype)
    torch.testing.assert_close(lengths, counted)


@pytest.mark.parametrize("build", [_vision_table, _dspark_table],
                         ids=["vision", "dspark"])
def test_the_old_default_was_wrong_on_every_row_of_these_tables(build):
    """Why changing the shared V4.0 call site is not a gamble.

    The argument for touching a path this line does not exercise was: the new
    behaviour differs from the old only where the old one was already wrong,
    because a table with no padding has count == width and the two agree.

    That argument is weaker than it needs to be here, and the measurement says
    so: neither builder can produce a padding-free row at realistic sizes. The
    vision width is window + max_image_tokens while a row's span is bounded by
    the larger of the two, and the dspark width is rounded up to a multiple of
    128. So on these tables the old default -- "the whole row is valid" -- was
    wrong on *every* row, and the change cannot be a regression on any of
    them. If this ever goes red, a padding-free table has become possible and
    the no-op half of the argument has to carry the weight again.
    """
    indices, lengths = build()
    width = indices.reshape(indices.shape[0], -1).shape[-1]
    assert int(lengths.max()) < width, (
        f"a row now fills all {width} columns; the old full-width default "
        "would have been correct for it, so the reasoning about the shared "
        "V4.0 path needs re-reading rather than this assertion relaxing")


def test_a_padding_free_row_would_have_declared_the_full_width():
    """The no-op half, stated directly rather than left implicit.

    Constructed rather than found in a builder, because the builders cannot
    produce one: this pins what the rule does in that case, which is nothing.
    """
    row = torch.arange(6, dtype=torch.int32).reshape(1, 6)
    count = int((row != -1).sum())
    assert count == row.shape[-1], (
        "with no -1 the true length is the full width, which is exactly what "
        "the operator assumes when no length is passed -- so supplying it "
        "changes nothing for such a table")


# ------------------------------- the compressed lengths, and which one they are

def test_the_compressed_lengths_are_sent_when_the_caller_has_them():
    lengths = torch.tensor([2, 3, 1], dtype=torch.int32)
    built = build_sparse_flash_mla_kwargs(
        **_common(attends_compressed=True, cmp_ratio=2,
                  cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
                  seqused_cmp_kv=lengths))
    torch.testing.assert_close(built["seqused_cmp_kv"], lengths)


def test_a_cumulative_array_is_refused_where_lengths_are_wanted():
    """cu_cmp_seqlen_list has num_requests + 1 entries and is what V4.0 sends
    as cu_seqlens_cmp_kv under layout TND. Passing it here would be accepted
    by the operator and read as lengths: same dtype, same device, plausible
    magnitudes. The count is the only cheap discriminator."""
    cumulative = torch.tensor([0, 2, 5, 6], dtype=torch.int32)
    assert cumulative.numel() == NUM_REQUESTS + 1
    with pytest.raises(ValueError, match="cumulative"):
        build_sparse_flash_mla_kwargs(
            **_common(attends_compressed=True, cmp_ratio=2,
                      cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
                      seqused_cmp_kv=cumulative))


def test_absent_lengths_pass_when_the_adapter_can_derive_them():
    """The precheck runs on pre-adapter kwargs.

    The bf16 plan's operator is the adapter, which fills the compressed
    lengths in itself when they are absent. Demanding them unconditionally
    rejected a call that would have completed -- the first smoke run to get
    past layer 1 died exactly there.
    """
    built = build_sparse_flash_mla_kwargs(
        **_common(attends_compressed=True, cmp_ratio=2,
                  cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM)))
    assert "seqused_cmp_kv" not in built


def test_the_two_conditions_the_adapter_needs_are_enforced_elsewhere():
    """Why there is no "requires seqused_cmp_kv" rule any more.

    Letting the adapter derive them is safe only while it has something to
    derive from: a positive cmp_ratio and a seqused_kv. Both are guaranteed
    here -- the first by an explicit rule, the second by the signature -- so a
    rule of my own would be unreachable, and an unreachable rule reads as
    coverage. These two assertions are what keep that true.
    """
    import inspect

    with pytest.raises(ValueError, match="cmp_ratio must be greater than 0"):
        build_sparse_flash_mla_kwargs(
            **_common(attends_compressed=True, cmp_ratio=0,
                      cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM)))

    signature = inspect.signature(build_sparse_flash_mla_kwargs)
    empty = inspect.Parameter.empty
    assert signature.parameters["seqused_kv"].default is empty, (
        "seqused_kv became optional; the adapter can no longer be relied on "
        "to derive the compressed lengths, so the rule removed above has to "
        "come back")


def test_the_builder_publishes_the_length_the_descriptor_computed():
    """One tensor, not two equal computations.

    dsa_v1 computes seq_lens // ratio for the descriptor; it now publishes
    that same object on the req metadata, so the descriptor and the call
    cannot hold different numbers.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1

    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder.build_req_metadata)
    assert 'sas_mask_kwargs["seqused_cmp_kv"] = seqused_cmp_kv' in source
    assert "seqused_cmp_kv=seqused_cmp_kv," in source


# ------------------------------------------------- cmp_sparse_indices' rank

def test_the_compressed_index_list_must_carry_the_kv_head_axis():
    """[T, KV_N, K2] under layout_q=TND.

    Upstream keeps topk_indices_buffer as [tokens, index_topk] because its own
    consumer reads it that way. The operator's rank is not a property of that
    buffer, and sending it unchanged is what #65 did: the refusal came back
    truncated -- "has inc[orrect shape" -- with no expected shape in it, so
    the launch cost a round trip and yielded no number.
    """
    two_d = torch.zeros(NUM_TOKENS, 512, dtype=torch.int32)
    with pytest.raises(ValueError, match="KV-head axis"):
        build_sparse_flash_mla_kwargs(
            **_common(attends_compressed=True, cmp_ratio=2,
                      cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
                      seqused_cmp_kv=torch.ones(NUM_REQUESTS, dtype=torch.int32),
                      cmp_sparse_indices=two_d))


def test_the_three_dimensional_form_is_accepted():
    three_d = torch.zeros(NUM_TOKENS, 1, 512, dtype=torch.int32)
    built = build_sparse_flash_mla_kwargs(
        **_common(attends_compressed=True, cmp_ratio=2,
                  cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
                  seqused_cmp_kv=torch.ones(NUM_REQUESTS, dtype=torch.int32),
                  cmp_sparse_indices=three_d,
                  cmp_topk_length=torch.full((NUM_TOKENS,), 512,
                                             dtype=torch.int32)))
    assert built["cmp_sparse_indices"].shape == (NUM_TOKENS, 1, 512)
    assert built["cmp_topk_length"].numel() == NUM_TOKENS


def test_compressed_indices_without_their_length_are_refused():
    """The same pairing the ori side enforces, and the same mechanism: the
    selection buffer is -1 padded, so declaring the whole row valid sends the
    kernel to gather at the padding. Measured single-variable on the ori
    side -- identical table, crashes at the full width, passes at its true
    length."""
    with pytest.raises(ValueError, match="without cmp_topk_length"):
        build_sparse_flash_mla_kwargs(
            **_common(attends_compressed=True, cmp_ratio=2,
                      cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
                      seqused_cmp_kv=torch.ones(NUM_REQUESTS, dtype=torch.int32),
                      cmp_sparse_indices=torch.zeros(NUM_TOKENS, 1, 512,
                                                     dtype=torch.int32)))


def test_the_length_is_counted_from_the_tensor_that_is_sent():
    """The one case where counting at the consumer is not a second source.

    It cannot disagree with the table it describes, because it *is* that
    table. The first version published a count from the indexer instead --
    right principle, wrong scope: topk_indices_buffer is one buffer shared by
    every layer while self.indexer is None on every layer that is not an index
    source, so consumers read shared indices and found no length.
    """
    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    selection = torch.full((4, 1, 8), -1, dtype=torch.int32)
    selection[0, 0, :3] = torch.tensor([5, 6, 7])
    selection[1, 0, :8] = torch.arange(8)
    lengths = AscendDeepseekV41Attention._compressed_topk_length(selection)
    assert lengths.tolist() == [[3], [8], [0], [0]]
    assert lengths.dtype == torch.int32


def test_the_indices_and_their_length_describe_one_object():
    """Derived from the source: two expressions producing "the selection" is
    how the pair comes apart, so it is bound once."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    source = inspect.getsource(AscendDeepseekV41Attention.forward_mqa)
    assert "cmp_selection = (" in source
    assert "cmp_sparse_indices=cmp_selection," in source
    assert "self._compressed_topk_length(cmp_selection)" in source
    assert source.count("self.topk_indices_buffer[:q.shape[0]]") == 1, (
        "the selection is built in more than one place again")


def test_a_row_count_that_is_not_the_token_count_is_refused():
    """Right rank, wrong length: the slice bound is the other thing that goes
    wrong here, and rank alone would not see it."""
    with pytest.raises(ValueError, match="one row per query token"):
        build_sparse_flash_mla_kwargs(
            **_common(attends_compressed=True, cmp_ratio=2,
                      cmp_kv=torch.zeros(4, 8, 1, HEAD_DIM),
                      seqused_cmp_kv=torch.ones(NUM_REQUESTS, dtype=torch.int32),
                      cmp_sparse_indices=torch.zeros(NUM_TOKENS + 3, 1, 512,
                                                     dtype=torch.int32)))


def test_the_call_site_adds_the_axis_rather_than_reshaping_the_buffer():
    """Derived from the source: the buffer is shared with upstream's own
    consumer, so the axis belongs at the call, not in the buffer."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    source = inspect.getsource(AscendDeepseekV41Attention.forward_mqa)
    assert "self.topk_indices_buffer[:q.shape[0]].unsqueeze(1)" in source


# ------------------------------------------- every KV argument, not the last one

def test_a_compressed_cache_one_axis_short_is_refused():
    """"Cmp_kv strideVec size[3] not match kvLayout expect len[4]".

    Third parameter of this call to arrive missing the same unit head axis --
    after indexer_compress_cache and cmp_sparse_indices. The operator happened
    to say so clearly this time; one launch earlier the same class of message
    came back truncated.
    """
    with pytest.raises(ValueError, match="cmp_kv must have 4 axes"):
        build_sparse_flash_mla_kwargs(
            **_common(attends_compressed=True, cmp_ratio=2,
                      cmp_kv=torch.zeros(4, 8, HEAD_DIM),     # 3-D
                      seqused_cmp_kv=torch.ones(NUM_REQUESTS, dtype=torch.int32)))


def test_the_original_cache_is_held_to_the_same_rule():
    """The guard covers every KV argument, not the one that failed last."""
    with pytest.raises(ValueError, match="ori_kv must have 4 axes"):
        build_sparse_flash_mla_kwargs(
            **_common(swa_kv_cache=torch.zeros(4, 32, HEAD_DIM)))


def test_both_caches_go_through_the_same_rank_helper():
    """Derived from the source. The helper existed and one of the two callers
    used it, which is how cmp_kv arrived short."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    source = inspect.getsource(AscendDeepseekV41Attention.forward_mqa)
    assert source.count("kv_with_layout_rank(") == 2, (
        "one of the KV cache arguments is not going through the rank helper")


def test_the_helper_is_no_longer_documented_as_being_for_one_parameter():
    from vllm_ascend.models.deepseek_v4_1.sparse_attn_args import \
        kv_with_layout_rank

    doc = " ".join(kv_with_layout_rank.__doc__.split())
    assert "a helper that one of two callers uses" in doc


# --------------------------------- the compressed cache's own block table

class _Layer:
    """Just enough of the attention layer to exercise the lookup."""

    prefix = "layers.2.attn"

    def __init__(self, is_kv_source, compressed_prefix=None):
        self.is_kv_source = is_kv_source
        self.compressed_cache_prefix = compressed_prefix

    _compressed_group_metadata = None  # bound below


def _bind():
    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    _Layer._compressed_group_metadata = \
        AscendDeepseekV41Attention._compressed_group_metadata
    return _Layer


def _meta(block_table, descriptor=None):
    return type("M", (), {"req_metadata": type("R", (), {
        "block_table": block_table, "sas_metadata": descriptor})()})()


def test_a_consumer_layer_uses_its_sources_block_table():
    """Not its own. The compressed cache belongs to another layer's group, and
    the two tables differ -- addressing one cache with the other's ids gives
    blocks that are in range and wrong."""
    layer = _bind()(is_kv_source=False, compressed_prefix="layers.0.attn")
    mine = torch.zeros(3, 4, dtype=torch.int32)
    theirs = torch.ones(3, 4, dtype=torch.int32)
    got = layer._compressed_group_metadata(
        {"layers.2.attn": _meta(mine), "layers.0.attn": _meta(theirs)})
    assert got.block_table is theirs


def test_a_kv_source_layer_uses_its_own():
    layer = _bind()(is_kv_source=True)
    mine = torch.zeros(3, 4, dtype=torch.int32)
    got = layer._compressed_group_metadata({"layers.2.attn": _meta(mine)})
    assert got.block_table is mine


def test_a_missing_group_is_refused_rather_than_falling_back():
    """The fallback is the defect: this layer's own table is always available
    and always wrong for that cache."""
    layer = _bind()(is_kv_source=False, compressed_prefix="layers.0.attn")
    with pytest.raises(ValueError, match="both well-formed, both wrong"):
        layer._compressed_group_metadata(
            {"layers.2.attn": _meta(torch.zeros(1))})


def test_a_consumer_with_no_source_prefix_is_refused():
    layer = _bind()(is_kv_source=False, compressed_prefix=None)
    with pytest.raises(ValueError, match="no compressed_cache_prefix"):
        layer._compressed_group_metadata({})


def test_the_call_site_does_not_reuse_the_swa_table():
    """Derived from the source: `cmp_block_table = req.block_table` is the
    defect, and `req` is the SWA group's metadata."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    source = inspect.getsource(AscendDeepseekV41Attention.forward_mqa)
    assert "cmp_block_table = req.block_table" not in source
    assert "cmp_req = self._compressed_group_metadata(" in source


def test_the_descriptor_comes_from_the_compressed_group_when_there_is_one():
    """The kernel takes its tiling from this. The SWA group's spec reports no
    compression, so its descriptor is built at ratio 1 -- well-formed, and the
    wrong tiling for a ratio-2 layer. V4.0 makes the same distinction
    (dsa_v1.py:2574-2580, common_attn_metadata vs swa_metadata)."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    source = inspect.getsource(AscendDeepseekV41Attention.forward_mqa)
    assert "metadata=(cmp_req or req).sas_metadata," in source, (
        "the descriptor is being taken from the SWA group unconditionally")
    assert "metadata=req.sas_metadata," not in source


def test_every_compressed_argument_comes_from_the_compressed_group():
    """Four arguments describe the compressed stream; all four must come from
    the group that owns it. Sourcing any of them from the SWA group is not a
    shape error and nothing downstream can see it."""
    import inspect

    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    source = inspect.getsource(AscendDeepseekV41Attention.forward_mqa)
    for arg in ("cmp_block_table = cmp_req.block_table",
                "cmp_req.seqused_cmp_kv",
                "cmp_req.qli_cmp_residual_k"):
        assert arg in source, arg
    for wrong in ("req.seqused_cmp_kv if attends",
                  "req.qli_cmp_residual_k if attends"):
        assert wrong not in source, wrong
