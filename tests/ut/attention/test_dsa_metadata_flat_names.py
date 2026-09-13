# SPDX-License-Identifier: Apache-2.0
"""The three names upstream's V4.1 indexer reads off the attention metadata.

Upstream reads them flat -- ``indexer_metadata.max_seq_len``,
``.slot_mapping``, ``.num_prefill_tokens`` -- from whatever object sits under
``attn_metadata[k_cache.prefix]``. On Ascend that object is AscendDSAMetadata,
which keeps the per-request payload one level down in ``req_metadata``. Two of
the three were simply absent and one had to be carried in from the builder, so
the V4.1 layer raised AttributeError before reaching any operator.

Pinned here rather than in the V4.1 tests because the class is shared: the
failure this prevents is someone tidying these away as unused, which they are
from inside this tree.
"""

import torch

from vllm_ascend.attention.dsa_v1 import AscendDSAMetadata


def _metadata(**kwargs):
    base = dict(num_actual_tokens=10, num_decodes=1, num_decode_tokens=2,
                num_prefills=1)
    base.update(kwargs)
    return AscendDSAMetadata(**base)


#: Every attribute upstream's V4.1 reads off the object under
#: attn_metadata[<swa prefix>]. Enumerated from the source rather than grown one
#: AttributeError at a time: the first two walls here were the same shape --
#: engram reading Ascend's metadata by upstream's field names -- and hitting
#: them one per launch costs a launch each.
#:
#: nvidia/model.py:583-595 (engram): num_decodes, num_prefills,
#:   query_start_loc, slot_mapping, block_table
#: attention.py:1153-1181 (indexer): slot_mapping, max_seq_len,
#:   num_decode_tokens, num_prefill_tokens
UPSTREAM_READS = ("num_decodes", "num_prefills", "num_decode_tokens",
                  "num_prefill_tokens", "max_seq_len", "slot_mapping",
                  "query_start_loc", "block_table", "block_size")


def test_upstream_reads_all_of_these_names():
    """The list itself, so a rename upstream is a red test and not a crash."""
    md = _metadata()
    missing = [n for n in UPSTREAM_READS if not hasattr(md, n)]
    assert not missing, f"upstream reads these and Ascend does not carry them: {missing}"


def test_block_size_is_the_logical_one_not_the_storage_page():
    """This was deliberately absent, on a premise that turned out to be false.

    The reasoning for leaving it out was sound -- storage_block_size equals the
    spec's block_size only at ratio 1, so a property returning it would be
    right for SWA and wrong by a factor of two for the ratio-2 groups. The
    premise was that nothing would reach it: its call sites are named
    _fused_qnorm_rope_kv_insert, which read as the CUDA prolog Ascend replaces.
    They are reached from _prepare_and_attn_eager. A function's name promised a
    path; the stack decides it.

    So the ambiguity is removed rather than worked around: the ratio is carried
    on the metadata and the logical block size reconstructed from it, which is
    correct for every group instead of for the ones where the two coincide.
    """
    class _Req:
        storage_block_size = 128
        slot_mapping = None

    assert _metadata(req_metadata=_Req(), tokens_per_state=1).block_size == 128
    assert _metadata(req_metadata=_Req(), tokens_per_state=2).block_size == 256


def test_block_size_is_none_without_a_payload():
    assert _metadata(req_metadata=None).block_size is None


def test_the_ratio_reaches_the_metadata_from_the_builder():
    """Both build sites must set it, or a group silently reports ratio 1.

    A missing one is not a crash: tokens_per_state defaults to 1 and the
    ratio-2 groups then report half their block size, which upstream's
    assert on the cache shape would catch -- on a card, at forward time.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1

    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder)
    assert source.count("tokens_per_state=max(self.compressor_ratio, 1)") == 2


def test_prefill_tokens_are_what_is_left_after_the_decodes():
    assert _metadata(num_actual_tokens=10, num_decode_tokens=2) \
        .num_prefill_tokens == 8


def test_slot_mapping_comes_from_the_request_payload():
    class _Req:
        slot_mapping = torch.arange(4, dtype=torch.int32)

    md = _metadata(req_metadata=_Req())
    assert torch.equal(md.slot_mapping, _Req.slot_mapping)


def test_slot_mapping_is_none_when_there_is_no_payload():
    """Profile and dummy runs build metadata with req_metadata unset.

    Returning None is what lets the caller's own guard fire; raising here would
    move the failure into a place with less context.
    """
    assert _metadata(req_metadata=None).slot_mapping is None


def test_max_seq_len_defaults_to_zero_rather_than_being_required():
    """Every existing construction site predates it.

    A required field would have turned an additive change into a broken V4.0.
    """
    assert _metadata().max_seq_len == 0


def test_max_seq_len_is_carried_not_recomputed():
    """It is a Python int on common_attn_metadata, already derived from the CPU
    copy of seq_lens. Reading it off the device tensor instead would add a sync
    per layer per step, and would be illegal under graph capture."""
    import inspect

    from vllm_ascend.attention import dsa_v1

    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder)
    assert source.count("max_seq_len=common_attn_metadata.max_seq_len") == 2, (
        "both build sites must carry it; a metadata built without it reports "
        "0, which reads as 'the batch is empty' at the one place it is used")


def test_the_two_new_names_come_from_the_request_payload():
    class _Req:
        query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
        block_table = torch.zeros((2, 4), dtype=torch.int32)
        slot_mapping = None

    md = _metadata(req_metadata=_Req())
    assert torch.equal(md.query_start_loc, _Req.query_start_loc)
    assert torch.equal(md.block_table, _Req.block_table)


def test_they_are_none_without_a_payload():
    """Profile and dummy runs build metadata with req_metadata unset; the
    caller's own guard is what should fire, not an AttributeError here."""
    md = _metadata(req_metadata=None)
    assert md.query_start_loc is None and md.block_table is None


def test_the_enumeration_was_taken_from_upstream_not_from_failures():
    """Both engram call sites must still read only names on the list.

    Grown one AttributeError per launch, this list is always one short. Read
    off the source, it goes red when upstream adds a read -- here, rather than
    on a card.
    """
    import re

    src = "/workspace/code/dsv4_single_card/vllm_project/vllm/vllm/models/deepseek_v4_1/nvidia/model.py"
    with open(src) as fh:
        text = fh.read()
    reads = set(re.findall(r"swa_metadata\.(\w+)", text))
    assert reads, "no swa_metadata reads found; the call site moved"
    unknown = reads - set(UPSTREAM_READS)
    assert not unknown, (
        f"upstream reads {sorted(unknown)} off the SWA metadata and this list "
        "does not mention them")


def test_slot_mapping_falls_back_when_the_field_is_nulled():
    """dsa_v1 sets the field to None for compressing groups, on purpose.

    Their compressed stream is scattered by its own operator, so a per-token
    mapping there would be misleading. V4.1's indexer K writer needs one at
    every ratio -- _produce_k reads it through this property -- so the null is
    kept as a field and answered here.
    """
    class _Req:
        slot_mapping = None
        token_slot_mapping = torch.arange(4, dtype=torch.int32)

    assert torch.equal(_metadata(req_metadata=_Req()).slot_mapping,
                       _Req.token_slot_mapping)


def test_an_explicit_slot_mapping_wins():
    class _Req:
        slot_mapping = torch.zeros(2, dtype=torch.int32)
        token_slot_mapping = torch.ones(2, dtype=torch.int32)

    assert torch.equal(_metadata(req_metadata=_Req()).slot_mapping,
                       _Req.slot_mapping)


def test_both_builder_sites_publish_the_per_token_mapping():
    """A site that forgets it leaves the property returning None again, which
    is the failure this replaced -- an AttributeError on .shape one frame in."""
    import inspect

    from vllm_ascend.attention import dsa_v1

    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder)
    assert source.count("token_slot_mapping=") == 2


def test_the_descriptor_defaults_are_v40s_hardcoded_values():
    """The one path by which V4.1's change could reach V4.0.

    Every V4.0 call site omits these parameters, so the defaults are what it
    gets. This is the exception to "do not write invariants against V4.0": the
    whole purpose of these defaults is that changing one silently changes V4.0's
    descriptor, which is the harm the parameterisation was chosen to avoid.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1

    sig = inspect.signature(dsa_v1.AscendDSAMetadataBuilder._build_sas_metadata)
    for name in ("ori_mask_mode", "ori_win_left", "ori_win_right",
                 "has_cmp_kv"):
        assert sig.parameters[name].default is None, (
            f"{name} must default to None so the body applies V4.0's value; a "
            "literal default here would be a second place to keep in step")

    body = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder._build_sas_metadata)
    assert "4 if ori_mask_mode is None else ori_mask_mode" in body
    assert "sliding_window - 1\n" in body or "sliding_window - 1" in body
    assert "0 if ori_win_right is None else ori_win_right" in body
    assert "self.compressor_ratio > 1\n" in body or "compressor_ratio > 1" in body


def test_v40_call_sites_pass_none_of_them():
    """B' holds only while V4.0's callers stay unchanged."""
    import inspect

    from vllm_ascend.attention import dsa_v1

    # Scoped to _build_sas_metadata: build_for_drafting constructs its own
    # descriptor with its own literals, and that path is V4.0's speculative
    # decoding, which this change does not touch.
    body = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder._build_sas_metadata)
    assert body.count("ori_mask_mode=") == 1, (
        "only the descriptor operator call may name ori_mask_mode here; a "
        "caller passing it directly would bypass the parameter")
    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder)
    assert source.count("**sas_mask_kwargs") == 2, (
        "both _build_sas_metadata call sites must forward the triple, or one "
        "path silently keeps V4.0's descriptor while the call uses V4.1's")


def test_the_descriptor_and_the_call_agree_on_the_mask_triple():
    """The invariant, stated without reference to V4.0.

    The kernel takes its tiling from the descriptor and then receives the real
    arguments; a disagreement is not rejected.

    This used to pin that the two sides spelled the same rule. Spelling it
    twice was itself the defect: the two expressions agreed, and agreement is
    also the acceptance criterion, so a second source in that state announces
    nothing. The builder now calls the call site's own functions, and what is
    pinned here is that it still does.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1
    from vllm_ascend.models.deepseek_v4_1 import sparse_attn_args

    builder = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder)
    # Descriptor side: mode 4 only when plain sliding window.
    assert '"ori_mask_mode": ori_mask_mode_for(' in builder, (
        "the descriptor spells the mode itself again. It was written as "
        '"4 if plain_window else 0" -- the two-configuration reading -- and '
        "the third configuration, no indices with a compressed stream, "
        "silently took the 0, which is NO_MASK.")
    # Call side: the same predicate object, not a copy of its text. Pinning
    # the expression was pinning a spelling -- it went red for a change that
    # kept the invariant and moved both sides together, which is the one
    # outcome this test should stay quiet for.
    assert "is_plain_sliding_window(" in builder
    assert sparse_attn_args.is_plain_sliding_window(True, object()) is False
    assert sparse_attn_args.is_plain_sliding_window(False, object()) is False
    # One source: the builder asks the predicate rather than restating it.
    assert "plain_window = is_plain_sliding_window(" in builder
    assert "select_ori_sparse_indices(" in builder, (
        "the builder derives the index list some other way again; the two "
        "sources agreed last time, which is how nobody noticed they were two")


def test_the_topk_promise_and_the_indices_are_one_decision():
    """cmp_topk in the descriptor and cmp_sparse_indices in the call.

    A descriptor promising a top-k the call does not supply is the same
    disagreement class that produced the aicore exception. They are computed in
    one place each and both gated on the same fact -- a compressed stream --
    rather than by two independent judgements that happen to agree.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1
    from vllm_ascend.models.deepseek_v4_1 import attention as v41_attention

    builder = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder)
    assert 'if attends_cmp:' in builder
    assert 'sas_mask_kwargs["cmp_topk"]' in builder
    assert 'sas_mask_kwargs["seqused_cmp_kv"]' in builder

    caller = inspect.getsource(v41_attention.AscendDeepseekV41Attention.forward_mqa)
    # The selection is bound once now and passed by name, so that the indices
    # and their length cannot be built from two different expressions.
    assert "cmp_selection = (" in caller
    assert "cmp_sparse_indices=cmp_selection," in caller
    assert "if attends_compressed and self.topk_indices_buffer is not None" in caller


def test_the_compressed_descriptor_extras_default_to_absent():
    """V4.0's descriptor keeps the argument list it has today.

    seqused_cmp_kv is omitted entirely when None rather than passed as None:
    the operator demands it only under PA_BBND with a declared compressed
    stream and a non-zero mask mode, and V4.0's FP8 plan uses PA_ND, so it has
    never had to supply one.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1

    sig = inspect.signature(dsa_v1.AscendDSAMetadataBuilder._build_sas_metadata)
    assert sig.parameters["cmp_topk"].default is None
    assert sig.parameters["seqused_cmp_kv"].default is None
    body = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder._build_sas_metadata)
    assert '{} if seqused_cmp_kv is None' in body
    assert "index_topk if self.compressor_ratio == 4 else 0" in body
