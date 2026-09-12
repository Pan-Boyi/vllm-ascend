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
                  "query_start_loc", "block_table")


def test_upstream_reads_all_of_these_names():
    """The list itself, so a rename upstream is a red test and not a crash."""
    md = _metadata()
    missing = [n for n in UPSTREAM_READS if not hasattr(md, n)]
    assert not missing, f"upstream reads these and Ascend does not carry them: {missing}"


def test_block_size_is_deliberately_absent():
    """attention.py:880 reads swa_metadata.block_size and this does not have it.

    Not an oversight and not safe to add by analogy. req_metadata carries
    storage_block_size, which equals the spec's block_size only at ratio 1 --
    for the ratio-2 groups they differ by a factor of two, and a property that
    is right for SWA and wrong for those would be read as authoritative by
    both. The call sites that want it are in the fused CUDA prolog, which
    Ascend replaces; if one is ever reached, it should fail here rather than
    receive a plausible number.
    """
    assert not hasattr(_metadata(), "block_size")


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
