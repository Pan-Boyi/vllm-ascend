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


def test_upstream_reads_these_three_names():
    """The list itself, so a rename upstream is a red test and not a crash."""
    for name in ("max_seq_len", "slot_mapping", "num_prefill_tokens"):
        assert hasattr(_metadata(), name), name


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
