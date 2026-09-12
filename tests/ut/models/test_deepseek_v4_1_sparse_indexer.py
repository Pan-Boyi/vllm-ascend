# SPDX-License-Identifier: Apache-2.0
"""forward_oot: the registration, the guards, and the buffer contract.

The registration half matters as much as the implementation. Upstream's base
forward_oot falls through to forward_native, which raises "only implemented for
CUDA, ROCm, XPU and CPU platforms" -- the identical message whether the Ascend
path does not exist or merely was not registered. A test that only calls the
method directly would pass in both worlds.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import sparse_indexer as mod
from vllm_ascend.models.deepseek_v4_1.sparse_indexer import (
    NO_TOKEN, AscendSparseAttnIndexer, unpack_indexer_cache)

TOPK = 8
HEADS = 4
HEAD_DIM = 128


def test_it_substitutes_upstreams_class_by_name():
    """CustomOp.__new__ looks up op_registry_oot[cls.__name__].

    So the key has to be the in-tree class name, not the registered op name
    "sparse_attn_indexer". Keyed wrong, nothing happens and the failure is the
    same NotImplementedError as before.
    """
    from vllm.model_executor.layers.sparse_attn_indexer import \
        SparseAttnIndexer

    table = {}
    mod.register(table)
    assert table["SparseAttnIndexer"] is AscendSparseAttnIndexer
    assert issubclass(AscendSparseAttnIndexer, SparseAttnIndexer)


def test_it_overrides_forward_oot_rather_than_inheriting_the_fallthrough():
    from vllm.model_executor.custom_op import CustomOp

    assert AscendSparseAttnIndexer.forward_oot is not CustomOp.forward_oot


def test_registration_is_scoped_to_v41():
    """V3.2 and V4.0 also build SparseAttnIndexer, and their Ascend paths reach
    their operators through the attention backend instead. Registering globally
    would redirect them into this."""
    import inspect

    from vllm_ascend import utils

    source = inspect.getsource(utils.register_ascend_customop)
    assert "is_v41" in source
    assert "sparse_indexer" in source


class _Indexer(AscendSparseAttnIndexer):
    """Constructed without upstream's __init__, which wants a DCP group and
    warms up Triton kernels. Only the attributes forward_oot reads are set."""

    def __init__(self, **kwargs):
        self.skip_k_cache_insert = True
        self.compress_ratio = 1
        self.topk_tokens = TOPK
        self.head_dim = HEAD_DIM
        self.candidate_blocks = None
        self.candidate_block_size = 0
        self.topk_indices_buffer = torch.zeros(16, TOPK, dtype=torch.int32)
        self.k_cache = SimpleNamespace(prefix="layers.3.indexer.k_cache",
                                       kv_cache=None)
        self.__dict__.update(kwargs)


def _metadata(num_tokens=4, num_reqs=2, max_seq_len=64):
    req = SimpleNamespace(
        seq_lens=torch.tensor([max_seq_len, max_seq_len // 2],
                              dtype=torch.int32)[:num_reqs],
        query_start_loc=torch.tensor([0, num_tokens // 2, num_tokens],
                                     dtype=torch.int32),
        block_table=torch.zeros((num_reqs, 4), dtype=torch.int32))
    return SimpleNamespace(req_metadata=req, max_seq_len=max_seq_len)


@pytest.fixture
def forward_context(monkeypatch):
    holder = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(mod, "get_forward_context", lambda: holder)
    return holder


def test_a_profile_run_returns_the_buffer_untouched(forward_context):
    """attn_metadata is None during profiling and _dummy_run.

    Raising here would make every startup fail; writing here would put values
    nobody asked for into a buffer that outlives the call.
    """
    indexer = _Indexer()
    indexer.topk_indices_buffer.fill_(7)
    forward_context.attn_metadata = None
    out = indexer.forward_oot(torch.zeros(4, 16), torch.zeros(4, HEADS,
                                                              HEAD_DIM),
                              None, torch.zeros(4, HEADS))
    assert out is indexer.topk_indices_buffer
    assert torch.all(out == 7)


def test_a_non_none_k_is_refused(forward_context):
    """V4.1 passes None and writes the cache itself. Anything else is a caller
    expecting the insert half, which this does not implement."""
    forward_context.attn_metadata = {"layers.3.indexer.k_cache": _metadata()}
    with pytest.raises(ValueError, match="writes its own K cache"):
        _Indexer().forward_oot(torch.zeros(4, 16),
                               torch.zeros(4, HEADS, HEAD_DIM),
                               torch.zeros(4, HEAD_DIM),
                               torch.zeros(4, HEADS))


def test_skip_k_cache_insert_false_is_refused(forward_context):
    forward_context.attn_metadata = {"layers.3.indexer.k_cache": _metadata()}
    with pytest.raises(ValueError, match="quantise k and write"):
        _Indexer(skip_k_cache_insert=False).forward_oot(
            torch.zeros(4, 16), torch.zeros(4, HEADS, HEAD_DIM), None,
            torch.zeros(4, HEADS))


def test_missing_req_metadata_names_the_layer(forward_context):
    md = _metadata()
    md.req_metadata = None
    forward_context.attn_metadata = {"layers.3.indexer.k_cache": md}
    with pytest.raises(ValueError, match="layers.3.indexer.k_cache"):
        _Indexer().forward_oot(torch.zeros(4, 16),
                               torch.zeros(4, HEADS, HEAD_DIM), None,
                               torch.zeros(4, HEADS))


def test_a_single_tensor_cache_is_now_split_into_the_two_arguments():
    """This used to refuse, and refusing was right while the offset was a guess.

    "One tensor cannot serve both" was never the problem -- the runner really
    does allocate one, because upstream builds this cache with head_dim already
    carrying the scale padding. The problem was that slicing it needed a byte
    layout nobody here had read, and a guessed one produces a plausible wrong
    answer rather than an error.

    The layout is now read out of the operator's kernel, so the split is done
    instead of refused. What it must not become is a guess again: the round
    trip in test_deepseek_v4_1_indexer_cache_view.py is what holds that, and a
    shape check here would not.
    """
    from vllm_ascend.models.deepseek_v4_1.indexer_cache_view import (
        SCALE_BYTES, SCALE_GROUP)

    width = SCALE_GROUP + SCALE_BYTES
    key, scale = unpack_indexer_cache(
        torch.zeros(2, 4, width, dtype=torch.uint8))
    assert key.shape == (2, 4, 1, SCALE_GROUP)
    assert scale.shape == (2, 4, 1, 1)


def test_a_single_tensor_of_an_impossible_width_is_still_refused():
    """The guard that survived: a width that is not values plus scales has no
    boundary to place, and placing one anyway is the guessed offset again."""
    with pytest.raises(ValueError, match="between a value and its scale"):
        unpack_indexer_cache(torch.zeros(2, 4, 8, dtype=torch.uint8))


def test_an_unexpected_tuple_arity_reports_what_it_got():
    with pytest.raises(ValueError, match="3 tensors"):
        unpack_indexer_cache((torch.zeros(1), torch.zeros(2), torch.zeros(3)))


def test_a_pair_unpacks_in_order():
    values, scales = torch.zeros(1), torch.ones(1)
    assert unpack_indexer_cache((values, scales)) == (values, scales)


def _run(indexer, forward_context, *, num_tokens=4, returned=None):
    calls = {}

    def _select(**kwargs):
        calls.update(kwargs)
        if returned is not None:
            return returned
        return torch.arange(num_tokens * TOPK, dtype=torch.int32).reshape(
            num_tokens, 1, TOPK)

    def _meta(**kwargs):
        calls["metadata_kwargs"] = kwargs
        return torch.zeros(1024, dtype=torch.int32)

    class _Ops:
        npu_quant_lightning_indexer_v2_metadata = staticmethod(_meta)

    import vllm_ascend.models.deepseek_v4_1.sparse_indexer as sp
    orig_select = sp.select_topk
    sp.select_topk = _select
    orig_ops = getattr(torch.ops, "_C_ascend", None)
    torch.ops._C_ascend = _Ops
    try:
        out = indexer.forward_oot(torch.zeros(num_tokens, 16),
                                  torch.zeros(num_tokens, HEADS, HEAD_DIM),
                                  None, torch.zeros(num_tokens, HEADS))
    finally:
        sp.select_topk = orig_select
        if orig_ops is not None:
            torch.ops._C_ascend = orig_ops
    return out, calls


def test_the_untouched_tail_of_a_row_is_the_no_token_sentinel(forward_context):
    """Upstream's buffer contract: -1 where there is no token.

    The rows this call covers are overwritten; the point of the clear is the
    ones it does not, which otherwise keep the previous step's indices and are
    read as real tokens.
    """
    indexer = _Indexer()
    indexer.k_cache.kv_cache = (torch.zeros(2, 4, HEAD_DIM,
                                            dtype=torch.float8_e4m3fn),
                                torch.zeros(2, 4, dtype=torch.float32))
    indexer.topk_indices_buffer.fill_(99)
    forward_context.attn_metadata = {"layers.3.indexer.k_cache": _metadata()}
    out, _ = _run(indexer, forward_context, num_tokens=4)
    assert torch.all(out[4:] == 99), "rows beyond the batch are not this call's"
    assert out[:4].max() >= 0


def test_a_short_row_keeps_the_sentinel(forward_context):
    indexer = _Indexer()
    indexer.k_cache.kv_cache = (torch.zeros(2, 4, HEAD_DIM,
                                            dtype=torch.float8_e4m3fn),
                                torch.zeros(2, 4, dtype=torch.float32))
    forward_context.attn_metadata = {"layers.3.indexer.k_cache": _metadata()}
    returned = torch.full((4, 1, TOPK), NO_TOKEN, dtype=torch.int32)
    out, _ = _run(indexer, forward_context, num_tokens=4, returned=returned)
    assert torch.all(out[:4] == NO_TOKEN)


def test_the_descriptor_and_the_call_agree_on_ratio_and_topk(forward_context):
    """Two places derive the same numbers; disagreeing builds a descriptor for
    one kernel variant and invokes another."""
    indexer = _Indexer(compress_ratio=2)
    indexer.k_cache.kv_cache = (torch.zeros(2, 4, HEAD_DIM,
                                            dtype=torch.float8_e4m3fn),
                                torch.zeros(2, 4, dtype=torch.float32))
    forward_context.attn_metadata = {
        "layers.3.indexer.k_cache": _metadata(max_seq_len=64)}
    _, calls = _run(indexer, forward_context)
    assert calls["compress_ratio"] == 2
    assert calls["index_topk"] == TOPK
    meta = calls["metadata_kwargs"]
    assert meta["cmp_ratio"] == 2
    assert meta["topk"] == TOPK
    assert meta["head_dim"] == HEAD_DIM
    assert meta["num_heads_q"] == HEADS, "heads, not tokens"
    assert meta["num_heads_k"] == 1
    assert meta["max_seqlen_k"] == 64 // 2
    assert meta["layout_q"] == "TND" and meta["layout_k"] == "PA_BBND"


def test_the_lengths_handed_to_both_are_the_same_tensors(forward_context):
    """seqused_k and cmp_residual_k describe the descriptor and parameterise
    the call; computing them twice would let them drift."""
    indexer = _Indexer(compress_ratio=2)
    indexer.k_cache.kv_cache = (torch.zeros(2, 4, HEAD_DIM,
                                            dtype=torch.float8_e4m3fn),
                                torch.zeros(2, 4, dtype=torch.float32))
    forward_context.attn_metadata = {"layers.3.indexer.k_cache": _metadata()}
    _, calls = _run(indexer, forward_context)
    assert calls["seqused_k"] is calls["metadata_kwargs"]["seqused_k"]
    assert calls["cmp_residual_k"] is calls["metadata_kwargs"]["cmp_residual_k"]


def test_a_context_past_the_candidate_boundary_refuses(forward_context):
    """The filter stops being vacuous and nothing else would say so."""
    indexer = _Indexer(compress_ratio=1, candidate_block_size=8,
                       candidate_blocks=torch.zeros((16, 64),
                                                    dtype=torch.int32))
    indexer.k_cache.kv_cache = (torch.zeros(2, 4, HEAD_DIM,
                                            dtype=torch.float8_e4m3fn),
                                torch.zeros(2, 4, dtype=torch.float32))
    # 64 blocks of 8 = 512 compressed columns; a 4096-token sequence needs 512.
    forward_context.attn_metadata = {
        "layers.3.indexer.k_cache": _metadata(max_seq_len=4096)}
    with pytest.raises(NotImplementedError, match="candidate filtering"):
        _run(indexer, forward_context)
