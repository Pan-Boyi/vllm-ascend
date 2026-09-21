# SPDX-License-Identifier: Apache-2.0
"""The two-level candidate filter (O6): argument assembly, roles, buffers.

Every failure this file is built around is silent on a card. The ds41
publisher returns a correctly shaped, correct selection while publishing
nothing (``candidate_topk_blocks`` defaults to -1); the consumer takes the
same tensors as the publisher under two different names, so a copy-paste
swaps the query and key scales without a shape complaining; the candidate
buffer is ``-1``-padded and never cleared, so a consumer reading rows nobody
wrote this step masks to the previous step's blocks and selects the wrong
tokens. None of those change a shape, a dtype or an exception.

What is **not** here: whether the operators compute the right numbers. That
needs a card, and it is
``tests/e2e/models/test_deepseek_v4_1_candidate_oracle.py``.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import indexer_candidate as cand
from vllm_ascend.models.deepseek_v4_1 import sparse_indexer as mod
from vllm_ascend.models.deepseek_v4_1.sparse_indexer import (
    AscendSparseAttnIndexer, candidate_lengths_buffer)

TOPK = 8
HEADS = 32
HEAD_DIM = 128
BLOCKS = 2048
BLOCK_SIZE = 8
TOKENS = 4


# --------------------------------------------------------------- fake operators

class _RecordingOp:
    """Stands in for a ds41 overload and keeps what it was called with."""

    def __init__(self, outputs):
        self.outputs = outputs
        self.kwargs = None
        self.args = None

    def __call__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        return self.outputs


def _publisher(rows=TOKENS, blocks=BLOCKS):
    return _RecordingOp((
        torch.zeros(rows, 1, TOPK, dtype=torch.int32),
        torch.zeros(0),
        torch.zeros(rows, blocks, dtype=torch.int32),
        torch.zeros(rows, dtype=torch.int32),
    ))


def _consumer(rows=TOKENS):
    return _RecordingOp(
        (torch.zeros(rows, 1, TOPK, dtype=torch.int32), torch.zeros(0)))


def _call_args(rows=TOKENS):
    """Byte planes and metadata in the shapes the operators take."""
    return dict(
        query=torch.zeros(rows, HEADS, HEAD_DIM // 2, dtype=torch.uint8),
        q_scale=torch.zeros(rows, HEADS, HEAD_DIM // 64, 2,
                            dtype=torch.uint8),
        key_cache=torch.zeros(3, 128, 1, HEAD_DIM // 2, dtype=torch.uint8),
        key_dequant_scale=torch.zeros(3, 128, 1, 2, 2, dtype=torch.uint8),
        weights=torch.zeros(rows, HEADS, dtype=torch.float32),
        index_topk=TOPK,
        compress_ratio=1,
        cu_seqlens_q=torch.tensor([0, 1, rows], dtype=torch.int32),
        seqused_k=torch.tensor([7, 9], dtype=torch.int32),
        cmp_residual_k=None,
        block_table=torch.zeros(2, 4, dtype=torch.int32),
    )


# ------------------------------------------------------------- argument assembly

def test_the_publisher_is_asked_to_publish():
    """🔴 The regression this whole module exists for.

    ``candidate_topk_blocks`` is keyword-only with a default of -1, and at -1
    the operator returns empty candidate tensors, raises nothing, and returns
    a correct selection of the correct shape. Omitting it degrades two-level
    filtering to no filtering with every other signal intact.
    """
    op = _publisher()
    cand.publish_candidates(**_call_args(), candidate_topk_blocks=BLOCKS,
                            candidate_block_size=BLOCK_SIZE, where="l20",
                            op=op)
    assert op.kwargs["candidate_topk_blocks"] == BLOCKS
    assert op.kwargs["candidate_block_size"] == BLOCK_SIZE


def test_an_empty_publication_is_refused_although_the_selection_is_fine():
    """What a forgotten candidate_topk_blocks actually looks like."""
    degenerate = _RecordingOp((
        torch.zeros(TOKENS, 1, TOPK, dtype=torch.int32),  # correct selection
        torch.zeros(0),
        torch.zeros(0, dtype=torch.int32),  # nothing published
        torch.zeros(0, dtype=torch.int32),
    ))
    with pytest.raises(RuntimeError, match="candidate_topk_blocks"):
        cand.publish_candidates(**_call_args(), candidate_topk_blocks=BLOCKS,
                                candidate_block_size=BLOCK_SIZE,
                                where="l20", op=degenerate)


def test_a_length_per_row_is_required():
    short = _RecordingOp((
        torch.zeros(TOKENS, 1, TOPK, dtype=torch.int32),
        torch.zeros(0),
        torch.zeros(TOKENS, BLOCKS, dtype=torch.int32),
        torch.zeros(TOKENS - 1, dtype=torch.int32),
    ))
    with pytest.raises(RuntimeError, match="candidate lengths"):
        cand.publish_candidates(**_call_args(), candidate_topk_blocks=BLOCKS,
                                candidate_block_size=BLOCK_SIZE,
                                where="l20", op=short)


def test_the_two_operators_disagree_on_the_scale_argument_names():
    """🔴 The publisher takes q_descale/k_descale and the consumer
    descale_q/descale_k. Both are uint8 E8M0 and the shapes do not reject
    each other, so a hand-written swap runs and returns garbage."""
    args = _call_args()
    pub, con = _publisher(), _consumer()
    cand.publish_candidates(**args, candidate_topk_blocks=BLOCKS,
                            candidate_block_size=BLOCK_SIZE, where="l20",
                            op=pub)
    cand.select_within_candidates(
        **args, candidate_indices=torch.zeros(TOKENS, 1, BLOCKS,
                                              dtype=torch.int32),
        candidate_lengths=torch.zeros(TOKENS, 1, dtype=torch.int32),
        candidate_block_size=BLOCK_SIZE, op=con)

    assert set(pub.kwargs) & {"descale_q", "descale_k"} == set()
    assert set(con.kwargs) & {"q_descale", "k_descale"} == set()
    # The same tensors in the same roles under both spellings -- identity,
    # not equality: two zero tensors of these shapes compare equal while
    # transposed.
    assert pub.kwargs["q_descale"] is args["q_scale"]
    assert pub.kwargs["k_descale"] is args["key_dequant_scale"]
    assert con.kwargs["descale_q"] is args["q_scale"]
    assert con.kwargs["descale_k"] is args["key_dequant_scale"]


@pytest.mark.parametrize("call", ["publish", "select"])
def test_seqused_q_is_passed_because_every_prebuilt_kernel_expects_it(call):
    """⚠ Optional in the signature, not optional in practice: the prebuilt
    variants were all exported with has_seqused_q=True, and the presence of
    each optional tensor is part of the key that picks one. Omitting it lands
    outside the prebuilt set and compiles a kernel on the spot."""
    args = _call_args()
    if call == "publish":
        op = _publisher()
        cand.publish_candidates(**args, candidate_topk_blocks=BLOCKS,
                                candidate_block_size=BLOCK_SIZE,
                                where="l20", op=op)
    else:
        op = _consumer()
        cand.select_within_candidates(
            **args,
            candidate_indices=torch.zeros(TOKENS, 1, BLOCKS,
                                          dtype=torch.int32),
            candidate_lengths=torch.zeros(TOKENS, 1, dtype=torch.int32),
            candidate_block_size=BLOCK_SIZE, op=op)
    assert torch.equal(op.kwargs["seqused_q"],
                       torch.tensor([1, TOKENS - 1], dtype=torch.int32))


@pytest.mark.parametrize("call", ["publish", "select"])
def test_the_fixed_contract_of_both_calls(call):
    """metadata None, mode 1, the paged compressed layout, no values.

    ``metadata`` is the one that would look most like an oversight: this
    version of the wheel refuses a non-None metadata outright, so the
    descriptor built for the v2 operator has no receiver here.
    """
    args = _call_args()
    if call == "publish":
        op = _publisher()
        cand.publish_candidates(**args, candidate_topk_blocks=BLOCKS,
                                candidate_block_size=BLOCK_SIZE,
                                where="l20", op=op)
    else:
        op = _consumer()
        cand.select_within_candidates(
            **args,
            candidate_indices=torch.zeros(TOKENS, 1, BLOCKS,
                                          dtype=torch.int32),
            candidate_lengths=torch.zeros(TOKENS, 1, dtype=torch.int32),
            candidate_block_size=BLOCK_SIZE, op=op)
    assert op.kwargs["metadata"] is None
    assert op.kwargs["quant_mode"] == 1
    assert op.kwargs["mask_mode"] == 3
    assert op.kwargs["cmp_ratio"] == 1
    assert (op.kwargs["layout_q"], op.kwargs["layout_kv"]) == ("TND",
                                                               "PA_BBND")
    assert op.kwargs["return_value"] is False
    # ⚠ uint8 reaches the operator as uint8. ``indexer_select`` reinterprets
    # the same planes as float4/e8m0 for the v2 operator, which derives its
    # mode from the dtype pair; ds41 type-checks the storage and rejects
    # exactly that reinterpretation.
    assert op.kwargs["q"].dtype is torch.uint8
    assert op.kwargs["k"].dtype is torch.uint8


def test_a_checkpoint_the_kernels_cannot_serve_is_named_up_front():
    """Both numbers are compiled into the kernels. A mismatch would surface
    as a shape complaint from inside the operator instead."""
    with pytest.raises(NotImplementedError, match="candidate_block_size"):
        cand.assert_candidate_geometry(16, BLOCKS)
    with pytest.raises(NotImplementedError, match="candidate axis"):
        cand.assert_candidate_geometry(BLOCK_SIZE, 1024)
    cand.assert_candidate_geometry(BLOCK_SIZE, BLOCKS)  # the checkpoint's


# ------------------------------------------------------------------- the roles

class _Indexer(AscendSparseAttnIndexer):
    """Upstream's __init__ wants a DCP group and warms up Triton; only what
    forward_oot reads is set. The candidate attributes are upstream's own."""

    def __init__(self, **kwargs):
        self.skip_k_cache_insert = True
        self.compress_ratio = 1
        self.topk_tokens = TOPK
        self.head_dim = HEAD_DIM
        self.candidate_blocks = None
        self.candidate_block_size = BLOCK_SIZE
        self.candidate_write = False
        self.topk_indices_buffer = torch.zeros(16, TOPK, dtype=torch.int32)
        self.k_cache = SimpleNamespace(prefix="layers.20.indexer.k_cache",
                                       kv_cache=None)
        self.__dict__.update(kwargs)


def _blocks(rows=16):
    return torch.full((rows, BLOCKS), -1, dtype=torch.int32)


def _metadata(num_tokens=TOKENS, max_seq_len=64):
    req = SimpleNamespace(
        seq_lens=torch.tensor([max_seq_len, max_seq_len // 2],
                              dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, num_tokens], dtype=torch.int32),
        block_table=torch.zeros((2, 4), dtype=torch.int32),
        storage_block_size=128)
    return SimpleNamespace(req_metadata=req, max_seq_len=max_seq_len)


@pytest.fixture
def forward_context(monkeypatch):
    holder = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(mod, "get_forward_context", lambda: holder)
    return holder


def _run(indexer, forward_context, num_tokens=TOKENS):
    indexer.k_cache.kv_cache = (
        torch.zeros(3, 128, 1, HEAD_DIM // 2, dtype=torch.uint8),
        torch.zeros(3, 128, 1, 4, dtype=torch.uint8),
    )
    forward_context.attn_metadata = {
        indexer.k_cache.prefix: _metadata(num_tokens)
    }
    q = (torch.zeros(num_tokens, HEADS, HEAD_DIM // 2, dtype=torch.uint8),
         torch.zeros(num_tokens, HEADS, HEAD_DIM // 64, 2, dtype=torch.uint8))
    return indexer.forward_oot(torch.zeros(num_tokens, 16), q, None,
                               torch.zeros(num_tokens, HEADS,
                                           dtype=torch.float32))


def test_the_publisher_fills_the_shared_buffers(forward_context, monkeypatch):
    blocks = _blocks()
    indexer = _Indexer(candidate_blocks=blocks, candidate_write=True)
    published = torch.arange(TOKENS * BLOCKS,
                             dtype=torch.int32).reshape(TOKENS, BLOCKS)
    lengths = torch.tensor([3, 4, 5, 6], dtype=torch.int32)
    monkeypatch.setattr(
        mod, "publish_candidates", lambda **kw:
        (torch.zeros(TOKENS, 1, TOPK, dtype=torch.int32), published, lengths))

    _run(indexer, forward_context)

    assert torch.equal(blocks[:TOKENS], published)
    # ⚠ The publisher returns [T] and the consumer takes [T, 1]; upstream's
    # Ascend branch bridges the same gap at dsa_v41.py:1060.
    assert indexer.candidate_lengths.shape == (16, 1)
    assert torch.equal(indexer.candidate_lengths[:TOKENS, 0], lengths)
    # Untouched rows keep the -1 padding rather than an earlier step's blocks.
    assert torch.all(blocks[TOKENS:] == -1)


def test_the_consumer_reads_this_steps_rows(forward_context, monkeypatch):
    blocks = _blocks()
    publisher = _Indexer(candidate_blocks=blocks, candidate_write=True)
    consumer = _Indexer(candidate_blocks=blocks,
                        k_cache=SimpleNamespace(
                            prefix="layers.24.indexer.k_cache", kv_cache=None))
    monkeypatch.setattr(
        mod, "publish_candidates", lambda **kw:
        (torch.zeros(TOKENS, 1, TOPK, dtype=torch.int32),
         torch.ones(TOKENS, BLOCKS, dtype=torch.int32),
         torch.full((TOKENS, ), 5, dtype=torch.int32)))
    seen = {}

    def _select(**kwargs):
        seen.update(kwargs)
        return torch.zeros(TOKENS, 1, TOPK, dtype=torch.int32)

    monkeypatch.setattr(mod, "select_within_candidates", _select)

    _run(publisher, forward_context)
    _run(consumer, forward_context)

    assert seen["candidate_indices"].shape == (TOKENS, 1, BLOCKS)
    assert seen["candidate_lengths"].shape == (TOKENS, 1)
    assert torch.all(seen["candidate_indices"] == 1)
    assert torch.all(seen["candidate_lengths"] == 5)


def test_a_consumer_whose_publisher_did_not_run_refuses(forward_context):
    """🔴 The buffer is one step's scratch. Reading rows nobody wrote this
    step masks to -1 padding or to the previous step's blocks -- a wrong
    selection with every shape correct."""
    consumer = _Indexer(candidate_blocks=_blocks())
    with pytest.raises(RuntimeError, match="not been written"):
        _run(consumer, forward_context)


def test_last_steps_publication_does_not_count(forward_context, monkeypatch):
    blocks = _blocks()
    publisher = _Indexer(candidate_blocks=blocks, candidate_write=True)
    consumer = _Indexer(candidate_blocks=blocks)
    monkeypatch.setattr(
        mod, "publish_candidates", lambda **kw:
        (torch.zeros(TOKENS, 1, TOPK, dtype=torch.int32),
         torch.ones(TOKENS, BLOCKS, dtype=torch.int32),
         torch.full((TOKENS, ), 5, dtype=torch.int32)))
    monkeypatch.setattr(
        mod, "select_within_candidates",
        lambda **kw: torch.zeros(TOKENS, 1, TOPK, dtype=torch.int32))

    _run(publisher, forward_context)
    # A new forward pass: set_forward_context builds a fresh context object
    # per model forward, and the buffer still holds last step's blocks.
    forward_context.attn_metadata = None
    monkeypatch.setattr(mod, "get_forward_context",
                        lambda: SimpleNamespace(attn_metadata={
                            consumer.k_cache.prefix: _metadata()
                        }))
    with pytest.raises(RuntimeError, match="not been written"):
        _run(consumer, forward_context)


def test_more_query_rows_than_were_published_refuses(forward_context,
                                                     monkeypatch):
    blocks = _blocks()
    publisher = _Indexer(candidate_blocks=blocks, candidate_write=True)
    consumer = _Indexer(candidate_blocks=blocks)
    monkeypatch.setattr(
        mod, "publish_candidates", lambda **kw:
        (torch.zeros(2, 1, TOPK, dtype=torch.int32),
         torch.ones(2, BLOCKS, dtype=torch.int32),
         torch.full((2, ), 5, dtype=torch.int32)))
    _run(publisher, forward_context, num_tokens=2)
    with pytest.raises(RuntimeError, match="filled 2 rows"):
        _run(consumer, forward_context, num_tokens=TOKENS)


def test_a_ratio_two_layer_stays_on_the_v2_operator(forward_context,
                                                    monkeypatch):
    """Layers 2/8/14 have no candidate buffer, were never under the ceiling,
    and are what batch 1 validated end to end."""
    indexer = _Indexer(compress_ratio=2, candidate_blocks=None)
    seen = {}
    monkeypatch.setattr(
        mod, "select_topk", lambda **kw:
        (seen.update(kw), torch.zeros(TOKENS, 1, TOPK,
                                      dtype=torch.int32))[1])
    monkeypatch.setattr(mod.AscendSparseAttnIndexer, "_qli_metadata",
                        lambda *a, **k: "descriptor")
    _run(indexer, forward_context)
    # The v2 descriptor is still built and still passed -- it is the ds41
    # path that has no receiver for it.
    assert seen["metadata"] == "descriptor"


def test_the_lengths_buffer_is_shared_and_keeps_its_address():
    """🔴 A graph replays against a fixed address; a buffer reallocated per
    step would be captured as whatever it was at capture time."""
    blocks = _blocks()
    first = candidate_lengths_buffer(blocks)
    assert candidate_lengths_buffer(blocks) is first
    publisher = _Indexer(candidate_blocks=blocks, candidate_write=True)
    consumer = _Indexer(candidate_blocks=blocks)
    assert publisher.candidate_lengths is first
    assert consumer.candidate_lengths is first
    assert first.shape == (blocks.shape[0], 1)
    assert first.dtype is torch.int32


def test_an_indexer_without_candidates_has_no_lengths_buffer():
    assert _Indexer().candidate_lengths is None


# ------------------------------------------------------------------ the guard

def test_the_guard_is_gone_when_the_operators_are_there(monkeypatch):
    """The 16k ceiling was never a property of the model: it is what an
    operator that cannot express the mask forces. With ds41 present the
    filter is applied and there is nothing to refuse."""
    from vllm_ascend.models.deepseek_v4_1 import indexer_select as sel

    model = _ModelWithIndexers()
    # ``sparse_indexer`` imported the name, so the module that asks is the
    # module to patch; patching ``indexer_candidate`` alone leaves the
    # already-bound reference in place and the test passes for a wrong reason.
    monkeypatch.setattr(mod, "load_ds41", lambda: True)
    assert sel.collect_candidate_guards(
        model, SimpleNamespace(model_type="deepseek_v41")) is None


def test_without_the_operators_the_ceiling_is_still_enforced(monkeypatch):
    """🔴 The regression guard for N3. Without ds41, a run past the budget
    would draw its selection from the whole context instead of the published
    blocks, and nothing else would say so."""
    from vllm_ascend.models.deepseek_v4_1 import indexer_select as sel

    monkeypatch.setattr(mod, "load_ds41", lambda: False)
    guards = sel.collect_candidate_guards(
        _ModelWithIndexers(), SimpleNamespace(model_type="deepseek_v41"))
    assert guards is not None
    # Only the five layers that take part; 2/8/14 were never under it.
    assert len(guards.entries) == 5
    guards.check(16384, "at the boundary")
    with pytest.raises(NotImplementedError, match="candidate filtering"):
        guards.check(16385, "one past it")


class _ModelWithIndexers:
    """The V4.1 indexer topology: 2/8/14 ratio 2, 20 publishes, 24-36 read."""

    def __init__(self):
        blocks = _blocks()
        self._indexers = []
        for layer in (2, 8, 14):
            self._indexers.append(
                _Indexer(compress_ratio=2,
                         candidate_blocks=None,
                         k_cache=SimpleNamespace(
                             prefix=f"layers.{layer}.indexer.k_cache",
                             kv_cache=None)))
        for layer in (20, 24, 28, 32, 36):
            self._indexers.append(
                _Indexer(candidate_blocks=blocks,
                         candidate_write=(layer == 20),
                         k_cache=SimpleNamespace(
                             prefix=f"layers.{layer}.indexer.k_cache",
                             kv_cache=None)))

    def modules(self):
        return iter(self._indexers)


def test_a_candidate_copy_that_would_cross_the_host_is_refused(
        forward_context, monkeypatch):
    """🔴 Upstream allocates the block buffer with a bare torch.empty and
    lets the construction-time device context place it. If that ever stops
    placing it on the card, these copies become device-to-host -- 8 KiB a
    row plus a sync, every step -- and nothing else would say so."""
    blocks = _blocks()
    indexer = _Indexer(candidate_blocks=blocks, candidate_write=True)

    class _ElsewhereTensor(torch.Tensor):
        """A published tensor that claims to live on another device."""

        @property
        def device(self):
            return torch.device("meta")

    published = torch.ones(TOKENS, BLOCKS,
                           dtype=torch.int32).as_subclass(_ElsewhereTensor)
    monkeypatch.setattr(
        mod, "publish_candidates", lambda **kw:
        (torch.zeros(TOKENS, 1, TOPK, dtype=torch.int32), published,
         torch.zeros(TOKENS, dtype=torch.int32)))

    with pytest.raises(RuntimeError, match="through the host"):
        _run(indexer, forward_context)
