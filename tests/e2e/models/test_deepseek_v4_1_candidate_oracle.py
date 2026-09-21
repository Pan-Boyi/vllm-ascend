# SPDX-License-Identifier: Apache-2.0
"""Do the ds41 candidate operators compute V4.1's two-level selection?

This is the half the unit tests cannot answer. They check that the arguments
are assembled and the buffers passed correctly; whether the kernels then
select the same tokens the model was trained to select is a question only a
card can answer, and "it ran" does not imply it.

The oracle is the reference implementation shipped with the checkpoint
(``dsv41_golden/ref/model.py``), in fp32, on the **dequantised** inputs --
the same bytes the operator gets, read back as numbers. Three properties, in
increasing strength:

1. the **publisher** publishes the reference's block set;
2. the **consumer** selects the reference's positions inside that set;
3. past the budget the two-level answer **differs** from the unfiltered one.

🔴 Property 3 is the bug arm and the only one that can fail for the right
reason. Below 16384 tokens the filter is a no-op *by construction* (every
block fits the budget), so agreement there is not evidence of anything; and
``candidate_topk_blocks`` defaults to -1, at which the operator publishes
nothing, raises nothing and returns a correct unfiltered selection. Without
property 3 this file would pass against an implementation that never
filtered.

🔴 Sets, not index-by-index. Both levels take a top-k over scores that tie --
the reference pins each row's newest block at ``+inf``, and MXFP4 leaves many
exact ties -- and tie order is not part of either contract. Rows whose
boundary score is tied are compared only on the unambiguous part, rather than
softened with a tolerance that would also admit a wrong answer.

⚠️ **This file has never run.** It needs the three ds41 wheels installed in
front of the CANN-bundled package (``cookbook/ds41_install_runbook.md``) and
a free card; at the time of writing all eight are held by another container.
Until it has run, nothing here is evidence about the operators.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest
import torch

CKPT = os.environ.get("DSV41_CKPT", "/workspace/models/DeepSeek-V4.1-Flash")
GOLDEN_REF = os.environ.get(
    "DSV41_GOLDEN_REF",
    "/workspace/code/dsv4_single_card/vllm_project/vllm-ascend/dsv41_golden/"
    "ref/model.py")

#: The kernels take the checkpoint's index shapes and no others, and only
#: these two page sizes were exported as prebuilt variants.
PA_BLOCK_SIZE = 128
#: Query rows per case. Small on purpose: the oracle materialises a dense
#: ``[rows, positions]`` fp32 logit matrix, which at 128k positions is what
#: bounds this test's memory rather than anything on the card.
QUERY_ROWS = 4

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="the ds41 indexer kernels only exist on an Ascend card")


@pytest.fixture(scope="module")
def cfg():
    with open(os.path.join(CKPT, "config.json")) as handle:
        return json.load(handle)["text_config"]


@pytest.fixture(scope="module")
def ds41():
    """The operators, or a skip naming what is missing.

    ⚠ ``import cann_ops_transformer`` succeeding is not evidence: the
    CANN-bundled copies have the same name and no ``ds41`` submodule, and
    they come first on PYTHONPATH unless the wheels were installed the way
    the runbook says.
    """
    from vllm_ascend.models.deepseek_v4_1 import indexer_candidate as cand

    if not cand.load_ds41():
        pytest.skip(
            f"ds41 operators not importable: {cand.ds41_import_error()}")
    return cand


@pytest.fixture(scope="module")
def reference_select_candidate_blocks():
    """The checkpoint's own ``select_candidate_blocks``, imported read-only.

    Reimplementing it here would make this file agree with my reading of the
    filter rather than with the model's definition of it -- and the rule most
    easily lost in a reimplementation (each row's newest block is pinned in
    at ``+inf``, because it is partly filled and would otherwise be outscored
    by an older full block) is also the one whose absence is nearly
    invisible: it changes one block in 2048.
    """
    if not os.path.exists(GOLDEN_REF):
        pytest.skip(f"the reference model is not at {GOLDEN_REF}")
    # ⚠ The reference is a package of flat sibling modules that import each
    # other by bare name (``from engram import ...``), so its own directory
    # has to be importable. Prepended and then removed: leaving it on the
    # path would make ``kernel``, ``vision`` and friends resolvable from
    # anywhere in the test session under names that collide easily.
    reference_dir = os.path.dirname(GOLDEN_REF)
    sys.path.insert(0, reference_dir)
    try:
        spec = importlib.util.spec_from_file_location("dsv41_ref", GOLDEN_REF)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(reference_dir)
    return module.select_candidate_blocks


# ------------------------------------------------------------------ the inputs

class _Case:
    """One prefill request of ``length`` positions, as both sides take it.

    The quantised planes are what the operator reads; ``q_ref``/``k_ref`` are
    those same planes dequantised back to fp32, which is what the oracle
    scores. Generating fp32 and quantising once -- rather than scoring the
    pre-quantisation values -- is what keeps a quantisation difference from
    being read as a selection difference.
    """

    def __init__(self, cfg, length: int, device):
        import torch_npu

        heads = cfg["index_n_heads"]
        width = cfg["index_head_dim"]
        self.length = length
        self.topk = cfg["index_topk"]
        self.topk_blocks = cfg["candidate_topk_blocks"]
        self.block_size = cfg["candidate_block_size"]
        self.ratio = cfg["compress_ratios"][cfg["candidate_source_layer_id"]]
        assert self.ratio == 1, "the candidate layers are the ratio-1 ones"

        generator = torch.Generator(device="cpu").manual_seed(length)
        k_bf16 = torch.randn(length, width,
                             generator=generator).bfloat16().to(device)

        # 🔴 Two independent (query, weights) pairs, because that is what
        # makes the filter mean anything. Every indexer has its own ``wq_b``
        # and ``weights_proj``, so layer 20 publishes blocks under **its**
        # scores and layers 24-36 pick inside them under **theirs**.
        # With one shared pair the filter is provably a no-op: a position in
        # the top ``index_topk`` (512) sits in a block whose maximum is at
        # least its own score, so that block is inside the top
        # ``candidate_topk_blocks`` (2048) by construction. A test built on a
        # shared pair therefore cannot fail, whatever the operator does --
        # which is exactly how this file first read as "the filter changes
        # nothing".
        self.q, self.q_scale, self.weights, self.q_ref = self._make_query(
            heads, width, device, generator, torch_npu)
        (self.consumer_q, self.consumer_q_scale, self.consumer_weights,
         self.consumer_q_ref) = self._make_query(heads, width, device,
                                                 generator, torch_npu)

        packed_k, scale_k = torch_npu.npu_dynamic_mx_quant(
            k_bf16, dst_type=torch_npu.float4_e2m1fn_x2)

        pages = -(-length // PA_BLOCK_SIZE)
        self.k = torch.zeros(pages, PA_BLOCK_SIZE, 1, width // 2,
                             dtype=torch.uint8, device=device)
        self.k_scale = torch.zeros(pages, PA_BLOCK_SIZE, 1, 4,
                                   dtype=torch.uint8, device=device)
        flat_k = self.k.view(-1, width // 2)
        flat_scale = self.k_scale.view(-1, 4)
        flat_k[:length] = packed_k.view(torch.uint8).reshape(length, -1)
        flat_scale[:length] = scale_k.contiguous().view(torch.uint8).reshape(
            length, -1)
        self.k_scale_paired = self.k_scale.view(pages, PA_BLOCK_SIZE, 1, 2, 2)

        self.block_table = torch.arange(pages, dtype=torch.int32,
                                        device=device).reshape(1, pages)
        self.cu_seqlens_q = torch.tensor([0, QUERY_ROWS], dtype=torch.int32,
                                         device=device)
        self.seqused_k = torch.tensor([length], dtype=torch.int32,
                                      device=device)

        # Dequantised, for the oracle. The operator is reading these bytes.
        self.k_ref = _dequantise(
            flat_k[:length],
            flat_scale[:length].reshape(length, 2, 2)).float()

    @staticmethod
    def _make_query(heads, width, device, generator, torch_npu):
        """One indexer's quantised query and weights, plus its dequantised
        twin. Quantising and then reading back -- rather than scoring the
        pre-quantisation values -- is what keeps a quantisation difference
        from being read as a selection difference."""
        q_bf16 = torch.randn(QUERY_ROWS, heads, width,
                             generator=generator).bfloat16().to(device)
        weights = torch.randn(QUERY_ROWS, heads,
                              generator=generator).float().to(device)
        packed, scale = torch_npu.npu_dynamic_mx_quant(
            q_bf16, dst_type=torch_npu.float4_e2m1fn_x2)
        packed = packed.view(torch.uint8).reshape(QUERY_ROWS, heads,
                                                  width // 2)
        scale = scale.contiguous().view(torch.uint8).reshape(
            QUERY_ROWS, heads, width // 64, 2)
        return packed, scale, weights, _dequantise(packed, scale).float()

    def call_args(self, *, consumer: bool = False) -> dict:
        """The publisher's arguments, or the consumer's own query/weights."""
        query = self.consumer_q if consumer else self.q
        q_scale = self.consumer_q_scale if consumer else self.q_scale
        weights = self.consumer_weights if consumer else self.weights
        return dict(query=query, q_scale=q_scale, key_cache=self.k,
                    key_dequant_scale=self.k_scale_paired,
                    weights=weights, index_topk=self.topk,
                    compress_ratio=self.ratio,
                    cu_seqlens_q=self.cu_seqlens_q,
                    seqused_k=self.seqused_k, cmp_residual_k=None,
                    block_table=self.block_table)

    def precision_band(self, *, consumer: bool = False) -> float:
        """How far this case's scores move under a different reduction."""
        q_ref = self.consumer_q_ref if consumer else self.q_ref
        per_head = torch.einsum("qhd,sd->qhs", q_ref, self.k_ref)
        reachable = torch.arange(self.length, device=per_head.device)
        last_position = self.length - QUERY_ROWS + torch.arange(
            QUERY_ROWS, device=per_head.device)
        finite = reachable.unsqueeze(0) <= last_position.unsqueeze(1)
        weights = self.consumer_weights if consumer else self.weights
        return _reduction_precision_band(per_head, weights, finite)

    def reference_logits(self, *, consumer: bool = False) -> torch.Tensor:
        """``[rows, length]`` fp32, relu'd per head then weighted -- the
        reference's ``index_score`` (``ref/model.py:556-557``), with the rows
        placed at the end of the sequence as TND prefill places them, so the
        causal bound each row gets is the operator's."""
        q_ref = self.consumer_q_ref if consumer else self.q_ref
        weights = self.consumer_weights if consumer else self.weights
        per_head = torch.einsum("qhd,sd->qhs", q_ref, self.k_ref)
        scores = (per_head.relu() * weights.unsqueeze(-1)).sum(dim=1)
        reachable = torch.arange(self.length, device=scores.device)
        last_position = self.length - QUERY_ROWS + torch.arange(
            QUERY_ROWS, device=scores.device)
        return scores.masked_fill(
            reachable.unsqueeze(0) > last_position.unsqueeze(1), -torch.inf)

    def compress_lens(self) -> torch.Tensor:
        """One per row: how many compressed positions that row can see."""
        return (self.length - QUERY_ROWS + 1 + torch.arange(
            QUERY_ROWS, device=self.q.device)).unsqueeze(-1)


def _dequantise(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 bytes back to numbers: two E2M1 values a byte, one E8M0 exponent
    per 32 values. Written out rather than taken from a helper because a
    helper that made the same mistake as the code under test would hide it."""
    e2m1 = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0,
         -3.0, -4.0, -6.0], dtype=torch.float32, device=packed.device)
    low = e2m1[(packed & 0x0F).long()]
    high = e2m1[(packed >> 4).long()]
    values = torch.stack((low, high), dim=-1).flatten(-2)
    exponents = scale.reshape(*scale.shape[:-2], -1).long()
    factors = torch.exp2((exponents - 127).float())
    return values.unflatten(-1, (factors.shape[-1], 32)).mul(
        factors.unsqueeze(-1)).flatten(-2)


# ----------------------------------------------------------------- comparisons

def _block_scores(logits: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-block maxima, the reference's level-one score.

    Padded out to a whole block first: a sequence length that is not a
    multiple of 8 is the normal case, not an edge case, and reshaping without
    the pad throws.
    """
    padded = torch.nn.functional.pad(logits, (0, -logits.shape[-1] % block_size),
                                     value=-torch.inf)
    return padded.unflatten(-1, (-1, block_size)).amax(-1)


def _reduction_precision_band(per_head: torch.Tensor,
                              weights: torch.Tensor,
                              finite: torch.Tensor) -> float:
    """How far a score can move purely by reducing it differently.

    🔴 This is what makes an inexact comparison defensible instead of a
    tolerance picked to make the test pass. The kernel states that it reduces
    in BF16 ("VEC1 applies BF16 ReLU, weighted reduction"); the oracle
    reduces in FP32. Both are correct implementations of the same formula, and
    the gap between them is measured here, on this case's own numbers, rather
    than assumed. A selection difference **inside** this band is a
    tie-break; one outside it is a difference in the rule, and that is what
    the assertions below distinguish.

    ⚠ It is a one-sided instrument: inside the band proves nothing about the
    rule, it only fails to disprove it. The rule itself is pinned down by the
    exact assertions -- the candidate count, and the newest-block test.
    """
    fp32 = (per_head.relu() * weights.unsqueeze(-1)).sum(1)
    bf16 = (per_head.relu().bfloat16() *
            weights.bfloat16().unsqueeze(-1)).sum(1).float()
    return float((fp32 - bf16)[finite].abs().max())


def _as_sets(indices: torch.Tensor) -> list[set]:
    return [set(row.tolist()) - {-1} for row in indices.cpu()]


# ------------------------------------------------------------------ properties

@pytest.fixture
def case(cfg, request):
    torch.npu.set_device(0)
    return _Case(cfg, request.param, torch.npu.current_device())


@pytest.mark.parametrize("case", [4096, 16384, 16385, 32768, 131072],
                         indirect=True)
def test_the_published_blocks_are_the_references(ds41, case,
                                                 reference_select_candidate_blocks):
    """Property 1: the publisher publishes the reference's block set.

    Exact on the two things that are the *rule* -- how many blocks are
    published, and that they are the reference's blocks -- and banded on the
    boundary, where a block that scores within reduction noise of the 2048nd
    could legitimately go either way. The band is measured from this case's
    own numbers (see ``_reduction_precision_band``), not chosen.
    """
    _, published, lengths = ds41.publish_candidates(
        **case.call_args(), candidate_topk_blocks=case.topk_blocks,
        candidate_block_size=case.block_size, where="oracle")

    logits = case.reference_logits()
    keep = reference_select_candidate_blocks(logits, case.compress_lens(),
                                             case.topk_blocks,
                                             case.block_size)
    expected = [
        set(row.nonzero().flatten().tolist())
        for row in keep[:, ::case.block_size]
    ]
    got = _as_sets(published.reshape(QUERY_ROWS, -1))
    scores = _block_scores(logits, case.block_size)
    band = case.precision_band()

    for row in range(QUERY_ROWS):
        # The count is exact: it is ``min(ceil(visible/8), 2048)`` and has
        # nothing to do with precision.
        assert len(got[row]) == len(expected[row]), (
            f"row {row}: published {len(got[row])} blocks, the reference "
            f"keeps {len(expected[row])}")
        assert int(lengths.reshape(-1)[row]) == len(expected[row])
        # 🔴 The newest block is excluded here and owned by
        # ``test_the_newest_block_is_always_published``: it is the one rule
        # the operator genuinely differs on, it moves a block whose score is
        # arbitrarily far from the boundary (measured: 32.2 at 131072), and
        # leaving it in would drown the boundary comparison this test is
        # for. Excluding it here is not forgiving it -- the other test fails
        # the day it is fixed.
        newest = (int(case.compress_lens()[row, 0]) - 1) // case.block_size
        disagreement = (got[row] ^ expected[row]) - {newest}
        if not disagreement:
            continue
        threshold = scores[row].sort(descending=True).values[
            len(expected[row]) - 1]
        worst = max(
            abs(float(scores[row, block]) - float(threshold))
            for block in disagreement)
        # Twice the band, and the factor is not slack for its own sake:
        # the band measures **one** alternative reduction (FP32 vs BF16),
        # while the kernel differs from the oracle in at least three places
        # -- where the MXFP4 dequant happens, the Cube's tile-wise FP32
        # accumulation order, and the BF16 relu and weighted reduce. One
        # such difference is a lower bound on the total, so the comparison
        # uses two of them. Measured worst case at 131072 was 1.21 against a
        # 1.03 band.
        assert worst <= 2 * band, (
            f"row {row}: {len(disagreement)} blocks differ from the "
            f"reference and one of them is {worst:.4f} away from the "
            f"{len(expected[row])}th score, outside twice the {band:.4f} "
            "that reducing the same formula differently can move it. That "
            "is a different rule, not a tie-break.")


@pytest.mark.parametrize("case", [16385, 32768, 131072], indirect=True)
def test_the_consumers_selection_matches_the_reference(ds41, case,
                                                       reference_select_candidate_blocks):
    """Property 2: and this is what lifting the ceiling rests on."""
    _, published, lengths = ds41.publish_candidates(
        **case.call_args(), candidate_topk_blocks=case.topk_blocks,
        candidate_block_size=case.block_size, where="oracle")
    selected = ds41.select_within_candidates(
        **case.call_args(consumer=True),
        candidate_indices=published.reshape(QUERY_ROWS, 1, -1),
        candidate_lengths=lengths.reshape(QUERY_ROWS, 1),
        candidate_block_size=case.block_size)

    # 🔴 The mask is the operator's **own** published set, not the
    # reference's. Level one and level two score with different weights, so
    # a block swapped at the boundary of level one -- a legitimate
    # tie-break, worth ~1 in block score there -- can carry a position that
    # is tens of points from the boundary at level two. Masking with the
    # reference's blocks would charge those amplified differences to the
    # consumer, which is not what this test is about; level one is property
    # 1's job. Measured: doing it the other way showed a 57.0 deviation that
    # was entirely a level-one boundary swap.
    keep = torch.zeros_like(case.reference_logits(), dtype=torch.bool)
    for row, blocks in enumerate(_as_sets(published.reshape(QUERY_ROWS,
                                                            -1))):
        for block in blocks:
            start = block * case.block_size
            keep[row, start:start + case.block_size] = True
    logits = case.reference_logits(consumer=True)
    masked = logits.masked_fill(~keep, -torch.inf)
    expected = masked.topk(case.topk, dim=-1)
    got = _as_sets(selected.reshape(QUERY_ROWS, -1))
    band = case.precision_band(consumer=True)

    for row in range(QUERY_ROWS):
        assert len(got[row]) == case.topk, (
            f"row {row}: the consumer returned {len(got[row])} positions, "
            f"not {case.topk}")
        # Same exclusion as property 1, at position granularity: the eight
        # positions of the unpinned newest block.
        newest = (int(case.compress_lens()[row, 0]) - 1) // case.block_size
        newest_positions = set(
            range(newest * case.block_size,
                  (newest + 1) * case.block_size))
        disagreement = (got[row] ^ set(
            expected.indices[row].tolist())) - newest_positions
        if not disagreement:
            continue
        threshold = float(expected.values[row][-1])
        worst = max(
            abs(float(logits[row, position]) - threshold)
            for position in disagreement)
        # 🔴 Two levels of top-k compose, so a block that changed hands at
        # level one moves eight positions at level two. The band is doubled
        # for that, and the count of differing positions is capped: a rule
        # difference shows up as a large fraction, not as a boundary fringe.
        assert worst <= 2 * band, (
            f"row {row}: a selected position is {worst:.4f} from the "
            f"{case.topk}th score, outside twice the {band:.4f} reduction "
            "band")
        assert len(disagreement) <= case.topk // 10, (
            f"row {row}: {len(disagreement)} of {case.topk} positions differ "
            "from the reference -- too many to be boundary tie-breaks")


@pytest.mark.parametrize("case", [16385, 32768], indirect=True)
def test_filtering_changes_the_answer_past_the_budget(ds41, case,
                                                      reference_select_candidate_blocks):
    """Property 3 -- the bug arm.

    🔴 If this fails, the run did not do two-level selection at all, and the
    two properties above passed for no reason: past the budget the filter
    must change at least one row's answer, or it was not applied.
    """
    keep = reference_select_candidate_blocks(case.reference_logits(),
                                             case.compress_lens(),
                                             case.topk_blocks,
                                             case.block_size)
    logits = case.reference_logits(consumer=True)
    unfiltered = logits.topk(case.topk, dim=-1).indices
    filtered = logits.masked_fill(~keep, -torch.inf).topk(case.topk,
                                                          dim=-1).indices
    assert any(
        set(a.tolist()) != set(b.tolist())
        for a, b in zip(unfiltered, filtered)), (
            f"at {case.length} tokens the reference itself selects the same "
            "tokens with and without the filter, so this case cannot tell a "
            "filtered run from an unfiltered one -- the case is wrong, not "
            "the operator")

    _, published, lengths = ds41.publish_candidates(
        **case.call_args(), candidate_topk_blocks=case.topk_blocks,
        candidate_block_size=case.block_size, where="oracle")
    two_level = _as_sets(
        ds41.select_within_candidates(
            **case.call_args(consumer=True),
            candidate_indices=published.reshape(QUERY_ROWS, 1, -1),
            candidate_lengths=lengths.reshape(QUERY_ROWS, 1),
            candidate_block_size=case.block_size).reshape(QUERY_ROWS, -1))
    one_level = _as_sets(unfiltered)
    assert any(a != b for a, b in zip(two_level, one_level)), (
        "the operators returned the unfiltered selection at "
        f"{case.length} tokens: the candidate path did not run")


@pytest.mark.xfail(
    strict=True,
    reason="ds41 pins the newest block only when it is partly filled "
           "(quant_lightning_indexer_dsl.py:1001-1005); the reference and "
           "upstream pin it unconditionally. Measured 2026-09-21, see "
           "DSV4.1_适配方案.md §6.9.7 ②. xfail(strict) so that this turns "
           "into a failure the day the operator is fixed.")
@pytest.mark.parametrize("case", [32768], indirect=True)
def test_the_newest_block_is_always_published(ds41, case):
    """The one rule difference found on a card, isolated from every other.

    The newest block is the one holding the tokens just produced. It is
    usually partly filled, so its block maximum is taken over fewer
    positions and can lose to an older full block -- which is why the
    reference pins it at ``+inf`` (``ref/model.py:604-605``) and upstream
    does the same unconditionally (``candidate_blocks.py:46-50``).

    🔴 Scoring cannot decide this question: in a random case the newest
    block usually wins on merit and the deviation is invisible. Here its
    keys are zeroed, so its score is the lowest there is -- the only thing
    that can keep it in the set is the pin. The rows differ only in how far
    each one can see, so ``visible % 8`` is the variable under test.
    """
    flat_keys = case.k.view(-1, case.k.shape[-1])
    flat_scales = case.k_scale.view(-1, 4)
    flat_keys[case.length - case.block_size:case.length] = 0
    flat_scales[case.length - case.block_size:case.length] = 0

    _, published, _ = ds41.publish_candidates(
        **case.call_args(), candidate_topk_blocks=case.topk_blocks,
        candidate_block_size=case.block_size, where="pin")
    got = _as_sets(published.reshape(QUERY_ROWS, -1))

    missing = []
    for row in range(QUERY_ROWS):
        visible = int(case.compress_lens()[row, 0])
        newest = (visible - 1) // case.block_size
        if newest not in got[row]:
            missing.append((row, visible, visible % case.block_size))
    assert not missing, (
        "the newest block was dropped for rows (row, visible, visible%8): "
        f"{missing}")


def test_the_budget_this_file_calls_the_boundary_is_the_checkpoints(cfg):
    """No literals for the boundary: 2048 blocks x 8 positions x ratio 1.

    Runs without a card on purpose. If the checkpoint ever changes these, the
    parametrisations above are quietly testing the wrong boundary -- 16384
    would no longer be "the last exact length" and 16385 no longer the first
    length that needs the filter.
    """
    source = cfg["candidate_source_layer_id"]
    budget = (cfg["candidate_topk_blocks"] * cfg["candidate_block_size"] *
              cfg["compress_ratios"][source])
    assert budget == 16384, budget
    assert cfg["index_source_layer_ids"][-4:] == [24, 28, 32, 36]
