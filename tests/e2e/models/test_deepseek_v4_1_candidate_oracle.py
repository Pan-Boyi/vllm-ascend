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
    spec = importlib.util.spec_from_file_location("dsv41_ref", GOLDEN_REF)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
        q_bf16 = torch.randn(QUERY_ROWS, heads, width,
                             generator=generator).bfloat16().to(device)
        k_bf16 = torch.randn(length, width,
                             generator=generator).bfloat16().to(device)
        self.weights = torch.randn(QUERY_ROWS, heads,
                                   generator=generator).float().to(device)

        packed_q, scale_q = torch_npu.npu_dynamic_mx_quant(
            q_bf16, dst_type=torch_npu.float4_e2m1fn_x2)
        self.q = packed_q.view(torch.uint8).reshape(QUERY_ROWS, heads,
                                                    width // 2)
        self.q_scale = scale_q.contiguous().view(torch.uint8).reshape(
            QUERY_ROWS, heads, width // 64, 2)
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
        self.q_ref = _dequantise(self.q, self.q_scale).float()
        self.k_ref = _dequantise(
            flat_k[:length],
            flat_scale[:length].reshape(length, 2, 2)).float()

    def call_args(self) -> dict:
        return dict(query=self.q, q_scale=self.q_scale, key_cache=self.k,
                    key_dequant_scale=self.k_scale_paired,
                    weights=self.weights, index_topk=self.topk,
                    compress_ratio=self.ratio,
                    cu_seqlens_q=self.cu_seqlens_q,
                    seqused_k=self.seqused_k, cmp_residual_k=None,
                    block_table=self.block_table)

    def reference_logits(self) -> torch.Tensor:
        """``[rows, length]`` fp32, relu'd per head then weighted -- the
        reference's ``index_score`` (``ref/model.py:556-557``), with the rows
        placed at the end of the sequence as TND prefill places them, so the
        causal bound each row gets is the operator's."""
        per_head = torch.einsum("qhd,sd->qhs", self.q_ref, self.k_ref)
        scores = (per_head.relu() * self.weights.unsqueeze(-1)).sum(dim=1)
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

def _unambiguous(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Rows whose k-th and (k+1)-th best score differ, so the top-k set is
    defined. A tied boundary makes either answer correct and the row is not
    evidence either way."""
    if scores.shape[-1] <= k:
        return torch.ones(scores.shape[0], dtype=torch.bool)
    ordered = scores.sort(dim=-1, descending=True).values
    return ordered[:, k - 1] != ordered[:, k]


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
    """Property 1: the publisher's block set is the reference's."""
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
    block_scores = logits.reshape(QUERY_ROWS, -1, case.block_size).amax(-1)
    defined = _unambiguous(block_scores, case.topk_blocks)
    for row, is_defined in enumerate(defined.tolist()):
        if not is_defined:
            continue
        assert got[row] == expected[row], f"row {row} published other blocks"
        assert int(lengths.reshape(-1)[row]) == len(expected[row])


@pytest.mark.parametrize("case", [16385, 32768, 131072], indirect=True)
def test_the_consumers_selection_matches_the_reference(ds41, case,
                                                       reference_select_candidate_blocks):
    """Property 2: and this is what lifting the ceiling rests on."""
    _, published, lengths = ds41.publish_candidates(
        **case.call_args(), candidate_topk_blocks=case.topk_blocks,
        candidate_block_size=case.block_size, where="oracle")
    selected = ds41.select_within_candidates(
        **case.call_args(),
        candidate_indices=published.reshape(QUERY_ROWS, 1, -1),
        candidate_lengths=lengths.reshape(QUERY_ROWS, 1),
        candidate_block_size=case.block_size)

    logits = case.reference_logits()
    keep = reference_select_candidate_blocks(logits, case.compress_lens(),
                                             case.topk_blocks,
                                             case.block_size)
    masked = logits.masked_fill(~keep, -torch.inf)
    expected = masked.topk(case.topk, dim=-1).indices
    defined = _unambiguous(masked, case.topk)
    got = _as_sets(selected.reshape(QUERY_ROWS, -1))
    for row, is_defined in enumerate(defined.tolist()):
        if not is_defined:
            continue
        assert got[row] == set(expected[row].tolist())


@pytest.mark.parametrize("case", [16385, 32768], indirect=True)
def test_filtering_changes_the_answer_past_the_budget(ds41, case,
                                                      reference_select_candidate_blocks):
    """Property 3 -- the bug arm.

    🔴 If this passes while the two above fail, the operators are wrong. If
    this **fails**, the run did not do two-level selection at all, and the
    other two passed for no reason: past the budget the filter must change
    at least one row's answer, or it was not applied.
    """
    logits = case.reference_logits()
    keep = reference_select_candidate_blocks(logits, case.compress_lens(),
                                             case.topk_blocks,
                                             case.block_size)
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
            **case.call_args(),
            candidate_indices=published.reshape(QUERY_ROWS, 1, -1),
            candidate_lengths=lengths.reshape(QUERY_ROWS, 1),
            candidate_block_size=case.block_size).reshape(QUERY_ROWS, -1))
    one_level = _as_sets(unfiltered)
    assert any(a != b for a, b in zip(two_level, one_level)), (
        "the operators returned the unfiltered selection at "
        f"{case.length} tokens: the candidate path did not run")


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
