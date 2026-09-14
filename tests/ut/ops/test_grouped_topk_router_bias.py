# SPDX-License-Identifier: Apache-2.0
"""The correction bias picks experts; it must not scale them.

Three sources agree, and two of them are in this repository:

  ref/model.py:820-822   "the bias picks experts but does not scale them:
                          weights come from the raw scores"
                              indices = (scores + bias).topk(topk)[1]
                              weights = scores.gather(1, indices)

  grouped_topk_router.py the grouped branch, with the same statement as a
  (grouped branch)       comment: "We use biased scores for expert selection
                         but original scores for routing weights"

  config                 topk_method = "noaux_tc", the routing this bias
                         belongs to

🔴 The ungrouped branch used the biased scores as the weights. The bias was
wired, passed, and applied -- to the wrong thing. That is harder to find than
a missing bias, because every "is the bias connected?" check answers yes; the
question has to be "does it reach the weights?", which is the thing it must
NOT do.

⚠️ Found by reading the sibling branch in the same file. Two branches doing
the same job differ only where one of them is wrong, so the difference is
itself a specification.
"""

import pytest
import torch

from vllm_ascend.ops.fused_moe.router.grouped_topk_router import \
    AscendGroupedTopKRouter


def _router(**kw):
    defaults = dict(top_k=2, global_num_experts=8, num_expert_group=None,
                    topk_group=None, use_grouped_topk=False,
                    renormalize=True, scoring_func="sqrtsoftplus",
                    routed_scaling_factor=1.0, e_score_correction_bias=None)
    defaults.update(kw)
    return AscendGroupedTopKRouter(**defaults)


def _reference(logits, bias, top_k, renormalize, scale):
    """ref/model.py:818-826, transcribed."""
    scores = torch.nn.functional.softplus(logits).sqrt()
    biased = scores if bias is None else scores + bias
    ids = biased.topk(top_k, dim=-1)[1]
    weights = scores.gather(1, ids)
    if renormalize and top_k > 1:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * scale, ids


def test_the_bias_does_not_reach_the_weights():
    """🔴 The defect. A bias large enough to change the selection, so that
    "selection moved" and "weights moved" are separately visible -- a small
    bias would hide half of it."""
    torch.manual_seed(0)
    logits = torch.randn(4, 8)
    bias = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0])

    router = _router(e_score_correction_bias=bias)
    weights, ids = router._select_experts(torch.zeros(4, 16), logits)
    want_w, want_ids = _reference(logits, bias, 2, True, 1.0)

    assert torch.equal(ids.to(torch.int64).sort(-1)[0],
                       want_ids.sort(-1)[0]), "the bias must pick the experts"
    torch.testing.assert_close(weights.sort(-1)[0], want_w.sort(-1)[0],
                               rtol=1e-4, atol=1e-4)


def test_the_bias_did_change_the_selection_in_that_test():
    """⚠️ Otherwise the test above passes on a bias too small to matter, and
    proves nothing about the half it exists for."""
    torch.manual_seed(0)
    logits = torch.randn(4, 8)
    bias = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0])
    _, with_bias = _reference(logits, bias, 2, True, 1.0)
    _, without = _reference(logits, None, 2, True, 1.0)
    assert not torch.equal(with_bias.sort(-1)[0], without.sort(-1)[0])


@pytest.mark.parametrize("bias", [None,
                                  torch.tensor([2.0, 0., 0., 0., 0., 0., 0., -2.])])
@pytest.mark.parametrize("renormalize", [True, False])
def test_the_two_branches_agree(bias, renormalize):
    """🔵 The check that actually prevents a recurrence.

    The grouped branch was right and the ungrouped one was wrong for the same
    input. Making disagreement between them fail turns "one of the two got
    edited" into something that goes red, rather than something a reader has
    to notice.
    """
    torch.manual_seed(1)
    logits = torch.randn(3, 8)
    common = dict(top_k=2, global_num_experts=8, renormalize=renormalize,
                  scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
                  e_score_correction_bias=bias)
    flat = _router(num_expert_group=None, topk_group=None,
                   use_grouped_topk=False, **common)
    grouped = _router(num_expert_group=1, topk_group=1,
                      use_grouped_topk=True, **common)

    w_flat, id_flat = flat._select_experts(torch.zeros(3, 16), logits)
    w_grp, id_grp = grouped._select_experts(torch.zeros(3, 16), logits)

    assert torch.equal(id_flat.to(torch.int64).sort(-1)[0],
                       id_grp.to(torch.int64).sort(-1)[0])
    torch.testing.assert_close(w_flat.sort(-1)[0], w_grp.sort(-1)[0],
                               rtol=1e-4, atol=1e-4)


def test_without_a_bias_nothing_changes():
    """The other arm: the fix must not move the no-bias case, which is what
    every model that passes no bias is running today."""
    torch.manual_seed(2)
    logits = torch.randn(5, 8)
    router = _router(routed_scaling_factor=1.5)
    weights, ids = router._select_experts(torch.zeros(5, 16), logits)
    want_w, want_ids = _reference(logits, None, 2, True, 1.5)
    assert torch.equal(ids.to(torch.int64).sort(-1)[0], want_ids.sort(-1)[0])
    torch.testing.assert_close(weights.sort(-1)[0], want_w.sort(-1)[0],
                               rtol=1e-4, atol=1e-4)
