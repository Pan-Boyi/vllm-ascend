# SPDX-License-Identifier: Apache-2.0
"""The two-call form of V4.1's per-row-bias MoE routing.

🔴 What this replaces and why it is two calls.

V4.1 needs a bias chosen **per token id** -- ``bias_vl`` for image rows,
``e_score_correction_bias`` for text rows. The fused operator this tree ships
with takes one global ``bias``, which is why all 40 MoE layers fell back to a
torch decomposition costing 702 kernels / 1.292 ms a step, paid on every step
including pure-text decode.

The operator is **per-row independent**, so running it once per bias table and
picking rows with ``where`` is not an approximation of per-row bias -- it is
per-row bias. These tests pin that the two calls get the two tables the right
way round, that the mask is the right width, and that the load-bearing
literals reach the operator.

They cannot check the arithmetic: the operator has no CPU implementation.
That was measured separately on card (2026-09-22, e=384/k=6, T=1/7/64,
**including mixed masks**): identical topk_ids, weights within 6e-8 absolute
and 2.3e-7 relative against the torch form.

⚪ A 6-input build of the same OpType (``additional_bias`` +
``additional_token_mask``) would make this one call. It is not used: the
kernel/binary-config layer does not fall through per operator, so adding that
vendor masked the production vendor's other 27 operators and the service
failed to start. See the perf report's 3.3.
"""

from unittest import mock

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.router import fused_topk_router as mod


class TestFusedVisionSelect(TestBase):

    E, K = 8, 3
    LO = 129264

    def _spy(self):
        """Record every operator call and return plausible outputs."""
        calls = []

        def _op(**kw):
            calls.append(kw)
            T = kw["x"].shape[0]
            # Distinguish the two calls by the bias they carry, so the
            # ``where`` below can be checked for picking the right one.
            tag = float(kw["bias"][0])
            return (torch.full((T, self.K), tag),
                    torch.full((T, self.K), int(tag), dtype=torch.int32),
                    torch.zeros(1))

        return calls, _op

    def _run(self, ids, text_bias=None, bias_vl=None):
        calls, _op = self._spy()
        # bias[0] is the tag: 1.0 for text, 2.0 for vision.
        text = torch.ones(self.E) if text_bias is None else text_bias
        vision = torch.full((self.E,), 2.0) if bias_vl is None else bias_vl
        with mock.patch.object(torch.ops._C_ascend,
                               "moe_gating_top_k_hash", _op, create=True):
            w, i = mod.select_deepseek_v4_vision_experts_fused(
                router_logits=torch.randn(ids.shape[0], self.E),
                input_ids=ids, bias_vl=vision, text_bias=text,
                top_k=self.K, routed_scaling_factor=1.5,
                image_sentinel_lo=self.LO)
        return calls, w, i

    def test_the_operator_is_called_once_per_bias_table(self):
        calls, _, _ = self._run(torch.tensor([42], dtype=torch.int64))
        assert len(calls) == 2, len(calls)
        assert float(calls[0]["bias"][0]) == 1.0   # text first
        assert float(calls[1]["bias"][0]) == 2.0   # vision second

    def test_image_rows_take_the_vision_result_and_text_rows_the_text_one(self):
        # 🔴 The assertion the whole change rests on. Getting the two
        # branches of the where backwards is a silent routing change: same
        # shapes, same dtypes, different experts.
        ids = torch.tensor([42, self.LO, 7, self.LO + 4], dtype=torch.int64)
        _, w, i = self._run(ids)
        assert w[:, 0].tolist() == [1.0, 2.0, 1.0, 2.0]
        assert i[:, 0].tolist() == [1, 2, 1, 2]

    def test_the_mask_is_five_ids_wide(self):
        # [lo, lo+5) is the width the upstream router uses. Reading it as 2
        # (upstream's engram/VL mask, a deliberately different rule) would
        # route three reserved placeholder tokens through the text bias.
        ids = torch.tensor([self.LO - 1, self.LO, self.LO + 4, self.LO + 5],
                           dtype=torch.int64)
        _, w, _ = self._run(ids)
        assert w[:, 0].tolist() == [1.0, 2.0, 2.0, 1.0]

    def test_an_all_text_batch_never_takes_the_vision_branch(self):
        # The steady state: pure-text decode. It still costs two calls, which
        # is the price of not branching on data under graph capture.
        ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        calls, w, _ = self._run(ids)
        assert len(calls) == 2
        assert w[:, 0].tolist() == [1.0, 1.0, 1.0]

    def test_a_missing_text_bias_becomes_zeros_not_none(self):
        # The operator has no "no bias" mode on this path: None would be a
        # TypeError, and zeros are what "no correction" means. Goes around
        # ``_run`` because that helper substitutes a tagged bias.
        calls, _op = self._spy()
        with mock.patch.object(torch.ops._C_ascend,
                               "moe_gating_top_k_hash", _op, create=True):
            mod.select_deepseek_v4_vision_experts_fused(
                router_logits=torch.randn(2, self.E),
                input_ids=torch.tensor([1, 2], dtype=torch.int64),
                bias_vl=torch.full((self.E,), 2.0), text_bias=None,
                top_k=self.K, routed_scaling_factor=1.0,
                image_sentinel_lo=self.LO)
        assert len(calls) == 2
        text_call = calls[0]["bias"]
        assert text_call is not None
        assert text_call.shape == (self.E,)
        assert torch.count_nonzero(text_call) == 0

    def test_the_bias_dtype_matches_x(self):
        # csrc enforces bias.scalar_type() == x.scalar_type() with a
        # TORCH_CHECK, so a mismatched bias throws rather than converting.
        calls, _, _ = self._run(torch.tensor([1, 2], dtype=torch.int64),
                                text_bias=torch.ones(self.E,
                                                     dtype=torch.float64),
                                bias_vl=torch.full((self.E,), 2.0,
                                                   dtype=torch.float64))
        for kw in calls:
            assert kw["bias"].dtype == kw["x"].dtype

    def test_the_hash_table_arguments_stay_none(self):
        # tid2eid is None on every V4.1 layer, and the tiling requires
        # input_ids and tid2eid to be given or omitted together.
        calls, _, _ = self._run(torch.tensor([1], dtype=torch.int64))
        for kw in calls:
            assert kw["input_ids"] is None and kw["tid2eid"] is None

    def test_the_load_bearing_literals(self):
        # subTest rather than pytest.parametrize: TestBase is unittest-based
        # and parametrize is silently ignored there.
        # norm_type=2 is SOFTPLUS *and* selects the in-kernel normalisation;
        # changing it to 0 quietly switches the scoring function to softmax.
        calls, _, _ = self._run(torch.tensor([1], dtype=torch.int64))
        for name, value in (("renorm", 0), ("norm_type", 2),
                            ("group_select_mode", 1), ("out_flag", False),
                            ("routed_scaling_factor", 1.5)):
            with self.subTest(argument=name):
                assert calls[0][name] == value
                assert calls[1][name] == value
