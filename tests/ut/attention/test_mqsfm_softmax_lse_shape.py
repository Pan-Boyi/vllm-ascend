# SPDX-License-Identifier: Apache-2.0
"""The softmax_lse buffer must be shaped from the KV head count, not from the query.

The meta kernel is shape arithmetic, so this runs anywhere -- no device, no CANN operator. Eager is
pinned separately by the operator itself, which rejects a wrongly shaped buffer outright; this pins
meta, so the two cannot drift apart while no card is free.

The rule, from CANN's own output constructors (`cann_ops_transformer/ops/csrc/
mixed_quant_sparse_flash_mla.cpp`, and its `sparse_flash_mla` and `quant_sparse_flash_mla`
siblings, which agree):

    BSND : {q0, n, q1, q2 // n}
    TND  : {n, q0, q1 // n}

with ``n`` the KV head count, read from kv dim 2 for BSND and PA_BBND and from kv dim 1 for the
other TND layouts. It is recomputed below rather than imported, so this is a check rather than a
restatement of the binding.

`return_softmax_lse` defaults to False and has no caller in-tree, which is how the previous rule --
the query's shape with a trailing 1 -- survived: it is wrong for every configuration where the
query and KV head counts differ, and nothing ever asked.
"""

import torch

from tests.ut.base import TestBase


def _expected_lse(q_shape, kv_shape, layout_q, layout_kv):
    n = kv_shape[2] if layout_kv in ("BSND", "PA_BBND") else kv_shape[1]
    if layout_q == "BSND":
        return (q_shape[0], n, q_shape[1], q_shape[2] // n)
    return (n, q_shape[0], q_shape[1] // n)


# Every case has a KV head count different from the query head count: that is precisely where the
# old rule and the correct one diverge, and a case with them equal would pass either way.
CASES = [
    ("TND", "PA_BBND", (4, 16, 512), (2, 128, 1, 608)),
    ("TND", "PA_BBND", (8, 32, 512), (4, 128, 2, 608)),
    ("BSND", "PA_BBND", (2, 4, 16, 512), (2, 128, 1, 608)),
    ("TND", "TND", (4, 16, 512), (4, 2, 608)),
]


class TestMqsfmSoftmaxLseShape(TestBase):
    def _run_meta(self, layout_q, layout_kv, q_shape, kv_shape):
        from torch._subclasses.fake_tensor import FakeTensorMode

        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env()
        import vllm_ascend.vllm_ascend_C  # noqa: F401

        with FakeTensorMode(allow_non_fake_inputs=True):
            q = torch.empty(q_shape, dtype=torch.bfloat16, device="npu")
            kv = torch.empty(kv_shape, dtype=torch.uint8,
                             device="npu").view(torch.float8_e4m3fn)
            tokens = q_shape[0] if layout_q == "TND" else q_shape[0] * q_shape[1]
            _, lse = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla(
                q,
                1,
                ori_kv=kv,
                cu_seqlens_q=torch.empty(tokens + 1, dtype=torch.int32, device="npu"),
                metadata=torch.empty(2048, dtype=torch.int32, device="npu"),
                rope_head_dim=64,
                softmax_scale=0.04,
                cmp_ratio=1,
                ori_mask_mode=0,
                cmp_mask_mode=0,
                ori_win_left=-1,
                ori_win_right=-1,
                layout_q=layout_q,
                layout_kv=layout_kv,
                topk_value_mode=1,
                return_softmax_lse=True,
            )
        return tuple(lse.shape)

    def test_lse_shape_follows_the_kv_head_count(self):
        # subTest rather than pytest.mark.parametrize: TestBase derives from unittest.TestCase,
        # where parametrize silently produces a method that pytest then calls without its
        # arguments -- it fails with a TypeError that says nothing about shapes.
        for layout_q, layout_kv, q_shape, kv_shape in CASES:
            with self.subTest(layout_q=layout_q, layout_kv=layout_kv, q=q_shape, kv=kv_shape):
                self.assertEqual(
                    self._run_meta(layout_q, layout_kv, q_shape, kv_shape),
                    _expected_lse(q_shape, kv_shape, layout_q, layout_kv))

    def test_lse_is_not_the_query_shape(self):
        # The rule this replaced. Kept as its own case so the suite says what it is guarding
        # against, not only what it expects.
        self.assertNotEqual(
            self._run_meta("TND", "PA_BBND", (4, 16, 512), (2, 128, 1, 608)), (4, 16, 1))
