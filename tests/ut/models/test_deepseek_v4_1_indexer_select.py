# SPDX-License-Identifier: Apache-2.0
"""Top-k selection for the V4.1 indexer: what it refuses, and what it passes.

The two halves are here for opposite reasons. The candidate-mask criterion is
arithmetic that decides whether skipping a semantic filter is exact -- it is
true today by a factor of two and stops being true at a longer context, with
no symptom. The argument assembly is transcription, and transcription of an
operator with twenty-one parameters is where a wrong mode or a dropped scale
lives.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.indexer_select import (
    assert_candidate_mask_is_vacuous, candidate_mask_is_vacuous,
    compressed_lengths, select_topk)

# The real checkpoint's values, read from
# /workspace/models/DeepSeek-V4.1-Flash/config.json -> text_config.
TOPK_BLOCKS = 2048
BLOCK = 8
INDEX_TOPK = 512


def test_the_filter_is_vacuous_at_the_context_we_serve():
    """8192 tokens at ratio 1 is 1024 blocks against a budget of 2048."""
    assert candidate_mask_is_vacuous(8192, 1, TOPK_BLOCKS, BLOCK)


def test_it_is_vacuous_by_a_factor_of_two_not_by_a_hair():
    """Worth stating: the margin is what makes 8192 safe, and it is finite.

    Pinned so that halving candidate_topk_blocks, or doubling max_model_len,
    turns this red rather than turning the selection subtly wrong.
    """
    compressed = 8192 // 1
    n_blocks = -(-compressed // BLOCK)
    assert n_blocks == 1024
    assert TOPK_BLOCKS == 2 * n_blocks


def test_the_boundary_is_where_the_arithmetic_says_it_is():
    limit = TOPK_BLOCKS * BLOCK  # 16384 compressed columns
    assert candidate_mask_is_vacuous(limit, 1, TOPK_BLOCKS, BLOCK)
    assert not candidate_mask_is_vacuous(limit + 1, 1, TOPK_BLOCKS, BLOCK)


def test_compression_buys_context_because_the_blocks_are_compressed_columns():
    """The blocks count compressed columns, so ratio 2 doubles the reach.

    Getting this backwards would refuse valid ratio-2 runs, or -- worse -- pass
    ratio-1 runs that are past the boundary.
    """
    limit = TOPK_BLOCKS * BLOCK
    assert candidate_mask_is_vacuous(2 * limit, 2, TOPK_BLOCKS, BLOCK)
    assert not candidate_mask_is_vacuous(2 * limit + 2, 2, TOPK_BLOCKS, BLOCK)


@pytest.mark.parametrize("ratio", [0, 1])
def test_ratio_zero_is_treated_as_one_here_too(ratio):
    """0 means "no compression", which is one state per token, not zero."""
    limit = TOPK_BLOCKS * BLOCK
    assert candidate_mask_is_vacuous(limit, ratio, TOPK_BLOCKS, BLOCK)
    assert not candidate_mask_is_vacuous(limit + 1, ratio, TOPK_BLOCKS, BLOCK)


def test_an_unconfigured_filter_is_vacuous():
    """A checkpoint without the knobs has no filter to skip."""
    assert candidate_mask_is_vacuous(1 << 20, 1, 0, 0)


def test_the_refusal_gives_the_numbers_and_the_way_out():
    with pytest.raises(NotImplementedError) as exc:
        assert_candidate_mask_is_vacuous(32768, 1, TOPK_BLOCKS, BLOCK,
                                         "model.layers.24.self_attn.indexer")
    msg = str(exc.value)
    assert "model.layers.24.self_attn.indexer" in msg
    # The numbers that decide it, not a paraphrase of them.
    assert "4096 blocks" in msg
    assert "2048" in msg
    # And the boundary, so the reader can pick a max_model_len rather than
    # guess one.
    assert "16384" in msg
    assert "max_model_len" in msg


def test_the_vacuous_case_does_not_raise():
    assert_candidate_mask_is_vacuous(8192, 1, TOPK_BLOCKS, BLOCK, "layer") \
        is None


@pytest.mark.parametrize("ratio,expect_used,expect_residual",
                         [(1, [7, 3], [0, 0]), (2, [3, 1], [1, 1]),
                          (0, [7, 3], [0, 0])])
def test_compressed_lengths(ratio, expect_used, expect_residual):
    seq_lens = torch.tensor([7, 3], dtype=torch.int32)
    used, residual = compressed_lengths(seq_lens, ratio)
    assert used.tolist() == expect_used
    assert residual.tolist() == expect_residual
    assert used.dtype == torch.int32 and residual.dtype == torch.int32


def test_compressed_lengths_accepts_a_non_int32_seq_lens():
    """seq_lens is int32 on the Ascend path, but nothing here depends on that.

    A silent int64 would reach the operator as the wrong element width.
    """
    used, residual = compressed_lengths(torch.tensor([9], dtype=torch.int64), 2)
    assert used.dtype == torch.int32 and residual.dtype == torch.int32
    assert used.tolist() == [4] and residual.tolist() == [1]


class _RecordingOp:
    """Stands in for npu_quant_lightning_indexer_v2 and keeps the kwargs."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        t = kwargs["query"]
        return torch.zeros((t.shape[0], 1, kwargs["topk"]),
                           dtype=torch.int32), torch.empty(0)


def _call(op, *, q_scale=None, compress_ratio=2, num_tokens=4, num_heads=32):
    query = torch.zeros((num_tokens, num_heads, 128),
                        dtype=torch.float8_e4m3fn)
    return select_topk(
        query=query,
        q_scale=q_scale,
        key_cache=torch.zeros((2, 64, 128), dtype=torch.float8_e4m3fn),
        key_dequant_scale=torch.zeros((2, 64), dtype=torch.float32),
        weights=torch.zeros((num_tokens, num_heads), dtype=torch.float32),
        index_topk=INDEX_TOPK,
        compress_ratio=compress_ratio,
        cu_seqlens_q=torch.zeros(3, dtype=torch.int32),
        seqused_k=torch.zeros(2, dtype=torch.int32),
        cmp_residual_k=torch.zeros(2, dtype=torch.int32),
        block_table=torch.zeros((2, 4), dtype=torch.int32),
        metadata=torch.zeros(1024, dtype=torch.int32),
        op=op,
    )


def test_the_layout_and_mask_mode_are_the_paged_compressed_ones():
    op = _RecordingOp()
    _call(op)
    (kwargs,) = op.calls
    assert kwargs["layout_q"] == "TND"
    assert kwargs["layout_k"] == "PA_BBND"
    assert kwargs["mask_mode"] == 3
    assert kwargs["return_value"] == 0


def test_the_layers_own_ratio_reaches_the_operator():
    """The whole reason this is not the V4.0 call site.

    V4.0 compresses by 4 everywhere; V4.1 carries 2 on layers 2..19 and 1 on
    20..39. A default of 4 here would run the operator against a cache laid out
    for a different group size and return plausible indices.
    """
    op = _RecordingOp()
    _call(op, compress_ratio=2)
    assert op.calls[-1]["cmp_ratio"] == 2


def test_ratio_zero_reaches_the_operator_as_one():
    """sparse_flash_mla rejects cmp_ratio=0; 0 configured means no compression."""
    op = _RecordingOp()
    _call(op, compress_ratio=0)
    assert op.calls[-1]["cmp_ratio"] == 1


def test_the_quant_mode_is_derived_from_the_tensors_not_defaulted():
    from vllm_ascend.attention.dsa_attn_kv_plan import qli_v2_quant_mode

    op = _RecordingOp()
    _call(op)
    expected = qli_v2_quant_mode(torch.float8_e4m3fn, torch.float32)
    assert op.calls[-1]["quant_mode"] == expected


def test_an_unsupported_dtype_pair_refuses_rather_than_picking_a_mode():
    """The kernel validates the q/k dtype against the mode but not the scale.

    A wrong mode with a plausible q/k dtype therefore runs and returns garbage,
    which is why the mapping raises instead of defaulting.
    """
    op = _RecordingOp()
    with pytest.raises(ValueError, match="no quant_mode"):
        select_topk(
            query=torch.zeros((1, 32, 128), dtype=torch.float8_e4m3fn),
            q_scale=None,
            key_cache=torch.zeros((2, 64, 128), dtype=torch.bfloat16),
            key_dequant_scale=torch.zeros((2, 64), dtype=torch.bfloat16),
            weights=torch.zeros((1, 32), dtype=torch.float32),
            index_topk=INDEX_TOPK,
            compress_ratio=1,
            cu_seqlens_q=torch.zeros(2, dtype=torch.int32),
            seqused_k=torch.zeros(1, dtype=torch.int32),
            cmp_residual_k=torch.zeros(1, dtype=torch.int32),
            block_table=torch.zeros((1, 4), dtype=torch.int32),
            metadata=torch.zeros(1024, dtype=torch.int32),
            op=op,
        )
    assert not op.calls, "nothing may reach the operator once the mode is unknown"


def test_a_folded_q_scale_becomes_ones_not_a_dropped_argument():
    """query_dequant_scale is mandatory in the schema, and V4.1 has none.

    fused_indexer_q_rope_quant folds the per-token q scale into `weights` and
    returns None for the scale. The logit is linear in it, so ones is exact --
    but only because the fold happened. This pins the shape and the value.
    """
    op = _RecordingOp()
    _call(op, q_scale=None, num_tokens=4, num_heads=32)
    scale = op.calls[-1]["query_dequant_scale"]
    assert scale.shape == (4, 32)
    assert torch.all(scale == 1)
    assert scale.dtype == torch.float32, "must match the key scale's dtype"


def test_a_real_q_scale_is_passed_through_untouched():
    """The MXFP4 path does have one, and substituting ones would be silent."""
    op = _RecordingOp()
    given = torch.full((4, 32), 0.5, dtype=torch.float32)
    _call(op, q_scale=given)
    assert op.calls[-1]["query_dequant_scale"] is given
