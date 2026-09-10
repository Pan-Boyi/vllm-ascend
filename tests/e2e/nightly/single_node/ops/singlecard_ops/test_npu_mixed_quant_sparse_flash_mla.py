"""MixedQuantSparseFlashMla binding, checked against CANN's own wrapper for the same operator.

The vendored KvQuantSparseAttnSharedkv accepts only cmp_ratio 1, 4 and 128
(csrc/attention/kv_quant_sparse_attn_sharedkv/op_host/
kv_quant_sparse_attn_sharedkv_check_existance.cpp:154-166), while DeepSeek-V4.1 asks for ratio 2 on
layers 2..19. This operator is Ascend950-only and takes any ratio, which is what these cases pin down.
"""

import importlib.util
import math
import os

import pytest
import torch
import torch_npu  # noqa: F401 -- registers the npu device

# torch.ops.cann_ops_transformer is empty until the package is imported, and CANN 9.1 ships a
# package of the same name that does not provide this operator. Resolving to that one raises a bare
# ImportError which says nothing about why, so check the tree first and skip with a reason.
_ASCEND_HOME = os.environ.get("ASCEND_HOME_PATH", "")
_spec = importlib.util.find_spec("cann_ops_transformer")
if _spec is None or not _spec.origin:
    pytest.skip("cann_ops_transformer is not importable", allow_module_level=True)
if not _ASCEND_HOME or not os.path.realpath(_spec.origin).startswith(os.path.realpath(_ASCEND_HOME)):
    pytest.skip(
        f"cann_ops_transformer resolves to {_spec.origin}, outside ASCEND_HOME_PATH "
        f"({_ASCEND_HOME or 'unset'}); CANN 9.1 has a same-named package without this operator",
        allow_module_level=True,
    )
import cann_ops_transformer  # noqa: E402,F401 -- registers torch.ops.cann_ops_transformer

from vllm_ascend.utils import bootstrap_custom_op_env  # noqa: E402

# A5 has no RUNTIME_CUSTOM_OPS, so enable_custom_op() never imports the extension. Import it here or
# torch.ops._C_ascend has none of these operators. include_vendor_lib is left off deliberately:
# stacking the vendor lib onto LD_LIBRARY_PATH as well as the module's rpath has aborted at exit.
bootstrap_custom_op_env()
import vllm_ascend.vllm_ascend_C  # noqa: E402,F401

# Shapes follow DeepSeek-V4.1-Flash: 64 query heads, 1 kv head, head_dim 512, rope_head_dim 64.
NUM_HEADS_Q = 64
NUM_HEADS_KV = 1
HEAD_DIM = 512
KV_DIM = 608
ROPE_HEAD_DIM = 64
BATCH = 1
SEQ_Q = 16
SEQ_ORI_KV = 2050
BLOCK_SIZE = 128
QUANT_MODE = 1

MASK_KWARGS = dict(
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
    layout_q="BSND",
    layout_kv="PA_BBND",
)


def _scatter_to_paged(kv_bnsd, block_size, num_blocks, max_blocks_per_batch):
    batch, num_kv_heads, seq, dim = kv_bnsd.shape
    expanded = torch.zeros(
        batch,
        num_kv_heads,
        max_blocks_per_batch * block_size,
        dim,
        dtype=kv_bnsd.dtype,
        device=kv_bnsd.device,
    )
    expanded[:, :, :seq, :] = kv_bnsd
    paged = torch.zeros(num_blocks, block_size, num_kv_heads, dim, dtype=kv_bnsd.dtype, device=kv_bnsd.device)
    block_table = torch.zeros(batch, max_blocks_per_batch, dtype=torch.int32, device=kv_bnsd.device)
    block_id = 0
    for b in range(batch):
        for i in range(max_blocks_per_batch):
            block_table[b, i] = block_id
            for h in range(num_kv_heads):
                paged[block_id, :, h, :] = expanded[b, h, i * block_size : (i + 1) * block_size, :]
            block_id += 1
    return paged, block_table


def _make_inputs(cmp_ratio):
    torch.manual_seed(1024)
    seq_cmp_kv = SEQ_ORI_KV // cmp_ratio
    inputs = {
        "seqused_q": torch.tensor([SEQ_Q], dtype=torch.int32, device="npu"),
        "seqused_ori_kv": torch.tensor([SEQ_ORI_KV], dtype=torch.int32, device="npu"),
        "seqused_cmp_kv": torch.tensor([seq_cmp_kv], dtype=torch.int32, device="npu"),
        # Constrained to be smaller than cmp_ratio; it recovers the pre-compression KV length.
        "cmp_residual_kv": torch.tensor([SEQ_ORI_KV % cmp_ratio], dtype=torch.int32, device="npu"),
        "q": torch.randn(BATCH, SEQ_Q, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="npu"),
        "sinks": torch.zeros(NUM_HEADS_Q, dtype=torch.float32, device="npu"),
        "cmp_ratio": cmp_ratio,
    }
    ori_blocks = math.ceil(SEQ_ORI_KV / BLOCK_SIZE)
    inputs["ori_kv"], inputs["ori_block_table"] = _scatter_to_paged(
        torch.randn(BATCH, NUM_HEADS_KV, SEQ_ORI_KV, KV_DIM, dtype=torch.bfloat16, device="npu").to(
            torch.float8_e4m3fn
        ),
        BLOCK_SIZE,
        ori_blocks * BATCH,
        ori_blocks,
    )
    cmp_blocks = math.ceil(seq_cmp_kv / BLOCK_SIZE)
    inputs["cmp_kv"], inputs["cmp_block_table"] = _scatter_to_paged(
        torch.randn(BATCH, NUM_HEADS_KV, seq_cmp_kv, KV_DIM, dtype=torch.bfloat16, device="npu").to(
            torch.float8_e4m3fn
        ),
        BLOCK_SIZE,
        cmp_blocks * BATCH,
        cmp_blocks,
    )
    return inputs


def _run_via_cann_wrapper(t):
    metadata = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata(
        num_heads_q=NUM_HEADS_Q,
        num_heads_kv=NUM_HEADS_KV,
        head_dim=HEAD_DIM,
        quant_mode=QUANT_MODE,
        seqused_q=t["seqused_q"],
        seqused_ori_kv=t["seqused_ori_kv"],
        seqused_cmp_kv=t["seqused_cmp_kv"],
        cmp_residual_kv=t["cmp_residual_kv"],
        batch_size=BATCH,
        max_seqlen_q=int(t["seqused_q"].max()),
        max_seqlen_ori_kv=int(t["seqused_ori_kv"].max()),
        max_seqlen_cmp_kv=int(t["seqused_cmp_kv"].max()),
        ori_topk=0,
        cmp_topk=0,
        rope_head_dim=ROPE_HEAD_DIM,
        cmp_ratio=t["cmp_ratio"],
        has_ori_kv=True,
        has_cmp_kv=True,
        **MASK_KWARGS,
    )
    attn_out, _ = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
        q=t["q"],
        ori_kv=t["ori_kv"],
        cmp_kv=t["cmp_kv"],
        ori_block_table=t["ori_block_table"],
        cmp_block_table=t["cmp_block_table"],
        seqused_q=t["seqused_q"],
        seqused_ori_kv=t["seqused_ori_kv"],
        seqused_cmp_kv=t["seqused_cmp_kv"],
        cmp_residual_kv=t["cmp_residual_kv"],
        sinks=t["sinks"],
        metadata=metadata,
        quant_mode=QUANT_MODE,
        rope_head_dim=ROPE_HEAD_DIM,
        softmax_scale=1.0 / math.sqrt(HEAD_DIM),
        cmp_ratio=t["cmp_ratio"],
        topk_value_mode=1,
        return_softmax_lse=False,
        **MASK_KWARGS,
    )
    return attn_out


def _run_via_binding(t):
    metadata = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla_metadata(
        NUM_HEADS_Q,
        NUM_HEADS_KV,
        HEAD_DIM,
        QUANT_MODE,
        None,
        None,
        None,
        t["seqused_q"],
        t["seqused_ori_kv"],
        t["seqused_cmp_kv"],
        t["cmp_residual_kv"],
        None,
        None,
        BATCH,
        int(t["seqused_q"].max()),
        int(t["seqused_ori_kv"].max()),
        int(t["seqused_cmp_kv"].max()),
        0,
        0,
        ROPE_HEAD_DIM,
        t["cmp_ratio"],
        MASK_KWARGS["ori_mask_mode"],
        MASK_KWARGS["cmp_mask_mode"],
        MASK_KWARGS["ori_win_left"],
        MASK_KWARGS["ori_win_right"],
        MASK_KWARGS["layout_q"],
        MASK_KWARGS["layout_kv"],
        True,
        True,
        "npu",
    )
    attn_out, _ = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla(
        t["q"],
        QUANT_MODE,
        t["ori_kv"],
        t["cmp_kv"],
        None,
        None,
        t["ori_block_table"],
        t["cmp_block_table"],
        None,
        None,
        None,
        t["seqused_q"],
        t["seqused_ori_kv"],
        t["seqused_cmp_kv"],
        t["cmp_residual_kv"],
        None,
        None,
        t["sinks"],
        metadata,
        ROPE_HEAD_DIM,
        1.0 / math.sqrt(HEAD_DIM),
        t["cmp_ratio"],
        MASK_KWARGS["ori_mask_mode"],
        MASK_KWARGS["cmp_mask_mode"],
        MASK_KWARGS["ori_win_left"],
        MASK_KWARGS["ori_win_right"],
        MASK_KWARGS["layout_q"],
        MASK_KWARGS["layout_kv"],
        1,
        False,
    )
    return attn_out


@torch.inference_mode()
@pytest.mark.parametrize("cmp_ratio", [1, 2, 4, 128])
def test_binding_matches_cann_wrapper(cmp_ratio):
    """The binding only forwards arguments, so it must reproduce CANN's wrapper exactly."""
    t = _make_inputs(cmp_ratio)
    expected = _run_via_cann_wrapper(t)
    actual = _run_via_binding(t)
    torch.npu.synchronize()
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected), (
        f"cmp_ratio={cmp_ratio}: max abs diff {(actual.float() - expected.float()).abs().max().item():.3e}"
    )


@torch.inference_mode()
def test_accepts_the_ratio_v41_needs():
    """DeepSeek-V4.1 compresses layers 2..19 by 2, which the vendored sharedkv operator rejects."""
    actual = _run_via_binding(_make_inputs(2))
    torch.npu.synchronize()
    assert actual.shape == (BATCH, SEQ_Q, NUM_HEADS_Q, HEAD_DIM)
    assert torch.isfinite(actual.float()).all()
