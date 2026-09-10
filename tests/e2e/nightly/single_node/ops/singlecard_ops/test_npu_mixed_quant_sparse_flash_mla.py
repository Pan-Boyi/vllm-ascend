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


# --- KV cache layout -------------------------------------------------------------------------
# The docs give each token's field layout (mixed_quant_sparse_flash_mla.md:189, :328):
#   quant_mode 1: rope(64, bf16) | nope(448, fp8_e4m3) | scale(7, bf16) | pad(18B) = 608
#   quant_mode 2: nope(448, fp8_e4m3) | rope(64, bf16) | scale(7, e8m0) | pad(1B)  = 584
# What they do not say is how those fields sit inside a paged block. Under PA_BBND, quant_mode 2
# splits them -- the whole block's KV first, then the whole block's scales -- while quant_mode 1 is
# one contiguous row per token. Laying quant_mode 2 out contiguously produces NaN.
#   csrc reference: csa_block_vector.h:471-476 in CANN's kernel source.

KV_D_BY_QUANT_MODE = {1: 608, 2: 584}
KV_BYTES_PER_TOKEN_QM2 = 576  # nope(448) + rope(64 bf16 = 128)
SCALE_BYTES_PER_TOKEN_QM2 = 8  # 7 e8m0 scales + 1 byte of padding
E8M0_ONE = 127  # e8m0 is a bias-127 exponent byte, so 2**0 is 127, not 1


def _kv_fields(num_tokens):
    torch.manual_seed(11)
    rope = (
        (torch.randn(1, num_tokens, NUM_HEADS_KV, ROPE_HEAD_DIM, dtype=torch.bfloat16) * 0.1)
        .view(torch.uint8)
        .reshape(1, num_tokens, NUM_HEADS_KV, 128)
    )
    nope = (
        (torch.randn(1, num_tokens, NUM_HEADS_KV, 448, dtype=torch.bfloat16) * 0.1)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    return rope, nope


def _build_paged_kv(quant_mode, num_tokens, partitioned):
    """Write each field at its offset. `partitioned` selects the two candidate quant_mode 2 layouts."""
    kv_d = KV_D_BY_QUANT_MODE[quant_mode]
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    rope, nope = _kv_fields(num_tokens)
    buf = torch.zeros(num_blocks, BLOCK_SIZE, NUM_HEADS_KV, kv_d, dtype=torch.uint8)
    flat = buf.view(num_blocks, -1)
    for token in range(num_tokens):
        block, offset = token // BLOCK_SIZE, token % BLOCK_SIZE
        if quant_mode == 1:
            base = offset * NUM_HEADS_KV * kv_d
            flat[block, base : base + 128] = rope[0, token, 0]
            flat[block, base + 128 : base + 576] = nope[0, token, 0]
            flat[block, base + 576 : base + 590] = torch.ones(7, dtype=torch.bfloat16).view(torch.uint8)
        elif partitioned:
            base = offset * NUM_HEADS_KV * KV_BYTES_PER_TOKEN_QM2
            flat[block, base : base + 448] = nope[0, token, 0]
            flat[block, base + 448 : base + 576] = rope[0, token, 0]
            scale_at = BLOCK_SIZE * NUM_HEADS_KV * KV_BYTES_PER_TOKEN_QM2 + offset * SCALE_BYTES_PER_TOKEN_QM2
            flat[block, scale_at : scale_at + 7] = E8M0_ONE
        else:
            base = offset * NUM_HEADS_KV * kv_d
            flat[block, base : base + 448] = nope[0, token, 0]
            flat[block, base + 448 : base + 576] = rope[0, token, 0]
            flat[block, base + 576 : base + 583] = E8M0_ONE
    block_table = torch.arange(num_blocks, dtype=torch.int32).view(1, num_blocks).npu()
    return buf.npu().view(torch.float8_e4m3fn), block_table


def _attend_over_layout(quant_mode, partitioned, num_tokens=256, topk=64):
    torch.manual_seed(11)
    q = torch.randn(1, SEQ_Q, NUM_HEADS_Q, HEAD_DIM, dtype=torch.bfloat16, device="npu") * 0.1
    kv, block_table = _build_paged_kv(quant_mode, num_tokens, partitioned)
    seqused_q = torch.tensor([SEQ_Q], dtype=torch.int32, device="npu")
    seqused_kv = torch.tensor([num_tokens], dtype=torch.int32, device="npu")
    sinks = torch.zeros(NUM_HEADS_Q, dtype=torch.float32, device="npu")
    indices = (
        torch.arange(topk, dtype=torch.int32, device="npu")
        .view(1, 1, 1, topk)
        .expand(1, SEQ_Q, NUM_HEADS_KV, topk)
        .contiguous()
    )
    topk_length = torch.full((1, SEQ_Q, NUM_HEADS_KV), topk, dtype=torch.int32, device="npu")
    dense = dict(
        ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=-1, ori_win_right=-1, layout_q="BSND", layout_kv="PA_BBND"
    )
    metadata = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla_metadata(
        NUM_HEADS_Q,
        NUM_HEADS_KV,
        HEAD_DIM,
        quant_mode,
        None,
        None,
        None,
        seqused_q,
        seqused_kv,
        None,
        None,
        topk_length,
        None,
        1,
        SEQ_Q,
        num_tokens,
        0,
        topk,
        0,
        ROPE_HEAD_DIM,
        1,
        dense["ori_mask_mode"],
        dense["cmp_mask_mode"],
        dense["ori_win_left"],
        dense["ori_win_right"],
        dense["layout_q"],
        dense["layout_kv"],
        True,
        False,
        "npu",
    )
    attn_out, _ = torch.ops._C_ascend.npu_mixed_quant_sparse_flash_mla(
        q,
        quant_mode,
        kv,
        None,
        indices,
        None,
        block_table,
        None,
        None,
        None,
        None,
        seqused_q,
        seqused_kv,
        None,
        None,
        topk_length,
        None,
        sinks,
        metadata,
        ROPE_HEAD_DIM,
        1.0 / math.sqrt(HEAD_DIM),
        1,
        dense["ori_mask_mode"],
        dense["cmp_mask_mode"],
        dense["ori_win_left"],
        dense["ori_win_right"],
        dense["layout_q"],
        dense["layout_kv"],
        1,
        False,
    )
    torch.npu.synchronize()
    return attn_out.float()


@torch.inference_mode()
def test_quant_mode_1_layout_is_contiguous_per_token():
    out = _attend_over_layout(quant_mode=1, partitioned=False)
    assert torch.isfinite(out).all()


@torch.inference_mode()
def test_quant_mode_2_layout_splits_kv_from_scales_within_a_block():
    """The layout the kernel source describes; the same fields laid out contiguously give NaN."""
    out = _attend_over_layout(quant_mode=2, partitioned=True)
    assert torch.isfinite(out).all()


@torch.inference_mode()
def test_quant_mode_2_laid_out_contiguously_does_not_work():
    """Pins the trap: reading the field list as "584 contiguous bytes per token" produces NaN.

    Kept as a test so that anyone tempted to "simplify" the split layout above sees why it is split.
    """
    out = _attend_over_layout(quant_mode=2, partitioned=False)
    assert torch.isnan(out).any()


@torch.inference_mode()
def test_both_quant_modes_agree_on_the_same_underlying_kv():
    """Same rope and nope bytes, different container: the attention output should match."""
    qm1 = _attend_over_layout(quant_mode=1, partitioned=False)
    qm2 = _attend_over_layout(quant_mode=2, partitioned=True)
    torch.testing.assert_close(qm1, qm2, rtol=2e-2, atol=2e-3)
