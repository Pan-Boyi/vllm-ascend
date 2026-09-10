"""Pure-PyTorch CPU replacements for the six tilelang kernels DeepSeek-V4.1-Flash needs.

Drop-in for ``inference/kernel.py``: same module-level names, same signatures, same
tensor layouts.  Nothing here touches CUDA, tilelang or torch_npu.

Design contract (read this before trusting a number that comes out of here)
--------------------------------------------------------------------------
This is a *numerical golden*, not a bit-exact simulator of the GPU kernel.

1. **Quantization semantics are reproduced exactly.**  Which fp8/fp4 code a value
   lands on, how the block scale is derived, where the clamps sit and what the
   ``amax`` floors are: all of that is a discrete decision that defines the
   arithmetic, so it is mirrored 1:1 from ``kernel.py``.  Getting it "more
   accurate" here would mean computing a *different* function.

2. **Reductions are made more accurate, not bit-identical.**  Both GEMMs
   dequantize each side to fp32 and run one dense fp32 matmul, instead of the
   kernel's per-32/128-block partial-sum-then-rescale.  The two agree
   mathematically; the fp32 path just carries less accumulation error.  Set
   ``GOLDEN_DOT_DTYPE = torch.float64`` for an even tighter reference.

   Note that the dequantization itself is *exact*: an e8m0 scale is a power of
   two and an e4m3 value has 4 mantissa bits, so ``fp8_value * scale`` is
   representable in fp32 with no rounding at all.  Every bit of GEMM error is
   accumulation error, and fp32/fp64 accumulation strictly dominates the
   kernel's.

3. **Softmax and Sinkhorn run in fp32** with the same max-subtraction and the
   same ``eps`` placement as the kernel, including the deliberately finite
   ``-1e30`` score floor that makes an all-masked attention row return zeros
   instead of NaN.

Faithfully reproduced quirk
---------------------------
``act_quant(round_scale=False, scale_dtype=e8m0)`` quantizes the activations with
the *unrounded* fp32 scale but stores the *rounded* e8m0 one, so the returned
scale does not reconstruct the returned values.  That is what the tilelang
kernel does.  ``model.py`` always passes ``scale_fmt="ue8m0"`` so it never hits
this, but the behaviour is kept rather than silently "fixed" -- see
``test_kernel_cpu.py::test_act_quant_unrounded_scale_storage_quirk``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "act_quant",
    "fp4_act_quant",
    "fp8_gemm",
    "fp4_gemm",
    "hc_split_sinkhorn",
    "sparse_attn",
    "fp4_unpack_to_float",
    "dequant_fp8_weight",
    "dequant_fp4_weight",
    "FP4_TABLE",
]

# Accumulation dtype for the two GEMMs.  fp32 by default; float64 for a tighter
# reference (~8x slower, ~2x the memory).
GOLDEN_DOT_DTYPE = torch.float32

FP8_MIN, FP8_MAX = -448.0, 448.0
FP4_MAX = 6.0

# e2m1 code -> value.  Codes 0..7 positive, 8..15 the sign-flipped mirror.
# Matches the FP4_TABLE in inference/convert.py.
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
# The 8 positive magnitudes, ascending -- the grid |x| is rounded onto.
_FP4_GRID = FP4_TABLE[:8].clone()


# --------------------------------------------------------------------------- #
# scale helpers                                                               #
# --------------------------------------------------------------------------- #
def _fast_round_scale(amax: torch.Tensor, max_inv: float) -> torch.Tensor:
    """``2 ** ceil(log2(amax * max_inv))``, via the same IEEE-754 bit trick the
    kernel uses (``fast_pow2(fast_log2_ceil(...))``).

    Done with integer ops rather than ``log2``/``ceil`` so the result is exactly
    the kernel's, with no chance of a floating-point log landing on the wrong
    side of an integer boundary.
    """
    x = (amax.float() * max_inv).contiguous()
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    e = exp - 127 + (mantissa != 0).to(torch.int32)  # ceil(log2(x))
    return ((e + 127) << 23).view(torch.float32)


def _cast_scale(s: torch.Tensor, scale_dtype: torch.dtype) -> torch.Tensor:
    """Store a fp32 scale in the requested scale dtype (fp32 / e8m0 / e4m3)."""
    if scale_dtype == torch.float32:
        return s.float()
    return s.float().to(scale_dtype)


def _blockify(x: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Size]:
    """[..., N] -> [..., N // block_size, block_size]."""
    assert x.size(-1) % block_size == 0, (x.shape, block_size)
    return x.unflatten(-1, (-1, block_size)), x.shape


# --------------------------------------------------------------------------- #
# fp8 / fp4 element casts                                                     #
# --------------------------------------------------------------------------- #
def _to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even into e4m3fn.

    The caller must have clamped to +-448 already: torch's CPU cast returns NaN
    on overflow rather than saturating, exactly like the hardware cvt the kernel
    relies on being fed pre-clamped values.
    """
    return x.float().to(torch.float8_e4m3fn)


def _round_to_fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """Round a pre-clamped fp32 tensor onto the e2m1 grid, returning uint8 codes.

    Ties go to the even code index, which for e2m1 is the even-mantissa value --
    i.e. IEEE round-to-nearest-even on the target format, matching the hardware
    cvt.  torch has no CPU cast for ``float4_e2m1fn_x2`` (it is storage-only), so
    this is done explicitly.
    """
    x = x.float()
    sign = torch.signbit(x)
    a = x.abs().clamp(0.0, FP4_MAX)

    grid = _FP4_GRID.to(a.device)
    hi = torch.searchsorted(grid, a.contiguous())          # first idx with grid[idx] >= a
    hi = hi.clamp(0, grid.numel() - 1)
    lo = (hi - 1).clamp_min(0)

    d_lo = a - grid[lo]
    d_hi = grid[hi] - a
    # tie -> even index; otherwise the strictly nearer neighbour
    take_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    idx = torch.where(take_hi, hi, lo).to(torch.uint8)
    return idx | (sign.to(torch.uint8) << 3)


def _fp4_codes_to_float(codes: torch.Tensor) -> torch.Tensor:
    return FP4_TABLE.to(codes.device)[codes.long()]


def _pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    """[..., N] uint8 codes -> [..., N // 2] packed, low nibble first.

    Low nibble holds the even element, high nibble the odd one -- the layout
    ``convert.py`` unpacks with ``low = x & 0x0F; high = (x >> 4) & 0x0F``.
    """
    assert codes.size(-1) % 2 == 0
    pair = codes.unflatten(-1, (-1, 2))
    return (pair[..., 0] & 0x0F) | ((pair[..., 1] & 0x0F) << 4)


def fp4_unpack_to_float(packed: torch.Tensor, out_dtype=torch.float32) -> torch.Tensor:
    """[..., K // 2] packed fp4 (``float4_e2m1fn_x2`` or uint8/int8) -> [..., K] float."""
    if packed.dtype == torch.float4_e2m1fn_x2:
        packed = packed.view(torch.uint8)
    elif packed.dtype == torch.int8:
        packed = packed.view(torch.uint8)
    assert packed.dtype == torch.uint8, packed.dtype
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    table = FP4_TABLE.to(packed.device).to(out_dtype)
    return torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(-2)


# --------------------------------------------------------------------------- #
# 1. act_quant                                                                #
# --------------------------------------------------------------------------- #
def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise fp8 (e4m3) quantization along the last dim.

    ``x``  : [..., N]        (bf16 in practice)
    returns ``(y, s)`` with ``y`` [..., N] e4m3fn and ``s`` [..., N // block_size],
    or -- when ``inplace`` -- writes ``dequant(quant(x))`` back into ``x`` and
    returns ``x``.

    ``scale_fmt`` not None rounds the scale up to a power of two (MXFP style).
    """
    N = x.size(-1)
    assert N % block_size == 0, (x.shape, block_size)
    round_scale = scale_fmt is not None

    xb, _ = _blockify(x.float(), block_size)                    # [..., G, B]
    amax = xb.abs().amax(dim=-1).clamp_min(1e-4)                # [..., G]
    if round_scale:
        s = _fast_round_scale(amax, 1.0 / FP8_MAX)
    else:
        s = amax * (1.0 / FP8_MAX)

    q = _to_fp8_e4m3(torch.clamp(xb / s.unsqueeze(-1), FP8_MIN, FP8_MAX))

    if inplace:
        deq = q.float() * s.unsqueeze(-1)
        x.copy_(deq.flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), _cast_scale(s, scale_dtype)


# --------------------------------------------------------------------------- #
# 2. fp4_act_quant                                                            #
# --------------------------------------------------------------------------- #
def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
):
    """Block-wise fp4 (e2m1) quantization along the last dim.

    Two scale flavours, and they differ in more than storage:

    * ``e8m0`` (the indexer): scale is ``2 ** ceil(log2(amax / 6))``, with
      ``amax`` floored at ``6 * 2**-126``.
    * ``e4m3`` (the compressed KV): scale is ``amax / 6`` *rounded to e4m3*, with
      ``amax`` floored at ``6 * 2**-9`` so an all-zero group still gets a nonzero
      scale -- the training kernel's convention.

    Returns ``(y, s)`` with ``y`` packed [..., N // 2] ``float4_e2m1fn_x2``, or
    writes the dequantized values back into ``x`` when ``inplace``.
    """
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn), scale_dtype
    N = x.size(-1)
    assert N % block_size == 0, (x.shape, block_size)

    xb, _ = _blockify(x.float(), block_size)
    amax = xb.abs().amax(dim=-1)

    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(FP4_MAX * (2.0**-9))
        s = _to_fp8_e4m3(amax / FP4_MAX).float()      # round-trip through e4m3
    else:
        amax = amax.clamp_min(FP4_MAX * (2.0**-126))
        s = _fast_round_scale(amax, 1.0 / FP4_MAX)

    codes = _round_to_fp4_codes(torch.clamp(xb / s.unsqueeze(-1), -FP4_MAX, FP4_MAX))

    if inplace:
        deq = _fp4_codes_to_float(codes) * s.unsqueeze(-1)
        x.copy_(deq.flatten(-2).to(x.dtype))
        return x

    packed = _pack_fp4_codes(codes.flatten(-2)).view(torch.float4_e2m1fn_x2)
    return packed, _cast_scale(s, scale_dtype)


# --------------------------------------------------------------------------- #
# weight dequantization helpers (shared by the two GEMMs)                     #
# --------------------------------------------------------------------------- #
def dequant_fp8_weight(w: torch.Tensor, w_s: torch.Tensor, block_size: int) -> torch.Tensor:
    """[N, K] e4m3 with a [ceil(N/bs), ceil(K/bs)] 2-D block scale -> fp32 [N, K].

    Exact: an e8m0 scale is a power of two and e4m3 has 4 mantissa bits.
    """
    N, K = w.shape
    # repeat_interleave rather than a reshape: N or K need not be a multiple of bs
    s = w_s.float().repeat_interleave(block_size, dim=0)
    s = s.repeat_interleave(block_size, dim=1)
    return w.float() * s[:N, :K]


def dequant_fp4_weight(w: torch.Tensor, w_s: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """[N, K//2] packed e2m1 with a [N, K/32] e8m0 scale -> fp32 [N, K]."""
    vals = fp4_unpack_to_float(w, torch.float32)          # [N, K]
    K = vals.size(-1)
    s = w_s.float().repeat_interleave(block_size, dim=-1)[..., :K]
    return vals * s


def _dequant_act(a: torch.Tensor, a_s: torch.Tensor, block_size: int) -> torch.Tensor:
    """[M, K] e4m3 with a [M, K/bs] per-row-per-block scale -> fp32 [M, K]. Exact."""
    K = a.size(-1)
    s = a_s.float().repeat_interleave(block_size, dim=-1)[..., :K]
    return a.float() * s


# --------------------------------------------------------------------------- #
# 3. fp8_gemm                                                                 #
# --------------------------------------------------------------------------- #
def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """``C[M, N] = A_fp8[M, K] @ B_fp8[N, K]^T`` with per-block scaling.

    ``a_s``: [M, K/bs] (per row, per K block).  ``b_s``: [ceil(N/bs), K/bs].
    Output takes ``torch.get_default_dtype()``, as the kernel's does.
    """
    assert block_size in (32, 128), block_size
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % block_size == 0, (K, block_size)
    assert a_s.numel() == M * (K // block_size), (a_s.shape, M, K, block_size)
    assert tuple(b_s.shape) == ((N + block_size - 1) // block_size, K // block_size), b_s.shape

    acc = GOLDEN_DOT_DTYPE
    a_deq = _dequant_act(a.reshape(M, K), a_s.reshape(M, -1), block_size).to(acc)
    b_deq = dequant_fp8_weight(b, b_s, block_size).to(acc)

    c = a_deq @ b_deq.T
    return c.view(*a.size()[:-1], N).to(torch.get_default_dtype())


# --------------------------------------------------------------------------- #
# 4. fp4_gemm                                                                 #
# --------------------------------------------------------------------------- #
def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """``C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T``.

    Activations carry a per-``act_block_size`` scale, weights a per-32 e8m0 one.
    ``b`` is [N, K//2] ``float4_e2m1fn_x2``, two values per byte packed along K.
    """
    assert act_block_size in (32, 128), act_block_size
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % act_block_size == 0, (K, act_block_size)
    assert a_s.numel() == M * (K // act_block_size), (a_s.shape, M, K, act_block_size)
    assert tuple(b_s.shape) == (N, K // 32), (b_s.shape, N, K)

    acc = GOLDEN_DOT_DTYPE
    a_deq = _dequant_act(a.reshape(M, K), a_s.reshape(M, -1), act_block_size).to(acc)
    b_deq = dequant_fp4_weight(b, b_s, 32).to(acc)

    c = a_deq @ b_deq.T
    return c.view(*a.size()[:-1], N).to(torch.get_default_dtype())


# --------------------------------------------------------------------------- #
# 5. hc_split_sinkhorn                                                        #
# --------------------------------------------------------------------------- #
def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Split one [b, s, (2 + hc) * hc] projection into the three Hyper-Connection
    coefficient sets, making ``comb`` doubly stochastic by Sinkhorn.

    Layout of the last axis: ``[0:hc]`` -> pre, ``[hc:2*hc]`` -> post,
    ``[2*hc:]`` -> comb (row-major hc x hc).

    Iteration 1's row step is a *softmax* (then ``+ eps``); the remaining
    ``sinkhorn_iters - 1`` rounds are plain row- then column-normalizations with
    ``eps`` in the denominator.  Column normalization runs on every round,
    iteration 1 included.  All fp32.
    """
    b, s, mix_hc = mixes.shape
    hc = hc_mult
    assert mix_hc == (2 + hc) * hc, (mix_hc, hc)

    m = mixes.float()
    scale = hc_scale.float()
    base = hc_base.float()

    pre = torch.sigmoid(m[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc: 2 * hc] * scale[1] + base[hc: 2 * hc])

    comb = m[..., 2 * hc:] * scale[2] + base[2 * hc:]
    comb = comb.view(b, s, hc, hc)

    # iteration 1: softmax over rows, + eps, then column normalize
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


# --------------------------------------------------------------------------- #
# 6. sparse_attn                                                              #
# --------------------------------------------------------------------------- #
# Queries processed per chunk, to bound the [chunk, h, topk] score tensor.
SPARSE_ATTN_Q_CHUNK = 64


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Gather-then-attend over the per-query top-k KV slots, with an attention sink.

    ``q``          [b, m, h, d]     bf16
    ``kv``         [b, n, d]        bf16 -- used as *both* K and V (latent attention)
    ``attn_sink``  [h]              fp32
    ``topk_idxs``  [b, m, topk]     int32; **-1 means "empty slot"**
    returns        [b, m, h, d]     same dtype as ``q``

    A ``-1`` slot contributes nothing: its KV row reads as zero and its score as
    ``-inf``.  The running max is floored at ``-1e30`` rather than ``-inf``, so a
    row whose indices are *all* ``-1`` yields ``exp(-inf - (-1e30)) = 0`` in the
    numerator and ``exp(sink + 1e30) = +inf`` in the denominator, i.e. an all-zero
    output instead of a NaN.  That is the training kernel's convention and the
    tilelang kernel reproduces it deliberately; so does this.

    The sink is a per-head learnable logit that joins the softmax denominator
    only -- it has no value vector, so it lets a head attend to "nothing".
    """
    b, m, h, d = q.shape
    n = kv.size(1)
    topk = topk_idxs.size(-1)
    assert topk_idxs.shape[:2] == (b, m), (topk_idxs.shape, b, m)
    assert kv.size(-1) == d, (kv.shape, d)
    assert attn_sink.numel() >= h, (attn_sink.shape, h)

    out = torch.empty_like(q)
    sink = attn_sink[:h].float()                       # [h]
    kv_f = kv.float()

    if topk == 0:
        # nothing to attend to at all: denominator is exp(sink + 1e30) = inf
        out.zero_()
        return out

    for start in range(0, m, SPARSE_ATTN_Q_CHUNK):
        stop = min(start + SPARSE_ATTN_Q_CHUNK, m)
        idx = topk_idxs[:, start:stop].long()          # [b, c, topk]
        valid = idx >= 0                               # [b, c, topk]
        safe = idx.clamp_min(0)

        # gather KV rows; invalid slots read row 0 then get zeroed
        gather_idx = safe.reshape(b, -1, 1).expand(-1, -1, d)
        kv_g = torch.gather(kv_f, 1, gather_idx).view(b, stop - start, topk, d)
        kv_g = kv_g * valid.unsqueeze(-1)

        qc = q[:, start:stop].float()                  # [b, c, h, d]
        scores = torch.einsum("bchd,bctd->bcht", qc, kv_g) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

        # finite floor, exactly as the kernel initializes scores_max
        mx = scores.amax(dim=-1).clamp_min(-1e30)       # [b, c, h]
        p = torch.exp(scores - mx.unsqueeze(-1))
        denom = p.sum(dim=-1) + torch.exp(sink.view(1, 1, h) - mx)

        o = torch.einsum("bcht,bctd->bchd", p, kv_g) / denom.unsqueeze(-1)
        out[:, start:stop] = o.to(q.dtype)

    return out
