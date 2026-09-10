"""Unit tests for kernel_cpu.py.

Run:  python test_kernel_cpu.py          (no pytest needed)
      pytest test_kernel_cpu.py -q       (also works)

Every kernel is checked against an *independently written* naive reference --
usually an explicit python loop -- rather than against a rearrangement of the
implementation itself, so a shared misreading of kernel.py cannot pass silently.
Shapes, dtypes and the awkward edges (all-zero blocks, overflow, all -1 indices,
padding, topk == 0) get their own cases.
"""

import math

import torch

import kernel_cpu as K

torch.manual_seed(0)
torch.set_num_threads(4)

_FAILURES = []


def check(cond, msg):
    if not cond:
        _FAILURES.append(msg)
        print(f"    FAIL: {msg}")
    return cond


def rel_err(a, b):
    a, b = a.float(), b.float()
    denom = b.abs().max().clamp_min(1e-12)
    return ((a - b).abs().max() / denom).item()


# ===========================================================================
# 1. act_quant
# ===========================================================================
def naive_act_quant(x, block_size, round_scale):
    """Explicit per-row-per-block loop mirroring act_quant_kernel_ line by line."""
    xf = x.float()
    *lead, N = xf.shape
    flat = xf.reshape(-1, N)
    M, G = flat.shape[0], N // block_size
    y = torch.zeros(M, N)
    s_out = torch.zeros(M, G)
    for i in range(M):
        for g in range(G):
            blk = flat[i, g * block_size:(g + 1) * block_size]
            amax = max(blk.abs().max().item(), 1e-4)
            # The kernel multiplies by the reciprocal constant rather than dividing,
            # and its compute dtype is fp32 -- doing this in python doubles and
            # rounding afterwards lands 1 ulp off on ~half the blocks.
            if round_scale:
                s = 2.0 ** math.ceil(math.log2(amax * (1.0 / 448.0)))
            else:
                s = (torch.tensor(amax, dtype=torch.float32)
                     * torch.tensor(1.0 / 448.0, dtype=torch.float32)).item()
            q = torch.clamp(blk / s, -448.0, 448.0).to(torch.float8_e4m3fn).float()
            y[i, g * block_size:(g + 1) * block_size] = q
            s_out[i, g] = s
    return y.reshape(*lead, N), s_out.reshape(*lead, G)


def test_act_quant():
    print("test_act_quant")
    x = (torch.randn(3, 7, 256) * 5).bfloat16()

    for bs in (32, 128):
        y, s = K.act_quant(x, bs, "ue8m0", torch.float8_e8m0fnu)
        check(y.shape == x.shape, f"shape {y.shape}")
        check(y.dtype == torch.float8_e4m3fn, f"dtype {y.dtype}")
        check(s.shape == (3, 7, 256 // bs), f"scale shape {s.shape}")
        check(s.dtype == torch.float8_e8m0fnu, f"scale dtype {s.dtype}")

        ny, ns = naive_act_quant(x, bs, True)
        check(torch.equal(y.float(), ny), f"bs={bs} values match naive loop")
        check(torch.equal(s.float(), ns), f"bs={bs} scales match naive loop")

        # e8m0 storage is lossless for a power-of-two scale
        check(torch.equal(s.float(), ns), "e8m0 round-trip exact")
        # every rounded scale really is a power of two
        check(torch.all(torch.log2(s.float()) == torch.log2(s.float()).round()),
              "round_scale gives powers of two")

    # accuracy of the dequantized round trip: e4m3 has 4 mantissa bits,
    # scale rounds *up*, so worst case is ~2 * 2^-4 relative
    y, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    deq = y.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)
    err = (deq.flatten(-2) - x.float()).abs().max() / x.float().abs().max()
    check(err < 0.15, f"round-trip rel err {err:.4f} < 0.15")

    # all-zero block: amax floors at 1e-4, so scale = 2^ceil(log2(1e-4/448))
    z = torch.zeros(1, 64).bfloat16()
    y, s = K.act_quant(z, 32, "ue8m0", torch.float8_e8m0fnu)
    check(torch.all(y.float() == 0), "zero input quantizes to zero")
    check(torch.all(s.float() > 0), "zero input still gets a positive scale")
    check(not torch.isnan(s.float()).any(), "no NaN scale on zero input")

    # extreme values must clamp, not overflow to NaN
    big = torch.tensor([[1e30, -1e30, 0.0, 1.0] * 8]).bfloat16()
    y, s = K.act_quant(big, 32, "ue8m0", torch.float8_e8m0fnu)
    check(not torch.isnan(y.float()).any(), "no NaN from huge inputs")
    check(y.float().abs().max() <= 448.0, "clamped to fp8 max")
    tiny = torch.full((1, 32), 1e-8).bfloat16()
    y, s = K.act_quant(tiny, 32, "ue8m0", torch.float8_e8m0fnu)
    check(not torch.isnan(y.float()).any(), "no NaN from tiny inputs")

    # inplace == dequantized non-inplace, and preserves dtype/identity
    a = x.clone()
    ptr = a.data_ptr()
    r = K.act_quant(a, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True)
    check(r.data_ptr() == ptr and a.dtype == torch.bfloat16, "inplace keeps buffer and dtype")
    y, s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    expect = (y.float().unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)).flatten(-2).bfloat16()
    check(torch.equal(a, expect), "inplace matches explicit dequant")


def test_act_quant_unrounded_scale_storage_quirk():
    """round_scale=False + e8m0 storage: the kernel quantizes with the unrounded
    fp32 scale but stores the rounded one, so `s` does not reconstruct `y`.
    Reproduced deliberately -- see the module docstring."""
    print("test_act_quant_unrounded_scale_storage_quirk")
    x = (torch.randn(2, 64) * 3).bfloat16()
    y, s = K.act_quant(x, 32, None, torch.float8_e8m0fnu)
    ny, ns = naive_act_quant(x, 32, False)
    check(torch.equal(y.float(), ny), "values use the unrounded scale")
    check(torch.equal(s.float(), ns.to(torch.float8_e8m0fnu).float()),
          "stored scale is the e8m0-rounded one")
    # and with fp32 storage the pair is self-consistent again
    y, s = K.act_quant(x, 32, None, torch.float32)
    check(torch.equal(s, ns), "fp32 storage keeps the exact scale")


# ===========================================================================
# 2. fp4_act_quant
# ===========================================================================
def naive_fp4_quant(x, block_size, e4m3_scale):
    """Explicit loop, rounding onto the e2m1 grid by exhaustive nearest search."""
    grid = K.FP4_TABLE[:8].tolist()
    xf = x.float()
    *lead, N = xf.shape
    flat = xf.reshape(-1, N)
    M, G = flat.shape[0], N // block_size
    y = torch.zeros(M, N)
    s_out = torch.zeros(M, G)
    for i in range(M):
        for g in range(G):
            blk = flat[i, g * block_size:(g + 1) * block_size]
            amax = blk.abs().max().item()
            if e4m3_scale:
                amax = max(amax, 6.0 * 2.0 ** -9)
                s = torch.tensor(amax / 6.0).to(torch.float8_e4m3fn).float().item()
            else:
                amax = max(amax, 6.0 * 2.0 ** -126)
                s = 2.0 ** math.ceil(math.log2(amax * (1.0 / 6.0)))
            for j in range(block_size):
                v = max(-6.0, min(6.0, blk[j].item() / s))
                a = abs(v)
                # nearest grid point, ties to the even index
                best, bd = 0, float("inf")
                for gi, gv in enumerate(grid):
                    d = abs(a - gv)
                    if d < bd - 1e-12 or (abs(d - bd) <= 1e-12 and gi % 2 == 0):
                        best, bd = gi, d
                y[i, g * block_size + j] = -grid[best] if v < 0 else grid[best]
            s_out[i, g] = s
    return y.reshape(*lead, N), s_out.reshape(*lead, G)


def test_fp4_act_quant():
    print("test_fp4_act_quant")
    x = (torch.randn(2, 5, 128) * 4).bfloat16()

    # --- e8m0 scale path (indexer) ---
    y, s = K.fp4_act_quant(x, 32, False, torch.float8_e8m0fnu)
    check(y.dtype == torch.float4_e2m1fn_x2, f"packed dtype {y.dtype}")
    check(y.shape == (2, 5, 64), f"packed shape {y.shape} (N // 2)")
    check(s.shape == (2, 5, 4) and s.dtype == torch.float8_e8m0fnu, f"scale {s.shape} {s.dtype}")

    unpacked = K.fp4_unpack_to_float(y)
    ny, ns = naive_fp4_quant(x, 32, False)
    check(torch.equal(unpacked, ny), "e8m0 fp4 values match naive loop")
    check(torch.equal(s.float(), ns), "e8m0 fp4 scales match naive loop")

    # every unpacked value is on the e2m1 grid
    check(torch.isin(unpacked.abs(), K.FP4_TABLE[:8]).all(), "values lie on the e2m1 grid")

    # --- e4m3 scale path (compressed KV), block 16 as model.py uses ---
    y, s = K.fp4_act_quant(x, 16, False, torch.float8_e4m3fn)
    check(s.shape == (2, 5, 8) and s.dtype == torch.float8_e4m3fn, f"e4m3 scale {s.shape} {s.dtype}")
    ny, ns = naive_fp4_quant(x, 16, True)
    check(torch.equal(K.fp4_unpack_to_float(y), ny), "e4m3 fp4 values match naive loop")
    check(torch.equal(s.float(), ns), "e4m3 fp4 scales match naive loop")

    # all-zero block: the e4m3 path must still produce a nonzero scale
    z = torch.zeros(1, 32).bfloat16()
    _, s = K.fp4_act_quant(z, 16, False, torch.float8_e4m3fn)
    check(torch.all(s.float() > 0), "all-zero block keeps a nonzero e4m3 scale")
    check(abs(s.float()[0, 0].item() - 2.0 ** -9) < 1e-12, f"floor scale is 2^-9, got {s.float()[0,0]}")
    _, s = K.fp4_act_quant(z, 32, False, torch.float8_e8m0fnu)
    check(torch.all(s.float() > 0), "all-zero block keeps a nonzero e8m0 scale")

    # extremes clamp rather than blow up
    big = torch.tensor([[1e30, -1e30] * 16]).bfloat16()
    y, s = K.fp4_act_quant(big, 32, False, torch.float8_e8m0fnu)
    u = K.fp4_unpack_to_float(y)
    check(not torch.isnan(u).any() and u.abs().max() <= 6.0, "fp4 clamps extremes")

    # inplace
    a = x.clone()
    ptr = a.data_ptr()
    K.fp4_act_quant(a, 32, True)
    check(a.data_ptr() == ptr and a.dtype == torch.bfloat16, "fp4 inplace keeps buffer/dtype")
    y, s = K.fp4_act_quant(x, 32, False, torch.float8_e8m0fnu)
    expect = (K.fp4_unpack_to_float(y).unflatten(-1, (-1, 32)) * s.float().unsqueeze(-1)).flatten(-2)
    check(torch.equal(a, expect.bfloat16()), "fp4 inplace matches explicit dequant")

    # fp4 is coarse: ~2^-1 relative worst case per element, but unbiased in aggregate
    err = rel_err(expect, x)
    check(err < 0.35, f"fp4 round-trip rel err {err:.4f} < 0.35")


def test_fp4_packing_layout():
    """Low nibble = even element, high nibble = odd -- the layout convert.py assumes."""
    print("test_fp4_packing_layout")
    codes = torch.tensor([[0, 7, 1, 15]], dtype=torch.uint8)   # 0.0, 6.0, 0.5, -6.0
    packed = K._pack_fp4_codes(codes)
    check(packed.tolist() == [[0x70, 0xF1]], f"packed nibbles {[hex(v) for v in packed[0].tolist()]}")
    back = K.fp4_unpack_to_float(packed)
    check(back.tolist() == [[0.0, 6.0, 0.5, -6.0]], f"unpacked {back.tolist()}")


# ===========================================================================
# 3 + 4. fp8_gemm / fp4_gemm
# ===========================================================================
def block_quant_weight_2d(w, bs):
    """Quantize a [N, K] weight the way the checkpoint stores one: a single
    power-of-two e8m0 scale per bs x bs block.  Returns (fp8 weight, [ceil(N/bs),
    ceil(K/bs)] e8m0 scale) -- the exact layout ``Linear`` and ``fp8_gemm`` expect."""
    N, Kk = w.shape
    nb, kb = (N + bs - 1) // bs, (Kk + bs - 1) // bs
    pad = torch.zeros(nb * bs, kb * bs)
    pad[:N, :Kk] = w.float()
    blocks = pad.unflatten(0, (nb, bs)).unflatten(-1, (kb, bs))      # [nb, bs, kb, bs]
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4)             # [nb, kb]
    s = 2.0 ** torch.ceil(torch.log2(amax * (1.0 / 448.0)))
    q = torch.clamp(blocks / s[:, None, :, None], -448.0, 448.0).to(torch.float8_e4m3fn)
    q = q.float().reshape(nb * bs, kb * bs)[:N, :Kk]
    return q.to(torch.float8_e4m3fn), s.to(torch.float8_e8m0fnu)


def naive_scaled_gemm(a_vals, a_s, b_vals, b_s_full, block_a, block_b):
    """C[m,n] = sum_k a[m,k]*sa[m,k//ba] * b[n,k]*sb[n,k//bb] -- explicit loop."""
    M, Kk = a_vals.shape
    N = b_vals.shape[0]
    c = torch.zeros(M, N, dtype=torch.float64)
    for m in range(M):
        for n in range(N):
            acc = 0.0
            for k in range(Kk):
                acc += (a_vals[m, k].item() * a_s[m, k // block_a].item()
                        * b_vals[n, k].item() * b_s_full[n, k // block_b].item())
            c[m, n] = acc
    return c


def test_fp8_gemm():
    print("test_fp8_gemm")
    torch.set_default_dtype(torch.bfloat16)
    try:
        M, N, Kk, bs = 5, 64, 64, 32
        x = (torch.randn(M, Kk) * 2).bfloat16()
        a, a_s = K.act_quant(x, bs, "ue8m0", torch.float8_e8m0fnu)

        w = (torch.randn(N, Kk) * 0.5).bfloat16()
        b, b_s = block_quant_weight_2d(w, bs)          # [N, K] fp8, [N/bs, K/bs] e8m0
        check(tuple(b_s.shape) == (N // bs, Kk // bs), f"weight scale shape {b_s.shape}")

        c = K.fp8_gemm(a, a_s, b, b_s, torch.float8_e8m0fnu, block_size=bs)
        check(c.shape == (M, N), f"shape {c.shape}")
        check(c.dtype == torch.bfloat16, f"out dtype {c.dtype} follows default dtype")

        b_s_full = b_s.float().repeat_interleave(bs, dim=0)[:N]
        ref = naive_scaled_gemm(a.float(), a_s.float(), b.float(), b_s_full, bs, bs)
        check(rel_err(c.float(), ref.float()) < 5e-3,
              f"vs naive loop rel err {rel_err(c.float(), ref.float()):.2e}")

        # quantized GEMM should track the unquantized one to within fp8 error
        ref_bf16 = (x.float() @ w.float().T)
        check(rel_err(c.float(), ref_bf16) < 0.1,
              f"vs unquantized rel err {rel_err(c.float(), ref_bf16):.4f}")

        # 3-D activations keep their leading dims
        x3 = (torch.randn(2, 3, Kk)).bfloat16()
        a3, a3_s = K.act_quant(x3, bs, "ue8m0", torch.float8_e8m0fnu)
        c3 = K.fp8_gemm(a3, a3_s, b, b_s, torch.float8_e8m0fnu, block_size=bs)
        check(c3.shape == (2, 3, N), f"3-D shape {c3.shape}")

        # all-zero activation -> all-zero output, no NaN
        z = torch.zeros(M, Kk).bfloat16()
        az, az_s = K.act_quant(z, bs, "ue8m0", torch.float8_e8m0fnu)
        cz = K.fp8_gemm(az, az_s, b, b_s, torch.float8_e8m0fnu, block_size=bs)
        check(torch.all(cz.float() == 0), "zero activations give zero output")

        # block_size 128 with N not a multiple of 128: the scale grid is ceil(N/128)
        Kk2 = 256
        x2 = torch.randn(M, Kk2).bfloat16()
        a2, a2_s = K.act_quant(x2, 128, "ue8m0", torch.float8_e8m0fnu)
        w2 = torch.randn(N, Kk2).bfloat16()
        b2, b2_s = block_quant_weight_2d(w2, 128)
        check(tuple(b2_s.shape) == (1, 2), f"ragged weight scale shape {b2_s.shape}")
        c2 = K.fp8_gemm(a2, a2_s, b2, b2_s, torch.float8_e8m0fnu, block_size=128)
        check(c2.shape == (M, N), f"bs=128 shape {c2.shape}")
        check(not torch.isnan(c2.float()).any(), "bs=128 no NaN")
        ref2 = naive_scaled_gemm(a2.float(), a2_s.float(), b2.float(),
                                 b2_s.float().repeat_interleave(128, dim=0)[:N], 128, 128)
        check(rel_err(c2.float(), ref2.float()) < 5e-3,
              f"bs=128 vs naive {rel_err(c2.float(), ref2.float()):.2e}")
    finally:
        torch.set_default_dtype(torch.float32)


def test_dequant_fp8_weight_axes():
    """Non-square, ragged shapes: the two block axes must not be confused.

    Regression guard -- with N == K, or with N < K, a scale expanded along the
    wrong axis still yields the right *shape*, so only asymmetric sizes catch it.
    """
    print("test_dequant_fp8_weight_axes")
    N, Kk, bs = 96, 64, 32                       # N/bs = 3, K/bs = 2, and N > K
    w = torch.randn(N, Kk).clamp(-400, 400).to(torch.float8_e4m3fn)
    s = (2.0 ** torch.randint(-4, 4, (N // bs, Kk // bs)).float()).to(torch.float8_e8m0fnu)

    got = K.dequant_fp8_weight(w, s, bs)
    check(got.shape == (N, Kk), f"shape {got.shape}")
    ref = torch.zeros(N, Kk)
    for i in range(N):
        for j in range(Kk):
            ref[i, j] = w.float()[i, j] * s.float()[i // bs, j // bs]
    check(torch.equal(got, ref), "2-D block scale expands along the right axes")

    # ragged: neither dim a multiple of the block
    N2, K2 = 70, 50
    w2 = torch.randn(N2, K2).clamp(-400, 400).to(torch.float8_e4m3fn)
    s2 = (2.0 ** torch.randint(-4, 4, ((N2 + bs - 1) // bs, (K2 + bs - 1) // bs)).float()
          ).to(torch.float8_e8m0fnu)
    got2 = K.dequant_fp8_weight(w2, s2, bs)
    ref2 = torch.zeros(N2, K2)
    for i in range(N2):
        for j in range(K2):
            ref2[i, j] = w2.float()[i, j] * s2.float()[i // bs, j // bs]
    check(torch.equal(got2, ref2), "ragged 2-D block scale is correct")

    # and the fp4 weight path, per-32 along K only
    N3, K3 = 48, 128
    packed = torch.randint(0, 255, (N3, K3 // 2), dtype=torch.uint8)
    s3 = (2.0 ** torch.randint(-3, 3, (N3, K3 // 32)).float()).to(torch.float8_e8m0fnu)
    g3 = K.dequant_fp4_weight(packed, s3, 32)
    vals = K.fp4_unpack_to_float(packed)
    ref3 = vals * s3.float().repeat_interleave(32, dim=-1)
    check(g3.shape == (N3, K3) and torch.equal(g3, ref3), "fp4 weight scale expands along K")


def test_fp4_gemm():
    print("test_fp4_gemm")
    torch.set_default_dtype(torch.bfloat16)
    try:
        M, N, Kk = 5, 32, 128
        x = (torch.randn(M, Kk) * 2).bfloat16()
        a, a_s = K.act_quant(x, 128, "ue8m0", torch.float8_e8m0fnu)

        w = (torch.randn(N, Kk) * 0.5).bfloat16()
        b, b_s = K.fp4_act_quant(w, 32, False, torch.float8_e8m0fnu)   # [N, K/2], [N, K/32]
        check(b.shape == (N, Kk // 2), f"fp4 weight shape {b.shape}")
        check(b_s.shape == (N, Kk // 32), f"fp4 weight scale shape {b_s.shape}")

        c = K.fp4_gemm(a, a_s, b, b_s, torch.float8_e8m0fnu, act_block_size=128)
        check(c.shape == (M, N), f"shape {c.shape}")
        check(c.dtype == torch.bfloat16, f"out dtype {c.dtype}")

        ref = naive_scaled_gemm(a.float(), a_s.float(),
                                K.fp4_unpack_to_float(b), b_s.float(), 128, 32)
        check(rel_err(c.float(), ref.float()) < 5e-3,
              f"vs naive loop rel err {rel_err(c.float(), ref.float()):.2e}")

        ref_bf16 = x.float() @ w.float().T
        e = rel_err(c.float(), ref_bf16)
        check(e < 0.25, f"vs unquantized rel err {e:.4f} (fp4 is coarse)")

        # act_block_size 32 path
        a32, a32_s = K.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        c32 = K.fp4_gemm(a32, a32_s, b, b_s, torch.float8_e8m0fnu, act_block_size=32)
        check(c32.shape == (M, N) and not torch.isnan(c32.float()).any(), "act_block_size=32 ok")

        # a real checkpoint expert weight is int8 in the file; .view() must work
        raw = torch.randint(-128, 127, (N, Kk // 2), dtype=torch.int8)
        bv = raw.view(torch.float4_e2m1fn_x2)
        cv = K.fp4_gemm(a, a_s, bv, b_s, torch.float8_e8m0fnu, act_block_size=128)
        check(cv.shape == (M, N) and not torch.isnan(cv.float()).any(),
              "int8-viewed checkpoint weights work")
    finally:
        torch.set_default_dtype(torch.float32)


# ===========================================================================
# 5. hc_split_sinkhorn
# ===========================================================================
def naive_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    """Row-by-row python reference."""
    b, s, _ = mixes.shape
    pre = torch.zeros(b, s, hc)
    post = torch.zeros(b, s, hc)
    comb = torch.zeros(b, s, hc, hc)
    for i in range(b):
        for j in range(s):
            m = mixes[i, j].float()
            for k in range(hc):
                pre[i, j, k] = torch.sigmoid(m[k] * hc_scale[0] + hc_base[k]) + eps
                post[i, j, k] = 2 * torch.sigmoid(m[k + hc] * hc_scale[1] + hc_base[k + hc])
            c = torch.zeros(hc, hc)
            for r in range(hc):
                for t in range(hc):
                    c[r, t] = m[r * hc + t + 2 * hc] * hc_scale[2] + hc_base[r * hc + t + 2 * hc]
            c = torch.softmax(c, dim=-1) + eps
            c = c / (c.sum(dim=0, keepdim=True) + eps)
            for _ in range(iters - 1):
                c = c / (c.sum(dim=1, keepdim=True) + eps)
                c = c / (c.sum(dim=0, keepdim=True) + eps)
            comb[i, j] = c
    return pre, post, comb


def test_hc_split_sinkhorn():
    print("test_hc_split_sinkhorn")
    hc, iters, eps = 4, 20, 1e-6
    b, s = 2, 6
    mixes = torch.randn(b, s, (2 + hc) * hc)
    hc_scale = torch.tensor([0.7, 1.3, 2.1])
    hc_base = torch.randn((2 + hc) * hc)

    pre, post, comb = K.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps)
    check(pre.shape == (b, s, hc), f"pre shape {pre.shape}")
    check(post.shape == (b, s, hc), f"post shape {post.shape}")
    check(comb.shape == (b, s, hc, hc), f"comb shape {comb.shape}")
    check(pre.dtype == torch.float32 and comb.dtype == torch.float32, "fp32 outputs")

    npre, npost, ncomb = naive_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps)
    check(rel_err(pre, npre) < 1e-6, f"pre vs naive {rel_err(pre, npre):.2e}")
    check(rel_err(post, npost) < 1e-6, f"post vs naive {rel_err(post, npost):.2e}")
    check(rel_err(comb, ncomb) < 1e-5, f"comb vs naive {rel_err(comb, ncomb):.2e}")

    # ranges
    check(bool(((pre > eps) & (pre < 1 + eps + 1e-6)).all()), "pre in (eps, 1+eps)")
    check(bool(((post > 0) & (post < 2)).all()), "post in (0, 2)")
    check(bool((comb >= 0).all()), "comb non-negative")

    # 20 rounds of alternating normalization should leave comb near doubly stochastic
    row = comb.sum(-1)
    col = comb.sum(-2)
    check((row - 1).abs().max() < 5e-2, f"row sums ~1, max dev {(row - 1).abs().max():.4f}")
    check((col - 1).abs().max() < 5e-2, f"col sums ~1, max dev {(col - 1).abs().max():.4f}")

    # more iterations must converge harder, never diverge
    _, _, c50 = K.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, 50, eps)
    check((c50.sum(-1) - 1).abs().max() <= (row - 1).abs().max() + 1e-6,
          "more Sinkhorn iterations converge further")

    # degenerate inputs
    _, _, cz = K.hc_split_sinkhorn(torch.zeros(1, 1, (2 + hc) * hc), hc_scale,
                                   torch.zeros((2 + hc) * hc), hc, iters, eps)
    check(not torch.isnan(cz).any(), "zero mixes give no NaN")
    check((cz - 1.0 / hc).abs().max() < 1e-3, "uniform input -> uniform doubly stochastic matrix")

    big = torch.full((1, 1, (2 + hc) * hc), 50.0)
    p, po, cb = K.hc_split_sinkhorn(big, hc_scale, hc_base, hc, iters, eps)
    check(not (torch.isnan(cb).any() or torch.isnan(p).any()), "large mixes give no NaN")


# ===========================================================================
# 6. sparse_attn
# ===========================================================================
def naive_sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """Per (batch, query, head) python reference, including the sink and -1 rule."""
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    out = torch.zeros(b, m, h, d, dtype=torch.float32)
    for i in range(b):
        for j in range(m):
            for hh in range(h):
                scores = []
                rows = []
                for t in range(topk):
                    idx = int(topk_idxs[i, j, t])
                    if idx < 0:
                        scores.append(float("-inf"))
                        rows.append(torch.zeros(d))
                    else:
                        row = kv[i, idx].float()
                        scores.append((q[i, j, hh].float() * row).sum().item() * scale)
                        rows.append(row)
                mx = max(max(scores) if scores else -1e30, -1e30)

                def _exp(v):
                    try:
                        return math.exp(v)
                    except OverflowError:
                        return math.inf

                ps = [_exp(sc - mx) if sc != float("-inf") else 0.0 for sc in scores]
                denom = sum(ps) + _exp(float(attn_sink[hh]) - mx)
                acc = torch.zeros(d)
                for t in range(topk):
                    acc += ps[t] * rows[t]
                out[i, j, hh] = acc / denom
    return out


def test_sparse_attn():
    print("test_sparse_attn")
    b, m, h, d, n, topk = 2, 4, 3, 16, 20, 6
    scale = d ** -0.5
    q = torch.randn(b, m, h, d).bfloat16()
    kv = torch.randn(b, n, d).bfloat16()
    sink = torch.randn(h)
    idxs = torch.randint(0, n, (b, m, topk), dtype=torch.int32)
    idxs[0, 0, 2] = -1              # one empty slot
    idxs[1, 3, :] = -1              # a fully empty row

    o = K.sparse_attn(q, kv, sink, idxs, scale)
    check(o.shape == q.shape, f"shape {o.shape}")
    check(o.dtype == q.dtype, f"dtype {o.dtype}")

    ref = naive_sparse_attn(q, kv, sink, idxs, scale)
    check(rel_err(o.float(), ref) < 2e-2, f"vs naive rel err {rel_err(o.float(), ref):.2e}")

    # all-(-1) row -> exactly zero, not NaN. This is the load-bearing edge case.
    check(torch.all(o[1, 3] == 0), "all -1 row returns zeros")
    check(not torch.isnan(o.float()).any(), "no NaN anywhere")

    # a -1 slot must behave exactly like dropping that slot
    idx_a = torch.tensor([[[0, 1, 2, -1]]], dtype=torch.int32)
    idx_b = torch.tensor([[[0, 1, 2]]], dtype=torch.int32)
    q1 = torch.randn(1, 1, h, d).bfloat16()
    kv1 = torch.randn(1, n, d).bfloat16()
    oa = K.sparse_attn(q1, kv1, sink, idx_a, scale)
    ob = K.sparse_attn(q1, kv1, sink, idx_b, scale)
    check(torch.equal(oa, ob), "-1 padding is equivalent to a shorter topk")

    # duplicate indices are summed independently, so they are NOT idempotent
    idx_d = torch.tensor([[[0, 0, 1]]], dtype=torch.int32)
    od = K.sparse_attn(q1, kv1, sink, idx_d, scale)
    check(not torch.isnan(od.float()).any(), "duplicate indices are handled")

    # a very negative sink should reduce to a plain softmax over the gathered rows
    big_neg = torch.full((h,), -1e4)
    o_ns = K.sparse_attn(q1, kv1, big_neg, idx_b, scale).float()
    kg = kv1[0, :3].float()
    sc = torch.einsum("hd,td->ht", q1[0, 0].float(), kg) * scale
    plain = torch.softmax(sc, dim=-1) @ kg
    check(rel_err(o_ns[0, 0], plain) < 2e-2, f"sink=-1e4 ~ plain softmax {rel_err(o_ns[0,0], plain):.2e}")

    # a very positive sink should crush the output towards zero
    big_pos = torch.full((h,), 30.0)
    o_ps = K.sparse_attn(q1, kv1, big_pos, idx_b, scale).float()
    check(o_ps.abs().max() < o_ns.abs().max(), "a large sink shrinks the output")

    # topk == 0 (the compress_len == 0 path in model.py)
    o0 = K.sparse_attn(q1, kv1, sink, torch.empty(1, 1, 0, dtype=torch.int32), scale)
    check(o0.shape == q1.shape and torch.all(o0 == 0), "topk=0 returns zeros")

    # head count below the kernel's 16-head padding threshold still works
    check(h < 16, "test exercises the h<16 padding path")

    # chunking over queries must not change the result
    old = K.SPARSE_ATTN_Q_CHUNK
    try:
        K.SPARSE_ATTN_Q_CHUNK = 1
        o_chunk = K.sparse_attn(q, kv, sink, idxs, scale)
        check(torch.equal(o_chunk, o), "query chunking is transparent")
    finally:
        K.SPARSE_ATTN_Q_CHUNK = old

    # larger, realistic-ish shapes
    o_big = K.sparse_attn(torch.randn(1, 33, 64, 512).bfloat16(),
                          torch.randn(1, 200, 512).bfloat16(),
                          torch.randn(64),
                          torch.randint(0, 200, (1, 33, 128), dtype=torch.int32),
                          512 ** -0.5)
    check(o_big.shape == (1, 33, 64, 512) and not torch.isnan(o_big.float()).any(),
          "realistic shapes ok")


# ===========================================================================
def main():
    tests = [
        test_act_quant,
        test_act_quant_unrounded_scale_storage_quirk,
        test_fp4_act_quant,
        test_fp4_packing_layout,
        test_fp8_gemm,
        test_dequant_fp8_weight_axes,
        test_fp4_gemm,
        test_hc_split_sinkhorn,
        test_sparse_attn,
    ]
    for t in tests:
        t()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILURES:")
        for f in _FAILURES:
            print("  -", f)
        raise SystemExit(1)
    print("all kernel_cpu tests passed")


if __name__ == "__main__":
    main()
