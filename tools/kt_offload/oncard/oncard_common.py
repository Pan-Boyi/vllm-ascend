# SPDX-License-Identifier: Apache-2.0
"""Pure-torch half of the on-card MoE checks: checkpoint bytes, torch references,
metrics, routing generation. Imports no torch_npu and opens no device, so the
CPU self-check (selftest_cpu.py) runs every function here without a card.

Reference numerics follow M0b (ktransformers kt-kernel/test/dsv41_mxfp4_offload_check.py,
PASS log build_artifacts/moe_offload_ref/m0_evidence/m0b_run4_292be81.log):
E2M1 LUT x 2^(byte-127), gate clamp one-sided / up clamp symmetric at 10.0,
route weights sqrtsoftplus-normalised x 1.5. Added here: an MXFP8 activation
variant for the NPU side (W4A8: x quantised in npu_moe_init_routing quant_mode=3,
the SwiGLU output quantised in npu_swiglu_group_quant), both per 32 along the
last dim with an E8M0 shared exponent (OCP MX: floor(log2(amax)) - 8 for E4M3).
Whether the NPU kernels match this emulation is exactly what T0 measures; the
gating criterion is therefore against the bf16-activation reference with a
floor that the MXFP8 emulation measures (see README "T0").
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

CKPT = os.environ.get("DSV41_CKPT", "/workspace/models/DeepSeek-V4.1-Flash")
HIDDEN, INTER, N_EXP, TOPK = 5120, 2304, 384, 6
LIMIT, ROUTE_SCALE = 10.0, 1.5
GROUP = 32
BF16_ULP = 2.0**-8  # relative spacing of bf16 at 1.0
E4M3_MAX = 448.0
E4M3_EMAX = 8  # 448 = 1.75 * 2^8

E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# Anomaly markers. Kept in one place: run_oncard.sh reads the same list through
# `python3 oncard_common.py --anomaly-regex`, and selftest_cpu feeds forged logs.
ANOMALY_PATTERNS = (
    r"AICore",
    r"[Aa][Ii][Cc]ore exception",
    r"aicore error",
    r"AIV(EC)? exception",
    r"MTE_ERROR",
    r"EZ9999",
    r"EE9999",
    r"EL0004",
    r"device task abort",
    r"Kernel task happen error",
)


def anomaly_regex() -> str:
    return "|".join(f"({p})" for p in ANOMALY_PATTERNS)


# ---------------------------------------------------------------------------
# checkpoint (byte views only: E8M0 scales are never numerically converted)
# ---------------------------------------------------------------------------


class Ckpt:
    def __init__(self, path: str = CKPT):
        from safetensors import safe_open  # noqa: F401  (fail early if missing)

        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.idx = json.load(f)["weight_map"]
        self.path, self.h = path, {}

    def raw(self, key: str) -> torch.Tensor:
        from safetensors import safe_open

        fn = self.idx[key]
        if fn not in self.h:
            self.h[fn] = safe_open(os.path.join(self.path, fn), framework="pt")
        return self.h[fn].get_tensor(key)

    def bytes_(self, key: str) -> torch.Tensor:
        return self.raw(key).contiguous().view(torch.uint8)

    def expert(self, layer: int, e: int) -> dict[str, torch.Tensor]:
        b = f"layers.{layer}.ffn.experts.{e}"
        out = {}
        for p in ("w1", "w3", "w2"):
            out[f"{p}w"] = self.bytes_(f"{b}.{p}.weight")
            out[f"{p}s"] = self.bytes_(f"{b}.{p}.scale")
        return out


def expert_shapes_ok(ex: dict[str, torch.Tensor]) -> list[str]:
    want = {
        "w1w": (INTER, HIDDEN // 2), "w3w": (INTER, HIDDEN // 2), "w2w": (HIDDEN, INTER // 2),
        "w1s": (INTER, HIDDEN // GROUP), "w3s": (INTER, HIDDEN // GROUP), "w2s": (HIDDEN, INTER // GROUP),
    }
    return [f"{k}: {tuple(ex[k].shape)} != {v}" for k, v in want.items() if tuple(ex[k].shape) != v]


def pack_npu_rows(ex: dict[str, torch.Tensor]):
    """Checkpoint bytes -> one expert's rows of the vLLM parameters.

    w13 = cat(w1, w3) along dim 0 (vllm fused_moe routed_experts._load_w13: shard
    w1 -> first half, w3 -> second half); scales the same, as uint8 (the DS
    method declares them uint8 so the copy is a byte move).
    """
    return (
        torch.cat([ex["w1w"], ex["w3w"]], 0),
        torch.cat([ex["w1s"], ex["w3s"]], 0),
        ex["w2w"],
        ex["w2s"],
    )


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------


def dequant(w_u8: torch.Tensor, s_u8: torch.Tensor, swap_nibbles: bool = False) -> torch.Tensor:
    lo, hi = (w_u8 & 0x0F).long(), (w_u8 >> 4).long()
    if swap_nibbles:
        lo, hi = hi, lo
    n, kh = w_u8.shape
    codes = torch.empty(n, kh * 2, dtype=torch.long)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    scale = torch.exp2(s_u8.to(torch.float32) - 127.0)
    return E2M1[codes] * scale.repeat_interleave(GROUP, dim=1)


def bf(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16).to(torch.float32)


def mx8_fake_quant(t: torch.Tensor, block: int = GROUP) -> torch.Tensor:
    """MXFP8 (E4M3 elements, E8M0 shared exponent per ``block`` along the last dim),
    returned dequantised in fp32. Shared exponent = floor(log2(amax)) - 8 (OCP MX),
    elements saturate at +-448, round-to-nearest-even through torch's e4m3 cast."""
    shp = t.shape
    if shp[-1] % block:
        raise ValueError(f"last dim {shp[-1]} not a multiple of {block}")
    x = t.float().reshape(*shp[:-1], shp[-1] // block, block)
    amax = x.abs().amax(-1, keepdim=True)
    safe = torch.where(amax > 0, amax, torch.ones_like(amax))
    e = torch.floor(torch.log2(safe)) - E4M3_EMAX
    e = torch.clamp(e, -127.0, 127.0)
    scale = torch.exp2(e)
    q = torch.clamp(x / scale, -E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float() * scale
    q = torch.where(amax > 0, q, torch.zeros_like(q))
    return q.reshape(shp)


VARIANTS = ("fp32", "bf16", "bf16_acc64", "mx8", "mx8_acc64", "mx8_symclamp")


def expert_out(x: torch.Tensor, w1, w3, w2, variant: str, limit: float = LIMIT) -> torch.Tensor:
    """One routed expert on ``x`` [n, HIDDEN] (fp32 holding bf16 values).

    fp32          no intermediate rounding (tier-A target of M0b)
    bf16          kt's storage points rounded to bf16 (gate/up out, act, down out)
    bf16_acc64    same, GEMMs in float64 (accumulation-order floor)
    mx8           NPU W4A8 path: x MXFP8, gate/up out bf16, clamp, SwiGLU, MXFP8, down out bf16
    mx8_acc64     same, GEMMs in float64
    mx8_symclamp  diagnostic: gate clamped symmetrically too (the other reading of clamp_value)
    """
    acc64 = variant.endswith("_acc64")
    base = variant.replace("_acc64", "")
    xin = mx8_fake_quant(x) if base.startswith("mx8") else x
    if acc64:
        xin, w1, w3, w2 = xin.double(), w1.double(), w3.double(), w2.double()
    g, u = (xin @ w1.T).float(), (xin @ w3.T).float()
    if base != "fp32":
        g, u = bf(g), bf(u)
    if limit > 0:
        if base == "mx8_symclamp":
            g = torch.clamp(g, min=-limit, max=limit)
        else:
            g = torch.clamp(g, max=limit)
        u = torch.clamp(u, min=-limit, max=limit)
    h = torch.nn.functional.silu(g) * u
    if base.startswith("mx8"):
        h = mx8_fake_quant(h)
    elif base == "bf16":
        h = bf(h)
    y = (h.double() @ w2.T).float() if acc64 else h @ w2.T
    return y if base == "fp32" else bf(y)


def moe_ref_multi(x, ids, wts, weight_fn, include, variants, limit: float = LIMIT) -> dict[str, torch.Tensor]:
    """Routed-expert sum for every variant in one pass (each expert dequantised once).

    ``weight_fn(e) -> (w1, w3, w2)`` fp32; ``include(e)`` selects the routes that
    count (e.g. only CPU experts). Routes with weight 0 still count (they add 0).
    Output rounded to bf16 for every variant but fp32 (the NPU combine writes bf16).
    """
    xf = x.float()
    outs = {v: torch.zeros(x.shape[0], HIDDEN, dtype=torch.float64) for v in variants}
    for e in sorted({int(v) for v in ids.flatten().tolist()}):
        if not include(e):
            continue
        tok, pos = torch.where(ids == e)
        w1, w3, w2 = weight_fn(e)
        for v in variants:
            y = expert_out(xf[tok], w1, w3, w2, v, limit)
            outs[v].index_add_(0, tok, wts[tok, pos, None].double() * y.double())
    return {v: (o.float() if v == "fp32" else bf(o.float())) for v, o in outs.items()}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    d = float((a.double() - b.double()).norm())
    n = float(b.double().norm())
    return d / n if n > 0 else (0.0 if d == 0 else float("inf"))


def rel_l2_to(a: torch.Tensor, b: torch.Tensor, ref_norm_of: torch.Tensor) -> float:
    n = float(ref_norm_of.double().norm())
    return float((a.double() - b.double()).norm()) / n if n > 0 else float("inf")


def col_rel_max(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.double() - b.double()).norm(dim=0)
    n = b.double().norm(dim=0)
    keep = n > 1e-3 * n.max()
    return float((d[keep] / n[keep]).max()) if bool(keep.any()) else float("inf")


def row_rel_max(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.double() - b.double()).norm(dim=1)
    n = b.double().norm(dim=1)
    keep = n > 0
    return float((d[keep] / n[keep]).max()) if bool(keep.any()) else float("inf")


def bit_identical(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    """Bytes equal OR values equal (memory: 逐位判据有两个方向相反的陷阱).

    Bytes-equal alone fails on -0/+0; value-equal alone fails on NaN (NaN != NaN).
    Values-equal here is NaN-aware: NaN at the same positions, the rest ==.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
        return True, "bytes equal"
    na, nb = torch.isnan(a), torch.isnan(b)
    if torch.equal(na, nb) and torch.equal(a[~na], b[~nb]):
        return True, "values equal (bytes differ: signed zero)"
    diff = int((a != b).sum() - (na & nb).sum())
    return False, f"{diff}/{a.numel()} elements differ"


def nan_inf_count(t: torch.Tensor) -> int:
    return int((~torch.isfinite(t.float())).sum())


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


def route_weights(rng: np.random.Generator, T: int) -> torch.Tensor:
    logits = torch.from_numpy(rng.standard_normal((T, TOPK)).astype(np.float32))
    s = torch.nn.functional.softplus(logits).sqrt()  # sqrtsoftplus
    return (s / (s.sum(-1, keepdim=True) + 1e-20) * ROUTE_SCALE).contiguous()


def make_routing(rng, T: int, cpu_ids, npu_ids, force_expert=None, force_all_rows=False):
    """Distinct logical ids per row (as the router produces), mixing CPU and NPU experts.

    T>=4: row0 all-CPU, row1 all-NPU, row2 3+3; T<4: row0 3 CPU + 3 NPU.
    ``force_expert`` goes into row 0 (or every row with ``force_all_rows``) so a
    mutation of that expert is guaranteed to be exercised.
    """
    cpu_ids, npu_ids = list(cpu_ids), list(npu_ids)
    pool = np.array(cpu_ids + npu_ids)
    ids = np.stack([rng.choice(pool, TOPK, replace=False) for _ in range(T)])
    if cpu_ids and npu_ids:
        if T >= 4:
            if len(cpu_ids) >= TOPK:
                ids[0] = rng.choice(cpu_ids, TOPK, replace=False)
            if len(npu_ids) >= TOPK:
                ids[1] = rng.choice(npu_ids, TOPK, replace=False)
            r = 2
        else:
            r = 0
        ids[r, :3] = rng.choice(cpu_ids, 3, replace=False)
        ids[r, 3:] = rng.choice(npu_ids, 3, replace=False)
    if force_expert is not None:
        rows = range(T) if force_all_rows else [0]
        for i in rows:
            if force_expert not in ids[i]:
                ids[i, 0] = force_expert
    if not all(len(set(r)) == TOPK for r in ids.tolist()):
        raise AssertionError("routing produced a duplicate id inside a row")
    return torch.from_numpy(ids.astype(np.int32)), route_weights(rng, T)


@dataclass
class Coverage:
    routes: int
    cpu_routes: int
    rows_ge2_cpu: int  # rows whose NPU half repeats the dummy slot (same-row duplicate id)
    all_cpu_rows: int
    all_npu_rows: int

    @property
    def cpu_share(self) -> float:
        return self.cpu_routes / self.routes if self.routes else 0.0

    def line(self) -> str:
        return (f"coverage: CPU routes {self.cpu_routes}/{self.routes} ({100 * self.cpu_share:.1f}%), "
                f"rows with >=2 CPU routes (dup slot on NPU) {self.rows_ge2_cpu}, "
                f"all-CPU rows {self.all_cpu_rows}, all-NPU rows {self.all_npu_rows}")

    def problems(self, need_dup: bool = True) -> list[str]:
        p = []
        if self.cpu_routes == 0:
            p.append("0% CPU routes: the split was not exercised")
        if self.cpu_routes == self.routes:
            p.append("100% CPU routes: the NPU half was not exercised")
        if need_dup and self.rows_ge2_cpu == 0:
            p.append("no row with >=2 CPU routes: same-row duplicate slot never reached the NPU kernel")
        return p


def coverage(ids: torch.Tensor, resident_mask: torch.Tensor) -> Coverage:
    cpu = ~resident_mask[ids.long()]
    per_row = cpu.sum(1)
    return Coverage(
        routes=int(ids.numel()),
        cpu_routes=int(cpu.sum()),
        rows_ge2_cpu=int((per_row >= 2).sum()),
        all_cpu_rows=int((per_row == ids.shape[1]).sum()),
        all_npu_rows=int((per_row == 0).sum()),
    )


# ---------------------------------------------------------------------------
# verdict bookkeeping (coverage counted; zero checks is a failure)
# ---------------------------------------------------------------------------


class Verdict:
    def __init__(self, name: str):
        self.name, self.required, self.failed, self.mutations, self.notes = name, 0, [], 0, []

    def check(self, ok: bool, what: str) -> bool:
        self.required += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}", flush=True)
        if not ok:
            self.failed.append(what)
        return ok

    def must_be_red(self, red: bool, what: str) -> bool:
        self.mutations += 1
        return self.check(red, f"mutation must be RED: {what} -> {'RED' if red else 'green'}")

    def note(self, what: str) -> None:
        print(f"  [note] {what}", flush=True)
        self.notes.append(what)

    def finish(self, out_json: str | None = None, extra: dict | None = None) -> int:
        ok = not self.failed and self.required > 0 and self.mutations > 0
        if self.required == 0:
            print("  [FAIL] zero required checks ran")
        if self.mutations == 0:
            print("  [FAIL] zero mutation arms ran")
        print(f"\n{self.name}: required checks {self.required}, failed {len(self.failed)}, "
              f"mutation arms {self.mutations}")
        print(f"VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)
        if out_json:
            with open(out_json, "w") as f:
                json.dump({"test": self.name, "pass": ok, "required": self.required, "failed": self.failed,
                           "mutations": self.mutations, "notes": self.notes, **(extra or {})}, f, indent=1)
        return 0 if ok else 1


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()



if __name__ == "__main__":
    if sys.argv[1:] == ["--anomaly-regex"]:
        print(anomaly_regex())
        sys.exit(0)
    print("usage: oncard_common.py --anomaly-regex", file=sys.stderr)
    sys.exit(2)
