"""Is the ~1.8e-3 offset really the P-to-bf16 rounding?  Single variable, on CPU.

`ALIGNMENT_PROTOCOL.md` section 10.6 P1 records a **systematic** offset between the
CPU golden and the original tilelang `sparse_attn`: the tilelang kernel casts the
softmax probabilities to bf16 before the P·V gemm
(``acc_s_cast = T.alloc_shared((h, block), BF16)`` in
``/workspace/models/DeepSeek-V4.1-Flash/inference/kernel.py:311+``), while the CPU
reference keeps them in fp32 (``ref/kernel_cpu.py``: ``p`` stays fp32 into the
einsum).  Section 6.0 (c) budgets ~1.8e-3 for it.

That number was **reasoned from the mechanism, not measured** -- and running the
real tilelang kernel needs a GPU, which this machine does not have.

But the mechanism itself is reproducible here: round ``p`` to bf16 and change
nothing else.  What that answers:

* difference lands at the budgeted magnitude  ⇒ the attribution is confirmed and the
  budget can be set from a measurement instead of an estimate;
* it does not                                 ⇒ **the more useful outcome** -- our
  attribution for that offset is wrong, and a budget derived from it is wrong too.

⚠️ What this does NOT establish: that the real tilelang kernel differs from the CPU
reference by exactly this and nothing else.  It establishes the size of **this one
mechanism**.  Any further difference would need the kernel itself.

    python measure_p1_pv_bf16.py [--dump DIR] [--layers 3]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import load_file

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "ref"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import kernel_cpu as kc                                       # noqa: E402

DUMPS = "/workspace/code/dsv4_single_card/vllm_project/golden_dumps"


def sparse_attn_pv(q, kv, attn_sink, topk_idxs, softmax_scale, p_dtype=None):
    """``kernel_cpu.sparse_attn`` with one knob: the dtype P carries into P·V.

    Deliberately a copy rather than a flag on the shipped function -- the golden's
    own path must stay exactly as it is while this measures an alternative.
    """
    b, m, h, d = q.shape
    topk = topk_idxs.size(-1)
    out = torch.empty_like(q)
    sink = attn_sink[:h].float()
    kv_f = kv.float()
    if topk == 0:
        out.zero_()
        return out
    for start in range(0, m, kc.SPARSE_ATTN_Q_CHUNK):
        stop = min(start + kc.SPARSE_ATTN_Q_CHUNK, m)
        idx = topk_idxs[:, start:stop].long()
        valid = idx >= 0
        safe = idx.clamp_min(0)
        gather_idx = safe.reshape(b, -1, 1).expand(-1, -1, d)
        kv_g = torch.gather(kv_f, 1, gather_idx).view(b, stop - start, topk, d)
        kv_g = kv_g * valid.unsqueeze(-1)
        qc = q[:, start:stop].float()
        scores = torch.einsum("bchd,bctd->bcht", qc, kv_g) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        mx = scores.amax(dim=-1).clamp_min(-1e30)
        p = torch.exp(scores - mx.unsqueeze(-1))
        denom = p.sum(dim=-1) + torch.exp(sink.view(1, 1, h) - mx)
        # 🔴 the one variable: tilelang materialises P in bf16 before the gemm
        p_pv = p.to(p_dtype).float() if p_dtype is not None else p
        o = torch.einsum("bcht,bctd->bchd", p_pv, kv_g) / denom.unsqueeze(-1)
        out[:, start:stop] = o.to(q.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(DUMPS, "len1100_fp64"))
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()

    print("P·V with P in fp32 (golden) vs P rounded to bf16 (tilelang's shape)")
    print(f"inputs: {a.dump}\n")
    print(f"{'layer':>5} {'rows':>6} {'max|diff|':>11} {'rel_l2':>11} "
          f"{'max rel':>11} {'signed?':>9}")
    worst = 0.0
    seen = 0
    for i in range(a.layers):
        p = os.path.join(a.dump, f"layer_{i:02d}.safetensors")
        if not os.path.exists(p):
            continue
        b = load_file(p)
        need = ("sa_in.q", "sa_in.kv", "sa_in.attn_sink", "sa_in.topk_idxs",
                "sa_in.softmax_scale")
        if not all(k in b for k in need):
            print(f"{i:>5}  (dump has no sa_in.* -- regenerate with the current taps)")
            continue
        seen += 1
        q, kv = b["sa_in.q"], b["sa_in.kv"]
        sink, idxs = b["sa_in.attn_sink"], b["sa_in.topk_idxs"]
        scale = float(b["sa_in.softmax_scale"][0])
        o32 = sparse_attn_pv(q, kv, sink, idxs, scale, None)
        o16 = sparse_attn_pv(q, kv, sink, idxs, scale, torch.bfloat16)
        d = (o32.float() - o16.float())
        rel = float(d.norm() / max(o32.float().norm().item(), 1e-30))
        nz = o32.float().abs() > 1e-6
        maxrel = float((d.abs()[nz] / o32.float().abs()[nz]).max()) if nz.any() else 0.0
        pos = float((d[d != 0] > 0).float().mean()) if (d != 0).any() else float("nan")
        worst = max(worst, rel)
        print(f"{i:>5} {q.shape[1]:>6} {float(d.abs().max()):>11.3e} {rel:>11.3e} "
              f"{maxrel:>11.3e} {100*pos:>8.1f}%")
        del b

    if not seen:
        print("\nNOTHING CHECKED -- no layer had the sa_in.* taps")
        sys.exit(1)
    print(f"\nworst rel_l2 over {seen} layers: {worst:.3e}")
    print(f"section 6.0 (c) budgets ~1.8e-3 for this mechanism")
    lo, hi = 5e-4, 6e-3
    if lo <= worst <= hi:
        print(f"  => within a factor of ~3 of the budget: the attribution holds, and the")
        print(f"     budget can now be set from {worst:.3e} instead of from an estimate.")
    else:
        print(f"  => 🔴 NOT the budgeted magnitude. The attribution for that offset is")
        print(f"     wrong, and any threshold derived from it is wrong too. Do not")
        print(f"     'adjust the budget' -- find what else differs.")
    print("\n⚠️ This measures ONE mechanism, not 'CPU golden vs tilelang'. Any further")
    print("   difference between the two would still need the kernel itself.")
    sys.exit(0)


if __name__ == "__main__":
    main()
