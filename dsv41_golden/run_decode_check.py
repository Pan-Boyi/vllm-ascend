"""Criterion G-1: prefill and decode must agree on the same text.

Feed the golden prompt two ways -- once as a single prefill, once as a shorter
prefill followed by one-token steps -- and compare.  Anything the decode path does
differently shows up here: the sliding-window ring buffer, the compressor's
partial group carried across steps, the indexer's single-query branch, and the
engram hash cache.

None of that is exercised by prefill, and all of it lives in buffers registered
``persistent=False`` inside a block that the streaming runner frees as soon as the
layer is done -- which is why ``golden.LayerStateStore`` exists.

Two modes, because one of them cannot answer the question on its own
-------------------------------------------------------------------
**End-to-end** (default) compares the decode logits against the full prefill's.
It answers "does decode produce the same answer", but its noise floor is not
small: a 20-token prefill and the first 20 positions of a 27-token prefill are
*already* different, because the sliding-window index tensor is a different width
and its reduction rounds differently, and bf16 plus causal attention spreads that (see
ALIGNMENT_PROTOCOL.md section 0.2).  So this mode measures that floor in the same
run -- the shared prefill prefix -- and reports it as the control.  Its verdict is
**top-1 agreement**, which is the business-level reading of G-1.

**Isolated** (``--force-layer-in GOLDEN_DUMP``) feeds every block the golden's own
``layer.in`` for the position being decoded, so cross-layer accumulation is gone
and each layer is judged on its own.  Its verdict is per-layer numerical
agreement, and it is the mode that can actually localise a decode bug.

Routing is forced in both (``--routing-plan``).  Prefill and decode reach the same
value by different arithmetic, and a ~1e-7 difference flips MoE routing for a
fifth of the tokens by layer 2; left free, either mode would measure routing
divergence instead of the decode path.  The plan is addressed by absolute
sequence position, which is exactly why it works unchanged for a decode step.

    DU=/workspace/code/dsv4_single_card/vllm_project/golden_dumps
    python run_decode_check.py --cut 20 --routing-plan $DU/plan_full40
    python run_decode_check.py --cut 20 --routing-plan $DU/plan_full40 \
                               --force-layer-in $DU/full40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_golden import GOLDEN_PROMPT  # noqa: E402  the one fixed input

# Per-layer limits for the isolated mode, from ALIGNMENT_PROTOCOL.md section 6.2.
ISOLATED_LIMITS = {
    "layer.out": 1.5e-2, "hc_post": 1.5e-2, "attn.out": 1.5e-2,
    "moe.out": 4e-2, "hc_mixes": 5e-3, "logits": 1e-2,
    "_default": 5e-3,
}
# Never gated.
#   moe_routing_margin -- a difference of two adjacent selection scores, so its
#     relative noise is large by construction; it is a diagnostic, not a threshold.
#   indexer_kv_offset  -- metadata, and it is *supposed* to differ between the two
#     paths (the window tensor is `seqlen` long in prefill, `window_size` in
#     decode).  It is the thing that makes indexer_topk.out comparable, not a
#     quantity to compare.
NEVER_GATE = ("moe_routing_margin", "indexer_kv_offset")

# Below this, a differing expert selection is a numerical near-tie rather than a
# bug -- ALIGNMENT_PROTOCOL.md section 5 step 2.  Measured for scale: at layer 22
# the median margin is 2.7e-2 and the one token whose routing flipped between the
# prefill and decode paths had 9.3e-5, the smallest of all 27.
NEAR_TIE_MARGIN = 1e-3


# See compare_dumps.BF16_FLOOR: 1 bf16 ULP is 3.9e-3 relative per element, so a
# purely systematic one-ULP shift already reaches that with nothing wrong.
BF16_FLOOR = 3 * 2 ** -8         # 1.17e-2
BF16_TENSORS = (
    "layer.in", "layer.out", "hc_pre", "hc_post", "attn_norm.out", "attn.in",
    "attn.out", "ffn_norm.out", "moe.in", "moe.out", "engram.out",
    "compressor.out",
)


def limit_for(name: str) -> float:
    best, best_len = ISOLATED_LIMITS["_default"], -1
    for k, v in ISOLATED_LIMITS.items():
        if k != "_default" and name.startswith(k) and len(k) > best_len:
            best, best_len = v, len(k)
    if name.startswith(BF16_TENSORS):
        best = max(best, BF16_FLOOR)
    return best


def _unoffset(idxs: torch.Tensor, offset: torch.Tensor | None):
    """Turn [window ; compressed] KV indices back into compressed positions.

    ``-1`` is an empty slot and stays ``-1``.  Without the recorded offset there is
    nothing to normalise against, so the caller skips the comparison rather than
    guessing.
    """
    if offset is None:
        return None
    off = int(offset.flatten()[0])
    return torch.where(idxs < 0, idxs, idxs - off)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    n = b.norm()
    return float((a - b).norm() / n) if n > 0 else float((a - b).norm())


class MemoryRecorder:
    """``golden.Recorder`` without the filesystem -- one decode step is small."""

    def __init__(self):
        self.enabled = True
        self.buf: dict[str, torch.Tensor] = {}
        self.tags: dict[str, dict[str, torch.Tensor]] = {}

    def add(self, name, t):
        if t is None:
            return
        if isinstance(t, (tuple, list)):
            for k, sub in enumerate(t):
                self.add(f"{name}.{k}", sub)
            return
        if not isinstance(t, torch.Tensor):
            return
        self.buf[name] = t.detach().clone()

    def flush(self, tag):
        self.tags[tag] = self.buf
        self.buf = {}

    def write_manifest(self, extra):
        pass


def compare_step_to_golden(rec: MemoryRecorder, dump_dir: str, pos: int, n_layers: int,
                           forced: bool = False):
    """Every recorded layer tensor against the golden prefill's slice at ``pos``."""
    from safetensors.torch import load_file

    rows, failures, near_ties = [], [], []
    for i in range(n_layers):
        tag = f"layer_{i:02d}"
        if tag not in rec.tags:
            continue
        ref = load_file(os.path.join(dump_dir, f"{tag}.safetensors"))
        for name, got in rec.tags[tag].items():
            if name not in ref:
                continue
            want = ref[name]
            # The token axis is dim 1 for [1, L, ...] activations and dim 0 for the
            # flat [L, topk] gate tensors.  Compressor latents live on a strided
            # axis (one per compress_ratio positions) and a decode step emits one
            # only when its group closes, so they are compared separately.
            if want.dim() >= 3 and want.size(0) == 1 and want.size(1) > 1:
                if want.size(1) <= pos:
                    continue
                want = want[:, pos:pos + 1]
            elif want.dim() >= 1 and want.size(0) > 1 and got.size(0) == 1:
                if want.size(0) <= pos:
                    continue
                want = want[pos:pos + 1]
            if tuple(want.shape) != tuple(got.shape):
                continue                       # shape differs by design; skip quietly
            if want.dtype != got.dtype:
                continue
            if not want.is_floating_point():
                wi, gi = want.to(torch.int64), got.to(torch.int64)
                if name == "indexer_topk.out":
                    # Values index the concatenated [window ; compressed] KV, and the
                    # window part is `seqlen` long in prefill but `window_size` long
                    # in decode -- so the same compressed position gets a different
                    # number in each path.  Normalise by the recorded offset;
                    # comparing raw would fail on a correct implementation.
                    wi = _unoffset(wi, ref.get("indexer_kv_offset"))
                    gi = _unoffset(gi, rec.tags[tag].get("indexer_kv_offset"))
                    if wi is None or gi is None:
                        continue
                elif name == "moe_gate.out.1":
                    # Expert selection is a set: topk order does not affect the
                    # result, so an order swap is not a mismatch.
                    wi, gi = wi.sort(-1).values, gi.sort(-1).values
                same = bool(torch.equal(wi, gi))
                gated = not name.startswith(NEVER_GATE)
                if not same and name == "moe_gate.out.1" and forced:
                    # Under forced routing the layer's own top-k is not what reached
                    # the experts, so its disagreement cannot break the numerical
                    # comparison.  Run the protocol's own step-2 diagnostic instead
                    # of either gating on it or waving it through: a near-tie is
                    # expected, a comfortable margin picked differently is a bug.
                    margin = ref.get("moe_routing_margin")
                    mval = None if margin is None else float(margin.float()[pos])
                    med = None if margin is None else float(margin.float().median())
                    near_tie = mval is not None and mval < NEAR_TIE_MARGIN
                    note = (f"own gate differs (injection overrode it): "
                            f"{gi.flatten().tolist()} vs golden {wi.flatten().tolist()}; "
                            f"margin={mval:.2e} vs layer median {med:.2e}"
                            if mval is not None else "own gate differs, no margin recorded")
                    if near_tie:
                        rows.append((i, name, 0.0, 0.0, False))
                        near_ties.append((i, name, note))
                        continue
                    failures.append((i, name, f"NOT a near-tie: {note}"))
                    rows.append((i, name, float("nan"), 0.0, True))
                    continue
                if not same and name == "moe_gate.out.0" and forced:
                    # Weights follow the selection, so they inherit its verdict.
                    near_ties.append((i, name, f"own gate weights differ, rel_l2="
                                               f"{rel_l2(got, want):.2e} (follows out.1)"))
                    rows.append((i, name, 0.0, 0.0, False))
                    continue
                if not same and gated:
                    failures.append((i, name, f"int mismatch: {gi.flatten()[:8].tolist()} "
                                              f"vs golden {wi.flatten()[:8].tolist()}"))
                rows.append((i, name, 0.0 if same else float("nan"), 0.0, gated))
                continue
            r = rel_l2(got, want)
            lim = limit_for(name)
            gated = not name.startswith(NEVER_GATE)
            if forced and name == "moe_gate.out.0" and r > lim:
                # Same reasoning as out.1: these are the layer's own weights, not
                # the ones the experts used.  A near-tie flip moves them a lot.
                near_ties.append((i, name, f"own gate weights differ, rel_l2={r:.2e} "
                                           "(follows out.1; the injected weights are "
                                           "what the experts used)"))
                rows.append((i, name, r, lim, False))
                continue
            rows.append((i, name, r, lim, gated))
            if gated and r > lim:
                failures.append((i, name, f"rel_l2={r:.3e} > {lim:.1e}"))
    return rows, failures, near_ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", type=int, default=20,
                    help="prefill tokens [0:cut], then step the rest one at a time")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--window-size", type=int, default=0, metavar="N",
                    help="override args.window_size, to make the ring buffer wrap inside a "
                         "27-token prompt. Self-consistency still holds -- both paths use "
                         "the same config -- so this is a valid structural test even though "
                         "the numbers are not the real model's")
    ap.add_argument("--floor-gb", type=float, default=150.0)
    ap.add_argument("--fp64", action="store_true")
    ap.add_argument("--routing-plan", type=str, default=None)
    ap.add_argument("--force-layer-in", type=str, default=None, metavar="GOLDEN_DUMP",
                    help="isolated mode: feed every block the golden's layer.in for "
                         "the decoded position, and judge each layer on its own")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.bfloat16)

    import kernel_cpu
    if a.fp64:
        kernel_cpu.GOLDEN_DOT_DTYPE = torch.float64

    import golden as G
    from ckpt import Checkpoint
    from transformers import AutoTokenizer

    plan = None
    if a.routing_plan:
        from routing_plan import RoutingPlan
        plan = RoutingPlan.load(a.routing_plan)

    ckpt = Checkpoint(G.MODEL_DIR)
    overrides = {"window_size": a.window_size} if a.window_size else {}
    args = G.load_args(max_batch_size=1, max_seq_len=a.max_seq_len, **overrides)
    tokenizer = AutoTokenizer.from_pretrained(G.MODEL_DIR)
    ids = tokenizer(a.prompt or GOLDEN_PROMPT)["input_ids"]
    input_ids = torch.tensor([ids])
    n = input_ids.size(1)
    cut = a.cut
    assert 0 < cut < n, f"--cut must be in 1..{n - 1} (prompt has {n} tokens)"

    isolated = a.force_layer_in is not None
    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"accum={kernel_cpu.GOLDEN_DOT_DTYPE}")
    print(f"mode            {'ISOLATED (per-layer)' if isolated else 'END-TO-END (logits)'}")
    print(f"prompt          {n} tokens; prefill[0:{cut}] then {n - cut} decode steps; "
          f"layers={a.layers}")
    print(f"forced routing  {a.routing_plan or 'NO -- would measure routing divergence'}")
    if isolated:
        print(f"forced layer.in {a.force_layer_in}")
    print(f"window_size={args.window_size}  kv_source={args.kv_source_layers}  "
          f"index_source={args.index_source_layers}")
    print()

    guard = G.MemoryGuard(a.floor_gb)
    result: dict = {}

    # -- reference: one prefill over the whole prompt ----------------------
    # Skipped in isolated mode, where the golden dump on disk already is the
    # reference and re-running it would only add its own rounding.
    ref_logits = None
    if not isolated:
        model = G.GoldenModel(args, ckpt, tokenizer, guard, verbose=False)
        t0 = time.time()
        ref_logits = model.forward(input_ids, None, n_layers=a.layers, routing_plan=plan)
        print(f"[ref] full prefill [0:{n}]      {time.time() - t0:6.1f}s  "
              f"logits {tuple(ref_logits.shape)}  rss {G.rss_gb():.1f} GB")
        del model

    # -- short prefill, then one token at a time --------------------------
    model2 = G.GoldenModel(args, ckpt, tokenizer, guard, verbose=False)
    state = G.LayerStateStore()
    t0 = time.time()
    pre = model2.forward(input_ids[:, :cut], None, n_layers=a.layers,
                         routing_plan=plan, state=state,
                         layer_in_dir=a.force_layer_in)
    st = state.summary()
    print(f"[run] prefill [0:{cut}]         {time.time() - t0:6.1f}s  "
          f"state: {st['layers']} layers / {st['mb']} MB / "
          f"{st['buffers_min']}-{st['buffers_max']} buffers per layer")
    for nm, cnt in sorted(st["carried"].items()):
        print(f"        carried {nm:36s} on {cnt:2d} layers")

    control = None
    if not isolated:
        control = rel_l2(pre[0, :cut], ref_logits[0, :cut])
        print(f"        CONTROL: prefill[0:{cut}] vs prefill[0:{n}][:{cut}] "
              f"logits rel_l2 = {control:.3e}")
        print("        (two pure prefills, no decode involved -- this is the floor)")
    print()

    per_pos: dict[int, dict] = {}
    all_failures: list = []
    all_near_ties: list = []
    for p in range(cut, n):
        t0 = time.time()
        rec = MemoryRecorder() if isolated else None
        step = model2.forward(input_ids[:, p:p + 1], rec, n_layers=a.layers,
                              start_pos=p, routing_plan=plan, state=state,
                              layer_in_dir=a.force_layer_in)
        assert step.shape[1] == 1, step.shape
        dt = time.time() - t0

        if isolated:
            rows, failures, near_ties = compare_step_to_golden(
                rec, a.force_layer_in, p, a.layers, forced=plan is not None)
            all_failures += [(p, *f) for f in failures]
            all_near_ties += [(p, *t) for t in near_ties]
            gated = [r for r in rows if r[4] and r[2] == r[2]]
            worst = max(gated, key=lambda r: r[2] / max(r[3], 1e-30)) if gated else None
            per_pos[p] = {"seconds": round(dt, 1), "n_tensors": len(rows),
                          "n_failures": len(failures),
                          "worst": None if worst is None else
                          {"layer": worst[0], "name": worst[1],
                           "rel_l2": worst[2], "limit": worst[3]}}
            flag = "ok  " if not failures else "FAIL"
            w = per_pos[p]["worst"]
            print(f"      {flag} decode pos {p:3d}  {len(rows):4d} tensors  "
                  f"{len(failures)} over limit  "
                  f"worst={w['name']}@L{w['layer']} {w['rel_l2']:.2e}/{w['limit']:.0e}"
                  f"  ({dt:.1f}s)")
        else:
            r = rel_l2(step[0, 0], ref_logits[0, p])
            t_ref = int(ref_logits[0, p].argmax())
            t_got = int(step[0, 0].argmax())
            per_pos[p] = {"seconds": round(dt, 1), "rel_l2": r,
                          "top1_ref": t_ref, "top1_got": t_got,
                          "top1_same": t_ref == t_got,
                          "vs_control": (r / control) if control else None}
            flag = "ok  " if t_ref == t_got else "FAIL"
            vc = f"{r / control:5.1f}x control" if control else "control is 0"
            print(f"      {flag} decode pos {p:3d}  rel_l2={r:.3e} "
                  f"({vc})  top1 {t_got}"
                  f"{'' if t_ref == t_got else f' != {t_ref}'}  ({dt:.1f}s)")

    print()
    if isolated:
        ok = not all_failures
        print(f"tensors over their per-layer limit: {len(all_failures)}")
        for p, i, name, why in all_failures[:15]:
            print(f"  pos {p} layer {i} {name}: {why}")
        if all_near_ties:
            print(f"near-ties (informational, not gated): {len(all_near_ties)}")
            for p, i, name, why in all_near_ties[:10]:
                print(f"  pos {p} layer {i} {name}: {why}")
        result = {"failures": [[p, i, nm, w] for p, i, nm, w in all_failures],
                  "near_ties": [[p, i, nm, w] for p, i, nm, w in all_near_ties]}
    else:
        all_top1 = all(v["top1_same"] for v in per_pos.values())
        worst = max(per_pos.values(), key=lambda v: v["rel_l2"])
        ok = all_top1
        print(f"control (two prefills)   {control:.3e}")
        if control:
            print(f"worst decode rel_l2      {worst['rel_l2']:.3e}  "
                  f"({worst['rel_l2'] / control:.1f}x control)")
        else:
            # A zero control is informative, not a division problem: it means the
            # two prefills agreed bit for bit, so any decode difference is the
            # decode path's alone rather than sitting on top of a shared floor.
            print(f"worst decode rel_l2      {worst['rel_l2']:.3e}  "
                  f"(control is exactly 0 -- the two prefills were bit-identical, "
                  f"so this is decode's own contribution with nothing subtracted)")
        print(f"top-1 all agree          {all_top1}")
        print("(rel_l2 is reported against the control, not gated: prefill and decode")
        print(" reach the same value by different arithmetic. top-1 is the verdict.)")
        result = {"control_rel_l2": control, "top1_all_agree": all_top1}

    print(f"peak RSS                 {guard.peak_rss:.1f} GB "
          f"({guard.peak_anon:.1f} anon + {guard.peak_file:.1f} file-backed)   "
          f"min avail {guard.min_avail:.1f} GB")
    if plan is not None:
        print(f"routing                  {plan.summary()}")
    print()
    print(f"G-1 prefill/decode self-consistency "
          f"({'isolated' if isolated else 'end-to-end'}): {'PASS' if ok else 'FAIL'}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"mode": "isolated" if isolated else "end_to_end",
                       "n_tokens": n, "cut": cut, "layers": a.layers,
                       "accum": str(kernel_cpu.GOLDEN_DOT_DTYPE),
                       "routing_plan": a.routing_plan,
                       "force_layer_in": a.force_layer_in,
                       "routing_summary": plan.summary() if plan else None,
                       "state": st, "per_position": per_pos,
                       "pass": ok, **result}, f, indent=2)
        print(f"wrote {a.json}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
