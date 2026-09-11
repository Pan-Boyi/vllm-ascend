"""Driver: run the CPU golden over a fixed prompt and dump per-layer activations.

    python run_golden.py --layers 3 --dump dumps/smoke
    python run_golden.py --layers 40 --dump dumps/full --seq-len 16

Always run with LD_PRELOAD unset (jemalloc segfaults) and no NPU env vars set.
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

# Fixed reference prompt for the alignment protocol.  Short, deterministic, ASCII,
# and long enough that layer 2's compressor (ratio 2) produces several latents and
# the sliding window (128) is not the only KV source.
GOLDEN_PROMPT = (
    "The capital of France is Paris. "
    "Ascend NPU kernels must match this reference bit for bit in structure "
    "and within tolerance in value."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3, help="how many backbone layers to run")
    ap.add_argument("--dump", type=str, default=None, help="directory for per-layer safetensors")
    ap.add_argument("--seq-len", type=int, default=0,
                    help="truncate the prompt to this many tokens (truncate only -- "
                         "it does not pad; use --random-tokens N for a longer input)")
    ap.add_argument("--random-tokens", type=int, default=0, help="use N seeded random token ids instead")
    ap.add_argument("--prompt", type=str, default=None, help="override the fixed golden prompt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--perturb-ulp", type=str, default=None, metavar="TOK,DIM",
                    help="move one element of embed.out by exactly one bf16 ULP "
                         "(the smallest difference the residual stream can hold, and the "
                         "size two different implementations start with). Measures what "
                         "that becomes after N layers")
    ap.add_argument("--no-quant", action="store_true",
                    help="bypass act_quant/fp4_act_quant so the two KV streams carry "
                         "NO quantization noise. The dump is then NOT the model's real "
                         "path -- it isolates the attention merge semantics from the "
                         "per-stream quantization, so a reference can be checked one "
                         "thing at a time")
    ap.add_argument("--indexer-post-rope", action="store_true",
                    help="DELIBERATELY WRONG ORDER: let the indexer see the latent "
                         "AFTER RoPE instead of before (ref/model.py:749 requires "
                         "before). Produces no error, only different top-k -- which is "
                         "the point: it is the negative arm for that ordering")
    ap.add_argument("--ep-sim", type=int, default=0,
                    help="also sum the experts the way EP=N would (per-rank partial "
                         "sums + all_reduce) and report the spread against the "
                         "golden's expert-id-order sum. Diagnostic; does not change "
                         "the dump")
    ap.add_argument("--batch", type=int, default=1, metavar="N",
                    help="replicate the prompt into a batch of N identical sequences. "
                         "Every row must then come out bit-identical to every other; "
                         "anything else is a batch-handling bug (a cache slot indexed by "
                         "the wrong dimension, a broadcast that should have been a gather). "
                         "The check is bit-exact, so it needs no tolerance")
    ap.add_argument("--window-size", type=int, default=0, metavar="N",
                    help="override args.window_size. Shrinking it below the prompt length "
                         "is how the ring buffer's wrap-around gets exercised at all: with "
                         "the real 128 and a 27-token prompt, prefill never takes the "
                         "seqlen>window branch and decode never wraps")
    ap.add_argument("--floor-gb", type=float, default=150.0)
    ap.add_argument("--fp64", action="store_true", help="accumulate the GEMMs in float64")
    ap.add_argument("--no-engram", action="store_true",
                    help="skip the engram lookup in layers 1 and 14 (no tokenizer needed). "
                         "The result is NOT a numerical reference -- those layers are "
                         "computed without their engram contribution. Use it for capacity "
                         "and timing measurements only")
    ap.add_argument("--routing-plan", type=str, default=None,
                    help="force the MoE expert choice from this plan (file or dir); "
                         "see ROUTING_INJECTION.md")
    ap.add_argument("--force-layer-in", type=str, default=None, metavar="GOLDEN_DUMP_DIR",
                    help="overwrite layer.in/pre_mix.in at each block entry from this "
                         "golden dump, making every layer an independent comparison")
    args_cli = ap.parse_args()

    torch.set_num_threads(args_cli.threads)
    torch.manual_seed(args_cli.seed)
    torch.set_default_dtype(torch.bfloat16)      # must precede any module construction

    import kernel_cpu
    if args_cli.fp64:
        kernel_cpu.GOLDEN_DOT_DTYPE = torch.float64

    import golden as G
    from ckpt import Checkpoint

    print(f"torch {torch.__version__}  threads={torch.get_num_threads()}  "
          f"avail={G.sys_available_gb():.1f} GB", flush=True)

    t0 = time.time()
    ckpt = Checkpoint(G.MODEL_DIR)
    overrides = {"window_size": args_cli.window_size} if args_cli.window_size else {}
    args = G.load_args(max_batch_size=args_cli.batch, max_seq_len=args_cli.max_seq_len,
                       **overrides)
    if overrides:
        print(f"OVERRIDE window_size={args.window_size} (real config is 128)", flush=True)
    print(f"config: n_layers={args.n_layers} dim={args.dim} experts={args.n_routed_experts} "
          f"engram_layers={args.engram_layer_ids} kv_source={args.kv_source_layers}", flush=True)

    tokenizer = None
    if not args_cli.no_engram:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(G.MODEL_DIR)

    if args_cli.random_tokens:
        g = torch.Generator().manual_seed(args_cli.seed)
        input_ids = torch.randint(0, args.vocab_size, (1, args_cli.random_tokens), generator=g)
    else:
        assert tokenizer is not None, "--no-engram needs --random-tokens"
        ids = tokenizer(args_cli.prompt or GOLDEN_PROMPT)["input_ids"]
        if args_cli.seq_len:
            ids = ids[: args_cli.seq_len]
        input_ids = torch.tensor([ids])
    if args_cli.batch > 1:
        input_ids = input_ids.repeat(args_cli.batch, 1)
    print(f"input_ids {tuple(input_ids.shape)}: {input_ids[0].tolist()}", flush=True)

    guard = G.MemoryGuard(args_cli.floor_gb)
    model = G.GoldenModel(args, ckpt, tokenizer, guard)
    model.ep_sim = args_cli.ep_sim

    if args_cli.no_quant:
        # Monkeypatched rather than edited into ref/kernel_cpu.py on purpose: the
        # golden's real numerics must stay untouched, and a flag leaves a trace in
        # the manifest where an edit would not.
        # Patch the names in the MODEL module, not in kernel_cpu: model.py does
        # `from kernel import act_quant, fp4_act_quant` (ref/model.py:13-14), so the
        # call sites resolve against model's own namespace and rebinding the source
        # module would have no effect.
        import model as _mm
        _real_aq, _real_fq = _mm.act_quant, _mm.fp4_act_quant

        def _aq(x, *a, **k):
            return x if (k.get("inplace") or (len(a) >= 4 and a[3])) else _real_aq(x, *a, **k)

        def _fq(x, *a, **k):
            return x if (k.get("inplace") or (len(a) >= 2 and a[1])) else _real_fq(x, *a, **k)

        _mm.act_quant, _mm.fp4_act_quant = _aq, _fq
        print("!!! --no-quant: act_quant/fp4_act_quant bypassed. This dump is NOT the "
              "model's real path and is NOT comparable to a normal dump. It exists to "
              "isolate the attention merge semantics from per-stream quantization.",
              flush=True)

    if args_cli.indexer_post_rope:
        model.indexer_post_rope = True
        print("!!! --indexer-post-rope: the indexer will see the latent AFTER RoPE. "
              "ref/model.py:749 requires BEFORE. This is the negative arm for that "
              "ordering -- the run will not error, it will just select different "
              "tokens.", flush=True)
    print(f"setup {time.time() - t0:.1f}s, rss {G.rss_gb():.1f} GB", flush=True)

    plan = None
    if args_cli.routing_plan:
        from routing_plan import RoutingPlan
        plan = RoutingPlan.load(args_cli.routing_plan)
        print(f"forced routing: {args_cli.routing_plan}  layers={len(plan.layers)} "
              f"topk={plan.topk} plan_accum={plan.meta.get('accum_dtype')}", flush=True)

    rec = G.Recorder(args_cli.dump, enabled=args_cli.dump is not None)
    t1 = time.time()
    pert = None
    if args_cli.perturb_ulp:
        pert = tuple(int(x) for x in args_cli.perturb_ulp.split(","))
    logits = model.forward(input_ids, rec, n_layers=args_cli.layers, routing_plan=plan,
                           layer_in_dir=args_cli.force_layer_in, perturb_ulp=pert)
    dt = time.time() - t1
    if plan is not None:
        print(f"  routing   {plan.summary()}")

    st = model.stats
    print()
    print(f"forward: {dt:.1f}s over {args_cli.layers} layers "
          f"({dt / max(args_cli.layers, 1):.1f}s/layer)")
    print(f"  read      {st['bytes_read'] / 2**30:.1f} GB of weights")
    print(f"  experts   {st['experts_loaded']} loaded "
          f"({st['experts_loaded'] / max(args_cli.layers, 1):.0f}/layer of {args.n_routed_experts})")
    print(f"  engram    {st['engram_rows']} rows gathered")
    print(f"  peak RSS  {guard.peak_rss:.1f} GB "
          f"({guard.peak_anon:.1f} anon + {guard.peak_file:.1f} file-backed/reclaimable)"
          f"    min avail {guard.min_avail:.1f} GB")
    print(f"  logits    {tuple(logits.shape)} dtype={logits.dtype}")
    print(f"  finite    {bool(torch.isfinite(logits).all())}")
    if args_cli.batch > 1:
        # Identical inputs must give identical outputs, row for row.  Compared as
        # bits, not within a tolerance: there is nothing here that could
        # legitimately differ between two rows carrying the same tokens.
        ref = logits[0]
        same = [bool(torch.equal(logits[b], ref)) for b in range(args_cli.batch)]
        print(f"  batch     {args_cli.batch} identical sequences; "
              f"rows bit-identical to row 0: {sum(same)}/{args_cli.batch}")
        if not all(same):
            bad = [b for b, ok in enumerate(same) if not ok]
            for b in bad[:3]:
                d = (logits[b] - ref).abs()
                print(f"    row {b}: max|diff| = {d.max().item():.3e} at "
                       f"token {int(d.max(dim=-1).values.argmax())}")
            print("  => BATCH BUG: identical inputs produced different outputs")
        else:
            print("  => batch handling is consistent")
    print(f"  last-tok  top5 {logits[0, -1].topk(5).indices.tolist()}")
    print(f"            vals {[round(v, 3) for v in logits[0, -1].topk(5).values.tolist()]}")
    if tokenizer is not None:
        print(f"            toks {[tokenizer.decode([i]) for i in logits[0, -1].topk(5).indices.tolist()]}")

    rec.write_manifest({
        # Provenance.  A dump is only usable as a baseline if you can tell what
        # produced it -- and "which stack" is the wrong question here: the golden
        # imports neither torch_npu nor CANN nor vllm, so it does not move when
        # those do.  What it *does* depend on is this repo's code and the
        # checkpoint bytes, so both are pinned by identity rather than by path.
        **G.provenance(),
        "prompt": None if args_cli.random_tokens else (args_cli.prompt or GOLDEN_PROMPT),
        "input_ids": input_ids[0].tolist(),
        "n_layers_run": args_cli.layers,
        "seed": args_cli.seed,
        "accum_dtype": str(kernel_cpu.GOLDEN_DOT_DTYPE),
        # Consumed by compare_dumps.py to pick the judgment order: with routing
        # forced, a routing mismatch is expected and must not gate the comparison.
        "force_route": ("off" if plan is None else "moe") + (
            "+layer_in" if args_cli.force_layer_in else ""),
        "force_layer_in": args_cli.force_layer_in,
        # Recorded so a comparison can run the causal-mask check automatically:
        # tokens before a perturbation can never see it, so they must stay
        # bit-exact for all 40 layers. See ALIGNMENT_PROTOCOL.md section 0.2.
        "perturb_ulp": list(pert) if pert else None,
        # Both of these make the dump deliberately unlike the model; recorded so a
        # comparison cannot mistake one for a normal reference.
        "no_quant": args_cli.no_quant,
        "indexer_post_rope": args_cli.indexer_post_rope,
        "routing_plan": args_cli.routing_plan,
        "routing_plan_accum_dtype": None if plan is None else plan.meta.get("accum_dtype"),
        "torch_version": torch.__version__,
        "model_dir": G.MODEL_DIR,
        "wall_seconds": round(dt, 2),
        "peak_rss_gb": round(guard.peak_rss, 2),
        "layer_seconds": [round(x, 2) for x in st["layer_times"]],
    })
    if args_cli.ep_sim:
        st_ep = G.EP_SIM_STATS
        print()
        print(f"--- EP={args_cli.ep_sim} simulation: same expert outputs, summed the "
              f"way {args_cli.ep_sim} ranks would")
        print("    (per-rank partial sums + all_reduce, vs the golden's expert-id-order sum;")
        print("     worst case over every all_reduce association, since it is not promised)")
        print(f"    {'layer':>5} {'max|diff| pre-shared':>21} {'relative':>10} "
              f"{'moe.out, fp32 reduce':>21} {'moe.out, lowp reduce':>21}")
        worst, worst16 = 0.0, 0.0
        for r in st_ep:
            worst = max(worst, r["max_abs_moe_out"])
            worst16 = max(worst16, r["max_abs_moe_out_reduce_in_out_dtype"])
            print(f"    {r['layer']:>5} {r['max_abs_pre_shared']:>21.3e} "
                  f"{r['rel_pre_shared']:>10.3e} {r['max_abs_moe_out']:>21.3e} "
                  f"{r['max_abs_moe_out_reduce_in_out_dtype']:>21.3e}")
        print(f"    worst over {len(st_ep)} layers:  fp32 reduce {worst:.3e}   "
              f"low-precision reduce {worst16:.3e}")
        if worst == 0.0:
            print("    => with the all_reduce in fp32, expert parallelism is INVISIBLE at")
            print("       the dumped precision: the cast to the residual dtype absorbs it")
            print("       entirely.  So EP sharding is not an excuse for a moe.out mismatch.")
        if worst16 > 0.0:
            print("    => but reducing in the residual dtype does show up. If the NPU")
            print("       all_reduce carries the low precision, that is a real floor and")
            print("       the criterion has to allow it -- so keep the reduce in fp32.")

    if args_cli.dump:
        print(f"  dumped to {args_cli.dump}")
        print("  files:", sorted(os.listdir(args_cli.dump))[:6], "...")


if __name__ == "__main__":
    main()
