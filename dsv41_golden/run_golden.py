"""Driver: run the CPU golden over a fixed prompt and dump per-layer activations.

    python run_golden.py --layers 3 --dump dumps/smoke
    python run_golden.py --layers 40 --dump dumps/full --prompt-tokens 16

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
    ap.add_argument("--seq-len", type=int, default=0, help="truncate/pad the prompt to this many tokens")
    ap.add_argument("--random-tokens", type=int, default=0, help="use N seeded random token ids instead")
    ap.add_argument("--prompt", type=str, default=None, help="override the fixed golden prompt")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--window-size", type=int, default=0, metavar="N",
                    help="override args.window_size. Shrinking it below the prompt length "
                         "is how the ring buffer's wrap-around gets exercised at all: with "
                         "the real 128 and a 27-token prompt, prefill never takes the "
                         "seqlen>window branch and decode never wraps")
    ap.add_argument("--floor-gb", type=float, default=150.0)
    ap.add_argument("--fp64", action="store_true", help="accumulate the GEMMs in float64")
    ap.add_argument("--no-engram", action="store_true", help="skip engram layers (no tokenizer needed)")
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
    args = G.load_args(max_batch_size=1, max_seq_len=args_cli.max_seq_len, **overrides)
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
    print(f"input_ids {tuple(input_ids.shape)}: {input_ids[0].tolist()}", flush=True)

    guard = G.MemoryGuard(args_cli.floor_gb)
    model = G.GoldenModel(args, ckpt, tokenizer, guard)
    print(f"setup {time.time() - t0:.1f}s, rss {G.rss_gb():.1f} GB", flush=True)

    plan = None
    if args_cli.routing_plan:
        from routing_plan import RoutingPlan
        plan = RoutingPlan.load(args_cli.routing_plan)
        print(f"forced routing: {args_cli.routing_plan}  layers={len(plan.layers)} "
              f"topk={plan.topk} plan_accum={plan.meta.get('accum_dtype')}", flush=True)

    rec = G.Recorder(args_cli.dump, enabled=args_cli.dump is not None)
    t1 = time.time()
    logits = model.forward(input_ids, rec, n_layers=args_cli.layers, routing_plan=plan,
                           layer_in_dir=args_cli.force_layer_in)
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
    print(f"  last-tok  top5 {logits[0, -1].topk(5).indices.tolist()}")
    print(f"            vals {[round(v, 3) for v in logits[0, -1].topk(5).values.tolist()]}")
    if tokenizer is not None:
        print(f"            toks {[tokenizer.decode([i]) for i in logits[0, -1].topk(5).indices.tolist()]}")

    rec.write_manifest({
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
        "routing_plan": args_cli.routing_plan,
        "routing_plan_accum_dtype": None if plan is None else plan.meta.get("accum_dtype"),
        "torch_version": torch.__version__,
        "model_dir": G.MODEL_DIR,
        "wall_seconds": round(dt, 2),
        "peak_rss_gb": round(guard.peak_rss, 2),
        "layer_seconds": [round(x, 2) for x in st["layer_times"]],
    })
    if args_cli.dump:
        print(f"  dumped to {args_cli.dump}")
        print("  files:", sorted(os.listdir(args_cli.dump))[:6], "...")


if __name__ == "__main__":
    main()
