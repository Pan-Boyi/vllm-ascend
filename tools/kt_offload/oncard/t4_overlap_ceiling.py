# SPDX-License-Identifier: Apache-2.0
"""T4: is form C worth building? Three measured numbers, one layer, T=1 decode.

  1. node_us    one host-func node's occupancy of the compute stream
                (graph with the node vs without; node empty, and with one kt
                submit; plus the two-node placement form C would actually use)
  2. window_us  the NPU work that sits between the submit and the wait -- this
                layer's dispatch + GMM + combine, measured, not taken from a
                profile average
  3. ceiling    min(cpu_us, window_us) - node_us, per layer and per step

and, because the question was put as "ours versus sglang's":

  * eager, three arms, comparable: serial (today's shape) / sglang's shape
    (submit and the copies on a side stream, join before the merge) / form C's
    shape (one stream, submit then wait around the NPU work);
  * graph, sglang's shape is attempted too. If capture_end raises, the error
    text is reported verbatim as a result -- not as a crash, and not as
    "ours runs and theirs does not".

THIS TEST MEASURES TIME ONLY. It validates no numerical result: the arms hand
their outputs nowhere, form C is emulated by hand-placed nodes, and nothing
here may be cited as evidence that form C is correct (t2/t3 are that shape of
test). The eager numbers also do not carry over to graph mode: capture and
replay have costs of their own and the node cost only exists in a graph.

Every number is a distribution over >= 20 interleaved rounds (median/p10/p90),
never a single point, and the arithmetic lives in t4_model.py so the CPU
self-test can exercise every branch, including the refusals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oncard_common as C  # noqa: E402
import t4_model as M  # noqa: E402

PLAN = """T4 plan (one layer, T=1 decode shape, timing only):
  arms, eager    : serial | sglang-shape (side stream fork/join) | formC-shape (one stream, submit..wait)
  arms, graph    : no-node | 1 empty node | 2 empty nodes (form C placement)
                 | 1 deliberately fat node (instrument control) | sglang-shape capture (may raise 107025)
  numbers        : node_us, window_us, ceiling = min(cpu_us, window_us) - node_us
  required checks: negative control (graph no-node ~= eager NPU-only), instrument control (fat node
                   must be measurable), >= 20 samples per arm, 1-node vs 2-node additivity
  mutation arm   : feed the fat node's measured cost into the decision model; the verdict must flip
  refusals       : fewer than 20 samples, a non-finite sample, or a cross-source disagreement are
                   reported as failures/inconsistencies, never averaged away
  NOT checked    : any numerical result. This test cannot show form C is correct."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n", type=int, default=100, help="resident experts per layer")
    ap.add_argument("--tokens", type=int, default=1, help="decode shape")
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--numa", default="1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fat-us", type=float, default=500.0, help="instrument control: a node that burns this long")
    ap.add_argument("--tol", type=float, default=0.35, help="negative-control tolerance")
    ap.add_argument("--layers", type=int, default=40, help="layers in the model, for the per-step figure")
    ap.add_argument("--ab-node-us", type=float, default=None, help="per-operation cost derived from the packed A/B")
    ap.add_argument("--packed-gain-ms", type=float, default=None, help="measured packed gain per step, if known")
    ap.add_argument("--profile-cpu-us", type=float, default=None, help="per-layer CPU wait from a profile, to compare")
    ap.add_argument("--require-cross-check", action="store_true", help="fail if no independent source is given")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without touching the device")
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()
    if args.reps < 20:
        ap.error("--reps must be >= 20: a single point cannot show whether a difference means anything")

    print(PLAN, flush=True)
    if args.dry_run or os.environ.get("ONCARD_DRY") == "1":
        print("t4: --dry-run, nothing was measured and no device was touched", flush=True)
        return 0

    import npu_moe as N
    import numpy as np
    import torch

    import vllm_ascend.kt_offload.layer as KL

    v = C.Verdict("T4 overlap ceiling (timing only)")
    v.note("this test validates no numerical result; form C's correctness is not in scope")
    v.note("eager numbers do not carry to graph mode (capture/replay costs, node cost exists only in a graph)")

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    L, T = args.layer, args.tokens
    numa = [int(s) for s in args.numa.split(",")]
    cfg = N.kt_cfg("prefix", args.n, None, args.threads, numa, graph=True)
    off, _ = N.make_offload(cfg, L, capture_sizes=[T])
    pl = off.placement
    raw = N.load_raw_layer(ck, L, list(pl.resident_ids))
    lay = N.build_layer(raw, pl.num_slots, dev, offload=off)
    comm = N.comm_method(pl.num_slots)

    rng = np.random.default_rng(args.seed + 4000 + L)
    x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16).to(dev)
    ids, w = C.make_routing(rng, T, pl.cpu_ids, list(pl.resident_ids))
    ids, w = ids.to(dev), w.to(dev)
    print(f"input: {C.coverage(ids.cpu(), pl.resident_mask()).line()}", flush=True)

    # the NPU half's routing, computed once: the graph arms must contain the
    # same NPU work production's replay does
    routing = off.route(x, w, ids, in_profile_run=False)
    off.merge(torch.zeros_like(x))  # drain the submit this route just made
    s_ids = routing.topk_ids.clone()
    s_w = routing.topk_weights.clone()
    s_x = x.clone()

    def npu_only():
        return N.npu_moe(lay, comm, s_x, s_w, s_ids)

    def kt_only():
        off.route(x, w, ids, in_profile_run=False)
        off.merge(torch.zeros_like(x))

    def eager_serial():
        r = off.route(x, w, ids, in_profile_run=False)
        cpu_half = off.merge(torch.zeros_like(x))  # wait before any NPU work is enqueued
        return N.npu_moe(lay, comm, x, r.topk_weights, r.topk_ids) + cpu_half

    def eager_form_c():
        r = off.route(x, w, ids, in_profile_run=False)  # submit
        pre = N.npu_moe(lay, comm, x, r.topk_weights, r.topk_ids)
        return off.merge(pre)  # wait, same stream

    def eager_sglang():
        side = KL._side_stream(dev)
        with torch.npu.stream(side):
            r = off.route(x, w, ids, in_profile_run=False)  # kt's copies go on the side stream
        pre = N.npu_moe(lay, comm, x, r.topk_weights, r.topk_ids)
        ev_c = torch.npu.Event()
        ev_c.record()
        side.wait_event(ev_c)
        with torch.npu.stream(side):
            out = off.merge(pre)
        ev_s = torch.npu.Event()
        ev_s.record(side)
        torch.npu.current_stream(dev).wait_event(ev_s)
        return out

    def timed(fns: dict, reps: int) -> dict:
        """One round runs every arm once; the order rotates so a drift that
        moves all arms together drops out of the paired differences."""
        out = {k: [] for k in fns}
        names = list(fns)
        for r in range(reps + 2):  # first two rounds are warm-up, discarded
            k = r % len(names)
            for name in names[k:] + names[:k]:
                torch.npu.synchronize()
                t0 = time.perf_counter()
                fns[name]()
                torch.npu.synchronize()
                if r >= 2:
                    out[name].append((time.perf_counter() - t0) * 1e6)
        return out

    # ---- eager: the three comparable arms ---------------------------------
    eager = timed(
        {"serial": eager_serial, "sglang": eager_sglang, "formC": eager_form_c,
         "npu_only": npu_only, "kt_only": kt_only},
        args.reps,
    )
    eager_stats = {k: M.stats(s) for k, s in eager.items()}
    for k, s in eager_stats.items():
        print(f"  eager {k:9s} median {s['median']:9.1f} us  p10 {s['p10']:9.1f}  p90 {s['p90']:9.1f}  n={s['n']}",
              flush=True)
    v.check(all(s["n"] >= 20 for s in eager_stats.values()), f"every eager arm has >= 20 samples ({args.reps})")
    saved_sglang = M.paired_delta(eager["serial"], eager["sglang"])
    saved_formc = M.paired_delta(eager["serial"], eager["formC"])
    print(f"  eager saved by sglang shape: median {saved_sglang['median']:.1f} us "
          f"(p10 {saved_sglang['p10']:.1f}, p90 {saved_sglang['p90']:.1f})", flush=True)
    print(f"  eager saved by formC shape : median {saved_formc['median']:.1f} us "
          f"(p10 {saved_formc['p10']:.1f}, p90 {saved_formc['p90']:.1f})", flush=True)

    # ---- graph: node cost and the window ---------------------------------
    payload = ("t4", 0)
    KL._subscribe_stream(torch.npu.current_stream(dev))

    def noop(_payload):
        return None

    def fat(_payload):
        end = time.perf_counter() + args.fat_us / 1e6
        while time.perf_counter() < end:
            pass

    # The "node does one kt submit" arm is deliberately absent: it would leave
    # a task submitted inside a replay and synced on the host afterwards, and
    # kt has not confirmed that is legal for its FIFO worker (production keeps
    # submit and sync in one node so a task's exception is re-raised). An extra
    # number is not worth risking kt's state; the empty-node arms give the
    # node cost, which is what the decision needs.

    def build(nodes) -> tuple:
        g = torch.npu.NPUGraph()
        stream = torch.npu.current_stream(dev)
        with torch.npu.graph(g):
            for fn in nodes.get("before", []):
                KL._launch_host_func(stream, fn, payload)
            out = npu_only()
            for fn in nodes.get("after", []):
                KL._launch_host_func(stream, fn, payload)
        torch.npu.synchronize()
        return g, out

    graphs, build_errors = {}, {}
    wanted = {
        "g_none": {},
        "g_one": {"before": [noop]},
        "g_two": {"before": [noop], "after": [noop]},
        "g_fat": {"before": [fat]},
    }
    for name, nodes in wanted.items():
        try:
            graphs[name] = build(nodes)
        except Exception as exc:  # noqa: BLE001
            build_errors[name] = f"{type(exc).__name__}: {str(exc)[:400]}"
            v.note(f"{name}: capture failed -- {build_errors[name]}")

    def replay_of(name):
        g, _ = graphs[name]

        def run():
            g.replay()
            torch.npu.synchronize()

        return run

    gr = timed({name: replay_of(name) for name in graphs}, args.reps)
    graph_stats = {k: M.stats(s) for k, s in gr.items()}
    for k, s in graph_stats.items():
        print(f"  graph {k:9s} median {s['median']:9.1f} us  p10 {s['p10']:9.1f}  p90 {s['p90']:9.1f}  n={s['n']}",
              flush=True)

    ok, line = M.negative_control(gr["g_none"], eager["npu_only"], args.tol)
    v.check(ok, f"negative control: {line}")

    d_one = M.paired_delta(gr["g_one"], gr["g_none"])
    d_two = M.paired_delta(gr["g_two"], gr["g_none"])
    d_fat = M.paired_delta(gr["g_fat"], gr["g_none"])
    print(f"  node cost: 1 empty node {d_one['median']:.1f} us (p10 {d_one['p10']:.1f}, p90 {d_one['p90']:.1f}); "
          f"2 nodes {d_two['median']:.1f} us; fat node {d_fat['median']:.1f} us", flush=True)
    ok, line = M.instrument_control(d_fat["median"], args.fat_us)
    v.check(ok, f"instrument control: {line}")
    per_node_from_two = d_two["median"] / 2.0
    add_ok = abs(per_node_from_two - d_one["median"]) <= max(20.0, 0.5 * max(per_node_from_two, d_one["median"]))
    v.check(add_ok, f"additivity: 1 node {d_one['median']:.1f} us vs half of 2 nodes {per_node_from_two:.1f} us")

    # ---- sglang's shape, in a graph ---------------------------------------
    sg = {"captured": None, "error": None, "where": None}
    try:
        gs = torch.npu.NPUGraph()
        side = KL._side_stream(dev)
        KL._subscribe_stream(side)
        with torch.npu.graph(gs):
            ev1 = torch.npu.Event()
            ev1.record()
            side.wait_event(ev1)
            with torch.npu.stream(side):
                KL._launch_host_func(side, noop, payload)
            ev2 = torch.npu.Event()
            ev2.record(side)
            torch.npu.current_stream(dev).wait_event(ev2)
            npu_only()
        torch.npu.synchronize()
        sg["captured"] = True
        v.note("UNEXPECTED: sglang's fork/join shape captured on this platform; re-open the M3 conclusion")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()
        ours = [ln.strip() for ln in tb if "t4_overlap_ceiling.py" in ln or "kt_offload" in ln]
        sg["captured"] = False
        sg["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
        sg["where"] = ours[-1] if ours else tb[-2] if len(tb) > 1 else ""
        v.note(f"sglang's fork/join shape does NOT capture here: {sg['error']}")
        v.note(f"  raised at {sg['where']}")

    # ---- the three numbers and the verdict --------------------------------
    cpu_us = eager_stats["kt_only"]["median"]
    window_us = graph_stats["g_none"]["median"]
    node_us = d_one["median"]
    ceil = M.ceiling(cpu_us, window_us, node_us, args.layers)
    print(f"\n  1) node_us   = {node_us:.1f} us/node   (form C adds {2 * node_us:.1f} us/layer for its two nodes)")
    print(f"  2) window_us = {window_us:.1f} us     (this layer's NPU work between submit and wait)")
    print(f"  3) cpu_us    = {cpu_us:.1f} us     (kt's per-layer CPU latency, submit..sync)")
    print(f"     ceiling   = {ceil['per_layer_us']:.1f} us/layer = {ceil['per_step_ms']:.2f} ms/step "
          f"(bound by {ceil['bound_by']})", flush=True)
    if args.profile_cpu_us is not None:
        prof = M.cross_check(cpu_us, args.profile_cpu_us)
        v.note(f"cpu_us vs the profile's per-layer wait: {prof['line']}")
    xc = M.cross_check(node_us, args.ab_node_us)
    v.note(f"cross-source: {xc['line']}")
    if args.require_cross_check:
        v.check(xc["state"] == "agree", f"independent sources agree: {xc['line']}")
    verdict = M.verdict(ceil, args.packed_gain_ms)
    print(f"\n  FORM C WORTH BUILDING: {verdict['worth_it']} -- {verdict['why']}", flush=True)

    # mutation arm on the decision rule, driven by measured data: with the fat
    # node's cost in place of the real one, the verdict must flip to "no"
    mutated = M.verdict(M.ceiling(cpu_us, window_us, d_fat["median"], args.layers), args.packed_gain_ms)
    v.must_be_red(
        mutated["worth_it"] is False,
        f"decision model with the fat node's {d_fat['median']:.1f} us in place of the node cost "
        f"(got worth_it={mutated['worth_it']})",
    )

    extra = {
        "validates_correctness": False,
        "eager_us": eager_stats,
        "eager_saved_vs_serial": {"sglang": saved_sglang, "formC": saved_formc},
        "graph_us": graph_stats,
        "graph_build_errors": build_errors,
        "node_cost_us": {"one_empty": d_one, "two_empty": d_two, "fat": d_fat},
        "numbers": {"node_us": node_us, "window_us": window_us, "cpu_us": cpu_us},
        "ceiling": ceil,
        "verdict": verdict,
        "cross_check": xc,
        "sglang_in_graph": sg,
        "args": vars(args),
    }
    return v.finish(os.path.join(args.out, "t4_result.json"), extra)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        print(json.dumps({"test": "T4", "pass": False, "error": traceback.format_exc()[-2000:]}))
        raise
