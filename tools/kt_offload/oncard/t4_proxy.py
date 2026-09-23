# SPDX-License-Identifier: Apache-2.0
"""T4p: a 2-4 minute proxy for "how many ms/token would this mechanism save?".

Why this file exists (MOE_OFFLOAD_PLAN 4.19, 4.22, 4.25-4.28)
-------------------------------------------------------------
One whole-network A/B costs 1-2 hours: 3-5 min of start-up per arm (kt loads
284 experts x 40 layers), 9 min of perf per arm, four arms, and a bit-identity
pass on top. That is the right price for a *decision*, and far too high for
"is this idea worth coding at all".

t4 already knows how to measure the device timeline correctly: a blocker
kernel is queued first so the host runs ahead, ``Event(enable_timing=True)``
brackets the arm's own execution, and two controls say the instrument works --
a negative one (a graph with no node reads like eager NPU-only) and a positive
one (a deliberately fat node must be measured). This file points that same
instrument at the *mechanisms* instead of at form C:

    serial   today's captured path: D2H + host-func node + H2D on the compute
             stream (``_submit_captured``)
    fork     the side stream (``_submit_captured_forked``)
    fork2    🔴 the side stream AGAIN, captured a second time. No mechanism
             difference at all, so whatever the proxy "measures" between fork
             and fork2 is the proxy's own floor -- the analogue of the same-arm
             floor every whole-network reading is judged against.
    p1a      side stream + ``VLLM_ASCEND_DSV41_KT_ASYNC_WAIT``: the node only
             submits, the device waits on an ExternalEvent
    p1b      side stream + ``VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL``: no host-func
             node at all, an 8-byte counter D2H starts a polling thread

Every switch is production's (``vllm_ascend.kt_offload``): this file sets the
same env vars before the layers are built and flips the same instance
attributes per arm, exactly the way ``t5_side_reuse.py`` does. Nothing here
reimplements a mechanism, so nothing here can measure a mechanism this line
does not actually ship.

What it reports, and what it refuses to
---------------------------------------
Per arm: the device us of one replay of an ``--layers``-deep stack, over
``--reps`` interleaved rounds, as median/p10/p90. Per mechanism: the PAIRED
per-round difference, divided by the stack depth, times ``--model-layers``:

    predicted ms/token = (dev_us[A] - dev_us[B]) / layers * 40 / 1000

and then, next to it, whatever the whole-network A/B already measured for that
mechanism. Whether the proxy may be used at all is decided by
``PROXY_CRITERIA.md``, which was written and committed BEFORE this file first
ran, and is implemented in ``t4_model.calibrate`` / ``proxy_verdict``.

🔴 THIS FILE MEASURES TIME ONLY, AND ON THE DEVICE TIMELINE ONLY.
  * It validates no numerical result. Bit-identity is t5r's job and the whole
    network's; a mechanism that is faster and wrong looks perfect here.
  * A mechanism whose gain is on the HOST's serial chain (a Python frame, a
    GIL wait, a thread hand-off that the device never sees) is invisible to
    it -- the proxy will say "0" and be wrong. See RUNBOOK_proxy.md 4.
  * The number that goes in the benefit ledger is the whole-network one. The
    proxy's number is for ordering and screening candidates.
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

ARM_NAMES = ("serial", "fork", "fork2", "p1a", "p1b")

# mechanism -> (arm A, arm B); the prediction is A - B, so a saving is negative
MECHANISMS = {
    "sidestream": ("fork", "serial"),
    "p1a": ("p1a", "fork"),
    "p1b": ("p1b", "fork"),
    "null": ("fork2", "fork"),
}

PLAN = """T4p plan (a stack of --layers real layers, T=1 decode shape, timing only):
  arms          : serial | fork | fork2 (the null mechanism = the proxy's own floor) | p1a | p1b
                  + g_npu_only (no kt at all) and g_fat (a --fat-us node per layer) as the two controls
  measured on   : the DEVICE timeline (blocker ahead + Event(enable_timing=True)), never wall time
  reported      : per arm median/p10/p90 over >= 20 INTERLEAVED rounds; per mechanism the PAIRED
                  per-round difference / layers * --model-layers = predicted ms/token
  calibrated by : the whole-network A/B numbers passed in --measured-*, judged by PROXY_CRITERIA.md
                  (declared before the first flight; t4_model.PROXY_REL_TOL / PROXY_MIN_CALIBRATED)
  refusals      : any control red, fewer than 20 samples, a non-finite sample, a host that did not run
                  ahead, or a mechanism whose paired p10/p90 straddle zero -> no prediction is printed
  NOT checked   : any numerical result, and anything that happens off the device timeline.
🔴 This file validates no numerical result: a mechanism that is faster and WRONG reads as a win
   here. Correctness is t5r's job and the whole network's, and the benefit ledger takes the
   whole-network number, never this one."""


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0, help="first layer of the window")
    ap.add_argument("--layers", type=int, default=8, help="stack depth actually replayed")
    ap.add_argument("--model-layers", type=int, default=40, help="layers in the model, for the ms/token figure")
    ap.add_argument("--tokens", type=int, default=1, help="bs; decode is 1")
    ap.add_argument("--n", type=int, default=100, help="resident experts per layer (production: 100)")
    ap.add_argument("--threads", type=int, default=32, help="cpuinfer_threads (production: 96)")
    ap.add_argument("--numa", default="1", help="numa nodes (production: 0,1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=24, help=">= 20; one round replays every arm once")
    ap.add_argument("--arms", default="serial,fork,fork2,p1a",
                    help="comma list of " + ",".join(ARM_NAMES) + "; p1a and p1b are alternatives")
    ap.add_argument("--fat-us", type=float, default=500.0, help="positive control: a node that burns this long")
    ap.add_argument("--tol", type=float, default=0.35, help="negative-control tolerance")
    ap.add_argument("--measured-sidestream", type=float, default=None,
                    help="whole-network A-B for side stream vs serial, ms/token (PLAN 4.25: -3.00)")
    ap.add_argument("--measured-p1a", type=float, default=None,
                    help="whole-network A-B for P1a vs side stream, ms/token (PLAN 4.28: -0.53)")
    ap.add_argument("--measured-p1b", type=float, default=None,
                    help="whole-network A-B for P1b vs side stream, ms/token (leave unset if not measured)")
    ap.add_argument("--prev", default=None,
                    help="a previous flight's t4p_result.json: report the BETWEEN-FLIGHT spread of the "
                         "predictions as well as this flight's within-flight one")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without touching the device")
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()

    if args.reps < 20:
        ap.error("--reps must be >= 20: a single point cannot show whether a difference means anything")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ARM_NAMES:
            ap.error(f"--arms: unknown arm {a!r} (known: {', '.join(ARM_NAMES)})")
    if "p1a" in arms and "p1b" in arms:
        # They share ``self._ext_events[bs]`` (layer.py:_enable_device_signal reuses whatever
        # _enable_async_wait made), so both the completion thread and the polling thread would
        # record on one event. t5_side_reuse.py refuses the same combination for the same reason.
        ap.error("--arms: p1a and p1b are alternatives, not a pair; run them in separate invocations")
    if "fork2" in arms and "fork" not in arms:
        ap.error("--arms: fork2 is the null mechanism against fork; it needs fork in the same run")
    for dep in ("p1a", "p1b"):
        if dep in arms and "fork" not in arms:
            ap.error(f"--arms: {dep} is measured against fork; add fork")
    if args.layers < 2:
        ap.error("--layers must be >= 2: a one-layer stack makes the per-layer figure a single sample")

    print(PLAN, flush=True)
    print(f"t4p: arms {arms}, layers {args.layers} (model {args.model_layers}), tokens {args.tokens}, "
          f"reps {args.reps}, n {args.n}, threads {args.threads}, numa {args.numa}", flush=True)
    if args.dry_run or os.environ.get("ONCARD_DRY") == "1":
        print("t4p: --dry-run, nothing was measured and no device was touched", flush=True)
        return 0

    # Before ANY layer is built: the ExternalEvents, the completion thread, the
    # signal words and the polling thread are all made in after_weights_loaded,
    # and none of them may be made inside a capture. Same order as t5r.
    if "p1a" in arms:
        os.environ["VLLM_ASCEND_DSV41_KT_ASYNC_WAIT"] = "1"
    if "p1b" in arms:
        os.environ["VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL"] = "1"

    import npu_moe as N
    import numpy as np
    import torch

    import vllm_ascend.kt_offload.device_signal as DS
    import vllm_ascend.kt_offload.layer as KL

    v = C.Verdict("T4p mechanism proxy (timing only)")
    v.note("this file validates no numerical result; 'faster and wrong' reads as a win here")
    v.note("device timeline only: a mechanism that shortens the HOST's serial chain is invisible to it")
    v.note("predictions are for ordering and screening; the benefit ledger takes whole-network numbers only")

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    numa = [int(s) for s in args.numa.split(",")]
    T = args.tokens
    layer_ids = list(range(args.layer, args.layer + args.layers))
    if layer_ids[-1] >= 40:
        raise SystemExit(f"t4p: layers {layer_ids[0]}-{layer_ids[-1]} run past the model's 40")

    # ---------------------------------------------------------------- stack
    # Built ONCE and shared by every arm, exactly as t5r does it: overlap=True
    # at build time because after_weights_loaded is where the side stream is
    # created and subscribed, and that must happen outside any capture. The
    # serial arm flips the instance attribute afterwards.
    cfg = N.kt_cfg("prefix", args.n, None, args.threads, numa, graph=True, overlap=True)
    stack = []
    t_load0 = time.monotonic()
    for lid in layer_ids:
        off, _ = N.make_offload(cfg, lid, capture_sizes=[T])
        if not off.graph_enabled:
            raise RuntimeError("KTLayerOffload did not come up in graph mode")
        off.overlap = True
        pl = off.placement
        raw = N.load_raw_layer(ck, lid, list(pl.resident_ids))
        lay = N.build_layer(raw, pl.num_slots, dev, offload=off)
        if not off._ids_i64:
            off._ids_i64 = {bs: torch.empty((bs, off.top_k), dtype=torch.int64, device=dev)
                            for bs in off.capture_sizes}
        stack.append((off, lay, N.comm_method(pl.num_slots)))
        del raw
    print(f"stack: {len(stack)} layers loaded in {time.monotonic() - t_load0:.0f}s; {N.hbm_line()}", flush=True)
    pl0 = stack[0][0].placement

    # ------------------------------------------------------------- the data
    rng = np.random.default_rng(args.seed + 9000 + layer_ids[0] + T)
    sx = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16).to(dev)
    sids, sw = C.make_routing(rng, T, pl0.cpu_ids, list(pl0.resident_ids))
    sids, sw = sids.to(dev), sw.to(dev)
    print(f"input: {C.coverage(sids.cpu(), pl0.resident_mask()).line()}", flush=True)

    # The NPU half's routing per layer, computed once and OUTSIDE any capture:
    # the control arms must contain the same NPU work the kt arms do, and a
    # route() inside the npu-only graph would put kt back in it.
    npu_routing = []
    for off, _lay, _comm in stack:
        r = off.route(sx, sw, sids, in_profile_run=False)
        npu_routing.append((r.topk_ids.clone(), r.topk_weights.clone()))
        off.merge(torch.zeros_like(sx))  # drain the submit that route just made
    torch.npu.synchronize()

    def npu_chain(h):
        for (off, lay, comm), (ids_n, w_n) in zip(stack, npu_routing):
            h = h + N.npu_moe(lay, comm, h, w_n, ids_n)
        return h

    def kt_chain_eager():
        """kt's per-layer CPU latency, eager, submit..sync -- t4's ``cpu_us``."""
        for off, _lay, _comm in stack:
            off.route(sx, sw, sids, in_profile_run=False)
            off.merge(torch.zeros_like(sx))

    # ---------------------------------------------- the two control arms
    payload = ("t4p", 0)
    calls = {"fat": 0}

    def fat(_payload):
        calls["fat"] += 1
        end = time.perf_counter() + args.fat_us / 1e6
        while time.perf_counter() < end:
            pass

    graphs: dict = {}
    build_errors: dict = {}

    def capture(name: str, body) -> None:
        g = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(g):
                # The capture runs on torch_npu's own capture stream, NOT on
                # whatever was current outside the ``with``. Reading the stream
                # outside is what made the 2026-09-22 14:15 flight measure a
                # graph with no node in it (PLAN 4.19 (1)); production takes it
                # inside (layer.py:347) and so does this.
                stream = torch.npu.current_stream(dev)
                KL._subscribe_stream(stream)
                out = body(stream)
            torch.npu.synchronize()
            graphs[name] = (g, out)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()
            mine = [ln.strip() for ln in tb if "kt_offload" in ln or "t4_proxy" in ln]
            build_errors[name] = {"error": f"{type(exc).__name__}: {str(exc)[:600]}",
                                  "where": mine[-1] if mine else (tb[-2] if len(tb) > 1 else "")}
            print(f"  arm {name}: capture raised -- {build_errors[name]['error']}\n"
                  f"    at {build_errors[name]['where']}", flush=True)

    KL._subscribe_stream(torch.npu.current_stream(dev))
    capture("g_npu_only", lambda _s: npu_chain(sx))

    def _fat_body(stream):
        h = sx
        for (off, lay, comm), (ids_n, w_n) in zip(stack, npu_routing):
            KL._launch_host_func(stream, fat, payload)
            h = h + N.npu_moe(lay, comm, h, w_n, ids_n)
        return h

    capture("g_fat", _fat_body)

    # ------------------------------------------------------ the kt arms
    def kt_body(_stream):
        h = sx
        for off, lay, comm in stack:
            r = off.route(h, sw, sids, in_profile_run=False)
            pre = N.npu_moe(lay, comm, h, r.topk_weights, r.topk_ids)
            h = h + off.merge(pre)
        return h

    def set_arm(name: str) -> None:
        """Production's own switches, set the way t5r sets them."""
        forked = name != "serial"
        for off, _lay, _comm in stack:
            off.overlap = forked
            off.async_wait = (name == "p1a")
            off.device_signal = (name == "p1b")
            off._ops_prefix = []
            off._pending = None
            off._pending_captured = False
            off._pending_join = None
            off._side_inflight = False
            off._submit_ops = None
        getattr(KL, "_SIDE_BUFFER_OWNER", getattr(KL, "_SIDE_INFLIGHT", {})).clear()

    arm_flags = {}
    calls_before_capture = dict(KL.HOST_FUNC_CALLS)
    for name in arms:
        set_arm("fork" if name == "fork2" else name)
        arm_flags[name] = {"overlap": stack[0][0].overlap, "async_wait": stack[0][0].async_wait,
                           "device_signal": stack[0][0].device_signal}
        capture(name, kt_body)
    missing = [a for a in (*arms, "g_npu_only", "g_fat") if a not in graphs]
    v.check(not missing, f"every arm captured (missing: {missing or 'none'}; errors: {build_errors or 'none'})")
    if missing:
        return v.finish(os.path.join(args.out, "t4p_result.json"),
                        {"validates_correctness": False, "graph_build_errors": build_errors, "args": vars(args)})

    # Did anything run DURING the capture? A node that never entered the graph
    # and a node that costs nothing read exactly the same afterwards.
    grew_in_capture = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - calls_before_capture.get(lid, 0) for lid in layer_ids}
    v.check(calls["fat"] == 0, f"no fat node ran during capture (ran {calls['fat']}x)")
    v.check(all(n == 0 for n in grew_in_capture.values()),
            f"no kt host-func node ran during capture (grew {grew_in_capture})")

    # ---- one replay each: did the mechanism actually run? ----------------
    # An arm that silently did nothing is the fastest arm there is.
    per_replay = {}
    for name in (*arms, "g_fat"):
        hf0 = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) for lid in layer_ids}
        st0 = {lid: DS.STEPS.get(lid, 0) for lid in layer_ids}
        fat0 = calls["fat"]
        if name in arms:
            set_arm("fork" if name == "fork2" else name)
        graphs[name][0].replay()
        torch.npu.synchronize()
        hf = sum(KL.HOST_FUNC_CALLS.get(lid, 0) - hf0[lid] for lid in layer_ids)
        st = sum(DS.STEPS.get(lid, 0) - st0[lid] for lid in layer_ids)
        per_replay[name] = {"host_func": hf, "device_signal_steps": st, "fat": calls["fat"] - fat0}
        print(f"  arm {name:11s} one replay: host-func {hf}, device-signal steps {st}, fat {calls['fat'] - fat0}",
              flush=True)
    v.check(per_replay["g_fat"]["fat"] == args.layers,
            f"the fat control's node ran once per layer per replay ({per_replay['g_fat']['fat']}/{args.layers})")
    for name in arms:
        want_hf = 0 if name == "p1b" else args.layers
        want_sig = args.layers if name == "p1b" else 0
        got = per_replay[name]
        v.check(got["host_func"] == want_hf and got["device_signal_steps"] == want_sig,
                f"arm {name}: one replay drives its mechanism {want_hf} host-func / {want_sig} signal steps "
                f"(got {got['host_func']} / {got['device_signal_steps']}); a mechanism that did nothing "
                "would be the fastest arm here")

    # ---- device time: the only comparison that means anything ------------
    # Wall time around a replay and wall time around an eager arm are not the
    # same instrument (eager at T=1 is launch-bound), so both sides go on the
    # device timeline: a blocker is queued first, the arm is queued behind it
    # while the device is still busy, and two events bracket the arm itself.
    blk = torch.randn((2048, 2048), device=dev, dtype=torch.bfloat16)
    torch.npu.synchronize()
    # 🔴 The blocker's cost has to be its DEVICE time, not the wall time of a
    # launch-and-synchronize loop. Wall time includes the host's launch and the
    # synchronize, so it overstates what the device actually spends: on the
    # 2026-09-23 07:11 flight the wall figure said 589 us per mm and 14 of them
    # would be 8.2 ms of cover, while the device got through them fast enough
    # to reach e0 before the eager arm had finished launching (host-ahead 42%).
    for _ in range(3):  # warm the kernel before it is timed
        torch.mm(blk, blk)
    torch.npu.synchronize()
    eb0, eb1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    eb0.record()
    for _ in range(20):
        torch.mm(blk, blk)
    eb1.record()
    torch.npu.synchronize()
    per_blk_us = eb0.elapsed_time(eb1) * 1000.0 / 20.0

    def replay_of(name):
        g = graphs[name][0]
        arm = "fork" if name == "fork2" else name

        def run():
            # The per-arm instance flags decide what the *callback* and the
            # completion/polling threads do at replay time, so they are set on
            # every sample, not once per capture.
            if name in arms:
                set_arm(arm)
            g.replay()

        return run

    dev_arms = {name: replay_of(name) for name in (*arms, "g_npu_only", "g_fat")}
    dev_arms["eager_npu_only"] = lambda: npu_chain(sx)

    # 🔴 How much blocker: enough that the HOST is still enqueueing when the
    # device reaches e0, for EVERY arm -- including the eager one. t4's fixed
    # ~5 ms was sized for a one-layer eager arm; an ``--layers``-deep eager
    # chain issues layers x ~a hundred small ops and takes far longer to launch
    # than that, so on the first t4p flight (2026-09-23 07:08) the eager arm's
    # host-ahead came out at 12% and the whole run was void. The size is
    # therefore measured, not guessed: four times the slowest arm's own host
    # launch, and never under 10 ms. Anything smaller makes the control arm
    # read launch gaps and the negative control compare two instruments.
    launch_us = {}
    for name, fn in dev_arms.items():
        worst = 0.0
        for _ in range(3):
            torch.npu.synchronize()
            t_l = time.perf_counter()
            fn()
            worst = max(worst, (time.perf_counter() - t_l) * 1e6)
            torch.npu.synchronize()
        launch_us[name] = worst
    need_us = max(10000.0, 4.0 * max(launch_us.values()), 2.0 * args.layers * args.fat_us)
    blk_reps = max(1, int(round(need_us / max(per_blk_us, 1.0))))
    print("  host launch (worst of 3): " + ", ".join(f"{k} {v / 1000:.1f} ms" for k, v in launch_us.items()),
          flush=True)
    print(f"  blocker: {per_blk_us:.1f} us per 2048^3 mm, {blk_reps} of them "
          f"(~{blk_reps * per_blk_us / 1000:.1f} ms) queued ahead of every device-timed sample "
          f"(needed >= {need_us / 1000:.1f} ms)", flush=True)

    def dev_timed(fns: dict, reps: int) -> tuple[dict, dict]:
        out = {k: [] for k in fns}
        ahead = {k: [] for k in fns}
        names = list(fns)
        for r in range(reps + 2):  # the first two rounds are warm-up, discarded
            k = r % len(names)     # rotate, so a drift that moves every arm drops out of the pairs
            for name in names[k:] + names[:k]:
                torch.npu.synchronize()
                e0, e1 = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                for _ in range(blk_reps):
                    torch.mm(blk, blk)
                e0.record()
                fns[name]()
                e1.record()
                host_ran_ahead = not e0.query()  # the device has not reached e0 yet
                torch.npu.synchronize()
                if r >= 2:
                    out[name].append(e0.elapsed_time(e1) * 1000.0)
                    ahead[name].append(host_ran_ahead)
        return out, ahead

    dv, ahead = dev_timed(dev_arms, args.reps)
    dev_stats = {k: M.stats(s) for k, s in dv.items()}
    print("", flush=True)
    for k, s in dev_stats.items():
        frac = sum(ahead[k]) / len(ahead[k])
        print(f"  DEV {k:15s} median {s['median']:10.1f} us  p10 {s['p10']:10.1f}  p90 {s['p90']:10.1f} "
              f" n={s['n']}  per-layer {s['median'] / args.layers:8.1f} us  host-ahead {frac:.0%}", flush=True)
    worst = min(sum(a) / len(a) for a in ahead.values())
    v.check(worst >= 0.9,
            f"the host ran ahead of the device on >= 90% of device-timed samples (worst arm {worst:.0%}); "
            "below that the events bracket launch gaps, not execution")
    v.check(all(s["n"] >= 20 for s in dev_stats.values()), f"every arm has >= 20 interleaved samples ({args.reps})")

    # ---- the two controls -------------------------------------------------
    nc_ok, nc_line = M.negative_control(dv["g_npu_only"], dv["eager_npu_only"], args.tol)
    v.check(nc_ok, f"negative control (device time): {nc_line}")
    d_fat = M.per_layer_delta(dv["g_fat"], dv["g_npu_only"], args.layers)
    ic_ok, ic_line = M.instrument_control(d_fat["median"], args.fat_us)
    v.check(ic_ok, f"positive control, per layer: {ic_line}")
    controls_ok = nc_ok and ic_ok and worst >= 0.9

    # ---- what kt costs, and what window there is to hide it in ------------
    # Not a criterion; it is what says whether this run is in the regime the
    # whole network is in (kt's CPU latency longer than the NPU window, so the
    # side stream's saving is bounded by the window and not by kt's speed).
    t_kt0 = time.perf_counter()
    for _ in range(5):
        kt_chain_eager()
    torch.npu.synchronize()
    cpu_us = (time.perf_counter() - t_kt0) * 1e6 / (5 * args.layers)
    window_us = dev_stats["g_npu_only"]["median"] / args.layers
    print(f"\n  kt CPU latency {cpu_us:.1f} us/layer (eager submit..sync, wall) vs NPU window "
          f"{window_us:.1f} us/layer (device) -> bound by "
          f"{'the window' if window_us < cpu_us else 'kt'}", flush=True)
    v.note(f"cpu_us {cpu_us:.1f} us/layer, window_us {window_us:.1f} us/layer: the side stream's saving is "
           f"bounded by {'the window (so it does not depend on cpuinfer_threads)' if window_us < cpu_us else 'kt'}")

    # ---- the predictions --------------------------------------------------
    measured = {"sidestream": args.measured_sidestream, "p1a": args.measured_p1a, "p1b": args.measured_p1b,
                "null": None}
    preds: dict = {}
    for mech, (a, b) in MECHANISMS.items():
        if a not in graphs or b not in graphs:
            continue
        per_layer = M.per_layer_delta(dv[a], dv[b], args.layers)
        pred = M.predict_ms_per_token(per_layer, args.model_layers)
        ok, why = M.resolved(per_layer)
        preds[mech] = {"arms": [a, b], "per_layer_us": per_layer, "pred_ms_per_token": pred,
                       "resolved": ok, "resolved_why": why}

    if not controls_ok:
        print("\n  🔴 a control is RED: no prediction is printed. (PROXY_CRITERIA.md G0)", flush=True)
    else:
        print(f"\n  预测（代理）：Δ per layer -> ms/token over {args.model_layers} layers", flush=True)
        for mech, p in preds.items():
            pl, pm = p["per_layer_us"], p["pred_ms_per_token"]
            print(f"    {mech:11s} {p['arms'][0]:7s} - {p['arms'][1]:7s}  "
                  f"{pl['median']:+8.1f} us/layer (p10 {pl['p10']:+8.1f}, p90 {pl['p90']:+8.1f})  "
                  f"=> {pm['median']:+6.2f} ms/token (p10 {pm['p10']:+6.2f}, p90 {pm['p90']:+6.2f})  "
                  f"{'resolved' if p['resolved'] else '🔴 NOT RESOLVED'}", flush=True)

    # ---- the proxy's own floor (the null mechanism) -----------------------
    floor_ms = abs(preds["null"]["pred_ms_per_token"]["median"]) if "null" in preds else None
    if floor_ms is not None:
        print(f"\n  代理地板（fork vs fork2，无任何机制差）: {floor_ms:.2f} ms/token", flush=True)
        judged = [abs(preds[m]["pred_ms_per_token"]["median"]) for m in preds
                  if m != "null" and measured.get(m) is not None]
        if judged:
            smallest = min(judged)
            v.check(floor_ms < smallest / 3.0,
                    f"proxy floor {floor_ms:.2f} ms/token < 1/3 of the smallest calibrated prediction "
                    f"{smallest:.2f} (PROXY_CRITERIA G2.1)")

    # ---- calibration ------------------------------------------------------
    cals = {}
    for mech, p in preds.items():
        if mech == "null":
            continue
        if not p["resolved"]:
            cals[mech] = {"state": "inconclusive",
                          "line": f"the proxy did not resolve {mech} in this run: {p['resolved_why']}"}
            continue
        cals[mech] = M.calibrate(p["pred_ms_per_token"]["median"], measured.get(mech))
    print("\n  标定（代理预测 vs 整网实测）：", flush=True)
    for mech, c in cals.items():
        print(f"    {mech:11s} {c['line']}", flush=True)
    pv = M.proxy_verdict(cals)
    print(f"\n  PROXY USABLE: {pv['state']} -- {pv['why']}", flush=True)
    if not controls_ok:
        pv = {"state": "unusable", "n_measured": pv["n_measured"],
              "why": "a control was red, so no prediction may be read at all (PROXY_CRITERIA G0)"}
        print(f"  overridden by the controls -> {pv['state']}", flush=True)

    # mutation arm on the decision rule, driven by this run's own data: the
    # null mechanism (two captures of one configuration) must NOT calibrate as
    # the side stream. If +-50% is wide enough to accept "nothing was changed"
    # as "-3.00 ms/token", the criterion has no discriminating power.
    if "null" in preds:
        null_cal = M.calibrate(preds["null"]["pred_ms_per_token"]["median"], -3.00)
        v.must_be_red(null_cal["state"] != "agree",
                      f"the null mechanism ({preds['null']['pred_ms_per_token']['median']:+.2f} ms/token) "
                      f"calibrated against -3.00 -> {null_cal['state']}")
    # and the positive control's per-layer cost, fed in as if it were a
    # mechanism, must come out with the wrong sign against a saving
    fat_as_mech = M.calibrate(d_fat["median"] * args.model_layers / 1000.0, -3.00)
    v.must_be_red(fat_as_mech["state"] == "sign_disagrees",
                  f"the fat node's {d_fat['median']:.1f} us/layer read as a mechanism -> {fat_as_mech['state']}")

    # ---- between-flight spread -------------------------------------------
    between = None
    if args.prev:
        try:
            with open(args.prev) as f:
                prev = json.load(f)
            between = {}
            for mech, p in preds.items():
                q = (prev.get("predictions") or {}).get(mech)
                if not q:
                    continue
                a, b = p["pred_ms_per_token"]["median"], q["pred_ms_per_token"]["median"]
                between[mech] = {"this": a, "prev": b, "spread_ms": abs(a - b)}
            print("\n  发间离散度（与 --prev 那一发比）：", flush=True)
            for mech, d in between.items():
                print(f"    {mech:11s} this {d['this']:+.2f}  prev {d['prev']:+.2f}  "
                      f"spread {d['spread_ms']:.2f} ms/token", flush=True)
        except Exception as exc:  # noqa: BLE001
            v.note(f"--prev could not be read ({type(exc).__name__}: {exc}); no between-flight spread")

    extra = {
        "validates_correctness": False,
        "arms": arms,
        "arm_flags": arm_flags,
        "device_us": dev_stats,
        "device_us_per_layer": {k: s["median"] / args.layers for k, s in dev_stats.items()},
        "device_host_ahead_frac": {k: sum(a) / len(a) for k, a in ahead.items()},
        "blocker": {"per_mm_us": per_blk_us, "reps": blk_reps},
        "per_replay_counters": per_replay,
        "host_func_during_capture": grew_in_capture,
        "controls": {"negative": {"ok": nc_ok, "line": nc_line},
                     "positive": {"ok": ic_ok, "line": ic_line, "per_layer_us": d_fat},
                     "host_ahead_worst": worst, "all_ok": controls_ok},
        "kt_cpu_us_per_layer": cpu_us,
        "npu_window_us_per_layer": window_us,
        "predictions": preds,
        "proxy_floor_ms": floor_ms,
        "measured_whole_network": measured,
        "calibration": cals,
        "proxy_verdict": pv,
        "between_flight": between,
        "criteria": {"file": "tools/kt_offload/oncard/PROXY_CRITERIA.md",
                     "rel_tol": M.PROXY_REL_TOL, "min_calibrated": M.PROXY_MIN_CALIBRATED},
        "graph_build_errors": build_errors,
        "args": vars(args),
    }
    return v.finish(os.path.join(args.out, "t4p_result.json"), extra)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        print(json.dumps({"test": "T4p", "pass": False, "error": traceback.format_exc()[-2000:]}))
        raise
