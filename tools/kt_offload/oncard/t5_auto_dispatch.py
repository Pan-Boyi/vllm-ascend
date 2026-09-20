# SPDX-License-Identifier: Apache-2.0
"""T5: does ``auto_dispatch_capture=True`` bring the side-stream route back?

The named difference: sglang captures with
``torch.npu.graph(..., auto_dispatch_capture=True)``
(``npu_graph_runner.py:154``); vllm-ascend's wrapper takes the default False
(``vllm_ascend/compilation/acl_graph.py:187``). With it on, torch_npu enters
``_GraphDispatchMode`` for the capture, which wraps dispatched ops in
``graph_task_group_begin/end(stream)`` -- i.e. exactly the machinery that
decides how work issued on another stream enters the graph. Our three
capture_end 107011/107025 flights all ran with it off, so the platform
conclusion ("a side stream cannot be captured here") has never been tested
against this parameter.

Two arms, one changed parameter, on the **real layer** capture scenario that
t3 uses -- never a miniature graph: a scalar-plus-trivial-host-func capture
passed on 2026-09-20 while every real-layer capture failed, and it is that
false green this file exists to avoid.

  arm "off"  auto_dispatch_capture=False  -> expected to RAISE at capture_end.
             This is the positive control: if it does not raise, the machine
             is not in the state the earlier flights measured, and arm "on"
             proves nothing.
  arm "on"   auto_dispatch_capture=True   -> if it raises, the side-stream
             route stays dead and form C is decided by t4's ceiling. If it
             captures, that is NOT the answer yet: capture success and replay
             correctness are judged separately, because on 2026-09-20 a
             capture "succeeded" while the side-stream work never entered the
             graph. The replay criteria are t3's: >= 3 different inputs, each
             replay bit-identical to the serial arm, replays pairwise
             different, and every layer's host-func node running exactly once
             per replay (0 during capture).

The overlap switch itself stays refused in the config; this test flips the
instance attribute on the layer it built, in-process, so no production path
changes and nothing here is a way to turn overlap on for a service.

A failed capture can leave the stream in a state the next capture inherits,
so the arms run in separate processes by default (``--arm on`` / ``--arm off``,
run_oncard.sh t5 twice). ``--arm both`` runs them in one process and says in
its own output that the second arm carries that caveat.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oncard_common as C  # noqa: E402

PLAN = """T5 plan (real-layer capture, one parameter changed):
  arm off : torch.npu.graph(g)                          -> expected RAISE (positive control)
  arm on  : torch.npu.graph(g, auto_dispatch_capture=True)
            if it captures: replay >= 3 inputs, each == the serial arm bit for bit,
            replays pairwise different,每 layer host-func +1 per replay, 0 during capture
  refusals: capture success alone is NOT a pass; a positive control that does not go red
            voids the comparison; fewer than 3 inputs is refused
  scope   : this says nothing about performance. It answers one question: can the
            side-stream shape be captured and replayed correctly on this platform."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--layers", type=int, default=3, help=">=3 so two layers share kt's layer_idx % 2 slot")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tokens", type=int, default=1, help="one capture size, decode shape")
    ap.add_argument("--inputs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--numa", default="1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=("on", "off", "both"), default="both")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()
    if args.inputs < 3:
        ap.error("--inputs must be >= 3 (a stale replay output repeats the previous one)")
    if args.layers < 3:
        ap.error("--layers must be >= 3 (kt's slot reuse needs layers L and L+2)")

    print(PLAN, flush=True)
    if args.dry_run or os.environ.get("ONCARD_DRY") == "1":
        print("t5: --dry-run, nothing was captured and no device was touched", flush=True)
        return 0

    import npu_moe as N
    import numpy as np
    import torch

    import vllm_ascend.kt_offload.layer as KL

    v = C.Verdict("T5 auto_dispatch_capture")
    v.note("performance is not measured here; this is a can-it-capture-and-replay question")
    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    T = args.tokens
    numa = [int(s) for s in args.numa.split(",")]
    layer_ids = list(range(args.layer, args.layer + args.layers))
    rng = np.random.default_rng(args.seed + 5000)

    def build(forked: bool):
        """A stack of real offload layers. ``forked`` flips the *instance*
        attribute the captured path reads, so the refused config switch stays
        refused and no production code changes."""
        cfg = N.kt_cfg("prefix", args.n, None, args.threads, numa, graph=True)
        stack = []
        for lid in layer_ids:
            off, _ = N.make_offload(cfg, lid, capture_sizes=[T])
            if not off.graph_enabled:
                raise RuntimeError("KTLayerOffload did not come up in graph mode")
            if forked:
                off.overlap = True  # test-only; config.overlap_side_stream stays refused
            pl = off.placement
            raw = N.load_raw_layer(ck, lid, list(pl.resident_ids))
            lay = N.build_layer(raw, pl.num_slots, dev, offload=off)
            stack.append((off, lay, N.comm_method(pl.num_slots)))
        return stack

    def run_stack(stack, x, ids, w):
        out = None
        for off, lay, comm in stack:
            _, merged, _ = N.offload_forward(off, lay, comm, x, w, ids)
            out = merged if out is None else out + merged
        return out

    # reference: the serial shape, eager (what a correct replay must equal)
    serial = build(forked=False)
    pl = serial[0][0].placement
    inputs = []
    for _ in range(args.inputs):
        x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16)
        ids, w = C.make_routing(rng, T, pl.cpu_ids, list(pl.resident_ids))
        inputs.append((x.to(dev), ids.to(dev), w.to(dev)))
        print(f"input {len(inputs)}: {C.coverage(ids, pl.resident_mask()).line()}", flush=True)
    ref = [run_stack(serial, *inp).float().cpu() for inp in inputs]
    torch.npu.synchronize()
    for off, _, _ in serial:
        del off
    del serial
    torch.npu.empty_cache()

    results: dict[str, dict] = {}

    def attempt(auto: bool) -> dict:
        """One capture of the forked shape, with auto_dispatch_capture=auto."""
        name = "on" if auto else "off"
        stack = build(forked=True)
        sx, sids, sw = (t.clone() for t in inputs[0])
        calls0 = dict(KL.HOST_FUNC_CALLS)
        out: dict = {"auto_dispatch_capture": auto, "captured": None, "error": None, "where": None}
        g = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(g, auto_dispatch_capture=auto):
                sout = run_stack(stack, sx, sids, sw)
            torch.npu.synchronize()
            out["captured"] = True
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()
            mine = [ln.strip() for ln in tb if "kt_offload" in ln or "t5_auto_dispatch" in ln]
            out.update(captured=False, error=f"{type(exc).__name__}: {str(exc)[:600]}",
                       where=(mine[-1] if mine else tb[-2] if len(tb) > 1 else ""))
            print(f"  arm {name}: capture raised -- {out['error']}", flush=True)
            print(f"    at {out['where']}", flush=True)
            return out

        ran = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - calls0.get(lid, 0) for lid in layer_ids}
        out["host_func_during_capture"] = ran
        v.check(all(n == 0 for n in ran.values()), f"arm {name}: capture ran no host-func node (ran {ran})")

        # capture success is not the answer: judge the replay separately
        replays, checks = [], []
        for k, (x, ids, w) in enumerate(inputs):
            sx.copy_(x), sids.copy_(ids), sw.copy_(w)
            before = dict(KL.HOST_FUNC_CALLS)
            g.replay()
            torch.npu.synchronize()
            grew = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - before.get(lid, 0) for lid in layer_ids}
            ok_calls = all(n == 1 for n in grew.values())
            v.check(ok_calls, f"arm {name}: replay {k}: every layer's host-func ran once (got {grew})")
            r = sout.float().cpu().clone()
            replays.append(r)
            same, how = C.bit_identical(r, ref[k])
            v.check(same, f"arm {name}: replay {k} == serial {k}: {how}"
                          + ("" if same else f"; rel_l2 {C.rel_l2(r, ref[k]):.3e}"))
            if k:
                v.check(not C.bit_identical(r, ref[k - 1])[0],
                        f"arm {name}: replay {k} != serial {k - 1} (stale-output fingerprint)")
            checks.append({"replay": k, "host_func": grew, "bit_identical": same, "how": how})
        for i in range(len(replays)):
            for j in range(i + 1, len(replays)):
                v.check(not C.bit_identical(replays[i], replays[j])[0], f"arm {name}: replay {i} != replay {j}")
        out["replays"] = checks
        return out

    order = {"both": [True, False], "on": [True], "off": [False]}[args.arm]
    for idx, auto in enumerate(order):
        name = "on" if auto else "off"
        if idx:
            v.note(f"arm {name} runs after a capture in the same process; a failed capture can leave state "
                   f"behind, so re-run it with --arm {name} in a fresh process before treating it as final")
        print(f"\n===== arm {name} (auto_dispatch_capture={auto}) =====", flush=True)
        results[name] = attempt(auto)

    # the positive control: the status-quo arm must reproduce the known failure
    if "off" in results:
        off_failed = results["off"]["captured"] is False
        v.must_be_red(off_failed, "status quo (auto_dispatch_capture=False) capture of the forked shape")
        if not off_failed:
            v.note("🔴 the positive control captured: this machine is NOT in the state the three earlier "
                   "flights measured, so nothing here can be attributed to auto_dispatch_capture")
    else:
        v.note("no positive control in this run (--arm on): run --arm off before concluding")

    if "on" in results:
        if results["on"]["captured"]:
            v.note("arm on CAPTURED: the side-stream route is alive if and only if the replay checks above "
                   "passed; form C drops to fallback in that case")
        else:
            v.note("arm on did NOT capture: the side-stream route stays dead on this platform and form C is "
                   "decided by t4's ceiling")

    return v.finish(os.path.join(args.out, "t5_result.json"),
                    {"arms": results, "layers": layer_ids, "tokens": T, "inputs": args.inputs,
                     "measures_performance": False})


if __name__ == "__main__":
    sys.exit(main())
