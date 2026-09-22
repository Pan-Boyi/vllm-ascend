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

  arm "off"  auto_dispatch_capture=False. Until 2026-09-22 this was "expected
             to RAISE" (the positive control). Re-read on 09-22: the earlier
             107025s were tripped by an empty capture buffer; the real trigger
             is an unjoined tail record on the side stream. Now "off" (the
             production record+join) and "off_evtw" must CAPTURE and pass the
             replay checks; only "off_evt" (extra unjoined tail record) is the
             must-be-red positive control.
  arm "on"   auto_dispatch_capture=True   -> if it raises, the side-stream
             route stays dead and form C is decided by t4's ceiling. If it
             captures, that is NOT the answer yet: capture success and replay
             correctness are judged separately, because on 2026-09-20 a
             capture "succeeded" while the side-stream work never entered the
             graph. The replay criteria are t3's: >= 3 different inputs, each
             replay bit-identical to the serial arm, replays pairwise
             different, and every layer's host-func node running exactly once
             per replay (0 during capture).

2026-09-22, after the capture-buffer fix: this file also carries the
single-layer bit-identity arm (RUNBOOK_sidestream.md item 1). ``--layer`` and
``--tokens`` take lists, so one run sweeps several stack positions (both kt
pinned slots, ``layer_idx % 2``) and several token counts, and ``--mutant
drop-join`` removes the compute stream's wait on the side stream's join. The
mutant exists because "bit-identical" on its own does not prove the work was
overlapped: if nothing ever ran on the side stream the outputs would match
too. A mutant that keeps matching means the arm has no discriminating power.

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
  arm off : torch.npu.graph(g)                          -> clean arm: must capture + replay checks
  arm off_evt : off + unjoined tail record on the side stream -> must RAISE 107025 (positive control)
  arm on  : torch.npu.graph(g, auto_dispatch_capture=True)
            if it captures: replay >= 3 inputs, each == the serial arm bit for bit,
            replays pairwise different,每 layer host-func +1 per replay, 0 during capture
  arm *_evt: the same, plus an event record as the LAST task on the side stream before
            capture_end -- what error 107025's own text asks for ("Call aclrtRecordEvent
            on the sub stream to deliver an event record task")
  refusals: capture success alone is NOT a pass; a positive control that does not go red
            voids the comparison; fewer than 3 inputs is refused
  sweep   : --layer a,b,c x --tokens 1,4,32 -> one scenario each (own serial reference)
  mutant  : --mutant drop-join -> the compute stream does NOT wait on the side stream's
            join. Must NOT come out captured-and-bit-identical; either it fails the
            capture (107025: the sub stream's last record is unjoined) or the output
            differs. If it passes like the clean arm, the bit-identity criterion has no
            discriminating power in this run and the run is void.
  scope   : this says nothing about performance. It answers one question: can the
            side-stream shape be captured and replayed correctly on this platform."""


NUM_LAYERS = 40  # DSV4.1 main layers (npu_moe.fake_vllm_config)


def parse_int_list(text: str, what: str) -> list[int]:
    """"0,20,39" -> [0, 20, 39]. Order and duplicates are kept out."""
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if not part.lstrip("-").isdigit():
            raise ValueError(f"{what}: {part!r} is not an integer")
        value = int(part)
        if value < 0:
            raise ValueError(f"{what}: {value} is negative")
        if value not in out:
            out.append(value)
    if not out:
        raise ValueError(f"{what}: empty list")
    return out


def plan_scenarios(layer_text: str, tokens_text: str, layers: int, num_layers: int = NUM_LAYERS) -> list[dict]:
    """The (stack position x token count) matrix, resolved before any device is touched.

    A window that would run past the last layer is shifted down rather than
    dropped: ``--layer 39 --layers 3`` on a 40-layer model means "the last
    three layers", and the shift is reported so nobody reads the result as
    "layer 39 alone". Each scenario gets its own serial reference, so two
    scenarios can never compare against each other's.
    """
    if layers < 3:
        raise ValueError("layers must be >= 3 (kt's slot reuse needs layers L and L+2)")
    if layers > num_layers:
        raise ValueError(f"layers {layers} exceeds the model's {num_layers}")
    starts = parse_int_list(layer_text, "--layer")
    tokens = parse_int_list(tokens_text, "--tokens")
    if any(t < 1 for t in tokens):
        raise ValueError("--tokens must all be >= 1")
    scenarios = []
    for start in starts:
        if start >= num_layers:
            raise ValueError(f"--layer {start} is past the model's {num_layers} layers")
        first = min(start, num_layers - layers)
        for t in tokens:
            scenarios.append(
                {
                    "requested_start": start,
                    "layer_ids": list(range(first, first + layers)),
                    "tokens": t,
                    "shifted": first != start,
                    # both kt pinned slots covered iff the window holds L and L+2
                    "slots": sorted({lid % 2 for lid in range(first, first + layers)}),
                }
            )
    return scenarios


def scenario_key(name: str, sc: dict) -> str:
    return f"{name}@L{sc['layer_ids'][0]}_T{sc['tokens']}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", default="20",
                    help="comma list of stack START layers, e.g. 0,20,39; a window that would run past "
                         "the last layer is shifted down (and said so in the output)")
    ap.add_argument("--layers", type=int, default=3, help=">=3 so two layers share kt's layer_idx % 2 slot")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tokens", default="1",
                    help="comma list of token counts, e.g. 1,4,32; one capture size per scenario")
    ap.add_argument("--inputs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--numa", default="1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", choices=("on", "off", "both", "on_evt", "off_evt", "on_evtw", "off_evtw"), default="both",
                    help="on/off = auto_dispatch_capture; the _evt variants additionally make the LAST task "
                         "on the side stream an event record, which is what error 107025 asks for")
    ap.add_argument("--mutant", choices=("none", "drop-join"), default="none",
                    help="drop-join: the compute stream does not wait on the side stream's join. "
                         "Expected to be RED (capture failure or a different output).")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()
    if args.inputs < 3:
        ap.error("--inputs must be >= 3 (a stale replay output repeats the previous one)")
    if args.layers < 3:
        ap.error("--layers must be >= 3 (kt's slot reuse needs layers L and L+2)")

    try:
        scenarios = plan_scenarios(args.layer, args.tokens, args.layers)
    except ValueError as exc:
        ap.error(str(exc))

    print(PLAN, flush=True)
    print(f"t5 scenarios ({len(scenarios)}): "
          + ", ".join(f"L{sc['layer_ids'][0]}-{sc['layer_ids'][-1]} T{sc['tokens']}"
                      + (" (shifted)" if sc["shifted"] else "") for sc in scenarios), flush=True)
    if args.mutant != "none":
        print(f"t5 MUTANT ARM: {args.mutant} -- this arm must NOT come out captured-and-bit-identical",
              flush=True)
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
    numa = [int(s) for s in args.numa.split(",")]
    mutant = args.mutant

    ids_buffers_added: list[int] = []
    results: dict[str, dict] = {}

    def _ensure_ids_buffer(off) -> None:
        """Belt and braces: the layer must own its int64 id buffers.

        The production bug this used to work around (a process-global
        ``_CAPTURE_BUFFERS_READY`` skipping ``self._ids_i64`` for every layer
        after the first) was fixed on 2026-09-22; ``_prepare_capture_buffers``
        now fills them per instance. This test builds its layers by hand and
        does not call that method, so it still prepares the buffers itself --
        and records which layers it had to do it for, so a regression of the
        fix shows up in the result json instead of aborting the capture with
        "no preallocated int64 id buffer" (which on card reads as 107025).
        """
        if not off.graph_enabled or off._ids_i64:
            return
        off._ids_i64 = {bs: torch.empty((bs, off.top_k), dtype=torch.int64, device=dev) for bs in off.capture_sizes}
        ids_buffers_added.append(off.layer_idx)

    def _drop_join(self, device) -> None:
        """MUTANT: record the join on the side stream, never wait for it.

        Not the same thing as arm ``*_evt`` (which records an EXTRA event after
        the layers' work): here the layer's own join event is recorded and the
        compute stream simply does not wait, so the H2D may read the pinned
        output before the host func has written it. Two outcomes are both red:
        ACL refuses the capture (107025 -- the sub stream's last record is
        unjoined), or the replay output differs from the serial arm. The one
        outcome that must NOT happen is "captured and bit-identical": that
        would mean the join was never doing anything, i.e. nothing was running
        on the side stream at all.
        """
        pending, self._pending_join = self._pending_join, None
        self._check_capture_hostfunc()
        (getattr(self, "_release_side_buffers", None) or self._release_side_slot)()
        if pending is None:
            return
        join, side, _compute = pending
        join.record(side)

    def build(layer_ids, T, forked: bool):
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
            _ensure_ids_buffer(off)
            stack.append((off, lay, N.comm_method(pl.num_slots)))
        return stack

    def run_stack(stack, x, ids, w):
        out = None
        for off, lay, comm in stack:
            _, merged, _ = N.offload_forward(off, lay, comm, x, w, ids)
            out = merged if out is None else out + merged
        return out

    def attempt(sc, inputs, ref, auto: bool, record_evt, name: str) -> dict:
        """One capture of the forked shape, with auto_dispatch_capture=auto."""
        layer_ids, T = sc["layer_ids"], sc["tokens"]
        is_mutant = mutant != "none"
        # A mutant arm's checks are judged as a group at the end (it MUST fail);
        # its individual comparisons therefore record, not assert.
        def judge(ok: bool, what: str) -> bool:
            if is_mutant:
                return ok
            return v.check(ok, what)

        stack = build(layer_ids, T, forked=True)
        sx, sids, sw = (t.clone() for t in inputs[0])
        calls0 = dict(KL.HOST_FUNC_CALLS)
        out: dict = {"auto_dispatch_capture": auto, "record_event_on_substream": record_evt,
                     "layer_ids": layer_ids, "tokens": T, "mutant": mutant,
                     "captured": None, "error": None, "where": None, "all_bit_identical": None}
        tail_events: list = []  # kept alive: the graph's nodes reference them
        g = torch.npu.NPUGraph()
        original_join = KL.KTLayerOffload._join_side_stream
        if is_mutant:
            KL.KTLayerOffload._join_side_stream = _drop_join
        try:
            with torch.npu.graph(g, auto_dispatch_capture=auto):
                sout = run_stack(stack, sx, sids, sw)
                if record_evt:
                    # 107025 says verbatim: "In the cross-stream capture
                    # scenario, the last task on the sub stream is not an event
                    # record task. Call aclrtRecordEvent on the sub stream."
                    # So: one event record on the side stream, and NOTHING
                    # enqueued on that stream afterwards (a wait is recorded on
                    # the compute stream, which is a different stream's task).
                    side = KL._side_stream(dev)
                    tail = torch.npu.Event()
                    tail.record(side)
                    tail_events.append(tail)
                    if record_evt == "record_wait":
                        torch.npu.current_stream(dev).wait_event(tail)
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
        finally:
            KL.KTLayerOffload._join_side_stream = original_join

        ran = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - calls0.get(lid, 0) for lid in layer_ids}
        out["host_func_during_capture"] = ran
        judge(all(n == 0 for n in ran.values()), f"arm {name}: capture ran no host-func node (ran {ran})")

        # capture success is not the answer: judge the replay separately
        replays, checks, all_same = [], [], True
        for k, (x, ids, w) in enumerate(inputs):
            sx.copy_(x), sids.copy_(ids), sw.copy_(w)
            before = dict(KL.HOST_FUNC_CALLS)
            g.replay()
            torch.npu.synchronize()
            grew = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - before.get(lid, 0) for lid in layer_ids}
            ok_calls = all(n == 1 for n in grew.values())
            judge(ok_calls, f"arm {name}: replay {k}: every layer's host-func ran once (got {grew})")
            r = sout.float().cpu().clone()
            replays.append(r)
            same, how = C.bit_identical(r, ref[k])
            all_same = all_same and same
            judge(same, f"arm {name}: replay {k} == serial {k}: {how}"
                        + ("" if same else f"; rel_l2 {C.rel_l2(r, ref[k]):.3e}"))
            if k:
                judge(not C.bit_identical(r, ref[k - 1])[0],
                      f"arm {name}: replay {k} != serial {k - 1} (stale-output fingerprint)")
            checks.append({"replay": k, "host_func": grew, "bit_identical": same, "how": how})
        for i in range(len(replays)):
            for j in range(i + 1, len(replays)):
                judge(not C.bit_identical(replays[i], replays[j])[0], f"arm {name}: replay {i} != replay {j}")
        out["replays"] = checks
        out["all_bit_identical"] = all_same
        out["tail_events"] = len(tail_events)
        return out

    # (name, auto_dispatch_capture, record an event last on the side stream)
    order = {
        "both": [("on", True, False), ("off", False, False)],
        "on": [("on", True, False)],
        "off": [("off", False, False)],
        "on_evt": [("on_evt", True, "record")],
        "off_evt": [("off_evt", False, "record")],
        # ..._evtw: the same record, and the compute stream waits on it -- the
        # difference that tells "the sub stream's last task must be a record"
        # from "the sub stream must end on a record the main stream joins".
        "on_evtw": [("on_evtw", True, "record_wait")],
        "off_evtw": [("off_evtw", False, "record_wait")],
    }[args.arm]

    captures_done = 0
    for sc in scenarios:
        layer_ids, T = sc["layer_ids"], sc["tokens"]
        print(f"\n########## scenario layers {layer_ids} T={T} "
              f"(kt slots {sc['slots']}{', window shifted' if sc['shifted'] else ''}) ##########", flush=True)
        if sc["shifted"]:
            v.note(f"--layer {sc['requested_start']} was shifted to {layer_ids[0]} so the {args.layers}-layer "
                   f"window ends at the last layer ({NUM_LAYERS - 1})")
        rng = np.random.default_rng(args.seed + 5000 + 97 * layer_ids[0] + T)
        # reference: the serial shape, eager (what a correct replay must equal)
        serial = build(layer_ids, T, forked=False)
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

        for name, auto, record_evt in order:
            if captures_done:
                v.note(f"arm {name} (layers {layer_ids}, T={T}) runs after a capture in the same process; a "
                       f"failed capture can leave state behind, so re-run it alone in a fresh process before "
                       f"treating it as final")
            print(f"\n===== arm {name} (auto_dispatch_capture={auto}, "
                  f"record_event_on_substream={record_evt}, layers {layer_ids}, T={T}) =====", flush=True)
            results[scenario_key(name, sc)] = attempt(sc, inputs, ref, auto, record_evt, name)
            captures_done += 1

    # the mutant must not reproduce the clean arm: if it does, bit-identity has
    # no discriminating power here and nothing in this run may be called a pass.
    if mutant != "none":
        survived = [k for k, r in results.items() if r["captured"] and r["all_bit_identical"]]
        v.must_be_red(not survived,
                      f"mutant {mutant}: no arm came out captured-and-bit-identical (survived: {survived})")
        if survived:
            v.note("🔴 the mutant passed the same criteria as the clean arm ⇒ the criterion has no "
                   "discriminating power in this run (most likely nothing ran on the side stream at all); "
                   "the clean arm's PASS must NOT be reported")
        else:
            v.note("the mutant went red as required; this is what makes the clean arm's bit-identity mean "
                   "'the side-stream work really happened and was joined'")

    # the positive control (2026-09-22 re-read, RUNBOOK_sidestream.md §0/§2): 107025's real trigger is
    # "the side stream's last event record is not joined by the main stream". Only off_evt builds that
    # shape (an extra unjoined record after the layer's own record+join), so only off_evt must go red.
    # "off" (production record+join) and "off_evtw" (extra record, joined) are clean arms and must
    # capture -- the pre-09-22 rule "every off* arm must raise" came from the retired
    # "platform cannot capture a side stream" hypothesis and made the clean arm unpassable.
    for key in [k for k in results if k.startswith("off_evt@")]:
        off_failed = results[key]["captured"] is False
        v.must_be_red(off_failed, f"unjoined tail record on the side stream [{key}] (expect 107025)")
        if not off_failed:
            v.note("🔴 the positive control captured: an unjoined side-stream tail record no longer "
                   "triggers 107025 here, so the probe has no discriminating power -- stop and re-judge")
    if mutant == "none":
        for key in [k for k in results if k.split("@")[0] in ("off", "off_evtw")]:
            v.check(results[key]["captured"] is True,
                    f"clean arm [{key}] captured (error: {results[key]['error']})")
    if not [k for k in results if k.split("@")[0] == "off_evt"]:
        v.note("no 107025 positive control in this run: run --arm off_evt (separate process) before concluding")

    if ids_buffers_added:
        v.note(f"the test prepared kt's int64 id buffers itself for layers {sorted(set(ids_buffers_added))} "
               "(it builds KTLayerOffload objects by hand and never calls _prepare_capture_buffers). Since "
               "2026-09-22 production fills them per layer instance; if a layer here needed no help that is "
               "not evidence either way.")
    for key, res in results.items():
        if not key.split("@")[0].startswith("on"):
            continue
        if res["captured"]:
            v.note(f"arm {key} CAPTURED: the side-stream route is alive if and only if the replay checks "
                   "above passed; form C drops to fallback in that case")
        else:
            v.note(f"arm {key} did NOT capture: the side-stream route stays dead on this platform and form C "
                   "is decided by t4's ceiling")

    return v.finish(os.path.join(args.out, "t5_result.json"),
                    {"arms": results, "scenarios": scenarios, "inputs": args.inputs, "mutant": mutant,
                     "ids_buffers_added_by_test": sorted(set(ids_buffers_added)),
                     "measures_performance": False})


if __name__ == "__main__":
    sys.exit(main())
