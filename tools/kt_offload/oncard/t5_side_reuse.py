# SPDX-License-Identifier: Apache-2.0
"""T5r: the side stream in the shape a *service* runs it -- many layers, many
consecutive replays, no host synchronize in between, and routing tensors whose
storage the caller stops owning the moment ``route()`` returns.

Why this file exists (MOE_OFFLOAD_PLAN §4.23)
---------------------------------------------
``t5_auto_dispatch.py`` gives the forked path 9/9 bit-identical replays, and
the same code then got 90.1% of decode positions right in a service, with 3.1%
of the decode steps degenerate (every top-5 logprob equal to ``ln(1/129280)``).
A green single-layer check that cannot go red on the production failure is not
evidence, so this file removes, one at a time, the three things t5 does that a
service does not:

  1. **3 layers.** t5 builds three; a service runs 40 back to back. ``--layers``.
  2. **``torch.npu.synchronize()`` around every replay.** That drains the side
     stream between steps, so nothing of step N can still be in flight when
     step N+1 starts writing. A service never does it. ``--steps K`` enqueues K
     replays back to back and synchronizes **once**, at the end.
  3. 🔴 **persistent routing tensors.** This is the one that matters. t5 hands
     ``route()`` the same ``sids``/``sw`` placeholders at every replay, so the
     storage the side stream reads is alive and untouched for the whole
     process. In a service the router allocates ``topk_ids``/``topk_weights``
     per step and ``routed_experts.py:707-708`` rebinds both names to
     ``split_routing``'s results the moment ``route()`` returns -- the
     originals' last reference is gone, their blocks go back to the caching
     allocator, and inside a capture the allocator hands that address to the
     next allocation of the same size. Every write the compute stream makes
     afterwards is ordered against the compute stream's own reads. It is not
     ordered against the **side** stream's, and
     ``copy_inputs_to_cpu_buffers`` reads ``topk_weights`` from the side stream.

``--recycle`` is the knob for (3), and it is a deliberate amplifier rather than
a different bug: it makes the *same* missing ordering edge fire every time
instead of 3% of the time, which is what turns a coin flip into a verdict.

  ``off``      nothing; the natural shape. Says whether it diverges on its own.
  ``inplace``  the compute stream writes a poison value into the very buffer
               ``route()`` was handed, right after it returns. No dependence on
               what the allocator does -- the hazard under test is exactly
               "a compute-stream write to a block the side stream still has to
               read, with no event between them".
  ``realloc``  the faithful one: drop the reference (as the caller does), then
               allocate the same shape again and poison it. Reports how often
               the allocator really handed the freed address straight back; if
               it does not, this arm is not testing what it claims and says so.

The discriminating pair
-----------------------
Every arm runs the *same* schedule and the *same* poison; only the stream the
D2H and the host-func node sit on changes.

  ``serial``   captured, side stream off (``_submit_captured``). Its D2H is on
               the compute stream, so it is ordered before the poison by
               construction. **Must be bit-identical** -- that is what says the
               poison itself breaks no arithmetic.
  ``fork``     captured, side stream on (``_submit_captured_forked``).
  mutants      ``drop-join`` (the compute stream never waits for the side
               stream) and ``early-fork`` (the fork event recorded before the
               compute-stream fills instead of after -- the ordering edge this
               file's fix adds, taken out again). Both must be RED; a mutant
               that stays green means the criterion has no discriminating power
               in this run and nothing here is a pass.

``--probe`` additionally reads the pinned input buffers at the moment each
host-func node runs and compares them with what that (step, layer) was given.
That is what turns "the output is wrong" into "the weights buffer of step 3
layer 5 held 7.0 -- the poison -- instead of this step's routing weights".
The probe adds host work inside the callback, so the verdict run is probe-free
by default and the probe is a separate, named run.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oncard_common as C  # noqa: E402

NUM_LAYERS = 40  # DSV4.1 main layers (npu_moe.fake_vllm_config)

PLAN = """t5r plan (the service's shape, not the single-layer one):
  stack   : --layers L real offload layers chained as a residual stack
            (h = h + merge(layer_i(h))), so an error in layer i reaches the output
  steps   : --steps K captured replays enqueued BACK TO BACK; exactly one
            torch.npu.synchronize() at the very end (t5 synchronizes per replay,
            which is what hides a cross-step hazard)
  routing : the ids/weights handed to route() get written / freed right after it
            returns -- what routed_experts.py:707-708 does. --recycle says how.
  arms    : serial (side off, captured)  -> MUST be bit-identical: the negative
            control that says the poison alone breaks nothing
            fork   (side on,  captured)  -> the arm under test
  mutants : drop-join / early-fork -> MUST be RED
  judge   : bit-identity against an EAGER serial reference computed first, per step;
            plus host-func = K per layer, and the K replays pairwise different
  scope   : correctness only. No timing is measured here."""


def plan_layers(start: int, layers: int, num_layers: int = NUM_LAYERS) -> list[int]:
    """``start``, shifted down if the window would run past the last layer."""
    if layers < 8:
        raise ValueError("layers must be >= 8: three layers is the shape that was already green and wrong")
    if layers > num_layers:
        raise ValueError(f"layers {layers} exceeds the model's {num_layers}")
    if start >= num_layers:
        raise ValueError(f"--layer {start} is past the model's {num_layers} layers")
    first = min(start, num_layers - layers)
    return list(range(first, first + layers))


def classify(got, want, poison, previous) -> str:
    """Name what a pinned buffer held, so the evidence reads as a sentence."""
    import torch

    def same(a, b):
        return b is not None and a.shape == b.shape and bool(torch.equal(a, b))

    if same(got, want):
        return "this step's value"
    if poison is not None and got.numel() and bool((got == poison).all()):
        return f"the poison value ({poison})"
    if same(got, previous):
        return "the PREVIOUS step's value"
    return "neither this step's, the previous step's, nor the poison value"


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0, help="first layer of the window")
    ap.add_argument("--layers", type=int, default=8, help=">=8; 40 is the production stack")
    ap.add_argument("--steps", type=int, default=8, help="replays enqueued back to back, no sync in between")
    ap.add_argument("--tokens", type=int, default=1, help="bs; decode is 1")
    ap.add_argument("--n", type=int, default=100, help="resident experts per layer")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--numa", default="1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recycle", choices=("off", "inplace", "realloc"), default="inplace",
                    help="what happens to the routing buffers right after route() returns")
    ap.add_argument("--arms", default="serial,fork", help="comma list of serial,fork")
    ap.add_argument("--mutant", choices=("none", "drop-join", "early-fork"), default="drop-join",
                    help="the arm that MUST be red; --mutant none only to bisect (the verdict then fails "
                         "on 'zero mutation arms', by design)")
    ap.add_argument("--eager-every", type=int, default=0,
                    help="run an EAGER forward through the stack every N replays -- what a service does "
                         "when a prefill lands between two decode steps (eager never forks, but it writes "
                         "kt's pinned buffers and shares the one process-wide CPUInfer queue). 0 = never.")
    ap.add_argument("--eager-tokens", type=int, default=16,
                    help="token count of that eager forward; 16 is not a capture size (so kt uses its temp "
                         "buffer, as a real prefill chunk does), --eager-tokens 1 makes it share the "
                         "captured decode's pinned slots")
    ap.add_argument("--stress", choices=("none", "ids-delay"), default="none",
                    help="ids-delay: make the compute stream LATE writing this layer's int64 routing ids, "
                         "and leave everything else alone. One stressor, two code paths, and which one it "
                         "lands in is the whole point: where the fork event is recorded BEFORE that write "
                         "the side stream is already released and reads whatever the buffer held (RED); "
                         "where the write happens before the fork, the side stream waits for it however "
                         "late it is (GREEN). The serial arm never forks, so it must be unaffected either "
                         "way -- that pair is what says the delay is not itself the bug.")
    ap.add_argument("--stress-reps", type=int, default=4,
                    help="how many 16 MiB device-side adds the delay enqueues per layer")
    ap.add_argument("--probe", action="store_true",
                    help="read the pinned input buffers inside every host-func node and say which "
                         "(step, layer, buffer) held what. Adds host work in the callback.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()

    try:
        layer_ids = plan_layers(args.layer, args.layers)
    except ValueError as exc:
        ap.error(str(exc))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        if a not in ("serial", "fork"):
            ap.error(f"--arms: unknown arm {a!r}")
    if args.steps < 3:
        ap.error("--steps must be >= 3 (a stale replay output repeats the previous one)")
    if args.tokens < 1:
        ap.error("--tokens must be >= 1")

    print(PLAN, flush=True)
    print(f"t5r: layers {layer_ids[0]}-{layer_ids[-1]} ({len(layer_ids)}), steps {args.steps}, "
          f"tokens {args.tokens}, recycle {args.recycle}, arms {arms}, mutant {args.mutant}, "
          f"stress {args.stress}{'' if args.stress == 'none' else f'x{args.stress_reps}'}, "
          f"probe {'on' if args.probe else 'off'}", flush=True)
    if args.dry_run or os.environ.get("ONCARD_DRY") == "1":
        print("t5r: --dry-run, nothing was captured and no device was touched", flush=True)
        return 0

    import npu_moe as N
    import numpy as np
    import torch

    import vllm_ascend.kt_offload.layer as KL

    v = C.Verdict("T5r side stream under the service's shape")
    v.note("correctness only; this file measures no time")
    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    numa = [int(s) for s in args.numa.split(",")]
    T = args.tokens

    # ---------------------------------------------------------------- stack
    # Built ONCE and shared by every arm: _CAPTURED_PAYLOADS keeps a strong
    # reference to each wrapper for the life of the process (a graph node reads
    # it at every replay), so a second stack on the same layer indices would hit
    # _register_wrapper's "two live wrappers on one index" refusal.
    # overlap=True at build time on purpose: after_weights_loaded is where the
    # side stream is created and subscribed for host-func reports, and both must
    # happen outside any capture. The serial arm flips the instance attribute.
    cfg = N.kt_cfg("prefix", args.n, None, args.threads, numa, graph=True, overlap=True)
    stack = []
    for lid in layer_ids:
        off, _ = N.make_offload(cfg, lid, capture_sizes=[T])
        if not off.graph_enabled:
            raise RuntimeError("KTLayerOffload did not come up in graph mode")
        off.overlap = True
        pl = off.placement
        raw = N.load_raw_layer(ck, lid, list(pl.resident_ids))
        lay = N.build_layer(raw, pl.num_slots, dev, offload=off)
        if not off._ids_i64:  # belt and braces; production fills these per instance
            off._ids_i64 = {bs: torch.empty((bs, off.top_k), dtype=torch.int64, device=dev)
                            for bs in off.capture_sizes}
        stack.append((off, lay, N.comm_method(pl.num_slots)))
        del raw
    pl0 = stack[0][0].placement
    print(N.hbm_line(), flush=True)

    # ------------------------------------------------------------- the data
    rng = np.random.default_rng(args.seed + 7000 + 97 * layer_ids[0] + T)
    inputs = []
    for _ in range(args.steps):
        x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16)
        ids, w = C.make_routing(rng, T, pl0.cpu_ids, list(pl0.resident_ids))
        inputs.append((x.to(dev), ids.to(dev), w.to(dev)))
    print(f"coverage (step 0): {C.coverage(inputs[0][1].cpu(), pl0.resident_mask()).line()}", flush=True)

    # A poison that is a *valid* CPU expert id and a weight value no router
    # produces: the point is a wrong number reaching kt, never an out-of-range
    # index (that would be a host-side memory error, not an arithmetic one).
    poison_id = int(pl0.cpu_ids[-1])
    poison_w = 7.0

    # ------------------------------------------------- eager serial reference
    def run_eager(x0, ids0, w0):
        h = x0
        for off, lay, comm in stack:
            npu_routing = off.route(h, w0, ids0, in_profile_run=False)
            pre = N.npu_moe(lay, comm, h, npu_routing.topk_weights, npu_routing.topk_ids)
            h = h + off.merge(pre)
        return h

    ref = [run_eager(*inp).float().cpu() for inp in inputs]
    torch.npu.synchronize()
    for k in range(1, len(ref)):
        if C.bit_identical(ref[k], ref[k - 1])[0]:
            raise RuntimeError("two reference steps came out identical; the inputs do not discriminate")
    print(f"eager serial reference: {len(ref)} steps, pairwise different", flush=True)

    sx, sids, sw = (t.clone() for t in inputs[0])
    # A prefill-shaped eager forward for --eager-every (its own tensors: it must
    # not touch the captured graph's placeholders).
    P = args.eager_tokens
    e_ids, e_w = C.make_routing(rng, P, pl0.cpu_ids, list(pl0.resident_ids))
    eager_inputs = (
        torch.from_numpy(rng.standard_normal((P, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16).to(dev),
        e_ids.to(dev), e_w.to(dev),
    )

    # ----------------------------------------------------------- the stress
    # Pre-allocated: a delay that allocates would change the graph pool it is
    # supposed to leave alone. 2048x2048 float32 = 16 MiB per add.
    delay_dst = torch.zeros((2048, 2048), dtype=torch.float32, device=dev) if args.stress != "none" else None
    delay_src = torch.ones((2048, 2048), dtype=torch.float32, device=dev) if args.stress != "none" else None

    def compute_delay() -> None:
        """Device work on the CALLER's stream. Enqueued, never waited on."""
        for _ in range(args.stress_reps):
            delay_dst.add_(delay_src)

    original_device_fill = KL._device_fill

    def _device_fill_delayed(dst, src):
        """``_device_fill`` with the compute stream made late before the ids.

        This is where the fixed path writes the routing ids: on the compute
        stream, BEFORE ``fork.record(compute)``. The same delay as below, in
        the same place in the data flow, and here it must be harmless -- the
        side stream cannot start until the fork event, and the fork event is
        not recorded until this returns.
        """
        if getattr(src, "dtype", None) in (torch.int32, torch.int64):
            compute_delay()
        return original_device_fill(dst, src)

    original_ids_for_kt = KL.KTLayerOffload._ids_for_kt

    def _ids_for_kt_delayed(self, topk_ids):
        """``_ids_for_kt`` with the compute stream made late on purpose.

        layer.py:_submit_captured_forked records the fork event and only then
        calls this, which writes ``self._ids_i64[bs]`` on the compute stream --
        while the side stream, already released by that event, is on its way to
        read the same buffer. Nothing orders the two. This delay decides the
        race in the side stream's favour so the outcome can be read.
        """
        if topk_ids.dtype == torch.int64:
            return topk_ids
        compute_delay()
        return original_ids_for_kt(self, topk_ids)

    # ------------------------------------------------------------- the probe
    probe_log: list[dict] = []
    original_host_forward = KL._kt_host_forward
    original_packed_forward = KL._kt_packed_forward

    def _snapshot_kt(payload) -> dict:
        """kt's own pinned slot buffers (the layer_idx % 2 pair)."""
        _wrapper, shape_ref, _stream, layer_idx = payload
        bufs = KL._kt_buffer_class().capture_buffers[shape_ref.shape[0]]
        slot = layer_idx % 2
        return {"layer": int(layer_idx), "slot": slot, "where": "kt slot",
                "ids": bufs[1][slot].clone(), "w": bufs[3][slot].clone()}

    def _snapshot_packed(payload) -> dict:
        """Our own per-layer pinned buffers (the packed layout)."""
        _wrapper, buf, layer_idx = payload
        return {"layer": int(layer_idx), "slot": -1, "where": "own buffer",
                "ids": buf["pin_ids"].clone(), "w": buf["pin_weights"].clone()}

    def _probing(original, snapshot):
        def probing(payload):
            try:
                probe_log.append(snapshot(payload))
            except BaseException as exc:  # noqa: BLE001
                probe_log.append({"layer": -1, "error": repr(exc)})
            return original(payload)

        return probing

    # ------------------------------------------------------------- mutants
    def _drop_join(self, device) -> None:
        """MUTANT: record the join on the side stream, never wait for it."""
        pending, self._pending_join = self._pending_join, None
        self._check_capture_hostfunc()
        release = getattr(self, "_release_side_buffers", None)
        (release or self._release_side_slot)()
        if pending is None:
            return
        join, side, _compute = pending
        join.record(side)

    def _early_fork(self, hs, topk_ids, topk_weights) -> None:
        """MUTANT: the pre-fix order -- fork FIRST, fill the side buffers after.

        Byte for byte the fixed ``_submit_captured_forked`` except that
        ``fork.record(compute)`` happens before this layer's compute-stream
        fills instead of after them. That single edge is the fix; removing it
        must bring the failure back.
        """
        bs = hs.shape[0]
        device = hs.device
        compute, side = torch.npu.current_stream(device), KL._side_stream(device)
        buf = self._side_buffers(bs, device)
        self._claim_side_buffers(buf)
        self._hostfunc_at_capture = KL.HOST_FUNC_CALLS.get(self.layer_idx, 0)
        fork, join = KL._new_event(device), KL._new_event(device)
        KL._SIDE_EVENTS.append((fork, join))
        fork.record(compute)  # <-- the mutation: before the fills, not after
        side.wait_event(fork)
        if args.stress == "ids-delay":
            compute_delay()  # the same stressor, back in the place the fix took it out of
        original_device_fill(buf["dev_hidden"], hs)
        original_device_fill(buf["dev_ids"], topk_ids)
        original_device_fill(buf["dev_weights"], topk_weights)
        payload = self._packed_payload(bs)
        KL._CAPTURED_PAYLOADS.append(payload)
        with KL._stream_ctx(side, device):
            if not KL._capturing():
                raise RuntimeError("kt_offload: the graph capture did not propagate to the side stream")
            KL._host_copy(buf["pin"], buf["dev"])
            KL._launch_host_func(side, KL._kt_packed_forward, payload)
        self._pending_join = (join, side, compute)

    # --------------------------------------------------------- the arm runner
    def arm(name: str, forked: bool, mutant: str) -> dict:  # noqa: C901
        key = f"{name}{'' if mutant == 'none' else '+' + mutant}"
        is_mutant = mutant != "none"

        def judge(ok: bool, what: str) -> bool:
            return ok if is_mutant else v.check(ok, what)

        for off, _, _ in stack:
            off.overlap = forked
            off._pending = None
            off._pending_captured = False
            off._pending_join = None
            off._side_inflight = False  # harmless on a build without the guard
        getattr(KL, "_SIDE_BUFFER_OWNER", getattr(KL, "_SIDE_INFLIGHT", {})).clear()
        out: dict = {"forked": forked, "mutant": mutant, "recycle": args.recycle, "captured": None,
                     "error": None, "where": None, "all_bit_identical": None}
        ghost_hits: list = []

        def one_layer(off, lay, comm, h, ids_src, w_src):
            """One layer, with the caller-side buffer lifetime a service has."""
            ids = ids_src.clone()  # graph-pool allocations, as the router's are
            w = w_src.clone()
            npu_routing = off.route(h, w, ids, in_profile_run=False)
            w2, ids2 = npu_routing.topk_weights, npu_routing.topk_ids
            if args.recycle == "inplace":
                # The compute stream writes the block the side stream still has
                # to read. Nothing here depends on the allocator.
                ids.fill_(poison_id)
                w.fill_(poison_w)
            elif args.recycle == "realloc":
                ptr_ids, ptr_w = ids.data_ptr(), w.data_ptr()
                del ids, w  # routed_experts.py:707-708: the last reference goes
                ghost_ids = torch.empty((T, C.TOPK), dtype=torch.int32, device=h.device)
                ghost_w = torch.empty((T, C.TOPK), dtype=torch.float32, device=h.device)
                ghost_hits.append((ghost_ids.data_ptr() == ptr_ids, ghost_w.data_ptr() == ptr_w))
                ghost_ids.fill_(poison_id)
                ghost_w.fill_(poison_w)
                del ghost_ids, ghost_w
            pre = N.npu_moe(lay, comm, h, w2, ids2)
            return h + off.merge(pre)

        g = torch.npu.NPUGraph()
        saved_join = KL.KTLayerOffload._join_side_stream
        saved_forked = KL.KTLayerOffload._submit_captured_forked
        saved_ids = KL.KTLayerOffload._ids_for_kt
        saved_fill = KL._device_fill
        if args.stress == "ids-delay":
            # Both patches go in; which one the build actually reaches is the
            # answer. Pre-fix the ids are written through _ids_for_kt, AFTER
            # fork.record; with the fix they are a _device_fill BEFORE it.
            KL.KTLayerOffload._ids_for_kt = _ids_for_kt_delayed
            KL._device_fill = _device_fill_delayed
        if mutant == "drop-join":
            KL.KTLayerOffload._join_side_stream = _drop_join
        elif mutant == "early-fork":
            if not hasattr(KL.KTLayerOffload, "_side_buffers"):
                out["error"] = ("mutant early-fork needs the per-layer side buffers (the fix); this build "
                                "does not have _side_buffers")
                print(f"  arm {key}: {out['error']}", flush=True)
                return out
            KL.KTLayerOffload._submit_captured_forked = _early_fork
        if args.probe:
            KL._kt_host_forward = _probing(original_host_forward, _snapshot_kt)
            KL._kt_packed_forward = _probing(original_packed_forward, _snapshot_packed)
        probe_log.clear()
        calls0 = dict(KL.HOST_FUNC_CALLS)
        try:
            with torch.npu.graph(g):
                h = sx
                for off, lay, comm in stack:
                    h = one_layer(off, lay, comm, h, sids, sw)
                sout = h
            torch.npu.synchronize()
            out["captured"] = True
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()
            mine = [ln.strip() for ln in tb if "kt_offload" in ln or "t5_side_reuse" in ln]
            out.update(captured=False, error=f"{type(exc).__name__}: {str(exc)[:600]}",
                       where=(mine[-1] if mine else tb[-2] if len(tb) > 1 else ""))
            print(f"  arm {key}: capture raised -- {out['error']}\n    at {out['where']}", flush=True)
            KL._kt_host_forward, KL._kt_packed_forward = original_host_forward, original_packed_forward
            KL.KTLayerOffload._join_side_stream = saved_join
            KL.KTLayerOffload._submit_captured_forked = saved_forked
            KL.KTLayerOffload._ids_for_kt = saved_ids
            KL._device_fill = saved_fill
            return out
        finally:
            KL.KTLayerOffload._join_side_stream = saved_join
            KL.KTLayerOffload._submit_captured_forked = saved_forked
            KL.KTLayerOffload._ids_for_kt = saved_ids
            KL._device_fill = saved_fill

        if args.recycle == "realloc":
            hit_ids = sum(1 for a, _ in ghost_hits if a)
            hit_w = sum(1 for _, b in ghost_hits if b)
            out["ghost_reuse"] = {"layers": len(ghost_hits), "ids_same_address": hit_ids,
                                  "weights_same_address": hit_w}
            print(f"  arm {key}: the poison landed on the freed block in {hit_w}/{len(ghost_hits)} layers "
                  f"(weights) and {hit_ids}/{len(ghost_hits)} (ids)", flush=True)
            judge(hit_w == len(ghost_hits),
                  f"arm {key}: every layer's freed weights block was handed straight back "
                  f"({hit_w}/{len(ghost_hits)}); if not, --recycle realloc is not testing what it claims")

        ran = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - calls0.get(lid, 0) for lid in layer_ids}
        out["host_func_during_capture"] = ran
        judge(all(n == 0 for n in ran.values()), f"arm {key}: capture ran no host-func node (ran {ran})")

        # ---- K replays, back to back, ONE synchronize at the end ----------
        outs = [torch.empty_like(sout) for _ in range(args.steps)]
        probe_log.clear()
        before = dict(KL.HOST_FUNC_CALLS)
        eager_passes = 0
        try:
            for k, (x, ids, w) in enumerate(inputs):
                if args.eager_every and k and k % args.eager_every == 0:
                    run_eager(*eager_inputs)
                    eager_passes += 1
                sx.copy_(x)  # compute stream, async: no host sync anywhere
                sids.copy_(ids)
                sw.copy_(w)
                g.replay()
                outs[k].copy_(sout)
            torch.npu.synchronize()  # the only one
        finally:
            KL._kt_host_forward, KL._kt_packed_forward = original_host_forward, original_packed_forward
        grew = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - before.get(lid, 0) for lid in layer_ids}
        out["host_func_per_layer"] = grew
        out["eager_passes"] = eager_passes
        judge(all(n == args.steps for n in grew.values()),
              f"arm {key}: every layer's host-func ran once per replay (want {args.steps}, "
              f"got {min(grew.values())}..{max(grew.values())}; {eager_passes} eager passes do not "
              f"count -- eager runs kt's own submit/sync, not a host-func node)")

        results, all_same = [], True
        for k in range(args.steps):
            r = outs[k].float().cpu()
            same, how = C.bit_identical(r, ref[k])
            all_same = all_same and same
            judge(same, f"arm {key}: step {k} == eager serial step {k}: {how}"
                        + ("" if same else f"; rel_l2 {C.rel_l2(r, ref[k]):.3e}"
                                           f", nonfinite {C.nan_inf_count(r)}/{r.numel()}"))
            results.append({"step": k, "bit_identical": same, "how": how,
                            "rel_l2": None if same else C.rel_l2(r, ref[k]),
                            "nonfinite": C.nan_inf_count(r)})
        # The stale-output fingerprint. All pairs is O(K^2) device->host copies,
        # so past a dozen steps it degrades to consecutive pairs -- which is the
        # pair a stale replay would make identical anyway.
        pairs = ([(i, j) for i in range(args.steps) for j in range(i + 1, args.steps)]
                 if args.steps <= 12 else [(i, i + 1) for i in range(args.steps - 1)])
        bad_pairs = [(i, j) for i, j in pairs if C.bit_identical(outs[i], outs[j])[0]]
        judge(not bad_pairs, f"arm {key}: the {len(pairs)} compared step pairs are all different "
                             f"(identical: {bad_pairs[:4]})")
        out["steps"] = results
        out["all_bit_identical"] = all_same
        out["first_divergent_step"] = next((r["step"] for r in results if not r["bit_identical"]), None)
        print(f"  arm {key}: bit-identical {sum(r['bit_identical'] for r in results)}/{args.steps}"
              + ("" if out["first_divergent_step"] is None
                 else f", first divergence at step {out['first_divergent_step']}"), flush=True)
        if args.probe:
            out["probe"] = probe_report(key, list(probe_log))
        return out

    def probe_report(key: str, log: list[dict]) -> dict:
        """Name the (step, layer, buffer) whose pinned content was not this step's."""
        import torch as _t

        errs = [e for e in log if e.get("layer", 0) < 0]
        expected_n = args.steps * len(layer_ids)
        rep = {"entries": len(log), "expected": expected_n, "probe_errors": errs[:3], "mismatches": []}
        if errs or len(log) != expected_n:
            print(f"  probe[{key}]: {len(log)} callbacks recorded, expected {expected_n}"
                  f"{'; probe errors ' + str(errs[:3]) if errs else ''}", flush=True)
        for i, e in enumerate(log):
            step, pos = divmod(i, len(layer_ids))
            if step >= args.steps or e.get("layer") != layer_ids[pos]:
                rep["order_broken_at"] = i
                print(f"  probe[{key}]: callback {i} is layer {e.get('layer')}, not {layer_ids[pos]} -- "
                      f"the (step, layer) mapping past here is a guess", flush=True)
                break
            want_ids = inputs[step][1].cpu().to(_t.int64)
            want_w = inputs[step][2].cpu()
            prev_ids = inputs[step - 1][1].cpu().to(_t.int64) if step else None
            prev_w = inputs[step - 1][2].cpu() if step else None
            for what, got, want, pois, prev in (
                ("ids", e["ids"], want_ids, poison_id, prev_ids),
                ("weights", e["w"], want_w, poison_w, prev_w),
            ):
                if got.shape == want.shape and bool(_t.equal(got, want)):
                    continue
                verdict = classify(got, want, pois, prev)
                item = {"step": step, "layer": e["layer"], "slot": e["slot"], "where": e["where"],
                        "buffer": what, "held": verdict, "got": got.flatten()[:6].tolist(),
                        "want": want.flatten()[:6].tolist()}
                rep["mismatches"].append(item)
                if len(rep["mismatches"]) <= 8:
                    print(f"  🔴 probe[{key}]: step {step} layer {e['layer']} ({e['where']}"
                          + (f", kt slot {e['slot']}" if e["slot"] >= 0 else "")
                          + f") -- the {what} buffer held {verdict}: {item['got']} "
                          f"(this step wants {item['want']})", flush=True)
        rep["mismatch_count"] = len(rep["mismatches"])
        if not rep["mismatches"]:
            print(f"  probe[{key}]: every pinned input buffer held this step's value "
                  f"({len(log)} callbacks checked)", flush=True)
        return rep

    # --------------------------------------------------------------- run it
    results: dict[str, dict] = {}
    for name in arms:
        print(f"\n===== arm {name} (side stream {'ON' if name == 'fork' else 'off'}, "
              f"recycle {args.recycle}) =====", flush=True)
        results[name] = arm(name, forked=(name == "fork"), mutant="none")
        v.check(results[name]["captured"] is True,
                f"arm {name} captured (error: {results[name]['error']})")
    if args.mutant != "none":
        print(f"\n===== MUTANT arm fork+{args.mutant} (must be RED) =====", flush=True)
        m = arm("fork", forked=True, mutant=args.mutant)
        results[f"fork+{args.mutant}"] = m
        red = (m["captured"] is not True) or (m["all_bit_identical"] is not True)
        v.must_be_red(red, f"mutant {args.mutant}: captured={m['captured']} "
                           f"bit_identical={m['all_bit_identical']} error={m['error']}")
        if not red:
            v.note("🔴 the mutant passed the same criteria as the clean arm ⇒ the criterion has no "
                   "discriminating power in this run; no arm here may be called a pass")

    if "serial" in results and results["serial"].get("all_bit_identical") is not True:
        v.note("🔴 the NEGATIVE control (side stream off) is not bit-identical either ⇒ the harness itself "
               "is wrong (or the poison is reaching the compute path); the fork arm's result says nothing "
               "until this is green")
    if "fork" in results and "serial" in results:
        f_ok = results["fork"].get("all_bit_identical")
        s_ok = results["serial"].get("all_bit_identical")
        v.note(f"discriminating pair: serial {'green' if s_ok else 'RED'} / fork {'green' if f_ok else 'RED'} "
               f"-- same schedule, same poison, only the stream the D2H and the host-func sit on differs")

    return v.finish(os.path.join(args.out, "t5r_result.json"),
                    {"arms": results, "layer_ids": layer_ids, "steps": args.steps, "tokens": T,
                     "recycle": args.recycle, "mutant": args.mutant, "probe": args.probe,
                     "stress": args.stress, "stress_reps": args.stress_reps,
                     "poison": {"id": poison_id, "weight": poison_w},
                     "measures_performance": False})


if __name__ == "__main__":
    sys.exit(main())
