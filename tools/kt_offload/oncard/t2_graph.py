# SPDX-License-Identifier: Apache-2.0
"""T2: graph mode, one layer. The M2 captured path under a real NPUGraph.

One offload layer (prefix N, KTLayerOffload built with graph mode on and capture
size T), the forward_impl sequence (route -> apply -> merge) captured once into a
torch.npu.NPUGraph on static buffers, then replayed with >= 3 different inputs.

  replay == eager   for each input k: the replay output equals the eager output of the
                    same input (bit-identical; both run the same kt kernel and the same
                    NPU kernels -- if not bit-identical the rel_l2 is printed and the
                    check fails; the eager/replay floor is then the next thing to measure)
  all differ        the k replay outputs are pairwise different, and replay k differs
                    from eager k-1 (a stale host-func output repeats the previous one)
  host-func count   HOST_FUNC_CALLS[layer] grows by exactly 1 per replay, 0 at capture
  no 10701x         checked by run_oncard.sh over the log and the plog (reported there)
  mutation          the capture probe forced to False (route() takes the eager branch
                    inside the capture, kt's probe too): must go RED -- either the capture
                    raises, or replays repeat / disagree with eager.
The mutation runs last: a failed capture may leave the stream unusable.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oncard_common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tokens", type=int, default=4, help="the one capture size")
    ap.add_argument("--inputs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--numa", default="1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()
    if args.inputs < 3:
        ap.error("--inputs must be >= 3")

    import npu_moe as N
    import vllm_ascend.kt_offload.layer as KL

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    L, T = args.layer, args.tokens
    v = C.Verdict("T2 graph single layer")
    numa = [int(s) for s in args.numa.split(",")]

    cfg = N.kt_cfg("prefix", args.n, None, args.threads, numa, graph=True)
    off, _ = N.make_offload(cfg, L, capture_sizes=[T])
    if not off.graph_enabled:
        v.check(False, "KTLayerOffload did not come up in graph mode")
        return v.finish(os.path.join(args.out, "t2_result.json"))
    pl = off.placement
    raw = N.load_raw_layer(ck, L, list(pl.resident_ids))
    lay = N.build_layer(raw, pl.num_slots, dev, offload=off)
    comm = N.comm_method(pl.num_slots)
    rng = np.random.default_rng(args.seed + 2000 + L)
    inputs = []
    for _ in range(args.inputs):
        x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16)
        ids, w = C.make_routing(rng, T, pl.cpu_ids, list(pl.resident_ids))
        inputs.append((x.to(dev), ids.to(dev), w.to(dev)))
        print(f"input {len(inputs)}: {C.coverage(ids, pl.resident_mask()).line()}", flush=True)

    def step(x, ids, w):
        _, merged, _ = N.offload_forward(off, lay, comm, x, w, ids)
        return merged

    eager = [step(*inp).float().cpu() for inp in inputs]  # also the warm-up (vLLM warms up before capture)
    torch.npu.synchronize()
    sx, sids, sw = (t.clone() for t in inputs[0])
    calls0 = KL.HOST_FUNC_CALLS.get(L, 0)
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        sout = step(sx, sids, sw)
    torch.npu.synchronize()
    calls_cap = KL.HOST_FUNC_CALLS.get(L, 0)
    v.check(calls_cap == calls0, f"capture ran the host-func node 0 times (ran {calls_cap - calls0})")
    v.check(len(KL._CAPTURED_PAYLOADS) >= 1, f"a host-func payload was recorded ({len(KL._CAPTURED_PAYLOADS)})")

    replays = []
    for k, (x, ids, w) in enumerate(inputs):
        sx.copy_(x), sids.copy_(ids), sw.copy_(w)
        before = KL.HOST_FUNC_CALLS.get(L, 0)
        g.replay()
        torch.npu.synchronize()
        after = KL.HOST_FUNC_CALLS.get(L, 0)
        v.check(after - before == 1, f"replay {k}: HOST_FUNC_CALLS +{after - before} (want +1)")
        r = sout.float().cpu().clone()
        replays.append(r)
        same, how = C.bit_identical(r, eager[k])
        v.check(same, f"replay {k} == eager {k}: {how}" + ("" if same else f"; rel_l2 {C.rel_l2(r, eager[k]):.3e}"))
        if k:
            v.check(not C.bit_identical(r, eager[k - 1])[0], f"replay {k} != eager {k - 1} (stale-output fingerprint)")
    for i in range(len(replays)):
        for j in range(i + 1, len(replays)):
            v.check(not C.bit_identical(replays[i], replays[j])[0], f"replay {i} != replay {j}")

    # mutation: probe constant False (ours and kt's), fresh capture
    red, why = False, ""
    orig = KL._capturing
    try:
        KL._capturing = lambda: False
        KL._install_capture_probe(type(off.wrapper))
        g2 = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(g2):
                sout2 = step(sx, sids, sw)
            torch.npu.synchronize()
            outs = []
            for x, ids, w in inputs:
                sx.copy_(x), sids.copy_(ids), sw.copy_(w)
                g2.replay()
                torch.npu.synchronize()
                outs.append(sout2.float().cpu().clone())
            agree = all(C.bit_identical(o, e)[0] for o, e in zip(outs, eager))
            distinct = not any(C.bit_identical(outs[i], outs[j])[0]
                               for i in range(len(outs)) for j in range(i + 1, len(outs)))
            red = not (agree and distinct)
            why = f"captured without host-func: replay==eager {agree}, replays distinct {distinct}"
        except Exception as exc:  # noqa: BLE001
            red, why = True, f"capture/replay raised {type(exc).__name__}: {str(exc)[:200]}"
    finally:
        KL._capturing = orig
        KL._install_capture_probe(type(off.wrapper))
    print(f"   mutation probe=False: {why}", flush=True)
    v.must_be_red(red, "capture probe forced False")
    return v.finish(os.path.join(args.out, "t2_result.json"),
                    {"host_func_calls": dict(KL.HOST_FUNC_CALLS), "mutation": why})


if __name__ == "__main__":
    sys.exit(main())
