# SPDX-License-Identifier: Apache-2.0
"""T3: the side-stream overlap path (overlap_side_stream=true) in isolation.

Why in isolation: the 2026-09-20 whole-model A/B with the switch on produced
38/50 different GSM8K answers, and a whole-model run cannot say *which* of the
overlap's moving parts broke. This test captures a real NPUGraph over a few
offload layers with the switch on, replays it with several different inputs,
and compares against the eager arm of the same layers.

Covered here and NOT covered by T2 (which tests the non-forked captured path):
  * capture propagation onto the side stream (route() refuses if the platform
    does not follow the event wait; that refusal is reported, not swallowed);
  * kt's pinned-buffer slot reuse: >= 3 layers, so layers L and L+2 share
    ``layer_idx % 2`` and the graph's ordering is what keeps them apart;
  * more than one capture size, replayed alternately (a decode batch changes
    size between steps when requests come and go);
  * the host-func node really running per replay per layer (HOST_FUNC_CALLS).

Criteria (all bit-identity, never timing -- every failure mode here is faster):
  replay == eager      per input, per capture size
  replays differ       pairwise, and replay k != eager k-1 (stale-output shape)
  host-func count      +1 per layer per replay, 0 during capture
  overlap == serial    the same inputs through overlap=false layers, bit-identical
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
    ap.add_argument("--layer", type=int, default=20, help="first layer; --layers consecutive ones are built")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tokens", default="4,8", help="capture sizes, replayed alternately")
    ap.add_argument("--inputs", type=int, default=3)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--numa", default="1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("ONCARD_RUN_DIR", "."))
    args = ap.parse_args()
    if args.inputs < 3 or args.layers < 3:
        ap.error("--inputs and --layers must be >= 3 (slot reuse needs L and L+2)")

    import npu_moe as N

    import vllm_ascend.kt_offload.layer as KL

    rt = N.init_runtime()
    dev = rt.device
    ck = C.Ckpt()
    sizes = [int(s) for s in args.tokens.split(",")]
    v = C.Verdict("T3 side-stream overlap")
    numa = [int(s) for s in args.numa.split(",")]
    layer_ids = list(range(args.layer, args.layer + args.layers))
    rng = np.random.default_rng(args.seed + 3000)

    def build(overlap: bool):
        """One stack of offload layers, all with the same placement."""
        cfg = N.kt_cfg("prefix", args.n, None, args.threads, numa, graph=True, overlap=overlap)
        stack = []
        for lid in layer_ids:
            off, _ = N.make_offload(cfg, lid, capture_sizes=sizes)
            if not off.graph_enabled:
                raise RuntimeError("KTLayerOffload did not come up in graph mode")
            pl = off.placement
            raw = N.load_raw_layer(ck, lid, list(pl.resident_ids))
            lay = N.build_layer(raw, pl.num_slots, dev, offload=off)
            stack.append((off, lay, N.comm_method(pl.num_slots)))
        return stack

    def run_stack(stack, x, ids, w):
        """route -> apply -> merge for every layer, feeding each layer the same input."""
        out = None
        for off, lay, comm in stack:
            _, merged, _ = N.offload_forward(off, lay, comm, x, w, ids)
            out = merged if out is None else out + merged
        return out

    print(f"===== T3 layers {layer_ids}, capture sizes {sizes} =====", flush=True)
    serial = build(overlap=False)
    pl = serial[0][0].placement
    inputs = {}
    for T in sizes:
        per_size = []
        for _ in range(args.inputs):
            x = torch.from_numpy(rng.standard_normal((T, C.HIDDEN)).astype(np.float32)).to(torch.bfloat16)
            ids, w = C.make_routing(rng, T, pl.cpu_ids, list(pl.resident_ids))
            per_size.append((x.to(dev), ids.to(dev), w.to(dev)))
            print(f"T={T} input {len(per_size)}: {C.coverage(ids, pl.resident_mask()).line()}", flush=True)
        inputs[T] = per_size

    # reference: overlap off, eager
    ref = {T: [run_stack(serial, *inp).float().cpu() for inp in inputs[T]] for T in sizes}
    torch.npu.synchronize()
    for off, _, _ in serial:
        del off
    del serial
    torch.npu.empty_cache()

    # arm under test: overlap on
    stack = build(overlap=True)
    eager = {}
    for T in sizes:
        eager[T] = [run_stack(stack, *inp).float().cpu() for inp in inputs[T]]
        for k, (got, want) in enumerate(zip(eager[T], ref[T])):
            same, how = C.bit_identical(got, want)
            v.check(same, f"T={T} input {k} eager overlap == serial: {how}"
                          + ("" if same else f"; rel_l2 {C.rel_l2(got, want):.3e}"))
    torch.npu.synchronize()

    graphs, statics = {}, {}
    for T in sizes:
        sx, sids, sw = (t.clone() for t in inputs[T][0])
        calls0 = dict(KL.HOST_FUNC_CALLS)
        g = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(g):
                sout = run_stack(stack, sx, sids, sw)
        except RuntimeError as exc:
            v.check(False, f"T={T} capture with overlap raised: {str(exc)[:300]}")
            return v.finish(os.path.join(args.out, "t3_result.json"),
                            {"host_func_calls": dict(KL.HOST_FUNC_CALLS)})
        torch.npu.synchronize()
        ran = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - calls0.get(lid, 0) for lid in layer_ids}
        v.check(all(n == 0 for n in ran.values()), f"T={T} capture ran no host-func node (ran {ran})")
        graphs[T], statics[T] = g, (sx, sids, sw, sout)

    replays = {T: [] for T in sizes}
    for k in range(args.inputs):
        for T in sizes:  # alternate the capture sizes, as a changing decode batch does
            sx, sids, sw, sout = statics[T]
            x, ids, w = inputs[T][k]
            sx.copy_(x), sids.copy_(ids), sw.copy_(w)
            before = dict(KL.HOST_FUNC_CALLS)
            graphs[T].replay()
            torch.npu.synchronize()
            grew = {lid: KL.HOST_FUNC_CALLS.get(lid, 0) - before.get(lid, 0) for lid in layer_ids}
            v.check(all(n == 1 for n in grew.values()),
                    f"T={T} replay {k}: every layer's host-func ran once (got {grew})")
            r = sout.float().cpu().clone()
            replays[T].append(r)
            same, how = C.bit_identical(r, ref[T][k])
            v.check(same, f"T={T} replay {k} == serial {k}: {how}"
                          + ("" if same else f"; rel_l2 {C.rel_l2(r, ref[T][k]):.3e}"))
            if k:
                v.check(not C.bit_identical(r, ref[T][k - 1])[0],
                        f"T={T} replay {k} != serial {k - 1} (stale-output fingerprint)")
    for T in sizes:
        for i in range(len(replays[T])):
            for j in range(i + 1, len(replays[T])):
                v.check(not C.bit_identical(replays[T][i], replays[T][j])[0], f"T={T} replay {i} != replay {j}")

    return v.finish(os.path.join(args.out, "t3_result.json"),
                    {"host_func_calls": dict(KL.HOST_FUNC_CALLS), "layers": layer_ids, "sizes": sizes})


if __name__ == "__main__":
    sys.exit(main())
