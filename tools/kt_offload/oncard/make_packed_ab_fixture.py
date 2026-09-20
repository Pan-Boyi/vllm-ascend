#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a fake packed A/B run so the judge can be exercised without a card.

Modes: identical | onechar | emptyfield | nogain | deadlayers
The gap fixtures are built so the arithmetic has a known answer: arm A gets
80 extra gaps of 62.5 us per token, so the judge's implied C must come out at
62.5. That 62.5 is a fixture constant chosen to make the division checkable --
NOT a claim about the real per-operation cost, whose model the 2026-09-20
profile retired (see packed_ab_report.RETIRED_PRICE_US).
"""
from __future__ import annotations

import json
import os
import sys

TOKENS = 10
LAYERS = 40
GAP_US = 400.0  # one exposure gap per layer in arm A


def write_csv(path: str, big_gaps_per_token: int, extra_us: float) -> None:
    """Kernels with `big_gaps_per_token` gaps of GAP_US (+extra) per token."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = ["Device_id,Name,Type,Accelerator Core,Start Time(us),Duration(us),Wait Time(us),Block Num"]
    t = 0.0
    for _ in range(TOKENS):
        for layer in range(LAYERS):
            rows.append(f"3,aclnnMoeInitRoutingV3,MoeInitRoutingV3,MIX_AIC,{t:.3f}\t,10.0,0,1")
            t += 10.0
            rows.append(f"3,aclnnGroupedMatmul,GroupedMatmul,MIX_AIC,{t:.3f}\t,20.0,0,28")
            t += 20.0
            if layer < big_gaps_per_token:
                t += GAP_US + extra_us  # exposed host wait
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")


def write_answers(path: str, mutate: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in range(50):
            text = "" if mutate == "emptyfield" else f"Answer {i}: the result is \\boxed{{{i}}}"
            if mutate == "onechar" and i == 7:
                text = "Bnswer" + text[6:]
            f.write(json.dumps({"id": f"gsm8k-{i}", "generation": text, "finish_reason": "stop",
                                "num_generated_tokens": 100, "symbolic_correct": True}) + "\n")


def write_perf(arm: str, tpot: float) -> None:
    os.makedirs(os.path.join(arm, "perf"), exist_ok=True)
    for r in range(1, 5):
        with open(os.path.join(arm, "perf", f"bs1_r{r}.json"), "w") as f:
            json.dump({"median_tpot_ms": tpot + 0.1 * r, "mean_tpot_ms": tpot}, f)


def write_log(arm: str, dead: bool) -> None:
    line_ok = ("INFO kt_offload host-func over 5.0s: 6000 calls across 40 layers, "
               "per-layer min/max 150/150; uniform\n")
    line_dead = ("INFO kt_offload host-func over 5.0s: 1500 calls across 40 layers, "
                 "per-layer min/max 0/150; 30 layers never ran: [10, 11] -- their CPU-expert work is "
                 "NOT in the graph (a replay runs no Python, so a live layer's counter must grow every step)\n")
    with open(os.path.join(arm, "serve.log"), "w") as f:
        f.write("INFO starting\n" + (line_dead if dead else line_ok) * 3)
    with open(os.path.join(arm, "n_10701x.txt"), "w") as f:
        f.write("0\n")
    with open(os.path.join(arm, "dmesg_new.txt"), "w") as f:
        f.write("")


def main() -> int:
    run, mode = sys.argv[1], sys.argv[2]
    # arm A: 40 exposed gaps/token; arm B: 40 gaps shortened by 80 x 62.5 us / 40
    # per gap, i.e. the same count with each gap 125 us shorter -> delta 5.0 ms
    for name, gaps, extra, tpot in (("A", 40, 125.0, 39.8), ("B", 40, 0.0, 34.8)):
        arm = os.path.join(run, f"arm{name}")
        os.makedirs(arm, exist_ok=True)
        write_csv(os.path.join(arm, "prof_csv", "kernel_details.csv"), gaps, extra)
        write_answers(os.path.join(arm, "gsm8k50", "output-rs0.jsonl"),
                      mode if name == "B" else ("emptyfield" if mode == "emptyfield" else "none"))
        write_perf(arm, tpot if mode != "nogain" else 39.8)
        write_log(arm, dead=(mode == "deadlayers" and name == "B"))
    if mode == "nogain":  # arm B saves almost nothing: the model does not hold
        arm = os.path.join(run, "armB")
        write_csv(os.path.join(arm, "prof_csv", "kernel_details.csv"), 40, 120.0)
    print(f"fixture {mode} -> {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
