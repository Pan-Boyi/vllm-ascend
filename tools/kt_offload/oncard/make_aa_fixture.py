#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a fake A/A run so aa_report.py's decision table can be exercised.

One fixture per branch of the table, plus the branches that must REFUSE. A
judge whose three outcomes have never each been produced once is a judge with
one tested outcome.

Modes:
  batchcomp   c8 differs, c1 identical        -> "cause = batch composition"
  bothdiffer  c8 and c1 both differ           -> "another source, not packed"
  bothsame    neither differs                 -> "packed changed the numbers"
  inverted    c1 differs, c8 identical        -> must refuse (no hypothesis)
  missing_c1  no concurrency-1 eval at all    -> must refuse (cannot decide)
  emptyfield  generations empty               -> must refuse (nothing compared)
  otherquestions  the two arms answer different ids -> must refuse
"""

from __future__ import annotations

import json
import os
import sys

N = 50


def write_eval(path: str, differ_ids: set[int], ids: range | None = None) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "output-rs0.jsonl"), "w") as f:
        for i in ids if ids is not None else range(N):
            text = f"Step by step, the answer is \\boxed{{{i}}}"
            if i in differ_ids:
                text = text.replace("Step by step", "Let us see")
            f.write(json.dumps({"id": f"gsm8k-{i}", "generation": text}) + "\n")


def write_empty(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "output-rs0.jsonl"), "w") as f:
        for i in range(N):
            f.write(json.dumps({"id": f"gsm8k-{i}", "generation": ""}) + "\n")



def _bf():
    """The owning module, by path: fixtures read its constants, never copy them."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "build_fingerprint.py"), os.path.join(here, "..", "build_fingerprint.py")):
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("build_fingerprint", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("build_fingerprint.py not found: a fixture cannot guess its constants")


def _count_rule_id() -> str:
    return _bf().COUNT_RULE_ID


def _schema() -> str:
    """Read the current schema from build_fingerprint.py."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "build_fingerprint.py"), os.path.join(here, "..", "build_fingerprint.py")):
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("build_fingerprint", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.SCHEMA
    raise SystemExit("build_fingerprint.py not found: a fixture cannot guess the schema")


def fingerprint(arm: str) -> None:
    fp = {
        "schema": _schema(),
        "label": "aa-fixture",
        "so_primary_md5": "a" * 32,
        "so_primary_md5_short": "a" * 12,
        "vendors_sha256": "b" * 64,
        "vendors_files": 759,
        "vendors_bytes": 118_000_000,
        "vendors_ok": True,
        "vendors": {"count_rule": {"id": _count_rule_id()}},
        "kt_wheel_sha256": "c" * 64,
        "trees": {"vllm_ascend": {"head": "d" * 40}, "vllm": {"head": "e" * 40}},
        "backfilled": False,
        "fixture": True,
    }
    with open(os.path.join(arm, "build_fingerprint.json"), "w") as f:
        json.dump(fp, f, indent=1)


def main() -> int:
    run, mode = sys.argv[1], sys.argv[2]
    differ = set(range(31))  # 31/50, the number the real A/B produced
    a1, a2 = os.path.join(run, "armA1"), os.path.join(run, "armA2")
    for arm in (a1, a2):
        os.makedirs(arm, exist_ok=True)
        fingerprint(arm)
    # explicit per mode: a clever expression here once produced an "inverted"
    # fixture that was not inverted, and the branch it was meant to test read
    # as covered
    plans = {  # mode -> (A2 c8 differs, A2 c1 differs, A1 within-process differs)
        "batchcomp": (True, False, True),
        "bothdiffer": (True, True, True),
        "bothsame": (False, False, False),
        "inverted": (False, True, False),
        "missing_c1": (True, False, True),
    }
    if mode not in plans and mode not in ("emptyfield", "otherquestions"):
        print(f"unknown mode {mode}")
        return 2
    c8_differs, c1_differs, within_differs = plans.get(mode, (False, False, False))

    if mode == "emptyfield":
        for arm in (a1, a2):
            for key in ("gsm8k50", "gsm8k50_c8b", "gsm8k50_c1"):
                write_empty(os.path.join(arm, key))
        print(f"fixture {mode} -> {run}")
        return 0

    if mode == "otherquestions":
        write_eval(os.path.join(a1, "gsm8k50"), set(), ids=range(N))
        write_eval(os.path.join(a2, "gsm8k50"), set(), ids=range(1, N + 1))
        for arm in (a1, a2):
            write_eval(os.path.join(arm, "gsm8k50_c8b"), set())
            write_eval(os.path.join(arm, "gsm8k50_c1"), set())
        print(f"fixture {mode} -> {run}")
        return 0

    write_eval(os.path.join(a1, "gsm8k50"), set())
    write_eval(os.path.join(a2, "gsm8k50"), differ if c8_differs else set())
    write_eval(os.path.join(a1, "gsm8k50_c8b"), differ if within_differs else set())
    write_eval(os.path.join(a2, "gsm8k50_c8b"), differ if (c8_differs or within_differs) else set())
    if mode != "missing_c1":
        write_eval(os.path.join(a1, "gsm8k50_c1"), set())
        write_eval(os.path.join(a2, "gsm8k50_c1"), differ if c1_differs else set())
    print(f"fixture {mode} -> {run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
