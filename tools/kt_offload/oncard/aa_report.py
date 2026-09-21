#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the service-level bit-identity criterion have any discriminating power?

The packed A/B differed in 31 of 50 greedy answers. That reading has two
readings of its own, and until they are separated it cannot be used either
way:

  * the criterion is blunt -- two runs of the SAME configuration already
    differ, because 8 concurrent requests make batch composition vary and the
    reductions inside a batch are not associative;

It also measures something the A/B needed and never had: with both arms
identical, their TPOT difference IS this machine's drift over those ~20
minutes (it has been measured 2.5x slower on one commit 11 hours apart). That
number is printed as a noise floor, and any later A/B difference has to clear
it.
  * or packed really changed the numbers.

So: run the same arm twice, changing nothing, with the same harness (the
runner uses one eval function for both modes, so a parameter cannot drift),
and add a concurrency-1 arm -- without it, "batch composition did it" is a
story rather than a measurement.

    concurrency 8 A/A differs AND concurrency 1 A/A bit-identical
        => cause = batch composition; the service-level criterion has NO
           discriminating power, and bit-identity must be judged on the
           layer-level cases (t1/t2/t3) instead
    both differ
        => not batch composition; another source, to be found, and NOT to be
           attributed to packed
    both bit-identical
        => the A/B's 31/50 was packed changing the numbers

This script prints that table with the measured numbers filled in, so nobody
has to assemble the conclusion by hand inside a booked window.

Usage: aa_report.py <aa run dir> [--ab-run <packed A/B run dir>]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

EVALS = (
    ("gsm8k50", "concurrency 8", "the A/B's own parameters, in the A/B's position"),
    ("gsm8k50_c8b", "concurrency 8 (2nd eval, same service)", "same process: isolates batch composition"),
    ("gsm8k50_c1", "concurrency 1", "no concurrency at all"),
)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPORT = _load_module("packed_ab_report", os.path.join(HERE, "packed_ab_report.py"))
for candidate in (os.path.join(HERE, "build_fingerprint.py"), os.path.join(HERE, "..", "build_fingerprint.py")):
    if os.path.isfile(candidate):
        FP = _load_module("build_fingerprint", candidate)
        break
else:  # pragma: no cover - the tool is shipped next to this file
    FP = None


def answers(path: str) -> dict[str, str]:
    """Generations by question id, from one eval directory."""
    out: dict[str, str] = {}
    jsonl = os.path.join(path, "output-rs0.jsonl")
    if not os.path.exists(jsonl):
        return out
    with open(jsonl) as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row.get("generation", "")
    return out


def compare(a: dict[str, str], b: dict[str, str]) -> dict:
    cmp_ = REPORT.compare_answers(a, b)
    cmp_["same_questions"] = sorted(a) == sorted(b)
    return cmp_


def line(label: str, cmp_: dict) -> str:
    if not cmp_["n"]:
        return f"  {label:<44} no data"
    flags = []
    if not cmp_["field_has_content"]:
        flags.append("EMPTY FIELDS")
    if not cmp_["positive_control_detects_one_char"]:
        flags.append("NO DISCRIMINATING POWER IN THE COMPARISON ITSELF")
    if not cmp_["same_questions"]:
        flags.append("DIFFERENT QUESTION SETS")
    tail = ("   ⚠ " + "; ".join(flags)) if flags else ""
    return f"  {label:<44} {cmp_['identical']:>3}/{cmp_['n']:<3} identical, {len(cmp_['differ']):>3} differ{tail}"


def usable(cmp_: dict) -> bool:
    """A comparison only counts when it could have failed and could have passed."""
    return bool(
        cmp_["n"] and cmp_["field_has_content"] and cmp_["positive_control_detects_one_char"] and cmp_["same_questions"]
    )


def main() -> int:
    argv = sys.argv[1:]
    ab_run = None
    if "--ab-run" in argv:
        i = argv.index("--ab-run")
        ab_run = argv[i + 1]
        del argv[i : i + 2]
    if len(argv) != 1:
        print(__doc__)
        return 2
    run = argv[0]
    arms = {name: os.path.join(run, f"arm{name}") for name in ("A1", "A2")}
    missing = [n for n, p in arms.items() if not os.path.isdir(p)]
    if missing:
        print(f"❌ arm directories missing: {missing} (an arm did not run)")
        return 1

    print("=== 0. build fingerprint (both arms must be the same build; that is the whole point of an A/A)")
    gate = REPORT.build_gate(arms) if hasattr(REPORT, "build_gate") else {"ok": False, "why": "no gate"}
    for name in ("A1", "A2"):
        print(f"  arm {name}: {gate.get('lines', {}).get(name, 'no fingerprint')}")
    for reason in gate.get("reasons", []):
        print(f"  {reason}")
    if not gate["ok"]:
        print(f"  ⛔ REFUSING to draw a conclusion: {gate['why']}")
        return 1
    print("  ⇒ same build")

    print("\n=== 1. cross-start comparisons (two service starts, nothing else changed)")
    cross = {}
    for key, label, why in EVALS:
        cmp_ = compare(answers(os.path.join(arms["A1"], key)), answers(os.path.join(arms["A2"], key)))
        cross[key] = cmp_
        print(line(f"{label}  [{why}]", cmp_))

    print("\n=== 2. within-start comparisons (same process, two evals): batch composition alone")
    within = {}
    for name in ("A1", "A2"):
        cmp_ = compare(answers(os.path.join(arms[name], "gsm8k50")), answers(os.path.join(arms[name], "gsm8k50_c8b")))
        within[name] = cmp_
        print(line(f"arm {name}: concurrency 8, eval 1 vs eval 2", cmp_))

    print("\n=== 2.5 the machine's own drift, from these two arms (same configuration, ~20 min apart)")
    tpot = {name: REPORT.load_tpot(arms[name]) for name in ("A1", "A2")}
    for name in ("A1", "A2"):
        t = tpot[name]
        print(f"  arm {name}: " + (f"median {t['median']:.2f} ms  mean {t['mean']:.2f} +- {t['stdev']:.2f}  n={t['n']}"
                                   if t["n"] else "no TPOT samples"))
    if tpot["A1"]["n"] and tpot["A2"]["n"]:
        floor = abs(tpot["A1"]["median"] - tpot["A2"]["median"])
        print(f"  ⇒ NOISE FLOOR = {floor:.2f} ms/token. Two runs of the same configuration differ by this much,")
        print("    so no A/B difference smaller than it means anything -- including the packed A/B's")
        print("    -0.74 ms median, which was measured with the arms 40 minutes apart.")
    else:
        floor = None
        print("  ⇒ no floor measurable (an arm has no perf samples); run aa mode without skipping perf")

    c8, c1 = cross.get("gsm8k50", {}), cross.get("gsm8k50_c1", {})
    print("\n=== 3. verdict")
    if not usable(c8) or not usable(c1):
        which = [k for k, v in (("concurrency 8", c8), ("concurrency 1", c1)) if not usable(v)]
        print(f"  ⛔ cannot decide: the {', '.join(which)} comparison is not usable "
              "(missing data, empty fields, different question sets, or a comparison that cannot fail).")
        print("     Fix the run, not the conclusion: with AA_EVALS=c8,c8b,c1 both arms must produce all three.")
        return 1
    c8_differs = len(c8["differ"]) > 0
    c1_differs = len(c1["differ"]) > 0
    if c8_differs and not c1_differs:
        verdict = (
            "cause = BATCH COMPOSITION. The service-level bit-identity criterion has NO discriminating "
            "power at concurrency 8: two runs of one configuration already differ. Bit-identity must be "
            "judged on the layer-level cases (t1/t2/t3), and the A/B's 31/50 says nothing about packed."
        )
        rc = 0
    elif c8_differs and c1_differs:
        verdict = (
            "NOT batch composition: even concurrency 1 differs across two starts of the same build and "
            "configuration. There is another source (per-start state) and it must be found; the A/B's "
            "31/50 may NOT be attributed to packed until it is."
        )
        rc = 0
    elif not c8_differs and not c1_differs:
        verdict = (
            "both arms are bit-identical, so this criterion DOES discriminate at both concurrencies "
            "=> the A/B's 31/50 was packed changing the numbers. packed_transfer is a correctness "
            "question, not a performance one."
        )
        rc = 0
    else:
        verdict = (
            "concurrency 1 differs while concurrency 8 does not -- the opposite of every hypothesis on the "
            "table. Do not average this away: re-run before interpreting."
        )
        rc = 1
    print(f"  measured: concurrency 8 {len(c8['differ'])}/{c8['n']} differ, "
          f"concurrency 1 {len(c1['differ'])}/{c1['n']} differ")
    for name, cmp_ in within.items():
        if usable(cmp_):
            print(f"  supporting: arm {name} within one process {len(cmp_['differ'])}/{cmp_['n']} differ")
    print(f"  ⇒ {verdict}")

    if ab_run:
        print("\n=== 4. the A/B reading this control was built for")
        ab_arms = {n: os.path.join(ab_run, f"arm{n}") for n in ("A", "B")}
        if all(os.path.isdir(p) for p in ab_arms.values()):
            ab_cmp = compare(answers(os.path.join(ab_arms["A"], "gsm8k50")),
                             answers(os.path.join(ab_arms["B"], "gsm8k50")))
            print(line("packed A vs packed B (concurrency 8)", ab_cmp))
            if FP is not None:
                ok, reasons = FP.same_build(FP.load(ab_arms["A"]), FP.load(arms["A1"]))
                state = "same build as this control" if ok else "build match NOT proven"
                print(f"  provenance: {state}")
                for r in reasons:
                    print(f"    {r}")
                if not ok:
                    print("    (quoted for context only -- no number is subtracted across this boundary)")
        else:
            print(f"  A/B run not readable at {ab_run}")

    with open(os.path.join(run, "aa_result.json"), "w") as f:
        json.dump(
            {
                "cross_start": {k: {"identical": v["identical"], "n": v["n"], "differ": len(v["differ"])}
                                for k, v in cross.items()},
                "within_start": {k: {"identical": v["identical"], "n": v["n"], "differ": len(v["differ"])}
                                 for k, v in within.items()},
                "verdict": verdict,
                "tpot_ms": {k: {kk: v.get(kk) for kk in ("median", "mean", "stdev", "n")} for k, v in tpot.items()},
                "noise_floor_ms": floor,
                "criterion_discriminates_at_c8": not c8_differs,
                "criterion_discriminates_at_c1": not c1_differs,
            },
            f,
            indent=1,
        )
    return rc


if __name__ == "__main__":
    sys.exit(main())
