#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Judge a packed_transfer A/B run (arms A = off, B = on).

Six readings, in the order they decide anything:

  1. bit-identity of every answer, WITH A POSITIVE CONTROL. The comparison
     field has to be shown to discriminate first: on 2026-09-20 a comparison
     against a field that did not exist read as "0 differences" -- a green
     light from an empty set. So the run also compares arm B against arm B
     with one character changed, and the report fails if that does not show up.
  2. TPOT, n>=REPEAT: median, every sample, and the spread. Never one number.
  3. the gap table: device busy (union of intervals, never the sum -- parallel
     streams would be counted twice), gap ms/token, >100us gaps per token, and
     the median length of those gaps.
  4. the per-operation cost C, solved from the arms: packed removes 2 of 5
     host-device operations per layer, so 40 layers x 2 operations per token.
         C = delta(>100us gap ms/token) / 80
     Three outcomes, and the third is the one to watch:
       C ~= 62 us     the inferred model holds, form C is worth implementing
       C >= 300 us    one extra callback would eat the GMM window: form C dies
       delta does not match 80 x C  ->  "fixed cost x operation count" is the
                      wrong model and the 7.5 ms attribution has to be redone
                      per operation kind (D2H / H2D / host-func separately)
  5. the per-layer host-func counts the service logged (only an in-graph node
     increments them, so a layer that never grows has no CPU work in the graph).
  6. 10701x and the dmesg counts.

Usage: packed_ab_report.py <run dir>   (exit 1 if a required reading fails)
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import statistics
import sys

GAP_US = 100.0
OPS_SAVED_PER_TOKEN = 80  # 40 layers x (5 - 3) host-device operations
# 🔴 RETIRED (2026-09-20, profile_ops.py on the armA profile): 62.5 us was
# 7.5 ms / 120 operations, and 37.5 us the same aggregate over 200. The
# profile measures 1.168 us of device time per memcpy, so neither is a price,
# and a number built from one operation count cannot arbitrate between counts.
# Kept only so this report can say what the old figure was.
RETIRED_PRICE_US = 62.5




def arm_dir(run: str, name: str, round_no: int) -> str:
    """Round 1 keeps the historical name; later rounds carry _rN."""
    return os.path.join(run, f"arm{name}" if round_no == 1 else f"arm{name}_r{round_no}")


def discover_rounds(run: str) -> set[int]:
    rounds = set()
    for entry in os.listdir(run) if os.path.isdir(run) else []:
        m = re.fullmatch(r"arm([AB])(?:_r(\d+))?", entry)
        if m:
            rounds.add(int(m.group(2) or 1))
    return rounds or {1}


def drift_floor(arm_series: dict) -> dict | None:
    """How much the SAME configuration moved between rounds.

    This machine has been measured 2.5x slower on one commit 11 hours apart, so
    a difference smaller than the same-arm spread says nothing about the switch.
    The floor is the largest same-arm swing seen, not an average: it is the
    thing a claimed benefit has to clear.
    """
    swings = {}
    for name, series in arm_series.items():
        if len(series) >= 2:
            values = [series[r] for r in sorted(series)]
            swings[name] = max(values) - min(values)
    if not swings:
        return None
    worst = max(swings.values())
    detail = ", ".join(f"{n} {v:+.2f}" for n, v in sorted(swings.items()))
    return {"value": worst, "per_arm": swings, "line": f"{detail} ms/token -> floor {worst:.2f} ms/token"}

# --------------------------------------------------------- build fingerprint ---


def _fingerprint_module():
    """The tool lives next to this file in a run snapshot, one level up in the
    tree. Import by path so both layouts work."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "build_fingerprint.py"),
                      os.path.join(here, "..", "build_fingerprint.py")):
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("build_fingerprint", candidate)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def build_gate(arm_dirs: dict) -> dict:
    """May these two arms be subtracted at all?

    A difference between two builds is not a weaker result, it is a different
    question -- so this refuses rather than warning. It also refuses when a
    fingerprint is missing: "no evidence of a mismatch" is not evidence of a
    match.
    """
    fp_mod = _fingerprint_module()
    if fp_mod is None:
        return {"ok": False, "why": "build_fingerprint.py not found next to this script; cannot check builds",
                "fingerprints": {}}
    fps = {name: fp_mod.load(path) for name, path in arm_dirs.items()}
    lines = {name: fp_mod.short(fp) for name, fp in fps.items()}
    # by position, not by the names "A"/"B": the same gate judges the A/A
    # control, whose arms are A1/A2 (a gate that only knows two literal names
    # passes silently on every other pair)
    names = list(arm_dirs)
    if len(names) != 2:
        return {"ok": False, "why": f"expected exactly two arms, got {names}", "lines": lines,
                "fingerprints": fps, "reasons": [f"arms: {names}"]}
    ok, reasons = fp_mod.same_build(fps[names[0]], fps[names[1]])
    return {"ok": ok, "why": "; ".join(reasons) if reasons else "same build", "lines": lines,
            "fingerprints": fps, "reasons": reasons}

# ---------------------------------------------------------------- answers ---


def load_answers(arm: str) -> dict[str, str]:
    path = os.path.join(arm, "gsm8k50", "output-rs0.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["id"]] = row.get("generation", "")
    return out


def compare_answers(a: dict[str, str], b: dict[str, str]) -> dict:
    """Bit-identity plus the two properties that make it mean anything."""
    ids = sorted(set(a) | set(b))
    differ = [i for i in ids if a.get(i) != b.get(i)]
    nonempty = sum(1 for i in ids if (a.get(i) or "").strip())
    # positive control: one character changed must be reported
    control_src = next((i for i in ids if a.get(i)), None)
    control_ok = False
    if control_src is not None:
        mutated = dict(b)
        text = mutated.get(control_src) or a[control_src]
        mutated[control_src] = ("X" if text[:1] != "X" else "Y") + text[1:]
        control_ok = bool(compare_raw(a, mutated))
    return {
        "n": len(ids),
        "differ": differ,
        "identical": len(ids) - len(differ),
        "nonempty": nonempty,
        "field_has_content": nonempty == len(ids) and len(ids) > 0,
        "positive_control_detects_one_char": control_ok,
    }


def compare_raw(a: dict[str, str], b: dict[str, str]) -> list[str]:
    return [i for i in sorted(set(a) | set(b)) if a.get(i) != b.get(i)]


# ------------------------------------------------------------------- perf ---


def load_tpot(arm: str) -> dict:
    perf = os.path.join(arm, "perf")
    samples = []
    for name in sorted(os.listdir(perf)) if os.path.isdir(perf) else []:
        if not name.endswith(".json") or "warmup" in name or not name.startswith("bs1_r"):
            continue
        try:
            with open(os.path.join(perf, name)) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        for key in ("median_tpot_ms", "mean_tpot_ms"):
            if key in data:
                samples.append(float(data[key]))
                break
    if not samples:
        return {"samples": [], "n": 0}
    return {
        "samples": samples,
        "n": len(samples),
        "median": statistics.median(samples),
        "mean": statistics.mean(samples),
        "stdev": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min": min(samples),
        "max": max(samples),
    }


# -------------------------------------------------------------- gap table ---


def gap_table(arm: str) -> dict:
    csv_path = os.path.join(arm, "prof_csv", "kernel_details.csv")
    if not os.path.exists(csv_path):
        return {}
    intervals, steps = [], 0
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                start = float(row["Start Time(us)"].strip())
                dur = float(row["Duration(us)"])
            except (KeyError, ValueError):
                continue
            intervals.append((start, start + dur))
            if row.get("Type") == "MoeInitRoutingV3":
                steps += 1
    if not intervals:
        return {}
    intervals.sort()
    busy, gaps = 0.0, []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            gaps.append(s - cur_e)
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    span = intervals[-1][1] - intervals[0][0]
    big = [g for g in gaps if g >= GAP_US]
    tokens = steps / 40 if steps else 0  # one MoeInitRoutingV3 per layer per step
    out = {
        "tokens": tokens,
        "span_ms": span / 1000,
        "busy_share": busy / span if span else float("nan"),
        "busy_ms_per_token": busy / 1000 / tokens if tokens else float("nan"),
        "gap_ms_per_token": sum(gaps) / 1000 / tokens if tokens else float("nan"),
        "big_per_token": len(big) / tokens if tokens else float("nan"),
        "big_ms_per_token": sum(big) / 1000 / tokens if tokens else float("nan"),
        "big_median_us": statistics.median(big) if big else float("nan"),
    }
    return out


def solve_c(a: dict, b: dict, window_us: float | None = None) -> dict:
    """What the two arms actually moved, and the per-operation figure it implies.

    This is no longer a test of the "fixed cost x operation count" model: the
    profile retired that model (see RETIRED_PRICE_US). Delta is the result;
    the implied C is a derived number whose only remaining use is as one of
    the two independent sources for a host-func node's cost -- the other being
    t4's measured node delta. Agreeing is what makes either usable.
    """
    if not a or not b or math.isnan(a.get("big_ms_per_token", float("nan"))):
        return {"ok": False, "why": "a gap table is missing"}
    delta_ms = a["big_ms_per_token"] - b["big_ms_per_token"]
    c_us = delta_ms * 1000 / OPS_SAVED_PER_TOKEN
    if window_us is not None:
        verdict = (
            f"implied C = {c_us:.0f} us against a measured window of {window_us:.0f} us: "
            + ("one extra node would eat the window -> form C is not worth implementing"
               if c_us >= window_us
               else "an extra node still leaves part of the window -> form C stays a candidate, "
                    "decide on t4's own node measurement, not on this one")
        )
    else:
        verdict = (f"implied C = {c_us:.0f} us; no measured window given (--window-us, from t4 item 2), "
                   "so this cannot say whether an extra node eats it")
    return {
        "ok": True,
        "delta_big_ms_per_token": delta_ms,
        "C_us": c_us,
        "retired_price_us": RETIRED_PRICE_US,
        "verdict": verdict,
        "model_note": (
            "the 'fixed cost x operation count' model is retired (profile: 1.168 us of device time per "
            "memcpy); do not compare C against 62.5/37.5 us, compare it against t4's measured node cost"
        ),
    }


# ------------------------------------------------------- host-func counts ---

_HF = re.compile(r"kt_offload host-func over ([\d.]+)s: (\d+) calls across (\d+) layers, per-layer min/max (\d+)/(\d+)")


def host_func_report(arm: str) -> dict:
    log = os.path.join(arm, "serve.log")
    if not os.path.exists(log):
        return {"lines": 0}
    lines, never, uneven = 0, 0, 0
    last = ""
    with open(log, errors="replace") as f:
        for line in f:
            if not _HF.search(line):
                continue
            lines += 1
            last = line.strip()[-300:]
            if "never ran" in line:
                never += 1
            elif "uneven" in line:
                uneven += 1
    return {"lines": lines, "samples_with_dead_layers": never, "samples_uneven": uneven, "last": last}


# ----------------------------------------------------------------- report ---


def read_int(path: str) -> int:
    try:
        with open(path) as f:
            return int(f.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def main() -> int:
    argv = sys.argv[1:]
    window_us = None
    if "--window-us" in argv:  # t4 item 2, when it has been measured
        i = argv.index("--window-us")
        window_us = float(argv[i + 1])
        del argv[i : i + 2]
    if len(argv) != 1:
        print(__doc__)
        print("usage: packed_ab_report.py <run dir> [--window-us <t4 item 2, us>]")
        return 2
    run = argv[0]
    arms = {name: os.path.join(run, f"arm{name}") for name in ("A", "B")}
    missing = [n for n, p in arms.items() if not os.path.isdir(p)]
    if missing:
        print(f"❌ arm directories missing: {missing} (an arm did not run)")
        return 1

    fails: list[str] = []
    print("=== 0. build fingerprint (a difference across builds is not a result)")
    gate = build_gate(arms)
    for name in ("A", "B"):
        print(f"  arm {name}: {gate.get('lines', {}).get(name, 'no fingerprint')}")
    for reason in gate.get("reasons", []):
        print(f"  {reason}")
    if not gate["ok"]:
        print(f"  ⛔ REFUSING to report differences: {gate['why']}")
        print("     The two arms were not measured on the same build, so their difference answers no "
              "question anyone asked. Re-run both arms inside one build.")
        return 1
    print("  ⇒ same build: the arms may be compared")

    print("\n=== 1. answers, bit-identical (the criterion; TPOT decides nothing until this passes)")
    ans = {n: load_answers(p) for n, p in arms.items()}
    cmp_ = compare_answers(ans["A"], ans["B"])
    print(f"  {cmp_['identical']}/{cmp_['n']} identical, {len(cmp_['differ'])} differ: {cmp_['differ'][:8]}")
    print(f"  field non-empty for every answer: {cmp_['field_has_content']} ({cmp_['nonempty']}/{cmp_['n']})")
    print(f"  positive control (one character changed is reported): {cmp_['positive_control_detects_one_char']}")
    if not cmp_["n"]:
        fails.append("no answers to compare")
    if not cmp_["field_has_content"]:
        fails.append("the compared field is empty for some answers -- a comparison of nothing reads as agreement")
    if not cmp_["positive_control_detects_one_char"]:
        fails.append("the positive control did not fire: the comparison has no discriminating power")
    if cmp_["differ"]:
        fails.append(f"{len(cmp_['differ'])} answers differ between the arms")

    print("\n=== 2. TPOT (bs1), paired by round -- and the machine's own drift")
    rounds = discover_rounds(run)
    print(f"  rounds found: {sorted(rounds)}")
    per_round, arm_series = {}, {"A": {}, "B": {}}
    for r in sorted(rounds):
        row = {}
        for name in ("A", "B"):
            t = load_tpot(arm_dir(run, name, r))
            row[name] = t
            if t["n"]:
                arm_series[name][r] = t["median"]
        per_round[r] = row
        parts = []
        for name in ("A", "B"):
            t = row[name]
            parts.append(f"{name} {t['median']:.2f} (n={t['n']})" if t["n"] else f"{name} -")
        delta = (row["A"]["median"] - row["B"]["median"]) if row["A"]["n"] and row["B"]["n"] else None
        tail = f"  A-B = {delta:+.2f} ms/token" if delta is not None else "  A-B = n/a"
        print(f"  round {r}: " + "  ".join(parts) + tail)
        if not (row["A"]["n"] and row["B"]["n"]):
            fails.append(f"round {r} is missing an arm's TPOT samples")

    deltas = [
        per_round[r]["A"]["median"] - per_round[r]["B"]["median"]
        for r in sorted(rounds)
        if per_round[r]["A"]["n"] and per_round[r]["B"]["n"]
    ]
    floor = drift_floor(arm_series)
    if floor is None:
        print("  drift floor: NOT MEASURED (only one round per arm) -- a single-round difference cannot be "
              "told from this machine drifting; it has been seen 2.5x slower 11 hours apart")
    else:
        print(f"  drift floor (same arm, different rounds): {floor['line']}")
    if deltas:
        med = statistics.median(deltas)
        print(f"  paired A-B deltas: {[round(d, 2) for d in deltas]}  median {med:+.2f} ms/token")
        if floor is None and abs(med) > 0.05:
            # a difference with no yardstick is the dangerous case: it reads
            # like a benefit. No floor and no difference claims nothing, so it
            # is only a note.
            fails.append(
                f"a {med:+.2f} ms/token difference was measured with no drift floor (one round per arm): "
                "re-run with ROUNDS>=2 -- this machine has been 2.5x slower 11 hours apart, so the "
                "difference cannot be told from the clock"
            )
        if floor is not None:
            verdict = ("EXCEEDS the drift floor -> a real difference"
                       if abs(med) > floor["value"]
                       else "WITHIN the drift floor -> not distinguishable from the machine's own drift")
            print(f"  ⇒ |median delta| {abs(med):.2f} vs floor {floor['value']:.2f}: {verdict}")
    else:
        print("  no paired deltas at all")
        fails.append("no round produced both arms' TPOT")

    print("\n=== 3. gap table (union busy; the sum would double-count parallel streams)")
    gt = {n: gap_table(p) for n, p in arms.items()}
    for n in ("A", "B"):
        g = gt[n]
        if not g:
            print(f"  arm {n}: no kernel_details.csv")
            fails.append(f"arm {n} has no profile csv")
            continue
        print(f"  arm {n}: tokens {g['tokens']:.0f}  busy {100 * g['busy_share']:.1f}%  "
              f"busy {g['busy_ms_per_token']:.2f} ms/tok  gaps {g['gap_ms_per_token']:.2f} ms/tok  "
              f">{GAP_US:.0f}us {g['big_per_token']:.2f}/tok totalling {g['big_ms_per_token']:.2f} ms "
              f"(median {g['big_median_us']:.0f} us)")

    print("\n=== 4. per-operation cost C, solved from the arms (model retired; see the note)")
    c = solve_c(gt.get("A", {}), gt.get("B", {}), window_us)
    if not c["ok"]:
        print(f"  not solvable: {c['why']}")
        fails.append("C could not be solved")
    else:
        print(f"  delta(>100us ms/token) = {c['delta_big_ms_per_token']:+.2f} ms over {OPS_SAVED_PER_TOKEN} operations")
        print(f"  implied C = {c['C_us']:.1f} us   (the retired figure was {c['retired_price_us']}, do not compare)")
        print(f"  ⇒ {c['verdict']}")
        print(f"  ⇒ {c['model_note']}")

    print("\n=== 5. per-layer host-func counts (only an in-graph node increments them)")
    for n in ("A", "B"):
        h = host_func_report(arms[n])
        print(f"  arm {n}: {h.get('lines', 0)} samples, {h.get('samples_with_dead_layers', 0)} with layers that "
              f"never ran, {h.get('samples_uneven', 0)} uneven")
        if h.get("last"):
            print(f"     last: {h['last']}")
        if h.get("lines", 0) == 0:
            print("     (sampling was off or the service logged nothing -- set HFLOG)")

    print("\n=== 6. anomalies")
    for n in ("A", "B"):
        arm = arms[n]
        n10701 = read_int(os.path.join(arm, "n_10701x.txt"))
        dmesg_new = 0
        p = os.path.join(arm, "dmesg_new.txt")
        if os.path.exists(p):
            with open(p, errors="replace") as f:
                dmesg_new = sum(1 for _ in f)
        anomaly = os.path.getsize(os.path.join(arm, "anomaly.txt")) if os.path.exists(
            os.path.join(arm, "anomaly.txt")) else 0
        print(f"  arm {n}: 10701x {n10701}, new dmesg lines {dmesg_new}, anomaly markers {'yes' if anomaly else 'no'}")
        if n10701:
            fails.append(f"arm {n} logged 10701x")
        if anomaly:
            fails.append(f"arm {n} hit an anomaly marker")

    print("\n=== verdict")
    if fails:
        for f in fails:
            print(f"  ❌ {f}")
        print("  FAIL")
        return 1
    print("  ✅ bit-identical, both arms measured; read C above before deciding on form C")
    return 0


if __name__ == "__main__":
    sys.exit(main())
