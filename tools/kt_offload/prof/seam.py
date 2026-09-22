#!/usr/bin/env python3
"""Offline A..F decomposition of the per-layer host-func seam.

Reads the ``[kt-seam]`` records a running service writes when
``VLLM_ASCEND_DSV41_KT_SEAM_TIMING`` is on (vllm_ascend/kt_offload/
seam_timing.py), and optionally the two other instruments of §4.18:

  --ktphase FILE   the JSON ktphase.py writes as its 4th argument
                   ({"layer_median_us": {...}}): kt's own CPU compute, D.
  --nw-csv FILE    a profile's task_time.csv: NOTIFY_WAIT per layer, sliced
                   and *checked* (see "Aligning NOTIFY_WAIT" below).
  --nw-median-us X one number instead of the csv (e.g. 353.6 from §4.18).

The bookkeeping this prints:

  A  device reaches the host-func node -> the callback starts running
  B  pre        callback entry -> Python pre-checks done
  C  kt_pre + submit + between   task build and handing it to the worker
  D  kt compute (from --ktphase; the seam records cannot see inside sync)
  E  sync - D   worker wakeup + completion (cross-NUMA reduce, write-back)
  F  post       trailing Python -> callback returns, then the notify to device

Only B, C, sync, F are measured directly; the rest are differences:

  E            = sync - D                        (needs --ktphase)
  A + F_device = NOTIFY_WAIT - cb                (needs a NOTIFY_WAIT source)

which is why both must come from the SAME service and the same steps -- see
tools/kt_offload/oncard/RUNBOOK_seam.md. Every difference is printed with the
inputs it was made of, and a gap that does not close is printed as a gap
rather than folded into a neighbour.

Windowing (--since/--until/--bytes)
-----------------------------------
``[kt-seam]`` is written for the whole life of the service while a profile
covers only its last few seconds. Without a window the table mixes the two.
Each record covers ``interval_s`` seconds *ending* at its log timestamp; a
record is kept when at least ``--window-min-overlap`` of that interval falls
inside the window. The default is the whole log, and the number of records
actually used is always printed.

What the reconciliation judges (and what it cannot)
---------------------------------------------------
Inside one callback the segments are consecutive differences of one clock
(seam_timing.record), so **per callback the parts add up to cb exactly, by
construction** -- there is nothing there to check. What the records carry is
medians and p90s, and *medians are not additive*: summing the per-segment
medians and comparing against the median of cb compares two different
quantities, and on right-skewed segments the difference is systematically
positive. That comparison used to be this tool's red criterion; it is now
printed as ``resid_med`` and labelled "not a criterion".

The real failure mode this tool can see is a record whose middle segments were
not filled (the CPUInfer probe did not run for that callback) mixed into a run
reported as ``split``: those rows store zeros for kt_pre..kt_tail while ``kt``
and ``cb`` stay whole. So the criteria are necessary conditions of the
per-callback identity, each of them valid without assuming additivity:

  containment  every part's median and p90 must be <= cb's (a part is inside
               the callback, per callback -- hence at every quantile), and
               with ``split`` every middle part must be <= kt's;
  union bound  med(cb) <= p90(pre) + p90(kt) + p90(post): P(sum > sum of the
               p90s) <= 3 * 10% < 50%, so the median cannot exceed it. With
               ``split``, med(kt) <= sum of the five middle p90s (5 * 10%).
  non-zero     with ``split``, no middle segment may have a zero median --
               but ONLY as a stand-in for ``n_unsplit`` on records that do not
               carry it, and never where the split is ours rather than the
               probe's: on the packed and side-stream paths ``between`` and
               ``kt_tail`` are zero by construction, so this proxy is skipped
               whenever ``n_unsplit`` is present (see check_layer).
  n_unsplit    the record side reports, per layer and sample, how many rows of
               a split run had no middle (``n_unsplit``) and the per-callback
               residual of largest magnitude (``resid_us.max``); both are exact
               and are used when present (records from before 2026-09-23 do not
               carry them -- then only the necessary conditions above apply).

Resolution, stated because it is the point of an instrument: the union bound
only rejects a hole bigger than the slack ``sum(p90) - med(cb)``, which is
printed per layer, and only rejects unsplit contamination when it is a large
fraction of the callbacks. A single callback with an inflated residual is
invisible to any median/p90 record and needs the record side to count it.

Aligning NOTIFY_WAIT (--nw-csv)
-------------------------------
A step of this build contains 42 NOTIFY_WAIT, and the first 2 are not ours, so
position j is layer j-2. Feeding task_time.csv straight in shifts every layer
by two and still prints a full table. So the slicing is now part of the tool
(``--nw-skip``, default 2; only steps with exactly ``layers + skip`` waits are
used) and it carries its own check: the Spearman correlation between the
per-layer NOTIFY_WAIT medians and the per-layer cb medians must be at least
``--nw-align-min`` at the chosen shift and beat every other feasible shift by
``--nw-align-margin``. If it does not, the NW / A+F_dev columns are refused --
a misaligned number is worse than no number.

Splitting A from F (--nw-csv + ``[kt-seam-rt]`` lines, §4.22 P0)
---------------------------------------------------------------
When the log carries ``[kt-seam-rt]`` lines (absolute CLOCK_REALTIME ns of each
callback's entry and exit), the same task_time.csv gives A and F separately,
per callback, with no position slicing at all -- the join is by time:

  a HOSTFUNC_CALLBACK and the NOTIFY_WAIT that follows it on the same stream
  form one pair (in this build they start at the same ts); a callback belongs
  to the pair whose [HF.ts, NW.ts + NW.dur] holds its midpoint;
  A = entry_realtime - HF.ts        F = (NW.ts + NW.dur) - exit_realtime

The profile's ``task_start(us)`` is CLOCK_REALTIME microseconds (checked on
the box), read here as exact decimal, not float: at 1.79e15 us a double only
resolves 0.25 us.

The two columns are printed only if all of these hold, otherwise they are
refused and the tool returns 1:
  * callbacks fall inside the profile's span at all (a clock mix-up -- e.g.
    perf_counter written where CLOCK_REALTIME was meant -- lands nowhere);
  * at most --af-max-unmatched of those are outside every pair, and no pair
    takes two callbacks;
  * per layer med(A) > 0 and med(F) > 0, and at most --af-max-nonpos of the
    callbacks have A <= 0 or F <= 0;
  * per layer |med((A + F) - (NW.dur - cb))| < --af-tol-us (default 5): the
    new split must add back up to the existing NOTIFY_WAIT - T_cb column. A
    realtime stamp taken in the wrong place (inside kt instead of at entry)
    moves A by the part of cb it skipped, and this is where it shows.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import json
import re
import statistics as st
import sys
from collections import defaultdict

REC = re.compile(r"\[kt-seam\] (\{.*\})")
REC_RT = re.compile(r"\[kt-seam-rt\] (\{.*\})")
# vllm's log prefix: "INFO 09-22 15:01:51 [seam_timing.py:377]".
STAMP = re.compile(r"\b(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})\b")

SEGMENTS = ("pre", "kt_pre", "submit", "between", "sync", "kt_tail", "post", "kt", "cb", "gap")
ADDITIVE = ("pre", "kt_pre", "submit", "between", "sync", "kt_tail", "post")
ADDITIVE_LUMPED = ("pre", "kt", "post")
MIDDLE = ("kt_pre", "submit", "between", "sync", "kt_tail")


# --------------------------------------------------------------------- parsing


def parse_records(text: str) -> tuple[list[dict], int]:
    """Every ``[kt-seam]`` record in a log, and how many markers did not parse.

    Records share stdout with other writers (kt's own phase timing prints from
    C++), so a line can be cut in half; that is counted, never guessed at.

    Each parsed record gets a ``_stamp`` field: the ``MM-DD HH:MM:SS`` of its
    log line as (month, day, seconds-of-day), or None when the line had no
    prefix (a raw record in a test fixture). It is used only for windowing.
    """
    good, seen = [], 0
    for line in text.splitlines():
        m = REC.search(line)
        if not m:
            seen += line.count("[kt-seam]")
            continue
        seen += 1
        try:
            rec = json.loads(m.group(1))
        except ValueError:
            continue
        s = STAMP.search(line[: m.start()])
        rec["_stamp"] = (
            (int(s.group(1)), int(s.group(2)), int(s.group(3)) * 3600 + int(s.group(4)) * 60 + int(s.group(5)))
            if s
            else None
        )
        good.append(rec)
    return good, seen - len(good)


def parse_window_arg(value: str, year: int) -> tuple[int, int, int]:
    """``2026-09-22T15:01:44Z`` / ``09-22 15:01:44`` / ``15:01:44`` -> stamp key.

    The log prefix has no year and no zone, so neither has the comparison: the
    window must be given in the same clock the service logged in (both are the
    container's local time; the runbook writes ``date -u`` because that box
    runs on UTC).
    """
    v = value.strip().replace("T", " ").replace("Z", "").strip()
    for fmt, has_date in (("%Y-%m-%d %H:%M:%S", True), ("%m-%d %H:%M:%S", True), ("%H:%M:%S", False)):
        try:
            t = dt.datetime.strptime(v, fmt)
        except ValueError:
            continue
        if not has_date:
            today = dt.date.today()
            return (today.month, today.day, t.hour * 3600 + t.minute * 60 + t.second)
        return (t.month, t.day, t.hour * 3600 + t.minute * 60 + t.second)
    raise argparse.ArgumentTypeError(f"cannot read {value!r} as a time (try 2026-09-22T15:01:44Z)")


def _stamp_seconds(stamp: tuple[int, int, int], year: int) -> float:
    return dt.datetime(year, stamp[0], stamp[1]).timestamp() + stamp[2]


def select_window(records: list[dict], since, until, year: int, min_overlap: float = 0.5):
    """Records whose covered interval overlaps [since, until] enough.

    A record logged at t covers (t - interval_s, t]. Returns the kept records
    and a note describing what was done, so the table can print it.
    """
    if since is None and until is None:
        return records, f"window: whole log ({len(records)} records)"
    lo = _stamp_seconds(since, year) if since else float("-inf")
    hi = _stamp_seconds(until, year) if until else float("inf")
    kept, undated = [], 0
    for rec in records:
        stamp = rec.get("_stamp")
        if stamp is None:
            undated += 1
            continue
        end = _stamp_seconds(stamp, year)
        iv = float(rec.get("interval_s") or 0.0) or 1e-9
        overlap = max(0.0, min(end, hi) - max(end - iv, lo))
        if overlap / iv >= min_overlap:
            kept.append(rec)
    note = (
        f"window: {len(kept)}/{len(records)} records kept "
        f"(>= {min_overlap:.0%} of each record's interval inside the window)"
    )
    if undated:
        note += f"; {undated} record(s) had no log timestamp and were dropped"
    return kept, note


def read_text(path: str, byte_range: str | None) -> str:
    """The whole file, or the ``START:END`` byte slice of it."""
    with open(path, "rb") as f:
        if byte_range:
            start, _, end = byte_range.partition(":")
            start = int(start or 0)
            f.seek(start)
            blob = f.read(int(end) - start) if end else f.read()
        else:
            blob = f.read()
    return blob.decode("utf-8", "replace")


def merge(records: list[dict]) -> dict:
    """Combine the samples into one per-layer view.

    Each record already carries medians over its own interval, so this is a
    median of medians weighted by nothing -- stated plainly because it is not
    the same as the median over all callbacks, and drifts if the intervals
    have very different counts. ``n`` per layer is printed so that is visible.
    """
    per_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    p90_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[int, int] = defaultdict(int)
    # Per-callback facts from the record side, when it carries them: summed /
    # worst-of over the samples (they are counts and extremes, not medians).
    extra: dict[int, dict] = {}
    split = any(bool(r.get("split")) for r in records)
    for rec in records:
        for layer_s, v in (rec.get("layers") or {}).items():
            layer = int(layer_s)
            counts[layer] += int(v.get("n", 0))
            if "n_unsplit" in v:
                e = extra.setdefault(layer, {})
                e["n_unsplit"] = e.get("n_unsplit", 0) + int(v["n_unsplit"])
            rmax = (v.get("resid_us") or {}).get("max")
            if rmax is not None:
                e = extra.setdefault(layer, {})
                cur = (e.get("resid_us") or {}).get("max")
                if cur is None or abs(float(rmax)) > abs(cur):
                    e["resid_us"] = {"max": float(rmax)}
            for seg in SEGMENTS:
                if seg in (v.get("med_us") or {}):
                    per_layer[layer][seg].append(float(v["med_us"][seg]))
                if seg in (v.get("p90_us") or {}):
                    p90_layer[layer][seg].append(float(v["p90_us"][seg]))
    med = {
        layer: {seg: st.median(xs) for seg, xs in segs.items() if xs}
        for layer, segs in per_layer.items()
    }
    p90 = {
        layer: {seg: max(xs) for seg, xs in segs.items() if xs}
        for layer, segs in p90_layer.items()
    }
    return {"split": split, "median_us": med, "p90_us": p90, "n": dict(counts), "samples": len(records),
            "extra": extra}


# ------------------------------------------------------------- reconciliation


def reconcile(med_us: dict[str, float], split: bool, tol_us: float = 2.0) -> dict:
    """Sum of the per-segment medians against the median of cb.

    Kept because it is worth *printing*, not because it decides anything: the
    two sides are different statistics (medians do not add), so a non-zero
    residual is the normal state of a healthy run. ``ok`` is retained for
    callers that still look at it; the tool's verdict comes from
    :func:`check_layer`.
    """
    keys = ADDITIVE if split else ADDITIVE_LUMPED
    parts = sum(float(med_us.get(k, 0.0)) for k in keys)
    cb = float(med_us.get("cb", 0.0))
    return {
        "parts_us": round(parts, 3),
        "cb_us": round(cb, 3),
        "residual_us": round(cb - parts, 3),
        "ok": abs(cb - parts) <= tol_us,
    }


def check_layer(med: dict[str, float], p90: dict[str, float], split: bool,
                extra: dict | None = None, eps_us: float = 0.5) -> dict:
    """Necessary conditions of "the parts are inside the callback".

    Every check here follows from the per-callback identity *without* assuming
    medians are additive, so a violation means the segmentation (or the record
    side) is broken, not that the distribution is skewed. See the module
    docstring for what this cannot see.
    """
    bad: list[str] = []
    cb_med, cb_p90 = float(med.get("cb", 0.0)), float(p90.get("cb", 0.0))
    keys = ADDITIVE if split else ADDITIVE_LUMPED
    for k in keys + ("kt",):
        if k not in med:
            continue
        if float(med[k]) > cb_med + eps_us:
            bad.append(f"med({k})={med[k]:.2f} > med(cb)={cb_med:.2f}")
        if cb_p90 and k in p90 and float(p90[k]) > cb_p90 + eps_us:
            bad.append(f"p90({k})={p90[k]:.2f} > p90(cb)={cb_p90:.2f}")
    bound_lumped = sum(float(p90.get(k, 0.0)) for k in ADDITIVE_LUMPED)
    if bound_lumped and cb_med > bound_lumped + eps_us:
        bad.append(f"med(cb)={cb_med:.2f} > p90(pre)+p90(kt)+p90(post)={bound_lumped:.2f}")
    slack = bound_lumped - cb_med if bound_lumped else None
    slack_mid = None
    # 🔴 "a zero middle means the probe did not run" is a PROXY for unsplit
    # callbacks, and it is only sound on the probe-based split. Where WE make
    # the submit/sync calls -- ``packed``, and since 2026-09-22 the side stream
    # (layer.py:_kt_packed_forward) -- ``between`` and ``kt_tail`` are zero BY
    # CONSTRUCTION: sync is called right after submit with nothing between, and
    # kt_tail ends where sync ends. The proxy turned the first healthy forked
    # run red on 40/40 layers while that run's exact counters said
    # n_unsplit=0 over 661,285 callbacks and max|resid| = 0.0 us.
    # When the record carries ``n_unsplit`` the question is answered exactly a
    # few lines below, so the proxy is not applied at all.
    exact_unsplit = (extra or {}).get("n_unsplit") is not None
    if split:
        kt_med = float(med.get("kt", 0.0))
        for k in MIDDLE:
            if k in med and float(med[k]) > kt_med + eps_us:
                bad.append(f"med({k})={med[k]:.2f} > med(kt)={kt_med:.2f}")
            if not exact_unsplit and k in med and float(med[k]) <= 0.0:
                bad.append(f"med({k})=0 in a split run (the probe did not run for these callbacks)")
        bound_mid = sum(float(p90.get(k, 0.0)) for k in MIDDLE)
        if bound_mid and kt_med > bound_mid + eps_us:
            bad.append(f"med(kt)={kt_med:.2f} > sum p90(middle)={bound_mid:.2f}")
        slack_mid = bound_mid - kt_med if bound_mid else None
    # Exact, if the record side ever reports it (see the module docstring).
    n_unsplit = (extra or {}).get("n_unsplit")
    if n_unsplit is not None and split and int(n_unsplit) > 0:
        bad.append(f"n_unsplit={int(n_unsplit)} callbacks stored no middle segments")
    resid = (extra or {}).get("resid_us") or {}
    if resid.get("max") is not None and abs(float(resid["max"])) > eps_us:
        bad.append(f"per-callback residual max {float(resid['max']):+.2f} us")
    return {"ok": not bad, "violations": bad, "slack_us": slack, "slack_mid_us": slack_mid}


# ------------------------------------------------------------------ NOTIFY_WAIT


def notify_wait_steps(csv_path: str) -> tuple[list[list[float]], dict]:
    """NOTIFY_WAIT durations grouped by step, in step order.

    A step is the window between two MODEL_EXECUTE starts; the n-th
    NOTIFY_WAIT inside it is position n. Nothing is dropped here -- the caller
    decides which step shapes it trusts.
    """
    rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            k = (r.get("kernel_type") or "").strip()
            if k not in ("NOTIFY_WAIT", "MODEL_EXECUTE"):
                continue
            rows[k].append((float(r["task_start(us)"].strip()), float(r["task_time(us)"])))
    starts = sorted(s for s, _ in rows["MODEL_EXECUTE"])
    per_step: dict[int, list[float]] = defaultdict(list)
    for s, d in sorted(rows["NOTIFY_WAIT"]):
        i = bisect.bisect_right(starts, s) - 1
        if i >= 0:
            per_step[i].append(d)
    steps = [per_step[i] for i in sorted(per_step)]
    shapes: dict[int, int] = defaultdict(int)
    for v in steps:
        shapes[len(v)] += 1
    return steps, {"steps": len(steps), "shapes": dict(sorted(shapes.items()))}


def notify_wait_by_position(csv_path: str) -> dict[int, float]:
    """Per-position NOTIFY_WAIT medians over every step (the raw, unsliced view)."""
    steps, _ = notify_wait_steps(csv_path)
    pos: dict[int, list[float]] = defaultdict(list)
    for v in steps:
        for j, d in enumerate(v):
            pos[j].append(d)
    return {j: st.median(v) for j, v in pos.items()}


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation -- the one layer that runs 3x long (step-head jitter)
    must not be able to decide the alignment, and with Pearson it can."""
    if len(a) < 3 or len(a) != len(b):
        return 0.0
    return _pearson(_ranks(a), _ranks(b))


def notify_wait_ours(csv_path: str, n_layers: int, skip: int = 2, per_step: int | None = None):
    """Per-layer NOTIFY_WAIT medians for *our* layers, with the slicing stated.

    Only steps holding exactly ``per_step`` (default ``n_layers + skip``) waits
    are used -- a truncated step would silently renumber every layer after it
    -- and the first ``skip`` of each step are dropped, which is what makes
    position j equal layer j. This is the rule the first run's throwaway
    make_nw_ours40.py applied by hand.
    """
    per_step = per_step or (n_layers + skip)
    steps, meta = notify_wait_steps(csv_path)
    full = [v for v in steps if len(v) == per_step]
    by_pos: list[list[float]] = [[] for _ in range(per_step)]
    for v in full:
        for j, d in enumerate(v):
            by_pos[j].append(d)
    meta.update({"per_step": per_step, "skip": skip, "steps_used": len(full)})
    if not full:
        return {}, meta, {}
    pos_med = [st.median(v) for v in by_pos]
    shifted = {s: {i: pos_med[s + i] for i in range(n_layers)} for s in range(per_step - n_layers + 1)}
    return shifted.get(skip, {}), meta, shifted


def check_alignment(shifted: dict[int, dict[int, float]], cb_by_layer: dict[int, float], skip: int,
                    min_rho: float = 0.5, margin: float = 0.3) -> dict:
    """Is position j really layer j after dropping ``skip``?

    NOTIFY_WAIT is cb plus the dispatch/notify overhead, so across layers the
    two must move together; at a wrong shift they do not. The check is run
    against every feasible shift, and the chosen one has to win, not merely
    pass -- a threshold alone would accept two shifts at once.
    """
    layers = sorted(cb_by_layer)
    rho = {}
    for s, pos in shifted.items():
        if not all(i in pos for i in layers):
            continue
        rho[s] = spearman([pos[i] for i in layers], [cb_by_layer[i] for i in layers])
    chosen = rho.get(skip)
    others = [v for s, v in rho.items() if s != skip]
    best_other = max(others) if others else None
    ok = (
        chosen is not None
        and chosen >= min_rho
        and (best_other is None or chosen - best_other >= margin)
    )
    return {
        "ok": bool(ok),
        "rho_by_shift": {s: round(v, 3) for s, v in sorted(rho.items())},
        "chosen_shift": skip,
        "rho": None if chosen is None else round(chosen, 3),
        "best_other": None if best_other is None else round(best_other, 3),
        "min_rho": min_rho,
        "margin": margin,
    }


# ----------------------------------------------------------------- A and F split


def parse_realtime(text: str) -> tuple[list[tuple[int, int, int, int]], int]:
    """Every callback of the ``[kt-seam-rt]`` lines: (layer, entry_ns, exit_ns, cb_ns).

    entry/exit are absolute CLOCK_REALTIME ns (seam_timing.realtime_rows); cb
    is the perf_counter duration of the same callback. Returns the callbacks
    and the number of markers that did not parse (a cut line is counted).
    """
    out, bad = [], 0
    for line in text.splitlines():
        if "[kt-seam-rt]" not in line:
            continue
        m = REC_RT.search(line)
        try:
            rec = json.loads(m.group(1)) if m else None
            layer = int(rec["layer"])
            ent, ext, cbs = rec["entry_realtime_ns"], rec["exit_realtime_ns"], rec["cb_ns"]
            if not (len(ent) == len(ext) == len(cbs)):
                raise ValueError("ragged")
        except (ValueError, KeyError, TypeError, AttributeError):
            bad += 1
            continue
        out.extend((layer, int(a), int(b), int(c)) for a, b, c in zip(ent, ext, cbs))
    return out, bad


def _us_to_ns(field: str) -> int:
    """``1790089308256624.2`` (us, maybe padded) -> exact integer ns."""
    from decimal import Decimal

    return int(Decimal(field.strip()) * 1000)


def hostfunc_pairs(csv_path: str) -> list[tuple[int, int, int]]:
    """(HF.ts, NW.ts, NW.ts + NW.dur) in ns, one per HOSTFUNC_CALLBACK that is
    followed on its own stream by a NOTIFY_WAIT before the next HOSTFUNC."""
    by_stream: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            k = (r.get("kernel_type") or "").strip()
            if k not in ("NOTIFY_WAIT", "HOSTFUNC_CALLBACK"):
                continue
            ts = _us_to_ns(r["task_start(us)"])
            dur = _us_to_ns(r["task_time(us)"])
            # HF before NW when they share a ts (they do in this build).
            by_stream[(r.get("stream_id") or "").strip()].append((ts, 0 if k == "HOSTFUNC_CALLBACK" else 1, dur))
    pairs = []
    for rows in by_stream.values():
        rows.sort()
        hf = None
        for ts, kind, dur in rows:
            if kind == 0:
                hf = ts
            elif hf is not None:
                pairs.append((hf, ts, ts + dur))
                hf = None
    pairs.sort()
    return pairs


def _q(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def split_a_f(callbacks, pairs, tol_us: float = 5.0, max_unmatched: float = 0.01,
              max_nonpos: float = 0.01) -> dict:
    """Join callbacks to (HOSTFUNC, NOTIFY_WAIT) pairs by time; A, F per layer.

    Pure: ``callbacks`` as parse_realtime returns them, ``pairs`` as
    hostfunc_pairs. Returns ``ok`` and, only when ok, the per-layer columns;
    ``why`` lists every failed criterion (see the module docstring).
    """
    why: list[str] = []
    res = {"ok": False, "why": why, "layers": {}, "n_callbacks": len(callbacks), "n_pairs": len(pairs)}
    if not pairs:
        why.append("the profile has no HOSTFUNC_CALLBACK + NOTIFY_WAIT pair")
        return res
    lo, hi = pairs[0][0], max(p[2] for p in pairs)
    starts = [p[0] for p in pairs]
    in_span = [c for c in callbacks if lo <= (c[1] + c[2]) // 2 <= hi]
    res["n_in_span"] = len(in_span)
    if not in_span:
        why.append(f"none of the {len(callbacks)} callbacks falls inside the profile's span "
                   f"[{lo / 1e9:.3f}, {hi / 1e9:.3f}] s -- not the same run, or not CLOCK_REALTIME")
        return res
    used: dict[int, int] = defaultdict(int)
    per_layer: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    unmatched = 0
    for layer, ent, ext, cb in in_span:
        mid = (ent + ext) // 2
        i = bisect.bisect_right(starts, mid) - 1
        if i < 0 or mid > pairs[i][2]:
            unmatched += 1
            continue
        hf, nw_ts, nw_end = pairs[i]
        used[i] += 1
        a = (ent - hf) / 1000.0
        f = (nw_end - ext) / 1000.0
        ref = ((nw_end - nw_ts) - cb) / 1000.0
        d = per_layer[layer]
        d["A"].append(a)
        d["F"].append(f)
        d["AF"].append(a + f)
        d["ref"].append(ref)
        d["diff"].append(a + f - ref)
    res["unmatched"] = unmatched
    doubles = sum(1 for v in used.values() if v > 1)
    res["pairs_taken_twice"] = doubles
    if unmatched > max_unmatched * len(in_span):
        why.append(f"{unmatched}/{len(in_span)} callbacks inside the span sit in no HOSTFUNC..NOTIFY_WAIT "
                   f"window (> {max_unmatched:.0%})")
    if doubles:
        why.append(f"{doubles} pair(s) took more than one callback")
    for layer in sorted(per_layer):
        d = per_layer[layer]
        n = len(d["A"])
        nonpos = sum(1 for a, f in zip(d["A"], d["F"]) if a <= 0 or f <= 0)
        row = {
            "n": n,
            "A_med_us": round(st.median(d["A"]), 3), "A_p90_us": round(_q(d["A"], 0.9), 3),
            "F_med_us": round(st.median(d["F"]), 3), "F_p90_us": round(_q(d["F"], 0.9), 3),
            "AF_med_us": round(st.median(d["AF"]), 3),
            "NW_minus_cb_med_us": round(st.median(d["ref"]), 3),
            "diff_med_us": round(st.median(d["diff"]), 3),
            "nonpos": nonpos,
        }
        res["layers"][layer] = row
        if row["A_med_us"] <= 0 or row["F_med_us"] <= 0:
            why.append(f"layer {layer}: med(A)={row['A_med_us']:+.2f} med(F)={row['F_med_us']:+.2f} us, both must be > 0")
        if nonpos > max_nonpos * n:
            why.append(f"layer {layer}: {nonpos}/{n} callbacks with A <= 0 or F <= 0")
        if abs(row["diff_med_us"]) >= tol_us:
            why.append(f"layer {layer}: med((A+F) - (NW - cb)) = {row['diff_med_us']:+.2f} us, "
                       f"|.| must be < {tol_us:g}")
    # 🔴 Name the one failure that is NOT a mis-pairing: a constant offset
    # between the profile's timebase and the host's CLOCK_REALTIME shifts A up
    # and F down by the same amount, so A+F (and therefore ``diff``) survives
    # while the split does not. Its fingerprint is med(F) < 0 on essentially
    # every layer with a small spread -- a mis-pairing scatters instead. The
    # offset is then bounded without being known: it must be at least
    # max(-med(F)) for F to be non-negative and at most min(med(A)) for A to
    # stay non-negative. First seen 2026-09-22 on the forked path, where every
    # layer read med(F) = -84.6 us (-89.1..-65.7) -> offset in [89, 132] us.
    if res["layers"]:
        fs = [v["F_med_us"] for v in res["layers"].values()]
        as_ = [v["A_med_us"] for v in res["layers"].values()]
        neg = [x for x in fs if x < 0]
        if len(neg) >= 0.9 * len(fs) and max(fs) - min(fs) < abs(st.median(fs)):
            res["clock_offset_bound_us"] = [round(-max(fs), 2), round(min(as_), 2)]
            why.insert(0,
                f"med(F) < 0 on {len(neg)}/{len(fs)} layers with a spread of only "
                f"{max(fs) - min(fs):.1f} us: this is a CONSTANT offset between the profile's "
                f"timebase and CLOCK_REALTIME, not a mis-pairing. A+F survives it, the split does "
                f"not. The offset lies in [{-max(fs):.1f}, {min(as_):.1f}] us; until it is "
                f"calibrated, read A+F_dev and do not split it")
    res["ok"] = not why
    if res["ok"]:
        L = res["layers"]
        res["totals_ms"] = {
            "A": round(sum(v["A_med_us"] for v in L.values()) / 1000.0, 4),
            "F": round(sum(v["F_med_us"] for v in L.values()) / 1000.0, 4),
            "A_plus_F": round(sum(v["AF_med_us"] for v in L.values()) / 1000.0, 4),
        }
    else:
        res["layers"] = {}  # refused: no column survives a failed check
    return res


# --------------------------------------------------------------------- the CLI


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="serve.log (or any file holding the [kt-seam] records)")
    ap.add_argument("--ktphase", help="JSON from ktphase.py (4th arg): kt's own per-layer compute D")
    ap.add_argument("--nw-csv", help="profile task_time.csv: NOTIFY_WAIT per layer position")
    ap.add_argument("--nw-median-us", type=float, help="one NOTIFY_WAIT median for every layer")
    ap.add_argument("--nw-skip", type=int, default=2,
                    help="NOTIFY_WAIT at the head of a step that are not our layers (default 2)")
    ap.add_argument("--nw-per-step", type=int, default=0,
                    help="only use steps with exactly this many NOTIFY_WAIT (default layers + --nw-skip)")
    ap.add_argument("--nw-align-min", type=float, default=0.5, help="minimum Spearman(NW, cb) at the chosen shift")
    ap.add_argument("--nw-align-margin", type=float, default=0.3, help="how far it must beat the other shifts")
    ap.add_argument("--since", help="only records covering this time onward (2026-09-22T15:01:44Z / 15:01:44)")
    ap.add_argument("--until", help="only records covering up to this time")
    ap.add_argument("--year", type=int, default=dt.date.today().year, help="year for the log's MM-DD stamps")
    ap.add_argument("--window-min-overlap", type=float, default=0.5,
                    help="fraction of a record's interval that must be inside the window (default 0.5)")
    ap.add_argument("--bytes", dest="byte_range", help="read only this START:END byte slice of the log")
    ap.add_argument("--layers", type=int, default=0, help="expected layer count (0 = whatever reported)")
    ap.add_argument("--tol-us", type=float, default=2.0, help="informational residual tolerance (default 2 us)")
    ap.add_argument("--eps-us", type=float, default=0.5, help="slack on the containment/union checks")
    ap.add_argument("--af-tol-us", type=float, default=5.0,
                    help="max |med((A+F) - (NW - cb))| per layer before A/F are refused (default 5 us)")
    ap.add_argument("--af-max-unmatched", type=float, default=0.01,
                    help="fraction of in-span callbacks allowed outside every HF..NW pair (default 0.01)")
    ap.add_argument("--af-max-nonpos", type=float, default=0.01,
                    help="fraction of a layer's callbacks allowed with A <= 0 or F <= 0 (default 0.01)")
    ap.add_argument("--json-out", help="write the whole decomposition here")
    args = ap.parse_args(argv)

    since = parse_window_arg(args.since, args.year) if args.since else None
    until = parse_window_arg(args.until, args.year) if args.until else None

    text = read_text(args.log, args.byte_range)
    records, dropped = parse_records(text)
    if not records:
        print("no [kt-seam] records found -- was VLLM_ASCEND_DSV41_KT_SEAM_TIMING set?", file=sys.stderr)
        return 2
    records, window_note = select_window(records, since, until, args.year, args.window_min_overlap)
    print(window_note + (f"  [bytes {args.byte_range}]" if args.byte_range else ""))
    if not records:
        print("no [kt-seam] records inside the window -- check --since/--until against the log's clock",
              file=sys.stderr)
        return 2
    m = merge(records)
    split, med, p90 = m["split"], m["median_us"], m["p90_us"]
    layers = sorted(med)
    print(f"[kt-seam] samples {m['samples']}  layers {len(layers)}  dropped(unparsable) {dropped}  split={split}")
    if not split:
        print("  ⚠ split=false: kt's submit/sync were not separable on this run (no CPUInfer probe);")
        print("    only pre / kt / post are measured, so C and E cannot be told apart.")
    if args.layers and len(layers) != args.layers:
        print(f"  ⚠ {len(layers)} layers reported, expected {args.layers}: a layer whose callback never ran")
        print("    is missing here, and a missing layer and a fast layer read the same in a sum.")

    ktphase = {}
    if args.ktphase:
        raw = json.load(open(args.ktphase))
        ktphase = {int(k): float(v) for k, v in (raw.get("layer_median_us") or raw).items()}

    # ---- NOTIFY_WAIT, sliced and checked before it is allowed into the table
    nw: dict[int, float] = {}
    align = None
    if args.nw_csv:
        n_layers = args.layers or len(layers)
        nw, nwmeta, shifted = notify_wait_ours(
            args.nw_csv, n_layers, skip=args.nw_skip, per_step=args.nw_per_step or None)
        print(f"  NOTIFY_WAIT from {args.nw_csv}: {nwmeta['steps']} steps, shapes {nwmeta['shapes']}, "
              f"{nwmeta['steps_used']} with exactly {nwmeta['per_step']} used, first {args.nw_skip} dropped")
        if nw:
            align = check_alignment(shifted, {l: med[l].get("cb", 0.0) for l in layers},
                                    args.nw_skip, args.nw_align_min, args.nw_align_margin)
            print(f"  alignment: Spearman(NW, cb) by shift {align['rho_by_shift']} -> "
                  f"shift {align['chosen_shift']} rho {align['rho']} "
                  f"(need >= {align['min_rho']} and +{align['margin']} over the others): "
                  + ("✅" if align["ok"] else "❌ refusing the NW / A+F_dev columns"))
            if align["ok"]:
                print(f"  median-of-layers NW {st.median(list(nw.values())):.1f} us")
            else:
                nw = {}
        else:
            print("  ⚠ no step had exactly the expected number of NOTIFY_WAIT -- nothing to align")

    cols = ADDITIVE if split else ADDITIVE_LUMPED
    head = "layer     n  " + "".join(f"{c:>9s}" for c in cols) + "       cb      gap"
    if ktphase:
        head += "     D_kt   E=sync-D"
    if nw or args.nw_median_us:
        head += "       NW   A+F_dev"
    head += "  resid_med    slack"
    print("\n" + head)
    out_layers = {}
    bad = []
    for layer in layers:
        mm = med[layer]
        pp = p90.get(layer, {})
        row = f"{layer:5d} {m['n'].get(layer, 0):5d}  " + "".join(f"{mm.get(c, 0.0):9.2f}" for c in cols)
        row += f" {mm.get('cb', 0.0):8.2f} {mm.get('gap', 0.0):8.1f}"
        rec = reconcile(mm, split, args.tol_us)
        chk = check_layer(mm, pp, split, m["extra"].get(layer), args.eps_us)
        entry = {"med_us": mm, "p90_us": pp, "n": m["n"].get(layer, 0),
                 "resid_med": rec, "check": chk}
        if not chk["ok"]:
            bad.append((layer, chk))
        d = ktphase.get(layer)
        if ktphase:
            e = (mm.get("sync", 0.0) - d) if (d is not None and split) else None
            row += f" {d if d is not None else float('nan'):8.1f} " + (f"{e:9.1f}" if e is not None else "        -")
            entry["D_kt_us"] = d
            entry["E_us"] = e
        nwv = nw.get(layer, args.nw_median_us)
        if nw or args.nw_median_us:
            af = (nwv - mm.get("cb", 0.0)) if nwv is not None else None
            row += f" {nwv if nwv is not None else float('nan'):8.1f} " + (f"{af:9.1f}" if af is not None else "        -")
            entry["NW_us"] = nwv
            entry["A_plus_F_device_us"] = af
        slack = chk["slack_us"]
        row += f" {rec['residual_us']:+10.2f} {slack if slack is not None else float('nan'):8.1f}"
        out_layers[layer] = entry
        print(row)

    n_layers = len(layers)
    print("\nwhole net (sum of the per-layer medians), ms per token:")
    for c in cols + ("cb",):
        print(f"  {c:9s} {sum(med[l].get(c, 0.0) for l in layers)/1000.0:7.3f}")
    if ktphase:
        dsum = sum(ktphase.get(l, 0.0) for l in layers) / 1000.0
        print(f"  {'D_kt':9s} {dsum:7.3f}")
        if split:
            esum = sum(med[l].get("sync", 0.0) - ktphase.get(l, 0.0) for l in layers if l in ktphase) / 1000.0
            print(f"  {'E':9s} {esum:7.3f}   (sync - D_kt: worker wakeup + completion/write-back)")
    totals = {}
    if nw or args.nw_median_us:
        nwsum = sum((nw.get(l, args.nw_median_us) or 0.0) for l in layers) / 1000.0
        cbsum = sum(med[l].get("cb", 0.0) for l in layers) / 1000.0
        print(f"  {'NW':9s} {nwsum:7.3f}")
        print(f"  {'A+F_dev':9s} {nwsum - cbsum:7.3f}   (NOTIFY_WAIT - T_cb: dispatch to this thread + notify back)")
        totals = {"NW_ms": round(nwsum, 4), "A_plus_F_dev_ms": round(nwsum - cbsum, 4)}
        if ktphase:
            seam = nwsum - sum(ktphase.get(l, 0.0) for l in layers) / 1000.0
            print(f"  {'seam':9s} {seam:7.3f}   (NW - D_kt: the §4.18 107.5 us/layer, now broken down above)")
            totals["seam_ms"] = round(seam, 4)
    elif args.nw_csv:
        print("  NW / A+F_dev: refused (the alignment check did not pass; see above)")

    print()
    print("resid_med = med(cb) - sum of the segment medians. NOT a criterion: per callback the")
    print("  segments add up to cb by construction, and medians do not add. A skewed segment makes")
    print("  it positive on a perfectly healthy run.  slack = p90(pre)+p90(kt)+p90(post) - med(cb):")
    print("  how big a hole the union-bound check below could have seen on that layer.")
    if bad:
        print(f"\n❌ the segmentation check failed on {len(bad)}/{n_layers} layers:")
        for layer, chk in bad[:8]:
            print(f"   layer {layer}: " + "; ".join(chk["violations"]))
        print("   these are necessary conditions of 'each part lies inside the callback'; a violation")
        print("   means the records are wrong (a misplaced stamp, or callbacks whose middle was never")
        print("   filled mixed into a split run). Do not read the table above.")
    else:
        print(f"\n✅ segmentation check ok on all {n_layers} layers "
              "(containment at median and p90, union bound, no empty middle).")
        print("   Not judged: a hole smaller than each layer's slack, and unsplit callbacks below a")
        print("   large fraction -- both need the record side to count per-callback residuals.")
    if align is not None and not align["ok"]:
        print("❌ NOTIFY_WAIT alignment refused -- A+F_dev was not computed.")

    # ---- A and F separately (needs the [kt-seam-rt] lines and the csv)
    af = None
    rt_calls, rt_bad = parse_realtime(text)
    if args.nw_csv and rt_calls:
        af = split_a_f(rt_calls, hostfunc_pairs(args.nw_csv), args.af_tol_us,
                       args.af_max_unmatched, args.af_max_nonpos)
        print(f"\nA / F split: {af['n_callbacks']} callbacks with CLOCK_REALTIME stamps "
              f"({rt_bad} unparsable line(s)), {af['n_pairs']} HOSTFUNC+NOTIFY_WAIT pairs, "
              f"{af.get('n_in_span', 0)} callbacks inside the profile, {af.get('unmatched', '-')} unmatched")
        if af["ok"]:
            print("layer     n    A_med    A_p90    F_med    F_p90     A+F   NW-cb   diff")
            for layer, v in sorted(af["layers"].items()):
                print(f"{layer:5d} {v['n']:5d} {v['A_med_us']:8.2f} {v['A_p90_us']:8.2f} {v['F_med_us']:8.2f} "
                      f"{v['F_p90_us']:8.2f} {v['AF_med_us']:7.1f} {v['NW_minus_cb_med_us']:7.1f} "
                      f"{v['diff_med_us']:+6.2f}")
            t = af["totals_ms"]
            print(f"  whole net, ms/token: A {t['A']:.3f}  F {t['F']:.3f}  A+F {t['A_plus_F']:.3f}")
            print(f"✅ A/F reconcile with NW - cb on every layer (|diff| < {args.af_tol_us:g} us), A, F > 0")
        else:
            print("❌ refusing the A / F columns:")
            for w in af["why"][:10]:
                print(f"   {w}")
    elif rt_calls:
        print("\nA / F split: [kt-seam-rt] lines present but no --nw-csv to join them with")

    if args.json_out:
        json.dump(
            {"split": split, "samples": m["samples"], "dropped": dropped,
             "window": window_note, "alignment": align, "totals": totals, "a_f": af,
             "layers": {str(k): v for k, v in out_layers.items()}},
            open(args.json_out, "w"), indent=1, default=float)
        print(f"wrote {args.json_out}")
    return 1 if (bad or (align is not None and not align["ok"]) or (af is not None and not af["ok"])) else 0



if __name__ == "__main__":
    raise SystemExit(main())
