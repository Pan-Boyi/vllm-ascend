#!/usr/bin/env python3
"""Cut 2-3 steady-state decode steps out of an Ascend profile into a compact JSON
that a self-contained Gantt page can inline.

Why ``task_time.csv`` and not ``trace_view.json``
-------------------------------------------------
``trace_view.json`` is 0.6 GB per arm and is mostly *host* side: tens of
thousands of Python/aclnn/op-dispatch events that a device timeline has no use
for. ``task_time.csv`` in the same ``ASCEND_PROFILER_OUTPUT/`` carries exactly
the device tasks with the one column the timeline needs and
``kernel_details.csv`` lacks -- ``stream_id`` -- plus the non-kernel task types
a host-func seam is made of (``NOTIFY_WAIT``, ``HOSTFUNC_CALLBACK``,
``MEMCPY_ASYNC``, ``*_EVENT_WAIT``). Same rows, same clock, 1/6 the bytes and
no host noise. This is the same source ``devgaps.py`` reads, on purpose: the
slice has to reproduce devgaps' aggregates, and the ``checks`` block of the
output recomputes them from the slice so the two can be put side by side.

🔴 CLOCK. ``task_start(us)`` is CLOCK_REALTIME microseconds, i.e. wall clock of
the machine at capture time. Two profiles taken at 15:01 and 23:15 are eight
hours apart on that axis. **Slices from different runs must never be laid on a
shared time axis.** Every slice is therefore rebased to *its own* origin (the
first ``MODEL_EXECUTE`` start inside the slice) and the JSON carries
``origin_abs_us`` plus a ``clock`` note saying so. The renderer compares
*shapes*, never absolute positions.

Which steps
-----------
Steps are delimited by ``MODEL_EXECUTE`` starts (the same boundary
``devgaps.py`` uses). The first and last step are never eligible: the first can
still carry graph capture / first-token effects and the last is truncated by
profiler teardown. Among the remaining ones the script keeps the steps whose
per-step task census matches the run's mode (same number of
``HOSTFUNC_CALLBACK`` / ``NOTIFY_WAIT`` / wait tasks -- a prefill or a capture
step differs here) and whose period is within ``--period-tol`` of the run
median, then takes the ``--steps`` *consecutive* such steps nearest the middle
of the run. The choice and its reason land in ``step_pick``.

Size
----
Two lossy-but-accounted reductions, both reversible in aggregate:

  fragments   a task shorter than ``--frag-us`` (default 0.5 us) is not
              emitted individually; it is added to a per (stream, type)
              counter in ``merged[]`` with its count and summed duration, and
              to ``merged_total``. Nothing disappears silently: for any type,
              slice count == emitted + merged, and slice duration ==
              emitted + merged. ``HOSTFUNC_CALLBACK`` is exempt -- its nodes
              are 0.007 us each but they are the object of study, so they are
              kept as zero-width markers.
  runs        consecutive tasks of the same *class* on the same stream
              separated by <= ``--coalesce-gap`` us become one block carrying
              ``k`` (how many were folded in). Only the compute and memcpy
              classes are coalesced; the seam classes (host-func, notify-wait,
              event-wait) never are. A coalesced block spans exactly the
              wall-clock the folded tasks spanned, so the picture is unchanged
              at any zoom a browser can show -- 2440 AI_VECTOR_CORE tasks per
              step are ~0.05 px each.

    trace_slice.py <trace_dir|task_time.csv> --arm serial --out slice.json
"""
import argparse
import bisect
import csv
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

# class ids, also the colour buckets the renderer uses
C_HOSTFUNC, C_NOTIFY, C_EVTWAIT, C_MEMCPY, C_COMPUTE, C_OTHER = 0, 1, 2, 3, 4, 5
CLASS_NAMES = ["hostfunc", "notify_wait", "event_wait", "memcpy", "compute", "other"]

COMPUTE_TYPES = {"AI_CORE", "MIX_AIC", "AI_VECTOR_CORE", "MIX_AIV", "AI_CPU"}
# Types whose per-stream, per-step count says what kind of step this is. Only
# the two seam types: they are dead stable across decode steps (42/42 on the
# serial main stream, 40/40 on the side stream plus 2/2 on the side arm's main
# stream) and differ immediately on a capture or prefill step. Copy/event counts
# were tried too and are *not* usable here -- they wobble by one on the aux
# streams from step to step, which rejects perfectly ordinary decode steps.
CENSUS_TYPES = ("HOSTFUNC_CALLBACK", "NOTIFY_WAIT")
COALESCE_CLASSES = {C_COMPUTE, C_MEMCPY}


def klass(t: str) -> int:
    if t == "HOSTFUNC_CALLBACK":
        return C_HOSTFUNC
    if t == "NOTIFY_WAIT":
        return C_NOTIFY
    if t.endswith("EVENT_WAIT") or t == "MEM_WAIT_VALUE":
        return C_EVTWAIT
    if t == "MEMCPY_ASYNC":
        return C_MEMCPY
    if t in COMPUTE_TYPES:
        return C_COMPUTE
    return C_OTHER


def find_csv(path: str) -> str:
    if os.path.isfile(path):
        return path
    for cand in (
        os.path.join(path, "ASCEND_PROFILER_OUTPUT", "task_time.csv"),
        os.path.join(path, "task_time.csv"),
    ):
        if os.path.isfile(cand):
            return cand
    # one level of dp0_*_ascend_pt
    for d in sorted(os.listdir(path)):
        cand = os.path.join(path, d, "ASCEND_PROFILER_OUTPUT", "task_time.csv")
        if os.path.isfile(cand):
            return cand
    raise SystemExit(f"no task_time.csv under {path}")


def load(csv_path):
    """-> {stream_id: [(start_us, stop_us, kernel_type, kernel_name), ...] sorted}"""
    per = defaultdict(list)
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                s = float((r["task_start(us)"] or "").strip())
                e = float((r["task_stop(us)"] or "").strip())
            except (KeyError, ValueError, AttributeError):
                continue
            per[(r["stream_id"] or "").strip()].append(
                (s, e, (r["kernel_type"] or "").strip(), (r["kernel_name"] or "").strip())
            )
    for v in per.values():
        v.sort()
    return per


def pick_steps(per, n_steps, tol, skip_head, skip_tail):
    """Consecutive steady decode steps nearest the middle. -> (indices, bounds, why)"""
    marks = sorted(s for v in per.values() for s, _e, t, _n in v if t == "MODEL_EXECUTE")
    if len(marks) < n_steps + 3:
        raise SystemExit(f"only {len(marks)} MODEL_EXECUTE marks; cannot slice {n_steps} steps")
    bounds = list(zip(marks, marks[1:]))  # last mark has no end -> dropped
    periods = [b - a for a, b in bounds]
    med = st.median(periods)

    # census once, by bisect: 190 steps x 900k tasks would be quadratic
    cens_ev = sorted(
        (s, sid, t)
        for sid, v in per.items() for s, _e, t, _n in v if t in CENSUS_TYPES
    )
    cens_key = [x[0] for x in cens_ev]
    census = []
    for a, b in bounds:
        c = Counter()
        for _s, sid, t in cens_ev[bisect.bisect_left(cens_key, a):bisect.bisect_left(cens_key, b)]:
            c[(sid, t)] += 1
        census.append(tuple(sorted(c.items())))
    mode_census = Counter(census).most_common(1)[0][0]

    lo, hi = skip_head, len(bounds) - skip_tail
    # "middle section" = the middle half of the eligible range, so a run that
    # drifts (host loadavg climbing, a stall) at either end cannot be picked.
    mlo, mhi = lo + (hi - lo) // 4, hi - (hi - lo) // 4
    ok = set(
        i for i in range(mlo, mhi)
        if census[i] == mode_census and abs(periods[i] - med) <= tol * med
    )
    runs = [i for i in sorted(ok) if all((i + j) in ok for j in range(n_steps))]
    if not runs:
        raise SystemExit(
            "no run of %d consecutive steady steps in [%d,%d) (tol=%.2f, med=%.0f us)"
            % (n_steps, mlo, mhi, tol, med)
        )
    # among those, the run whose periods sit closest to the run median. Picking
    # "the middle index" alone is not enough: a profile can spend its middle in
    # a slow patch (seam/244 steps 90-102 sit 10%% above median and carry a 14%%
    # fatter NOTIFY_WAIT), and a slice taken there quietly disagrees with every
    # whole-run aggregate it is supposed to reproduce.
    mid = (mlo + mhi) // 2
    start = min(
        runs,
        key=lambda i: (
            sum(abs(periods[i + j] - med) for j in range(n_steps)) / n_steps,
            abs(i + n_steps / 2 - mid),
            i,
        ),
    )
    idx = list(range(start, start + n_steps))
    why = {
        "boundary": "MODEL_EXECUTE start",
        "n_steps_in_run": len(bounds),
        "eligible_range": [mlo, mhi],
        "excluded": "first %d and last %d steps (capture / teardown), then the outer "
                    "quarters of what is left (middle section only), then steps whose "
                    "per-stream census of %s differs from the run mode, then steps whose "
                    "period is more than %.0f%% off the run median"
                    % (skip_head, skip_tail, "/".join(CENSUS_TYPES), tol * 100),
        "run_period_median_us": round(med, 3),
        "picked_period_us": [round(periods[i], 3) for i in idx],
        "picked_because": "the consecutive run whose periods sit closest to the run median period (tie-break: nearest the middle)",
        "picked_period_dev_pct": [round(100 * (periods[i] - med) / med, 2) for i in idx],
    }
    return idx, bounds, why


def roles(per, t0, t1):
    """Name what each stream is, from what it carries inside the window."""
    out = {}
    for sid, v in per.items():
        c = Counter()
        busy = defaultdict(float)
        for s, e, t, _n in v:
            if t0 <= s < t1:
                c[t] += 1
                busy[klass(t)] += e - s
        if not c:
            continue
        out[sid] = (c, busy)
    step_sid = next((sid for sid, (c, _b) in out.items() if c.get("MODEL_EXECUTE")), None)
    compute = {sid: b[C_COMPUTE] for sid, (_c, b) in out.items() if sid != step_sid}
    main_sid = max(compute, key=compute.get) if compute else None
    res = {}
    for sid, (c, _b) in out.items():
        if sid == step_sid:
            r = "frame"          # carries MODEL_EXECUTE / MODEL_WAIT_COMPLETE
        elif sid == main_sid:
            r = "main"
        elif c.get("HOSTFUNC_CALLBACK"):
            r = "side"           # a second stream that owns host-func nodes
        else:
            r = "aux"
        res[sid] = r
    return res, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="trace dir (dp0_*_ascend_pt or its parent) or task_time.csv")
    ap.add_argument("--arm", required=True, help="label, e.g. serial / side")
    ap.add_argument("--run", default="", help="oncard_runs/... this profile came from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2,
                    help="2 keeps each arm near 60 KB; 3 is ~1.4x that")
    ap.add_argument("--frag-us", type=float, default=0.5)
    ap.add_argument("--coalesce-gap", type=float, default=1.0)
    ap.add_argument("--period-tol", type=float, default=0.15)
    ap.add_argument("--skip-head", type=int, default=5)
    ap.add_argument("--skip-tail", type=int, default=2)
    ap.add_argument("--max-names", type=int, default=140)
    ap.add_argument("--roles", default="main,side",
                    help="which stream roles get a lane: main/side/frame/aux. The checks "
                         "block always covers every stream, whatever is drawn.")
    a = ap.parse_args()

    csv_path = find_csv(a.trace)
    per = load(csv_path)
    idx, bounds, why = pick_steps(per, a.steps, a.period_tol, a.skip_head, a.skip_tail)
    win_t0, win_t1 = bounds[idx[0]][0], bounds[idx[-1]][1]
    role, _census = roles(per, win_t0, win_t1)

    # ---- raw slice, per stream, for both the checks and the emit -------------
    raw = {}
    for sid, v in per.items():
        w = [(s, e, t, n) for s, e, t, n in v if win_t0 <= s < win_t1]
        if w:
            raw[sid] = w

    # ---- checks: recomputed from the raw slice, before any merging ----------
    nsteps = len(idx)
    checks = {"n_steps": nsteps, "per_stream": {}}
    for sid, w in raw.items():
        durs, pregaps = defaultdict(list), defaultdict(list)
        last_end = None
        for s, e, t, _n in w:
            durs[t].append(e - s)
            if last_end is not None:
                pregaps[t].append(max(0.0, s - last_end))
            last_end = max(last_end, e) if last_end is not None else e
        row = {}
        for t in sorted(durs, key=lambda x: -len(durs[x])):
            row[t] = {
                "n": len(durs[t]),
                "n_per_step": round(len(durs[t]) / nsteps, 3),
                "dur_med_us": round(st.median(durs[t]), 3),
                "dur_ms_per_step": round(sum(durs[t]) / nsteps / 1000, 4),
                "pregap_med_us": round(st.median(pregaps[t]), 3) if pregaps.get(t) else None,
            }
        checks["per_stream"][sid] = {"role": role[sid], "types": row}

    # §4.20 note (2): on a stream that owns the 42 host-func nodes of a step, the
    # first 2 of each step are not MoE layers. The per-layer medians in the docs
    # are over the remaining 40, so recompute them that way too.
    ours = {}
    step_edges = [bounds[i] for i in idx]
    for sid, w in raw.items():
        for t in ("NOTIFY_WAIT", "HOSTFUNC_CALLBACK"):
            per_step = []
            for a0, b0 in step_edges:
                per_step.append([(s, e) for s, e, tt, _n in w if tt == t and a0 <= s < b0])
            if not any(per_step):
                continue
            n_each = [len(x) for x in per_step]
            drop = 2 if n_each and n_each[0] == 42 else 0
            sel = [x for lst in per_step for x in lst[drop:]]
            if not sel:
                continue
            ours.setdefault(sid, {})[t] = {
                "n_per_step_total": round(sum(n_each) / nsteps, 3),
                "dropped_head_per_step": drop,
                "n_per_step_ours": round(len(sel) / nsteps, 3),
                "dur_med_us_ours": round(st.median([e - s for s, e in sel]), 3),
            }
    checks["layer_columns"] = ours

    # ---- emit ---------------------------------------------------------------
    want = {r.strip() for r in a.roles.split(",") if r.strip()}
    order = {"main": 0, "side": 1, "frame": 2, "aux": 3}
    sids = sorted((s for s in raw if role[s] in want), key=lambda x: (order[role[x]], x))
    dropped = sorted(s for s in raw if role[s] not in want)
    if not sids:
        raise SystemExit(f"--roles {a.roles} matched no stream (roles present: {sorted(set(role.values()))})")
    sidx = {s: i for i, s in enumerate(sids)}

    name_freq = Counter()
    for sid in sids:
        w = raw[sid]
        for s, e, t, n in w:
            if e - s >= a.frag_us or t == "HOSTFUNC_CALLBACK":
                name_freq[(t, n if n and n != "N/A" else "")] += 1
    keep_names = set(k for k, _c in name_freq.most_common(a.max_names))

    names, name_cls, nidx = [], [], {}

    def name_id(t, n):
        key = (t, n if n and n != "N/A" else "")
        if key not in keep_names:
            key = (t, "")
        if key not in nidx:
            nidx[key] = len(names)
            names.append(key[0] if not key[1] else key[0] + " " + key[1])
            name_cls.append(klass(key[0]))
        return nidx[key]

    origin = win_t0
    out_tasks = []
    merged = defaultdict(lambda: [0, 0.0])   # (sid, type) -> [count, dur_sum]

    def num(x):
        # ts are quarter-us quantised; emit 4 rather than 4.0 -- lossless, ~3 KB/arm
        x = round(x, 2)
        return int(x) if x == int(x) else x

    def flush(si, pend):
        o = {"s": si, "n": pend[5], "t0": num(pend[1] - origin),
             "d": num(pend[2] - pend[1])}
        if pend[3] > 1:
            o["k"] = pend[3]
        out_tasks.append(o)

    for sid in sids:
        si = sidx[sid]
        pend = None  # [cls, start, end, k, best_dur, best_nid]
        for s, e, t, n in raw[sid]:
            d = e - s
            if d < a.frag_us and t != "HOSTFUNC_CALLBACK":
                merged[(sid, t)][0] += 1
                merged[(sid, t)][1] += d
                continue
            c = klass(t)
            if pend and c in COALESCE_CLASSES and pend[0] == c and s - pend[2] <= a.coalesce_gap:
                pend[2] = max(pend[2], e)
                pend[3] += 1
                if d > pend[4]:
                    pend[4], pend[5] = d, name_id(t, n)
                continue
            if pend:
                flush(si, pend)
            pend = [c, s, e, 1, d, name_id(t, n)]
        if pend:
            flush(si, pend)

    merged_out = []
    for (sid, t), (k, dsum) in sorted(merged.items(), key=lambda kv: -kv[1][0]):
        merged_out.append({"s": sidx[sid], "type": t, "k": k, "dur_us": round(dsum, 3)})

    doc = {
        "arm": a.arm,
        "run": a.run,
        "src": csv_path,
        "generated_by": "tools/kt_offload/prof/trace_slice.py",
        "clock": ("trace/task_time ts is CLOCK_REALTIME us. t0 below is rebased to this "
                  "slice's own origin (origin_abs_us = first MODEL_EXECUTE of the slice). "
                  "Slices from different runs are HOURS apart on the absolute axis and must "
                  "NOT be drawn on one shared axis -- compare shapes per arm only."),
        "origin_abs_us": origin,
        "span_us": round(win_t1 - win_t0, 2),
        "step_index": idx,
        "step_pick": why,
        "classes": CLASS_NAMES,
        "streams": [
            {"id": s, "role": role[s],
             "n_tasks": sum(1 for o in out_tasks if o["s"] == sidx[s])}
            for s in sids
        ],
        "streams_not_drawn": [
            {"id": s, "role": role[s], "n_tasks_in_slice": len(raw[s])} for s in dropped
        ],
        "names": names,
        "name_class": name_cls,
        "steps": [{"t0": round(x - origin, 2), "t1": round(y - origin, 2)} for x, y in step_edges],
        "tasks": out_tasks,
        "merged": merged_out,
        "merged_total": {
            "k": sum(m["k"] for m in merged_out),
            "dur_us": round(sum(m["dur_us"] for m in merged_out), 3),
            "rule": "tasks with dur < %.2f us are not emitted individually; "
                    "HOSTFUNC_CALLBACK is exempt (0.007 us nodes kept as markers)" % a.frag_us,
        },
        "coalesce": {
            "gap_us": a.coalesce_gap,
            "classes": [CLASS_NAMES[c] for c in sorted(COALESCE_CLASSES)],
            "note": "a task with k>1 is k same-class tasks folded into one block spanning "
                    "the same wall-clock; seam classes are never coalesced",
        },
        "checks": checks,
    }
    with open(a.out, "w") as f:
        json.dump(doc, f, separators=(",", ":"), ensure_ascii=False)
    sz = os.path.getsize(a.out)
    print(f"{a.arm}: steps {idx} span {(win_t1-win_t0)/1000:.1f} ms, "
          f"{len(out_tasks)} blocks, {len(names)} names, "
          f"merged {doc['merged_total']['k']} frags ({doc['merged_total']['dur_us']:.1f} us) "
          f"-> {a.out} {sz/1024:.1f} KB", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
