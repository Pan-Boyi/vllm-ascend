# SPDX-License-Identifier: Apache-2.0
"""The offline seam parser (tools/kt_offload/prof/seam.py) as an instrument.

The first on-card run of §4.20 produced two numbers that were wrong *in the
tool*, not on the card, and both are pinned here with a mutation arm:

  * the red verdict came from "sum of the segment medians vs the median of
    cb". Per callback the segments are consecutive differences of one clock,
    so they add up to cb exactly; medians do not add, so that comparison went
    red (24/40 layers) on a healthy run. What replaces it is a set of
    necessary conditions of the per-callback identity -- and a mutation that
    really breaks the segmentation has to make them red;
  * ``--nw-csv`` paired position j with layer j while the first 2
    NOTIFY_WAIT of each step are not ours. The slicing now lives in the tool
    and carries an alignment check; the mutation arm is a csv whose extra
    waits sit at the *end*, where dropping the first 2 is wrong, and the tool
    must refuse to print A+F_dev rather than print it shifted.

Plus the window (``--since/--until/--bytes``): a profile covers the last 8 s
of a service whose seam records cover all of it.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import random
from pathlib import Path

import pytest

SEAM_PY = Path(__file__).resolve().parents[3] / "tools" / "kt_offload" / "prof" / "seam.py"


def _load():
    spec = importlib.util.spec_from_file_location("seam_tool_ut", SEAM_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tool = _load()


# ------------------------------------------------------- ① the verdict itself


def _healthy(layer_scale: float = 1.0):
    """One layer's med/p90 as a healthy split run reports them.

    Every part's median and p90 is below cb's, and cb's median is below the
    sum of the lumped p90s -- which is what "the parts live inside the
    callback" looks like once only quantiles survive.
    """
    med = {"pre": 2.8, "kt_pre": 15.5, "submit": 2.4, "between": 1.4, "sync": 255.0 * layer_scale,
           "kt_tail": 2.4, "post": 2.6, "kt": 276.7 * layer_scale, "cb": 286.5 * layer_scale,
           "gap": 40200.0}
    p90 = {"pre": 4.0, "kt_pre": 22.0, "submit": 3.5, "between": 2.2, "sync": 320.0 * layer_scale,
           "kt_tail": 3.6, "post": 3.8, "kt": 350.0 * layer_scale, "cb": 356.0 * layer_scale,
           "gap": 41000.0}
    return med, p90


def test_a_healthy_layer_is_green_although_the_medians_do_not_add():
    med, p90 = _healthy()
    resid = tool.reconcile(med, True)
    assert abs(resid["residual_us"]) > 2.0, "the fixture must show the non-additivity, or it proves nothing"
    assert tool.check_layer(med, p90, True)["ok"], "the old criterion's false red must be gone"


def test_the_old_residual_is_still_printed_but_is_not_the_verdict():
    med, p90 = _healthy()
    out = tool.check_layer(med, p90, True)
    assert out["violations"] == []
    assert tool.reconcile(med, True)["residual_us"] == pytest.approx(286.5 - (2.8 + 15.5 + 2.4 + 1.4 + 255.0 + 2.4 + 2.6), abs=1e-6)


def test_a_swallowed_neighbour_arrives_as_an_unfilled_middle_and_is_red():
    """The record-side mutation of test_seam_timing.py, seen from here.

    Moving "submit returned" to after the sync makes the stamps non-monotonic,
    so seam_timing.record() refuses to split that callback and stores zeros
    for kt_pre..kt_tail (``kt`` and ``cb`` stay whole). That is what reaches
    this parser, and it must be red.
    """
    med, p90 = _healthy()
    for k in ("kt_pre", "submit", "between", "sync", "kt_tail"):
        med[k] = p90[k] = 0.0
    assert not tool.check_layer(med, p90, True)["ok"]


def test_an_overlap_that_still_fits_inside_the_callback_is_NOT_caught_here():
    """The honest floor. If some future record side ever reported an overlap
    that stayed within cb at every quantile, medians and p90s alone cannot
    reject it -- only a per-callback residual can. Pinned so nobody reads the
    green verdict as more than it is."""
    med, p90 = _healthy()
    med["submit"], p90["submit"] = 100.0, 120.0   # counted twice, still < kt
    assert tool.check_layer(med, p90, True)["ok"]


def test_callbacks_whose_middle_was_never_filled_are_red():
    """A split run into which unsplit rows leak: kt and cb stay whole while
    kt_pre..kt_tail are zeros (seam_timing.record's fallback)."""
    med, p90 = _healthy()
    for k in ("kt_pre", "submit", "between", "sync", "kt_tail"):
        med[k] = p90[k] = 0.0
    v = tool.check_layer(med, p90, True)
    assert not v["ok"]
    assert any("split run" in s or "sum p90(middle)" in s for s in v["violations"]), v


def test_a_hole_bigger_than_the_slack_is_red_and_the_slack_is_reported():
    """Residual amplified on the callbacks of one layer: time inside cb that
    belongs to no segment. Detected once it exceeds that layer's slack, which
    the tool prints for every layer instead of pretending it has no floor."""
    med, p90 = _healthy()
    slack = tool.check_layer(med, p90, True)["slack_us"]
    assert slack > 0
    small = dict(med, cb=med["cb"] + slack * 0.5)
    assert tool.check_layer(small, dict(p90, cb=p90["cb"] + slack * 0.5), True)["ok"], (
        "below the slack the union bound cannot see it -- this is the documented floor")
    big = dict(med, cb=med["cb"] + slack * 2)
    v = tool.check_layer(big, dict(p90, cb=p90["cb"] + slack * 2), True)
    assert not v["ok"] and any("union" in s or "p90(pre)" in s for s in v["violations"]), v


def test_a_per_callback_residual_from_the_record_side_is_used_when_it_exists():
    """Today's records carry only median/p90, so one bad callback is invisible.
    If the record side ever counts them, the check is exact and takes over."""
    med, p90 = _healthy()
    assert tool.check_layer(med, p90, True, {"n_unsplit": 0, "resid_us": {"max": 0.0}})["ok"]
    assert not tool.check_layer(med, p90, True, {"n_unsplit": 3})["ok"]
    assert not tool.check_layer(med, p90, True, {"resid_us": {"max": 41.0}})["ok"]


# ---------------------------------------------------- ② the NOTIFY_WAIT slice


def _write_csv(path: Path, steps, head_extra=2, extra_at_end=False):
    """A task_time.csv: one MODEL_EXECUTE per step, then its NOTIFY_WAIT."""
    lines = ["Device_id,kernel_name,kernel_type,stream_id,task_id,task_time(us),task_start(us),task_stop(us)"]
    t = 1_000_000.0
    for durations in steps:
        lines.append(f"2,N/A,MODEL_EXECUTE,65535,1,3.0,{t:.3f},{t + 3:.3f}")
        t += 10.0
        head = [40.0 + 10 * i for i in range(head_extra)]
        seq = (durations + head) if extra_at_end else (head + durations)
        for d in seq:
            lines.append(f"2,N/A,NOTIFY_WAIT,244,1,{d:.3f},{t:.3f},{t + d:.3f}")
            t += d + 1.0
        t += 5000.0
    path.write_text("\n".join(lines) + "\n")


def _synthetic_run(n_layers=8, n_steps=30, seed=7):
    """cb per layer, and a NOTIFY_WAIT per layer that is cb plus dispatch."""
    rng = random.Random(seed)
    # deliberately NOT monotone in the layer index: a ramp would correlate at
    # every shift and the alignment check would have nothing to decide.
    cb = {i: round(rng.uniform(200.0, 340.0), 1) for i in range(n_layers)}
    steps = [[cb[i] + 80.0 + rng.uniform(-5, 5) for i in range(n_layers)] for _ in range(n_steps)]
    return cb, steps


def test_the_slice_drops_the_two_waits_that_are_not_ours(tmp_path):
    cb, steps = _synthetic_run()
    csv_path = tmp_path / "task_time.csv"
    _write_csv(csv_path, steps)
    nw, meta, shifted = tool.notify_wait_ours(str(csv_path), len(cb), skip=2)
    assert meta["per_step"] == len(cb) + 2 and meta["steps_used"] == len(steps)
    for i in cb:
        assert nw[i] == pytest.approx(cb[i] + 80.0, abs=6.0)
    align = tool.check_alignment(shifted, cb, 2)
    assert align["ok"] and align["rho"] > 0.9, align


def test_a_truncated_step_cannot_renumber_the_layers(tmp_path):
    cb, steps = _synthetic_run()
    steps = steps + [[300.0, 310.0]]  # a step the profile only caught half of
    csv_path = tmp_path / "task_time.csv"
    _write_csv(csv_path, steps)
    _nw, meta, _sh = tool.notify_wait_ours(str(csv_path), len(cb), skip=2)
    assert meta["steps_used"] == len(steps) - 1
    assert meta["shapes"].get(4) == 1  # 2 head waits + the 2 that were caught


def test_the_mutation_arm_two_positions_off_is_refused(tmp_path):
    """The first run's bug: the extra waits are at the end, so dropping the
    first two shifts every layer. The alignment check must refuse."""
    cb, steps = _synthetic_run()
    csv_path = tmp_path / "task_time.csv"
    _write_csv(csv_path, steps, extra_at_end=True)
    _nw, _meta, shifted = tool.notify_wait_ours(str(csv_path), len(cb), skip=2)
    bad = tool.check_alignment(shifted, cb, 2)
    good = tool.check_alignment(shifted, cb, 0)
    assert not bad["ok"], bad
    assert good["ok"], good


def test_alignment_needs_a_margin_not_just_a_threshold(tmp_path):
    """Two shifts that both correlate must not be accepted as one answer."""
    cb = {i: 200.0 + 20.0 * i for i in range(8)}
    flat = {s: {i: 300.0 + 0.1 * i for i in range(8)} for s in (0, 1, 2)}
    out = tool.check_alignment(flat, cb, 2)
    assert not out["ok"] and out["rho"] == pytest.approx(1.0), out


# --------------------------------------------------------------- ③ the window


def _log_line(stamp: str, interval=5.0, cb=282.1, layers=2):
    rec = {
        "interval_s": interval, "split": True,
        "additive": list(tool.ADDITIVE),
        "layers": {
            str(i): {"n": 100,
                     "med_us": dict(zip(tool.SEGMENTS, [2.8, 15.5, 2.4, 1.4, 255.0, 2.4, 2.6, 276.7, cb, 40200.0])),
                     "p90_us": dict(zip(tool.SEGMENTS, [4.0, 22.0, 3.5, 2.2, 320.0, 3.6, 3.8, 350.0, 356.0, 41000.0]))}
            for i in range(layers)},
    }
    return f"(EngineCore pid=1) INFO {stamp} [seam_timing.py:377] [kt-seam] " + json.dumps(rec, separators=(",", ":"))


def test_the_window_keeps_only_the_records_covering_it():
    text = "\n".join(_log_line(f"09-22 15:0{m}:{s:02d}") for m, s in [(0, 11), (1, 46), (1, 51), (1, 56), (2, 11)])
    recs, dropped = tool.parse_records(text)
    assert dropped == 0 and len(recs) == 5
    since = tool.parse_window_arg("2026-09-22T15:01:48Z", 2026)
    # 🔴 the date is written out on BOTH ends. A bare "15:01:56" means "today"
    # (documented in parse_window_arg), so this test used to pass only while
    # today was 09-22 and went red by itself at midnight on 2026-09-23 --
    # which is also the trap for a human analysing at 00:05 a capture taken at
    # 23:15. The convenience stays; nothing may depend on it.
    until = tool.parse_window_arg("2026-09-22T15:01:56Z", 2026)
    kept, note = tool.select_window(recs, since, until, 2026)
    stamps = [r["_stamp"][2] for r in kept]
    assert stamps == [15 * 3600 + 1 * 60 + 51, 15 * 3600 + 1 * 60 + 56], (stamps, note)
    assert "2/5" in note


def test_a_bare_time_means_today_and_says_which_day_that_is():
    """The convenience form, pinned so it cannot silently mean another day."""
    today = datetime.date.today()
    mon, day, secs = tool.parse_window_arg("15:01:56", 2026)
    assert (mon, day) == (today.month, today.day)
    assert secs == 15 * 3600 + 1 * 60 + 56


def test_no_window_means_the_whole_log_and_says_so():
    text = "\n".join(_log_line("09-22 15:01:51") for _ in range(3))
    recs, _ = tool.parse_records(text)
    kept, note = tool.select_window(recs, None, None, 2026)
    assert len(kept) == 3 and "whole log (3 records)" in note


def test_a_byte_range_reads_only_that_slice(tmp_path):
    first, second = _log_line("09-22 15:01:51"), _log_line("09-22 15:01:56", cb=999.0)
    p = tmp_path / "serve.log"
    blob = (first + "\n" + second + "\n").encode()
    p.write_bytes(blob)
    text = tool.read_text(str(p), f"{len(first) + 1}:{len(blob)}")
    recs, _ = tool.parse_records(text)
    assert len(recs) == 1 and recs[0]["layers"]["0"]["med_us"]["cb"] == 999.0


# --------------------------------------------------------------- end to end


def _fixture(tmp_path, extra_at_end=False):
    cb, steps = _synthetic_run(n_layers=8)
    csv_path = tmp_path / "task_time.csv"
    _write_csv(csv_path, steps, extra_at_end=extra_at_end)
    lines = []
    for stamp, keep in (("15:00:11", False), ("15:01:51", True), ("15:01:56", True)):
        rec = {"interval_s": 5.0, "split": True, "additive": list(tool.ADDITIVE), "layers": {}}
        for i in cb:
            scale = cb[i] / 286.5 if keep else 3.0
            med, p90 = _healthy(scale)
            med["cb"] = cb[i]
            p90["cb"] = cb[i] * 1.26
            rec["layers"][str(i)] = {"n": 100, "med_us": med, "p90_us": p90}
        lines.append(f"(EngineCore pid=1) INFO 09-22 {stamp} [seam_timing.py:377] [kt-seam] "
                     + json.dumps(rec, separators=(",", ":")))
    log = tmp_path / "serve.log"
    log.write_text("\n".join(lines) + "\n")
    return log, csv_path


def test_end_to_end_green_and_the_window_is_honoured(tmp_path, capsys):
    log, csv_path = _fixture(tmp_path)
    rc = tool.main([str(log), "--nw-csv", str(csv_path), "--layers", "8",
                    "--since", "2026-09-22T15:01:48Z", "--until", "2026-09-22T15:01:56Z",
                    "--year", "2026", "--json-out", str(tmp_path / "out.json")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "2/3 records kept" in out
    assert "✅ segmentation check ok" in out
    got = json.load(open(tmp_path / "out.json"))
    assert got["alignment"]["ok"] and got["alignment"]["chosen_shift"] == 2
    assert got["totals"]["A_plus_F_dev_ms"] == pytest.approx(8 * 80.0 / 1000.0, abs=0.02)


def test_end_to_end_refuses_a_plus_f_when_the_alignment_is_wrong(tmp_path, capsys):
    log, csv_path = _fixture(tmp_path, extra_at_end=True)
    rc = tool.main([str(log), "--nw-csv", str(csv_path), "--layers", "8",
                    "--since", "2026-09-22T15:01:48Z", "--until", "2026-09-22T15:01:56Z", "--year", "2026"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "refusing the NW / A+F_dev columns" in out
    tail = out.split("whole net")[1]
    assert "A+F_dev: refused" in tail and "  A+F_dev " not in tail, tail


# ------------------------------------------------- ④ A and F, told apart (P0)
#
# §4.22 P0: the callback also stamps CLOCK_REALTIME at entry and at exit, so A
# (device reaches the node -> the callback runs) and F (the callback returns ->
# the device's NOTIFY_WAIT ends) stop being one 78 us number. The join is by
# time, per callback, against the same profile -- which means a stamp taken in
# the wrong place, or on the wrong clock, has to be caught here. Both arms are
# below; a split that cannot fail its own audit would just be two new columns.

AF_BASE_NS = 1_790_089_308_000_000_000  # a real profile's ts, so the span is realistic


def _af_run(n_layers=8, n_steps=30, seed=11, a_us=30.0, f_us=20.0, entry_extra_ns=0, clock_shift_ns=0):
    """One synthetic run: the csv rows, the callbacks, and the cb per layer.

    Every layer gets its own cb (not monotone in the index: the NOTIFY_WAIT
    alignment check must have something to decide), and every callback sits
    inside its pair with a known A and F.  ``entry_extra_ns`` is the mutation
    "the entry stamp was taken later than the entry"; ``clock_shift_ns`` is
    "the stamps are not CLOCK_REALTIME at all".
    """
    rng = random.Random(seed)
    cb_us = {i: round(rng.uniform(200.0, 340.0), 1) for i in range(n_layers)}
    rows, calls = [], []
    t = AF_BASE_NS
    for _ in range(n_steps):
        rows.append(("MODEL_EXECUTE", "65535", t, 3_000))
        t += 10_000
        for head in (40_000, 50_000):  # the two waits at the head of a step that are not ours
            rows.append(("HOSTFUNC_CALLBACK", "244", t, 7))
            rows.append(("NOTIFY_WAIT", "244", t, head))
            t += head + 1_000
        for i in range(n_layers):
            cb_ns = int(cb_us[i] * 1000) + rng.randint(-2000, 2000)
            a_ns, f_ns = int(a_us * 1000), int(f_us * 1000)
            hf = t
            entry = hf + a_ns
            exit_ = entry + cb_ns
            nw_end = exit_ + f_ns
            rows.append(("HOSTFUNC_CALLBACK", "244", hf, 7))
            rows.append(("NOTIFY_WAIT", "244", hf, nw_end - hf))
            calls.append((i, entry + entry_extra_ns + clock_shift_ns, exit_ + clock_shift_ns, cb_ns))
            t = nw_end + 1_000
        t += 5_000_000
    return rows, calls, cb_us


def _write_af_csv(path, rows):
    out = ["Device_id,kernel_name,kernel_type,stream_id,task_id,task_time(us),task_start(us),task_stop(us)"]
    for kind, stream, ts_ns, dur_ns in rows:
        out.append(f"2,N/A,{kind},{stream},1,{dur_ns / 1000.0:.3f},{ts_ns / 1000.0:.3f}\t,{(ts_ns + dur_ns) / 1000.0:.3f}\t")
    path.write_text("\n".join(out) + "\n")


def _rt_lines(calls, stamp="09-22 15:01:51"):
    by_layer = {}
    for layer, entry, exit_, cb in calls:
        d = by_layer.setdefault(layer, {"layer": layer, "entry_realtime_ns": [], "exit_realtime_ns": [], "cb_ns": []})
        d["entry_realtime_ns"].append(entry)
        d["exit_realtime_ns"].append(exit_)
        d["cb_ns"].append(cb)
    return [f"(EngineCore pid=1) INFO {stamp} [seam_timing.py:401] [kt-seam-rt] "
            + json.dumps(d, separators=(",", ":"), sort_keys=True) for d in by_layer.values()]


def test_the_pairs_are_one_hostfunc_and_the_wait_that_follows_it(tmp_path):
    rows, _calls, cb = _af_run(n_layers=4, n_steps=3)
    csv_path = tmp_path / "task_time.csv"
    _write_af_csv(csv_path, rows)
    pairs = tool.hostfunc_pairs(str(csv_path))
    assert len(pairs) == 3 * (len(cb) + 2), "every HOSTFUNC_CALLBACK must find its NOTIFY_WAIT"
    for hf, nw_ts, nw_end in pairs:
        assert hf == nw_ts < nw_end
    # exact ns, not float us: at 1.79e15 us a double only resolves 0.25 us
    assert pairs[0][0] % 1000 == 0 and isinstance(pairs[0][0], int)


def test_a_and_f_come_back_as_they_were_put_in(tmp_path):
    rows, calls, cb = _af_run(a_us=30.0, f_us=20.0)
    csv_path = tmp_path / "task_time.csv"
    _write_af_csv(csv_path, rows)
    out = tool.split_a_f(calls, tool.hostfunc_pairs(str(csv_path)))
    assert out["ok"], out["why"]
    assert out["unmatched"] == 0 and out["pairs_taken_twice"] == 0
    for layer, v in out["layers"].items():
        assert v["A_med_us"] == pytest.approx(30.0, abs=0.2), (layer, v)
        assert v["F_med_us"] == pytest.approx(20.0, abs=0.2), (layer, v)
        # and the split adds back up to the column that existed before it
        assert v["diff_med_us"] == pytest.approx(0.0, abs=0.05), (layer, v)
    assert out["totals_ms"]["A_plus_F"] == pytest.approx(len(cb) * 50.0 / 1000.0, abs=0.01)


def test_mutation_the_entry_stamp_taken_after_kt_is_red(tmp_path):
    """The arm that matters: the realtime stamp moved from the callback's
    entry to after kt ran. A looks ~250 us bigger and the reconciliation
    against NOTIFY_WAIT - cb must go red rather than print the new number."""
    rows, calls, _cb = _af_run(entry_extra_ns=250_000)
    csv_path = tmp_path / "task_time.csv"
    _write_af_csv(csv_path, rows)
    pairs = tool.hostfunc_pairs(str(csv_path))
    honest = tool.split_a_f([(l, e - 250_000, x, c) for l, e, x, c in calls], pairs)
    assert honest["ok"], honest["why"]  # the control arm of the same fixture
    out = tool.split_a_f(calls, pairs)
    assert not out["ok"]
    assert out["layers"] == {}, "a refused check must not leave the columns behind"
    assert any("med((A+F) - (NW - cb))" in w for w in out["why"]), out["why"]


def test_mutation_perf_counter_where_realtime_was_meant_is_caught(tmp_path):
    """Two clock sources mixed: perf_counter_ns is ns since boot, so the
    stamps land nowhere near the profile's span and nothing may be printed."""
    rows, calls, _cb = _af_run(clock_shift_ns=-1_789_000_000_000_000_000)
    csv_path = tmp_path / "task_time.csv"
    _write_af_csv(csv_path, rows)
    out = tool.split_a_f(calls, tool.hostfunc_pairs(str(csv_path)))
    assert not out["ok"] and out["n_in_span"] == 0
    assert any("not the same run, or not CLOCK_REALTIME" in w for w in out["why"]), out["why"]


def test_a_callback_outside_every_window_is_not_silently_dropped(tmp_path):
    """Between two pairs there is device time that belongs to no callback; a
    stamp landing there means the join is wrong, and enough of them refuse."""
    rows, calls, _cb = _af_run(n_layers=4, n_steps=10)
    csv_path = tmp_path / "task_time.csv"
    _write_af_csv(csv_path, rows)
    pairs = tool.hostfunc_pairs(str(csv_path))
    moved = [(l, e + 3_000_000, x + 3_000_000, c) if k % 3 == 0 else (l, e, x, c)
             for k, (l, e, x, c) in enumerate(calls)]
    out = tool.split_a_f(moved, pairs)
    assert not out["ok"] and out["unmatched"] > 0
    assert any("sit in no HOSTFUNC..NOTIFY_WAIT window" in w for w in out["why"]), out["why"]


def _af_log(tmp_path, calls, cb_us, stamp="15:01:51"):
    lines = []
    rec = {"interval_s": 5.0, "split": True, "additive": list(tool.ADDITIVE), "layers": {}}
    for i, c in cb_us.items():
        med, p90 = _healthy(c / 286.5)
        med["cb"] = c
        p90["cb"] = c * 1.26
        rec["layers"][str(i)] = {"n": 100, "med_us": med, "p90_us": p90,
                                 "n_unsplit": 0, "resid_us": {"max": 0.0}}
    lines.append(f"(EngineCore pid=1) INFO 09-22 {stamp} [seam_timing.py:377] [kt-seam] "
                 + json.dumps(rec, separators=(",", ":")))
    lines += _rt_lines(calls, f"09-22 {stamp}")
    log = tmp_path / "serve.log"
    log.write_text("\n".join(lines) + "\n")
    return log


def test_end_to_end_prints_a_and_f_and_the_json_carries_them(tmp_path, capsys):
    rows, calls, cb_us = _af_run()
    csv_path = tmp_path / "task_time.csv"
    _write_af_csv(csv_path, rows)
    log = _af_log(tmp_path, calls, cb_us)
    rc = tool.main([str(log), "--nw-csv", str(csv_path), "--layers", "8", "--year", "2026",
                    "--json-out", str(tmp_path / "out.json")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "A / F split" in out and "✅ A/F reconcile" in out
    got = json.load(open(tmp_path / "out.json"))["a_f"]
    assert got["ok"] and got["totals_ms"]["A"] == pytest.approx(8 * 30.0 / 1000.0, abs=0.01)
    assert got["totals_ms"]["F"] == pytest.approx(8 * 20.0 / 1000.0, abs=0.01)


def test_end_to_end_refuses_the_two_columns_and_returns_one(tmp_path, capsys):
    rows, calls, cb_us = _af_run(entry_extra_ns=250_000)
    csv_path = tmp_path / "task_time.csv"
    _write_af_csv(csv_path, rows)
    log = _af_log(tmp_path, calls, cb_us)
    rc = tool.main([str(log), "--nw-csv", str(csv_path), "--layers", "8", "--year", "2026"])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "refusing the A / F columns" in out
    assert "A_med" not in out, "a refused split must not print its columns"


def test_a_log_without_the_realtime_lines_still_works(tmp_path, capsys):
    """Records from before P0: the A/F block stays silent, nothing goes red."""
    log, csv_path = _fixture(tmp_path)
    rc = tool.main([str(log), "--nw-csv", str(csv_path), "--layers", "8",
                    "--since", "2026-09-22T15:01:48Z", "--until", "2026-09-22T15:01:56Z", "--year", "2026"])
    out = capsys.readouterr().out
    assert rc == 0 and "A / F split" not in out, out


# ------------------------------- ⑤ the exact per-callback fields (§4.20.1)


def test_the_record_sides_own_counts_reach_the_verdict(tmp_path, capsys):
    """n_unsplit / resid_us.max are exact; when the records carry them they
    decide, instead of the median-only necessary conditions."""
    _rows, calls, cb_us = _af_run(n_layers=2, n_steps=2)
    log = _af_log(tmp_path, calls, cb_us)
    text = log.read_text().replace('"n_unsplit":0', '"n_unsplit":7', 1)
    log.write_text(text)
    recs, _ = tool.parse_records(text)
    merged = tool.merge(recs)
    assert merged["extra"][0]["n_unsplit"] == 7
    rc = tool.main([str(log), "--layers", "2", "--year", "2026"])
    out = capsys.readouterr().out
    assert rc == 1 and "n_unsplit=7" in out, out


def test_the_zero_middles_of_our_own_split_are_not_a_missing_probe():
    """🔴 The forked (and packed) path stamps submit/sync itself, so
    ``between`` and ``kt_tail`` are zero BY CONSTRUCTION -- sync is called
    straight after submit, and kt_tail ends where sync ends. The
    "a zero middle means the probe did not run" rule was written for the
    probe-based split and turned the first healthy forked run red on 40/40
    layers, while that run's exact counters said n_unsplit=0 over 661,285
    callbacks and max|resid| = 0.0 us. So the proxy applies only where the
    exact count is missing.
    """
    med, p90 = _healthy()
    for k in ("between", "kt_tail"):
        med[k] = p90[k] = 0.0
    exact = tool.check_layer(med, p90, True, {"n_unsplit": 0, "resid_us": {"max": 0.0}})
    assert exact["ok"], exact["violations"]
    # a record from before the exact counters existed keeps the proxy
    old = tool.check_layer(med, p90, True)
    assert not old["ok"] and any("split run" in s for s in old["violations"]), old
    # and the exact counter still decides when it says there IS contamination
    assert not tool.check_layer(med, p90, True, {"n_unsplit": 2})["ok"]


def test_a_constant_clock_offset_is_named_as_such_and_bounded(tmp_path):
    """🔴 An offset between the profile's timebase and CLOCK_REALTIME moves A
    up and F down by the same amount: A+F (and ``diff``) survive, the split
    does not. Its fingerprint is med(F) < 0 on every layer with a small
    spread; a mis-pairing scatters instead. Seen on the card 2026-09-22 --
    med(F) = -84.6 us on 40/40 layers. The tool must say which of the two it
    is, and bound the offset, instead of listing forty layer complaints.
    """
    rows, calls, _cb = _af_run(a_us=30.0, f_us=20.0, clock_shift_ns=100_000)
    csv_path = tmp_path / "task_time.csv"
    _write_af_csv(csv_path, rows)
    out = tool.split_a_f(calls, tool.hostfunc_pairs(str(csv_path)))
    assert not out["ok"] and out["layers"] == {}
    assert "CONSTANT offset" in out["why"][0], out["why"]
    lo, hi = out["clock_offset_bound_us"]
    assert lo <= 100.0 <= hi, (lo, hi)
    # the control arm: no offset, same fixture, and the split comes back
    rows0, calls0, _ = _af_run(a_us=30.0, f_us=20.0, clock_shift_ns=0)
    csv0 = tmp_path / "t0.csv"
    _write_af_csv(csv0, rows0)
    assert tool.split_a_f(calls0, tool.hostfunc_pairs(str(csv0)))["ok"]


def _write_forked_csv(path: Path, steps, ours_stream="336", other_stream="337", head_extra=2,
                      interleave=False):
    """A forked-arm task_time.csv: OUR waits on the side stream, the two that
    are not ours on the main one. ``interleave`` drops the foreign pair into
    the middle of ours -- which the position slice would then get wrong and
    the stream selection still gets right."""
    lines = ["Device_id,kernel_name,kernel_type,stream_id,task_id,task_time(us),task_start(us),task_stop(us)"]
    t = 1_000_000.0
    for durations in steps:
        lines.append(f"2,N/A,MODEL_EXECUTE,65535,1,3.0,{t:.3f},{t + 3:.3f}")
        t += 10.0
        head = [(other_stream, 40.0 + 10 * i) for i in range(head_extra)]
        ours = [(ours_stream, d) for d in durations]
        seq = (ours[:2] + head + ours[2:]) if interleave else (head + ours)
        for sid, d in seq:
            lines.append(f"2,N/A,NOTIFY_WAIT,{sid},1,{d:.3f},{t:.3f},{t + d:.3f}")
            t += d + 1.0
        t += 5000.0
    path.write_text("\n".join(lines) + "\n")


def test_the_stream_that_holds_exactly_our_layers_is_found_and_used(tmp_path):
    """🔴 The forked path puts our host-func nodes on the side stream and
    leaves the two that are not ours on the main one, so they separate by
    stream id -- position j is then layer j with nothing to skip. The position
    slice only works there while the foreign pair happens to stay at the head
    of every step, which nothing enforces once the two streams run at once.
    """
    cb, steps = _synthetic_run()
    csv_path = tmp_path / "task_time.csv"
    _write_forked_csv(csv_path, steps, interleave=True)
    found = tool.notify_wait_stream_auto(str(csv_path), len(cb))
    assert found and found["stream"] == "336", found
    assert found["steps_with_exactly_layers"] == found["steps"]
    nw, meta, shifted = tool.notify_wait_ours(str(csv_path), len(cb), skip=0, stream="336")
    assert meta["per_step"] == len(cb) and list(shifted) == [0]
    assert tool.check_alignment(shifted, cb, 0)["ok"]
    for i in cb:
        assert nw[i] == pytest.approx(cb[i] + 80.0, abs=6.0), (i, nw[i], cb[i])
    # 🔴 the control, and the reason the selector exists: on the SAME csv the
    # position slice hands back the two foreign waits as layers 0 and 1 --
    # and the alignment check does not necessarily catch it, because the other
    # 6 layers still line up. A structurally correct pick beats a statistical
    # one that can be outvoted.
    bad, _m, _bs = tool.notify_wait_ours(str(csv_path), len(cb), skip=2)
    assert bad and bad != nw, "the fixture must actually separate the two readings"
    assert bad[0] != pytest.approx(cb[0] + 80.0, abs=6.0), (bad[0], cb[0])
    assert bad[1] != pytest.approx(cb[1] + 80.0, abs=6.0), (bad[1], cb[1])


def test_one_stream_holding_everything_is_left_to_the_position_slice(tmp_path):
    """The serial arm: one stream, layers + 2 per step, nothing to pick."""
    cb, steps = _synthetic_run()
    csv_path = tmp_path / "task_time.csv"
    _write_csv(csv_path, steps)
    assert tool.notify_wait_stream_auto(str(csv_path), len(cb)) == {}


def test_an_ambiguous_split_is_not_guessed_at(tmp_path):
    """Two streams both holding exactly n_layers per step: the selector has no
    ground to choose, so it declines instead of picking one."""
    cb, steps = _synthetic_run()
    csv_path = tmp_path / "task_time.csv"
    _write_forked_csv(csv_path, steps, head_extra=0)
    # a second stream with the same shape
    text = csv_path.read_text().splitlines()
    dup = [ln.replace(",NOTIFY_WAIT,336,", ",NOTIFY_WAIT,338,") for ln in text[1:] if ",NOTIFY_WAIT,336," in ln]
    csv_path.write_text("\n".join(text + dup) + "\n")
    assert tool.notify_wait_stream_auto(str(csv_path), len(cb)) == {}
