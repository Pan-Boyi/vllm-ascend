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
    until = tool.parse_window_arg("15:01:56", 2026)
    kept, note = tool.select_window(recs, since, until, 2026)
    stamps = [r["_stamp"][2] for r in kept]
    assert stamps == [15 * 3600 + 1 * 60 + 51, 15 * 3600 + 1 * 60 + 56], (stamps, note)
    assert "2/5" in note


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
