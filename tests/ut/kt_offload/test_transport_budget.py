# SPDX-License-Identifier: Apache-2.0
"""tools/kt_offload/transport_budget.py -- byte budget and profile attribution."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]


def _mod():
    spec = importlib.util.spec_from_file_location("transport_budget", ROOT / "tools/kt_offload/transport_budget.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_layer_bytes_dsv41_decode():
    tb = _mod()
    b = tb.layer_bytes(tokens=1, hidden=5120, topk=6)
    assert b["d2h"] == 5120 * 2 + 6 * 8 + 6 * 4  # hidden bf16 + ids int64 + weights fp32
    assert b["h2d"] == 5120 * 2
    assert b["total"] == 20552


def test_budget_says_the_bytes_are_not_the_cost():
    tb = _mod()
    r = tb.budget(tokens=1, layers=40, hidden=5120, topk=6, gbps=20.0)
    assert r["bytes_per_step"] == 40 * 20552
    assert r["us_per_step_at_bandwidth"] == pytest.approx(41.1, abs=0.5)  # microseconds, not milliseconds
    assert r["ops_per_step"] == 120
    # 7.5 ms observed against 41 us of bytes => a fixed per-operation cost
    assert tb.fixed_cost_per_op(7500.0, 40) == pytest.approx(62.5, abs=0.1)


def test_union_busy_never_double_counts_parallel_streams():
    tb = _mod()
    busy, span, gaps = tb._union([(0.0, 10.0), (2.0, 8.0), (20.0, 25.0)])
    assert busy == 15.0 and span == 25.0 and gaps == [10.0]  # not 10+6+5 = 21
    assert tb._union([])[0] == 0.0


def _csv(tmp_path, rows):
    p = tmp_path / "kernel_details.csv"
    p.write_text("Name,Start Time(us),Duration(us)\n" + "".join(f"{n},{s},{d}\n" for n, s, d in rows))
    return str(p)


def test_measure_attributes_copies_host_func_and_gaps(tmp_path):
    tb = _mod()
    rows = [
        ("aclnnMatmul", 0, 100),
        ("Memcpy DtoH", 100, 10),
        ("aclrtLaunchCallback", 110, 400),  # the host-func node
        ("Memcpy HtoD", 510, 10),
        ("aclnnMatmul", 1000, 50),  # a 480 us gap before this
    ]
    r = tb.measure(_csv(tmp_path, rows), steps=1, gap_us=100.0)
    assert r["busy_us_union"] == 570.0 and r["span_us"] == 1050.0
    assert r["by_kind"]["d2h"] == {"count": 1, "us": 10.0}
    assert r["by_kind"]["h2d"] == {"count": 1, "us": 10.0}
    assert r["by_kind"]["host_func"] == {"count": 1, "us": 400.0}
    assert r["gaps_over_threshold"] == 1 and r["gap_us_over_threshold"] == 480.0
    assert r["idle_us_per_step"] == pytest.approx(480.0)


def test_measure_sum_would_double_count(tmp_path):
    """Why busy is a union: two streams running at the same time.

    The sum of durations exceeds the span; only the union can produce the
    'device busy 57.1%' style number the plan reports.
    """
    tb = _mod()
    rows = [("aclnnA", 0, 100), ("aclnnB", 0, 100), ("Memcpy DtoH", 0, 100)]
    r = tb.measure(_csv(tmp_path, rows), steps=1, gap_us=100.0)
    assert r["busy_us_union"] == 100.0
    assert sum(k["us"] for k in r["by_kind"].values()) == 300.0
