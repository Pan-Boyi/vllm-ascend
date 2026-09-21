# SPDX-License-Identifier: Apache-2.0
"""Two artifacts from different builds must not be subtracted.

The rule used to live in someone's memory: "those numbers were measured on the
old .so". A line of defence in a head cannot be audited, cannot be handed
over, and fails silently -- two TPOT tables from different builds subtract
exactly as cleanly as two from the same build, and the difference reads like a
result.

So the fingerprint is written by the runner and the judge refuses. These tests
cover the refusal, both directions, and the three ways it can be wrong:
letting a mismatch through, blocking a legitimate comparison, and treating a
missing fingerprint as agreement.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools" / "kt_offload"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FP = _load("build_fingerprint", TOOLS / "build_fingerprint.py")
REPORT = _load("packed_ab_report", TOOLS / "oncard" / "packed_ab_report.py")


def fp(so="a" * 32, vendors="v" * 64, kt="k" * 64, head="1" * 40, vllm_head="2" * 40, backfilled=False):
    return {
        "schema": "kt_offload/build_fingerprint/1",
        "so_primary_md5": so,
        "so_primary_md5_short": so[:12],
        "vendors_sha256": vendors,
        "kt_wheel_sha256": kt,
        "trees": {"vllm_ascend": {"head": head}, "vllm": {"head": vllm_head}},
        "backfilled": backfilled,
    }


# -- the comparison rule -------------------------------------------------


def test_same_build_is_comparable():
    ok, reasons = FP.same_build(fp(), fp())
    assert ok and reasons == []


def test_a_different_so_blocks_the_comparison():
    ok, reasons = FP.same_build(fp(so="a" * 32), fp(so="b" * 32))
    assert not ok
    assert any("so_primary_md5 differs" in r for r in reasons)


@pytest.mark.parametrize(
    "key,kwargs",
    [("vendors_sha256", {"vendors": "x" * 64}), ("kt_wheel_sha256", {"kt": "y" * 64})],
)
def test_the_other_binary_inputs_block_too(key, kwargs):
    ok, reasons = FP.same_build(fp(), fp(**kwargs))
    assert not ok and any(key in r for r in reasons)


def test_a_missing_fingerprint_is_not_a_match():
    """'No evidence of a mismatch' is not evidence of a match."""
    for a, b in ((None, fp()), (fp(), None), (None, None)):
        ok, reasons = FP.same_build(a, b)
        assert not ok and any("no build fingerprint" in r for r in reasons)


def test_one_sided_unknown_blocks_but_symmetric_unknown_only_advises():
    """Two backfilled artifacts know their .so and nothing else. That is not a
    verified match, but it must not block a comparison the .so already
    settles -- while a one-sided unknown must block, because there the two
    sides genuinely differ in what was recorded."""
    one_sided = FP.same_build(fp(), fp(vendors=None))
    assert not one_sided[0] and any("recorded on one side only" in r for r in one_sided[1])
    both = FP.same_build(fp(vendors=None, kt=None), fp(vendors=None, kt=None))
    assert both[0], both[1]
    assert all(r.startswith("ADVISORY") for r in both[1]) and any("unknown on both sides" in r for r in both[1])


def test_a_tree_commit_difference_is_advisory_by_default():
    """A docs-only commit must not stop a valid comparison -- but it is always
    printed, and a caller who wants it blocking can ask."""
    ok, reasons = FP.same_build(fp(head="1" * 40), fp(head="3" * 40))
    assert ok and any(r.startswith("ADVISORY") and "vllm_ascend HEAD differs" in r for r in reasons)
    strict_ok, strict_reasons = FP.same_build(fp(head="1" * 40), fp(head="3" * 40), strict_trees=True)
    assert not strict_ok and any("HEAD differs" in r for r in strict_reasons)


def test_require_same_build_raises_with_both_fingerprints_in_the_message():
    with pytest.raises(FP.CrossBuildError) as exc:
        FP.require_same_build(fp(so="a" * 32), fp(so="b" * 32), "a TPOT delta")
    text = str(exc.value)
    assert "a TPOT delta" in text and "a" * 12 in text and "b" * 12 in text
    assert "not a result" in text  # it says why, not just that


def test_short_marks_a_backfilled_fingerprint():
    assert "(backfilled)" in FP.short(fp(backfilled=True))
    assert "(backfilled)" not in FP.short(fp())
    assert "UNKNOWN" in FP.short(None)


# -- the judge actually refuses ------------------------------------------


def _fixture_run(tmp_path: Path, mode: str = "identical", stamped: bool = True) -> Path:
    """A fixture run. Stamped by default, because a real run always is; the
    unstamped variant is the state every artifact was in before this change."""
    run = tmp_path / (mode if stamped else mode + "-unstamped")
    cmd = [sys.executable, str(TOOLS / "oncard" / "make_packed_ab_fixture.py"), str(run), mode]
    if not stamped:
        cmd.append("--no-fingerprint")
    subprocess.run(cmd, check=True, capture_output=True)
    return run


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_the_judge_refuses_and_prints_no_delta_across_builds(tmp_path, capsys):
    run = _fixture_run(tmp_path)
    _write(run / "armA" / FP.FILENAME, fp(so="a" * 32))
    _write(run / "armB" / FP.FILENAME, fp(so="b" * 32))
    argv = sys.argv
    sys.argv = ["packed_ab_report.py", str(run)]
    try:
        rc = REPORT.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "REFUSING to report differences" in out
    assert "delta" not in out.lower(), "a number was still subtracted across builds"
    assert "a" * 12 in out and "b" * 12 in out, "the refusal must print both fingerprints"


def test_the_judge_proceeds_within_one_build(tmp_path, capsys):
    # "realgain": two interleaved rounds, so the drift floor exists. A
    # single-round fixture is refused for a different reason (no yardstick for
    # the difference), which would make this test pass or fail for the wrong one.
    run = _fixture_run(tmp_path, mode="realgain")
    for arm in ("armA", "armB"):
        _write(run / arm / FP.FILENAME, fp())
    argv = sys.argv
    sys.argv = ["packed_ab_report.py", str(run)]
    try:
        rc = REPORT.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "same build: the arms may be compared" in out
    assert "delta" in out.lower(), "the judge stopped reporting differences even within one build"
    assert rc == 0, out


def test_the_judge_refuses_when_a_fingerprint_is_absent(tmp_path, capsys):
    """The state every artifact was in before this change."""
    run = _fixture_run(tmp_path, stamped=False)
    argv = sys.argv
    sys.argv = ["packed_ab_report.py", str(run)]
    try:
        rc = REPORT.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert rc == 1 and "no build fingerprint" in out


# -- the capture itself --------------------------------------------------


def test_capture_records_the_keys_the_rule_needs(tmp_path):
    got = FP.capture(label="unit-test", vendors=str(tmp_path))
    for key in ("so", "so_primary_md5", "vendors", "kt_wheel", "trees", "captured_at"):
        assert key in got, key
    assert got["backfilled"] is False
    # an empty directory hashes to a real (empty) fingerprint, while a package
    # that is not there must NOT get one invented for it
    assert got["vendors"]["present"] is True and got["vendors"]["files"] == 0
    missing = FP.vendors_fingerprint(str(tmp_path / "not-here"))
    assert missing["present"] is False and "sha256" not in missing


def test_vendors_fingerprint_follows_content_not_mtime(tmp_path):
    (tmp_path / "a").write_bytes(b"one")
    first = FP.vendors_fingerprint(str(tmp_path))["sha256"]
    os.utime(tmp_path / "a", (0, 0))  # a copy refreshes mtime without changing a byte
    assert FP.vendors_fingerprint(str(tmp_path))["sha256"] == first
    (tmp_path / "a").write_bytes(b"two")
    assert FP.vendors_fingerprint(str(tmp_path))["sha256"] != first


def test_vendors_fingerprint_notices_a_moved_symlink(tmp_path):
    """Shared operator products are reached through links between lines, and
    CANN refuses a package reached through one -- "the link moved" is a change."""
    (tmp_path / "real1").write_bytes(b"x")
    (tmp_path / "real2").write_bytes(b"x")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "real1")
    first = FP.vendors_fingerprint(str(tmp_path))["sha256"]
    link.unlink()
    link.symlink_to(tmp_path / "real2")
    assert FP.vendors_fingerprint(str(tmp_path))["sha256"] != first
