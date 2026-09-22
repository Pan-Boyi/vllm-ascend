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


def fp(so="a" * 32, vendors="v" * 64, kt="k" * 64, head="1" * 40, vllm_head="2" * 40, backfilled=False,
       files=759, nbytes=118_000_000, custom_ops="o" * 64, ds41="d" * 64, ds41_files=973,
       ds41_bytes=47_757_444, kt_site="s" * 64, kt_site_files=70, kt_site_bytes=9_518_073):
    return {
        # the kt site (schema /4): the package the run imports. Separate from
        # the wheel on purpose -- the wheel used to be the only kt input and it
        # was found by a glob that did not move when the site did.
        "kt_site_sha256": kt_site,
        "kt_site_files": kt_site_files,
        "kt_site_bytes": kt_site_bytes,
        "kt_site_ok": True,
        "kt_site_source": "/x/build_artifacts/moe_offload_ref/kt_site",
        # the ds41 operator package (schema /3): the distribution unit and
        # the installed site, both blocking
        "custom_ops_sha256": custom_ops,
        "custom_ops_files": 4,
        "custom_ops_ok": True,
        "ds41_sha256": ds41,
        "ds41_files": ds41_files,
        "ds41_bytes": ds41_bytes,
        "ds41_ok": True,
        "schema": FP.SCHEMA,
        "vendors_files": files,
        "vendors_bytes": nbytes,
        "vendors_ok": True,
        "vendors": {"count_rule": {"id": FP.COUNT_RULE_ID}},
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
    [("vendors_sha256", {"vendors": "x" * 64}), ("kt_wheel_sha256", {"kt": "y" * 64}),
     ("kt_site_sha256", {"kt_site": "z" * 64})],
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
    assert "NO DIFFERENCE WILL BE COMPUTED" in out
    # the sections that do not subtract still run (bit-identity, host-func
    # counts): suppressing them would throw away evidence that is still valid.
    assert "answers, bit-identical" in out
    # ... but nothing was actually differenced
    assert "paired A-B deltas: suppressed" in out or "skipped" in out
    import re as _re

    assert not _re.search(r"A-B = [+-]\d", out), "a delta was printed across builds"
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


# -- the object the fingerprint must watch -------------------------------


def test_the_loaded_package_is_the_one_that_decides():
    """The operator package is loaded from whichever tree is imported, it is
    gitignored, and neither git merge nor the distribution scripts touch it --
    so the tree's own copy and the shared baseline drift apart (759 files here
    against 786 in build_artifacts/vendor92 on 2026-09-21). Hashing the shared
    one let a swap in OUR tree read as "same build": the guard watched the
    wrong object."""
    got = FP.capture(label="unit-test")
    assert got["vendors_source"].endswith("vllm_ascend/_cann_ops_custom/vendors"), got["vendors_source"]
    assert "build_artifacts/vendor92" not in got["vendors_source"]
    # the shared baseline is still recorded, labelled, and out of the decision
    assert got["vendors_shared_baseline"]["role"] == "shared_baseline"
    assert "vendors_shared_baseline" not in FP.BLOCKING
    assert got["vendors"]["role"] == "loaded"


def test_two_trees_with_different_packages_are_different_builds():
    """The case that used to pass: only the loaded package differs."""
    a = fp()
    b = fp()
    b["vendors_sha256"] = "z" * 64
    b["vendors_files"] = 700
    ok, reasons = FP.same_build(a, b)
    assert not ok
    assert any("vendors_sha256 differs" in r for r in reasons)
    assert any("vendors_files differs" in r for r in reasons)


def test_a_difference_in_the_shared_baseline_alone_does_not_block():
    """It is context, not the thing that ran."""
    a = fp()
    b = fp()
    a["vendors_shared_baseline"] = {"sha256": "1" * 64, "files": 786, "role": "shared_baseline"}
    b["vendors_shared_baseline"] = {"sha256": "2" * 64, "files": 700, "role": "shared_baseline"}
    ok, _ = FP.same_build(a, b)
    assert ok


def test_an_empty_package_is_refused_not_hashed(tmp_path):
    """sha256 of nothing is a constant, so two failed reads would compare
    equal and print as "same build"."""
    empty = FP.vendors_fingerprint(str(tmp_path), role="loaded")
    assert empty["ok"] is False and empty["files"] == 0
    assert "sha256" not in empty, "an empty directory must not get a hash at all"
    assert "0 files" in empty["why"] and FP.EMPTY_SHA256[:16] in empty["why"]
    assert FP.EMPTY_SHA256[:16] == "e3b0c44298fc1c14"  # the sentinel to look for in a report
    missing = FP.vendors_fingerprint(str(tmp_path / "nope"), role="loaded")
    assert missing["ok"] is False and "sha256" not in missing


def test_an_unreadable_package_blocks_the_comparison():
    a, b = fp(), fp()
    for side in (a, b):
        side["vendors_ok"] = False
        side["vendors_files"] = 0
        side["vendors"] = {"why": "0 files under /gone"}
    ok, reasons = FP.same_build(a, b)
    assert not ok
    assert sum("not readable" in r for r in reasons) == 2, reasons
    assert any(FP.EMPTY_SHA256[:16] in r for r in reasons)


def test_the_override_is_honoured_and_recorded(tmp_path, monkeypatch):
    """DSV41_VENDORS points the fingerprint somewhere else on purpose; the
    artifact must say so, or a later reader cannot tell what was measured."""
    (tmp_path / "libfake.so").write_bytes(b"x" * 16)
    monkeypatch.setenv("DSV41_VENDORS", str(tmp_path))
    reloaded = _load("build_fingerprint_override", TOOLS / "build_fingerprint.py")
    got = reloaded.capture(label="override", vendors=reloaded.VENDORS)
    assert got["vendors_source"] == str(tmp_path)
    assert got["vendors_source_is_override"] is True
    assert got["vendors_files"] == 1 and got["vendors_bytes"] == 16


def test_the_old_schema_cannot_be_subtracted_from():
    """v1 recorded the shared baseline (or nothing), so a v1 artifact's numbers
    may be quoted but never differenced."""
    old = fp()
    old["schema"] = "kt_offload/build_fingerprint/1"
    ok, reasons = FP.same_build(old, fp())
    assert not ok
    assert any(r.startswith("OLDSCHEMA") for r in reasons)
    assert any("SHARED BASELINE" in r for r in reasons)


def test_counts_are_part_of_the_decision():
    assert "vendors_files" in FP.BLOCKING and "vendors_bytes" in FP.BLOCKING


# -- the kt site: the second time the guard watched the wrong object -----


def _fake_site(root: Path, name: str, ext: bytes, experts_base: bytes, wheel: bytes | None = b"WHEEL") -> Path:
    """A minimal kt site next to its wheel directory, laid out like the real
    ones: <root>/<name>/kt_kernel/{kt_kernel_ext*.so,experts_base.py,...} and
    the sibling <root>/kt_wheel<suffix>/*.whl."""
    pkg = root / name / "kt_kernel"
    pkg.mkdir(parents=True)
    (pkg / "kt_kernel_ext.cpython-312-x86_64-linux-gnu.so").write_bytes(ext)
    (pkg / "experts_base.py").write_bytes(experts_base)
    (pkg / "experts.py").write_bytes(b"# identical in both builds in circulation\n")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "experts.cpython-312.pyc").write_bytes(b"noise that must not decide")
    if wheel is not None:
        wheel_dir = root / ("kt_wheel" + name[len("kt_site"):])
        wheel_dir.mkdir(parents=True, exist_ok=True)
        (wheel_dir / "kt_kernel-0.7.0.post4-cp312-cp312-linux_x86_64.whl").write_bytes(wheel)
    return root / name


def test_the_kt_site_hashes_the_two_files_the_service_imports(tmp_path):
    site = _fake_site(tmp_path, "kt_site", ext=b"OLDSO", experts_base=b"old")
    got = FP.kt_site_fingerprint(str(site))
    assert got["ok"] and got["sha256"]
    decisive = got["decisive"]
    assert any(k.endswith("kt_kernel_ext.cpython-312-x86_64-linux-gnu.so") for k in decisive)
    assert "kt_kernel/experts_base.py" in decisive
    # __pycache__ is not part of the build: it is regenerated per interpreter
    # run and would make the same site hash differently from one day to the next
    assert got["files"] == 3, got


def test_a_site_swap_is_a_different_build_even_when_the_old_wheel_is_still_globbed(tmp_path):
    """The bug this schema bump fixes, in one test.

    Both builds ship as a *site* on PYTHONPATH; the wheel directory the old
    fingerprint globbed (kt_wheel/*.whl) had no link to the site at all. So
    switching serve/run_oncard to kt_site_v2 and missing that one glob left
    both runs reporting the same kt sha256, same_build() said "comparable",
    and two TPOT tables from two different .so builds subtracted in silence.

    Here: the site changes (new ext .so, new experts_base.py) while the wheel
    stays exactly the byte string the old glob would have found.
    """
    old = _fake_site(tmp_path, "kt_site", ext=b"OLDSO", experts_base=b"old", wheel=b"THE-OLD-WHEEL")
    new = _fake_site(tmp_path, "kt_site_v2", ext=b"NEWSO", experts_base=b"new", wheel=b"THE-OLD-WHEEL")
    a = FP.kt_site_fingerprint(str(old))
    b = FP.kt_site_fingerprint(str(new))
    assert a["sha256"] != b["sha256"], "the two sites hash the same: the site is not being read"
    # the wheel alone cannot tell them apart -- that is the whole point
    wa = FP.kt_wheel_fingerprint(str(old))
    wb = FP.kt_wheel_fingerprint(str(new))
    assert wa["sha256"] == wb["sha256"]
    # and the judge refuses on the site, not on the wheel
    ok, reasons = FP.same_build(fp(kt_site=a["sha256"]), fp(kt_site=b["sha256"]))
    assert not ok
    assert any("kt_site_sha256 differs" in r for r in reasons), reasons


def test_the_wheel_is_derived_from_the_site_not_named_separately(tmp_path):
    """kt_site -> kt_wheel, kt_site_v2 -> kt_wheel_v2. A wheel path of its own
    does not move when the site moves; that independence was the bug."""
    old = _fake_site(tmp_path, "kt_site", ext=b"OLDSO", experts_base=b"old", wheel=b"W1")
    new = _fake_site(tmp_path, "kt_site_v2", ext=b"NEWSO", experts_base=b"new", wheel=b"W2")
    assert FP.kt_wheel_fingerprint(str(old))["dir"].endswith("/kt_wheel")
    assert FP.kt_wheel_fingerprint(str(new))["dir"].endswith("/kt_wheel_v2")
    assert FP.kt_wheel_fingerprint(str(old))["sha256"] != FP.kt_wheel_fingerprint(str(new))["sha256"]
    # a site with no wheel beside it is still a usable fingerprint: the site
    # decides, the wheel is provenance
    lone = _fake_site(tmp_path, "kt_site_v3", ext=b"X", experts_base=b"y", wheel=None)
    assert FP.kt_site_fingerprint(str(lone))["ok"]
    assert FP.kt_wheel_fingerprint(str(lone))["sha256"] is None


@pytest.mark.parametrize("what", ["missing", "no-kt_kernel", "no-so", "no-experts_base", "unnamed"])
def test_a_site_that_cannot_be_read_is_refused_never_hashed(tmp_path, what):
    """As with the operator package: an unread input must not get a hash. The
    empty-input sha256 is a constant, so two failed reads would compare equal
    and print as "same build"."""
    if what == "missing":
        site = str(tmp_path / "not-here")
    elif what == "no-kt_kernel":
        (tmp_path / "bare").mkdir()
        site = str(tmp_path / "bare")
    elif what == "unnamed":
        site = None
    else:
        s = _fake_site(tmp_path, "kt_site", ext=b"e", experts_base=b"p")
        drop = ("kt_kernel_ext.cpython-312-x86_64-linux-gnu.so" if what == "no-so" else "experts_base.py")
        (s / "kt_kernel" / drop).unlink()
        site = str(s)
    got = FP.kt_site_fingerprint(site)
    assert got["ok"] is False and "sha256" not in got, got
    assert got["why"]


def test_the_site_is_named_in_exactly_one_place_and_the_env_overrides_it(tmp_path, monkeypatch):
    """kt_build.env is the single source: the shell drivers source it, this
    tool parses it. A second copy of the path is how the site and the
    fingerprint came apart."""
    assert FP.KT_BUILD_ENV.endswith("tools/kt_offload/kt_build.env")
    text = Path(FP.KT_BUILD_ENV).read_text()
    assert "KT_SITE_DEFAULT=" in text
    assert FP._kt_site_from_build_env() == FP.KT_SITE or os.environ.get("KT_SITE")
    site = _fake_site(tmp_path, "kt_site_v2", ext=b"NEWSO", experts_base=b"new")
    monkeypatch.setenv("KT_SITE", str(site))
    reloaded = _load("build_fingerprint_kt_override", TOOLS / "build_fingerprint.py")
    assert reloaded.KT_SITE == str(site)
    got = reloaded.kt_site_fingerprint(reloaded.KT_SITE)
    assert got["ok"] and got["is_override"] is True and got["source"] == "env KT_SITE"


def test_the_tool_refuses_to_write_a_fingerprint_without_a_site(tmp_path):
    """No site, no fingerprint. Falling back to some other path would put a
    build label on a reading nobody can attribute."""
    env = dict(os.environ, KT_SITE=str(tmp_path / "not-a-site"))
    out = tmp_path / "stamped"
    out.mkdir()
    proc = subprocess.run([sys.executable, str(TOOLS / "build_fingerprint.py"), "stamp", str(out)],
                          env=env, capture_output=True, text=True)
    assert proc.returncode != 0, proc.stdout
    assert "refusing to write a build fingerprint" in proc.stderr
    assert not (out / FP.FILENAME).exists(), "a fingerprint was written anyway"


def test_the_kt_site_is_part_of_the_decision():
    for key in ("kt_site_sha256", "kt_site_files", "kt_site_bytes"):
        assert key in FP.BLOCKING, key


def test_an_unreadable_kt_site_blocks_the_comparison():
    a, b = fp(), fp()
    for side in (a, b):
        side["kt_site_ok"] = False
        side["kt_site_files"] = 0
    ok, reasons = FP.same_build(a, b)
    assert not ok
    assert sum("kt site" in r for r in reasons) == 2, reasons


def test_the_previous_schema_cannot_be_subtracted_from():
    """A /3 artifact cannot say which kt site it ran on -- its kt hash is a
    wheel found by a glob that did not follow the site."""
    old = fp()
    old["schema"] = "kt_offload/build_fingerprint/3"
    ok, reasons = FP.same_build(old, fp())
    assert not ok and any(r.startswith("OLDSCHEMA") for r in reasons)


# -- the two principles this round produced ------------------------------


def test_the_current_schema_string_lives_in_exactly_one_place():
    """Test data must not copy a constant of the code under test.

    Both fixture builders once wrote ``schema: .../1`` as a literal. Bumping
    the schema then turned every artifact they produce into "not comparable",
    and eight downstream self-checks went red at once -- for a reason that had
    nothing to do with what they check. A fixture has no business knowing what
    the schema is; it reads it.

    The rule enforced here: the CURRENT schema string appears only in
    build_fingerprint.py. Historical values (``/1``) may be written literally
    -- they are frozen, and a test about migration has to name the old one.
    """
    root = Path(__file__).resolve().parents[3]
    owner = TOOLS / "build_fingerprint.py"
    current = FP.SCHEMA
    offenders = []
    scanned = 0
    for path in sorted(list((TOOLS / "oncard").glob("*.py")) + list(Path(__file__).parent.glob("*.py"))):
        if path.resolve() == owner.resolve():
            continue
        scanned += 1
        text = path.read_text()
        if current in text:
            offenders.append(str(path.relative_to(root)))
    assert scanned >= 10, f"the scan only looked at {scanned} files; it is pointing at the wrong place"
    assert not offenders, (
        f"these files copy the current schema {current!r} instead of reading build_fingerprint.SCHEMA: "
        f"{offenders}. Bump the schema and they will quietly declare every artifact incomparable."
    )


def test_fixtures_read_the_schema_rather_than_stating_it():
    """The positive half of the same rule: the builders must actually read it."""
    for name in ("make_packed_ab_fixture.py", "make_aa_fixture.py"):
        text = (TOOLS / "oncard" / name).read_text()
        assert "_schema()" in text, f"{name} does not read the schema from the tool"
        assert "build_fingerprint" in text, f"{name} does not reference the owning module"


def test_every_suppressed_difference_says_so_and_why(tmp_path, capsys):
    """Fine-grained refusal only works if each hole is labelled.

    Suppressing a difference silently looks like "that line happens to be
    missing", which is how someone ends up computing it by hand instead.
    """
    run = _fixture_run(tmp_path, mode="realgain")
    _write(run / "armA" / FP.FILENAME, fp(so="a" * 32))
    _write(run / "armB" / FP.FILENAME, fp(so="b" * 32))
    for r in (2,):
        _write(run / f"armA_r{r}" / FP.FILENAME, fp(so="a" * 32))
        _write(run / f"armB_r{r}" / FP.FILENAME, fp(so="b" * 32))
    argv = sys.argv
    sys.argv = ["packed_ab_report.py", str(run)]
    try:
        rc = REPORT.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert rc == 1
    # a reason, printed before the suppression, naming the differing key
    assert "so_primary_md5 differs" in out
    # every subtracting place is labelled
    assert out.count("suppressed") >= 2, out
    assert "⛔ skipped: a TPOT delta across builds" in out
    assert "⛔ skipped: C is a difference between the arms" in out
    # and the evidence that does not subtract is still there
    assert "answers, bit-identical" in out and "identical" in out
    import re as _re

    numeric_delta = _re.search(r"(A-B|delta[^:]*) = [+-]?\d+\.\d", out)
    assert numeric_delta is None, f"a difference survived suppression: {numeric_delta.group(0)!r}"


# -- counts are comparable only inside one counting convention -----------


def test_the_counting_rule_states_all_of_its_choices(tmp_path):
    """Four counts of "the operator package" were in circulation on
    2026-09-21 and they name different objects, not one off-by-one. A count
    without its rule cannot be told from another rule's count -- ours (760)
    even coincides with `find _cann_ops_custom -type f`."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "real").write_bytes(b"x" * 8)
    (tmp_path / "link").symlink_to(tmp_path / "sub" / "real")
    got = FP.vendors_fingerprint(str(tmp_path), role="loaded")
    rule = got["count_rule"]
    for key in ("root", "include_dirs", "include_symlinks", "symlink_mode", "includes_upper_gitkeep",
                "equivalent_shell", "files_regular_equivalent", "id"):
        assert key in rule, f"the rule does not say {key}"
    assert rule["root"] == str(tmp_path)  # the root is recorded verbatim
    assert got["files"] == 2 and got["files_regular"] == 1 and got["symlinks"] == 1
    assert "never dereferenced" in rule["symlink_mode"]


def test_counts_from_different_conventions_are_not_compared():
    """A different rule must read as "not comparable", never as "the package
    changed": two of the four counts would have been "fixed" into being wrong."""
    a = fp(files=760)
    b = fp(files=759)
    a["vendors"] = {"count_rule": {"id": FP.COUNT_RULE_ID}}
    b["vendors"] = {"count_rule": {"id": FP.COUNT_RULE_ID.replace("counted-as-link", "EXCLUDED")}}
    ok, reasons = FP.same_build(a, b)
    assert not ok
    assert any(r.startswith("CONVENTION") for r in reasons)
    assert any("NOT evidence that the package changed" in r for r in reasons)
    # and the count itself is not reported as a difference
    assert not any("vendors_files differs" in r for r in reasons), reasons


def test_counts_do_decide_within_one_convention():
    rule = {"count_rule": {"id": FP.COUNT_RULE_ID}}
    a, b = fp(files=760), fp(files=700)
    a["vendors"] = dict(rule)
    b["vendors"] = dict(rule)
    ok, reasons = FP.same_build(a, b)
    assert not ok and any("vendors_files differs" in r for r in reasons)


# -- the ds41 operator package (schema /3, 2026-09-22) ----------------------


@pytest.mark.parametrize("key,kwargs", [
    ("custom_ops_sha256", {"custom_ops": "z" * 64}),
    ("ds41_sha256", {"ds41": "z" * 64}),
    ("ds41_files", {"ds41_files": 972}),
    ("ds41_bytes", {"ds41_bytes": 47_757_443}),
])
def test_the_ds41_package_blocks_the_comparison(key, kwargs):
    """Without these in the decision, swapping the ds41 wheels or the
    installed site left the judge saying "same build" -- while the five
    candidate layers and >16k depend on them."""
    ok, reasons = FP.same_build(fp(), fp(**kwargs))
    assert not ok and any(key in r for r in reasons)


def test_one_flipped_byte_in_a_distributed_artifact_is_seen(tmp_path):
    """Not a model of the rule -- the real capture(), over real copies.

    The artifacts are copied to a temporary directory (the real ones are
    never touched), one byte of one wheel is flipped, and the two captures
    must not be comparable.
    """
    import shutil
    src = Path(FP.CUSTOM_OPS)
    if not src.is_dir():
        pytest.skip(f"{src} 不在")
    work = tmp_path / "custom_ops"
    work.mkdir()
    for name in FP.CUSTOM_OPS_FILES:
        shutil.copy2(src / name, work / name)
    # 传 root，不改默认值：默认参数在 def 时求值，monkeypatch 模块常量不生效
    before = FP.custom_ops_fingerprint(str(work))
    assert before["ok"] and before["files"] == 4

    victim = work / FP.CUSTOM_OPS_FILES[0]
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0x01
    victim.write_bytes(bytes(data))
    after = FP.custom_ops_fingerprint(str(work))
    assert after["sha256"] != before["sha256"]
    ok, reasons = FP.same_build(fp(custom_ops=before["sha256"]),
                                fp(custom_ops=after["sha256"]))
    assert not ok and any("custom_ops_sha256 differs" in r for r in reasons)


def test_the_run_is_recorded_as_shipped_but_not_installed(tmp_path):
    """The .run is not installed (指南 §3.9: it overrides seven OpTypes of the
    production vendor). It is still part of the unit, so it is hashed -- and
    the fact that it is not installed is written down next to it, because a
    hash with no statement of what it describes is how a guard ends up
    watching the wrong object (k.14)."""
    fpr = FP.custom_ops_fingerprint()
    if not fpr["ok"]:
        pytest.skip(fpr["why"])
    run = "cann-ops-transformer-custom_linux-x86_64.run"
    assert run in fpr["each"]
    assert fpr["not_installed"] == [run]
    assert len(fpr["each"]) == 4


def test_a_missing_ds41_artifact_is_refused_not_hashed(tmp_path):
    """A partial read must not produce a hash: two partial reads would
    compare equal."""
    fpr = FP.custom_ops_fingerprint(str(tmp_path))
    assert fpr["ok"] is False and "sha256" not in fpr
    assert len(fpr["missing"]) == 4
    ok, reasons = FP.same_build(fp(custom_ops=None, **{}) | {"custom_ops_ok": False,
                                                            "custom_ops_files": 0},
                                fp())
    assert not ok
    assert any("ds41 operator package" in r for r in reasons)


def test_an_unreadable_ds41_site_is_refused():
    broken = fp() | {"ds41_ok": False, "ds41_files": 0}
    ok, reasons = FP.same_build(broken, fp())
    assert not ok and any("installed ds41_site" in r for r in reasons)


def test_a_schema_2_artifact_cannot_be_subtracted_from_a_schema_3_one():
    """The bump is the point: a /2 fingerprint never recorded the ds41
    package, so "no difference seen" there means "never looked"."""
    old = fp() | {"schema": "kt_offload/build_fingerprint/2"}
    ok, reasons = FP.same_build(old, fp())
    assert not ok and any(r.startswith("OLDSCHEMA") for r in reasons)


def test_one_flipped_byte_in_the_installed_ds41_site_is_seen(tmp_path):
    """The installed site is fingerprinted by content with the same walker
    the operator package uses, so a swapped file inside it is visible even
    when the wheels next door are untouched."""
    site = tmp_path / "ds41_site"
    (site / "cann_ops_transformer").mkdir(parents=True)
    f = site / "cann_ops_transformer" / "__init__.py"
    f.write_text("x = 1\n")
    before = FP.vendors_fingerprint(str(site), role="ds41_site")
    assert before["ok"] and before["role"] == "ds41_site"
    f.write_text("x = 2\n")
    after = FP.vendors_fingerprint(str(site), role="ds41_site")
    assert after["sha256"] != before["sha256"]
    ok, _ = FP.same_build(fp(ds41=before["sha256"]), fp(ds41=after["sha256"]))
    assert not ok


def test_the_ds41_sources_can_be_overridden_and_the_override_is_recorded(
        tmp_path, monkeypatch):
    """As with DSV41_VENDORS: an override is allowed, and the fact that one
    was used is written into the artifact -- otherwise a fingerprint taken
    against a copy reads like one taken against production."""
    monkeypatch.setenv("DSV41_DS41_SITE", str(tmp_path))
    monkeypatch.setenv("DSV41_CUSTOM_OPS", str(tmp_path))
    mod = _load("build_fingerprint_override", TOOLS / "build_fingerprint.py")
    assert mod.DS41_SITE == str(tmp_path) and mod.CUSTOM_OPS == str(tmp_path)
    cap = mod.capture()
    assert cap["ds41_source_is_override"] and cap["custom_ops_source_is_override"]
    # ...and an empty override is a refusal, not a constant hash
    assert cap["ds41_ok"] is False and cap["custom_ops_ok"] is False
    ok, reasons = mod.same_build(cap, cap)
    assert not ok and any("not readable" in r for r in reasons)
