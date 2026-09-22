#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the build fingerprint watch the package that actually gets loaded?

The operator package is loaded from whichever tree is imported; it is
gitignored, so git merge does not carry it and the distribution scripts do not
touch it. The shared baseline under build_artifacts/vendor92 can therefore
drift away from the tree's own copy (786 files against 759 on 2026-09-21), and
the first version of this tool hashed the shared one -- so a swap inside OUR
tree would not have changed the fingerprint and the judge would have said "same
build". Used by selftest_packed_ab.sh.

Since 2026-09-22 the two are byte-identical: this tree was rebased onto
dsv41-clean a0e673186 and took over clean's already-accepted quadruple, which
is the 9.2.0-beta.2 package the shared baseline holds (verify_shared_artifacts.sh,
moe-offload pin). Identical content is the expected state, not drift -- see
_mutation_control below for where this tool's discriminating power moved to.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile


def _bf():
    """The build_fingerprint module this tree would run, loaded from its path.

    Loaded, never re-implemented: both the schema string and the hashing rule
    below have to come from the tool under test, or this file would be checking
    its own copy of them.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "build_fingerprint.py")
    spec = importlib.util.spec_from_file_location("build_fingerprint", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _schema() -> str:
    """Read the current schema from build_fingerprint.py.

    🔴 Was ``fp["schema"].endswith("/2")``. That is the very thing k.15②
    forbids -- a check copying a constant of the code it checks -- and it
    slipped past the "current schema appears in one file only" test because
    that test scans for the whole string, not for a suffix of it.
    """
    return _bf().SCHEMA


def _mutation_control(mod, watched: str, recorded: str) -> str:
    """Prove this fingerprint still has discriminating power, by mutation.

    🔴 This replaces ``assert fp["vendors_sha256"] != shared["sha256"]``, whose
    stated purpose was "this check can tell the loaded package from the shared
    baseline". That was never a property of the system -- it was a reading of
    one day (759 files here against 786 there on 2026-09-21). On 2026-09-22 the
    tree took over clean's accepted 9.2.0-beta.2 package and the two became
    byte-identical (787 entries / 135871736 B / sha 2eae934d033b on both), so
    the inequality stopped holding. Two things are true at once and must not be
    confused: the premise died, and a hash comparison between two directories
    that are *supposed* to be equal can prove nothing about this tool anyway.
    Relaxing the assertion to "they may be equal" would leave zero red arms.

    Where the discriminating power actually lives, now that content alone
    cannot separate the two directories:
      * *which object* is watched is settled by the path (``vendors_source``,
        asserted in main() -- it must be the tree's, never vendor92's);
      * *whether the hash is alive* is settled here: clone the watched
        directory, re-fingerprint the clone -- it must reproduce the recorded
        sha256, which is what a stale or hardcoded value cannot do -- then flip
        a single byte inside the clone. The sha256 must move while the file
        count and total size stay put, so it is the content that was seen and
        not the shape of the directory.
    The clone is made with ``shutil.copy``, not ``copy2``: the copies get fresh
    mtimes, so reproducing the recorded sha256 also demonstrates the rule
    build_fingerprint states for itself -- content, not mtime.
    The original object is never touched: the clone is a copy in TMPDIR, and
    the mutated file is rewritten in the clone only. Measured cost on this box:
    0.4 s and ~132 MB of scratch for the whole 787-entry package.
    """
    with tempfile.TemporaryDirectory(prefix="fp_object_check.") as tmp:
        clone = os.path.join(tmp, "vendors")
        shutil.copytree(watched, clone, symlinks=True, copy_function=shutil.copy)
        before = mod.vendors_fingerprint(clone, role="loaded")
        assert before["ok"], before
        assert before["sha256"] == recorded, (
            "a byte-for-byte clone of the watched directory does not reproduce the recorded "
            f"sha256 -- the fingerprint is not a live hash of {watched}: {before['sha256']} "
            f"vs recorded {recorded}"
        )
        victim = min(
            (os.path.join(d, n) for d, _s, fs in os.walk(clone) for n in fs
             if not os.path.islink(os.path.join(d, n)) and os.path.getsize(os.path.join(d, n)) > 0),
            key=os.path.getsize,
        )
        blob = bytearray(open(victim, "rb").read())
        blob[-1] ^= 0xFF                       # same length: files and bytes cannot move
        os.remove(victim)                      # never edit in place: a clone may share the inode
        with open(victim, "wb") as f:
            f.write(blob)
        after = mod.vendors_fingerprint(clone, role="loaded")
        assert after["sha256"] != recorded, (
            f"one flipped byte in {os.path.relpath(victim, clone)} did NOT change vendors_sha256 "
            "-- this fingerprint cannot tell two operator packages apart"
        )
        assert (after["files"], after["bytes"]) == (before["files"], before["bytes"]), (
            "the mutation changed the file count or the total size, so the sha256 moving proves "
            f"nothing about content: {after['files']}f/{after['bytes']}B vs "
            f"{before['files']}f/{before['bytes']}B"
        )
        return (f"mutation={os.path.relpath(victim, clone)}:1B -> "
                f"{recorded[:12]}→{after['sha256'][:12]}")


def main() -> int:
    with open(sys.argv[1]) as f:
        fp = json.load(f)
    src = fp["vendors_source"]
    assert src.endswith("vllm_ascend/_cann_ops_custom/vendors"), f"fingerprint watches {src}"
    assert "build_artifacts/vendor92" not in src, "still watching the shared baseline"
    assert fp["vendors_ok"] and fp["vendors_files"] > 0, fp.get("vendors")
    assert fp["vendors_shared_baseline"]["role"] == "shared_baseline"
    assert fp["schema"] == _schema(), (fp["schema"], _schema())
    # the ds41 operator package (schema /3): the distribution unit and the
    # installed site both have to be recorded and readable, or a swap of
    # them would again read as "same build"
    assert fp["custom_ops_ok"] and fp["custom_ops_files"] == 4, fp.get("custom_ops")
    assert fp["ds41_ok"] and fp["ds41_files"] > 0, fp.get("ds41_site")
    assert fp["ds41_sha256"] and fp["custom_ops_sha256"]
    # the kt site (schema /4): the fingerprint must watch the directory that
    # goes on PYTHONPATH, not a wheel in some unrelated directory. The wheel
    # glob was fixed at kt_wheel/*.whl and independent of the site, so a run on
    # kt_site_v2 carried the OLD wheel's sha256 and same_build() called the two
    # builds one.
    assert fp["kt_site_ok"] and fp["kt_site_files"] > 0, fp.get("kt_site")
    assert fp["kt_site_sha256"], "no kt site hash: the imported package is not being watched"
    decisive = fp["kt_site"]["decisive"]
    assert any(k.endswith(".so") for k in decisive) and any(k.endswith("experts_base.py") for k in decisive), decisive
    assert fp["kt_site_sha256"] != fp["kt_wheel_sha256"], "the site hash is just the wheel's again"
    assert fp["kt_wheel"]["is_override"] or fp["kt_wheel"]["derived_from"] == fp["kt_site_source"], (
        "the wheel is not derived from the site -- that independence is the bug")
    shared = fp["vendors_shared_baseline"]
    # The two directories being equal today is a reading, not a criterion: it is
    # reported so a later reader sees WHY there is no inequality test here, and
    # it decides nothing. The criterion is the mutation control.
    same_as_shared = fp["vendors_sha256"] == shared.get("sha256")
    mutation = _mutation_control(_bf(), fp["vendors_source"], fp["vendors_sha256"])
    print(f"loaded={fp['vendors_files']}f/{fp['vendors_bytes']}B  shared={shared['files']}f  "
          f"shared_identical={same_as_shared} (expected since the 09-22 package swap)  "
          f"discriminates: {mutation}  "
          f"ktsite={os.path.basename(fp['kt_site_source'])}:{fp['kt_site_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
