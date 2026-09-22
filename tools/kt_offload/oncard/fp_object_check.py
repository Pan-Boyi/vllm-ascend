#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does the build fingerprint watch the package that actually gets loaded?

The operator package is loaded from whichever tree is imported; it is
gitignored, so git merge does not carry it and the distribution scripts do not
touch it. The shared baseline under build_artifacts/vendor92 therefore drifts
away from the tree's own copy (786 files against 759 on 2026-09-21), and the
first version of this tool hashed the shared one -- so a swap inside OUR tree
would not have changed the fingerprint and the judge would have said "same
build". Used by selftest_packed_ab.sh.
"""
import importlib.util
import json
import os
import sys


def _schema() -> str:
    """Read the current schema from build_fingerprint.py.

    🔴 Was ``fp["schema"].endswith("/2")``. That is the very thing k.15②
    forbids -- a check copying a constant of the code it checks -- and it
    slipped past the "current schema appears in one file only" test because
    that test scans for the whole string, not for a suffix of it.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "build_fingerprint.py")
    spec = importlib.util.spec_from_file_location("build_fingerprint", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SCHEMA


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
    assert fp["vendors_sha256"] != shared.get("sha256"), (
        "the loaded package and the shared baseline hash the same today, so this check cannot tell "
        "them apart -- re-check by file count before trusting it"
    )
    print(f"loaded={fp['vendors_files']}f/{fp['vendors_bytes']}B  shared={shared['files']}f  distinct=True  "
          f"ktsite={os.path.basename(fp['kt_site_source'])}:{fp['kt_site_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
