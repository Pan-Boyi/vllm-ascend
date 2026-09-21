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
import json
import sys


def main() -> int:
    with open(sys.argv[1]) as f:
        fp = json.load(f)
    src = fp["vendors_source"]
    assert src.endswith("vllm_ascend/_cann_ops_custom/vendors"), f"fingerprint watches {src}"
    assert "build_artifacts/vendor92" not in src, "still watching the shared baseline"
    assert fp["vendors_ok"] and fp["vendors_files"] > 0, fp.get("vendors")
    assert fp["vendors_shared_baseline"]["role"] == "shared_baseline"
    assert fp["schema"].endswith("/2"), fp["schema"]
    shared = fp["vendors_shared_baseline"]
    assert fp["vendors_sha256"] != shared.get("sha256"), (
        "the loaded package and the shared baseline hash the same today, so this check cannot tell "
        "them apart -- re-check by file count before trusting it"
    )
    print(f"loaded={fp['vendors_files']}f/{fp['vendors_bytes']}B  shared={shared['files']}f  distinct=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
