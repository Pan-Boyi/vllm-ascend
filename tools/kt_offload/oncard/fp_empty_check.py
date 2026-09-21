#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""An unreadable operator package must be refused, never hashed.

sha256 of no input is the constant e3b0c44298fc1c14..., so two failed reads
would compare equal and print as "same build" -- three separate places were
bitten by exactly that on 2026-09-21. Used by selftest_packed_ab.sh.
"""
import importlib.util
import sys
import tempfile


def main() -> int:
    spec = importlib.util.spec_from_file_location("bf", sys.argv[1])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as d:
        out = m.vendors_fingerprint(d, role="loaded")
    assert out["ok"] is False and out["files"] == 0, out
    assert "sha256" not in out, "an empty package got a hash"
    assert m.EMPTY_SHA256[:16] == "e3b0c44298fc1c14", m.EMPTY_SHA256
    unread = {
        "schema": m.SCHEMA, "so_primary_md5": "x", "vendors_ok": False, "vendors_files": 0,
        "vendors_sha256": None, "vendors_bytes": 0, "kt_wheel_sha256": "k", "vendors": {"why": "0 files"},
    }
    ok, reasons = m.same_build(dict(unread), dict(unread))
    assert not ok and any("not readable" in r for r in reasons), reasons
    print(f"empty refused; two unread packages do NOT compare equal; sentinel {m.EMPTY_SHA256[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
