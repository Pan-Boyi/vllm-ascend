#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""What build produced this artifact, and may these two artifacts be compared?

Until now the rule "do not compare numbers measured on different builds" lived
in somebody's memory. It is a line of defence that cannot be audited, cannot
be handed over, and fails silently: two TPOT tables from different ``.so``
builds subtract just as cleanly as two from the same one, and the difference
reads like a result.

So: every performance or accuracy artifact carries a fingerprint, and every
script that *subtracts* two artifacts refuses when their builds differ. The
refusal prints both fingerprints; it never picks one, and it never degrades to
a warning, because a cross-build difference is not a weaker result -- it is a
different question.

What goes in the fingerprint:
  * ``so``       md5 of every ``vllm_ascend_C*.so`` found, and which one the
                 run would import (the operators' actual object code);
  * ``vendors``  one sha256 over (relative path, size, md5) of every file in
                 the operator package -- content, not mtime, because a copy
                 refreshes mtime without changing a byte;
  * ``kt_wheel`` sha256 (the CPU MoE);
  * ``trees``    both worktrees' HEAD and dirty-file count.

Blocking vs advisory: the first three decide numerics, so they block a
comparison. Tree commits are advisory -- a docs-only commit must not stop a
valid comparison -- but they are always printed, and a caller that wants them
blocking passes ``strict_trees=True``.

Usage:
  build_fingerprint.py capture [--out PATH] [--label L]
  build_fingerprint.py stamp <dir> [--label L]      # writes <dir>/build_fingerprint.json
  build_fingerprint.py compare <A> <B>              # rc 0 same build, rc 1 different
  build_fingerprint.py backfill <dir> --so-md5 <md5> [--note ...]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.environ.get("DSV41_ROOT", "/workspace/code/dsv4_single_card/vllm_project")
TREE = os.environ.get("DSV41_VLLM_ASCEND_TREE", os.path.join(ROOT, "worktrees/moe-offload"))
VLLM_TREE = os.environ.get("DSV41_VLLM_TREE", os.path.join(ROOT, "worktrees/moe-offload-vllm"))
VENDORS = os.environ.get("DSV41_VENDORS", os.path.join(ROOT, "build_artifacts/vendor92"))
KT_WHEEL_GLOB = os.path.join(ROOT, "build_artifacts/moe_offload_ref/kt_wheel/*.whl")
SO_GLOBS = (
    os.path.join(TREE, "vllm_ascend/vllm_ascend_C*.so"),
    os.path.join(ROOT, "build_artifacts/vllm_ascend_C*.so"),
    os.path.join(ROOT, "vllm-ascend/vllm_ascend/vllm_ascend_C*.so"),
)
FILENAME = "build_fingerprint.json"
BACKFILL_NAME = "build_fingerprint.backfilled.json"
BLOCKING = ("so_primary_md5", "vendors_sha256", "kt_wheel_sha256")


class CrossBuildError(RuntimeError):
    """Raised when two artifacts from different builds would be subtracted."""


def _md5(path: str) -> str:
    h = hashlib.md5()  # noqa: S324 - an identity tag, not a security boundary
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vendors_fingerprint(root: str = VENDORS) -> dict:
    """One hash over the operator package's *content*.

    Symlinks are followed and recorded as such: a shared build tree links
    products between lines, and "the link moved" is a real change here (CANN
    also refuses to load an operator package reached through a symlink, which
    is how a whole afternoon once went).
    """
    if not os.path.isdir(root):
        return {"path": root, "present": False}
    entries, links, unreadable = [], 0, 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if os.path.islink(full):
                links += 1
                entries.append((rel, -1, "link:" + os.readlink(full)))
                continue
            try:
                entries.append((rel, os.path.getsize(full), _md5(full)))
            except OSError:
                unreadable += 1
                entries.append((rel, -2, "unreadable"))
    entries.sort()
    h = hashlib.sha256()
    for rel, size, digest in entries:
        h.update(f"{rel}\0{size}\0{digest}\n".encode())
    return {
        "path": root,
        "present": True,
        "sha256": h.hexdigest(),
        "files": len(entries),
        "symlinks": links,
        "unreadable": unreadable,
    }


def _git(tree: str, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", tree, *args], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def capture(label: str | None = None, vendors: str = VENDORS) -> dict:
    so = {}
    for pattern in SO_GLOBS:
        for path in sorted(glob.glob(pattern)):
            try:
                so[path] = _md5(path)
            except OSError:
                so[path] = "unreadable"
    primary = next((md5 for path, md5 in so.items() if path.startswith(TREE)), None)
    if primary is None and so:
        primary = next(iter(so.values()))
    vend = vendors_fingerprint(vendors)  # once: it hashes ~130 MB
    wheels = sorted(glob.glob(KT_WHEEL_GLOB))
    kt_sha = _sha256(wheels[-1]) if wheels else None
    trees = {}
    for name, tree in (("vllm_ascend", TREE), ("vllm", VLLM_TREE)):
        trees[name] = {
            "path": tree,
            "head": _git(tree, "rev-parse", "HEAD"),
            "branch": _git(tree, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([ln for ln in _git(tree, "status", "--porcelain").splitlines() if ln]),
        }
    return {
        "schema": "kt_offload/build_fingerprint/1",
        "label": label,
        "so": so,
        "so_primary_md5": primary,
        "so_primary_md5_short": (primary or "")[:12] or None,
        "vendors": vend,
        "vendors_sha256": vend.get("sha256"),
        "kt_wheel": {"path": wheels[-1] if wheels else None, "sha256": kt_sha},
        "kt_wheel_sha256": kt_sha,
        "trees": trees,
        "captured_at": subprocess.run(["date", "-Is"], capture_output=True, text=True).stdout.strip(),
        "backfilled": False,
    }


def load(path_or_dir: str | None) -> dict | None:
    """Read a fingerprint from a file, or from a directory holding one.

    A backfilled fingerprint is accepted and stays marked as such: it says
    which build the numbers came from, but it was written after the fact, so
    it is evidence of provenance, not of measurement.
    """
    if not path_or_dir:
        return None
    if os.path.isfile(path_or_dir) and os.path.basename(path_or_dir) in (FILENAME, BACKFILL_NAME):
        with open(path_or_dir) as f:
            return json.load(f)
    # walk up: a profile csv sits several directories below the run that was
    # stamped (armA/prof_out/csv/<rank>/task_time.csv), and the nearest
    # fingerprint above it is the one that describes it
    here = os.path.abspath(path_or_dir if os.path.isdir(path_or_dir) else os.path.dirname(path_or_dir))
    for _ in range(8):
        for name in (FILENAME, BACKFILL_NAME):
            candidate = os.path.join(here, name)
            if os.path.isfile(candidate):
                with open(candidate) as f:
                    return json.load(f)
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def short(fp: dict | None) -> str:
    if not fp:
        return "build: UNKNOWN (no fingerprint)"
    v = (fp.get("vendors_sha256") or "")[:12] or "?"
    kt = (fp.get("kt_wheel_sha256") or "")[:12] or "?"
    heads = "/".join((fp.get("trees", {}).get(n, {}).get("head") or "?")[:9] for n in ("vllm_ascend", "vllm"))
    tag = " (backfilled)" if fp.get("backfilled") else ""
    return f"build: so {fp.get('so_primary_md5_short') or '?'} vendors {v} kt {kt} trees {heads}{tag}"


def same_build(a: dict | None, b: dict | None, strict_trees: bool = False) -> tuple[bool, list[str]]:
    """(comparable, reasons). A missing fingerprint is never 'the same build'."""
    reasons = []
    for name, fp in (("first", a), ("second", b)):
        if fp is None:
            reasons.append(f"the {name} artifact has no build fingerprint, so it cannot be shown to match")
    if reasons:
        return False, reasons
    for key in BLOCKING:
        va, vb = a.get(key), b.get(key)
        if va is None and vb is None:
            # Symmetrically unknown: usually two backfilled fingerprints, where
            # the .so is known and the rest was never recorded. Not a match and
            # not a mismatch -- say so, and let the known keys decide.
            reasons.append(f"ADVISORY {key} unknown on both sides (never recorded; not verified equal)")
        elif va is None or vb is None:
            reasons.append(f"{key} recorded on one side only ({va} vs {vb}): cannot be shown to match")
        elif va != vb:
            reasons.append(f"{key} differs: {va} vs {vb}")
    for name in ("vllm_ascend", "vllm"):
        ha = a.get("trees", {}).get(name, {}).get("head")
        hb = b.get("trees", {}).get(name, {}).get("head")
        if ha and hb and ha != hb:
            note = f"{name} HEAD differs: {ha[:9]} vs {hb[:9]}"
            if strict_trees:
                reasons.append(note)
            else:
                reasons.append("ADVISORY " + note)
    blocking = [r for r in reasons if not r.startswith("ADVISORY ")]
    return (not blocking), reasons


def require_same_build(a: dict | None, b: dict | None, what: str, strict_trees: bool = False) -> list[str]:
    """Raise unless the two artifacts may be subtracted. Returns advisories."""
    ok, reasons = same_build(a, b, strict_trees)
    if not ok:
        raise CrossBuildError(
            f"refusing to compute {what}: these artifacts come from different builds, so a difference "
            "between them is not a result.\n  A: "
            + short(a)
            + "\n  B: "
            + short(b)
            + "\n  "
            + "\n  ".join(reasons)
        )
    return [r for r in reasons if r.startswith("ADVISORY ")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--out", default="")
    cap.add_argument("--label", default=None)
    st = sub.add_parser("stamp")
    st.add_argument("dir")
    st.add_argument("--label", default=None)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--strict-trees", action="store_true")
    bf = sub.add_parser("backfill")
    bf.add_argument("dir")
    bf.add_argument("--so-md5", required=True)
    bf.add_argument("--vendors-sha256", default=None)
    bf.add_argument("--kt-wheel-sha256", default=None)
    bf.add_argument("--note", default="")
    args = ap.parse_args()

    if args.cmd in ("capture", "stamp"):
        fp = capture(label=args.label)
        out = args.out if args.cmd == "capture" and args.out else (
            os.path.join(args.dir, FILENAME) if args.cmd == "stamp" else ""
        )
        text = json.dumps(fp, indent=1)
        if out:
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w") as f:
                f.write(text + "\n")
            print(f"{short(fp)}\n-> {out}")
        else:
            print(text)
        return 0

    if args.cmd == "compare":
        a, b = load(args.a), load(args.b)
        ok, reasons = same_build(a, b, args.strict_trees)
        print(f"A {short(a)}\nB {short(b)}")
        for r in reasons:
            print("  " + r)
        print("SAME BUILD: comparable" if ok else "DIFFERENT BUILD: not comparable (do not subtract these)")
        return 0 if ok else 1

    fp = {
        "schema": "kt_offload/build_fingerprint/1",
        "label": os.path.basename(os.path.normpath(args.dir)),
        "so": {},
        "so_primary_md5": args.so_md5,
        "so_primary_md5_short": args.so_md5[:12],
        "vendors_sha256": args.vendors_sha256,
        "kt_wheel_sha256": args.kt_wheel_sha256,
        "trees": {},
        "backfilled": True,
        "backfill_note": args.note
        or "written after the run; the md5 is the build these numbers were measured on, recorded from "
           "outside the run, not captured by it",
    }
    path = os.path.join(args.dir, BACKFILL_NAME)
    with open(path, "w") as f:
        f.write(json.dumps(fp, indent=1) + "\n")
    print(f"{short(fp)}\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
