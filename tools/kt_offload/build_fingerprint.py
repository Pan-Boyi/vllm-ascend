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
  * ``vendors``  the operator package **that this tree imports**
                 (``<tree>/vllm_ascend/_cann_ops_custom/vendors``): one sha256
                 over (relative path, size, md5) of every file -- content, not
                 mtime, because a copy refreshes mtime without changing a byte
                 -- plus the file count and total bytes, which are part of the
                 decision. The shared baseline under
                 ``build_artifacts/vendor92`` is recorded too, labelled, and
                 takes NO part in the same-build decision: the package is
                 loaded from whichever tree is imported, it is gitignored, and
                 git merge and the distribution scripts do not touch it, so
                 the two can differ (they do today: 759 files here against
                 786 there) and hashing the shared one would let a swap in
                 OUR tree pass unnoticed;
  * ``custom_ops`` the ds41 distribution unit under ``custom_ops_dsv41``:
                 the three installed wheels and the shipped-but-not-installed
                 ``.run``, each by (name, size, sha256);
  * ``ds41_site`` the installed ds41 package by content -- what a run actually
                 imports, and without which the five candidate layers fall
                 back to the v2 operators and >16k stays blocked, silently;
  * ``kt_site``  the kt package **that this run imports** -- the directory
                 appended to PYTHONPATH, hashed by content over its
                 ``kt_kernel/`` tree, with ``kt_kernel/kt_kernel_ext*.so`` and
                 ``kt_kernel/experts_base.py`` required and recorded one by
                 one. The site is named in exactly one place,
                 ``tools/kt_offload/kt_build.env``, which the shell drivers
                 source and this file parses;
  * ``kt_wheel`` sha256 of the wheel that site was installed from, **derived
                 from the site directory** (kt_site -> kt_wheel, kt_site_v2 ->
                 kt_wheel_v2). Provenance, recorded, and no longer the only
                 thing watched: it used to be globbed from a fixed
                 ``kt_wheel/*.whl`` with no link to the site, so switching the
                 site to kt_site_v2 and forgetting that glob left two builds
                 reporting one sha256 -- same_build() said "comparable" and
                 the numbers subtracted in silence;
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
import fnmatch
import glob
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.environ.get("DSV41_ROOT", "/workspace/code/dsv4_single_card/vllm_project")
TREE = os.environ.get("DSV41_VLLM_ASCEND_TREE", os.path.join(ROOT, "worktrees/moe-offload"))
VLLM_TREE = os.environ.get("DSV41_VLLM_TREE", os.path.join(ROOT, "worktrees/moe-offload-vllm"))
# The package that actually gets loaded: the one inside the tree being
# imported. DSV41_VENDORS overrides it (and the override is recorded).
VENDORS = os.environ.get("DSV41_VENDORS", os.path.join(TREE, "vllm_ascend/_cann_ops_custom/vendors"))
VENDORS_SHARED_BASELINE = os.path.join(ROOT, "build_artifacts/vendor92")
# The ds41 operator package: four artifacts distributed as one unit, and the
# directory they are installed into. They decide numerics exactly like the
# .so does -- the five candidate layers fall back to the v2 operators without
# them and >16k stays blocked -- and until 2026-09-22 nothing here watched
# them, so swapping them left the judge saying "same build".
#   * the three wheels are the ones actually installed (指南 §3.9:
#     pip install --no-deps --no-index --target $DS41 <the three>);
#   * the .run is NOT installed (指南 §3.9 says do not: it overrides seven
#     OpTypes of the production vendor). It is fingerprinted anyway because
#     it ships in the same unit: if it changed, the unit changed, and a
#     reading taken across that change is not comparable without someone
#     having looked. Recorded next to the fact that it is not installed.
#   * ds41_site is the installed result, by content -- what a run imports.
CUSTOM_OPS = os.environ.get("DSV41_CUSTOM_OPS", os.path.join(ROOT, "custom_ops_dsv41"))
CUSTOM_OPS_FILES = (
    "cannbotdsl-0.5.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    "cann_ops_transformer-1.0.0-py3-none-any.whl",
    "cannbot_arena_net_ops-0.1.0-cp312-cp312-linux_x86_64.whl",
    "cann-ops-transformer-custom_linux-x86_64.run",   # shipped, not installed
)
CUSTOM_OPS_NOT_INSTALLED = ("cann-ops-transformer-custom_linux-x86_64.run",)
DS41_SITE = os.environ.get("DSV41_DS41_SITE", "/workspace/code/dsv4_single_card/ds41_site")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()  # the hash of no input: a constant, and constants always "match"

# The kt site: the directory the drivers append to PYTHONPATH, i.e. the
# package a run actually imports. It is named ONCE, in kt_build.env, which the
# shell drivers source and this file parses -- see that file for why. There is
# deliberately no hard-coded fallback here: a second copy of the path is how
# the site and the fingerprint came apart in the first place.
KT_BUILD_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kt_build.env")
KT_SITE_REQUIRED_FILE = "kt_kernel/experts_base.py"   # the python half of the two that differ between builds
KT_SITE_REQUIRED_GLOB = "kt_kernel/kt_kernel_ext*.so"  # the compiled half


def _kt_site_from_build_env() -> str | None:
    """The default site, read from the file the shell drivers source."""
    try:
        with open(KT_BUILD_ENV) as f:
            for line in f:
                line = line.strip()
                if line.startswith("KT_SITE_DEFAULT="):
                    return line.split("=", 1)[1].strip().strip("\"'") or None
    except OSError:
        return None
    return None


KT_SITE = os.environ.get("KT_SITE") or _kt_site_from_build_env()
SO_GLOBS = (
    os.path.join(TREE, "vllm_ascend/vllm_ascend_C*.so"),
    os.path.join(ROOT, "build_artifacts/vllm_ascend_C*.so"),
    os.path.join(ROOT, "vllm-ascend/vllm_ascend/vllm_ascend_C*.so"),
)
# /3 (2026-09-22): adds the ds41 operator package -- the four custom_ops_dsv41
# artifacts and the installed ds41_site -- to the same-build decision. A /2
# fingerprint never recorded them, so it may not be subtracted against a /3
# one; that is the point of the bump, not a side effect.
# /4 (2026-09-22): adds the kt SITE -- the package the run imports -- and
# derives the wheel from it. Up to /3 the kt half of the fingerprint was a
# wheel found by a fixed glob that was independent of the site on PYTHONPATH,
# so a run on kt_site_v2 and a run on kt_site carried the SAME kt hash. A /3
# artifact cannot say which site it ran on, so it may not be subtracted
# against a /4 one; again the point of the bump, not a side effect.
SCHEMA = "kt_offload/build_fingerprint/4"
# The identity of the counting convention. Counts may only be compared when
# this matches; a different value means "counted differently", never "the
# package changed". Anything that writes a fingerprint reads this from here.
COUNT_RULE_ID = "root=<recorded>;dirs=0;symlink=counted-as-link;deref=0;v2"
FILENAME = "build_fingerprint.json"
BACKFILL_NAME = "build_fingerprint.backfilled.json"
# Every key here decides. The counts are in on purpose: the hash of an empty
# directory is a constant, so a hash alone cannot tell "identical packages"
# from "read nothing twice".
BLOCKING = ("so_primary_md5", "vendors_sha256", "vendors_files", "vendors_bytes",
            "kt_site_sha256", "kt_site_files", "kt_site_bytes", "kt_wheel_sha256",
            "custom_ops_sha256", "custom_ops_files", "ds41_sha256", "ds41_files", "ds41_bytes")


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


def vendors_fingerprint(root: str = VENDORS, role: str = "loaded") -> dict:
    """Content fingerprint of one operator package, with its size.

    ``role`` says what this directory IS: "loaded" (the tree's own package,
    which decides) or "shared_baseline" (build_artifacts/vendor92, recorded
    for context only). A fingerprint that does not say which one it measured
    is how a guard ends up watching the wrong object.

    Symlinks are followed only as names: a shared build tree links products
    between lines, "the link moved" is a real change here, and CANN refuses an
    operator package reached through a symlink anyway.

    An absent or empty directory returns ``ok: False`` and no sha256. It must
    not return the empty-input hash: that value is a constant, so two failed
    reads would compare equal and print as "same build".
    """
    out = {"path": root, "role": role, "present": os.path.isdir(root)}
    if not out["present"]:
        out.update(ok=False, why=f"not a directory: {root}", files=0, bytes=0)
        return out
    entries, links, unreadable, total = [], 0, 0, 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if os.path.islink(full):
                links += 1
                entries.append((rel, -1, "link:" + os.readlink(full)))
                continue
            try:
                size = os.path.getsize(full)
                total += size
                entries.append((rel, size, _md5(full)))
            except OSError:
                unreadable += 1
                entries.append((rel, -2, "unreadable"))
    entries.sort()
    h = hashlib.sha256()
    for rel, size, digest in entries:
        h.update(f"{rel}\0{size}\0{digest}\n".encode())
    out.update(
        files=len(entries),
        bytes=total,
        files_regular=len(entries) - links,
        symlinks=links,
        unreadable=unreadable,
        # Say exactly what was counted, next to the number. Four counts of these
        # two directories were in circulation on 2026-09-21 -- 787/786 for the
        # shared baseline and 760/759 for this tree -- and the whole difference
        # is this convention: `find -type f` counts regular files only, while
        # this walker also counts the one symlink in each tree (759+1, 786+1,
        # measured). Without the rule printed here, the right answer ("two
        # conventions") reads as the wrong one ("a package was swapped").
        # What was counted, in full. Four counts of "the operator package" were
        # in circulation on 2026-09-21 and they are NOT one off-by-one: they
        # name different objects (integration, 2026-09-21):
        #   786  <tree>/.../vendors/custom_transformer, find -type f   <- clean's new pin
        #   787  the same, counting the one symlink as well
        #   787  one level up at _cann_ops_custom (adds a .gitkeep)
        #   787  build/lib copy -- a DIFFERENT directory, where setuptools
        #        dereferenced the symlink into a 3.25 MB regular file
        # Aligning them would make two of the four wrong. So the rule is
        # recorded, never inferred: root, whether directories count, whether
        # symlinks count and how, and whether the upper .gitkeep is inside the
        # root. Our own 760 vs flash_dev's 759 on this tree is the symlink item
        # (759 + 1); and 760 happens to equal `find _cann_ops_custom -type f`
        # as well, so the number alone cannot say which rule produced it.
        count_rule={
            "root": root,
            "walk": "os.walk from root",
            "include_dirs": False,
            "include_symlinks": True,
            "symlink_mode": "counted, hashed by target name, never dereferenced",
            "includes_upper_gitkeep": os.path.basename(os.path.normpath(root)) != "vendors",
            "equivalent_shell": f"find {root} \\( -type f -o -type l \\) | wc -l  == files",
            "files_regular_equivalent": f"find {root} -type f | wc -l  == files_regular",
            "id": COUNT_RULE_ID,
        },
    )
    if not entries:
        out.update(ok=False, why=f"0 files under {root}: an empty package hashes to the constant "
                                 f"{EMPTY_SHA256[:16]}, which would compare equal to any other empty read")
        return out
    out.update(ok=True, sha256=h.hexdigest())
    return out


def custom_ops_fingerprint(root: str = CUSTOM_OPS) -> dict:
    """The ds41 distribution unit: each named artifact by content.

    Named explicitly rather than walking the directory: the four files are
    what is distributed and what gets installed, while the directory also
    holds unpacked experiments (``ops_vector_0920/``) whose churn must not
    read as "the operator package changed". A missing file is a refusal, not
    a skip -- as with the operator package, an unread input must never
    compare equal.
    """
    out = {"path": root, "role": "ds41_distribution_unit",
           "not_installed": list(CUSTOM_OPS_NOT_INSTALLED),
           "rule": "the four named files, by (name, size, sha256); the rest of the directory is ignored"}
    entries, missing, total = [], [], 0
    for name in CUSTOM_OPS_FILES:
        full = os.path.join(root, name)
        try:
            size = os.path.getsize(full)
            entries.append((name, size, _sha256(full)))
            total += size
        except OSError:
            missing.append(name)
    out["files"] = len(entries)
    out["bytes"] = total
    out["missing"] = missing
    if missing:
        out.update(ok=False, why=f"{len(missing)} of {len(CUSTOM_OPS_FILES)} ds41 artifacts missing under "
                                 f"{root}: {missing}. Not hashed: a partial read must not compare equal.")
        return out
    h = hashlib.sha256()
    for name, size, digest in entries:
        h.update(f"{name}\0{size}\0{digest}\n".encode())
    out.update(ok=True, sha256=h.hexdigest(),
               each={name: {"bytes": size, "sha256": digest} for name, size, digest in entries})
    return out


def kt_site_fingerprint(site: str | None = KT_SITE) -> dict:
    """The kt package this run imports, by content.

    Reads the judged object: ``<site>/kt_kernel`` is what ends up on
    PYTHONPATH, so it -- not a wheel sitting in some other directory -- is
    what decides the CPU MoE numerics. The two files that differ between the
    builds in circulation (``kt_kernel_ext*.so`` and ``experts_base.py``) are
    required and recorded individually, so a reader can eyeball which build a
    reading came from; the hash covers the whole ``kt_kernel`` tree, minus
    ``__pycache__``, because anything else in the package is imported too.

    Refusals -- all of them ``ok: False`` with no sha256, never a fallback to
    another path:
      * the site is not named at all (no ``KT_SITE``, no ``KT_SITE_DEFAULT``
        in kt_build.env);
      * the site or its ``kt_kernel`` directory does not exist;
      * either required file is missing.
    A refusal must not produce a hash: the empty-input sha256 is a constant,
    so two failed reads would compare equal and print as "same build" -- which
    is the exact failure this whole file exists to stop.
    """
    out = {"role": "kt_site", "path": site,
           "source": "env KT_SITE" if os.environ.get("KT_SITE") else f"KT_SITE_DEFAULT in {KT_BUILD_ENV}",
           "is_override": bool(os.environ.get("KT_SITE")),
           "rule": "sha256 over (relpath, size, sha256) of every file under <site>/kt_kernel, "
                   "__pycache__ excluded; kt_kernel_ext*.so and experts_base.py required",
           "files": 0, "bytes": 0}
    if not site:
        out.update(ok=False, why=f"no kt site named: KT_SITE is unset and {KT_BUILD_ENV} has no "
                                 "KT_SITE_DEFAULT. Refusing to guess -- a guessed site is how the "
                                 "fingerprint stopped describing what ran.")
        return out
    pkg = os.path.join(site, "kt_kernel")
    if not os.path.isdir(pkg):
        out.update(ok=False, why=f"not a kt site: {pkg} is not a directory")
        return out
    entries, total = [], 0
    for dirpath, dirnames, filenames in os.walk(pkg):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, site)
            try:
                size = os.path.getsize(full)
                total += size
                entries.append((rel, size, _sha256(full)))
            except OSError:
                entries.append((rel, -2, "unreadable"))
    entries.sort()
    by_rel = {rel: digest for rel, _size, digest in entries}
    ext = sorted(rel for rel in by_rel if fnmatch.fnmatch(rel, KT_SITE_REQUIRED_GLOB))
    out["files"] = len(entries)
    out["bytes"] = total
    out["ext_so"] = ext
    missing = []
    if not ext:
        missing.append(KT_SITE_REQUIRED_GLOB)
    if KT_SITE_REQUIRED_FILE not in by_rel:
        missing.append(KT_SITE_REQUIRED_FILE)
    if missing:
        out.update(ok=False, missing=missing,
                   why=f"{site} does not look like a kt site: missing {missing}. Not hashed -- the "
                       f"service imports these two, so a fingerprint taken without them describes "
                       f"nothing that ran.")
        return out
    h = hashlib.sha256()
    for rel, size, digest in entries:
        h.update(f"{rel}\0{size}\0{digest}\n".encode())
    out.update(ok=True, sha256=h.hexdigest(),
               decisive={rel: by_rel[rel] for rel in [*ext, KT_SITE_REQUIRED_FILE]},
               ext_so_sha256=by_rel[ext[0]],
               experts_base_sha256=by_rel[KT_SITE_REQUIRED_FILE])
    return out


def kt_wheel_fingerprint(site: str | None = KT_SITE) -> dict:
    """The wheel the site was installed from -- found THROUGH the site.

    kt_site -> kt_wheel, kt_site_v2 -> kt_wheel_v2: the wheel directory is the
    site's sibling under the same name with ``kt_site`` swapped for
    ``kt_wheel``. Never a path of its own, because a path of its own does not
    move when the site does -- which is how the old fixed
    ``kt_wheel/*.whl`` glob kept reporting the previous build's sha256 after
    the site had been switched. ``KT_WHEEL`` overrides with an explicit file
    and is recorded as an override.
    """
    explicit = os.environ.get("KT_WHEEL")
    if explicit:
        out = {"role": "kt_wheel", "path": explicit, "is_override": True, "derived_from": None}
        try:
            out["sha256"] = _sha256(explicit)
        except OSError:
            out.update(sha256=None, why=f"KT_WHEEL={explicit} is not readable")
        return out
    out = {"role": "kt_wheel", "is_override": False, "derived_from": site,
           "rule": "sibling of the site with 'kt_site' replaced by 'kt_wheel' in the basename"}
    if not site:
        out.update(path=None, sha256=None, why="no kt site named, so no wheel to derive")
        return out
    base = os.path.basename(os.path.normpath(site))
    sibling = "kt_wheel" + base[len("kt_site"):] if base.startswith("kt_site") else base
    wheel_dir = os.path.join(os.path.dirname(os.path.normpath(site)), sibling)
    out["dir"] = wheel_dir
    wheels = sorted(glob.glob(os.path.join(wheel_dir, "*.whl")))
    if not wheels:
        out.update(path=None, sha256=None,
                   why=f"no wheel under {wheel_dir}; the site is what decides, the wheel is provenance")
        return out
    out["path"] = wheels[-1]
    out["sha256"] = _sha256(wheels[-1])
    return out


def host_facts() -> dict:
    """CPU counts and the thread env, because one of the readings lies.

    ``nproc`` returns min(affinity, $OMP_NUM_THREADS) and OMP_NUM_THREADS=1
    here, so it reports 1 on a 256-core machine. An artifact that records only
    that number reads as "the machine had one CPU", which looks like a major
    finding rather than a units bug -- so record the true count, the misleading
    one, and the variable that makes them differ, side by side.
    """
    def sh(*cmd) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    online = sh("getconf", "_NPROCESSORS_ONLN")
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = sum(1 for line in f if line.startswith("processor"))
    except OSError:
        cpuinfo = None
    try:
        with open("/proc/loadavg") as f:
            loadavg = f.read().split()[:3]
    except OSError:
        loadavg = []
    return {
        "cpus_online_getconf": int(online) if online.isdigit() else None,
        "cpus_in_cpuinfo": cpuinfo,
        "nproc_reports": int(sh("nproc") or 0) or None,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "note": "nproc = min(affinity, OMP_NUM_THREADS); use cpus_online_getconf as the machine's size",
        "loadavg": loadavg,
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
    vend = vendors_fingerprint(vendors, role="loaded")  # once: it hashes ~120 MB
    shared = vendors_fingerprint(VENDORS_SHARED_BASELINE, role="shared_baseline")
    custom_ops = custom_ops_fingerprint()
    # the installed result, by content: this is what a run imports
    ds41 = vendors_fingerprint(DS41_SITE, role="ds41_site")
    # the kt half: the site is the judged object, the wheel is derived from it
    site = kt_site_fingerprint(KT_SITE)
    wheel = kt_wheel_fingerprint(KT_SITE)
    trees = {}
    for name, tree in (("vllm_ascend", TREE), ("vllm", VLLM_TREE)):
        trees[name] = {
            "path": tree,
            "head": _git(tree, "rev-parse", "HEAD"),
            "branch": _git(tree, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": len([ln for ln in _git(tree, "status", "--porcelain").splitlines() if ln]),
        }
    return {
        "schema": SCHEMA,
        "label": label,
        "so": so,
        "so_primary_md5": primary,
        "so_primary_md5_short": (primary or "")[:12] or None,
        "vendors": vend,
        "vendors_sha256": vend.get("sha256"),
        "vendors_files": vend.get("files"),
        "vendors_bytes": vend.get("bytes"),
        "vendors_ok": vend.get("ok", False),
        "vendors_source": vend.get("path"),
        "vendors_source_is_override": bool(os.environ.get("DSV41_VENDORS")),
        # context only: NOT part of the same-build decision (see the docstring)
        "vendors_shared_baseline": {
            k: shared.get(k)
            for k in ("path", "role", "files", "files_regular", "symlinks", "bytes", "sha256", "ok", "count_rule")
        },
        "custom_ops": custom_ops,
        "custom_ops_sha256": custom_ops.get("sha256"),
        "custom_ops_files": custom_ops.get("files"),
        "custom_ops_ok": custom_ops.get("ok", False),
        "custom_ops_source_is_override": bool(os.environ.get("DSV41_CUSTOM_OPS")),
        "ds41_site": ds41,
        "ds41_sha256": ds41.get("sha256"),
        "ds41_files": ds41.get("files"),
        "ds41_bytes": ds41.get("bytes"),
        "ds41_ok": ds41.get("ok", False),
        "ds41_source": ds41.get("path"),
        "ds41_source_is_override": bool(os.environ.get("DSV41_DS41_SITE")),
        "kt_site": site,
        "kt_site_sha256": site.get("sha256"),
        "kt_site_files": site.get("files"),
        "kt_site_bytes": site.get("bytes"),
        "kt_site_ok": site.get("ok", False),
        "kt_site_source": site.get("path"),
        "kt_site_source_is_override": site.get("is_override", False),
        "kt_wheel": wheel,
        "kt_wheel_sha256": wheel.get("sha256"),
        "trees": trees,
        "host": host_facts(),
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
    files = fp.get("vendors_files")
    v = f"{v}/{files}f" if files is not None else f"{v}(shared-baseline?)"
    kt = (fp.get("kt_wheel_sha256") or "")[:12] or "?"
    # the site is what got imported; print it FIRST and next to its directory,
    # because "kt <wheel sha>" alone is what used to read the same on two builds
    ktsite = (fp.get("kt_site_sha256") or "")[:12] or "?"
    ktdir = os.path.basename(os.path.normpath(fp.get("kt_site_source") or "")) or "?"
    ds41 = (fp.get("ds41_sha256") or "")[:12] or "?"
    ops = (fp.get("custom_ops_sha256") or "")[:12] or "?"
    heads = "/".join((fp.get("trees", {}).get(n, {}).get("head") or "?")[:9] for n in ("vllm_ascend", "vllm"))
    tag = " (backfilled)" if fp.get("backfilled") else ""
    return (f"build: so {fp.get('so_primary_md5_short') or '?'} vendors {v} "
            f"ktsite {ktdir}:{ktsite} ktwheel {kt} "
            f"ds41ops {ops} ds41site {ds41}/{fp.get('ds41_files')}f trees {heads}{tag}")


def same_build(a: dict | None, b: dict | None, strict_trees: bool = False) -> tuple[bool, list[str]]:
    """(may these be subtracted, reasons). A missing fingerprint is never 'same'.

    Reasons prefixed ``ADVISORY`` do not block. Everything else does, including
    an operator package that could not be read: a failed read must not look
    like a match (its hash would be the constant ``sha256("")``).
    """
    reasons = []
    for name, fp in (("first", a), ("second", b)):
        if fp is None:
            reasons.append(f"the {name} artifact has no build fingerprint, so it cannot be shown to match")
    if reasons:
        return False, reasons

    for name, fp in (("first", a), ("second", b)):
        schema = fp.get("schema", "?")
        if schema != SCHEMA:
            reasons.append(
                f"OLDSCHEMA the {name} artifact is {schema}, not {SCHEMA}: back then the vendors field held the "
                "SHARED BASELINE (build_artifacts/vendor92), not the operator package the run actually loaded "
                "from its own tree -- the two differ, so no difference may be computed across this boundary"
            )
        for what, okkey, cnt in (("ds41 operator package (custom_ops_dsv41)", "custom_ops_ok", "custom_ops_files"),
                                 ("installed ds41_site", "ds41_ok", "ds41_files"),
                                 ("kt site (the kt_kernel package the run imports)", "kt_site_ok", "kt_site_files")):
            if fp.get(okkey) is False or (cnt in fp and fp.get(cnt) == 0):
                reasons.append(
                    f"the {name} artifact's {what} was not readable (files={fp.get(cnt)}): an unread input "
                    f"hashes to the constant {EMPTY_SHA256[:16]} and would compare equal to any other unread one"
                )
        if fp.get("vendors_ok") is False or (fp.get("vendors_files") == 0 and "vendors_files" in fp):
            reasons.append(
                f"the {name} artifact's operator package was not readable "
                f"(files={fp.get('vendors_files')}, why={fp.get('vendors', {}).get('why', '?')}): an unread "
                f"package hashes to the constant {EMPTY_SHA256[:16]} and would compare equal to any other "
                "unread package"
            )

    # Counts are only comparable within one counting convention. A different
    # rule is not evidence that the package changed -- reporting it as
    # "vendors_files differs" would be exactly the wrong conclusion (two of the
    # four counts in circulation on 2026-09-21 would have been "fixed" into
    # being wrong).
    rule_a = (a.get("vendors", {}) or {}).get("count_rule", {})
    rule_b = (b.get("vendors", {}) or {}).get("count_rule", {})
    id_a = rule_a.get("id") if isinstance(rule_a, dict) else str(rule_a)
    id_b = rule_b.get("id") if isinstance(rule_b, dict) else str(rule_b)
    same_convention = id_a == id_b and id_a is not None
    if not same_convention:
        reasons.append(
            f"CONVENTION the two artifacts counted the operator package by different rules "
            f"({id_a!r} vs {id_b!r}): the counts are not comparable, and this is NOT evidence that the "
            "package changed. Treated like an old schema: no difference may be computed."
        )

    for key in BLOCKING:
        if key in ("vendors_files", "vendors_bytes") and not same_convention:
            continue  # comparing across conventions would read as a package change
        va, vb = a.get(key), b.get(key)
        if va is None and vb is None:
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
            reasons.append(note if strict_trees else "ADVISORY " + note)

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
        # A fingerprint that cannot say which kt site ran is not a weaker
        # fingerprint, it is the failure mode this file exists to stop: two
        # builds would carry the same kt hash and their numbers would subtract
        # without a word. So: no site, no fingerprint -- and no silent
        # fallback to some other path.
        if not fp.get("kt_site_ok"):
            site = fp.get("kt_site", {})
            print(f"refusing to write a build fingerprint: {site.get('why', 'the kt site is unreadable')}\n"
                  f"  site   : {site.get('path')}\n"
                  f"  source : {site.get('source')}\n"
                  f"  fix    : point KT_SITE at the site the run imports, or set KT_SITE_DEFAULT in "
                  f"{KT_BUILD_ENV}", file=sys.stderr)
            return 2
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
        # deliberately the OLD schema: these numbers were taken before the
        # fingerprint existed, and the judge must treat them as not comparable
        # rather than pretend they carry a v2 vendors reading.
        "schema": "kt_offload/build_fingerprint/1",
        "label": os.path.basename(os.path.normpath(args.dir)),
        "so": {},
        "so_primary_md5": args.so_md5,
        "so_primary_md5_short": args.so_md5[:12],
        "vendors_sha256": args.vendors_sha256,
        "kt_wheel_sha256": args.kt_wheel_sha256,
        "trees": {},
        "backfilled": True,
        "vendors_provenance": "UNRECORDED. Nothing was written about the operator package at the time. "
        "Note that the first fingerprint tool hashed the SHARED BASELINE (build_artifacts/vendor92, 786 "
        "files) while a run loads the package from its own tree "
        "(worktrees/<tree>/vllm_ascend/_cann_ops_custom/vendors, 759 here) -- so even a v1 vendors hash "
        "would not have described what ran. Left empty on purpose; guessing from today's tree would read "
        "as verified.",
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
