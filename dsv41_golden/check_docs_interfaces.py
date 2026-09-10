"""Check that every interface the docs promise actually exists.

The audit of 2026-09-10 found three layers of stale documentation, in increasing
order of difficulty:

1. **A stale number** conflicts with the corrected number.  Grep the old value.
2. **A stale explanation** conflicts with the new evidence.  Grep the old wording.
3. **A promised interface that was never implemented conflicts with nothing.**
   It is not wrong, it is *absent*, so no search for a contradiction can find it.
   ``engram_lookup_ref.py``'s docstring advertised ``--verify DUMP_DIR`` while its
   argparse had no such argument; it read perfectly and the only way to notice was
   to run it.

This script is layer 3 made executable.  It reads every flag, script name, function
and dump path the markdown and the module docstrings promise, and checks each one
against the code.  Run it after editing docs.

    python check_docs_interfaces.py            # scans the current directory
    python check_docs_interfaces.py --root DIR # scan somewhere else
    python check_docs_interfaces.py -v         # also list what was verified

Exit 1 means either something promised is missing, **or nothing was checked**.
A check that scans an empty tree and prints a pass is the exact failure this
script exists to catch, so zero coverage is a failure, not a silent success.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys

# The tree to scan.  Set from --root; it deliberately does NOT default to the
# directory holding this script.  "scan wherever the script happens to sit" is an
# implicit convention that silently changes behaviour when the file is copied --
# it made a copy under build_artifacts/gate_checks/ report a clean pass over the
# three files next to it while ignoring all 8 markdown documents it was meant to
# check.  See the zero-coverage guard in main().
HERE = os.getcwd()


def _find_dumps() -> str:
    """Walk up for golden_dumps rather than hard-coding a depth."""
    d = HERE
    for _ in range(6):
        cand = os.path.join(d, "golden_dumps")
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return os.path.join(HERE, "golden_dumps")


DUMPS = ""      # set in main(), after --root is known

# Files the docs legitimately name that are not ours to define: vendor code under
# ref/, and upstream sources quoted from the CANN / HF trees.  Naming one is a
# reference, not a promise, so absence here is not a finding.
EXTERNAL_SCRIPTS = {"modeling_deepseek.py", "elastic_buffer.py", "convert.py",
                    "fp8_cast_bf16.py", "setup.py"}

# Only our own docs and scripts; ref/ is vendor code and engram_research/ is rescued
# upstream material, neither of which we promise anything about.
SKIP_DIRS = {"ref", "engram_research", "__pycache__", ".git"}

FLAG_RE = re.compile(r"(?<![\w-])--[a-z][a-z0-9-]*")
SCRIPT_RE = re.compile(r"\b([a-z_][a-z0-9_]*\.py)\b")
# ``mod.func`` / ``Class.method`` inside backticks, with or without parentheses
DOTTED_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\(\)`")
DUMPDIR_RE = re.compile(r"golden_dumps/([A-Za-z0-9_]+)")


def our_files(ext: str) -> list[str]:
    out = []
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in files:
            if f.endswith(ext):
                out.append(os.path.join(root, f))
    return sorted(out)


def argparse_flags(path: str) -> set[str] | None:
    """Flags a script actually defines, by reading its AST -- no import, no run."""
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return None
    flags: set[str] = set()
    found_parser = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in ("add_argument", "add_arg"):
            found_parser = True
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and a.value.startswith("--"):
                    flags.add(a.value)
        elif isinstance(fn, ast.Attribute) and fn.attr == "ArgumentParser":
            found_parser = True
        elif isinstance(fn, ast.Name) and fn.id == "ArgumentParser":
            found_parser = True
    return flags if found_parser else None


def defined_names(path: str) -> set[str]:
    """Top-level and class-level defs, so ``Class.method()`` mentions can be checked."""
    try:
        tree = ast.parse(open(path).read())
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(f"{node.name}.{sub.name}")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
    return names


def doc_sources() -> list[tuple[str, str]]:
    """(label, text) for every markdown file and every module docstring."""
    out = []
    for p in our_files(".md"):
        out.append((os.path.relpath(p, HERE), open(p).read()))
    for p in our_files(".py"):
        try:
            doc = ast.get_docstring(ast.parse(open(p).read()))
        except SyntaxError:
            doc = None
        if doc:
            out.append((os.path.relpath(p, HERE) + " (docstring)", doc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--root", default=None,
                    help="tree to scan (default: the current directory). Explicit "
                         "because 'scan next to the script' breaks silently when copied")
    a = ap.parse_args()

    global HERE, DUMPS
    if a.root:
        HERE = os.path.abspath(a.root)
        if not os.path.isdir(HERE):
            print(f"--root {a.root} is not a directory")
            sys.exit(1)
    DUMPS = _find_dumps()
    print(f"root    {HERE}")

    scripts = {os.path.basename(p): p for p in our_files(".py")}
    # vendor scripts resolve, but we make no promises about their flags
    vendor = set()
    for root, _dirs, files in os.walk(os.path.join(HERE, "ref")):
        vendor |= {f for f in files if f.endswith(".py")}
    flags = {n: argparse_flags(p) for n, p in scripts.items()}
    names = {n: defined_names(p) for n, p in scripts.items()}
    all_names = set()
    for s in names.values():
        all_names |= s

    problems: list[str] = []
    checked = 0

    for label, text in doc_sources():
        lines = text.splitlines()
        # A flag belongs to the last script named at or above it, within the same
        # block -- how these docs actually read, e.g. a heredoc of commands.
        current = None
        in_fence = False
        for i, line in enumerate(lines):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                current = None
                continue
            if not line.strip():
                current = None
            # Only a line that actually invokes the script owns the flags on it.
            # A prose line naming `routing_plan.py` and `--force-layer-in` in the
            # same sentence owns neither -- that heuristic produced a false finding.
            invoked = re.search(r"(?:^|[$\s(])python[0-9.]*\s+([a-z_][a-z0-9_]*\.py)", line)
            if invoked and invoked.group(1) in scripts:
                current = invoked.group(1)
            for mt in SCRIPT_RE.finditer(line):
                name = mt.group(1)
                if name in scripts or name in vendor or name in EXTERNAL_SCRIPTS:
                    continue
                # A citation is not a promise.  `torch_npu/.../distributed_c10d.py:500`
                # and `w4a8_mxfp4.py:87` point at someone else's source; only a bare
                # invocable name claims that we ship the thing.
                before = line[:mt.start()][-1:]
                after = line[mt.end():mt.end() + 2]
                if before == "/" or re.match(r":\d", after):
                    continue
                problems.append(f"{label}:{i+1}: names script `{name}` which does not exist")
            # A single inline-code span that holds both a script and flags is a
            # command, even without a leading `python`: `engram_lookup_ref.py
            # --verify` in a table cell promises that flag.  The false finding this
            # rule must NOT resurrect had the script and the flag in *separate*
            # spans, which is why the unit is the span and not the line.
            for span in re.findall(r"`([^`]+)`", line):
                sm = SCRIPT_RE.search(span)
                if not sm or sm.group(1) not in scripts:
                    continue
                owner, defined = sm.group(1), flags.get(sm.group(1))
                if defined is None:
                    continue
                for f in FLAG_RE.findall(span):
                    checked += 1
                    if f not in defined:
                        problems.append(
                            f"{label}:{i+1}: `{owner} {f}` -- {owner} defines no {f} "
                            f"(has: {' '.join(sorted(defined)) or 'none'})")
                    elif a.verbose:
                        print(f"  ok  {label}:{i+1}  {owner} {f}  [inline span]")

            for f in FLAG_RE.findall(line):
                if current is None:
                    continue
                defined = flags.get(current)
                if defined is None:
                    continue                    # script takes no argparse args at all
                checked += 1
                if f not in defined:
                    problems.append(
                        f"{label}:{i+1}: `{current} {f}` -- {current} defines no {f} "
                        f"(has: {' '.join(sorted(defined)) or 'none'})")
                elif a.verbose:
                    print(f"  ok  {label}:{i+1}  {current} {f}")

        for cls, meth in DOTTED_RE.findall(text):
            dotted = f"{cls}.{meth}"
            if cls in ("torch", "np", "os", "sys", "F", "M", "self"):
                continue
            if cls not in all_names and dotted not in all_names:
                continue                        # not one of ours; vendor or stdlib
            checked += 1
            if dotted not in all_names and meth not in all_names:
                problems.append(f"{label}: promises `{dotted}()` which is not defined")
            elif a.verbose:
                print(f"  ok  {label}  {dotted}()")

        for d in set(DUMPDIR_RE.findall(text)):
            checked += 1
            if not os.path.isdir(os.path.join(DUMPS, d)):
                problems.append(f"{label}: references golden_dumps/{d}, which is absent")
            elif a.verbose:
                print(f"  ok  {label}  golden_dumps/{d}")

    nmd = len(our_files(".md"))
    ndocs = len(doc_sources())
    print(f"checked {checked} promises across {ndocs} documents "
          f"({nmd} markdown + {ndocs - nmd} docstrings) and {len(scripts)} scripts")

    # Zero coverage is a failure.  This tool's whole argument is that silence is
    # not correctness, so it must not pass by having looked at nothing -- and the
    # signal is that the count is implausible, not that a check failed.  The
    # coverage number is printed unconditionally for the same reason: a drop in
    # what gets checked otherwise looks exactly like an improvement in quality.
    # The markdown count is guarded separately from the total.  A copy of this
    # script sitting alone in a directory still supplies its OWN docstring, so
    # "documents > 0" and "promises > 0" can both hold while not one of the
    # markdown files this is meant to check was seen.  Markdown is the subject;
    # docstrings alone are the tool talking to itself.
    if nmd == 0 or not scripts or checked == 0:
        print()
        print(f"NOTHING CHECKED -- {nmd} markdown files, {len(scripts)} scripts, "
              f"{checked} promises under {HERE}")
        print("This is a failure, not a pass.  Most likely --root points at the")
        print("wrong tree (it defaults to the CURRENT DIRECTORY, not to wherever")
        print("this script was copied).  Point --root at the directory that holds")
        print("the markdown you want checked.")
        sys.exit(1)

    if problems:
        print()
        for p in problems:
            print(f"MISSING  {p}")
        print()
        print(f"{len(problems)} promised interface(s) do not exist.")
        print("A promised-but-absent interface contradicts nothing, so grepping for")
        print("stale claims cannot find it -- only running it can.  That is why this")
        print("check exists.")
        sys.exit(1)
    print("every flag, script, function and dump path the docs promise exists")
    sys.exit(0)


if __name__ == "__main__":
    main()
