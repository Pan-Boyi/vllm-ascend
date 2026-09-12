"""The negative controls, made permanent and self-asserting.

Every criterion in ``compare_dumps.py`` was validated once, by hand, by building a
mutant dump and reading the output.  That proves the criterion worked *that day*.
It does not stop someone from widening a threshold, dropping an attribution branch,
or short-circuiting a check -- all of which leave the positive path green.

Two things each control asserts, and the second is the one usually skipped:

1. the comparison **fails** on the mutant, and
2. **the reason it gives is the right one.**

A control that only checks the exit code cannot tell "red because of the fault I
injected" from "red because the dump path was wrong" -- those have the same exit
code.  That is not hypothetical: a check here returned exit 2 once for
``python: can't open file`` while the intended reading was "the criterion fired".

    python test_negative_controls.py [--base DUMP] [-v]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
DUMPS = "/workspace/code/dsv4_single_card/vllm_project/golden_dumps"
DEFAULT_BASE = os.path.join(DUMPS, "smoke3_v5")

results: list[tuple[str, bool, str]] = []


def run_compare(golden: str, cand: str, tol: str = "isolated") -> tuple[int, str]:
    p = subprocess.run([sys.executable, os.path.join(HERE, "compare_dumps.py"),
                        golden, cand, "--tolerance", tol],
                       capture_output=True, text=True, cwd=HERE)
    return p.returncode, p.stdout + p.stderr


def check(name: str, rc: int, out: str, want_rc: int, want_text: str, verbose: bool):
    """Assert both the verdict and the stated reason."""
    ok_rc = (rc == want_rc)
    ok_txt = (want_text in out) if want_text else True
    ok = ok_rc and ok_txt
    detail = ""
    if not ok_rc:
        detail = f"exit {rc}, expected {want_rc}"
        # exit 2 is almost always a crash, not a verdict -- say so rather than
        # letting it read as "the criterion fired"
        if rc == 2:
            detail += "  (exit 2 usually means the tool crashed, not that it judged)"
    elif not ok_txt:
        detail = f"right verdict, WRONG REASON -- {want_text!r} not in the output"
    results.append((name, ok, detail))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
    if verbose and not ok:
        print("        " + "\n        ".join(out.strip().splitlines()[-8:]))


def mutate(base: str, layer: int, fn) -> str:
    """Copy ``base`` to a temp dir and apply ``fn`` to one layer's tensors."""
    d = tempfile.mkdtemp(prefix="negctl_")
    shutil.rmtree(d)
    shutil.copytree(base, d)
    p = os.path.join(d, f"layer_{layer:02d}.safetensors")
    blob = load_file(p)
    fn(blob)
    save_file(blob, p)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(a.base):
        print(f"base dump {a.base} missing -- regenerate with "
              f"`run python run_golden.py --layers 3 --dump {a.base}`")
        sys.exit(1)
    print(f"base = {a.base}")

    tmp: list[str] = []

    print("\n1. a known-good pair must pass (the positive arm -- without it the rest "
          "proves nothing)")
    rc, out = run_compare(a.base, a.base)
    check("identical dumps pass", rc, out, 0, "", a.verbose)

    print("\n2. row permutation must be named as an ORDERING bug, not as arithmetic")
    def perm(b):
        t = b["moe.out"]; f = t.reshape(-1, t.shape[-1]).clone()
        f[[4, 11]] = f[[11, 4]]
        b["moe.out"] = f.view_as(t)
    d = mutate(a.base, 1, perm); tmp.append(d)
    rc, out = run_compare(a.base, d)
    check("permutation -> 'ORDERING bug'", rc, out, 1,
          "dispatch/combine ORDERING bug", a.verbose)

    print("\n3. a changed VALUE must NOT be called a permutation")
    def val(b):
        t = b["moe.out"].clone(); t[0, 4, :] = t[0, 4, :] * 1.5
        b["moe.out"] = t
    d = mutate(a.base, 1, val); tmp.append(d)
    rc, out = run_compare(a.base, d)
    check("value change -> not a permutation", rc, out, 1,
          "row multiset differs", a.verbose)

    print("\n4. non-finite values: the three arms must land in three different places")
    base_l2 = load_file(os.path.join(a.base, "layer_02.safetensors"))
    if "indexer_score" not in base_l2:
        results.append(("non-finite arms", False, "base dump has no indexer_score"))
        print("  FAIL non-finite arms   base dump has no indexer_score")
    else:
        def nan_where_finite(b):
            t = b["indexer_score"].clone()
            t[tuple(torch.isfinite(t).nonzero()[0].tolist())] = float("nan")
            b["indexer_score"] = t
        d = mutate(a.base, 2, nan_where_finite); tmp.append(d)
        rc, out = run_compare(a.base, d)
        check("NaN where golden is finite -> 'non-finite values disagree'", rc, out, 1,
              "non-finite values disagree", a.verbose)
        # direction matters: the message must name the CANDIDATE side
        check("  ...and names the candidate side", rc, out, 1,
              "positions are non-finite in the candidate but finite in the golden",
              a.verbose)

    print("\n5. a single-token error must be caught by the per-token gate")
    # 1.5x on one token: the whole-tensor rel_l2 stays well inside its limit, so this
    # passes unless max_token_rel_l2 is actually gated.  That is the point -- it is
    # the regression test for section 6.7 being deleted as "redundant".
    def one_token(b):
        t = b["layer.out"].clone(); t[:, 3] = t[:, 3] * 1.5
        b["layer.out"] = t
    d = mutate(a.base, 1, one_token); tmp.append(d)
    rc, out = run_compare(a.base, d)
    check("one bad token -> 'max_token_rel_l2' in the reason", rc, out, 1,
          "max_token_rel_l2", a.verbose)

    for d in tmp:
        shutil.rmtree(d, ignore_errors=True)

    print()
    bad = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(bad)}/{len(results)} controls behaved as specified")
    if bad:
        print("FAILED: " + ", ".join(bad))
        sys.exit(1)
    print("every control fails for the reason it was written to detect")
    sys.exit(0)


if __name__ == "__main__":
    main()
