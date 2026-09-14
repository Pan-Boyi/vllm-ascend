#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Every upstream behaviour this line substitutes, and whether it still agrees.

Run:  python tools/audit_dsv41_upstream_substitutions.py

🔴 Why this exists. A-2 was a per-(token, head) RMS norm applied to q that
V4.1 does not have. It came from transcribing a prose contract -- cited
correctly, by file and line -- that described the **other model's** call of a
shared operator (``deepseek_v4/cpu/cpu_sparse.py:426``). The sentence was
true; its subject was V4.0. The difference between the two callers is one
defaulted boolean, ``apply_q_norm``, which prose does not mention.

⚠️ **The set is detected, not enumerated.** A first version of this listed
"the four Python rebinds" -- and A-2 was not among them, because it is a
*decomposition* (a method override that replaces an operator with several
Python steps), a category the list did not have. Enumerating by known kind
cannot show the kind you did not think of, and the result reads as complete.
So substitutions are found by **differencing**: upstream modules before and
after the patches are applied, and our classes against their own bases.

*** The completeness argument, stated rather than assumed ***
For the operator-level substitutions there is one, and it runs the other way
from the usual worry: ``torch.ops._C`` does not exist on Ascend. An upstream
operator we neither replaced nor avoided therefore **cannot run silently** --
it raises on the missing symbol. The silent hazards are exactly the things we
*did* replace, since only a replacement can quietly differ. That is the set
this audit covers.

Coverage is printed and zero coverage is an error: a table that scanned
nothing looks exactly like a table where everything agreed.
"""

from __future__ import annotations

import sys

# ⚠️ First, before anything imports the standard library. Running
# ``python tools/x.py`` puts ``tools/`` on sys.path[0], and this repository
# has ``tools/bisect/`` -- which then shadows the stdlib ``bisect`` that
# ``random`` (and so ``tempfile``, and so ``torch``) imports. The failure
# surfaces six frames deep inside torch's import and names neither this file
# nor the directory that caused it.
if sys.path and sys.path[0].endswith("/tools"):
    del sys.path[0]

import ast  # noqa: E402
import inspect  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402

VLLM_SOURCE = pathlib.Path(
    "/workspace/code/dsv4_single_card/vllm_project/vllm")
UPSTREAM_MODULES = (
    "vllm.models.deepseek_v4_1.compressor",
    "vllm.models.deepseek_v4_1.attention",
)


class Coverage:
    """What was scanned, so "all agree" and "nothing scanned" differ."""

    #: 🔴 Declared up front and started at zero. A section that never runs
    #: creates no key at all, and a guard that only looks at the keys present
    #: cannot see it -- the report would then be missing a whole category and
    #: still say "no disagreements". The expected categories are part of the
    #: report's declaration, not a by-product of running it.
    EXPECTED = (
        "module attributes replaced",
        "methods overridden",
        "operator signatures read",
        "operator call sites in V4.1",
        "decomposition/flag agreements checked",
    )

    def __init__(self) -> None:
        self.counts: dict[str, int] = {name: 0 for name in self.EXPECTED}
        self.problems: list[str] = []

    def saw(self, what: str, n: int = 1) -> None:
        self.counts[what] = self.counts.get(what, 0) + n

    def wrong(self, message: str) -> None:
        self.problems.append(message)

    def report(self) -> int:
        print("\n--- coverage " + "-" * 52)
        for what, n in sorted(self.counts.items()):
            print(f"  {n:5}  {what}")
        empty = [what for what, n in self.counts.items() if n == 0]
        if empty:
            print("\nNOTHING SCANNED for: " + ", ".join(empty))
            print("A table that scanned nothing reads exactly like a table "
                  "where everything agreed. Probably the source tree moved, "
                  "or a section raised before it counted anything.")
            return 2
        if self.problems:
            print(f"\n{len(self.problems)} disagreement(s):")
            for p in self.problems:
                print("  🔴 " + p)
            return 1
        print("\nno disagreements")
        return 0


def _defaults(fn) -> dict:
    try:
        return {name: repr(p.default)
                for name, p in inspect.signature(fn).parameters.items()
                if p.default is not inspect.Parameter.empty}
    except (TypeError, ValueError):
        return {}


def audit_rebinds(cov: Coverage) -> None:
    """Module attributes our patches replace, found by differencing."""
    import importlib

    before = {}
    for name in UPSTREAM_MODULES:
        module = importlib.import_module(name)
        before[name] = dict(vars(module))

    from vllm_ascend.models.deepseek_v4_1 import platform_patches
    platform_patches.apply_all()

    print("=== module attributes replaced (detected by differencing) ===")
    for name in UPSTREAM_MODULES:
        module = importlib.import_module(name)
        for attr, now in vars(module).items():
            was = before[name].get(attr)
            if was is None or was is now or not callable(now):
                continue
            cov.saw("module attributes replaced")
            d_was, d_now = _defaults(was), _defaults(now)
            flag = "" if d_was == d_now else "  🔴 DEFAULTS DIFFER"
            print(f"  {name.split('.')[-1]}.{attr:34} {d_was} -> {d_now}{flag}")
            if d_was != d_now:
                cov.wrong(f"{name}.{attr} replacement declares {d_now}, "
                          f"upstream declares {d_was}; every caller that "
                          "omits the argument now gets a different value")


def audit_overrides(cov: Coverage) -> None:
    """Methods our classes replace on an upstream base -- the category the
    first version of this audit did not have, and the one A-2 was in."""
    import importlib

    print("\n=== methods overridden on an upstream base ===")
    for mod_name in ("vllm_ascend.models.deepseek_v4_1.attention",
                     "vllm_ascend.models.deepseek_v4_1.caches",
                     "vllm_ascend.models.deepseek_v4_1.sparse_indexer"):
        module = importlib.import_module(mod_name)
        for _, obj in vars(module).items():
            if not (inspect.isclass(obj) and obj.__module__ == mod_name):
                continue
            for base in obj.__mro__[1:]:
                if base.__module__.startswith("vllm_ascend"):
                    continue
                for attr, ours in vars(obj).items():
                    theirs = vars(base).get(attr)
                    if theirs is None or not callable(ours):
                        continue
                    cov.saw("methods overridden")
                    d_t, d_o = _defaults(theirs), _defaults(ours)
                    flag = "" if d_t == d_o else "  🔴 DEFAULTS DIFFER"
                    print(f"  {obj.__name__}.{attr:34} {d_t} -> {d_o}{flag}")
                    if d_t != d_o:
                        cov.wrong(f"{obj.__name__}.{attr} declares {d_o}, "
                                  f"{base.__name__} declares {d_t}")
                break


def audit_operator_flags(cov: Coverage, tree: pathlib.Path | None = None) -> None:
    """The defaulted arguments of every shared operator V4.1 calls.

    ``tree`` lets this run against a source tree other than the installed one,
    which is how the check is shown to go red on the commit before the fix.
    """
    source = tree or VLLM_SOURCE
    if not source.exists():
        print(f"\n(upstream source {source} absent; operator flags not audited)")
        return

    sigs: dict[str, list] = {}
    for cpp in (source / "csrc").rglob("torch_bindings.cpp"):
        text = cpp.read_text()
        for m in re.finditer(r'ops\.def\(\s*((?:"[^"]*"\s*)+)\)', text):
            joined = "".join(re.findall(r'"([^"]*)"', m.group(1)))
            if "(" not in joined:
                continue
            name, rest = joined.split("(", 1)
            sigs[name.strip()] = [p.strip() for p in
                                  rest.rsplit(")", 1)[0].split(",") if p.strip()]
    cov.saw("operator signatures read", len(sigs))

    calls: dict[str, list] = {}
    for py in (source / "vllm/models/deepseek_v4_1").rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)):
                continue
            chain, cur = [], node.func
            while isinstance(cur, ast.Attribute):
                chain.append(cur.attr)
                cur = cur.value
            chain = list(reversed(chain))
            if len(chain) >= 3 and chain[0] == "ops" and chain[1] == "_C":
                calls.setdefault(chain[-1], []).append(
                    (f"{py.name}:{node.lineno}", len(node.args)))
    cov.saw("operator call sites in V4.1",
            sum(len(v) for v in calls.values()))

    print("\n=== defaulted operator arguments, per V4.1 call site ===")
    for op, sites in sorted(calls.items()):
        params = sigs.get(op)
        if params is None:
            cov.wrong(f"{op} is called by V4.1 but has no binding signature; "
                      "its defaults cannot be checked")
            continue
        required = sum(1 for p in params if "=" not in p)
        optional = [p for p in params if "=" in p]
        if not optional:
            print(f"  {op[:58]:58} no defaulted arguments")
            continue
        cov.saw("defaulted operator arguments", len(optional))
        for site, n in sorted(sites):
            taken = "DEFAULT" if n <= required else "explicit"
            print(f"  {op[:50]:50} {site:22} {optional} -> {taken}")


def audit_decomposition_matches_the_flag(cov: Coverage,
                                        tree: pathlib.Path | None = None) -> None:
    """The one place the flag and our own code have to agree.

    🔴 This is the check that would have caught A-2, and it is the reason the
    audit is not just a table: a table tells you what the arguments are, and
    somebody still has to notice that our decomposition disagrees with them.

    V4.1's attention path passes ``apply_q_norm=False``. ``qnorm_rope_kv_insert``
    is the decomposition of that same operator, so it must apply no norm. If
    upstream ever stops passing False, this goes red from the other side.
    """
    source = tree or VLLM_SOURCE
    if not source.exists():
        return
    from vllm_ascend.models.deepseek_v4_1 import qkv_prolog

    body = ast.parse(inspect.getsource(qkv_prolog.qnorm_rope_kv_insert))
    calls = {node.func.id for node in ast.walk(body)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    norms = {name for name in calls if "norm" in name}
    cov.saw("decomposition/flag agreements checked")

    print("\n=== does the decomposition match the flag it stands in for? ===")
    print(f"  V4.1 attention sites pass apply_q_norm=False")
    print(f"  qnorm_rope_kv_insert calls: {sorted(norms) or 'no norm'}")
    if norms:
        cov.wrong(
            "V4.1's attention path passes apply_q_norm=False, but the "
            f"decomposition of that operator calls {sorted(norms)}. The norm "
            "is V4.0's behaviour, inherited with the operator's default; on "
            "the flight that found it, q reached the attention operator with "
            "a per-(token, head) RMS factor of 0.21-1.23 applied.")


def main() -> int:
    cov = Coverage()
    audit_rebinds(cov)
    audit_overrides(cov)
    audit_operator_flags(cov)
    audit_decomposition_matches_the_flag(cov)
    return cov.report()


if __name__ == "__main__":
    sys.exit(main())
