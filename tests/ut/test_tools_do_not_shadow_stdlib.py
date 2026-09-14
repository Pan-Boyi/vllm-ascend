# SPDX-License-Identifier: Apache-2.0
"""``tools/`` must not shadow a stdlib module that start-up needs.

🔴 Structural, not incidental. Running ``python tools/<x>.py`` puts
``tools/`` at ``sys.path[0]``, so a package in there outranks the standard
library. ``tools/bisect/`` shadows the stdlib ``bisect``, which ``random``
imports, which ``tempfile`` imports, which ``torch`` imports -- the script
dies before its first statement.

⚠️ This has bitten twice. A per-script guard at the top of each file relies
on remembering, and remembering is what already failed; so the check lives
here, where a new directory trips it without anyone having to recall the
rule.

🔴 And the second time it cost more than a broken run: the L0 gate's own
red-first verification ran both arms and got exit 1 from each. The
broken-version arm looked exactly like the gate correctly refusing, and it
was the gate being dead -- a dead instrument impersonating its own negative
control.
"""

import ast
import pathlib
import sys

_TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"

#: Imported while the interpreter or torch starts, so shadowing one of these
#: breaks a script before any of its own code runs. Not an exhaustive list of
#: the stdlib -- an exhaustive list is not the point; these are the ones whose
#: shadowing is fatal rather than merely surprising.
_FATAL_IF_SHADOWED = frozenset({
    "bisect", "random", "tempfile", "types", "enum", "typing", "functools",
    "collections", "io", "os", "sys", "re", "json", "logging", "warnings",
    "abc", "copy", "math", "time", "platform", "struct", "hashlib", "base64",
    "pathlib", "inspect", "traceback", "importlib", "dataclasses", "queue",
    "socket", "select", "signal", "threading", "subprocess", "weakref",
    "contextlib", "itertools", "operator", "string", "textwrap", "shutil",
})


def test_no_tools_package_shadows_a_startup_stdlib_module():
    if not _TOOLS.is_dir():
        return
    shadowing = sorted(
        d.name for d in _TOOLS.iterdir()
        if d.is_dir() and (d / "__init__.py").exists()
        and d.name in _FATAL_IF_SHADOWED)
    # A known one is grandfathered: it is a real tool and renaming it would
    # break callers. It is listed rather than silently excluded so the list
    # is the record of what is still dangerous.
    grandfathered = {"bisect"}
    new = [n for n in shadowing if n not in grandfathered]
    assert not new, (
        f"tools/{new} shadow stdlib modules that are imported during "
        "start-up. Any script run as `python tools/<x>.py` will fail to "
        "import torch, with an error that names the stdlib module and not "
        "this directory. Either rename the directory, or strip the script's "
        "own directory from sys.path before importing torch.")
    assert shadowing == sorted(grandfathered), (
        f"the grandfathered list is stale: found {shadowing}, expected "
        f"{sorted(grandfathered)}. A name that left the tree should leave "
        "this list, or the list stops describing the tree.")


def test_every_tools_script_that_imports_torch_strips_its_own_directory():
    """The scripts that must survive the grandfathered shadow.

    🔴 Walked as an AST, not searched as text, and that is not fussiness.
    The first version used ``text.find("import torch")`` -- and the comment
    this file's own fix added, which *explains* the deferred ``import
    torch``, contains that literal earlier in the file than the fix itself.
    So the check failed on a file it had just been used to repair.

    ⚠️ Third time today that a substring check met a sentence about the thing
    it was looking for: once made a check falsely red, once falsely green,
    once this. A substring cannot tell a statement from prose describing a
    statement; the parser can.
    """
    if not _TOOLS.is_dir():
        return
    offenders = []
    for path in sorted(_TOOLS.glob("*.py")):
        tree = ast.parse(path.read_text())
        imports = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Import)
            and any(a.name.split(".")[0] == "torch" for a in n.names)
        ]
        if not imports:
            continue
        strips = [
            n.lineno for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Subscript)
                    and ast.unparse(t).startswith("sys.path") for t in n.targets)
        ]
        if not strips or min(strips) > min(imports):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} import torch without first stripping their own "
        "directory from sys.path, and tools/bisect shadows the stdlib "
        "bisect that torch's import chain needs. The failure is total -- the "
        "script cannot run one line -- and it reads like the script's own "
        "logic failing.")
