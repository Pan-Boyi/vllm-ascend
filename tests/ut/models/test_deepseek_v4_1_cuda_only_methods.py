# SPDX-License-Identifier: Apache-2.0
"""Every base method that reaches for a CUDA C++ extension must be handled.

The module docstring of vllm_ascend.models.deepseek_v4_1.attention already
said this: the base class calls ``torch.ops._C.fused_deepseek_v4_*`` in
``_fused_qnorm_rope_kv_insert``, those symbols do not exist on Ascend, and
"methods still to be filled in are marked below and raise rather than silently
doing the CUDA thing."

The documentation was accurate and was not treated as a criterion. The method
was neither overridden nor marked, so a launch reached it and failed with
``'_OpNamespace' '_C' object has no attribute ...`` -- the CUDA thing, minus
the silence.

So the promise is re-derived from upstream's source here rather than restated.
A base method that starts calling one of those symbols joins this list on its
own, and goes red until it is either implemented or declared.
"""

import inspect

import pytest
from vllm.models.deepseek_v4_1.attention import DeepseekV4Attention

from vllm_ascend.models.deepseek_v4_1.attention import \
    AscendDeepseekV41Attention

CUDA_EXTENSION = "torch.ops._C.fused_deepseek_v4_"

#: Base methods reaching a CUDA extension that Ascend has not implemented yet,
#: each with why it is still here. Emptying this list is the goal; adding to it
#: is a decision, which is the point of it being written down.
DECLARED_UNIMPLEMENTED: dict[str, str] = {}


def _methods_calling_the_cuda_extension() -> set[str]:
    found = set()
    for name, member in vars(DeepseekV4Attention).items():
        if not callable(member):
            continue
        try:
            source = inspect.getsource(member)
        except (OSError, TypeError):
            continue
        if CUDA_EXTENSION in source:
            found.add(name)
    return found


def test_the_enumeration_finds_something():
    """Zero coverage here would pass every assertion below.

    If upstream renames the op namespace, this list empties and the checks
    become vacuous -- which looks exactly like compliance.
    """
    assert _methods_calling_the_cuda_extension(), (
        f"no base method references {CUDA_EXTENSION!r}; the spelling changed "
        "and every check in this file just became vacuous")


@pytest.mark.parametrize("name", sorted(_methods_calling_the_cuda_extension()))
def test_each_is_overridden_or_declared(name):
    """Overridden, or written down as not yet implemented. Not neither.

    "Neither" is what happened: the method fell through to the base class and
    called an operator that does not exist here, several frames below anything
    that mentions Ascend.
    """
    overridden = name in vars(AscendDeepseekV41Attention)
    declared = name in DECLARED_UNIMPLEMENTED
    assert overridden or declared, (
        f"DeepseekV4Attention.{name} calls {CUDA_EXTENSION}*, which does not "
        f"exist on Ascend. Override it in AscendDeepseekV41Attention, or add "
        f"it to DECLARED_UNIMPLEMENTED with the reason. Falling through gives "
        f"\"'_OpNamespace' '_C' object has no attribute\" at forward time.")


@pytest.mark.parametrize("name", sorted(DECLARED_UNIMPLEMENTED))
def test_a_declared_method_is_not_quietly_implemented(name):
    """The list has to shrink when the work lands, not linger as a stale note."""
    assert name not in vars(AscendDeepseekV41Attention), (
        f"{name} is implemented but still listed as unimplemented; remove it "
        "from DECLARED_UNIMPLEMENTED so the list keeps meaning something")


def test_the_docstring_still_makes_the_promise_this_checks():
    """If the promise is dropped, this file is enforcing nothing anyone claims.

    Worth catching: a check whose stated basis has disappeared is one nobody
    can argue with and nobody can rely on.
    """
    from vllm_ascend.models.deepseek_v4_1 import attention

    promise = " ".join(attention.__doc__.split())
    assert "raise rather than silently doing the CUDA thing" in promise


def test_every_attribute_forward_reads_is_stored_at_construction():
    """nn.Module defers a missing attribute to the first forward.

    forward_mqa reads self.vllm_config three times and nothing assigned it, so
    the failure arrived on a card, inside upstream frames, as
    "'AscendDeepseekV41Attention' object has no attribute 'vllm_config'".
    Construction-time state that only the forward reads has no earlier check --
    so the check is here, derived from the source rather than from a list.
    """
    import ast
    import inspect

    import textwrap

    # dedent, not cleandoc: cleandoc strips the leading indent of the *first*
    # line only, which for a class leaves the body indented under an
    # unindented `class` line and fails to parse.
    tree = ast.parse(textwrap.dedent(
        inspect.getsource(AscendDeepseekV41Attention)))

    read, assigned = set(), set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            continue
        if isinstance(node.ctx, ast.Store):
            assigned.add(node.attr)
        else:
            read.add(node.attr)

    # Methods are read through self too and are defined, not assigned; so are
    # class attributes (e.g. the backend_cls ClassVar).
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assigned.add(node.name)
    for node in tree.body[0].body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
        elif isinstance(node, ast.Assign):
            assigned.update(t.id for t in node.targets
                            if isinstance(t, ast.Name))

    # Names the base class provides are legitimately read without assignment
    # here. dir() is not enough: most of them are instance attributes set in
    # the base's __init__, which is exactly the kind this is looking for, so
    # the base's stores have to be parsed the same way.
    base_tree = ast.parse(textwrap.dedent(
        inspect.getsource(DeepseekV4Attention)))
    base_names = set(dir(DeepseekV4Attention))
    for node in ast.walk(base_tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self"
                and isinstance(node.ctx, ast.Store)):
            base_names.add(node.attr)

    missing = {n for n in read - assigned if n not in base_names}
    assert not missing, (
        f"read on self but assigned nowhere and absent from the base class: "
        f"{sorted(missing)}. These surface as AttributeError at forward time.")
