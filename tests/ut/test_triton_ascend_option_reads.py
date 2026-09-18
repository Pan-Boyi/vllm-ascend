"""`enable_fp_fusion` is accepted by the Ascend Triton backend and never read.

Upstream V4.1 passes ``enable_fp_fusion=False`` to ``hc_collapse_triton``
(``vllm/model_executor/kernels/mhc/triton.py``) with the comment "Preserve the separate FP32
multiply and sum in the Torch reference". On CUDA and ROCm that flag reaches the compiler. On the
Ascend backend it is a dataclass field and nothing else: declared once, loaded nowhere. Passing it
raises no error and changes no code, so the numerical intent in that comment is silently absent.

This test is NOT here because that is a bug. It is here because it decides a downstream
question. Today the flag is inert, so the kernel's multiply-add may be contracted to an FMA, which
makes the result *more* accurate than the Torch reference rather than less -- the consequence is
that this operator cannot be compared bit-for-bit against a CUDA/Torch golden, and dsv41-golden's
protocol absorbs it in tolerance instead of budgeting for it (the deviation is data-dependent and
two-sided, not one-directional). If the backend ever starts reading the flag, that reasoning has to
be redone: ``enable_fp_fusion=False`` would then be honoured and a bit-exact comparison might
become correct again.

So a failure here means "re-examine the tolerance decision", not "something broke". It is deliberately
phrased that way in the assertion message, because a test that fails on an improvement gets deleted
if nobody wrote down why it exists.

The scan is an AST walk rather than a grep: a grep cannot tell the field declaration apart from a
read, and the whole point is the difference between those two.

Card-free. Skips cleanly when the Ascend Triton backend is not installed.
"""

import ast
import pathlib

import pytest

from tests.ut.base import TestBase

OPTION = "enable_fp_fusion"


def _ascend_backend_dir() -> pathlib.Path | None:
    try:
        import triton
    except ImportError:
        return None
    candidate = pathlib.Path(triton.__file__).parent / "backends" / "ascend"
    return candidate if candidate.is_dir() else None


def _scan(root: pathlib.Path) -> tuple[list[str], list[str]]:
    """Return (declarations, reads) of OPTION across the backend's sources."""
    declarations: list[str] = []
    reads: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root)
        for node in ast.walk(tree):
            # `enable_fp_fusion: bool = True` inside the options dataclass.
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == OPTION:
                declarations.append(f"{rel}:{node.lineno}")
            elif isinstance(node, ast.Name) and node.id == OPTION and isinstance(node.ctx, ast.Load):
                reads.append(f"{rel}:{node.lineno} (name load)")
            elif isinstance(node, ast.Attribute) and node.attr == OPTION:
                reads.append(f"{rel}:{node.lineno} (attribute access)")
    return declarations, reads


class TestTritonAscendOptionReads(TestBase):
    def test_enable_fp_fusion_is_declared_and_never_read(self):
        root = _ascend_backend_dir()
        if root is None:
            pytest.skip("the Ascend Triton backend is not installed in this environment")

        declarations, reads = _scan(root)

        # Zero coverage must fail: if the field disappears entirely, this test would otherwise pass
        # while checking nothing, and the option would be silently rejected instead of ignored.
        self.assertTrue(
            declarations,
            f"{OPTION} is no longer declared anywhere under {root}. The backend may now REJECT it, "
            f"which is a different failure from ignoring it: upstream passes it, so that would be "
            f"a hard error rather than a silent no-op.",
        )

        self.assertEqual(
            reads,
            [],
            f"{OPTION} is now READ by the Ascend Triton backend at {reads} (declared at "
            f"{declarations}). This is probably an IMPROVEMENT, not a regression -- do not delete "
            f"this test to make it green. It means enable_fp_fusion=False is honoured now, so "
            f"hc_collapse_triton's multiply and add stay separate and the operator may have become "
            f"bit-comparable against the Torch reference. Re-examine the tolerance that was granted "
            f"on the assumption that it was not, and tell dsv41-golden.",
        )
