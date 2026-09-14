# SPDX-License-Identifier: Apache-2.0
"""A scale parameter holds bytes, so it must be declared as bytes.

🔴 The defect this pins cost the whole MoE output at every layer. The DS
w4a8 MoE scheme declared its two expert scale parameters ``float8_e8m0fnu``
while every other scale parameter in the tree is ``uint8``. The loader moves
the checkpoint's scale as raw bytes, so a copy into an E8M0 parameter is a
*value* conversion rather than a byte move: every byte in 96..191 rounds to
the same power of two and lands on 134. The checkpoint's own bytes are
119..122 (w1/w3) and 119..123 (w2) -- all inside that window -- so all
70,778,880 of them collapsed to one value.

⚠️ Both dtypes are one byte wide and every shape matched, which is why
nothing downstream complained: the declaration turned a byte move into an
arithmetic one without changing anything that gets checked.
"""

import ast
import pathlib

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parents[3] / "vllm_ascend"
_QUANT = _ROOT / "quantization"


def _mx_block_scale_params():
    """Every MX block-scale parameter declared with ``torch.empty``.

    🔴 Found by walking the AST rather than by listing the names I know: a
    hand-written list cannot show the row it is missing, and the defect here
    was exactly one entry nobody had thought to look at.

    ⚠️ Two conditions, and both are load-bearing -- my first version used
    only "the name contains scale" and failed on seven parameters that are
    *correctly* float:

      · the module is an MX one (``mxfp`` in its name) -- MX block scales are
        E8M0 exponent **bytes**; w4a16/w8a16/w8a8 scales are genuine numbers
        and belong in a float dtype
      · the declared shape divides by ``group_size`` -- the block-scale
        signature. ``w8a8_mxfp8.py:532`` lives in an MX module but is a
        ``block_size`` fp8 scale and is rightly float32

    A criterion wider than the defect is not the safe direction it looks
    like: it fails on correct code, and the fix for that is to loosen it,
    which is how a criterion ends up unable to catch anything.
    """
    found = []
    for path in sorted(_QUANT.rglob("*.py")):
        if "mxfp" not in path.name:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            call = node.value
            if not (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "empty"):
                continue
            names = [
                t.slice.value for t in node.targets
                if isinstance(t, ast.Subscript)
                and isinstance(t.slice, ast.Constant)
                and isinstance(t.slice.value, str)
            ]
            if not names or "scale" not in names[0]:
                continue
            if "group_size" not in " ".join(ast.unparse(a) for a in call.args):
                continue
            dtype = next((ast.unparse(kw.value) for kw in call.keywords
                          if kw.arg == "dtype"), None)
            found.append((path.relative_to(_ROOT).as_posix(), node.lineno,
                          names[0], dtype))
    return found


def test_every_mx_block_scale_is_declared_as_a_byte_buffer():
    entries = _mx_block_scale_params()
    # Zero coverage must fail: a walk that found nothing looks exactly like a
    # walk in which everything passed.
    assert len(entries) >= 10, f"only found {len(entries)}: {entries}"

    wrong = [e for e in entries if e[3] != "torch.uint8"]
    assert not wrong, (
        "an MX block-scale parameter is declared with a dtype that gives it "
        "arithmetic meaning. The loader copies raw bytes into it, so the "
        "copy becomes a value conversion and the bytes are silently "
        "rewritten:\n  "
        + "\n  ".join(f"{f}:{ln} {name} declared {dt}"
                      for f, ln, name, dt in wrong))


def test_the_conversion_that_actually_happened():
    """The mechanism, with the checkpoint's real byte range.

    Kept separate from the dtype scan because the scan alone would let a
    reader think the rule is stylistic. It is not: this is what the wrong
    declaration does to the numbers.
    """
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("this torch build has no float8_e8m0fnu")

    # The bytes the checkpoint holds for layer 0's routed experts, read from
    # model.safetensors: w1/w3 119..122, w2 119..123.
    ckpt = torch.tensor([119, 120, 121, 122, 123], dtype=torch.uint8)

    as_bytes = torch.empty(5, dtype=torch.uint8)
    as_bytes.copy_(ckpt)
    assert as_bytes.tolist() == ckpt.tolist(), "a byte buffer moves bytes"

    as_e8m0 = torch.empty(5, dtype=torch.float8_e8m0fnu)
    as_e8m0.copy_(ckpt)
    got = as_e8m0.view(torch.uint8).tolist()
    assert got == [134] * 5, got
    assert len(set(got)) == 1, (
        "the signature of this defect is a CONSTANT, not noise and not a "
        "shift -- which is why 'all zero / shifted / reshaped' did not "
        "describe it")


def test_the_collapse_window_covers_the_whole_checkpoint_range():
    """Why not one byte survived.

    ⚠️ The reason this produced a single value rather than a few is that the
    checkpoint's entire range sits inside one power-of-two bucket. Stated as
    a property so it stays true if the checkpoint's range moves.
    """
    if not hasattr(torch, "float8_e8m0fnu"):
        pytest.skip("this torch build has no float8_e8m0fnu")

    allb = torch.arange(256, dtype=torch.uint8)
    collapsed = allb.to(torch.float8_e8m0fnu).view(torch.uint8)
    window = set((collapsed == 134).nonzero().flatten().tolist())
    assert set(range(119, 125)).issubset(window)
    assert len(window) == 96, len(window)
