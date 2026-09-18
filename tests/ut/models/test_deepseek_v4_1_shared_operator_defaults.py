# SPDX-License-Identifier: Apache-2.0
"""One operator, two models, one boolean -- checked rather than remembered.

A-2 was not a typo. The fused q-norm/RoPE/insert operator is shared by V4.0
and V4.1 and its signature ends ``bool apply_q_norm=True``; V4.0 omits the
argument and takes the norm, V4.1's attention path passes ``False``. This
line's decomposition was transcribed from a prose contract -- correctly cited,
by file and line -- that described **V4.0's** call
(``deepseek_v4/cpu/cpu_sparse.py:426``). The sentence was true; its subject was
the other model. q reached the attention operator with a per-(token, head) RMS
factor of 0.21-1.23 applied to it.

The lesson that generalises is not "read the contract". It is that a prose
contract for a shared operator describes **one caller's arguments**, and the
argument that differs between callers is exactly the kind prose omits -- a
defaulted flag. So the scope is derived here instead of recalled: signatures
and their defaults come from the bindings, actual arguments from an AST walk
of both models' call sites.

Everything here is static. It says what the source declares, not what ran.
"""

import ast
import importlib
import inspect
import pathlib
import re

import pytest

VLLM_SOURCE = pathlib.Path(
    "/workspace/code/dsv4_single_card/vllm_project/vllm")

#: Every upstream callable this line rebinds, and what it rebinds to. Not a
#: hand list: a test below re-derives it from the modules' own tables and
#: fails if this falls behind.
REBINDS = (
    ("vllm.models.deepseek_v4_1.compressor", "rope_quant_insert",
     "vllm_ascend.models.deepseek_v4_1.compressor_insert",
     "rope_quant_insert_ascend"),
    ("vllm.models.deepseek_v4_1.compressor", "fused_save_compress_norm",
     "vllm_ascend.models.deepseek_v4_1.compressor_insert",
     "fused_save_compress_norm_ascend"),
    ("vllm.models.deepseek_v4_1.attention", "indexer_k_norm_rope_store",
     "vllm_ascend.models.deepseek_v4_1.indexer_k_store",
     "indexer_k_norm_rope_store_ascend"),
    ("vllm.models.deepseek_v4_1.attention", "fused_indexer_q_rope_quant",
     "vllm_ascend.models.deepseek_v4_1.indexer_q",
     "fused_indexer_q_rope_quant_ascend"),
)


def _defaults(fn) -> dict:
    return {name: repr(p.default)
            for name, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty}


@pytest.mark.parametrize("upmod,upname,ourmod,ourname", REBINDS)
def test_a_replacement_declares_the_same_defaults_as_what_it_replaces(
        upmod, upname, ourmod, ourname):
    """The mechanism that produced A-2, at the Python level.

    A replacement that declares a different default silently changes what
    every caller that omits the argument gets -- and the callers that omit it
    are the ones nobody will re-read.
    """
    upstream = getattr(importlib.import_module(upmod), upname)
    ours = getattr(importlib.import_module(ourmod), ourname)
    assert _defaults(ours) == _defaults(upstream), (
        f"{ourname} declares different defaults from {upname}; a caller that "
        "omits the argument now gets a different value on this platform")


def test_the_rebind_list_here_matches_the_modules_own_tables():
    """A list that falls behind is worse than no list: it reads as complete."""
    from vllm_ascend.models.deepseek_v4_1 import compressor_insert

    declared = {name for name, _ in compressor_insert._REBINDS}
    listed = {upname for _, upname, ourmod, _ in REBINDS
              if ourmod.endswith("compressor_insert")}
    assert declared == listed, (
        f"compressor_insert._REBINDS is {sorted(declared)} but this file "
        f"checks {sorted(listed)}")


# ------------------------------- the operator-level flag, derived not recalled

def _binding_defaults() -> dict:
    """``op name -> [params]`` from the bindings, defaults included."""
    sigs = {}
    for cpp in (VLLM_SOURCE / "csrc").rglob("torch_bindings.cpp"):
        text = cpp.read_text()
        for m in re.finditer(r'ops\.def\(\s*((?:"[^"]*"\s*)+)\)', text):
            joined = "".join(re.findall(r'"([^"]*)"', m.group(1)))
            if "(" not in joined:
                continue
            name, rest = joined.split("(", 1)
            sigs[name.strip()] = [p.strip() for p in
                                  rest.rsplit(")", 1)[0].split(",") if p.strip()]
    return sigs


def _ops_calls(root: pathlib.Path) -> list:
    """``(op, file:line, n_positional)`` for every ``torch.ops._C.*`` call."""
    found = []
    for py in root.rglob("*.py"):
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
                found.append((chain[-1], f"{py.name}:{node.lineno}",
                              len(node.args)))
    return found


@pytest.mark.skipif(not VLLM_SOURCE.exists(),
                    reason="upstream source tree not present")
def test_the_attention_path_of_v41_turns_the_shared_q_norm_off():
    """The fact the decomposition has to match, derived from the source.

    If upstream ever stops passing False here, this decomposition -- which
    applies no norm -- becomes wrong in the other direction, and nothing else
    in this repository would notice.
    """
    sigs = _binding_defaults()
    family = [op for op in sigs
              if op.startswith("fused_deepseek_v4_qnorm_rope_kv_rope")]
    assert family, "the operator family was renamed; this check is vacuous"

    for op in family:
        params = sigs[op]
        required = sum(1 for p in params if "=" not in p)
        optional = [p for p in params if "=" in p]
        # The whole point: exactly one defaulted parameter, and it is this one.
        assert len(optional) == 1 and "apply_q_norm" in optional[0], (
            f"{op} now has optional parameters {optional}; each one is a "
            "default this line could inherit without noticing")

        calls = [(site, n) for name, site, n in
                 _ops_calls(VLLM_SOURCE / "vllm/models/deepseek_v4_1")
                 if name == op]
        attention = [(site, n) for site, n in calls
                     if site.startswith("attention.py")]
        assert attention, f"no V4.1 attention call site found for {op}"
        for site, n in attention:
            assert n > required, (
                f"V4.1 {site} now omits apply_q_norm for {op}, so it takes "
                "the default True -- and qkv_prolog applies no norm at all")


@pytest.mark.skipif(not VLLM_SOURCE.exists(),
                    reason="upstream source tree not present")
def test_the_dspark_sites_are_known_to_differ_and_are_not_ours():
    """Scope, written down because the first version of this claim said
    'V4.1 passes False at every call site'.

    DSpark omits the argument and takes True, exactly as V4.0 does. It does
    not reach this line's decomposition -- it calls torch.ops._C directly, so
    on Ascend it fails on the missing symbol rather than silently norming --
    but a claim about 'V4.1' that is false of six call sites is the same
    over-generalisation that caused the defect.
    """
    sigs = _binding_defaults()
    for op, params in sigs.items():
        if not op.startswith("fused_deepseek_v4_qnorm_rope_kv_rope"):
            continue
        required = sum(1 for p in params if "=" not in p)
        dspark = [(site, n) for name, site, n in
                  _ops_calls(VLLM_SOURCE / "vllm/models/deepseek_v4_1")
                  if name == op and site.startswith("dspark.py")]
        for site, n in dspark:
            assert n == required, (
                f"V4.1 {site} now passes apply_q_norm explicitly; the "
                "attention path and DSpark no longer differ, and this "
                "file's scope note is stale")
