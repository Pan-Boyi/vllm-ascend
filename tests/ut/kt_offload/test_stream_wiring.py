# SPDX-License-Identifier: Apache-2.0
"""The streaming fork in forward_impl: when it fires, and what it must skip.

The decision half is driven directly. The skip half is checked structurally, by
an AST scan of the production file: a streamed layer that also runs ``route()``
/ ``merge()`` counts every offloaded expert twice without raising, and the
numbers stay plausible, so no tolerance check can see it. The scan carries
mutation arms proving it goes red when a guard is removed.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan

ROUTED_EXPERTS = pathlib.Path(kl.__file__).resolve().parents[1] / "ops/fused_moe/routed_experts.py"
H, TOPK, E = 8, 2, 16


def _vcfg(eager=True, sizes=(1, 2, 4, 8)):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=TOPK, num_hidden_layers=2
            ),
            model="/m",
            enforce_eager=eager,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE" if eager else "FULL_AND_PIECEWISE"),
            cudagraph_capture_sizes=list(sizes),
        ),
    )


class _SchemeStub:
    """What production keeps INSIDE layer.quant_method (an AscendMoEScheme).

    A bare ``object()`` would not catch a wrapper/scheme mix-up: nothing would
    check that what is handed down the chain carries what the chain reads.
    """

    quant_type = "w4a8_mxfp4"
    act_quant_type = "mxfp8"

    def get_weight(self, *a, **kw):
        return {}

    def get_dynamic_quant_param(self, *a, **kw):
        return {}


class _WrapperStub:
    """What production hands as ``layer.quant_method`` (AscendFusedMoEMethod)."""

    def __init__(self):
        self.quant_method = _SchemeStub()

    def process_weights_after_loading(self, module):
        pass


def _rt(monkeypatch, *, stream=True, min_tokens=512, chunk=4, eager=True, provider=object()):
    plan = build_plan([[1, 4, 6]], E)
    cfg = KTOffloadConfig(
        enabled=True, cpuinfer_threads=4, num_resident_experts=1,
        stream_prefill=stream, stream_prefill_min_tokens=min_tokens, stream_prefill_chunk_experts=chunk,
    )
    rt = kl.KTLayerOffload(cfg, plan.layer(0), 0, _vcfg(eager))
    rt._stream_slot_provider = provider
    monkeypatch.setattr(kl, "_capturing", lambda: False)
    return rt


def _call(rt, tokens=1024, in_profile_run=False):
    x = torch.zeros(tokens, H)
    ids = torch.zeros(tokens, TOPK, dtype=torch.int32)
    w = torch.ones(tokens, TOPK, dtype=torch.float32)
    return rt.maybe_stream_prefill(torch.nn.Module(), x, w, ids, in_profile_run=in_profile_run)


def test_switch_off_never_streams(monkeypatch):
    rt = _rt(monkeypatch, stream=False)
    assert _call(rt) is None


def test_below_the_threshold_takes_the_hybrid_path(monkeypatch):
    rt = _rt(monkeypatch, min_tokens=512)
    assert _call(rt, tokens=511) is None


def test_profile_run_is_not_streamed(monkeypatch):
    """Random routing measures a shape production never serves."""
    rt = _rt(monkeypatch)
    assert _call(rt, in_profile_run=True) is None


def test_a_captured_forward_is_not_streamed(monkeypatch):
    rt = _rt(monkeypatch)
    monkeypatch.setattr(kl, "_capturing", lambda: True)
    assert _call(rt) is None


def test_switch_on_without_a_slot_provider_raises(monkeypatch):
    """Fail closed: quietly returning None would make every 'streaming on'
    reading a hybrid reading."""
    rt = _rt(monkeypatch, provider=None)
    with pytest.raises(RuntimeError, match="no slot provider"):
        _call(rt)


def test_refusals_are_decided_before_any_work(monkeypatch):
    """No provider call, no chunk plan, for a forward that will not stream."""
    calls = []
    rt = _rt(monkeypatch, provider=lambda *a, **k: calls.append(a))
    assert _call(rt, tokens=8) is None
    assert calls == []


def _guarded_by_not_streamed(tree: ast.AST) -> dict[str, list[bool]]:
    """For each hook call, whether it sits under a ``streamed_out is None`` test."""
    found: dict[str, list[bool]] = {"route": [], "merge": [], "apply": []}

    def is_guard(test: ast.AST) -> bool:
        src = ast.unparse(test)
        return "streamed_out is None" in src

    def is_antiguard(test: ast.AST) -> bool:
        return "streamed_out is not None" in ast.unparse(test)

    def visit(node, guarded: bool, in_else_of_stream: bool):
        if isinstance(node, ast.If):
            g = guarded or is_guard(node.test)
            for child in node.body:
                visit(child, g, in_else_of_stream or is_antiguard(node.test))
            # the else of ``if streamed_out is not None`` is the hybrid branch
            for child in node.orelse:
                visit(child, guarded or is_antiguard(node.test), in_else_of_stream)
            return
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if name in found:
                target = ast.unparse(node.func.value)
                if name in ("route", "merge") and "kt_offload" not in target:
                    pass
                elif name == "apply" and "quant_method" not in target:
                    pass
                else:
                    found[name].append(guarded)
        for child in ast.iter_child_nodes(node):
            visit(child, guarded, in_else_of_stream)

    visit(tree, False, False)
    return found


def test_route_and_apply_are_skipped_when_streaming():
    found = _guarded_by_not_streamed(ast.parse(ROUTED_EXPERTS.read_text()))
    for name in ("route", "apply"):
        assert found[name], f"no {name}() call found -- the scan is looking at the wrong thing"
        assert all(found[name]), f"{name}() can run on a streamed forward: its result would be added twice"


# ``merge`` is checked separately: ``kt_offload.merge`` sits behind a single
# entry point, ``kt_merge()``, not inside ``forward_impl`` under
# ``streamed_out is None``. Same invariant -- a streamed forward must not merge
# -- but it now rests on three structural facts, all checked below.


def _func(tree: ast.AST, name: str) -> ast.FunctionDef:
    """The one function of this name, or a failure that says which is missing."""
    hits = [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
    assert len(hits) == 1, f"expected exactly one {name}(), found {len(hits)}"
    return hits[0]


def _merge_facts(src: str) -> dict[str, bool]:
    tree = ast.parse(src)

    def enclosing_funcs(node) -> set[str]:
        out = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and node in set(ast.walk(fn)):
                out.add(fn.name)
        return out

    kt_merge_calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "merge" and "kt_offload" in ast.unparse(n.func.value)
    ]
    kt_merge = _func(tree, "kt_merge")
    fwd = _func(tree, "forward_impl")

    # (1) every kt_offload.merge() lives inside kt_merge() -- the single entry point
    only_in_kt_merge = bool(kt_merge_calls) and all(
        "kt_merge" in enclosing_funcs(c) for c in kt_merge_calls)

    # (2) kt_merge() returns early on a streamed forward, before reaching merge()
    early_return = False
    for node in kt_merge.body:
        if isinstance(node, ast.If) and "_kt_streamed_forward" in ast.unparse(node.test):
            if any(isinstance(b, ast.Return) for b in node.body):
                early_return = True
            break
        if any(isinstance(c, ast.Call) and getattr(c.func, "attr", None) == "merge"
               for c in ast.walk(node)):
            break            # merge reached before any such guard

    # (3) forward_impl's own self.kt_merge() sits under ``streamed_out is None``
    inline_guarded = []

    def visit(node, guarded):
        if isinstance(node, ast.If):
            g = guarded or "streamed_out is None" in ast.unparse(node.test)
            for c in node.body:
                visit(c, g)
            for c in node.orelse:
                visit(c, guarded)
            return
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "kt_merge"):
            inline_guarded.append(guarded)
        for c in ast.iter_child_nodes(node):
            visit(c, guarded)

    visit(fwd, False)
    return {"only_in_kt_merge": only_in_kt_merge,
            "early_return": early_return,
            "inline_guarded": bool(inline_guarded) and all(inline_guarded)}


def test_a_streamed_forward_cannot_merge():
    f = _merge_facts(ROUTED_EXPERTS.read_text())
    assert f["only_in_kt_merge"], "kt_offload.merge() escaped kt_merge(): there is more than one merge site"
    assert f["early_return"], "kt_merge() does not return early on a streamed forward"
    assert f["inline_guarded"], "forward_impl calls kt_merge() outside ``streamed_out is None``"


def _mutate(old: str, new: str) -> str:
    """Replace, and refuse to return a source the replacement did not touch: a
    mutation arm whose mutation silently no-ops tests nothing."""
    src = ROUTED_EXPERTS.read_text()
    assert old in src, f"mutation target is gone from the source, so this arm tests nothing: {old!r}"
    return src.replace(old, new, 1)


def test_mut_merge_without_the_streamed_early_return_is_red():
    """Drop kt_merge()'s early return and the check must catch it."""
    src = _mutate('if getattr(self, "_kt_streamed_forward", False):', "if False:")
    assert not _merge_facts(src)["early_return"]


def test_mut_unguarded_inline_merge_is_red():
    """Hoist forward_impl's kt_merge() out of ``streamed_out is None`` -> red."""
    src = _mutate("if streamed_out is None:\n            # Nested rather than one ``and``",
                  "if True:\n            # Nested rather than one ``and``")
    assert not _merge_facts(src)["inline_guarded"]


def test_mut_a_second_merge_site_is_red():
    """A merge outside kt_merge() -> red, however well guarded it looks."""
    src = _mutate("        # The vLLM FusedMoE forward_impl does not return events.",
                  "        routed_out = self.kt_offload.merge(routed_out)\n"
                  "        # The vLLM FusedMoE forward_impl does not return events.")
    assert not _merge_facts(src)["only_in_kt_merge"]


def test_mut_unguarded_route_is_red():
    src = ROUTED_EXPERTS.read_text().replace(
        "if kt_offload is not None and streamed_out is None:\n            # Logical ids", "if kt_offload is not None:\n            # Logical ids"
    )
    found = _guarded_by_not_streamed(ast.parse(src))
    assert not all(found["route"])


def test_the_fork_runs_before_route():
    """Order matters: route() submits CPU work, so the decision must precede it."""
    src = ROUTED_EXPERTS.read_text()
    assert src.index("maybe_stream_prefill") < src.index("kt_offload.route(")


def test_success_path_binds_against_the_real_chain_signature(monkeypatch):
    """Every arm above returns before the success path, so the call is driven for
    real here and its captured kwargs are bound against the *real* signature: a
    rename on either side goes red instead of raising TypeError only on-card."""
    import inspect

    from vllm_ascend.kt_offload import stream_chain

    real_sig = inspect.signature(stream_chain.run_streamed_layer)
    seen = {}

    def fake(**kwargs):
        seen.update(kwargs)
        real_sig.bind(**kwargs)          # raises TypeError on a name mismatch
        return SimpleNamespace(output=torch.zeros(1), coverage="cov")

    monkeypatch.setattr(stream_chain, "run_streamed_layer", fake)
    rt = _rt(monkeypatch, provider=lambda k, ids: None)

    layer = SimpleNamespace(quant_method=_WrapperStub(), moe_config=object())
    out = rt.maybe_stream_prefill(layer, torch.zeros(1024, H), torch.ones(1024, TOPK),
                                  torch.zeros(1024, TOPK, dtype=torch.int32), in_profile_run=False)

    assert out is not None and rt.stream_coverage == "cov"
    # The chain gets the SCHEME, not the wrapper. Asserting identity with the
    # inner object rather than "it is not None" catches both directions: handing
    # down the wrapper, and handing down some third object.
    assert seen["quant_method"] is layer.quant_method.quant_method
    assert seen["quant_method"] is not layer.quant_method
    assert seen["resident_layer"] is layer
    assert seen["slot_for"] is rt._stream_slot_provider
    assert seen["strict"] is True
    # The fork skips apply(), so an expert outside the plan contributes nothing
    # at all: the plan must cover all of them, not just the offloaded ones.
    assert sorted(seen["plan"].covered) == list(range(E))


def test_stream_modules_import_standalone():
    """Import each streaming module first in a fresh process, like a script does.

    Under pytest, conftest has already initialised ``vllm_ascend.ops`` before any
    test imports anything, so a module-scope import cycle inside kt_offload is
    invisible here but kills a script on its first import.
    """
    import subprocess
    import sys

    for mod in ("vllm_ascend.kt_offload.stream_chain",
                "vllm_ascend.kt_offload.stream_prefill",
                "vllm_ascend.kt_offload.resident_swap",
                "vllm_ascend.kt_offload.weight_source"):
        r = subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"{mod} cannot be imported on its own:\n{r.stderr[-2000:]}"


def test_the_streamed_plan_covers_every_expert(monkeypatch):
    """Whatever the plan misses contributes nothing, because the fork skips
    apply(). A plan-vs-plan coverage check cannot see that, so the expectation
    comes from the layer's expert count."""
    from vllm_ascend.kt_offload import stream_chain

    seen = {}
    monkeypatch.setattr(stream_chain, "run_streamed_layer",
                        lambda **kw: (seen.update(kw), SimpleNamespace(output=torch.zeros(1), coverage=None))[1])
    rt = _rt(monkeypatch, provider=lambda k, ids: None)
    layer = SimpleNamespace(quant_method=_WrapperStub(), moe_config=object())
    rt.maybe_stream_prefill(layer, torch.zeros(1024, H), torch.ones(1024, TOPK),
                            torch.zeros(1024, TOPK, dtype=torch.int32), in_profile_run=False)
    assert sorted(seen["plan"].covered) == list(range(E)), (
        "the plan must cover every expert of the layer; anything it misses is dropped")
    assert sorted(seen["expected_expert_ids"]) == list(range(E))
