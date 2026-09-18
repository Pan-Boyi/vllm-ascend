# SPDX-License-Identifier: Apache-2.0
"""Routed-expert frequency counter (M0c) -- CPU only.

Covers: off by default and guarded at every call site, placement before
log2phy, profile-run exclusion, decode/prefill split, DP padding and the
ALLGATHER DP-gathered rows, one file per rank, the report's invariant, and the
graph-mode refusal. Mutation arms show the checks can fail.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import expert_counter as ec

ROOT = pathlib.Path(__file__).resolve().parents[3]
ROUTED_EXPERTS = ROOT / "vllm_ascend/ops/fused_moe/routed_experts.py"
REPORT = ROOT / "tools/kt_offload/expert_count_report.py"


@pytest.fixture
def counter(monkeypatch, tmp_path):
    ctx = SimpleNamespace(split=(0, 0), rank=0, dp=1, gathered=(False, None), capturing=False)
    monkeypatch.setattr(ec, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(ec, "ENABLED", True)
    monkeypatch.setattr(ec, "_S", None)
    monkeypatch.setattr(ec, "_capturing", lambda: ctx.capturing)
    monkeypatch.setattr(ec, "_batch_split", lambda: ctx.split)
    monkeypatch.setattr(ec, "_dp_info", lambda: (ctx.rank, ctx.dp))
    monkeypatch.setattr(ec, "_rows_are_dp_gathered", lambda: ctx.gathered)
    monkeypatch.setattr(ec.atexit, "register", lambda fn: None)
    return ctx


def _layer(i):
    return SimpleNamespace(layer_name=f"model.layers.{i}.ffn.experts")


def _ids(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.argsort(torch.rand(rows, 384, generator=g), dim=1)[:, :6].to(torch.int32)


def _expected(ids):
    return torch.bincount(ids.reshape(-1).long(), minlength=384)


# ---------------------------------------------------------------------------
# off path
# ---------------------------------------------------------------------------


def test_env_default_is_off(monkeypatch):
    from vllm_ascend import envs

    monkeypatch.delenv("VLLM_ASCEND_DSV41_EXPERT_COUNT", raising=False)
    assert envs.VLLM_ASCEND_DSV41_EXPERT_COUNT == ""


def _guarded_counter_calls(src: str):
    guarded, unguarded = [], []

    def is_guard(test):
        return (
            isinstance(test, ast.Attribute) and test.attr == "ENABLED" and ast.unparse(test.value) == "_expert_counter"
        )

    def visit(node, inside):
        if isinstance(node, ast.If) and is_guard(node.test):
            for c in node.body:
                visit(c, True)
            for c in node.orelse:
                visit(c, inside)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if ast.unparse(node.func.value) == "_expert_counter":
                (guarded if inside else unguarded).append(node.lineno)
        for c in ast.iter_child_nodes(node):
            visit(c, inside)

    visit(ast.parse(src), False)
    return guarded, unguarded


def test_every_counter_call_is_guarded():
    guarded, unguarded = _guarded_counter_calls(ROUTED_EXPERTS.read_text())
    assert unguarded == [] and len(guarded) == 2, (guarded, unguarded)  # check_config + record


def test_guard_mutation_is_red():
    guarded, unguarded = _guarded_counter_calls("def f(x):\n    _expert_counter.record(x, 1, False)\n")
    assert unguarded == [2]


def _record_before_log2phy(src: str) -> bool:
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_select_experts")
    rec = [
        n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "_expert_counter.record"
    ]
    l2p = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Subscript) and ast.unparse(n.value) == "self.log2phy"]
    return len(rec) == 1 and len(l2p) == 1 and rec[0] < l2p[0]


def test_record_sits_before_log2phy():
    assert _record_before_log2phy(ROUTED_EXPERTS.read_text())


def test_record_before_log2phy_mutation_is_red():
    """Mutant: the guarded record() block moved below the log2phy mapping (physical ids)."""
    src = ROUTED_EXPERTS.read_text()
    block_start = src.index("        if _expert_counter.ENABLED:\n            # Logical ids")
    block_end = src.index("        if self.log2phy is not None:\n", block_start)
    block = src[block_start:block_end]
    rest = src[:block_start] + src[block_end:]
    anchor = "            topk_ids = self.log2phy[topk_ids]\n"
    mutated = rest.replace(anchor, anchor + block, 1)
    assert mutated != src and len(mutated) == len(src)
    assert not _record_before_log2phy(mutated)


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------


def test_decode_prefill_split_and_padding(counter):
    counter.split = (3, 10)  # 3 decode tokens, 10 real tokens; router saw 16 rows (padding)
    ids = _ids(16, 1)
    ec.record(_layer(5), ids, False)
    s = ec._S
    assert torch.equal(s["counts"]["decode"][5], _expected(ids[:3]))
    assert torch.equal(s["counts"]["prefill"][5], _expected(ids[3:10]))
    assert s["tokens"]["decode"][5] == 3 and s["tokens"]["prefill"][5] == 7
    assert int(s["counts"]["decode"].sum() + s["counts"]["prefill"].sum()) == 10 * 6  # padding not counted


def test_force_load_balance_is_excluded_and_counted(counter):
    counter.split = (4, 4)
    ec.record(_layer(0), _ids(4), True)
    ec.record(_layer(0), _ids(4), False)
    s = ec._S
    assert sum(s["skipped"].values()) == 1 and "force_load_balance" in next(iter(s["skipped"]))
    assert s["tokens"]["decode"][0] == 4 and s["steps"] == 1


def test_non_main_layers_skipped(counter):
    counter.split = (1, 1)
    ec.record(SimpleNamespace(layer_name="model.layers.41.ffn.experts"), _ids(1), False)
    ec.record(SimpleNamespace(layer_name="dspark.stages.0.ffn.experts"), _ids(1), False)
    assert sum(ec._S["skipped"].values()) == 2 and int(ec._S["counts"]["decode"].sum()) == 0


@pytest.mark.parametrize("rank", [0, 2, 3])
def test_allgather_dp_counts_only_own_slice(counter, rank):
    counter.rank, counter.dp, counter.gathered = rank, 4, (True, 8)
    counter.split = (5, 6)
    ids = _ids(32, 7)  # 4 ranks x 8 rows (max_tokens_across_dp)
    ec.record(_layer(1), ids, False)
    own = ids[rank * 8 : rank * 8 + 6]
    total = ec._S["counts"]["decode"][1] + ec._S["counts"]["prefill"][1]
    assert torch.equal(total, _expected(own))


def test_allgather_slice_mutation_is_red(counter):
    """Mutant: take the first rows as the M0c original did -> rank 2 would count rank 0's tokens."""
    ids = _ids(32, 7)
    own, mutant = ids[16:22], ids[0:6]
    assert not torch.equal(_expected(own), _expected(mutant))


def test_allgather_shape_mismatch_is_skipped(counter):
    counter.rank, counter.dp, counter.gathered = 1, 4, (True, 8)
    counter.split = (1, 1)
    ec.record(_layer(0), _ids(30), False)
    assert int(ec._S["counts"]["decode"].sum()) == 0 and sum(ec._S["skipped"].values()) == 1


def test_graph_mode_refused(counter):
    with pytest.raises(ec.ExpertCounterError):
        ec.check_config(SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False)))
    ec.check_config(SimpleNamespace(model_config=SimpleNamespace(enforce_eager=True)))
    counter.capturing = True
    with pytest.raises(ec.ExpertCounterError):
        ec.record(_layer(0), _ids(1), False)


# ---------------------------------------------------------------------------
# files per rank + report invariant
# ---------------------------------------------------------------------------


def _load_report():
    spec = importlib.util.spec_from_file_location("expert_count_report", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_one_file_per_rank_and_report_invariant(counter, tmp_path, monkeypatch):
    paths = []
    for rank in range(4):
        monkeypatch.setattr(ec, "_S", None)
        monkeypatch.setattr(ec.os, "getpid", lambda r=rank: 1000 + r)  # distinct processes
        counter.rank, counter.dp = rank, 4
        for step in range(3):
            counter.split = (2, 5)
            for layer in range(ec.N_LAYERS):
                ec.record(_layer(layer), _ids(5, seed=rank * 100 + step * 10 + layer), False)
        paths.append(ec.dump())
    assert len(set(paths)) == 4 and all(f"rank{r}_pid{1000 + r}.pt" in p for r, p in enumerate(paths))
    tot, tok, bad = _load_report().load(str(tmp_path))
    assert bad == 0
    assert torch.equal(tot["all"].sum(1), tok["all"] * 6)
    assert int(tok["all"][0]) == 4 * 3 * 5


def test_report_invariant_mutation_is_red(counter, tmp_path):
    counter.split = (1, 2)
    ec.record(_layer(0), _ids(2), False)
    path = ec.dump()
    d = torch.load(path)
    d["tokens"]["decode"][0] += 1  # tokens no longer match counts
    torch.save(d, path)
    _, _, bad = _load_report().load(str(tmp_path))
    assert bad == 1


def test_report_output_feeds_frequency_placement(counter, tmp_path):
    from vllm_ascend.kt_offload.placement import load_frequency_table

    counter.split = (1, 3)
    for layer in range(ec.N_LAYERS):
        ec.record(_layer(layer), _ids(3, seed=layer), False)
    ec.dump()
    rep = _load_report()
    tot, _, bad = rep.load(str(tmp_path))
    assert bad == 0
    import torch as _t

    _t.save({"counts": tot, "tokens": {}}, tmp_path / "expert_counts_total.pt")
    freq = load_frequency_table(str(tmp_path / "expert_counts_total.pt"))
    assert tuple(freq.shape) == (40, 384) and _t.equal(freq, tot["all"])
