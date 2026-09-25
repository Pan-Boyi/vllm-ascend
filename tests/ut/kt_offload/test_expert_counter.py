# SPDX-License-Identifier: Apache-2.0
"""Routed-expert frequency counter -- CPU only.

The capture predicate runs on the NPU in production, so a test that stubs
``_capturing`` away has zero coverage of it: the fixture instead installs a fake
``torch.npu`` whose ``is_current_stream_capturing`` answers what the test says
(True / False / raises / a non-bool) or is absent, and the real predicate runs
against it. Not claimed here: that on the card the predicate is True exactly for
the stream the counter's ``index_add_`` is enqueued on.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.kt_offload import expert_counter as ec

ROOT = pathlib.Path(__file__).resolve().parents[3]
ROUTED_EXPERTS = ROOT / "vllm_ascend/ops/fused_moe/routed_experts.py"


def _install_npu(monkeypatch, ctx):
    """Model the NPU side of the predicate: ``torch.npu.is_current_stream_capturing`` answers ``ctx.capturing``
    (an exception instance is raised). Undone by monkeypatch, whether or not torch_npu is installed here."""

    def is_current_stream_capturing():
        if isinstance(ctx.capturing, BaseException):
            raise ctx.capturing
        return ctx.capturing

    fake = SimpleNamespace(is_current_stream_capturing=is_current_stream_capturing)
    monkeypatch.setattr(torch, "npu", fake, raising=False)


def _arm(mod, monkeypatch, tmp_path, ctx):
    monkeypatch.setattr(mod, "OUT_DIR", str(tmp_path))
    monkeypatch.setattr(mod, "ENABLED", True)
    monkeypatch.setattr(mod, "_S", None)
    monkeypatch.setattr(mod, "_UNKNOWN_WARNED", False)
    monkeypatch.setattr(mod, "_CONFIG_ENFORCE_EAGER", None)
    monkeypatch.setattr(mod, "_batch_split", lambda: ctx.split)
    monkeypatch.setattr(mod, "_dp_info", lambda: (ctx.rank, ctx.dp))
    monkeypatch.setattr(mod, "_rows_are_dp_gathered", lambda: ctx.gathered)
    monkeypatch.setattr(mod.atexit, "register", lambda fn: None)
    return mod


def _ctx():
    return SimpleNamespace(split=(0, 0), rank=0, dp=1, gathered=(False, None), capturing=False)


@pytest.fixture
def env(monkeypatch):
    """The runtime environment of the predicate. ``env.capturing`` is what the fake NPU answers;
    ``env.no_npu()`` removes torch.npu from the process."""
    ctx = _ctx()
    _install_npu(monkeypatch, ctx)
    ctx.no_npu = lambda: monkeypatch.delattr(torch, "npu", raising=False)
    return ctx


@pytest.fixture
def counter(env, monkeypatch, tmp_path):
    _arm(ec, monkeypatch, tmp_path, env)
    return env


def _layer(i):
    return SimpleNamespace(layer_name=f"model.layers.{i}.ffn.experts")


def _ids(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.argsort(torch.rand(rows, 384, generator=g), dim=1)[:, :6].to(torch.int32)


def _expected(ids):
    return torch.bincount(ids.reshape(-1).long(), minlength=384)


# off path
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


# counting
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
    """Mutant: take the first rows instead of the rank's own -> rank 2 would count rank 0's tokens."""
    ids = _ids(32, 7)
    own, mutant = ids[16:22], ids[0:6]
    assert not torch.equal(_expected(own), _expected(mutant))


def test_allgather_shape_mismatch_is_skipped(counter):
    counter.rank, counter.dp, counter.gathered = 1, 4, (True, 8)
    counter.split = (1, 1)
    ec.record(_layer(0), _ids(30), False)
    assert int(ec._S["counts"]["decode"].sum()) == 0 and sum(ec._S["skipped"].values()) == 1


# graph mode: count only provably-eager forwards, with a denominator. Each
# check_* takes the module under test so that the same criterion runs on the
# real module and on source-level mutants of it (below); a check is red by
# raising AssertionError.


def _all_layers(mod, ctx, split, rows, seed, force=False):
    """One model forward: the same router call on every main layer."""
    ctx.split = split
    ids = {layer: _ids(rows, seed + layer) for layer in range(mod.N_LAYERS)}
    for layer in range(mod.N_LAYERS):
        mod.record(_layer(layer), ids[layer], force)
    return ids


def _dumped(mod):
    return torch.load(mod.dump())


def _ledger_closes(fw):
    return fw["seen"] == fw["counted"] + sum(fw["excluded"].values())


def check_captured_not_counted(mod, ctx):
    """A forward on a capturing stream adds nothing, and is accounted as captured."""
    ctx.capturing = True
    _all_layers(mod, ctx, (8, 8), 8, 10)
    out = _dumped(mod)
    fw = out["forwards"]
    assert int(out["counts"]["decode"].sum() + out["counts"]["prefill"].sum()) == 0, "counted under capture"
    assert int(out["tokens"]["decode"].sum() + out["tokens"]["prefill"].sum()) == 0
    assert fw["counted"] == 0 and fw["excluded"].get("captured") == 1, fw
    assert fw["captured_token_sizes"] == [8] and fw["replayed"] is None, fw
    assert _ledger_closes(fw), fw


def check_eager_counted(mod, ctx):
    """A >=512-token prefill with the predicate saying False is counted, exactly."""
    ctx.capturing = False
    ids = _all_layers(mod, ctx, (0, 512), 512, 20)
    out = _dumped(mod)
    for layer in (0, 17, 39):
        assert torch.equal(out["counts"]["prefill"][layer], _expected(ids[layer])), layer
    fw = out["forwards"]
    assert fw["seen"] == 1 and fw["counted"] == 1 and fw["excluded"] == {}, fw
    assert fw["replayed"] == 0 and out["steps"] == 1, fw  # no capture seen -> no graph -> no replay, provably


def check_unknown_not_counted(mod, ctx, make_unknown):
    """Predicate cannot be evaluated -> not counted (fail closed), accounted, replayed unknowable."""
    make_unknown()
    _all_layers(mod, ctx, (0, 512), 512, 30)
    out = _dumped(mod)
    fw = out["forwards"]
    assert int(out["counts"]["prefill"].sum()) == 0, "counted although the capture state was unknown"
    assert fw["counted"] == 0 and fw["excluded"].get("capture_state_unknown") == 1, fw
    assert fw["replayed"] is None and _ledger_closes(fw), fw


def check_ledger(mod, ctx):
    """Every layer-0 call has exactly one disposition, and each exclusion is under its own key."""
    ctx.capturing = False
    _all_layers(mod, ctx, (4, 4), 4, 40, force=True)  # profile run
    _all_layers(mod, ctx, (8, 8), 8, 50)  # capture warm-up (eager dummy, looks like decode)
    ctx.capturing = True
    _all_layers(mod, ctx, (8, 8), 8, 50)  # the capture
    ctx.capturing = False
    ctx.split = None  # an eager dummy without attention metadata
    mod.record(_layer(0), _ids(1), False)
    ids_p = _all_layers(mod, ctx, (0, 600), 600, 60)  # real prefill
    ids_d = _all_layers(mod, ctx, (3, 3), 3, 70)  # real (eager) decode
    ctx.capturing = RuntimeError("driver gone")
    _all_layers(mod, ctx, (0, 600), 600, 80)
    fw = _dumped(mod)["forwards"]
    ex = fw["excluded"]
    assert fw["seen"] == 7 and fw["counted"] == 2, fw
    assert ex.get("profile_run") == 1 and ex.get("captured") == 1, ex
    assert ex.get("discarded_before_capture") == 1 and ex.get("capture_state_unknown") == 1, ex
    assert ex.get("no attn metadata (dummy run?)") == 1, ex
    assert _ledger_closes(fw), fw
    assert fw["replayed"] is None and fw["captured_token_sizes"] == [8], fw
    return ids_p, ids_d


def check_warmup_discarded(mod, ctx):
    """Capture warm-ups (eager, FULL => with attn metadata) must not survive into the table."""
    ctx.capturing = False
    for seed in (100, 200):  # cudagraph_num_of_warmups = 2
        _all_layers(mod, ctx, (8, 8), 8, seed)
    ctx.capturing = True
    _all_layers(mod, ctx, (8, 8), 8, 100)
    ctx.capturing = False
    ids = _all_layers(mod, ctx, (0, 600), 600, 300)
    out = _dumped(mod)
    assert int(out["counts"]["decode"].sum()) == 0, "capture warm-up decode counted as traffic"
    for layer in (0, 39):
        assert torch.equal(out["counts"]["prefill"][layer], _expected(ids[layer])), layer
    assert int(out["tokens"]["prefill"][0]) == 600 and int(out["tokens"]["decode"][0]) == 0
    assert out["forwards"]["excluded"].get("discarded_before_capture") == 2, out["forwards"]
    # the report's sum invariant still holds after the discard
    assert torch.equal(out["counts"]["prefill"].sum(1), out["tokens"]["prefill"] * 6)
    assert torch.equal(out["counts"]["decode"].sum(1), out["tokens"]["decode"] * 6)


def check_capture_touches_no_device_state(mod, ctx):
    """The captured path may not write a device tensor: it would be recorded and
    re-run on every replay. CPU model of that: the counts tensors must be
    bit-identical before and after a captured call.
    """
    ctx.capturing = False
    _all_layers(mod, ctx, (8, 8), 8, 400)
    before = {b: t.clone() for b, t in mod._S["counts"].items()}
    ctx.capturing = True
    _all_layers(mod, ctx, (8, 8), 8, 400)
    for b, t in mod._S["counts"].items():
        assert torch.equal(t, before[b]), f"device counts[{b}] written under capture"
    ctx.capturing = False
    ids = _all_layers(mod, ctx, (0, 512), 512, 500)
    assert int(mod._S["counts"]["decode"].sum()) == 0  # stale counts dropped by the eager call
    assert torch.equal(mod._S["counts"]["prefill"][0], _expected(ids[0]))


def check_first_call_captured_allocates_nothing(mod, ctx):
    """If the very first call is under capture, no device state may be allocated (graph pool)."""
    ctx.capturing = True
    mod.record(_layer(0), _ids(8), False)
    assert mod._S["counts"] is None, "allocated counts under capture"
    ctx.capturing = False
    ctx.split = (0, 512)
    ids = _ids(512, 9)
    mod.record(_layer(0), ids, False)
    assert torch.equal(mod._S["counts"]["prefill"][0], _expected(ids))


def _no_npu(monkeypatch, ctx):
    return lambda: monkeypatch.delattr(torch, "npu", raising=False)


def _unknown_envs(monkeypatch, ctx):
    """Every way the predicate can fail to give an answer."""

    def set_(v):
        return lambda: setattr(ctx, "capturing", v)

    return {
        "no_torch_npu": _no_npu(monkeypatch, ctx),
        "no_predicate": lambda: monkeypatch.setattr(torch, "npu", SimpleNamespace(), raising=False),
        "raises": set_(RuntimeError("ACL not initialised")),
        "returns_mock": set_(mock.MagicMock()),
        "returns_int": set_(1),
        "returns_none": set_(None),
    }


UNKNOWN_KINDS = ("no_torch_npu", "no_predicate", "raises", "returns_mock", "returns_int", "returns_none")

CHECKS = {
    "captured_not_counted": check_captured_not_counted,
    "eager_counted": check_eager_counted,
    "ledger": check_ledger,
    "warmup_discarded": check_warmup_discarded,
    "capture_touches_no_device_state": check_capture_touches_no_device_state,
    "first_call_captured_allocates_nothing": check_first_call_captured_allocates_nothing,
}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_graph_mode_checks_green_on_real_module(counter, name):
    CHECKS[name](ec, counter)


@pytest.mark.parametrize("kind", UNKNOWN_KINDS)
def test_cannot_evaluate_fails_closed(counter, monkeypatch, kind):
    check_unknown_not_counted(ec, counter, _unknown_envs(monkeypatch, counter)[kind])


def test_predicate_is_the_production_one(env, monkeypatch):
    """True/False only from a real bool; the capture half matches layer.py's ``_capturing`` (``is True``)."""
    from vllm_ascend.kt_offload import layer as kt_layer  # not the model package: no platform patches

    for answer, want in ((True, True), (False, False), (1, None), (0, None), (None, None), (mock.MagicMock(), None)):
        env.capturing = answer
        assert ec._capturing() is want, answer
        # "captured" agrees with the production path's own predicate, evaluated on the same fake stream
        assert (want is True) == (kt_layer._capturing() is True), answer
    env.capturing = RuntimeError("x")
    assert ec._capturing() is None
    env.no_npu()
    assert ec._capturing() is None


def test_plain_cpu_process_counts_nothing(counter, monkeypatch, caplog):
    """Without an NPU the real predicate cannot say "eager"."""
    counter.no_npu()
    counter.split = (0, 512)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            ec.record(_layer(0), _ids(512), False)
    fw = ec._S["forwards"]
    assert fw["counted"] == 0 and fw["excluded"] == {"capture_state_unknown": 3}
    warned = [r for r in caplog.records if "cannot tell whether the current stream is being captured" in r.getMessage()]
    assert len(warned) <= 1  # logged once, not per call (0 if the vllm logger does not propagate to caplog)


def test_check_config_records_instead_of_refusing(counter):
    ec.check_config(SimpleNamespace(model_config=SimpleNamespace(enforce_eager=False)))
    assert ec._CONFIG_ENFORCE_EAGER is False
    counter.split = (1, 1)
    ec.record(_layer(0), _ids(1), False)
    out = _dumped(ec)
    assert out["config_enforce_eager"] is False and out["forwards"]["counted"] == 1


def test_non_main_layer_calls_are_not_forwards(counter):
    counter.capturing = True
    ec.record(SimpleNamespace(layer_name="dspark.stages.0.ffn.experts"), _ids(8), False)
    ec.record(_layer(3), _ids(8), False)  # captured, but not layer 0: per-call only
    fw = ec._S["forwards"]
    assert fw["seen"] == 0 and fw["excluded"] == {}
    assert ec._S["skipped"].get("captured") == 1


# mutation arms: each mutant is the real source with one edit, and must turn a check red

COUNTER_SRC = pathlib.Path(ec.__file__)


def _mutant(tag, *subs):
    src = COUNTER_SRC.read_text()
    for old, new in subs:
        assert src.count(old) == 1, f"mutant {tag}: anchor not found exactly once: {old!r}"
        src = src.replace(old, new)
    mod = types.ModuleType(f"vllm_ascend.kt_offload.expert_counter__mutant_{tag}")
    mod.__file__ = str(COUNTER_SRC)
    exec(compile(src, f"{COUNTER_SRC} [mutant {tag}]", "exec"), mod.__dict__)
    return mod


MUTANTS = {
    # counts a captured forward: the capture verdict is ignored
    "count_during_capture": (
        [("    cap = _capturing()\n", "    cap = _capturing()\n    cap = False if cap is True else cap\n")],
        ["captured_not_counted"],
    ),
    # the runtime condition as a gate, both constants
    "predicate_const_false": (
        [("def _capturing() -> bool | None:\n", "def _capturing() -> bool | None:\n    return False\n")],
        ["captured_not_counted", "unknown:no_torch_npu", "unknown:raises"],
    ),
    "predicate_const_true": (
        [("def _capturing() -> bool | None:\n", "def _capturing() -> bool | None:\n    return True\n")],
        ["eager_counted", "unknown:no_torch_npu"],
    ),
    # layer.py's predicate verbatim: fine for "am I capturing", fails OPEN for "am I provably eager"
    "fail_open_like_layer_py": (
        [
            (
                "        return None\n    try:\n        result = predicate()",
                "        return False\n    try:\n        result = predicate()",
            ),
            ("    if result is False:\n        return False\n", "    if result is not True:\n        return False\n"),
        ],
        ["unknown:no_torch_npu", "unknown:returns_mock", "unknown:returns_none"],
    ),
    # the denominator: drop the excluded-count bookkeeping
    "drop_excluded_bookkeeping": (
        [("        ex[why] = ex.get(why, 0) + 1\n", "        pass\n")],
        ["ledger", "captured_not_counted"],
    ),
    # capture warm-ups kept in the table
    "keep_capture_warmups": (
        [("        dropped = _discard_before_capture(s)\n", "        dropped = 0\n")],
        ["warmup_discarded", "ledger"],
    ),
    # zero the device counts inside the captured call (would be recorded into the graph)
    "zero_under_capture": (
        [
            (
                '    s["counts_stale"] = True\n',
                '    s["counts_stale"] = True\n    for t in (s["counts"] or {}).values():\n        t.zero_()\n',
            )
        ],
        ["capture_touches_no_device_state"],
    ),
    # allocate the state's device tensors regardless of the capture verdict
    "allocate_under_capture": (
        [
            (
                "    s = _state(topk_ids.device, allocate=cap is False)\n",
                "    s = _state(topk_ids.device, allocate=True)\n",
            )
        ],
        ["first_call_captured_allocates_nothing"],
    ),
}


def _run_check(check, mod, ctx, monkeypatch):
    if check.startswith("unknown:"):
        check_unknown_not_counted(mod, ctx, _unknown_envs(monkeypatch, ctx)[check.split(":", 1)[1]])
    else:
        CHECKS[check](mod, ctx)


@pytest.mark.parametrize(
    "tag,check", [(tag, check) for tag, (_, checks) in MUTANTS.items() for check in checks], ids=lambda x: x
)
def test_mutation_arm_is_red(monkeypatch, tmp_path, tag, check):
    subs, _ = MUTANTS[tag]
    ctx = _ctx()
    _install_npu(monkeypatch, ctx)
    mod = _arm(_mutant(tag, *subs), monkeypatch, tmp_path, ctx)
    with pytest.raises(AssertionError):
        _run_check(check, mod, ctx, monkeypatch)


@pytest.mark.parametrize("check", sorted({c for _, checks in MUTANTS.values() for c in checks}), ids=lambda x: x)
def test_same_checks_green_on_unmutated_source(monkeypatch, tmp_path, check):
    """Positive control: the same harness on the unedited source is green, so a red
    arm is the mutation and not the exec-loading."""
    ctx = _ctx()
    _install_npu(monkeypatch, ctx)
    mod = _arm(_mutant("none"), monkeypatch, tmp_path, ctx)
    _run_check(check, mod, ctx, monkeypatch)


# files per rank + report invariant












