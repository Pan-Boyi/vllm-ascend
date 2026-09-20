# SPDX-License-Identifier: Apache-2.0
"""The single-card KV byte ledger -- CPU only, no device, no service.

``tools/kt_offload/kv_budget_singlecard.py`` recomputes the KV cache geometry
from config alone and reconciles it against what our own two bring-up flights
printed. The criterion is the reconciliation, so the tests here are (a) it
passes on both real logs and (b) mutation arms on the model -- one per
mechanism claim -- each of which must make the reconciliation fail. Without
(b) a reconciliation that is insensitive to the mechanism reads exactly like
one that confirms it.

The logs live under build_artifacts/ (shared, read-only here).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
PROJECT = ROOT.parents[1]           # .../vllm_project
SCRIPT = ROOT / "tools/kt_offload/kv_budget_singlecard.py"
LOGS = [PROJECT / p for p in (
    "build_artifacts/moe_offload_ref/bringup_0920_034907/serve.log",
    "build_artifacts/moe_offload_ref/bringup_graph_0920_040749/serve.log",
)]


@pytest.fixture
def kvb():
    spec = importlib.util.spec_from_file_location("kv_budget_singlecard",
                                                  SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules[cls.__module__];
    # a module that is not registered there fails with an opaque
    # AttributeError inside dataclasses.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    yield mod
    sys.modules.pop(spec.name, None)


@pytest.fixture
def readings(kvb):
    missing = [p for p in LOGS if not p.exists()]
    if missing:
        pytest.skip(f"serve.log 不在：{missing}")
    return [kvb.parse_log(str(p)) for p in LOGS]


# -- the reconciliation itself ----------------------------------------------


def test_both_of_our_flights_reconcile(kvb, readings, capsys):
    for r in readings:
        layout, avail = kvb.check(r)
        assert layout.swa_groups == 40
        assert layout.swa_blocks_per_req == 257
        assert layout.bytes_per_block == 542208
        assert layout.blocks_per_req == 10377
        assert layout.num_blocks(avail) == 54131
        assert layout.allocated(avail) == 29350261248


def test_the_per_token_charge_is_the_measured_one(kvb, readings):
    layout, _ = kvb.check(readings[0])
    # 10377 blocks/request * 542208 B / 8192 tokens
    assert layout.bytes_per_token() == pytest.approx(686827.5, abs=1.0)
    assert layout.bytes_per_token() / 1024 == pytest.approx(670.7, abs=0.1)


def test_the_charge_is_not_constant_per_token(kvb, readings):
    """The single-card flight is charged ~1.95x the four-card one, and it is
    max_model_len that does it, not the card count: the SWA reservation is
    dominated by in-flight tokens, which do not grow with the context.

    If this were a single/multi-card property the two would not land on the
    same curve.
    """
    r = readings[0]
    inflight = 2 * r.args["max_num_batched_tokens"]
    at_8192 = kvb.build_layout(8192, inflight).bytes_per_token()
    at_16384 = kvb.build_layout(16384, inflight).bytes_per_token()
    assert at_8192 / 1024 == pytest.approx(670.7, abs=0.1)
    # the four-card flights print 24.54 GiB / 74,872 tokens = 343.8 KiB
    assert at_16384 / 1024 == pytest.approx(343.7, abs=0.3)
    assert at_8192 / at_16384 == pytest.approx(1.95, abs=0.01)


def test_the_sliding_window_is_essentially_all_of_it(kvb, readings):
    layout, _ = kvb.check(readings[0])
    swa = layout.swa_groups * layout.swa_blocks_per_req
    assert swa / layout.blocks_per_req > 0.99


def test_the_ring_region_is_18_percent_and_almost_never_touched(kvb, readings):
    layout, avail = kvb.check(readings[0])
    nb = layout.num_blocks(avail)
    share = layout.ring_bytes_per_block / layout.bytes_per_block
    assert share == pytest.approx(0.1813, abs=0.0005)
    assert layout.ring_bytes_per_block * nb / (1 << 30) == pytest.approx(
        4.956, abs=0.01)
    # one block per request, max_num_seqs=8 on this flight
    assert readings[0].args["max_num_seqs"] * 100 / nb < 0.05


# -- mutation arms: each breaks one mechanism claim --------------------------

MUTATIONS = [
    ("每个 SWA 组 5 层而不是 1 层", dict(N_LAYERS=40), dict(swa_layers=5)),
    ("ring 层数 3→4", dict(N_RING=4), {}),
    ("indexer K 页少算 scale（fp32 4B）", dict(P_IDX=128 * 128), {}),
    ("主 MLA 页按 fp8 而不是 bf16", dict(P_MLA=128 * 512), {}),
    ("滑窗预留只算窗口，不算 in-flight", dict(SW=128, _inflight_zero=True), {}),
    ("层数 40→43（V4.0 的数）", dict(N_LAYERS=43), {}),
]


@pytest.mark.parametrize("why,consts,extra", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_mutation_arms_break_the_reconciliation(kvb, readings, why, consts,
                                                extra, monkeypatch):
    """Every one of these must make check() raise.

    A reconciliation that survives them would be measuring the log format, not
    the mechanism.
    """
    for name, value in consts.items():
        if name == "_inflight_zero":
            monkeypatch.setattr(
                kvb, "swa_admission_blocks",
                lambda L, inflight, bs: kvb.cdiv(min(kvb.SW - 1, L), bs) + 1)
            continue
        monkeypatch.setattr(kvb, name, value)
    if extra.get("swa_layers"):
        real = kvb.build_layout
        monkeypatch.setattr(
            kvb, "build_layout",
            lambda L, inflight, **kw: real(
                L, inflight, **{**kw, "swa_layers_per_group": extra["swa_layers"]}))
    with pytest.raises(AssertionError):
        kvb.check(readings[0])


def test_the_in_flight_fit_has_discriminating_power(kvb, readings):
    """The script pins max_in_flight_tokens by asking which of the two
    possible values reproduces the log. That is only worth anything if the
    other one does not: at mnbt (async off) it lands at 14.27 GiB, half the
    printed 27.34.
    """
    r = readings[0]
    mnbt = r.args["max_num_batched_tokens"]
    lay = kvb.build_layout(r.max_model_len, mnbt)
    nb, _, _ = kvb.solve_avail_bytes(r, lay)
    assert nb * lay.bytes_per_block / (1 << 30) == pytest.approx(14.27,
                                                                 abs=0.01)


# -- the levers --------------------------------------------------------------


def test_L3_quadruples_capacity_and_frees_20_GiB(kvb, readings):
    r = readings[0]
    base, avail = kvb.check(r)
    l3 = kvb.build_layout(r.max_model_len, base.inflight, swa_bs=128)
    assert l3.swa_blocks_per_req == 65          # 257 -> 65
    assert l3.bytes_per_block == base.bytes_per_block   # 池块字节不变
    assert l3.tokens(avail) / base.tokens(avail) == pytest.approx(3.85,
                                                                  abs=0.02)
    # 保住基线 token 数需要多少 KV
    need = kvb.cdiv(base.tokens(avail) * l3.blocks_per_req,
                    r.max_model_len) * l3.bytes_per_block
    assert (avail - need) / (1 << 30) == pytest.approx(20.2, abs=0.2)


def test_L3_removes_the_page_padding(kvb):
    assert kvb.swa_page_padded(32) / kvb.swa_page(32) == 4.0
    assert kvb.swa_page_padded(64) / kvb.swa_page(64) == 2.0
    assert kvb.swa_page_padded(128) / kvb.swa_page(128) == 1.0


def test_L4_alone_is_the_small_one(kvb, readings):
    r = readings[0]
    base, avail = kvb.check(r)
    l4 = kvb.build_layout(r.max_model_len, base.inflight,
                          ring_capped_seqs=r.args["max_num_seqs"])
    assert l4.tokens(avail) / base.tokens(avail) == pytest.approx(1.22,
                                                                  abs=0.02)


def test_the_cpu_curve_is_the_measured_one(kvb):
    """§4.3: T(k)=66.8+43.95k us/layer x 40 layers, k=6(1-N/384)."""
    assert kvb.cpu_ms_per_token(100) == pytest.approx(10.47, abs=0.01)
    assert kvb.cpu_ms_per_token(128) == pytest.approx(9.70, abs=0.01)
    # 20 GiB of KV buys 28 experts and 0.77 ms -- the capacity axis is the
    # one that moves, not TPOT.
    assert (kvb.cpu_ms_per_token(100) - kvb.cpu_ms_per_token(128)) < 1.0


# -- the theory ledger -------------------------------------------------------


def test_the_card_ledger_recomputes_to_890(kvb, capsys):
    ours_bt, ours_swa = kvb.theory_table()
    # bf16 is 3.25x the card's FP4 number, not 4x: the main cache goes
    # 0.5625 -> 2 B/element (3.556x) but index_k is fp8+fp32 scale, only 1.94x.
    assert ours_bt == pytest.approx(2890, abs=1)
    assert ours_bt / 890 == pytest.approx(3.25, abs=0.01)
    assert ours_swa == 40 * 128 * 512 * 2
