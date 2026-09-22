# SPDX-License-Identifier: Apache-2.0
"""The single-card KV byte ledger -- CPU only, no device, no service.

``tools/kt_offload/kv_budget_singlecard.py`` recomputes the KV cache geometry
and, when handed a serve.log, reconciles it against what that flight printed.

What changed on 2026-09-22 (rebase onto f85267ab8, index-K row fp8 -> MXFP4):
the script used to hard-code the page sizes (``P_IDX = 16896``) and to read
two default logs from our 2026-09-20 flights. Those logs describe the OLD
build, so they kept reconciling the old constant green after the code under
it had moved to an 8,704 B page -- the ledger was checking itself against a
record of a build that no longer exists.

So the criteria here are now:

(a) the script's layout equals the real patched planner, run on the specs the
    real cache objects report (no log involved; it checks this tree's code);
(b) mutation arms -- one per mechanism claim, including putting 16896 back --
    each of which must make (a) fail. Without (b) a check that is insensitive
    to the mechanism reads exactly like one that confirms it;
(c) a log is only reconciled if it carries a build fingerprint that matches
    the current build; our two old logs carry none and must be refused.

The two old logs live under build_artifacts/ (shared, read-only here). They
are used only by the test that shows why they must be refused.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
PROJECT = ROOT.parents[1]           # .../vllm_project
SCRIPT = ROOT / "tools/kt_offload/kv_budget_singlecard.py"
OLD_LOGS = [PROJECT / p for p in (
    "build_artifacts/moe_offload_ref/bringup_0920_034907/serve.log",
    "build_artifacts/moe_offload_ref/bringup_graph_0920_040749/serve.log",
)]
# An input to the planner, not an expectation: the backing size our
# 2026-09-20 flight allocated. What it maps to today is the output.
AVAIL = 29350261248


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
def old_readings(kvb):
    missing = [p for p in OLD_LOGS if not p.exists()]
    if missing:
        pytest.skip(f"serve.log 不在：{missing}")
    return [kvb.parse_log(str(p)) for p in OLD_LOGS]


# -- (a) the ledger against the real planner ---------------------------------


@pytest.mark.parametrize("max_model_len", [8192, 16384])
@pytest.mark.parametrize("swa_bs", [32, 128])
def test_the_ledger_matches_the_real_planner(kvb, max_model_len, swa_bs):
    lay = kvb.check_against_planner(max_model_len, 8192, swa_bs, AVAIL)
    assert lay.swa_groups == 40


def test_the_pages_come_from_the_real_objects(kvb):
    """The page sizes the script uses, and where they come from today
    (f85267ab8 + ours, 2026-09-22). Read off the objects, then checked against
    the arithmetic that explains them -- not the other way round.

    * index K: 128 x (64 packed E2M1 + 4 E8M0) = 8,704 (was 16,896, fp8 +
      one fp32 scale, before the MXFP4 index-K cache);
    * main MLA: 128 x 512 x bf16 = 131,072;
    * ring: 8 x 1024 x fp32 = 32,768, three ratio-2 sources.
    """
    specs = kvb.real_specs(32)
    assert kvb.P_IDX == specs["L20.idx"].page_size_bytes == 128 * (64 + 4)
    assert kvb.P_MLA == specs["L20.attn"].page_size_bytes == 128 * 512 * 2
    assert kvb.P_RING == specs["L2.ring"].page_size_bytes == 8 * 1024 * 4
    assert kvb.N_RING == 3
    assert kvb.swa_page(32) == specs["L0.swa"].page_size_bytes == 32 * 512 * 2


def test_the_pool_block_on_this_build(kvb):
    """Expected values computed from the real objects (see the test above):
    pool block 3 x (8,704 + 131,072) + 3 x 32,768 = 517,632 B (542,208 on the
    old build); 10,377 blocks/request at 8192 is unchanged -- it counts
    blocks, not bytes; 29,350,261,248 // 517,632 = 56,701 blocks.
    """
    lay = kvb.check_against_planner(8192, 8192, 32, AVAIL)
    assert lay.bytes_per_block == 517632
    assert lay.swa_blocks_per_req == 257
    assert lay.blocks_per_req == 10377
    assert lay.num_blocks(AVAIL) == 56701
    assert lay.allocated(AVAIL) == 56701 * 517632


def test_the_per_token_charge(kvb):
    """10,377 blocks/request x 517,632 B / 8192 = 640.3 KiB per token
    (670.7 on the old build, which is the number §4.5 quotes)."""
    lay = kvb.check_against_planner(8192, 8192, 32, AVAIL)
    assert lay.bytes_per_token() == pytest.approx(10377 * 517632 / 8192)
    assert lay.bytes_per_token() / 1024 == pytest.approx(640.3, abs=0.1)


def test_the_charge_is_not_constant_per_token(kvb):
    """max_model_len, not the card count, drives the per-token charge: the
    SWA reservation is dominated by in-flight tokens, which do not grow with
    the context. 8192 -> 16384 roughly halves it (1.95x, unchanged by the
    page change because both terms scale with the pool block)."""
    at_8192 = kvb.check_against_planner(8192, 8192, 32, AVAIL)
    at_16384 = kvb.check_against_planner(16384, 8192, 32, AVAIL)
    ratio = at_8192.bytes_per_token() / at_16384.bytes_per_token()
    assert ratio == pytest.approx(1.95, abs=0.01)


def test_the_sliding_window_is_essentially_all_of_it(kvb):
    lay = kvb.check_against_planner(8192, 8192, 32, AVAIL)
    swa = lay.swa_groups * lay.swa_blocks_per_req
    assert swa / lay.blocks_per_req > 0.99


def test_the_ring_region_share(kvb):
    """3 x 32,768 / 517,632 = 19.0% of the pool (18.1% on the old build: the
    ring did not shrink, the tuple did)."""
    lay = kvb.check_against_planner(8192, 8192, 32, AVAIL)
    share = lay.ring_bytes_per_block / lay.bytes_per_block
    assert share == pytest.approx(98304 / 517632)
    assert share == pytest.approx(0.1899, abs=0.0005)


def test_the_in_flight_term_has_discriminating_power(kvb):
    """The planner check pins in-flight tokens: at mnbt (async off, 4096)
    the SWA reservation is roughly halved, so a ledger that ignored the
    in-flight term would disagree with the planner."""
    at_async = kvb.check_against_planner(8192, 8192, 32, AVAIL)
    at_sync = kvb.check_against_planner(8192, 4096, 32, AVAIL)
    assert at_sync.blocks_per_req < at_async.blocks_per_req
    assert (at_sync.blocks_per_req, at_async.blocks_per_req) == (5417, 10377)


# -- (b) mutation arms: each breaks one mechanism claim ----------------------

MUTATIONS = [
    ("index-K 页塞回旧构建的 16896（fp8+fp32 scale）", dict(P_IDX=16896), {}),
    ("每个 SWA 组 5 层而不是 1 层", {}, dict(swa_layers=5)),
    ("ring 层数 3→4", dict(N_RING=4), {}),
    ("主 MLA 页按 fp8 而不是 bf16", dict(P_MLA=128 * 512), {}),
    ("滑窗预留只算窗口，不算 in-flight", dict(_inflight_zero=True), {}),
    ("层数 40→43（V4.0 的数）", dict(N_LAYERS=43), {}),
]


@pytest.mark.parametrize("why,consts,extra", MUTATIONS,
                         ids=[m[0] for m in MUTATIONS])
def test_mutation_arms_break_the_planner_check(kvb, why, consts, extra,
                                               monkeypatch):
    """Every one of these must make check_against_planner() raise.

    A check that survives them would be comparing the script with itself.
    The first arm is the one this file was rewritten for: 16896 put back
    must go red on this tree.
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
        kvb.check_against_planner(8192, 8192, 32, AVAIL)


# -- (c) logs must prove their build -----------------------------------------


def _write_log_dir(tmp_path, fingerprint=None):
    d = tmp_path / "flight"
    d.mkdir()
    log = d / "serve.log"
    log.write_text("placeholder serve.log\n")
    if fingerprint is not None:
        (d / "build_fingerprint.json").write_text(json.dumps(fingerprint))
    return log


def _fp(kvb, **over):
    bf = kvb._fingerprint_module()
    rule = {"id": bf.COUNT_RULE_ID}
    fp = {"schema": bf.SCHEMA, "so_primary_md5": "a" * 32,
          "vendors_sha256": "b" * 64, "vendors_files": 759,
          "vendors_bytes": 1, "vendors_ok": True,
          "vendors": {"count_rule": rule}, "kt_wheel_sha256": "c" * 64,
          "trees": {}}
    fp.update(over)
    return fp


def test_a_log_without_a_fingerprint_is_refused(kvb, tmp_path):
    log = _write_log_dir(tmp_path)
    with pytest.raises(kvb.StaleLogError, match="build_fingerprint.json"):
        kvb.require_log_build(str(log), current=_fp(kvb))


def test_a_log_from_the_same_build_is_accepted(kvb, tmp_path):
    fp = _fp(kvb)
    log = _write_log_dir(tmp_path, fp)
    kvb.require_log_build(str(log), current=dict(fp))


def test_a_log_from_another_so_is_refused(kvb, tmp_path):
    """Positive control for the test above: one blocking field differs."""
    log = _write_log_dir(tmp_path, _fp(kvb))
    with pytest.raises(kvb.StaleLogError, match="so_primary_md5 differs"):
        kvb.require_log_build(str(log),
                              current=_fp(kvb, so_primary_md5="d" * 32))


def test_the_old_default_logs_are_refused_and_why(kvb, old_readings,
                                                  monkeypatch):
    """Our two 2026-09-20 logs carry no fingerprint, so the gate refuses
    them. The rest of this test is the reason the gate exists: under this
    tree's real pages they do not reconcile, and with 16896 put back they
    reconcile again -- i.e. those logs are exactly what kept the stale
    constant green.
    """
    for path in OLD_LOGS:
        with pytest.raises(kvb.StaleLogError):
            kvb.require_log_build(str(path), current=_fp(kvb))
    for r in old_readings:
        with pytest.raises(AssertionError):
            kvb.check(r)
    monkeypatch.setattr(kvb, "P_IDX", 16896)
    for r in old_readings:
        layout, avail = kvb.check(r)
        assert layout.bytes_per_block == 542208


def test_main_refuses_to_run_without_an_explicit_log(kvb, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(sys, "argv", ["kv_budget_singlecard.py"])
    assert kvb.main() == 2
    assert "没有默认日志" in capsys.readouterr().err


# -- the levers --------------------------------------------------------------


def test_L3_quadruples_capacity_and_frees_20_GiB(kvb):
    """3.85x tokens and ~20.2 GiB at the same available bytes. Both are
    ratios of block counts at one pool block, so the page change leaves them
    (3.8477 / 20.23 GiB on both builds)."""
    base = kvb.check_against_planner(8192, 8192, 32, AVAIL)
    l3 = kvb.check_against_planner(8192, 8192, 128, AVAIL)
    assert l3.swa_blocks_per_req == 65          # 257 -> 65
    assert l3.bytes_per_block == base.bytes_per_block   # 池块字节不变
    assert l3.tokens(AVAIL) / base.tokens(AVAIL) == pytest.approx(3.85,
                                                                  abs=0.02)
    need = kvb.cdiv(base.tokens(AVAIL) * l3.blocks_per_req,
                    8192) * l3.bytes_per_block
    assert (AVAIL - need) / (1 << 30) == pytest.approx(20.2, abs=0.2)


def test_L3_removes_the_page_padding(kvb):
    assert kvb.swa_page_padded(32) / kvb.swa_page(32) == 4.0
    assert kvb.swa_page_padded(64) / kvb.swa_page(64) == 2.0
    assert kvb.swa_page_padded(128) / kvb.swa_page(128) == 1.0


def test_L4_alone_is_the_small_one(kvb):
    """1.23x (1.22x on the old build: the ring is a larger share now)."""
    base = kvb.check_against_planner(8192, 8192, 32, AVAIL)
    l4 = kvb.build_layout(8192, 8192, ring_capped_seqs=8)
    assert l4.tokens(AVAIL) / base.tokens(AVAIL) == pytest.approx(1.23,
                                                                  abs=0.01)


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
    # main cache bf16: 1024 B/state x (3/2 + 1) = 2560 B/token; index K now
    # MXFP4 68 B/state x 2.5 = 170 B/token (was fp8+fp32 132 x 2.5 = 330)
    # => 2730 B/token, 3.07x the card's 890 (3.25x on the old build).
    assert ours_bt == pytest.approx(2730, abs=1)
    assert ours_bt / 890 == pytest.approx(3.07, abs=0.01)
    assert ours_swa == 40 * 128 * 512 * 2
