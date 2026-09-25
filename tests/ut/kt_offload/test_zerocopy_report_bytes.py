"""``_report_zerocopy_effect`` must expect BYTES, and must include the scales.

Every expected number below is a literal computed from the checkpoint's own
quantisation parameters and kt's allocation sites, never by calling the function
under test.
"""

from __future__ import annotations

import pytest

from vllm_ascend.kt_offload.layer import _kt_staging_bytes, _report_zerocopy_effect

# From config.json: hidden_size, moe_intermediate_size, and
# quantization_config.config_groups.group_1 (MoEGMM) group_size.
H = 5120
I = 2304
G = 32

# kt's per-expert allocations (operators/amx/fp4-moe.hpp, do_numa_job): each of
# gate/up/down_proj is `inter_tp * hidden / 2` bytes (uint8, two FP4 nibbles per
# byte); each scale is `(hidden / group_size) * inter_tp * 2` bytes (ggml_bf16_t,
# not the 1-byte E8M0 the NPU side stores). do_numa_job runs once per TP part
# with that part's inter_tp, so the parts sum to the full `inter`.
ONE_PROJ = I * H // 2  # 5_898_240
ONE_SCALE = (H // G) * I * 2  # 737_280

EXPECT_W13 = 2 * ONE_PROJ  # gate + up: what KT_ZEROCOPY_SCOPE=w13 skips
EXPECT_OTHER = ONE_PROJ + 3 * ONE_SCALE  # down + all three scale stagings


def test_literals_are_what_the_formula_says():
    """Guard the guard: if these drift, every other case here is meaningless."""
    assert ONE_PROJ == 5_898_240
    assert ONE_SCALE == 737_280
    assert EXPECT_W13 == 11_796_480
    assert EXPECT_OTHER == 8_110_080


def test_staging_bytes_are_bytes_and_include_the_scales():
    w13, other = _kt_staging_bytes(H, I)
    assert (w13, other) == (EXPECT_W13, EXPECT_OTHER)


def test_not_the_old_element_count():
    """An element-count `other` drops the scales, 27.3% of what zero-copy leaves."""
    _, other = _kt_staging_bytes(H, I)
    old_other_at_tp2 = H * I // 2  # 5_898_240 -- no scales
    assert other != old_other_at_tp2
    assert other - old_other_at_tp2 == 3 * ONE_SCALE
    # The scales are not a rounding detail.
    assert (other - old_other_at_tp2) / other > 0.25


def test_no_tp_division():
    """``do_numa_job`` allocates per part, so the process-wide cost is the full
    ``inter``; dividing by the part count reports one part's share as the whole."""
    w13_full, other_full = _kt_staging_bytes(H, I)
    # Halving `inter` must halve the answer; that is the only way the result can
    # be linear in the real (undivided) intermediate size.
    w13_half, other_half = _kt_staging_bytes(H, I // 2)
    assert (w13_half, other_half) == (w13_full // 2, other_full // 2)


def test_group_size_moves_only_the_scales():
    w13_32, other_32 = _kt_staging_bytes(H, I, group_size=32)
    w13_16, other_16 = _kt_staging_bytes(H, I, group_size=16)
    assert w13_16 == w13_32  # weights do not depend on the group size
    # Halving the group doubles the number of groups, hence the scale bytes.
    assert other_16 - other_32 == 3 * ONE_SCALE


def _capture(monkeypatch):
    """Capture the module's own ``logger``: it does not propagate to the root
    handler pytest installs, so caplog reads empty and every "it warned"
    assertion would pass on a silent build."""
    from vllm_ascend.kt_offload import layer as mod
    warns: list[str] = []
    infos: list[str] = []
    monkeypatch.setattr(mod.logger, "warning", lambda m, *a: warns.append(m % a if a else m))
    monkeypatch.setattr(mod.logger, "info", lambda m, *a: infos.append(m % a if a else m))
    return warns, infos


def test_the_capture_helper_can_actually_see_a_warning(monkeypatch):
    """Positive control for _capture: a helper that sees nothing would make every
    assertion below vacuous."""
    from vllm_ascend.kt_offload import layer as mod
    warns, _ = _capture(monkeypatch)
    mod.logger.warning("canary %d", 1)
    assert warns == ["canary 1"]


@pytest.mark.parametrize("n_cpu", [1, 284, 384])
def test_warns_when_anon_growth_matches_the_copy_path(monkeypatch, n_cpu):
    """The failure this proxy exists to catch: zerocopy asked for, copy path
    delivered."""
    warns, _ = _capture(monkeypatch)
    w13, other = n_cpu * EXPECT_W13, n_cpu * EXPECT_OTHER
    _report_zerocopy_effect(7, True, w13 + other, w13, other)
    assert warns, "a copy-path-sized growth under zerocopy_w13=True must warn"
    assert any("looks like it was copied anyway" in m for m in warns), warns


def test_silent_when_anon_growth_matches_zero_copy(monkeypatch):
    warns, infos = _capture(monkeypatch)
    n_cpu = 284
    w13, other = n_cpu * EXPECT_W13, n_cpu * EXPECT_OTHER
    _report_zerocopy_effect(7, True, other, w13, other)
    assert warns == [], warns
    assert infos, "the INFO line is the record of what was measured; it must still be emitted"


def test_the_old_expectation_was_too_small_not_too_large():
    """Pin the direction: an element-count expectation is ~11% SMALL, so the
    ``0.8 * expect_copy`` threshold sat too low and the warning was quick to
    fire. Correcting it can only make the warning fire less often, never more."""
    tp = 2
    old_expect_copy = 2 * H * I // tp + H * I // tp
    new_expect_copy = EXPECT_W13 + EXPECT_OTHER
    assert old_expect_copy == 17_694_720
    assert new_expect_copy == 19_906_560
    assert old_expect_copy < new_expect_copy
    assert 0.8 * old_expect_copy < 0.8 * new_expect_copy


def test_bufferb_is_not_in_either_figure():
    """Both values are FLOORS: kt also allocates AMX BufferB per CPU expert
    (``moe_base.hpp``), which neither number includes."""
    w13, other = _kt_staging_bytes(H, I)
    # Exactly the three staging terms zero-copy leaves behind, and nothing else.
    assert other == ONE_PROJ + 3 * ONE_SCALE
    assert w13 == 2 * ONE_PROJ


def test_no_report_without_a_measurement(monkeypatch):
    """``d_anon is None`` means /proc could not be read, not 'zero growth':
    reporting 0 would claim zero-copy saved everything on an unmeasured run."""
    warns, infos = _capture(monkeypatch)
    _report_zerocopy_effect(7, True, None, EXPECT_W13, EXPECT_OTHER)
    assert (warns, infos) == ([], [])
