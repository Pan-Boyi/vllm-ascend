# SPDX-License-Identifier: Apache-2.0
"""Off has to be provably off, and on has to record the right rows.

Thirty taps on the forward path become a permanent extra variable in every
later A/B unless "off" is structural. Hooks make it so: with no dump directory
set, nothing is registered, so no tap function is entered -- which is a
stronger statement than "the tap returns early", and it is the one these tests
pin.
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1 import align_taps


class _Inner(torch.nn.Module):
    def __init__(self, record):
        super().__init__()
        self.record = record

    def forward(self, x):
        self.record.append("inner-ran")
        return x * 2


class _Layer(torch.nn.Module):
    def __init__(self, record):
        super().__init__()
        self.attn = _Inner(record)
        self.ffn = _Inner(record)

    def forward(self, x, positions, input_ids=None):
        return self.ffn(self.attn(x))


class _Model(torch.nn.Module):
    def __init__(self, record, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_Layer(record) for _ in range(num_layers)])
        self.start_layer = 0

    def forward(self, x, positions):
        for layer in self.layers:
            x = layer(x, positions)
        return x


@pytest.fixture
def no_recorder(monkeypatch):
    monkeypatch.setattr(align_taps, "_CURRENT_POSITIONS", None)
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align
    monkeypatch.setattr(debug_align, "get_recorder", lambda: None)


# ------------------------------------------------------------------- off

def test_nothing_is_registered_when_there_is_no_recorder(no_recorder):
    """The whole argument for hooks. Not "the tap returns early" -- there is
    no tap to enter."""
    model = _Model([])
    assert align_taps.install_alignment_taps(model) == []


def test_off_leaves_the_modules_hook_dicts_empty(no_recorder):
    """Structural, and checkable without running anything."""
    model = _Model([])
    align_taps.install_alignment_taps(model)
    for layer in model.layers:
        for module in (layer, layer.attn, layer.ffn):
            assert not module._forward_pre_hooks
            assert not module._forward_hooks


def test_off_produces_the_same_output_as_an_untapped_model(no_recorder):
    """Bit-identical, not close. With no hooks this is structural; the test
    exists so that a future tap added to the forward path breaks it."""
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    plain = _Model([])(x.clone(), torch.arange(4))
    model = _Model([])
    align_taps.install_alignment_taps(model)
    tapped = model(x.clone(), torch.arange(4))
    assert torch.equal(plain, tapped)


def test_a_tap_function_is_never_entered_when_off(no_recorder, monkeypatch):
    """Counted, not reasoned about."""
    entered = []
    monkeypatch.setattr(align_taps, "_record",
                        lambda *a, **k: entered.append(a))
    model = _Model([])
    align_taps.install_alignment_taps(model)
    model(torch.randn(2, 4), torch.arange(2))
    assert entered == []


# -------------------------------------------------------------------- on

@pytest.fixture
def with_recorder(monkeypatch):
    calls = []

    class _Recorder:
        def record_layer(self, layer_id, key, tensor, row_to_pos):
            calls.append((layer_id, key, tuple(tensor.shape),
                          row_to_pos.tolist()))

    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align
    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Recorder())
    monkeypatch.setattr(debug_align, "current_row_to_pos",
                        lambda positions: positions.to(torch.int64))
    monkeypatch.setattr(align_taps, "_CURRENT_POSITIONS", None)
    return calls


def test_on_records_both_edges_of_both_modules(with_recorder):
    model = _Model([])
    handles = align_taps.install_alignment_taps(model)
    assert handles, "nothing was registered with a recorder present"
    model(torch.randn(3, 4), torch.arange(3))

    keys = {(layer_id, key) for layer_id, key, _, _ in with_recorder}
    assert keys == {(0, "attn.in"), (0, "attn.out"), (0, "moe.in"),
                    (0, "moe.out"), (1, "attn.in"), (1, "attn.out"),
                    (1, "moe.in"), (1, "moe.out")}


def test_the_layer_index_is_global_not_local(with_recorder):
    """start_layer offsets it, because golden's tags are global and a
    pipeline rank holding layers 20..39 would otherwise write layer_00."""
    model = _Model([])
    model.start_layer = 20
    align_taps.install_alignment_taps(model)
    model(torch.randn(2, 4), torch.arange(2))
    assert {layer_id for layer_id, _, _, _ in with_recorder} == {20, 21}


def test_positions_come_from_the_layer_being_tapped(with_recorder):
    """The value the hooks use is the argument of the layer they sit in.

    The attention metadata does not carry positions; a tap that looked there
    would find None every time and record nothing, which is indistinguishable
    from a model that never ran.
    """
    model = _Model([])
    align_taps.install_alignment_taps(model)
    model(torch.randn(3, 4), torch.tensor([5, 6, 7]))
    for _, _, _, row_to_pos in with_recorder:
        assert row_to_pos == [5, 6, 7]


def test_a_forward_with_no_positions_records_nothing_rather_than_guessing(
        with_recorder, monkeypatch):
    model = _Model([])
    align_taps.install_alignment_taps(model)
    monkeypatch.setattr(align_taps, "_current_positions", lambda: None)
    model(torch.randn(2, 4), torch.arange(2))
    assert with_recorder == []


def test_removing_the_handles_restores_the_untapped_model(with_recorder):
    model = _Model([])
    for handle in align_taps.install_alignment_taps(model):
        handle.remove()
    model(torch.randn(2, 4), torch.arange(2))
    assert with_recorder == []


# ------------------------------------------------------- the expected key set

def test_the_expected_set_is_read_from_goldens_own_dump(tmp_path):
    """Not a list written here: a hand-written one cannot show the row it is
    missing, and this is the same set the comparison will use."""
    from safetensors.torch import save_file

    save_file({"attn.out": torch.zeros(1, 2, 2)},
              str(tmp_path / "layer_00.safetensors"))
    save_file({"embed.out": torch.zeros(1, 2, 2)},
              str(tmp_path / "prologue.safetensors"))

    got = align_taps.expected_keys(str(tmp_path), (0,))
    assert got == {"layer_00/attn.out", "prologue/embed.out"}


def test_an_empty_reference_dump_is_refused_not_treated_as_agreement(tmp_path):
    """Zero expected keys would make every coverage comparison vacuous: the
    difference in both directions is empty, which reads as full agreement."""
    with pytest.raises(ValueError, match="vacuous"):
        align_taps.expected_keys(str(tmp_path), (0,))


def test_the_norm_modules_are_not_tapped_as_boundaries():
    """attn_norm / ffn_norm are real submodules whose norm is computed inside
    mhc_pre_delayed_tilelang -- it only borrows their weights. A hook on
    either would never fire, and "no rows" reads as "the layer did not run"."""
    tapped = {attribute for attribute, _, _ in align_taps._BOUNDARY_TAPS}
    assert "attn_norm" not in tapped and "ffn_norm" not in tapped
