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

def _manifest(tmp_path, tensors):
    import json

    (tmp_path / "manifest.json").write_text(json.dumps({"tensors": tensors}))
    return str(tmp_path)


def test_the_expected_set_comes_from_the_producers_declaration(tmp_path):
    """Not a scan, in either direction.

    Scanning the directory picks up files that are not outputs; scanning by
    name pattern misses tags the pattern does not cover. Both were hit within
    a day. The manifest has neither blind spot because the side that produced
    the tensors is the side that says which ones they are.
    """
    got = align_taps.expected_keys(
        _manifest(tmp_path, {"layer_00": {"attn.out": {}},
                             "prologue": {"embed.out": {}}}), (0,))
    assert got == {"layer_00/attn.out", "prologue/embed.out"}


def test_a_tag_that_is_not_a_layer_is_kept_whatever_it_is_called(tmp_path):
    """The filter excludes other layers; it does not decide what a tag is.

    An allow-list of {prologue, epilogue, layer_NN} would drop a protocol tag
    added later, and a dropped tag looks exactly like a tag with nothing in
    it.
    """
    got = align_taps.expected_keys(
        _manifest(tmp_path, {"something_new": {"a": {}},
                             "layer_00": {"b": {}}}), (0,))
    assert got == {"something_new/a", "layer_00/b"}


def test_other_layers_are_excluded(tmp_path):
    got = align_taps.expected_keys(
        _manifest(tmp_path, {"layer_00": {"a": {}}, "layer_07": {"b": {}}}),
        (0,))
    assert got == {"layer_00/a"}


def test_a_dump_that_declares_nothing_is_refused(tmp_path):
    """Zero expected keys would make every coverage comparison vacuous: the
    difference in both directions is empty, which reads as full agreement."""
    with pytest.raises(ValueError, match="vacuous"):
        align_taps.expected_keys(_manifest(tmp_path, {}), (0,))


def test_a_missing_manifest_is_refused_rather_than_scanned(tmp_path):
    """Falling back to a scan would reintroduce the blind spot silently,
    because a scan always returns something."""
    with pytest.raises(ValueError, match="no declaration"):
        align_taps.expected_keys(str(tmp_path), (0,))


def test_the_real_golden_dump_declares_the_count_golden_states():
    """67 for layers 0 and 1 -- and no routing_plan, which is an input fixture
    that a directory scan did pick up."""
    import os

    dump = "/workspace/code/dsv4_single_card/vllm_project/golden_dumps/len1100"
    if not os.path.exists(os.path.join(dump, "manifest.json")):
        pytest.skip("golden dump not present on this host")
    got = align_taps.expected_keys(dump, (0, 1))
    assert len(got) == 67
    assert not any(k.startswith("routing_plan") for k in got)


def test_the_norm_modules_are_not_tapped_as_boundaries():
    """attn_norm / ffn_norm are real submodules whose norm is computed inside
    mhc_pre_delayed_tilelang -- it only borrows their weights. A hook on
    either would never fire, and "no rows" reads as "the layer did not run"."""
    tapped = {attribute for attribute, _, _ in align_taps._BOUNDARY_TAPS}
    assert "attn_norm" not in tapped and "ffn_norm" not in tapped


# ------------------------------------------------------------- the prologue

class _Embed(torch.nn.Module):
    def forward(self, input_ids):
        return input_ids.unsqueeze(-1).float().expand(-1, 4)


class _ModelWithEmbed(torch.nn.Module):
    def __init__(self, record, num_layers=1):
        super().__init__()
        self.embed_tokens = _Embed()
        self.layers = torch.nn.ModuleList(
            [_Layer(record) for _ in range(num_layers)])
        self.start_layer = 0

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, positions)
        return x


@pytest.fixture
def prologue_recorder(monkeypatch):
    """A recorder that refuses record() before set_seq_len(), like the real one."""
    state = {"seq_len": None, "calls": [], "order": []}

    class _Recorder:
        def set_seq_len(self, seq_len):
            state["seq_len"] = seq_len
            state["order"].append("set_seq_len")

        def record(self, tag, key, tensor, row_to_pos):
            if state["seq_len"] is None:
                raise RuntimeError("set_seq_len() must be called before record()")
            state["order"].append(f"{tag}/{key}")
            state["calls"].append((tag, key))

        def record_layer(self, layer_id, key, tensor, row_to_pos):
            self.record(f"layer_{layer_id:02d}", key, tensor, row_to_pos)

    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align
    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Recorder())
    monkeypatch.setattr(debug_align, "current_row_to_pos",
                        lambda positions: positions.to(torch.int64))
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: 1100)
    monkeypatch.setattr(align_taps, "_CURRENT_POSITIONS", None)
    return state


def test_seq_len_is_set_before_the_first_record(prologue_recorder):
    """The failure this hook exists for.

    Without it the first tap reached -- sa_in.q on the first layer -- raises
    "set_seq_len() must be called before record()", six taps are entered, none
    record, and the dump is empty. Measured on the first real run.
    """
    model = _ModelWithEmbed([])
    align_taps.install_alignment_taps(model)
    model(torch.arange(3), torch.arange(3))
    assert prologue_recorder["order"][0] == "set_seq_len"


def test_the_prologue_keys_are_tagged_prologue_not_layer_00(prologue_recorder):
    model = _ModelWithEmbed([])
    align_taps.install_alignment_taps(model)
    model(torch.arange(3), torch.arange(3))
    assert ("prologue", "input_ids") in prologue_recorder["calls"]
    assert ("prologue", "embed.out") in prologue_recorder["calls"]


def test_positions_are_available_before_the_embedding_runs(prologue_recorder):
    """The prologue runs before any decoder layer, so a positions capture
    attached only to the layers would give None here -- or, worse, the
    previous step's, which is in range and wrong."""
    model = _ModelWithEmbed([])
    align_taps.install_alignment_taps(model)
    model(torch.arange(3), torch.tensor([7, 8, 9]))
    assert ("prologue", "embed.out") in prologue_recorder["calls"]


def test_a_forward_without_positions_clears_the_previous_ones(no_recorder):
    """Assigned unconditionally, including to None: a leftover position
    vector from the last step is in range and wrong."""
    align_taps._CURRENT_POSITIONS = torch.tensor([1, 2, 3])
    align_taps._capture_positions(torch.nn.Identity(), (torch.zeros(3),))
    assert align_taps._current_positions() is None


# --------------------------------------------- the premise, not the convention

def test_the_production_model_installs_the_taps():
    """"This function is correct" and "this function is called" are two things.

    The taps were written, tested and merged with zero production call sites,
    and the symptom was not "no hooks": the six inline taps in attention.py
    still ran, reached record() before any set_seq_len(), and raised -- so the
    dump came back empty and the cause was two inferences away. Twice.

    Derived from the source, because the failure is an absence and an absence
    is what a test of behaviour cannot see.
    """
    import inspect

    from vllm_ascend.models.deepseek_v4_1.model import \
        AscendDeepseekV41ForCausalLM

    source = inspect.getsource(AscendDeepseekV41ForCausalLM.__init__)
    assert "install_alignment_taps(self)" in source, (
        "nothing in the production model installs the alignment taps; the "
        "hooks will never register and every inline tap will raise")


def test_a_tap_before_set_seq_len_says_what_is_missing(prologue_recorder,
                                                       monkeypatch):
    """The invariant itself, not "who should call whom".

    Guarding the convention ("the model must call install") only catches the
    wiring that was forgotten last time. This catches any path that reaches a
    tap with no sequence length set, however it got there.
    """
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align

    monkeypatch.setattr(align_taps, "_CURRENT_POSITIONS", torch.arange(3))

    class _Unprepared:
        def record_layer(self, *a, **k):
            raise RuntimeError(
                "set_seq_len() must be called before record()")

    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Unprepared())
    with pytest.raises(RuntimeError, match="set_seq_len"):
        align_taps.tap(0, "sa_in.q", torch.zeros(3, 4), torch.arange(3))


def test_zero_handles_with_a_recorder_present_is_reported_as_a_defect(
        with_recorder, monkeypatch):
    """A model with nothing to hook is not a quiet no-op.

    Both diagnoses so far went "the dump is empty" -> "the hooks were never
    installed", two steps from the fact. This makes the fact itself appear.
    """
    import logging

    from vllm.logger import logger as vllm_logger

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = Capture()
    vllm_logger.addHandler(handler)
    try:
        # A model with no decoder layers at all now returns before
        # registering anything, including the positions capture -- there is
        # nothing to capture positions for.
        assert align_taps.install_alignment_taps(torch.nn.Identity()) == []
    finally:
        vllm_logger.removeHandler(handler)

    assert any("zero taps were installed" in m for m in records), records
    assert any("no module under" in m for m in records), records


# ------------------------------------------- found by capability, not by path

class _Wrapped(torch.nn.Module):
    """The real shape: a multimodal wrapper one level deeper."""

    def __init__(self, record):
        super().__init__()
        self.language_model = _ModelWithEmbed(record)

    def forward(self, input_ids, positions):
        return self.language_model(input_ids, positions)


def test_taps_are_installed_through_a_multimodal_wrapper(prologue_recorder):
    """model.model was hard-coded and found nothing: V4.1-Flash is a
    multimodal checkpoint and the real path is language_model.model."""
    model = _Wrapped([])
    align_taps.install_alignment_taps(model)
    model(torch.arange(3), torch.arange(3))
    assert ("prologue", "embed.out") in prologue_recorder["calls"]
    assert any(tag.startswith("layer_") for tag, _ in prologue_recorder["calls"])


def test_the_search_is_by_capability_not_by_a_named_level():
    """A path naming language_model would work today and go silently empty
    behind the next wrapper -- the same failure at a different address."""
    import inspect

    source = inspect.getsource(align_taps._find_inner_model)
    body = source.split('"""')[2]
    assert "language_model" not in body, (
        "the search hard-codes a wrapper name again")
    assert "layers" in body


def test_a_rank_without_an_embedding_still_taps_its_layers(prologue_recorder):
    """A pipeline rank that is not the first has no usable embed_tokens.

    Requiring both capabilities would find nothing there and tap nothing --
    silently, which is exactly the failure this search exists to stop.
    """
    model = _Model([])          # layers, no embed_tokens
    handles = align_taps.install_alignment_taps(model)
    assert handles, "a model with layers but no embedding was not tapped"


def test_an_unrecognisable_model_reports_what_it_looked_at(with_recorder):
    """Zero taps is a defect, not a quiet no-op, and the message has to carry
    enough to act on -- the last diagnosis needed the children's names."""
    import logging

    from vllm.logger import logger as vllm_logger

    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    class _Opaque(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.something_else = torch.nn.Identity()

    handler = Capture()
    vllm_logger.addHandler(handler)
    try:
        assert align_taps.install_alignment_taps(_Opaque()) == []
    finally:
        vllm_logger.removeHandler(handler)

    assert any("something_else" in m for m in records), records
