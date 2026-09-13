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

    # ⚠️ `positions` first, `hidden_states` second -- the real attention
    # module's order (attention.py:545). A double taking one tensor made
    # "which parameter" untestable, and "the first positional argument" is
    # exactly the assumption that put an int64 position vector into the key
    # golden uses for a bf16 hidden state.
    def forward(self, positions, hidden_states):
        self.record.append("inner-ran")
        return hidden_states * 2


class _Layer(torch.nn.Module):
    def __init__(self, record):
        super().__init__()
        self.attn = _Inner(record)
        self.ffn = _Inner(record)

    # ⚠️ The real signature, argument for argument
    # (vllm/models/deepseek_v4_1/nvidia/model.py:285-296). The fixture used to
    # stop at input_ids, which modelled a decoder layer that cannot exist --
    # and a tap bound to `residual` by name could not be tested against it at
    # all. A double that is missing the thing under test is not a double.
    def forward(self, x, positions, input_ids=None, pre_mix=None,
                post_mix=None, res_mix=None, residual=None,
                engram_hashes=None, engram_mask=None):
        return self.ffn(positions, self.attn(positions, x))


class _Model(torch.nn.Module):
    def __init__(self, record, num_layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [_Layer(record) for _ in range(num_layers)])
        self.start_layer = 0

    def forward(self, x, positions, residual=None, pre_mix=None):
        # Positionally, exactly as the real model calls it (model.py:626-636),
        # because the taps resolve names to positions once and then index.
        for layer in self.layers:
            x = layer(x, positions, None, pre_mix, None, None, residual)
            # After the first layer the streams exist, as upstream does.
            residual = residual if residual is not None else x
            pre_mix = pre_mix if pre_mix is not None else x[..., :1]
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
        # ⚠️ Carries a seq_len like the real one. Recording is armed per
        # forward on "can this recorder size its buffer", so a double without
        # it disarms every tap and the test then asserts about a run in which
        # nothing was ever going to be recorded.
        seq_len = 1100

        def set_seq_len(self, seq_len):
            self.seq_len = seq_len

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
    # The module edges, on both layers...
    assert {k for k in keys if k[1] in ("attn.in", "attn.out", "moe.in",
                                       "moe.out")} == {
        (0, "attn.in"), (0, "attn.out"), (0, "moe.in"), (0, "moe.out"),
        (1, "attn.in"), (1, "attn.out"), (1, "moe.in"), (1, "moe.out")}
    # ...and the layer's own inputs, which exist only once the streams do.
    # Layer 0 receives residual=None and pre_mix=None and records neither:
    # "layer 0 has no pre_mix" is a fact the dump has to be able to show, and
    # a fabricated identity would erase it.
    assert (1, "layer.in") in keys and (1, "pre_mix.in") in keys
    assert (0, "layer.in") not in keys and (0, "pre_mix.in") not in keys


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
    tapped = {entry[0] for entry in align_taps._BOUNDARY_TAPS}
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
    # ⚠️ Armed on purpose. An *unarmed* tap is now silent -- that is the fix
    # for the forwards that must not be recorded at all. What is still a
    # programming error, and still has to raise, is reaching a tap while
    # armed against a recorder that was never given a length: the two cases
    # were one before, and only one of them was a defect.
    monkeypatch.setattr(align_taps, "_ARMED", True)

    class _Unprepared:
        seq_len = None

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


# ------------------------------------ did we record everything golden compares

def test_golden_layers_come_from_goldens_own_manifest(tmp_path):
    """Not from what we produced. A layer we failed to record entirely has no
    tag of ours to be discovered from, so deriving the list from our own
    output would make the omission we are looking for invisible."""
    from vllm_ascend.models.deepseek_v4_1.align_taps import golden_layers

    d = _manifest(tmp_path, {
        "prologue": ["input_ids"],
        "layer_00": ["attn.out"],
        "layer_02": ["attn.out"],
        "layer_20": ["attn.out"],
        "epilogue": ["logits"],
    })
    assert golden_layers(d) == (0, 2, 20)


def test_a_complete_dump_reports_ok(tmp_path):
    from vllm_ascend.models.deepseek_v4_1.align_taps import coverage_report

    d = _manifest(tmp_path, {"prologue": ["input_ids"],
                             "layer_00": ["attn.out", "sa_in.q"]})
    recorded = {"prologue/input_ids", "layer_00/attn.out", "layer_00/sa_in.q"}
    report = coverage_report(recorded, d)
    assert report["state"] == "ok"
    assert report["missing"] == []
    assert report["expected_count"] == 3


def test_a_key_never_recorded_is_reported_as_missing(tmp_path):
    """🔴 The case the position-coverage assertion structurally cannot see.

    That assertion walks ``self._coverage``, which only has an entry for a key
    some call recorded. A key nothing ever wrote has no buffer, no coverage
    vector, and therefore nothing to be checked -- it passes by not being
    looked at. A whole tensor silently absent and a whole tensor perfectly
    recorded produce the same green flush.
    """
    from vllm_ascend.models.deepseek_v4_1.align_taps import coverage_report

    d = _manifest(tmp_path, {"layer_00": ["attn.out", "moe_gate.out.1"]})
    report = coverage_report({"layer_00/attn.out"}, d)
    assert report["state"] == "incomplete"
    assert report["missing"] == ["layer_00/moe_gate.out.1"]


def test_keys_golden_does_not_compare_are_reported_too(tmp_path):
    """The other direction. Not a failure -- bytes and seconds for nothing,
    and it moves the tensor count, which is the number everyone reads first."""
    from vllm_ascend.models.deepseek_v4_1.align_taps import coverage_report

    d = _manifest(tmp_path, {"layer_00": ["attn.out"]})
    report = coverage_report({"layer_00/attn.out", "layer_00/scratch"}, d)
    assert report["state"] == "ok"
    assert report["extra"] == ["layer_00/scratch"]


def test_no_golden_directory_is_not_checked_rather_than_agreement():
    """⚠️ The three-state. Two empty differences read as full agreement, and
    that reading must not be available to something that did not run."""
    from vllm_ascend.models.deepseek_v4_1.align_taps import coverage_report

    report = coverage_report({"layer_00/attn.out"}, None)
    assert report["state"] == "not_checked"
    assert "not a pass" in report["reason"].lower()
    assert "missing" not in report and "extra" not in report, (
        "an unchecked run published differences, and empty differences are "
        "what agreement looks like")


def test_an_unreadable_golden_manifest_is_not_checked_either(tmp_path):
    from vllm_ascend.models.deepseek_v4_1.align_taps import coverage_report

    report = coverage_report({"layer_00/attn.out"}, str(tmp_path))
    assert report["state"] == "not_checked"
    assert "not a pass" in report["reason"].lower()


def test_the_report_never_raises_on_a_bad_golden_dir(tmp_path):
    """The dump costs a launch. Nothing in this path may throw it away."""
    from vllm_ascend.models.deepseek_v4_1.align_taps import coverage_report

    (tmp_path / "manifest.json").write_text("{not json")
    assert coverage_report(set(), str(tmp_path))["state"] == "not_checked"


def test_a_layer_missing_in_its_entirety_is_still_reported(tmp_path):
    """🔴 The case that says why the layer list comes from golden.

    Mutation-tested: deriving the layers from our own recorded keys passes
    every other test in this file and turns this one green-by-omission --
    layer_02 is filtered out of the expected set by the very fact that we
    recorded nothing for it, so "we recorded nothing for layer 2" and "layer 2
    is complete" become the same answer. The missing row cannot show itself in
    an enumeration built from the rows that are present.
    """
    from vllm_ascend.models.deepseek_v4_1.align_taps import coverage_report

    d = _manifest(tmp_path, {"layer_00": ["attn.out"],
                             "layer_02": ["attn.out"]})
    report = coverage_report({"layer_00/attn.out"}, d)
    assert report["state"] == "incomplete"
    assert report["missing"] == ["layer_02/attn.out"]


# ------------------- recording is armed once per forward, or not at all

def test_a_forward_that_cannot_be_sized_records_nothing(monkeypatch,
                                                        with_recorder):
    """🔴 The failure this replaced: recording left half-armed.

    set_seq_len never ran, every later tap called record(), and the run died
    forty taps away with "set_seq_len() must be called before record(),
    normally from the prologue hook" -- a message naming the hook that did
    *not* fail. It cost three launches once, and another the first time a
    request came in through /v1/completions.
    """
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align

    class _Unsized:
        seq_len = None

        def set_seq_len(self, seq_len):
            raise AssertionError("must not be reached")

        def record_layer(self, *a):
            raise AssertionError("recording was left armed")

    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Unsized())
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: None)

    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))   # must not raise
    finally:
        for h in handles:
            h.remove()
    # The fixture's list also collects the model's own "inner-ran" markers,
    # so what is asserted is that no *tap* wrote, not that nothing happened.
    assert not [c for c in with_recorder if isinstance(c, tuple)]


def test_the_disarming_says_so(monkeypatch, with_recorder):
    """Said where it is discovered, not forty taps later."""
    import logging

    from vllm.logger import logger

    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align
    from vllm_ascend.models.deepseek_v4_1.switch_log import reset_for_test

    class _Unsized:
        seq_len = None

        def record_layer(self, *a):
            pass

    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Unsized())
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: None)

    lines = []

    class _Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    reset_for_test()
    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
        logger.removeHandler(handler)
        reset_for_test()

    assert any("alignment recording is disabled for this forward" in line
               for line in lines), lines


def test_a_recorder_that_already_has_a_length_stays_armed(monkeypatch,
                                                          with_recorder):
    """⚠️ The opposite failure, and just as silent: refusing to record a run
    that is perfectly recordable. The precondition is "can this recorder size
    its buffer", not "did the forward context tell us"."""
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: None)

    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
    assert [c for c in with_recorder if isinstance(c, tuple)], \
        "the fixture's recorder carries seq_len = 1100, so taps must record"


def test_the_arming_hook_runs_before_every_tap():
    """Ordering, stated as an observable: it is registered on the module that
    always runs, and before the positions capture."""
    import inspect

    source = inspect.getsource(align_taps.install_alignment_taps)
    assert source.index("_arm_recording") < source.index("_capture_positions")


def test_arming_is_reported_too(monkeypatch, with_recorder):
    """🔴 "disarmed" alone cannot be read.

    The first forward of a server is a profile run with no request, so it
    disarms; without a line for the success case, "disarmed" is the only
    thing in the log whether or not a real forward ever armed -- and those
    are opposite readings.
    """
    import logging

    from vllm.logger import logger

    from vllm_ascend.models.deepseek_v4_1.switch_log import reset_for_test

    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: 77)
    lines = []

    class _Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    reset_for_test()
    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
        logger.removeHandler(handler)
        reset_for_test()

    assert any("recording armed at seq_len=77" in line for line in lines), lines


def test_a_recorded_prefill_flushes_and_rotates(monkeypatch, with_recorder):
    """Flushed at the end of the forward, which is what makes two dumps in
    one server possible."""
    rotations = []
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align

    monkeypatch.setattr(debug_align, "rotate_recorder",
                        lambda: rotations.append(1) or True)
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: 5)

    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
    assert rotations == [1]


def test_a_disarmed_forward_does_not_flush(monkeypatch, with_recorder):
    """⚠️ A decode step disarms, and flushing there would write an empty dump
    over a real one -- a well-formed artifact that says nothing and reads as
    if it did, which is the trace-sentinel failure again."""
    rotations = []
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align

    class _Unsized:
        seq_len = None

        def record_layer(self, *a):
            pass

    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Unsized())
    monkeypatch.setattr(debug_align, "rotate_recorder",
                        lambda: rotations.append(1) or True)
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: None)

    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
    assert rotations == []


def test_the_disarm_line_says_which_step_came_up_empty(monkeypatch,
                                                       with_recorder):
    """🔴 "No sequence length" has four causes needing four different next
    actions. Reporting only the conclusion makes a flight that disarms cost a
    second flight to find out why -- and this path has already been debugged
    twice from the outside."""
    import logging

    from vllm.logger import logger

    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align
    from vllm_ascend.models.deepseek_v4_1.switch_log import reset_for_test

    class _Unsized:
        seq_len = None

        def record_layer(self, *a):
            pass

    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Unsized())
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: None)

    lines = []

    class _Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    reset_for_test()
    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
        logger.removeHandler(handler)
        reset_for_test()

    disarm = [line for line in lines if "recording is disabled" in line]
    assert disarm, lines
    # Off the model path there is no forward context, which is one of the
    # four and must be named as that one rather than as "not available".
    assert "no forward context here at all" in disarm[0], disarm


def test_a_forward_whose_rows_are_all_padding_records_nothing(monkeypatch,
                                                              with_recorder):
    """🔴 What stopped the dump flight.

    The run armed on a real request and then met a forward whose rows all
    mapped to no position. Recording it created buffers whose coverage was
    zero everywhere, and the flush reported the whole key as missing -- which
    is the same artifact as a broken row mapping, and a different defect.
    """
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align

    monkeypatch.setattr(debug_align, "current_row_to_pos",
                        lambda positions: torch.full_like(
                            positions.to(torch.int64), -1))
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: 8)

    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
    assert not [c for c in with_recorder if isinstance(c, tuple)], (
        "an all-padding forward still wrote, which is what produced a dump "
        "whose keys were entirely missing")


def test_a_forward_with_some_real_rows_still_records(monkeypatch,
                                                     with_recorder):
    """Two arms: the skip must not swallow a forward that has real rows."""
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align

    monkeypatch.setattr(debug_align, "current_row_to_pos",
                        lambda positions: torch.tensor([-1, 1, -1]))
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: 8)

    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
    assert [c for c in with_recorder if isinstance(c, tuple)]


# ------------- only a forward that can fill the buffer opens one

def _armed_with(monkeypatch, with_recorder, seq_len, rows):
    monkeypatch.setattr(align_taps, "_single_request_seq_len",
                        lambda: seq_len)
    monkeypatch.setattr(align_taps, "_single_request_query_rows",
                        lambda: rows)
    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
    return [c for c in with_recorder if isinstance(c, tuple)]


def test_a_decode_step_is_not_recorded(monkeypatch, with_recorder):
    """🔴 What killed the dump flight.

    A warm-up request recorded its prefill correctly and then its first
    decode armed at seq_len=202 -- 201 + 1, which only a decode can be --
    recorded one row, and the flush raised. The assertion was firing on an
    object its premise does not cover: one row can never write 202 positions.
    """
    assert _armed_with(monkeypatch, with_recorder, seq_len=202, rows=1) == []


def test_a_chunked_prefill_is_not_recorded(monkeypatch, with_recorder):
    """Same shape, different cause: some of the positions, not one. Also
    normal, also produces the artifact a broken mapping produces."""
    with_recorder.clear()
    assert _armed_with(monkeypatch, with_recorder, seq_len=201, rows=64) == []


def test_a_full_prefill_is_recorded(monkeypatch, with_recorder):
    """Two arms. The one forward that can write every position exactly once
    is the one that opens a buffer."""
    with_recorder.clear()
    assert _armed_with(monkeypatch, with_recorder, seq_len=3, rows=3)


def test_the_refusal_names_both_numbers(monkeypatch, with_recorder):
    """⚠️ "cannot be recorded" without the pair cannot be acted on: 1 of 202
    is a decode, 64 of 201 is a chunk, and they lead different places."""
    import logging

    from vllm.logger import logger

    from vllm_ascend.models.deepseek_v4_1.switch_log import reset_for_test

    lines = []

    class _Collect(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())

    handler = _Collect()
    logger.addHandler(handler)
    reset_for_test()
    try:
        _armed_with(monkeypatch, with_recorder, seq_len=202, rows=1)
    finally:
        logger.removeHandler(handler)
        reset_for_test()

    assert any("1 query rows against a sequence length of 202" in line
               for line in lines), lines
    assert any("not a defect" in line for line in lines), lines


def test_an_unknown_row_count_does_not_block_recording(monkeypatch,
                                                       with_recorder):
    """⚠️ Three states. "Cannot tell how many rows" must not silently refuse a
    recordable forward -- that is the same wrong-direction failure as
    demanding the length come from the forward context."""
    with_recorder.clear()
    assert _armed_with(monkeypatch, with_recorder, seq_len=3, rows=None)


# ------------------------------- every writer goes through the same door

def test_every_recording_call_comes_through_the_armed_door():
    """🔴 Enumerating writers by pattern finds the ones that look like the
    writers you already know about.

    _record_prologue was a second writer of the same shape and a text match
    caught it. _record_with -- the inline tap API -- had a different shape,
    the same text match did not reach it, and a decode step's tap called
    record() against a recorder with no sequence length. This walks the AST
    instead, so a writer that looks like nothing already here still has to
    come through _armed_recorder.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(align_taps)))
    writers = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        records = [n for n in ast.walk(func)
                   if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ("record", "record_layer",
                                       "record_whole")]
        if records:
            writers.append(func)

    assert writers, "no recording calls found; this check is not checking"
    for func in writers:
        names = {n.func.id for n in ast.walk(func)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert "_armed_recorder" in names, (
            f"{func.name} writes to the recorder without going through "
            "_armed_recorder; a forward that must not be recorded will reach "
            "it, and record() raises when the buffer has no sequence length")


def test_the_inline_tap_is_silent_when_not_armed(monkeypatch, with_recorder):
    """The specific path that raised: tap() during a decode step."""
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align

    class _Explodes:
        seq_len = None

        def record_layer(self, *a):
            raise AssertionError("recorded while disarmed")

        def record_whole(self, *a):
            raise AssertionError("recorded while disarmed")

    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Explodes())
    monkeypatch.setattr(align_taps, "_ARMED", False)

    align_taps.tap(0, "sa_in.q", torch.zeros(1, 4), torch.tensor([0]))
    align_taps.tap_whole(0, "sa_in.attn_sink", torch.zeros(4))


def test_the_inline_tap_records_when_armed(monkeypatch, with_recorder):
    """Two arms: the door must open."""
    import vllm_ascend.models.deepseek_v4_1.debug_align as debug_align

    written = []

    class _Recorder:
        seq_len = 4

        def record_layer(self, *a):
            written.append("layer")

        def record_whole(self, *a):
            written.append("whole")

    monkeypatch.setattr(debug_align, "get_recorder", lambda: _Recorder())
    monkeypatch.setattr(debug_align, "current_row_to_pos",
                        lambda positions: positions.to(torch.int64))
    monkeypatch.setattr(align_taps, "_ARMED", True)

    align_taps.tap(0, "sa_in.q", torch.zeros(1, 4), torch.tensor([0]))
    align_taps.tap_whole(0, "sa_in.attn_sink", torch.zeros(4))
    assert written == ["layer", "whole"]


# ------------------------------- a key name is not a contract

def test_attn_in_records_the_hidden_state_not_the_positions():
    """🔴 It shipped wrong.

    DeepseekV4Attention.forward(self, positions, hidden_states, ...) takes
    positions first, and "the module's first positional input" put an int64
    [T] position vector into the key golden uses for a bf16 [T, 5120] hidden
    state. Both sides had a tensor called attn.in; they were different
    tensors; nothing said so, because a key name is not a contract.
    """
    entries = {key: parameter
               for _, key, side, parameter in align_taps._BOUNDARY_TAPS
               if side == "in"}
    assert entries == {"attn.in": "hidden_states",
                       "attn.positions": "positions",
                       "moe.in": "hidden_states"}


def test_a_module_without_that_parameter_fails_installation():
    """Loudly, rather than recording the neighbour."""
    class _Wrong(torch.nn.Module):
        def forward(self, positions):
            return positions

    with pytest.raises(RuntimeError, match="has no 'hidden_states'"):
        align_taps._parameter_index(_Wrong(), "hidden_states", "attn.in")


def test_the_tapped_tensor_is_the_named_one(with_recorder):
    """Two arms, on a double whose two arguments are distinguishable."""
    model = _Model(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        hidden = torch.randn(3, 4)
        model(hidden, torch.arange(3))
    finally:
        for h in handles:
            h.remove()
    shapes = {entry[1]: entry[2] for entry in with_recorder
              if isinstance(entry, tuple)}
    # The two candidates are distinguishable by shape alone: the hidden state
    # is [T, H] and positions is [T]. That is the whole difference the bug
    # turned on, and it is what the dump showed -- I64 (1,5) against BF16
    # (1,5,5120).
    assert shapes["attn.in"] == tuple(hidden.shape), shapes
    assert shapes["attn.in"] != tuple(torch.arange(3).shape)


def test_every_recorded_key_has_a_declared_meaning():
    """⚠️ Two keys have already turned out to mean different things on the
    two sides. Once that happens to one key, every other key is unverified
    too -- so each one says what it is rather than leaving a reader to infer
    it from a name two teams chose separately."""
    declared = set(align_taps.KEY_SEMANTICS)
    tapped = {key for _, key, _, _ in align_taps._BOUNDARY_TAPS}
    tapped |= {key for _, key in align_taps._ARG_TAPS}
    missing = tapped - declared
    assert not missing, missing
    assert "⚠️" in align_taps.KEY_SEMANTICS["sa_in.kv"], (
        "the one key already known to differ in meaning must say so")


def test_positions_are_recorded_under_their_own_key():
    """🔵 In its own right, not for compatibility. positions reaches layer 0,
    and the earliest divergence is at layer 0 -- "input_ids matched but
    positions did not" produces exactly the symptom of "the embedding is
    wrong", and only this key separates them."""
    entries = {key: parameter
               for _, key, side, parameter in align_taps._BOUNDARY_TAPS
               if side == "in"}
    assert entries["attn.positions"] == "positions"
    assert entries["attn.in"] == "hidden_states"


def test_the_kv_layout_is_declared_with_what_is_not_established():
    """⚠️ A comparison against golden that does not know the layout produces
    a number that looks like a finding -- a cosine of 0.15 between two
    tensors that are not the same thing."""
    text = align_taps.KEY_SEMANTICS["read.ori_kv"]
    assert "post-RoPE" in text and "448 NoPE | 64 RoPE" in text
    source = align_taps.KEY_SEMANTICS.__str__()
    del source
    import inspect
    module = inspect.getsource(align_taps)
    assert "NOT established" in module, (
        "the layout block states only what is known; the half that is not "
        "known is what stops a low cosine from being read as a finding")


def test_the_renamed_key_says_dumps_across_it_are_not_comparable():
    text = align_taps.KEY_SEMANTICS["attn.in"]
    assert "must not be compared" in text


# ------------------------- the forward-level arguments

class _EngramLayer(_Layer):
    def forward(self, x, positions, input_ids=None, pre_mix=None,
                post_mix=None, res_mix=None, residual=None,
                engram_hashes=None, engram_mask=None):
        return self.ffn(positions, self.attn(positions, x))


class _EngramModel(torch.nn.Module):
    def __init__(self, record, n=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(_EngramLayer(record)
                                          for _ in range(n))
        self.start_layer = 0

    def forward(self, x, positions, engram_hashes=None):
        for layer in self.layers:
            x = layer(x, positions, engram_hashes=engram_hashes)
        return x


def test_engram_hash_ids_are_recorded_once_under_prologue(monkeypatch,
                                                          with_recorder):
    """🔵 The engram tables are gathered *by these ids*: a wrong id returns a
    different row of a 189 GiB table and the layer then computes correctly on
    the wrong weights -- structural, present from the first layer, identical
    on every request, independent of cross-request state. That is the whole
    description of the defect being hunted, and it has never been recorded.
    """
    recorded = []
    monkeypatch.setattr(align_taps, "_record_prologue",
                        lambda key, tensor: recorded.append(key))
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: 3)
    monkeypatch.setattr(align_taps, "_single_request_query_rows", lambda: 3)
    align_taps._FORWARD_RECORDED.clear()

    model = _EngramModel(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3),
              engram_hashes=torch.arange(6).reshape(3, 2))
    finally:
        for h in handles:
            h.remove()

    # Once, not once per layer: the model computes it for the whole forward,
    # and three copies under three layer tags would make it look
    # layer-specific and put it where golden does not compare.
    assert recorded == ["engram_hash_ids"], recorded


def test_a_forward_without_engram_hashes_records_nothing(monkeypatch,
                                                         with_recorder):
    """Two arms. Absent is absent -- a model with no engram layers has no
    hashes, and that is not a defect."""
    recorded = []
    monkeypatch.setattr(align_taps, "_record_prologue",
                        lambda key, tensor: recorded.append(key))
    monkeypatch.setattr(align_taps, "_single_request_seq_len", lambda: 3)
    monkeypatch.setattr(align_taps, "_single_request_query_rows", lambda: 3)
    align_taps._FORWARD_RECORDED.clear()

    model = _EngramModel(with_recorder)
    handles = align_taps.install_alignment_taps(model)
    try:
        model(torch.randn(3, 4), torch.arange(3))
    finally:
        for h in handles:
            h.remove()
    assert recorded == []


def test_the_next_forward_records_its_own(monkeypatch, with_recorder):
    """⚠️ Cleared at the flush. Otherwise the second forward keeps the
    first's ids, and a stale tensor is in range and wrong -- the failure this
    module has already had once with positions."""
    align_taps._FORWARD_RECORDED.add("engram_hash_ids")
    align_taps._ARMED = True
    align_taps._flush_and_rotate(None, (), None)
    assert "engram_hash_ids" not in align_taps._FORWARD_RECORDED
