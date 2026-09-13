# SPDX-License-Identifier: Apache-2.0
"""Where the alignment dump is taken from, and why it costs nothing when off.

The tensors golden compares are module inputs and outputs, so they are read
with torch forward hooks rather than with calls placed in the forward path.
That is not a style preference -- it is what makes "off" provable:

* when the dump directory is unset, :func:`install_alignment_taps` registers
  **no hooks at all**, so no tap function is entered, nothing is branched on
  per layer, and the module graph is byte-identical to the untapped one;
* the alternative -- ``if recording: tap(...)`` at thirty points -- is off by
  early return, which is a claim about what the branch does rather than about
  whether it runs.

Not every protocol tensor is a module boundary. ``sa_in.*``, ``hc_*``,
``moe_gate.*`` and ``pre_mix.*`` are intermediates inside functions this line
owns, and those are tapped inline in their own files. The split is by what the
value *is*, not by convenience: a hook can only see what crosses a module
edge, and pretending otherwise would mean hooking a module that does not
compute the thing whose name it carries.

⚠️ One such trap is already visible. ``attn_norm`` and ``ffn_norm`` are real
submodules, but the norm they are named after is computed *inside*
``mhc_pre_delayed_tilelang``, which only borrows their weights -- a hook on
either would never fire, and "no rows recorded" reads exactly like "this
layer did not run".
"""

from __future__ import annotations

import re
from typing import Any

import torch

#: Per-layer dump tags. The only naming rule this module knows,
#: and it is used to *exclude* other layers -- never to decide
#: what counts as a tag.
_LAYER_TAG = re.compile(r"layer_\d+")

#: Protocol key -> what to read at that module edge. "in" takes the module's
#: first positional input, "out" its output.
_BOUNDARY_TAPS: tuple[tuple[str, str, str], ...] = (
    ("attn", "attn.in", "in"),
    ("attn", "attn.out", "out"),
    ("ffn", "moe.in", "in"),
    ("ffn", "moe.out", "out"),
)


#: Protocol key -> the decoder layer's *argument* carrying it. These are the
#: keys the injector replaces, so they have to be the tensors the layer was
#: actually handed, not anything reconstructed from them.
#:
#: 🔴 Bound by parameter **name**, resolved once at install time. The call site
#: passes all nine positionally (``model.py:626-636``), so ``args[6]`` would
#: work today and would silently record the wrong tensor the day an argument
#: is inserted ahead of it. Resolving the name and failing loudly if it is
#: gone is the difference between a wrong dump and no dump.
_ARG_TAPS: tuple[tuple[str, str], ...] = (
    ("residual", "layer.in"),
    ("pre_mix", "pre_mix.in"),
)


def _arg_positions(layer: torch.nn.Module) -> dict[str, int]:
    """Where each tapped argument sits in this layer's positional signature."""
    import inspect

    parameters = list(
        inspect.signature(type(layer).forward).parameters)
    positions = {}
    for name, key in _ARG_TAPS:
        if name not in parameters:
            raise RuntimeError(
                f"{type(layer).__name__}.forward has no {name!r} parameter, so "
                f"{key!r} cannot be recorded from the tensor the layer is "
                "handed. Recording it from anywhere else would be recording a "
                "reconstruction: the injector would then be replacing "
                "something the layer never received. Its parameters are "
                f"{parameters}.")
        # -1 for self, which a hook's args tuple does not carry.
        positions[key] = parameters.index(name) - 1
    return positions


def _first_tensor(value: Any) -> torch.Tensor | None:
    """The tensor a hook actually got, out of whatever shape it arrived in."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _record(tag_layer: int, key: str, tensor: torch.Tensor) -> None:
    """One tensor into the recorder, addressed by global sequence position.

    The row mapping is rebuilt here rather than cached because it belongs to
    the forward, not to the tap: a cached one would survive into the next
    step and scatter this step's rows onto last step's positions.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import (current_row_to_pos,
                                                              get_recorder)

    recorder = get_recorder()
    if recorder is None or not _ARMED:
        return
    positions = _current_positions()
    if positions is None:
        return
    row_to_pos = current_row_to_pos(positions)
    if row_to_pos is None:
        # Profile run, dummy run or graph capture. Reported by the resolver,
        # not swallowed here.
        return
    rows = tensor.reshape(row_to_pos.numel(), -1) if tensor.dim() > 2 else tensor
    recorder.record_layer(tag_layer, key, rows.reshape(tensor.shape), row_to_pos)


#: The positions argument of the decoder layer currently executing. Captured
#: by a pre-hook on the layer itself rather than looked up elsewhere.
_CURRENT_POSITIONS: torch.Tensor | None = None


#: Whether this forward's recording is armed. False until the sequence length
#: is known, because every ``record()`` scatters into a [seq_len, ...] buffer
#: and raises without one.
_ARMED = False


def _arm_recording(_module: torch.nn.Module, _args: tuple) -> None:
    """Fix the sequence length for this forward, or disarm it and say so.

    🔴 This used to live in the prologue hook and to ``return`` when the length
    could not be determined. The result was recording left half-armed:
    ``set_seq_len`` never ran, every later tap called ``record()``, and the run
    died forty taps away with

        RuntimeError: set_seq_len() must be called before record(),
                      normally from the prologue hook

    -- a message that names the hook which did *not* fail. It cost three
    launches once, and another the first time a request arrived through
    ``/v1/completions``, where the prologue's own precondition happened not to
    hold. The fix is not a better message: "cannot size the buffer" has to
    disable recording **here**, where it is discovered, instead of arming a
    trap for the next forty taps.

    Installed on the inner model, which always runs -- ``embed_tokens`` does
    not exist on a non-first pipeline rank -- and which is strictly earlier
    than every tap, the ordering the recorder needs.
    """
    global _ARMED

    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    recorder = get_recorder()
    if recorder is None:
        _ARMED = False
        return
    try:
        seq_len = _single_request_seq_len()
    except ValueError as exc:
        _ARMED = False
        _report_disarmed(str(exc))
        return
    if seq_len is None:
        # ⚠️ The precondition is "the recorder can size its buffer", not "the
        # forward context told us the length". A recorder constructed with a
        # seq_len already can, and disarming it would refuse to record a run
        # that is perfectly recordable -- the opposite failure to the one
        # this function exists for, and just as silent.
        if getattr(recorder, "seq_len", None) is not None:
            _ARMED = True
            return
        _ARMED = False
        _report_disarmed(
            "no single request's sequence length was available from the "
            "forward context and the recorder has none of its own (profile "
            "run, dummy run, or a batch this dump cannot be addressed "
            "against)")
        return
    recorder.set_seq_len(seq_len)
    _ARMED = True


def _report_disarmed(reason: str) -> None:
    from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

    announce_once(
        "align_disarmed",
        "alignment recording is disabled for this forward -- %s. Nothing will "
        "be written, so the dump will be short rather than wrong. Said here, "
        "where it is discovered: the previous behaviour left recording "
        "half-armed and failed forty taps later with a message naming the "
        "hook that did not fail.", reason)


def _capture_positions(_module: torch.nn.Module, args: tuple) -> None:
    """Remember this layer's ``positions`` for the boundary hooks below.

    The attention metadata does **not** carry positions -- checked, not
    assumed -- so a tap that looked for them there would find None on every
    forward and record nothing, which reads exactly like "this model never
    ran". The decoder layer's own signature has them as the second positional
    argument, which is the only place they exist without deriving them a
    second time from query_start_loc and seq_lens.
    """
    global _CURRENT_POSITIONS
    # Assigned unconditionally, including to None: leaving the previous
    # forward's positions in place is the failure mode this guards. A stale
    # position vector is in range and wrong, so every row lands somewhere
    # real and the coverage assertion may not even fire.
    _CURRENT_POSITIONS = args[1] if len(args) > 1 else None


def _current_positions() -> torch.Tensor | None:
    """The positions of the layer whose submodule is being tapped."""
    return _CURRENT_POSITIONS


def install_alignment_taps(model: torch.nn.Module) -> list:
    """Register the boundary hooks, or none at all.

    Returns the handles so a caller can remove them; an empty list is the
    normal case and is what makes the off state provable -- there is nothing
    to enter, not merely nothing to do.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    if get_recorder() is None:
        return []

    handles: list = []
    inner = _find_inner_model(model)
    if inner is None:
        _report_nothing_found(model)
        return []
    # 🔴 First, and on the model rather than on a layer. The layer-level
    # capture below runs only once the first decoder layer starts, which is
    # *after* embed_tokens -- so a prologue tap reading positions would get
    # either None or, worse, the previous step's, which is in range and wrong.
    # This also clears the value, so a forward that never reaches the capture
    # cannot leave the last one visible to the next.
    # Before the positions capture and before every tap: nothing records
    # until the buffer can be sized.
    handles.append(inner.register_forward_pre_hook(_arm_recording))
    handles.append(inner.register_forward_pre_hook(_capture_positions))
    # Counted apart from the positions capture, which always registers. Using
    # len(handles) would have made the "nothing was tapped" branch below
    # unreachable -- a guard that cannot fire, which is worse than none
    # because it reads as coverage.
    taps_installed = 0
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and type(embed).__name__ != "PPMissingLayer":
        # 🔴 The prologue is not optional plumbing. AlignRecorder.record()
        # raises until set_seq_len() has been called, so without this hook the
        # very first tap -- sa_in.q -- fails and the whole dump is empty. That
        # is what happened on the first real run: six taps were entered, none
        # recorded, and the error correctly said so.
        #
        # It lives on embed_tokens because that is the one module that runs
        # once per forward, before any layer, and whose output *is* a protocol
        # key. A separate "prologue" module does not exist to hook.
        handles.append(embed.register_forward_pre_hook(_prologue_pre))
        handles.append(embed.register_forward_hook(_prologue_post))
        taps_installed += 2
    for layer_id, layer in _decoder_layers(model):
        # First, so the boundary hooks below it in the same layer see this
        # layer's positions rather than the previous layer's.
        handles.append(layer.register_forward_pre_hook(_capture_positions))
        # The layer's own inputs, before its submodules see anything. These
        # are what the injector replaces, so they are read from the argument
        # the layer was handed.
        #
        # ⚠️ Layer 0 is a different shape of thing and is left to produce
        # nothing here: it receives ``x`` as 2-D [T, 5120] and builds the
        # 4-stream tensor itself (``model.py:303-305``), and its ``pre_mix``
        # is None. The prologue's ``embed.out`` is what layer 0 was actually
        # handed. Recording an expanded [T, 4, 5120] for it would be
        # equivalent *today* -- expand loses nothing -- and that is exactly
        # the trap: equivalence is a property of the current code, not of the
        # interface, so the day that line stops being a broadcast nothing
        # goes red and the injector quietly compares something else.
        handles.append(
            layer.register_forward_pre_hook(
                _make_argument_tap(layer_id, _arg_positions(layer))))
        taps_installed += 1
        for attribute, key, side in _BOUNDARY_TAPS:
            module = getattr(layer, attribute, None)
            if module is None:
                continue
            handles.append(_attach(module, layer_id, key, side))
            taps_installed += 1

    # Said out loud, because the two failures so far were both diagnosed by
    # inferring "the hooks were never installed" from "the dump is empty" --
    # two steps away from the fact. Zero handles with a recorder present is a
    # defect, not a quiet no-op, so it is reported as one.
    from vllm.logger import logger

    if taps_installed:
        logger.info(
            "DSV4.1 alignment: installed %d taps (%d decoder layers, "
            "prologue %s)", taps_installed,
            sum(1 for _ in _decoder_layers(model)),
            "yes" if embed is not None else "MISSING")
    else:
        logger.error(
            "DSV4.1 alignment: found %s (it has embed_tokens and layers) but "
            "installed zero taps -- it holds no decoder layers this "
            "recognises. Every inline tap will reach record() before "
            "set_seq_len() and raise.", type(inner).__name__)
    return handles


def _make_argument_tap(layer_id: int, positions: dict[str, int]):
    """Record the layer's own inputs, by name, from the args it was given."""

    def _tap(_module: torch.nn.Module, args: tuple) -> None:
        for key, index in positions.items():
            if index >= len(args):
                continue
            value = args[index]
            # None is the honest answer for layer 0's residual and pre_mix,
            # and it is recorded as absence rather than as a fabricated
            # identity: "layer 0 has no pre_mix" is a fact the dump has to be
            # able to show, and an identity would erase it.
            if isinstance(value, torch.Tensor):
                _record(layer_id, key, value)

    return _tap


def _prologue_pre(_module: torch.nn.Module, args: tuple) -> None:
    """Fix the sequence length, then record input_ids.

    The order matters and is the whole point of this hook: every record()
    scatters into a [seq_len, ...] buffer, so seq_len has to be known before
    the first one. Set from the request's own total length rather than from
    the number of rows in this chunk -- under chunked prefill those differ,
    and sizing the buffer by the chunk would drop every later row off the end.
    """
    input_ids = _first_tensor(args)
    if input_ids is not None:
        _record_prologue("input_ids", input_ids)


def _prologue_post(_module: torch.nn.Module, _args: tuple,
                   output: Any) -> None:
    tensor = _first_tensor(output)
    if tensor is not None:
        _record_prologue("embed.out", tensor)


def _record_prologue(key: str, tensor: torch.Tensor) -> None:
    """One prologue tensor, scattered by position like any other.

    Tagged "prologue" rather than by layer: golden keeps the model-level
    tensors in their own file, and a per-layer tag here would put them in
    layer_00 where nothing compares them.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import (
        current_row_to_pos, get_recorder)

    recorder = get_recorder()
    if recorder is None or not _ARMED:
        return
    positions = _current_positions()
    if positions is None:
        return
    row_to_pos = current_row_to_pos(positions)
    if row_to_pos is None:
        return
    recorder.record("prologue", key, tensor, row_to_pos)


def _single_request_seq_len() -> int | None:
    """The one request's total sequence length, or None if there is no batch.

    None on a profile or dummy run -- there is nothing to size a buffer with,
    and guessing would produce a dump shaped like a real one.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return None
    metadata = getattr(context, "attn_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for entry in metadata.values():
        req = getattr(entry, "req_metadata", None)
        seq_lens = getattr(req, "seq_lens", None)
        if seq_lens is None or seq_lens.numel() == 0:
            continue
        if seq_lens.numel() != 1:
            raise ValueError(
                f"alignment recording needs a single request, got "
                f"{seq_lens.numel()} sequence lengths. positions are "
                "per-request, so two requests both start at 0 and their rows "
                "scatter onto each other -- silently, since every value stays "
                "in range.")
        return int(seq_lens[0])
    return None


def _attach(module: torch.nn.Module, layer_id: int, key: str,
            side: str) -> Any:
    def _pre(_mod: torch.nn.Module, args: tuple) -> None:
        tensor = _first_tensor(args)
        if tensor is not None:
            _record(layer_id, key, tensor)

    def _post(_mod: torch.nn.Module, _args: tuple, output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is not None:
            _record(layer_id, key, tensor)

    if side == "in":
        return module.register_forward_pre_hook(_pre)
    return module.register_forward_hook(_post)


def _find_inner_model(model: torch.nn.Module) -> torch.nn.Module | None:
    """The module that actually holds ``embed_tokens`` and ``layers``.

    Searched by capability, not by path. ``model.model`` was hard-coded here
    and found nothing, because V4.1-Flash is a multimodal checkpoint and the
    real path is ``model.language_model.model`` -- one wrapper deeper.

    🔴 That wrapper has now broken three different assumptions, each one
    written for the shape of a text-only model, and each failing silently:

        the config      use_compress read hf_config where compress_ratios
                        lives under hf_text_config
        the request     mm_req_doc_ranges is ``{0: []}`` for a text-only
                        request -- a truthy dict of empty lists -- which put
                        every text prefill through the vision index path and
                        was the aicore exception's root cause
        the module tree this function, returning zero taps

    So the fix is not to hard-code one more level. A path that names
    ``language_model`` would work today and go silently empty behind the next
    wrapper, which is the same failure with a different address. Breadth-first
    so the outermost match wins, and depth-limited so a pathological tree
    cannot make model construction crawl.
    """
    queue = [(model, 0)]
    while queue:
        module, depth = queue.pop(0)
        # ``layers`` alone, not layers *and* embed_tokens. A pipeline rank
        # that is not the first has a PPMissingLayer where the embedding
        # would be, and requiring both would find nothing there and tap
        # nothing -- silently, which is the failure this function exists to
        # stop. The prologue is looked up separately and is allowed to be
        # absent for exactly that reason.
        if getattr(module, "layers", None) is not None:
            return module
        if depth < 4:
            queue.extend((child, depth + 1) for child in module.children())
    return None


def _report_nothing_found(model: torch.nn.Module) -> None:
    from vllm.logger import logger

    logger.error(
        "DSV4.1 alignment: a recorder is active but no module under %s has "
        "decoder layers, so zero taps were installed. Every "
        "inline tap will reach record() before set_seq_len() and raise. "
        "Children searched: %s", type(model).__name__,
        [name for name, _ in model.named_children()])


def _decoder_layers(model: torch.nn.Module):
    """``(layer_index, module)`` for each decoder layer, or nothing.

    Indices come from the layer's position in ``layers``, offset by
    ``start_layer`` so they are global under pipeline parallelism rather than
    local to this rank -- golden's tags are global.
    """
    inner = _find_inner_model(model) or model
    layers = getattr(inner, "layers", None)
    if layers is None:
        return
    start = int(getattr(inner, "start_layer", 0))
    for offset, layer in enumerate(layers):
        if isinstance(layer, torch.nn.Module) and type(layer).__name__ != \
                "PPMissingLayer":
            yield start + offset, layer


def expected_keys(dump_dir: str, layers: tuple[int, ...]) -> set[str]:
    """What golden will compare, from golden's own manifest.

    Read from the producer's declaration rather than discovered, because the
    two ways of discovering it have opposite blind spots and both were hit
    within a day:

        scanning the directory   picks up things that are not outputs
                                 (golden's routing_plan fixture, 8 extra keys)
        scanning by name pattern misses anything the pattern does not cover
                                 (this function's own hand-written tag list --
                                 a new protocol tag would silently drop a
                                 whole file from the comparison)

    A manifest has neither, because the side that produced the tensors is the
    side that says which ones they are. Returned as ``tag/key`` so the caller
    can difference in both directions: keys never recorded are a comparison
    that cannot run, keys recorded that golden does not compare are bytes and
    seconds spent for nothing -- and they also make the tensor count disagree,
    which is the number everyone reads first.
    """
    import json
    import os

    path = os.path.join(dump_dir, "manifest.json")
    if not os.path.exists(path):
        raise ValueError(
            f"{path} does not exist, so there is no declaration of what this "
            "dump contains. Falling back to a scan here would reintroduce "
            "exactly the blind spot the manifest removes -- and it would do "
            "so silently, since a scan always returns something.")
    declared = json.loads(open(path).read()).get("tensors")
    if not isinstance(declared, dict) or not declared:
        raise ValueError(
            f"{path} declares no 'tensors'. An empty expected set makes every "
            "coverage comparison vacuous: the difference in both directions "
            "is empty, which reads as full agreement.")

    # An exclusion rule, not an allow-list. Only the per-layer tags are
    # filtered -- by which layers were asked for -- and everything else the
    # manifest declares is kept whatever it is called. An allow-list of
    # {prologue, epilogue, layer_NN} would be this function's own version of
    # the blind spot above: a protocol that grows a new top-level tag would
    # drop it silently, and a dropped tag looks exactly like a tag with
    # nothing in it.
    keep = {f"layer_{i:02d}" for i in layers}
    wanted = {f"{tag}/{key}"
              for tag, keys in declared.items()
              if not _LAYER_TAG.fullmatch(tag) or tag in keep
              for key in keys}
    if not wanted:
        raise ValueError(
            f"{path} declares tags {sorted(declared)}; none survived selecting "
            f"layers {sorted(layers)}. Comparing against an empty set would "
            "report agreement against nothing.")
    return wanted



def golden_layers(golden_dir: str) -> tuple[int, ...]:
    """Which layers golden actually declares, from its own manifest.

    Not a configured range and not a scan of our own output. A layer that we
    failed to record entirely has no tag of ours to be discovered from, so
    deriving the layer list from what we produced would make exactly the
    omission we are looking for invisible -- the missing row cannot show
    itself in a hand-built enumeration of the rows that are present.
    """
    import json
    import os

    path = os.path.join(golden_dir, "manifest.json")
    declared = json.loads(open(path).read()).get("tensors")
    if not isinstance(declared, dict):
        raise ValueError(
            f"{path} declares no 'tensors', so there is no list of layers to "
            "compare against.")
    return tuple(sorted(int(tag.split("_")[1]) for tag in declared
                        if _LAYER_TAG.fullmatch(tag)))


def coverage_report(recorded: set[str], golden_dir: str | None) -> dict:
    """Did this run record everything golden will compare, and nothing else.

    🔴 ``AlignRecorder``'s flush already asserts position coverage -- every
    token of every recorded key written exactly once. What it cannot see is a
    key that was **never recorded at all**: there is no buffer, so there is
    no coverage vector, so there is nothing for that assertion to look at.
    A whole tensor silently absent and a whole tensor perfectly recorded
    produce the same green flush.

    Both directions are reported because they mean different things and only
    one of them is obvious:

        missing   golden compares it, we never wrote it
                  -> that comparison cannot run, and its absence looks
                     exactly like a comparison that passed
        extra     we wrote it, golden does not compare it
                  -> bytes and seconds for nothing, and it moves the tensor
                     count, which is the number everyone reads first

    ⚠️ Three states. Without a golden directory this is **not checked**, and
    that is returned as its own state rather than as two empty differences --
    empty differences read as full agreement, which is the one reading that
    must not be available to something that did not run.
    """
    if not golden_dir:
        return {
            "state": "not_checked",
            "reason": ("VLLM_ASCEND_DSV41_GOLDEN_DIR is not set, so there is "
                       "no declaration of what this dump should contain. "
                       "This is not a pass."),
            "recorded_count": len(recorded),
        }
    try:
        wanted = expected_keys(golden_dir, golden_layers(golden_dir))
    except Exception as exc:                       # noqa: BLE001
        return {
            "state": "not_checked",
            "reason": (f"golden's manifest could not be read ({exc!r}), so "
                       "the expected set is unknown. This is not a pass."),
            "recorded_count": len(recorded),
        }

    missing = sorted(wanted - recorded)
    extra = sorted(recorded - wanted)
    return {
        "state": "ok" if not missing else "incomplete",
        "missing": missing,
        "extra": extra,
        "expected_count": len(wanted),
        "recorded_count": len(recorded),
    }


def alignment_active() -> bool:
    """Whether anything should be recorded at all.

    Called at each inline tap site; the tap function itself is not entered
    when this is False, which is the same guarantee the hooks give
    structurally. Kept as one call rather than an attribute lookup so the tap
    sites read the same way the hook installer does.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    return get_recorder() is not None


def layer_index(prefix: str, hf_text_config) -> int:
    """The layer number golden tags with, out of the module prefix.

    Uses the project's own extractor rather than a split on ".", because MTP
    layers keep their original namespace and are addressed after the main
    layers -- a naive parse tags them as 0..n and silently overwrites the
    real layer 0.
    """
    from vllm_ascend.utils import extract_dsv4_layer_index

    return extract_dsv4_layer_index(hf_text_config, prefix)


def tap(layer_id: int, key: str, tensor: torch.Tensor,
        positions: torch.Tensor) -> None:
    """One position-sharded tensor, from inside a forward that owns it.

    For values that are not module boundaries -- ``sa_in.*``, ``hc_*``,
    ``moe_gate.*``, ``pre_mix.*``. A hook cannot see these: they never cross a
    module edge, and hooking the module whose name they carry is how you get a
    tap that never fires (see this module's docstring on attn_norm).
    """
    _record_with(layer_id, key, tensor, positions)


def tap_whole(layer_id: int, key: str, tensor: torch.Tensor) -> None:
    """One tensor that is not sharded by position.

    ``sa_in.attn_sink`` is per head and ``sa_in.softmax_scale`` is a scalar --
    neither has a token axis, so scattering them by row would be meaningless
    and the coverage assertion would never be satisfied.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    recorder = get_recorder()
    if recorder is None:
        return
    recorder.record_whole(f"layer_{layer_id:02d}", key, tensor)


def _record_with(layer_id: int, key: str, tensor: torch.Tensor,
                 positions: torch.Tensor) -> None:
    from vllm_ascend.models.deepseek_v4_1.debug_align import (
        current_row_to_pos, get_recorder)

    recorder = get_recorder()
    if recorder is None:
        return
    row_to_pos = current_row_to_pos(positions)
    if row_to_pos is None:
        return
    recorder.record_layer(layer_id, key, tensor, row_to_pos)
