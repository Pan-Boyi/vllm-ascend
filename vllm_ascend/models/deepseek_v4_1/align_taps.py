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
    if recorder is None:
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
    inner = getattr(model, "model", model)
    # 🔴 First, and on the model rather than on a layer. The layer-level
    # capture below runs only once the first decoder layer starts, which is
    # *after* embed_tokens -- so a prologue tap reading positions would get
    # either None or, worse, the previous step's, which is in range and wrong.
    # This also clears the value, so a forward that never reaches the capture
    # cannot leave the last one visible to the next.
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
            "DSV4.1 alignment: a recorder is active but zero taps were "
            "installed -- no embed_tokens and no decoder layers were found "
            "under %s. Every inline tap will reach record() before "
            "set_seq_len() and raise.", type(model).__name__)
    return handles


def _prologue_pre(_module: torch.nn.Module, args: tuple) -> None:
    """Fix the sequence length, then record input_ids.

    The order matters and is the whole point of this hook: every record()
    scatters into a [seq_len, ...] buffer, so seq_len has to be known before
    the first one. Set from the request's own total length rather than from
    the number of rows in this chunk -- under chunked prefill those differ,
    and sizing the buffer by the chunk would drop every later row off the end.
    """
    seq_len = _single_request_seq_len()
    if seq_len is None:
        return

    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    recorder = get_recorder()
    if recorder is None:
        return
    recorder.set_seq_len(seq_len)

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
    if recorder is None:
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


def _decoder_layers(model: torch.nn.Module):
    """``(layer_index, module)`` for each decoder layer, or nothing.

    Indices come from the layer's position in ``model.layers``, offset by
    ``start_layer`` so they are global under pipeline parallelism rather than
    local to this rank -- golden's tags are global.
    """
    inner = getattr(model, "model", model)
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
