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

from typing import Any

import torch

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
    for layer_id, layer in _decoder_layers(model):
        # First, so the boundary hooks below it in the same layer see this
        # layer's positions rather than the previous layer's.
        handles.append(layer.register_forward_pre_hook(_capture_positions))
        for attribute, key, side in _BOUNDARY_TAPS:
            module = getattr(layer, attribute, None)
            if module is None:
                continue
            handles.append(_attach(module, layer_id, key, side))
    return handles


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
    """What golden will compare, evaluated from golden's own dump.

    Taken from the reference dump rather than from a list written here: a
    hand-written list cannot show the row it is missing, and this one is the
    same set the comparison will use. Returned as ``tag/key`` pairs so the
    caller can take the difference in both directions -- keys we never record
    are a comparison that cannot run, and keys we record that golden does not
    compare are bytes and seconds spent for nothing.
    """
    import os

    from safetensors import safe_open

    wanted = set()
    for tag in [f"layer_{i:02d}" for i in layers] + ["prologue", "epilogue"]:
        path = os.path.join(dump_dir, f"{tag}.safetensors")
        if not os.path.exists(path):
            continue
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                wanted.add(f"{tag}/{key}")
    if not wanted:
        raise ValueError(
            f"no reference tensors found under {dump_dir}. An empty expected "
            "set makes every coverage comparison vacuous -- it would report "
            "full agreement against nothing.")
    return wanted
