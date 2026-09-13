# SPDX-License-Identifier: Apache-2.0
"""Where two identical requests stop agreeing.

Twelve identical requests to one server produced nine different completions.
Every hypothesis about *why* -- which rank served it, how many requests came
before, which buffer was not reset -- has been tested against the **final
text**, which is forty layers and a sampler away from wherever the divergence
starts. That distance is the problem: a cause at layer 2 and a cause at layer
39 produce the same observable, so each round of guessing costs a flight.

This turns that into a bisection. One fingerprint per decoder layer per
forward, in the log, comparable across requests:

    DSV4.1 trace step=3 layer=00 out=0x9f3a... coarse=0x71c2... rows=519
    DSV4.1 trace step=3 layer=02 out=0x11be... coarse=0x71c2... rows=519

Two requests with the same prompt should produce identical lines. The first
layer where they differ is the first layer that read something this request
did not write -- and everything before it is exonerated.

🔴 The fingerprint is **position-weighted and bitwise**, not a summary
statistic. A sum, a mean or a norm collapses different tensors onto the same
number often enough to matter: fp32 and bf16 versions of one computation have
matched on 67% of elements here while differing in mechanism, and a reduction
that cannot see a permutation cannot see a misaligned read either. So the
bytes are read as integers and folded against their own index.

⚠️ Default off, and off means no hooks registered -- there is nothing to
enter, not merely nothing to do. Three launches were lost to instrumentation
that was on; the rule this line now runs on is that anything opened for
observation must first have a reading with it closed.
"""

from __future__ import annotations

import torch

_ENV = "VLLM_ASCEND_DSV41_TRACE_LAYERS"

#: Bumped once per model forward, so lines from different steps cannot be
#: mistaken for each other when a comparison is done by grep.
_STEP = 0


def trace_spec() -> str:
    """The raw selection, or "" when tracing is off."""
    from vllm_ascend import envs

    return str(getattr(envs, _ENV, "") or "").strip()


def selected_layers(spec: str, available: tuple[int, ...]) -> tuple[int, ...]:
    """Which layers this spec asks for, out of the ones that exist.

    ``"all"`` is every layer; otherwise a comma list of indices. A layer named
    that does not exist is an error rather than a silent omission: asking for
    layer 44 and getting a quiet 40-layer trace is how a comparison ends up
    missing the only layer someone cared about.
    """
    if not spec:
        return ()
    if spec.lower() == "all":
        return tuple(available)
    wanted = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            index = int(piece)
        except ValueError:
            raise ValueError(
                f"{_ENV}={spec!r}: {piece!r} is not a layer index or 'all'. "
                "Refusing rather than skipping it -- a spec that is partly "
                "ignored produces a trace that looks complete.") from None
        if index not in available:
            raise ValueError(
                f"{_ENV}={spec!r} asks for layer {index}, which this rank does "
                f"not hold (it has {available[0]}..{available[-1]}). A quiet "
                "omission here reads as 'that layer never diverged'.")
        wanted.append(index)
    return tuple(dict.fromkeys(wanted))


def fingerprint(tensor: torch.Tensor) -> int:
    """A position-sensitive, bit-exact digest of ``tensor``'s contents.

    Reads the raw bytes as integers and folds each against its own index, so
    two tensors holding the same multiset of values in a different order do
    not collide -- which a sum, a mean or a norm all do. Bit-exact rather than
    tolerant on purpose: the question is "is this the same computation", not
    "is this close".

    ``-1`` when the tensor cannot be digested (an unsupported dtype, a meta
    tensor). That is a third state and it prints as one; a fingerprint that
    silently becomes a constant is a check bit with no discriminating power,
    which reads as agreement everywhere.
    """
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return 0
        width = flat.element_size()
        as_int = {1: torch.int8, 2: torch.int16,
                  4: torch.int32, 8: torch.int64}.get(width)
        if as_int is None:
            return -1
        raw = flat.view(as_int).to(torch.int64)
        index = torch.arange(raw.numel(), device=raw.device,
                             dtype=torch.int64)
        # ⚠️ The first version XOR-ed the index in and multiplied by one
        # constant. It collided on the very first permutation tested --
        # [1,2,3,4] and [4,3,2,1] -- because that form is linear in the sum
        # once the constant factors out, which is the property it was
        # supposed not to have. Weighting each element by a *distinct odd
        # multiplier* and folding the high bits back down does not factor
        # out. The test that caught it is the one that compares a tensor with
        # its own reverse.
        mixed = (raw ^ 0x9E3779B97F4A7C15) * (index * 2 + 1)
        return int((mixed ^ (mixed >> 29)).sum())
    except Exception:                                        # noqa: BLE001
        return -1


#: Quantisation levels for the coarse digest. 1/64 is about 1.6e-2 relative,
#: which is comfortably above bf16's ULP of 3.9e-3 -- a threshold at or below
#: the representation floor cannot separate "the same computation, reassociated"
#: from "a different computation".
COARSE_LEVELS = 64


def coarse_fingerprint(tensor: torch.Tensor) -> int:
    """A digest that survives reassociation but not a real difference.

    🔴 The exact fingerprint answers "are these the same bits", which is the
    right question for a stale buffer and the wrong one for a sum added up in
    a different order. Both print as "different", and the two have opposite
    remedies: one is a memory bug, the other is a non-deterministic reduction
    whose 1-ULP wobble is flipping an argmax between near-tied tokens.

    So the trace carries both. Read together:

        exact differs, coarse same   the computation is the same and the
                                     arithmetic was reassociated
        both differ                  something structural changed

    Scaled by the tensor's own maximum before quantising, so the grid is
    relative and one outlier element does not decide the whole digest.
    """
    try:
        flat = tensor.detach().reshape(-1).to(torch.float32)
        if flat.numel() == 0:
            return 0
        scale = flat.abs().max()
        if not bool(torch.isfinite(scale)) or float(scale) == 0.0:
            return 0
        quantised = (flat / scale * COARSE_LEVELS).round().to(torch.int64)
        index = torch.arange(quantised.numel(), device=quantised.device,
                             dtype=torch.int64)
        mixed = (quantised ^ 0x9E3779B97F4A7C15) * (index * 2 + 1)
        return int((mixed ^ (mixed >> 29)).sum())
    except Exception:                                        # noqa: BLE001
        return -1


def install_layer_trace(model: torch.nn.Module) -> list:
    """Register the per-layer fingerprint hooks, or none at all.

    Returns the handles, so an empty list is the provable off state.
    """
    from vllm_ascend.models.deepseek_v4_1.align_taps import (_decoder_layers,
                                                             _find_inner_model)

    spec = trace_spec()
    if not spec:
        return []

    layers = dict(_decoder_layers(model))
    if not layers:
        raise RuntimeError(
            f"{_ENV}={spec!r} but no decoder layers were found on this model, "
            "so the trace would be empty -- which looks exactly like a run "
            "where nothing diverged.")
    wanted = set(selected_layers(spec, tuple(sorted(layers))))

    handles = []
    inner = _find_inner_model(model)
    if inner is not None:
        handles.append(inner.register_forward_pre_hook(_bump_step))
    for index, layer in layers.items():
        if index in wanted:
            handles.append(layer.register_forward_hook(_make_hook(index)))

    from vllm.logger import logger

    logger.info(
        "DSV4.1 trace: fingerprinting %d of %d decoder layers (%s). Two "
        "requests with the same prompt should produce identical lines; the "
        "first layer that differs is the first that read something the "
        "request did not write.", len(wanted), len(layers), spec)
    return handles


def _bump_step(_module: torch.nn.Module, _args: tuple) -> None:
    global _STEP
    _STEP += 1


def _make_hook(index: int):
    def _hook(_module: torch.nn.Module, _args: tuple, output) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            return
        from vllm.logger import logger

        logger.info(
            "DSV4.1 trace step=%d layer=%02d out=0x%016x coarse=0x%016x "
            "rows=%d", _STEP, index,
            fingerprint(tensor) & 0xFFFFFFFFFFFFFFFF,
            coarse_fingerprint(tensor) & 0xFFFFFFFFFFFFFFFF,
            tensor.shape[0] if tensor.ndim else 0)

    return _hook
