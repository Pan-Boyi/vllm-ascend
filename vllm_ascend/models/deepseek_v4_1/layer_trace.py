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

#: The mixing constant, written as the **signed** value with the same 64 bits
#: as the golden-ratio constant 0x9E3779B97F4A7C15.
#:
#: 🔴 Spelled unsigned it is 11400714819323198485, larger than int64's maximum
#: of 9223372036854775807. CPU torch accepts it; the NPU does not --
#: ``aclnnBitwiseXorScalar`` fails -- and the failure landed in an ``except``
#: that returned the -1 sentinel. The result was a trace whose lines were 100%
#: sentinel, and a sentinel is a *constant*, so the obvious reading of that log
#: is "every layer agrees".
#:
#: ⚠️ Twenty-four unit tests were green while the instrument was dead on the
#: target, because every one of them ran on CPU. "Can be computed on CPU" and
#: "can be computed on Ascend" are two claims, and the first passing looks
#: exactly like the second passing. The check that catches it without a device
#: is the one on DEVICE_INT_CONSTANTS: the value has to fit in int64.
MIX = 0x9E3779B97F4A7C15 - (1 << 64)

#: Every integer constant this module hands to a torch op, so one test can
#: check the property that was violated rather than this one value.
DEVICE_INT_CONSTANTS = (MIX, )




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
    value = fingerprint_tensor(tensor)
    return int(value)


def fingerprint_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """:func:`fingerprint` without bringing the answer back to the host.

    🔴 The int-returning form forces a device-to-host transfer, and the trace
    calls it once per layer. Forty of those inside one forward is forty host
    syncs that the other ranks are not waiting for, and the first launch with
    the trace on died in ``aicpu`` 507018 -- the same error code as the last
    instrument that synchronised inside the forward.

    So the digest is computed and **kept on the device**; the trace stores it
    in a preallocated slot and takes the whole buffer back once, after the
    model forward, when there is nothing left to be out of step with.
    """
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() == 0:
            return torch.zeros((), dtype=torch.int64, device=tensor.device)
        width = flat.element_size()
        as_int = {1: torch.int8, 2: torch.int16,
                  4: torch.int32, 8: torch.int64}.get(width)
        if as_int is None:
            return torch.full((), -1, dtype=torch.int64, device=tensor.device)
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
        mixed = (raw ^ MIX) * (index * 2 + 1)
        return (mixed ^ (mixed >> 29)).sum()
    except Exception as exc:                                 # noqa: BLE001
        _report_sentinel("exact", exc)
        return torch.full((), -1, dtype=torch.int64)



#: Fingerprint points before the first decoder layer, so "layer 0 already
#: differs" can be taken one step further back. Negative so they sort ahead of
#: the layers and cannot collide with a real index.
#:
#: 🔴 ``input_ids`` is here first and it is not a formality. Everything
#: downstream is being read as "identical input, different output", and that
#: premise has never been checked against the tensor the model actually
#: received -- only against the prompt string that was sent. A chat template,
#: a timestamp or a differing BOS would make the whole investigation a search
#: for a defect that is not there.
PRE_LAYER_POINTS = {-3: "input_ids", -2: "positions", -1: "embed_out"}


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
    return int(coarse_fingerprint_tensor(tensor))


def coarse_fingerprint_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """:func:`coarse_fingerprint`, kept on the device. See
    :func:`fingerprint_tensor` for why that matters."""
    try:
        flat = tensor.detach().reshape(-1).to(torch.float32)
        zero = torch.zeros((), dtype=torch.int64, device=flat.device)
        if flat.numel() == 0:
            return zero
        scale = flat.abs().max()
        # ⚠️ Branchless on purpose: ``if float(scale) == 0`` would be a host
        # read, which is the thing this form exists to avoid. A zero or
        # non-finite scale yields a zero digest through arithmetic instead.
        safe = torch.where(torch.isfinite(scale) & (scale != 0), scale,
                           torch.ones_like(scale))
        quantised = (flat / safe * COARSE_LEVELS).round().to(torch.int64)
        index = torch.arange(quantised.numel(), device=quantised.device,
                             dtype=torch.int64)
        mixed = (quantised ^ MIX) * (index * 2 + 1)
        digest = (mixed ^ (mixed >> 29)).sum()
        return torch.where(torch.isfinite(scale) & (scale != 0), digest, zero)
    except Exception as exc:                                 # noqa: BLE001
        _report_sentinel("coarse", exc)
        return torch.full((), -1, dtype=torch.int64)


#: The last reason a digest could not be computed, so the refusal below can
#: quote it. Without this the raise says "all sentinel" and the caller has to
#: go find the one log line that said why.
_LAST_SENTINEL_REASON: str = ""

#: Sentinel proportion above which the trace is reported as unusable rather
#: than merely incomplete. Matches the threshold integration's analyser uses,
#: so the two sides do not disagree about what counts as a reading.
SENTINEL_LIMIT = 0.05


def _report_sentinel(which: str, exc: BaseException) -> None:
    """Say why a digest could not be computed, the first time it happens.

    🔴 The sentinel is a *constant*, so a trace made entirely of sentinels
    reads as "every layer agrees" -- the strongest possible wrong conclusion,
    and it is produced by the failure being silent. The only line that could
    have said otherwise was the one the except clause was swallowing.
    """
    from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

    global _LAST_SENTINEL_REASON
    _LAST_SENTINEL_REASON = f"{which}: {exc!r}"
    announce_once(
        "trace_sentinel_" + which,
        "the %s layer digest could not be computed (%r). Every line from here "
        "carries the -1 sentinel, which is a constant and therefore reads as "
        "'all layers agree'. Treat this trace as no reading at all.",
        which, exc)


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

    ordered = tuple(sorted(PRE_LAYER_POINTS)) + tuple(sorted(wanted))
    _SLOTS.prepare(ordered)

    handles = []
    inner = _find_inner_model(model)
    if inner is None:
        raise RuntimeError(
            f"{_ENV}={spec!r} but the module holding the decoder layers was "
            "not found, so there is nowhere to drain the digests from. The "
            "trace would collect forty values and print none of them, which "
            "reads as a forward that never ran.")
    handles.append(inner.register_forward_pre_hook(_bump_step))
    handles.append(inner.register_forward_pre_hook(_digest_model_inputs))
    handles.append(inner.register_forward_hook(_flush))
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and type(embed).__name__ != "PPMissingLayer":
        handles.append(embed.register_forward_hook(_digest_embedding))
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


class _Slots:
    """Where the per-layer digests wait until the forward is over.

    🔴 One device buffer, one transfer. The first version read each digest
    back with ``int(...)`` inside the layer hook -- forty host syncs inside a
    forward that other ranks are not waiting at -- and the launch died in
    aicpu 507018, the same error code as the previous instrument that
    synchronised inside the forward. An instrument that stalls the thing it
    measures is not measuring it.

    Filled by index-assignment on the device, which queues like any other
    kernel, and drained once from the model-level hook where the forward has
    already finished.
    """

    def __init__(self):
        self.layers: tuple[int, ...] = ()
        self.buffer: torch.Tensor | None = None
        self.rows = [0] * 0
        self.written: list[int] = []

    def prepare(self, layers: tuple[int, ...]) -> None:
        self.layers = layers
        self.buffer = None
        self.rows = [0] * len(layers)
        self.written: list[int] = []

    def slot_of(self, layer: int) -> int:
        return self.layers.index(layer)

    def store(self, slot: int, exact: torch.Tensor, coarse: torch.Tensor,
              rows: int) -> None:
        if self.buffer is None:
            self.buffer = torch.full((len(self.layers), 2), -1,
                                     dtype=torch.int64, device=exact.device)
        self.buffer[slot, 0] = exact
        self.buffer[slot, 1] = coarse
        self.rows[slot] = rows
        self.written.append(self.layers[slot])

    def drain(self) -> list[tuple[int, int, int, int]]:
        """``(layer, exact, coarse, rows)`` per slot, in one transfer.

        Slots nothing wrote this forward are dropped rather than reported as
        sentinels: a decode step does not run the prologue, and calling that
        "could not be digested" would make every decode step look like a dead
        instrument.
        """
        if self.buffer is None:
            return []
        values = self.buffer.to("cpu").tolist()
        written = set(self.written)
        self.buffer = None
        self.written = []
        return [(layer, values[i][0], values[i][1], self.rows[i])
                for i, layer in enumerate(self.layers) if layer in written]


_SLOTS = _Slots()


def _bump_step(_module: torch.nn.Module, _args: tuple) -> None:
    global _STEP
    _STEP += 1


def _flush(_module: torch.nn.Module, _args: tuple, _output) -> None:
    """One transfer and all the lines, after the forward.

    Here rather than per layer: at this point the forward is over, so a
    transfer costs a wait that nothing else is racing.
    """
    from vllm.logger import logger

    drained = _SLOTS.drain()
    if not drained:
        return
    sentinels = sum(1 for _, exact, coarse, _ in drained
                    if exact == -1 or coarse == -1)
    for layer, exact, coarse, rows in drained:
        name = PRE_LAYER_POINTS.get(layer, "%02d" % layer)
        logger.info(
            "DSV4.1 trace step=%d layer=%s out=0x%016x coarse=0x%016x "
            "rows=%d", _STEP, name, exact & 0xFFFFFFFFFFFFFFFF,
            coarse & 0xFFFFFFFFFFFFFFFF, rows)
    _refuse_a_trace_that_is_not_a_reading(sentinels, len(drained))


def _refuse_a_trace_that_is_not_a_reading(sentinels: int, total: int) -> None:
    """Stop, loudly, when the trace is sentinels rather than digests.

    🔴 The failure this exists for has already happened: a whole run's trace
    was 100% sentinel, and because the sentinel is a *constant*, the lines all
    matched and the log read as "every layer agrees" -- the strongest possible
    wrong conclusion, produced by a dead instrument rather than by the model.

    Raising is safe here in a way it would not normally be: nothing registers
    these hooks unless the trace was explicitly asked for, so this can only
    stop a run that was started to collect a trace. It fires on the first
    forward, which is warmup, so the cost is the cheapest one available.

    ⚠️ Three states, not two. Some sentinels is not the same as all of them:
    a minority means particular layers could not be digested and the rest of
    the reading stands; past SENTINEL_LIMIT the reading does not.
    """
    if not sentinels:
        return
    from vllm.logger import logger

    share = sentinels / total
    if share <= SENTINEL_LIMIT:
        logger.error(
            "DSV4.1 trace: %d of %d layers came back as the -1 sentinel (%s). "
            "Those lines are not digests; the rest of the reading stands.",
            sentinels, total, _LAST_SENTINEL_REASON or "reason not recorded")
        return
    raise RuntimeError(
        f"DSV4.1 trace: {sentinels} of {total} layers came back as the -1 "
        f"sentinel ({_LAST_SENTINEL_REASON or 'reason not recorded'}). The "
        "sentinel is a constant, so these lines all match each other and the "
        "log reads as 'every layer agrees' -- which is the opposite of a "
        "finding. Refusing to produce that trace: it would be spent as "
        "evidence, and it is not a reading at all.")


def _digest_model_inputs(_module: torch.nn.Module, args: tuple) -> None:
    """The tensors the model was actually handed.

    🔴 Not a formality. Everything downstream is read as "identical input,
    different output", and that premise has only ever been checked against the
    prompt *string* that was sent -- never against the tensor the model
    received. A chat template, a timestamp or a differing BOS would make the
    whole investigation a search for a defect that is not there, and nothing
    else in the trace can tell that apart from a real divergence at layer 0.

    Positional, because the inner model's signature is
    ``forward(input_ids, positions, ...)`` and both are what they are called.
    Anything that is not a tensor is skipped rather than guessed at.
    """
    for label, value in zip((-3, -2), args[:2]):
        if isinstance(value, torch.Tensor):
            _store(label, value)


def _digest_embedding(_module: torch.nn.Module, _args: tuple, output) -> None:
    """What the first decoder layer receives.

    Splits "layer 0's output differs" into its two possible causes: the input
    it was given, or what it did with it.
    """
    tensor = output[0] if isinstance(output, tuple) else output
    if isinstance(tensor, torch.Tensor):
        _store(-1, tensor)


def _store(label: int, tensor: torch.Tensor) -> None:
    _SLOTS.store(_SLOTS.slot_of(label), fingerprint_tensor(tensor),
                 coarse_fingerprint_tensor(tensor),
                 tensor.shape[0] if tensor.ndim else 0)


def _make_hook(index: int):
    slot = _SLOTS.slot_of(index)

    def _hook(_module: torch.nn.Module, _args: tuple, output) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            return
        _SLOTS.store(slot, fingerprint_tensor(tensor),
                     coarse_fingerprint_tensor(tensor),
                     tensor.shape[0] if tensor.ndim else 0)

    return _hook
