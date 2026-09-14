# SPDX-License-Identifier: Apache-2.0
"""A layout change moves bytes; it must not change them.

``process_weights_after_loading`` reshapes and transposes the MXFP4 block
scales. Every one of those steps is a *layout* change, so the multiset of
scale bytes has to come out exactly as it went in. If it does not, something
in there is editing values while claiming to rearrange them -- and a wrong
scale is not visible downstream: the block structure is cut along the
reduction dimension and the GEMM's sum flattens it, so the output carries no
trace of which block scale was applied.

🔵 The reason this check is worth its weight: it needs nothing. No flight, no
reference implementation, no operator experiment -- just one function's input
and its output. When the question is "did we hand the operator the right
scale", this answers the half that can be answered on the ground.

⚠️ Padding is subtracted explicitly rather than by loosening the comparison.
The linear path pads an odd ``k_dim`` with a column of zeros, which adds real
bytes to the multiset; the count and the value are named by the caller and
reported, because "the histogram differs by N zeros because we padded" and
"the histogram differs" must not produce the same output.
"""

from __future__ import annotations

import torch

#: Scale bytes are E8M0 exponents held as uint8, so 256 buckets covers them.
_BUCKETS = 256


def _readable(data: torch.Tensor,
              view_as: torch.dtype | None = None) -> tuple[torch.Tensor, str]:
    """A host-side flat copy of these bytes, and how it was obtained.

    ``view_as`` re-interprets the dtype, and is applied *after* whichever
    materialisation path succeeded -- see the comment on the cast below for
    why the order is not free.

    🔴 The **whole** chain is inside the try, deliberately. Measured on the
    device (card 4, L0): ``reshape(-1)`` *succeeds* on a FRACTAL_NZ tensor --
    the step that actually fails is the host copy::

        RuntimeError: Current device only support aclnn operator, and current
        operator ...::copy_ do not support internal format

    The first version guarded only the reshape, so on the real failure it
    returned happily, labelled the reading ``"direct"``, and raised from
    outside itself one line later. A guard whose trigger point is not the
    failure point is not a guard: it was dead code on the only case it was
    written for, and it was lying about the route as well.

    ⚠️ Measured, not assumed: the ND fallback does not change the bytes.
    ND -> FRACTAL_NZ -> ND round trips came back byte-identical, 0/10240
    differing, for int8, float16 and bfloat16 at [64, 160] on this card.
    SCOPE: that shape, those dtypes, this card, this PTA build -- it is a
    measurement of the route we take, not a general property of
    ``npu_format_cast``. FRACTAL_Z is not castable here at all (361001), so
    nothing is claimed about it.
    """
    def _finish(t):
        return (t if view_as is None else t.view(view_as)).reshape(
            -1).to(torch.int64).cpu()

    try:
        return _finish(data.detach()), "direct"
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import torch_npu

        # 🔴 Cast first, re-dtype second. The other order fails on the
        # device: an internal-format tensor viewed as a narrower dtype has a
        # format descriptor that no longer matches its storage, and the cast
        # then refuses it -- "Parameter c0 of aclnnNpuFormatCast has
        # incorrect value 16. It should be 32", because NZ's inner block is
        # counted in elements. Measured at L0 on bf16 and fp16.
        nd = torch_npu.npu_format_cast(data.detach(), 2)  # 2 = ACL_FORMAT_ND
        return _finish(nd), "via npu_format_cast to ND"
    except Exception as exc:                                 # noqa: BLE001
        raise RuntimeError(
            f"the tensor is in a format this host cannot read and the ND cast "
            f"also failed ({exc!r})") from exc


def _storage_format(data: torch.Tensor) -> str:
    """Ascend's private storage format for this tensor, named if it can be.

    ⚠️ Reported beside every reading because it is the thing that decides
    *which* reading was possible. "shape=[512,160] dtype=uint8" is identical
    text for a tensor this host can read directly and for one it can only
    read after a cast, and those are two different claims.
    """
    try:
        import torch_npu

        fmt = int(torch_npu.get_npu_format(data))
    except Exception:                                        # noqa: BLE001
        return "host/ND"
    named = {2: "ND", 3: "NC1HWC0", 4: "FRACTAL_Z", 29: "FRACTAL_NZ"}
    return named.get(fmt, f"acl_format={fmt}")


class _Reading(tuple):
    """Bytes, plus how they were obtained and what they were stored as.

    🔴 The three travel together on purpose. The invariant below compares a
    reading taken before the transform with one taken after, and if those two
    were obtained by different routes the comparison is no longer about the
    transform -- it is about the routes. An instrument that differs between
    the two arms is a variable in the experiment, not a measurement of it.
    """

    __slots__ = ()

    def __new__(cls, data, how, fmt):
        return super().__new__(cls, (data, how, fmt))

    bytes = property(lambda self: self[0])
    how = property(lambda self: self[1])
    fmt = property(lambda self: self[2])


def _as_bytes(data: torch.Tensor) -> _Reading:
    """The raw bytes of a scale tensor, whatever dtype it is wearing.

    🔴 ``.to(torch.int64)`` would be wrong for the dtype this actually
    arrives in. The DS scheme declares ``w13_weight_scale`` as
    ``float8_e8m0fnu`` and views it as uint8 inside the transform
    (w4a8_mxfp4.py:329-334, :357-361), so a conversion would read the
    *decoded exponent* on one side and the *stored byte* on the other, and
    the comparison would be between two different quantities -- the exact
    mistake this module exists to catch, made by the module itself.
    """
    fmt = _storage_format(data)
    view_as = None
    if data.dtype not in (torch.int8, torch.uint8, torch.int32, torch.int64):
        view_as = torch.uint8
    flat, how = _readable(data, view_as=view_as)
    return _Reading(flat, how, fmt)


class ScaleLayoutRefusal(ValueError):
    """A deliberate refusal by the criterion, not a failure of the instrument.

    🔴 These are opposite things and the guard below has to tell them apart.
    Without a distinct type, the net that stops a broken *reading* from
    killing a launch also swallows the criterion's own verdict -- a guard
    sharing the failure mode of the thing it guards, which is how the first
    version of this behaved: armed, and silently unable to refuse anything.
    """


def _never_fails_the_launch(fn):
    """An instrument that can stop a launch is not an instrument.

    🔴 Learned here at full price: the byte-reading line raised inside weight
    loading, all four workers failed to start, and a whole-network flight was
    lost -- for eight printed bytes. Everything in this module is read-only
    and optional; nothing in it has any business propagating.

    Says so once per entry point rather than going quiet: a dump missing the
    scale report and a dump where the scale report found nothing to say are
    different situations.
    """
    import functools

    @functools.wraps(fn)
    def _guarded(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ScaleLayoutRefusal:
            # The criterion speaking, not the instrument breaking. Straight
            # through: swallowing this would leave it armed and mute.
            raise
        except Exception as exc:                             # noqa: BLE001
            if fn.__name__ not in _FAILED:
                _FAILED.add(fn.__name__)
                from vllm.logger import logger

                logger.warning(
                    "DSV4.1 scale instrument %s could not read the scales "
                    "(%r) and was skipped. The run continues: this is an "
                    "observation, and it has already cost one flight to let "
                    "one of these propagate.", fn.__name__, exc)
            return None

    return _guarded


_FAILED: set = set()


@_never_fails_the_launch
def snapshot_scales(layer, names: tuple[str, ...]) -> dict:
    """The bytes of each named scale, before any layout change."""
    out = {}
    for name in names:
        tensor = getattr(layer, name, None)
        data = getattr(tensor, "data", tensor)
        if isinstance(data, torch.Tensor):
            out[name] = _as_bytes(data)
    return out


def _histogram(flat: torch.Tensor) -> torch.Tensor:
    return torch.bincount(flat.clamp(min=0, max=_BUCKETS - 1),
                          minlength=_BUCKETS)


@_never_fails_the_launch
def verify_scale_layout_preserved(before: dict, layer,
                                  names: tuple[str, ...],
                                  padded: dict | None = None,
                                  expanded: dict | None = None,
                                  what: str = "") -> None:
    """Raise if a layout change altered any scale's byte multiset.

    ``padded`` maps a name to ``(count, value)`` -- elements the transform
    added and what they hold.

    ``expanded`` maps a name to an integer factor, for a step that *copies*
    each element a fixed number of times. The real path does this: the tile
    scales are broadcast from ``[out/block, in/block]`` to ``[out, in/group]``
    with ``repeat_interleave`` (w8a8_mxfp8.py:548-550). A copy multiplies
    every bucket by the same factor and introduces no new byte value, so the
    invariant stays exact rather than being relaxed into a tolerance --
    which matters, because the failure it has to catch is precisely a
    "broadcast" that averages or requantises instead of copying, and that
    shows up as **new byte values**, not as a different count.
    """
    if not before:
        return
    padded = padded or {}
    expanded = expanded or {}
    problems = []
    problems_undecidable = []
    for name in names:
        was = before.get(name)
        if was is None:
            continue
        tensor = getattr(layer, name, None)
        data = getattr(tensor, "data", tensor)
        if not isinstance(data, torch.Tensor):
            continue
        now = _as_bytes(data)
        pad_count, pad_value = padded.get(name, (0, 0))
        factor = int(expanded.get(name, 1))

        if was.how != now.how:
            # 🔴 Not a verdict either way. The two arms were read by
            # different routes, so any difference between them is equally
            # explained by the routes -- and npu_format_cast is documented
            # to move bytes, not only to relabel them. Saying "OK" here
            # would be the stronger of the two wrong answers.
            problems_undecidable.append(
                f"{name}: read {was.how} (stored {was.fmt}) before and "
                f"{now.how} (stored {now.fmt}) after. The instrument is not "
                "the same in the two arms, so this pair cannot be graded; "
                "the invariant is skipped for this name, not passed.")
            continue

        reading_how, fmt_before, fmt_after = was.how, was.fmt, now.fmt
        was, now = was.bytes, now.bytes
        if now.numel() != was.numel() * factor + pad_count:
            problems.append(
                f"{name}: {was.numel()} elements in, {now.numel()} out, "
                f"with a declared copy factor of {factor} and {pad_count} "
                "declared as padding -- the counts do not reconcile, so the "
                "step is not the layout change it is documented as")
            continue

        want = _histogram(was) * factor
        if pad_count:
            want = want.clone()
            want[min(max(int(pad_value), 0), _BUCKETS - 1)] += pad_count
        got = _histogram(now)
        if not torch.equal(want, got):
            differing = (want != got).nonzero().flatten().tolist()
            new_values = [v for v in differing if int(want[v]) == 0]
            problems.append(
                f"{name}: the byte histogram changed at values {differing[:8]} "
                f"(in x{factor}: {[int(want[v]) for v in differing[:8]]}, out: "
                f"{[int(got[v]) for v in differing[:8]]}). A reshape, a "
                "transpose or a repeat cannot do that."
                + (f" ⚠️ Values {new_values[:8]} are NEW -- they were not in "
                   "the input at all, which is what an averaging or "
                   "requantising 'broadcast' produces." if new_values else ""))
        else:
            _report_ok(what or name, was, now, pad_count, pad_value, factor,
                       how=reading_how, fmt_before=fmt_before,
                       fmt_after=fmt_after)

    if problems_undecidable:
        from vllm.logger import logger

        logger.warning(
            "DSV4.1 MXFP4 scale layout NOT CHECKED for %d name(s):\n  %s",
            len(problems_undecidable), "\n  ".join(problems_undecidable))

    if problems:
        complaint = ("MXFP4 scale layout changed values, not just positions:"
                     "\n  " + "\n  ".join(problems))
        if strict_scale_layout():
            raise ScaleLayoutRefusal(complaint)
        from vllm.logger import logger

        logger.error(
            "%s\n⚠️ NOT raising: this criterion has not yet run clean on the "
            "device, and the last instrument here that could stop a launch "
            "did. Set VLLM_ASCEND_DSV41_SCALE_LAYOUT_STRICT=1 to make it a "
            "refusal once a flight has shown it passing.", complaint)


def strict_scale_layout() -> bool:
    """Whether a changed scale multiset refuses the run.

    Default off, with a trigger rather than a date: arm it once a flight has
    logged the OK line from the device.
    """
    import os

    return os.environ.get("VLLM_ASCEND_DSV41_SCALE_LAYOUT_STRICT", "0") == "1"


_REPORTED: set = set()


def _report_ok(what: str, was: torch.Tensor, now: torch.Tensor,
               pad_count: int, pad_value: int, factor: int = 1, *,
               how: str = "direct", fmt_before: str = "?",
               fmt_after: str = "?") -> None:
    """Say what was checked, once per name, with the numbers.

    🔴 Including the byte range, because the *other* half of this question is
    whether the bytes are the ones the checkpoint holds -- measured there as
    119..122 for w1/w3 and 119..124 for w2. This check proves they were not
    changed here; printing them is what lets a reader see whether they were
    right to begin with, which no invariant on this function can decide.
    """
    if what in _REPORTED:
        return
    _REPORTED.add(what)
    from vllm.logger import logger

    values = was.to(torch.float32)
    logger.info(
        "DSV4.1 MXFP4 scale layout OK for %s: %d bytes preserved exactly"
        "%s. Byte range min=%d median=%d max=%d. Read %s, stored %s -> %s. "
        "⚠️ This says the layout step did not change them; whether they were "
        "the right bytes on arrival is a different question, and the "
        "checkpoint's are 119..124.",
        what, was.numel(),
        (f" (x{factor} by declared copy)" if factor != 1 else "")
        + (f" (+{pad_count} declared padding of value {pad_value})"
           if pad_count else ""),
        int(values.min()), int(values.median()), int(values.max()),
        how, fmt_before, fmt_after)


def reset_reports_for_test() -> None:
    _REPORTED.clear()


@_never_fails_the_launch
def report_scale_bytes(layer, names: tuple[str, ...], *, where: str,
                       weights: tuple[str, ...] = ()) -> None:
    """The bytes the operator will actually receive, once per name.

    🔴 Read from the parameter as it stands **after** every transform, not
    from an intermediate: the question is what the operator gets, and every
    step between here and it is a step that could change the answer.

    The invariant beside this one proves the layout step did not alter the
    bytes. It cannot say whether they were the right bytes when they arrived,
    and nothing on this side of the loader can -- that needs the checkpoint's
    own numbers, measured as 119..122 for w1/w3 (median 121) and 119..124 for
    w2 (median 120). So the numbers are printed and the comparison is left to
    a reader who has both.

    ⚠️ A histogram summary rather than a verdict: "all zero" means the scale
    was never populated, "shifted as a block" means the exponent bias is
    wrong, and "reshaped distribution" means the rearrangement is -- three
    different defects that a single pass/fail would flatten into one.
    """
    from vllm.logger import logger

    for name in names:
        key = f"{where}:{name}"
        if key in _REPORTED:
            continue
        tensor = getattr(layer, name, None)
        data = getattr(tensor, "data", tensor)
        if not isinstance(data, torch.Tensor):
            continue
        _REPORTED.add(key)
        flat, how, fmt = _as_bytes(data)
        counts = _histogram(flat)
        present = counts.nonzero().flatten().tolist()
        top = sorted(present, key=lambda v: -int(counts[v]))[:6]
        values = flat.to(torch.float32)
        logger.info(
            "DSV4.1 scale bytes at %s %s: shape=%s dtype=%s stored=%s n=%d "
            "read %s, min=%d median=%d max=%d, %d distinct values, most "
            "common %s. ⚠️ Compare with the checkpoint's own: w1/w3 119..122 "
            "(median 121), w2 119..124 (median 120). All zero => never "
            "populated; shifted as a block => wrong exponent bias; reshaped "
            "distribution => wrong rearrangement. ⚠️ If this says anything "
            "but 'direct', the numbers came through npu_format_cast, which "
            "may change bytes and not only layout -- a reading through a "
            "conversion is a weaker claim than a direct one.",
            where, name, tuple(data.shape), data.dtype, fmt, flat.numel(),
            how, int(values.min()), int(values.median()), int(values.max()),
            len(present), [(v, int(counts[v])) for v in top])

    for name in weights:
        key = f"{where}:{name}"
        if key in _REPORTED:
            continue
        tensor = getattr(layer, name, None)
        data = getattr(tensor, "data", tensor)
        if not isinstance(data, torch.Tensor):
            continue
        _REPORTED.add(key)
        fmt_before = _storage_format(data)
        flat, how = _readable(data)
        sample = flat[:8].tolist()
        logger.info(
            "DSV4.1 packed weight at %s %s: shape=%s dtype=%s stored=%s n=%d, "
            "first bytes %s, read %s. ⚠️ A reading taken through a conversion "
            "is not the same claim as a direct one. The ND round trip was "
            "measured byte-identical for int8/fp16/bf16 at [64,160] on this "
            "card -- that is the route, not a general property. Two 4-bit "
            "nibbles per byte while packed.",
            where, name, tuple(data.shape), data.dtype, fmt_before,
            flat.numel(), sample, how)
