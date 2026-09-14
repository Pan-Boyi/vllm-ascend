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


def _as_bytes(data: torch.Tensor) -> torch.Tensor:
    """The raw bytes of a scale tensor, whatever dtype it is wearing.

    🔴 ``.to(torch.int64)`` would be wrong for the dtype this actually
    arrives in. The DS scheme declares ``w13_weight_scale`` as
    ``float8_e8m0fnu`` and views it as uint8 inside the transform
    (w4a8_mxfp4.py:329-334, :357-361), so a conversion would read the
    *decoded exponent* on one side and the *stored byte* on the other, and
    the comparison would be between two different quantities -- the exact
    mistake this module exists to catch, made by the module itself.
    """
    if data.dtype not in (torch.int8, torch.uint8, torch.int32, torch.int64):
        data = data.view(torch.uint8)
    return data.detach().reshape(-1).to(torch.int64).cpu()


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
    padded = padded or {}
    expanded = expanded or {}
    problems = []
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
            _report_ok(what or name, was, now, pad_count, pad_value, factor)

    if problems:
        raise ValueError(
            "MXFP4 scale layout changed values, not just positions:\n  "
            + "\n  ".join(problems))


_REPORTED: set = set()


def _report_ok(what: str, was: torch.Tensor, now: torch.Tensor,
               pad_count: int, pad_value: int, factor: int = 1) -> None:
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
        "%s. Byte range min=%d median=%d max=%d. ⚠️ This says the layout step "
        "did not change them; whether they were the right bytes on arrival is "
        "a different question, and the checkpoint's are 119..124.",
        what, was.numel(),
        (f" (x{factor} by declared copy)" if factor != 1 else "")
        + (f" (+{pad_count} declared padding of value {pad_value})"
           if pad_count else ""),
        int(values.min()), int(values.median()), int(values.max()))


def reset_reports_for_test() -> None:
    _REPORTED.clear()


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
        flat = _as_bytes(data)
        counts = _histogram(flat)
        present = counts.nonzero().flatten().tolist()
        top = sorted(present, key=lambda v: -int(counts[v]))[:6]
        values = flat.to(torch.float32)
        logger.info(
            "DSV4.1 scale bytes at %s %s: shape=%s dtype=%s n=%d "
            "min=%d median=%d max=%d, %d distinct values, most common "
            "%s. ⚠️ Compare with the checkpoint's own: w1/w3 119..122 "
            "(median 121), w2 119..124 (median 120). All zero => never "
            "populated; shifted as a block => wrong exponent bias; reshaped "
            "distribution => wrong rearrangement.",
            where, name, tuple(data.shape), data.dtype, flat.numel(),
            int(values.min()), int(values.median()), int(values.max()),
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
        sample = data.detach().reshape(-1)[:8].to(torch.int64).cpu().tolist()
        logger.info(
            "DSV4.1 packed weight at %s %s: shape=%s dtype=%s, first bytes "
            "%s. Two 4-bit nibbles per byte while still packed; after "
            "format_cast this is one value per byte.",
            where, name, tuple(data.shape), data.dtype, sample)
