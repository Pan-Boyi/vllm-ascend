# SPDX-License-Identifier: Apache-2.0
"""A plain-torch reference for the indexer's selection (test oracle).

The logits are ``sum_h w_h * (q_h . k_h)`` over the compressed K rows, computed
here in float32, followed by a causal top-k. It takes dequantised inputs, so it
does not validate the cache layout K is read from.

It also reads the operator's fill value off a short row's tail: upstream's
contract is ``-1``, and the Ascend operator's convention is not documented.
"""

from __future__ import annotations

import torch


def indexer_logits(q: torch.Tensor, k: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    """``[T, S]`` scores of every query token against every compressed state.

    Args:
        q: ``[T, H, D]`` dequantised query.
        k: ``[S, D]`` dequantised compressed keys, in position order.
        weights: ``[T, H]`` per-head weights, with every scale already folded
            in -- which on the FP8 path includes the per-token q scale. Folding
            is what the operator assumes; a reference that re-applied the scale
            would disagree with a correct implementation.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, D], got {tuple(q.shape)}")
    if k.ndim != 2:
        raise ValueError(f"k must be [S, D], got {tuple(k.shape)}")
    per_head = torch.einsum("thd,sd->ths", q.to(torch.float32),
                            k.to(torch.float32))
    return torch.einsum("ths,th->ts", per_head, weights.to(torch.float32))


def causal_topk(logits: torch.Tensor, seq_positions: torch.Tensor,
                compress_ratio: int, topk: int,
                pad: int = -1) -> torch.Tensor:
    """Reference selection: causal mask, then top-k per row, ``-1`` padded.

    ``seq_positions`` is each query token's position in its own sequence, so
    the number of visible states is ``(position + 1) // ratio`` -- the same
    group-boundary arithmetic the K writer uses. Rows with fewer visible states
    than ``topk`` are padded, and *that* is the case the operator's convention
    has to be read off.

    Ties are not broken here. Two states with equal logits make the selected
    set ambiguous, and a comparison that demands an exact index match would
    report a difference that is not one. Compare sets, and compare the selected
    *values* when the sets differ -- see ``selection_disagreement``.
    """
    ratio = max(compress_ratio, 1)
    num_states = logits.shape[1]
    visible = torch.div(seq_positions + 1, ratio,
                        rounding_mode="floor").clamp(max=num_states)
    columns = torch.arange(num_states, device=logits.device)
    masked = logits.masked_fill(columns[None, :] >= visible[:, None],
                                -float("inf"))
    k = min(topk, num_states)
    top = masked.topk(k, dim=-1)
    out = torch.full((logits.shape[0], topk), pad, dtype=torch.int32,
                     device=logits.device)
    out[:, :k] = top.indices.to(torch.int32)
    # A row with fewer visible states than k has -inf in the surplus slots;
    # those are not selections, they are the pad case this exists to expose.
    out[:, :k] = torch.where(top.values > -float("inf"), out[:, :k], pad)
    return out


def selection_disagreement(got: torch.Tensor, expected: torch.Tensor,
                           logits: torch.Tensor, pad: int = -1) -> dict:
    """Compare two selections as sets, and say whether the difference matters.

    A top-k is a set, so element-wise equality reports order differences as
    errors. But set equality alone hides the case that matters most: the two
    selections differ *and* the scores differ, meaning tokens were genuinely
    ranked differently rather than tied.

    Returns per-row counts and, for rows that differ, the worst logit gap
    between a state one side kept and the other dropped. A gap of zero is a
    tie; anything else is a real disagreement.
    """
    rows = got.shape[0]
    differing, worst_gap = [], 0.0
    for r in range(rows):
        a = {int(v) for v in got[r].tolist() if int(v) != pad}
        b = {int(v) for v in expected[r].tolist() if int(v) != pad}
        if a == b:
            continue
        differing.append(r)
        only_a, only_b = a - b, b - a
        if only_a and only_b:
            gap = abs(float(logits[r, list(only_a)].max())
                      - float(logits[r, list(only_b)].max()))
            worst_gap = max(worst_gap, gap)
    return {
        "rows": rows,
        "differing_rows": differing,
        "worst_logit_gap": worst_gap,
        "all_ties": bool(differing) and worst_gap == 0.0,
    }


def pad_value_of(buffer_row: torch.Tensor, num_valid: int) -> int | None:
    """Read the operator's fill value off a row that could not be filled.

    The point of the short-context batch: with fewer visible states than topk,
    whatever sits past ``num_valid`` is the convention, and it is observed
    rather than assumed. Returns None if the tail is not constant, which is
    itself an answer -- it would mean the tail is uninitialised rather than
    padded, and reading it as token ids would select arbitrary tokens.
    """
    tail = buffer_row[num_valid:]
    if tail.numel() == 0:
        return None
    first = int(tail[0])
    return first if bool((tail == first).all()) else None
