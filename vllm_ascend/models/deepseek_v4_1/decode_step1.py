# SPDX-License-Identifier: Apache-2.0
"""Record the **first decode step**, so the two arms can be compared there.

Why this step and no other. Integration's same-side comparison on the
existing dumps is negative and complete: DP3's req_000 (group A) against
req_001 (group B), same process, same rank, adjacent, 923 shared keys,
923/923 covered, and **zero real model tensors differ**. Prefill is still
bit-identical, and it stayed bit-identical across both the A-2 fix and the
scale root-cause fix. Meanwhile the two arms already disagree numerically at
decode step 1 -- ' p' at -1.032261 against -0.433779, which is 0.6 logprob
and not last-bit jitter -- while the text only diverges at position 6.

So the divergence is born in one forward, and this is that forward.

🔵 What makes it cheap: the comparison is **same-side**. Two requests in one
server on one rank, no golden, no cross-side alignment, no G-1. The criterion
is the one that proved prefill identical -- key-by-key, and among the keys
that differ, the first in forward order is the source.

⚠️ Grouping, established by integration over 6/6 requests: req1 DP1#1,
req2 DP2#1, req3 DP3#1, req5 DP0#1 are all group A; req4 DP3#2 and req6
DP0#2 are all group B. The grouping is by **which prefill this is for that
rank**, and it does not group by rank identity. So this is the old
odd/even phenomenon, not a new one -- which says the quantity to look for is
one that **carries state across requests**.

🔴 Four candidates, and this records all four rather than the two with the
higher prior: the ring buffer's state, the engram hash cache, the
compressor's partial-group state, and the indexer's single-query branch.
Integration explicitly refused to rank them, because there is no decode-side
reading of any of them and a ranking would be prior rather than measurement.
That refusal is right, and ordering them here would quietly undo it.
"""

from __future__ import annotations

_SEEN_PREFILL: set = set()
_RECORDED_DECODE: set = set()
_ARM_CALLS = 0
_ARMED_DECODE = 0
_ARMED_DECODE_AT_LAST_FLUSH = 0
_NO_IDENTITY_REPORTED = False


def request_identity() -> str | None:
    """A name for the request this forward belongs to, or None.

    🔴 By identity, never by arrival order. "The forward after the prefill"
    is only the same request while nothing interleaves, and when something
    does interleave the mistake is silent: a decode step of one request gets
    recorded under another's dump, with every shape and every key name
    correct.

    ⚠️ Returns None rather than a guess. A synthesised identity would make
    this function total and its callers wrong.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except Exception:                                        # noqa: BLE001
        return None
    metadata = getattr(context, "attn_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for entry in metadata.values():
        req = getattr(entry, "req_metadata", None)
        ids = getattr(req, "req_ids_tensor", None)
        if ids is not None and ids.numel() == 1:
            return f"req:{int(ids.flatten()[0])}"
    return None


def note_prefill(identity: str | None) -> None:
    """This request's prefill was recorded; its next single-row forward is
    decode step 1."""
    if identity is not None:
        _SEEN_PREFILL.add(identity)


def is_first_decode_step(identity: str | None, rows: int | None,
                         seq_len: int | None) -> bool:
    """Whether this forward is the first decode step of a recorded prefill.

    All four conditions are required and each excludes a real case:

    · a known identity        -- otherwise this cannot be attributed at all
    · whose prefill we saw    -- a request whose prefill was not recorded has
                                 nothing on the other side to compare against
    · exactly one query row   -- two rows is a chunked prefill or a batched
                                 decode, neither of which is this step
    · not already recorded    -- step 2 also has one row, and recording it
                                 under the same name would overwrite step 1
                                 with something that looks just like it
    """
    global _ARM_CALLS
    _ARM_CALLS += 1
    if identity is None or rows != 1 or seq_len is None:
        return False
    if identity not in _SEEN_PREFILL:
        return False
    if identity in _RECORDED_DECODE:
        return False
    return True


def note_decode_recorded(identity: str | None) -> None:
    global _ARMED_DECODE
    if identity is not None:
        _RECORDED_DECODE.add(identity)
    _ARMED_DECODE += 1


def report_decode_step1_coverage() -> None:
    """Print what this accounting bit holds, every flush, **including zero**.

    🔴 Two lessons, both paid for on the routing tap, and they are different
    failures:

    · that counter existed for three flights with **no caller anywhere**, so
      "it never ran" and "nobody printed it" produced the same output
    · when a caller was added, the line said "this dump" while the counter
      was process-cumulative, so every reading was a running total wearing a
      per-dump label

    So: a reader that is actually called, and two numbers whose labels each
    match their own scope.
    """
    global _ARMED_DECODE_AT_LAST_FLUSH
    from vllm.logger import logger

    this_dump = _ARMED_DECODE - _ARMED_DECODE_AT_LAST_FLUSH
    _ARMED_DECODE_AT_LAST_FLUSH = _ARMED_DECODE
    logger.info(
        "DSV4.1 decode step-1 recording: armed %d time(s) this dump "
        "(%d since this process started; %d prefills seen, %d arming checks)%s",
        this_dump, _ARMED_DECODE, len(_SEEN_PREFILL), _ARM_CALLS,
        "" if this_dump else
        " -- ZERO this dump. That is correct for a dump that IS the prefill: "
        "the decode step comes after it. It is a defect only if a whole "
        "request produced no decode dump, which shows up as prefills seen "
        "exceeding decode dumps by more than one.")


def reset_for_test() -> None:
    global _ARM_CALLS, _ARMED_DECODE, _ARMED_DECODE_AT_LAST_FLUSH
    global _NO_IDENTITY_REPORTED
    _SEEN_PREFILL.clear()
    _RECORDED_DECODE.clear()
    _ARM_CALLS = 0
    _ARMED_DECODE = 0
    _ARMED_DECODE_AT_LAST_FLUSH = 0
    _NO_IDENTITY_REPORTED = False
