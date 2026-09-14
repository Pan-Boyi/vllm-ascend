# SPDX-License-Identifier: Apache-2.0
"""The first token's full-vocabulary logits, which nothing else records.

Needed because judgement v2 has a real threshold now -- F_KL(t201) =
2.0792e-03 nat, pass at <=3x, fail at >10x -- and computing a KL needs the
candidate's distribution over the **whole** vocabulary.

🔴 Not approximated from logprobs, and the reason decides the direction of
the error rather than its size. ``logprobs`` returns the candidate's *own*
top-k, while the KL needs the candidate's probability on the tokens where
**golden** has mass. Those two sets disagree exactly in the tail, and
summing over only the overlap makes the KL systematically **too small** --
biased toward "pass", which is the direction that never gets found by later
work. So the tap is the cheap option even though it looks like the expensive
one: 129280 x 4B = 517 KB per dump, against a whole-dump cost far larger.

⚠️ Attribution matters here in a way it does not for the layer taps. Under
DP the logits exist only on the rank serving that request, so a dump without
this key is the normal state of three ranks out of four, and "this rank did
not serve the request" must not read like "the tap did not fire".
"""

from __future__ import annotations

import torch

_RECORDED = 0
_RECORDED_AT_LAST_FLUSH = 0
_CALLS = 0
_SKIPPED_UNARMED = 0
_ROWS_SEEN: set = set()

#: golden's reference is (1, T, 129280) float32 and the comparison takes
#: ``[0, -1]``. This key holds exactly that slice: the next-token logits for
#: the last position of this forward, as a 1-D float32 tensor of vocabulary
#: size. 🔴 Written down because "logits" has named four different tensors in
#: this port already -- pre/post gather, per-row and per-request.
KEY = "epilogue.logits.last"


def record_logits(logits: torch.Tensor | None) -> None:
    """Record the last row of ``logits``, when this forward is being dumped.

    Called after ``compute_logits`` has returned, never around it: a tap that
    reads before the call gets the previous invocation's tensor, which is in
    range, correctly shaped, and wrong.
    """
    global _RECORDED, _CALLS, _SKIPPED_UNARMED

    _CALLS += 1
    from vllm_ascend.models.deepseek_v4_1.align_taps import recording_armed

    if not recording_armed():
        # Dummy runs, profile runs, decode steps that are not step 1, and
        # every forward on a rank that is not dumping. Counted, not silent.
        _SKIPPED_UNARMED += 1
        return
    if logits is None or not isinstance(logits, torch.Tensor):
        return

    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    recorder = get_recorder()
    if recorder is None:
        return

    flat = logits.detach()
    if flat.dim() == 3:                      # [B, T, V] -- take [0, -1]
        flat = flat[0, -1]
    elif flat.dim() == 2:                    # [rows, V] -- the sampled rows
        _ROWS_SEEN.add(int(flat.shape[0]))
        flat = flat[-1]
    elif flat.dim() != 1:
        return
    # 🔴 Cast named rather than assumed. golden's reference is float32; if
    # ours arrives in bf16 the comparison is against a coarser quantity and
    # the manifest has to say so, because a silent upcast here would make a
    # storage difference look like an agreement.
    was = flat.dtype
    recorder.record_whole("epilogue", KEY, flat.to(torch.float32).cpu())
    recorder.add_manifest(logits_source_dtype=str(was),
                          logits_vocab=int(flat.numel()),
                          logits_sampled_rows=sorted(_ROWS_SEEN))
    _RECORDED += 1


def report_logits_tap_coverage() -> None:
    """Print what this tap did, every flush, **including zero**.

    🔴 Both halves of the routing tap's lesson, which were two different
    failures: that counter had no caller anywhere for three flights, and when
    it got one the line said "this dump" over a process-cumulative number.

    ⚠️ And zero is spelled out, because under DP zero is the **correct**
    reading on three ranks out of four. Without that sentence a normal dump
    and a broken tap produce the same line.
    """
    global _RECORDED_AT_LAST_FLUSH
    from vllm.logger import logger

    this_dump = _RECORDED - _RECORDED_AT_LAST_FLUSH
    _RECORDED_AT_LAST_FLUSH = _RECORDED
    logger.info(
        "DSV4.1 logits tap: recorded %d time(s) this dump (%d since this "
        "process started; compute_logits entered %d time(s), %d of them not "
        "armed; sampled row counts seen %s)%s",
        this_dump, _RECORDED, _CALLS, _SKIPPED_UNARMED,
        sorted(_ROWS_SEEN) or "none",
        "" if this_dump else
        " -- ZERO this dump. Under DP that is CORRECT on every rank that did "
        "not serve this request, and it is a defect only if no rank recorded "
        "it. Compare across ranks before concluding anything. If "
        "compute_logits was entered 0 times, the override is not on the "
        "object the runner holds.")


def reset_for_test() -> None:
    global _RECORDED, _RECORDED_AT_LAST_FLUSH, _CALLS, _SKIPPED_UNARMED
    _RECORDED = 0
    _RECORDED_AT_LAST_FLUSH = 0
    _CALLS = 0
    _SKIPPED_UNARMED = 0
    _ROWS_SEEN.clear()
