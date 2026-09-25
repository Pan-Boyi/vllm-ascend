# SPDX-License-Identifier: Apache-2.0
"""Which request is prefilling this step, and whether this step finishes its prompt.

The folded swap-in accumulates heat over a WHOLE request's prefill and commits
once, on the last chunk, so it needs "is this the last prefill forward of this
request?". A layer cannot work that out: neither
``vllm.forward_context.ForwardContext`` nor the per-layer attention metadata
carries the prompt length, and "this chunk is short" -- the only thing a layer
could test alone -- is *wrong* whenever the prompt length divides the token
budget, the round numbers a benchmark uses. So the signal comes from the
scheduler output (``vllm/v1/core/sched/output.py``):

* ``scheduled_new_reqs[i]`` is a ``NewRequestData`` with ``req_id``,
  ``num_computed_tokens`` and ``prompt_len``;
* ``scheduled_cached_reqs`` is a ``CachedRequestData`` with ``req_ids`` and a
  parallel ``num_computed_tokens`` -- and no prompt length, which is why the
  length is remembered here from the step the request was new;
* ``num_scheduled_tokens[req_id]`` is what this step computes;
* ``finished_req_ids`` is when to forget one.

A request is prefilling this step when ``computed < prompt_len``; the step is
its last prefill chunk when ``computed + scheduled >= prompt_len``. ``first``
(``computed == 0``) and the request id tell the fold's accumulator that a new
request started.

``observe`` reads ONLY the scheduler output, so it does not care whether it runs
before or after ``_update_states``, which advances
``CachedRequestState.num_computed_tokens`` and
``InputBatch.num_computed_tokens_cpu`` during the step. Same reason the runner's
own ``is_prefilling`` / ``discard_request_mask`` are not reused: the same three
numbers, but taking them means reaching into ``model_runner_v1``'s buffers at
exactly the right line, and they carry no request id.

Nothing here is on unless ``arm()`` was called (``layer.install_resident_fold``,
i.e. ``dynamic_resident``): with the switch off ``observe`` sets one bool and
returns. Not a fallback, though -- if the runner never calls ``observe`` at all,
``require`` RAISES rather than letting the fold fall back to "decide on every
chunk", the behaviour this module replaces, which would look in every counter
like a fold that is working.

``last`` is not enough to place the commit, because the forward carrying the
last chunk may never reach the fold: ``layer.maybe_stream_prefill`` declines any
forward of fewer than ``stream_prefill_min_tokens`` tokens, and the LAST chunk
(``prompt_len mod max_num_batched_tokens``) is the one most likely to be short.
The symptom is silent -- heat accumulated over the whole prefill, nothing
swapped in. So the window also reports ``remaining``, the prompt tokens still to
come AFTER this forward, and the fold compares it against the same threshold the
layer gates on (``fold_swap.ResidentFold._decides_now``). ``on_prefill_end``
exists for the same reason: "this prefill committed nothing" cannot be detected
on the forward that ends the prefill, because the failure IS that forward not
happening.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

__all__ = [
    "PrefillWindow",
    "arm",
    "armed",
    "current",
    "observe",
    "on_prefill_end",
    "publish",
    "require",
    "reset",
]

#: Above this many remembered prompt lengths, drop everything this step does not
#: schedule. bs=1 keeps one; the cap only bounds a leak if ``finished_req_ids``
#: ever stops naming a request we saw.
_MAX_TRACKED = 4096


@dataclass(frozen=True)
class PrefillWindow:
    """One request's position in its own prefill, as of this step.

    ``seq`` counts ``observe`` calls -- not a vLLM step number; it lets a
    consumer tell a freshly published window from the previous step's.
    """

    key: str
    first: bool
    last: bool
    computed: int
    scheduled: int
    prompt_len: int
    seq: int

    @property
    def remaining(self) -> int:
        """Prompt tokens still to be prefilled AFTER this forward. ``last`` <=> this is 0.

        What decides whether this forward is the last chance to fold anything
        into a streamed pass (``fold_swap.ResidentFold._decides_now``): whatever
        the scheduler does, no later forward of this prefill can carry more than
        ``remaining`` tokens.
        """
        return max(0, self.prompt_len - (self.computed + self.scheduled))

    def __str__(self) -> str:
        return (f"request {self.key}: tokens {self.computed}..{self.computed + self.scheduled} of "
                f"{self.prompt_len}{' (first)' if self.first else ''}{' (last)' if self.last else ''} "
                f"[step {self.seq}]")


_armed = False
_hook_ran = False
_window: PrefillWindow | None = None
_seq = 0
_prompt_len: dict[str, int] = {}
_warned: set[str] = set()
#: The request whose prefill the last published window belonged to. The step
#: that publishes anything else (another request, or no window at all) is the
#: step AFTER that prefill ended.
_prefilling: str | None = None
_end_hooks: list[Any] = []


def arm() -> None:
    """Switch the derivation on. Called by the ``dynamic_resident`` install."""
    global _armed
    _armed = True


def armed() -> bool:
    return _armed


def reset() -> None:
    """Forget everything (a fresh process, or a test)."""
    global _armed, _hook_ran, _window, _seq, _prefilling
    _armed = False
    _hook_ran = False
    _window = None
    _seq = 0
    _prefilling = None
    _prompt_len.clear()
    _warned.clear()
    _end_hooks.clear()


def on_prefill_end(hook: Any) -> None:
    """Call ``hook(req_id)`` on the first step after that request stopped prefilling.

    Deliberately NOT in the fold: the failure it detects ("a whole prefill went
    by and the fold moved nothing") is a last chunk too short to stream, i.e. a
    forward on which the fold's code never runs, while ``observe`` runs once per
    step whatever the layers do. Registered once, by
    ``layer.install_resident_fold``. The hook fires on the step that publishes a
    DIFFERENT window (another request, or none), so a request that is still the
    last one this process saw is never reported -- "we cannot know yet", not
    silence about a known outcome.
    """
    if hook not in _end_hooks:
        _end_hooks.append(hook)


def publish(window: PrefillWindow | None) -> None:
    """Set this step's window directly. ``observe`` is how production gets one."""
    global _hook_ran
    _hook_ran = True
    _publish(window)


def current() -> PrefillWindow | None:
    """This step's window, or ``None`` if the step is not a single request's prefill."""
    return _window


def require() -> PrefillWindow | None:
    """``current()``, but refuse if the runner hook is not installed at all.

    ``None`` means "this step is not a prefill I can attribute to one request"
    and the caller declines to fold. A missing hook would make every step look
    like that, so it raises instead.
    """
    if not _hook_ran:
        raise RuntimeError(
            "kt_offload: dynamic_resident is on but nothing has published a prefill window. The folded "
            "swap-in accumulates a request's heat across its prefill chunks and commits once, on the last "
            "one, and only the scheduler output knows which chunk that is -- so "
            "vllm_ascend/kt_offload/prefill_window.observe(scheduler_output) has to run once per step from "
            "the model runner's execute_model. It has never run in this process. Refusing rather than "
            "deciding on every chunk, which is the behaviour this replaces.")
    return _window


def _forget(finished) -> None:
    for req_id in finished or ():
        _prompt_len.pop(req_id, None)


def _prompt_len_of(req: Any) -> int | None:
    """``NewRequestData.prompt_len``, or ``None`` if this request has no prompt tokens."""
    n = getattr(req, "prompt_len", None)
    if n is None:
        ids = getattr(req, "prompt_token_ids", None)
        n = None if ids is None else len(ids)
    n = None if n is None else int(n)
    return n if n and n > 0 else None


def observe(scheduler_output: Any) -> PrefillWindow | None:
    """Derive and publish this step's prefill window. Reads the scheduler output only.

    Called once per ``execute_model``. Returns what it published, for callers
    that assert on it; production ignores the return value.
    """
    global _window, _seq, _hook_ran
    _hook_ran = True
    if not _armed:
        return None
    _seq += 1

    scheduled: dict[str, int] = dict(getattr(scheduler_output, "num_scheduled_tokens", None) or {})
    _forget(getattr(scheduler_output, "finished_req_ids", None))

    computed: dict[str, int] = {}
    for req in getattr(scheduler_output, "scheduled_new_reqs", None) or ():
        req_id = str(req.req_id)
        computed[req_id] = int(req.num_computed_tokens)
        n = _prompt_len_of(req)
        if n is not None:
            _prompt_len[req_id] = n
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is not None:
        ids = list(getattr(cached, "req_ids", None) or ())
        nums = list(getattr(cached, "num_computed_tokens", None) or ())
        # Two parallel lists in the dataclass; a length mismatch would silently
        # pair a request with another one's progress.
        if len(ids) != len(nums):
            raise RuntimeError(
                f"kt_offload prefill_window: scheduled_cached_reqs has {len(ids)} req_ids and {len(nums)} "
                "num_computed_tokens; they are parallel lists and pairing them by position is the only way "
                "to read them")
        for req_id, n in zip(ids, nums):
            computed[str(req_id)] = int(n)

    windows = []
    for req_id, n_sched in scheduled.items():
        n_sched = int(n_sched)
        done = computed.get(req_id)
        n_prompt = _prompt_len.get(req_id)
        if done is None or n_prompt is None or n_sched <= 0:
            # A request whose prompt length was never seen (process joined
            # mid-flight, or embeddings rather than tokens) cannot be placed in
            # its own prefill, so it makes the whole step unattributable rather
            # than being skipped quietly.
            _warn_once(f"missing-{req_id}" if n_prompt is None else f"empty-{req_id}",
                       f"request {req_id}: no prompt length on record (computed={done}, scheduled={n_sched}); "
                       "this step cannot be attributed to a request's prefill")
            return _publish(None)
        if done >= n_prompt:
            continue                                   # decoding, not prefilling
        windows.append(PrefillWindow(key=req_id, first=done == 0, last=done + n_sched >= n_prompt,
                                     computed=done, scheduled=n_sched, prompt_len=n_prompt, seq=_seq))

    if len(windows) != 1:
        # Zero: a pure decode step. More than one: one request's heat is not
        # separable from another's, which is why dynamic_resident requires
        # max_num_seqs == 1 (config.py). Either way no window, and the fold
        # declines rather than guessing.
        if len(windows) > 1:
            _warn_once("multi", f"{len(windows)} requests are prefilling in one step "
                                f"({[w.key for w in windows]}); dynamic_resident accumulates heat per request "
                                "and has no commit point for a mixed step")
        return _publish(None)

    if len(_prompt_len) > _MAX_TRACKED:
        for req_id in [k for k in _prompt_len if k not in scheduled]:
            del _prompt_len[req_id]
    return _publish(windows[0])


def _publish(window: PrefillWindow | None) -> PrefillWindow | None:
    global _window, _prefilling
    _window = window
    key = None if window is None else window.key
    ended = _prefilling if (_prefilling is not None and key != _prefilling) else None
    _prefilling = key
    if ended is not None:
        # After the assignment: a hook that raises must not leave this module
        # believing the finished request is still prefilling.
        for hook in list(_end_hooks):
            hook(ended)
    return window


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning("kt_offload prefill_window: %s", message)
