# SPDX-License-Identifier: Apache-2.0
"""Say, once, that a switch's branch was actually taken.

🔴 A switch with no observable artifact cannot be used as an experiment.
Setting ``VLLM_ASCEND_DSV41_ENGRAM_BLOCKING_H2D=1`` and having the branch
never run produces the same log as setting it and having the branch run and
not help: in both cases the only line mentioning it is vLLM's own

    WARNING Unknown vLLM environment variable detected: VLLM_ASCEND_DSV41_...

which proves the variable was **set**, not that the code path was **taken**.
Integration nearly reported a hypothesis as refuted on exactly that reading.

So every switch on this line announces itself from the place where its branch
is taken -- not from where the value is parsed, because parsing happens
whether or not the branch is ever reached, and that is the distinction being
made.

⚠️ This is evidence, not a guard. It changes no behaviour and it must never
be the thing that makes a configuration work; if a line here is deleted the
run is identical and only the reading of it gets worse. The counterpart
failure is the other half of the same hole: a typo in the value silently
selecting the default, which the switches refuse rather than absorb.

Once per process per message, so a per-token decision point does not flood
the log -- the question it answers ("did this branch ever run") is answered
by the first occurrence.
"""

from __future__ import annotations

_ANNOUNCED: set[str] = set()


def announce_once(key: str, message: str, *args) -> bool:
    """Log ``message`` the first time ``key`` is announced.

    Returns whether it logged, so a test can assert the once-ness rather than
    take it on trust.
    """
    if key in _ANNOUNCED:
        return False
    _ANNOUNCED.add(key)
    from vllm.logger import logger

    logger.info("DSV4.1 switch in effect: " + message, *args)
    return True


def reset_for_test() -> None:
    """Forget what has been announced. Tests only."""
    _ANNOUNCED.clear()
