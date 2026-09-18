# SPDX-License-Identifier: Apache-2.0
"""Log, once per process, that a behaviour branch was actually taken.

An environment variable being set, or a config value being parsed, does not
show that the code path it selects ever ran. Calling ``announce_once`` from
the branch itself gives that evidence without flooding per-token paths. It is
purely informational and must never change behaviour.
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
