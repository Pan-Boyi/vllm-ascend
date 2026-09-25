# SPDX-License-Identifier: Apache-2.0
"""The four kt offload optimisation switches, resolved in ONE place.

Three are ON by default -- the side stream, the event wait (P1a) and the shared
expert in the window. The device signal (P1b) stays off.

"Unset" means "default", not "0", and the default of P1a and of the shared
window depends on the other switches. That makes it easy for two readers to
disagree about what "unset" means, so nobody reads these variables except
:func:`resolve` -- config validation, the layer runtime and the start-up log
all call it and get the same answer, plus where the answer came from.

Values accepted in the environment: unset or "" (= default), "1", "0".
Anything else raises: a value we cannot read must not quietly mean "off" when
the default is "on".

Semantics (on/off, and the ``source`` the start-up log prints)::

  side stream    (VLLM_ASCEND_DSV41_KT_SIDE_STREAM, cfg overlap_side_stream)
      env "0"                       -> off  (env)   the kill switch
      cfg overlap_side_stream=false -> off  (cfg)
      env "1"                       -> on   (env)
      otherwise                     -> on   (default; cfg default is true)
  device signal  P1b (VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL)
      env "1"                       -> on   (env)   precondition: packed_transfer or side stream
      otherwise                     -> off  (default / env)
  async wait     P1a (VLLM_ASCEND_DSV41_KT_ASYNC_WAIT)
      env "0"                       -> off  (env)
      env "1"                       -> on   (env)   precondition as P1b; P1b also "1" -> refused
      unset, P1b on                 -> off  (auto)  yields to P1b
      unset, packed or side stream  -> on   (auto)
      unset, neither                -> off  (auto)  no error
  shared in window (VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW)
      env "0" / "1"                 -> off / on (env)  ("1" without side stream: warning)
      unset                         -> = side stream (auto)

Eager (enforce_eager / cudagraph_mode NONE): the side stream, P1a and P1b are
captured-path changes -- eager never forks, never splits submit from sync and
keeps its host-func-free submit+sync. The layer runtime additionally turns an
*auto* P1a and an *auto* shared-window move off in eager (see KTLayerOffload).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_ascend.kt_offload import shared_window, side_stream

# Names only (async_wait / device_signal import vllm's logger at module level;
# keep this module importable on its own).
ASYNC_WAIT_ENV = "VLLM_ASCEND_DSV41_KT_ASYNC_WAIT"
DEVICE_SIGNAL_ENV = "VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL"
SIDE_STREAM_ENV = side_stream.ENV_NAME
SHARED_IN_WINDOW_ENV = shared_window.ENV_NAME

DEFAULT, ENV, CFG, AUTO = "default", "env", "cfg", "auto"


def raw(env_name: str) -> str | None:
    """The tri-state reading of one variable: None (unset/empty), "0" or "1".

    Read through ``vllm_ascend.envs`` (not os.environ) so a test that
    monkeypatches the ``envs`` attribute is seen the same way as a real
    environment variable. Raises ValueError on any other value.
    """
    from vllm_ascend import envs

    value = getattr(envs, env_name, "")
    value = "" if value is None else str(value)
    if value == "":
        return None
    if value in ("0", "1"):
        return value
    raise ValueError(
        f"{env_name}={value!r}: only '1' (on), '0' (off) or unset (the default) are "
        "accepted. An earlier version read any value other than '1' as off; with the default now "
        "on, a value that cannot be read must not quietly mean either."
    )


@dataclass(frozen=True)
class Switch:
    on: bool
    source: str  # default | env | cfg | auto
    why: str = ""

    def __str__(self) -> str:
        text = f"{'on' if self.on else 'off'}({self.source}"
        return text + (f": {self.why})" if self.why else ")")


@dataclass(frozen=True)
class Switches:
    side_stream: Switch
    async_wait: Switch
    shared_in_window: Switch
    device_signal: Switch

    def summary(self) -> str:
        return (
            f"side_stream={self.side_stream} async_wait={self.async_wait} "
            f"shared_in_window={self.shared_in_window} device_signal={self.device_signal}"
        )


def resolve(cfg: Any, *, validate: bool = True) -> Switches:
    """Effective value and source of each switch for this process.

    ``validate=True`` (config validation) raises on the combinations an
    operator asked for explicitly and cannot have: P1a or P1b "1" without
    their precondition, and P1a and P1b both "1". ``validate=False`` (the
    layer runtime, and harnesses that build a layer without going through
    config validation) never raises on those: it honours explicit values as
    they are, and the layer's own start-up checks still refuse what it cannot
    run. The *auto* results are identical in both modes.
    """
    packed = bool(getattr(cfg, "packed_transfer", False))
    cfg_overlap = bool(getattr(cfg, "overlap_side_stream", False))

    # -- side stream -----------------------------------------------------
    r = raw(SIDE_STREAM_ENV)
    if r == "0":
        side = Switch(False, ENV, f"{SIDE_STREAM_ENV}=0")
    elif not cfg_overlap:
        side = Switch(
            False,
            CFG,
            "overlap_side_stream=false" + (f"; {SIDE_STREAM_ENV}=1 does not override it" if r == "1" else ""),
        )
    elif r == "1":
        side = Switch(True, ENV, f"{SIDE_STREAM_ENV}=1")
    else:
        side = Switch(True, DEFAULT)
    own_buffers = packed or side.on
    precondition = "packed_transfer" if packed else ("side stream" if side.on else "")

    # -- P1b (explicit only) -----------------------------------------------
    r = raw(DEVICE_SIGNAL_ENV)
    if r == "1":
        if validate and not own_buffers:
            raise ValueError(
                f"{DEVICE_SIGNAL_ENV}=1 needs kt_offload_config.packed_transfer or the side stream "
                f"({_side_off_why(side)}). With no callback in the graph, the polling thread is what calls "
                "moe.forward_task + cpu_infer.submit/sync, which only exists where the layer owns its "
                "pinned buffers; kt's own captured entry point (run_pinned_forward_sync) needs a "
                "callback to call it, and its buffers are shared by layer_idx % 2."
            )
        p1b = Switch(True, ENV, f"{DEVICE_SIGNAL_ENV}=1")
    else:
        p1b = Switch(False, ENV if r == "0" else DEFAULT)

    # -- P1a ---------------------------------------------------------------
    r = raw(ASYNC_WAIT_ENV)
    if r == "0":
        p1a = Switch(False, ENV, f"{ASYNC_WAIT_ENV}=0")
    elif r == "1":
        if validate and not own_buffers:
            raise ValueError(
                f"{ASYNC_WAIT_ENV}=1 needs kt_offload_config.packed_transfer or the side stream "
                f"({_side_off_why(side)}). Splitting kt's submit from its sync is only possible where WE "
                "build the task (moe.forward_task + cpu_infer.submit/sync on this layer's own pinned "
                "buffers); kt's own captured entry point (run_pinned_forward_sync) does both in one call, "
                "and its pinned buffers are shared by layer_idx % 2, which is exactly the sharing the "
                "device-side wait would stop ordering. Turn on one of the two, or unset the variable "
                "(unset = automatic: on exactly when the precondition holds)."
            )
        if validate and p1b.on:
            raise ValueError(
                f"{DEVICE_SIGNAL_ENV}=1 and {ASYNC_WAIT_ENV}=1 are both set explicitly, and they are two "
                "ways to start the same work (P1b replaces the host-func node; P1a keeps it and only moves "
                f"the waiting). P1a is on by default and steps aside by itself when P1b "
                f"is on: to run P1b, unset {ASYNC_WAIT_ENV} (or set it to 0); to run P1a, unset "
                f"{DEVICE_SIGNAL_ENV} (or set it to 0)."
            )
        p1a = Switch(True, ENV, f"{ASYNC_WAIT_ENV}=1")
    elif p1b.on:
        p1a = Switch(False, AUTO, f"yields to {DEVICE_SIGNAL_ENV}=1")
    elif own_buffers:
        p1a = Switch(True, AUTO, f"{precondition} on")
    else:
        p1a = Switch(False, AUTO, "needs packed_transfer or the side stream")

    # -- shared expert in the window ---------------------------------------
    r = raw(SHARED_IN_WINDOW_ENV)
    if r is not None:
        shared = Switch(r == "1", ENV, f"{SHARED_IN_WINDOW_ENV}={r}")
    else:
        shared = Switch(side.on, AUTO, "follows the side stream")

    return Switches(side_stream=side, async_wait=p1a, shared_in_window=shared, device_signal=p1b)


def _side_off_why(side: Switch) -> str:
    return f"side stream {side}"
