# SPDX-License-Identifier: Apache-2.0
"""``additional_config["kt_offload_config"]``.

Every field is a start-up setting. The resident expert set in particular is
fixed for the life of the process: changing it means editing this section and
restarting, never editing code. (kt-kernel reads ``gpu_experts_mask`` through
a raw pointer from ``load_weights`` on; mutating it afterwards segfaults, so an
online change is out of scope -- see docs/moe_offload_design.md §b.4.)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from vllm_ascend.config_utils import config

ResidentStrategy = Literal["prefix", "frequency", "explicit"]


@config
class KTOffloadConfig:
    """Single-card MoE offload to kt-kernel. Off by default.

    Example::

        "kt_offload_config": {
            "enabled": true,
            "resident_strategy": "frequency",
            "num_resident_experts": 100,
            "frequency_table_path": "/path/gsm8k_topk_freq.pt",
            "cpuinfer_threads": 96,
            "threadpool_count": 2,
            "numa_nodes": [0, 1]
        }

    Resident set, one of:
      * ``prefix``    -- experts ``0..N-1`` on every layer.
      * ``frequency`` -- per layer, the ``N`` experts with the highest count in
        ``frequency_table_path`` (``[num_layers, num_experts]``, global expert
        ids; ties broken by lower id).
      * ``explicit``  -- ``resident_experts_path``: a JSON file listing the
        resident logical ids of every layer. Layers may differ in set and in
        size.
    """

    enabled: bool = False

    # -- where the CPU experts come from --------------------------------
    # Checkpoint directory kt-kernel reads the CPU experts from. None = the
    # served model's own directory (the native MXFP4 safetensors).
    weight_path: str | None = None
    method: str = "MXFP4"

    # -- which experts stay on the NPU ---------------------------------
    resident_strategy: ResidentStrategy = "prefix"
    # Resident experts per layer for prefix/frequency. Ignored by explicit.
    num_resident_experts: int = 0
    frequency_table_path: str | None = None
    resident_experts_path: str | None = None

    # -- kt-kernel CPU runtime -------------------------------------------
    # Required when enabled (no silent default: it decides the host cores used).
    cpuinfer_threads: int = 0
    threadpool_count: int = 2
    numa_nodes: list[int] | None = None
    # Largest token count one forward hands to kt. 0 = the scheduler's
    # max_num_batched_tokens. kt raises if a forward exceeds it.
    chunked_prefill_size: int = 0

    # -- behaviour ---------------------------------------------------------
    # The profile run routes every token to random experts (force load
    # balance). The CPU half of that is skipped: it does not change the NPU
    # memory peak the profile measures, and it would push ~3/4 of the routes
    # through the host for nothing.
    skip_cpu_in_profile_run: bool = True
    # Eager (M1): run kt's synchronous submit/sync path rather than ACL
    # stream callbacks. Sets KT_FORCE_SYNC_SUBMIT=1 for the process.
    force_sync_submit: bool = True
    # The side stream (CPU-expert D2H + host-func node captured onto a second
    # stream, joined back in merge). ON by default since PLAN §4.33, after
    # §4.25/§4.26 (whole-network bit-identity, GSM8K 50/50, 1319 = 1275/1319)
    # and the merged-version acceptances §4.32 (with P1a) and §4.33 (with P1a
    # and the shared expert in the window: decode 1170/1170 bit-identical,
    # GSM8K 50/50, 1319 = 1283/1319). Captured path only: eager never forks.
    # Off: set this false, or VLLM_ASCEND_DSV41_KT_SIDE_STREAM=0 (which wins).
    # The effective value -- and the automatic defaults of P1a and of the
    # shared-expert move, which follow it -- come from kt_offload/switches.py,
    # never from reading this field directly.
    # History: refused 2026-09-20 (ACL 107025 at capture_end), withdrawn
    # 2026-09-22 (the on-card case was failing on our own capture-buffer
    # guard; a real three-layer side-stream capture captures and replays).
    overlap_side_stream: bool = True
    # Fold kt's three separate host-device input copies into one, and use our
    # own pinned buffers instead of kt's slot-reuse pair. Off by default; the
    # criterion is bit-identical output with the switch flipped.
    packed_transfer: bool = False

    @model_validator(mode="after")
    def _validate_values(self):
        if not self.enabled:
            return self
        if self.cpuinfer_threads <= 0:
            raise ValueError("kt_offload_config.cpuinfer_threads must be set (> 0) when enabled.")
        if self.threadpool_count <= 0:
            raise ValueError("kt_offload_config.threadpool_count must be > 0.")
        if self.numa_nodes is not None and len(self.numa_nodes) != self.threadpool_count:
            raise ValueError(
                f"kt_offload_config.numa_nodes lists {len(self.numa_nodes)} node(s) "
                f"but threadpool_count is {self.threadpool_count}."
            )
        if self.chunked_prefill_size < 0:
            raise ValueError("kt_offload_config.chunked_prefill_size must be >= 0.")
        if self.resident_strategy in ("prefix", "frequency") and self.num_resident_experts <= 0:
            raise ValueError(
                f"kt_offload_config.resident_strategy={self.resident_strategy!r} needs num_resident_experts > 0."
            )
        if self.resident_strategy == "frequency" and not self.frequency_table_path:
            raise ValueError("kt_offload_config.resident_strategy='frequency' needs frequency_table_path.")
        if self.resident_strategy == "explicit" and not self.resident_experts_path:
            raise ValueError("kt_offload_config.resident_strategy='explicit' needs resident_experts_path.")
        return self

    def _validate_preconditions(self, vllm_config: Any) -> None:
        """Parallel/runtime settings the M1 implementation depends on."""
        if not self.enabled:
            return
        # PLAN §4.33: the side stream, P1a and the shared-expert move are on by
        # default; P1b is off. What each switch actually is -- cfg, env and
        # the automatic defaults together -- comes from ONE function, so this
        # check, the layer runtime and the log below cannot disagree about
        # what "unset" means. It raises on what was asked for explicitly and
        # cannot be had (P1a/P1b "1" without packed_transfer or the side
        # stream; P1a and P1b both "1").
        #
        # packed_transfer + overlap_side_stream used to be refused here
        # (2026-09-22, morning): ``route()`` tested ``packed`` first, so with
        # both on nothing was ever forked. ``route()`` now tests ``overlap``
        # first and the forked path uses the packed per-layer buffers, so the
        # combination is accepted; "on means on" is asserted in
        # tests/ut/kt_offload/test_side_stream_overlap.py.
        from vllm.logger import logger

        from vllm_ascend.kt_offload import device_signal, shared_window, switches

        sw = switches.resolve(self, validate=True)
        eager = bool(getattr(getattr(vllm_config, "model_config", None), "enforce_eager", False))
        logger.info(
            "kt_offload switches (PLAN §4.33; value(source[: why])): %s%s",
            sw.summary(),
            " -- enforce_eager: the side stream / async wait / device signal are captured-path "
            "changes and an automatic shared-window move is not made, so eager runs its original path"
            if eager
            else "",
        )
        # P1b (VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL): no host-func node at all --
        # an 8-byte copy of a per-step device counter is the start signal and a
        # polling thread does the rest. Its precondition and its conflict with
        # an explicit P1a are refused in switches.resolve; this one is its own.
        if sw.device_signal.on:
            from vllm_ascend import envs

            if envs.VLLM_ASCEND_DSV41_KT_SEAM_TIMING:
                raise ValueError(
                    f"{device_signal.ENV_NAME}=1 removes the host-func callback, and the seam instrument "
                    "(VLLM_ASCEND_DSV41_KT_SEAM_TIMING) measures segments *inside* that callback: with P1b "
                    "on it would report on a callback that never runs and its A/F columns would be joined "
                    "against HOSTFUNC_CALLBACK rows that do not exist. Refusing rather than printing a "
                    "report about nothing (§4.26 cost two nights to a pair of instruments that did)."
                )
        # §4.31: the shared-expert MLP moves in front of the kt merge. An
        # explicit "1" without the side stream is not refused -- the order is
        # still correct there, and an A/B with the side stream off is how
        # "correct, just pointless" gets shown -- but say so. (Unset follows
        # the side stream, so it cannot land here.)
        if sw.shared_in_window.on and not sw.side_stream.on:
            logger.warning(
                "%s=1 without the side stream (%s): the merge still moves behind the "
                "shared-expert MLP and the result is unchanged, but there is no idle window to hide it in "
                "(the main stream waits for the CPU experts inside merge() either way), so expect no "
                "TPOT difference.",
                shared_window.ENV_NAME,
                sw.side_stream,
            )
        pc = vllm_config.parallel_config
        sizes = {
            "tensor_parallel_size": pc.tensor_parallel_size,
            "data_parallel_size": pc.data_parallel_size,
            "pipeline_parallel_size": pc.pipeline_parallel_size,
        }
        wrong = {k: v for k, v in sizes.items() if v != 1}
        if wrong:
            raise ValueError(
                f"kt_offload_config needs a single card (TP=DP=PP=1); got {wrong}. "
                "The NPU half is placed as one EP rank that owns every resident expert."
            )
        if pc.enable_eplb:
            raise ValueError("kt_offload_config is incompatible with EPLB (it owns the expert placement).")
        if vllm_config.speculative_config is not None:
            raise ValueError("kt_offload_config does not support speculative decoding (MTP experts stay unhandled).")
