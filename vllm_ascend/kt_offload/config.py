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
    # M3, REFUSED (2026-09-20). Narrowly: forking work onto a SECOND stream
    # and joining it back inside a capture fails at capture_end with ACL
    # 107025 -- three independent on-card runs (three layers in one capture,
    # one capture per layer, and the whole-model A/B whose captured path then
    # served 34/50 GSM8K with 38/50 answers different from the serial arm).
    # This says nothing against host work *in* a graph: a host-func node on the
    # stream being captured is a delivered production path (engram's host
    # gather, models/deepseek_v4_1/engram_embedding.py, and our own overlap=off
    # arm, 50/50). init_ascend_config rejects this switch; the code path and
    # its tests are kept for whatever mechanism replaces it. A test harness may
    # still build the config object directly to bisect the limit (tools/.../t3).
    overlap_side_stream: bool = False
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
        if self.overlap_side_stream:
            from vllm_ascend.kt_offload import side_stream

            # packed_transfer + overlap_side_stream used to be refused here
            # (2026-09-22, morning): ``route()`` tested ``packed`` first, so with
            # both on the packed branch ran and nothing was ever forked -- a
            # switch that looked set and was not. That reason is gone: the
            # forked path now *is* the packed per-layer buffers (it was moved
            # onto them to stop the side stream reading kt's shared pinned slots
            # and the caller's short-lived routing tensors), and ``route()``
            # tests ``overlap`` first. With both on, the captured path forks and
            # the eager path is packed; with only overlap on, the captured path
            # is identical and the eager path is kt's own. So the combination is
            # accepted, and the thing the refusal protected -- "on means on" --
            # is asserted where it belongs, in
            # tests/ut/kt_offload/test_side_stream_overlap.py.
            if not side_stream.enabled_by_env():
                raise ValueError(
                    "kt_offload_config.overlap_side_stream is experimental and off by default. The 2026-09-20 "
                    "refusal (ACL 107025 at capture_end) was withdrawn on 2026-09-22: the on-card case that "
                    "produced it was failing on our own capture-buffer guard, and with that fixed a real "
                    "three-layer capture on the side stream captures and replays correctly. The one arm that "
                    "still fails with 107025 is the one whose side-stream event record nobody joins, which is "
                    "not what this code does. It is still unproven in a service: no TPOT reading, and the ACL "
                    f"capture itself has no CPU coverage. Set {side_stream.ENV_NAME}=1 to run the B arm "
                    "(tools/kt_offload/oncard/RUNBOOK_sidestream.md), and compare bit-for-bit against the "
                    "serial arm -- every known failure of this path is FASTER and wrong."
                )
        # P1a (VLLM_ASCEND_DSV41_KT_ASYNC_WAIT): the captured host-func node
        # only submits, and a completion thread records the ExternalEvent the
        # graph waits on. It is an env-only switch on purpose -- the A/B it
        # exists for must differ in exactly ONE thing, and the cfg json is part
        # of an arm's fingerprint -- but it still has a precondition, and the
        # place to refuse is here, at start-up, not at the first decode step.
        from vllm_ascend.kt_offload import async_wait

        if async_wait.enabled_by_env() and not (self.packed_transfer or self.overlap_side_stream):
            raise ValueError(
                f"{async_wait.ENV_NAME}=1 needs kt_offload_config.packed_transfer or "
                "overlap_side_stream. Splitting kt's submit from its sync is only possible where WE build "
                "the task (moe.forward_task + cpu_infer.submit/sync on this layer's own pinned buffers); "
                "kt's own captured entry point (run_pinned_forward_sync) does both in one call, and its "
                "pinned buffers are shared by layer_idx % 2, which is exactly the sharing the device-side "
                "wait would stop ordering. Turn on one of the two, or unset the variable."
            )
        # P1b (VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL): no host-func node at all --
        # an 8-byte copy of a per-step device counter is the start signal and a
        # polling thread does the rest. Same precondition as P1a and for the
        # same reason (only where WE build the task), plus two of its own.
        from vllm_ascend.kt_offload import device_signal

        if device_signal.enabled_by_env():
            if not (self.packed_transfer or self.overlap_side_stream):
                raise ValueError(
                    f"{device_signal.ENV_NAME}=1 needs kt_offload_config.packed_transfer or "
                    "overlap_side_stream. With no callback in the graph, the polling thread is what calls "
                    "moe.forward_task + cpu_infer.submit/sync, which only exists where the layer owns its "
                    "pinned buffers; kt's own captured entry point (run_pinned_forward_sync) needs a "
                    "callback to call it, and its buffers are shared by layer_idx % 2."
                )
            if async_wait.enabled_by_env():
                raise ValueError(
                    f"{device_signal.ENV_NAME}=1 and {async_wait.ENV_NAME}=1 are two ways to start the same "
                    "work (P1b replaces the host-func node; P1a keeps it and only moves the waiting), and "
                    "an A/B arm must differ in exactly one thing. Turn on one of them."
                )
            from vllm_ascend import envs

            if envs.VLLM_ASCEND_DSV41_KT_SEAM_TIMING:
                raise ValueError(
                    f"{device_signal.ENV_NAME}=1 removes the host-func callback, and the seam instrument "
                    "(VLLM_ASCEND_DSV41_KT_SEAM_TIMING) measures segments *inside* that callback: with P1b "
                    "on it would report on a callback that never runs and its A/F columns would be joined "
                    "against HOSTFUNC_CALLBACK rows that do not exist. Refusing rather than printing a "
                    "report about nothing (§4.26 cost two nights to a pair of instruments that did)."
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
