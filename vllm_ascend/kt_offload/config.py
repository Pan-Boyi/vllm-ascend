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
    # M3: run the captured CPU-expert submit on a side stream so the resident
    # NPU experts overlap it (sglang's KT_SIDE_STREAM). Interface only; the
    # default, serial path is the one M2 implements.
    overlap_side_stream: bool = False

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
        if self.overlap_side_stream:
            raise ValueError("kt_offload_config.overlap_side_stream is reserved for M3 (not implemented).")
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
