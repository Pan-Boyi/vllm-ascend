# SPDX-License-Identifier: Apache-2.0
"""Single-card MoE expert offload to the host CPU through kt-kernel.

A subset of each MoE layer's routed experts stays resident on the NPU; the
rest are computed by kt-kernel on the host and added back after the NPU
combine. Off by default (``additional_config.kt_offload_config.enabled``);
when off, no code in this package runs on the forward path.

Layout:
  config.py     -- the user-facing ``kt_offload_config`` section
  placement.py  -- which experts are resident, logical id <-> NPU weight slot,
                   routing split. Pure torch; unit-tested on CPU.
  layer.py      -- the per-layer runtime (kt wrapper, eager submit/sync) and
                   the hooks ``AscendRoutedExperts`` calls.

See docs/moe_offload_design.md.
"""
