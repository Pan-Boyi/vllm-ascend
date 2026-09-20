# SPDX-License-Identifier: Apache-2.0
"""packed_transfer: one host-device input copy per layer instead of three.

The target is the measured 7.5 ms per token of transport + callback on card
(~62.5 us x 120 operations, 2026-09-20). kt fills three separate pinned
allocations, so a layer pays three host-device copies for the inputs; one
combined buffer with the same three regions pays one. The switch also gives
each layer its own buffers, which removes kt's ``layer_idx % 2`` slot sharing.

Criterion here is the same as on card: **the output is bit-identical with the
switch flipped**, and the host-device operation count drops. Speed is measured
on card, never asserted here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.kt_offload import layer as kl
from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.kt_offload.placement import build_plan, host_ops_per_layer, packed_layout

H, TOPK, NEXP = 8, 2, 16


def test_layout_regions_are_aligned_and_sized():
    lay = packed_layout(4, 5120, 6)
    assert lay["hidden"] == (0, 4 * 5120 * 2)
    assert lay["ids"][0] % 512 == 0 and lay["weights"][0] % 512 == 0
    assert lay["ids"][1] == 4 * 6 * 8 and lay["weights"][1] == 4 * 6 * 4
    assert lay["nbytes"] % 512 == 0
    # the regions do not overlap
    spans = sorted((lay[r][0], lay[r][0] + lay[r][1]) for r in ("hidden", "ids", "weights"))
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))


def test_op_accounting():
    assert host_ops_per_layer(False) == {"d2h": 3, "host_func": 1, "h2d": 1, "total": 5}
    assert host_ops_per_layer(True) == {"d2h": 1, "host_func": 1, "h2d": 1, "total": 3}
    # 40 layers: 200 -> 120 host-device operations per step
    assert 40 * host_ops_per_layer(False)["total"] == 200
    assert 40 * host_ops_per_layer(True)["total"] == 120


class KtMoE:
    """kt's C++ MoE task factory: reads the caller's pointers."""

    def __init__(self, tensors):
        self.tensors = tensors  # data_ptr -> tensor, so the fake can compute

    def forward_task(self, bsz_ptr, topk, ids_ptr, w_ptr, in_ptr, out_ptr, incremental):
        return (bsz_ptr, topk, ids_ptr, w_ptr, in_ptr, out_ptr, incremental)


class FakeWrapper:
    """kt wrapper exposing moe.forward_task + cpu_infer, as the packed path needs."""

    _layer_has_pending_deferred: dict = {}

    def __init__(self, registry):
        self.registry = registry
        self.moe = KtMoE(registry)
        self.submitted = []
        self.syncs = 0
        self.cpu_infer = SimpleNamespace(submit=self._submit, sync=self._sync)
        self.calls = []

    def _submit(self, task):
        self.submitted.append(task)

    def _sync(self, allow_pending):
        self.syncs += 1
        for bsz_ptr, topk, ids_ptr, w_ptr, in_ptr, out_ptr, _ in self.submitted:
            src, dst = self.registry[in_ptr], self.registry[out_ptr]
            weights = self.registry[w_ptr]
            # the CPU experts: 3 * hidden, scaled by the first routing weight,
            # so a wrong pointer or a stale buffer shows up in the value
            dst.copy_(3 * src * weights[:, :1])
        self.submitted.clear()

    # kt's own path, for the arm with the switch off
    def submit_forward(self, hs, ids, w, stream):
        self.calls.append("submit_forward")
        self._kt_pin = 3 * hs.float() * w[:, :1]

    def sync_forward(self, hs, stream):
        self.calls.append("sync_forward")
        return self._kt_pin.to(hs.dtype)


def _runtime(monkeypatch, packed: bool, layers: int = 4, sizes=(1, 2)):
    registry: dict[int, torch.Tensor] = {}
    plan = build_plan([[0, 3]] * layers, NEXP)
    cfg = KTOffloadConfig(
        enabled=True, cpuinfer_threads=4, num_resident_experts=1, packed_transfer=packed
    )
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=TOPK),
            model="/m",
            enforce_eager=True,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"), cudagraph_capture_sizes=[]),
    )
    counts = {"host_copy": 0, "device_fill": 0}
    monkeypatch.setattr(kl, "_stream_handle", lambda device: 0)
    monkeypatch.setattr(kl, "_wait_for_copy", lambda device: None)
    monkeypatch.setattr(kl, "_capturing", lambda: False)

    def host_copy(dst, src):
        counts["host_copy"] += 1
        dst.copy_(src)

    def device_fill(dst, src):
        counts["device_fill"] += 1
        dst.copy_(src)

    monkeypatch.setattr(kl, "_host_copy", host_copy)
    # pinned memory needs a real accelerator; the layout and the pointers are
    # what these tests are about, so plain host memory stands in
    monkeypatch.setattr(kl, "_pinned_empty", lambda shape, dtype: torch.empty(shape, dtype=dtype))
    monkeypatch.setattr(kl, "_device_fill", device_fill)
    rts = []
    for i in range(layers):
        rt = kl.KTLayerOffload(cfg, plan.layer(i), i, vcfg)
        wrapper = FakeWrapper(registry)
        monkeypatch.setattr(rt, "_build_wrapper", lambda w=wrapper: w)
        mod = torch.nn.Module()
        mod.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3), requires_grad=False)
        rt.after_weights_loaded(mod)
        if packed:
            for bs in sizes:  # eager: prepared lazily, do it up front for the test
                buf = rt._packed_buffers(bs, torch.device("cpu"))
                for key in ("pin_ids", "pin_weights", "pin_hidden", "pin_out", "bsz"):
                    registry[buf[key].data_ptr()] = buf[key]
        rts.append(rt)
    return rts, counts


def _step(rts, bs, step):
    """One forward over every layer; returns each layer's CPU contribution."""
    out = []
    for i, rt in enumerate(rts):
        hs = torch.full((bs, H), float(10 * step + i), dtype=torch.bfloat16)
        ids = torch.zeros(bs, TOPK, dtype=torch.int32)
        w = torch.full((bs, TOPK), 0.5)
        rt.route(hs, w, ids, in_profile_run=False)
        merged = rt.merge(torch.zeros(bs, H, dtype=torch.bfloat16))
        out.append(merged.float().clone())
    return out


def test_packed_is_bit_identical_to_kt_own_path(monkeypatch):
    packed_rts, packed_counts = _runtime(monkeypatch, packed=True)
    plain_rts, plain_counts = _runtime(monkeypatch, packed=False)
    for step, bs in enumerate((1, 2, 1, 2)):
        got = _step(packed_rts, bs, step)
        want = _step(plain_rts, bs, step)
        for i, (g, w) in enumerate(zip(got, want)):
            assert torch.equal(g, w), f"step {step} layer {i}: {g.flatten()[:4]} vs {w.flatten()[:4]}"


def test_packed_pays_one_host_copy_per_layer_for_the_inputs(monkeypatch):
    rts, counts = _runtime(monkeypatch, packed=True, layers=4)
    _step(rts, 1, 0)
    # per layer: one combined input copy + one output copy back
    assert counts["host_copy"] == 4 * 2
    # and three device-side fills, which are not host-device operations
    assert counts["device_fill"] == 4 * 3


def test_every_layer_keeps_its_own_buffers(monkeypatch):
    """No kt slot sharing: layer L and L+2 must not write the same pinned bytes."""
    rts, _ = _runtime(monkeypatch, packed=True, layers=4)
    ptrs = [rt._packed_buffers(1, torch.device("cpu"))["pin"].data_ptr() for rt in rts]
    assert len(set(ptrs)) == len(ptrs)
    outs = [rt._packed_buffers(1, torch.device("cpu"))["pin_out"].data_ptr() for rt in rts]
    assert len(set(outs)) == len(outs)


def test_values_follow_the_input_every_step(monkeypatch):
    """A stale buffer would repeat the previous step's value."""
    rts, _ = _runtime(monkeypatch, packed=True, layers=3)
    seen = [_step(rts, 1, step) for step in range(3)]
    for i in range(3):
        vals = [float(seen[s][i].flatten()[0]) for s in range(3)]
        assert len(set(vals)) == 3, f"layer {i} repeated a value across steps: {vals}"


def test_refuses_a_kt_build_without_the_entry_points(monkeypatch):
    plan = build_plan([[0, 3]], NEXP)
    cfg = KTOffloadConfig(enabled=True, cpuinfer_threads=4, num_resident_experts=1, packed_transfer=True)
    vcfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=H, moe_intermediate_size=4, num_experts_per_tok=TOPK),
            model="/m",
            enforce_eager=True,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="NONE"), cudagraph_capture_sizes=[]),
    )
    monkeypatch.setattr(kl, "_pinned_empty", lambda shape, dtype: torch.empty(shape, dtype=dtype))
    rt = kl.KTLayerOffload(cfg, plan.layer(0), 0, vcfg)
    monkeypatch.setattr(rt, "_build_wrapper", lambda: SimpleNamespace())  # no moe / cpu_infer
    mod = torch.nn.Module()
    mod.w13_weight = torch.nn.Parameter(torch.ones(plan.num_slots, 3), requires_grad=False)
    with pytest.raises(RuntimeError, match="packed_transfer needs kt's moe.forward_task"):
        rt.after_weights_loaded(mod)


def test_capture_of_an_unprepared_size_is_refused(monkeypatch):
    rts, _ = _runtime(monkeypatch, packed=True, layers=1, sizes=(1,))
    monkeypatch.setattr(kl, "_capturing", lambda: True)
    hs = torch.zeros(4, H, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="no buffer was prepared"):
        rts[0].route(hs, torch.ones(4, TOPK), torch.zeros(4, TOPK, dtype=torch.int32), in_profile_run=False)


# ---------------------------------------------------------------------------
# the four properties of kt's raw task we now depend on (kt, 2026-09-20)
# ---------------------------------------------------------------------------


def test_refuses_more_tokens_than_kt_sized_its_output_for(monkeypatch):
    """kt's own wrappers check qlen; calling the C++ task directly makes it ours.

    Past chunked_prefill_size the C++ output pointer runs outside the
    allocation and the heap damage shows up somewhere else entirely.
    """
    rts, _ = _runtime(monkeypatch, packed=True, layers=1, sizes=(1,))
    rts[0].chunked_prefill_size = 4
    rts[0]._packed_buffers(4, torch.device("cpu"))  # at the limit: fine
    with pytest.raises(ValueError, match="exceeds chunked_prefill_size"):
        rts[0]._packed_buffers(5, torch.device("cpu"))


def test_task_arguments_are_in_kt_positional_order(monkeypatch):
    """(qlen_ptr, k, ids, weights, input, output, incremental) -- and qlen is a pointer."""
    rts, _ = _runtime(monkeypatch, packed=True, layers=1, sizes=(1,))
    rt = rts[0]
    buf = rt._packed_buffers(1, torch.device("cpu"))
    seen = {}

    class Recorder:
        def forward_task(self, *args):
            seen["args"] = args
            return ("task",)

    rt.wrapper.moe = Recorder()
    kl._kt_packed_forward(rt._packed_payload(1))
    qlen_ptr, k, ids, w, inp, out, incremental = seen["args"]
    assert qlen_ptr == buf["bsz"].data_ptr() and int(buf["bsz"][0]) == 1
    assert k == TOPK
    assert (ids, w, inp, out) == (
        buf["pin_ids"].data_ptr(),
        buf["pin_weights"].data_ptr(),
        buf["pin_hidden"].data_ptr(),
        buf["pin_out"].data_ptr(),
    )
    assert incremental is False  # overwrite; True would accumulate into output


def test_every_layer_submits_and_syncs_in_the_same_node(monkeypatch):
    """kt re-raises a task's exception only at the next sync, so a submit
    without a sync would hide the error for ever."""
    rts, _ = _runtime(monkeypatch, packed=True, layers=2, sizes=(1,))
    _step(rts, 1, 0)
    for rt in rts:
        assert rt.wrapper.syncs == 1 and rt.wrapper.submitted == []


def test_buffers_are_dtype_and_contiguity_checked(monkeypatch):
    rts, _ = _runtime(monkeypatch, packed=True, layers=1, sizes=(1,))
    rt = rts[0]
    rt._packed.clear()
    monkeypatch.setattr(kl, "packed_layout", lambda t, h, k: {  # ids region one byte short
        "hidden": (0, t * h * 2), "ids": (512, t * k * 8 - 1), "weights": (1536, t * k * 4), "nbytes": 2048
    })
    with pytest.raises(RuntimeError):  # a view of the wrong size cannot be built
        rt._packed_buffers(1, torch.device("cpu"))
