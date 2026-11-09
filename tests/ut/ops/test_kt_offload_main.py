# SPDX-License-Identifier: Apache-2.0
"""Mainline MoE contracts for the optional single-card expert offload."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm_ascend.kt_offload.config import KTOffloadConfig
from vllm_ascend.ops.fused_moe import moe_comm_method as comm_module
from vllm_ascend.ops.fused_moe import routed_experts as routed_module
from vllm_ascend.ops.fused_moe import token_dispatcher as dispatcher_module
from vllm_ascend.ops.fused_moe.dataclass.shared_experts import PreparedSharedExpertInput, RoutedMoEMilestones
from vllm_ascend.ops.fused_moe.dataclass.token_dispatcher import MoEAllGatherCombineMetadata
from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner
from vllm_ascend.ops.fused_moe.moe_comm_method import AllGatherCommImpl, FusedExpertsResult
from vllm_ascend.ops.fused_moe.routed_experts import AscendRoutedExperts
from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAllGather


@pytest.mark.parametrize("preserve_fp32", [None, False, True])
def test_allgather_combine_keeps_fp32_router_precision_only_when_requested(monkeypatch, preserve_fp32):
    hidden_states = torch.tensor([[1024.0], [-1024.0]], dtype=torch.bfloat16)
    router_weights = torch.tensor([[0.5001, 0.4999]], dtype=torch.float32)
    metadata = MoEAllGatherCombineMetadata(
        topk_weights=router_weights,
        expanded_row_idx=torch.tensor([0, 1], dtype=torch.int32),
        restore_shape=torch.Size([1, 1]),
    )
    calls = []

    def unpermute(*, permuted_tokens, sorted_indices, probs):
        calls.append(probs)
        products = permuted_tokens.float().reshape(1, 2, 1) * probs.float().unsqueeze(-1)
        return products.sum(dim=1).to(permuted_tokens.dtype)

    monkeypatch.setattr(dispatcher_module.DeviceOperator, "npu_moe_token_unpermute", unpermute)
    kwargs = {} if preserve_fp32 is None else {"preserve_router_weights_fp32": preserve_fp32}
    dispatcher = TokenDispatcherWithAllGather(top_k=2, num_experts=2, num_local_experts=2, **kwargs)

    result = dispatcher.token_combine(hidden_states, metadata)

    assert len(calls) == 1
    expected_probs = router_weights if preserve_fp32 else router_weights.to(hidden_states.dtype)
    assert calls[0].dtype == expected_probs.dtype
    expected = (hidden_states.float().reshape(1, 2, 1) * expected_probs.float().unsqueeze(-1)).sum(1)
    torch.testing.assert_close(result, expected.to(hidden_states.dtype), rtol=0, atol=0)
    # Both probabilities round to 0.5 in BF16. Keeping FP32 preserves the
    # nonzero contribution, matching the CPU and streaming expert paths.
    assert bool(result.ne(0).any()) is (preserve_fp32 is True)
    if preserve_fp32:
        assert calls[0] is router_weights


@pytest.mark.parametrize("enabled", [False, True])
def test_allgather_factory_configures_router_precision_from_offload_config(monkeypatch, enabled):
    cfg = KTOffloadConfig(enabled=enabled, cpuinfer_threads=8, num_resident_experts=2)
    monkeypatch.setattr(comm_module, "get_ascend_config", lambda: SimpleNamespace(kt_offload_config=cfg))
    comm = AllGatherCommImpl.__new__(AllGatherCommImpl)
    comm.moe_config = SimpleNamespace(experts_per_token=6, num_experts=384, num_local_experts=2)

    dispatcher = comm._get_token_dispatcher()

    assert dispatcher.preserve_router_weights_fp32 is enabled
    assert dispatcher.top_k == 6
    assert dispatcher.num_experts == 384
    assert dispatcher.num_experts_local == 2


def test_allgather_factory_keeps_default_precision_without_initialized_config(monkeypatch):
    def uninitialized_config():
        raise RuntimeError("Ascend config is not initialized")

    monkeypatch.setattr(comm_module, "get_ascend_config", uninitialized_config)
    comm = AllGatherCommImpl.__new__(AllGatherCommImpl)
    comm.moe_config = SimpleNamespace(experts_per_token=6, num_experts=384, num_local_experts=2)

    assert comm._get_token_dispatcher().preserve_router_weights_fp32 is False


@pytest.mark.parametrize("expert_id,slot", [(3, 0), (7, 1), (4, -1)])
def test_offload_weight_loader_maps_global_experts_to_compact_slots(expert_id, slot):
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    layer.kt_offload = SimpleNamespace(placement=SimpleNamespace(slot_of=lambda eid: {3: 0, 7: 1}.get(eid, -1)))

    assert layer._map_global_expert_id_to_local_expert_id(expert_id) == slot


def test_disabled_offload_keeps_upstream_weight_mapping(monkeypatch):
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    upstream_mapping = MagicMock(return_value=9)
    monkeypatch.setattr(routed_module.RoutedExperts, "_map_global_expert_id_to_local_expert_id", upstream_mapping)

    assert layer._map_global_expert_id_to_local_expert_id(4) == 9
    upstream_mapping.assert_called_once_with(4)


@pytest.mark.parametrize("defer_merge", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("num_tokens", [0, 2])
def test_mainline_routed_forward_merges_cpu_once_and_streaming_skips_cpu(
    monkeypatch, defer_merge, streamed, num_tokens
):
    hidden_states = torch.full((num_tokens, 4), 2.0)
    router_logits = torch.empty((num_tokens, 8))
    # Include the model's routed scaling in these weights. Both execution
    # halves must receive them unchanged; the merge must not scale them again.
    logical_weights = torch.full((num_tokens, 2), 0.75, dtype=torch.float32)
    logical_ids = torch.tensor([[3, 7]], dtype=torch.int32).expand(num_tokens, -1)
    npu_weights = logical_weights * 0.5
    npu_ids = torch.zeros_like(logical_ids)
    resident_out = torch.full_like(hidden_states, 3.0)
    cpu_out = torch.full_like(hidden_states, 5.0)
    streamed_out = resident_out + cpu_out if streamed else None
    offload = SimpleNamespace(
        maybe_stream_prefill=MagicMock(return_value=streamed_out),
        route=MagicMock(return_value=SimpleNamespace(topk_weights=npu_weights, topk_ids=npu_ids)),
        merge=MagicMock(side_effect=lambda output: output + cpu_out),
    )
    quant_method = SimpleNamespace(
        apply=MagicMock(return_value=FusedExpertsResult(routed_out=resident_out)),
        quant_method=None,
    )
    comm = SimpleNamespace(
        prepare=MagicMock(
            return_value=SimpleNamespace(
                hidden_states=hidden_states,
                router_logits=router_logits,
                mc2_mask=None,
                padded_hidden_states_shape=hidden_states.shape,
                pertoken_scale=None,
            )
        ),
        finalize=MagicMock(side_effect=lambda **kwargs: kwargs["hidden_states"]),
    )
    layer = AscendRoutedExperts.__new__(AscendRoutedExperts)
    layer.kt_offload = offload
    layer.enable_npugraph_ex_static_kernel = False
    layer.moe_config = SimpleNamespace(is_sequence_parallel=False)
    object.__setattr__(layer, "quant_method", quant_method)
    layer.router = SimpleNamespace(eplb_state=None)
    layer._select_experts = MagicMock(return_value=(logical_weights, logical_ids))
    layer._use_v2_model_runner = True
    layer.dynamic_eplb = False
    layer.return_with_event = True
    monkeypatch.setattr(routed_module, "get_forward_context", lambda: SimpleNamespace(all_moe_layers=None))
    monkeypatch.setattr(routed_module, "get_moe_comm_method", lambda *_: comm)
    monkeypatch.setattr(
        routed_module,
        "_EXTRA_CTX",
        SimpleNamespace(in_profile_run=False, moe_comm_type=None, moe_comm_method=comm),
    )

    result, milestones = layer.forward_impl(
        hidden_states=hidden_states,
        router_logits=router_logits,
        defer_kt_merge=defer_merge,
    )
    assert isinstance(milestones, RoutedMoEMilestones)
    if defer_merge:
        offload.merge.assert_not_called()
        result = layer.kt_merge(result)

    torch.testing.assert_close(result, resident_out + cpu_out)
    if streamed:
        offload.route.assert_not_called()
        quant_method.apply.assert_not_called()
        offload.merge.assert_not_called()
    else:
        offload.route.assert_called_once_with(hidden_states, logical_weights, logical_ids, in_profile_run=False)
        assert quant_method.apply.call_args.kwargs["topk_weights"] is npu_weights
        assert quant_method.apply.call_args.kwargs["topk_ids"] is npu_ids
        offload.merge.assert_called_once_with(resident_out)
    assert layer.ascend_pertoken_scale is None
    assert layer.ascend_mc2_mask is None


@pytest.mark.parametrize("defer_merge", [False, True])
def test_runner_preserves_shared_milestones_around_optional_cpu_merge(defer_merge):
    hidden_states = torch.ones(2, 4)
    router_logits = torch.ones(2, 8)
    resident_out = torch.full_like(hidden_states, 3.0)
    shared_out = torch.full_like(hidden_states, 4.0)
    merged_out = torch.full_like(hidden_states, 5.0)
    milestones = RoutedMoEMilestones()
    prepared_input = PreparedSharedExpertInput(hidden_states)
    order = []

    def routed_forward(**kwargs):
        order.append("routed")
        assert kwargs.get("defer_kt_merge", False) is defer_merge
        return resident_out, milestones

    def shared_forward(prepared, events, *, defer_output_wait):
        order.append("shared")
        assert prepared is prepared_input
        assert events is milestones
        return shared_out

    def merge(output):
        order.append("merge")
        assert output is resident_out
        return merged_out

    runner = SimpleNamespace(
        _sequence_parallel_context=nullcontext,
        ascend_shared_experts=SimpleNamespace(
            prepare_input_before_routed=lambda hidden: prepared_input,
            forward=shared_forward,
        ),
        routed_experts=SimpleNamespace(
            kt_shared_in_window=defer_merge,
            forward_impl=routed_forward,
            kt_merge=MagicMock(side_effect=merge),
        ),
        _can_overlap_sp_shared_with=lambda transform: False,
        _prepare_router_and_milestones=lambda *args: (router_logits, None, None, None),
        routed_input_transform=None,
        routed_output_transform=None,
    )

    shared_result, routed_result = AscendMoERunner._forward_impl(runner, hidden_states, router_logits, None)

    assert shared_result is shared_out
    assert routed_result is (merged_out if defer_merge else resident_out)
    assert order == (["routed", "shared", "merge"] if defer_merge else ["routed", "shared"])
    if not defer_merge:
        runner.routed_experts.kt_merge.assert_not_called()
