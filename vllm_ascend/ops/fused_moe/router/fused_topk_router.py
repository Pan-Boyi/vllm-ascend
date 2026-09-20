#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from collections.abc import Callable

import torch
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.models.utils import sequence_parallel_chunk

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter

DEEPSEEK_V4_IMAGE_SENTINEL_BASE_ID = 129257
DEEPSEEK_V4_IMAGE_SENTINEL_COUNT = 5


def select_deepseek_v4_vision_experts(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor | None,
    bias_vl: torch.Tensor,
    text_bias: torch.Tensor | None,
    top_k: int,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
    image_sentinel_lo: int = DEEPSEEK_V4_IMAGE_SENTINEL_BASE_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select text experts and apply the vision route to image rows.

    DeepSeek-V4 vision checkpoints borrow five consecutive in-vocabulary
    sentinel ids for IMAGE_START..IMAGE_END. Text rows retain the deterministic
    ``tid2eid`` lookup used by the text-only model, while image rows use the
    checkpoint's ``bias_vl`` with the sqrt-softplus router scores.
    """
    scores = torch.nn.functional.softplus(router_logits).sqrt()
    image_hi = image_sentinel_lo + DEEPSEEK_V4_IMAGE_SENTINEL_COUNT
    image_mask = (input_ids >= image_sentinel_lo) & (input_ids < image_hi)
    row_bias = torch.where(
        image_mask.unsqueeze(-1),
        bias_vl.to(scores.dtype).unsqueeze(0),
        (text_bias.to(scores.dtype).unsqueeze(0) if text_bias is not None else torch.zeros_like(scores)),
    )
    dynamic_ids = torch.topk(
        scores + row_bias,
        k=top_k,
        dim=-1,
        sorted=True,
    ).indices
    if tid2eid is None:
        topk_ids = dynamic_ids
    else:
        lookup_ids = torch.where(image_mask, 0, input_ids)
        text_ids = tid2eid[lookup_ids].to(torch.int64)
        topk_ids = torch.where(image_mask.unsqueeze(-1), dynamic_ids, text_ids)
    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(topk_weights.dtype).tiny
        )
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_ids


def reject_unrenormalizable_hash_routing(
    scoring_func: str,
    renormalize: bool,
    bias_vl: torch.Tensor | None,
) -> None:
    """Refuse a checkpoint the hash routing operator would silently renormalize.

    ``moe_gating_top_k_hash`` normalises the selected top-k weights on every
    call. Its ``renorm`` attribute is not a switch: the host tiling accepts
    ``renorm == 0`` and nothing else
    (``csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_tiling.cpp``
    :239 and ``..._tiling_arch35.h``:255, which is the A5 path), the operator's
    own prototype documents ``renorm`` as choosing the ORDER of norm and top-k
    (``moe_gating_top_k_hash_proto.cpp``:41), and all four kernel variants run
    ReduceSum -> Div -> Muls with no test on ``renorm_`` at all
    (``..._e_k_fullload.h``:295-307, ``..._generalized.h``:533-543,
    ``..._without_group.h``:257-267, ``..._regbase.h``:1110-1113). ``renorm_``
    is declared in two kernels and read in none.

    So for ``scoring_func == "sqrtsoftplus"`` a checkpoint that asks for
    ``norm_topk_prob=false`` gets renormalised weights anyway. Nothing raises,
    nothing logs; the only symptom is an accuracy shift, which is the most
    expensive kind of failure this port has had to chase. Hence a raise here,
    at layer construction, before any weight is read.

    A warning was considered and rejected: a policy that only prints is not an
    enforcement, and a run that prints one line at construction and then serves
    for hours is indistinguishable from a correct run.

    WHAT THIS DOES NOT COVER -- deliberately, so nobody reads a green start-up
    as more than it is:

    1. It reads the ``renormalize``/``scoring_func``/``bias_vl`` ARGUMENTS, not
       the checkpoint file. If a future call site loses the
       ``config.norm_topk_prob -> renormalize`` wiring and passes the default
       ``True``, this guard sees ``True`` and stays silent. It cannot detect a
       broken resolver, only a hostile configuration.
    2. It does not cover the non-hash path. ``scoring_func`` softmax/sigmoid go
       through ``DeviceOperator.moe_gating_top_k``, a different operator that
       does honour ``renorm``; those are not judged here.
    3. It is a construction-time check, so it cannot know the shapes. When
       ``is_fused_supported()`` is false at forward time this router delegates
       to ``AscendGroupedTopKRouter``, which honours ``renormalize`` correctly
       -- for such a (degenerate-shape) model this guard raises although the
       run would have been right. That is the conservative direction and it is
       on purpose; a loose guard issues a safety certificate nobody rechecks.
    4. It does not check the ``renorm=`` argument actually passed at the call
       site below. If someone changes that call to ``renorm=1`` the tiling
       rejects it on the device, not here.
    5. It says nothing about ``routed_scaling_factor`` or the ``eps`` added to
       the sum inside the kernel. Those are separate numerical questions.
    6. It is the Ascend router only. CPU/CUDA/310p routers are elsewhere and
       reach different operators.

    ``bias_vl is not None`` is exempt on purpose: that checkpoint takes the
    ``select_deepseek_v4_vision_experts`` path below, which applies
    ``norm_topk_prob`` in Python and never reaches the hash operator. Removing
    that exemption would make this guard reject a configuration that is
    correct today.
    """
    if scoring_func != "sqrtsoftplus":
        return
    if renormalize:
        return
    if bias_vl is not None:
        return
    raise ValueError(
        "Refusing to build this MoE router: the checkpoint asks for "
        "unnormalised routing weights and the Ascend operator cannot provide "
        "them.\n"
        "  checkpoint field: norm_topk_prob = False  (read by "
        "vllm/models/deepseek_v4/nvidia/model.py:1003 as "
        "renormalize=config.norm_topk_prob, and it arrives here as "
        f"renormalize={renormalize!r} with scoring_func={scoring_func!r})\n"
        "  operator behaviour: torch.ops._C_ascend.moe_gating_top_k_hash "
        "divides the selected top-k weights by their sum on EVERY call. Its "
        "`renorm` attribute selects the order of norm and top-k, not whether "
        "to normalise -- the tiling accepts renorm==0 only "
        "(csrc/moe/moe_gating_top_k_hash/op_host/"
        "moe_gating_top_k_hash_tiling.cpp:239, ..._tiling_arch35.h:255) and "
        "every kernel variant runs ReduceSum->Div unconditionally "
        "(..._e_k_fullload.h:295-307).\n"
        "  consequence if this were allowed to run: weights this checkpoint "
        "did not ask for, with no error and no log line. The only symptom is "
        "an accuracy shift.\n"
        "  fix it on one of two sides:\n"
        "    (a) checkpoint side -- if this model really is renormalised, set "
        "norm_topk_prob=true in its config.json (under text_config for the "
        "V4.1 multimodal config). DeepSeek-V4.1-Flash ships "
        "norm_topk_prob=true and is unaffected.\n"
        "    (b) operator side -- if the model really is NOT renormalised, "
        "this needs a moe_gating_top_k_hash variant that leaves the top-k "
        "weights alone. Ask the operator owner for it, then relax this "
        "check. Do not force renormalize=True here: that silently changes "
        "the model, which is the failure this raise exists to prevent."
    )


class AscendFusedTopKRouter(AscendGroupedTopKRouter):
    """Router adapter that uses Ascend's existing expert-selection path."""

    def __init__(
        self,
        top_k: int,
        global_num_experts: int,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        eplb_state: EplbLayerState | None = None,
        num_logical_experts: int | None = None,
        tid2eid: torch.Tensor | None = None,
        bias_vl: torch.Tensor | None = None,
        image_sentinel_lo: int = DEEPSEEK_V4_IMAGE_SENTINEL_BASE_ID,
        select_experts_fn: Callable[..., tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        # First statement on purpose: this is the load-time gate, and every
        # path that reaches the hash operator goes through this constructor.
        # It is here rather than in ``check_npu_moe_gating_top_k`` because that
        # function returns a bool and a False there means "fall back to
        # AscendGroupedTopKRouter" -- which for sqrtsoftplus drops tid2eid and
        # bias_vl, i.e. it would answer a silent numerical change with a
        # different silent numerical change. See router_factory.py.
        reject_unrenormalizable_hash_routing(
            scoring_func=scoring_func,
            renormalize=renormalize,
            bias_vl=bias_vl,
        )
        super().__init__(
            top_k=top_k,
            global_num_experts=global_num_experts,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            eplb_state=eplb_state,
        )
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.num_logical_experts = num_logical_experts if num_logical_experts is not None else global_num_experts
        self.tid2eid = tid2eid
        self.bias_vl = bias_vl
        self.image_sentinel_lo = image_sentinel_lo

    def is_fused_supported(
        self,
        hidden_states: torch.Tensor,
    ) -> bool:
        topk_group = self.topk_group if self.topk_group is not None else 1
        num_expert_group = self.num_expert_group if self.num_expert_group is not None else 1
        if not (
            num_expert_group > 0
            and hidden_states.shape[-1] % num_expert_group == 0
            and hidden_states.shape[-1] // num_expert_group > 2
        ):
            return False
        if self.top_k > (hidden_states.shape[-1] / (num_expert_group * topk_group)):
            return False
        if topk_group * hidden_states.shape[-1] / num_expert_group < self.top_k:  # noqa: SIM103
            return False
        return True

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.bias_vl is None and not self.is_fused_supported(hidden_states):
            return super()._compute_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                indices_type=indices_type,
                input_ids=input_ids,
            )

        topk_group = self.topk_group if self.topk_group is not None else 1
        num_expert_group = self.num_expert_group if self.num_expert_group is not None else 1
        renorm = int(self.renormalize)
        if self.scoring_func == "sqrtsoftplus":
            if self.tid2eid is not None or self.bias_vl is not None:
                if input_ids is None:
                    raise ValueError("DeepSeek V4 vision/hash MoE routing requires input_ids.")
                input_ids = input_ids.to(torch.int64)
                tid2eid_ones = self.tid2eid.to(torch.int32) if self.tid2eid is not None else None
                if _EXTRA_CTX.moe_comm_type == MoECommType.ALLGATHER:
                    prepare_finalize = _EXTRA_CTX.moe_comm_method.prepare_finalize
                    input_ids = prepare_finalize.all_gather_input_id_with_dp_group(input_ids)
                else:
                    input_ids = _EXTRA_CTX.moe_comm_method.pad_and_split_input_ids(input_ids)
                if _EXTRA_CTX.moe_comm_type != MoECommType.ALLGATHER and input_ids.numel() != router_logits.shape[0]:
                    # Native MoE SP chunks hidden states before MC2/All2All,
                    # while their replace-allreduce paths retain full token
                    # ids. Apply the identical TP chunk only when communication
                    # has not already aligned ids with local router rows.
                    input_ids = sequence_parallel_chunk(input_ids.reshape(-1, 1)).reshape(-1)
                input_ids = torch.where(input_ids == -1, 0, input_ids)
            else:
                input_ids = None
                tid2eid_ones = None
            if self.bias_vl is not None and input_ids is not None:
                topk_weights, topk_ids = select_deepseek_v4_vision_experts(
                    router_logits=router_logits,
                    input_ids=input_ids,
                    tid2eid=tid2eid_ones,
                    bias_vl=self.bias_vl,
                    text_bias=self.e_score_correction_bias,
                    top_k=self.top_k,
                    renormalize=self.renormalize,
                    routed_scaling_factor=self.routed_scaling_factor,
                    image_sentinel_lo=self.image_sentinel_lo,
                )
                return topk_weights.to(torch.float32), topk_ids.to(
                    torch.int32 if indices_type is None else indices_type
                )
            topk_weights, topk_ids, _ = torch.ops._C_ascend.moe_gating_top_k_hash(
                x=router_logits,
                k=self.top_k,
                bias=self.e_score_correction_bias,
                input_ids=input_ids,
                tid2eid=tid2eid_ones,
                k_group=topk_group,
                group_count=num_expert_group,
                routed_scaling_factor=self.routed_scaling_factor,
                eps=1e-20,
                group_select_mode=1,
                # renorm=0 is the only value the tiling accepts, and it does
                # not mean "do not renormalise": the kernel divides the
                # selected weights by their sum either way. Nothing is applied
                # in Python afterwards and nothing needs to be -- the operator
                # has already done it. This is only correct because the
                # constructor refused every checkpoint that asked for
                # norm_topk_prob=False; see
                # reject_unrenormalizable_hash_routing above.
                renorm=0,
                norm_type=2,
                out_flag=False,
            )
            return topk_weights, topk_ids
        norm_type = 0 if self.scoring_func == "softmax" else 1
        if self.e_score_correction_bias is not None and self.e_score_correction_bias.dtype != router_logits.dtype:
            self.e_score_correction_bias = self.e_score_correction_bias.to(router_logits.dtype)
        topk_weights, topk_ids, _ = DeviceOperator.moe_gating_top_k(
            router_logits,
            k=self.top_k,
            k_group=topk_group,
            group_count=num_expert_group,
            group_select_mode=1,
            renorm=renorm,
            norm_type=norm_type,  # 0: softmax; 1: sigmoid
            out_flag=False,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
            bias_opt=self.e_score_correction_bias,
        )

        return topk_weights, topk_ids
