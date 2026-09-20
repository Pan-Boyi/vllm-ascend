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

    WHAT DECIDES THE NORMALISATION IS ``norm_type``, NOT ``renorm``.

    ``renorm`` cannot switch it on and cannot switch it off, because no kernel
    variant reads ``renorm`` at all. On A5 the only reachable kernel is
    ``MoeGatingTopKHashRegbase`` -- ``moe_gating_top_k_hash_def.cpp``:141
    registers ``ascend950`` with ``regbaseCfg`` and the arch35 tiling keys are
    10000-10004 -- and ``renorm`` has zero references in
    ``..._kernel/moe_gating_top_k_hash_regbase.h``. In the two non-A5 kernels
    that mention it (``..._e_k_fullload.h``, ``..._generalized.h``) it is a
    member that is declared and then never assigned and never read. That is
    why the host tiling rejects every value except 0 instead of implementing
    it (``..._tiling_arch35.h``:255 for A5, ``..._tiling.cpp``:239 otherwise):
    there is nothing behind the attribute.

    ``norm_type`` is the switch, and the regbase kernel branches on it in all
    six gather variants (``..._regbase.h``:161/654/703/1081/1133/1226/1299):

      ``norm_type == 0``  (SOFTMAX)   -> NO post-top-k normalisation, only
                                         ``Muls(routed_scaling_factor)``
      ``norm_type != 0``  (SIGMOID=1, SOFTPLUS=2)
                                      -> ``ReduceSum -> Adds(eps) -> Div ->
                                         Muls``, unconditionally

    THIS GUARD'S SCOPE IS THE ``norm_type != 0`` SIDE ONLY. The mapping it
    depends on is not a function of ``scoring_func``: the one call site below
    passes ``norm_type=2`` as a LITERAL, inside the ``sqrtsoftplus`` branch, so
    every call this file makes to ``moe_gating_top_k_hash`` takes the
    normalising branch and ``norm_topk_prob=false`` is unsatisfiable on it.
    (The ``norm_type = 0 if softmax else 1`` line further down belongs to a
    DIFFERENT operator, ``DeviceOperator.moe_gating_top_k``, which is not
    judged here.) If that literal ever becomes derived, or a second hash call
    site appears with ``norm_type=0``, then ``norm_topk_prob=false`` is legal
    on that path and this guard would be a FALSE KILL on it -- which is why
    ``test_operator_is_called_with_a_normalising_norm_type`` pins the literal
    rather than leaving the scope as prose.

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
       through ``DeviceOperator.moe_gating_top_k``, a different operator with
       its own ``renorm`` handling; those are not judged here.
    3. It does not read ``norm_type`` -- it infers it from ``scoring_func``,
       which is only sound while the call site below keeps the literal. See
       the scope paragraph above; the test arm is what makes that assumption
       fail loudly instead of silently.
    4. It is a construction-time check, so it cannot know the shapes. When
       ``is_fused_supported()`` is false at forward time this router delegates
       to ``AscendGroupedTopKRouter``, which honours ``renormalize`` correctly
       -- for such a (degenerate-shape) model this guard raises although the
       run would have been right. That is the conservative direction and it is
       on purpose; a loose guard issues a safety certificate nobody rechecks.
    5. It says nothing about ``routed_scaling_factor`` or the ``eps`` added to
       the sum inside the kernel. Those are separate numerical questions. Nor
       about ``norm_type=2`` combined with ``group_count > 1``, which the
       arch35 tiling does not check at all while the non-arch35 one rejects it
       -- that is an open question that only a run can answer.
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
        "  operator behaviour: in torch.ops._C_ascend.moe_gating_top_k_hash "
        "the normalisation is decided by norm_type, and the `renorm` "
        "attribute decides nothing -- no kernel variant reads it (zero "
        "references in moe_gating_top_k_hash_regbase.h, the only kernel A5 "
        "reaches; a declared-but-never-assigned member in the other two), "
        "which is why the tiling rejects every value but 0 rather than "
        "implementing it (csrc/moe/moe_gating_top_k_hash/op_host/"
        "moe_gating_top_k_hash_tiling_arch35.h:255 on A5, "
        "..._tiling.cpp:239 otherwise). What norm_type does: norm_type==0 "
        "(SOFTMAX) skips the post-top-k normalisation and only applies "
        "routed_scaling_factor; norm_type!=0 (SIGMOID=1, SOFTPLUS=2) runs "
        "ReduceSum->Adds(eps)->Div->Muls unconditionally, in all six gather "
        "variants (..._kernel/moe_gating_top_k_hash_regbase.h:654, 703, "
        "1081, 1133, 1226, 1299). This router's only call to that operator "
        "passes norm_type=2 (SOFTPLUS) as a literal, so it always takes the "
        "normalising branch and norm_topk_prob=false cannot be honoured on "
        "it.\n"
        "  consequence if this were allowed to run: weights this checkpoint "
        "did not ask for, with no error and no log line. The only symptom is "
        "an accuracy shift.\n"
        "  fix it on one of two sides:\n"
        "    (a) checkpoint side -- if this model really is renormalised, set "
        "norm_topk_prob=true in its config.json (under text_config for the "
        "V4.1 multimodal config). DeepSeek-V4.1-Flash ships "
        "norm_topk_prob=true and is unaffected.\n"
        "    (b) operator side -- if the model really is NOT renormalised, "
        "this needs a norm_type that scores with softplus WITHOUT the "
        "ReduceSum->Div, which no current norm_type provides (norm_type "
        "couples the scoring function and the normalisation; 0 gives you "
        "softmax, not softplus). Ask the moe_gating_top_k_hash owner for it, "
        "then widen this check's scope. Do not force renormalize=True here "
        "and do not pass norm_type=0 to dodge the normalisation: both "
        "silently change the model, which is the failure this raise exists "
        "to prevent."
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
                # not mean "do not renormalise": no kernel variant reads
                # `renorm` at all. The normalisation is decided by norm_type
                # below.
                renorm=0,
                # 2 = SOFTPLUS, and on this operator norm_type couples the
                # scoring function to the normalisation: != 0 means the
                # kernel runs ReduceSum -> Adds(eps) -> Div -> Muls on the
                # selected weights. So nothing is applied in Python
                # afterwards and nothing needs to be -- the operator has
                # already done it.
                #
                # This literal is load-bearing twice over. It is the premise
                # reject_unrenormalizable_hash_routing is built on (see its
                # scope paragraph), and changing it to 0 would not merely
                # skip the normalisation, it would change the scoring
                # function to softmax. Both halves are pinned by
                # test_operator_is_called_with_a_normalising_norm_type.
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
