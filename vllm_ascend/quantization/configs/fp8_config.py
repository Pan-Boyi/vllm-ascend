import inspect
from typing import Optional

import torch
from compressed_tensors.quantization import QuantizationArgs
from vllm.logger import logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.models.deepseek_v4 import DeepseekV4FP8Config

from vllm_ascend.utils import FP8_METHOD

from ..methods import get_scheme_class
from ..utils import is_fused_moe_layer

# vLLM 0.27.1's is_layer_skipped has no match_mode; newer trees default to exact.
_IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE = "match_mode" in inspect.signature(is_layer_skipped).parameters


QUANTIZATION_SCHEME_MAP_TYPE = dict[str, dict[str, QuantizationArgs] | None]


@register_quantization_config(FP8_METHOD)
class AscendFp8Config(Fp8Config):
    """Serve checkpoints published under the generic HF ``fp8`` quant method.

    Only the block-wise flavour is supported: weights in ``float8_e4m3fn`` paired
    with a float32 ``weight_scale_inv`` holding one scale per
    ``weight_block_size`` tile. Layers the checkpoint left in bfloat16 are named
    in ``ignored_layers`` / ``modules_to_not_convert`` and fall back to the
    unquantized methods.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}

    @classmethod
    def from_config(cls, config: dict) -> "AscendFp8Config":
        # ``Fp8Config.from_config`` keeps only the keys the generic block-wise scheme understands and
        # drops the rest, so by the time a mis-dispatched DeepSeek V4 checkpoint reaches
        # ``get_quant_method`` there is nothing left to recognise it by: the routed experts simply come
        # out FP8 instead of MXFP4, roughly doubling the weights (~296 GB -> ~592 GB) with no error
        # anywhere. Catch it here, while the raw checkpoint dict is still in hand.
        expert_dtype = config.get("expert_dtype")
        if expert_dtype is not None and expert_dtype != "fp8":
            raise ValueError(
                f"This checkpoint declares `quantization_config.expert_dtype={expert_dtype!r}`, so its "
                f"routed experts are not plain block-wise FP8, yet it was dispatched to the generic "
                f"Ascend FP8 path. That path would quantize the experts to FP8 anyway and roughly "
                f"double their memory footprint. It should have been claimed by "
                f"AscendDeepseekV4FP8Config instead -- check that its `override_quantization_method` "
                f"recognises this checkpoint's model_type."
            )
        # ``scale_fmt: ue8m0`` marks DeepSeek-native e8m0 scales; the generic HF fp8 scheme stores
        # float32 ``weight_scale_inv`` instead. Landing here with e8m0 scales is not fatal on its own,
        # but it is a strong hint that a DeepSeek checkpoint missed its dispatch, so say so out loud.
        if config.get("scale_fmt") == "ue8m0":
            logger.warning(
                "Serving a checkpoint with `scale_fmt: ue8m0` through the generic Ascend FP8 path "
                "(weight_block_size=%s). DeepSeek V4-family checkpoints belong on "
                "AscendDeepseekV4FP8Config -- verify this is really the intended path.",
                config.get("weight_block_size"),
            )
        return super().from_config(config)

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError('Ascend hardware dose not support "get_min_capability" feature.')

    def _verify_block_quantization(self) -> None:
        if self.weight_block_size is None:
            raise NotImplementedError(
                "vLLM Ascend serves native FP8 checkpoints only when their scales are stored per "
                "weight block, that is when `quantization_config.weight_block_size` is present in "
                "config.json. This checkpoint uses per-tensor or per-channel FP8 scales, which have "
                "no Ascend execution path yet. Re-quantize it with ModelSlim and serve with "
                "`--quantization ascend`, or use an llm-compressor checkpoint with "
                "`--quantization compressed-tensors`."
            )
        if self.activation_scheme != "dynamic":
            raise NotImplementedError(
                f"vLLM Ascend supports dynamic activation quantization for native FP8 checkpoints, "
                f"but this one declares `activation_scheme: {self.activation_scheme}`."
            )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        # Delayed imports: both modules reach back into the quantization package.
        from vllm_ascend.ops.fused_moe.routed_experts import AscendUnquantizedFusedMoEMethod
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        from ..method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        is_linear = isinstance(layer, LinearBase)
        is_moe = is_fused_moe_layer(layer)
        if not is_linear and not is_moe:
            return None

        skip_kwargs = {}
        if _IS_LAYER_SKIPPED_SUPPORTS_MATCH_MODE:
            skip_kwargs["match_mode"] = getattr(self, "ignored_layers_match_mode", "exact")
        if is_layer_skipped(
            prefix,
            self.ignored_layers,
            self.packed_modules_mapping,
            **skip_kwargs,
        ):
            if is_linear:
                return AscendUnquantizedLinearMethod()
            return AscendUnquantizedFusedMoEMethod(layer.moe_config, tid2eid)

        self._verify_block_quantization()
        logger.info_once("Using the vLLM Ascend block-wise fp8 Quantization now!")

        if is_linear:
            scheme_class = get_scheme_class(FP8_METHOD, "linear")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/linear"
            return AscendLinearMethod(scheme_class(self.weight_block_size))

        if is_moe:
            scheme_class = get_scheme_class(FP8_METHOD, "moe")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/moe"
            return AscendFusedMoEMethod(
                scheme_class(self.weight_block_size, layer.moe_config),
                layer.moe_config,
                tid2eid=tid2eid,
            )

        return None


@register_quantization_config("deepseek_v4_fp8")
class AscendDeepseekV4FP8Config(DeepseekV4FP8Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_description = {}

    # DeepSeek ships each V4 revision under its own model_type ("deepseek_v4", "deepseek_v41", plus the
    # "..._text" text-only towers). Match the family prefix rather than one exact name -- upstream's
    # exact ``model_type == "deepseek_v4"`` test is precisely what let V4.1 regress unnoticed.
    _DS_V4_FAMILY_PREFIX = "deepseek_v4"

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        """Claim DeepSeek V4-family native FP8 checkpoints.

        Upstream ``DeepseekV4FP8Config.override_quantization_method``
        (``vllm/models/deepseek_v4/quant_config.py``) hard-codes ``model_type == "deepseek_v4"``, so
        DeepSeek-V4.1-Flash -- which ships as ``deepseek_v41`` -- was never claimed and fell through to
        the generic ``AscendFp8Config``, turning the MXFP4 routed experts into FP8 ones.
        """
        claimed = super().override_quantization_method(hf_quant_cfg, user_quant, hf_config=hf_config)
        if claimed is not None:
            return claimed
        if not isinstance(hf_quant_cfg, dict) or hf_quant_cfg.get("quant_method") not in ("fp8", "deepseek_v4_fp8"):
            return None

        model_type = getattr(hf_config, "model_type", None) or ""
        if model_type.startswith(cls._DS_V4_FAMILY_PREFIX):
            return "deepseek_v4_fp8"

        # Structural fallback, for a revision that renames itself out of the "deepseek_v4*" prefix:
        # DeepSeek-native checkpoints carry e8m0 scales and name their expert dtype, and the generic HF
        # fp8 scheme emits neither.
        if (
            "deepseek" in model_type.lower()
            and hf_quant_cfg.get("scale_fmt") == "ue8m0"
            and hf_quant_cfg.get("expert_dtype") is not None
        ):
            logger.warning(
                "Claiming model_type=%r for the DeepSeek V4 FP8 path on checkpoint layout alone "
                "(scale_fmt=ue8m0, expert_dtype=%r). Add it to _DS_V4_FAMILY_PREFIX if this revision "
                "is here to stay.",
                model_type,
                hf_quant_cfg.get("expert_dtype"),
            )
            return "deepseek_v4_fp8"
        return None

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
        tid2eid=None,
    ) -> Optional["QuantizeMethodBase"]:
        from ..method_adapters import (
            AscendFusedMoEMethod,
            AscendLinearMethod,
        )

        if isinstance(layer, LinearBase):
            scheme_class = get_scheme_class(FP8_METHOD, "ds_linear")
            assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_linear"
            quant_method = AscendLinearMethod(scheme_class(self.weight_block_size))
            return quant_method
        if is_fused_moe_layer(layer):
            if self.expert_dtype == "fp4":
                scheme_class = get_scheme_class(FP8_METHOD, "ds_w4a8_moe")
                assert scheme_class is not None, f"No scheme registered for {FP8_METHOD}/ds_w4a8_moe"
            else:
                raise NotImplementedError
            quant_method = AscendFusedMoEMethod(scheme_class(), layer.moe_config, tid2eid=tid2eid)
            return quant_method
        return None
