from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from tests.ut.base import TestBase
from tests.ut.quantization.conftest_quantization import (
    create_mock_ascend_config,
    create_mock_vllm_config,
    create_mxfp_moe_layer,
)
from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
    AscendW8A8MXFP8DSDynamicLinearMethod,
    AscendW8A8MXFP8DynamicFusedMoEMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)


class TestAscendW8A8MXFP8LinearMethod(TestBase):
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config")
    def setUp(self, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        self.scheme = AscendW8A8MXFP8DynamicLinearMethod()

    def test_modelopt_config_defaults_group_size(self):
        vllm_config = create_mock_vllm_config()
        vllm_config.quant_config = SimpleNamespace()
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
                return_value=vllm_config,
            ),
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
                return_value=0,
            ),
        ):
            scheme = AscendW8A8MXFP8DynamicLinearMethod()

        self.assertEqual(scheme.group_size, 32)

    def test_get_weight_various_input_sizes(self):
        sizes = [(128, 64), (512, 256), (1024, 512)]
        for input_size, output_size in sizes:
            result = self.scheme.get_weight(input_size, output_size, torch.bfloat16)
            self.assertEqual(result["weight"].shape, (output_size, input_size))
            self.assertEqual(result["weight"].dtype, torch.float8_e4m3fn)

    def test_get_pergroup_param_group_size_variations(self):
        group_sizes = [16, 32, 64, 128]
        for gs in group_sizes:
            self.scheme.group_size = gs
            result = self.scheme.get_pergroup_param(256, 128, torch.bfloat16)
            self.assertEqual(result["weight_scale"].shape, (128, 256 // gs))
            self.assertEqual(result["weight_scale"].dtype, torch.uint8)

    def test_process_weights_stores_original_shapes(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(128, 256).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "_mxfp8_original_shapes"))
        self.assertEqual(layer._mxfp8_original_shapes["weight"], (128, 256))
        self.assertTrue(layer._mxfp8_transformed)
        self.assertEqual(layer.weight_scale.shape, (4, 128, 2))
        self.assertTrue(layer.weight.data.is_contiguous())
        self.assertTrue(layer.weight_scale.data.is_contiguous())

    def test_restore_after_process_returns_original_shape(self):
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(128, 256).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        original_weight_shape = layer.weight.shape
        original_scale_shape = layer.weight_scale.shape
        self.scheme.process_weights_after_loading(layer)
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(layer.weight.shape, original_weight_shape)
        self.assertEqual(layer.weight_scale.shape, original_scale_shape)
        self.assertFalse(layer._mxfp8_transformed)

    def test_transform_buffer_data_ptr_stable_across_reloads(self):
        # The transformed buffer is what the ACL graph captures and replays.
        # It must keep a stable data_ptr across RL weight reloads so graph
        # replay never reads stale/freed memory and produces garbled output.
        # The transformed weight/scale must also stay contiguous, since ACL
        # graph capture expects contiguous weight/scale tensors.
        layer = nn.Module()
        layer.weight = nn.Parameter(
            torch.randint(0, 255, (128, 256), dtype=torch.uint8).to(torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (128, 8), dtype=torch.uint8), requires_grad=False)
        self.scheme.process_weights_after_loading(layer)
        transform_weight_ptr = layer._mxfp8_weight_buf.data_ptr()
        transform_scale_ptr = layer._mxfp8_scale_buf.data_ptr()
        # Buffers and the weight/scale views over them must be contiguous.
        self.assertTrue(layer._mxfp8_weight_buf.is_contiguous())
        self.assertTrue(layer._mxfp8_scale_buf.is_contiguous())
        self.assertTrue(layer.weight.data.is_contiguous())
        self.assertTrue(layer.weight_scale.data.is_contiguous())
        for _ in range(3):
            self.scheme.restore_weights_for_rl_loading(layer)
            # Simulate model.load_weights() writing new data via copy_.
            new_w = torch.randint(0, 255, layer.weight.shape, dtype=torch.uint8).to(torch.float8_e4m3fn)
            new_s = torch.randint(0, 255, layer.weight_scale.shape, dtype=torch.uint8)
            layer.weight.data.copy_(new_w)
            layer.weight_scale.data.copy_(new_s)
            self.scheme.process_weights_after_loading(layer)
            self.assertEqual(layer._mxfp8_weight_buf.data_ptr(), transform_weight_ptr)
            self.assertEqual(layer._mxfp8_scale_buf.data_ptr(), transform_scale_ptr)
            self.assertEqual(layer.weight.data.data_ptr(), transform_weight_ptr)
            self.assertEqual(layer.weight_scale.data.data_ptr(), transform_scale_ptr)
            # Contiguity must be preserved across reloads.
            self.assertTrue(layer._mxfp8_weight_buf.is_contiguous())
            self.assertTrue(layer._mxfp8_scale_buf.is_contiguous())
            self.assertTrue(layer.weight.data.is_contiguous())
            self.assertTrue(layer.weight_scale.data.is_contiguous())

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.torch_npu")
    def test_apply(self, mock_torch_npu):
        dynamic_scale = torch.randint(0, 255, (32, 8), dtype=torch.uint8)
        mock_torch_npu.npu_dynamic_mx_quant.return_value = (
            torch.randint(0, 255, (32, 256), dtype=torch.uint8),
            dynamic_scale,
        )
        mock_torch_npu.npu_quant_matmul.return_value = torch.randn(32, 128, dtype=torch.float16)
        layer = nn.Module()
        layer.weight = nn.Parameter(torch.randn(256, 128).to(torch.float8_e4m3fn), requires_grad=False)
        layer.weight_scale = nn.Parameter(torch.randint(0, 255, (4, 128, 2), dtype=torch.uint8), requires_grad=False)
        x = torch.randn(32, 1, 256, dtype=torch.float16)
        bias = torch.randn(128, dtype=torch.float16)
        output = self.scheme.apply(layer, x, bias)
        self.assertEqual(output.shape, (32, 1, 128))
        dynamic_quant_kwargs = mock_torch_npu.npu_dynamic_mx_quant.call_args.kwargs
        self.assertEqual(dynamic_quant_kwargs["scale_alg"], self.scheme.dynamic_mx_quant_scale_alg)
        call_kwargs = mock_torch_npu.npu_quant_matmul.call_args.kwargs
        self.assertEqual(call_kwargs["bias"].dtype, torch.float32)
        self.assertEqual(call_kwargs["group_sizes"], [1, 1, self.scheme.group_size])
        self.assertEqual(call_kwargs["output_dtype"], torch.float16)


class TestAscendW8A8MXFP8MoEMethod(TestBase):
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config")
    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_ascend_config")
    def setUp(self, mock_ascend, mock_vllm):
        mock_vllm.return_value = create_mock_vllm_config()
        mock_ascend.return_value = create_mock_ascend_config()
        self.scheme = AscendW8A8MXFP8DynamicFusedMoEMethod()

    def test_modelopt_config_defaults_group_size(self):
        vllm_config = create_mock_vllm_config()
        vllm_config.quant_config = SimpleNamespace()
        vllm_config.use_v2_model_runner = True
        with (
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
                return_value=vllm_config,
            ),
            patch(
                "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_ascend_config",
                return_value=create_mock_ascend_config(),
            ),
        ):
            scheme = AscendW8A8MXFP8DynamicFusedMoEMethod()

        self.assertEqual(scheme.group_size, 32)

    def test_get_weight_various_expert_counts(self):
        for num_experts in [4, 8, 16]:
            result = self.scheme.get_weight(num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16)
            self.assertEqual(result["w13_weight"].shape[0], num_experts)
            self.assertEqual(result["w2_weight"].dtype, torch.float8_e4m3fn)

    def test_get_dynamic_quant_param_dtype_uint8(self):
        result = self.scheme.get_dynamic_quant_param(
            self.num_experts, self.intermediate_size, self.hidden_size, torch.bfloat16
        )
        self.assertEqual(result["w13_weight_scale"].shape, (8, 512, 4))
        self.assertEqual(result["w2_weight_scale"].dtype, torch.uint8)

    def test_process_weights_stores_original_shapes(self):
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        original_shape = layer.w13_weight.shape
        self.scheme.process_weights_after_loading(layer)
        self.assertTrue(hasattr(layer, "_mxfp8_original_shapes"))
        self.assertIn("w13_weight", layer._mxfp8_original_shapes)
        self.assertEqual(layer.w13_weight.shape, (original_shape[0], original_shape[2], original_shape[1]))
        self.assertFalse(layer.w13_weight.data.is_contiguous())
        self.assertFalse(layer.w2_weight.data.is_contiguous())
        self.assertFalse(layer.w13_weight_scale.data.is_contiguous())
        self.assertFalse(layer.w2_weight_scale.data.is_contiguous())

        weight_views = self.scheme.get_eplb_weight_views(layer)
        self.assertTrue(self.scheme.supports_eplb)
        self.assertEqual(len(weight_views), 4)
        for source, weight_view in zip(
            [layer.w13_weight, layer.w2_weight, layer.w13_weight_scale, layer.w2_weight_scale],
            weight_views,
        ):
            self.assertTrue(weight_view.is_contiguous())
            self.assertEqual(weight_view.shape[0], self.num_experts)
            self.assertEqual(weight_view.untyped_storage().data_ptr(), source.untyped_storage().data_ptr())

    def test_restore_weights_for_rl_loading(self):
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        original_w13_shape = layer.w13_weight.shape
        self.scheme.process_weights_after_loading(layer)
        self.assertNotEqual(layer.w13_weight.shape, original_w13_shape)
        self.scheme.restore_weights_for_rl_loading(layer)
        self.assertEqual(layer.w13_weight.shape, original_w13_shape)

    @patch("vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8._EXTRA_CTX")
    def test_apply_full_params(self, mock_ctx):
        tokens = 4
        layer = create_mxfp_moe_layer(
            num_experts=self.num_experts, hidden_size=self.hidden_size, intermediate_size=self.intermediate_size
        )
        self.scheme.process_weights_after_loading(layer)
        layer.swiglu_limit = 1000000
        x = torch.randn(tokens, self.hidden_size, dtype=torch.bfloat16)
        topk_weights = torch.randn(tokens, 2)
        topk_ids = torch.randint(0, self.num_experts, (tokens, 2))
        layer.activation = "silu"
        layer.ascend_pertoken_scale = torch.randn(tokens)
        layer.apply_router_weight_on_input = True
        layer.ascend_expert_map = None
        layer.global_redundant_expert_num = 0
        layer.log2phy = None
        layer.ascend_mc2_mask = None
        layer.swiglu_alpha = 1.0
        layer.swiglu_beta = 0.0
        mock_comm = Mock()
        mock_comm.fused_experts.return_value = torch.randn(tokens, self.hidden_size)
        mock_ctx.moe_comm_method = mock_comm
        mock_ctx.moe_comm_type = Mock()
        self.scheme.apply(
            layer,
            x,
            topk_weights,
            topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
        mock_comm.fused_experts.assert_called_once()


# Real (weight, scale) shapes read out of the checkpoint safetensors headers. Both revisions store one
# e8m0 scale per square weight_block_size tile; only the tile size differs.
V40_REAL_SHAPES = [  # DeepSeek-V4-Flash-0731, weight_block_size [128, 128]
    ("layers.0.attn.wkv", (512, 4096), (4, 32)),
    ("layers.0.attn.wo_b", (4096, 8192), (32, 64)),
    ("layers.0.attn.wq_b", (32768, 1024), (256, 8)),
]
V41_REAL_SHAPES = [  # DeepSeek-V4.1-Flash, weight_block_size [32, 32]
    ("layers.0.attn.wkv", (512, 5120), (16, 160)),
    ("layers.0.attn.wo_b", (5120, 8192), (160, 256)),
    ("layers.0.ffn.shared_experts.w1", (2304, 5120), (72, 160)),
    ("layers.0.ffn.shared_experts.w2", (5120, 2304), (160, 72)),
]

GROUP_SIZE = 32


def _ds_linear_scheme(block_size, o_groups=8, o_lora_rank=1024, tp_size=1):
    """Build AscendW8A8MXFP8DSDynamicLinearMethod without a live vllm config."""
    return _ds_linear_scheme_from_block_size([block_size, block_size], o_groups, o_lora_rank, tp_size)


def _ds_linear_scheme_from_block_size(weight_block_size, o_groups=8, o_lora_rank=1024, tp_size=1):
    vllm_config = SimpleNamespace(
        quant_config=SimpleNamespace(quant_description={"group_size": GROUP_SIZE}),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(o_groups=o_groups, o_lora_rank=o_lora_rank)),
    )
    with (
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_current_vllm_config",
            return_value=vllm_config,
        ),
        patch(
            "vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8.get_dynamic_mx_quant_scale_alg",
            return_value=0,
        ),
    ):
        return AscendW8A8MXFP8DSDynamicLinearMethod(weight_block_size)


def _run_process_weights(scheme, weight_shape, scale_shape, prefix="layers.0.attn.wkv", scale_dtype=None):
    """Feed one layer through process_weights_after_loading; returns it plus the uint8 exponents fed in.

    The checkpoint stores ``.scale`` as F8_E8M0, which is what the e8m0 branch of
    process_weights_after_loading expects; ``scale_dtype=torch.float32`` exercises the other branch,
    which recovers the same exponents out of float32 scales.
    """
    scale_dtype = scale_dtype or torch.float8_e8m0fnu
    out_features, in_features = weight_shape
    layer = nn.Module()
    layer.prefix = prefix
    layer.weight = nn.Parameter(
        torch.zeros(out_features, in_features).to(torch.float8_e4m3fn), requires_grad=False
    )
    # A distinct value per tile, so a mis-sized broadcast cannot pass by accident. Exponents are kept
    # in [1, 200] so they stay exactly representable both as e8m0 and as a float32 power of two.
    exponents = (torch.arange(scale_shape[0] * scale_shape[1], dtype=torch.int32).reshape(scale_shape) % 200) + 1
    if scale_dtype == torch.float32:
        # float32 with mantissa 0: the bias-127 exponent field is what the else branch reads back.
        scale = (exponents.to(torch.int32) << 23).view(torch.float32)
    else:
        scale = exponents.to(torch.uint8).view(scale_dtype)
    layer.weight_scale = nn.Parameter(scale, requires_grad=False)
    scheme.process_weights_after_loading(layer)
    return layer, exponents.to(torch.uint8)


class TestAscendW8A8MXFP8DSDynamicLinearMethodBlockSize(TestBase):
    """The block-scale broadcast must follow weight_block_size, not V4.0's constants.

    ``process_weights_after_loading`` used to hard-code ``repeat_interleave(4, dim=1)`` and
    ``repeat_interleave(128, dim=0)`` -- the factors for V4.0's 128 x 128 tiles. DeepSeek-V4.1-Flash
    ships 32 x 32 tiles, so both factors were wrong by 4x.
    """

    def _expected_expansion(self, scale, block_size):
        """The layout npu_quant_matmul consumes: one scale per group_size elements along K."""
        return scale.repeat_interleave(block_size // GROUP_SIZE, dim=1).repeat_interleave(block_size, dim=0)

    def test_v41_real_shapes_expand_to_the_gemm_layout(self):
        scheme = _ds_linear_scheme(32)
        for name, weight_shape, scale_shape in V41_REAL_SHAPES:
            with self.subTest(tensor=name):
                out_features, in_features = weight_shape
                # The allocated parameter must match the checkpoint exactly, or loading cannot work.
                allocated = scheme.get_pergroup_param(in_features, out_features, torch.bfloat16)
                self.assertEqual(tuple(allocated["weight_scale"].shape), scale_shape)

                layer, scale = _run_process_weights(scheme, weight_shape, scale_shape)
                expected = self._expected_expansion(scale, 32)
                self.assertEqual(tuple(expected.shape), (out_features, in_features // GROUP_SIZE))
                # process_weights_after_loading finishes with reshape(N, K//2, 2) then transpose(0, 1).
                expected = expected.reshape(out_features, in_features // GROUP_SIZE // 2, 2).transpose(0, 1)
                self.assertEqual(layer.weight_scale.data.shape, expected.shape)
                self.assertTrue(torch.equal(layer.weight_scale.data, expected))

    def test_v40_real_shapes_still_match_the_old_hard_coded_factors(self):
        """V4.0 behaviour must be bit-identical to the constants that were replaced."""
        scheme = _ds_linear_scheme(128)
        for name, weight_shape, scale_shape in V40_REAL_SHAPES:
            with self.subTest(tensor=name):
                out_features, in_features = weight_shape
                allocated = scheme.get_pergroup_param(in_features, out_features, torch.bfloat16)
                self.assertEqual(tuple(allocated["weight_scale"].shape), scale_shape)

                layer, scale = _run_process_weights(scheme, weight_shape, scale_shape)
                # Literally the replaced expression.
                legacy = scale.repeat_interleave(4, dim=1).repeat_interleave(128, dim=0)
                legacy = legacy.reshape(out_features, in_features // GROUP_SIZE // 2, 2).transpose(0, 1)
                self.assertTrue(torch.equal(layer.weight_scale.data, legacy))

    def test_each_tile_scale_lands_on_its_own_rows_and_k_groups(self):
        """Shape alone is not enough -- check the values land where the GEMM will look for them."""
        block_size = 32
        scheme = _ds_linear_scheme(block_size)
        out_features, in_features = 128, 256
        scale_shape = (out_features // block_size, in_features // block_size)
        layer, scale = _run_process_weights(scheme, (out_features, in_features), scale_shape)

        expanded = layer.weight_scale.data.transpose(0, 1).reshape(out_features, in_features // GROUP_SIZE)
        for tile_row in range(scale_shape[0]):
            for tile_col in range(scale_shape[1]):
                rows = slice(tile_row * block_size, (tile_row + 1) * block_size)
                groups_per_tile = block_size // GROUP_SIZE
                cols = slice(tile_col * groups_per_tile, (tile_col + 1) * groups_per_tile)
                self.assertTrue(
                    torch.equal(
                        expanded[rows, cols],
                        scale[tile_row, tile_col].expand(block_size, groups_per_tile),
                    ),
                    f"tile ({tile_row}, {tile_col}) was broadcast to the wrong place",
                )

    def test_float32_scales_recover_the_same_exponents(self):
        """The non-e8m0 branch reads the exponent field out of float32 and must agree."""
        scheme = _ds_linear_scheme(32)
        weight_shape, scale_shape = (512, 5120), (16, 160)
        e8m0_layer, _ = _run_process_weights(scheme, weight_shape, scale_shape)
        f32_layer, _ = _run_process_weights(scheme, weight_shape, scale_shape, scale_dtype=torch.float32)
        self.assertTrue(torch.equal(e8m0_layer.weight_scale.data, f32_layer.weight_scale.data))

    def test_wo_a_keeps_its_block_diagonal_grouping(self):
        """wo_a is a grouped (block-diagonal) projection, so its scales get regrouped as well."""
        o_groups, o_lora_rank = 8, 1024
        scheme = _ds_linear_scheme(32, o_groups=o_groups, o_lora_rank=o_lora_rank)
        layer, _ = _run_process_weights(scheme, (8192, 4096), (256, 128), prefix="layers.0.attn.wo_a")
        self.assertEqual(layer.weight_scale.data.shape[0], o_groups)
        self.assertEqual(layer.weight.data.shape[0], o_groups)

    def test_rejects_non_square_blocks(self):
        with self.assertRaises(NotImplementedError):
            _ds_linear_scheme_from_block_size([32, 64])

    def test_rejects_block_smaller_than_the_group(self):
        with self.assertRaises(NotImplementedError):
            _ds_linear_scheme_from_block_size([16, 16])
