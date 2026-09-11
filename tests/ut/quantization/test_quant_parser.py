import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.quant_parser import (
    QuantTypeMapping,
    get_rollback_quant_type,
    parse_mxfp_quant_params,
    parse_quant_moe_down_proj_params,
)

# The one branch whose round mode the caller is allowed to choose. Everything else is forced to
# "rint" (round-half-to-even), and that forcing is load-bearing rather than tidy -- see
# TestRoundModeForcingIsExhaustive.
ROUND_MODE_IS_CALLER_CHOSEN = {"W4A4_MXFP4"}


class TestGetRollbackQuantType(TestBase):
    def test_returns_down_proj_quant_type(self):
        config = {
            "model.layers.0.mlp.gate_proj": "W8A8_MXFP8",
            "model.layers.0.mlp.down_proj": "W4A4_MXFP4",
        }
        result = get_rollback_quant_type(config)
        self.assertEqual(result, "W4A4_MXFP4")

    def test_returns_default_when_no_down_proj(self):
        config = {"model.layers.0.mlp.gate_proj": "W4A8_MXFP"}
        result = get_rollback_quant_type(config)
        self.assertEqual(result, "W8A8_MXFP8")


class TestParseMxfpQuantParams(TestBase):
    def test_default_values(self):
        act, weight, scale, per_token, round_mode = parse_mxfp_quant_params()
        self.assertIsNotNone(act)
        self.assertIsNotNone(weight)
        self.assertIsNotNone(round_mode)


class TestParseQuantMoeDownProjParams(TestBase):
    def test_w8a8_mxfp8_uses_rint_round_mode(self):
        act, weight, scale, per_token, round_mode = parse_quant_moe_down_proj_params("W8A8_MXFP8", "round")
        self.assertEqual(round_mode, "rint")

    def test_w4a4_mxfp4_respects_parsed_round_mode(self):
        act, weight, scale, per_token, round_mode = parse_quant_moe_down_proj_params("W4A4_MXFP4", "round")
        self.assertEqual(round_mode, "round")


class TestRoundModeForcingIsExhaustive(TestBase):
    """Every non-W4A4 path must discard the caller's round mode.

    The two hand-written cases above name W8A8_MXFP8 and W4A4_MXFP4 and stop there. That leaves
    W4A8_MXFP -- the type DeepSeek-V4.1 actually runs -- with no assertion at all, and it leaves
    any type added to the mapping later with none either. These iterate the mapping instead, so a
    new entry is covered the moment it exists.

    🔴 Why the forcing matters. On the non-W4A4 paths the activation is FP8 and
    ``npu_dynamic_mx_quant`` is called without a round mode, so "rint" is what the hardware does
    anyway and the reference's round-half-to-even matches the real path. Hand it "round" (round
    away from zero) and the quantisation error stops being two-sided: every tie moves the same
    way, the deviation accumulates with depth instead of cancelling, and it presents as a model
    bug rather than as a rounding choice. dsv41-golden's protocol carries this as a dormant item
    for exactly that reason -- it is not currently reachable, and it becomes reachable the day
    someone deletes the else-branch below or routes a round mode into W4A8.

    So this is the thing that has to go red on that day.
    """

    def test_every_non_w4a4_type_discards_the_callers_round_mode(self):
        forced = []
        for quant_type in QuantTypeMapping.get_quant_settings():
            if quant_type in ROUND_MODE_IS_CALLER_CHOSEN:
                continue
            with self.subTest(quant_type=quant_type):
                *_, round_mode = parse_quant_moe_down_proj_params(quant_type, "round")
                self.assertEqual(
                    round_mode,
                    "rint",
                    f"{quant_type} let the caller's round mode through; on an FP8 activation path "
                    f"that turns a two-sided rounding error into a one-directional bias",
                )
                forced.append(quant_type)
        # Zero coverage must fail: an empty mapping, or one where every type became caller-chosen,
        # would otherwise pass this test without checking anything.
        self.assertGreater(len(forced), 1, f"only {forced} were checked; expected several types")

    def test_w4a8_mxfp_is_the_v41_path_and_quantises_activations_to_fp8(self):
        """The premise behind "rint is the only reachable mode" on our path.

        If this mapping ever put an FP4 activation type here, the round mode would start reaching
        a kernel that honours it, and the assertion above would stop being sufficient on its own.
        """
        config = QuantTypeMapping.get_quant_settings()["W4A8_MXFP"]
        self.assertIs(config["act_quant_type"], torch.float8_e4m3fn)

    def test_w4a4_is_still_the_only_caller_chosen_branch(self):
        """Pins the exception set itself, so widening it is a deliberate, visible edit."""
        caller_chosen = {
            quant_type
            for quant_type in QuantTypeMapping.get_quant_settings()
            if parse_quant_moe_down_proj_params(quant_type, "round")[-1] == "round"
        }
        self.assertEqual(caller_chosen, ROUND_MODE_IS_CALLER_CHOSEN)
