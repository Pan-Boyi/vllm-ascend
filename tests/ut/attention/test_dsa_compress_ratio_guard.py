# SPDX-License-Identifier: Apache-2.0
"""The compression-ratio guard must reject what no dispatch path implements.

Before this guard, an unsupported ratio fell through to a default at every
dispatch site: the cmp_ratio mapping sent it to 128 and the backend selection
sent it to SWA. DeepSeek-V4.1 needs ratio 2 on layers 2..19, so the model
would have run at 128:1 compression and produced quietly wrong numbers.
"""

import pytest

from vllm_ascend.attention.dsa_v1 import (SUPPORTED_COMPRESS_RATIOS,
                                          validate_compress_ratio)


@pytest.mark.parametrize("ratio", [0, 1, 4, 128])
def test_accepts_ratios_v40_actually_uses(ratio):
    assert validate_compress_ratio(ratio, "layer") == ratio


def test_rejects_v41_ratio_2():
    # The whole point: this used to be silently mapped to 128.
    with pytest.raises(ValueError, match="compress_ratio=2 is not supported"):
        validate_compress_ratio(2, "model.layers.2.self_attn.attn")


def test_error_names_the_layer_and_the_supported_set():
    with pytest.raises(ValueError) as exc:
        validate_compress_ratio(2, "model.layers.7.self_attn.attn")
    msg = str(exc.value)
    assert "model.layers.7.self_attn.attn" in msg
    assert "[0, 1, 4, 128]" in msg


def test_error_does_not_read_as_a_hardware_limit():
    """Ratio 2 is a binding gap, not something A5 cannot do.

    CANN's MixedQuantSparseFlashMla accepts ratio 2 on A5; it simply is not
    bound in this repo yet. Without saying so, the next person to hit this
    error concludes the hardware cannot do it and designs around a limit that
    does not exist.
    """
    with pytest.raises(ValueError) as exc:
        validate_compress_ratio(2, "layer")
    msg = str(exc.value)
    assert "not a hardware limit" in msg
    assert "MixedQuantSparseFlashMla" in msg


@pytest.mark.parametrize("ratio", [3, 8, 16, 32, 64, 256])
def test_rejects_other_unimplemented_ratios(ratio):
    with pytest.raises(ValueError):
        validate_compress_ratio(ratio, "layer")


def test_supported_set_matches_operator_check():
    # csrc/.../kv_quant_sparse_attn_sharedkv_check_existance.cpp:154-166 --
    # SWA accepts {1, 4, 128}, CFA and SCFA accept {4, 128}; 0 means no
    # compressed stream and never reaches the operator.
    assert set(SUPPORTED_COMPRESS_RATIOS) == {0, 1, 4, 128}
