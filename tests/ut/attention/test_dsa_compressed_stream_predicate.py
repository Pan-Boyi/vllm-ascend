# SPDX-License-Identifier: Apache-2.0
"""Which layers read a compressed stream, asked once instead of three times.

Three call sites decide the same thing -- whether to pass the compressed
cumulative lengths, whether to supply the stream, and whether the mask may be
sliding-window-only -- and each used its own comparison. Two of them kept
V4.0's answer after the third was corrected, and the disagreement is silent:
the operator accepts a layer that was handed no compressed stream.
"""

import pytest

from vllm_ascend.attention.dsa_v1 import attends_compressed_stream

#: Every ratio V4.0 uses. Small enough to enumerate, which is the point: the
#: claim "V4.0 behaviour is unchanged" is checked rather than argued.
V40_RATIOS = (0, 1, 4, 128)
#: V4.1's, from compress_ratios: 0 on layers 0-1, 2 on 2-19, 1 on 20-39.
V41_RATIOS = (0, 1, 2)


class _HfConfig:
    def __init__(self, kv_source_layer_ids=None):
        if kv_source_layer_ids is not None:
            self.kv_source_layer_ids = kv_source_layer_ids


class _ModelConfig:
    def __init__(self, hf_config):
        self.hf_config = hf_config


class _VllmConfig:
    def __init__(self, kv_source_layer_ids=None):
        self.model_config = _ModelConfig(_HfConfig(kv_source_layer_ids))


V40 = _VllmConfig()
V41 = _VllmConfig([2, 8, 14, 20])


@pytest.mark.parametrize("ratio", V40_RATIOS)
def test_v40_behaviour_is_unchanged(ratio):
    """Enumerated, not argued.

    The expression being replaced was ``compress_ratio <= 1`` guarding the
    *absence* of a stream, so the predicate must equal its negation on every
    value V4.0 can produce.
    """
    assert attends_compressed_stream(ratio, V40) == (not (ratio <= 1))


@pytest.mark.parametrize("ratio,expected", [(0, False), (1, True), (2, True)])
def test_v41_reads_a_stream_at_ratio_one(ratio, expected):
    """Ratio 1 means "compressed, pooling disabled" here, not "no stream".

    Layers 20..39 carry it -- half the network. Under V4.0's reading they were
    handed no compressed stream and the operator did not complain.
    """
    assert attends_compressed_stream(ratio, V41) is expected


def test_the_two_readings_differ_exactly_at_ratio_one():
    differ = [r for r in (0, 1, 2, 4, 128)
              if attends_compressed_stream(r, V40)
              != attends_compressed_stream(r, V41)]
    assert differ == [1]


def test_keyed_on_the_field_that_answers_the_question():
    """Not on a version label.

    A config naming its source layers has the V4.1 topology whatever its
    model_type says, and labels have drifted from the code repeatedly here.
    """
    unlabelled = _VllmConfig([2, 8, 14, 20])
    assert attends_compressed_stream(1, unlabelled) is True
    assert not hasattr(unlabelled.model_config.hf_config, "model_type")


def test_missing_config_falls_back_to_v40():
    """A half-built config must not silently claim the V4.1 topology."""

    class _Empty:
        pass

    assert attends_compressed_stream(1, _Empty()) is False
    assert attends_compressed_stream(4, _Empty()) is True
