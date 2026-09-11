# SPDX-License-Identifier: Apache-2.0
"""RoPE group registration: the half that raises, and the half that does not.

The half that raises is the group name. dsa_v1 asks get_full_cos_and_sin_dsa
for f"c{ratio}" on every layer with compressor_ratio > 1, and V4.1 registered
nothing at all, so ratio-2 layers died with "RoPE group c2 is not registered".

The half that does not raise is the base. Upstream picks it per layer --
compress_rope_theta when the layer compresses, rope_theta when it does not, and
in this checkpoint that is 160000 against 10000. Register the name and build
its table from the wrong one and there is no error anywhere: every compressed
key is simply rotated by the wrong angle. That is the test that matters here.
"""

from types import SimpleNamespace

import pytest

from vllm_ascend.models.deepseek_v4_1.rope_groups import (register_rope_groups,
                                                          rope_group_names)

ROPE_THETA = 10000
COMPRESS_ROPE_THETA = 160000
ROPE_DIM = 64


def _rope(base):
    """Stands in for what build_deepseek_v4_rope returns."""
    return SimpleNamespace(rotary_dim=ROPE_DIM, max_position_embeddings=4096,
                           base=base, scaling_factor=1.0, beta_fast=32,
                           beta_slow=1)


@pytest.mark.parametrize("ratio,expected", [
    (0, ["default"]),
    (1, ["default"]),
    (2, ["default", "c2"]),
    (4, ["default", "c4"]),
])
def test_which_groups_a_layer_registers(ratio, expected):
    assert rope_group_names(ratio) == expected


def test_the_compressed_group_is_named_what_the_builder_asks_for():
    """dsa_v1.py builds the name with f"c{self.compressor_ratio}".

    Pinned as a string because the two sites are in different files and
    nothing connects them but this spelling.
    """
    assert "c2" in rope_group_names(2)


@pytest.fixture
def registry(monkeypatch):
    from vllm_ascend.ops import rope_dsv4

    recorded = []

    class _Fake:
        def __init__(self, **kwargs):
            recorded.append(kwargs)

    monkeypatch.setattr(rope_dsv4, "ComplexExpRotaryEmbedding", _Fake)
    return recorded


def _register(registry, ratio, base):
    register_rope_groups(SimpleNamespace(), layername="layers.2.attn",
                         rotary_emb=_rope(base), compress_ratio=ratio,
                         head_size=ROPE_DIM)
    return registry[-1]


def test_a_compressing_layer_registers_c2(registry):
    call = _register(registry, 2, COMPRESS_ROPE_THETA)
    assert call["rope_groups"] == ["default", "c2"]
    assert call["layername"] == "layers.2.attn"


def test_the_table_is_built_from_the_base_upstream_chose(registry):
    """The half that does not raise.

    Nothing downstream compares these numbers; a table built from rope_theta
    for a compressed layer is a valid table of the wrong frequencies.
    """
    call = _register(registry, 2, COMPRESS_ROPE_THETA)
    assert call["base"] == COMPRESS_ROPE_THETA
    assert call["base"] != ROPE_THETA


def test_a_non_compressing_layer_keeps_its_own_base(registry):
    """The same guard in the other direction.

    Layers 20..39 carry ratio 1 and upstream builds them with rope_theta.
    Reading the compressed base here would be equally silent.
    """
    call = _register(registry, 1, ROPE_THETA)
    assert call["base"] == ROPE_THETA
    assert call["rope_groups"] == ["default"]


def test_every_parameter_comes_from_upstreams_rope(registry):
    """Not a second derivation of build_deepseek_v4_rope's choices.

    Two derivations agree on the day the second is written. This pins that the
    values are read off the object upstream built.
    """
    rope = _rope(COMPRESS_ROPE_THETA)
    register_rope_groups(SimpleNamespace(), layername="layers.2.attn",
                         rotary_emb=rope, compress_ratio=2, head_size=ROPE_DIM)
    call = registry[-1]
    assert call["rotary_dim"] == rope.rotary_dim
    assert call["max_position_embeddings"] == rope.max_position_embeddings
    assert call["base"] == rope.base
    assert call["scaling_factor"] == rope.scaling_factor
    assert call["beta_fast"] == rope.beta_fast
    assert call["beta_slow"] == rope.beta_slow


def test_a_missing_parameter_is_refused_rather_than_defaulted(registry):
    """Every default available here is a plausible number.

    32 and 1 are the beta defaults, 1.0 is a plausible scaling factor. Each
    would build a table that differs from the model's own and raise nothing, so
    an absent attribute has to stop the layer instead.
    """
    incomplete = SimpleNamespace(rotary_dim=ROPE_DIM,
                                 max_position_embeddings=4096,
                                 base=COMPRESS_ROPE_THETA, scaling_factor=1.0,
                                 beta_fast=32)  # beta_slow missing
    with pytest.raises(AttributeError, match="beta_slow"):
        register_rope_groups(SimpleNamespace(), layername="layers.2.attn",
                             rotary_emb=incomplete, compress_ratio=2,
                             head_size=ROPE_DIM)
    assert not registry, "nothing may be registered from incomplete parameters"


def test_the_refusal_says_what_a_default_would_cost(registry):
    incomplete = SimpleNamespace(rotary_dim=ROPE_DIM)
    with pytest.raises(AttributeError) as exc:
        register_rope_groups(SimpleNamespace(), layername="layers.2.attn",
                             rotary_emb=incomplete, compress_ratio=2,
                             head_size=ROPE_DIM)
    msg = str(exc.value)
    assert "layers.2.attn" in msg
    assert "wrong angle" in msg


def test_the_consumer_asks_for_the_name_this_produces():
    """Close the loop against the caller, not against a remembered string.

    dsa_v1 builds the group name itself; if that spelling changes, the two
    sides drift with no error until a ratio-2 layer runs.
    """
    import inspect

    from vllm_ascend.attention import dsa_v1

    source = inspect.getsource(dsa_v1.AscendDSAMetadataBuilder)
    assert 'f"c{self.compressor_ratio}"' in source, (
        "the metadata builder no longer spells the group name this way; "
        "rope_group_names has to follow it")
