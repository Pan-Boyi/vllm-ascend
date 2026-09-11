# SPDX-License-Identifier: Apache-2.0
"""Ascend RoPE group registration for DeepSeek-V4.1 layers.

``AscendDSAMetadataBuilder`` asks for the compressed stream's frequency table by
group name -- ``dsa_v1.py`` builds ``f"c{ratio}"`` and hands it to
``get_full_cos_and_sin_dsa`` for every layer with ``compressor_ratio > 1``. That
registry is populated only by constructing an Ascend
``ComplexExpRotaryEmbedding``, which V4.0's model does and V4.1's does not: V4.1
builds upstream's rotary embedding through ``build_deepseek_v4_rope`` and never
touches the Ascend one. So a ratio-2 layer reaches the compressor metadata and
gets ``KeyError: RoPE group c2 is not registered.``

Two halves, and only the first one raises.

The second is the base. Upstream selects it per layer
(``models/deepseek_v4/common/rope.py:28``)::

    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta)

and in this checkpoint those are 160000 and 10000 -- a factor of sixteen.
Registering the group name while building its table from ``rope_theta`` raises
nothing and rotates every compressed key by the wrong angle.

So nothing here re-derives those parameters. They are read off the rotary
embedding upstream already built, which is the single place that decision is
made. A second derivation would be a second thing to keep in step, and the two
would agree on the day it was written.
"""

from __future__ import annotations

from typing import Any

#: Read off upstream's rotary embedding rather than recomputed. A missing one
#: is refused rather than defaulted: every default here is a plausible number
#: that produces a silently wrong table.
_ROPE_ATTRS = ("rotary_dim", "max_position_embeddings", "base",
               "scaling_factor", "beta_fast", "beta_slow")


def rope_group_names(compress_ratio: int) -> list[str]:
    """The groups a layer has to register.

    ``default`` always; the compressed stream's own group only when there is a
    compressed stream. Mirrors V4.0's ``model.py``, which is what the metadata
    builder was written against.
    """
    if compress_ratio > 1:
        return ["default", f"c{compress_ratio}"]
    return ["default"]


def _rope_parameters(rotary_emb: Any, where: str) -> dict[str, Any]:
    values = {}
    for name in _ROPE_ATTRS:
        if not hasattr(rotary_emb, name):
            raise AttributeError(
                f"{where}: the rotary embedding upstream built "
                f"({type(rotary_emb).__name__}) has no {name!r}, so the Ascend "
                "RoPE group cannot be registered from it. Supplying a default "
                "would build a frequency table that differs from the one the "
                "model's own RoPE uses -- which raises nothing and rotates "
                "every key by the wrong angle.")
        values[name] = getattr(rotary_emb, name)
    return values


def register_rope_groups(vllm_config, layername: str, rotary_emb: Any,
                         compress_ratio: int, head_size: int):
    """Register this layer's groups against upstream's own RoPE parameters.

    Returns the constructed ``ComplexExpRotaryEmbedding``. It is not used for
    the forward pass -- V4.1 keeps calling upstream's -- it exists so the
    Ascend registry has the table the compressor metadata reads.
    """
    from vllm_ascend.ops.rope_dsv4 import ComplexExpRotaryEmbedding

    params = _rope_parameters(rotary_emb, layername)
    return ComplexExpRotaryEmbedding(
        vllm_config=vllm_config,
        layername=layername,
        head_size=head_size,
        rotary_dim=params["rotary_dim"],
        max_position_embeddings=params["max_position_embeddings"],
        base=params["base"],
        scaling_factor=params["scaling_factor"],
        beta_fast=params["beta_fast"],
        beta_slow=params["beta_slow"],
        is_neox_style=False,
        rope_groups=rope_group_names(compress_ratio),
    )
