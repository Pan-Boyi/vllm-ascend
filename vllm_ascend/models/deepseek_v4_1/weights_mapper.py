# SPDX-License-Identifier: Apache-2.0
"""Checkpoint -> vLLM parameter name mapping for DeepSeek-V4.1-Flash.

The released checkpoint uses DeepSeek's own naming (``layers.N.attn.wq_a``,
``ffn``, ``attn_norm``, ...); vLLM expects its own (``model.layers.N.self_attn``,
``mlp``, ``input_layernorm``, ...). V4.0 does this rewriting inline in
``load_weights``; pulling it out here makes it testable against the real
checkpoint's 96085 tensor names without loading a single weight or touching an
NPU.

The property worth protecting is **exhaustiveness**. A name that matches no rule
must raise, never be skipped: a silently dropped weight leaves a randomly
initialised parameter behind, the model still loads, still runs, and is simply
wrong -- the same failure shape as V4.0's ``spec_layer`` ``continue``, which is
how DSpark's ``mtp.1``/``mtp.2`` came to be allocated but never written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

# Where a checkpoint tensor belongs. DSpark is a separate 3-layer module loaded
# into its own drafter model, not into the 40-layer backbone.
MAIN = "main"
DSPARK = "dspark"
VISION = "vision"


@dataclass(frozen=True)
class MappedWeight:
    target: str
    name: str
    #: DSpark block index (0/1/2) for ``target == DSPARK``, else None.
    spec_layer: int | None = None


class UnmappedWeightError(KeyError):
    """A checkpoint tensor matched no rule.

    Raised rather than skipped -- see the module docstring.
    """


_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_MTP_RE = re.compile(r"^mtp\.(\d+)\.")

# MoE expert matrices. DeepSeek numbers them w1/w2/w3; vLLM wants the
# gate/down/up names its fused-MoE loader keys off.
_EXPERT_MATS = {
    ".w1.": ".gate_proj.",
    ".w2.": ".down_proj.",
    ".w3.": ".up_proj.",
}

# Top-level tensors that are not inside a layer.
_TOPLEVEL = {
    "embed.weight": "model.embed_tokens.weight",
    "head.weight": "lm_head.weight",
    "norm.weight": "model.norm.weight",
    # Multimodal sentinels, carried on the language model.
    "image_start": "model.image_start",
    "image_end": "model.image_end",
    "image_newline": "model.image_newline",
}


def _rewrite_submodules(rest: str) -> str:
    """Rewrite the part of a name after the ``layers.N.`` / ``mtp.N.`` prefix."""
    # Order matters: ``.attn_norm.`` and ``.ffn_norm.`` must be consumed before
    # the bare ``.attn.`` / ``.ffn.`` rules, or ``attn_norm`` would survive into
    # ``self_attn_norm``.
    if rest.startswith("attn_norm."):
        return "input_layernorm." + rest[len("attn_norm."):]
    if rest.startswith("ffn_norm."):
        return "post_attention_layernorm." + rest[len("ffn_norm."):]

    if rest.startswith("attn."):
        # Covers attn_sink, q_norm, kv_norm, wq_a/wq_b/wkv/wo_a/wo_b, and the
        # nested compressor.* / indexer.* trees.
        rest = "self_attn." + rest[len("attn."):]
    elif rest.startswith("ffn."):
        rest = "mlp." + rest[len("ffn."):]
        for ckpt, vllm in _EXPERT_MATS.items():
            if ckpt in rest:
                rest = rest.replace(ckpt, vllm)
                break
    # engram.* and the hc_* parameters keep their checkpoint names; they are
    # attributes of the decoder layer itself.
    return rest


def _rewrite_scale(name: str) -> str:
    """``.scale`` is vLLM's ``.weight_scale``.

    Anchored at the end so that a module legitimately named ``...scale...``
    (``hc_attn_scale``) is left alone.
    """
    if name.endswith(".scale"):
        return name[:-len(".scale")] + ".weight_scale"
    return name


def map_weight_name(name: str) -> MappedWeight:
    """Map one checkpoint tensor name, or raise :class:`UnmappedWeightError`."""
    if name in _TOPLEVEL:
        return MappedWeight(MAIN, _TOPLEVEL[name])

    if name.startswith("vision.") or name.startswith("aligner."):
        return MappedWeight(VISION, _rewrite_scale(name))

    m = _MTP_RE.match(name)
    if m is not None:
        idx = int(m.group(1))
        rest = name[m.end():]
        # main_proj/main_norm/markov_head/confidence_head live only on mtp.0;
        # they are left in place rather than folded into the block, because the
        # drafter reads them separately from the transformer block itself.
        mapped = _rewrite_submodules(rest)
        return MappedWeight(DSPARK, _rewrite_scale(f"model.layers.{idx}.{mapped}"),
                            spec_layer=idx)

    m = _LAYER_RE.match(name)
    if m is not None:
        idx = int(m.group(1))
        rest = name[m.end():]
        mapped = _rewrite_submodules(rest)
        return MappedWeight(MAIN, _rewrite_scale(f"model.layers.{idx}.{mapped}"))

    raise UnmappedWeightError(
        f"{name!r} matches no DeepSeek-V4.1 naming rule. Add a rule rather "
        "than skipping it: an unmapped tensor leaves a randomly initialised "
        "parameter in the model, which loads and runs but is silently wrong.")


def map_all(names: Iterable[str]) -> dict[str, MappedWeight]:
    return {n: map_weight_name(n) for n in names}
