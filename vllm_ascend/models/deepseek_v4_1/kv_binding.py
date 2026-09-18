# SPDX-License-Identifier: Apache-2.0
"""How a V4.1 layer's KV cache has to be shaped before upstream reads it.

Three facts that only make sense together:

* the Ascend backend allocates ``(num_blocks, block_size, num_kv_heads,
  head_size)`` with ``num_kv_heads == 1``;
* upstream's V4.1 code asserts ``kv_cache.shape[1:] == (block_size, head_dim)``
  -- three dimensions -- and calls ``.numel()`` on it directly;
* the Ascend runner binds ``[kv_cache]``, a list, for the whole V4 family.

So the list is wrong for V4.1, and so is upstream's own remedy. Its
``bind_kv_cache`` does ``kv_cache.squeeze(1)``, commented ``[B, H=1, N, C] ->
[B, N, C]``, which assumes the head axis is at index 1. Here it is at index 2,
and ``squeeze`` on a non-unit dimension is a silent no-op -- it would leave the
four-dimensional tensor untouched and the assertion would still fail, with
nothing to say why.

The axis to drop is the one that is actually the head axis. V4.0's consumers
are untouched: they read the list, by fixed index in the indexer's case.
"""

from __future__ import annotations

import torch

#: Where the Ascend KV cache shape puts num_kv_heads.
HEAD_AXIS = -2


def is_v41(hf_text_config) -> bool:
    """V4.1 only. V4.0's Ascend consumers expect the list and index into it."""
    return str(getattr(hf_text_config, "model_type", "")).startswith("deepseek_v41")


def bind_value(kv_cache):
    """What to store on the layer.

    A tensor stays a tensor: V4.1 consumers read a single-tensor cache as one
    (``kv_cache.numel()`` in ``engram.ensure_cache``, an assert on
    ``shape[1:]`` in the SWA path). The unit head axis is dropped when there is
    one; a tensor that arrives without it is handed over unchanged.

    A cache that genuinely is several tensors (e.g. a (values, scales) pair
    addressed by fixed slot) is handed over as-is.
    """
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) == 1:
        # A one-element list is not "several tensors": _adjust_kv_layout
        # always returns a list, so SWA arrives wrapped.
        inner = kv_cache[0]
        if isinstance(inner, torch.Tensor):
            return _drop_head_axis(inner)
    if isinstance(kv_cache, torch.Tensor):
        return _drop_head_axis(kv_cache)
    # Several tensors, addressed by fixed slot: handed over exactly as it
    # came (wrapping again would shift every slot index down a level).
    return kv_cache


def _drop_head_axis(kv_cache: torch.Tensor) -> torch.Tensor:
    """The unit head axis, where there is one. Never any other axis."""
    if kv_cache.ndim >= 3 and kv_cache.shape[HEAD_AXIS] == 1:
        return kv_cache.squeeze(HEAD_AXIS)
    return kv_cache
