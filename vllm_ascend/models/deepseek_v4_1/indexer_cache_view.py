# SPDX-License-Identifier: Apache-2.0
"""The indexer K cache, addressed the way the read path needs it.

One block of memory, two conventions. The writer hands
``indexer_compress_epilog_v2`` the whole packed tensor and the operator places
values and scales itself; the reader has to pass ``key`` and
``key_dequant_scale`` to ``npu_quant_lightning_indexer_v2`` as two separate
arguments. So "the layout of this cache" is not one property -- it is a pair of
agreements, and a tensor that satisfies one of them says nothing about the
other.

🔴 The packing is **segmented within a block**, not per token. Read out of the
operator's kernel (``op_kernel/..._multi_row.h:99-101``, identical in
``..._single_row.h:85-87``)::

    valueOffset = blkNumIdx*blockStride + blkSizeIdx*d
    scaleOffset = blkNumIdx*blockStride + cacheBs*d + blkSizeIdx*scaleCol*4

so within one block of ``cacheBs`` tokens:

    bytes [0, cacheBs*d)            every token's values, token t at t*d
    bytes [cacheBs*d, blockStride)  every token's scales, token t at t*4

⚠️ The obvious reading -- "each token owns ``cacheD`` contiguous bytes, values
then scale" -- is wrong and **silent**. Under it token 1 would read bytes
132..263, which is the tail of token 1's values plus the head of token 2's.
Nothing raises: the shape is right, the dtype is right, the strides are right,
and the values are real numbers belonging to other tokens. That is why the
round-trip test in ``tests/ut/models/test_deepseek_v4_1_indexer_cache_view.py``
writes distinguishable marks and reads them back, rather than checking shapes.

The width formula is upstream's own (``attention.py:90``):
``head_dim = index_head_dim + index_head_dim // 128 * 4`` -- 128 fp8 values
plus one fp32 scale column -- and it is why the cache's last axis is 132 while
``d`` is 128. Both numbers are taken from the tensor and the config here, never
written down.
"""

from __future__ import annotations

import torch

#: Values per scale column, from the operator's ``scaleCol_ = ceil(d_ / 128)``
#: (``indexer_compress_epilog_v2_tiling.cpp:49``, PER_BLOCK).
SCALE_GROUP = 128

#: Bytes per scale column: fp32, from ``..._single_row.h:45``
#: (``RoundUp<float>(scaleCol) * sizeof(float)``), cross-checked against
#: ``..._base.h:39`` (``B32_INTERPRE_TO_B8_RATIO = 4``).
SCALE_BYTES = torch.finfo(torch.float32).bits // 8

#: What the scale column is, as a torch dtype. Named so the view below reads
#: from one place rather than pairing a 4 with a float32 in two.
SCALE_DTYPE = torch.float32


def value_width(cache_width: int) -> int:
    """Values per token, inverted from upstream's packed width.

    ``cache_width = d + SCALE_BYTES * ceil(d / SCALE_GROUP)``. For every width
    this cache actually takes, d is a multiple of SCALE_GROUP, so the inverse
    is exact rather than a search: one scale column per SCALE_GROUP values.
    """
    per_group = SCALE_GROUP + SCALE_BYTES
    if cache_width <= 0 or cache_width % per_group:
        raise ValueError(
            f"indexer cache row is {cache_width} bytes, which is not "
            f"{SCALE_GROUP} values plus {SCALE_BYTES} scale bytes per group. "
            "The split below would land between a value and its scale, and "
            "nothing downstream would say so -- the shapes stay right.")
    return cache_width // per_group * SCALE_GROUP


def split_indexer_cache(cache: torch.Tensor) -> tuple[torch.Tensor,
                                                      torch.Tensor]:
    """``(key, key_dequant_scale)`` as views of ``cache``, never copies.

    ``cache`` is the packed form the runner allocates and the writer fills:
    ``[num_blocks, tokens_per_block, cache_width]``, byte-addressable.

    Views rather than copies because the cost of a copy here is per token and
    per step -- it would sit on the decode path for the life of the process,
    not once at startup. The assertions below are about identity, not shape:
    two tensors of the right shape that do not share storage with the cache
    are a working, permanently wrong implementation.
    """
    if cache.ndim == 4 and cache.shape[-2] == 1:
        # The unit head axis, if binding left it on.
        cache = cache.squeeze(-2)
    if cache.ndim != 3:
        raise ValueError(
            f"indexer cache is {tuple(cache.shape)}; this takes "
            "[num_blocks, tokens_per_block, packed_width] (optionally with a "
            "unit head axis) and cannot place the value/scale boundary in "
            "anything else.")

    num_blocks, tokens_per_block, cache_width = cache.shape
    d = value_width(int(cache_width))
    scale_cols = d // SCALE_GROUP
    block_stride = tokens_per_block * cache_width
    values_bytes = tokens_per_block * d

    raw = cache.view(torch.uint8) if cache.dtype != torch.uint8 else cache
    base = raw.storage_offset()

    key = torch.as_strided(
        raw.view(cache.dtype),
        size=(num_blocks, tokens_per_block, 1, d),
        stride=(block_stride, d, d, 1),
        storage_offset=base,
    )

    if base % SCALE_BYTES or (base + values_bytes) % SCALE_BYTES:
        raise ValueError(
            f"the scale segment starts at byte {base + values_bytes}, which "
            f"is not {SCALE_BYTES}-byte aligned; a float32 view cannot start "
            "there and a realigned one would read another token's bytes.")
    scale = torch.as_strided(
        raw.view(SCALE_DTYPE),
        size=(num_blocks, tokens_per_block, 1, scale_cols),
        stride=(block_stride // SCALE_BYTES, scale_cols, scale_cols, 1),
        storage_offset=(base + values_bytes) // SCALE_BYTES,
    )

    _assert_same_memory(cache, key, scale, values_bytes)
    return key, scale


def _assert_same_memory(cache: torch.Tensor, key: torch.Tensor,
                        scale: torch.Tensor, values_bytes: int) -> None:
    """Identity, not shape.

    The failure this excludes is a pair of correctly shaped tensors that are
    copies, or views of something else: the read would succeed, the numbers
    would be wrong or stale, and every shape check would pass. So what is
    asserted is that both views address the cache's own storage, and that the
    scale segment begins exactly ``tokens_per_block * d`` bytes in -- the
    offset the operator's kernel computes. An implementation that read each
    token as ``cache_width`` contiguous bytes would put that gap at ``d``, so
    this distinguishes the two readings rather than merely bounding them.
    """
    cache_storage = cache.untyped_storage()
    for name, view in (("key", key), ("key_dequant_scale", scale)):
        if view.untyped_storage().data_ptr() != cache_storage.data_ptr():
            raise ValueError(
                f"{name} does not share the indexer cache's storage; it is a "
                "copy or a view of something else. A copy here costs per "
                "token per step, and a view of the wrong buffer reads stale "
                "keys with nothing to show for it.")

    scale_start = scale.storage_offset() * SCALE_BYTES
    expected = key.storage_offset() * key.element_size() + values_bytes
    if scale_start != expected:
        raise ValueError(
            f"the scale view starts {scale_start - expected} bytes from where "
            "the operator looks for it. The kernel computes "
            "blkNumIdx*blockStride + cacheBs*d; a per-token reading would put "
            f"it {values_bytes} bytes earlier and would silently return other "
            "tokens' values.")
