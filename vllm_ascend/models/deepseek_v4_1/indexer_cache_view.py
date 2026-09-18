# SPDX-License-Identifier: Apache-2.0
"""The indexer K cache, addressed the way the read path needs it.

The writer hands ``indexer_compress_epilog_v2`` the whole packed tensor and the
operator places values and scales itself; the reader has to pass ``key`` and
``key_dequant_scale`` to ``npu_quant_lightning_indexer_v2`` separately.

The packing is segmented within a block, not per token. From the operator's
kernel::

    valueOffset = blkNumIdx*blockStride + blkSizeIdx*d
    scaleOffset = blkNumIdx*blockStride + cacheBs*d + blkSizeIdx*scaleCol*4

so within one block of ``cacheBs`` tokens:

    bytes [0, cacheBs*d)            every token's values, token t at t*d
    bytes [cacheBs*d, blockStride)  every token's scales, token t at t*4

Reading each token as ``cacheD`` contiguous bytes (values then scale) would be
silently wrong: shapes and strides stay valid while the bytes belong to other
tokens.

The packed width is upstream's ``index_head_dim + index_head_dim // 128 * 4``
(128 fp8 values plus one fp32 scale column = 132 bytes); both numbers are
taken from the tensor here.
"""

from __future__ import annotations

import torch

#: Values per scale column, from the operator's ``scaleCol_ = ceil(d_ / 128)``.
SCALE_GROUP = 128

#: Bytes per scale column (fp32).
SCALE_BYTES = torch.finfo(torch.float32).bits // 8

#: The scale column's torch dtype.
SCALE_DTYPE = torch.float32


def value_width(cache_width: int) -> int:
    """Values per token, inverted from upstream's packed width.

    ``cache_width = d + SCALE_BYTES * ceil(d / SCALE_GROUP)``; d is a multiple
    of SCALE_GROUP for every width this cache takes, so the inverse is exact.
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

    Views rather than copies: a copy would cost per token per step on the
    decode path, and a correctly shaped copy would silently read stale data.
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
    """Both views address the cache's own storage, and the scale segment
    begins exactly ``tokens_per_block * d`` bytes in, where the kernel looks.
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
