# SPDX-License-Identifier: Apache-2.0
"""The indexer K cache, addressed the way the read path needs it.

The writer hands ``indexer_quant_cache`` two tensors -- the values and the
scales -- and the reader has to pass ``key`` and ``key_dequant_scale`` to
``npu_quant_lightning_indexer_v2`` separately. Both live inside one paged
allocation, so both are views of it.

The packing is segmented within a block, not per token. From the operator's
kernel::

    valueOffset = blkNumIdx*blockStride + blkSizeIdx*valueBytes
    scaleOffset = blkNumIdx*blockStride + cacheBs*valueBytes + blkSizeIdx*scaleCol

so within one block of ``cacheBs`` tokens::

    bytes [0, cacheBs*valueBytes)        every token's values
    bytes [cacheBs*valueBytes, stride)   every token's scales

**That segmentation rule is unchanged** from the fp8 row this replaced; what
changed is the arithmetic inside it. MXFP4 stores two E2M1 values per byte and
one E8M0 scale (one byte) per 32 logical values, so a 128-wide key is
``128 // 2 = 64`` value bytes plus ``128 // 32 = 4`` scale bytes -- **68 bytes
per token**, where the fp8 row was ``128 + 4 = 132``.

Reading each token as ``cache_width`` contiguous bytes (its values followed by
its scales) would be silently wrong: shapes and strides stay valid while the
bytes belong to other tokens.

Both planes are byte planes. The values are ``uint8`` because two E2M1 values
share a byte and the scales are ``uint8`` because E8M0 *is* a byte; neither
dtype says what the bytes mean, which is why the read path re-views them as
``float4_e2m1fn_x2`` / ``float8_e8m0fnu`` at the call that needs the meaning
(see ``indexer_select.select_topk``).

Every number here is derived from the row width, not written down: a different
``index_head_dim`` moves the value bytes, the scale bytes and the split point
together or not at all.
"""

from __future__ import annotations

import torch

#: Logical values covered by one MXFP4 scale -- the E8M0 block size. Upstream
#: writes the same number as ``MXFP4_BLOCK_SIZE``
#: (``vllm/model_executor/layers/sparse_attn_indexer.py:51``).
SCALE_GROUP = 32

#: Bytes per scale. E8M0 is one byte: a bare exponent, no mantissa.
SCALE_BYTES = 1

#: The scale plane's torch dtype. The byte, not what it means -- see the module
#: docstring.
SCALE_DTYPE = torch.uint8

#: E2M1 values per stored byte.
VALUES_PER_BYTE = 2

#: Modes 3 and 5 give the dequant scales a trailing ``(D/64, 2)`` axis, so the
#: scales are addressed in pairs and a segment may not start mid-pair.
SCALE_PAIR = 2

#: Upstream's fp8 row, which is still the width it constructs the V4.1 indexer
#: cache with: 128 fp8 values plus one fp32 scale column per 128 values.
#: ``vllm/models/deepseek_v4_1/attention.py:88-90``.
UPSTREAM_FP8_SCALE_GROUP = 128
UPSTREAM_FP8_SCALE_BYTES = torch.finfo(torch.float32).bits // 8


def value_width(cache_width: int) -> int:
    """Logical values per token, inverted from the MXFP4 row width.

    ``cache_width = d // VALUES_PER_BYTE + SCALE_BYTES * (d // SCALE_GROUP)``;
    d is a multiple of SCALE_GROUP for every width this cache takes, so the
    inverse is exact.
    """
    per_group = SCALE_GROUP // VALUES_PER_BYTE + SCALE_BYTES
    if cache_width <= 0 or cache_width % per_group:
        raise ValueError(
            f"indexer cache row is {cache_width} bytes, which is not "
            f"{SCALE_GROUP // VALUES_PER_BYTE} value bytes plus {SCALE_BYTES} "
            f"scale byte per group of {SCALE_GROUP} values. The split below "
            "would land between a value and its scale, and nothing downstream "
            "would say so -- the shapes stay right.")
    return cache_width // per_group * SCALE_GROUP


def value_bytes_per_token(logical_width: int) -> int:
    """Bytes of packed E2M1 values for one token of ``logical_width`` values."""
    if logical_width <= 0 or logical_width % SCALE_GROUP:
        raise ValueError(
            f"logical key width {logical_width} is not a multiple of the "
            f"MXFP4 block size {SCALE_GROUP}; the last block would be part "
            "of a scale group that has no scale.")
    return logical_width // VALUES_PER_BYTE


def scale_bytes_per_token(logical_width: int) -> int:
    """Bytes of E8M0 scales for one token of ``logical_width`` values."""
    if logical_width <= 0 or logical_width % SCALE_GROUP:
        raise ValueError(
            f"logical key width {logical_width} is not a multiple of the "
            f"MXFP4 block size {SCALE_GROUP}; the last block would be part "
            "of a scale group that has no scale.")
    return logical_width // SCALE_GROUP * SCALE_BYTES


def logical_width_from_head_dim(head_dim: int) -> int:
    """The logical key width behind a packed per-token row width.

    ``DeepseekV4IndexerCache`` is constructed with
    ``_indexer_k_cache_head_dim(index_head_dim, use_fp4_kv)``
    (``vllm/models/deepseek_v4_1/attention.py:82-90``), and on Ascend
    ``dsa_indexer_uses_fp4`` is always False -- it raises outside Blackwell --
    so the number that reaches the Ascend spec is upstream's **fp8** row,
    ``d + d // 128 * 4`` (132 for d=128). Upstream's own fp4 row,
    ``d // 2 + d // 32``, is inverted too, so this keeps working if that gate
    ever opens.

    Both inversions are exact and the two row formulas do not collide at any
    width this cache takes; a width that satisfies both is refused rather than
    resolved by precedence, because picking one would put the wrong logical
    width into the page size with nothing to show for it.
    """
    fp8_group = UPSTREAM_FP8_SCALE_GROUP + UPSTREAM_FP8_SCALE_BYTES
    mxfp4_group = SCALE_GROUP // VALUES_PER_BYTE + SCALE_BYTES
    is_fp8 = head_dim > 0 and head_dim % fp8_group == 0
    is_mxfp4 = head_dim > 0 and head_dim % mxfp4_group == 0
    if is_fp8 and is_mxfp4:
        raise ValueError(
            f"packed row width {head_dim} inverts to "
            f"{head_dim // fp8_group * UPSTREAM_FP8_SCALE_GROUP} logical "
            "values as an fp8 row and to "
            f"{head_dim // mxfp4_group * SCALE_GROUP} as an MXFP4 row. Both "
            "readings are arithmetically exact, so there is nothing in the "
            "width to choose between them, and the wrong one sizes every page "
            "in the cache.")
    if is_fp8:
        return head_dim // fp8_group * UPSTREAM_FP8_SCALE_GROUP
    if is_mxfp4:
        return head_dim // mxfp4_group * SCALE_GROUP
    raise ValueError(
        f"packed row width {head_dim} is neither upstream's fp8 row "
        f"(d + d // {UPSTREAM_FP8_SCALE_GROUP} * {UPSTREAM_FP8_SCALE_BYTES}) "
        f"nor its MXFP4 row (d // {VALUES_PER_BYTE} + d // {SCALE_GROUP}). "
        "The logical key width is not written down anywhere else on this "
        "path, so it cannot be recovered.")


def split_indexer_cache(cache: torch.Tensor) -> tuple[torch.Tensor,
                                                      torch.Tensor]:
    """``(key, key_dequant_scale)`` as views of ``cache``, never copies.

    ``cache`` is the packed form the runner allocates and the writer fills:
    ``[num_blocks, tokens_per_block, cache_width]``, byte-addressable. Both
    returned views are ``uint8``.

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
    value_bytes = value_bytes_per_token(d)
    scale_cols = scale_bytes_per_token(d)
    block_stride = tokens_per_block * cache_width
    values_bytes = tokens_per_block * value_bytes

    raw = cache.view(torch.uint8) if cache.dtype != torch.uint8 else cache
    base = raw.storage_offset()

    key = torch.as_strided(
        raw,
        size=(num_blocks, tokens_per_block, 1, value_bytes),
        stride=(block_stride, value_bytes, value_bytes, 1),
        storage_offset=base,
    )

    _assert_mxfp4_placement(base, values_bytes, block_stride)
    scale = torch.as_strided(
        raw,
        size=(num_blocks, tokens_per_block, 1, scale_cols),
        stride=(block_stride, scale_cols, scale_cols, 1),
        storage_offset=base + values_bytes,
    )

    _assert_same_memory(cache, key, scale, values_bytes)
    return key, scale


def _assert_mxfp4_placement(base: int, values_bytes: int,
                            block_stride: int) -> None:
    """Where MXFP4 allows the two segments to start, and at what stride.

    This guard is the fp8 one's successor, not its survivor: the fp8 split
    checked 4-byte alignment because the scale segment was read through a
    ``float32`` view and torch refuses a misaligned one. **That reason
    retired with the fp32 scales** -- a ``uint8`` view has no alignment
    requirement at all, and torch would build this split at any offset.

    Two requirements took its place, and both are the operator's rather than
    torch's. ``libophost_transformer.so``, which is the tiling that actually
    runs, carries them verbatim::

        When quant_mode is 5 and layout_k is PA_BBND, key stride0 must be
        positive and satisfy 2-element FP4 packing alignment, but got
        When quant_mode is 3 or 5 and layout_k is PA_BBND,
        key_dequant_scale stride0 must be positive and satisfy 2-element
        E8M0 packing alignment, but got

    and CANN's own torch binding additionally checks ``storage_offset() % 2``
    on the E8M0 scale before the call (the wheel's ``FixQLIV2AclDtypes()``).
    🔴 **Our** ``csrc/torch_binding.cpp`` has no equivalent of that function,
    so nothing between here and the kernel performs that check -- which is
    precisely why it is performed here.

    A misplaced segment is not rejected by torch: the views are byte-typed,
    every stride stays valid, and a scale pair that straddles would dequantise
    each group with its neighbour's exponent.
    """
    if block_stride <= 0 or block_stride % SCALE_PAIR:
        raise ValueError(
            f"the indexer cache block stride is {block_stride} bytes; both "
            f"planes are addressed at that stride and the operator requires "
            f"it to be positive and a multiple of {SCALE_PAIR} (\"2-element "
            "FP4 packing alignment\" for the key, \"2-element E8M0 packing "
            "alignment\" for the scale).")
    scale_start = base + values_bytes
    if scale_start % SCALE_PAIR:
        raise ValueError(
            f"the scale segment starts at byte {scale_start}, which is not a "
            f"multiple of {SCALE_PAIR}. Modes 3 and 5 read the dequant scale "
            "with a trailing (D/64, 2) axis, so a pair would straddle the "
            "start and every group would be dequantised with the neighbouring "
            "exponent. CANN's own binding refuses this (FixQLIV2AclDtypes); "
            "ours does not, so it is refused here.")


def _assert_same_memory(cache: torch.Tensor, key: torch.Tensor,
                        scale: torch.Tensor, values_bytes: int) -> None:
    """Both views address the cache's own storage, and the scale segment
    begins exactly ``tokens_per_block * value_bytes`` in, where the kernel
    looks.
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
            "blkNumIdx*blockStride + cacheBs*valueBytes; a per-token reading "
            f"would put it {values_bytes} bytes earlier and would silently "
            "return other tokens' values.")


def planes_from_bound_cache(kv_cache) -> tuple[torch.Tensor, torch.Tensor]:
    """``(key, key_dequant_scale)`` out of whatever the runner bound.

    Three forms reach here, and which one arrives is decided by the spec, not
    by this module:

    * a **single packed tensor** -- what the runner produces when the spec
      carries ``scale_dim == 0``; split here at the operator's byte offsets;
    * a **pair**, ``(values, scales)`` -- already the two planes;
    * a **triple**, ``(values, scales, packed)`` -- what
      ``_reshape_kv_cache_tensors`` produces for a ``scale_dim != 0`` spec on
      a DSV4_COMPRESSED_CACHE part (``worker/model_runner_v1.py:5078-5106``,
      ``worker/v2/attn_utils.py:456-476``): the same storage viewed three
      ways, the third overlapping the first two. The first two *are* the
      planes; the third is checked against them rather than ignored, because
      "the runner already split it" and "the runner split it somewhere else"
      look identical from the pair alone.
    """
    if isinstance(kv_cache, torch.Tensor):
        return split_indexer_cache(kv_cache)
    if isinstance(kv_cache, (tuple, list)):
        if len(kv_cache) == 2:
            return kv_cache[0], kv_cache[1]
        if len(kv_cache) == 3:
            key, scale, packed = kv_cache
            _assert_triple_agrees(key, scale, packed)
            return key, scale
    raise TypeError(
        f"the indexer cache came back as {type(kv_cache).__name__}"
        + (f" of {len(kv_cache)} tensors" if isinstance(
            kv_cache, (tuple, list)) else "")
        + "; this takes the packed tensor, the (values, scales) pair, or the "
        "(values, scales, packed) triple the runner builds for a spec with "
        "scale_dim != 0. npu_quant_lightning_indexer_v2 takes key and "
        "key_dequant_scale as two arguments and there is nothing here to "
        "derive them from.")


def _assert_triple_agrees(key: torch.Tensor, scale: torch.Tensor,
                          packed: torch.Tensor) -> None:
    """The runner's three views really are one page split where we split it.

    The triple is built from the spec's ``head_size`` and ``scale_dim``, which
    are computed in ``caches.py`` from the same row arithmetic as
    :func:`split_indexer_cache`. If the two ever come apart, every shape here
    stays valid and the reader reads the writer's scales as values.
    """
    if not all(isinstance(t, torch.Tensor) for t in (key, scale, packed)):
        raise TypeError(
            "the indexer cache triple holds something that is not a tensor; "
            f"got {[type(t).__name__ for t in (key, scale, packed)]}.")
    ptrs = {t.untyped_storage().data_ptr() for t in (key, scale, packed)}
    if len(ptrs) != 1:
        raise ValueError(
            "the indexer cache triple does not share one allocation "
            f"({len(ptrs)} distinct storages). The third tensor is supposed "
            "to be the whole page the first two are segments of.")
    width = int(packed.shape[-1]) * packed.element_size()
    d = value_width(width)
    expected_key = value_bytes_per_token(d)
    expected_scale = scale_bytes_per_token(d)
    got_key = int(key.shape[-1]) * key.element_size()
    got_scale = int(scale.shape[-1]) * scale.element_size()
    if (got_key, got_scale) != (expected_key, expected_scale):
        raise ValueError(
            f"the runner split a {width}-byte indexer row into "
            f"{got_key} value bytes + {got_scale} scale bytes, but that row "
            f"is {expected_key} + {expected_scale}. The spec's head_size and "
            "scale_dim disagree with the packing this module reads, and every "
            "shape stays valid either way.")
    offset_bytes = (scale.storage_offset() * scale.element_size()
                    - key.storage_offset() * key.element_size())
    tokens_per_block = int(key.shape[1]) if key.ndim >= 2 else 0
    if offset_bytes != tokens_per_block * expected_key:
        raise ValueError(
            f"the runner's scale plane starts {offset_bytes} bytes into the "
            f"page; the operator reads it at cacheBs*valueBytes = "
            f"{tokens_per_block * expected_key}. A whole-token reading would "
            "land here too, so the shapes cannot tell the two apart.")
