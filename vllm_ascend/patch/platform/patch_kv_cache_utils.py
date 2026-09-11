# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import math
from collections import defaultdict

import vllm.v1.core.kv_cache_utils
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.attention.backend import MultipleOf
from vllm.utils.math_utils import cdiv, round_up
from vllm.v1.core.kv_cache_utils import _approximate_gcd, may_override_num_blocks
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)

from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_config,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_max_memory_usage,
    get_glm5_next_pool_bytes_per_block,
)
from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.utils import vllm_version_is

_KIMI_K3_TARGET_LAYER_PREFIX = "language_model.model.layers."
_KIMI_K3_DRAFT_LAYER_PREFIX = "model.layers."
_orig_resolve_kv_cache_block_sizes = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes
_orig_get_kv_cache_groups_uniform_page_size = vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size
_orig_get_kv_cache_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_groups
_orig_get_kv_cache_config_from_groups = vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups
_orig_max_memory_usage_bytes_from_groups = vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups
_orig_pool_bytes_per_block = vllm.v1.core.kv_cache_utils._pool_bytes_per_block


if UniformTypeKVCacheSpecs.max_num_blocks_per_req is KVCacheSpec.max_num_blocks_per_req:

    def _uniform_type_max_num_blocks_per_req(
        self: UniformTypeKVCacheSpecs,
        vllm_config: VllmConfig,
        max_len: int,
    ) -> int:
        """Preserve the inner spec's block-table width."""
        widths = {spec.max_num_blocks_per_req(vllm_config, max_len) for spec in self.kv_cache_specs.values()}
        assert len(widths) == 1, (
            "All layers in the same KV cache group must need the same number "
            f"of block table entries, got {sorted(widths)}."
        )
        return next(iter(widths))

    UniformTypeKVCacheSpecs.max_num_blocks_per_req = (  # type: ignore[method-assign]
        _uniform_type_max_num_blocks_per_req
    )


def _page_sizes(spec: UniformTypeKVCacheSpecs) -> set[int]:
    """Distinct page sizes across a group's cache specs.

    vLLM #51718 removed ``UniformTypeKVCacheSpecs.get_page_sizes()`` on main;
    the underlying ``kv_cache_specs``/``page_size_bytes`` are unchanged on both
    lanes, so compute it inline.
    """
    return {s.page_size_bytes for s in spec.kv_cache_specs.values()}


def _ascend_resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Ascend-compatible resolve_kv_cache_block_sizes.

    vLLM PR #40860 added a restriction that hybrid KV cache groups with
    multiple block sizes do not support DCP.
    This restriction is correct for CUDA but not for Ascend, which implements
    context parallelism for MLA and SWA-MLA layers independently.

    For multiple KV cache groups with DCP, compute scheduler_block_size as
    lcm(group_block_sizes) * dcp to maintain alignment.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    group_block_sizes = [group.kv_cache_spec.block_size for group in groups]
    if dcp != 1:
        # Ascend supports CP with multiple KV cache groups; compute
        # scheduler_block_size using the LCM of all group block sizes
        # multiplied by DCP for proper alignment.
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp
        if not cache_config.enable_prefix_caching:
            return scheduler_block_size, scheduler_block_size
        hash_block_size = math.gcd(*group_block_sizes)
        return scheduler_block_size, hash_block_size

    return _orig_resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)


def _get_kimi_k3_dspark_mixed_kv_cache_groups(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec] | None:
    """Build topology-independent Kimi K3 DSpark scheduler groups.

    Target and causal draft attention layers require the same full-sequence
    block ownership. Putting them in one UniformType group lets them share one
    scheduler block table while preserving a separate physical page per layer.
    Recurrent layers are split into the fewest balanced groups whose size does
    not exceed the attention group. This minimizes scheduler groups while
    keeping the recurrent groups balanced.

    Block and page sizes are resolved by the runtime and intentionally not
    fixed here: TP8 and TP16 produce different sizes but the same ownership
    relation. An unrecognized or incompatible signature falls back to vLLM's
    generic hybrid grouping.
    """
    target_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    draft_attention_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_DRAFT_LAYER_PREFIX) and isinstance(spec, FullAttentionSpec)
    }
    mamba_specs = {
        name: spec
        for name, spec in kv_cache_spec.items()
        if name.startswith(_KIMI_K3_TARGET_LAYER_PREFIX) and isinstance(spec, MambaSpec)
    }

    matched_layer_count = len(target_attention_specs) + len(draft_attention_specs) + len(mamba_specs)
    if (
        not target_attention_specs
        or not draft_attention_specs
        or not mamba_specs
        or matched_layer_count != len(kv_cache_spec)
    ):
        return None

    attention_specs = [*target_attention_specs.values(), *draft_attention_specs.values()]
    all_specs = [*attention_specs, *mamba_specs.values()]
    # Only attention layers share a scheduler block table. Mamba keeps its
    # own groups and may use max_model_len as block_size when cache_mode=none.
    if (
        len({spec.block_size for spec in attention_specs}) != 1
        or len({spec.page_size_bytes for spec in all_specs}) != 1
    ):
        return None

    first_mamba_spec = next(iter(mamba_specs.values()))
    if any(spec != first_mamba_spec for spec in mamba_specs.values()):
        return None

    # Insert target attention first. generate_scheduler_kv_cache_config unwraps a
    # UniformType group to its first spec, and this representative is registered
    # with the FullAttentionManager needed by both target and draft attention.
    mixed_attention_specs = {**target_attention_specs, **draft_attention_specs}
    mixed_attention_spec = UniformTypeKVCacheSpecs.from_specs(mixed_attention_specs)
    if mixed_attention_spec is None:
        return None

    groups = [
        KVCacheGroupSpec(
            layer_names=list(mixed_attention_specs),
            kv_cache_spec=mixed_attention_spec,
        )
    ]
    mamba_layer_names = list(mamba_specs)
    mamba_group_count = cdiv(len(mamba_layer_names), len(mixed_attention_specs))
    for group_idx in range(mamba_group_count):
        layer_names = mamba_layer_names[group_idx::mamba_group_count]
        group_specs = {name: mamba_specs[name] for name in layer_names}
        uniform_mamba_spec = UniformTypeKVCacheSpecs.from_specs(group_specs)
        assert uniform_mamba_spec is not None
        groups.append(
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=uniform_mamba_spec,
            )
        )

    logger.info(
        "Using Kimi K3 DSpark mixed KV grouping: %d target + %d draft attention layers, followed by Mamba groups %s",
        len(target_attention_specs),
        len(draft_attention_specs),
        [len(group.layer_names) for group in groups[1:]],
    )
    return groups


def _get_kv_cache_groups_uniform_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
    if kimi_k3_groups is not None:
        return kimi_k3_groups
    return _orig_get_kv_cache_groups_uniform_page_size(kv_cache_spec)


def _kv_cache_config_has_mamba_layers(self: KVCacheConfig) -> bool:
    """Recognize Mamba layers nested in UniformType cache groups."""
    return any(get_kv_cache_spec_kind(group.kv_cache_spec) == KVCacheSpecKind.MAMBA for group in self.kv_cache_groups)


def _num_layer_tuples(group_spec: "UniformTypeKVCacheSpecs") -> int:
    """Number of layer tuples in one uniform-type group.

    vLLM #53896 (e126687a9a) renamed
    ``UniformTypeKVCacheSpecs.get_num_layer_tuples`` to
    ``get_max_layers_per_page_size``; the body is unchanged
    (``vllm/v1/kv_cache_interface.py:1144``).
    """
    if vllm_version_is("0.28.0"):
        return group_spec.get_num_layer_tuples()
    return group_spec.get_max_layers_per_page_size()


def has_deepseek_v4_specs(kv_cache_spec: dict[str, KVCacheSpec]) -> bool:
    """Spec-level counterpart of :func:`_is_deepseek_v4_groups`.

    That one answers the same question about groups, so it cannot be used
    before grouping has happened.
    """
    return any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in kv_cache_spec.values())


def group_and_unify_kv_cache_specs(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[UniformTypeKVCacheSpecs] | None:
    """
    Group the KV cache specs and unify each group into one UniformTypeKVCacheSpecs.
    Currently, this is only used for DeepseekV4.
    """
    if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
        return None

    logical_block_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[spec.block_size][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            logical_block_specs[spec.block_size][name] = spec

    mla_uniform_specs = []
    for block_size in sorted(logical_block_specs):
        spec_dict = logical_block_specs[block_size]
        assert len(spec_dict) > 0
        mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
    assert mla_uniform_specs is not None

    swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    for spec_dict in grouped_swa_mla_specs.values():
        uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
        assert uniform_spec is not None
        swa_uniform_specs.append(uniform_spec)

    return [*mla_uniform_specs, *swa_uniform_specs]


def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """
    Generate the KV cache groups from the grouped specs.
    """
    assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
    # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
    # containing only MLAAttentionSpec.
    full_mla_spec = grouped_specs[0]
    full_mla_c128_spec = grouped_specs[1]

    assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
    full_mla_group = KVCacheGroupSpec(
        layer_names=list(full_mla_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_spec,
    )
    full_mla_c128_group = KVCacheGroupSpec(
        layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
        kv_cache_spec=full_mla_c128_spec,
    )

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.
    # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
    # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
    # The other uniform KV cache specs will be similarly partitioned into layer tuples.
    # Say we have 21 SWA layers, all with the same page size, then we will have "21"
    # layer tuples.
    num_layer_tuples_per_group: list[int] = [_num_layer_tuples(g_spec) for g_spec in grouped_specs]
    # Choose `num_layer_tuples` to minimize total padding across groups.
    num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
    # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
    num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

    # TODO(cmq): this is not general enough
    swa_mla_specs = grouped_specs[2:]

    assert all(
        isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
    )

    # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
    # Possibly padding layer tuples for this.
    # Additionally, we also pad KV blocks in each SWA layer, to align the page size
    # with the corresponding layer in the full-MLA group.
    all_page_sizes = _page_sizes(full_mla_spec)
    swa_mla_groups = []
    for sm_spec in swa_mla_specs:
        sm_page_sizes = _page_sizes(sm_spec)
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        assert max(sm_page_sizes) <= max(all_page_sizes)

        # Unify page size by padding layers' page_size to the nearest larger page_size.
        # Compute candidate (nearest larger page_size) for each unique page size.
        size_to_candidate: dict[int, int] = {}
        for ps in sm_page_sizes:
            size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
        # Pad and collect layer names per page size.
        for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
            current_size = layer_spec.page_size_bytes
            candidate = size_to_candidate[current_size]
            if current_size < candidate:
                object.__setattr__(layer_spec, "page_size_padded", candidate)
            layers_per_size[candidate].append(layer_name)
        # NOTE(yifan): for now, inside a UniformKV group, each page_size should
        # have the same number of layers. This also means we don't need to pad layers
        # inside a partial-full layer tuple.
        assert len(set(len(layers) for layers in layers_per_size.values())) == 1
        num_layers_per_size = len(next(iter(layers_per_size.values())))

        # Split layers inside each UniformKV group for aligned #(layers).
        # See `_get_kv_cache_groups_uniform_page_size` for more details.
        num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
        layer_tuples = list(zip(*layers_per_size.values()))
        for i in range(num_tuple_groups):
            group_layer_tuples = layer_tuples[i::num_tuple_groups]
            # Flatten tuples and build dict for from_specs
            group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
            group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
            sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_sm_spec is not None
            swa_mla_groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_sm_spec,
                )
            )

    return [full_mla_group, full_mla_c128_group, *swa_mla_groups]


def _ring_layer_pages(kv_cache_groups: list[KVCacheGroupSpec]) -> list[tuple[str, int]]:
    """compressor 环形 buffer 的 (层名, page_size)，按层名定序。

    环不属于 DSV4 的 layer-tuple 契约：tuple 的两个页大小来自
    ``DSV4_BLOCK_SIZES`` 里硬写的 (page_size_padded_t1, page_size_padded_t2)
    对，那是主 MLA cache 和 indexer K cache 的几何；环的 32768 不在其中。
    ⇒ 它要单独排在同一块 backing 的尾部，不能塞进 tuple。
    """
    out: list[tuple[str, int]] = []
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, CircularBufferSpec):
            out.extend((name, spec.page_size_bytes) for name in group.layer_names)
    return sorted(out)


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroupSpec],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and rank normalization.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream (see _get_kv_cache_groups_uniform_groups) so
    every layer's page_size matches one of the full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    page_sizes = sorted(_page_sizes(full_mla_spec))

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        if isinstance(group.kv_cache_spec, CircularBufferSpec):
            continue  # 环单独排在 backing 尾部，见 _ring_layer_pages
        assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        specs = group.kv_cache_spec.kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            if "mtp" not in name:
                b[specs[name].page_size_bytes].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = specs[name].page_size_bytes
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    # 🔴 这条守卫原本【不存在】，而它挡的正是一个不会报错的错。
    #    下面两个 planner 都是 `for ps in page_sizes:` 再从各组的桶里取层 ——
    #    页大小不在 page_sizes 里的桶会被【静默跳过】，那些层一个 KVCacheTensor
    #    都拿不到，num_blocks 和报出来的显存却完全正常。
    #    docstring 里那句 "every layer's page_size matches one of the full-MLA
    #    bucket sizes" 一直只是一句描述；描述不会变红，判据才会。
    strays: dict[int, list[str]] = defaultdict(list)
    for b in bucketed:
        for ps, names in b.items():
            if ps not in page_sizes:
                strays[ps].extend(names)
    if strays:
        raise NotImplementedError(
            f"DSV4 layer-tuple 的页大小集合是 {page_sizes}（取自 full-MLA 组），"
            f"但有层的 page_size 不在其中："
            f"{ {k: v[:4] for k, v in sorted(strays.items())} }。"
            "这些层会被 planner 静默跳过、拿不到任何 KV cache tensor。"
        )

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


def _get_kv_cache_config_deepseek_v4(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    """Plan v0.28.0 DSV4 tensors using the shared_by contract."""
    page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples = _get_deepseek_v4_cache_layout(
        kv_cache_groups
    )
    # 0.28.0 的 shared_by 契约没有排环的位置。这里【拒绝】而不是照常返回：
    # 照常返回的结果是那几层拿不到 tensor，而这条路径我们跑不到、也就测不到。
    if ring := _ring_layer_pages(kv_cache_groups):
        raise NotImplementedError(
            f"v0.28.0 的 shared_by DSV4 planner 没有实现 compressor 环形 buffer 的排布，"
            f"涉及层: {[n for n, _ in ring]}"
        )
    layer_tuple_page_bytes = sum(page_sizes)

    num_blocks = available_memory // (layer_tuple_page_bytes * num_layer_tuples)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors: list[KVCacheTensor] = []
    for tuple_idx in range(num_layer_tuples - len(mtp_layer_names)):
        for ps in page_sizes:
            shared_by: list[str] = []
            for b in bucketed:
                bucket = b.get(ps)
                if bucket is not None and tuple_idx < len(bucket):
                    shared_by.append(bucket[tuple_idx])
            kv_cache_tensors.append(KVCacheTensor(size=ps * num_blocks, shared_by=shared_by))
    for i in range(len(mtp_layer_names)):
        kv_cache_tensors.append(
            KVCacheTensor(
                size=mtp_page_size * num_blocks,
                shared_by=[mtp_layer_names[i]],
            )
        )

    return num_blocks, kv_cache_tensors


def _get_kv_cache_config_deepseek_v4_main(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> tuple[int, list[KVCacheTensor]]:
    (
        page_sizes,
        bucketed,
        mtp_layer_names,
        mtp_page_size,
        num_tuple_slots,
    ) = _get_deepseek_v4_cache_layout(kv_cache_groups)

    # 环形 buffer 的页大小不在 tuple 里（见 _ring_layer_pages），所以它的字节
    # 必须【在解 num_blocks 之前】就计入分母；事后再往 backing 后面追加，
    # 得到的总量会超过 available_memory。
    ring_pages = _ring_layer_pages(kv_cache_groups)
    ring_bytes_per_block = sum(ps for _, ps in ring_pages)

    bytes_per_tuple = sum(page_sizes)
    num_blocks = available_memory // (bytes_per_tuple * num_tuple_slots + ring_bytes_per_block)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    tuple_stride = bytes_per_tuple * num_blocks
    tuple_region_bytes = tuple_stride * num_tuple_slots
    backing_size = tuple_region_bytes + ring_bytes_per_block * num_blocks

    # Within every tuple slot, page-size buckets are placed consecutively.
    page_offsets: dict[int, int] = {}
    page_prefix = 0
    for page_size in page_sizes:
        page_offsets[page_size] = page_prefix * num_blocks
        page_prefix += page_size

    tensors: list[KVCacheTensor] = []

    # Keep each descriptor inside one cache group. Descriptors from different
    # groups alias corresponding tuple slots by using identical geometry.
    for group_buckets in bucketed:
        for page_size in page_sizes:
            layer_names = group_buckets.get(page_size)
            if not layer_names:
                continue

            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=list(layer_names),
                    offset=page_offsets[page_size],
                    layer_stride=tuple_stride,
                    block_stride=page_size,
                )
            )

    # MTP layers receive trailing tuple slots. Unused page buckets remain
    # padding so num_blocks and memory accounting retain the existing contract.
    normal_tuple_slots = num_tuple_slots - len(mtp_layer_names)
    for index, layer_name in enumerate(mtp_layer_names):
        slot = normal_tuple_slots + index
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                offset=slot * tuple_stride + page_offsets[mtp_page_size],
                layer_stride=0,
                block_stride=mtp_page_size,
            )
        )

    # 环排在 tuple 区之后，每层一段连续区域。
    #
    # ⚠️ 为什么不能另开一块 backing：运行时 model_runner_v1 的 is_dsv4_main 分支
    #    要求【所有 descriptor 的 size 相同】("DeepSeek-V4 KV cache descriptors
    #    must share one backing allocation")，它只按那个 size 分配一次再切视图。
    #    给环单独一个小 descriptor 会当场被那条检查拒掉。
    ring_offset = tuple_region_bytes
    for layer_name, page_size in ring_pages:
        tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[layer_name],
                offset=ring_offset,
                layer_stride=0,
                block_stride=page_size,
            )
        )
        ring_offset += page_size * num_blocks
    assert ring_offset == backing_size, (
        f"环区排完后 offset={ring_offset} 与 backing_size={backing_size} 不符"
    )

    return num_blocks, tensors


def _is_deepseek_v4_groups(kv_cache_groups: list[KVCacheGroupSpec]) -> bool:
    """DSV4 的组集合：full-MLA 组打头，其余是 Uniform 组或 compressor 环组。

    🔴 这里原来要求【每一个】组都是 UniformTypeKVCacheSpecs。2026-09-11 第十三次
       起飞时，compressor 的环组（裸 CircularBufferSpec）让这个 all() 变成 False，
       于是整个 Ascend DSV4 planner 被【静默】绕过，落到上游通用 planner，
       症状却出现在下游一句和分组无关的 layout 报错上
       （"KV cache layout LBNHC cannot express ... [16896, 32768, 65536, 131072]"）。
       「走错了路」和「路上有堵墙」在日志里长得一模一样 ——
       所以这个分支现在会把自己为什么没被选中打出来。
    """
    if not kv_cache_groups:
        return False
    # 第 0 组必须是 full-MLA 的 Uniform 组：_get_deepseek_v4_cache_layout 用它
    # 定义整个 page_size 桶集合，这是它的前提而不是巧合。
    if not isinstance(kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
        return False
    if not all(
        isinstance(group.kv_cache_spec, (UniformTypeKVCacheSpecs, CircularBufferSpec))
        for group in kv_cache_groups
    ):
        return False
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        if not isinstance(group_spec, UniformTypeKVCacheSpecs):
            continue
        specs = group_spec.kv_cache_specs.values()
        if any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs):
            return True
    return False


def _ascend_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroupSpec]) -> int:
    """Use the same DSV4 divisor as Ascend's shared-tuple planner.

    vLLM #51718 re-plans ranks with more KV memory using
    ``min_num_blocks * _pool_bytes_per_block(groups)``. Its standardized
    per-group layout has a different divisor from Ascend's DSV4 shared-tuple
    layout, so using the upstream value changes ``num_blocks`` during the
    re-plan and leaves ranks inconsistent.
    """
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
    if not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_pool_bytes_per_block(kv_cache_groups)

    page_sizes, _, _, _, num_layer_tuples = _get_deepseek_v4_cache_layout(kv_cache_groups)
    # 必须和 _get_kv_cache_config_deepseek_v4_main 的分母【逐字节一致】：
    # #51718 的跨 rank 重排用 min_num_blocks * 这个值，除数不一致会让各 rank
    # 算出不同的 num_blocks。
    ring_bytes_per_block = sum(ps for _, ps in _ring_layer_pages(kv_cache_groups))
    return sum(page_sizes) * num_layer_tuples + ring_bytes_per_block


def _ascend_max_memory_usage_bytes_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
) -> int:
    """Keep the pre-#51718 DSV4 admission formula for its shared tuples."""
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_max_memory_usage(vllm_config, kv_cache_groups)
    if vllm_version_is("0.28.0") or not _is_deepseek_v4_groups(kv_cache_groups):
        return _orig_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)

    # 组里允许有 compressor 的环组（裸 CircularBufferSpec）；tuple 的几何只由
    # Uniform 组决定，环单独计在下面。
    # 🔴 这条断言原来写死「每组都是 Uniform」，和 _is_deepseek_v4_groups 放宽后的
    #    前提对不上 —— 第十五次起飞就死在这里。同一个假设散在三处
    #    （detector / layout / 这里），改了两处是改不完的。
    assert all(
        isinstance(group.kv_cache_spec, (UniformTypeKVCacheSpecs, CircularBufferSpec))
        for group in kv_cache_groups
    )
    full_mla_spec = kv_cache_groups[0].kv_cache_spec
    assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)
    layer_tuple_bytes = sum(_page_sizes(full_mla_spec))
    num_layer_tuples = max(
        _num_layer_tuples(group.kv_cache_spec)
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )
    uniform_bytes = sum(
        num_layer_tuples * group.kv_cache_spec.max_memory_usage_pages(vllm_config) * layer_tuple_bytes
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
    )
    # 环不在 tuple 里，单独计：一个请求整段生命周期占 1 个 block
    # （CircularBufferSpec.max_memory_usage_bytes 就返回 page_size_bytes）。
    # 这个 if 过滤原本把环整个漏掉了 —— 漏掉一项和它恰好为 0，在数上分不开。
    ring_bytes = sum(
        group.kv_cache_spec.max_memory_usage_bytes(vllm_config) * len(group.layer_names)
        for group in kv_cache_groups
        if isinstance(group.kv_cache_spec, CircularBufferSpec)
    )
    return uniform_bytes + ring_bytes




def _block_size_supported(block_size: int, supported: list) -> bool:
    """supported 里可能混着 int 和 MultipleOf(base)，两种都要认。

    上游 ``CompressorBackend.get_supported_kernel_block_sizes()`` 返回
    ``[MultipleOf(1)]`` —— 拿 ``bs not in supported`` 判会把任何块大小都判成不支持。
    """
    for item in supported:
        if isinstance(item, MultipleOf):
            if item.base > 0 and block_size % item.base == 0:
                return True
        elif item == block_size:
            return True
    return False


def _ring_groups(ring: dict[str, KVCacheSpec], vllm_config: VllmConfig) -> list[KVCacheGroupSpec]:
    """compressor 的 state_cache 各自成组，按 (block_size, page_size) 归并。

    ⚠️ 断言 block_size 落在该 backend 支持的 kernel block size 集合里 ——
       这是 select_common_block_size 的必要条件，不该默认它成立。
       （AscendCompressorStateBackend 故意不覆盖 get_supported_kernel_block_sizes，
        基类给 [2,4,8,16,32,64,128]；无投机时环容量 8、七个草稿 token 时 16，都在里面。）
    """
    ctx = vllm_config.compilation_config.static_forward_context
    buckets: dict[tuple, list[str]] = {}
    for name, spec in ring.items():
        buckets.setdefault((spec.block_size, spec.page_size_bytes), []).append(name)
    out: list[KVCacheGroupSpec] = []
    for (bs, _ps), names in sorted(buckets.items()):
        spec = ring[names[0]]
        # 🔴 backend 必须从【层】上拿。上一版写的是
        #    getattr(spec, "attn_backend", None) or getattr(spec, "backend", None)
        #    —— CircularBufferSpec 这两个属性都没有，于是 backend 恒为 None，
        #    整条断言【一次都没有执行过】。零覆盖的守卫和通过的守卫，日志里一样绿。
        # 🔴 这个函数在 EngineCore 和 Worker 【两个】进程里都会跑，而 EngineCore
        #    进程里没有模型 ⇒ static_forward_context 是空的。
        #    「本进程没有模型」和「这一层不见了」是两件事：前者这条判据在这里
        #    本来就不适用（worker 进程会做），后者是真的不一致。
        #    第十七次起飞就是把两者混成一条 ⇒ 我自己的 fail-safe 在 EngineCore 里打死了它。
        if not ctx:
            logger.info(
                "本进程没有模型（static_forward_context 为空），环组 block_size 检查"
                "留给 worker 进程做；本次【跳过】不是【通过】（涉及 %d 层）", len(names),
            )
        else:
            layer = ctx.get(names[0])
            if layer is None or not hasattr(layer, "get_attn_backend"):
                raise NotImplementedError(
                    f"static_forward_context 有 {len(ctx)} 层，但"
                    f"{'没有 ' + names[0] if layer is None else names[0] + ' 没有 get_attn_backend'}"
                    " —— 判据构造不出来就不放行，不能当成检查通过。"
                )
            backend = layer.get_attn_backend()
            supported = backend.get_supported_kernel_block_sizes()
            if supported and not _block_size_supported(bs, supported):
                raise NotImplementedError(
                    f"compressor state_cache block_size={bs} 不在 {backend.__name__} 支持的 "
                    f"kernel block sizes {supported} 里 —— select_common_block_size 会失败。"
                    f"涉及层: {names}"
                )
            logger.info("环组 block_size=%d 已对 %s 的支持集 %s 核过（%d 层）",
                        bs, backend.__name__, supported, len(names))
        out.append(KVCacheGroupSpec(layer_names=sorted(names), kv_cache_spec=spec))
    return out


def _assert_covers(
    groups: list[KVCacheGroupSpec],
    kv_cache_spec: dict[str, KVCacheSpec],
    branch: str,
) -> list[KVCacheGroupSpec]:
    """每一层都必须落进某个组，落不进就【当场点名】。

    🔴 上游在 kv_cache_utils.py:2649 有这条断言，但它只说
        "Some layers are not assigned to any group."
    —— 不说是哪些层、也不说它们是什么 spec。而那两样正是唯一能定位问题的信息，
    且分组是在【我们这个补丁里】做的 ⇒ 该由这里说出来，不该让上游去说它不知道的事。
    2026-09-11 第十一次起飞就撞在那条裸断言上，日志里一个层名都没有。

    ⚠️ 这条只改【报错内容】，不改分组行为：覆盖全时原样返回。
    """
    assigned = {name for g in groups for name in g.layer_names}
    missing = [n for n in kv_cache_spec if n not in assigned]
    if not missing:
        return groups
    def _desc(n: str) -> str:
        sp = kv_cache_spec[n]
        bits = [type(sp).__name__]
        for attr in ("block_size", "page_size_bytes", "model_version", "compress_ratio"):
            v = getattr(sp, attr, None)
            if v is not None:
                bits.append(f"{attr}={v}")
        return f"{n}  [{', '.join(bits)}]"
    raise NotImplementedError(
        f"Ascend {branch} KV-cache grouping left {len(missing)} of {len(kv_cache_spec)} "
        f"layers unassigned; upstream's assert would only say 'Some layers are not "
        f"assigned to any group.' The unassigned layers and their specs are:\n  "
        + "\n  ".join(_desc(n) for n in missing)
        + f"\nGrouped {len(assigned)} layers into {len(groups)} group(s): "
        + ", ".join(f"{len(g.layer_names)}x{type(g.kv_cache_spec).__name__}" for g in groups)
    )


def _log_groups(groups: list[KVCacheGroupSpec], branch: str) -> list[KVCacheGroupSpec]:
    """把分组结果【无条件】打出来：组号 / spec 类 / block_size / page_size / 层数。

    ⚠️ 只在出错时才打的诊断，等于在「一切正常」和「正常得可疑」之间不给任何信息。
    第十三次起飞就死在这上面：报错说 page size 有四种
    ([16896, 32768, 65536, 131072])，而【没有任何一处】说得出这四种各属于哪个组、
    哪些层 —— 光看报错无法判断该改 layout 还是该改分组。
    这条 INFO 是判据的输入，不是装饰。
    """
    def _page_sizes_of(g: KVCacheGroupSpec) -> dict[int, int]:
        sp = g.kv_cache_spec
        specs = sp.kv_cache_specs.values() if isinstance(sp, UniformTypeKVCacheSpecs) else [sp] * len(g.layer_names)
        out: dict[int, int] = defaultdict(int)
        for one in specs:
            out[one.page_size_bytes] += 1
        return dict(sorted(out.items()))

    logger.info("Ascend KV-cache 分组 [%s]: %d 组, 共 %d 层", branch, len(groups),
                sum(len(g.layer_names) for g in groups))
    for i, g in enumerate(groups):
        sp = g.kv_cache_spec
        inner = ""
        if isinstance(sp, UniformTypeKVCacheSpecs):
            kinds = sorted({type(one).__name__ for one in sp.kv_cache_specs.values()})
            inner = f" of {{{', '.join(kinds)}}}"
        logger.info(
            "  组%d: %s%s  block_size=%s  page_size->层数=%s  层数=%d  例: %s",
            i, type(sp).__name__, inner, getattr(sp, "block_size", "?"),
            _page_sizes_of(g), len(g.layer_names), g.layer_names[:2],
        )
    return groups


def _ascend_get_kv_cache_groups(
    vllm_config: VllmConfig,
    kv_cache_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Route GLM-5.3-Next and DeepSeek-V4 to their Ascend grouping.

    DSV4 deliberately runs two page sizes side by side -- the main MLA cache
    and the indexer K cache -- and they do not divide each other:

        main MLA   storage_block_size * num_kv_heads * head_size * itemsize
                   = 128 * 1 * (512 + 128) * 1                  = 81920
        indexer    128 * 1 * (128 * 1 + scale_dim 1 * fp32 4)   = 16896
        81920 % 16896 = 14336

    Both land exactly on the (page_size_padded_t1, page_size_padded_t2) pair
    hardcoded in ``DSV4_BLOCK_SIZES``
    (``vllm_ascend/models/layer/attention/layer.py:40``), so two page sizes
    is the design, not an accident to be unified away.

    vLLM used to give this its own branch in ``get_kv_cache_groups``, which
    ``group_and_unify_kv_cache_specs`` / ``_get_kv_cache_groups_uniform_groups``
    above replace. vLLM #53896 (e126687a9a, "Support Qwen3.8-Flash-Next")
    deleted both functions and their call site, leaving the two patches at the
    bottom of this module bound to names nothing calls any more. DSV4 then
    falls through to ``unify_kv_cache_spec_page_size``, whose two escape
    hatches both decline it:

      * ``_get_packed_kv_cache_groups`` wants a block-outermost layout
        (``vllm/v1/core/kv_cache_utils.py:1955``); Ascend resolves to LBNHC,
        which is layer-outermost.
      * ``_try_get_full_allocation_fallback_groups`` returns None for any
        SlidingWindowMLASpec (``kv_cache_utils.py:1852``), and DSV4's SWA and
        compressor caches are exactly that.

    so it raises

        NotImplementedError: Layer model.layers.2.self_attn.indexer.k_cache:
        page size is not divisible by the maximum page size and cannot be
        padded. Padding is only supported for non-MLA attention layers.

    Restore the branch here instead of upstream. It is gated on the DSV4
    marker rather than on ``group_and_unify_kv_cache_specs`` alone, whose
    SlidingWindowMLASpec test is broad enough to catch other models.

    The downstream config-stage planner already assumes this grouping ran:
    ``_get_deepseek_v4_cache_layout`` documents "Non-full-MLA groups must have
    been page_size-padded upstream (see _get_kv_cache_groups_uniform_groups)".
    """
    if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
        return _assert_covers(get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec), kv_cache_spec, "GLM5-Next")
    if not vllm_version_is("0.28.0") and has_deepseek_v4_specs(kv_cache_spec):
        # 🔴 V4.1 的 compressor state_cache 是【第三种】spec，必须单独成组 ——
        #    而这不是我们的偏好，是上游用代码要求的：
        #      CircularBufferSpec.is_uniform_with_collection（kv_cache_interface.py:784-789）
        #      覆盖了基类的默认实现，改成「这一组里【每一个】都必须是 CircularBufferSpec」，
        #      而 SameTypeKVCacheSpecs.is_uniform_type 正是调它来决定能不能成组。
        #    ⇒ 把它和任何别的 spec 放进同一组，uniformity 判定直接为假。
        #    ⚪ 它没有跨层语义（max_num_blocks_per_req()→1、max_memory_usage_bytes()→page_size、
        #      prefix_cacheable=False）：那个「环」是单层内部跨 token 的，三个环互不相干。
        #    ⚪ 为什么恰好 3 个：kv_source_layer_ids=[2,8,14,20] 共 4 个，
        #      而第 20 层 compress_ratio=1（不压缩）⇒ 没有「还在压缩中的组」⇒ 没有 state。
        #      4 - 1 = 3，和实际看到的一致，不是漏了一个。
        ring = {n: sp for n, sp in kv_cache_spec.items() if isinstance(sp, CircularBufferSpec)}
        rest = {n: sp for n, sp in kv_cache_spec.items() if n not in ring}
        if ring and rest:
            if grouped_specs := group_and_unify_kv_cache_specs(rest):
                groups = _get_kv_cache_groups_uniform_groups(grouped_specs)
                groups = groups + _ring_groups(ring, vllm_config)
                return _log_groups(_assert_covers(groups, kv_cache_spec, "DSV4+ring"), "DSV4+ring")
        elif grouped_specs := group_and_unify_kv_cache_specs(kv_cache_spec):
            return _log_groups(_assert_covers(_get_kv_cache_groups_uniform_groups(grouped_specs), kv_cache_spec, "DSV4"), "DSV4")
    return _log_groups(_orig_get_kv_cache_groups(vllm_config, kv_cache_spec), "upstream")


# Kept as an alias: this was the patch's name while it only handled GLM5.
_get_glm5_next_kv_cache_groups = _ascend_get_kv_cache_groups


def _ascend_get_kv_cache_config_from_groups(
    vllm_config: VllmConfig,
    kv_cache_groups: list[KVCacheGroupSpec],
    available_memory: int,
) -> KVCacheConfig:
    """Restore Ascend's DSV4 shared-tuple planner removed by vLLM #51718."""
    if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
        return get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, available_memory)
    if vllm_version_is("0.28.0") or not _is_deepseek_v4_groups(kv_cache_groups):
        # 🔴 这条分支【必须】说出自己为什么被选中。第十三次起飞时它被静默选中
        #    （_ring_groups 返回的裸 CircularBufferSpec 让 _is_deepseek_v4_groups 的
        #     all(isinstance(..., UniformTypeKVCacheSpecs)) 变 False），
        #    于是 Ascend 的 DSV4 planner 整个被绕过，症状却出现在上游的 layout 检查里
        #    ——「走错了路」和「路上有堵墙」在日志里长得一模一样。
        non_uniform = [
            f"组{i}={type(g.kv_cache_spec).__name__}"
            for i, g in enumerate(kv_cache_groups)
            if not isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs)
        ]
        logger.info(
            "Ascend DSV4 planner 未启用，走上游通用 planner（version_is_0.28.0=%s, "
            "is_dsv4_groups=%s, 非 UniformTypeKVCacheSpecs 的组: %s）",
            vllm_version_is("0.28.0"), _is_deepseek_v4_groups(kv_cache_groups),
            non_uniform or "无",
        )
        return _orig_get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)

    num_blocks, kv_cache_tensors = _get_kv_cache_config_deepseek_v4_main(
        vllm_config,
        kv_cache_groups,
        available_memory,
    )
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
        prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
    )


vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
# On vLLM > 0.28.0 these two names no longer exist upstream (deleted by
# #53896), so these assignments only add unread attributes; the functions are
# reached through _ascend_get_kv_cache_groups instead. Kept for 0.28.0, where
# get_kv_cache_groups still calls them.
vllm.v1.core.kv_cache_utils.group_and_unify_kv_cache_specs = group_and_unify_kv_cache_specs
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_groups = _get_kv_cache_groups_uniform_groups
vllm.v1.core.kv_cache_utils._get_kv_cache_groups_uniform_page_size = _get_kv_cache_groups_uniform_page_size
# vLLM v0.24.0 renamed _get_kv_cache_config_deepseek_v4 to
# _get_kv_cache_config_packed. The v0.28.0 planner still consumes shared_by;
# main uses _ascend_get_kv_cache_config_from_groups and the stride-aware planner.
if vllm_version_is("0.28.0"):
    vllm.v1.core.kv_cache_utils._get_kv_cache_config_packed = _get_kv_cache_config_deepseek_v4
vllm.v1.core.kv_cache_utils.get_kv_cache_groups = _ascend_get_kv_cache_groups
KVCacheConfig.has_mamba_layers = property(  # type: ignore[assignment]
    _kv_cache_config_has_mamba_layers
)
vllm.v1.core.kv_cache_utils.get_kv_cache_config_from_groups = _ascend_get_kv_cache_config_from_groups
vllm.v1.core.kv_cache_utils._max_memory_usage_bytes_from_groups = _ascend_max_memory_usage_bytes_from_groups
if not vllm_version_is("0.28.0"):
    vllm.v1.core.kv_cache_utils._pool_bytes_per_block = _ascend_pool_bytes_per_block

# Also patch the reference used by engine/core.py which imports the function directly.
import vllm.v1.engine.core  # noqa: E402

vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve_kv_cache_block_sizes
