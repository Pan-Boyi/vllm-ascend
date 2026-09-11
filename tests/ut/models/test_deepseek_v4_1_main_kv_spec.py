# SPDX-License-Identifier: Apache-2.0
"""The compressed-KV spec has to reproduce the DSV4 block-size table.

What this guards is the spec drifting from that table. The table is the
kernels' contract -- it is what select_common_block_size is handed and what the
runner reshapes against -- so "matches the table" is the criterion, not "differs
from upstream by N bytes". Upstream's number is not depended on by anything;
the table's is depended on by everything.

The reason a test exists at all is that the two conventions disagree in a way
that looks like a wash and is not. Upstream fixes block_size at the cache
config's value and lets storage_block_size shrink by the compression ratio;
Ascend keeps the storage page at the table's value and multiplies block_size
up. At ratio 1 they agree, so any check written only against a ratio-1 layer
passes under both and proves nothing.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec
from vllm_ascend.models.deepseek_v4_1.attention import \
    AscendDeepseekV41Attention
from vllm_ascend.models.layer.attention.layer import dsv4_block_sizes

HEAD_DIM = 512
CACHE_BLOCK = 128
STORAGE_BLOCK = 128  # DSV4 block-size table, cache block 128, entry [0][0]

#: The two layouts, written out rather than read from DSV4_BLOCK_SIZES.
#: That table is built at *import* time from the hardware profile
#: (layer.py:58, ``DSV4_BLOCK_SIZES = get_dsv4_block_sizes()``), so on a host
#: without DSV4_COMPRESSED_CACHE it is the uncompressed table and the numbers
#: below would be compared against the wrong row. Stating them keeps the test
#: honest about which configuration it is describing.
BF16_PAGE = STORAGE_BLOCK * HEAD_DIM * 2            # 512 bf16 elements: 131072
FP8_PAGE = STORAGE_BLOCK * (HEAD_DIM + 128) * 1     # 640 fp8 bytes:      81920

#: Every kv-source layer in the checkpoint and the ratio it carries.
#: kv_source_layer_ids = [2, 8, 14, 20]; compress_ratios gives 2, 2, 2, 1.
KV_SOURCE_RATIOS = {2: 2, 8: 2, 14: 2, 20: 1}


def _layer(ratio, *, is_kv_source=True, kv_cache_dtype="auto"):
    return SimpleNamespace(
        is_kv_source=is_kv_source,
        head_dim=HEAD_DIM,
        compress_ratio=ratio,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_torch_dtype=torch.bfloat16,
    )


def _config(block_size=CACHE_BLOCK, cache_dtype="auto"):
    return SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size,
                                     cache_dtype=cache_dtype))


def _spec(monkeypatch, ratio, *, bf16, compressed, **layer_kwargs):
    from vllm_ascend.models.deepseek_v4_1 import attention as mod

    monkeypatch.setattr(
        "vllm_ascend.attention.dsa_attn_kv_plan.is_a5_bf16_kv_enabled",
        lambda cfg: bf16)

    class _Profile:
        @staticmethod
        def supports(cap):
            return compressed

    monkeypatch.setattr(
        "vllm_ascend.device.hardware_profile.get_current_hardware_profile",
        lambda: _Profile)
    cfg = _config()
    return mod.AscendDeepseekV41Attention.get_kv_cache_spec(
        _layer(ratio, **layer_kwargs), cfg), cfg


def test_a_non_source_layer_owns_no_compressed_cache(monkeypatch):
    """Consumers read the source's cache; ratio-0 layers are pure SWA.

    Returning a spec here would allocate a cache for every layer in the
    sharing group -- 51 of them instead of 4.
    """
    spec, _ = _spec(monkeypatch, 2, bf16=True, compressed=True,
                    is_kv_source=False)
    assert spec is None


def test_it_is_the_ascend_spec_not_upstreams(monkeypatch):
    """Three runner predicates test this class by isinstance and take a
    non-Ascend branch when it fails."""
    spec, _ = _spec(monkeypatch, 2, bf16=True, compressed=True)
    assert isinstance(spec, AscendMLAAttentionSpec)


@pytest.mark.parametrize("ratio", sorted(set(KV_SOURCE_RATIOS.values())))
def test_bf16_page_matches_the_table(monkeypatch, ratio):
    spec, _ = _spec(monkeypatch, ratio, bf16=True, compressed=True)
    assert spec.real_page_size_bytes == BF16_PAGE == 131072


@pytest.mark.parametrize("ratio", sorted(set(KV_SOURCE_RATIOS.values())))
def test_fp8_page_matches_the_table(monkeypatch, ratio):
    """The compressed layout is 640 B per state: 448 NoPE + 128 RoPE + scale
    + pad, expressed as 128 extra head elements at one byte."""
    spec, _ = _spec(monkeypatch, ratio, bf16=False, compressed=True)
    assert spec.real_page_size_bytes == FP8_PAGE == 81920


def test_the_page_does_not_depend_on_the_ratio(monkeypatch):
    """The whole point of the Ascend convention, and the thing a ratio-1-only
    test cannot see.

    The storage page is fixed by the table; the ratio moves block_size instead.
    Under upstream's convention the ratio-2 page would be half the ratio-1 one,
    and every layer would still allocate, reshape and run.
    """
    pages = set()
    for ratio in (1, 2):
        spec, _ = _spec(monkeypatch, ratio, bf16=True, compressed=True)
        pages.add(spec.real_page_size_bytes)
    assert len(pages) == 1, f"page must not vary with the ratio, got {pages}"


@pytest.mark.parametrize("ratio,expected_block", [(1, 128), (2, 256)])
def test_block_size_is_the_storage_page_times_the_ratio(monkeypatch, ratio,
                                                        expected_block):
    """256 at ratio 2 is why AscendDSAC2Backend lists [64, 128, 256].

    Table, spec convention and backend support set are three exits from one
    decision. If this number moves, that backend stops accepting the layer.
    """
    spec, _ = _spec(monkeypatch, ratio, bf16=True, compressed=True)
    assert spec.block_size == expected_block
    assert spec.storage_block_size == STORAGE_BLOCK


def test_the_ratio_two_block_size_is_one_the_c2_backend_accepts(monkeypatch):
    from vllm_ascend.attention.dsa_v1 import AscendDSAC2Backend

    spec, _ = _spec(monkeypatch, 2, bf16=True, compressed=True)
    assert spec.block_size in AscendDSAC2Backend.get_supported_kernel_block_sizes()


def test_ratio_zero_cannot_produce_a_zero_block_size(monkeypatch):
    """0 % n == 0 for every n, so select_common_block_size would not reject a
    block size of 0 -- it would silently return the largest supported page.

    No kv-source layer carries ratio 0 today; this is the guard for the day one
    does, because the symptom is a page nobody asked for rather than an error.
    """
    spec, _ = _spec(monkeypatch, 0, bf16=True, compressed=True)
    assert spec.block_size > 0
    assert spec.tokens_per_state == 1


def test_compressed_fp8_pins_the_cache_dtype_string(monkeypatch):
    """The spec and the cache config have to agree, or the reshape path and the
    kernels read the same bytes as different layouts."""
    spec, cfg = _spec(monkeypatch, 2, bf16=False, compressed=True)
    assert cfg.cache_config.cache_dtype == "float8_e4m3fn"
    assert spec.dtype == torch.float8_e4m3fn
    assert spec.cache_dtype_str == "float8_e4m3fn"


def test_bf16_leaves_the_cache_dtype_alone(monkeypatch):
    spec, cfg = _spec(monkeypatch, 2, bf16=True, compressed=True)
    assert cfg.cache_config.cache_dtype == "auto"
    assert spec.dtype == torch.bfloat16


def test_the_head_size_grows_only_for_the_compressed_layout(monkeypatch):
    """448 NoPE + 128 RoPE + scale + pad is 640; bf16 stores 512 and no scale.

    Carrying the +128 into bf16 would over-allocate by 25% and leave the tail
    unread -- which is not an error anywhere, just memory that never comes back.
    """
    bf16_spec, _ = _spec(monkeypatch, 2, bf16=True, compressed=True)
    fp8_spec, _ = _spec(monkeypatch, 2, bf16=False, compressed=True)
    assert bf16_spec.head_size == HEAD_DIM
    assert fp8_spec.head_size == HEAD_DIM + 128


def test_the_storage_block_the_spec_uses_is_the_tables(monkeypatch):
    """Pin the one number this spec does read from the table.

    Everything else above is stated as a literal on purpose; this is the single
    value the implementation looks up, so it is the single place the table can
    drift out from under it.
    """
    spec, cfg = _spec(monkeypatch, 2, bf16=True, compressed=True)
    table = dsv4_block_sizes(cfg)
    assert spec.storage_block_size == table[CACHE_BLOCK][0][0]
