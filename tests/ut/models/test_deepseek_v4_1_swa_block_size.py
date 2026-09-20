# SPDX-License-Identifier: Apache-2.0
"""The SWA cache's block size and the switch that changes it -- CPU only.

Upstream hard-codes 32 at the construction site
(``vllm/models/deepseek_v4_1/attention.py:456``) and the Ascend subclass
overrode ``get_kv_cache_spec`` and not the block size, so V4.1 runs the sliding
window at a 32-token page while Ascend's own V4.0 path runs the same backend
and the same spec class at ``DSV4_BLOCK_SIZES[128][0][1] = 128``
(``models/deepseek_v4/model.py:124``).

``VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE`` is 0 by default, so the identity arm below
is the one that runs in production today. Every other test states a criterion
and is followed by a mutation arm showing that criterion can go red.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backend import MultipleOf

from vllm_ascend.models.deepseek_v4_1 import caches

HEAD_DIM, WINDOW = 512, 2048
UPSTREAM_BLOCK = 32          # vllm/models/deepseek_v4_1/attention.py:456
MAIN_STORAGE_BLOCK = 128     # layer.py:46 _DSV4_BLOCK_SIZES_A5_BF16[128][0][0]


class _SWABackend:
    """Stands in for AscendDSASWABackend: plain ints, 32/64/128."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [32, 64, 128]


class _MultipleOfBackend:
    """Upstream's DeepseekSparseSWABackend surface: MultipleOf(32) only."""

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(32)]


def _resolve(env_value, backend=_SWABackend, constructed=UPSTREAM_BLOCK,
             main_block=MAIN_STORAGE_BLOCK, monkeypatch=None, fn=None):
    from vllm_ascend import envs
    monkeypatch.setattr(envs, "VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE", env_value,
                        raising=False)
    return (fn or caches.resolve_swa_block_size)(
        constructed, backend, "model.layers.0.self_attn.swa_cache", main_block)


# -- the real supported list, read rather than assumed ----------------------


def test_the_ascend_swa_backend_really_lists_128():
    """The switch is only usable because the backend this cache returns lists
    128 verbatim. Read it off the class, not off the docstring."""
    from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend

    assert 128 in AscendDSASWABackend.get_supported_kernel_block_sizes()


def test_ascend_v40_already_runs_the_sliding_window_at_128():
    """Not a preference: the sibling model on this hardware, on this backend
    and this spec class, takes its SWA block size from the block-size table's
    swa column, which is 128 at the default cache block size."""
    from vllm_ascend.models.layer.attention.layer import (
        DSV4_BLOCK_SIZES, DSV4_BLOCK_SIZES_A5_BF16)

    assert DSV4_BLOCK_SIZES[128][0][1] == 128
    assert DSV4_BLOCK_SIZES_A5_BF16[128][0][1] == 128


# -- the switch --------------------------------------------------------------


def test_off_by_default_keeps_the_constructed_block_size(monkeypatch):
    """The identity arm. Off must be byte-identical to before the switch
    existed, or an A/B cannot attribute a launch to this change."""
    assert _resolve(0, monkeypatch=monkeypatch) == UPSTREAM_BLOCK


def test_a_supported_value_replaces_it(monkeypatch):
    assert _resolve(128, monkeypatch=monkeypatch) == 128
    assert _resolve(64, monkeypatch=monkeypatch) == 64


def test_an_unsupported_value_raises_and_names_the_backend(monkeypatch):
    with pytest.raises(NotImplementedError) as e:
        _resolve(96, monkeypatch=monkeypatch)
    assert "_SWABackend" in str(e.value)
    assert "96" in str(e.value)


def test_divisible_is_not_verbatim(monkeypatch):
    """128 is a multiple of 32, and a MultipleOf(32) backend would accept it
    under a divisibility test. It must still be refused: the list has no
    verbatim entry, so select_common_block_size would pick something else and
    the planner and the operator would disagree with nothing raised."""
    with pytest.raises(NotImplementedError):
        _resolve(128, backend=_MultipleOfBackend, monkeypatch=monkeypatch)


def test_divisible_is_not_verbatim_MUTANT_goes_red(monkeypatch):
    """Mutation arm for the test above: a resolver that tests divisibility
    instead of membership accepts the MultipleOf backend, so the criterion
    fails. If it did not, that test would be passing for another reason."""

    def mutant(constructed, backend_cls, prefix, main_storage_block_size):
        from vllm_ascend import envs
        override = envs.VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE
        if not override:
            return constructed
        for item in backend_cls.get_supported_kernel_block_sizes():
            base = item.base if isinstance(item, MultipleOf) else item
            if override % base == 0 or override == base:
                return override
        raise NotImplementedError("mutant")

    assert _resolve(128, backend=_MultipleOfBackend, monkeypatch=monkeypatch,
                    fn=mutant) == 128


def test_a_page_larger_than_the_main_page_raises(monkeypatch):
    """Past the ceiling the planner fails on a bare assert with no
    attribution, so the refusal has to happen here."""
    with pytest.raises(NotImplementedError) as e:
        _resolve(128, main_block=32, monkeypatch=monkeypatch)
    assert "32" in str(e.value)


def test_the_ceiling_MUTANT_goes_red(monkeypatch):
    """Mutation arm: drop the ceiling check and the value sails through."""

    def mutant(constructed, backend_cls, prefix, main_storage_block_size):
        from vllm_ascend import envs
        override = envs.VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE
        if not override:
            return constructed
        if override not in [s for s in
                            backend_cls.get_supported_kernel_block_sizes()
                            if isinstance(s, int)]:
            raise NotImplementedError("mutant")
        return override

    assert _resolve(128, main_block=32, monkeypatch=monkeypatch,
                    fn=mutant) == 128


def test_a_negative_value_raises(monkeypatch):
    with pytest.raises(ValueError):
        _resolve(-1, monkeypatch=monkeypatch)


# -- the block size must land on the layer, not only in the spec -------------


def _cache(block_size):
    """An AscendDeepseekV41SWACache without running __init__.

    __init__ registers a prefix in the compilation context and needs a current
    vllm_config, neither of which this file has; the fields it would set are
    set here instead. The block-size resolution is exercised through
    ``resolve_swa_block_size`` directly above and through ``_construct`` below.
    """
    cache = object.__new__(caches.AscendDeepseekV41SWACache)
    torch.nn.Module.__init__(cache)
    cache.head_dim = HEAD_DIM
    cache.window_size = WINDOW
    cache.block_size = block_size
    cache.dtype = torch.bfloat16
    cache.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    cache.backend_cls = _SWABackend
    cache.stream_compress_ratio = 0
    return cache


def test_the_spec_reports_the_layer_block_size():
    """The prolog checks ``swa_kv_cache.shape[1:] == (metadata.block_size,
    head_dim)`` and the JIT warmup registers kernels with
    ``swa_cache_layer.block_size``. Spec and layer disagreeing writes rows at
    the wrong offset rather than failing, so they are one value."""
    for bs in (32, 64, 128):
        cache = _cache(bs)
        assert cache.get_kv_cache_spec(None).block_size == cache.block_size == bs


def test_the_spec_reports_the_layer_block_size_MUTANT_goes_red():
    """Mutation arm: a subclass that keeps the new size in a private field and
    leaves ``self.block_size`` at 32 -- the shape the change is meant to
    prevent -- fails the equality above."""

    class Mutant(caches.AscendDeepseekV41SWACache):
        def get_kv_cache_spec(self, vllm_config):
            spec = super().get_kv_cache_spec(vllm_config)
            object.__setattr__(spec, "block_size", self._spec_block_size)
            return spec

    cache = object.__new__(Mutant)
    torch.nn.Module.__init__(cache)
    cache.head_dim, cache.window_size = HEAD_DIM, WINDOW
    cache.block_size = UPSTREAM_BLOCK
    cache._spec_block_size = 128
    cache.dtype = torch.bfloat16
    cache.cache_config = SimpleNamespace(cache_dtype="bfloat16")
    cache.backend_cls = _SWABackend
    cache.stream_compress_ratio = 0
    assert cache.get_kv_cache_spec(None).block_size != cache.block_size


def test_128_makes_the_page_exactly_the_full_mla_page():
    """Why 128 and not 64: at 128 the SWA page equals the full-MLA page, so
    ``size_to_candidate`` pads it by 1.00x instead of 4.00x (at 32) or 2.00x
    (at 64). Same head_dim, same bf16 element size on this path."""
    mla_page = MAIN_STORAGE_BLOCK * HEAD_DIM * 2
    assert _cache(128).get_kv_cache_spec(None).page_size_bytes == mla_page
    assert _cache(32).get_kv_cache_spec(None).page_size_bytes * 4 == mla_page


# -- the switch is wired into __init__, not only available as a function -----


def test_the_constructor_calls_the_resolver():
    """A resolver nobody calls is the defect it claims to fix, under a name
    that says it is fixed. Checked on the class's own __init__ source."""
    import ast
    import inspect

    src = inspect.getsource(caches.AscendDeepseekV41SWACache.__init__)
    tree = ast.parse(src.lstrip())
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "resolve_swa_block_size" in called
    # ... and its result is assigned to self.block_size, not to a local.
    targets = {ast.unparse(t) for n in ast.walk(tree)
               if isinstance(n, ast.Assign) for t in n.targets}
    assert "self.block_size" in targets


# -- the real planner, not a model of it ------------------------------------


def _dsv41_specs(swa_block_size):
    """The V4.1 spec dict as the flight builds it: 4 kv-source layers
    (ratios 2,2,2,1) with their indexer K caches, 3 compressor rings, and 40
    sliding-window layers. Values match the 2026-09-20 single-card flight
    (block_size 128, kv_cache_dtype bfloat16, A5 bf16 KV path).
    """
    from vllm.v1.kv_cache_interface import CircularBufferSpec

    from vllm_ascend.core.kv_cache_interface import (
        AscendMLAAttentionSpec, AscendSlidingWindowMLASpec)

    ratios = {2: 2, 8: 2, 14: 2, 20: 1}
    specs = {}
    for layer, ratio in ratios.items():
        common = dict(num_kv_heads=1, model_version="deepseek_v4",
                      cache_dtype_str="bfloat16", scale_dtype=torch.float,
                      tokens_per_state=ratio, block_size=128 * ratio)
        specs[f"L{layer}.attn"] = AscendMLAAttentionSpec(
            head_size=HEAD_DIM, dtype=torch.bfloat16, scale_dim=0, **common)
        specs[f"L{layer}.idx"] = AscendMLAAttentionSpec(
            head_size=128, dtype=torch.float8_e4m3fn, scale_dim=1, **common)
        if ratio > 1:
            specs[f"L{layer}.ring"] = CircularBufferSpec(
                block_size=8, num_kv_heads=1, head_size=1024, head_size_v=0,
                dtype=torch.float32)
    for layer in range(40):
        specs[f"L{layer}.swa"] = AscendSlidingWindowMLASpec(
            block_size=swa_block_size, num_kv_heads=1, head_size=HEAD_DIM,
            dtype=torch.bfloat16, sliding_window=128,
            cache_dtype_str="bfloat16", model_version="deepseek_v4",
            alignment=None)
    return specs


def _plan(swa_block_size, available_bytes=29350261248):
    """Run the patched planner that actually runs in the worker."""
    import math
    from types import SimpleNamespace

    from vllm.v1.kv_cache_interface import CircularBufferSpec

    import vllm_ascend.patch.platform.patch_kv_cache_utils as P

    cfg = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None,
                                     block_size=128),
        model_config=SimpleNamespace(max_model_len=8192),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=8192,
        compilation_config=SimpleNamespace(static_forward_context={}))
    specs = _dsv41_specs(swa_block_size)
    groups = P._get_kv_cache_groups_uniform_groups(
        P.group_and_unify_kv_cache_specs(specs))
    rings = {n: s for n, s in specs.items()
             if isinstance(s, CircularBufferSpec)}
    groups = groups + P._ring_groups(rings, cfg)
    num_blocks, tensors = P._get_kv_cache_config_deepseek_v4_main(
        cfg, groups, available_bytes)
    swa_groups = [g for g in groups if g.layer_names[0].endswith(".swa")]
    return SimpleNamespace(
        groups=groups,
        block_sizes=[g.kv_cache_spec.block_size for g in groups],
        lcm=math.lcm(*[g.kv_cache_spec.block_size for g in groups]),
        pool_bytes_per_block=P._ascend_pool_bytes_per_block(groups),
        num_blocks=num_blocks,
        backing_size=tensors[0].size,
        swa_group_count=len(swa_groups),
        swa_page=swa_groups[0].kv_cache_spec.page_size_bytes,
        swa_blocks_per_req=len(swa_groups) * P.cdiv(
            swa_groups[0].kv_cache_spec.max_memory_usage_bytes(cfg),
            swa_groups[0].kv_cache_spec.page_size_bytes),
        admission_bytes=P._ascend_max_memory_usage_bytes_from_groups(cfg,
                                                                     groups),
    )


def test_the_shipped_plan_reproduces_our_flight():
    """Positive control for the two tests below: at the shipped 32 the real
    planner lands exactly on what our 2026-09-20 flights printed -- 43 groups,
    [128, 256, 32x40, 8], lcm 256, 27.3346 GiB backing.
    """
    p = _plan(32)
    assert len(p.groups) == 43
    assert p.block_sizes[:3] == [128, 256, 32]
    assert p.lcm == 256
    assert p.pool_bytes_per_block == 542208
    assert p.num_blocks == 54131
    assert p.backing_size == 29350261248
    assert p.swa_group_count == 40          # 40 single-layer SWA groups
    assert p.swa_page == 131072             # 32*512*2 = 32 KiB, padded 4.00x
    assert p.swa_blocks_per_req == 40 * 257


def test_at_128_the_planner_keeps_the_same_pool_and_quarters_the_admission():
    """What L3 does and does not do, from the planner itself: the pool block
    and the pool size are untouched; what shrinks is blocks per request.
    """
    a, b = _plan(32), _plan(128)
    assert b.block_sizes[:3] == [128, 256, 128]
    assert b.lcm == a.lcm == 256                          # scheduler unchanged
    assert b.pool_bytes_per_block == a.pool_bytes_per_block
    assert b.num_blocks == a.num_blocks
    assert b.backing_size == a.backing_size
    assert b.swa_group_count == a.swa_group_count == 40
    assert b.swa_page == 131072 == 128 * HEAD_DIM * 2     # exact, no padding
    # 40 x (cdiv(8192, bs) + 1): 40x257 -> 40x65. Not exactly 4x -- the "+1"
    # for a window that may not start on a block boundary is per group and
    # does not shrink.
    assert (a.swa_blocks_per_req, b.swa_blocks_per_req) == (40 * 257, 40 * 65)
    assert a.swa_blocks_per_req / b.swa_blocks_per_req == pytest.approx(3.95,
                                                                        abs=0.01)
    # the startup admission check gets looser, not tighter
    assert b.admission_bytes < a.admission_bytes


def test_off_returns_before_touching_the_block_size_table():
    """The identity arm must not gain a new way to fail. The table lookup
    ``dsv4_block_sizes(cfg)[cache_config.block_size]`` raises KeyError on a
    cache block size outside {32,64,128}; with the switch off it must not run
    at all.
    """
    import ast
    import inspect

    src = inspect.getsource(caches.AscendDeepseekV41SWACache.__init__)
    tree = ast.parse(src.lstrip())
    early_returns = [n.lineno for n in ast.walk(tree)
                     if isinstance(n, ast.Return) and n.value is None]
    table_uses = [n.lineno for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and n.id == "dsv4_block_sizes"]
    assert early_returns and table_uses
    assert min(early_returns) < min(table_uses)
