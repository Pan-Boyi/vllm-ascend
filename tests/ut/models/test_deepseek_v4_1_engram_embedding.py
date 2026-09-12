# SPDX-License-Identifier: Apache-2.0
"""The engram table must not be allocated, and that must be enforced.

The first real launch died in upstream's ``ParallelEngramEmbedding.__init__``
with ``rtsMallocHost ... out of memory``. It was trying the pinned-host path --
the one that took an NPU down -- and what stopped it was the host being too
small, not any decision of ours. On a larger machine it would have succeeded.

So the tests that matter here are not "does it fail on this host". They are
"does it fail for the right reason, including when the allocation would have
worked".
"""

import pytest
import torch

from vllm_ascend.models.deepseek_v4_1.engram_embedding import (
    AscendStagedEngramEmbedding, tables_not_allocated)


def test_allocations_are_diverted_to_meta():
    with tables_not_allocated():
        t = torch.empty(1024, 256, dtype=torch.float8_e4m3fn)
    assert t.device.type == "meta"
    assert t.shape == (1024, 256) and t.dtype == torch.float8_e4m3fn


def test_pin_memory_is_dropped_not_honoured():
    """The specific argument that reaches aclrtMallocHostWithCfg."""
    with tables_not_allocated():
        t = torch.empty(8, 8, dtype=torch.uint8, device="cpu", pin_memory=True)
    assert t.device.type == "meta"
    assert not t.is_pinned()


def test_torch_empty_is_restored_afterwards():
    before = torch.empty
    with tables_not_allocated():
        pass
    assert torch.empty is before
    assert torch.empty(4).device.type != "meta"


def test_it_is_restored_even_if_construction_raises():
    before = torch.empty
    with pytest.raises(ValueError):
        with tables_not_allocated():
            raise ValueError("constructor blew up")
    assert torch.empty is before


# -- the guard, which must not depend on running out of memory ---------------


class _Fake(AscendStagedEngramEmbedding):
    """Bypasses upstream's __init__; only the guard is under test."""

    def __init__(self, weight, scale):
        self.weight = weight
        self.weight_scale_inv = scale


def test_meta_parameters_are_accepted():
    f = _Fake(torch.empty(1024, 256, dtype=torch.float8_e4m3fn, device="meta"),
              torch.empty(1024, 8, dtype=torch.uint8, device="meta"))
    f._assert_nothing_was_allocated()


def test_the_guard_fires_on_an_allocation_that_SUCCEEDED():
    """The whole point.

    Last time the banned path was stopped by OOM. A guard that only triggers
    when the allocation fails is not a guard -- it is the machine's memory
    limit wearing one. This allocation is tiny and succeeds, and the guard must
    still refuse.
    """
    f = _Fake(torch.empty(16, 8, dtype=torch.float8_e4m3fn),   # real, succeeds
              torch.empty(16, 1, dtype=torch.uint8, device="meta"))
    with pytest.raises(RuntimeError) as exc:
        f._assert_nothing_was_allocated()
    msg = str(exc.value)
    assert "weight" in msg
    # and it has to say that OOM is not the guard, or the next person reads
    # "it failed on this host" as "it is prevented"
    assert "running out of memory here is not the guard" in msg


def test_the_scale_parameter_is_checked_too():
    f = _Fake(torch.empty(16, 8, dtype=torch.float8_e4m3fn, device="meta"),
              torch.empty(16, 1, dtype=torch.uint8))          # real
    with pytest.raises(RuntimeError, match="weight_scale_inv"):
        f._assert_nothing_was_allocated()


def test_storage_is_a_dead_end():
    """_storage() is upstream's UVA / host-registration entry point."""
    f = _Fake(torch.empty(1, 1, device="meta"), torch.empty(1, 1, device="meta"))
    with pytest.raises(RuntimeError) as exc:
        f._storage()
    msg = str(exc.value)
    assert "must not be reached" in msg
    assert "read-only mapping" in msg


def test_unbound_tables_say_how_to_bind_them():
    f = _Fake(torch.empty(1, 1, device="meta"), torch.empty(1, 1, device="meta"))
    f._layer_id = 14
    with pytest.raises(RuntimeError, match="bind_staged_tables"):
        f.staged_tables()


def test_the_mixin_precedes_upstream_so_its_init_wins():
    from vllm_ascend.models.deepseek_v4_1.caches import AscendEngramEmbedding

    mro = [c.__name__ for c in AscendEngramEmbedding.__mro__[:3]]
    assert mro == ["AscendEngramEmbedding", "AscendStagedEngramEmbedding",
                   "ParallelEngramEmbedding"], (
        "upstream's __init__ must not win -- it allocates the shard before "
        f"anything else gets a say; got {mro}")
    assert "__init__" in vars(AscendStagedEngramEmbedding)


# -- the lookup path, up to the kernel seam ----------------------------------

from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout

LAYOUT = EngramLayout(n_hash_cols=24, dim=256, quant_block=32,
                      layer_ids=(1,), num_embeddings=(512,))


class _Lookup(AscendStagedEngramEmbedding):
    """Enough of upstream's surface to exercise lookup(); no NPU involved."""

    def __init__(self, tables, part=24, total=24, start=0):
        import numpy as np
        self.weight = torch.empty(1, 1, device="meta")
        self.weight_scale_inv = torch.empty(1, 1, device="meta")
        self.part_n_hash_cols, self.n_hash_cols = part, total
        self.head_start = start
        self._tables = tables
        self.dequantised_rows = None
        self._np = np

    def staged_tables(self):
        return self._tables

    def _dequantize_staged(self, rows, out):     # the seam, stubbed
        self.dequantised_rows = rows


def _tables(rows=512):
    import numpy as np
    w = np.zeros((rows, 256), dtype=np.uint8)
    s = np.zeros((rows, 8), dtype=np.uint8)
    w[:, 0] = np.arange(rows) % 256
    s[:, 0] = np.arange(rows) % 251
    return w, s


def test_lookup_stages_the_rows_each_token_asked_for():
    import numpy as np

    tables = _tables()
    layer = _Lookup(tables)
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")

    rng = np.random.default_rng(11)
    ids = rng.integers(0, 512, size=(5, 24)).astype(np.int32)
    layer.lookup(torch.from_numpy(ids), out=None)

    assert layer.dequantised_rows == 5 * 24
    for t in range(5):
        for c in range(24):
            r = LAYOUT.row_index(t, c)
            assert layer._dev_weight[r, 0].item() == tables[0][ids[t, c], 0]
            assert layer._dev_scale[r, 0].item() == tables[1][ids[t, c], 0]


def test_flattening_indices_matches_upstreams_row_numbering():
    """Upstream numbers rows token * LOCAL_HEADS + head and reads
    ``indices + token * TOTAL_HEADS + head``. With the columns undivided that
    is row-major order, which is what lookup() relies on -- so this pins the
    two numbering schemes against each other rather than trusting they agree.
    """
    import numpy as np

    ids = np.arange(5 * 24, dtype=np.int32).reshape(5, 24)
    flat = ids.reshape(-1)
    for t in range(5):
        for c in range(24):
            assert flat[LAYOUT.row_index(t, c)] == ids[t, c]


def test_an_empty_batch_does_nothing():
    layer = _Lookup(_tables())
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
    layer.lookup(torch.zeros(0, 24, dtype=torch.int32), out=None)
    assert layer.dequantised_rows is None


def test_more_tokens_than_the_buffers_were_sized_for_is_refused():
    layer = _Lookup(_tables())
    layer.allocate_staging(LAYOUT, max_tokens=4, device="cpu")
    with pytest.raises(ValueError, match="staging buffers were sized for"):
        layer.lookup(torch.zeros(5, 24, dtype=torch.int32), out=None)


def test_lookup_without_staging_says_what_to_call():
    layer = _Lookup(_tables())
    with pytest.raises(RuntimeError, match="allocate_staging"):
        layer.lookup(torch.zeros(2, 24, dtype=torch.int32), out=None)


def test_lookup_refuses_a_sharded_configuration():
    """The row numbering is only right while the hash columns are undivided."""
    layer = _Lookup(_tables(), part=12, total=24, start=0)
    layer.allocate_staging(LAYOUT, max_tokens=8, device="cpu")
    with pytest.raises(ValueError, match="undivided hash columns"):
        layer.lookup(torch.zeros(2, 24, dtype=torch.int32), out=None)


def test_the_kernel_resolves_and_takes_the_agreed_interface():
    import inspect

    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _staged_dequant_launcher

    launch = _staged_dequant_launcher()
    params = inspect.signature(launch).parameters
    assert list(params)[:3] == ["staging_weight", "staging_scale", "out"]
    assert "dim" in params and "quant_block" in params


def test_the_binding_records_the_kernels_verification_status():
    """Binding is not endorsing, so the status is written down next to it.

    The arithmetic is settled on the CPU against golden's rows. The kernel has
    run -- the Triton compile cache holds two compilations of it, and Triton
    only compiles at first launch. Its file header still claims otherwise,
    which is exactly what produced the wrong conclusion here on first reading:
    an artefact settles this, a comment does not.
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _staged_dequant_launcher

    doc = _staged_dequant_launcher.__doc__
    assert "arithmetic is settled" in doc
    assert "has run on hardware" in doc
    assert "stale" in doc
    assert "compile cache is" in doc


def test_the_resolver_prefers_a_real_package_location():
    """It lives under a directory named for the design it replaced. When it
    moves somewhere sane this keeps working without an edit."""
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _KERNEL_LOCATIONS

    assert _KERNEL_LOCATIONS[0][1] is None, "the package path must be tried first"
    assert _KERNEL_LOCATIONS[0][0].startswith("vllm_ascend.")


def test_an_unresolvable_kernel_says_what_is_already_done(monkeypatch):
    from vllm_ascend.models.deepseek_v4_1 import engram_embedding as mod

    monkeypatch.setattr(mod, "_KERNEL_LOCATIONS",
                        (("nowhere.at.all", None),
                         ("also_nowhere", "/nonexistent/path.py")))
    with pytest.raises(RuntimeError) as exc:
        mod._staged_dequant_launcher()
    msg = str(exc.value)
    assert "already\nin HBM" in msg or "already in HBM" in msg
    assert "nowhere.at.all" in msg and "/nonexistent/path.py" in msg


# -- the log line that became untrue -----------------------------------------


def test_the_stale_offload_line_is_dropped():
    """Upstream logs it from inside `if cpu_offload:` regardless of what the
    allocation did, so with the allocation diverted it reports an intention."""
    import logging

    from vllm_ascend.models.deepseek_v4_1.engram_embedding import (
        _STALE_OFFLOAD_LOG, _upstream_offload_line_suppressed)

    logger = logging.getLogger("vllm.models.deepseek_v4_1.common.engram")
    seen = []
    sink = logging.Handler()
    sink.emit = lambda record: seen.append(record.getMessage())
    logger.addHandler(sink)
    logger.setLevel(logging.INFO)
    try:
        with _upstream_offload_line_suppressed():
            logger.info("Engram table %s: 384006168 rows x 256, 94.42 GiB per rank",
                        _STALE_OFFLOAD_LOG)
            logger.info("Built engram token map (129280 -> 99092 ids)")
        logger.info("after the context, %s", _STALE_OFFLOAD_LOG)
    finally:
        logger.removeHandler(sink)

    # the untrue line, identified by its number rather than by position
    assert not any("94.42 GiB per rank" in m for m in seen), (
        f"the untrue line reached the log: {seen}")
    # the filter is surgical: other lines from the same logger survive
    assert any("Built engram token map" in m for m in seen)
    # and it is removed with the context, not left on the logger forever
    assert any(m.startswith("after the context") for m in seen), (
        "the filter must come off again, or it would hide a line that becomes "
        "true again if the design changes back")


def test_the_number_it_printed_was_specific_enough_to_believe():
    """Why suppressing it beats leaving it.

    94.42 GiB per rank over four ranks and two layers is 755.4 GiB, against a
    755 GiB host. A reader would conclude host memory had just been exhausted.
    The measured figure is VmPin = 0.
    """
    per_rank_per_layer = 94.42
    assert round(per_rank_per_layer * 4 * 2, 1) == 755.4


def test_the_replacement_line_says_what_actually_happened():
    import inspect

    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        AscendStagedEngramEmbedding

    source = inspect.getsource(AscendStagedEngramEmbedding.__init__)
    assert "0 bytes pinned per rank" in source
    assert "does not happen" in source


# -- the torch route, used to attribute a compiler abort ----------------------


def test_torch_dequant_matches_an_independent_computation():
    """The kernel's arithmetic, checked against a separate expression.

    Not a refactor of the same lines: the reference below multiplies by
    2**(byte-127) explicitly, while the implementation bit-casts the byte into
    the exponent field. Those agreeing is the point -- it is the same trick
    verified two ways.
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        dequantize_staged_torch

    rows, dim, qb = 4, 64, 32
    torch.manual_seed(5)
    values = (torch.randn(rows, dim) * 3).to(torch.float8_e4m3fn)
    weight_u8 = values.view(torch.uint8)
    # ue8m0 exponents around 1.0, avoiding the extremes
    scale_u8 = torch.randint(120, 134, (rows, dim // qb), dtype=torch.uint8)

    out = torch.zeros(rows, dim, dtype=torch.bfloat16)
    dequantize_staged_torch(weight_u8, scale_u8, out, quant_block=qb)

    factors = torch.pow(2.0, scale_u8.to(torch.float32) - 127.0)
    expected = (values.to(torch.float32).view(rows, dim // qb, qb)
                * factors.unsqueeze(-1)).view(rows, dim).to(torch.bfloat16)
    torch.testing.assert_close(out, expected)


def test_each_scale_covers_its_own_block_only():
    """The broadcast, checked positionally.

    A scale applied to the wrong block is the failure that produced maxdiff 28
    on this data with no error raised, so it is checked by construction rather
    than by a tolerance.
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        dequantize_staged_torch

    rows, dim, qb = 2, 64, 32
    ones = torch.ones(rows, dim, dtype=torch.float8_e4m3fn).view(torch.uint8)
    scale_u8 = torch.tensor([[127, 128], [129, 127]], dtype=torch.uint8)

    out = torch.zeros(rows, dim, dtype=torch.float32)
    dequantize_staged_torch(ones, scale_u8, out, quant_block=qb)

    assert torch.all(out[0, :32] == 1.0)    # 2**0
    assert torch.all(out[0, 32:] == 2.0)    # 2**1
    assert torch.all(out[1, :32] == 4.0)    # 2**2
    assert torch.all(out[1, 32:] == 1.0)


def test_a_scale_buffer_of_the_wrong_width_is_refused():
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        dequantize_staged_torch

    with pytest.raises(ValueError, match="scale buffer is"):
        dequantize_staged_torch(
            torch.zeros(4, 64, dtype=torch.uint8),
            torch.zeros(4, 4, dtype=torch.uint8),      # should be 2
            torch.zeros(4, 64), quant_block=32)


def test_the_switch_is_off_by_default_and_selects_the_torch_route():
    """It is a bisection instrument, not a fallback: the kernel does one pass
    where this materialises several intermediates."""
    import vllm_ascend.envs as envs
    from vllm_ascend.models.deepseek_v4_1 import engram_embedding as mod

    assert envs.VLLM_ASCEND_DSV41_ENGRAM_DEQUANT_TORCH == "0"
    assert not mod.envs_dsv41_engram_dequant_torch()

    import inspect
    source = inspect.getsource(mod.AscendStagedEngramEmbedding._dequantize_staged)
    assert "envs_dsv41_engram_dequant_torch()" in source
    assert source.index("envs_dsv41_engram_dequant_torch()") < source.index(
        "_staged_dequant_launcher()"), (
        "the switch must be checked before the kernel is even resolved -- "
        "resolving it imports the module whose compilation is under suspicion")


def test_staging_is_allocated_at_construction_not_on_first_lookup():
    """The ordering constraint, stated where it can go red.

    Graph replay reads the device addresses recorded at capture. A buffer
    allocated on the first lookup is a different address than the one the graph
    replays against, and that mismatch reports nothing -- it reads whatever is
    at the captured address. Construction is the only point guaranteed to
    precede capture, so this pins that the call is there.
    """
    import inspect

    from vllm_ascend.models.deepseek_v4_1 import engram_embedding as mod

    init = inspect.getsource(mod.AscendStagedEngramEmbedding.__init__)
    assert "_allocate_staging_at_construction" in init, (
        "staging must be sized during __init__; anywhere later is after "
        "graph capture")
    body = inspect.getsource(mod.AscendStagedEngramEmbedding
                             ._allocate_staging_at_construction)
    assert "max_num_batched_tokens" in body
    assert "max_model_len" not in body.split('"""')[2], (
        "the bound is tokens per forward, not tokens per sequence")


def test_an_oversized_batch_is_refused_by_name():
    """What makes taking the tight bound safe.

    max_num_batched_tokens and max_model_len differ by a factor of two here.
    Choosing the smaller one is defensible only because exceeding it raises
    with both numbers in the message rather than writing past the buffer.
    """
    from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout

    f = _Lookup(_tables())
    f._layout = EngramLayout(n_hash_cols=24, dim=256, quant_block=32,
                             layer_ids=(1, 14), num_embeddings=(512, 512))
    f._max_tokens = 16
    with pytest.raises(ValueError, match="exceeds the 16"):
        f.lookup(torch.zeros((32, f.part_n_hash_cols), dtype=torch.int64),
                 torch.zeros((32, 8)))
