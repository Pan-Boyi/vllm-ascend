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
