# SPDX-License-Identifier: Apache-2.0
"""The engram table, never allocated and never registered.

Upstream's ``ParallelEngramEmbedding.__init__`` allocates the whole shard up
front, and on this platform both of its branches are impossible::

    kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}

``cpu_offload=True`` takes ``aclrtMallocHostWithCfg`` -- the pinned-host path,
which is banned, and which at TP=1 (no sharding, so no reduction per rank)
wants 188.83 GiB per rank -- 755.3 GiB across four ranks, against a 755 GiB
host that already has 188.8 GiB of it in /dev/shm. ``False`` puts the same
188.83 GiB in HBM. The first real launch died here, and it is worth being
precise
about how: it died of ``rtsMallocHost ... out of memory``. **The banned path was
stopped by the machine running out of memory, not by us.** On a larger host it
would have succeeded and then done the thing that took a card down.

So this does two separate jobs, and the second is not a consequence of the
first:

1. The tables are never allocated. The Parameters are created on the ``meta``
   device -- correct shape and dtype, zero bytes -- because the real data is
   already staged in ``/dev/shm`` and is read from there.
2. The banned path is made unreachable and loud. ``_storage()`` raises, and
   construction refuses if anything managed to allocate real storage. A policy
   that is only written down is enforced by whatever happens to fail first.
"""

from __future__ import annotations

import contextlib

import torch

from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout
from vllm_ascend.models.deepseek_v4_1.engram_staging import (DEFAULT_STAGE_DIR,
                                                             staged_file_name)


@contextlib.contextmanager
def tables_not_allocated():
    """Force ``torch.empty`` onto ``meta`` for the duration.

    Scoped to upstream's ``__init__``, whose only two ``torch.empty`` calls are
    the table and its scales. Narrow and ugly, but the alternative is copying
    upstream's constructor here so the two allocations can be dropped, and a
    copied constructor drifts silently while this fails loudly if the shape of
    that code changes.
    """
    real_empty = torch.empty

    def empty_on_meta(*sizes, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "meta"
        return real_empty(*sizes, **kwargs)

    torch.empty = empty_on_meta
    try:
        yield
    finally:
        torch.empty = real_empty


def _read_only_table(path: str, rows: int, width: int):
    """A read-only view of one staged table. Never writable: registration is
    what needed write access, and registration is gone."""
    import mmap
    import os

    import numpy as np

    size = os.path.getsize(path)
    if size != rows * width:
        raise ValueError(
            f"{path}: expected {rows:,} x {width} = {rows * width:,} bytes, "
            f"found {size:,}. The config's shapes and the file's size are the "
            "same fact by two routes.")
    handle = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(handle, size, flags=mmap.MAP_SHARED,
                            prot=mmap.PROT_READ)
    finally:
        os.close(handle)
    return np.frombuffer(mapping, dtype=np.uint8).reshape(rows, width), mapping


class AscendStagedEngramEmbedding:
    """Mixin over upstream's ``ParallelEngramEmbedding``.

    Mixed in ahead of it so these win the MRO::

        class AscendEngramEmbedding(AscendStagedEngramEmbedding,
                                    ParallelEngramEmbedding):
            pass
    """

    #: Set once construction has bound the staged tables.
    _staged_tables: tuple | None = None

    def __init__(self, *args, **kwargs) -> None:
        with tables_not_allocated():
            super().__init__(*args, **kwargs)

        self._assert_nothing_was_allocated()
        self._layer_id = None          # bound by the owner; see bind_layer()

    # -- guards --------------------------------------------------------------

    def _assert_nothing_was_allocated(self) -> None:
        """The interception, not the intention.

        Checked rather than trusted because the failure mode of the allocation
        succeeding is not an error -- it is 188.83 GiB of pinned host memory per
        rank and, on a host large enough to provide it, the registration that
        took a card down.
        """
        for name in ("weight", "weight_scale_inv"):
            param = getattr(self, name, None)
            if param is None:
                continue
            if param.device.type != "meta":
                raise RuntimeError(
                    f"engram {name} was allocated on {param.device} with "
                    f"{param.numel() * param.element_size() / 1024**3:.1f} GiB "
                    "of real storage. It must not be: the table is read from "
                    "the staged mapping, and the pinned-host path this would "
                    "otherwise take is the one that took an NPU down. Note "
                    "that running out of memory here is not the guard -- a "
                    "larger host would have let it through.")

    def _storage(self):
        raise RuntimeError(
            "engram _storage() is the UVA / host-registration entry point and "
            "must not be reached. The device does not address the engram "
            "tables on this platform: the CPU gathers the rows a step needs "
            "out of a read-only mapping and one H2D copy carries them. If this "
            "raised during a forward pass, something is still on upstream's "
            "direct-read path -- see engram_layout for the replacement.")

    # -- the staged tables ---------------------------------------------------

    def bind_staged_tables(self, layout: EngramLayout, layer_id: int,
                           stage_dir: str = DEFAULT_STAGE_DIR) -> None:
        """Map this layer's staged weight and scale tables, read-only."""
        rows = dict(zip(layout.layer_ids, layout.num_embeddings))[layer_id]
        weight, wmap = _read_only_table(
            f"{stage_dir}/{staged_file_name(f'layers.{layer_id}.engram.embed.weight')}",
            rows, layout.dim)
        scale, smap = _read_only_table(
            f"{stage_dir}/{staged_file_name(f'layers.{layer_id}.engram.embed.scale')}",
            rows, layout.n_scales)
        self._staged_tables = (weight, scale, wmap, smap)
        self._layer_id = layer_id

    def staged_tables(self):
        if self._staged_tables is None:
            raise RuntimeError(
                f"engram tables for layer {self._layer_id} are not bound; call "
                "bind_staged_tables() after construction. They are not "
                "allocated by the constructor on purpose -- see this module's "
                "docstring.")
        return self._staged_tables[0], self._staged_tables[1]
