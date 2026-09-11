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


#: Upstream logs this from inside ``if cpu_offload:`` regardless of what the
#: allocation actually did, so with the allocation diverted it describes an
#: intention rather than an event.
_STALE_OFFLOAD_LOG = "offloaded to pinned host memory"


class _DropStaleOffloadLine:
    """Filter out upstream's now-untrue INFO line.

    It reports "94.42 GiB per rank", which over four ranks and two layers is
    755.4 GiB against a 755 GiB host -- a reader would conclude host memory had
    just been exhausted, when the measured figure is VmPin = 0. Left in place it
    is worse than no log line, because the number is specific enough to be
    believed.
    """

    def filter(self, record) -> bool:
        return _STALE_OFFLOAD_LOG not in record.getMessage()


@contextlib.contextmanager
def _upstream_offload_line_suppressed():
    import logging

    logger = logging.getLogger(
        "vllm.models.deepseek_v4_1.common.engram")
    drop = _DropStaleOffloadLine()
    logger.addFilter(drop)
    try:
        yield
    finally:
        logger.removeFilter(drop)


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
        with tables_not_allocated(), _upstream_offload_line_suppressed():
            super().__init__(*args, **kwargs)

        self._assert_nothing_was_allocated()
        self._layer_id = None          # bound by the owner; see bind_layer()

        import logging
        logging.getLogger(__name__).info(
            "Engram table not allocated: %d rows x %d stays in the staged "
            "read-only mapping, 0 bytes resident and 0 bytes pinned per rank. "
            "Upstream's \"%s\" line is suppressed here because the allocation "
            "it describes does not happen.",
            getattr(self, "part_num_embeddings", -1),
            getattr(self, "dim", -1), _STALE_OFFLOAD_LOG)

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

    # -- the lookup path -----------------------------------------------------

    def allocate_staging(self, layout: EngramLayout, max_tokens: int,
                         device) -> None:
        """Pre-allocate the per-step buffers, once, at the maximum batch.

        Allocated once and reused so the device addresses stay put: ACL graph
        replay requires the addresses recorded at capture, and a fresh buffer
        per step desynchronises it.

        The host side is **not** pinned. Pinning would make the H2D faster and
        ``aclrtMallocHost`` at this size (about 104 MB at 8192 tokens) is not
        the call that took a card down -- that was ``aclrtHostRegisterV2``. But
        the two are close enough in this area that turning it on should be a
        decision someone makes deliberately rather than something that arrives
        inside an optimisation, so it is left off and noted here.
        """
        import numpy as np

        (wshape, sshape) = layout.staging_shapes(max_tokens)
        self._layout = layout
        self._max_tokens = max_tokens
        self._host_weight = np.empty(wshape, dtype=np.uint8)
        self._host_scale = np.empty(sshape, dtype=np.uint8)
        self._dev_weight = torch.empty(wshape, dtype=torch.uint8, device=device)
        self._dev_scale = torch.empty(sshape, dtype=torch.uint8, device=device)

    def lookup(self, indices: torch.Tensor, out: torch.Tensor,
               background: bool = False) -> None:
        """Gather on the host, copy once, let the kernel read HBM.

        ``indices`` is ``[num_tokens, n_hash_cols]`` of table rows. Flattened
        row-major it is already in staging order: upstream numbers its rows
        ``token * LOCAL_HEADS + head`` and, with the hash columns undivided,
        that is exactly ``EngramLayout.row_index``. The two agreeing is checked
        rather than assumed -- see ``assert_head_sharding_is_degenerate``.
        """
        import numpy as np

        from vllm_ascend.models.deepseek_v4_1.engram_layout import \
            gather_into_staging

        num_tokens = indices.shape[0]
        rows = num_tokens * self.part_n_hash_cols
        if not rows:
            return

        layout = getattr(self, "_layout", None)
        if layout is None:
            raise RuntimeError(
                "engram staging buffers are not allocated; call "
                "allocate_staging() after construction. The tables are not "
                "device-addressable on this platform, so there is no path that "
                "reads them without this.")
        layout.assert_head_sharding_is_degenerate(
            self.part_n_hash_cols, self.n_hash_cols, self.head_start)
        if num_tokens > self._max_tokens:
            raise ValueError(
                f"{num_tokens} tokens exceeds the {self._max_tokens} the "
                "staging buffers were sized for. They are allocated once so "
                "their device addresses stay stable for graph replay, so this "
                "cannot grow on demand.")

        weight_table, scale_table = self.staged_tables()

        # The one synchronising transfer on this path: the hash ids have to
        # reach the host to be gathered. Small (24 int32 per token) but it is a
        # sync point, and the way to remove it is to hash on the host from the
        # token ids the runner already holds, rather than to make it faster.
        row_ids = indices.reshape(-1).to("cpu", torch.int64).numpy()

        gather_into_staging(weight_table, scale_table, row_ids,
                            self._host_weight[:rows], self._host_scale[:rows])

        self._dev_weight[:rows].copy_(
            torch.from_numpy(self._host_weight[:rows]), non_blocking=True)
        self._dev_scale[:rows].copy_(
            torch.from_numpy(self._host_scale[:rows]), non_blocking=True)

        self._dequantize_staged(rows, out)

    def _dequantize_staged(self, rows: int, out: torch.Tensor) -> None:
        """Read the staged rows out of HBM and write the embedding.

        The seam the device-side kernel binds to. It reads
        ``self._dev_weight[:rows]`` and ``self._dev_scale[:rows]``, which are
        contiguous, so it needs no hash ids, no vocabulary offset and -- this
        is the load-bearing one -- no ``cols // QUANT_BLOCK`` scale gather: the
        scales are ``n_scales`` contiguous bytes per row, read with
        ``tl.arange`` and broadcast.
        """
        if envs_dsv41_engram_dequant_torch():
            dequantize_staged_torch(
                self._dev_weight[:rows], self._dev_scale[:rows], out,
                quant_block=self._layout.quant_block)
            return

        launch = _staged_dequant_launcher()
        launch(self._dev_weight[:rows], self._dev_scale[:rows], out,
               dim=self._layout.dim, quant_block=self._layout.quant_block)


#: Where the kernel lives. The package path is where it belongs; the
#: build_artifacts path is where it is today, under a directory named for the
#: UVA design that this replaced.
_KERNEL_LOCATIONS = (
    ("vllm_ascend.ops.triton.engram_dequant_staging", None),
    ("engram_dequant_staging",
     "/workspace/code/dsv4_single_card/vllm_project/build_artifacts/npu_view/"
     "engram_dequant_staging.py"),
)


def _staged_dequant_launcher():
    """Resolve the kernel, preferring a real package location.

    Verification status, stated here because binding is not endorsing:

    - The arithmetic is settled. ``test_engram_dequant_reference`` checks it on
      the CPU against golden's authoritative rows, byte for byte, including the
      negative zero a value comparison would report as equal.
    - The kernel has run on hardware. Triton compiles at first launch, and
      ``/root/.triton/cache`` holds two compilations of ``engram_dequant_staging``
      (2026-09-11 10:48:43 and 11:35:35).
    - Its file header still says "nothing here has run". That is stale -- it
      was written before the run -- and it is worth knowing that reading it is
      what produced the opposite conclusion here first. The compile cache is
      the artefact that settles it; a comment is not.

    Still open: performance at production scale, and getting that run's output
    written down somewhere other than a chat log.
    """
    import importlib

    errors = []
    for module_name, path in _KERNEL_LOCATIONS:
        try:
            if path is None:
                module = importlib.import_module(module_name)
            else:
                import importlib.util
                import os
                if not os.path.isfile(path):
                    errors.append(f"{path}: not present")
                    continue
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            return module.launch
        except Exception as exc:                      # noqa: BLE001
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "the staged-dequantise kernel could not be loaded from any known "
        "location:\n  " + "\n  ".join(errors)
        + "\n\nEverything ahead of it is wired: this step's rows are already "
        "in HBM, laid out as EngramLayout describes. Only the kernel is "
        "missing.")


# -- a torch route through the same arithmetic -------------------------------
#
# Exists to answer one question: whether a triton-ascend compiler assertion is
# coming from this kernel. That assertion is an MLIR abort(), not an exception
# -- the worker vanishes with exit code None, no traceback, nothing in dmesg --
# so it cannot be caught, narrowed by a stack, or attributed by reading a log.
# The only cheap way to attribute it is to take one Triton kernel out of the
# picture and see whether the process still dies.
#
# This is not a fallback anybody should run in production: it materialises the
# dequantised rows through several ordinary torch ops where the kernel does one
# pass. It is a bisection instrument.


def envs_dsv41_engram_dequant_torch() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_ENGRAM_DEQUANT_TORCH == "1"


def dequantize_staged_torch(staging_weight: torch.Tensor,
                            staging_scale: torch.Tensor,
                            out: torch.Tensor,
                            quant_block: int) -> None:
    """The staging kernel's arithmetic, in torch.

    Same contract: row r of the weight and scale buffers is row r of the
    output. The scales are read contiguously and broadcast over their block --
    never indexed by ``cols // quant_block``, which is the form that reads the
    same values from HBM in a 2-D block and the wrong ones everywhere else.
    """
    rows, dim = staging_weight.shape
    n_scales = dim // quant_block
    if staging_scale.shape != (rows, n_scales):
        raise ValueError(
            f"scale buffer is {tuple(staging_scale.shape)}, expected "
            f"({rows}, {n_scales}) for dim={dim} and quant_block={quant_block}")

    values = staging_weight.view(torch.float8_e4m3fn).to(torch.float32)
    # ue8m0 is a power of two, so the byte *is* the fp32 exponent field.
    scales = (staging_scale.to(torch.int32) << 23).view(torch.float32)

    scaled = values.view(rows, n_scales, quant_block) * scales.unsqueeze(-1)
    out.copy_(scaled.view(rows, dim).to(out.dtype))
