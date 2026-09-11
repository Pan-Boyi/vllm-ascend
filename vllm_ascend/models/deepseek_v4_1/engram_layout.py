# SPDX-License-Identifier: Apache-2.0
"""The staging-buffer convention, defined once.

The engram tables are 188.83 GiB and cannot be reached from the device
directly -- see the deprecation note in ``engram_gather`` for what happened
when we tried. Instead the CPU gathers the rows a step needs out of a
read-only mapping, writes them into a small staging buffer, and one H2D copy
puts that buffer in HBM. The kernel then reads HBM, contiguously, with no
indirection at all: no hash ids, no vocabulary offset, no scale gather.

That trade removes one silent failure and introduces another, and this module
exists because of the second one. The CPU writes rows in an order the device
recomputes from its program id. If the two disagree, **both sides still handle
well-formed data** -- the bytes are valid, they are simply paired with the
wrong token -- so nothing raises and the only symptom is that accuracy is off.
The convention therefore lives in exactly one place, and both sides import it
rather than each restating it.

The shape of the rows is taken from the model config and then checked against
the staged files, which is the same fact arriving by two routes: the config
says 384,006,168 rows of 256, and the file is 98,305,579,008 bytes, which is
that product exactly. A mismatch means one of them is describing a different
checkpoint.

.. note:: What this change cost in evidence, as distinct from in code.

    "Scales must be read contiguously and broadcast, never gathered by
    ``cols // QUANT_BLOCK``" used to rest on a cross-check: the same bytes read
    through two different address spaces had to agree, and the defective form
    was wrong in both. On this path the kernel only ever reads HBM, so **that
    second arm no longer exists**. The conclusion is unchanged -- the indexed
    form is wrong on HBM too, which is one of the measured cells -- but it is
    now carried by fewer independent observations than it was.

    Converting the host arm into a test of the CPU gather does not restore it.
    That is a useful new test of a different property; it says nothing about
    the kernel. Recorded because a narrowed support and an unchanged conclusion
    read identically a week later, and nothing fails when evidence goes stale.

.. note:: ``n_hash_cols`` is 24 and has three independent sources, but the
    kernel has only been measured at 8.

    ``engram_n_heads`` is 8; ``n_hash_cols`` is 24. They are different
    quantities and the config names only the first, so taking ``n_heads``
    directly is the natural mistake. The value 24 is confirmed by
    ``(max_ngram_size - 1) * n_heads``, by the staged tables' row arithmetic,
    and by the shape of golden's dumped ``engram_hash_ids``. The kernel is
    parameterised over it, but its verification ran at 8 -- so 24, the value
    production uses, is covered by derivation and not by measurement.

"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EngramLayout:
    """How a step's gathered rows are laid out for the device.

    ``staging_weight`` is ``[num_tokens * n_hash_cols, dim]`` and
    ``staging_scale`` is ``[num_tokens * n_hash_cols, n_scales]``, both uint8
    holding the checkpoint's raw fp8 / ue8m0 bytes. Row ``r`` of one
    corresponds to row ``r`` of the other.
    """

    n_hash_cols: int
    dim: int
    quant_block: int
    layer_ids: tuple[int, ...]
    num_embeddings: tuple[int, ...]

    @property
    def n_scales(self) -> int:
        return self.dim // self.quant_block

    @property
    def row_bytes(self) -> int:
        """Weight plus scale, per gathered row."""
        return self.dim + self.n_scales

    @classmethod
    def from_config(cls, config) -> "EngramLayout":
        """Derive the layout from the model config.

        ``n_hash_cols`` is ``(max_ngram_size - 1) * n_heads``: a position is
        hashed as one n-gram per size from 2 up to the maximum, each split over
        the heads, and every (size, head) pair owns a column.

        The quantisation block is derived from the shapes rather than named in
        the config -- ``dim // n_scales`` -- so it is checked against the files
        below rather than assumed.
        """
        dim = int(config.engram_head_dim)
        return cls(
            n_hash_cols=(int(config.engram_max_ngram_size) - 1)
            * int(config.engram_n_heads),
            dim=dim,
            # 256 elements share 8 scale bytes in this checkpoint; asserted
            # against the staged files by validate_against_files.
            quant_block=32,
            layer_ids=tuple(config.engram_layer_ids),
            num_embeddings=tuple(config.engram_num_embeddings),
        )

    # -- the convention both sides use ---------------------------------------

    def row_index(self, token_index: int, hash_col: int) -> int:
        """Where a (token, hash column) pair lives in the staging buffer.

        The single definition. The CPU writes here; the kernel recomputes the
        same value from its program id. Anything that restates this arithmetic
        somewhere else is the bug this module exists to prevent.
        """
        return token_index * self.n_hash_cols + hash_col

    def staging_shapes(self, num_tokens: int) -> tuple[tuple[int, int],
                                                        tuple[int, int]]:
        rows = num_tokens * self.n_hash_cols
        return (rows, self.dim), (rows, self.n_scales)

    def staging_bytes(self, num_tokens: int) -> int:
        return num_tokens * self.n_hash_cols * self.row_bytes

    # -- the two guards ------------------------------------------------------

    def assert_head_sharding_is_degenerate(self, local_heads: int,
                                           total_heads: int,
                                           head_start: int) -> None:
        """``row_index`` is only that simple because TP=1 degenerates sharding.

        Upstream shards the hash columns across tensor-parallel ranks, and this
        port runs TP=1, where every rank owns all of them. The row numbering
        above silently means something else the moment that stops being true --
        and it would keep producing well-formed output.

        This is the several-th time a parameterised upstream mechanism has
        degenerated under TP=1. The difference here is that we are *relying* on
        the degenerate case rather than merely noticing it, which is why it is
        pinned rather than commented.
        """
        if not (local_heads == total_heads == self.n_hash_cols
                and head_start == 0):
            raise ValueError(
                f"engram staging assumes undivided hash columns, but this rank "
                f"has local_heads={local_heads}, total_heads={total_heads}, "
                f"head_start={head_start} against n_hash_cols="
                f"{self.n_hash_cols}. row_index() numbers rows as "
                "token * n_hash_cols + head, which is only the whole picture "
                "when the columns are not split. Under sharding it would keep "
                "returning valid row numbers for the wrong rows, so this "
                "refuses rather than continuing.")

    def validate_against_files(self, stage_dir: str,
                               file_name) -> None:
        """Check the config's shapes against the staged files' actual sizes.

        The same fact by two routes. ``file_name(tensor_name)`` maps a table to
        its staged path; passing it in keeps this module free of the naming
        convention, which belongs to the staging code.
        """
        problems = []
        for layer_id, rows in zip(self.layer_ids, self.num_embeddings):
            for suffix, width in (("weight", self.dim),
                                  ("scale", self.n_scales)):
                name = f"layers.{layer_id}.engram.embed.{suffix}"
                path = os.path.join(stage_dir, file_name(name))
                if not os.path.isfile(path):
                    problems.append(f"{name}: missing at {path}")
                    continue
                actual = os.path.getsize(path)
                expected = rows * width
                if actual != expected:
                    problems.append(
                        f"{name}: config implies {rows:,} x {width} = "
                        f"{expected:,} bytes, file is {actual:,}")
        if problems:
            raise ValueError(
                "engram layout does not match the staged tables:\n  "
                + "\n  ".join(problems)
                + "\n\nThe config's shapes and the files' sizes are the same "
                "fact by two routes; disagreeing means one of them describes a "
                "different checkpoint. Note that head_dim is 256 for engram "
                "and 512 for attention in this model -- reading the wrong one "
                "makes every gathered row straddle two real rows, with no "
                "error anywhere.")


#: Threads for the gather. Measured on this host (256 cores) against the real
#: 98 GiB table, one layer of an 8192-token chunk: 1 thread 14.75ms, 4 2.28ms,
#: 8 1.39ms, 16 1.27ms, 32 2.37ms. It is latency-bound on random reads into
#: resident tmpfs, so it scales well until it stops; past 16 it gets worse, so
#: this is a measured optimum rather than a core count.
GATHER_THREADS = 16

#: Below this many rows the pool costs more than it saves (a 256-token decode
#: step is 6144 rows and takes 0.22ms single-threaded, for both layers).
GATHER_THREAD_THRESHOLD = 16384


def gather_into_staging(weight_table, scale_table, row_ids,
                        out_weight, out_scale,
                        threads: int = GATHER_THREADS) -> None:
    """Copy the rows a step needs into the staging buffers.

    ``row_ids[r]`` is the table row wanted at staging row ``r``, and ``r`` is
    ``layout.row_index(token, hash_col)`` -- the caller builds ``row_ids`` in
    that order and the kernel recomputes ``r`` from its program id. Neither
    side may restate that arithmetic; both take it from ``EngramLayout``.

    ``weight_table`` and ``scale_table`` are read-only views of the staged
    tables. Read-only is the point: it is what the previous design could not
    have, and it is what makes a stray write impossible rather than guarded.

    No de-duplication. Hash ids repeat, so a row wanted by several tokens is
    copied several times; the measured costs assume that. De-duplicating would
    add an index indirection back, which is the kind of thing this design was
    reached by removing, so it needs its own measurement before being worth it.
    """
    import numpy as np

    # Written out rather than chained: `a != b != c` is `(a != b) and
    # (b != c)`, which is true only when the two buffers disagree with each
    # other -- exactly the case that cannot happen here, so the chained form
    # passed everything.
    if (row_ids.shape[0] != out_weight.shape[0]
            or row_ids.shape[0] != out_scale.shape[0]):
        raise ValueError(
            f"row_ids has {row_ids.shape[0]} entries but the staging buffers "
            f"hold {out_weight.shape[0]} and {out_scale.shape[0]} rows. These "
            "index each other positionally, so a mismatch would pair rows with "
            "the wrong tokens rather than fail.")

    n = row_ids.shape[0]
    if n < GATHER_THREAD_THRESHOLD or threads <= 1:
        np.take(weight_table, row_ids, axis=0, out=out_weight)
        np.take(scale_table, row_ids, axis=0, out=out_scale)
        return

    from concurrent.futures import ThreadPoolExecutor

    bounds = [(i * n // threads, (i + 1) * n // threads) for i in range(threads)]
    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(
            lambda lo_hi: (
                np.take(weight_table, row_ids[lo_hi[0]:lo_hi[1]], axis=0,
                        out=out_weight[lo_hi[0]:lo_hi[1]]),
                np.take(scale_table, row_ids[lo_hi[0]:lo_hi[1]], axis=0,
                        out=out_scale[lo_hi[0]:lo_hi[1]]),
            ),
            bounds))
