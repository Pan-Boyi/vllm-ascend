# SPDX-License-Identifier: Apache-2.0
"""The engram staging-buffer convention, defined once.

The engram tables do not fit in device memory. The CPU gathers the rows a step
needs from a read-only mapping into a small staging buffer, one H2D copy puts
it in HBM, and the kernel reads it contiguously (no hash ids, no vocabulary
offset, no scale gather).

The CPU writes rows in an order the device recomputes from its program id. If
the two disagree the data is still well-formed, only paired with the wrong
token, so nothing raises. The convention therefore lives here and both sides
use it.

``n_hash_cols`` is ``(max_ngram_size - 1) * n_heads`` (24), not
``engram_n_heads`` (8).
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

        ``n_hash_cols`` is ``(max_ngram_size - 1) * n_heads``: one n-gram per
        size from 2 up to the maximum, each split over the heads.

        The quantisation block is not in the config; ``validate_against_files``
        checks it against the staged scale files.
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

        The single definition: the CPU writes here and the kernel recomputes
        the same value from its program id.
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
        """``row_index`` assumes every rank owns all hash columns (TP=1).

        Under sharding it would return valid row numbers for the wrong rows,
        so this refuses instead.
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
        """Check the config's shapes against the staged files' sizes.

        ``file_name(tensor_name)`` maps a table to its staged file name.
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


#: Threads for the gather. It is latency-bound on random reads; 16 was the
#: measured optimum, more threads got slower.
GATHER_THREADS = 16

#: Below this many rows the thread pool costs more than it saves.
GATHER_THREAD_THRESHOLD = 16384


def gather_into_staging(weight_table, scale_table, row_ids,
                        out_weight, out_scale,
                        threads: int = GATHER_THREADS) -> None:
    """Copy the rows a step needs into the staging buffers.

    ``row_ids[r]`` is the table row wanted at staging row ``r``, where ``r`` is
    ``layout.row_index(token, hash_col)``. No de-duplication: repeated ids are
    copied repeatedly, which keeps the row count a function of the token count
    (the capture route relies on that).
    """
    import numpy as np

    # Not chained: `a != b != c` would pass when both buffers agree.
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
