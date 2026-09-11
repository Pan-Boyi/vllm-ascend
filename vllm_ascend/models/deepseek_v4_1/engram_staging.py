# SPDX-License-Identifier: Apache-2.0
"""Preconditions on the staged engram tables.

The engram lookup tables are 188.83 GiB and do not fit in device memory, so
they are staged as raw files in ``/dev/shm`` and mapped. Everything downstream
-- registration, the device address, the Triton gather -- assumes those files
are complete. Nothing downstream can tell that they are not: a short table
reads as whatever the mapping's tail holds, which is zeros, and zeros are
valid engram payload.

This is the second, independent check, and the reason it exists is worth
recording. The staging side already compares what it wrote against the source
and reports a byte-for-byte match. That report was true and useless at the same
time: it compared *the part that had been written*, and the writer had stopped
7,184 bytes early on all four files -- each truncated to a page boundary,
invisible at the GiB granularity the report used. A check whose extent is
derived from the thing it is checking cannot see a short write.

So this one derives the expected size from the *checkpoint* instead: the
safetensors header's ``data_offsets``, which is the authority on how long the
tensor is. Integer equality, no tolerance. Anything else -- a missing
directory, a missing file, a zero-length file -- takes the same refusal path
rather than a special case; a reboot empties ``/dev/shm`` (it is tmpfs, in
memory), and that lands here as "expected N, found 0", which is exactly right.
"""

from __future__ import annotations

import json
import os
import struct

#: Where the staging step writes the raw tables. One file per tensor, named
#: for the tensor with "/" replaced -- "." is kept, so the name still reads as
#: the checkpoint's.
DEFAULT_STAGE_DIR = "/dev/shm/engram_stage"

#: safetensors: 8-byte little-endian header length, then that many bytes of
#: JSON, then the data section that ``data_offsets`` indexes into.
_HEADER_LEN_BYTES = 8


def staged_file_name(tensor_name: str) -> str:
    return tensor_name.replace("/", "_") + ".bin"


def expected_nbytes(checkpoint_dir: str, tensor_name: str) -> int:
    """Length of one tensor according to the checkpoint's own header.

    ``data_offsets`` is the authority here, not the file size of the shard and
    not anything the staging step recorded -- the point of this module is to
    compare against something the staging step did not produce.
    """
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path, "r") as handle:
        weight_map = json.load(handle)["weight_map"]
    if tensor_name not in weight_map:
        raise KeyError(
            f"{tensor_name!r} is not in {index_path}. The staged table does "
            "not correspond to a tensor in this checkpoint.")

    shard_path = os.path.join(checkpoint_dir, weight_map[tensor_name])
    with open(shard_path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(_HEADER_LEN_BYTES))[0]
        header = json.loads(handle.read(header_len))
    start, end = header[tensor_name]["data_offsets"]
    return end - start


def verify_staged_tables(tensor_names: list[str], checkpoint_dir: str,
                         stage_dir: str = DEFAULT_STAGE_DIR) -> dict[str, int]:
    """Refuse to start unless every staged table is exactly the right length.

    Returns the verified sizes, so a caller that wants to report coverage has
    the numbers rather than a bare True.
    """
    if not tensor_names:
        raise ValueError(
            "verify_staged_tables() was given no tensor names. Checking "
            "nothing and checking everything both return without complaint, "
            "so this refuses instead.")

    if not os.path.isdir(stage_dir):
        raise RuntimeError(
            f"engram staging directory {stage_dir!r} does not exist. It is "
            "tmpfs, so a reboot empties it and the tables have to be staged "
            "again; this is not a sign that they are damaged.")

    verified: dict[str, int] = {}
    problems: list[str] = []
    for name in tensor_names:
        expected = expected_nbytes(checkpoint_dir, name)
        path = os.path.join(stage_dir, staged_file_name(name))
        if not os.path.isfile(path):
            problems.append(f"{name}: expected {expected:,} bytes, file is missing ({path})")
            continue
        actual = os.path.getsize(path)
        if actual != expected:
            short_by = expected - actual
            problems.append(
                f"{name}: expected {expected:,} bytes, found {actual:,} "
                f"({short_by:+,} bytes)")
            continue
        verified[name] = actual

    if problems:
        raise RuntimeError(
            "staged engram tables do not match the checkpoint:\n  "
            + "\n  ".join(problems)
            + f"\n\n{len(verified)} of {len(tensor_names)} tables verified. "
            "Sizes come from the safetensors data_offsets, so this is not a "
            "tolerance that can be loosened -- a table that is short by any "
            "number of bytes reads zeros past the end, and zeros are valid "
            "engram payload. If the staging directory is empty or the files "
            "are missing, /dev/shm was most likely cleared by a reboot and "
            "the tables need staging again.")

    return verified


# -- content, not just length -------------------------------------------------
#
# Registration for device access requires the mapping to be writable: the
# driver pins pages with write intent, and a PROT_READ mapping is refused
# outright (aclrtHostRegisterV2 rc=507899, with the kernel reporting
# get_user_pages_fast -EFAULT). So these tables -- read-only data, shared as
# one set of physical pages across every rank -- are necessarily mapped
# writable in production. A stray write through any of those mappings corrupts
# the table for all ranks at once, silently, and the size check above cannot
# see it: a write changes content, not length.
#
# Hence a fingerprint. Two independent folds per file, so a change that happens
# to preserve one is still caught by the other. Reading is memory-bandwidth
# bound -- the whole 188.83 GiB folds in about 11 seconds -- which is cheap
# enough to do at startup.

_FOLD_CHUNK = 1 << 30
_U64_MASK = (1 << 64) - 1


def content_fingerprint(path: str) -> tuple[int, int]:
    """``(sum, xor)`` over the file's 64-bit words, both mod 2**64.

    Mapped read-only on purpose: the check must never be the thing that
    corrupts the table it is checking.
    """
    import mmap

    import numpy as np

    size = os.path.getsize(path)
    handle = os.open(path, os.O_RDONLY)
    total, xor, done = 0, 0, 0
    try:
        while done < size:
            length = min(_FOLD_CHUNK, size - done)
            length -= length % 8          # whole words only
            if not length:
                break
            mapping = mmap.mmap(handle, length, flags=mmap.MAP_SHARED,
                                prot=mmap.PROT_READ, offset=done)
            words = np.frombuffer(mapping, dtype=np.uint64)
            total = (total + int(words.sum(dtype=np.uint64))) & _U64_MASK
            xor ^= int(np.bitwise_xor.reduce(words))
            del words                     # numpy holds a view; close() refuses otherwise
            mapping.close()
            done += length
    finally:
        os.close(handle)
    return total, xor


def fingerprint_staged_tables(tensor_names: list[str],
                              stage_dir: str = DEFAULT_STAGE_DIR
                              ) -> dict[str, tuple[int, int]]:
    """Fingerprint every staged table, for recording after staging."""
    return {
        name: content_fingerprint(os.path.join(stage_dir, staged_file_name(name)))
        for name in tensor_names
    }


def verify_staged_content(expected: dict[str, tuple[int, int]],
                          stage_dir: str = DEFAULT_STAGE_DIR) -> None:
    """Refuse to start if any table's content has changed since it was staged.

    ``expected`` is what ``fingerprint_staged_tables`` returned when the tables
    were written. Sizes are checked separately by ``verify_staged_tables``;
    this is the arm that catches a write, which leaves the size untouched.
    """
    if not expected:
        raise ValueError(
            "verify_staged_content() was given no fingerprints. Checking "
            "nothing returns just as quietly as checking everything.")

    problems = []
    for name, (want_sum, want_xor) in expected.items():
        path = os.path.join(stage_dir, staged_file_name(name))
        if not os.path.isfile(path):
            problems.append(f"{name}: file is missing ({path})")
            continue
        got_sum, got_xor = content_fingerprint(path)
        if (got_sum, got_xor) != (want_sum, want_xor):
            problems.append(
                f"{name}: fingerprint changed\n      expected sum={want_sum:016x} "
                f"xor={want_xor:016x}\n      found    sum={got_sum:016x} "
                f"xor={got_xor:016x}")

    if problems:
        raise RuntimeError(
            "staged engram tables have changed since they were written:\n  "
            + "\n  ".join(problems)
            + "\n\nThe size check passing alongside this means the length is "
            "unchanged and the bytes are not -- which is what a stray write "
            "through the writable mapping looks like. Registration requires "
            "PROT_WRITE, so every rank holds one, and they all address the "
            "same physical pages. Re-stage rather than continuing.")
