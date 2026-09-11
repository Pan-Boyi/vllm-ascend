# SPDX-License-Identifier: Apache-2.0
"""The staged engram tables must be exactly the length the checkpoint says.

The failure this exists for actually happened: the staging step reported a
byte-for-byte match against the source while every one of the four files was
7,184 bytes short, each truncated to a page boundary. The report was true --
it compared the part that had been written. A check whose extent comes from
the thing being checked cannot see a short write, which is why this one takes
its expected length from the checkpoint's safetensors header instead.
"""

import json
import os
import struct

import pytest

from vllm_ascend.models.deepseek_v4_1 import engram_staging as staging

TENSOR = "model.layers.1.engram.embed.weight"
OTHER = "model.layers.1.engram.embed.scale"
NBYTES, OTHER_NBYTES = 4096, 1024

#: The real thing, if this machine has it staged.
REAL_CHECKPOINT = "/workspace/models/DeepSeek-V4.1-Flash"
REAL_STAGE_DIR = staging.DEFAULT_STAGE_DIR


def _write_shard(directory, entries):
    """A minimal safetensors shard: 8-byte header length, JSON, then data."""
    header, offset = {}, 0
    for name, nbytes in entries.items():
        header[name] = {"dtype": "U8", "shape": [nbytes],
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    blob = json.dumps(header).encode()
    path = os.path.join(directory, "model-00001-of-00001.safetensors")
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\0" * offset)
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as handle:
        json.dump({"weight_map": {n: os.path.basename(path) for n in entries}},
                  handle)


@pytest.fixture
def checkpoint(tmp_path):
    d = tmp_path / "ckpt"
    d.mkdir()
    _write_shard(str(d), {TENSOR: NBYTES, OTHER: OTHER_NBYTES})
    return str(d)


@pytest.fixture
def stage(tmp_path):
    d = tmp_path / "stage"
    d.mkdir()
    return str(d)


def _stage_file(stage_dir, name, nbytes):
    with open(os.path.join(stage_dir, staging.staged_file_name(name)), "wb") as h:
        h.write(b"\0" * nbytes)


def test_expected_size_comes_from_the_data_offsets(checkpoint):
    assert staging.expected_nbytes(checkpoint, TENSOR) == NBYTES
    assert staging.expected_nbytes(checkpoint, OTHER) == OTHER_NBYTES


def test_correct_tables_verify(checkpoint, stage):
    _stage_file(stage, TENSOR, NBYTES)
    _stage_file(stage, OTHER, OTHER_NBYTES)
    assert staging.verify_staged_tables([TENSOR, OTHER], checkpoint, stage) == {
        TENSOR: NBYTES, OTHER: OTHER_NBYTES}


def test_the_page_boundary_truncation_is_caught(checkpoint, stage):
    """The historical failure, to the byte.

    7,184 short of 4,096 is not representable, so this uses the shape of the
    bug rather than its exact number: truncated to a page boundary, a size that
    looks entirely plausible next to the expected one.
    """
    _stage_file(stage, TENSOR, NBYTES - 4096 % NBYTES - 3072)  # 1024 bytes short
    _stage_file(stage, OTHER, OTHER_NBYTES)
    with pytest.raises(RuntimeError) as exc:
        staging.verify_staged_tables([TENSOR, OTHER], checkpoint, stage)
    msg = str(exc.value)
    assert "4,096" in msg and "1,024" in msg      # both integers printed
    assert "+3,072" in msg                        # and the difference
    assert "1 of 2 tables verified" in msg        # and the coverage


def test_a_single_missing_byte_is_refused(checkpoint, stage):
    """No tolerance. A table short by one byte reads one zero past the end,
    and zero is valid engram payload."""
    _stage_file(stage, TENSOR, NBYTES - 1)
    _stage_file(stage, OTHER, OTHER_NBYTES)
    with pytest.raises(RuntimeError, match=r"\+1 bytes"):
        staging.verify_staged_tables([TENSOR, OTHER], checkpoint, stage)


def test_an_oversized_table_is_refused_too(checkpoint, stage):
    """Not only short writes: a longer file means the layout is not what the
    row arithmetic assumes."""
    _stage_file(stage, TENSOR, NBYTES + 8)
    _stage_file(stage, OTHER, OTHER_NBYTES)
    with pytest.raises(RuntimeError, match=r"-8 bytes"):
        staging.verify_staged_tables([TENSOR, OTHER], checkpoint, stage)


def test_a_missing_file_is_not_a_special_case(checkpoint, stage):
    _stage_file(stage, OTHER, OTHER_NBYTES)
    with pytest.raises(RuntimeError, match="file is missing"):
        staging.verify_staged_tables([TENSOR, OTHER], checkpoint, stage)


def test_an_empty_directory_says_why(checkpoint, stage):
    """What a reboot looks like: tmpfs is in memory, so /dev/shm comes back
    empty rather than damaged."""
    with pytest.raises(RuntimeError) as exc:
        staging.verify_staged_tables([TENSOR], checkpoint, stage)
    assert "file is missing" in str(exc.value)
    assert "reboot" in str(exc.value)


def test_a_missing_staging_directory_says_reboot(checkpoint, tmp_path):
    with pytest.raises(RuntimeError, match="reboot"):
        staging.verify_staged_tables([TENSOR], checkpoint,
                                     str(tmp_path / "never_created"))


def test_checking_nothing_is_refused(checkpoint, stage):
    """Zero coverage and full coverage both return without complaint."""
    with pytest.raises(ValueError, match="no tensor names"):
        staging.verify_staged_tables([], checkpoint, stage)


def test_a_tensor_absent_from_the_checkpoint_is_refused(checkpoint, stage):
    with pytest.raises(KeyError, match="not in"):
        staging.expected_nbytes(checkpoint, "model.layers.99.engram.embed.weight")


# -- against the real staged tables, when this machine has them --------------

#: Note the absence of a "model." prefix -- the engram tables are named from
#: the checkpoint root, unlike most tensors in this checkpoint. Writing the
#: prefixed form here is what the first version of this test did, and the
#: checkpoint-derived lookup refused it rather than quietly finding nothing.
REAL_TENSORS = [
    "layers.1.engram.embed.weight",
    "layers.1.engram.embed.scale",
    "layers.14.engram.embed.weight",
    "layers.14.engram.embed.scale",
]


@pytest.mark.skipif(
    not os.path.isdir(REAL_STAGE_DIR) or not os.path.isdir(REAL_CHECKPOINT),
    reason="engram tables are not staged on this machine")
def test_the_real_staged_tables_match_the_checkpoint():
    """A second source for the staging report, derived independently.

    Same evidence (the safetensors headers), different code path and different
    author from the staging step's own comparison -- which is the part that was
    wrong last time. Reads sizes only; it never opens the 188 GiB of payload.
    """
    verified = staging.verify_staged_tables(REAL_TENSORS, REAL_CHECKPOINT)
    assert set(verified) == set(REAL_TENSORS)
    assert sum(verified.values()) > 0
