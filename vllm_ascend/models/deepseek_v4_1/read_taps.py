# SPDX-License-Identifier: Apache-2.0
"""What a layer *read*, as opposed to what it computed.

Two requests with byte-identical ``sa_in.q``, ``sa_in.kv``,
``sa_in.attn_sink`` and ``sa_in.softmax_scale`` produced byte-different
``attn.o_raw`` at layer 2. Same inputs, different output, so the difference
entered through an input that was **not in the dump** -- and the list of
those is the dump's own ``not_isolated``: the paged KV cache, the compressed
latents, the candidate blocks. Plus, at layer 2, the indexer's selection,
which had never been recorded at all and is therefore NOT MEASURED rather
than measured-and-equal.

🔴 These are **caches**, so the moment of recording decides what is recorded:

    before the operator call   what this layer read        <- what we want
    after the forward          what the cache holds now    <- a different
                                                              question

Everything here is taken immediately before the call, and the manifest says
so. A tensor that does not declare which of the two it is can be read as
either, and the two support opposite conclusions.

⚠️ Cache reads are gathered **by this request's block table**, not copied
whole. The full cache is gigabytes and shared by every request; what matters
is the handful of blocks this call actually addressed, which is also the only
part two requests can be meaningfully compared on.
"""

from __future__ import annotations

import torch

_ENV = "VLLM_ASCEND_DSV41_TAP_READ_LAYERS"

#: Blocks gathered per cache. A cap rather than a slice of unknown size: the
#: dump is read tensor by tensor and a single multi-gigabyte entry would make
#: the comparison unusable. Overflow is reported, never silently dropped.
MAX_BLOCKS = 8

#: Declared in the manifest so the dump says what its "read.*" keys are.
MOMENT = ("immediately before the operator call: what this layer read, not "
          "what the cache held after the forward")


def read_tap_layers() -> tuple[int, ...]:
    """Which layers record their reads.

    Defaults to layer 2 -- the first layer with a compressed stream and the
    first that diverged. Recording all forty would multiply the dump by the
    cache blocks of every layer, and the comparison that is being run needs
    one layer.
    """
    from vllm_ascend import envs

    spec = str(getattr(envs, _ENV, "") or "").strip()
    if not spec:
        return ()
    if spec.lower() == "all":
        return tuple(range(64))
    out = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(int(piece))
        except ValueError:
            raise ValueError(
                f"{_ENV}={spec!r}: {piece!r} is not a layer index or 'all'. "
                "Refusing rather than skipping it -- a spec that is partly "
                "ignored produces a dump that looks complete.") from None
    return tuple(dict.fromkeys(out))


def records_reads(layer_id: int) -> bool:
    return layer_id in read_tap_layers()


def tap_read(layer_id: int, key: str, tensor) -> None:
    """Record one thing this layer read, under a ``read.`` key.

    Not position-sharded: a cache block is addressed by slot, not by token
    position, so scattering it into a ``[seq_len, ...]`` buffer would be
    meaningless and the coverage check could never be satisfied.
    """
    if not isinstance(tensor, torch.Tensor) or not records_reads(layer_id):
        return
    from vllm_ascend.models.deepseek_v4_1.align_taps import tap_whole

    _declare_moment()
    tap_whole(layer_id, f"read.{key}", tensor)


def tap_cache_read(layer_id: int, key: str, cache, block_table) -> None:
    """Record the cache blocks this request's block table points at.

    ⚠️ The blocks, not the cache. Two requests share one cache and differ in
    which blocks they were given; copying the whole thing would compare
    mostly other requests' data and would be gigabytes of it.
    """
    if not records_reads(layer_id):
        return
    if not isinstance(cache, torch.Tensor) or \
            not isinstance(block_table, torch.Tensor):
        return
    ids = _block_ids(block_table)
    if ids is None:
        return
    kept = ids[:MAX_BLOCKS]
    tap_read(layer_id, key, cache[kept])
    # The ids themselves, because "which blocks" is half the question: two
    # requests reading identical bytes out of different blocks and two
    # reading different bytes out of the same block are different findings.
    tap_read(layer_id, f"{key}.block_ids", kept)
    if ids.numel() > MAX_BLOCKS:
        _report_truncated(layer_id, key, int(ids.numel()))


def _block_ids(block_table: torch.Tensor):
    """Valid block ids in this table, de-duplicated, in order."""
    flat = block_table.reshape(-1)
    valid = flat[flat >= 0]
    if valid.numel() == 0:
        return None
    seen = torch.zeros_like(valid, dtype=torch.bool)
    seen[0] = True
    if valid.numel() > 1:
        seen[1:] = valid[1:] != valid[:-1]
    return valid[seen]


_DECLARED = False


def _declare_moment() -> None:
    global _DECLARED
    if _DECLARED:
        return
    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    recorder = get_recorder()
    if recorder is None:
        return
    _DECLARED = True
    recorder.add_manifest(read_tap_moment=MOMENT,
                          read_tap_layers=list(read_tap_layers()),
                          read_tap_max_blocks=MAX_BLOCKS)


def reset_for_test() -> None:
    global _DECLARED
    _DECLARED = False


def _report_truncated(layer_id: int, key: str, total: int) -> None:
    from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

    announce_once(
        f"read_tap_truncated_{key}",
        "layer %d read %d blocks for %r and only the first %d are in the "
        "dump. A difference in the blocks beyond that will not appear, so "
        "treat a match on this key as partial.", layer_id, total, key,
        MAX_BLOCKS)
