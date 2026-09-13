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


def _never_breaks_the_run(fn):
    """An instrument that can kill the run is not an instrument.

    🔴 ``tap_cache_read`` raised ``aclnnIndex`` 561103 and the flight came
    back 0/4 with an empty dump -- the cost of a server start and a warm-up,
    spent on the observer rather than on the thing observed. Everything in
    this module is read-only and optional by construction, so nothing here
    has any business propagating.

    Disables itself after the first failure and says so once. Continuing to
    throw every call would fill the log with the same line and keep paying
    the cost; going quiet without a word would leave a dump that is missing
    keys for no stated reason, which reads as "this layer had nothing".
    """
    import functools

    @functools.wraps(fn)
    def _guarded(*args, **kwargs):
        if _DISABLED.get(fn.__name__):
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:                             # noqa: BLE001
            _DISABLED[fn.__name__] = True
            from vllm_ascend.models.deepseek_v4_1.switch_log import \
                announce_once

            announce_once(
                "read_taps_disabled_" + fn.__name__,
                "%s failed (%r) and has disabled itself for the rest of "
                "this process. The run continues and the other read taps are "
                "unaffected; the keys this one writes are absent from the "
                "dump, and that is why -- not because the layer had nothing "
                "to read.", fn.__name__, exc)
            return None

    return _guarded


#: Disabled per entry point, not for the module.
#:
#: 🔴 This was one shared flag, and that was wrong in the direction that
#: cost a flight. ``tap_cache_read`` failed -- it gathers from a cache the
#: device may hold in an internal format -- and took ``tap_read`` down with
#: it. But ``tap_read`` only records small tensors that were already passed
#: as arguments: the selections, the lengths, the indexer's topk_idxs. Those
#: are exactly the keys the investigation needs, they never touch the
#: gather, and they were lost to a failure in a part they do not depend on.
#:
#: ⚠️ The earlier reasoning -- "a failure in one is a reason to stop trusting
#: the other" -- is right about *trust* and wrong about *coupling*. The two
#: read different things through different operations; only the shared
#: conclusion ("the cache contents are unavailable") is shared.
_DISABLED: dict = {}


@_never_breaks_the_run
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


@_never_breaks_the_run
def tap_cache_read(layer_id: int, key: str, cache, block_table,
                   seqused=None, block_size: "int | None" = None) -> None:
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
    ids = _block_ids(block_table, seqused, block_size)
    if ids is None:
        return
    kept = ids[:MAX_BLOCKS]
    # The ids themselves first, and through their own tap, because "which
    # blocks" is half the question -- two requests reading identical bytes
    # out of different blocks and two reading different bytes out of the same
    # block are different findings -- and because this part cannot fail the
    # way the contents can.
    tap_read(layer_id, f"{key}.block_ids", kept)
    _tap_block_contents(layer_id, key, cache, kept)
    if ids.numel() > MAX_BLOCKS:
        _report_truncated(layer_id, key, int(ids.numel()))


@_never_breaks_the_run
def _tap_block_contents(layer_id: int, key: str, cache, kept) -> None:
    """A digest of each block, taken one block at a time.

    🔴 Two gather operators have now failed on this tensor with the same
    error: ``aclnnIndex`` (from ``cache[ids]``) and ``aclnnIndexSelect``
    (from ``index_select``). The same code from two different kernels points
    away from "which gather" and at the tensor -- a KV cache the device may
    be holding in an internal format, which neither gather accepts.

    So this takes ``cache[int(i)]``, which is **basic** indexing: a view, not
    a kernel, and the most primitive access there is. And it records a
    **digest** rather than the bytes, because the question is "did the two
    requests read the same thing", and a digest answers it without moving
    megabytes per layer per request off the device.

    ⚠️ The digest is the trace's, so a difference here and a difference in
    the layer trace mean the same thing and can be compared directly.
    """
    from vllm_ascend.models.deepseek_v4_1.layer_trace import (
        coarse_fingerprint_tensor, fingerprint_tensor)

    _describe_once("cmp/ori cache", cache, kept)
    exact, coarse = [], []
    for index in kept.reshape(-1).tolist():
        # 🔴 The view is free; consuming it is not. The failure that killed
        # two flights was not the index operator -- the traceback's third
        # line names ``AsStridedAiCore`` and its last line says "Convert
        # tensor to contiguous failed", so what failed is the contiguity
        # conversion op-plugin inserts for a non-contiguous input. A basic
        # index returns a view, and the next op that needs contiguous memory
        # inserts the same AsStrided. So the landing is attempted here,
        # where it can be reported as its own step.
        block = _landed(cache[int(index)], index)
        if block is None:
            return
        exact.append(fingerprint_tensor(block).reshape(1))
        coarse.append(coarse_fingerprint_tensor(block).reshape(1))
    if not exact:
        return
    tap_read(layer_id, f"{key}.block_digest", torch.cat(exact))
    tap_read(layer_id, f"{key}.block_digest_coarse", torch.cat(coarse))


@_never_breaks_the_run
def tap_operator_rows(layer_id: int, cache, block_table, seqused) -> None:
    """The compressed entries this request hands the operator, as rows.

    🔵 golden's ``sa_in.kv`` is ``[original rows] ++ [compressed entries]`` --
    one tensor of width ``T + seqused_cmp_kv`` (measured: 5+2=7 at ratio 2,
    5+5=10 at ratio 1). On this side the two halves are **two arguments**:
    the SWA cache and the compressed cache, passed separately. So the
    concatenation golden records does not exist here as a tensor.

    ⚠️ Which is why this records the missing half under **its own key** and
    the manifest declares the relation, rather than building golden's tensor.
    Fabricating it would put a constructed object under a name the operator
    never saw as one block -- the same move already refused for
    ``sa_in.topk_idxs`` and for layer 0's ``layer.in``. The relation is
    stated so the comparison can still be made:

        golden sa_in.kv  ==  ours sa_in.kv  ++  ours sa_in.cmp_kv_rows

    Read by **basic indexing** into the block the table names, which is the
    access that works: two gather kernels have failed on this tensor.
    """
    if not records_reads(layer_id):
        return
    if not isinstance(cache, torch.Tensor) or \
            not isinstance(block_table, torch.Tensor):
        return
    ids = _block_ids(block_table)
    if ids is None:
        return
    rows = None if not isinstance(seqused, torch.Tensor) or \
        seqused.numel() == 0 else int(seqused.reshape(-1)[0])
    block = cache[int(ids.reshape(-1)[0])]
    tap_read(layer_id, "sa_in.cmp_kv_rows",
             block if rows is None else block[:rows])
    # How many rows were taken, because "the first seqused_cmp_kv of the
    # first block" is an assumption about layout, and a reader comparing
    # widths needs to know which assumption produced this one.
    tap_read(layer_id, "sa_in.cmp_kv_rows.count",
             torch.tensor([-1 if rows is None else rows]))


def _landed(view, index: int):
    """A copy of ``view`` that later ops can consume, or ``None`` with a say.

    Tried as ``clone``; a failure here is the AsStrided wall and is reported
    as **that step**, not folded into the caller's "tap failed". The two have
    different remedies: a failure to land means the tensor cannot be read at
    all on this device, while a failure in the caller means the addressing
    was wrong.
    """
    try:
        return view.clone()
    except Exception as exc:                                 # noqa: BLE001
        from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

        announce_once(
            "read_taps_land_failed",
            "could not materialise block %d of the cache (%r). This is the "
            "contiguity conversion, not the indexing: a basic index gives a "
            "view and the first op needing contiguous memory inserts "
            "AsStrided. Block digests are absent from this dump; the block "
            "ids are not.", index, exc)
        return None


_DESCRIBED: set = set()


def _describe_once(what: str, tensor, ids) -> None:
    """Shape, strides, contiguity and the index range -- before anything can
    fail on them.

    🔴 Printed unconditionally rather than only on failure, because the two
    candidate causes are decided by exactly these numbers and one of them
    does not raise:

        not contiguous, ids in range   the conversion is the problem
        ids out of range               this call's addressing is wrong,
                                       which also explains why it is
                                       intermittent

    ⚠️ Self-contained on purpose. The quantities that would have answered
    this were already being logged elsewhere, under another switch, on
    another line -- and a diagnosis split across two lines that are not
    always both present is one nobody assembles.
    """
    if what in _DESCRIBED:
        return
    _DESCRIBED.add(what)
    from vllm.logger import logger

    try:
        ids_list = [int(v) for v in ids.reshape(-1)[:8]]
        logger.info(
            "DSV4.1 read tap %s: shape=%s stride=%s contiguous=%s "
            "storage_offset=%d numel=%d dtype=%s | ids=%s max_id=%s "
            "dim0=%d -> in_range=%s", what, tuple(tensor.shape),
            tuple(tensor.stride()), tensor.is_contiguous(),
            tensor.storage_offset(), tensor.numel(), tensor.dtype,
            ids_list, max(ids_list) if ids_list else None,
            tensor.shape[0],
            all(0 <= i < tensor.shape[0] for i in ids_list))
    except Exception as exc:                                 # noqa: BLE001
        logger.info("DSV4.1 read tap %s: could not be described (%r)",
                    what, exc)


def _block_ids(block_table: torch.Tensor, seqused=None,
               block_size: "int | None" = None):
    """The blocks this request actually uses, in order.

    🔴 Bounded by the sequence length, not by ``>= 0``. Block tables are
    padded with **zero**, not with -1 (``dsa_v1.py:1570`` fills unused rows
    with 0), so filtering on non-negative keeps the padding -- and block 0 is
    a real block belonging to somebody else. That is what produced
    ``ori=[5, 0]`` and ``cmp=[84, 85, 0]`` in the dumps: a trailing 0 that
    reads as "this request also touched block 0" and would have had its
    contents digested and attributed here.

    ⚠️ When the length or the page size is unavailable the old behaviour is
    kept and said out loud, because an unbounded list is still better than no
    list -- but a reader must know which of the two they are holding.
    """
    flat = block_table.reshape(-1)
    if flat.numel() == 0:
        return None
    needed = _blocks_needed(seqused, block_size)
    if needed is None:
        _report_unbounded()
        valid = flat[flat >= 0]
    else:
        valid = flat[:needed]
    if valid.numel() == 0:
        return None
    seen = torch.zeros_like(valid, dtype=torch.bool)
    seen[0] = True
    if valid.numel() > 1:
        seen[1:] = valid[1:] != valid[:-1]
    return valid[seen]


def _blocks_needed(seqused, block_size: "int | None"):
    """How many blocks a sequence of this length occupies, or ``None``."""
    if not isinstance(seqused, torch.Tensor) or seqused.numel() == 0:
        return None
    if not block_size or block_size <= 0:
        return None
    longest = int(seqused.reshape(-1).max())
    return max(1, -(-longest // block_size))


def _report_unbounded() -> None:
    from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

    announce_once(
        "read_taps_unbounded_blocks",
        "the block list could not be bounded by a sequence length, so it is "
        "every non-negative entry in the table -- including the zero padding, "
        "and block 0 belongs to somebody else. Treat the ids and any digest "
        "of them as an upper bound, not as this request's blocks.")


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
    _DISABLED.clear()
    _DESCRIBED.clear()


def _report_truncated(layer_id: int, key: str, total: int) -> None:
    from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

    announce_once(
        f"read_tap_truncated_{key}",
        "layer %d read %d blocks for %r and only the first %d are in the "
        "dump. A difference in the blocks beyond that will not appear, so "
        "treat a match on this key as partial.", layer_id, total, key,
        MAX_BLOCKS)
