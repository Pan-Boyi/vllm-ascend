# SPDX-License-Identifier: Apache-2.0
"""Per-layer routed-expert frequency counter.

Builds the ``[40, 384]`` table the ``frequency`` resident strategy reads. Off
unless ``VLLM_ASCEND_DSV41_EXPERT_COUNT=<output dir>``; the flag is read once
at import (``ENABLED``) and every call site in routed_experts.py is guarded by
it, so with the variable unset nothing here runs.

Output: ``<dir>/rank<dp>_pid<pid>.pt`` per process (one file per DP rank; no
two ranks write the same file)::

    counts[bucket]  [40, 384] int64  bucket in {decode, prefill}
    tokens[bucket]  [40]      int64  tokens counted for that layer
    skipped         {reason: n}      per-layer calls not counted, by reason
    forwards        the denominator (see below)
    steps           == forwards["counted"] (kept for the report tool)
    topk, rank, dp_size, config_enforce_eager
    dump_every      forwards between periodic dumps, i.e. how coarsely two
                    snapshots of this file can be diffed (see dump())

Invariants: for every rank, layer and bucket, ``counts.sum() == tokens * topk``;
and ``forwards["seen"] == forwards["counted"] + sum(forwards["excluded"].values())``.

The denominator (``forwards``). A forward is one call on main layer 0 (as
``steps`` always was). Every such call gets exactly one disposition:

    seen                  layer-0 calls that reached record()
    counted               ... whose tokens are in ``counts`` right now
    excluded[reason]      ... that are not, by reason. Fixed keys:
        captured                  the current stream was being captured
        capture_state_unknown     the capture predicate could not be evaluated
        profile_run               enable_force_load_balance (random ids)
        discarded_before_capture  counted, then dropped when a capture followed
      plus the older per-call reasons (no attn metadata, shape, ...) verbatim.
    replayed              0, or None = not observable (see below); NOT in the sum
    replay_note           why replayed is what it is
    captured_token_sizes  router row counts seen under capture: the batch sizes
                          whose later forwards may be replays nobody counted

What is counted, and why:
  * the router's global *logical* expert ids 0..383, before log2phy and before
    mix_placement appends the shared expert -- so DP ranks add up directly;
  * only when ``enable_force_load_balance`` is False: the profile/dummy run
    replaces the ids with a random permutation (excluded as ``profile_run``);
  * only this rank's real tokens: DP padding rows are dropped, and under an
    ALLGATHER step with DP>1 (the router sees every rank's rows) only this
    rank's slice ``[dp_rank * max_tokens_across_dp, + num_actual_tokens)``;
  * only in forwards that are provably neither captured nor replayed. Graph
    mode is allowed; the counter does not refuse it.

Graph mode, precisely:
  * Captured: the predicate is the one the kt_offload production path uses
    (layer.py ``_capturing``: ``torch.npu.is_current_stream_capturing() is
    True``), copied, not imported. It reads the *current stream* -- the stream
    this call's ``index_add_`` would be enqueued on -- so it judges the very
    work that would otherwise end up in the graph, not "some capture is in
    progress somewhere". A captured call counts nothing: if it did, every
    replay would re-run the ``index_add_`` on device while the Python token
    totals stayed put.
  * Replayed: a replay runs no Python, so record() never sees one and cannot
    count them. What it can see is the capture that every replay needs: a graph
    containing this layer can only be replayed after that layer was captured,
    and capturing it runs record() under capture. Hence ``replayed == 0`` is
    provable when this process saw no captured call and no unknown one; else
    ``replayed is None`` (unobservable) and ``captured_token_sizes`` names the
    batch sizes that may have gone to a graph.
  * Capture warm-ups: before each capture vLLM runs ``cudagraph_num_of_warmups``
    eager dummy forwards of the same size (``_warmup_and_capture``), and for a
    FULL graph with attention metadata, so they pass every other filter and
    look like real decode traffic. Captures happen at start-up, before any
    request, so on every captured call everything counted so far is dropped
    and moved to ``discarded_before_capture``. If a capture ever happened after
    real traffic this under-counts, visibly -- never over-counts.
  * Cannot evaluate (no ``torch.npu``, no ``is_current_stream_capturing``, it
    raises, or it returns anything but a real bool): the call is NOT counted,
    it is excluded as ``capture_state_unknown`` and one WARNING is logged. Not
    counting is the only direction that cannot put graph-captured work into
    the table; and it does not raise, because this is a statistics hook inside
    the serving forward. The failure is still loud where it matters: counted
    stays 0 while seen grows, and replayed becomes None. The same rule means a
    plain CPU process (no NPU) counts nothing unless it supplies a predicate.
"""
from __future__ import annotations

import atexit
import os
import re

import torch
from vllm.logger import logger  # the vllm logger: a vllm_ascend.* name loses INFO in the worker

from vllm_ascend import envs

OUT_DIR: str = envs.VLLM_ASCEND_DSV41_EXPERT_COUNT
ENABLED: bool = bool(OUT_DIR)

N_LAYERS, N_EXPERTS, TOPK = 40, 384, 6


def _dump_every() -> int:
    """``VLLM_ASCEND_DSV41_EXPERT_COUNT_EVERY``, or 200.

    Not a validator that raises: this module is imported by routed_experts.py
    unconditionally, so a raise here would take down a service that never asked
    for the counter. Unparsable or < 1 keeps 200 and logs one WARNING -- and the
    value in force is written into every dump (``dump_every``), so a reading can
    never be wrong about which granularity produced it.
    """
    raw = envs.VLLM_ASCEND_DSV41_EXPERT_COUNT_EVERY
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 0
    if n < 1:
        logger.warning(
            "kt_offload expert counter: VLLM_ASCEND_DSV41_EXPERT_COUNT_EVERY=%r is not an integer >= 1; "
            "keeping the default 200 forwards between dumps",
            raw,
        )
        return 200
    return n


DUMP_EVERY = _dump_every()  # steps (layer-0 calls) between dumps; also dumped at exit
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_S: dict | None = None


_CONFIG_ENFORCE_EAGER: bool | None = None  # what check_config saw; informational only
_UNKNOWN_WARNED = False


def check_config(vllm_config) -> None:
    """Called at MoE construction when enabled. Records the config; no longer refuses graph mode.

    Informational only: whether graphs exist is judged per call from the
    capture predicate (see the module docstring), not from this flag.
    """
    global _CONFIG_ENFORCE_EAGER
    _CONFIG_ENFORCE_EAGER = bool(getattr(vllm_config.model_config, "enforce_eager", False))
    if not _CONFIG_ENFORCE_EAGER:
        logger.info(
            "kt_offload expert counter with graph mode: counting only forwards the capture predicate proves "
            "eager; replays are not observable and are reported as such (forwards.replayed=None)."
        )


def _alloc_counts(device) -> dict:
    return {b: torch.zeros(N_LAYERS, N_EXPERTS, dtype=torch.int64, device=device) for b in BUCKETS}


def _state(device, allocate: bool) -> dict:
    """The per-process state. Device tensors only when ``allocate`` (i.e. provably not under capture):
    an allocation inside a capture would come from the graph's private pool."""
    global _S
    if _S is None:
        _S = {
            "counts": _alloc_counts(device) if allocate else None,
            "counts_stale": False,  # a capture dropped the counts; zero them on the next eager call
            "tokens": {b: [0] * N_LAYERS for b in BUCKETS},
            "skipped": {},
            "steps": 0,  # == forwards["counted"]
            "forwards": {"seen": 0, "counted": 0, "excluded": {}},
            "captured_token_sizes": set(),
            "rank": None,
            "dp_size": None,
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        atexit.register(dump)
        logger.info(
            "kt_offload expert counter on: %d layers x %d experts, device %s, out_dir %s, dump every %d forwards",
            N_LAYERS,
            N_EXPERTS,
            device,
            OUT_DIR,
            DUMP_EVERY,
        )
    return _S


BUCKETS = ("decode", "prefill")


def _skip(s: dict, why: str) -> None:
    s["skipped"][why] = s["skipped"].get(why, 0) + 1


def _exclude(s: dict, is_forward: bool, why: str, call_reason: str | None = None) -> None:
    """One call not counted: per-call reason in ``skipped``; a layer-0 call also in the denominator."""
    _skip(s, call_reason or why)
    if is_forward:
        ex = s["forwards"]["excluded"]
        ex[why] = ex.get(why, 0) + 1


def _discard_before_capture(s: dict) -> int:
    """A capture was observed: everything counted so far was start-up dummy traffic. Drop it, keep the tally.

    Host-side only. Runs under capture, so it must not touch a device tensor:
    a ``zero_()`` here would be recorded into the graph and re-run on every
    replay. The device counts are marked stale and zeroed by the next eager
    call (``_live_counts``); dump() writes zeros for stale counts.
    """
    n = s["forwards"]["counted"]
    s["counts_stale"] = True
    for b in BUCKETS:
        s["tokens"][b] = [0] * N_LAYERS
    if n:
        ex = s["forwards"]["excluded"]
        ex["discarded_before_capture"] = ex.get("discarded_before_capture", 0) + n
    s["forwards"]["counted"] = 0
    s["steps"] = 0
    return n


def _live_counts(s: dict, device) -> dict:
    """The device counts, for an eager (provably not captured) call only."""
    if s["counts"] is None:
        s["counts"] = _alloc_counts(device)
    elif s["counts_stale"]:
        for t in s["counts"].values():
            t.zero_()
    s["counts_stale"] = False
    return s["counts"]


def _replay_accounting(s: dict) -> tuple[int | None, str]:
    ex = s["forwards"]["excluded"]
    captured, unknown = ex.get("captured", 0), ex.get("capture_state_unknown", 0)
    if captured == 0 and unknown == 0 and not s["captured_token_sizes"]:
        return 0, "no call was captured and none was undecidable in this process, so no graph holds this layer"
    return None, (
        f"not observable: a replay runs no Python ({captured} captured, {unknown} undecidable layer-0 calls); "
        f"batch sizes that may have been replayed: {sorted(s['captured_token_sizes'])}"
    )


# -- runtime context (indirections so the CPU tests can supply them) ---------


_UNKNOWN_WHY = ""


def _capturing() -> bool | None:
    """Is the *current stream* being captured? True / False, or None when that cannot be told.

    The capture test is layer.py's ``_capturing`` (``is_current_stream_capturing()
    is True``), copied, not imported (importing the model package applies the
    V4.1 platform patches). Unlike layer.py, "not capturing" must also be
    positive: only a real ``False`` proves it. No ``torch.npu``, no predicate,
    an exception, or a non-bool (e.g. a Mock) all give None, and record() then
    does not count (fail closed; see the module docstring).
    """
    global _UNKNOWN_WHY
    try:
        predicate = torch.npu.is_current_stream_capturing
    except AttributeError as exc:
        _UNKNOWN_WHY = f"torch.npu.is_current_stream_capturing unavailable ({exc})"
        return None
    try:
        result = predicate()
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot tell"
        _UNKNOWN_WHY = f"is_current_stream_capturing() raised {type(exc).__name__}: {exc}"
        return None
    if result is True:
        return True
    if result is False:
        return False
    _UNKNOWN_WHY = f"is_current_stream_capturing() returned {type(result).__name__}, not a bool"
    return None


def _batch_split() -> tuple[int | None, int] | None:
    """(num_decode_tokens, num_actual_tokens) of this rank, from attn metadata. No device sync."""
    from vllm.forward_context import get_forward_context

    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = next((m for m in md.values() if getattr(m, "num_actual_tokens", None) is not None), None)
    if md is None:
        return None
    n_act = int(md.num_actual_tokens)
    n_dec = getattr(md, "num_decode_tokens", None)
    if n_dec is None:
        n_dec = getattr(md, "num_decodes", None)
    return (None if n_dec is None else int(n_dec)), n_act


def _dp_info() -> tuple[int, int]:
    """(dp_rank, dp_size)."""
    from vllm.distributed import get_dp_group

    g = get_dp_group()
    return int(g.rank_in_group), int(g.world_size)


def _rows_are_dp_gathered() -> tuple[bool, int | None]:
    """Whether the router saw every DP rank's rows (ALLGATHER, DP>1), and the per-rank row count."""
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

    if _EXTRA_CTX.moe_comm_type != MoECommType.ALLGATHER:
        return False, None
    return True, _EXTRA_CTX.max_tokens_across_dp


# -- the hook ------------------------------------------------------------------


def record(layer, topk_ids: torch.Tensor, force_load_balance: bool) -> None:
    """Count one router output. Call only when ENABLED (the call site checks)."""
    global _UNKNOWN_WARNED
    # Judged first and per call: the stream this call's index_add_ would go to.
    cap = _capturing()
    s = _state(topk_ids.device, allocate=cap is False)
    name = getattr(layer, "layer_name", "") or ""
    m = _LAYER_RE.search(name)
    if m is None or "dspark" in name or "mtp" in name:
        _skip(s, "not a main layer (no layer index, or dspark/mtp)")
        return
    layer_idx = int(m.group(1))
    if layer_idx >= N_LAYERS:
        _skip(s, f"layer index {layer_idx} >= {N_LAYERS}")
        return
    fwd = layer_idx == 0
    if fwd:
        s["forwards"]["seen"] += 1

    if cap is True:
        s["captured_token_sizes"].add(int(topk_ids.shape[0]))
        dropped = _discard_before_capture(s)
        _exclude(s, fwd, "captured")
        if fwd:
            logger.info(
                "kt_offload expert counter: layer-0 call of %d rows under graph capture, not counted; "
                "%d earlier counted forwards dropped as start-up warm-up",
                int(topk_ids.shape[0]),
                dropped,
            )
        return
    if cap is not False:
        if not _UNKNOWN_WARNED:
            _UNKNOWN_WARNED = True
            logger.warning(
                "kt_offload expert counter: cannot tell whether the current stream is being captured (%s); "
                "not counting such calls (forwards.excluded.capture_state_unknown)",
                _UNKNOWN_WHY,
            )
        _exclude(s, fwd, "capture_state_unknown")
        return
    if force_load_balance:
        _exclude(s, fwd, "profile_run", "force_load_balance (random topk: profile/dummy run)")
        return
    if topk_ids.dim() != 2 or int(topk_ids.shape[1]) != TOPK:
        _exclude(s, fwd, f"topk shape {tuple(topk_ids.shape)} is not [T, {TOPK}]")
        return
    try:
        split = _batch_split()
    except Exception as exc:  # noqa: BLE001
        _exclude(s, fwd, f"no forward context ({type(exc).__name__})")
        return
    if split is None:
        _exclude(s, fwd, "no attn metadata (dummy run?)")
        return
    n_dec, n_act = split
    if n_dec is None:
        _exclude(s, fwd, "attn metadata has no decode count")
        return

    if s["rank"] is None:
        s["rank"], s["dp_size"] = _dp_info()
    rank, dp_size = s["rank"], s["dp_size"]
    lo_row = 0
    gathered, per_rank = _rows_are_dp_gathered()
    if gathered and dp_size > 1:
        if per_rank is None or topk_ids.shape[0] != per_rank * dp_size:
            _exclude(s, fwd, f"allgather rows {topk_ids.shape[0]} != max_tokens_across_dp {per_rank} x dp {dp_size}")
            return
        lo_row = rank * per_rank
    if topk_ids.shape[0] < lo_row + n_act:
        _exclude(s, fwd, f"router rows {topk_ids.shape[0]} < rank rows end {lo_row + n_act}")
        return

    counts = _live_counts(s, topk_ids.device)
    ids = topk_ids[lo_row : lo_row + n_act]
    n_dec = min(n_dec, n_act)
    for bucket, lo, hi in (("decode", 0, n_dec), ("prefill", n_dec, n_act)):
        if hi > lo:
            flat = ids[lo:hi].reshape(-1).to(torch.int64)
            counts[bucket][layer_idx].index_add_(0, flat, torch.ones_like(flat))
            s["tokens"][bucket][layer_idx] += hi - lo
    if fwd:
        s["forwards"]["counted"] += 1
        s["steps"] += 1
        if s["steps"] % DUMP_EVERY == 0:
            dump()


def output_path(rank) -> str:
    return os.path.join(OUT_DIR, f"rank{rank}_pid{os.getpid()}.pt")


def dump() -> str | None:
    """Write the whole cumulative state to ``output_path``. Called every
    ``DUMP_EVERY`` counted forwards and once at exit.

    Diffing two periodic dumps to attribute counts to one request is skewed: a
    dump is triggered from *layer 0*'s record(), i.e. at the START of a forward,
    so at that instant layer 0 holds F forwards while layers 1..39 still hold
    F-1. The exit dump has no such skew.
    """
    s = _S
    if s is None:
        return None
    rank = s["rank"] if s["rank"] is not None else "x"
    replayed, replay_note = _replay_accounting(s)
    out = {
        "counts": {
            b: (
                torch.zeros(N_LAYERS, N_EXPERTS, dtype=torch.int64, device="cpu")
                if s["counts"] is None or s["counts_stale"]
                else s["counts"][b].detach().cpu()
            )
            for b in BUCKETS
        },
        "tokens": {b: torch.tensor(v, dtype=torch.int64, device="cpu") for b, v in s["tokens"].items()},
        "skipped": dict(s["skipped"]),
        "steps": s["steps"],
        "forwards": {
            "seen": s["forwards"]["seen"],
            "counted": s["forwards"]["counted"],
            "excluded": dict(s["forwards"]["excluded"]),
            "replayed": replayed,
            "replay_note": replay_note,
            "captured_token_sizes": sorted(s["captured_token_sizes"]),
        },
        "config_enforce_eager": _CONFIG_ENFORCE_EAGER,
        "dump_every": DUMP_EVERY,
        "topk": TOPK,
        "rank": rank,
        "dp_size": s["dp_size"],
    }
    path = output_path(rank)
    torch.save(out, path + ".tmp")
    os.replace(path + ".tmp", path)
    logger.info(
        "kt_offload expert counter dumped %s: %d/%d forwards counted, excluded %s, replayed %s; "
        "%d decode + %d prefill tokens on layer 0, per-call skipped %s",
        path,
        s["forwards"]["counted"],
        s["forwards"]["seen"],
        dict(s["forwards"]["excluded"]) or "none",
        "unobservable" if replayed is None else replayed,
        s["tokens"]["decode"][0],
        s["tokens"]["prefill"][0],
        dict(s["skipped"]) or "nothing",
    )
    return path
