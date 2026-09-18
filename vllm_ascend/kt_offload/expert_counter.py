# SPDX-License-Identifier: Apache-2.0
"""Per-layer routed-expert frequency counter (DSV4.1 MoE offload, M0c).

Builds the ``[40, 384]`` table the ``frequency`` resident strategy reads. Off
unless ``VLLM_ASCEND_DSV41_EXPERT_COUNT=<output dir>``; the flag is read once
at import (``ENABLED``) and every call site in routed_experts.py is guarded by
it, so with the variable unset nothing here runs.

Output: ``<dir>/rank<dp>_pid<pid>.pt`` per process (one file per DP rank; no
two ranks write the same file)::

    counts[bucket]  [40, 384] int64  bucket in {decode, prefill}
    tokens[bucket]  [40]      int64  tokens counted for that layer
    skipped         {reason: n}      calls not counted, by reason
    steps, topk, rank, dp_size

Invariant (checked by tools/kt_offload/expert_count_report.py): for every
rank, layer and bucket, ``counts.sum() == tokens * topk``.

What is counted, and why (adapted from integration's tools/kt_offload version):
  * the router's global *logical* expert ids 0..383, before log2phy and before
    mix_placement appends the shared expert -- so DP ranks add up directly;
  * only when ``enable_force_load_balance`` is False: the profile/dummy run
    replaces the ids with a random permutation. Those calls go to ``skipped``;
  * only this rank's real tokens: DP padding rows are dropped, and under an
    ALLGATHER step with DP>1 (the router sees every rank's rows) only this
    rank's slice ``[dp_rank * max_tokens_across_dp, + num_actual_tokens)``;
  * eager only. A graph replay runs no Python, so under graph mode only the
    capture run would be counted. ``check_config`` refuses a non-eager config
    at model construction and ``record`` refuses to run inside a capture.
"""

from __future__ import annotations

import atexit
import os
import re

import torch

from vllm_ascend import envs

OUT_DIR: str = envs.VLLM_ASCEND_DSV41_EXPERT_COUNT
ENABLED: bool = bool(OUT_DIR)

N_LAYERS, N_EXPERTS, TOPK = 40, 384, 6
DUMP_EVERY = 200  # steps (layer-0 calls) between dumps; also dumped at exit
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_S: dict | None = None


class ExpertCounterError(RuntimeError):
    pass


def check_config(vllm_config) -> None:
    """Called at MoE construction when enabled: refuse a config it cannot count."""
    if not getattr(vllm_config.model_config, "enforce_eager", False):
        raise ExpertCounterError(
            "VLLM_ASCEND_DSV41_EXPERT_COUNT needs --enforce-eager (EAGER=1): a graph replay runs no "
            "Python, so the counter would only see the capture run and report a table of zeros "
            "that passes its own sum check."
        )


def _state(device) -> dict:
    global _S
    if _S is None:
        _S = {
            "counts": {b: torch.zeros(N_LAYERS, N_EXPERTS, dtype=torch.int64, device=device) for b in BUCKETS},
            "tokens": {b: [0] * N_LAYERS for b in BUCKETS},
            "skipped": {},
            "steps": 0,
            "rank": None,
            "dp_size": None,
        }
        os.makedirs(OUT_DIR, exist_ok=True)
        atexit.register(dump)
    return _S


BUCKETS = ("decode", "prefill")


def _skip(s: dict, why: str) -> None:
    s["skipped"][why] = s["skipped"].get(why, 0) + 1


# -- runtime context (indirections so the CPU tests can supply them) ---------


def _capturing() -> bool:
    predicate = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    return predicate is not None and predicate() is True


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
    if _capturing():
        raise ExpertCounterError(
            "expert counter reached inside a graph capture; it only works with --enforce-eager "
            "(replays would not be counted)."
        )
    s = _state(topk_ids.device)
    name = getattr(layer, "layer_name", "") or ""
    m = _LAYER_RE.search(name)
    if m is None or "dspark" in name or "mtp" in name:
        _skip(s, "not a main layer (no layer index, or dspark/mtp)")
        return
    layer_idx = int(m.group(1))
    if layer_idx >= N_LAYERS:
        _skip(s, f"layer index {layer_idx} >= {N_LAYERS}")
        return
    if force_load_balance:
        _skip(s, "force_load_balance (random topk: profile/dummy run)")
        return
    if topk_ids.dim() != 2 or int(topk_ids.shape[1]) != TOPK:
        _skip(s, f"topk shape {tuple(topk_ids.shape)} is not [T, {TOPK}]")
        return
    try:
        split = _batch_split()
    except Exception as exc:  # noqa: BLE001
        _skip(s, f"no forward context ({type(exc).__name__})")
        return
    if split is None:
        _skip(s, "no attn metadata (dummy run?)")
        return
    n_dec, n_act = split
    if n_dec is None:
        _skip(s, "attn metadata has no decode count")
        return

    if s["rank"] is None:
        s["rank"], s["dp_size"] = _dp_info()
    rank, dp_size = s["rank"], s["dp_size"]
    lo_row = 0
    gathered, per_rank = _rows_are_dp_gathered()
    if gathered and dp_size > 1:
        if per_rank is None or topk_ids.shape[0] != per_rank * dp_size:
            _skip(s, f"allgather rows {topk_ids.shape[0]} != max_tokens_across_dp {per_rank} x dp {dp_size}")
            return
        lo_row = rank * per_rank
    if topk_ids.shape[0] < lo_row + n_act:
        _skip(s, f"router rows {topk_ids.shape[0]} < rank rows end {lo_row + n_act}")
        return

    ids = topk_ids[lo_row : lo_row + n_act]
    n_dec = min(n_dec, n_act)
    for bucket, lo, hi in (("decode", 0, n_dec), ("prefill", n_dec, n_act)):
        if hi > lo:
            flat = ids[lo:hi].reshape(-1).to(torch.int64)
            s["counts"][bucket][layer_idx].index_add_(0, flat, torch.ones_like(flat))
            s["tokens"][bucket][layer_idx] += hi - lo
    if layer_idx == 0:
        s["steps"] += 1
        if s["steps"] % DUMP_EVERY == 0:
            dump()


def output_path(rank) -> str:
    return os.path.join(OUT_DIR, f"rank{rank}_pid{os.getpid()}.pt")


def dump() -> str | None:
    s = _S
    if s is None:
        return None
    rank = s["rank"] if s["rank"] is not None else "x"
    out = {
        "counts": {b: t.detach().cpu() for b, t in s["counts"].items()},
        "tokens": {b: torch.tensor(v, dtype=torch.int64) for b, v in s["tokens"].items()},
        "skipped": dict(s["skipped"]),
        "steps": s["steps"],
        "topk": TOPK,
        "rank": rank,
        "dp_size": s["dp_size"],
    }
    path = output_path(rank)
    torch.save(out, path + ".tmp")
    os.replace(path + ".tmp", path)
    return path
