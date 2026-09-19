# SPDX-License-Identifier: Apache-2.0
"""Startup topology check for DeepSeek-V4.1 (kv / index / candidate sources).

Modeled on the official ``build_layer_plan`` validator, but:
  * reads this checkpoint's key names (``kv_source_layer_ids`` /
    ``index_source_layer_ids`` / ``candidate_source_layer_id``). The official
    validator reads the Aurora names (``kv_source_layers`` ...) and would reject
    this checkpoint with "config is missing 'kv_source_layers'".
  * derives roles with the SAME rules the running code uses (upstream
    ``vllm/models/deepseek_v4_1/attention.py``), so the validator and the model
    cannot disagree about who owns what:
        is_kv_source    = layer < num_hidden_layers and layer in kv_source_layer_ids
        is_index_source = layer < num_hidden_layers and layer in index_source_layer_ids
        kv/index source of a compressed layer = max(s in sources if s <= layer)
        indexer exists  iff is_index_source; it owns the index-K cache iff is_kv_source
        candidate write iff is_index_source and layer == candidate_source_layer_id
  * checks every entry of compress_ratios (MTP layers included), as the model
    constructs and checks every layer, and requires the list to cover exactly
    num_hidden_layers + num_nextn_predict_layers.

Without it the running code accepts several bad configs silently (source out of
the backbone, candidate source that is not an index source, a fifth index-K
owner, a short compress_ratios list) and misattributes two others (a bare
"max() arg is an empty sequence"; a kv source that is not an index source
reported as a pipeline-parallel limit).

Every failure raises ``V41TopologyError`` naming the field, the layer and the rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ALLOWED_RATIOS = (0, 1, 2)
#: The index-K cache/memory plan is sized for this many owners (memory
#: dsv41-index-k-cache-count: owns_k by kv_source_layers -> 4). Change together with
#: the plan; the validator refuses a checkpoint that needs a different count.
EXPECTED_INDEX_K_OWNERS = 4


class V41TopologyError(ValueError):
    pass


@dataclass(frozen=True)
class LayerRole:
    layer: int
    ratio: int
    is_backbone: bool
    is_kv_source: bool
    is_index_source: bool
    kv_source: int | None
    index_source: int | None
    has_indexer: bool
    owns_index_k: bool
    candidate_write: bool
    uses_candidates: bool


def _get(cfg: Any, name: str):
    if isinstance(cfg, dict):
        cfg = cfg.get("text_config", cfg)
        if name not in cfg:
            raise V41TopologyError(f"DeepSeek-V4.1 config is missing {name!r}")
        return cfg[name]
    if not hasattr(cfg, name):
        raise V41TopologyError(f"DeepSeek-V4.1 config is missing {name!r}")
    return getattr(cfg, name)


def _int_list(cfg, name) -> tuple[int, ...]:
    v = _get(cfg, name)
    if not isinstance(v, (list, tuple)) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise V41TopologyError(f"DeepSeek-V4.1 {name} must be a list of ints, got {v!r}")
    return tuple(v)


def derive_roles(cfg) -> list[LayerRole]:
    """Roles exactly as the running code derives them (see module docstring).
    Raises only where the running code cannot even form the role."""
    n = int(_get(cfg, "num_hidden_layers"))
    ratios = _int_list(cfg, "compress_ratios")
    kv = _int_list(cfg, "kv_source_layer_ids")
    idx = _int_list(cfg, "index_source_layer_ids")
    cand = int(_get(cfg, "candidate_source_layer_id"))
    roles = []
    for layer, r in enumerate(ratios):
        backbone = layer < n
        is_kv = backbone and layer in kv
        is_idx = backbone and layer in idx
        kvs = max((s for s in kv if s <= layer), default=None) if r > 0 else None
        ixs = max((s for s in idx if s <= layer), default=None) if r > 0 else None
        roles.append(
            LayerRole(
                layer=layer,
                ratio=r,
                is_backbone=backbone,
                is_kv_source=is_kv,
                is_index_source=is_idx,
                kv_source=kvs,
                index_source=ixs,
                has_indexer=is_idx,
                owns_index_k=is_idx and is_kv,
                candidate_write=is_idx and layer == cand,
                uses_candidates=is_idx and 0 <= cand < layer,
            )
        )
    return roles


def validate_v41_topology(cfg, *, expected_index_k_owners: int = EXPECTED_INDEX_K_OWNERS) -> list[LayerRole]:
    n = int(_get(cfg, "num_hidden_layers"))
    n_mtp = int(_get(cfg, "num_nextn_predict_layers"))
    ratios = _int_list(cfg, "compress_ratios")
    kv = _int_list(cfg, "kv_source_layer_ids")
    idx = _int_list(cfg, "index_source_layer_ids")
    cand = int(_get(cfg, "candidate_source_layer_id"))

    # --- layer count ------------------------------------------------------------
    if n <= 0:
        raise V41TopologyError(f"num_hidden_layers must be positive, got {n}")
    if len(ratios) != n + n_mtp:
        raise V41TopologyError(
            f"compress_ratios has {len(ratios)} entries but num_hidden_layers + "
            f"num_nextn_predict_layers = {n} + {n_mtp} = {n + n_mtp}; every constructed "
            "layer reads compress_ratios[layer_id] (cvllm attention.py:252-257)"
        )
    # --- ratio set --------------------------------------------------------------
    bad = [(i, r) for i, r in enumerate(ratios) if r not in ALLOWED_RATIOS]
    if bad:
        raise V41TopologyError(
            f"compress_ratios must be in {set(ALLOWED_RATIOS)}; offending (layer, ratio): {bad}"
        )
    # --- source lists -----------------------------------------------------------
    for name, src in (("kv_source_layer_ids", kv), ("index_source_layer_ids", idx)):
        if not src:
            raise V41TopologyError(f"{name} is empty but compressed layers exist")
        if tuple(sorted(set(src))) != src:
            raise V41TopologyError(f"{name} must be sorted and unique, got {list(src)}")
        out = [s for s in src if not 0 <= s < n]
        if out:
            raise V41TopologyError(
                f"{name} contains {out}, outside the backbone [0, {n}); the running code "
                "silently treats such a layer as not-a-source (is_backbone guard)"
            )
        local = [s for s in src if ratios[s] == 0]
        if local:
            raise V41TopologyError(
                f"{name} contains {local}, which have compress_ratio 0 (sliding-window only); "
                "a source must own a compressed stream"
            )
    not_idx = sorted(set(kv) - set(idx))
    if not_idx:
        raise V41TopologyError(
            f"kv sources {not_idx} are not index sources: the index-K cache is only built "
            "for index sources (attention.py:367-380), so nobody would own it"
        )
    # --- consumers --------------------------------------------------------------
    roles = derive_roles(cfg)
    for r in roles:
        if r.ratio == 0:
            continue
        if r.kv_source is None or r.index_source is None:
            first = min(kv[0], idx[0])
            raise V41TopologyError(
                f"layer {r.layer} (ratio {r.ratio}) is a consumer before its source: the first "
                f"kv/index source is layer {first}; attention.py:284-289 would fail with a bare "
                "'max() arg is an empty sequence'"
            )
        if ratios[r.kv_source] != r.ratio:
            raise V41TopologyError(
                f"layer {r.layer} has ratio {r.ratio} but its kv source layer {r.kv_source} has "
                f"ratio {ratios[r.kv_source]} (a consumer must read a stream of its own ratio)"
            )
        # (No separate index-source ratio check: it is implied. kv ⊆ index gives
        #  kv_source <= index_source <= layer, and index_source's own kv source is the
        #  same kv_source, whose ratio was already checked when index_source was visited.)
    # --- index-K owners ---------------------------------------------------------
    owners = [r.layer for r in roles if r.owns_index_k]
    if len(owners) != expected_index_k_owners:
        raise V41TopologyError(
            f"{len(owners)} index-K cache owners {owners} (owns_k = kv source that is also an "
            f"index source); the cache/memory plan expects exactly {expected_index_k_owners}"
        )
    # --- candidate source -------------------------------------------------------
    writers = [r.layer for r in roles if r.candidate_write]
    readers = [r.layer for r in roles if r.uses_candidates]
    if cand >= 0 and writers != [cand]:
        raise V41TopologyError(
            f"candidate_source_layer_id={cand} is not an index source "
            f"(index sources {list(idx)}); nothing would write the candidate-block buffer "
            f"that layers {readers} read (attention.py:398-418)"
        )
    return roles


def is_v41_config(cfg) -> bool:
    model_type = getattr(cfg, "model_type", None)
    return isinstance(model_type, str) and model_type.startswith("deepseek_v41")


def validate_if_v41(cfg) -> list[LayerRole] | None:
    """Run the check for a V4.1 text config; return None for anything else."""
    if not is_v41_config(cfg):
        return None
    roles = validate_v41_topology(cfg)
    from vllm.logger import logger
    logger.info(
        "DeepSeek-V4.1 topology OK: %d layers, index-K owners %s, indexers %s, "
        "candidate writer %s", len(roles),
        [r.layer for r in roles if r.owns_index_k],
        [r.layer for r in roles if r.has_indexer],
        [r.layer for r in roles if r.candidate_write])
    return roles
