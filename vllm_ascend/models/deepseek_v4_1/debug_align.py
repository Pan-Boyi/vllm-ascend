# SPDX-License-Identifier: Apache-2.0
"""Layer-by-layer alignment plumbing for DeepSeek-V4.1-Flash on Ascend.

NPU side of the CPU-golden alignment protocol, per
``worktrees/golden/dsv41_golden/ALIGNMENT_PROTOCOL.md`` and the injection
contract in ``ROUTING_INJECTION.md``.

Three debug facilities, all off unless their env var is set:

* :class:`AlignRecorder` writes ``layer_NN.safetensors`` dumps whose tensor
  names match the protocol byte for byte, so golden's ``compare_dumps.py``
  consumes them unmodified.
* :class:`RouteInjector` overrides the MoE top-k with golden's decision.
  ``topk(6)`` is a discrete verdict on continuous scores, so a near-tie
  flipped by rounding noise changes which experts run, and the O(1) output
  difference propagates. Measured under an fp32->fp64 accumulator change
  (golden, 27 tokens, *set* comparison): 0% of tokens re-routed at layer 0,
  0% at layer 1, 22.2% at layer 2.
* The same injector can also overwrite each block's *input* residual stream
  (``moe+layer_in``). Forcing routing alone is necessary but **not
  sufficient**: golden measured layer 2 ``layer.out`` at rel_l2 2.27e-2 even
  with routing forced, above the protocol's own 2e-2 bound. The dominant path
  is not operator noise but a single bf16 ULP spreading through causal
  attention -- one token lands on a rounding boundary, and attention pollutes
  every later token, saturating over layers. Overwriting ``layer.in`` per
  block stops the cross-layer accumulation and brings per-layer deviation back
  to layer-0 scale (~3.97e-4), which is where a strict tolerance becomes
  meaningful and where any layer can be diagnosed independently.

  Note this is *partial* isolation: KV cache, compressor latents (layers
  2/8/14/20), layer 20 candidate blocks and indexer selections still come from
  this implementation's own earlier layers.

Everything is addressed by **global sequence position**, never by row index
into the MoE input tensor. Those coincide only for golden's single-chunk
batch-1 prefill; here chunked prefill splits sequences, padding inserts dead
rows, and expert dispatch permutes tokens. A row-indexed implementation would
surface as "routing mismatch" -- exactly the phenomenon under test -- and
would pass every smoke case.

Protocol keys that are easy to forget
-------------------------------------
``indexer_kv_offset`` ``[1]`` int32
    ``indexer_topk.out`` does not hold compressed-position numbers; it holds
    indices into the concatenated ``[window ; compressed]`` KV, so every
    non-``-1`` value carries a base offset equal to the length of the sliding
    window tensor. That length is path dependent -- during prefill it is the
    sequence length, during decode the full ring size -- so two correct
    implementations disagree by a constant. Record the offset alongside the
    indices and compare ``where(raw < 0, raw, raw - offset)``; comparing raw
    values reports 100% mismatch on a correct implementation.

Env vars
--------
``VLLM_ASCEND_DSV41_DUMP_DIR``
    Where to write NPU dumps.
``VLLM_ASCEND_DSV41_GOLDEN_DIR``
    Golden's full dump dir (``golden_dumps/full40``). Needed for
    ``layer_in`` injection, and used as the routing source if no plan is set.
``VLLM_ASCEND_DSV41_ROUTING_PLAN``
    ``golden_dumps/plan_full40`` (or the ``.safetensors`` directly). 58 KB
    versus 271 MB for the full dump, so prefer it for routing injection.
``VLLM_ASCEND_DSV41_FORCE_ROUTE``
    ``off`` (default) | ``moe`` | ``moe+layer_in`` | ``moe+indexer``.
    Flags combine with ``+``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Set, Tuple

import torch

from vllm_ascend import envs

PROLOGUE = "prologue"
EPILOGUE = "epilogue"

# safetensors cannot store fp8/fp4; the protocol says widen to fp32 (exact).
_WIDEN_TO_FP32 = (torch.float8_e4m3fn, torch.float8_e5m2)

_KNOWN_FLAGS = {"moe", "layer_in", "indexer"}


def _env(name: str) -> Optional[str]:
    """Read a registered vllm-ascend env var.

    Names live in ``vllm_ascend/envs.py`` rather than being spelled out at each
    use site, so that the set of knobs is discoverable in one place.
    """
    v = str(getattr(envs, name, "") or "").strip()
    return v or None


def _to_storable(t: torch.Tensor) -> torch.Tensor:
    t = t.detach()
    if t.dtype in _WIDEN_TO_FP32:
        t = t.to(torch.float32)
    return t.cpu().contiguous()


#: Matches the tags _layer_tag produces, so the scan below cannot be fooled
#: by a non-layer tag that happens to start with "layer".
_LAYER_TAG_RE = re.compile(r"layer_\d+")


def _layer_tag(layer_id: int) -> str:
    return f"layer_{layer_id:02d}"


def _log_coverage(report: dict, dump_dir: str) -> None:
    """Put the verdict in the log, in the words the reader needs.

    ⚠️ "not checked" is logged at warning level and says so in those words.
    The failure this prevents is the cheapest one in this whole area: a dump
    produced with no golden directory, a manifest whose differences are both
    empty, and a reader concluding the run recorded everything.
    """
    from vllm.logger import logger

    state = report.get("state")
    if state == "not_checked":
        logger.warning(
            "DSV4.1 alignment: dump written to %s with %d tensors, and "
            "whether that is everything golden compares was NOT CHECKED -- %s",
            dump_dir, report.get("recorded_count", 0), report.get("reason"))
        return
    if state == "ok":
        logger.info(
            "DSV4.1 alignment: dump written to %s -- all %d keys golden "
            "declares are present (%d extra recorded).", dump_dir,
            report.get("expected_count", 0), len(report.get("extra", ())))
        return
    missing = report.get("missing", [])
    logger.error(
        "DSV4.1 alignment: dump written to %s is INCOMPLETE -- %d of the %d "
        "keys golden declares were never recorded, so those comparisons "
        "cannot run and their absence looks exactly like a comparison that "
        "passed. Missing: %s%s", dump_dir, len(missing),
        report.get("expected_count", 0), ", ".join(missing[:12]),
        " ..." if len(missing) > 12 else "")


class AlignRecorder:
    """Collects per-layer tensors and writes protocol-shaped safetensors.

    Tensors arrive one forward chunk at a time, holding whichever rows the
    scheduler batched. ``row_to_pos`` maps each row to its global sequence
    position (``-1`` for padding), and rows are scattered into a
    ``[1, seq_len, ...]`` buffer. At flush every position of every recorded key
    must have been written exactly once; that assertion is the guard rail
    against the misalignment described in the module docstring.
    """

    def __init__(self, dump_dir: str, seq_len: Optional[int] = None):
        self.dump_dir = dump_dir
        self.seq_len = seq_len
        self.enabled = True
        self._bufs: Dict[str, Dict[str, torch.Tensor]] = {}
        self._coverage: Dict[str, Dict[str, torch.Tensor]] = {}
        self._meta: Dict[str, object] = {}
        os.makedirs(dump_dir, exist_ok=True)

    # -- recording ---------------------------------------------------------

    def set_seq_len(self, seq_len: int) -> None:
        if self.seq_len is not None and self.seq_len != seq_len:
            raise ValueError(
                f"seq_len already fixed at {self.seq_len}, got {seq_len}; an "
                "alignment run must be a single sequence")
        self.seq_len = seq_len

    def record_whole(self, tag: str, key: str, tensor: torch.Tensor) -> None:
        """Record a tensor that is already complete (not position-sharded)."""
        self._bufs.setdefault(tag, {})[key] = _to_storable(tensor)

    def record(self, tag: str, key: str, tensor: torch.Tensor,
               row_to_pos: torch.Tensor) -> None:
        """Scatter ``tensor``'s rows into the ``[1, seq_len, ...]`` buffer.

        ``tensor`` is ``[num_rows, *feat]``; ``row_to_pos`` is ``[num_rows]``
        int64 with ``-1`` marking padding rows, which are dropped.
        """
        if self.seq_len is None:
            raise RuntimeError(
                "set_seq_len() must be called before record(), normally from "
                "the prologue hook")
        if row_to_pos.numel() != tensor.shape[0]:
            raise ValueError(
                f"{tag}/{key}: row_to_pos has {row_to_pos.numel()} entries but "
                f"tensor has {tensor.shape[0]} rows")

        src = _to_storable(tensor)
        pos = row_to_pos.detach().cpu().to(torch.int64)
        valid = pos >= 0
        pos_v = pos[valid]

        keys = self._bufs.setdefault(tag, {})
        cov = self._coverage.setdefault(tag, {})
        if key not in keys:
            keys[key] = torch.zeros((self.seq_len, ) + tuple(src.shape[1:]),
                                    dtype=src.dtype)
            cov[key] = torch.zeros(self.seq_len, dtype=torch.int32)
        keys[key][pos_v] = src[valid]
        cov[key].index_add_(0, pos_v,
                            torch.ones_like(pos_v, dtype=torch.int32))

    def record_layer(self, layer_id: int, key: str, tensor: torch.Tensor,
                     row_to_pos: torch.Tensor) -> None:
        self.record(_layer_tag(layer_id), key, tensor, row_to_pos)

    # -- flushing ----------------------------------------------------------

    def add_manifest(self, **kw) -> None:
        self._meta.update(kw)

    def flush(self, strict: bool = True) -> None:
        """Write every tag, then judge what was written.

        🔴 The order used to be the other way round, and it cost a flight. The
        coverage assertion ran first and raised, so **nothing reached disk** --
        a dump that says "I am incomplete, here is how" is far more useful
        than no dump, and the run that would have produced it is a server
        start and a warm-up away.

        Same shape as the trace's sentinel refusal and the golden-coverage
        report: the artifact is written, and then the artifact declares what
        is wrong with it. A report must not destroy the thing it reports on.
        """
        from safetensors.torch import save_file

        for tag, keys in self._bufs.items():
            scattered = self._coverage.get(tag, {})
            out = {}
            for key, t in keys.items():
                # Protocol shapes carry a batch axis; golden is always B=1.
                # moe_gate.* and moe_routing_margin are the exceptions -- they
                # are stored without it, so only re-add it where we scattered
                # by position and the protocol asks for it.
                out[key] = t.unsqueeze(0) if (key in scattered
                                              and not _is_flat_key(key)) else t
            save_file(out, os.path.join(self.dump_dir, f"{tag}.safetensors"))

        manifest = dict(self._meta)
        manifest.setdefault("seq_len", self.seq_len)
        manifest.setdefault("tags", sorted(self._bufs))
        manifest["tensors"] = {tag: sorted(keys)
                               for tag, keys in self._bufs.items()}
        manifest["golden_coverage"] = self._golden_coverage()
        manifest["injection_scope"] = self._injection_scope()
        self._report_missing_prologue()
        problems = self._coverage_problems()
        manifest["position_coverage"] = {
            "state": "ok" if not problems else "incomplete",
            "problems": problems,
            "note": ("Every position of every recorded key must be written "
                     "exactly once. Anything else means the row->position "
                     "mapping is wrong, and the symptom downstream looks like "
                     "a routing mismatch rather than a mapping bug."),
        }
        with open(os.path.join(self.dump_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        if problems and strict:
            raise AssertionError(
                "position coverage is not exactly 1 for every token -- the "
                "row->position mapping is wrong, and the symptom would "
                "otherwise look like a routing mismatch. The tensors and a "
                f"manifest saying so were written to {self.dump_dir} first, "
                "so this dump is readable and self-describing rather than "
                "absent:\n" + "\n".join(problems))

    #: Keys an injector can replace, in the order it should prefer them. Layer
    #: 0 is the exception the whole structure exists for: it receives a 2-D
    #: embedding and builds the 4-stream tensor itself, so ``layer.in`` is not
    #: a tensor it was handed and ``embed.out`` is.
    INJECTABLE_KEYS = ("layer.in", "pre_mix.in", "embed.out")

    def _injection_scope(self) -> dict:
        """Per layer: what an injector can replace here, and what it cannot.

        🔴 Per layer, not one shared statement. What is injectable differs by
        layer -- layer 0 has no ``layer.in`` and no ``pre_mix`` at all -- so
        any single sentence about "the injected keys" is already false on
        layer 0. That is also why this is written by the side that produced
        the tensors: a reader that composes the sentence itself has to guess,
        and its guess is wrong on the layer that differs.

        Derived from what was actually recorded, never from a list of what
        was meant to be. A key that failed to record would otherwise still be
        claimed as covered, which is the failure this file already had once.

        ⚠️ ``not_isolated`` is the load-bearing half. Injection replaces a
        layer's inputs and nothing else; the KV cache, the compressor latents
        and the candidate blocks still come from this run. An error that
        travels through those is invisible to injection, and the report for
        that layer reads "input the same, output the same" -- which is what
        "no defect here" looks like.
        """
        per_layer: dict[str, dict] = {}
        for tag, keys in sorted(self._bufs.items()):
            if not _LAYER_TAG_RE.fullmatch(tag):
                continue
            index = int(tag.split("_")[1])
            injectable = [k for k in self.INJECTABLE_KEYS if k in keys]
            if index == 0 and "embed.out" not in injectable:
                # Recorded in the prologue tag, not in layer_00's.
                if "embed.out" in self._bufs.get("prologue", {}):
                    injectable = ["embed.out (prologue tag)"] + injectable
            per_layer[str(index)] = {
                "injectable_keys": injectable,
                "absent_here": [k for k in self.INJECTABLE_KEYS
                                if k not in keys and not (
                                    index == 0 and k == "embed.out")],
                "not_isolated": self._not_isolated(index),
            }
        return {
            "per_layer": per_layer,
            "source": "vllm_ascend AlignRecorder.flush",
            "note": ("PARTIAL isolation. Injection replaces the keys listed "
                     "per layer and nothing else; everything under "
                     "not_isolated still comes from this run, so an error "
                     "travelling through it is invisible here and the layer "
                     "reads as clean."),
        }

    def _not_isolated(self, index: int) -> list:
        """What still comes from this run at this layer.

        ⚠️ Three states on the compressor layers. Which layers produce the
        compressed latents is a property of the checkpoint, so it is reported
        only when the caller supplied it through ``add_manifest``; otherwise
        this says it does not know rather than naming a set from memory. A
        frozen 2/8/14/20 would be right for today's config and silently wrong
        for the next one.
        """
        items = ["KV cache (this run's own)"]
        sources = self._meta.get("compressor_source_layers")
        if sources is None:
            items.append(
                "compressor latents (which layers produce them was not "
                "recorded in this dump -- pass compressor_source_layers to "
                "add_manifest; NOT a claim that there are none)")
        else:
            items.append(
                "compressor latents produced by layers "
                + ", ".join(str(x) for x in sources))
        candidates = self._meta.get("candidate_block_layers")
        if candidates is None:
            items.append("candidate blocks (producing layers not recorded)")
        elif index in set(candidates):
            items.append("this layer's candidate blocks")
        return items

    def _golden_coverage(self) -> dict:
        """Say whether this dump holds everything golden will compare.

        🔴 Runs **after** the tensors are on disk and never raises. A dump run
        costs a launch; throwing it away at the last step because the report
        is bad would destroy the only copy of the thing the report is about.
        So the verdict is written into the manifest and logged, and the bytes
        are kept whatever it says.

        The coverage assertion above it is about *positions within a key*.
        This is about *which keys exist at all*, and the two cannot substitute
        for each other: a key that was never recorded has no coverage vector,
        so it passes the first check by not being looked at.
        """
        from vllm_ascend.models.deepseek_v4_1.align_taps import coverage_report

        recorded = {f"{tag}/{key}"
                    for tag, keys in self._bufs.items() for key in keys}
        report = coverage_report(recorded, _env("VLLM_ASCEND_DSV41_GOLDEN_DIR")
                                 or None)
        _log_coverage(report, self.dump_dir)
        return report

    def _report_missing_prologue(self) -> None:
        """Say when this dump has no prologue, because nothing else will.

        🔴 The prologue is the only observation point before layer 0. A dump
        without it cannot answer whether the embedding is right, and its
        absence is indistinguishable from "there was nothing to record" --
        which is how this section stayed at zero coverage for a day while
        every flight reported success.

        Not an error: a pipeline rank that does not hold the embedding has
        no prologue and is behaving correctly. What must not happen is that
        it passes unremarked.
        """
        if "prologue" in self._bufs and self._bufs["prologue"]:
            return
        from vllm.logger import logger

        logger.error(
            "DSV4.1 alignment: %s has no prologue section. input_ids, "
            "positions and embed.out are not in it, so nothing in this dump "
            "speaks about the model before layer 0 -- and the earliest known "
            "divergence is *at* layer 0. This is expected on a pipeline rank "
            "that does not hold the embedding and is a defect anywhere else; "
            "either way it is not 'there was nothing to record'.",
            self.dump_dir)

    def _coverage_problems(self) -> list:
        """Keys whose positions were not written exactly once."""
        problems = []
        for tag, keys in sorted(self._coverage.items()):
            for key, cov in sorted(keys.items()):
                if not torch.all(cov == 1):
                    missing = (cov == 0).nonzero().flatten().tolist()
                    dupes = (cov > 1).nonzero().flatten().tolist()
                    problems.append(
                        f"  {tag}/{key}: seq_len={cov.numel()} "
                        f"missing={missing[:8]} duplicated={dupes[:8]}")
        return problems


def build_row_to_pos(positions: torch.Tensor, *, num_actual_tokens: int,
                     num_reqs: int) -> torch.Tensor:
    """The ``row_to_pos`` that :meth:`AlignRecorder.record` wants.

    The right quantity is ``positions`` -- not ``cu_seqlens_q`` (boundaries)
    and not ``seq_lens`` (per-request totals). But it cannot be handed over as
    it arrives, for a reason that is invisible in the tensor:

        model_runner_v1.py:1523 / 1530  write self.positions[:total_num_scheduled_tokens]
        model_runner_v1.py:3889         read  self.positions[:num_tokens_padded]

    and no line anywhere clears the gap between them (the whole file writes
    ``self.positions[...]`` at four places, none of which touch the tail). So
    the padding rows carry *last step's* positions: in range, not 0, not -1.
    Passed through, those rows scatter onto real positions and overwrite真值.

    The recorder's flush-time coverage assertion does catch it -- a position
    written twice -- but only after the run, and these runs cost a launch. A
    guard being correct is not a reason to let the defect travel that far.

    Two facts are asserted rather than assumed:

    * **Single request.** ``positions`` is per-request-relative, so two
      requests both starting at 0 scatter onto each other. ``set_seq_len()``
      would eventually raise on a second sequence length, but only if they
      differ; the cheap, exact check is here.
    * **Row count.** ``num_actual_tokens`` must not exceed the rows, or the
      slice that is supposed to be padding is real data.
    """
    if num_reqs != 1:
        raise ValueError(
            f"alignment recording needs a single request, got {num_reqs}. "
            "positions are per-request, so two requests both start at 0 and "
            "their rows scatter onto each other -- silently, since every "
            "value stays in range.")
    total = int(positions.shape[-1]) if positions.ndim else 0
    if not 0 <= num_actual_tokens <= total:
        raise ValueError(
            f"num_actual_tokens {num_actual_tokens} is outside the {total} "
            "rows of positions; the tail this marks as padding would then "
            "cover real tokens.")
    row_to_pos = positions.detach().reshape(-1).clone().to(torch.int64)
    row_to_pos[num_actual_tokens:] = -1
    return row_to_pos


def current_row_to_pos(positions: torch.Tensor) -> Optional[torch.Tensor]:
    """``row_to_pos`` for this forward, or ``None`` when it must not record.

    ``None`` is a third state and not a failure: a profile or dummy run has no
    metadata dict, and during ACL graph capture ``positions`` is a placeholder
    (measured: every entry 127), which would map every row to one position and
    trip the coverage assertion with nothing wrong. Recording in either case
    produces a dump that looks real.
    """
    if torch.cuda.is_current_stream_capturing():
        return None

    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return None
    attn_metadata = getattr(context, "attn_metadata", None)
    if not isinstance(attn_metadata, dict) or not attn_metadata:
        # Profile / dummy run: no caches bound, nothing worth recording.
        return None

    for metadata in attn_metadata.values():
        num_actual_tokens = getattr(metadata, "num_actual_tokens", None)
        req = getattr(metadata, "req_metadata", None)
        starts = getattr(req, "query_start_loc", None)
        if num_actual_tokens is None or starts is None:
            continue
        return build_row_to_pos(positions,
                                num_actual_tokens=int(num_actual_tokens),
                                num_reqs=int(starts.numel()) - 1)

    raise ValueError(
        "no attention metadata in this forward carries both "
        "num_actual_tokens and a query_start_loc, so the padding boundary is "
        "unknown. Recording anyway would scatter stale positions over real "
        f"rows. Metadata seen: {sorted(attn_metadata)[:4]}")


def _is_flat_key(key: str) -> bool:
    """True for protocol keys stored as ``[S, ...]`` with no batch axis."""
    return key.startswith("moe_gate.") or key == "moe_routing_margin"


class RouteInjector:
    """Injects golden's MoE routing, and optionally its block-input residual.

    Both expert ids and routing weights are injected. Injecting ids alone
    would be wrong: ``norm_topk_prob=true`` renormalises over the selected set,
    so a different selection changes the denominator and the numbers stay
    incomparable. Golden's dumped weights are already normalised and already
    scaled by ``routed_scaling_factor`` (1.5), so they are substituted
    wholesale -- no renormalisation, no rescaling here.

    Expert ids are in the **global** id space (``routing_plan.json`` says
    ``expert_id_space: "global"``), so injection must happen before any
    global->local mapping for expert parallelism. TP=1 makes that a no-op
    today; the structure is kept so it need not be retrofitted.
    """

    def __init__(self, mode: str, golden_dir: Optional[str],
                 plan_path: Optional[str]):
        self.mode = mode
        self.flags = _parse_mode(mode)
        self.moe = "moe" in self.flags
        self.layer_in = "layer_in" in self.flags
        # Staged behind MoE: golden measures indexer top-k as fully stable on
        # the 27-token case, but only because K collapses to
        # ``min(index_topk, end_pos // ratio)`` = 13 there -- "13 chosen out of
        # 13", i.e. no freedom at all. On long sequences K approaches 512 with
        # far more candidates and it becomes a real discrete verdict again.
        self.indexer = "indexer" in self.flags
        self.enabled = bool(self.flags)

        self.golden_dir = golden_dir
        self.plan_path = plan_path
        self.plan_meta: Dict[str, object] = {}
        self._plan: Optional[Dict[str, torch.Tensor]] = None
        self._layer_cache: Dict[int, Dict[str, torch.Tensor]] = {}

        if self.layer_in and not golden_dir:
            raise ValueError(
                "VLLM_ASCEND_DSV41_FORCE_ROUTE includes 'layer_in' but "
                "VLLM_ASCEND_DSV41_GOLDEN_DIR is not set; layer.in / pre_mix.in come "
                "from the full dump, not the routing plan")
        if self.moe and not (plan_path or golden_dir):
            raise ValueError(
                "VLLM_ASCEND_DSV41_FORCE_ROUTE includes 'moe' but neither "
                "VLLM_ASCEND_DSV41_ROUTING_PLAN nor VLLM_ASCEND_DSV41_GOLDEN_DIR is set")

    # -- sources -----------------------------------------------------------

    def _load_plan(self) -> Optional[Dict[str, torch.Tensor]]:
        """Load the compact routing plan, if one was configured."""
        if self._plan is not None or not self.plan_path:
            return self._plan
        from safetensors.torch import load_file

        path = self.plan_path
        if os.path.isdir(path):
            meta_path = os.path.join(path, "routing_plan.json")
            path = os.path.join(path, "routing_plan.safetensors")
        else:
            meta_path = os.path.splitext(path)[0] + ".json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"routing plan not found: {path}")

        self._plan = load_file(path)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.plan_meta = json.load(f)
        return self._plan

    def _load_layer(self, layer_id: int) -> Dict[str, torch.Tensor]:
        if layer_id not in self._layer_cache:
            from safetensors.torch import load_file

            path = os.path.join(self.golden_dir or "",
                                f"{_layer_tag(layer_id)}.safetensors")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"injection needs golden dump {path}; point "
                    "VLLM_ASCEND_DSV41_GOLDEN_DIR at the directory holding "
                    "layer_NN.safetensors")
            self._layer_cache[layer_id] = load_file(path)
        return self._layer_cache[layer_id]

    def _routing_for(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        plan = self._load_plan()
        tag = _layer_tag(layer_id)
        if plan is not None:
            ids = plan[f"{tag}.indices"]
            weights = plan[f"{tag}.weights"]
        else:
            d = self._load_layer(layer_id)
            ids = d["moe_gate.out.1"]
            weights = d["moe_gate.out.0"]
        return ids.to(torch.int64), weights.to(torch.float32)

    # -- guards ------------------------------------------------------------

    def check_input_ids(self, input_ids: torch.Tensor) -> None:
        """Fail loudly if this run is not golden's fixed 27-token prompt.

        A routing plan is only valid for the tokens it was recorded on, and
        feeding different text would show up as wholesale routing mismatch.
        """
        plan = self._load_plan()
        ref = None
        if plan is not None and "input_ids" in plan:
            ref = plan["input_ids"]
        elif self.golden_dir:
            from safetensors.torch import load_file
            p = os.path.join(self.golden_dir, f"{PROLOGUE}.safetensors")
            if os.path.exists(p):
                ref = load_file(p)["input_ids"].flatten()
        if ref is None:
            return
        got = input_ids.detach().cpu().flatten().to(torch.int64)
        ref = ref.cpu().flatten().to(torch.int64)
        if got.numel() != ref.numel() or not torch.equal(got, ref):
            raise AssertionError(
                "input_ids do not match the golden routing plan; feed the "
                f"recorded token ids directly (expected {ref.numel()} tokens "
                f"starting {ref[:6].tolist()}, got {got.numel()} starting "
                f"{got[:6].tolist()})")

    def accum_dtype(self) -> Optional[str]:
        """Accumulator precision the plan was recorded with.

        Plans from different accumulator dtypes are not interchangeable --
        routing itself differs by ~22% of tokens at layer 2, so using the
        wrong one injects noise rather than removing it.
        """
        self._load_plan()
        return self.plan_meta.get("accum_dtype")

    # -- injection ---------------------------------------------------------

    def inject_moe(
        self,
        layer_id: int,
        row_to_pos: torch.Tensor,
        own_ids: torch.Tensor,
        own_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(ids, weights)`` with golden's values on non-padding rows.

        Padding rows (``row_to_pos < 0``) get zero routing weight so dispatch
        is deterministic; their ids are left alone since they select nothing.
        """
        if not self.moe:
            return own_ids, own_weights

        g_ids, g_w = self._routing_for(layer_id)
        pos = row_to_pos.detach().to(torch.int64)
        valid = pos >= 0
        safe_pos = torch.where(valid, pos, torch.zeros_like(pos))

        sel_ids = g_ids.to(own_ids.device)[safe_pos]
        sel_w = g_w.to(own_weights.device)[safe_pos]
        if tuple(sel_ids.shape[1:]) != tuple(own_ids.shape[1:]):
            raise ValueError(
                f"layer {layer_id}: golden top-k is {tuple(sel_ids.shape[1:])} "
                f"but the model produced {tuple(own_ids.shape[1:])}")

        m = valid.view(-1, *([1] * (own_ids.dim() - 1)))
        ids = torch.where(m, sel_ids.to(own_ids.dtype), own_ids)
        weights = torch.where(m, sel_w.to(own_weights.dtype),
                              torch.zeros_like(own_weights))
        return ids, weights

    def inject_layer_in(
        self,
        layer_id: int,
        row_to_pos: torch.Tensor,
        own_layer_in: torch.Tensor,
        own_pre_mix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Overwrite a block's input residual stream with golden's.

        ``own_layer_in`` is ``[num_rows, hc_mult, hidden]`` and
        ``own_pre_mix`` is ``[num_rows, hc_mult]``.
        """
        if not self.layer_in:
            return own_layer_in, own_pre_mix

        d = self._load_layer(layer_id)
        g_in = d["layer.in"].squeeze(0)      # [S, 4, 5120]
        g_pre = d["pre_mix.in"].squeeze(0)   # [S, 4]

        pos = row_to_pos.detach().to(torch.int64)
        valid = pos >= 0
        safe_pos = torch.where(valid, pos, torch.zeros_like(pos))

        sel_in = g_in.to(own_layer_in.device)[safe_pos].to(own_layer_in.dtype)
        sel_pre = g_pre.to(own_pre_mix.device)[safe_pos].to(own_pre_mix.dtype)

        m_in = valid.view(-1, *([1] * (own_layer_in.dim() - 1)))
        m_pre = valid.view(-1, *([1] * (own_pre_mix.dim() - 1)))
        return (torch.where(m_in, sel_in, own_layer_in),
                torch.where(m_pre, sel_pre, own_pre_mix))


def _parse_mode(mode: str) -> Set[str]:
    mode = (mode or "off").strip()
    if mode == "off":
        return set()
    flags = {f for f in mode.split("+") if f}
    unknown = flags - _KNOWN_FLAGS
    if unknown:
        raise ValueError(
            f"VLLM_ASCEND_DSV41_FORCE_ROUTE has unknown flag(s) {sorted(unknown)}; "
            f"known flags are {sorted(_KNOWN_FLAGS)}, combined with '+' "
            "(e.g. 'moe+layer_in'), or 'off'")
    if "moe" not in flags:
        raise ValueError(
            f"VLLM_ASCEND_DSV41_FORCE_ROUTE={mode!r}: 'moe' is required whenever "
            "injection is on; the other flags refine it")
    return flags


# -- singletons ------------------------------------------------------------

_recorder: Optional[AlignRecorder] = None
_injector: Optional[RouteInjector] = None
_initialised = False


def _init() -> None:
    global _recorder, _injector, _initialised
    if _initialised:
        return
    _initialised = True

    dump_dir = _env("VLLM_ASCEND_DSV41_DUMP_DIR")
    if dump_dir:
        _recorder = _new_recorder(dump_dir, 0)

    mode = _env("VLLM_ASCEND_DSV41_FORCE_ROUTE") or "off"
    if mode != "off":
        _injector = RouteInjector(mode, _env("VLLM_ASCEND_DSV41_GOLDEN_DIR"),
                                  _env("VLLM_ASCEND_DSV41_ROUTING_PLAN"))

    if _recorder is not None:
        # Recorded so a dump can never be read as "injection was on" when it
        # was not. Without this, a typo'd env var looks exactly like a real
        # operator bug: strict tolerance fails across every layer.
        _recorder.add_manifest(
            force_route=mode,
            routing_plan=(_injector.plan_path if _injector else None),
            routing_plan_accum_dtype=(_injector.accum_dtype()
                                      if _injector else None),
        )


#: How many prefills to record, each into its own directory. Default 1 keeps
#: today's single dump.
#:
#: 🔴 Two is the number this exists for. The strongest available comparison
#: is not "our dump against golden" -- it is **our dump against our own**:
#: the same server, the same weights, the same prompt, one request before a
#: different one has run and one after. A correct forward makes those two
#: byte-identical, and they are already known to produce different text, so
#: the first tensor that differs is the entry point for cross-request state,
#: located to a layer and a key. That comparison needs no reference, no
#: tolerance and no discussion of precision, and it cannot come back empty.
def _dump_request_limit() -> int:
    raw = _env("VLLM_ASCEND_DSV41_DUMP_REQUESTS") or "1"
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            f"VLLM_ASCEND_DSV41_DUMP_REQUESTS={raw!r} is not a number. "
            "Refusing rather than falling back to 1: a typo that silently "
            "records one dump makes the two-dump comparison read as 'the "
            "second request was never served'.") from None
    if value < 1:
        raise ValueError(
            f"VLLM_ASCEND_DSV41_DUMP_REQUESTS={value} would record nothing.")
    return value


def _writer_identity() -> dict:
    """Who is writing this dump: rank first, pid always.

    🔴 Four data-parallel ranks run this code in four processes and wrote to
    the same ``base/req_000``. Each one made the same complete-sounding
    statement -- "I am dump 0 of 3" -- into the same file, and the last
    writer won. The artifacts that survived were well formed, internally
    consistent and belonged to two different ranks, and that went unnoticed
    for a day.

    ⚠️ ``pid`` is recorded even when the rank is known, and the path falls
    back to it when the rank is not. A path that can collide is the whole
    defect; "the rank could not be determined" must not reintroduce it.
    """
    identity = {"pid": os.getpid(), "dp_rank": None, "dp_rank_source": None}
    for source, getter in (
            ("get_dp_group", _dp_rank_from_group),
            ("parallel_config", _dp_rank_from_config),
            ("VLLM_DP_RANK", lambda: _env_int("VLLM_DP_RANK"))):
        try:
            rank = getter()
        except Exception:                                    # noqa: BLE001
            rank = None
        if rank is not None:
            identity["dp_rank"] = int(rank)
            identity["dp_rank_source"] = source
            break
    return identity


def _dp_rank_from_group():
    from vllm.distributed.parallel_state import get_dp_group

    return get_dp_group().rank_in_group


def _dp_rank_from_config():
    from vllm.config import get_current_vllm_config

    return get_current_vllm_config().parallel_config.data_parallel_rank


def _env_int(name: str):
    raw = os.environ.get(name)
    return None if raw is None or not raw.strip() else int(raw)


def _writer_tag(identity: dict) -> str:
    rank = identity.get("dp_rank")
    if rank is None:
        return f"pid_{identity['pid']}"
    return f"rank_{rank:02d}"


def _dump_dir_for(base: str, index: int) -> str:
    """One directory per recorded prefill, under the writer that wrote it.

    Numbered rather than named after the request id: the comparison is
    positional ("the first apple" against "the third"), and a directory named
    after an id that the log also has to be joined against is one more step
    between the reader and the answer.

    🔴 But the number alone is not an address. Four ranks each produce a
    "req_000", so the writer goes in the path -- and it goes in even for a
    single dump, where the old layout collided just as silently.
    """
    return os.path.join(base, _writer_tag(_writer_identity()),
                        f"req_{index:03d}")


def _new_recorder(base: str, index: int) -> AlignRecorder:
    """A recorder that says, in its own manifest, which dump it is.

    🔴 The directory name is a *convention*, and conventions are lost in a
    copy, an archive or a rename. A manifest is the artifact's own
    declaration. A dump that does not say what it is can only be identified
    by inference from outside, and inference from outside is what went wrong
    the last three times something was read as the thing next to it.

    ``dump_requests`` is here for the same reason: "I am dump 0 of 3" is a
    complete statement, "I am req_000 of some run that recorded an unknown
    number" is not.
    """
    identity = _writer_identity()
    directory = _dump_dir_for(base, index)
    _refuse_to_overwrite(directory)
    recorder = AlignRecorder(directory)
    recorder.add_manifest(request_index=index,
                          dump_requests=_dump_request_limit(),
                          **identity)
    return recorder


def _refuse_to_overwrite(directory: str) -> None:
    """A second writer for one directory is a defect, not a retry.

    🔴 Silently overwriting produced artifacts that were well formed,
    internally consistent, and belonged to a different writer than the reader
    believed -- and nothing about them looked wrong. Raising here costs a
    launch; the alternative has already cost a day.
    """
    marker = os.path.join(directory, "manifest.json")
    if not os.path.exists(marker):
        return
    raise RuntimeError(
        f"{directory} already holds a dump. Something else has written it -- "
        "another data-parallel rank sharing this path, or a re-run into the "
        "same directory. Refusing rather than overwriting: the result would "
        "be well formed, internally consistent, and attributed to the wrong "
        "writer, which is exactly what went unnoticed for a day.")


#: How many prefills have been flushed so far this process.
_DUMPS_DONE = 0


def rotate_recorder() -> bool:
    """Flush the current dump and open the next one, if any are left.

    Returns whether recording continues. Called at the end of a prefill
    forward, which is where "this request's tensors are all in" is true and
    nothing later can still scatter into the buffers.

    ⚠️ One dump per *prefill forward*, not per request. Under chunked prefill
    a single request spans several forwards and would be split across
    directories. That is stated rather than guarded against: at the lengths
    this is used for there is one chunk, and a guard that cannot be exercised
    is worse than a sentence that can be read.
    """
    global _recorder, _DUMPS_DONE

    if _recorder is None:
        return False
    base = _env("VLLM_ASCEND_DSV41_DUMP_DIR")
    limit = _dump_request_limit()
    _recorder.flush()
    _DUMPS_DONE += 1

    from vllm.logger import logger

    if _DUMPS_DONE >= limit:
        logger.info(
            "DSV4.1 alignment: %d of %d requested dumps written; recording "
            "stops here. A later request will NOT overwrite them.",
            _DUMPS_DONE, limit)
        _recorder = None
        return False
    _recorder = _new_recorder(base, _DUMPS_DONE)
    logger.info(
        "DSV4.1 alignment: dump %d of %d written; next request records into "
        "%s.", _DUMPS_DONE, limit, _recorder.dump_dir)
    return True


def reset_for_test() -> None:
    global _recorder, _injector, _initialised, _DUMPS_DONE
    _recorder = None
    _injector = None
    _initialised = False
    _DUMPS_DONE = 0


def get_recorder() -> Optional[AlignRecorder]:
    _init()
    return _recorder


def get_injector() -> Optional[RouteInjector]:
    _init()
    return _injector


def alignment_active() -> bool:
    """True when either dumping or injection is on.

    Call sites use this to skip building ``row_to_pos`` on the production path.
    """
    _init()
    return _recorder is not None or _injector is not None
