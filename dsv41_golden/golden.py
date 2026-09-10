"""CPU golden runner for DeepSeek-V4.1-Flash.

Runs the vendor's own reference implementation (verbatim copies in ``ref/``, with
only ``kernel.py`` swapped for the CPU one) against the real HF checkpoint, and
dumps per-layer activations for NPU comparison.

The whole design turns on one number: the checkpoint is 510 GB, of which 290 GB
is MoE experts and 203 GB is the two engram tables.  Nothing like that fits
alongside the vllm baseline on this box, so weights are never all resident:

* **Layers stream.**  Build layer i, fill it from the checkpoint, run it, free it.
  Peak is one layer.
* **Experts stream inside a layer.**  The gate runs first, so only the experts a
  token actually routes to are ever read -- for a short prompt that is a small
  fraction of the 384.
* **Engram tables are never materialized at all.**  A forward touches a few
  thousand of their 384 million rows, gathered straight out of the mmap.

Everything the layers hand each other (``shared_attn.compress_kv``, ``index_k``,
``topk_idxs``, ``candidates``) is a small cache tensor that outlives the layer
that produced it, so streaming does not break the cross-layer wiring.

Not covered: the vision tower and the DSpark/MTP draft head.  See REPORT.md.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import time
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
if REF not in sys.path:
    sys.path.insert(0, REF)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import model as M                     # noqa: E402  the vendor reference, CPU kernels
from ckpt import Checkpoint, dequant_fp8_2d   # noqa: E402
from engram import EngramLayout       # noqa: E402

MODEL_DIR = "/workspace/models/DeepSeek-V4.1-Flash"


# --------------------------------------------------------------------------- #
# memory reporting                                                            #
# --------------------------------------------------------------------------- #
def rss_gb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGESIZE") / 2**30


def sys_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    return float("nan")


class MemoryGuard:
    """Abort before we squeeze anything else on this box (the vllm baseline lives here).

    Checked at every layer boundary rather than relying on the OOM killer, which
    would be free to pick the wrong victim.
    """

    def __init__(self, floor_gb: float = 150.0):
        self.floor_gb = floor_gb
        self.peak_rss = 0.0
        self.min_avail = float("inf")

    def check(self, where: str):
        avail = sys_available_gb()
        rss = rss_gb()
        self.peak_rss = max(self.peak_rss, rss)
        self.min_avail = min(self.min_avail, avail)
        if avail < self.floor_gb:
            raise MemoryError(
                f"aborting at {where}: {avail:.1f} GB available, floor is {self.floor_gb:.0f} GB "
                f"(own RSS {rss:.1f} GB). Nothing was killed; reduce the prompt or layer count."
            )
        return rss, avail


# --------------------------------------------------------------------------- #
# config                                                                      #
# --------------------------------------------------------------------------- #
def load_args(max_batch_size=1, max_seq_len=512, **overrides) -> "M.ModelArgs":
    with open(os.path.join(MODEL_DIR, "inference", "config.json")) as f:
        cfg = json.load(f)
    known = {f.name for f in fields(M.ModelArgs)}
    kwargs = {k: v for k, v in cfg.items() if k in known}
    kwargs.update(max_batch_size=max_batch_size, max_seq_len=max_seq_len, temperature=0.0)
    kwargs.update(overrides)
    return M.ModelArgs(**kwargs)


# --------------------------------------------------------------------------- #
# streaming replacements for the two memory-hostile modules                   #
# --------------------------------------------------------------------------- #
class StreamingEngramEmbedding(nn.Module):
    """``ParallelEngramEmbedding`` without the 98 GB allocation.

    Identical arithmetic to the reference (dequantize the fp8 row with its e8m0
    per-32 scale, return bf16), but the rows are gathered from the mmap instead
    of from a resident table.
    """

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = M.fp8_block_size
        # world_size is 1, so this rank owns the whole table
        self.vocab_start_idx = 0
        self.vocab_end_idx = num_embeddings
        self.ckpt = None
        self.wname = self.sname = None
        self.rows_read = 0

    def bind(self, ckpt: Checkpoint, wname: str, sname: str):
        self.ckpt, self.wname, self.sname = ckpt, wname, sname
        _, wshape = ckpt.meta(wname)
        assert wshape[1] == self.dim, (wshape, self.dim)
        return self

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        assert self.ckpt is not None, "StreamingEngramEmbedding.bind() was never called"
        mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
        local = (indices - self.vocab_start_idx).masked_fill(mask, 0)

        flat, inverse = torch.unique(local.reshape(-1), return_inverse=True)
        self.rows_read += flat.numel()
        vals = self.ckpt.gather_rows(self.wname, flat)      # [u, dim]   fp8
        scales = self.ckpt.gather_rows(self.sname, flat)    # [u, dim/32] e8m0

        v = vals.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        v = v.flatten(-2).to(torch.bfloat16)                # [u, dim]
        values = v[inverse].view(*local.shape, self.dim)
        return values.masked_fill(mask.unsqueeze(-1), 0)


class StreamingMoE(nn.Module):
    """``model.MoE`` with the routed experts read on demand.

    The arithmetic below is a line-for-line copy of ``MoE.forward`` (world_size 1,
    so the all_reduce drops out); the only change is that ``self.experts[i]`` is
    replaced by a template module whose weights are re-pointed at expert ``i``
    just before it runs.  The gate decides which experts that is, so a short
    prompt reads far fewer than all 384.
    """

    def __init__(self, layer_id: int, args, ckpt: Checkpoint, prefix: str):
        super().__init__()
        n_routed, n_act = args.get_moe_config(layer_id)
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_routed_experts = n_routed
        self.n_activated_experts = n_act
        self.ckpt = ckpt
        self.prefix = prefix
        self.gate = M.Gate(layer_id, args)
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        # One template, re-pointed per expert -- avoids 384 x 18 MB of allocation.
        # Deliberately kept out of the module registry (object.__setattr__ bypasses
        # nn.Module.__setattr__), so it never shows up in state_dict and the weight
        # loader does not try to find "ffn._template.*" in the checkpoint.
        object.__setattr__(self, "_template",
                           M.Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype,
                                    swiglu_limit=args.swiglu_limit))
        self.shared_experts = M.Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        self.experts_loaded = 0
        self.expert_bytes = 0
        self.record_margin = None      # set by _attach_taps when dumping
        # Forced routing (ROUTING_INJECTION.md).  Both are set together by
        # GoldenModel.forward when a plan is supplied; see the note in forward().
        self.routing_override = None   # (layer_id, positions, weights, indices) -> (weights, indices)
        self.routing_positions = None  # [n_rows] absolute sequence position of each gate row
        self.record_forced = None      # set by _attach_taps when dumping

    def _bind_expert(self, i: int):
        p = f"{self.prefix}ffn.experts.{i}."
        for wn in ("w1", "w2", "w3"):
            lin = getattr(self._template, wn)
            w = self.ckpt.get(p + wn + ".weight")
            if lin.weight.dtype == torch.float4_e2m1fn_x2:
                w = w.view(torch.float4_e2m1fn_x2)
            lin.weight.data = w
            s = self.ckpt.get(p + wn + ".scale")
            lin.scale.data = s
            lin.weight.scale = lin.scale
            self.expert_bytes += self.ckpt.nbytes(p + wn + ".weight") + self.ckpt.nbytes(p + wn + ".scale")
        self.experts_loaded += 1
        return self._template

    def routing_margin(self, x: torch.Tensor, image_mask=None) -> torch.Tensor:
        """Per-token gap between the last selected expert and the first rejected one.

        DIAGNOSTIC ONLY -- nothing in the golden's numerical path reads this.  It
        recomputes the gate's *selection* logits (``scores + bias``, the quantity
        ``Gate.forward`` takes its top-k of) purely so a comparison run can tell a
        real arithmetic bug from a near-tie that any tiny perturbation would flip.

        A small margin means the token's routing is numerically fragile: two
        implementations that agree to 1e-7 can still pick different experts, and
        the resulting hidden state differs by O(1).  See ALIGNMENT_PROTOCOL.md.
        """
        g = self.gate
        scores = M.linear(x.float(), g.weight.float()) / g.gate_temp
        if g.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif g.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = g.bias
        if image_mask is not None and g.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), g.bias_vl, bias)
        sel = scores + bias
        top = sel.topk(min(g.topk + 1, sel.size(-1)), dim=-1).values
        return top[:, g.topk - 1] - top[:, g.topk]        # >= 0; 0 means an exact tie

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        if self.record_margin is not None:
            self.record_margin(self.routing_margin(x, None if image_mask is None else image_mask.flatten()))
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        if self.routing_override is not None:
            # Substitute the plan's expert choice for this layer's own top-k.  This
            # sits before the dispatch loop below and before any expert id is used,
            # which is the injection point the NPU side has to match: an override
            # applied after dispatch would not change which experts ran.
            assert self.routing_positions is not None, (
                "routing_override was set without routing_positions -- a row-index "
                "lookup is exactly the misalignment ROUTING_INJECTION.md forbids"
            )
            weights, indices = self.routing_override(
                self.layer_id, self.routing_positions, weights, indices
            )
            if self.record_forced is not None:
                self.record_forced(weights, indices)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self._bind_expert(i)
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)


class _NullMoE(nn.Module):
    """Placeholder so ``Block.__init__`` does not build 384 resident experts."""

    def __init__(self, layer_id, args):
        super().__init__()


@contextlib.contextmanager
def _lean_block_construction():
    """Swap the two allocation-hostile classes out for the duration of a Block build."""
    moe, eng = M.MoE, M.ParallelEngramEmbedding
    M.MoE, M.ParallelEngramEmbedding = _NullMoE, StreamingEngramEmbedding
    try:
        yield
    finally:
        M.MoE, M.ParallelEngramEmbedding = moe, eng


# --------------------------------------------------------------------------- #
# activation recording                                                        #
# --------------------------------------------------------------------------- #
class Recorder:
    """Collects tagged activations and writes one safetensors file per layer."""

    def __init__(self, out_dir: str | None, enabled: bool = True):
        self.out_dir = out_dir
        self.enabled = enabled and out_dir is not None
        self.buf: dict[str, torch.Tensor] = {}
        self.manifest: dict[str, dict] = {}
        if self.enabled:
            os.makedirs(out_dir, exist_ok=True)

    def add(self, name: str, t):
        if not self.enabled or t is None:
            return
        if isinstance(t, (tuple, list)):
            for k, sub in enumerate(t):
                self.add(f"{name}.{k}", sub)
            return
        if not isinstance(t, torch.Tensor):
            return
        t = t.detach()
        # safetensors cannot serialize fp8/fp4/complex; widen to something portable
        if t.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.float64):
            t = t.float()
        elif t.dtype == torch.float4_e2m1fn_x2:
            t = t.view(torch.uint8)
        elif t.is_complex():
            t = torch.view_as_real(t)
        self.buf[name] = t.contiguous().clone()

    def flush(self, tag: str):
        if not self.enabled or not self.buf:
            self.buf = {}
            return
        from safetensors.torch import save_file
        path = os.path.join(self.out_dir, f"{tag}.safetensors")
        save_file(self.buf, path)
        self.manifest[tag] = {
            k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
            for k, v in self.buf.items()
        }
        self.buf = {}

    def write_manifest(self, extra: dict):
        if not self.enabled:
            return
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump({"tensors": self.manifest, **extra}, f, indent=2)


def _attach_taps(block, rec: Recorder):
    """Hook every sublayer boundary worth comparing, plus the three hc helpers.

    Module hooks give the sublayer inputs/outputs without touching the reference
    code.  ``hc_pre``/``hc_post``/``hc_mixes`` are plain methods, so those get
    wrapped instead -- each runs twice per block (attention, then FFN), and the
    call index is part of the tag.
    """
    handles = []

    def hook(tag, want_input=False):
        def fn(_mod, inp, out):
            if want_input and inp:
                rec.add(f"{tag}.in", inp[0])
            rec.add(f"{tag}.out", out)
        return fn

    handles.append(block.attn_norm.register_forward_hook(hook("attn_norm")))
    handles.append(block.attn.register_forward_hook(hook("attn", want_input=True)))
    handles.append(block.ffn_norm.register_forward_hook(hook("ffn_norm")))
    handles.append(block.ffn.register_forward_hook(hook("moe", want_input=True)))
    handles.append(block.ffn.gate.register_forward_hook(hook("moe_gate")))
    block.ffn.record_margin = lambda t: rec.add("moe_routing_margin", t)

    def _forced(w, i):
        # Recorded AFTER the override returns, so it is evidence the experts really
        # ran on these ids -- not merely that a plan file was readable.  Absent when
        # routing is free, which is how a comparison tells the two phases apart.
        rec.add("moe_gate.forced.0", w)
        rec.add("moe_gate.forced.1", i)

    block.ffn.record_forced = _forced
    if block.engram is not None:
        handles.append(block.engram.register_forward_hook(hook("engram")))
    if block.attn.compressor is not None:
        handles.append(block.attn.compressor.register_forward_hook(hook("compressor")))
    if block.attn.indexer is not None:
        handles.append(block.attn.indexer.register_forward_hook(hook("indexer_topk")))

    counters = {"hc_mixes": 0, "hc_pre": 0, "hc_post": 0}
    for nm in list(counters):
        orig = getattr(block, nm)

        def wrapper(*a, _orig=orig, _nm=nm, **kw):
            out = _orig(*a, **kw)
            rec.add(f"{_nm}.{counters[_nm]}", out)
            counters[_nm] += 1
            return out

        setattr(block, nm, wrapper)
    return handles


# --------------------------------------------------------------------------- #
# the runner                                                                  #
# --------------------------------------------------------------------------- #
class GoldenModel:
    def __init__(self, args, ckpt: Checkpoint, tokenizer=None, guard: MemoryGuard | None = None,
                 verbose: bool = True):
        self.args = args
        self.ckpt = ckpt
        self.guard = guard or MemoryGuard()
        self.verbose = verbose
        self.stats = {"bytes_read": 0, "experts_loaded": 0, "engram_rows": 0, "layer_times": []}

        self.engram_layout = EngramLayout.from_args(args)
        self.engram_hash = (
            M.NgramHashState(args, self.engram_layout, tokenizer)
            if self.engram_layout is not None and tokenizer is not None else None
        )
        # small, always-resident tensors
        self.embed_weight = ckpt.get("embed.weight")            # [vocab, dim] bf16, 1.3 GB
        self.final_norm_weight = ckpt.get("norm.weight")
        self._head_weight = None                                # 1.3 GB, loaded on first use

    # -- weight loading ----------------------------------------------------
    def _assign(self, param: nn.Parameter, name: str):
        src = self.ckpt.get(name)
        self.stats["bytes_read"] += self.ckpt.nbytes(name)
        tgt = param.data
        if tgt.dtype == torch.float4_e2m1fn_x2:
            assert src.dtype == torch.int8, f"{name}: expected packed int8, got {src.dtype}"
            src = src.view(torch.float4_e2m1fn_x2)
        elif src.dtype != tgt.dtype:
            src = src.to(tgt.dtype)
        assert tuple(src.shape) == tuple(tgt.shape), (
            f"{name}: checkpoint {tuple(src.shape)} vs module {tuple(tgt.shape)}"
        )
        param.data = src

    def load_block(self, block: nn.Module, layer_id: int):
        prefix = f"layers.{layer_id}."
        sd = dict(block.state_dict(keep_vars=True))

        # wo_a is bf16 in the module but fp8 + a 32x32 block scale in the checkpoint;
        # convert.py dequantizes it at conversion time, so do the same here.
        wo_a = block.attn.wo_a.weight
        w = self.ckpt.get(prefix + "attn.wo_a.weight")
        s = self.ckpt.get(prefix + "attn.wo_a.scale")
        self.stats["bytes_read"] += (self.ckpt.nbytes(prefix + "attn.wo_a.weight")
                                     + self.ckpt.nbytes(prefix + "attn.wo_a.scale"))
        deq = dequant_fp8_2d(w, s, M.fp8_block_size)
        assert deq.shape == wo_a.shape, (deq.shape, wo_a.shape)
        wo_a.data = deq
        sd.pop("attn.wo_a.weight", None)
        del w, s, deq

        missing = []
        for key, param in sd.items():
            name = prefix + key
            if name in self.ckpt:
                self._assign(param, name)
            else:
                missing.append(name)
        if missing:
            raise KeyError(f"layer {layer_id}: not in checkpoint: {missing[:8]}")

        # the engram table is never resident; point the streaming lookup at the mmap
        if block.engram is not None:
            block.engram.embed.bind(
                self.ckpt, prefix + "engram.embed.weight", prefix + "engram.embed.scale"
            )

    def build_block(self, layer_id: int) -> nn.Module:
        with _lean_block_construction():
            block = M.Block(layer_id, self.args, self.engram_layout)
        block.ffn = StreamingMoE(layer_id, self.args, self.ckpt, f"layers.{layer_id}.")
        return block

    # -- forward -----------------------------------------------------------
    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, rec: Recorder | None = None,
                n_layers: int | None = None, full_logits: bool = True,
                start_pos: int = 0, routing_plan=None, layer_in_dir: str | None = None):
        """``layer_in_dir`` turns every block into an independent comparison.

        Forced routing removes the discrete verdict but not the *accumulation*: the
        residual stream is bf16, so a ~1e-7 arithmetic difference occasionally moves
        one token by one ULP, and causal attention then spreads that token into every
        later one.  Measured over 3 layers, that alone takes ``layer.out`` from 4.0e-4
        to 2.3e-2 with routing pinned -- past the protocol's own 2e-2 threshold, with
        no bug present.

        Pointing this at a golden dump overwrites ``layer.in``/``pre_mix.in`` at each
        block entry, so each layer is judged on the golden's input instead of on its
        own predecessors' output.  It does NOT isolate cross-layer state (KV cache,
        the compressor latents from layers 2/8/14/20, layer 20's candidate blocks),
        which still come from this run -- partial isolation, not total.
        """
        args = self.args
        rec = rec or Recorder(None, enabled=False)
        n_layers = args.n_layers if n_layers is None else n_layers

        # Absolute sequence position of every token in this call -- the key the
        # forced-routing plan is addressed by.  Deliberately *not* the row index:
        # see section 0 of ROUTING_INJECTION.md.
        positions = torch.arange(start_pos, start_pos + input_ids.size(1))
        if routing_plan is not None:
            assert input_ids.size(0) == 1, (
                f"a routing plan describes one sequence; got batch {input_ids.size(0)}. "
                "Use one plan per request."
            )
            if start_pos == 0:
                routing_plan.check_input_ids(input_ids)

        engram_hashes = None
        if self.engram_hash is not None:
            engram_hashes = self.engram_hash(input_ids, 0, None)

        h = F.embedding(input_ids, self.embed_weight)
        rec.add("embed.out", h)
        h = h.unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        pre_mix = M.make_identity_pre_mix(h, args.hc_mult)

        rec.add("input_ids", input_ids.int())
        if engram_hashes is not None:
            rec.add("engram_hash_ids", engram_hashes)
        rec.flush("prologue")

        last_block = None
        for i in range(n_layers):
            t0 = time.time()
            rss, avail = self.guard.check(f"before layer {i}")
            block = self.build_block(i)
            self.load_block(block, i)
            t_load = time.time() - t0

            if layer_in_dir is not None:
                from safetensors.torch import load_file
                ref = load_file(os.path.join(layer_in_dir, f"layer_{i:02d}.safetensors"))
                assert tuple(ref["layer.in"].shape) == tuple(h.shape), (
                    f"layer {i}: reference layer.in {tuple(ref['layer.in'].shape)} "
                    f"vs this run {tuple(h.shape)}"
                )
                h = ref["layer.in"].to(h.dtype).clone()
                pre_mix = ref["pre_mix.in"].to(pre_mix.dtype).clone()

            handles = _attach_taps(block, rec)
            if routing_plan is not None:
                block.ffn.routing_override = routing_plan.override
                block.ffn.routing_positions = positions
            rec.add("layer.in", h)
            rec.add("pre_mix.in", pre_mix)

            if block.engram is not None:
                idx = block.engram.layer_hash_index
                h = block.engram(h, engram_hashes[:, :, idx, :], None)

            h, pre_mix = block(h, 0, pre_mix, None)

            rec.add("layer.out", h)
            rec.add("pre_mix.out", pre_mix)
            rec.flush(f"layer_{i:02d}")
            for hd in handles:
                hd.remove()

            self.stats["experts_loaded"] += block.ffn.experts_loaded
            self.stats["bytes_read"] += block.ffn.expert_bytes
            if block.engram is not None:
                self.stats["engram_rows"] += block.engram.embed.rows_read
            dt = time.time() - t0
            self.stats["layer_times"].append(dt)
            if self.verbose:
                print(f"  layer {i:2d}  {dt:6.1f}s (load {t_load:5.1f}s)  "
                      f"experts {block.ffn.experts_loaded:3d}  "
                      f"rss {rss:5.1f}G  avail {avail:5.1f}G", flush=True)

            if i == n_layers - 1:
                last_block = block
                h = block.hc_pre(h, pre_mix)
            else:
                del block
                gc.collect()

        # final norm + head
        norm = M.RMSNorm(args.dim, args.norm_eps)
        norm.weight.data = self.final_norm_weight
        hn = norm(h)
        rec.add("final_norm.out", hn)
        if self._head_weight is None:
            self._head_weight = self.ckpt.get("head.weight")
        x = hn if full_logits else hn[:, -1]
        logits = F.linear(x.float(), self._head_weight.float())
        rec.add("logits", logits)
        rec.flush("epilogue")
        del last_block
        gc.collect()
        return logits
