"""Forced MoE routing: the golden's expert choice, replayable in any implementation.

Why this exists
---------------
``Gate.forward`` picks experts with ``(scores + bias).topk(k)`` -- a *discrete*
verdict read off continuous values.  Measured on this model (``dumps/smoke3`` vs
``dumps/smoke3_fp64``, i.e. the same code with the GEMM accumulator widened from
fp32 to fp64, a change that is strictly *more* accurate and moves a single GEMM
by ~1e-7): 0% of tokens route differently at layer 0, 7.4% at layer 1, and
**55.6% at layer 2**.  Two different experts differ by O(1), so once routing
forks the hidden state is incomparable, and by layer 2 the selection margin no
longer separates "near-tie" from "real bug" -- the two distributions overlap.

So layer-by-layer numerical comparison past layer 2 is measuring noise unless the
discrete verdict is taken out of the equation.  This module takes it out: the
golden's per-token expert choice is exported once, then *injected* into whatever
implementation is under test, which skips its own top-k.  What remains is a pure
operator comparison, and any deviation is then a genuine bug.

Addressing: by absolute sequence position, never by row index
-------------------------------------------------------------
The golden runs batch=1 prefill, where the j-th row of the gate input is token j.
Nothing else does: chunked prefill splits the sequence, padding inserts dead
rows, and expert dispatch permutes tokens to group them by expert.  A plan keyed
by row index therefore passes on a single-chunk unpadded smoke test and silently
misaligns under real scheduling -- and a misalignment *presents as a routing
mismatch*, the exact symptom being investigated.  Hence every lookup goes through
the caller-supplied absolute positions, and an unknown position is an error
rather than a pass-through.

One plan file describes one sequence.  A batch of several requests uses one plan
per request.
"""

from __future__ import annotations

import json
import os

import torch

PLAN_TENSORS = "routing_plan.safetensors"
PLAN_META = "routing_plan.json"

# Fields a consumer may rely on being present in routing_plan.json.
SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# building a plan out of a golden dump                                        #
# --------------------------------------------------------------------------- #
def build_from_dump(dump_dir: str, out_dir: str | None = None) -> str:
    """Distil ``layer_NN.safetensors`` routing into one small plan file.

    The full dump is ~300 MB and holds 40 files of activations; a consumer only
    needs two tensors per layer.  Extracting them once keeps the injection path
    from having to parse (or ship) the whole dump.
    """
    from safetensors.torch import load_file, save_file

    out_dir = out_dir or dump_dir
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(dump_dir, "manifest.json")) as f:
        manifest = json.load(f)
    input_ids = torch.tensor(manifest["input_ids"], dtype=torch.int32)
    n_tok = int(input_ids.numel())

    tensors: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        # Prefill of a fresh sequence: row j is position j.  Written out
        # explicitly so a consumer never has to infer it.
        "positions": torch.arange(n_tok, dtype=torch.int32),
    }

    layers: list[int] = []
    topk = None
    for name in sorted(os.listdir(dump_dir)):
        if not (name.startswith("layer_") and name.endswith(".safetensors")):
            continue
        layer_id = int(name[len("layer_"):-len(".safetensors")])
        blob = load_file(os.path.join(dump_dir, name))
        if "moe_gate.out.1" not in blob:
            continue                      # a dense layer, or taps were not attached
        idx = blob["moe_gate.out.1"]
        wts = blob["moe_gate.out.0"]
        assert idx.shape == wts.shape, f"{name}: {idx.shape} vs {wts.shape}"
        assert idx.shape[0] == n_tok, (
            f"{name}: {idx.shape[0]} gate rows but manifest says {n_tok} tokens -- "
            "the dump was not produced by a batch=1 prefill of these input_ids"
        )
        if topk is None:
            topk = int(idx.shape[1])
        assert idx.shape[1] == topk, f"{name}: topk {idx.shape[1]} != {topk}"
        tensors[f"layer_{layer_id:02d}.indices"] = idx.to(torch.int32).contiguous()
        tensors[f"layer_{layer_id:02d}.weights"] = wts.float().contiguous()
        layers.append(layer_id)

    if not layers:
        raise ValueError(f"{dump_dir}: no layer dump held 'moe_gate.out.1'")

    save_file(tensors, os.path.join(out_dir, PLAN_TENSORS))
    meta = {
        "schema_version": SCHEMA_VERSION,
        "source_dump": os.path.abspath(dump_dir),
        "n_tokens": n_tok,
        "layers": layers,
        "topk": topk,
        "input_ids": input_ids.tolist(),
        # Expert ids are global.  Golden runs world_size=1 with all 384 experts
        # resident-on-demand, so there is no EP shard mapping to undo.
        "expert_id_space": "global",
        # Provenance: which numerical variant of the golden produced this plan.
        # Two plans from different accumulators are NOT interchangeable -- that
        # difference is exactly what flips routing.
        "accum_dtype": manifest.get("accum_dtype"),
        "n_layers_run": manifest.get("n_layers_run"),
        "prompt": manifest.get("prompt"),
        "torch_version": manifest.get("torch_version"),
    }
    with open(os.path.join(out_dir, PLAN_META), "w") as f:
        json.dump(meta, f, indent=2)
    return os.path.join(out_dir, PLAN_TENSORS)


# --------------------------------------------------------------------------- #
# consuming a plan                                                            #
# --------------------------------------------------------------------------- #
class RoutingPlan:
    """A golden routing decision, looked up by absolute sequence position.

    Reference consumer for the contract in ROUTING_INJECTION.md.  The NPU side
    does not have to use this class, but it must reproduce these semantics --
    in particular the two refusals, which exist because a lenient version of
    either turns a wiring bug into a plausible-looking accuracy bug.
    """

    def __init__(self, tensors: dict[str, torch.Tensor], meta: dict):
        self.meta = meta
        self.topk = int(meta["topk"])
        self.layers = set(int(x) for x in meta["layers"])
        self.input_ids = tensors["input_ids"].to(torch.int64)
        positions = tensors["positions"].to(torch.int64)
        self._indices = {
            int(k[len("layer_"):-len(".indices")]): v.to(torch.int64)
            for k, v in tensors.items() if k.endswith(".indices")
        }
        self._weights = {
            int(k[len("layer_"):-len(".weights")]): v.float()
            for k, v in tensors.items() if k.endswith(".weights")
        }
        # position -> row, as a dense table: lookups happen once per layer per
        # step, so a searchsorted is not worth the subtlety.  -1 = not in plan.
        self._row_of = torch.full((int(positions.max()) + 2,), -1, dtype=torch.int64)
        self._row_of[positions] = torch.arange(positions.numel(), dtype=torch.int64)
        self.n_calls = 0
        self.n_rows = 0
        self.n_flipped = 0        # rows whose expert set the plan actually changed

    @classmethod
    def load(cls, path: str) -> "RoutingPlan":
        """``path`` may be the plan file itself or the directory holding it."""
        from safetensors.torch import load_file

        if os.path.isdir(path):
            path = os.path.join(path, PLAN_TENSORS)
        with open(os.path.join(os.path.dirname(path), PLAN_META)) as f:
            meta = json.load(f)
        return cls(load_file(path), meta)

    def check_input_ids(self, input_ids: torch.Tensor):
        """Refuse a plan built for different text.

        Cheap, and it catches the mistake that would otherwise look like a
        catastrophic accuracy regression.
        """
        got = input_ids.flatten().to(torch.int64)
        want = self.input_ids
        if got.numel() != want.numel() or not torch.equal(got, want):
            raise ValueError(
                f"routing plan is for {want.numel()} tokens {want[:8].tolist()}... "
                f"but this run has {got.numel()} tokens {got[:8].tolist()}... -- "
                "plans are per-sequence and not transferable"
            )

    def has_layer(self, layer_id: int) -> bool:
        return layer_id in self.layers

    def override(self, layer_id: int, positions: torch.Tensor,
                 weights: torch.Tensor, indices: torch.Tensor):
        """Replace ``(weights, indices)`` with the golden's, row by row.

        ``positions[j]`` is the absolute sequence position of row j of the gate
        output -- whatever permutation or padding the caller applied, this is how
        it says which token each row is.  Returns tensors of the caller's own
        dtype and device so it can substitute them in place.
        """
        if layer_id not in self._indices:
            raise KeyError(
                f"layer {layer_id} is not in this plan (has {sorted(self.layers)}) -- "
                "refusing to fall through to the implementation's own top-k, "
                "which would silently un-force the routing for this layer"
            )
        gi = self._indices[layer_id]
        gw = self._weights[layer_id]
        assert indices.shape == weights.shape, (indices.shape, weights.shape)
        if indices.shape[-1] != self.topk:
            raise ValueError(
                f"layer {layer_id}: implementation selects {indices.shape[-1]} experts "
                f"but the plan holds {self.topk}"
            )

        pos = positions.flatten().to(torch.int64)
        if pos.numel() != indices.shape[0]:
            raise ValueError(
                f"layer {layer_id}: {pos.numel()} positions for {indices.shape[0]} gate rows"
            )
        oob = (pos < 0) | (pos >= self._row_of.numel())
        rows = torch.where(oob, torch.zeros_like(pos), self._row_of[pos.clamp(min=0, max=self._row_of.numel() - 1)])
        missing = oob | (rows < 0)
        if missing.any():
            bad = pos[missing][:8].tolist()
            raise KeyError(
                f"layer {layer_id}: positions {bad} are not in the plan "
                f"(plan covers 0..{int(self.input_ids.numel()) - 1}). Refusing to "
                "route these rows freely: a partially forced layer produces a "
                "mismatch that looks like an operator bug. Extend the plan, or "
                "exclude these rows from the comparison explicitly."
            )

        new_i = gi[rows]
        new_w = gw[rows]
        self.n_calls += 1
        self.n_rows += int(pos.numel())
        # Order within the top-k is not itself meaningful (the sum over experts is
        # order-independent), so compare as sets to count real changes.
        same = (new_i.sort(dim=-1).values == indices.to(torch.int64).sort(dim=-1).values).all(dim=-1)
        self.n_flipped += int((~same).sum())
        return new_w.to(weights.dtype), new_i.to(indices.dtype)

    def summary(self) -> dict:
        return {
            "calls": self.n_calls,
            "rows": self.n_rows,
            "rows_changed": self.n_flipped,
            "frac_changed": (self.n_flipped / self.n_rows) if self.n_rows else 0.0,
        }


def main():
    import argparse

    ap = argparse.ArgumentParser(description="build a forced-routing plan from a golden dump")
    ap.add_argument("dump_dir")
    ap.add_argument("--out", default=None, help="where to write (default: alongside the dump)")
    a = ap.parse_args()
    path = build_from_dump(a.dump_dir, a.out)
    plan = RoutingPlan.load(path)
    print(f"wrote {path}")
    print(f"  layers {sorted(plan.layers)}")
    print(f"  tokens {plan.input_ids.numel()}  topk {plan.topk}")
    print(f"  accum  {plan.meta.get('accum_dtype')}")


if __name__ == "__main__":
    main()
