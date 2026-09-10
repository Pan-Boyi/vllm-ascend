"""Standalone reference for ``engram_hash_ids`` -- no weights, no forward pass.

The n-gram hash is pure integer arithmetic (a compressed-vocab remap, multiply,
XOR, then a prime modulus per bucket range).  There is no float anywhere in it, so
it must match **bit for bit** -- there is no tolerance to hide behind, and a
mismatch means the hash, the compressed token map, or the prime layout is wrong,
not that something rounded differently.

It is also the earliest dump point that can disagree: ``engram.out`` at layer 1
feeds every later layer.  Pinning it first removes it from the search space.

Running the full golden to get these ids costs 216 s and 55 GB.  This script needs
only the tokenizer and config, so it takes about a second -- which is why it exists
separately.

    python engram_hash_ref.py                      # the golden prompt, prefill
    python engram_hash_ref.py --verify DUMP_DIR    # check against a golden dump
    python engram_hash_ref.py --decode-from 20     # prefill 0..19 then step 20..26
    python engram_hash_ref.py --out ref_hashes     # write safetensors + json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
for p in (REF, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

MODEL_DIR = "/workspace/models/DeepSeek-V4.1-Flash"

# Kept in step with run_golden.GOLDEN_PROMPT; see ALIGNMENT_PROTOCOL.md section 2.
GOLDEN_PROMPT = (
    "The capital of France is Paris. "
    "Ascend NPU kernels must match this reference bit for bit in structure "
    "and within tolerance in value."
)


def build(max_seq_len: int = 512):
    """Everything needed for the hash, and nothing else -- no checkpoint touched."""
    import model as M
    from engram import EngramLayout, NgramHashState
    from transformers import AutoTokenizer

    from golden import load_args

    args = load_args(max_batch_size=1, max_seq_len=max_seq_len)
    layout = EngramLayout.from_args(args)
    assert layout is not None, "config has no engram_layer_ids"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    state = NgramHashState(args, layout, tokenizer)
    return args, layout, state, tokenizer, M


def digest(t: torch.Tensor) -> str:
    """Short stable fingerprint, so two sides can compare without shipping tensors."""
    a = t.to(torch.int64).contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--verify", default=None, metavar="DUMP_DIR",
                    help="compare against DUMP_DIR/prologue.safetensors")
    ap.add_argument("--decode-from", type=int, default=0, metavar="N",
                    help="prefill tokens 0..N-1, then feed the rest one at a time; "
                         "the concatenation must equal the pure-prefill result")
    ap.add_argument("--out", default=None, help="write the ids as safetensors + json")
    ap.add_argument("--max-seq-len", type=int, default=512)
    a = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    args, layout, state, tokenizer, M = build(a.max_seq_len)

    ids = tokenizer(a.prompt or GOLDEN_PROMPT)["input_ids"]
    input_ids = torch.tensor([ids])
    n = input_ids.size(1)

    print(f"engram_layer_ids   {layout.layer_ids}   (dim=2 of the output is in THIS order)")
    print(f"max_ngram_size     {layout.max_ngram_size}   n_heads {layout.n_heads}   "
          f"head_dim {layout.head_dim}")
    print(f"n_hash_cols        {(layout.max_ngram_size - 1) * layout.n_heads}")
    print(f"num_embeddings     {layout.num_embeddings}")
    print(f"compressed_vocab   {args.engram_compressed_vocab_size}   "
          f"pad_id {args.engram_pad_id} -> compressed {state.pad_id}")
    print(f"input_ids ({n})     {input_ids[0].tolist()}")
    print()

    hashes = state(input_ids, 0, None)
    print(f"engram_hash_ids    {tuple(hashes.shape)}  {hashes.dtype}")
    print(f"  sha256[:16]      {digest(hashes)}")
    for k, lid in enumerate(layout.layer_ids):
        h = hashes[0, :, k, :]
        print(f"  layer {lid:2d}  min {int(h.min()):>10}  max {int(h.max()):>10}  "
              f"sha {digest(h)}")
    print(f"  token 0 / layer {layout.layer_ids[0]}: {hashes[0, 0, 0].tolist()}")
    print(f"  token {n - 1} / layer {layout.layer_ids[0]}: {hashes[0, n - 1, 0].tolist()}")

    # prefill/decode self-consistency of the hash state.  The cache carries the
    # look-back across the split, so a decode step at position p must produce the
    # same ids as the prefill did -- this is the integer half of criterion G-1, and
    # it is checkable without any weights.
    if a.decode_from:
        cut = a.decode_from
        assert 0 < cut < n, f"--decode-from must be in 1..{n - 1}"
        _, _, state2, _, _ = build(a.max_seq_len)
        parts = [state2(input_ids[:, :cut], 0, None)]
        for p in range(cut, n):
            parts.append(state2(input_ids[:, p:p + 1], p, None))
        stepped = torch.cat(parts, dim=1)
        ok = torch.equal(stepped, hashes)
        print()
        print(f"prefill[0:{cut}] + {n - cut} decode steps  vs  pure prefill[0:{n}]: "
              f"{'IDENTICAL' if ok else 'DIFFER'}")
        if not ok:
            bad = (stepped != hashes).any(-1).any(-1)[0].nonzero().flatten().tolist()
            print(f"  differing positions: {bad[:16]}")
            sys.exit(1)

    if a.verify:
        from safetensors.torch import load_file
        blob = load_file(os.path.join(a.verify, "prologue.safetensors"))
        got = blob["engram_hash_ids"].to(torch.int64)
        print()
        if tuple(got.shape) != tuple(hashes.shape):
            print(f"SHAPE MISMATCH: dump {tuple(got.shape)} vs reference {tuple(hashes.shape)}")
            sys.exit(1)
        same = torch.equal(got, hashes)
        print(f"vs {a.verify}/prologue.safetensors: {'BIT-IDENTICAL' if same else 'DIFFER'}")
        if not same:
            d = (got != hashes)
            print(f"  {int(d.sum())} of {d.numel()} ids differ; "
                  f"positions {d.any(-1).any(-1)[0].nonzero().flatten().tolist()[:16]}")
            sys.exit(1)

    if a.out:
        from safetensors.torch import save_file
        os.makedirs(a.out, exist_ok=True)
        save_file({"input_ids": input_ids.int().contiguous(),
                   "engram_hash_ids": hashes.contiguous()},
                  os.path.join(a.out, "engram_hash_ref.safetensors"))
        meta = {
            "input_ids": input_ids[0].tolist(),
            "engram_layer_ids": list(layout.layer_ids),
            "max_ngram_size": layout.max_ngram_size,
            "n_heads": layout.n_heads,
            "head_dim": layout.head_dim,
            "n_hash_cols": (layout.max_ngram_size - 1) * layout.n_heads,
            "num_embeddings": list(layout.num_embeddings),
            "compressed_vocab_size": args.engram_compressed_vocab_size,
            "pad_id_compressed": int(state.pad_id),
            "shape": list(hashes.shape),
            "sha256_16": digest(hashes),
            "per_layer_sha256_16": {
                str(lid): digest(hashes[0, :, k, :])
                for k, lid in enumerate(layout.layer_ids)
            },
            # The primes are the whole layout: a single mismatched modulus rehashes
            # one bucket range and nothing else, which is easy to miss.
            "primes": [[list(pn) for pn in per_layer] for per_layer in layout.primes],
        }
        with open(os.path.join(a.out, "engram_hash_ref.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nwrote {a.out}/engram_hash_ref.safetensors + .json")


if __name__ == "__main__":
    main()
