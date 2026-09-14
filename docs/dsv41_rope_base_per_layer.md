# `rope.kv.out` at a ratio-2 layer: which stream, and which rotation

Answered 2026-09-14 by reading the code. No card, no flight.

## The question splits in two, and the halves have different answers
"Which **stream**" and "which **rotation**" are separate questions. Merging
them is how this stayed open: **the stream does not depend on the ratio, the
rotation does.**

## 1. Which stream — the ORIGINAL (sliding-window) one, at every ratio
```
qkv_prolog.py:261  kv_out = rope_tail(kv, cos, sin, rope_head_dim)
qkv_prolog.py:271  tap("rope.kv.out", kv_out)
qkv_prolog.py:272  scatter_rows(swa_kv_cache, kv_out, slot_mapping, ...)

attention.py:510   swa_kv_cache = self.swa_cache_layer.kv_cache   <- AscendDeepseekV41SWACache
attention.py:552   slot_mapping  = swa_metadata.slot_mapping      <- the SWA cache group
```
=> one row **per query token**, into the **sliding-window** cache.

**It is not the compressed stream.** That one is written by a separate path,
`compressor_insert.rope_quant_insert`, into
`AscendDeepseekV41CompressorStateCache` (the ring). Two different cache
objects, two different slot mappings.

The prolog **does not branch on the ratio**. The only `compress_ratio == 2`
test on this path (`attention.py:569`) is a compile-readiness assert in
`__init__`, not a data-path branch. So the above holds at ratio 0, 1 and 2.

## 2. Which rotation — depends on the ratio, and the boundary is `> 0`
`self.rotary_emb` is built **per layer**, and the prolog rotates with exactly
that object (`cos_sin_cache=self.rotary_emb.cos_sin_cache`):
```
upstream attention.py:347   self.rotary_emb = build_deepseek_v4_rope(..., compress_ratio=self.compress_ratio)
upstream common/rope.py:29  rope_theta = compress_rope_theta if compress_ratio > 0 else rope_theta
upstream common/rope.py:31  YaRN when compress_ratio > 0
```

| `compress_ratios` index | ratio | base | scaling |
|---|---|---|---|
| 0, 1 | 0 | `rope_theta` = **10000** | `deepseek_yarn` with **factor 1.0** (plain) |
| 2 – 19 | 2 | `compress_rope_theta` = **160000** | YaRN, factor 16, beta_fast 32, beta_slow 1, orig_max 65536 |
| 20 – 39 | 1 | `compress_rope_theta` = **160000** | YaRN, same |
| 40 – 42 | 0 | **10000** | factor 1.0 |

`mscale = mscale_all_dim = 0` on every row, so no scaled-rope amplitude
correction anywhere.

Note the ratio-0 row: "plain RoPE" is *implemented* as `deepseek_yarn` with
`factor = 1.0`, not as a different code path. A check that looks for the
rope **type** will not separate the two; only the base and the factor do.

## The trap: V4.0 and V4.1 disagree about ratio **1**
```
deepseek_v4  /common/rope.py:29   compress_ratio > 1
deepseek_v4_1/common/rope.py:29   compress_ratio > 0     <- ours
```
Same function name, same-shaped file, one character different — and
`compress_ratios` has **three** values (0, 1, 2), so the third value is
exactly where the two versions disagree.

=> **Layers 20–39 (ratio 1) are on base 160000 + YaRN under V4.1**, and would
be on 10000 plain under V4.0's rule. Any assumption of the form "the
compressed rope is the ratio-2 layers" is wrong for twenty layers, and
nothing raises.

## 3. The two streams at one layer share the rope table
`register_rope_groups` (attention.py:614) registers the `c{ratio}` group from
**the same `self.rotary_emb` object** — deliberately: "never from a second
derivation of them: the base differs per layer and getting it wrong raises
nothing."

=> At a ratio-2 layer the compressed stream and the SWA stream use the
**same** base. **A difference between the two streams is therefore not a
rotation difference**, and should not be explained as one.

## Not verified
`compress_ratios` has **43** entries while the model has **40** decoder
layers. Indices 0–39 line up with the decoder layers; 40–42 are presumably
the MTP / draft heads, but that mapping was **not** checked. Do not read rows
40–42 of the table as being about decoder layers.
