# DeepSeek-V4.1: RoPE base per layer

## Which stream the prolog rotates

`qkv_prolog` applies RoPE to the KV row of every query token and scatters it
into the sliding-window (SWA) cache, using the SWA cache group's slot mapping.
The compressed stream is written by a separate path
(`compressor_insert.rope_quant_insert_ascend`) into the compressor state cache. The
prolog does not branch on the compress ratio.

## Which rotation each layer uses

`self.rotary_emb` is built per layer by `build_deepseek_v4_rope(...,
compress_ratio=...)`, and the prolog rotates with that object. In V4.1's
`common/rope.py` the switch is `compress_ratio > 0`:

| `compress_ratios` index | ratio | base | scaling |
|---|---|---|---|
| 0, 1 | 0 | `rope_theta` = 10000 | `deepseek_yarn` with factor 1.0 (plain) |
| 2 - 19 | 2 | `compress_rope_theta` = 160000 | YaRN, factor 16, beta_fast 32, beta_slow 1, orig_max 65536 |
| 20 - 39 | 1 | `compress_rope_theta` = 160000 | YaRN, same |
| 40 - 42 | 0 | 10000 | factor 1.0 |

`mscale = mscale_all_dim = 0` everywhere, so there is no amplitude correction.
Plain RoPE is implemented as `deepseek_yarn` with `factor = 1.0`, so the rope
type alone does not distinguish the two cases; the base and factor do.

`compress_ratios` has 43 entries for 40 decoder layers. Indices 40 - 42 are
presumably the MTP heads; that mapping has not been verified.

## V4.0 and V4.1 differ at ratio 1

V4.0's `common/rope.py` uses `compress_ratio > 1`, V4.1's uses
`compress_ratio > 0`. Under V4.1 the ratio-1 layers (20 - 39) therefore use
base 160000 with YaRN, not 10000 plain. Nothing raises if the wrong rule is
applied.

## Both streams of a layer share the rope table

`register_rope_groups` registers the compressed-stream (`c{ratio}`) group from
the same `self.rotary_emb` object rather than deriving it again, because the
base differs per layer and a wrong base raises nothing. A difference between
the two streams of one layer is therefore not a rotation difference.
