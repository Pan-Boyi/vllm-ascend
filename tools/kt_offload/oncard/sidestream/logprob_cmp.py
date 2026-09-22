"""Bitwise comparison of two logprob runs. Position 0 = prefill (eager, outside the captured path);
positions 1.. = decode steps (FULL graph, bs1 -> captured path, where the side stream runs).
A position is compared only while all earlier chosen tokens agree (same input)."""
import json, sys, math
def load(p): return {r["i"]: r for r in map(json.loads, open(p))}
argv = [a for a in sys.argv[1:] if not a.startswith("--skip-first=")]
skip = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--skip-first=")), 12)
A, B = load(argv[0]), load(argv[1])
label = argv[2] if len(argv) > 2 else ""
# --skip-first=N: drop the first N requests of each run. The first few requests after a
# service starts are not comparable to later ones (JIT / warmup).
#
# 🔴 The cutoff is calibrated PER ARM, and 10 was too tight. 2026-09-22 first pass
# (seed 20260922, 4 processes): the SERIAL arm's own cross-process floor still
# differed on requests 2,3,4,6,7,8,9, and the side-stream arm's on 7,10,11. A
# cutoff taken from the serial arm and applied to the side-stream arm therefore
# reads the side stream's longer warm-up tail as "the side stream computes
# different numbers" -- which is exactly the mistake this default now prevents.
# From request 12 on, all six pairs (both floors and all four A/B crosses) were
# 100% bit-identical, 1149/1149 decode positions, nonzero differences 0.
# Default 12; pass --skip-first=N explicitly to look inside the warm-up window.
keys = sorted(k for k in set(A) & set(B) if k >= skip)
stat = {"prefill": [0, 0, []], "decode": [0, 0, []]}  # compared, bit-identical, |d| list (over all top-k entries)
tok_div = []; key_set_diff = 0; prefill_pt = []
for i in keys:
    a, b = A[i], B[i]
    prefill_pt.append(a["prompt_tokens"])
    n = min(len(a["tokens"] or []), len(b["tokens"] or []))
    for pos in range(n):
        ph = "prefill" if pos == 0 else "decode"
        ta, tb = a["top_logprobs"][pos] or {}, b["top_logprobs"][pos] or {}
        same = (a["tokens"][pos] == b["tokens"][pos] and a["token_logprobs"][pos] == b["token_logprobs"][pos] and ta == tb)
        stat[ph][0] += 1; stat[ph][1] += same
        if set(ta) != set(tb): key_set_diff += 1
        ds = [abs(ta[k] - tb[k]) for k in set(ta) & set(tb)]
        ds.append(abs(a["token_logprobs"][pos] - b["token_logprobs"][pos]) if a["tokens"][pos] == b["tokens"][pos] else math.nan)
        stat[ph][2].extend(d for d in ds if not math.isnan(d))
        if a["tokens"][pos] != b["tokens"][pos]:
            tok_div.append((i, a["kind"], pos)); break
def q(v, p):
    if not v: return float("nan")
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]
print(f"== {label}  prompts={len(keys)} (A-only {len(set(A)-set(B))}, B-only {len(set(B)-set(A))}); "
      f"prompt_tokens max={max(prefill_pt)} (>4096: {sum(1 for x in prefill_pt if x > 4096)})")
for ph, (n, same, ds) in stat.items():
    nz = [d for d in ds if d > 0]
    print(f"  {ph:7s}: positions compared {n}, bit-identical {same} ({same/max(n,1):.1%}); "
          f"|dlogprob| entries {len(ds)}, nonzero {len(nz)}, max {max(ds) if ds else float('nan'):.3e}, "
          f"p50 {q(ds,.5):.3e} p90 {q(ds,.9):.3e} p99 {q(ds,.99):.3e}; nonzero-only p50 {q(nz,.5):.3e}")
print(f"  top-k key-set differences: {key_set_diff}; chosen-token divergences: {len(tok_div)} {tok_div[:10]}")
