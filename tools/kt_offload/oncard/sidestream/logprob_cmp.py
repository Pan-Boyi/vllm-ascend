"""Bitwise comparison of two logprob runs. Position 0 = prefill (eager, outside the captured path);
positions 1.. = decode steps (FULL graph, bs1 -> captured path, where the side stream runs).
A position is compared only while all earlier chosen tokens agree (same input)."""
import json, sys, math
def load(p): return {r["i"]: r for r in map(json.loads, open(p))}
argv = [a for a in sys.argv[1:] if not a.startswith("--skip-first=")]
skip = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--skip-first=")), 0)
A, B = load(argv[0]), load(argv[1])
label = argv[2] if len(argv) > 2 else ""
# --skip-first=N: drop the first N requests of each run. The first few requests after a
# service starts are not comparable to later ones (JIT / warmup); 2026-09-22 measured:
# in-process A1 pass1 vs pass2 differed on exactly 4 prompts, all at request index 2-5.
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
