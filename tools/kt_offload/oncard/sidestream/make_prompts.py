"""Fixed ~200-prompt batch for the side-stream logprob comparison (seeded, deterministic).

usage: make_prompts.py <out.jsonl> [seed]

The seed is an argument so a *confirmation* run can be made on a batch the
criterion was never tuned on. Same count, same short/mid/long split, different
draw: 2026-09-22 the first A/B pair was judged on seed 20260922, and the
confirmation pair that fixed --skip-first=12 in advance used another seed.
"""
import json, random, sys, glob
out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260922
rng = random.Random(seed)
gsm = [json.loads(l) for l in open("/workspace/code/dsv4_single_card/eval/gsm8k/sgl_eval_gsm8k_20260904-161029/output-rs0.jsonl")]
gsm.sort(key=lambda r: r["id"])
rng.shuffle(gsm)
P = []
# short: a single GSM8K problem, raw completion prompt
for r in gsm[:67]:
    P.append({"kind": "short", "src": r["id"], "prompt": f"Question: {r['problem']}\nAnswer:"})
# mid: 6-8 problems with answers as few-shot, then one question (~512 tok)
k = 67
for i in range(67):
    n = 7
    shots = gsm[k:k + n]; k += n
    q = gsm[k]; k += 1
    body = "".join(f"Question: {s['problem']}\nAnswer: {s['expected_answer']}\n\n" for s in shots)
    P.append({"kind": "mid", "src": q["id"], "prompt": body + f"Question: {q['problem']}\nAnswer:"})
# long: ~11k-char windows of local docs/code + a question (~3000 tok, < 4096 so no chunk boundary)
texts = []
for f in ["/workspace/models/DeepSeek-V4.1-Flash/README.md"] + sorted(glob.glob("/workspace/code/dsv4_single_card/vllm_project/DSV4.1_*.md")):
    texts.append((f, open(f, encoding="utf-8").read()))
for f in sorted(glob.glob("/workspace/code/dsv4_single_card/vllm_project/worktrees/moe-offload/vllm_ascend/kt_offload/*.py")):
    texts.append((f, open(f, encoding="utf-8").read()))
qs = ["Summarize the key points of the text above in three sentences.",
      "What is the main technical problem discussed above?",
      "List the numbers mentioned above and what they measure."]
for i in range(66):
    f, t = texts[i % len(texts)]
    zh = sum(1 for c in t[:2000] if ord(c) > 0x2E80) > 400
    width = 4200 if zh else 10000
    if len(t) <= width:
        start, w = 0, t
    else:
        start = rng.randrange(0, len(t) - width); w = t[start:start + width]
    P.append({"kind": "long", "src": f"{f}@{start}", "prompt": w + "\n\n" + qs[i % 3] + "\n"})
with open(out, "w") as fo:
    for i, p in enumerate(P):
        p["i"] = i
        fo.write(json.dumps(p, ensure_ascii=False) + "\n")
print(len(P), f"prompts (seed {seed}) ->", out)
