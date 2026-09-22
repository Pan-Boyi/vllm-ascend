"""Sequential (concurrency 1) greedy completions with top-k logprobs; one JSON line per prompt."""
import argparse, json, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--max-tokens", type=int, default=8)
ap.add_argument("--logprobs", type=int, default=5); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
P = [json.loads(l) for l in open(a.prompts)]
if a.limit: P = P[:a.limit]
url = f"http://127.0.0.1:{a.port}/v1/completions"
t0 = time.time()
with open(a.out, "w") as fo:
    for p in P:
        body = {"model": "dsv41", "prompt": p["prompt"], "max_tokens": a.max_tokens, "temperature": 0,
                "top_p": 1.0, "logprobs": a.logprobs, "seed": 0}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        raw = urllib.request.urlopen(req, timeout=600).read()
        r = json.loads(raw)
        ch = r["choices"][0]; lp = ch.get("logprobs") or {}
        fo.write(json.dumps({"i": p["i"], "kind": p["kind"], "prompt_tokens": r.get("usage", {}).get("prompt_tokens"),
                             "text": ch.get("text"), "finish": ch.get("finish_reason"),
                             "tokens": lp.get("tokens"), "token_logprobs": lp.get("token_logprobs"),
                             "top_logprobs": lp.get("top_logprobs")}, ensure_ascii=False) + "\n")
        fo.flush()
print(f"{len(P)} prompts in {time.time()-t0:.1f}s -> {a.out}")
