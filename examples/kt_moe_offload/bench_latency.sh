#!/usr/bin/env bash
# Measure TTFT and TPOT against an already-running server.
#
#   Window 1:  bash serve_lowlatency.sh          # leave it running
#   Window 2:  PORT=8100 bash bench_latency.sh   # measure, as often as you like
#
# This does not touch the server. Start it once and benchmark repeatedly.
#
#   PORT=8100 CONCURRENCY=1 REQUESTS=10 PROMPT_TOKENS=4096 GEN_TOKENS=128 \
#     PROMPT_FILE=/path/to/your/own/text.txt \
#     bash bench_latency.sh
#
# 🔴 Read this before trusting a number.
#
# A single reading is an order of magnitude, not a measurement. The machine this
# was developed on drifts 2.5x across time windows at the *same* commit
# (242 s -> 603 s, same benchmark, twelve hours apart), while the differences
# people usually want to measure are a few percent. So:
#
#   - Never build a ratio out of two readings taken at different times. Run both
#     arms of any A/B interleaved, in the same window.
#   - At concurrency > 1, send many more requests than the concurrency, or you
#     measure the tail drain rather than the steady state (concurrency 4 with
#     only 4 requests read 68 ms/token against a steady-state 27.7 ms).
#
# At concurrency 1 each request is its own wave, so there is no tail to distort
# and REQUESTS is just the sample count. The per-request spread is printed for
# exactly that reason: read it before quoting the mean.
set -u

PORT=${PORT:-8100}
HOST=${HOST:-127.0.0.1}
MODEL=${MODEL_NAME:-dsv41}
CONCURRENCY=${CONCURRENCY:-1}
REQUESTS=${REQUESTS:-10}
PROMPT_TOKENS=${PROMPT_TOKENS:-4096}
GEN_TOKENS=${GEN_TOKENS:-128}

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

python3 - "$HOST" "$PORT" "$MODEL" "$CONCURRENCY" "$REQUESTS" "$PROMPT_TOKENS" "$GEN_TOKENS" <<'PY'
import io, json, os, statistics, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

host, port, model = sys.argv[1], int(sys.argv[2]), sys.argv[3]
conc, nreq, ptok, gtok = (int(x) for x in sys.argv[4:8])
url = f"http://{host}:{port}/v1/completions"

tok_path = f"http://{host}:{port}/v1/models"
with urllib.request.urlopen(tok_path, timeout=30) as r:
    served = [m["id"] for m in json.load(r)["data"]]
if model not in served:
    # Every request would 404, and a 404 still produces a complete-looking
    # report of zeros -- failure here is shaped exactly like a measurement.
    sys.exit(f"model {model!r} is not served; server offers {served}")

# "4096 tokens" has to mean 4096 tokens, or the reading describes a different
# workload than the label claims. Loading the model's tokenizer here would mean
# knowing the checkpoint path, so calibrate against the server instead, growing
# the text until the prompt_tokens it reports back hits the target.
def prompt_tokens(text):
    body = json.dumps({"model": model, "prompt": text, "max_tokens": 1,
                       "temperature": 0.0}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["usage"]["prompt_tokens"]
    except urllib.error.HTTPError as e:
        if e.code == 400:
            sys.exit(f"server rejected a {len(text)}-character probe prompt: "
                     f"{e.read().decode()[:300]}\n"
                     f"PROMPT_TOKENS={ptok} plus GEN_TOKENS={gtok} most likely "
                     f"exceeds the server's max_model_len.")
        raise

# 🔴 The prompt is English prose, not a repeated filler token, and for this
# model that is not cosmetic. Routed experts are picked per token: a prompt made
# of one token id over and over hits a far narrower set of experts than real
# text, and on this path a narrow set is a cheap one -- more of the work lands
# on the experts kept resident on the NPU instead of falling through to the CPU.
# A repeated-token prompt therefore reports a better TPOT than the same length of
# real text would. vLLM's own `random` dataset errs the other way (consecutive
# vocabulary ids, wider and less natural than prose). Prose sits between them.
#
# PROMPT_FILE overrides this with your own text, which is the honest thing to do
# if your traffic looks nothing like the sample below. It is cycled if short.
SAMPLE = """
The scheduler keeps a queue of requests and decides, on every step, which of
them make progress. A request that has not yet produced its first token is in
the prefill phase: its whole prompt must be read before anything can be emitted,
and the cost of that grows with the length of the prompt. Once the first token
is out, the request moves into decoding, where each further token costs roughly
the same as the one before it, and the pressure moves from arithmetic to memory.
Serving systems are judged on both, and the two rarely improve together.
Memory is the harder half. Keys and values computed during prefill are kept so
that decoding does not recompute them, and that cache grows with every token in
flight. Capacity is finite, so a scheduler that admits too many requests will
find itself evicting work it has already paid for, recomputing it later, and
delivering worse latency than if it had admitted fewer. Admission control, block
size, and the eviction policy all trade throughput against tail latency, and the
right balance depends on the shape of the traffic rather than on the model.
Mixture-of-experts models complicate this further. Each token is routed to a
small subset of the available experts, so the weights actually touched depend on
the content of the request rather than its length alone. Two prompts of equal
size can differ substantially in cost when one concentrates on a handful of
experts and the other spreads across many. Systems that keep only part of the
expert set in fast memory are especially sensitive: a request whose tokens route
to resident experts runs at one speed, and a request that reaches for the rest
runs at another, without anything in the request itself announcing which it is.
Measurement has to respect all of this. A benchmark that sends the same token
thousands of times measures a routing pattern no real workload produces; one
that samples the vocabulary uniformly measures a different unreal pattern. What
the operator wants to know is the behaviour under traffic resembling their own,
which is why published serving numbers are usually quoted against a corpus of
real conversations rather than against synthetic filler, and why a number taken
under one corpus should not be carried to another without saying so.
Latency figures also depend on how they are defined. Time to first token covers
queueing, prompt processing, and the first forward pass; time per output token
covers everything after. Reporting a single average hides the distribution, and
the distribution is what users experience: a service whose median is excellent
and whose tail is poor will be described as slow. Percentiles, the number of
requests behind them, and the conditions under which they were gathered belong
next to any figure that is meant to be acted upon.
"""


def build_prompt(target):
    """Return text whose prompt_tokens equals target exactly, by calibrating
    against the server -- the tokenizer lives there, not here."""
    src = SAMPLE
    if os.environ.get("PROMPT_FILE"):
        src = io.open(os.environ["PROMPT_FILE"], encoding="utf-8").read()
        if not src.strip():
            sys.exit(f"PROMPT_FILE {os.environ['PROMPT_FILE']} is empty")
    src = " ".join(src.split())
    per_char = prompt_tokens(src[:4000]) / len(src[:4000])
    chars = max(1, round(target / per_char))
    # Step by the measured rate: it is stable for prose and lands in two or three
    # probes. Two guards, both of which a stub-tokenizer test hit before this ran
    # against a server: a rounded step of zero would sit still one token short,
    # and not every target is reachable when a token spans several characters, so
    # revisiting a length means no exact answer exists and the closest is used.
    best, tried = None, set()
    for _ in range(10):
        if chars in tried:
            break
        tried.add(chars)
        text = (src * (chars // len(src) + 1))[:chars]
        got = prompt_tokens(text)
        if got == target:
            return text, got, src
        if best is None or abs(got - target) < abs(best[1] - target):
            best = (text, got)
        step = round((target - got) / per_char)
        chars = max(1, chars + (step if step else (1 if got < target else -1)))
    text, got = best
    print(f"\u26a0\ufe0f  closest reachable is {got} prompt tokens, not {target} "
          f"-- reported as measured, and a comparison must use the same figure")
    return text, got, src


prompt, real_ptok, source = build_prompt(ptok)
print(f"prompt: {real_ptok} tokens of English prose "
      f"({'PROMPT_FILE' if os.environ.get('PROMPT_FILE') else 'built-in sample'}, "
      f"{len(set(source.split()))} distinct words in the source)")

def one(_):
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": gtok,
        "temperature": 0.0, "stream": True,
        # ignore_eos pins the generated length: total tokens = requests x
        # max_tokens regardless of sampling, so runs stay comparable.
        "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, ntok, last = None, 0, t0
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break
            chunk = json.loads(line[6:])
            text = chunk["choices"][0].get("text") or ""
            if not text:
                continue
            if ttft is None:
                ttft = time.perf_counter() - t0
            ntok += 1
            last = time.perf_counter()
    if ttft is None or ntok < 2:
        return None
    # TPOT excludes the first token: it is (decode wall time) / (tokens - 1),
    # so prefill does not leak into the per-token figure.
    return ttft, (last - t0 - ttft) / (ntok - 1) * 1000, ntok

t_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=conc) as ex:
    rows = [r for r in ex.map(one, range(nreq)) if r]
wall = time.perf_counter() - t_start

if not rows:
    sys.exit("no request produced tokens -- nothing measured")
bad = nreq - len(rows)
short = [r for r in rows if r[2] != gtok]
ttfts = sorted(r[0] for r in rows)
tpots = sorted(r[1] for r in rows)
def pct(v, q):
    return v[min(len(v) - 1, int(q * len(v)))]

print(f"\nconcurrency {conc}   requests {len(rows)}/{nreq}"
      f"   prompt {real_ptok}   gen {gtok}   wall {wall:.1f}s")
print(f"  TTFT  mean {statistics.fmean(ttfts):8.2f} s   "
      f"min {ttfts[0]:6.2f}   p50 {pct(ttfts,.5):6.2f}   max {ttfts[-1]:6.2f}")
print(f"  TPOT  mean {statistics.fmean(tpots):8.2f} ms  "
      f"min {tpots[0]:6.2f}   p50 {pct(tpots,.5):6.2f}   max {tpots[-1]:6.2f}")
print(f"  output {sum(r[2] for r in rows) / wall:.1f} tok/s aggregate")
if len(rows) > 1:
    print(f"  spread: TTFT sd {statistics.stdev(ttfts):.2f} s, "
          f"TPOT sd {statistics.stdev(tpots):.2f} ms"
          f"   <- quote this next to any mean")
if bad:
    print(f"  🔴 {bad} request(s) produced no tokens -- the means above are over "
          f"the rest, which is a different workload than requested")
if short:
    print(f"  🔴 {len(short)} request(s) stopped before {gtok} tokens; "
          f"ignore_eos did not hold and the run is not comparable")
PY
