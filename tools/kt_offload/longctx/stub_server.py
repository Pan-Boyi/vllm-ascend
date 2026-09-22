#!/usr/bin/env python3
"""Fake OpenAI-compatible /v1/chat/completions for CPU self-test of
run_longctx_check.py (and cookbook/niah_scan.py). NOT a model.

STUB_MODE:
  oracle          answers the target iff the item's evidence probe is in the prompt,
                  else a wrong letter  (=> full right / trunc wrong)
  always_correct  always the target
  always_A        always "ANSWER: A"
  garbage         unparseable prose
  no_usage        like oracle but omits the usage block
  http500         every request -> HTTP 500
NIAH prompts ("registration number") are answered by echoing the needle id.
prompt_tokens = DSV4.1 tokenizer count of the chat-rendered prompt (same as the
runner's local count) + STUB_TOKEN_SKEW (default 0).
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_longctx_check as R  # noqa: E402

MODE = os.environ.get("STUB_MODE", "oracle")
SKEW = int(os.environ.get("STUB_TOKEN_SKEW", "0"))
ITEMS = json.load(open(R.ITEMS))["items"]


def answer(prompt: str) -> str:
    if "registration number" in prompt:
        m = re.search(r"\d{4}-ZEBRA-\d{4}", prompt)
        return m.group(0) if m else "unknown"
    item = next((x for x in ITEMS if x["question"] in prompt), None)
    if item is None:
        return "2"
    tgt = item["target"]
    if MODE == "always_correct":
        return f"ANSWER: {tgt}"
    if MODE == "always_A":
        return "ANSWER: A"
    if MODE == "garbage":
        return "I cannot determine this from the text."
    wrong = next(L for L in "DCBA" if L != tgt)
    # look for the probe in the <text> block only: some probes also occur in the question
    doc = prompt.split("<text>\n", 1)[-1].split("\n</text>", 1)[0]
    return f"ANSWER: {tgt if R.probe_in(doc, item['evidence_probe']) else wrong}"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"data": [{"id": "dsv41"}]})

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if MODE == "http500":
            return self._send(500, {"error": "stub 500"})
        # the runner must send the agreed protocol; refuse otherwise so the test catches drift
        if (req.get("temperature") != 0.0 or
                req.get("chat_template_kwargs", {}).get("thinking") is not False):
            return self._send(400, {"error": f"protocol drift: {req.get('temperature')} "
                                              f"{req.get('chat_template_kwargs')}"})
        prompt = req["messages"][0]["content"]
        text = answer(prompt)
        out = {"choices": [{"message": {"role": "assistant", "content": text},
                            "finish_reason": "stop"}]}
        if MODE != "no_usage":
            n = R.local_tokens(prompt)
            out["usage"] = {"prompt_tokens": n + SKEW, "completion_tokens": 4,
                            "total_tokens": n + SKEW + 4}
        self._send(200, out)


if __name__ == "__main__":
    port = int(sys.argv[1])
    R.local_tokens("warm")
    print(f"stub {MODE} on {port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
