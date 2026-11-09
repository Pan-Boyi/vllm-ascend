# SPDX-License-Identifier: Apache-2.0
"""Run deterministic functional requests against the single-card service."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def request_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    # The local validation service should bypass unrelated proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def cases(long_lines):
    yield "arithmetic", [{"role": "user", "content": "What is 17 + 25? Reply with only the integer."}], "42"
    yield "chinese", [{"role": "user", "content": "请只输出这六个字：单卡测试成功"}], "单卡测试成功"
    yield "copy", [{"role": "user", "content": "Reply with exactly: violet-317"}], "violet-317"
    yield (
        "conversation",
        [
            {"role": "user", "content": "Remember the word orchid. Reply OK."},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "What word did I ask you to remember? Reply with only that word."},
        ],
        "orchid",
    )
    if long_lines:
        prompt = (
            "Remember this code: violet-317.\n"
            + "This is an ordinary filler line.\n" * long_lines
            + "What code did I ask you to remember? Reply with only the code."
        )
        yield "long_context", [{"role": "user", "content": prompt}], "violet-317"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18141")
    parser.add_argument("--model", default="dsv41-singlecard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-lines", type=int, default=600)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.long_lines < 0 or args.timeout <= 0:
        parser.error("long-lines must be nonnegative and timeout must be positive")

    results = []
    for name, messages, expected in cases(args.long_lines):
        started = time.monotonic()
        result = {"name": name, "expected": expected, "passed": False}
        try:
            response = request_json(
                args.base_url.rstrip("/") + "/v1/chat/completions",
                {
                    "model": args.model,
                    "messages": messages,
                    "temperature": 0,
                    "seed": 0,
                    "max_tokens": 512,
                    "chat_template_kwargs": {"reasoning_effort": 1},
                },
                args.timeout,
            )
            choice = response["choices"][0]
            content = choice["message"].get("content") or ""
            normalized = content.strip().strip("`*\"' .。！!\n").casefold()
            result.update(
                content=content,
                usage=response.get("usage"),
                finish_reason=choice.get("finish_reason"),
                passed=normalized == expected.casefold(),
            )
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_seconds"] = time.monotonic() - started
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n")
        print(f"{name}: {'PASS' if result['passed'] else 'FAIL'} ({result['elapsed_seconds']:.2f}s)", flush=True)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
