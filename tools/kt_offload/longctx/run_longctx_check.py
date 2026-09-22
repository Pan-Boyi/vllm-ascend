#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 单卡 >16k prompt 功能验通：LongBench-v2 真题 × {完整, 只留尾部 ~12k} 两臂。

它回答的问题：服务在 >16384 token（两级候选筛选 O6 生效的那条线）的真实长文上，
能不能答对**只有在尾部 12k 之外才找得到依据**的题。

判读（每题）：
  完整对 & 截断错  ⇒ DISCRIMINATIVE_PASS  模型确实用到了尾部 12k 之外的内容
  完整对 & 截断对  ⇒ NO_DISCRIMINATION    这题**不能**当 16k 验通证据（可能靠参数记忆/常识）
  完整错           ⇒ FULL_WRONG           记录，单题不下结论
  任一臂 未解析/报错/prompt 未跨线 ⇒ INCONCLUSIVE（不当成对或错）

用法：
  PORT=8010 python3 run_longctx_check.py                         # 主测（4 题 × 2 臂 = 8 请求）
  PORT=8010 python3 run_longctx_check.py --ids all               # 含备选
  PORT=8010 python3 run_longctx_check.py --niah                  # 主测后接 niah_scan
  python3 run_longctx_check.py --dry-run                         # 不发请求：构 prompt、本地计数、probe 检查
  python3 run_longctx_check.py --selftest                        # 解析/判定自测（无网络）

口径：chat/completions，thinking 关（chat_template_kwargs.thinking=False），
**temperature=0.0 显式传**（harness 默认 T=1.0 是已知坑），top_p=1.0，max_tokens 小。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(HERE, "longctx_items.json")
TOKENIZER = "/workspace/models/DeepSeek-V4.1-Flash"
ENCODING_DIR = "/workspace/models/DeepSeek-V4.1-Flash/encoding"
NIAH = "/workspace/code/dsv4_single_card/vllm_project/cookbook/niah_scan.py"
#: 候选筛选开始生效的线：candidate_topk_blocks(2048) × block(8) × ratio(1)
THRESHOLD = 16384

O15_CAVEAT = ("⚠ 前提（O15）：ds41 qli 在 valid_s2 % 8 == 0 时不把最新块钉进候选集，"
              "本次所有 >16k 读数都带着这条已知算子缺陷，不是干净基线。")

TEMPLATE = """Please read the following text and answer the questions below.

<text>
{document}
</text>

Answer the following multiple choice question. Respond with exactly one line of the format 'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A,B,C,D. Do not explain.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""

# ---------------------------------------------------------------- answer parsing
# 最后一个 "ANSWER: X"（大小写不敏感的 answer，字母必须是大写 A-D 且后面不跟字母/数字）；
# 兜底：整段只有一个字母（可带括号/星号/句点），或首行形如 "C) ..."。其它一律 None。
_ANS_RE = re.compile(r"(?i:answer)\s*(?:(?i:is)\s*)?[:：]?\s*[\*\(\[]*\s*([ABCD])(?![A-Za-z0-9])")
_BARE_RE = re.compile(r"^[\s\*\(\[]*([ABCD])[\s\*\)\]\.]*$")
_LEAD_RE = re.compile(r"^\s*[\*\(\[]*([ABCD])[\)\.:：]\s")


def parse_answer(text: str | None) -> str | None:
    if not text:
        return None
    hits = _ANS_RE.findall(text)
    if hits:
        return hits[-1]
    m = _BARE_RE.match(text)
    if m:
        return m.group(1)
    m = _LEAD_RE.match(text)
    if m:
        return m.group(1)
    return None


PARSE_POS = [
    ("ANSWER: C", "C"), ("The table says 6144.\n\nANSWER: C", "C"), ("ANSWER: (B)", "B"),
    ("**ANSWER: D**", "D"), ("Answer: A", "A"), ("The answer is B.", "B"), ("C", "C"),
    ("(D)", "D"), ("C) 6144", "C"), ("ANSWER：B", "B"), ("ANSWER: A\nwait\nANSWER: B", "B"),
    ("answer: [D]", "D"), ("**C**", "C"), ("ANSWER: C.", "C"),
]
PARSE_NEG = [
    "", None, "Answer: The answer depends on context.", "I think the answer is unclear.",
    "ANSWER: E", "Absolutely.", "A lot of text without a letter", "ANSWER: [LETTER]",
    "answer: a", "ANSWER: Across", "Based on the text, both could apply", "ANSWER: AB",
    "Dear reader",
]


# ---------------------------------------------------------------- verdict
def verdict(target: str, full: dict, trunc: dict) -> str:
    """Only compare fields that are PRESENT. A missing prediction is never 'wrong'."""
    for arm_name, arm in (("full", full), ("trunc", trunc)):
        if arm.get("error"):
            return f"INCONCLUSIVE({arm_name}:error)"
        if arm.get("pred") is None:
            return f"INCONCLUSIVE({arm_name}:unparsed)"
    ptf = full.get("prompt_tokens")
    if ptf is None:
        return "INCONCLUSIVE(full:no_usage)"
    if ptf <= THRESHOLD:
        return "INCONCLUSIVE(full:prompt<=16384)"
    fc, tc = full["pred"] == target, trunc["pred"] == target
    if fc and not tc:
        return "DISCRIMINATIVE_PASS"
    if fc and tc:
        return "NO_DISCRIMINATION"
    if not fc and tc:
        return "FULL_WRONG(trunc_right!)"
    return "FULL_WRONG(both_wrong)"


def selftest() -> int:
    bad = 0
    for text, want in PARSE_POS:
        got = parse_answer(text)
        if got != want:
            bad += 1
            print(f"  ✗ POS {text!r}: got {got!r} want {want!r}")
    for text in PARSE_NEG:
        got = parse_answer(text)
        if got is not None:
            bad += 1
            print(f"  ✗ NEG {text!r}: got {got!r} want None")
    # 区分力：解析器的输出必须随输入变 —— 四个字母各自可被读出，且不是常量
    letters = {parse_answer(f"ANSWER: {L}") for L in "ABCD"}
    if letters != set("ABCD"):
        bad += 1
        print(f"  ✗ parser is not discriminative: {letters}")
    # 判定函数：四种组合必须给出四种不同判定；缺字段必须 INCONCLUSIVE 而不是「错」
    ok = {"pred": "C", "prompt_tokens": 20000}
    wrong = {"pred": "A", "prompt_tokens": 20000}
    tr_ok = {"pred": "C", "prompt_tokens": 12300}
    tr_wrong = {"pred": "B", "prompt_tokens": 12300}
    cases = [
        (ok, tr_wrong, "DISCRIMINATIVE_PASS"), (ok, tr_ok, "NO_DISCRIMINATION"),
        (wrong, tr_wrong, "FULL_WRONG(both_wrong)"), (wrong, tr_ok, "FULL_WRONG(trunc_right!)"),
        ({"pred": None, "prompt_tokens": 20000}, tr_wrong, "INCONCLUSIVE(full:unparsed)"),
        (ok, {"pred": None}, "INCONCLUSIVE(trunc:unparsed)"),
        ({"pred": "C", "prompt_tokens": None}, tr_wrong, "INCONCLUSIVE(full:no_usage)"),
        ({"pred": "C", "prompt_tokens": 16000}, tr_wrong, "INCONCLUSIVE(full:prompt<=16384)"),
        ({"error": "HTTP 500"}, tr_wrong, "INCONCLUSIVE(full:error)"),
    ]
    for f, t, want in cases:
        got = verdict("C", f, t)
        if got != want:
            bad += 1
            print(f"  ✗ verdict {f} {t}: got {got} want {want}")
    if len({verdict("C", f, t) for f, t, _ in cases[:4]}) != 4:
        bad += 1
        print("  ✗ verdict not discriminative over the 4 core combos")
    n = len(PARSE_POS) + len(PARSE_NEG) + 1 + len(cases) + 1
    print(f"selftest: {n - bad}/{n} checks passed")
    return 1 if bad else 0


# ---------------------------------------------------------------- prompts & tokens
_tok = None
_enc = None


def local_tokens(prompt: str) -> int | None:
    """Local count of the chat-rendered prompt (encoding.py chat mode + tokenizer).
    Only a cross-check; the authority is the server's usage.prompt_tokens."""
    global _tok, _enc
    try:
        if _tok is None:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained(TOKENIZER)
            sys.path.insert(0, ENCODING_DIR)
            from encoding import encode_messages
            _enc = encode_messages
        rendered = _enc([{"role": "user", "content": prompt}], thinking_mode="chat")
        return len(_tok.encode(rendered, add_special_tokens=False))
    except Exception as exc:  # noqa: BLE001
        print(f"  (local tokenizer unavailable: {type(exc).__name__}: {exc})", file=sys.stderr)
        return None


def truncate_tail(doc: str, keep_tokens: int) -> str:
    """Keep only the last `keep_tokens` tokens of `doc` (token-exact), then drop the
    partial first line/sentence so the arm does not start mid-word."""
    global _tok
    if _tok is None:
        local_tokens("x")
    ids = _tok.encode(doc, add_special_tokens=False)
    if len(ids) <= keep_tokens:
        return doc
    tail = _tok.decode(ids[-keep_tokens:])
    cut = -1
    for sep in ("\n", ". ", "。"):
        j = tail.find(sep)
        if 0 <= j < 400 and (cut < 0 or j < cut):
            cut = j + len(sep)
    return tail[cut:] if cut > 0 else tail


def build_prompt(item: dict, doc: str) -> str:
    c = item["choices"]
    return TEMPLATE.format(document=doc, question=item["question"],
                           A=c["A"], B=c["B"], C=c["C"], D=c["D"])


def probe_in(doc: str, probe: str) -> bool:
    pat = re.compile(r"\s+".join(re.escape(w) for w in probe.split()))
    return pat.search(doc) is not None


# ---------------------------------------------------------------- HTTP
#: 容器里有 http_proxy=127.0.0.1:1057，urllib 会照用它 ⇒ 访问本机服务「超时而不是报错」
#: （指南 §1.3 同一颗雷）。本机请求一律绕过代理。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(base: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with _OPENER.open(req, timeout=timeout) as resp:
        return json.load(resp)


def ask(base: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    }
    t0 = time.time()
    try:
        payload = post(base, body, timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        return {"error": f"HTTP {exc.code}: {detail}", "seconds": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "seconds": round(time.time() - t0, 2)}
    secs = round(time.time() - t0, 2)
    try:
        choice = payload["choices"][0]
        text = choice["message"].get("content") or ""
        usage = payload.get("usage") or {}
    except (KeyError, IndexError, TypeError) as exc:
        return {"error": f"malformed response: {exc}", "seconds": secs}
    return {
        "text": text, "pred": parse_answer(text),
        "finish_reason": choice.get("finish_reason"),
        # never default to 0: a missing field must stay None (and make the verdict INCONCLUSIVE)
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seconds": secs,
    }


def preflight(base: str, model: str, timeout: float) -> str | None:
    """O16 lesson: 'startup complete' + /v1/models 200 does not mean serving."""
    r = ask(base, model, "1+1=? Reply with the number only.", 8, timeout)
    if r.get("error"):
        return r["error"]
    return None


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8010")))
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "dsv41"))
    ap.add_argument("--items", default=ITEMS)
    ap.add_argument("--ids", default="main", help="main | all | comma-separated ids")
    ap.add_argument("--tail-tokens", type=int, default=12000)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("REQ_TIMEOUT", "3600")))
    ap.add_argument("--out", default=os.environ.get("OUT", "/dev/shm/longctx_check.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--niah", action="store_true", help="run cookbook/niah_scan.py afterwards")
    ap.add_argument("--niah-lengths", default=os.environ.get("NIAH_LENGTHS", "16500,24000,30000"))
    ap.add_argument("--niah-depths", default=os.environ.get("NIAH_DEPTHS", "0,50,100"))
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    # parser must prove it discriminates before any reading is trusted
    if selftest() != 0:
        print("❌ 解析/判定自测失败，拒绝运行", file=sys.stderr)
        return 2

    base = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    data = json.load(open(args.items))
    items = data["items"]
    if args.ids == "main":
        items = [x for x in items if x["role"] == "main"]
    elif args.ids != "all":
        want = args.ids.split(",")
        items = [x for x in items if x["id"] in want]
    print(f"items={len(items)}  base={base}  model={args.model}  tail={args.tail_tokens}  "
          f"max_tokens={args.max_tokens}  timeout={args.timeout}s  T=0.0 thinking=off")
    print(f"数据：{data['source']}\n")

    if not args.dry_run and not args.skip_preflight:
        err = preflight(base, args.model, min(args.timeout, 120))
        if err:
            print(f"❌ 预检（1+1=?）失败：{err} —— 服务没在服务（见 O16），不跑长请求")
            return 3
        print("✓ 预检 1+1=? 通过\n")

    results = []
    for it in items:
        full_doc = it["doc"]
        tail_doc = truncate_tail(full_doc, args.tail_tokens)
        rec = {"id": it["id"], "domain": it["domain"], "difficulty": it["difficulty"],
               "target": it["target"], "evidence_probe": it["evidence_probe"],
               "probe_dist_from_end_tok": it["probe_dist_from_end_tok"],
               "probe_in_full": probe_in(full_doc, it["evidence_probe"]),
               "probe_in_trunc": probe_in(tail_doc, it["evidence_probe"])}
        if not rec["probe_in_full"]:
            rec["warning"] = "probe missing from full doc"
        if rec["probe_in_trunc"]:
            rec["warning"] = "probe survives truncation ⇒ 构造上无区分力"
        for arm, doc in (("full", full_doc), ("trunc", tail_doc)):
            prompt = build_prompt(it, doc)
            local = local_tokens(prompt)
            if args.dry_run:
                r = {"pred": None, "dry_run": True}
            else:
                print(f"  → {it['id']} {arm:5} (local {local} tok) ...", flush=True)
                r = ask(base, args.model, prompt, args.max_tokens, args.timeout)
            r["local_prompt_tokens"] = local
            if r.get("prompt_tokens") is not None and local is not None:
                r["server_minus_local"] = r["prompt_tokens"] - local
            rec[arm] = r
            if not args.dry_run:
                print(f"    {arm:5} pred={r.get('pred')} finish={r.get('finish_reason')} "
                      f"server={r.get('prompt_tokens')} local={local} {r.get('seconds')}s "
                      f"{('ERR ' + r['error']) if r.get('error') else repr((r.get('text') or '')[:80])}",
                      flush=True)
        rec["verdict"] = ("DRY_RUN" if args.dry_run
                          else verdict(it["target"], rec["full"], rec["trunc"]))
        if rec.get("warning") and rec["verdict"] == "DISCRIMINATIVE_PASS":
            rec["verdict"] = "NO_DISCRIMINATION(probe_in_tail)"
        results.append(rec)

    # ------------------------------------------------------------ table
    print("\n| id | 领域 | 证据距结尾(tok) | 完整 prompt tok (服务/本地) | 截断 prompt tok (服务/本地) "
          "| 标准 | 完整 | 截断 | finish(完整/截断) | 耗时 s(完整/截断) | 判定 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        f, t = r["full"], r["trunc"]
        print(f"| {r['id']} | {r['domain']} | {min(r['probe_dist_from_end_tok'])} "
              f"| {f.get('prompt_tokens')}/{f.get('local_prompt_tokens')} "
              f"| {t.get('prompt_tokens')}/{t.get('local_prompt_tokens')} | {r['target']} "
              f"| {f.get('pred')} | {t.get('pred')} | {f.get('finish_reason')}/{t.get('finish_reason')} "
              f"| {f.get('seconds')}/{t.get('seconds')} | {r['verdict']} |")
        if r.get("warning"):
            print(f"|  ⚠ {r['warning']} |")
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    npass = counts.get("DISCRIMINATIVE_PASS", 0)
    nfw = sum(v for k, v in counts.items() if k.startswith("FULL_WRONG"))
    nnd = sum(v for k, v in counts.items() if k.startswith("NO_DISCRIMINATION"))
    ninc = sum(v for k, v in counts.items() if k.startswith("INCONCLUSIVE"))
    print(f"\n判定计数：{counts}")
    for r in results:
        if r["verdict"].startswith("FULL_WRONG"):
            print(f"  ✗ 完整臂答错 {r['id']}: pred={r['full'].get('pred')} target={r['target']} "
                  f"text={r['full'].get('text', '')[:200]!r}")

    if args.dry_run:
        conclusion = "DRY RUN —— 未发请求，无结论。"
    else:
        max_pt = max((r["full"].get("prompt_tokens") or 0) for r in results) if results else 0
        conclusion = (
            f"结论：单卡 MoE offload 服务上，LongBench-v2 真题 {len(results)} 道（完整 prompt 最长 "
            f"{max_pt} tok，服务 usage 计），{npass} 道「完整对 & 只留尾部 {args.tail_tokens} tok 时错」"
            f"⇒ {'模型确实用到了尾部之外（>16k 区间）的内容，>16k 功能验通' if npass else '没有一道题证明用到了尾部之外的内容，>16k 功能【未】验通'}；"
            f"{nnd} 道无区分力（不计入证据），{nfw} 道完整臂答错（单题不下结论），{ninc} 道不可判。"
            f"T=0、thinking 关、每臂单发 ⇒ 每道 PASS 仍有 ~25% 的「完整臂蒙对」概率，"
            f"证据强度随 PASS 题数叠加。{O15_CAVEAT}")
    print("\n" + conclusion)

    out = {"config": {"base": base, "model": args.model, "tail_tokens": args.tail_tokens,
                      "max_tokens": args.max_tokens, "temperature": 0.0, "top_p": 1.0,
                      "thinking": False, "threshold": THRESHOLD, "items_file": args.items},
           "results": results, "counts": counts, "conclusion": conclusion,
           "o15_caveat": O15_CAVEAT}
    with open(args.out, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"读数 → {args.out}")

    rc = 0
    if args.niah and not args.dry_run:
        env = dict(os.environ, PORT=str(args.port), LENGTHS=args.niah_lengths,
                   DEPTHS=args.niah_depths, MODEL=args.model,
                   OUT=os.environ.get("NIAH_OUT", "/dev/shm/niah_singlecard.json"))
        if args.base_url:
            env["BASE_URL"] = args.base_url
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
            env.pop(k, None)  # niah_scan uses plain urllib, which honours the proxy
        print(f"\n▶ niah_scan LENGTHS={env['LENGTHS']} DEPTHS={env['DEPTHS']} OUT={env['OUT']}")
        rc = subprocess.call([sys.executable, NIAH], env=env)
    errors = sum(1 for r in results for a in ("full", "trunc") if r[a].get("error"))
    return 1 if (errors or rc) else 0


if __name__ == "__main__":
    raise SystemExit(main())
