#!/usr/bin/env python3
"""CPU-only: extract LongBench-v2 items with FULL original context from the
evalscope review files of the 09-08 run (pre middle-truncation patch, so the
<text> block is the untruncated dataset context), count tokens with the
DSV4.1 tokenizer, locate answer evidence heuristically, write candidates.json.

Sources (read-only):
  eval/f_longbench_v2/20260908_162336/reviews/DeepSeek-V4-Flash-0731/*.jsonl
      -> full prompt + target + that run's prediction (V4-Flash-0731, T=1.0,
         nothink, max_tokens 32768 -- a WEAK prior only, other model/protocol)
  eval/lb_v2_official/20260909_151655/reviews/... -> second weak prior
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter

ROOT = "/workspace/code/dsv4_single_card/eval"
RUN_A = f"{ROOT}/f_longbench_v2/20260908_162336/reviews/DeepSeek-V4-Flash-0731"
RUN_B = f"{ROOT}/lb_v2_official/20260909_151655/reviews/DeepSeek-V4-Flash-0731"
TOK = "/workspace/models/DeepSeek-V4.1-Flash"
OUT = sys.argv[1] if len(sys.argv) > 1 else "candidates.json"
LO, HI = 16000, 34000  # doc-token window to keep (final pick narrows to 18k-30k prompt)

PROMPT_RE = re.compile(
    r"^Please read the following text and answer the questions below\.\n\n<text>\n"
    r"(?P<doc>.*)\n</text>\n\nAnswer the following multiple choice question\..*?"
    r"Think step by step before answering\.\n\n(?P<q>.*?)\n\nA\) (?P<A>.*?)\nB\) (?P<B>.*?)"
    r"\nC\) (?P<C>.*?)\nD\) (?P<D>.*)$", re.S)

STOP = set("""a an the of to in on for and or is are was were be been by with as at from that this
these those it its which what who whom how why when where does do did not no can could would should
will may might than then there their they them he she his her we our you your i me my about into
over under between among following according text document based most more less also only other
which while such all any each both above below one two first second""".split())


def words(s: str) -> list[str]:
    s = s.lower()
    toks = re.findall(r"[a-z0-9][a-z0-9\-']+", s)
    toks = [t for t in toks if t not in STOP and len(t) > 2]
    # CJK: character bigrams
    cjk = re.findall(r"[一-鿿]+", s)
    for run in cjk:
        toks += [run[i:i + 2] for i in range(len(run) - 1)]
    return toks


def load(path: str) -> dict:
    out = {}
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            md = d["sample_score"]["sample_metadata"]
            out[md["_id"]] = d
    return out


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    cands = []
    for sub in ("short", "medium"):
        a = load(f"{RUN_A}/longbench_v2_{sub}.jsonl")
        try:
            b = load(f"{RUN_B}/longbench_v2_{sub}.jsonl")
        except FileNotFoundError:
            b = {}
        for _id, d in a.items():
            prompt = d["messages"][0]["content"]
            m = PROMPT_RE.match(prompt)
            if not m:
                print("unparsed", _id, file=sys.stderr)
                continue
            doc = m["doc"]
            if len(doc) < LO * 2 or len(doc) > HI * 6:  # cheap char prefilter
                continue
            ids = tok.encode(doc, add_special_tokens=False)
            n = len(ids)
            if not (LO <= n <= HI):
                continue
            md = d["sample_score"]["sample_metadata"]
            target = d["target"]
            sc = d["sample_score"]["score"]
            rb = b.get(_id)
            # ---- evidence localisation (heuristic) ----
            # score 512-token chunks by idf-weighted overlap with question+correct
            # choice terms that do NOT appear in the wrong choices.
            chunk = 512
            chunks = [tok.decode(ids[i:i + chunk]) for i in range(0, n, chunk)]
            cw = [Counter(words(c)) for c in chunks]
            df = Counter()
            for c in cw:
                df.update(set(c))
            q_terms = set(words(m["q"]))
            right = set(words(m[target]))
            wrong = set()
            for L in "ABCD":
                if L != target:
                    wrong |= set(words(m[L]))
            terms = (q_terms | (right - wrong))
            N = len(chunks)
            scores = []
            for c in cw:
                s = sum(math.log(1 + N / df[t]) for t in terms if t in c and df[t] < N)
                scores.append(s)
            order = sorted(range(N), key=lambda i: -scores[i])
            top = order[:3]
            best = order[0] if order else 0
            # tokens from the start of best chunk to end of doc
            dist_from_end = n - best * chunk
            share_in_tail = sum(scores[i] for i in range(N) if (n - i * chunk) <= 12000)
            share_total = sum(scores) or 1.0
            cands.append({
                "id": _id, "subset": sub, "domain": md["domain"],
                "sub_domain": md["sub_domain"], "difficulty": md["difficulty"],
                "target": target, "doc_tokens": n,
                "question": m["q"], "choices": {L: m[L] for L in "ABCD"},
                "doc": doc,
                "prior_A": {"pred": sc.get("extracted_prediction"),
                            "correct": sc["value"].get("accuracy")},
                "prior_B": ({"pred": rb["sample_score"]["score"].get("extracted_prediction"),
                             "correct": rb["sample_score"]["score"]["value"].get("accuracy")}
                            if rb else None),
                "evidence": {"best_chunk_start_tok": best * chunk,
                             "best_dist_from_end_tok": dist_from_end,
                             "top3_dist_from_end_tok": [n - i * chunk for i in top],
                             "top3_scores": [round(scores[i], 2) for i in top],
                             "tail12k_score_frac": round(share_in_tail / share_total, 3),
                             "n_terms": len(terms)},
            })
            print(f"{_id} {sub:6} {md['domain'][:22]:22} {n:6} tgt={target} "
                  f"A={sc.get('extracted_prediction')} best_dist={dist_from_end}",
                  file=sys.stderr, flush=True)
    with open(OUT, "w") as fh:
        json.dump(cands, fh, ensure_ascii=False)
    print(f"{len(cands)} candidates -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
