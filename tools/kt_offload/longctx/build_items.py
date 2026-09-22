#!/usr/bin/env python3
"""CPU-only: from candidates.json (prep_candidates.py) write longctx_items.json,
the fixed item set run_longctx_check.py consumes. Each item carries an
`evidence_probe`: a verbatim substring of the doc that carries the answer
evidence. Its token distance from the doc end is computed here and the runner
re-checks that the probe is ABSENT from the truncated arm (otherwise the
control arm would not be a control)."""
import json
import re
import os
import sys

from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = "/workspace/models/DeepSeek-V4.1-Flash"

PICKS = [
    # (id, role, probe, why)
    ("66ebce825a08c7b9b35dfc75", "main", "64 6144 8.0 41.0%",
     "Nemotron-4-340B 报告 Table 2 的一行；四个选项 768/1536/3072/6144 全部出自同一张表"
     "（干扰项就是相邻行）⇒ 必须读到表才能选对；该表只出现一次。风险：模型可能凭参数记忆答出。"),
    ("66ec0f3c821e116aacb19d70", "main", "new APIs for efficient state checkpointing and restoration",
     "Metis/RefFS 论文（冷门，记忆风险低）；答案依据是引言里的一句"
     "「RefFS is an in-memory ... with new APIs for efficient state checkpointing」。"),
    ("66ec3a46821e116aacb1c49f", "main", "push the right coupling element upward and the left downward",
     "METTLER TOLEDO AG 天平说明书（记忆风险低）；问「哪条不对」，A 把左右手弄反了，"
     "依据是 §3.2 Adapting the draft shield 那一段。"),
    ("66ebc0c95a08c7b9b35de7f3", "main", "Github [4.5%]",
     "LLaMA 论文数据配比表 Github 4.5%。🔴 高记忆风险（名篇）——列入主测是有意的："
     "若截断臂也答对，正好演示对照臂能识别「靠记忆答对 ≠ 用到了长上下文」。"),
    ("6708a096bb02136c067d1789", "spare", "def allocate_chunks(self, size: int, file_path: str)",
     "代码仓库理解：答案由 Scheduler.allocate_chunks 的分配逻辑决定（按剩余容量降序贪心、"
     "每份副本排除上一份的首选机）。🔴 实测 probe 在距结尾 9,776 tok ⇒ 落在截断臂保留的尾部 12k 内 ⇒ 构造上无区分力，降为备选。无 thinking 下要在脑内模拟代码，答对率可能低 ⇒ 若完整臂答错，按规则只记录不下结论。"),
    # spares (not run by default)
    ("66ebc95f5a08c7b9b35df497", "spare", "Exiled God",
     "解释性题，依据集中在 ~27k from end。"),
    ("66fc0152bb02136c067c8826", "spare", "QULIPTA",
     "多文档计数（10 个药名里有几个治神经疾病），证据分散 11k–26k from end，部分在尾部 ⇒ 区分力预期弱。"),
]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(TOK)
    cands = {x["id"]: x for x in json.load(open(os.path.join(HERE, "candidates.json")))}
    items = []
    for _id, role, probe, why in PICKS:
        x = cands[_id]
        doc = x["doc"]
        n = x["doc_tokens"]
        # whitespace-insensitive (PDF-extracted docs break lines anywhere)
        pat = re.compile(r"\s+".join(re.escape(w) for w in probe.split()))
        pos = [m.start() for m in pat.finditer(doc)]
        dists = []
        for p in pos[:20]:
            dists.append(n - len(tok.encode(doc[:p], add_special_tokens=False)))
        if not pos:
            print(f"probe not found for {_id}: {probe!r}", file=sys.stderr)
            return 1
        items.append({
            "id": _id, "role": role, "domain": x["domain"], "sub_domain": x["sub_domain"],
            "difficulty": x["difficulty"], "target": x["target"], "question": x["question"],
            "choices": x["choices"], "doc": doc, "doc_tokens": n,
            "evidence_probe": probe, "probe_hits": len(pos),
            "probe_dist_from_end_tok": dists,
            "heuristic_evidence": x["evidence"], "why": why,
            "prior_weak": {"V4-Flash-0731 nothink T=1.0 run 09-08": x["prior_A"],
                           "V4-Flash-0731 run 09-09": x["prior_B"]},
        })
        print(f"{_id} {role:5} doc={n} probe_hits={len(pos)} "
              f"min_dist_from_end={min(dists)} max={max(dists)}")
    out = os.path.join(HERE, "longctx_items.json")
    json.dump({"source": "eval/f_longbench_v2/20260908_162336/reviews/DeepSeek-V4-Flash-0731/"
                         "longbench_v2_{short}.jsonl (full, untruncated <text> block)",
               "tokenizer": TOK, "items": items}, open(out, "w"), ensure_ascii=False)
    print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
