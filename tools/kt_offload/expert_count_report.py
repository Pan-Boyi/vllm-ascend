#!/usr/bin/env python3
"""汇总 dsv41_expert_counter 的产物(单卡 MoE offload 线 M0c)。

用法:
  python3 expert_count_report.py <计数目录>                  # 汇总一份:判据 + [40,384] + 每层 top-K
  python3 expert_count_report.py <目录A> --compare <目录B>    # 再报两份 top-K 的逐层重合度
判据(逐 rank、逐层、逐 bucket):counts.sum() == tokens * topk;任何一格不满足 ⇒ 退出码 1。
覆盖量和结论一起报:各 rank 的 steps 与 skipped 原样打印。
产出:<目录>/expert_counts_total.pt  {counts:{decode,prefill,all}:[40,384], tokens:{…}:[40]}
      <目录>/expert_topk.tsv          层 \t bucket \t top-K 专家号(按频次降序)
"""
import argparse, glob, os, sys
import torch


def load(d):
    files = sorted(glob.glob(os.path.join(d, "rank*_pid*.pt")))
    if not files:
        sys.exit(f"❌ {d} 下没有 rank*_pid*.pt —— 计数器没开(VLLM_ASCEND_DSV41_EXPERT_COUNT)或没 dump")
    ranks = [torch.load(f) for f in files]
    bad = 0
    print(f"=== {d}: {len(files)} 个 rank 文件 ===")
    for f, r in zip(files, ranks):
        print(f"  {os.path.basename(f)}  steps={r['steps']}  skipped={r['skipped']}")
        for b in ("decode", "prefill"):
            s = r["counts"][b].sum(dim=1)
            t = r["tokens"][b] * r["topk"]
            wrong = (s != t).nonzero().flatten().tolist()
            if wrong:
                bad += 1
                print(f"   ❌ {b} 层 {wrong[:8]} 计数和≠token×{r['topk']}")
    tot = {b: sum(r["counts"][b] for r in ranks) for b in ("decode", "prefill")}
    tok = {b: sum(r["tokens"][b] for r in ranks) for b in ("decode", "prefill")}
    tot["all"], tok["all"] = tot["decode"] + tot["prefill"], tok["decode"] + tok["prefill"]
    print(f"  合计 token(按层 0):decode={int(tok['decode'][0])} prefill={int(tok['prefill'][0])}")
    if int(tok["all"].min()) == 0:
        print("  ⚠️ 有层计入 token 为 0 —— 覆盖不全,top-K 对该层无意义")
    return tot, tok, bad


def topk(counts, k):
    return [row.argsort(descending=True)[:k].tolist() for row in counts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--compare"); ap.add_argument("-k", type=int, default=100)
    a = ap.parse_args()
    tot, tok, bad = load(a.dir)
    torch.save({"counts": tot, "tokens": tok}, os.path.join(a.dir, "expert_counts_total.pt"))
    with open(os.path.join(a.dir, "expert_topk.tsv"), "w") as f:
        for b in ("decode", "prefill", "all"):
            for L, row in enumerate(topk(tot[b], a.k)):
                f.write(f"{L}\t{b}\t{','.join(map(str, row))}\n")
    share = (tot["all"].sort(dim=1, descending=True).values[:, :a.k].sum(1).double()
             / tot["all"].sum(1).clamp(min=1).double())
    print(f"  top-{a.k} 专家承接的路由占比(按层):中位 {share.median():.3f}  最低 {share.min():.3f} (层 {int(share.argmin())})")
    if a.compare:
        tot2, _, bad2 = load(a.compare); bad += bad2
        ov = [len(set(x) & set(y)) / a.k for x, y in zip(topk(tot["all"], a.k), topk(tot2["all"], a.k))]
        o = torch.tensor(ov)
        print(f"  两份 top-{a.k} 重合度(按层):中位 {o.median():.3f}  最低 {o.min():.3f} (层 {int(o.argmin())})")
    print("✅ 判据全部通过" if not bad else f"❌ {bad} 个 rank×bucket 不满足判据")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
