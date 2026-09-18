#!/usr/bin/env python3
"""汇总 dsv41_expert_counter 的产物(单卡 MoE offload 线 M0c)。

用法:
  python3 expert_count_report.py <计数目录>                  # 汇总一份:判据 + [40,384] + 每层 top-K
  python3 expert_count_report.py <目录A> --compare <目录B>    # 再报两份 top-K 的逐层重合度
判据(逐 rank、逐层、逐 bucket):
  ① 和判据 counts.sum() == tokens * topk;任何一格不满足 ⇒ 退出码 1。
  ② 身份判据:同层同 bucket 下,两个都有 token 的 rank 的 [384] 计数完全相同 ⇒ 退出码 1 并点名。
     ①对"数错了哪几行"没有区分力(数别的 rank 的行,和照样对);ALLGATHER 切片 bug 的指纹是各 rank 计数相同。
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
    if identity_check([os.path.basename(f) for f in files], ranks):
        bad += 1
    tot = {b: sum(r["counts"][b] for r in ranks) for b in ("decode", "prefill")}
    tok = {b: sum(r["tokens"][b] for r in ranks) for b in ("decode", "prefill")}
    tot["all"], tok["all"] = tot["decode"] + tot["prefill"], tok["decode"] + tok["prefill"]
    print(f"  合计 token(按层 0):decode={int(tok['decode'][0])} prefill={int(tok['prefill'][0])}")
    if int(tok["all"].min()) == 0:
        print("  ⚠️ 有层计入 token 为 0 —— 覆盖不全,top-K 对该层无意义")
    return tot, tok, bad


def identity_check(names, ranks):
    """身份判据:同层同 bucket,两个都有 token 的 rank 计数完全相同 ⇒ 违例。返回违例格数。"""
    bad_cells, compared = [], 0
    for b in ("decode", "prefill"):
        n_layers = ranks[0]["counts"][b].shape[0]
        for L in range(n_layers):
            live = [i for i, r in enumerate(ranks) if int(r["tokens"][b][L]) > 0]
            if len(live) < 2:
                continue
            compared += 1
            same = [(names[i], names[j]) for k, i in enumerate(live) for j in live[k + 1:]
                    if torch.equal(ranks[i]["counts"][b][L], ranks[j]["counts"][b][L])]
            if same:
                bad_cells.append((b, L, same))
    print("  各 rank token 数(层 0):" + "  ".join(
        f"{n}: decode={int(r['tokens']['decode'][0])} prefill={int(r['tokens']['prefill'][0])}"
        for n, r in zip(names, ranks)))
    print(f"  身份判据:比较了 {compared} 个 层×bucket 格(≥2 个 rank 有 token),"
          f"其中 {len(bad_cells)} 格有 rank 两两计数完全相同")
    if compared == 0 and len(ranks) > 1:
        print("  ⚠️ 身份判据零覆盖:没有任何一格有 ≥2 个 rank 有 token")
    for b, L, same in bad_cells[:8]:
        print(f"   ❌ {b} 层 {L}:计数完全相同的 rank 对 {same[:6]}")
    return len(bad_cells)


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
    print("✅ 判据全部通过" if not bad else f"❌ {bad} 项判据不满足(和判据按 rank×bucket 计,身份判据按目录计)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
