"""Apply the PLAN §4.27 benefit criterion to a RUN directory, mechanically.

usage: perf_verdict.py <RUN dir> [--rounds N] [--arms XY] [--signed]

``--arms CD`` (second merge acceptance, C = baseline, D = candidate): the
first letter is always the baseline, so Δ = mean(first) − mean(second) keeps
its sign convention (positive = the candidate is faster). Default ``AB``.

Why a script and not arithmetic in the report: the criterion is
``|A-B| >= 3 x floor AND |A-B| >= 0.3 ms/token``, where the floor is THIS run's
own same-arm swing across rounds. Every time that has been done by hand the
temptation was to drop "the noisy round" or to read the floor off the other
arm. Here the inputs are fixed (every round of every arm, no exclusions) and
the verdict prints before anyone gets to interpret it.

🔴 The reading is the ``decode ms/token(TPOT 中位)`` row's mean -- the mean over
REPEAT rounds of each round's MEDIAN TPOT. Not the ``TPOT 均值`` row: that one is
a mean of means and carries the tail (§4.30: sigma has outliers of 3.09/4.74/7.27).
§4.28 and §4.29 both quote the median row, so quoting the other one would make
this run incomparable with them.

🔴 A missing arm is an error, never a skipped row. Six arms were declared in
EXPECTED.txt; averaging over five and calling it a verdict is exactly the shape
of error this file exists to stop. It exits 3 for that, not 1 -- "one arm never
produced a reading" and "the criterion was applied and came out negative" are
different outcomes and must not share an exit code.
"""
import re
import statistics as st
import sys

ROW = "decode ms/token(TPOT 中位)"


def read_arm(run: str, arm: str) -> tuple[float, float, float, int]:
    """(mean, min, max, n) of the median-TPOT row for one arm."""
    path = f"{run}/{arm}/perf/summary.txt"
    try:
        fh = open(path, encoding="utf-8")
    except OSError as exc:
        print(f"❌ 臂 {arm} 没有 perf/summary.txt（{exc}）—— 少一臂就没有判定，"
              "不许用剩下的臂凑一个均值")
        raise SystemExit(3) from exc
    with fh:
        for line in fh:
            if ROW not in line:
                continue
            nums = re.findall(r"-?\d+\.\d+", line)
            n = re.search(r"\(n=(\d+)\)", line)
            if len(nums) < 4 or not n:
                print(f"❌ {path}: 这一行解析不出四个数 + n=：{line.strip()}")
                raise SystemExit(3)
            mean, _sd, lo, hi = (float(x) for x in nums[:4])
            return mean, lo, hi, int(n.group(1))
    print(f"❌ {path}: 没有 '{ROW}' 这一行 —— 这一臂没有读数，不能当成 0 或跳过")
    raise SystemExit(3)


def main() -> None:
    run = sys.argv[1]
    rounds = 3
    if "--rounds" in sys.argv:
        rounds = int(sys.argv[sys.argv.index("--rounds") + 1])
    pair = "AB"
    if "--arms" in sys.argv:
        pair = sys.argv[sys.argv.index("--arms") + 1]
        if len(pair) != 2 or pair[0] == pair[1]:
            print(f"❌ --arms 要两个不同字母（基线在前），得到 {pair!r}")
            raise SystemExit(3)
    P, Q = pair
    arms = {X: [f"{X}{r}" for r in range(1, rounds + 1)] for X in pair}
    vals: dict[str, list[float]] = {}
    print(f"== §4.27 判定  run={run}  轮次={rounds}  读数行='{ROW}' 的均值")
    for X in pair:
        vals[X] = []
        for arm in arms[X]:
            mean, lo, hi, n = read_arm(run, arm)
            vals[X].append(mean)
            print(f"  {arm}: {mean:7.2f}  (轮内 {lo:.2f}–{hi:.2f}, n={n})")
    mA, mB = st.mean(vals[P]), st.mean(vals[Q])
    fA = max(vals[P]) - min(vals[P])
    fB = max(vals[Q]) - min(vals[Q])
    floor = max(fA, fB)
    delta = mA - mB
    print(f"  {P} 臂均值 {mA:.3f}  ({' / '.join(f'{v:.2f}' for v in vals[P])})  同臂摆动 {fA:.3f}")
    print(f"  {Q} 臂均值 {mB:.3f}  ({' / '.join(f'{v:.2f}' for v in vals[Q])})  同臂摆动 {fB:.3f}")
    print(f"  Δ = {P} − {Q} = {delta:+.3f} ms/token   （正 = {Q} 更快）")
    print(f"  同臂地板 floor = max({fA:.3f}, {fB:.3f}) = {floor:.3f}")
    # --signed（第二版验收起用）：收益 = 候选臂【更快】，Δ 按带符号判。|Δ| 的旧写法下
    # 候选臂慢 3×地板 也会打出「收益成立」—— 前几发 Δ 都为正所以没咬到，第二版不赌这个。
    signed = "--signed" in sys.argv
    d, dn = (delta, "Δ") if signed else (abs(delta), "|Δ|")
    c1 = d >= 3 * floor
    c2 = d >= 0.3
    print(f"  判据①  {dn} {d:+.3f} ≥ 3×floor {3*floor:.3f} ? {'✅' if c1 else '❌'}")
    print(f"  判据②  {dn} {d:+.3f} ≥ 0.3      ? {'✅' if c2 else '❌'}")
    ok = c1 and c2
    print(f"  ⇒ {'🟢 收益成立' if ok else '🔴 判负'}"
          + ("" if ok else "（按 EXPECTED.txt §4：不许调参数重试、不许放宽门槛）"))
    if not ok and not c1 and c2:
        print("     注：②过①不过 ⇒ 差值存在但被本发地板吃掉 —— 「判负」不等于「没有收益」，"
              "要不要另排安静时段是用户的决定。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
