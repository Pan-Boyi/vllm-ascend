"""Single-round (A1 vs B1) verdict, per EXPECTED_addendum_0936.txt.

usage: perf_verdict_single.py <RUN dir> [--floor2 0.96]

🔴 This is NOT the PLAN §4.27 verdict and must never be recorded as one.
§4.27's floor is the SAME ARM's swing across rounds/restarts, which a single
round does not measure. The addendum knowingly substitutes two stand-ins:

  floor1 = max over both arms of (that arm's per-round median max - min)
           -- spread WITHIN one service. §4.18 measured this at 0.06-0.19 ms
           while the true cross-restart floor was 0.24-1.0 ms, i.e. floor1 runs
           roughly 5-10x optimistic. It is not a restart floor.
  floor2 = a constant passed in, taken from |this run's A1 - the voided run's A1|.
           That is a CROSS-RUN, CROSS-CARD subtraction, which §4.25 forbids for
           readings; and it is one observation, not a distribution.

Both are stand-ins for a floor this run never measured. The verdict line says so
on every run, so a reader who sees only the output cannot mistake it for §4.27.

达预期 ⇔ Δ ≥ 0.3 且 Δ ≥ 3×floor1 且 Δ ≥ 3×floor2   (all three)
Exit 0 = 达预期 (stop interleaving, go to bitwise), 1 = 不达 (keep interleaving),
3 = an arm is missing / unparsable (never silently treated as either).
"""
import re
import sys

ROW = "decode ms/token(TPOT 中位)"


def read_arm(run: str, arm: str) -> tuple[float, float, float]:
    """(mean, min, max) of the median-TPOT row for one arm."""
    path = f"{run}/{arm}/perf/summary.txt"
    try:
        fh = open(path, encoding="utf-8")
    except OSError as exc:
        print(f"❌ 臂 {arm} 没有 perf/summary.txt（{exc}）")
        raise SystemExit(3) from exc
    with fh:
        for line in fh:
            if ROW in line:
                nums = re.findall(r"-?\d+\.\d+", line)
                if len(nums) < 4:
                    print(f"❌ {path}: 这一行解析不出四个数：{line.strip()}")
                    raise SystemExit(3)
                mean, _sd, lo, hi = (float(x) for x in nums[:4])
                return mean, lo, hi
    print(f"❌ {path}: 没有 '{ROW}' 这一行 —— 这一臂没有读数，不能当成 0 或跳过")
    raise SystemExit(3)


def main() -> None:
    run = sys.argv[1]
    floor2 = 0.96
    if "--floor2" in sys.argv:
        floor2 = float(sys.argv[sys.argv.index("--floor2") + 1])
    a_mean, a_lo, a_hi = read_arm(run, "A1")
    b_mean, b_lo, b_hi = read_arm(run, "B1")
    floor1 = max(a_hi - a_lo, b_hi - b_lo)
    delta = a_mean - b_mean
    print("== 单轮判定（EXPECTED_addendum_0936）—— 🔴 不是 §4.27 判定，地板为代用")
    print(f"  A1: {a_mean:7.2f}  (轮内 {a_lo:.2f}–{a_hi:.2f}, 摆动 {a_hi-a_lo:.2f})")
    print(f"  B1: {b_mean:7.2f}  (轮内 {b_lo:.2f}–{b_hi:.2f}, 摆动 {b_hi-b_lo:.2f})")
    print(f"  Δ = A1 − B1 = {delta:+.3f} ms/token   （正 = B 更快）")
    print(f"  floor1（同服务内摆动，代用）= {floor1:.3f} ⇒ 3×floor1 = {3*floor1:.3f}")
    print(f"  floor2（跨发跨卡一次观测，代用）= {floor2:.3f} ⇒ 3×floor2 = {3*floor2:.3f}")
    c = [delta >= 0.3, delta >= 3 * floor1, delta >= 3 * floor2]
    print(f"  ① Δ ≥ 0.3        {'✅' if c[0] else '❌'}")
    print(f"  ② Δ ≥ 3×floor1   {'✅' if c[1] else '❌'}")
    print(f"  ③ Δ ≥ 3×floor2   {'✅' if c[2] else '❌'}")
    ok = all(c)
    print(f"  ⇒ {'🟢 达预期 —— 停止 perf 交错，进入逐位' if ok else '🔴 不达 —— 按原 §4 继续 A2,B2,A3,B3，原 §5 判据照旧'}")
    print("  🔴 口径声明：**单轮、地板为代用**。本行不得写成「§4.27 收益成立」；"
          "本发没有测到同臂跨重启地板。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
