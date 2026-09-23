#!/usr/bin/env bash
# seq_merge.sh <RUN dir> [card] [steps] [rounds]
#   合版本（侧流 + P1a）对【现状基线】（两个开关都关 = §4.18 的串行形态）的总收益 A/B。
#   交错 A1,B1,A2,B2,...（默认 3 轮）；任一臂非 0 就停在那一臂。
#
# 与 seq_ab.sh / seq_p1a.sh / seq_p1b.sh 的区别，也是这份脚本单独存在的理由：
#   · seq_ab.sh   两臂只差侧流（P1A 恒 0）      —— 量的是侧流单独的账（§4.26）
#   · seq_p1a.sh  两臂【都】开侧流，只差 P1a    —— 量的是 P1a 单独的账（§4.28）
#   · seq_merge.sh 两臂【一起】开/关侧流 + P1a  —— 量的是【合版本的总收益】
#     ⇒ 这一发**故意**一次动两个开关。它不是在分谁的功劳（那两笔账已经分别结过），
#       而是回答「合起来相对现状基线值多少」。别拿它去反推单项收益。
#
# 🔴 cfg 不可能逐字节相同，且这不是本线的选择：
#     `kt_offload/config.py:_validate_preconditions` 在 `overlap_side_stream=true` 而
#     `VLLM_ASCEND_DSV41_KT_SIDE_STREAM` 未置 1 时直接 raise；而 `layer.py:221` 的
#     `self.overlap` 只读 cfg、不读 env ⇒ 侧流「关」必须由 cfg 关。
#     所以两臂的最小差 = cfg 的 `overlap_side_stream` 一行 + 两个 env。
#     下面断言的正是这个最小差：**除那一行外逐字节相同**，且 A=false / B=true。
#     （seq_p1a.sh 的 `cmp -s` 在这里会误拦，所以不能复用它。）
#
# 🔴 交错 ≥3 轮不是形式：判据（PLAN §4.27）是 |A−B| ≥ 3 × 同臂地板 且 ≥ 0.3 ms/token，
#    地板＝【本发】同臂跨轮的最大摆动。轮次不够 ⇒ 没有地板 ⇒ 按 §4.15③ 判 fail。
#    口径 `CASES="bs1 1 16" REPEAT=3` 不许砍半（§4.30）。
#
# 用法：
#   bash seq_merge.sh $RUN 0            # 默认 perf，3 轮 -> A1,B1,A2,B2,A3,B3
#   bash seq_merge.sh $RUN 0 perf 3
#   ARM_OFFSET=3 bash seq_merge.sh $RUN 0 lp 2   # -> A4,B4,A5,B5（逐位那四臂，不覆盖 perf 的臂目录）
#     🔴 ARM_OFFSET 存在的理由只有一个：臂目录名就是读数的出处，重名 = 后一发静默覆盖前一发。
#        不许为了省事把 lp 跑进已经有 perf 读数的臂目录里。
set -u
# 🔴 pipefail：下面 `card_clear ... | tee` 的 `||` 否则判的是 tee 的退出码，
#    而 tee 几乎永远成功 ⇒ 预检红了也会照样起飞，且看起来一切正常。
set -o pipefail
RUN="${1:?usage: seq_merge.sh <RUN dir> [card] [steps] [rounds]}"
CARD="${2:-0}"; STEPS="${3:-perf}"; ROUNDS="${4:-3}"; OFF="${ARM_OFFSET:-0}"
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ── cfg 断言：除 overlap_side_stream 一行外逐字节相同 ──────────────────────
strip() { sed 's/"overlap_side_stream" *: *\(true\|false\)/"overlap_side_stream":X/' "$1"; }
diff <(strip "$RUN/cfg_A.json") <(strip "$RUN/cfg_B.json") > /dev/null || {
  echo "❌ cfg_A.json 与 cfg_B.json 除 overlap_side_stream 外还有别的差异 —— 那样两臂就不止差合版本那两项了"
  diff <(strip "$RUN/cfg_A.json") <(strip "$RUN/cfg_B.json"); exit 2; }
grep -q '"overlap_side_stream" *: *false' "$RUN/cfg_A.json" || {
  echo "❌ cfg_A.json 的 overlap_side_stream 不是 false —— A 臂是【现状基线/串行】，侧流必须由 cfg 关"; exit 2; }
grep -q '"overlap_side_stream" *: *true' "$RUN/cfg_B.json" || {
  echo "❌ cfg_B.json 的 overlap_side_stream 不是 true —— B 臂是合版本，侧流必须开"; exit 2; }

# ── 每臂起飞前的占卡预检 ────────────────────────────────────────────────────
# 🔴 为什么每一臂都要预检，而不只在整段开头预检一次（2026-09-23 实测，不是推测）：
#    第一发 `20260923_082414_merge` 的 A2 起服务失败 ——
#      `ValueError: Free memory on device (96.31/123.16 GiB) on startup is less than
#       desired GPU memory utilization (0.9, 110.84 GiB)`
#    整段开头的预检是绿的（卡 0/1/2 全 4%/0%、无趋势），是**跑到一半**外部租户上来的：
#      B1 的 card_after（09:02:53）卡 0 = 16,056 MB **且卡 1 = 16,061 MB**，
#      而卡 1 我们从头到尾没碰过 ⇒ 卡 0 上那 16 GB 不是我们自己的拆解残留；
#      A2 的 card_before（09:05:03）涨到 17,079 / 18,109 MB，到 09:05:53 卡 0 已约 27 GiB。
#      ⇒ 外部租户正在卡 0+1 上【加载权重】，正是 cookbook 预检的「HBM 在上升」那一格。
#      同期卡 2 全程 5,253 MB 没动。
#    代价：每臂多约 40 秒。换掉的是「一臂死在起服务 + 整段交错作废」。
# 🔴 判的是【最大值与趋势】，不是某一瞬间：单采分不开「卡是空的」和「别人正在加载权重」
#    —— 加载那几分钟 HBM 可以低到和空闲一样（口径同 cookbook/serve_multicard.sh ③）。
# ⚪ 这一格只决定【什么时候起飞】，不碰口径、不碰判据、不碰任何被测的东西。
PF_N="${PF_N:-5}"; PF_GAP="${PF_GAP:-8}"; PF_MAXPCT="${PF_MAXPCT:-8}"; PF_WAIT="${PF_WAIT:-1200}"
card_clear() {  # $1=卡号；0=干净可起飞，1=等超时仍不干净
  local c=$1 t0 umax amax ufirst ulast series i r u a
  t0=$SECONDS
  while :; do
    umax=-1; amax=-1; ufirst=-1; ulast=-1; series=""
    for i in $(seq 1 "$PF_N"); do
      r=$(timeout 20 npu-smi info -t usages -i "$c" 2>/dev/null)
      u=$(grep -oE 'HBM Usage Rate\(%\)[^0-9]+[0-9]+' <<<"$r" | grep -oE '[0-9]+$')
      a=$(grep -oE 'Aicore Usage Rate\(%\)[^0-9]+[0-9]+' <<<"$r" | grep -oE '[0-9]+$')
      # 🔴 读不到 ≠ 空闲。当成「不干净」继续等，绝不放行。
      [ -z "$u" ] || [ -z "$a" ] && { u=99; a=99; }
      [ "$u" -gt "$umax" ] && umax=$u; [ "$a" -gt "$amax" ] && amax=$a
      [ "$i" -eq 1 ] && ufirst=$u; ulast=$u; series="$series $u/$a"
      [ "$i" -lt "$PF_N" ] && sleep "$PF_GAP"
    done
    echo "  ⚪ 预检卡 $c 序列(HBM/Aicore):$series ⇒ HBM峰值 ${umax}% Aicore峰值 ${amax}% 趋势 ${ufirst}→${ulast}"
    if [ "$umax" -le "$PF_MAXPCT" ] && [ "$ulast" -le $((ufirst + 3)) ]; then return 0; fi
    [ $((SECONDS - t0)) -ge "$PF_WAIT" ] && {
      echo "🔴 卡 $c 等了 $((SECONDS - t0))s 仍不干净（峰值 ${umax}% 趋势 ${ufirst}→${ulast}）——"
      echo "   HBM 高 + Aicore 非零 = 有人正在跑（别 kill 别 reset，去协调或等）；"
      echo "   HBM 高 + Aicore 全 0 = 上一发的拆解还没走完；HBM 在涨 = 有人正在加载权重。"
      echo "   拒绝起飞：抢进去几分钟后会两个进程一起 OOM，而且归因扯不清。"; return 1; }
    echo "  ⏳ 卡 $c 还不干净，等 30s 再采（已等 $((SECONDS - t0))s / 上限 ${PF_WAIT}s）"; sleep 30
  done
}

for r in $(seq 1 "$ROUNDS"); do
  for X in A B; do
    ARM="$X$((OFF + r))"
    [ -e "$RUN/$ARM/arm_summary.txt" ] && {
      echo "❌ $RUN/$ARM 已经有读数了 —— 重名会静默覆盖出处。换 ARM_OFFSET，别覆盖。"; exit 2; }
    card_clear "$CARD" | tee -a "$RUN/preflight_arms.txt" || {
      echo "ARM $ARM 预检未通过，未起飞" >> "$RUN/seq.txt"
      echo "STOPPED at $ARM (preflight)" >> "$RUN/seq.txt"; exit 6; }
    P=$([ "$X" = B ] && echo 1 || echo 0)   # 侧流 env 与 P1a env 一起走
    SIDE=$P P1A=$P P1B=0 bash "$T/run_arm.sh" "$RUN" "$ARM" "$CARD" "$STEPS"; rc=$?
    echo "ARM $ARM SIDE=$P P1A=$P P1B=0 rc=$rc" >> "$RUN/seq.txt"
    [ $rc = 0 ] || { echo "STOPPED at $ARM" >> "$RUN/seq.txt"; exit $rc; }
  done
done
echo ALLDONE >> "$RUN/seq.txt"
