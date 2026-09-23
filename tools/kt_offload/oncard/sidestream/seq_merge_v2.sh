#!/usr/bin/env bash
# seq_merge_v2.sh <RUN dir> <card> <STEPS_C> <STEPS_D> <rounds>
#   第二版合版本验收（用户 2026-09-23 10:40 定）：
#     C = 已验收的第一版（侧流 + 事件等待）
#     D = C + 共享专家挪进窗口（§4.31，VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW=1）
#   交错 C1,D1,C2,D2,...；任一臂非 0 就停在那一臂（不跳过、不补跑同名臂）。
#
# 与 seq_merge.sh 的区别，也是这份脚本单独存在的理由：
#   · seq_merge.sh 两臂一起开/关侧流 + P1a，cfg 必然差 overlap_side_stream 一行；
#   · 本脚本两臂【都】开侧流 + P1a，只差一个纯 env ⇒ cfg 必须【逐字节相同】（cmp -s），
#     并且还要与第一版 B 臂的 cfg 逐字节相同（C 就是第一版 B 那一臂）。
#
# 每臂：SIDE=1 P1A=1 P1B=0，SIW = C:0 / D:1，REQUIRE_CLEAN_TREE=1（起服务前本树 git status
# 必须为空、指纹 dirty_files 两棵树都 0 —— 第一版 A4 的教训）。每臂起飞前做占卡预检（同 seq_merge.sh）。
# 另：RUNTIME_BASE（默认 d1c5b7faf = 验收交付的 HEAD）—— 主会话会往本树提交文档、HEAD 会动；
#     每臂核 `git diff --name-only $RUNTIME_BASE HEAD -- vllm_ascend/` 为空，否则拒飞。
#
# 用法：
#   bash seq_merge_v2.sh $RUN 2 perf perf 3                          # C1,D1,C2,D2,C3,D3
#   ARM_OFFSET=3 bash seq_merge_v2.sh $RUN 2 lp lp,gsm,gsm1319if50 1 # C4,D4
#   ARM_OFFSET=4 bash seq_merge_v2.sh $RUN 2 lp lp 1                 # C5,D5（仅当 C4 vs D4 有差）
#   rounds=0 ⇒ 只做 cfg 断言（干跑）。
set -u
set -o pipefail   # 🔴 承重：否则 `card_clear | tee` 的 || 判的是 tee 的退出码（见 seq_merge.sh）
RUN="${1:?usage}"; CARD="${2:?card}"; STEPS_C="${3:?}"; STEPS_D="${4:?}"; ROUNDS="${5:-3}"; OFF="${ARM_OFFSET:-0}"
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ "$CARD" = 3 ] && { echo "❌ 绝不用卡 3"; exit 2; }

# ── cfg 断言：C 与 D 逐字节相同，且与第一版 B 臂逐字节相同 ─────────────────
cmp -s "$RUN/cfg_C.json" "$RUN/cfg_D.json" || { echo "❌ cfg_C.json 与 cfg_D.json 不是逐字节相同 —— 两臂就不止差那一个 env 了"; exit 2; }
cmp -s "$RUN/cfg_C.json" "$RUN/cfg_v1B.json" || { echo "❌ cfg_C.json 与第一版 B 臂 cfg（cfg_v1B.json）不逐字节相同 —— C 就不是已验收的那一版"; exit 2; }
grep -q '"overlap_side_stream" *: *true' "$RUN/cfg_C.json" || { echo "❌ cfg 的 overlap_side_stream 不是 true"; exit 2; }
grep -q '"packed_transfer" *: *false' "$RUN/cfg_C.json" || { echo "❌ cfg 的 packed_transfer 不是 false"; exit 2; }
echo "✅ cfg 断言过：cfg_C = cfg_D = cfg_v1B（cmp -s）"

# ── 每臂起飞前的占卡预检（与 seq_merge.sh:card_clear 逐行同口径）────────────
PF_N="${PF_N:-5}"; PF_GAP="${PF_GAP:-8}"; PF_MAXPCT="${PF_MAXPCT:-8}"; PF_WAIT="${PF_WAIT:-1200}"
card_clear() {
  local c=$1 t0 umax amax ufirst ulast series i r u a
  t0=$SECONDS
  while :; do
    umax=-1; amax=-1; ufirst=-1; ulast=-1; series=""
    for i in $(seq 1 "$PF_N"); do
      r=$(timeout 20 npu-smi info -t usages -i "$c" 2>/dev/null)
      u=$(grep -oE 'HBM Usage Rate\(%\)[^0-9]+[0-9]+' <<<"$r" | grep -oE '[0-9]+$')
      a=$(grep -oE 'Aicore Usage Rate\(%\)[^0-9]+[0-9]+' <<<"$r" | grep -oE '[0-9]+$')
      [ -z "$u" ] || [ -z "$a" ] && { u=99; a=99; }   # 🔴 读不到 ≠ 空闲
      [ "$u" -gt "$umax" ] && umax=$u; [ "$a" -gt "$amax" ] && amax=$a
      [ "$i" -eq 1 ] && ufirst=$u; ulast=$u; series="$series $u/$a"
      [ "$i" -lt "$PF_N" ] && sleep "$PF_GAP"
    done
    echo "  ⚪ [$(date +%T)] 预检卡 $c 序列(HBM/Aicore):$series ⇒ HBM峰值 ${umax}% Aicore峰值 ${amax}% 趋势 ${ufirst}→${ulast}"
    if [ "$umax" -le "$PF_MAXPCT" ] && [ "$ulast" -le $((ufirst + 3)) ]; then return 0; fi
    [ $((SECONDS - t0)) -ge "$PF_WAIT" ] && { echo "🔴 卡 $c 等了 $((SECONDS - t0))s 仍不干净 —— 拒绝起飞"; return 1; }
    echo "  ⏳ 卡 $c 还不干净，等 30s 再采（已等 $((SECONDS - t0))s / 上限 ${PF_WAIT}s）"; sleep 30
  done
}

for r in $(seq 1 "$ROUNDS"); do
  for X in C D; do
    ARM="$X$((OFF + r))"
    [ -e "$RUN/$ARM/arm_summary.txt" ] && { echo "❌ $RUN/$ARM 已经有读数了 —— 重名会静默覆盖出处"; exit 2; }
    card_clear "$CARD" | tee -a "$RUN/preflight_arms.txt" || {
      echo "ARM $ARM 预检未通过，未起飞" >> "$RUN/seq.txt"; echo "STOPPED at $ARM (preflight)" >> "$RUN/seq.txt"; exit 6; }
    df -h / | tail -1 | sed "s/^/[$ARM 起飞前 df] /" >> "$RUN/seq.txt"
    W=$([ "$X" = D ] && echo 1 || echo 0); S=$([ "$X" = D ] && echo "$STEPS_D" || echo "$STEPS_C")
    SIDE=1 P1A=1 P1B=0 SIW=$W REQUIRE_CLEAN_TREE=1 RUNTIME_BASE=${RUNTIME_BASE:-d1c5b7faf} bash "$T/run_arm.sh" "$RUN" "$ARM" "$CARD" "$S"; rc=$?
    echo "ARM $ARM card=$CARD SIDE=1 P1A=1 P1B=0 SIW=$W steps=$S rc=$rc [$(date +%T)]" >> "$RUN/seq.txt"
    [ $rc = 0 ] || { echo "STOPPED at $ARM" >> "$RUN/seq.txt"; exit $rc; }
  done
done
echo "ALLDONE off=$OFF rounds=$ROUNDS" >> "$RUN/seq.txt"
