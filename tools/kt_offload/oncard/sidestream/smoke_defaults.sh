#!/usr/bin/env bash
# smoke_defaults.sh <CARD> [RUN]  —— PLAN §4.33「默认开」上卡冒烟（RUNBOOK_defaults_on.md）。
#   E1：cfg_E = 第二版 cfg_D 去掉 overlap_side_stream 一项（吃代码默认 true），四个开关【一个都不传】
#       （run_arm.sh DEFAULTS=1，env -u 剥掉父 shell 残留）→ probe + lp。
#   判：① 必检 SIDE STREAM / ASYNC WAIT / SHARED IN WINDOW 各 40、DEVICE SIGNAL 0、ACL 错码 0、
#          启动行「kt_offload switches」逐字 = 默认语义；② lp_verdict D4 vs E1（skip=13）全过。
#   E1 必检不过 ⇒ rc=5（run_arm）；lp 红 ⇒ rc=11，并按 RUNBOOK §3 补 F1（新代码 + 第二版显式开关）分因。
# 卡：调用方给；卡 3 硬拦。只用一张卡、一个服务、一次加载。
set -u
set -o pipefail   # 🔴 承重：否则 `card_clear | tee` 的 || 判的是 tee 的退出码（干跑实证：卡 2 HBM 69% 时照样起飞）
ROOT=/workspace/code/dsv4_single_card/vllm_project
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CARD="${1:?usage: smoke_defaults.sh <CARD> [RUN]}"
case "$CARD" in 0|1|2) ;; *) echo "❌ 卡只认 0/1/2（卡 3 不许用；得到 '$CARD'）"; exit 2;; esac
V2=${V2:-$ROOT/oncard_runs/20260923_105413_merge_v2}
RUN=${2:-$ROOT/oncard_runs/$(date +%Y%m%d_%H%M%S)_defaults_on}
for f in "$V2/D4/lp.jsonl" "$V2/prompts.jsonl" "$V2/cfg_D.json"; do
  [ -s "$f" ] || { echo "❌ $f 不在 —— 第二版产物未被 git 跟踪、会消失；没有 D4 就没有对照，不起飞"; exit 2; }
done
[ -e "$RUN/E1/arm_summary.txt" ] && { echo "❌ $RUN/E1 已经有读数了 —— 重名会静默覆盖出处"; exit 2; }
mkdir -p "$RUN"
cp "$V2/prompts.jsonl" "$RUN/prompts.jsonl"
python3 - "$V2/cfg_D.json" "$RUN/cfg_E.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
k = d["kt_offload_config"]
assert k.pop("overlap_side_stream") is True, "cfg_D 的 overlap_side_stream 不是 true"
assert k.get("packed_transfer") is False, "cfg_D 的 packed_transfer 不是 false"
json.dump(d, open(sys.argv[2], "w"), separators=(",", ":"))
EOF
[ $? = 0 ] || { echo "❌ 由 cfg_D 生成 cfg_E 失败"; exit 2; }
grep -q overlap_side_stream "$RUN/cfg_E.json" && { echo "❌ cfg_E 里还有 overlap_side_stream —— 就测不到代码默认值了"; exit 2; }
echo "✅ cfg_E = cfg_D − overlap_side_stream：$(cat "$RUN/cfg_E.json")" | tee "$RUN/seq.txt"
{ echo "D4 lp md5 $(md5sum < "$V2/D4/lp.jsonl" | cut -c1-32)"; echo "prompts md5 $(md5sum < "$RUN/prompts.jsonl" | cut -c1-32)"; } >> "$RUN/seq.txt"

# ── 起飞前占卡预检（与 seq_merge_v2.sh / seq_merge.sh 的 card_clear 逐行同口径）──
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

card_clear "$CARD" | tee -a "$RUN/preflight_arms.txt" || { echo "STOPPED at E1 (preflight)" >> "$RUN/seq.txt"; exit 6; }
DEFAULTS=1 REQUIRE_CLEAN_TREE=${REQUIRE_CLEAN_TREE:-1} bash "$T/run_arm.sh" "$RUN" E1 "$CARD" probe,lp; rc=$?
echo "ARM E1 card=$CARD DEFAULTS=1 steps=probe,lp rc=$rc [$(date +%T)]" >> "$RUN/seq.txt"
[ $rc = 0 ] || { echo "STOPPED at E1 rc=$rc" >> "$RUN/seq.txt"; exit $rc; }
python3 "$T/lp_verdict.py" "$V2/D4/lp.jsonl" "$RUN/E1/lp.jsonl" "D4 vs E1 (第二版 D 显式 vs §4.33 默认)" --skip-first=13 \
  > "$RUN/lp_cmp_D4_E1.txt" 2>&1; lrc=$?
cat "$RUN/lp_cmp_D4_E1.txt" | tee -a "$RUN/seq.txt"
[ $lrc = 0 ] || { echo "🔴 D4 vs E1 lp_verdict rc=$lrc ⇒ 不下结论，按 RUNBOOK_defaults_on.md §3 补 F1" | tee -a "$RUN/seq.txt"; exit 11; }
echo "ALLDONE 🟢 默认开 = 第二版 D 逐位相同" | tee -a "$RUN/seq.txt"
