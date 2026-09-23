#!/usr/bin/env bash
# accept_v2.sh <RUN dir> perf|rest   —— 第二版合版本验收（C vs D）的整段驱动。判据见 $RUN/EXPECTED.txt（冻结）。
#   ① perf 三轮交错 C1,D1,C2,D2,C3,D3 → perf_verdict.py --arms CD --signed；🔴 判负即停（用户决定）
#   ② C4（lp）、D4（lp → gsm50 → 满分才 1319，同一服务）→ lp_verdict.py skip=13
#   ③ C4 vs D4 有差 ⇒ 补 C5/D5（lp），给出自比地板与跨臂三对比较，判读人工按 EXPECTED 做
# 卡：先卡 2；只在【C1 起飞前】卡 2 等满 20 min 仍不干净时改卡 1（此时尚无任何读数）。
#     之后任一臂预检失败 ⇒ 停（不中途换卡：跨卡读数不得相减）。绝不用卡 3（seq_merge_v2.sh 也拦）。
# 分两段调用（协调方要求：perf 段结束、判定写完就先报一声，再继续逐位/GSM8K）：
#   bash accept_v2.sh $RUN perf   → ①，rc 0 = 成立；10 = 判负/少臂（停）；其它 = 没跑完
#   bash accept_v2.sh $RUN rest   → ②③，只在 VERDICT_perf.txt 为成立时才肯跑
set -u
RUN="${1:?usage: accept_v2.sh <RUN> perf|rest}"; STAGE="${2:?perf|rest}"
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); S=$T/seq_merge_v2.sh
ST=$RUN/DRIVER_STATUS.txt
log(){ echo "[$(date +%FT%T)] $*" | tee -a $ST; }
log "start; md5 EXPECTED $(md5sum < $RUN/EXPECTED.txt | cut -c1-32) (落盘 $(cat $RUN/EXPECTED.md5 | cut -c1-32))"
cmp -s <(md5sum < $RUN/EXPECTED.txt | cut -c1-32) <(cut -c1-32 $RUN/EXPECTED.md5) || { log "❌ EXPECTED.txt 与落盘 md5 不符 —— 拒跑"; exit 2; }
log "df: $(df -h / | tail -1)"

if [ "$STAGE" = perf ]; then
CARD=2
bash $S $RUN 2 perf perf 3; rc=$?
if [ $rc = 6 ] && [ ! -e $RUN/C1 ] && tail -1 $RUN/seq.txt | grep -q 'STOPPED at C1 (preflight)'; then
  log "卡 2 在 C1 起飞前等满 20 min 仍不干净 ⇒ 按用户口径改卡 1（此时尚无任何读数）"
  CARD=1; bash $S $RUN 1 perf perf 3; rc=$?
fi
echo $CARD > $RUN/card.txt
[ $rc = 0 ] || { log "🔴 perf 交错未跑完 rc=$rc（见 seq.txt）⇒ 停"; exit $rc; }
python3 $T/perf_verdict.py $RUN --rounds 3 --arms CD --signed > $RUN/VERDICT_perf.txt 2>&1; vrc=$?
cat $RUN/VERDICT_perf.txt | tee -a $ST
[ $vrc = 0 ] || { log "🔴 perf 判定 rc=$vrc（1=判负 3=少臂）⇒ 按用户决定停下报告，不跑逐位/GSM8K"; exit 10; }

log "perf 段结束：成立"; exit 0
fi
[ "$STAGE" = rest ] || { echo "stage 只认 perf|rest"; exit 2; }
# REST_OVERRIDE=<run 目录里的 addendum 文件>：用户决定跳过 perf 判定、直接放行 rest 时用。
# 🔴 不伪造 VERDICT_perf.txt —— 放行依据是那份 addendum（须只读、在本 run 目录、带 md5 落盘），日志里点名它。
if [ -n "${REST_OVERRIDE:-}" ]; then
  case "$REST_OVERRIDE" in "$RUN"/*) ;; *) log "❌ REST_OVERRIDE 必须是本 run 目录里的文件"; exit 2;; esac
  [ -f "$REST_OVERRIDE" ] || { log "❌ REST_OVERRIDE 不存在"; exit 2; }
  M=${REST_OVERRIDE%.txt}.md5
  cmp -s <(md5sum < "$REST_OVERRIDE" | cut -c1-32) <(cut -c1-32 "$M" 2>/dev/null) || { log "❌ $REST_OVERRIDE 与其 md5 不符/缺 md5"; exit 2; }
  log "⚠ 不依 perf 判定、依用户决定放行 rest：$REST_OVERRIDE（md5 $(cut -c1-32 "$M")）"
else
  grep -q '🟢 收益成立' $RUN/VERDICT_perf.txt 2>/dev/null || { log "❌ VERDICT_perf.txt 不是「收益成立」—— rest 段拒跑"; exit 10; }
fi
CARD=$(cat $RUN/card.txt 2>/dev/null || echo 2)
[ "$CARD" = 2 ] || [ "$CARD" = 1 ] || { log "❌ card.txt 里是 '$CARD'"; exit 2; }
log "进入逐位 + GSM8K（卡 $CARD；放行依据见上一行）"
ARM_OFFSET=3 bash $S $RUN $CARD lp lp,gsm,gsm1319if50 1; rc=$?
if [ -s $RUN/C4/lp.jsonl ] && [ -s $RUN/D4/lp.jsonl ]; then
  python3 $T/lp_verdict.py $RUN/C4/lp.jsonl $RUN/D4/lp.jsonl "C4 vs D4 (跨臂)" --skip-first=13 > $RUN/lp_cmp_C4_D4.txt 2>&1; lrc=$?
  cat $RUN/lp_cmp_C4_D4.txt | tee -a $ST
else lrc=3; log "⚠ C4/D4 lp.jsonl 不全，逐位未判"; fi
[ $rc = 7 ] && { log "🔴 D4 GSM8K 50 < 50 ⇒ 停下报告（1319 未跑）"; exit 7; }
[ $rc = 0 ] || { log "🔴 C4/D4 rc=$rc ⇒ 停"; exit $rc; }
if [ $lrc = 1 ]; then
  log "C4 vs D4 有差 ⇒ 补 C5/D5 自比地板"
  ARM_OFFSET=4 bash $S $RUN $CARD lp lp 1; rc=$?
  [ $rc = 0 ] || { log "🔴 C5/D5 rc=$rc ⇒ 停"; exit $rc; }
  for p in "C4 C5 C自比" "D4 D5 D自比" "C5 D5 跨臂2"; do set -- $p
    python3 $T/lp_verdict.py $RUN/$1/lp.jsonl $RUN/$2/lp.jsonl "$1 vs $2 ($3)" --skip-first=13 > $RUN/lp_cmp_$1_$2.txt 2>&1
    log "lp_cmp $1 vs $2 rc=$?"; cat $RUN/lp_cmp_$1_$2.txt >> $ST; done
fi
log "ALLDONE (perf rc=$vrc, lp rc=$lrc)"
