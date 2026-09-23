#!/usr/bin/env bash
# seq_p1a.sh <RUN dir> [card] [steps]  —— P1a 的整网 A/B：交错 A1,B1,A2,B2；任一臂非 0 就停。
#
# 与 seq_ab.sh 的唯一区别，也是这份脚本单独存在的理由：
#   🔴 两臂【都】开侧流（SIDE=1），只差 VLLM_ASCEND_DSV41_KT_ASYNC_WAIT 这一个开关。
#      侧流开 + P1a 开是将来要合的形态，所以 A/B 必须在那个形态上量，
#      否则量到的是"侧流 + P1a"的合计，而侧流的 −3.00 ms 早就单独结过账（§4.26）。
#   ⇒ cfg_A.json 与 cfg_B.json 必须【完全相同】（都带 overlap_side_stream: true）。
#      下面会核对；不同就拒绝跑，因为那样两臂就不止差一个开关了。
#
# 用法：
#   bash seq_p1a.sh $RUN 0            # 默认 perf（CASES="bs1 1 16" REPEAT=3）
#   bash seq_p1a.sh $RUN 0 lp,perf    # 加 logprob 逐位（要先有 prompts.jsonl）
RUN="${1:?usage: seq_p1a.sh <RUN dir> [card] [steps]}"; CARD="${2:-0}"; STEPS="${3:-perf}"
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cmp -s "$RUN/cfg_A.json" "$RUN/cfg_B.json" || {
  echo "❌ cfg_A.json != cfg_B.json —— P1a 的两臂只许差 VLLM_ASCEND_DSV41_KT_ASYNC_WAIT 一个开关"; exit 2; }
grep -q '"overlap_side_stream" *: *true' "$RUN/cfg_A.json" || {
  echo "❌ cfg 里没有 overlap_side_stream:true —— 这一轮要在【侧流开】的形态上比 P1a"; exit 2; }
for spec in "A1 $STEPS" "B1 $STEPS" "A2 $STEPS" "B2 $STEPS"; do
  set -- $spec
  X=${1:0:1}; P=$([ "$X" = B ] && echo 1 || echo 0)
  SIDE=1 P1A=$P bash $T/run_arm.sh $RUN $1 $CARD $2; rc=$?
  echo "ARM $1 SIDE=1 P1A=$P rc=$rc" >> $RUN/seq.txt
  [ $rc = 0 ] || { echo "STOPPED at $1" >> $RUN/seq.txt; exit $rc; }
done
echo ALLDONE >> $RUN/seq.txt
