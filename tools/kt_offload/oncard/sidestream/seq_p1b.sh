#!/usr/bin/env bash
# seq_p1b.sh <RUN dir> [card] [steps]  —— P1b 的整网 A/B：交错 A1,B1,A2,B2；任一臂非 0 就停。
#
# 与 seq_p1a.sh 一样的形状，换一个开关：
#   🔴 两臂【都】开侧流（SIDE=1），只差 VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL 这一个开关。
#      侧流的 −3.00 ms 早在 §4.26 单独结过账（§4.25/§4.26），混在一起量到的是合计。
#   🔴 A 臂是【现在的默认形态】（host-func 节点还在），不是 P1a —— P1a 与 P1b 互斥，
#      而这一发要回答的是「把节点整个拆掉值多少」，基线就得是没拆之前那个。
#   ⇒ cfg_A.json 与 cfg_B.json 必须逐字节相同（都带 overlap_side_stream: true）。
#
# 🔴 交错 ≥2 轮不是形式：判据（PLAN §4.27）是 |A−B| ≥ 3 × 同臂地板 且 ≥ 0.3 ms/token，
#    地板＝【本发】同臂跨轮的最大摆动。只跑一轮 ⇒ 没有地板 ⇒ 按 §4.15③ 判 fail。
#
# 用法：
#   bash seq_p1b.sh $RUN 0            # 默认 perf（CASES="bs1 1 16" REPEAT=3）
#   bash seq_p1b.sh $RUN 0 lp,perf    # 加 logprob 逐位（要先有 prompts.jsonl）
RUN="${1:?usage: seq_p1b.sh <RUN dir> [card] [steps]}"; CARD="${2:-0}"; STEPS="${3:-perf}"
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cmp -s "$RUN/cfg_A.json" "$RUN/cfg_B.json" || {
  echo "❌ cfg_A.json != cfg_B.json —— P1b 的两臂只许差 VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL 一个开关"; exit 2; }
grep -q '"overlap_side_stream" *: *true' "$RUN/cfg_A.json" || {
  echo "❌ cfg 里没有 overlap_side_stream:true —— 这一轮要在【侧流开】的形态上比 P1b"; exit 2; }
for spec in "A1 $STEPS" "B1 $STEPS" "A2 $STEPS" "B2 $STEPS"; do
  set -- $spec
  X=${1:0:1}; P=$([ "$X" = B ] && echo 1 || echo 0)
  SIDE=1 P1A=0 P1B=$P bash $T/run_arm.sh $RUN $1 $CARD $2; rc=$?
  echo "ARM $1 SIDE=1 P1B=$P rc=$rc" >> $RUN/seq.txt
  [ $rc = 0 ] || { echo "STOPPED at $1" >> $RUN/seq.txt; exit $rc; }
done
echo ALLDONE >> $RUN/seq.txt
