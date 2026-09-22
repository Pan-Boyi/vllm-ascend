#!/usr/bin/env bash
# seq_ab.sh <RUN dir> [card]  —— 交错 A1,B1,A2,B2；任一臂非 0 就停在那一臂。
# RUN 目录里要先有 prompts.jsonl / cfg_A.json / cfg_B.json / tools/（见 RUNBOOK_sidestream.md §3-§4）。
RUN="${1:?usage: seq_ab.sh <RUN dir> [card]}"; CARD="${2:-0}"
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
for spec in "A1 probe,lp,lp2,perf,gsm" "B1 probe,lp,perf,gsm" "A2 lp,perf,gsm" "B2 lp,perf,gsm"; do
  set -- $spec
  bash $T/run_arm.sh $RUN $1 $CARD $2; rc=$?
  echo "ARM $1 rc=$rc" >> $RUN/seq.txt
  [ $rc = 0 ] || { echo "STOPPED at $1" >> $RUN/seq.txt; exit $rc; }
done
echo ALLDONE >> $RUN/seq.txt
