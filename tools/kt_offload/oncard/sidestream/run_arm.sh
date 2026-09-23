#!/usr/bin/env bash
# run_arm.sh <RUN> <ARM: A1|B1|A2|B2|...> <CARD> <STEPS: comma list of probe,lp,lp2,perf,gsm,gsm1319,hold> [extra env assignments...]
set -u
ROOT=/workspace/code/dsv4_single_card/vllm_project; cd $ROOT
RUN=$1; ARM=$2; CARD=$3; STEPS=$4; shift 4
T=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # 本脚本所在目录 = logprob_run.py 等的出处
PORT=${PORT:-8041}; OWNER=dsv41_vllm_singlecard
export TMPDIR=/dev/shm/dsv41_tmp TMP=/dev/shm/dsv41_tmp
D=$RUN/$ARM; mkdir -p $D; SUM=$D/arm_summary.txt; : > $SUM
say(){ echo "[$(date +%T)] $*" | tee -a $SUM; }
X=${ARM:0:1}; CFG=$RUN/cfg_$X.json
# 默认：A 臂侧流关、B 臂侧流开（§4.26 那一轮的口径）。
# 🔴 P1a 的 A/B 是【两臂都开侧流】、只差 async wait 这一个开关，所以两者都可被调用方
#    显式覆盖（SIDE=1 P1A=0 / SIDE=1 P1A=1）。覆盖之后下面的必检也跟着这两个值走 ——
#    检查项必须判【这一臂实际要的】，否则一个写死 40/0 的必检会把正确的臂判成作废。
SIDE=${SIDE:-$([ "$X" = B ] && echo 1 || echo 0)}
P1A=${P1A:-0}
# 🔴 Read the score from metrics.json, never by grepping the progress log: the
# runner prints "[n/N] acc=..." every 50 items, so `grep -m1 acc` picks the
# score after the FIRST 50 and calls it the result. With --limit 50 that is
# accidentally the final line; with 1319 it reported acc=1.0000 for a run that
# finished at 1275/1319. A reader cannot tell those two apart.
score(){ python3 - "$1" <<'PYEOF' 2>/dev/null || echo "score unreadable"
import json,sys
m=json.load(open(sys.argv[1]+"/metrics.json"))
print(f'{m["num_correct"]}/{m["num_examples"]} acc={m["aggregate"]["score"]:.4f} '
      f'err_rate={m["aggregate"]["error_rate"]} trunc={m["aggregate"]["truncated_rate"]} '
      f'{m["latency_seconds"]:.0f}s')
PYEOF
}
say "arm $ARM card $CARD cfg $(cat $CFG) SIDE_STREAM=$SIDE ASYNC_WAIT=$P1A extra=[$*] steps=$STEPS"
python3 worktrees/moe-offload/tools/kt_offload/build_fingerprint.py stamp $D --label sidestream_$ARM > $D/fp.txt 2>&1
grep -q 85331298bdeffee02b7c42c5658030ac $D/build_fingerprint.json || { say "❌ .so mismatch"; exit 3; }
npu-smi info > $D/card_before.txt 2>&1
( cd $ROOT && setsid env \
    WT=moe-offload DSV41_VLLM_TREE=$ROOT/worktrees/moe-offload-vllm LEASE_OWNER=$OWNER \
    KT_OFFLOAD=1 NPUS=$CARD PORT=$PORT EAGER=0 VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    KT_ADDITIONAL_CONFIG="$(cat $CFG)" MAXLEN=8192 MAXSEQS=8 MAXBATCH=4096 \
    VLLM_ASCEND_DSV41_KT_SIDE_STREAM=$SIDE VLLM_ASCEND_DSV41_KT_ASYNC_WAIT=$P1A \
    VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=30 "$@" \
    bash worktrees/moe-offload/tools/kt_offload/serve_v41_singlecard.sh ) > $D/serve.log 2>&1 &
SP=$!; sleep 2; PGID=$(ps -o pgid= -p $SP | tr -d ' '); echo "pgid=$PGID" > $D/pgid.txt; say "serve pid=$SP pgid=$PGID"
stop(){ say "stopping"; OWNED_CARDS=$CARD STOP_PGID=$PGID bash build_artifacts/stop_by_port.sh $PORT > $D/stop.log 2>&1; say "stop rc=$? $(tail -1 $D/stop.log)"
  for i in $(seq 1 60); do kill -0 -$PGID 2>/dev/null || break; sleep 2; done; kill -0 -$PGID 2>/dev/null && say "⚠ pgid $PGID still alive"
  bash build_artifacts/card_lease.sh status > $D/lease_after.txt 2>&1; npu-smi info > $D/card_after.txt 2>&1; }
ready=0; w=0
while [ $w -lt 900 ]; do
  curl -s -m 5 http://127.0.0.1:$PORT/v1/models -o $D/models.json 2>/dev/null && { ready=1; break; }
  grep -qE '(^|\) )Traceback|EngineCore failed|进程退出' $D/serve.log && { say "❌ failure marker in serve.log"; break; }
  kill -0 $SP 2>/dev/null || { say "❌ serve process exited"; break; }
  sleep 5; w=$((w+5))
done
[ $ready = 1 ] || { [ $w -ge 900 ] && say "❌ curl 超时 900s"; grep -nE '(^|\) )Traceback|EngineCore failed|进程退出|Error' $D/serve.log | tail -5 | tee -a $SUM; stop; exit 4; }
say "ready after ${w}s"
VA=$(grep -m1 'vllm_ascend = ' $D/serve.log); NSS=$(grep -c 'SIDE STREAM ENABLED' $D/serve.log); # 🔴 grep the ACL error codes on the lines that could carry one -- NOT on our
# own instrument's output. With VLLM_ASCEND_DSV41_KT_SEAM_TIMING on, the P0
# [kt-seam-rt] lines dump raw CLOCK_REALTIME nanoseconds and callback
# durations in ns, and a 19-digit integer contains "1070[12][0-9]" by pure
# chance: 136 / 145 / 176 hits on three healthy arms of 2026-09-23, and one of
# them (BD0r2) was VOIDED by a match inside 1790123465521070247. Word anchors
# do not help -- a cb_ns of exactly 107015 is comma-delimited. So the check
# excludes our own dumps; a real 107025 in a t5 log still matches (checked).
# An arm that passed this check while rt lines existed was not checked at all.
# 🔴 并且要有【非数字边界】：19 位纳秒时间戳里碰巧含 1070[12]x，09-22 夜就有一臂
# （BD0r2）被 1790123465521070247 里的一段误判作废。两条一起用才干净；口径与
# oncard_lib.sh 的 ONCARD_ERRCODE_RE 一致，selftest_cpu.sh 里有钉住它的变异臂。
ERRRE='(^|[^0-9])(107011|107015|107025|107027|107030)([^0-9]|$)'
NE=$(grep -v 'kt-seam' $D/serve.log | grep -cE "$ERRRE")
NAW=$(grep -c 'ASYNC WAIT ENABLED' $D/serve.log)
say "check vllm_ascend: $VA"; say "check SIDE STREAM ENABLED count=$NSS (want $([ $SIDE = 1 ] && echo 40 || echo 0))"
say "check ASYNC WAIT ENABLED count=$NAW (want $([ $P1A = 1 ] && echo 40 || echo 0))"; say "check ACL errcode lines count=$NE (want 0)"
bad=0; echo "$VA" | grep -q 'worktrees/moe-offload/vllm_ascend/__init__.py' || bad=1
[ $SIDE = 1 ] && [ $NSS -ne 40 ] && bad=1; [ $SIDE = 0 ] && [ $NSS -ne 0 ] && bad=1; [ $NE -ne 0 ] && bad=1
[ $P1A = 1 ] && [ $NAW -ne 40 ] && bad=1; [ $P1A = 0 ] && [ $NAW -ne 0 ] && bad=1
[ $bad = 1 ] && { say "❌ mandatory check failed -> arm void"; stop; exit 5; }
for s in ${STEPS//,/ }; do case $s in
  probe) curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"dsv41","prompt":"1+1=","max_tokens":1,"temperature":0,"logprobs":5}' > $D/probe.json; say "probe: $(head -c 400 $D/probe.json)";;
  lp|lp2) python3 $T/logprob_run.py --port $PORT --prompts $RUN/prompts.jsonl --out $D/$s.jsonl > $D/$s.log 2>&1; say "$s rc=$? $(tail -1 $D/$s.log)";;
  perf) PORT=$PORT LABEL=$ARM CASES="bs1 1 16" REPEAT=3 PERF_OUT=$D/perf bash build_artifacts/perf_baseline.sh > $D/perf.log 2>&1
      say "perf rc=$?"; grep -E 'bs1 +decode|bs1 +TPOT' $D/perf/summary.txt | tee -a $SUM;;
  gsm) EV_THINKING=0 python3 build_artifacts/eval_runners/gsm8k_eval_effort.py --base-url "http://127.0.0.1:$PORT/v1" --model dsv41 \
      --out-dir $D/gsm8k50 --max-tokens 4096 --temperature 0 --top-p 1.0 --num-threads 8 --limit 50 > $D/gsm8k.log 2>&1
      say "gsm8k50 rc=$? $(score $D/gsm8k50)";;
  # M4: the whole 1319. Same knobs as `gsm`, only --limit drops away (0 = all), plus
  # --resume so a retry continues from partial.jsonl instead of starting over.
  # 🔴 Read it against the band, not against a number: single-card offload has NO
  # 1319 history, the only comparable figure is multi-card 1276-1281/1319
  # (96.74-97.12%), and same-code-same-package restart noise is 11 questions
  # ⇒ inside the band = no detectable regression, <=11 apart = INDISTINGUISHABLE.
  gsm1319) EV_THINKING=0 python3 build_artifacts/eval_runners/gsm8k_eval_effort.py --base-url "http://127.0.0.1:$PORT/v1" --model dsv41 \
      --out-dir $D/gsm8k1319 --max-tokens 4096 --temperature 0 --top-p 1.0 --num-threads 8 --resume > $D/gsm1319.log 2>&1
      say "gsm8k1319 rc=$? $(score $D/gsm8k1319)";;
  hold) say "hold: sleeping until $D/release exists"; while [ ! -e $D/release ]; do sleep 10; done;;
esac; done
grep 'host-func over' $D/serve.log | sed 's/.*kt_offload host-func/host-func/' > $D/hostfunc.txt
say "hostfunc reports $(wc -l < $D/hostfunc.txt); uniform $(grep -c 'uniform$' $D/hostfunc.txt); uneven $(grep -c uneven $D/hostfunc.txt); never-ran $(grep -c 'never ran' $D/hostfunc.txt)"
stop; say "done"
