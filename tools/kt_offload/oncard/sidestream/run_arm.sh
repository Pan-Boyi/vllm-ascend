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
X=${ARM:0:1}; CFG=$RUN/cfg_$X.json; SIDE=$([ "$X" = B ] && echo 1 || echo 0)
say "arm $ARM card $CARD cfg $(cat $CFG) SIDE_STREAM=$SIDE extra=[$*] steps=$STEPS"
python3 worktrees/moe-offload/tools/kt_offload/build_fingerprint.py stamp $D --label sidestream_$ARM > $D/fp.txt 2>&1
grep -q 85331298bdeffee02b7c42c5658030ac $D/build_fingerprint.json || { say "❌ .so mismatch"; exit 3; }
npu-smi info > $D/card_before.txt 2>&1
( cd $ROOT && setsid env \
    WT=moe-offload DSV41_VLLM_TREE=$ROOT/worktrees/moe-offload-vllm LEASE_OWNER=$OWNER \
    KT_OFFLOAD=1 NPUS=$CARD PORT=$PORT EAGER=0 VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    KT_ADDITIONAL_CONFIG="$(cat $CFG)" MAXLEN=8192 MAXSEQS=8 MAXBATCH=4096 \
    VLLM_ASCEND_DSV41_KT_SIDE_STREAM=$SIDE VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=30 "$@" \
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
VA=$(grep -m1 'vllm_ascend = ' $D/serve.log); NSS=$(grep -c 'SIDE STREAM ENABLED' $D/serve.log); NE=$(grep -cE '1070[12][0-9]' $D/serve.log)
say "check vllm_ascend: $VA"; say "check SIDE STREAM ENABLED count=$NSS (want $([ $SIDE = 1 ] && echo 40 || echo 0))"; say "check 1070[12]x count=$NE (want 0)"
bad=0; echo "$VA" | grep -q 'worktrees/moe-offload/vllm_ascend/__init__.py' || bad=1
[ $SIDE = 1 ] && [ $NSS -ne 40 ] && bad=1; [ $SIDE = 0 ] && [ $NSS -ne 0 ] && bad=1; [ $NE -ne 0 ] && bad=1
[ $bad = 1 ] && { say "❌ mandatory check failed -> arm void"; stop; exit 5; }
for s in ${STEPS//,/ }; do case $s in
  probe) curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
      -d '{"model":"dsv41","prompt":"1+1=","max_tokens":1,"temperature":0,"logprobs":5}' > $D/probe.json; say "probe: $(head -c 400 $D/probe.json)";;
  lp|lp2) python3 $T/logprob_run.py --port $PORT --prompts $RUN/prompts.jsonl --out $D/$s.jsonl > $D/$s.log 2>&1; say "$s rc=$? $(tail -1 $D/$s.log)";;
  perf) PORT=$PORT LABEL=$ARM CASES="bs1 1 16" REPEAT=3 PERF_OUT=$D/perf bash build_artifacts/perf_baseline.sh > $D/perf.log 2>&1
      say "perf rc=$?"; grep -E 'bs1 +decode|bs1 +TPOT' $D/perf/summary.txt | tee -a $SUM;;
  gsm) EV_THINKING=0 python3 build_artifacts/eval_runners/gsm8k_eval_effort.py --base-url "http://127.0.0.1:$PORT/v1" --model dsv41 \
      --out-dir $D/gsm8k50 --max-tokens 4096 --temperature 0 --top-p 1.0 --num-threads 8 --limit 50 > $D/gsm8k.log 2>&1
      say "gsm8k50 rc=$? $(grep -m1 -oE 'acc[^,]*' $D/gsm8k.log | head -1)";;
  # M4: the whole 1319. Same knobs as `gsm`, only --limit drops away (0 = all), plus
  # --resume so a retry continues from partial.jsonl instead of starting over.
  # 🔴 Read it against the band, not against a number: single-card offload has NO
  # 1319 history, the only comparable figure is multi-card 1276-1281/1319
  # (96.74-97.12%), and same-code-same-package restart noise is 11 questions
  # ⇒ inside the band = no detectable regression, <=11 apart = INDISTINGUISHABLE.
  gsm1319) EV_THINKING=0 python3 build_artifacts/eval_runners/gsm8k_eval_effort.py --base-url "http://127.0.0.1:$PORT/v1" --model dsv41 \
      --out-dir $D/gsm8k1319 --max-tokens 4096 --temperature 0 --top-p 1.0 --num-threads 8 --resume > $D/gsm1319.log 2>&1
      say "gsm8k1319 rc=$? $(grep -m1 -oE 'acc[^,]*' $D/gsm1319.log | head -1)";;
  hold) say "hold: sleeping until $D/release exists"; while [ ! -e $D/release ]; do sleep 10; done;;
esac; done
grep 'host-func over' $D/serve.log | sed 's/.*kt_offload host-func/host-func/' > $D/hostfunc.txt
say "hostfunc reports $(wc -l < $D/hostfunc.txt); uniform $(grep -c 'uniform$' $D/hostfunc.txt); uneven $(grep -c uneven $D/hostfunc.txt); never-ran $(grep -c 'never ran' $D/hostfunc.txt)"
stop; say "done"
