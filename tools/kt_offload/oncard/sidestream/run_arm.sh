#!/usr/bin/env bash
# run_arm.sh <RUN> <ARM: A1|B1|A2|B2|...> <CARD> <STEPS: comma list of probe,lp,lp2,perf,gsm,gsm1319,gsm1319if50,hold>  (env: SIDE P1A P1B SIW REQUIRE_CLEAN_TREE PORT) [extra env assignments...]
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
# P1b（VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL）：图里不再有 host-func 节点。与 P1A 互斥
# （config 起服务就 raise），所以这里也拦一道：两个都开的那一发不是任何一条路的读数。
P1B=${P1B:-0}
[ "$P1A" = 1 ] && [ "$P1B" = 1 ] && { echo "❌ P1A 与 P1B 只能开一个"; exit 2; }
# §4.31 共享专家进窗口（VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW）。默认 0 = 旧各发的形态不变。
# 🔴 显式传 0 而不是「不设」：父 shell 里若残留 =1，「不设」会让 C 臂悄悄变成 D 臂，
#    而两臂逐位全同 + perf 无差 正是那种错误的样子。必检同样按【这一臂要的值】判 40/0。
SIW=${SIW:-0}
case "$SIW" in 0|1) ;; *) echo "❌ SIW 只认 0/1（得到 '$SIW'）"; exit 2;; esac
# 起服务前核树（第二版验收起加，第一版 A4 的教训：另一 agent 在跑期间改了本树的已跟踪文件）。
# REQUIRE_CLEAN_TREE=1 ⇒ 本树 git status 必须为空、指纹 dirty_files 必须两棵树都是 0，否则拒飞 rc=3。
REQUIRE_CLEAN_TREE=${REQUIRE_CLEAN_TREE:-0}
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
say "arm $ARM card $CARD cfg $(cat $CFG) SIDE_STREAM=$SIDE ASYNC_WAIT=$P1A DEVICE_SIGNAL=$P1B SHARED_IN_WINDOW=$SIW extra=[$*] steps=$STEPS"
python3 worktrees/moe-offload/tools/kt_offload/build_fingerprint.py stamp $D --label sidestream_$ARM > $D/fp.txt 2>&1
grep -q 85331298bdeffee02b7c42c5658030ac $D/build_fingerprint.json || { say "❌ .so mismatch"; exit 3; }
WTREE=$ROOT/worktrees/moe-offload
git -C $WTREE status --short > $D/git_status.txt 2>&1
say "tree HEAD $(git -C $WTREE rev-parse HEAD) git-status-lines $(wc -l < $D/git_status.txt) (全文 $D/git_status.txt)"
DIRTY=$(python3 -c "import json,sys;t=json.load(open(sys.argv[1]))['trees'];print(' '.join(f'{k}={v[\"dirty_files\"]}' for k,v in sorted(t.items())))" $D/build_fingerprint.json 2>&1)
say "fingerprint dirty_files: $DIRTY"
# RUNTIME_BASE=<commit>：本臂 HEAD 相对它在 vllm_ascend/ 下必须 0 个文件差（主会话会往本树提交文档，
# HEAD 会动；动的只许是文档）。
if [ -n "${RUNTIME_BASE:-}" ]; then
  RTD=$(git -C $WTREE diff --name-only "$RUNTIME_BASE" HEAD -- vllm_ascend/ 2>&1)
  say "runtime diff vs $RUNTIME_BASE (vllm_ascend/): $(echo -n "$RTD" | grep -c . ) files"
  [ -n "$RTD" ] && { say "❌ HEAD 相对 $RUNTIME_BASE 改了运行时代码 —— 拒飞"; echo "$RTD" | tee -a $SUM; exit 3; }
fi
if [ "$REQUIRE_CLEAN_TREE" = 1 ]; then
  [ -s $D/git_status.txt ] && { say "❌ 验收树不干净（git status 非空）—— 拒飞"; cat $D/git_status.txt | tee -a $SUM; exit 3; }
  [ "$DIRTY" = "vllm=0 vllm_ascend=0" ] || { say "❌ 指纹 dirty_files 不是两棵树都 0 —— 拒飞"; exit 3; }
fi
# DRY_RUN=1：只做到这里（cfg/env 组装、.so 与树的核对），不碰卡、不起服务。干跑用。
[ "${DRY_RUN:-0}" = 1 ] && { say "DRY_RUN: 起飞前核对全过，未起服务"; exit 0; }
npu-smi info > $D/card_before.txt 2>&1
( cd $ROOT && setsid env \
    WT=moe-offload DSV41_VLLM_TREE=$ROOT/worktrees/moe-offload-vllm LEASE_OWNER=$OWNER \
    KT_OFFLOAD=1 NPUS=$CARD PORT=$PORT EAGER=0 VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    KT_ADDITIONAL_CONFIG="$(cat $CFG)" MAXLEN=8192 MAXSEQS=8 MAXBATCH=4096 \
    VLLM_ASCEND_DSV41_KT_SIDE_STREAM=$SIDE VLLM_ASCEND_DSV41_KT_ASYNC_WAIT=$P1A \
    VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL=$P1B VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW=$SIW \
    VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=30 "$@" \
    bash worktrees/moe-offload/tools/kt_offload/serve_v41_singlecard.sh ) > $D/serve.log 2>&1 &
SP=$!; sleep 2; PGID=$(ps -o pgid= -p $SP | tr -d ' '); echo "pgid=$PGID" > $D/pgid.txt; say "serve pid=$SP pgid=$PGID"
stop(){ say "stopping"; OWNED_CARDS=$CARD STOP_PGID=$PGID bash build_artifacts/stop_by_port.sh $PORT > $D/stop.log 2>&1; say "stop rc=$? $(tail -1 $D/stop.log)"
  for i in $(seq 1 60); do kill -0 -$PGID 2>/dev/null || break; sleep 2; done; kill -0 -$PGID 2>/dev/null && say "⚠ pgid $PGID still alive"
  bash build_artifacts/card_lease.sh status > $D/lease_after.txt 2>&1; npu-smi info > $D/card_after.txt 2>&1
  git -C $ROOT/worktrees/moe-offload status --short > $D/git_status_after.txt 2>&1
  say "tree HEAD at stop $(git -C $ROOT/worktrees/moe-offload rev-parse HEAD) git-status-lines $(wc -l < $D/git_status_after.txt)"; }
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
NDS=$(grep -c 'DEVICE SIGNAL ENABLED' $D/serve.log)
NSW=$(grep -c 'SHARED IN WINDOW for layer' $D/serve.log)
say "check vllm_ascend: $VA"; say "check SIDE STREAM ENABLED count=$NSS (want $([ $SIDE = 1 ] && echo 40 || echo 0))"
say "check ASYNC WAIT ENABLED count=$NAW (want $([ $P1A = 1 ] && echo 40 || echo 0))"
say "check DEVICE SIGNAL ENABLED count=$NDS (want $([ $P1B = 1 ] && echo 40 || echo 0))"; say "check ACL errcode lines count=$NE (want 0)"
say "check SHARED IN WINDOW for layer count=$NSW (want $([ $SIW = 1 ] && echo 40 || echo 0))"
bad=0; echo "$VA" | grep -q 'worktrees/moe-offload/vllm_ascend/__init__.py' || bad=1
[ $SIDE = 1 ] && [ $NSS -ne 40 ] && bad=1; [ $SIDE = 0 ] && [ $NSS -ne 0 ] && bad=1; [ $NE -ne 0 ] && bad=1
[ $P1A = 1 ] && [ $NAW -ne 40 ] && bad=1; [ $P1A = 0 ] && [ $NAW -ne 0 ] && bad=1
[ $P1B = 1 ] && [ $NDS -ne 40 ] && bad=1; [ $P1B = 0 ] && [ $NDS -ne 0 ] && bad=1
[ $SIW = 1 ] && [ $NSW -ne 40 ] && bad=1; [ $SIW = 0 ] && [ $NSW -ne 0 ] && bad=1
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
  # 第二版：1319 只在同一服务的 GSM8K 50 满分时跑（用户：50 满分后直接跑；<50 停下报告）。
  gsm1319if50) python3 -c "import json,sys;m=json.load(open(sys.argv[1]+'/metrics.json'));sys.exit(0 if (m['num_correct'],m['num_examples'])==(50,50) else 1)" $D/gsm8k50 2>/dev/null \
      && { EV_THINKING=0 python3 build_artifacts/eval_runners/gsm8k_eval_effort.py --base-url "http://127.0.0.1:$PORT/v1" --model dsv41 \
             --out-dir $D/gsm8k1319 --max-tokens 4096 --temperature 0 --top-p 1.0 --num-threads 8 --resume > $D/gsm1319.log 2>&1
           say "gsm8k1319 rc=$? $(score $D/gsm8k1319)"; } \
      || { say "🔴 gsm8k50 不是 50/50（$(score $D/gsm8k50)）⇒ 不跑 1319，停下报告"; GSM_STOP=1; };;
  hold) say "hold: sleeping until $D/release exists"; while [ ! -e $D/release ]; do sleep 10; done;;
esac; done
# 🔴 采样器的行名跟着它采的那个计数器走：P1b 开着时 HOST_FUNC_CALLS 永远是 0，行名是
# 'device-signal over'。只 grep 'host-func over' 会让 P1b 那一臂"一行都没有"——而
# "每层都跑了"和"一层都没跑"在这个输出里长得一模一样。两个都收，并核对条数。
grep -E 'kt_offload (host-func|device-signal) over' $D/serve.log | sed 's/.*kt_offload //' > $D/hostfunc.txt
say "per-layer reports $(wc -l < $D/hostfunc.txt) (counter: $([ $P1B = 1 ] && echo device-signal || echo host-func)); uniform $(grep -c 'uniform$' $D/hostfunc.txt); uneven $(grep -c uneven $D/hostfunc.txt); never-ran $(grep -c 'never ran' $D/hostfunc.txt)"
[ "$(wc -l < $D/hostfunc.txt)" -eq 0 ] && say '⚠ 采样器一行都没出 —— 这一臂的每层计数没有被看过，不是「都好」' 
stop; say "done"
[ "${GSM_STOP:-0}" = 1 ] && exit 7 || true
