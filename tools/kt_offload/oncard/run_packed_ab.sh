#!/usr/bin/env bash
# ============================================================================
#  run_packed_ab.sh <card>        —— packed_transfer A/B, one command, one card
#  MODE=aa run_packed_ab.sh <card> —— A/A control: the SAME script, both arms
#                                    packed_transfer=false. Same harness by
#                                    construction, not by copy: the eval call
#                                    is one function used by both modes, so a
#                                    parameter cannot drift between them (if it
#                                    could, the control would not control the
#                                    variable it claims to).
#
#  Two arms on one card, differing in ONE switch (kt_offload_config.packed_transfer):
#    A  packed_transfer=false   the path in service today (5 host-device ops/layer)
#    B  packed_transfer=true    one combined input copy       (3 ops/layer)
#  Each arm: serve -> GSM8K 50 (greedy) -> perf_baseline (bs1, REPEAT=4) ->
#            decode profile -> teardown, and the card is proven back before the
#            next arm starts. Then packed_ab_report.py judges both arms.
#
#  Why this shape (each line is a past failure):
#   * VLLM_USE_BREAKABLE_CUDAGRAPH=1 is passed explicitly and printed: the
#     platform defaults it to 0 at import, and clean's O15 assertion refuses a
#     graph-mode launch that did not pass it. A missing value runs a path that
#     was never validated, silently.
#   * The front end has twice survived stop_by_port. This script kills the
#     process group it started (never pkill -f), waits for it to be gone, waits
#     for the port to be free, releases the lease and then re-reads the lease
#     table -- and says "card returned" only when all four hold.
#   * The profile capture prints "captured" even on a 404, so the ascend_pt
#     directories are counted here and a zero count fails the arm.
#   * Nothing here enables overlap_side_stream or form C: one variable.
#
#  MODE=aa detail: each arm runs the eval set AA_EVALS (default "c8,c8b,c1"):
#    c8  num-threads 8  -> positionally identical to the A/B's only eval
#    c8b num-threads 8  -> a SECOND eval inside the same service start, so
#                          "batch composition" and "a different process" stop
#                          being one explanation
#    c1  num-threads 1  -> no concurrency at all. Without this档 "batch
#                          composition caused it" is a story, not a measurement.
#  profile is skipped in aa mode (nothing compares kernels); perf DOES run in
#  every arm -- the A-vs-A' TPOT difference is the drift floor. The c8 eval
#  still runs first, in the same position as in ab mode.
#
#  rc: 0 both arms done and judged | 2 usage | 3 preflight (no card touched)
#      4 lease refused | 5 an arm failed | 6 the card was not returned
#  Env: PORT (8041), CARD via argv, HFLOG (5 = host-func sampling interval,
#       0 disables), LIMIT (50), REPEAT (4), CAPTURE_S (8), N_RESIDENT (100),
#       THREADS (96), NUMA (0,1), ONCARD_DRY=1 (stop before the first acquire).
# ============================================================================
set -u
set -o pipefail

ROOT=/workspace/code/dsv4_single_card/vllm_project
WT="$ROOT/worktrees/moe-offload"
OWNER=dsv41_vllm_singlecard
CARD="${1:-}"
[[ "$CARD" =~ ^[0-9]+$ ]] || { echo "usage: $0 <card>   (env: PORT HFLOG LIMIT REPEAT CAPTURE_S N_RESIDENT THREADS NUMA)"; exit 2; }
PORT="${PORT:-8041}"
HFLOG="${HFLOG:-5}"
LIMIT="${LIMIT:-50}"
REPEAT="${REPEAT:-4}"
CAPTURE_S="${CAPTURE_S:-8}"
N_RESIDENT="${N_RESIDENT:-100}"
THREADS="${THREADS:-96}"
NUMA="${NUMA:-0,1}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
MODE="${MODE:-ab}"
case "$MODE" in
  ab) AA_EVALS="${AA_EVALS:-c8}"; ROUNDS="${ROUNDS:-2}";;
  aa) AA_EVALS="${AA_EVALS:-c8,c8b,c1}"; ROUNDS="${ROUNDS:-1}";;
  *) echo "usage: MODE=ab|aa $0 <card>"; exit 2;;
esac
[ "$ROUNDS" -ge 1 ] || { echo "ROUNDS must be >= 1"; exit 2; }

# ---- snapshot first: bash reads a script by byte offset, so editing this file
#      while it runs corrupts that run. exec keeps the PID, so LEASE_PID=$$ is
#      the same process either way.
if [ -z "${PACKED_AB_SNAPSHOT:-}" ]; then
  SRC_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
  STAMP="$(date -u +%Y%m%d_%H%M%S)"
  RUN="${PACKED_AB_RUNS_DIR:-$ROOT/build_artifacts/moe_offload_ref/packed_ab}/${STAMP}"
  mkdir -p "$RUN/code" || { echo "❌ cannot create $RUN"; exit 3; }
  cp -p "$SRC_DIR"/*.sh "$SRC_DIR"/*.py "$RUN/code/" || { echo "❌ snapshot failed"; exit 3; }
  cp -p "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../build_fingerprint.py" "$RUN/code/" 2>/dev/null || true
  export PACKED_AB_SNAPSHOT=1 PACKED_AB_RUN="$RUN" PACKED_AB_ARGV="MODE=$MODE $0 $CARD"
  exec bash "$RUN/code/run_packed_ab.sh" "$CARD"
fi
RUN="$PACKED_AB_RUN"; HERE="$RUN/code"; ONCARD_DIR="$HERE"
# shellcheck source=oncard_lib.sh
source "$HERE/oncard_lib.sh"
# shellcheck source=packed_ab_lib.sh
source "$HERE/packed_ab_lib.sh"
SUM="$RUN/summary.txt"
say() { echo "$*" | tee -a "$SUM"; }
say "▶ packed ${MODE^^} card=$CARD port=$PORT run=$RUN pid=$$ argv=[$PACKED_AB_ARGV]"
if [ "$MODE" = "aa" ]; then
  say "  A/A control: both arms packed_transfer=false, evals=$AA_EVALS (c8 first, in the A/B's position)"
  say "  it answers ONE question: does this service-level bit-identity criterion have any discriminating power"
fi
say "  VLLM_USE_BREAKABLE_CUDAGRAPH=1 (explicit; the platform defaults it to 0 and O15 refuses graph mode without it)"
say "  host-func sampling: VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=${HFLOG}s in BOTH arms (A is the baseline shape)"
say "  rounds: ${ROUNDS} (arms interleaved A,B,A,B...; two same-arm rounds give the machine's drift floor)"
say "  eval parameters (identical in both modes, one function): 50 questions, greedy (temperature 0, top-p 1.0), max-tokens 4096, thinking off"

# ---- preflight: host only, before any lease -------------------------------
pf() { say "❌ preflight: $1"; exit 3; }
command -v npu-smi >/dev/null 2>&1 || pf "no npu-smi here -> no card environment"
[ -x "$ROOT/build_artifacts/card_lease.sh" ] || pf "card_lease.sh missing"
[ -x "$ROOT/build_artifacts/stop_by_port.sh" ] || [ -f "$ROOT/build_artifacts/stop_by_port.sh" ] || pf "stop_by_port.sh missing"
[ -f "$ROOT/build_artifacts/perf_baseline.sh" ] || pf "perf_baseline.sh missing"
[ -f "$ROOT/build_artifacts/eval_runners/gsm8k_eval_effort.py" ] || pf "gsm8k harness missing"
[ -f "$ROOT/build_artifacts/eval_runners/prof_decode_capture.py" ] || pf "prof_decode_capture.py missing"
[ -f "$ROOT/build_artifacts/eval_runners/prof_collect.sh" ] || pf "prof_collect.sh missing"
[ -f "$WT/tools/kt_offload/serve_v41_singlecard.sh" ] || pf "serve_v41_singlecard.sh missing"
[ -f "$HERE/packed_ab_report.py" ] || pf "packed_ab_report.py missing (it judges the run)"
[ -d "$ROOT/build_artifacts/moe_offload_ref/kt_site/kt_kernel" ] || pf "kt_site missing"
python3 -m py_compile "$HERE/packed_ab_report.py" || pf "packed_ab_report.py does not compile"
AVAIL_KB=$(oc_meminfo_avail_kb)
[ "${AVAIL_KB:-0}" -ge $(( ${MIN_AVAIL_GB:-150} * 1024 * 1024 )) ] || pf "MemAvailable $((AVAIL_KB/1048576)) GiB < ${MIN_AVAIL_GB:-150} GiB"
say "  preflight ok (MemAvailable $((AVAIL_KB/1048576)) GiB, kt_site $(sha256sum "$ROOT/build_artifacts/moe_offload_ref/kt_site/kt_kernel/experts.py" 2>/dev/null | cut -c1-12))"
# ---- build fingerprint: the two arms are only comparable inside one build,
# and so is any later comparison against this run
FPTOOL="$HERE/build_fingerprint.py"; [ -f "$FPTOOL" ] || FPTOOL="$ROOT/worktrees/moe-offload/tools/kt_offload/build_fingerprint.py"
if python3 "$FPTOOL" stamp "$RUN" --label "packed_ab" > "$RUN/build_fingerprint.txt" 2>&1; then
  say "  $(head -n 1 "$RUN/build_fingerprint.txt")"
else
  say "❌ build fingerprint failed: $(tail -n 2 "$RUN/build_fingerprint.txt" | tr '\n' ' ')"; exit 3
fi
FP_FACTS="$(python3 -c '
import json,sys
d=json.load(open(sys.argv[1])); v=d.get("vendors",{}); t=d.get("trees",{}).get("vllm_ascend",{})
print("\t".join([d.get("so_primary_md5") or "", t.get("path") or "?", str(v.get("files")),
                 str(v.get("files_regular")), str(v.get("bytes")), (d.get("vendors_sha256") or "?")[:12]]))
' "$RUN/build_fingerprint.json")"
SO_MD5="$(printf %s "$FP_FACTS" | cut -f1)"; EXPECT_TREE="$(printf %s "$FP_FACTS" | cut -f2)"
say "  build under test: tree=$EXPECT_TREE  .so=$SO_MD5"
say "    vendors(loaded)=$(printf %s "$FP_FACTS" | cut -f6) files=$(printf %s "$FP_FACTS" | cut -f3)"\
"    (regular=$(printf %s "$FP_FACTS" | cut -f4)) bytes=$(printf %s "$FP_FACTS" | cut -f5)"
if [ -n "${ONCARD_EXPECT_SO_MD5:-}" ]; then
  if [ "${ONCARD_EXPECT_SO_MD5}" = "$SO_MD5" ]; then
    say "  ✅ ONCARD_EXPECT_SO_MD5 matches THIS tree's .so (per-tree expectation; other trees may differ)"
  else
    say "❌ this tree's .so is $SO_MD5, expected ${ONCARD_EXPECT_SO_MD5} (tree=$EXPECT_TREE);"
    say "   refusing to run across a build swap. Another tree's pin moving is a different fact."
    exit 3
  fi
fi

if [ "${ONCARD_DRY:-0}" = "1" ]; then
  say "⚪ ONCARD_DRY=1: preflight passed, build fingerprint written; would acquire card $CARD as $OWNER (LEASE_PID=$$) and run arms A,B. Stopping before acquire."
  exit 0
fi

cfg_json() {  # $1 = true|false for packed_transfer
  printf '{"kt_offload_config":{"enabled":true,"resident_strategy":"prefix","num_resident_experts":%s,' "$N_RESIDENT"
  printf '"cpuinfer_threads":%s,"threadpool_count":2,"numa_nodes":[%s],"packed_transfer":%s}}' "$THREADS" "$NUMA" "$1"
}

teardown() {  # $1 = arm dir, $2 = serve pgid
  local arm="$1" pgid="$2" ok=1
  say "  ── teardown"
  OWNED_CARDS="$CARD" bash "$ROOT/build_artifacts/stop_by_port.sh" "$PORT" >"$arm/stop.log" 2>&1
  # The front end has survived stop_by_port twice. Kill the group this script
  # started -- our own pgid, never a name match -- and then prove it is gone.
  if [ -n "$pgid" ] && kill -0 "-$pgid" 2>/dev/null; then
    say "     front end still alive in pgid $pgid: sending TERM"
    kill -TERM "-$pgid" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "-$pgid" 2>/dev/null || break; sleep 1; done
    if kill -0 "-$pgid" 2>/dev/null; then
      say "     still alive after 60 s: sending KILL"
      kill -KILL "-$pgid" 2>/dev/null; sleep 3
    fi
  fi
  for _ in $(seq 1 60); do pa_port_busy "$PORT" || break; sleep 1; done
  # pid gone is not context freed: wait until BOTH the card's process table and
  # its HBM say the card is free (pa_card_busy; the old lower-case grep here
  # never matched npu-smi's "Process id", so this loop used to exit at once).
  local waited=0
  while [ "$waited" -lt "${PA_TEARDOWN_WAIT_S:-300}" ] && pa_card_busy "$CARD"; do
    if [ $((waited % 60)) -eq 0 ]; then
      say "     ⚪ card $CARD still busy: state=$PA_LAST_STATE ($PA_LAST_SAMPLES), waited ${waited}s"
      say "        $(pa_advice "$PA_LAST_STATE")"
    fi
    sleep 5; waited=$((waited + 5))
  done
  if pa_card_busy "$CARD"; then
    say "     ❌ card $CARD still reads busy after ${waited}s: state=$PA_LAST_STATE ($PA_LAST_SAMPLES)"
    say "        $(pa_advice "$PA_LAST_STATE")"
    ok=0
  else
    say "     card free after ${waited}s (process table empty and HBM < ${PA_IDLE_HBM_PCT:-10}%)"
  fi
  bash "$ROOT/build_artifacts/card_lease.sh" release "$CARD" "$OWNER" >>"$arm/stop.log" 2>&1
  bash "$ROOT/build_artifacts/card_lease.sh" status > "$arm/lease_after.txt" 2>&1
  local why
  why=$(pa_card_returned "$(cat "$arm/lease_after.txt")" "$CARD" "$OWNER" "$PORT" "${pgid:-}") || {
    say "     ❌ card $CARD NOT returned: $why"; ok=0; }
  npu-smi info > "$arm/card_after.txt" 2>&1
  if [ "$ok" = 1 ]; then say "     ✅ card $CARD returned (group gone, port free, lease released)"; else say "     ❌ card $CARD NOT proven returned -- stop and look before the next arm"; fi
  return $((1 - ok))
}

run_evals() {  # $1 = arm dir. The ONLY place a GSM8K eval is launched.
  local arm="$1" ev out th
  for ev in ${AA_EVALS//,/ }; do
    case "$ev" in
      c8)  out="$arm/gsm8k50";     th=8;;
      c8b) out="$arm/gsm8k50_c8b"; th=8;;
      c1)  out="$arm/gsm8k50_c1";  th=1;;
      *) say "  ❌ unknown eval '$ev' in AA_EVALS"; return 5;;
    esac
    ( cd "$ROOT" && EV_THINKING=0 python3 build_artifacts/eval_runners/gsm8k_eval_effort.py \
        --base-url "http://127.0.0.1:$PORT/v1" --model dsv41 --out-dir "$out" \
        --max-tokens 4096 --temperature 0 --top-p 1.0 --num-threads "$th" --limit "$LIMIT" ) \
        >"$out.log" 2>&1
    say "  gsm8k[$ev threads=$th] rc=$? $(command grep -m1 'acc=' "$out.log" || true)"
  done
}

run_arm() {  # $1 = A|B|A1|A2, $2 = packed true|false, $3 = round (1-based)
  local name="$1" packed="$2" round="${3:-1}"
  local arm="$RUN/arm$name"
  [ "$round" -gt 1 ] && arm="$RUN/arm${name}_r${round}"
  mkdir -p "$arm/cap"
  cp -p "$RUN/build_fingerprint.json" "$arm/" 2>/dev/null || true
  local profdir="/dev/shm/dsv41_prof_packed_$name"
  rm -rf "$profdir"; mkdir -p "$profdir"
  cfg_json "$packed" > "$arm/cfg.json"
  say "═══ arm $name: packed_transfer=$packed"
  say "  cfg $(cat "$arm/cfg.json")"
  dmesg > "$arm/dmesg_before.txt" 2>/dev/null

  LEASE_PID=$$ bash "$ROOT/build_artifacts/card_lease.sh" acquire "$CARD" "$OWNER" "$PORT" >>"$SUM" 2>&1 || {
    say "  ❌ lease refused for card $CARD"; return 4; }

  # 🔴 WT / DSV41_VLLM_TREE / LEASE_OWNER 必须显式传（2026-09-22 首次上卡发现）：
  #    本脚本的 $WT 是【路径】且不 export，launcher 的 WT 默认是 "base" ⇒ 不传就会起
  #    主 checkout 那棵树（不是本线的树），vllm 树也退回 $ROOT/vllm。那次是 launcher
  #    自己再申请租约（owner "dsv41-base" ≠ 本脚本的 $OWNER）撞上本脚本的租约才被拦下，
  #    纯属运气。LEASE_OWNER 与本脚本一致 ⇒ launcher 那次申请是同一持有者续租。
  #    起来之后还要读 serve.log 里 assert_trees 打印的 vllm_ascend 路径（读真对象）。
  ( cd "$ROOT" && setsid env \
      WT=moe-offload DSV41_VLLM_TREE="$ROOT/worktrees/moe-offload-vllm" LEASE_OWNER="$OWNER" \
      KT_OFFLOAD=1 NPUS="$CARD" PORT="$PORT" EAGER=0 \
      VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
      KT_ADDITIONAL_CONFIG="$(cat "$arm/cfg.json")" \
      VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG="$HFLOG" \
      MAXLEN="${MAXLEN:-8192}" MAXSEQS="${MAXSEQS:-8}" MAXBATCH="${MAXBATCH:-4096}" \
      EXTRA_ARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$profdir\"}" \
      bash "$WT/tools/kt_offload/serve_v41_singlecard.sh" ) >"$arm/serve.log" 2>&1 &
  local serve_pid=$! pgid
  pgid=$(ps -o pgid= -p "$serve_pid" 2>/dev/null | tr -d ' ')
  say "  serve pid=$serve_pid pgid=${pgid:-?} log=$arm/serve.log"

  local ready=0 waited=0 want_va="vllm_ascend = $WT/vllm_ascend/__init__.py"
  while [ "$waited" -lt "$READY_TIMEOUT_S" ]; do
    if curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" -o "$arm/models.json" 2>/dev/null; then ready=1; break; fi
    kill -0 "$serve_pid" 2>/dev/null || { say "  ❌ serve exited before it was ready (see serve.log)"; break; }
    sleep 5; waited=$((waited + 5))
  done
  if [ "$ready" != 1 ]; then
    say "  ❌ arm $name: service not ready after ${waited}s"
    teardown "$arm" "${pgid:-}" ; return 5
  fi
  say "  service ready after ${waited}s"
  if ! command grep -qF "$want_va" "$arm/serve.log"; then
    say "  ❌ arm $name: serve.log does not show '$want_va' -- the service is not running this tree"
    command grep -m1 "vllm_ascend = " "$arm/serve.log" | sed 's/^/     got: /' | tee -a "$SUM" >/dev/null
    teardown "$arm" "${pgid:-}" ; return 5
  fi
  say "  tree check: $want_va ✅"
  command grep -m1 "BREAKABLE_CUDAGRAPH" "$arm/serve.log" | sed 's/^/     /' | tee -a "$SUM" >/dev/null

  # ---- positive control for the busy reading: the service is up right now,
  # so pa_card_busy MUST say busy, and it should say "running" (aicore non-zero)
  # rather than "residue". If it does not, every later "the card is free"
  # reading is worthless, and the run says so instead of trusting it.
  if pa_card_busy "$CARD"; then
    echo "busy-while-serving: yes state=$PA_LAST_STATE $PA_LAST_SAMPLES" > "$arm/card_busy_control.txt"
    say "  busy-reading positive control: card reads BUSY while serving ✅ (state=$PA_LAST_STATE)"
    [ "$PA_LAST_STATE" = running ] || say "     ⚠ expected state=running while serving, got $PA_LAST_STATE"
  else
    echo "busy-while-serving: NO" > "$arm/card_busy_control.txt"
    say "  ❌ busy-reading positive control FAILED: the card reads free while the service is serving."
    say "     Every 'card returned' verdict in this run is therefore unproven (npu-smi format changed?)."
  fi

  # ---- GSM8K 50, greedy, thinking off (same function in both modes).
  # Correctness does not drift with the hour, so later rounds skip it and spend
  # their minutes on the thing that does: the paired timing.
  if [ "$round" -eq 1 ]; then
    run_evals "$arm" || { teardown "$arm" "${pgid:-}"; return 5; }
    cp -p "$arm/gsm8k50.log" "$arm/gsm8k.log" 2>/dev/null || true
  else
    say "  (round $round: GSM8K skipped -- answers do not drift with the clock, timing does)"
  fi


  # ---- TPOT: same script, same cases, n=REPEAT
  ( cd "$ROOT" && PORT="$PORT" LABEL="packed_arm${name}_r${round}" NOTE="packed_transfer=$packed round=$round" \
      CASES="bs1 1 16" REPEAT="$REPEAT" PERF_OUT="$arm/perf" \
      bash build_artifacts/perf_baseline.sh ) >"$arm/perf.log" 2>&1
  say "  perf rc=$? -> $arm/perf/summary.txt"

  if [ "$MODE" = "aa" ] || [ "$round" -gt 1 ]; then
    # A/A compares no kernels, and a later round only needs its perf numbers.
    say "  (profile skipped: mode=$MODE round=$round)"
    teardown "$arm" "${pgid:-}" || return 6
    return 0
  fi

  # ---- decode profile (the capture script prints success even on 404)
  ( cd "$ROOT" && python3 build_artifacts/eval_runners/prof_decode_capture.py \
      --port "$PORT" --out "$arm/cap" --bs 1 --capture-s "$CAPTURE_S" ) >"$arm/capture.log" 2>&1
  say "  capture rc=$?"
  sleep 5
  local n_pt
  n_pt=$(ls -d "$profdir"/*_ascend_pt 2>/dev/null | wc -l)
  echo "$profdir $n_pt" > "$arm/profdir.txt"
  if [ "$n_pt" -eq 0 ]; then
    say "  ❌ no *_ascend_pt under $profdir -- the profile was NOT written (the capture script says 'captured' anyway)"
  else
    say "  profile dirs: $n_pt"
    ( cd "$ROOT" && bash build_artifacts/eval_runners/prof_collect.sh "$profdir" "$arm/prof_out" ) >"$arm/collect.log" 2>&1
    say "  collect rc=$? (breakdown may refuse: we aggregate kernel_details.csv ourselves)"
    mkdir -p "$arm/prof_csv"
    find "$arm/prof_out/csv" -name kernel_details.csv -exec cp {} "$arm/prof_csv/" \; 2>/dev/null
    [ -s "$arm/prof_csv/kernel_details.csv" ] || say "  ❌ kernel_details.csv did not come back"
  fi

  # oc_scan_anomaly takes the regex FIRST: rc 0 = hit, 1 = none, 2 = nothing
  # scannable (zero coverage is not "clean", so 2 is reported as a problem).
  oc_scan_anomaly "$(oc_anomaly_regex)" "$arm/serve.log" "$arm/gsm8k.log" "$arm/perf.log" > "$arm/anomaly.txt" 2>&1
  case $? in
    0) say "  ⚠️ anomaly marker: $(head -c 200 "$arm/anomaly.txt")";;
    1) say "  anomaly scan: none in serve/gsm8k/perf logs";;
    *) say "  ❌ anomaly scan had nothing to scan (zero coverage is not 'clean')";;
  esac
  command grep -cE "10701[0-9]" "$arm/serve.log" > "$arm/n_10701x.txt" 2>/dev/null || echo 0 > "$arm/n_10701x.txt"
  say "  10701x lines in serve.log: $(cat "$arm/n_10701x.txt")"

  teardown "$arm" "${pgid:-}" || return 6
  dmesg > "$arm/dmesg_after.txt" 2>/dev/null
  oc_dmesg_new "$arm/dmesg_before.txt" "$arm/dmesg_after.txt" "$arm/dmesg_new.txt"
  say "  dmesg new lines: $(wc -l < "$arm/dmesg_new.txt")"
  return 0
}

rc=0
if [ "$MODE" = "aa" ]; then
  ARM_SPECS=("A1 false" "A2 false")   # one variable changed: none
else
  ARM_SPECS=("A false" "B true")
fi
# Interleaved, not blocked. This machine has been measured 2.5x slower 11 hours
# apart on one commit, and the old shape put arm A at 09:48 and arm B at 10:28
# -- 40 minutes during which the machine itself is a second variable. A,B,A,B
# gives paired differences plus, from the two A rounds, a drift floor a real
# benefit has to clear. Service restarts cannot be interleaved (the switch
# lives in the config), so the round is the smallest unit that can be.
for round in $(seq 1 "$ROUNDS"); do
  for spec in "${ARM_SPECS[@]}"; do
    set -- $spec
    say "── round $round, arm $1"
    run_arm "$1" "$2" "$round" || { rc=$?; say "❌ round $round arm $1 failed (rc=$rc); stopping"; break 2; }
  done
done

say "── report"
if [ "$MODE" = "aa" ]; then
  python3 "$HERE/aa_report.py" "$RUN" ${AB_RUN:+--ab-run "$AB_RUN"} 2>&1 | tee -a "$SUM"
else
  python3 "$HERE/packed_ab_report.py" "$RUN" 2>&1 | tee -a "$SUM"
fi
report_rc=${PIPESTATUS[0]}
say "■ arms rc=$rc report rc=$report_rc  run=$RUN"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$report_rc"
