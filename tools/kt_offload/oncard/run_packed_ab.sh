#!/usr/bin/env bash
# ============================================================================
#  run_packed_ab.sh <card>        —— packed_transfer A/B, one command, one card
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

# ---- snapshot first: bash reads a script by byte offset, so editing this file
#      while it runs corrupts that run. exec keeps the PID, so LEASE_PID=$$ is
#      the same process either way.
if [ -z "${PACKED_AB_SNAPSHOT:-}" ]; then
  SRC_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
  STAMP="$(date -u +%Y%m%d_%H%M%S)"
  RUN="${PACKED_AB_RUNS_DIR:-$ROOT/build_artifacts/moe_offload_ref/packed_ab}/${STAMP}"
  mkdir -p "$RUN/code" || { echo "❌ cannot create $RUN"; exit 3; }
  cp -p "$SRC_DIR"/*.sh "$SRC_DIR"/*.py "$RUN/code/" || { echo "❌ snapshot failed"; exit 3; }
  export PACKED_AB_SNAPSHOT=1 PACKED_AB_RUN="$RUN" PACKED_AB_ARGV="$0 $CARD"
  exec bash "$RUN/code/run_packed_ab.sh" "$CARD"
fi
RUN="$PACKED_AB_RUN"; HERE="$RUN/code"; ONCARD_DIR="$HERE"
# shellcheck source=oncard_lib.sh
source "$HERE/oncard_lib.sh"
# shellcheck source=packed_ab_lib.sh
source "$HERE/packed_ab_lib.sh"
SUM="$RUN/summary.txt"
say() { echo "$*" | tee -a "$SUM"; }
say "▶ packed A/B card=$CARD port=$PORT run=$RUN pid=$$ argv=[$PACKED_AB_ARGV]"
say "  VLLM_USE_BREAKABLE_CUDAGRAPH=1 (explicit; the platform defaults it to 0 and O15 refuses graph mode without it)"
say "  host-func sampling: VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=${HFLOG}s in BOTH arms (A is the baseline shape)"

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
if [ "${ONCARD_DRY:-0}" = "1" ]; then
  say "⚪ ONCARD_DRY=1: preflight passed; would acquire card $CARD as $OWNER (LEASE_PID=$$) and run arms A,B. Stopping before acquire."
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
  # pid gone is not context freed: wait for the card's process table too
  for _ in $(seq 1 60); do
    npu-smi info -t proc-mem -i "$CARD" 2>/dev/null | command grep -qE "process id|pid" || break
    sleep 2
  done
  bash "$ROOT/build_artifacts/card_lease.sh" release "$CARD" "$OWNER" >>"$arm/stop.log" 2>&1
  bash "$ROOT/build_artifacts/card_lease.sh" status > "$arm/lease_after.txt" 2>&1
  local why
  why=$(pa_card_returned "$(cat "$arm/lease_after.txt")" "$CARD" "$OWNER" "$PORT" "${pgid:-}") || {
    say "     ❌ card $CARD NOT returned: $why"; ok=0; }
  npu-smi info > "$arm/card_after.txt" 2>&1
  if [ "$ok" = 1 ]; then say "     ✅ card $CARD returned (group gone, port free, lease released)"; else say "     ❌ card $CARD NOT proven returned -- stop and look before the next arm"; fi
  return $((1 - ok))
}

run_arm() {  # $1 = A|B, $2 = packed true|false
  local name="$1" packed="$2"
  local arm="$RUN/arm$name"; mkdir -p "$arm/cap"
  local profdir="/dev/shm/dsv41_prof_packed_$name"
  rm -rf "$profdir"; mkdir -p "$profdir"
  cfg_json "$packed" > "$arm/cfg.json"
  say "═══ arm $name: packed_transfer=$packed"
  say "  cfg $(cat "$arm/cfg.json")"
  dmesg > "$arm/dmesg_before.txt" 2>/dev/null

  LEASE_PID=$$ bash "$ROOT/build_artifacts/card_lease.sh" acquire "$CARD" "$OWNER" "$PORT" >>"$SUM" 2>&1 || {
    say "  ❌ lease refused for card $CARD"; return 4; }

  ( cd "$ROOT" && setsid env \
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

  local ready=0 waited=0
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
  command grep -m1 "BREAKABLE_CUDAGRAPH" "$arm/serve.log" | sed 's/^/     /' | tee -a "$SUM" >/dev/null

  # ---- GSM8K 50, greedy, thinking off
  ( cd "$ROOT" && EV_THINKING=0 python3 build_artifacts/eval_runners/gsm8k_eval_effort.py \
      --base-url "http://127.0.0.1:$PORT/v1" --model dsv41 --out-dir "$arm/gsm8k50" \
      --max-tokens 4096 --temperature 0 --top-p 1.0 --num-threads 8 --limit "$LIMIT" ) \
      >"$arm/gsm8k.log" 2>&1
  say "  gsm8k rc=$? $(command grep -m1 'acc=' "$arm/gsm8k.log" || true)"

  # ---- TPOT: same script, same cases, n=REPEAT
  ( cd "$ROOT" && PORT="$PORT" LABEL="packed_arm$name" NOTE="packed_transfer=$packed" \
      CASES="bs1 1 16" REPEAT="$REPEAT" PERF_OUT="$arm/perf" \
      bash build_artifacts/perf_baseline.sh ) >"$arm/perf.log" 2>&1
  say "  perf rc=$? -> $arm/perf/summary.txt"

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
for spec in "A false" "B true"; do
  set -- $spec
  run_arm "$1" "$2" || { rc=$?; say "❌ arm $1 failed (rc=$rc); stopping before the next arm"; break; }
done

say "── report"
python3 "$HERE/packed_ab_report.py" "$RUN" 2>&1 | tee -a "$SUM"
report_rc=${PIPESTATUS[0]}
say "■ arms rc=$rc report rc=$report_rc  run=$RUN"
[ "$rc" -ne 0 ] && exit "$rc"
exit "$report_rc"
