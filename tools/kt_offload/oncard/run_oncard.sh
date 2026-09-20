#!/usr/bin/env bash
# ============================================================================
#  run_oncard.sh <card> <test> [test args...]    test = t0 | t1 | t2 | t3 | t4
#
#  The only way the single-layer on-card checks touch a card. The shell enforces:
#   1. lease first: build_artifacts/card_lease.sh acquire <card> dsv41_vllm_singlecard
#      with LEASE_PID=$$ (this shell lives until the test and its cleanup are done);
#      no lease -> exit, nothing touched. Before the lease only host-side checks run.
#   2. lease and test live and die together: EXIT trap stops the test's process group,
#      waits for the card's process table to drop our pid (pid gone != context freed),
#      then releases. A watchdog releases if this shell is SIGKILLed; it exits as soon
#      as this shell is gone (zombie counts as gone), never outlives it.
#   3. one card: ASCEND_RT_VISIBLE_DEVICES=<card>, no torch.distributed / HCCL init.
#   4. private caches under /dev/shm/dsv41_singlecard/ (TMPDIR, TMP, TORCH_EXTENSIONS_DIR,
#      TRITON_CACHE_DIR); TEMP is NOT exported (the GSM8K runner uses it as temperature).
#   5. anomaly markers (oncard_common.ANOMALY_PATTERNS: AICore / aicore exception /
#      MTE_ERROR / EZ9999 ...) in the test output or plog -> stop at once, no retry,
#      keep everything, rc 20.
#  Artifacts: build_artifacts/moe_offload_ref/oncard_runs/<UTC stamp>_<test>/
#   provenance.json, test.log, ascend_log/ (plog), card_before/after.txt,
#   dmesg_before/after/new/card.txt, meminfo, summary.txt, t*_result.json, code/ (snapshot).
#
#  rc: 0 pass | 1 test verdict FAIL | 2 usage | 3 preflight (no card touched) |
#      4 lease refused | 5 card not idle / dmesg gate | 20 anomaly | 124 timeout | other = test rc
#  Env: ONCARD_TIMEOUT_S (default per test), ONCARD_SKIP_DMESG_GATE=1 (logged),
#       ONCARD_DRY=1 (stop right before acquire: prints the plan; used by selftest).
# ============================================================================
set -u
set -o pipefail

ROOT=/workspace/code/dsv4_single_card/vllm_project
OWNER=dsv41_vllm_singlecard
CARD="${1:-}"; TEST="${2:-}"
[ $# -ge 2 ] && shift 2
usage() { echo "usage: $0 <card> <t0|t1|t2|t3|t4> [test args...]"; exit 2; }
[[ "$CARD" =~ ^[0-9]+$ ]] || usage
case "$TEST" in
  t0) PY=t0_offload_off.py; DEF_TIMEOUT=2400;;
  t1) PY=t1_offload_on.py;  DEF_TIMEOUT=3600;;
  t2) PY=t2_graph.py;       DEF_TIMEOUT=2400;;
  t3) PY=t3_side_stream.py; DEF_TIMEOUT=2400;;
  t4) PY=t4_overlap_ceiling.py; DEF_TIMEOUT=1800;;
  *) usage;;
esac

# ---- snapshot first (bash reads scripts by byte offset: editing a running copy breaks it).
#      exec keeps the PID, so LEASE_PID=$$ below is the same process either way.
if [ -z "${ONCARD_SNAPSHOT:-}" ]; then
  SRC_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
  STAMP="$(date -u +%Y%m%d_%H%M%S)"
  RUN="${ONCARD_RUNS_DIR:-$ROOT/build_artifacts/moe_offload_ref/oncard_runs}/${STAMP}_${TEST}"
  mkdir -p "$RUN/code" || { echo "❌ cannot create $RUN"; exit 3; }
  cp -p "$SRC_DIR"/*.sh "$SRC_DIR"/*.py "$RUN/code/" || { echo "❌ snapshot failed"; exit 3; }
  export ONCARD_SNAPSHOT=1 ONCARD_RUN_DIR="$RUN" ONCARD_SRC_DIR="$SRC_DIR" ONCARD_ARGV="$0 $CARD $TEST $*"
  exec bash "$RUN/code/run_oncard.sh" "$CARD" "$TEST" "$@"
fi
RUN="$ONCARD_RUN_DIR"; HERE="$RUN/code"
ONCARD_DIR="$HERE"
# shellcheck source=oncard_lib.sh
source "$HERE/oncard_lib.sh"
LOG="$RUN/test.log"; SUM="$RUN/summary.txt"
say() { echo "$*" | tee -a "$SUM"; }
say "▶ run_oncard card=$CARD test=$TEST run=$RUN pid=$$ argv=[$ONCARD_ARGV]"

# ---- preflight: host-side only, before any lease (safe to fail here: nothing touched)
command -v npu-smi >/dev/null 2>&1 || { say "❌ preflight: no npu-smi here -> no card environment; stopping before acquire"; exit 3; }
[ -x "$ROOT/build_artifacts/card_lease.sh" ] || { say "❌ preflight: card_lease.sh missing"; exit 3; }
[ -f "$ROOT/build_artifacts/env_dsv41.sh" ] || { say "❌ preflight: env_dsv41.sh missing"; exit 3; }
[ -d "$ROOT/worktrees/moe-offload" ] && [ -d "$ROOT/worktrees/moe-offload-vllm" ] || { say "❌ preflight: trees missing"; exit 3; }
KT_SITE="$ROOT/build_artifacts/moe_offload_ref/kt_site"
KT_WHEEL=$(ls "$ROOT"/build_artifacts/moe_offload_ref/kt_wheel/kt_kernel-*.whl 2>/dev/null | head -n 1)
[ -d "$KT_SITE/kt_kernel" ] && [ -n "$KT_WHEEL" ] || { say "❌ preflight: kt_site / kt wheel missing"; exit 3; }
AVAIL_KB=$(oc_meminfo_avail_kb)
NEED_KB=$(( ${ONCARD_MIN_AVAIL_GB:-150} * 1024 * 1024 ))
[ "${AVAIL_KB:-0}" -ge "$NEED_KB" ] || { say "❌ preflight: MemAvailable $((AVAIL_KB/1048576)) GiB < ${ONCARD_MIN_AVAIL_GB:-150} GiB (plan §2 rule)"; exit 3; }
if [ "${ONCARD_DRY:-0}" = "1" ]; then
  say "⚪ ONCARD_DRY=1: preflight passed; would acquire card $CARD as $OWNER (LEASE_PID=$$) and run $PY. Stopping before acquire."
  exit 0
fi

# ---- environment (sourcing touches no device)
export DSV41_VLLM_TREE="$ROOT/worktrees/moe-offload-vllm"
set +u
# shellcheck source=/dev/null
source "$ROOT/build_artifacts/env_dsv41.sh" moe-offload "$CARD" >> "$SUM" 2>&1 || { say "❌ env_dsv41.sh failed"; exit 3; }
set -u
unset LD_PRELOAD
export ASCEND_RT_VISIBLE_DEVICES="$CARD"
PRIV=/dev/shm/dsv41_singlecard
export TMPDIR="$PRIV/tmp" TMP="$PRIV/tmp" TORCH_EXTENSIONS_DIR="$PRIV/torch_extensions" TRITON_CACHE_DIR="$PRIV/triton_cache"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" || { say "❌ cannot create $PRIV"; exit 3; }
export ASCEND_PROCESS_LOG_PATH="$RUN/ascend_log"; mkdir -p "$ASCEND_PROCESS_LOG_PATH"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$KT_SITE"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
PYBIN="${ONCARD_PYTHON:-/usr/local/python3.12.13/bin/python3}"
ANOM="$(oc_anomaly_regex)" || { say "❌ cannot read anomaly patterns"; exit 3; }

# ---- provenance (before the card: the record exists even if the run dies)
"$PYBIN" - "$RUN/provenance.json" <<EOF || { say "❌ provenance failed"; exit 3; }
import json, os, subprocess, sys, hashlib, socket
def git(t, *a):
    return subprocess.run(["git", "-C", t, *a], capture_output=True, text=True).stdout.strip()
def sha(p):
    h = hashlib.sha256(); h.update(open(p, "rb").read()); return h.hexdigest()
va, vl = "$ROOT/worktrees/moe-offload", "$ROOT/worktrees/moe-offload-vllm"
code = "$HERE"
json.dump({
  "argv": os.environ.get("ONCARD_ARGV"), "card": "$CARD", "test": "$TEST", "script": "$PY",
  "host": socket.gethostname(), "owner": "$OWNER", "lease_pid": $$,
  "vllm_ascend": {"path": va, "head": git(va, "rev-parse", "HEAD"), "branch": git(va, "rev-parse", "--abbrev-ref", "HEAD"),
                  "dirty_files": len([l for l in git(va, "status", "--porcelain").splitlines() if l])},
  "vllm": {"path": vl, "head": git(vl, "rev-parse", "HEAD"), "branch": git(vl, "rev-parse", "--abbrev-ref", "HEAD"),
           "dirty_files": len([l for l in git(vl, "status", "--porcelain").splitlines() if l])},
  "kt_wheel": {"path": "$KT_WHEEL", "sha256": sha("$KT_WHEEL")},
  "kt_site": "$KT_SITE",
  "code_snapshot_sha256": {f: sha(os.path.join(code, f)) for f in sorted(os.listdir(code))},
  "env": {k: os.environ.get(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "TMPDIR", "TMP", "TEMP", "TORCH_EXTENSIONS_DIR",
          "TRITON_CACHE_DIR", "ASCEND_PROCESS_LOG_PATH", "PYTHONPATH", "LD_PRELOAD", "DSV41_VLLM_TREE", "VLLM_VERSION")},
  "python": "$PYBIN",
}, open(sys.argv[1], "w"), indent=1)
EOF

# ---- lease
LEASED=0; TEST_PID=""; WD_PID=""; RC=""
cleanup() {
  local rc=$?
  [ -n "$RC" ] && rc=$RC
  trap - EXIT INT TERM
  if [ -n "$TEST_PID" ] && oc_alive "$TEST_PID"; then
    say "⚪ stopping test process group $TEST_PID"
    kill -TERM -- "-$TEST_PID" 2>/dev/null
    for _ in $(seq 1 60); do oc_alive "$TEST_PID" || break; sleep 2; done
    oc_alive "$TEST_PID" && kill -KILL -- "-$TEST_PID" 2>/dev/null
  fi
  if [ "$LEASED" = 1 ]; then
    # pid gone != device context freed: wait until the card's process table drops it.
    local waited=0
    while [ -n "$TEST_PID" ] && [ "$waited" -lt "${ONCARD_TEARDOWN_WAIT_S:-1800}" ]; do
      out=$(timeout 20 npu-smi info -t proc-mem -i "$CARD" 2>/dev/null) || break
      command grep -qE "Process id *: *${TEST_PID}\b|Process id:${TEST_PID}\b" <<< "$out" || break
      [ $((waited % 60)) -eq 0 ] && say "⚪ card $CARD still lists pid $TEST_PID (teardown), waited ${waited}s"
      sleep 5; waited=$((waited + 5))
    done
    timeout 20 npu-smi info -t usages -i "$CARD" > "$RUN/card_after.txt" 2>&1
    timeout 20 npu-smi info -t proc-mem -i "$CARD" >> "$RUN/card_after.txt" 2>&1
    dmesg 2>/dev/null | tail -n 200 > "$RUN/dmesg_after.txt"
    oc_dmesg_new "$RUN/dmesg_before.txt" "$RUN/dmesg_after.txt" "$RUN/dmesg_new.txt"
    oc_dmesg_card "$CARD" "$RUN/dmesg_new.txt" > "$RUN/dmesg_card.txt"
    say "⚪ dmesg: $(head -n 1 "$RUN/dmesg_card.txt")"
    echo "after $(date -Is) MemAvailable_kB $(oc_meminfo_avail_kb)" >> "$RUN/meminfo.txt"
    [ -n "$WD_PID" ] && kill "$WD_PID" 2>/dev/null
    "$ROOT/build_artifacts/card_lease.sh" release "$CARD" "$OWNER" >> "$SUM" 2>&1
    say "⚪ lease released at $(oc_ms) ms (acquired at ${ACQ_MS:-?} ms)"
  elif [ -n "$WD_PID" ]; then
    kill "$WD_PID" 2>/dev/null
  fi
  say "■ rc=$rc"
  exit "$rc"
}
trap cleanup EXIT
trap 'RC=${RC:-130}; exit' INT TERM

say "▶ acquire card $CARD as $OWNER (LEASE_PID=$$)"
if ! LEASE_PID=$$ "$ROOT/build_artifacts/card_lease.sh" acquire "$CARD" "$OWNER" >> "$SUM" 2>&1; then
  say "❌ lease refused (see summary); nothing ran"; RC=4; exit 4
fi
LEASED=1; ACQ_MS=$(oc_ms)
say "✅ lease acquired at $ACQ_MS ms"

# watchdog: if this shell dies without its trap (SIGKILL), stop the test and release.
oc_watchdog "$$" "$RUN/test.pid" "$ROOT/build_artifacts/card_lease.sh" release "$CARD" "$OWNER" >> "$SUM" 2>&1 &
WD_PID=$!

# ---- card state before
timeout 20 npu-smi info -t usages -i "$CARD" > "$RUN/card_before.txt" 2>&1 || { say "❌ npu-smi usages timed out"; RC=5; exit 5; }
timeout 20 npu-smi info -t proc-mem -i "$CARD" >> "$RUN/card_before.txt" 2>&1
PCT=$(awk -F: '/HBM Usage Rate/{gsub(/ /,"",$2); print $2; exit}' "$RUN/card_before.txt")
[[ "$PCT" =~ ^[0-9]+$ ]] || { say "❌ cannot parse HBM usage of card $CARD"; RC=5; exit 5; }
[ "$PCT" -lt "${IDLE_PCT:-20}" ] || { say "❌ card $CARD HBM ${PCT}% busy before start"; RC=5; exit 5; }
if command grep -qE "Process id *: *[0-9]+|Process id:[0-9]+" "$RUN/card_before.txt"; then
  say "❌ card $CARD process table not empty before start"; RC=5; exit 5
fi
dmesg 2>/dev/null | tail -n 200 > "$RUN/dmesg_before.txt"
if [ "${ONCARD_SKIP_DMESG_GATE:-0}" = "1" ]; then
  say "⚠️ ONCARD_SKIP_DMESG_GATE=1: shared dmesg gate skipped (decision recorded here)"
else
  "$ROOT/build_artifacts/check_card_dmesg.sh" "$CARD" >> "$SUM" 2>&1; g=$?
  [ "$g" -eq 0 ] || { say "❌ dmesg gate rc=$g for card $CARD (3 = errors in window, 1 = cannot judge); not running"; RC=5; exit 5; }
fi
echo "before $(date -Is) MemAvailable_kB $(oc_meminfo_avail_kb)" > "$RUN/meminfo.txt"
say "  card $CARD HBM ${PCT}% idle; MemAvailable $(( $(oc_meminfo_avail_kb) / 1048576 )) GiB"

# ---- the test, in its own process group (killable as a group, never by name)
T_START=$(oc_ms)
setsid "$PYBIN" "$HERE/$PY" --out "$RUN" "$@" > "$LOG" 2>&1 < /dev/null &
TEST_PID=$!
echo "$TEST_PID" > "$RUN/test.pid"
# setsid takes effect once the child has exec'd; give it up to 5 s before judging
for _ in $(seq 1 50); do
  PGID=$(ps -o pgid= -p "$TEST_PID" 2>/dev/null | tr -d ' ')
  [ "$PGID" = "$TEST_PID" ] && break; sleep 0.1
done
if [ "$PGID" != "$TEST_PID" ]; then
  # group kill would hit the wrong group; stop by pid and refuse to continue
  say "❌ test pid $TEST_PID is not its own process group (pgid '$PGID')"; kill -TERM "$TEST_PID" 2>/dev/null; RC=3; exit 3
fi
say "▶ test pid $TEST_PID started at $T_START ms: $PY $*"
TIMEOUT_S="${ONCARD_TIMEOUT_S:-$DEF_TIMEOUT}"
ANOMALY=""; min_avail=$(oc_meminfo_avail_kb); hbm_max="$PCT"
oc_sample_hook() {  # every 8th poll: HBM usage, host MemAvailable, test VmHWM
  local a p
  a=$(oc_meminfo_avail_kb); [ "$a" -lt "$min_avail" ] && min_avail=$a
  [ $(( $2 % 8 )) -eq 0 ] || return 0
  p=$(timeout 10 npu-smi info -t usages -i "$CARD" 2>/dev/null | awk -F: '/HBM Usage Rate/{gsub(/ /,"",$2); print $2; exit}')
  [[ "$p" =~ ^[0-9]+$ ]] && [ "$p" -gt "$hbm_max" ] && hbm_max=$p
  command grep -hE "VmHWM" "/proc/$1/status" > "$RUN/test_vmhwm.txt" 2>/dev/null
  return 0
}
# not in $( ): the sampling hook must update this shell's variables
oc_monitor "$TEST_PID" "$ANOM" "$TIMEOUT_S" "$LOG" "$ASCEND_PROCESS_LOG_PATH" > "$RUN/monitor.txt"; MRC=$?
MON=$(cat "$RUN/monitor.txt")
case $MRC in
  0) ;;
  20) ANOMALY="${MON#ANOMALY }"; say "🛑 anomaly marker: $ANOMALY -> stopped at once, no retry; plog kept at $ASCEND_PROCESS_LOG_PATH";;
  *) RC=$MRC; say "🛑 $MON -> stopped";;
esac
wait "$TEST_PID" 2>/dev/null; TRC=$?
T_END=$(oc_ms)
# the final scan also covers a marker written in the last two seconds
if [ -z "$ANOMALY" ] && hit=$(oc_scan_anomaly "$ANOM" "$LOG" "$ASCEND_PROCESS_LOG_PATH"); then
  ANOMALY="$hit"; say "🛑 anomaly marker (final scan): $hit"
fi
N1070=$(oc_count "10701[0-9]" "$LOG" "$ASCEND_PROCESS_LOG_PATH")
VERD=$(command grep -E "^VERDICT:" "$LOG" | tail -n 1)
say "  test rc=$TRC wall $(( (T_END - T_START) / 1000 ))s; $VERD; 10701x lines (log+plog) $N1070"
say "  MemAvailable min during run $((min_avail/1048576)) GiB; HBM usage max ${hbm_max}% (sampled); $(cat "$RUN/test_vmhwm.txt" 2>/dev/null)"
if [ -n "$ANOMALY" ]; then RC=20
elif [ -n "$RC" ]; then :
elif [ "$TRC" -ne 0 ]; then RC=$TRC
elif [ -z "$VERD" ]; then say "❌ test exited 0 without a VERDICT line"; RC=1
else RC=0; fi
exit "$RC"
