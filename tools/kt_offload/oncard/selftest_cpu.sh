#!/usr/bin/env bash
# CPU self-check for tools/kt_offload/oncard/. Touches no card, takes no lease, calls no npu-smi.
#   bash tools/kt_offload/oncard/selftest_cpu.sh
# Each check prints PASS/FAIL with what it saw; rc 0 only if all pass and the count is non-zero.
set -u
set -o pipefail
D="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
TREE="$(cd "$D/../../.." && pwd)"
VLLM_TREE="$(cd "$TREE/../moe-offload-vllm" && pwd)"
PY="${ONCARD_PYTHON:-/usr/local/python3.12.13/bin/python3}"
ONCARD_DIR="$D"
# shellcheck source=oncard_lib.sh
source "$D/oncard_lib.sh"
W="$(mktemp -d "${TMPDIR:-/tmp}/oncard_selftest.XXXXXX")"
N=0; F=0
ok() { N=$((N+1)); if [ "$1" = 0 ]; then echo "  [PASS] $2"; else F=$((F+1)); echo "  [FAIL] $2"; fi; }

echo "== 1. syntax"
for f in "$D"/*.sh; do bash -n "$f"; ok $? "bash -n $(basename "$f")"; done
for f in "$D"/*.py; do "$PY" -m py_compile "$f" 2> "$W/pyc.err"; ok $? "py_compile $(basename "$f") $(cat "$W/pyc.err")"; done

echo "== 2. pure-torch half (no device)"
ASCEND_RT_VISIBLE_DEVICES="" TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH="$TREE:$VLLM_TREE" \
  env -u LD_PRELOAD "$PY" "$D/selftest_cpu.py" > "$W/py.log" 2>&1; rc=$?
command grep -E "^\s+\[(PASS|FAIL)\]|^VERDICT" "$W/py.log"
ok $rc "selftest_cpu.py rc=$rc ($(command grep -c '\[PASS\]' "$W/py.log") pass lines)"

echo "== 3. anomaly markers: positive controls on forged logs"
RE="$(oc_anomaly_regex)"; ok $? "anomaly regex readable: $RE"
i=0
for line in "[ERROR] RUNTIME(1234): AICore Kernel task happen error, retCode=0x31" \
            "rtStreamSynchronize failed: aicore exception" \
            "[ERROR] 507015 aicore error" \
            "AIVEC exception on core 3" \
            "fault: MTE_ERROR at addr 0x1" \
            "EZ9999: Inner Error!" \
            "EE9999: internal" \
            "EL0004: task abort" \
            "device task abort, stream 3" \
            "Kernel task happen error"; do
  i=$((i+1)); printf 'start\nnormal line\n%s\nend\n' "$line" > "$W/forged_$i.log"
  hit=$(oc_scan_anomaly "$RE" "$W/forged_$i.log"); ok $? "forged line $i detected -> ${hit}"
done
printf 'start\nkt_offload plan: 40 layers\nVERDICT: PASS\n' > "$W/clean.log"
oc_scan_anomaly "$RE" "$W/clean.log" > /dev/null; r=$?; [ $r -eq 1 ]; ok $? "clean log -> no hit (rc $r, want 1)"
oc_scan_anomaly "$RE" "$W/does_not_exist" > /dev/null; r=$?; [ $r -eq 2 ]; ok $? "nothing scannable -> rc $r (want 2, not 'clean')"
mkdir -p "$W/plog/debug"; printf 'x\n[ERROR] EZ9999 from plog\n' > "$W/plog/debug/p.log"
hit=$(oc_scan_anomaly "$RE" "$W/clean.log" "$W/plog"); ok $? "marker inside a plog directory detected -> $hit"
c=$(oc_count "10701[0-9]" "$W/clean.log" "$W/plog"); [ "$c" = 0 ]; ok $? "10701x count on clean inputs = $c"
printf 'rt error 107011 x\n107027 y\n' > "$W/plog/debug/q.log"
c=$(oc_count "10701[0-9]" "$W/plog"); [ "$c" = 1 ]; ok $? "10701x count sees 107011 only (107027 is not 10701x): $c"

echo "== 4. monitor stops a test that prints a marker (process group, no retry)"
setsid bash -c "sleep 2; echo 'EZ9999 fake'; sleep 60" > "$W/mon.log" 2>&1 < /dev/null &
tp=$!; sleep 0.3
t0=$(oc_ms); out=$(OC_MONITOR_POLL_S=1 oc_monitor "$tp" "$RE" 60 "$W/mon.log"); r=$?; dt=$(( $(oc_ms) - t0 ))
sleep 1; oc_alive "$tp"; alive=$?
[ $r -eq 20 ] && [ $alive -ne 0 ] && [ $dt -lt 10000 ]; ok $? "monitor rc=$r ('$out') after ${dt} ms; test group gone: $([ $alive -ne 0 ] && echo yes || echo NO)"
wait "$tp" 2>/dev/null
setsid bash -c "sleep 1; echo fine" > "$W/mon2.log" 2>&1 < /dev/null &
tp=$!; out=$(OC_MONITOR_POLL_S=1 oc_monitor "$tp" "$RE" 60 "$W/mon2.log"); r=$?
[ $r -eq 0 ]; ok $? "monitor on a clean test returns 0 (rc=$r)"; wait "$tp" 2>/dev/null
setsid bash -c "sleep 30" > /dev/null 2>&1 < /dev/null &
tp=$!; out=$(OC_MONITOR_POLL_S=1 oc_monitor "$tp" "$RE" 2 "$W/mon2.log"); r=$?
sleep 1; oc_alive "$tp"; alive=$?
[ $r -eq 124 ] && [ $alive -ne 0 ]; ok $? "monitor timeout -> rc=$r, group stopped"; wait "$tp" 2>/dev/null

echo "== 5. watchdog dies with the main shell and releases"
sleep 30 & fmain=$!
setsid sleep 100 < /dev/null > /dev/null 2>&1 & ftest=$!
echo "$ftest" > "$W/test.pid"
rel() { echo "released $*" > "$W/released"; }
( OC_WATCHDOG_POLL_S=1 oc_watchdog "$fmain" "$W/test.pid" rel card9 selftest ) > "$W/wd.log" 2>&1 &
wd=$!
sleep 2.5; oc_alive "$wd"; ok $? "watchdog alive while main alive"
[ ! -e "$W/released" ]; ok $? "no release while main alive"
kill "$fmain"; wait "$fmain" 2>/dev/null
for _ in $(seq 1 10); do oc_alive "$wd" || break; sleep 0.5; done
oc_alive "$wd"; wa=$?; oc_alive "$ftest"; ta=$?
[ $wa -ne 0 ]; ok $? "watchdog exited after main died"
[ $ta -ne 0 ]; ok $? "watchdog stopped the test group"
[ -s "$W/released" ]; ok $? "watchdog ran the release: $(cat "$W/released" 2>/dev/null)"
wait "$ftest" 2>/dev/null; wait "$wd" 2>/dev/null

echo "== 6. run_oncard.sh stops before acquire without a card environment"
LD="$W/leases"; mkdir -p "$LD"
PATH=/usr/bin:/bin CARD_LEASE_DIR="$LD" ONCARD_RUNS_DIR="$W/runs_nocard" bash "$D/run_oncard.sh" 0 t0 > "$W/nocard.out" 2>&1; r=$?
[ $r -eq 3 ]; ok $? "no npu-smi on PATH -> rc $r (want 3)"
command grep -q "stopping before acquire" "$W/nocard.out"; ok $? "says it stopped before acquire"
! command grep -q "acquire card" "$W/nocard.out"; ok $? "never reached the acquire step"
[ -z "$(ls -A "$LD")" ]; ok $? "lease dir untouched ($(ls -A "$LD" | wc -l) files)"
[ -f "$(ls -d "$W"/runs_nocard/*_t0 2>/dev/null | head -n 1)/code/run_oncard.sh" ]; ok $? "run dir has the code snapshot"
CARD_LEASE_DIR="$LD" ONCARD_DRY=1 ONCARD_RUNS_DIR="$W/runs_dry" bash "$D/run_oncard.sh" 0 t1 > "$W/dry.out" 2>&1; r=$?
{ [ $r -eq 0 ] && command grep -q "Stopping before acquire" "$W/dry.out"; } || { [ $r -eq 3 ] && command grep -q "preflight" "$W/dry.out"; }
ok $? "ONCARD_DRY=1 stops before acquire (rc $r: $(tail -n 1 "$W/dry.out"))"
[ -z "$(ls -A "$LD")" ]; ok $? "lease dir still untouched"
bash "$D/run_oncard.sh" x t0 > /dev/null 2>&1; r=$?; [ $r -eq 2 ]; ok $? "bad card -> usage rc $r"
bash "$D/run_oncard.sh" 0 t9 > /dev/null 2>&1; r=$?; [ $r -eq 2 ]; ok $? "unknown test -> usage rc $r"

echo
echo "selftest: $N checks, $F failed; scratch $W"
[ "$N" -gt 0 ] && [ "$F" -eq 0 ] && { echo "SELFTEST: PASS"; rm -rf "$W"; exit 0; }
echo "SELFTEST: FAIL"; exit 1
