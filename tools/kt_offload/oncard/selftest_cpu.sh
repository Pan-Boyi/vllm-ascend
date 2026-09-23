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
ROOT_VLLM_ASCEND="$(cd "$D/../../.." && pwd)/vllm_ascend"
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
ERRC="107011|107015|107025|107027|107030"
c=$(oc_count "$ERRC" "$W/clean.log" "$W/plog"); [ "$c" = 0 ]; ok $? "ACL err count on clean inputs = $c"
printf 'rt error 107011 x\n107027 y\n' > "$W/plog/debug/q.log"
c=$(oc_count "$ERRC" "$W/plog"); [ "$c" = 2 ]; ok $? "the list sees both 107011 and 107027 (design 要求这五个都为 0): $c"
# 变异臂：旧口径在同一份输入上只数到 1 ⇒ 它会漏掉 10702x 这一档，而 107025（侧流捕获失败）就在这一档。
c=$(oc_count "10701[0-9]" "$W/plog"); [ "$c" = 1 ]; ok $? "the old pattern misses 107027 (counts $c, not 2) -- that is the bug"
printf 'capture failed 107025 z
' > "$W/plog/debug/r.log"
c=$(oc_count "$ERRC" "$W/plog"); [ "$c" = 3 ]; ok $? "107025 is counted by the list ($c, was 2) -- the whole point of the fix"
c=$(oc_count "10701[0-9]" "$W/plog"); [ "$c" = 1 ]; ok $? "the old pattern still misses 107025 (counts $c) -- that is what shipped as \"0 lines\""
rm -f "$W/plog/debug/r.log"

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
echo "== 7. t4 (overlap ceiling): refuses what it cannot read, and is honest about what it does not check"
ONCARD_DRY=1 python3 "$D/t4_overlap_ceiling.py" > "$W/t4dry.out" 2>&1; r=$?
[ $r -eq 0 ]; ok $? "t4 dry run exits 0 without touching a device (rc $r)"
command grep -q "no device was touched" "$W/t4dry.out"; ok $? "t4 dry run says it touched nothing"
command grep -q "This test cannot show form C is correct" "$W/t4dry.out"; ok $? "t4 prints the not-a-correctness-test line"
command grep -q "sglang-shape" "$W/t4dry.out"; ok $? "t4 plan names the sglang arm"
python3 "$D/t4_overlap_ceiling.py" --reps 5 > "$W/t4few.out" 2>&1; r=$?
[ $r -eq 2 ]; ok $? "t4 refuses fewer than 20 reps (rc $r)"
python3 - "$D" <<'PY' > "$W/t4model.out" 2>&1; r=$?
import importlib.util, sys
spec = importlib.util.spec_from_file_location("t4_model", sys.argv[1] + "/t4_model.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
assert not M.instrument_control(3.0, 500.0)[0], "instrument control stayed green with an invisible fat node"
assert not M.negative_control([5.0]*24, [100.0]*24, 0.35)[0], "negative control stayed green on a hollow graph"
assert M.verdict(M.ceiling(400.0, 100.0, 500.0, 40))["worth_it"] is False, "a 500us node did not kill form C"
assert M.cross_check(5.0, 62.5)["state"] == "inconsistent", "two sources 12x apart read as agreement"
try:
    M.stats([1.0]*19); raise SystemExit("stats accepted 19 samples")
except ValueError:
    pass
print("t4_model positive controls all red where they must be")
PY
[ $r -eq 0 ]; ok $? "t4 decision model: every control goes red when it must ($(tail -n 1 "$W/t4model.out"))"

echo "== 8. t5 (auto_dispatch_capture): refuses a run that could not tell a stale replay from a fresh one"
ONCARD_DRY=1 python3 "$D/t5_auto_dispatch.py" > "$W/t5dry.out" 2>&1; r=$?
[ $r -eq 0 ]; ok $? "t5 dry run exits 0 without touching a device (rc $r)"
command grep -q "no device was touched" "$W/t5dry.out"; ok $? "t5 dry run says it touched nothing"
command grep -q "positive control" "$W/t5dry.out"; ok $? "t5 plan names the positive control arm"
command grep -q "capture success alone is NOT a pass" "$W/t5dry.out"; ok $? "t5 separates capture from replay"
python3 "$D/t5_auto_dispatch.py" --inputs 2 > "$W/t5few.out" 2>&1; r=$?
[ $r -eq 2 ]; ok $? "t5 refuses fewer than 3 inputs (rc $r)"
python3 "$D/t5_auto_dispatch.py" --layers 1 > "$W/t5lay.out" 2>&1; r=$?
[ $r -eq 2 ]; ok $? "t5 refuses fewer than 3 layers, so kt slot reuse stays covered (rc $r)"
command grep -q "auto_dispatch_capture=auto" "$D/t5_auto_dispatch.py"; ok $? "t5 passes the parameter it is testing"
! command grep -q "auto_dispatch_capture" "$ROOT_VLLM_ASCEND/compilation/acl_graph.py" 2>/dev/null \
  || command grep -q "auto_dispatch_capture=True" "$ROOT_VLLM_ASCEND/compilation/acl_graph.py" 2>/dev/null
ok $? "production capture path untouched by this test (acl_graph.py still takes the default)"

bash "$D/run_oncard.sh" x t0 > /dev/null 2>&1; r=$?; [ $r -eq 2 ]; ok $? "bad card -> usage rc $r"
bash "$D/run_oncard.sh" 0 t9 > /dev/null 2>&1; r=$?; [ $r -eq 2 ]; ok $? "unknown test -> usage rc $r"

echo
echo "selftest: $N checks, $F failed; scratch $W"
[ "$N" -gt 0 ] && [ "$F" -eq 0 ] && { echo "SELFTEST: PASS"; rm -rf "$W"; exit 0; }
echo "SELFTEST: FAIL"; exit 1
