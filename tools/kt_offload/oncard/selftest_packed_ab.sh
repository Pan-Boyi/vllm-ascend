#!/usr/bin/env bash
# Self-check for run_packed_ab.sh + packed_ab_report.py. Touches no card.
#   bash selftest_packed_ab.sh
# Every check names what would go wrong without it. The red arms matter most:
# a judge that cannot fail is not a judge.
set -u
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/packed_ab_selftest.XXXXXX")"
PASS=0; FAIL=0
ok()  { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
chk() { if [ "$1" = "$2" ]; then ok "$3 (rc=$1)"; else bad "$3 (rc=$1, want $2)"; fi; }

echo "── syntax"
bash -n "$HERE/run_packed_ab.sh" && ok "run_packed_ab.sh parses" || bad "run_packed_ab.sh does not parse"
bash -n "$HERE/packed_ab_lib.sh" && ok "packed_ab_lib.sh parses" || bad "packed_ab_lib.sh does not parse"
python3 -m py_compile "$HERE/packed_ab_report.py" && ok "packed_ab_report.py compiles" || bad "report does not compile"

echo "── usage and dry run"
bash "$HERE/run_packed_ab.sh" >/dev/null 2>&1; chk $? 2 "no card argument -> usage"
bash "$HERE/run_packed_ab.sh" x9 >/dev/null 2>&1; chk $? 2 "non-numeric card -> usage"
out=$(ONCARD_DRY=1 PACKED_AB_RUNS_DIR="$TMP/runs" bash "$HERE/run_packed_ab.sh" 3 2>&1); rc=$?
chk "$rc" 0 "ONCARD_DRY=1 stops before acquire"
echo "$out" | command grep -q "Stopping before acquire" && ok "dry run says where it stopped" || bad "dry run message missing"
echo "$out" | command grep -q "VLLM_USE_BREAKABLE_CUDAGRAPH=1" && ok "breakable value is printed" || bad "breakable value not printed"
ls "$TMP"/runs/*/code/run_packed_ab.sh >/dev/null 2>&1 && ok "code snapshot taken" || bad "no code snapshot"

echo "── teardown predicates (the 'card not returned' check must be able to go red)"
source "$HERE/packed_ab_lib.sh"
held="card 3 owner dsv41_vllm_singlecard port 8041"
free="card 7 owner someone_else port 8010"
pa_lease_still_held "$held" 3 dsv41_vllm_singlecard && ok "a held lease is detected" || bad "held lease missed"
pa_lease_still_held "$free" 3 dsv41_vllm_singlecard && bad "a free lease read as held" || ok "a free lease reads as free"
pa_lease_still_held "$held" 3 someone_else && bad "another owner read as ours" || ok "another owner is not ours"
pa_group_alive "" && bad "empty pgid read as alive" || ok "empty pgid is not alive"
# our own *process group*, not $$: this script is not always its group
# leader (it is not when a tool launches it inside another group), and
# kill -0 on a pid that is no group id reads as dead
self_pgid=$(ps -o pgid= -p $$ | tr -d " ")
pa_group_alive "$self_pgid" && ok "our own group reads as alive" || bad "our own group read as dead"
why=$(pa_card_returned "$held" 3 dsv41_vllm_singlecard 65000 ""); rc=$?
chk "$rc" 1 "card NOT returned when the lease is still held"
echo "$why" | command grep -q "lease still held" && ok "it names why" || bad "it does not name why"
pa_card_returned "$free" 3 dsv41_vllm_singlecard 65000 "" >/dev/null; chk $? 0 "card returned when nothing holds it"

echo "── anomaly scan positive control"
printf 'all good\naicore exception happened\n' > "$TMP/anom.log"
SCAN="source '$HERE/oncard_lib.sh'; oc_scan_anomaly \"\$(oc_anomaly_regex)\""
ONCARD_DIR="$HERE" bash -c "$SCAN '$TMP/anom.log'" | command grep -qi "aicore" \
  && ok "anomaly marker is detected" || bad "anomaly marker missed (the scanner cannot see a real one)"
printf 'all good\nnothing to see\n' > "$TMP/clean.log"
out=$(ONCARD_DIR="$HERE" bash -c "$SCAN '$TMP/clean.log'"); rc=$?
[ -z "$out" ] && [ "$rc" = 1 ] && ok "a clean log reports nothing (rc=1)" || bad "a clean log gave rc=$rc out='$out'"
ONCARD_DIR="$HERE" bash -c "$SCAN '$TMP/does_not_exist.log'" >/dev/null 2>&1
chk $? 2 "nothing scannable -> rc 2 (zero coverage is not 'clean')"
command grep -q 'oc_scan_anomaly "$(oc_anomaly_regex)"' "$HERE/run_packed_ab.sh" \
  && ok "the runner passes the regex first (its rc semantics are used)" || bad "the runner calls oc_scan_anomaly wrongly"

echo "── report: missing arms, and the fixtures"
python3 "$HERE/packed_ab_report.py" "$TMP/nothing" >/dev/null 2>&1; chk $? 1 "missing arm dirs -> fail, not crash"
python3 "$HERE/make_packed_ab_fixture.py" "$TMP/good" identical || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/good" > "$TMP/good.txt" 2>&1; chk $? 0 "identical arms -> pass"
command grep -q "implied C = 62" "$TMP/good.txt" && ok "C is solved from the arms (fixture is built for 62 us)" \
  || { bad "C not solved as expected"; command grep -i "C = " "$TMP/good.txt" | sed 's/^/     /'; }
command grep -q "model is retired" "$TMP/good.txt" && ok "the report says the unit-price model is retired" \
  || bad "the retired-model note is missing"
command grep -q "the model holds" "$TMP/good.txt" && bad "the report still endorses the retired model" \
  || ok "the report no longer compares C against 62.5 as if it were a target"
command grep -q "no measured window given" "$TMP/good.txt" \
  && ok "without --window-us it refuses to say whether a node eats the window" \
  || bad "it drew a form-C conclusion with no measured window"
python3 "$HERE/packed_ab_report.py" "$TMP/good" --window-us 40 > "$TMP/good_w40.txt" 2>&1
command grep -q "not worth implementing" "$TMP/good_w40.txt" \
  && ok "C above the measured window -> form C not worth implementing" || bad "window comparison did not fire"
python3 "$HERE/packed_ab_report.py" "$TMP/good" --window-us 4000 > "$TMP/good_w4k.txt" 2>&1
command grep -q "stays a candidate" "$TMP/good_w4k.txt" \
  && ok "C well below the window -> still a candidate (decided by t4, not here)" \
  || bad "the both-directions arm did not fire"

python3 "$HERE/make_packed_ab_fixture.py" "$TMP/onechar" onechar || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/onechar" > "$TMP/onechar.txt" 2>&1; chk $? 1 "one character changed -> FAIL"
command grep -q "answers differ" "$TMP/onechar.txt" && ok "it names the bit-identity failure" || bad "failure not named"

python3 "$HERE/make_packed_ab_fixture.py" "$TMP/empty" emptyfield || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/empty" > "$TMP/empty.txt" 2>&1; chk $? 1 "an empty compared field -> FAIL (no false green)"
command grep -q "comparison of nothing" "$TMP/empty.txt" && ok "it says the comparison was empty" || bad "empty comparison not named"

python3 "$HERE/make_packed_ab_fixture.py" "$TMP/slow" nogain || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/slow" > "$TMP/slow.txt" 2>&1
command grep -q "implied C" "$TMP/slow.txt" && ok "a no-gain arm still reports its implied C (no silent skip)" \
  || { bad "the no-gain fixture produced no C line"; command grep -i "C = " "$TMP/slow.txt" | sed 's/^/     /'; }
command grep -q "model is retired" "$TMP/slow.txt" && ok "and repeats that the model behind it is retired" \
  || bad "the retired-model note is missing on the no-gain arm"

python3 "$HERE/make_packed_ab_fixture.py" "$TMP/dead" deadlayers || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/dead" > "$TMP/dead.txt" 2>&1
command grep -q "with layers that never ran" "$TMP/dead.txt" && ok "dead-layer samples are reported" || bad "dead layers not reported"

echo
echo "selftest: $((PASS+FAIL)) checks, $FAIL failed; scratch $TMP"
[ "$FAIL" -eq 0 ] && { echo "SELFTEST: PASS"; exit 0; } || { echo "SELFTEST: FAIL"; exit 1; }
