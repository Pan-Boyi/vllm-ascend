#!/usr/bin/env bash
# Self-check for run_packed_ab.sh + packed_ab_report.py. Touches no card.
#   bash selftest_packed_ab.sh
# Every check names what would go wrong without it. The red arms matter most:
# a judge that cannot fail is not a judge.
set -u
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/packed_ab_selftest.XXXXXX")"
ROOT_ARTIFACTS=/workspace/code/dsv4_single_card/vllm_project/build_artifacts/moe_offload_ref
PASS=0; FAIL=0
# 🔴 ok() used to be `ok() { echo "[PASS] $1"; ... }` -- it took the MESSAGE and
# printed PASS unconditionally. Every `<predicate>; ok "..."` call site therefore
# threw its predicate away and every `ok $? "..."` printed "[PASS] 0". This is the
# same shape as selftest_cpu.sh:17, which takes the rc: ok <rc> <message>.
#   * judged form:   <predicate>; ok $? "message"      (rc decides)
#   * branched form: <predicate> && ok 0 "message" || bad "other message"
#                    (the && / || already decided; the 0 says so out loud)
# Nothing else is allowed -- `ok "message"` now counts as a FAIL with rc="message".
# ${2-...}: an old-shape `ok "text"` must be REPORTED as a failure, not abort the
# run under set -u before it can be counted.
ok()  { if [ "${1-}" = 0 ]; then echo "  [PASS] ${2-<no message>}"; PASS=$((PASS+1));
        else echo "  [FAIL] ${2-<no message>} (rc=${1-<no rc given>})"; FAIL=$((FAIL+1)); fi; }
bad() { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
chk() { if [ "$1" = "$2" ]; then ok 0 "$3 (rc=$1)"; else bad "$3 (rc=$1, want $2)"; fi; }

echo "── the judge itself (an ok() that cannot go red turns every check below into a no-op)"
# This is the meta-check the 2026-09-22 false green needed: exercise ok()/chk()
# in a subshell with their own counters and assert on the counts.
#
# 🔴 meta_must does NOT go through ok(). The first version of this block did, and
# it was green against the exact mutant it exists to catch: with the old
# unconditional ok(), the meta arms produced the wrong counts AND their verdicts
# were printed by that same ok() as [PASS]. A judge cannot be its own witness.
# It also exits on the spot: if the judge is broken, every line below is noise,
# and a "N checks, 0 failed" tail printed under a broken judge is the bug itself.
meta_must() {  # $1 got  $2 want  $3 what
  if [ "$1" = "$2" ]; then echo "  [PASS] $3 (got $1)"; PASS=$((PASS+1)); return 0; fi
  echo "  [FAIL] $3 (got $1, want $2)"; FAIL=$((FAIL+1))
  echo "  ⛔ the judge itself does not work -- every check below would be meaningless."
  echo "selftest: $((PASS+FAIL)) checks, $FAIL failed; scratch $TMP"; echo "SELFTEST: FAIL"; exit 1
}
meta=$( PASS=0; FAIL=0
        ok 0 "zero rc"      >/dev/null   # must land in PASS
        ok 1 "non-zero rc"  >/dev/null   # must land in FAIL
        ok "" "empty rc"    >/dev/null   # unreadable rc is not a pass either
        ok 'a message'      >/dev/null   # the OLD call shape must now go red
        echo "$PASS/$FAIL" )             # ^ single-quoted on purpose, see the grep below
meta_must "$meta" "1/3" "meta: ok() counts 1 pass / 3 fails for rc 0,1,'',<message>"
meta=$( PASS=0; FAIL=0; false; ok $? "forced" >/dev/null; echo "$PASS/$FAIL" )
meta_must "$meta" "0/1" "meta: a failing command reaching ok \$? is counted FAIL"
meta=$( PASS=0; FAIL=0; chk 0 0 "want" >/dev/null; chk 3 0 "want" >/dev/null; echo "$PASS/$FAIL" )
meta_must "$meta" "1/1" "meta: chk() has both arms -- matching rc passes, other rc fails"
# And the call sites themselves: on a non-comment line, an `ok` whose first
# argument opens a double quote with anything but a closing quote is the old
# shape (a message in the rc slot). `ok 0`, `ok $?` and the meta's own `ok ""`
# do not match. The quote is written ["] so that this line cannot match itself
# -- a grep that is its own only hit is green whatever the rest of the file says.
stale=$(command grep -cE '^[^#]*\bok ["][^"]' "$(readlink -f "${BASH_SOURCE[0]}")" || true)
meta_must "$stale" 0 "meta: no call site still passes a message where the rc goes"

echo "── syntax"
bash -n "$HERE/run_packed_ab.sh" && ok 0 "run_packed_ab.sh parses" || bad "run_packed_ab.sh does not parse"
bash -n "$HERE/packed_ab_lib.sh" && ok 0 "packed_ab_lib.sh parses" || bad "packed_ab_lib.sh does not parse"
python3 -m py_compile "$HERE/packed_ab_report.py" && ok 0 "packed_ab_report.py compiles" || bad "report does not compile"

echo "── usage and dry run"
bash "$HERE/run_packed_ab.sh" >/dev/null 2>&1; chk $? 2 "no card argument -> usage"
bash "$HERE/run_packed_ab.sh" x9 >/dev/null 2>&1; chk $? 2 "non-numeric card -> usage"
out=$(ONCARD_DRY=1 PACKED_AB_RUNS_DIR="$TMP/runs" bash "$HERE/run_packed_ab.sh" 3 2>&1); rc=$?
chk "$rc" 0 "ONCARD_DRY=1 stops before acquire"
echo "$out" | command grep -q "Stopping before acquire" && ok 0 "dry run says where it stopped" || bad "dry run message missing"
echo "$out" | command grep -q "VLLM_USE_BREAKABLE_CUDAGRAPH=1" && ok 0 "breakable value is printed" || bad "breakable value not printed"
ls "$TMP"/runs/*/code/run_packed_ab.sh >/dev/null 2>&1 && ok 0 "code snapshot taken" || bad "no code snapshot"

echo "── teardown predicates (the 'card not returned' check must be able to go red)"
source "$HERE/packed_ab_lib.sh"
held="card 3 owner dsv41_vllm_singlecard port 8041"
free="card 7 owner someone_else port 8010"
pa_lease_still_held "$held" 3 dsv41_vllm_singlecard && ok 0 "a held lease is detected" || bad "held lease missed"
pa_lease_still_held "$free" 3 dsv41_vllm_singlecard && bad "a free lease read as held" || ok 0 "a free lease reads as free"
pa_lease_still_held "$held" 3 someone_else && bad "another owner read as ours" || ok 0 "another owner is not ours"
pa_group_alive "" && bad "empty pgid read as alive" || ok 0 "empty pgid is not alive"
# our own *process group*, not $$: this script is not always its group
# leader (it is not when a tool launches it inside another group), and
# kill -0 on a pid that is no group id reads as dead
self_pgid=$(ps -o pgid= -p $$ | tr -d " ")
pa_group_alive "$self_pgid" && ok 0 "our own group reads as alive" || bad "our own group read as dead"
why=$(pa_card_returned "$held" 3 dsv41_vllm_singlecard 65000 ""); rc=$?
chk "$rc" 1 "card NOT returned when the lease is still held"
echo "$why" | command grep -q "lease still held" && ok 0 "it names why" || bad "it does not name why"
pa_card_returned "$free" 3 dsv41_vllm_singlecard 65000 "" >/dev/null; chk $? 0 "card returned when nothing holds it"

echo "── the busy-card reading (a wait that never waits looks like a wait that worked)"
source "$HERE/packed_ab_lib.sh"
IDLE_SAMPLE="$(cat "$ROOT_ARTIFACTS/oncard_runs/20260920_024812_t0/card_before.txt" 2>/dev/null)"
[ -n "$IDLE_SAMPLE" ] && ok 0 "a recorded idle npu-smi sample is available" || bad "no recorded idle sample to test against"
# The busy sample follows the format two independently written shared scripts
# parse (build_artifacts/wait_cards_free.sh:49, card_owners.sh:43) -- not a
# format invented here.
BUSY_SAMPLE="$(printf '\tNPU ID                         : 3\n\tProcess id                     : 1234567\n\tProcess memory(MB)             : 88123\n')"
pa_proc_table_busy "$IDLE_SAMPLE" && bad "the idle sample reads as busy" || ok 0 "idle sample -> free"
pa_proc_table_busy "$BUSY_SAMPLE" && ok 0 "busy sample -> busy" || bad "the busy sample reads as free"
echo "$BUSY_SAMPLE" | command grep -qE "process id|pid" \
  && bad "the old lower-case pattern would have matched (then it was not the bug)" \
  || ok 0 "the old lower-case pattern MISSES a busy card -> it had no discriminating power"
echo "$IDLE_SAMPLE" | command grep -q "No process in device" \
  && ok 0 "an idle card prints 'No process in device.' (neither pattern matches it either)" \
  || bad "the idle sample does not say what we think it says"
pa_hbm_busy 50 && ok 0 "HBM 50% -> busy" || bad "HBM 50% read as free"
pa_hbm_busy 4 && bad "HBM 4% read as busy" || ok 0 "HBM 4% (a real idle reading) -> free"
pa_hbm_busy "" && ok 0 "unreadable HBM -> treated as busy (fail-safe direction)" || bad "unreadable HBM read as free"
pa_hbm_busy "n/a" && ok 0 "garbage HBM -> treated as busy" || bad "garbage HBM read as free"
command grep -q 'grep -qE "process id|pid"' "$HERE/run_packed_ab.sh" \
  && bad "the lower-case wait is still in run_packed_ab.sh" || ok 0 "the lower-case wait is gone from the runner"
command grep -q "busy-reading positive control" "$HERE/run_packed_ab.sh" \
  && ok 0 "the runner checks its busy reading against a serving card" || bad "no positive control for the busy reading"
command grep -q "ROUNDS" "$HERE/run_packed_ab.sh" && ok 0 "arms are interleaved in rounds" || bad "no rounds"
for m in drift realgain; do
  python3 "$HERE/make_packed_ab_fixture.py" "$TMP/fl_$m" "$m" >/dev/null || bad "fixture $m failed"
  python3 "$HERE/packed_ab_report.py" "$TMP/fl_$m" > "$TMP/fl_$m.out" 2>&1
done
command grep -q "WITHIN the drift floor" "$TMP/fl_drift.out"; ok $? "a 0.5 ms delta under a 3 ms drift -> not distinguishable"
command grep -q "EXCEEDS the drift floor" "$TMP/fl_realgain.out"; ok $? "a 5 ms delta over a 0.3 ms drift -> a real difference"
# 🔴 this used to read "$TMP/good", a fixture that is not built until ~140 lines
# further down: the report was run against a directory that did not exist, the
# grep never matched, and the old ok() printed PASS anyway. Own fixture, built
# here, so the check no longer depends on where it sits in the file.
python3 "$HERE/make_packed_ab_fixture.py" "$TMP/oneround" identical >/dev/null || bad "fixture identical failed"
python3 "$HERE/packed_ab_report.py" "$TMP/oneround" > "$TMP/onerounds.out" 2>&1
command grep -q "drift floor: NOT MEASURED" "$TMP/onerounds.out"; ok $? "one round per arm -> the floor is reported as not measured"

echo "── card busy: three dimensions (HBM alone cannot tell 'running' from 'residue')"
# Recorded shapes, 2026-09-21: four cards at 92% HBM for 12 minutes while aicore
# moved 7..77% (a live deployment in another container) vs a teardown holding
# HBM with aicore flat at 0.
[ "$(pa_classify 1 "92,92,92" "7,0,77")" = running ]; ok $? "HBM high + aicore fluctuating -> running"
[ "$(pa_classify 1 "92,92,92" "0,0,0")" = residue ]; ok $? "HBM high + aicore flat zero -> residue"
[ "$(pa_classify 0 "4,4,4" "0,0,0")" = free ]; ok $? "HBM low -> free"
[ "$(pa_classify 1 "92,92,92" "0,?,0")" = running-unknown ]; ok $? "aicore unreadable -> treated as running (conservative)"
[ "$(pa_classify 1 "92,92" "0,0")" = running-unknown ]; ok $? "fewer than 3 samples -> not called idle"
[ "$(pa_classify 1 "4,4,4" "0,0,0")" = residue ]; ok $? "process table busy alone is still busy"
# the missing dimension, as executable evidence: the old HBM-only rule gives the
# SAME answer to the two states that need opposite actions
old_running=$(pa_hbm_busy 92 && echo busy || echo free)
old_residue=$(pa_hbm_busy 92 && echo busy || echo free)
[ "$old_running" = "$old_residue" ]; ok $? "HBM-only cannot separate them (both read '$old_running') -- that is the missing dimension"
[ "$(pa_classify 1 "92,92,92" "7,0,77")" != "$(pa_classify 1 "92,92,92" "0,0,0")" ]
ok $? "with aicore they are different verdicts"
pa_advice running | command grep -q "never kill, never reset"; ok $? "the 'running' advice forbids kill/reset"
pa_advice residue | command grep -q "drain"; ok $? "the 'residue' advice is to let it drain"
[ "$(pa_advice running)" != "$(pa_advice residue)" ]; ok $? "the two busy states advise different actions"
# The pre-flight must be EXERCISED, not grepped for: a name check stayed green while
# `CARD_STATE=$(pa_card_state ...)` lost PA_LAST_SAMPLES in the subshell and the
# first real t0 (2026-09-22) died under set -u before anything ran.
mkdir -p "$TMP/bin"
cat > "$TMP/bin/npu-smi" <<'FAKE'
#!/usr/bin/env bash
case "$*" in
  *proc-mem*) printf '\tNPU ID : 0\n\tNo process in device.\n';;
  *usages*)   printf '\tHBM Usage Rate(%%)              : 4\n\tAicore Usage Rate(%%)           : 0\n';;
esac
FAKE
chmod +x "$TMP/bin/npu-smi"
pre=$(command grep -E '^pa_sample_card "\$CARD"' "$HERE/run_oncard.sh")
[ -n "$pre" ] && ok 0 "run_oncard's pre-flight calls pa_sample_card directly (not in \$(...))" \
  || bad "run_oncard's pre-flight does not call pa_sample_card directly"
out=$(PATH="$TMP/bin:$PATH" bash -uc 'source "$1"; CARD=0; ONCARD_BUSY_SLEEP=0
  '"$pre"'
  CARD_STATE=$PA_LAST_STATE; echo "state=$CARD_STATE ($PA_LAST_SAMPLES)"' _ "$HERE/packed_ab_lib.sh" 2>&1); rc=$?
chk "$rc" 0 "the pre-flight line runs under set -u against a fake idle card"
echo "$out" | command grep -q "state=free (proc_busy=0 hbm=\[4,4,4\] aicore=\[0,0,0\])" \
  && ok 0 "it reads free and the samples reach the caller ($out)" || bad "unexpected pre-flight reading: $out"
# mutant: the old subshell form must die exactly the way the card run did
out=$(PATH="$TMP/bin:$PATH" bash -uc 'source "$1"; CARD_STATE=$(pa_card_state 0 3 0)
  echo "state=$CARD_STATE ($PA_LAST_SAMPLES)"' _ "$HERE/packed_ab_lib.sh" 2>&1); rc=$?
[ "$rc" -ne 0 ] && echo "$out" | command grep -q "PA_LAST_SAMPLES: unbound variable" \
  && ok 0 "mutant: the old \$(pa_card_state) form reproduces the on-card failure" \
  || bad "mutant did not reproduce 'PA_LAST_SAMPLES: unbound variable' (rc=$rc: $out)"
PATH="$TMP/bin:$PATH" pa_card_busy 0 && bad "pa_card_busy reads the fake idle card as busy" \
  || { [ -n "${PA_LAST_SAMPLES:-}" ] && ok 0 "pa_card_busy: free, and PA_LAST_SAMPLES survives for its caller" \
       || bad "pa_card_busy left PA_LAST_SAMPLES empty"; }
command grep -q "pa_advice" "$HERE/run_packed_ab.sh"; ok $? "run_packed_ab reports which kind of busy it waited on"
command grep -q "lease table is a convention INSIDE this container" "$HERE/run_oncard.sh"
ok $? "the gate says a lease cannot answer 'is somebody else on this card'"

echo "── the fingerprint must watch the package that is loaded, not the shared baseline"
python3 "$HERE/../build_fingerprint.py" capture > "$TMP/fp_now.json" 2>/dev/null
python3 "$HERE/fp_object_check.py" "$TMP/fp_now.json" "$HERE/../build_fingerprint.py" > "$TMP/fp_obj.out" 2>&1; r=$?
# the tail is hoisted out of the message: a $(...) in the same argument list as
# `ok $?` is one expansion order away from overwriting the very rc being judged
last=$(tail -n1 "$TMP/fp_obj.out")
[ $r -eq 0 ]; ok $? "fingerprint watches the tree's own package ($last)"
python3 "$HERE/fp_empty_check.py" "$HERE/../build_fingerprint.py" > "$TMP/fp_empty.out" 2>&1; r=$?
last=$(tail -n1 "$TMP/fp_empty.out")
[ $r -eq 0 ]; ok $? "an unread package is refused, not hashed to the empty constant ($last)"
command grep -q "EMPTY_SHA256" "$HERE/../build_fingerprint.py"; ok $? "the empty-hash sentinel is named in the tool"

echo "── A/A control: MODE=aa and every branch of its decision table"
out=$(MODE=aa ONCARD_DRY=1 PACKED_AB_RUNS_DIR="$TMP/aaruns" bash "$HERE/run_packed_ab.sh" 3 2>&1); rc=$?
chk "$rc" 0 "MODE=aa dry run stops before acquire"
echo "$out" | command grep -q "packed AA" && ok 0 "it says which mode it is" || bad "mode not announced"
echo "$out" | command grep -q "evals=c8,c8b,c1" && ok 0 "all three evals are planned by default" || bad "eval set not announced"
echo "$out" | command grep -q "one function" && ok 0 "it states the harness cannot drift between modes" || bad "harness note missing"
command grep -q 'AA_EVALS' "$HERE/run_packed_ab.sh" && ok 0 "the eval list is one variable" || bad "eval list not centralised"
# 🔴 the pattern used to be the bare name, so it counted every MENTION -- including
# the `[ -f .../gsm8k_eval_effort.py ] || pf "gsm8k harness missing"` pre-flight,
# which is not a launch site. It has read 2 since the check was written (6b7e84e1d,
# verified against that commit's own run_packed_ab.sh): it never had a green state,
# the old ok() just never asked. A launch site is a line that RUNS the harness.
n_eval_calls=$(command grep -cE 'python3 .*gsm8k_eval_effort\.py' "$HERE/run_packed_ab.sh")
[ "$n_eval_calls" -eq 1 ]; ok $? "exactly one GSM8K launch site in the runner (found $n_eval_calls) -- parameters cannot drift"
for m in batchcomp bothdiffer bothsame; do
  python3 "$HERE/make_aa_fixture.py" "$TMP/aa_$m" "$m" >/dev/null || bad "aa fixture $m failed"
  python3 "$HERE/aa_report.py" "$TMP/aa_$m" > "$TMP/aa_$m.out" 2>&1; r=$?
  [ $r -eq 0 ]; ok $? "aa verdict for $m decides (rc $r)"
done
command grep -q "BATCH COMPOSITION" "$TMP/aa_batchcomp.out"; ok $? "c8 differs + c1 identical -> batch composition"
command grep -q "NO discriminating power" "$TMP/aa_batchcomp.out"; ok $? "and it says the service criterion is blunt"
command grep -q "NOT batch composition" "$TMP/aa_bothdiffer.out"; ok $? "both differ -> another source"
command grep -q "may NOT be attributed to packed" "$TMP/aa_bothdiffer.out"; ok $? "and it refuses to blame packed"
command grep -q "packed changing the numbers" "$TMP/aa_bothsame.out"; ok $? "both identical -> packed changed the numbers"
for m in inverted missing_c1 emptyfield otherquestions; do
  python3 "$HERE/make_aa_fixture.py" "$TMP/aa_$m" "$m" >/dev/null || bad "aa fixture $m failed"
  python3 "$HERE/aa_report.py" "$TMP/aa_$m" > "$TMP/aa_$m.out" 2>&1; r=$?
  [ $r -eq 1 ]; ok $? "aa refuses to conclude on '$m' (rc $r)"
done
command grep -q "opposite of every hypothesis" "$TMP/aa_inverted.out"; ok $? "the inverted reading is named, not averaged away"
command grep -q "cannot decide" "$TMP/aa_missing_c1.out"; ok $? "a missing concurrency-1 arm blocks the verdict"
python3 "$HERE/make_aa_fixture.py" "$TMP/aa_gate" bothsame >/dev/null
python3 "$HERE/../build_fingerprint.py" backfill "$TMP/aa_gate/armA2" --so-md5 ffffffffffffffffffffffffffffffff >/dev/null
rm -f "$TMP/aa_gate/armA2/build_fingerprint.json"
python3 "$HERE/aa_report.py" "$TMP/aa_gate" > "$TMP/aa_gate.out" 2>&1; r=$?
{ [ $r -eq 1 ] && command grep -q "REFUSING" "$TMP/aa_gate.out"; }
ok $? "an A/A across two builds is refused too (rc $r)"

echo "── build fingerprint gate (cross-build comparison must be refused)"
python3 "$HERE/make_packed_ab_fixture.py" "$TMP/fpsame" identical >/dev/null || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/fpsame" > "$TMP/fpsame.out" 2>&1; r=$?
# 🔴 this used to demand rc 0 as well. The `identical` fixture is one round per
# arm, so since 8573fb9d1 the report legitimately exits 1 on the MISSING DRIFT
# FLOOR -- a different criterion, which this file asserts separately at
# "$TMP/good". Tying the build gate to the overall rc made it a check of the
# wrong thing, and it has been red-in-principle ever since; the old ok() hid it.
# What belongs here is only: did the BUILD gate let the arms be compared?
command grep -q "same build: the arms may be compared" "$TMP/fpsame.out"
ok $? "same build -> the build gate lets the arms be compared (report rc $r, decided elsewhere)"
! command grep -q "NO DIFFERENCE WILL BE COMPUTED" "$TMP/fpsame.out"
ok $? "and nothing is suppressed on the same-build side"
command grep -qE "A-B = [+-][0-9]" "$TMP/fpsame.out"; ok $? "and a difference is actually reported, as a number"
python3 "$HERE/make_packed_ab_fixture.py" "$TMP/fpdiff" identical >/dev/null || bad "fixture builder failed"
python3 "$HERE/../build_fingerprint.py" backfill "$TMP/fpdiff/armB" --so-md5 ffffffffffffffffffffffffffffffff >/dev/null
rm -f "$TMP/fpdiff/armB/build_fingerprint.json"
python3 "$HERE/packed_ab_report.py" "$TMP/fpdiff" > "$TMP/fpdiff.out" 2>&1; r=$?
[ $r -eq 1 ]; ok $? "different .so -> refused (rc $r)"
# 🔴 the string was "REFUSING to report differences". 9d0a88b5a (2026-09-21)
# reworded it to "NO DIFFERENCE WILL BE COMPUTED (<n> blocking reason(s))" in the
# same change that narrowed the refusal: refuse every subtraction, still run the
# criteria that do not subtract. The grep went stale that day and stayed green.
command grep -q "NO DIFFERENCE WILL BE COMPUTED" "$TMP/fpdiff.out"; ok $? "it says it is refusing, not warning"
command grep -q "ffffffffffff" "$TMP/fpdiff.out"; ok $? "it prints both fingerprints"
# 🔴 was `! grep -qi "delta"`. After the same commit the report still PRINTS the
# delta lines with "suppressed" where the number would be, so the word is there
# by design. The rule is "no NUMBER survives", and the line must still be printed
# -- a silently dropped line is how a suppressed difference becomes invisible.
! command grep -qE "A-B = [+-][0-9]" "$TMP/fpdiff.out"; ok $? "and no numeric difference was printed anyway"
command grep -q "paired A-B deltas: suppressed" "$TMP/fpdiff.out"
ok $? "the delta line is still printed, marked suppressed (not dropped)"
python3 "$HERE/make_packed_ab_fixture.py" "$TMP/fpnone" identical --no-fingerprint >/dev/null || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/fpnone" > "$TMP/fpnone.out" 2>&1; r=$?
{ [ $r -eq 1 ] && command grep -q "no build fingerprint" "$TMP/fpnone.out"; }
ok $? "no fingerprint at all -> refused, not assumed equal (rc $r)"

echo "── anomaly scan positive control"
printf 'all good\naicore exception happened\n' > "$TMP/anom.log"
SCAN="source '$HERE/oncard_lib.sh'; oc_scan_anomaly \"\$(oc_anomaly_regex)\""
ONCARD_DIR="$HERE" bash -c "$SCAN '$TMP/anom.log'" | command grep -qi "aicore" \
  && ok 0 "anomaly marker is detected" || bad "anomaly marker missed (the scanner cannot see a real one)"
printf 'all good\nnothing to see\n' > "$TMP/clean.log"
out=$(ONCARD_DIR="$HERE" bash -c "$SCAN '$TMP/clean.log'"); rc=$?
[ -z "$out" ] && [ "$rc" = 1 ] && ok 0 "a clean log reports nothing (rc=1)" || bad "a clean log gave rc=$rc out='$out'"
ONCARD_DIR="$HERE" bash -c "$SCAN '$TMP/does_not_exist.log'" >/dev/null 2>&1
chk $? 2 "nothing scannable -> rc 2 (zero coverage is not 'clean')"
command grep -q 'oc_scan_anomaly "$(oc_anomaly_regex)"' "$HERE/run_packed_ab.sh" \
  && ok 0 "the runner passes the regex first (its rc semantics are used)" || bad "the runner calls oc_scan_anomaly wrongly"

echo "── report: missing arms, and the fixtures"
python3 "$HERE/packed_ab_report.py" "$TMP/nothing" >/dev/null 2>&1; chk $? 1 "missing arm dirs -> fail, not crash"
python3 "$HERE/make_packed_ab_fixture.py" "$TMP/good" identical || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/good" > "$TMP/good.txt" 2>&1; r=$?
[ $r -eq 1 ]; ok $? "identical answers but ONE round and a 5 ms delta -> refused for lack of a drift floor (rc $r)"
command grep -q "no drift floor" "$TMP/good.txt"; ok $? "and it names the missing yardstick"
python3 "$HERE/make_packed_ab_fixture.py" "$TMP/twor" realgain >/dev/null || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/twor" > "$TMP/twor.txt" 2>&1; chk $? 0 "identical answers + two rounds + delta over the floor -> pass"
command grep -q "implied C = 62" "$TMP/good.txt" && ok 0 "C is solved from the arms (fixture is built for 62 us)" \
  || { bad "C not solved as expected"; command grep -i "C = " "$TMP/good.txt" | sed 's/^/     /'; }
command grep -q "model is retired" "$TMP/good.txt" && ok 0 "the report says the unit-price model is retired" \
  || bad "the retired-model note is missing"
command grep -q "the model holds" "$TMP/good.txt" && bad "the report still endorses the retired model" \
  || ok 0 "the report no longer compares C against 62.5 as if it were a target"
command grep -q "no measured window given" "$TMP/good.txt" \
  && ok 0 "without --window-us it refuses to say whether a node eats the window" \
  || bad "it drew a form-C conclusion with no measured window"
python3 "$HERE/packed_ab_report.py" "$TMP/good" --window-us 40 > "$TMP/good_w40.txt" 2>&1
command grep -q "not worth implementing" "$TMP/good_w40.txt" \
  && ok 0 "C above the measured window -> form C not worth implementing" || bad "window comparison did not fire"
python3 "$HERE/packed_ab_report.py" "$TMP/good" --window-us 4000 > "$TMP/good_w4k.txt" 2>&1
command grep -q "stays a candidate" "$TMP/good_w4k.txt" \
  && ok 0 "C well below the window -> still a candidate (decided by t4, not here)" \
  || bad "the both-directions arm did not fire"

python3 "$HERE/make_packed_ab_fixture.py" "$TMP/onechar" onechar || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/onechar" > "$TMP/onechar.txt" 2>&1; chk $? 1 "one character changed -> FAIL"
command grep -q "answers differ" "$TMP/onechar.txt" && ok 0 "it names the bit-identity failure" || bad "failure not named"

python3 "$HERE/make_packed_ab_fixture.py" "$TMP/empty" emptyfield || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/empty" > "$TMP/empty.txt" 2>&1; chk $? 1 "an empty compared field -> FAIL (no false green)"
command grep -q "comparison of nothing" "$TMP/empty.txt" && ok 0 "it says the comparison was empty" || bad "empty comparison not named"

python3 "$HERE/make_packed_ab_fixture.py" "$TMP/slow" nogain || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/slow" > "$TMP/slow.txt" 2>&1
command grep -q "implied C" "$TMP/slow.txt" && ok 0 "a no-gain arm still reports its implied C (no silent skip)" \
  || { bad "the no-gain fixture produced no C line"; command grep -i "C = " "$TMP/slow.txt" | sed 's/^/     /'; }
command grep -q "model is retired" "$TMP/slow.txt" && ok 0 "and repeats that the model behind it is retired" \
  || bad "the retired-model note is missing on the no-gain arm"

python3 "$HERE/make_packed_ab_fixture.py" "$TMP/dead" deadlayers || bad "fixture builder failed"
python3 "$HERE/packed_ab_report.py" "$TMP/dead" > "$TMP/dead.txt" 2>&1
command grep -q "with layers that never ran" "$TMP/dead.txt" && ok 0 "dead-layer samples are reported" || bad "dead layers not reported"

echo
echo "selftest: $((PASS+FAIL)) checks, $FAIL failed; scratch $TMP"
[ "$FAIL" -eq 0 ] && { echo "SELFTEST: PASS"; exit 0; } || { echo "SELFTEST: FAIL"; exit 1; }
