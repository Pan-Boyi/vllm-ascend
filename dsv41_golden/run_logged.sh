#!/usr/bin/env bash
# Correctly capturing an exit code, as the default path instead of a rule to remember.
#
# Four people on this project know "no head/tail in a diagnostic pipeline" and all
# four broke it in one day.  A rule that everyone knows and everyone breaks is not a
# discipline problem, so this makes the right thing the easy thing.
#
# What goes wrong without it:
#
#     mycmd | tail -8 ; echo "rc=$?"      # rc is tail's, always 0
#     mycmd | head -40                    # SIGPIPE truncates the output you needed
#
# Both fail in the same direction: they report success and show you less than
# happened.  A failing command reads as a passing one.
#
# Usage:
#     source run_logged.sh
#     run pytest -x tests/            # rc is the command's; full output on disk
#     run --lines 20 ./long_thing.sh  # show more of the tail after rc is recorded
#     echo $RUN_RC   $RUN_LOG         # also left in variables for scripting
#
#     bash run_logged.sh --self-test  # proves rc propagates and nothing truncates

RUN_LOG_DIR="${RUN_LOG_DIR:-${TMPDIR:-/tmp}/run_logged}"

run() {
    local lines=8
    while [ "$1" = "--lines" ]; do lines="$2"; shift 2; done
    if [ $# -eq 0 ]; then
        echo "run: no command given" >&2
        return 2
    fi

    mkdir -p "$RUN_LOG_DIR" || return 2
    # mktemp, not a name built from time+pid: two runs in the same second from the
    # same shell collided and the second overwrote the first, which quietly undoes
    # the "full output is on disk" guarantee this helper exists for.
    local tag; tag=$(basename -- "$1" | tr -c 'A-Za-z0-9_.-' '_')
    RUN_LOG=$(mktemp "$RUN_LOG_DIR/$(date +%H%M%S)_${tag}_XXXXXX.log") || return 2

    # The whole point: a plain redirect, never a pipe.  $? is then the command's own
    # status, and the file holds every byte -- no SIGPIPE, no truncation.
    "$@" > "$RUN_LOG" 2>&1
    RUN_RC=$?

    local n; n=$(wc -l < "$RUN_LOG")
    echo "--- rc=$RUN_RC  ${n} lines  log=$RUN_LOG"
    # The excerpt is printed only AFTER the status is recorded, so looking at it
    # cannot change what gets reported.  This is the part people reach for a pipe for.
    if [ "$n" -gt "$lines" ]; then
        echo "--- last $lines of $n lines (full output is in the log):"
    fi
    tail -n "$lines" "$RUN_LOG"
    return $RUN_RC
}

self_test() {
    local fails=0
    echo "== 1. a failing command must report its own status, not the pipeline's"
    ( set +e; run sh -c 'echo out; echo err >&2; exit 7' >/dev/null 2>&1; echo "$?" ) > /tmp/.rt1
    local rc; rc=$(cat /tmp/.rt1)
    if [ "$rc" = "7" ]; then echo "   ok   run -> rc=7"; else echo "   FAIL run -> rc=$rc, expected 7"; fails=1; fi
    # the same command through the pipeline this helper exists to replace
    local bad; bad=$( sh -c 'echo out; exit 7' | tail -n 2 >/dev/null 2>&1; echo $? )
    if [ "$bad" = "0" ]; then
        echo "   ok   demonstrated the failure mode: '| tail' reports rc=0 for the same exit 7"
    else
        echo "   note '| tail' reported rc=$bad on this shell"
    fi

    echo "== 2. a succeeding command must report 0"
    ( set +e; run true >/dev/null 2>&1; echo "$?" ) > /tmp/.rt2
    rc=$(cat /tmp/.rt2)
    if [ "$rc" = "0" ]; then echo "   ok   run true -> rc=0"; else echo "   FAIL rc=$rc"; fails=1; fi

    echo "== 3. long output must be complete on disk, not truncated to the excerpt"
    # NOT in a subshell: RUN_LOG must survive into this scope.  Running it in one
    # is what the first version of this test did, and it failed for that reason
    # rather than for anything wrong with run().
    run sh -c 'i=0; while [ $i -lt 5000 ]; do echo "line $i"; i=$((i+1)); done' >/dev/null 2>&1
    local kept; kept=$(wc -l < "$RUN_LOG")
    if [ "$kept" = "5000" ]; then echo "   ok   5000 lines kept"; else echo "   FAIL kept $kept of 5000"; fails=1; fi

    echo "== 4. stderr must be captured too"
    run sh -c 'echo only-on-stderr >&2' >/dev/null 2>&1
    if grep -q only-on-stderr "$RUN_LOG"; then echo "   ok   stderr in the log"; else echo "   FAIL stderr lost"; fails=1; fi

    echo "== 5. no command must fail safe, not silently pass"
    ( set +e; run >/dev/null 2>&1; echo "$?" ) > /tmp/.rt5
    rc=$(cat /tmp/.rt5)
    if [ "$rc" = "2" ]; then echo "   ok   empty invocation -> rc=2"; else echo "   FAIL rc=$rc"; fails=1; fi

    echo "== 6. two runs in the same second must not share a log file"
    run true >/dev/null 2>&1; local l1="$RUN_LOG"
    run true >/dev/null 2>&1; local l2="$RUN_LOG"
    if [ "$l1" != "$l2" ] && [ -f "$l1" ] && [ -f "$l2" ]; then
        echo "   ok   distinct logs kept"
    else
        echo "   FAIL same log reused: $l1"; fails=1
    fi

    rm -f /tmp/.rt1 /tmp/.rt2 /tmp/.rt5
    echo
    if [ "$fails" = "0" ]; then echo "self-test PASSED"; else echo "self-test FAILED"; fi
    return $fails
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi
