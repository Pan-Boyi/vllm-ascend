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

# Waiting for a background job.  `kill -0 $pid` is NOT a liveness test: a zombie
# satisfies it, so `until ! kill -0 $pid; do sleep; done` waits forever on a job that
# finished.  That happened three times in one session -- the jobs had long completed,
# the loops were still spinning, and the harness eventually killed them citing memory,
# which hid the real cause.  (`nohup cmd &` from a one-shot shell leaves the child
# unreaped, so the zombie persists.)
#
# So do not consult the process table at all: wait for the job's OWN completion
# marker.  run_bg writes one when the command exits, whatever the exit code.
#
#     run_bg python long_thing.py --flag      # prints its log path and sentinel
#     wait_bg                                 # blocks until it is really done
#     echo $RUN_RC                             # the command's status
run_bg() {
    mkdir -p "$RUN_LOG_DIR" || return 2
    local tag; tag=$(basename -- "${1:-job}" | tr -c 'A-Za-z0-9_.-' '_')
    RUN_LOG=$(mktemp "$RUN_LOG_DIR/$(date +%H%M%S)_${tag}_XXXXXX.log") || return 2
    RUN_DONE="$RUN_LOG.done"
    rm -f "$RUN_DONE"
    # the sentinel carries the exit code, so "finished" and "succeeded" stay separate
    # `sh -c script a b c` binds a to $0, so the command must be shifted past the two
    # paths explicitly -- getting this wrong yields rc=127 and an empty log.
    nohup sh -c 'log=$1; done=$2; shift 2; "$@" > "$log" 2>&1; echo $? > "$done"' sh "$RUN_LOG" "$RUN_DONE" "$@" > /dev/null 2>&1 &
    echo "--- started  log=$RUN_LOG  sentinel=$RUN_DONE"
}

wait_bg() {
    local timeout="${1:-3600}" waited=0
    while [ ! -f "$RUN_DONE" ]; do
        sleep 5
        waited=$((waited + 5))
        if [ "$waited" -ge "$timeout" ]; then
            echo "--- wait_bg TIMED OUT after ${timeout}s; log=$RUN_LOG" >&2
            return 124
        fi
    done
    RUN_RC=$(cat "$RUN_DONE")
    local n; n=$(wc -l < "$RUN_LOG")
    echo "--- rc=$RUN_RC  ${n} lines  log=$RUN_LOG  (waited ${waited}s)"
    return "$RUN_RC"
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

    echo "== 6b. a finished background job must be detected as finished"
    run_bg sh -c 'echo bg-out; exit 5' >/dev/null 2>&1
    ( set +e; wait_bg 60 >/dev/null 2>&1; echo "$?" ) > /tmp/.rt6
    local brc; brc=$(cat /tmp/.rt6)
    if [ "$brc" = "5" ]; then echo "   ok   wait_bg returned the job's rc=5"; else echo "   FAIL wait_bg -> $brc, expected 5"; fails=1; fi
    if grep -q bg-out "$RUN_LOG"; then echo "   ok   background output captured"; else echo "   FAIL output lost"; fails=1; fi
    # the point of the sentinel: show that the process-table test would have hung
    local zpid; zpid=$(cat "$RUN_DONE" >/dev/null 2>&1; ps -eo pid,stat | awk '$2 ~ /^Z/ {print $1; exit}')
    if [ -n "$zpid" ] && kill -0 "$zpid" 2>/dev/null; then
        echo "   ok   demonstrated the trap: kill -0 succeeds on zombie pid $zpid"
    else
        echo "   note no zombie present right now to demonstrate against"
    fi
    rm -f /tmp/.rt6

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
