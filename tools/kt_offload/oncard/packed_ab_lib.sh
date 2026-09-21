#!/usr/bin/env bash
# Teardown predicates for run_packed_ab.sh, split out so the self-test can make
# each one go red without a card. Each answers exactly one question and takes
# its evidence as an argument, so a self-test can supply the evidence.

# Is the lease table still showing this card held by this owner?
# Input is the status TEXT, not a command: the test has to be able to hand it
# the case that must be refused.
pa_lease_still_held() {  # <status text> <card> <owner>  -> rc 0 = still held
  local text="$1" card="$2" owner="$3"
  printf '%s\n' "$text" | command grep -E "(^|[^0-9])${card}([^0-9]|$)" | command grep -q -- "$owner"
}

# Is anything still listening on the port?
pa_port_busy() {  # <port> -> rc 0 = busy
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# Is the process group we started still alive? (our own pgid, never a name match)
pa_group_alive() {  # <pgid> -> rc 0 = alive
  [ -n "${1:-}" ] && kill -0 "-$1" 2>/dev/null
}

# All three must hold before the card counts as returned.
pa_card_returned() {  # <status text> <card> <owner> <port> <pgid>
  local why=""
  pa_group_alive "$5" && why="$why process group $5 alive;"
  pa_port_busy "$4" && why="$why port $4 busy;"
  pa_lease_still_held "$1" "$2" "$3" && why="$why lease still held;"
  [ -z "$why" ] && return 0
  echo "$why"
  return 1
}

# Is the card still in use? Two independent readings; either one says "busy"
# and we keep waiting.
#
# Why not one: the process-table reading is what this script had, spelled
# `grep -qE "process id|pid"` in lower case. npu-smi prints `Process id` with a
# capital P (see build_artifacts/wait_cards_free.sh:49 and card_owners.sh:43,
# written independently of this file), and an idle card prints
# `No process in device.` -- which matches neither pattern either. So the old
# loop broke on its first iteration in both states: it waited on a busy card
# for zero seconds and read as if the card were free. A wait that never waits
# looks exactly like a wait that succeeded.
#
# Unreadable output counts as busy: the fail-safe direction for "release the
# card to the next arm" is to wait, not to proceed.
pa_aicore_of() {  # <usages text> -> the Aicore Usage Rate field, or "" if absent
  awk -F: '/Aicore Usage Rate/{gsub(/ /,"",$2); print $2; exit}' <<< "${1:-}"
}

pa_hbm_of() {  # <usages text> -> the HBM Usage Rate field, or ""
  awk -F: '/HBM Usage Rate/{gsub(/ /,"",$2); print $2; exit}' <<< "${1:-}"
}

# Classify a card from samples. Pure: the inputs are strings, so the self-test
# can feed recorded readings without a card.
#
# Why aicore is a separate dimension: HBM alone cannot tell "someone is running
# on this card" from "a teardown has not finished" -- the two read identically
# (2026-09-21: four cards at 92% HBM, 114861 MB each, unchanged for 12 minutes,
# while aicore moved between 7% and 77% -- a live deployment in ANOTHER
# CONTAINER). The two call for opposite actions: wait/coordinate versus wait for
# it to drain. A single zero sample proves nothing either (it can land in a gap
# between kernels), hence >= 3 samples.
#
# echoes one of: free | running | residue | running-unknown
pa_classify() {  # <proc_busy 0|1> <hbm csv> <aicore csv>
  local proc_busy="${1:-0}" hbm_list="${2:-}" ai_list="${3:-}" v hbm_busy=0 any_ai=0 unknown=0 n=0
  for v in ${hbm_list//,/ }; do
    pa_hbm_busy "$v" && hbm_busy=1
  done
  for v in ${ai_list//,/ }; do
    n=$((n + 1))
    if [[ "$v" =~ ^[0-9]+$ ]]; then
      [ "$v" -gt "${PA_AICORE_IDLE_PCT:-0}" ] && any_ai=1
    else
      unknown=1     # unreadable: the conservative reading is "someone is running"
    fi
  done
  [ "$n" -ge "${PA_AICORE_SAMPLES:-3}" ] || unknown=1   # too few samples to call it idle
  if [ "$proc_busy" != 1 ] && [ "$hbm_busy" != 1 ]; then
    echo free; return 0
  fi
  if [ "$any_ai" = 1 ]; then echo running; return 0; fi
  if [ "$unknown" = 1 ]; then echo running-unknown; return 0; fi
  echo residue
}

pa_advice() {  # <state> -> what to do about it (the two busy states differ)
  case "${1:-}" in
    running) echo "someone is RUNNING on this card (aicore non-zero). Wait or coordinate -- never kill, never reset. The pids npu-smi lists may belong to another container and not exist in ours.";;
    running-unknown) echo "aicore could not be read; treated as RUNNING (the conservative side). Wait or coordinate.";;
    residue) echo "teardown RESIDUE (HBM held, aicore flat zero). Let it drain -- 113 GiB x 4 has taken over 20 minutes.";;
    free) echo "card looks free (process table empty, HBM below the idle threshold).";;
    *) echo "unclassified";;
  esac
}

# Sample a real card and classify it. Prints the state; the samples land in
# PA_LAST_SAMPLES for the caller to log.
pa_card_state() {  # <card> [samples] [sleep_s]
  local card="$1" n="${2:-${PA_AICORE_SAMPLES:-3}}" nap="${3:-2}" i out hbm ai proc_busy=0
  local hbm_list="" ai_list=""
  out=$(timeout 20 npu-smi info -t proc-mem -i "$card" 2>/dev/null) || out=""
  if [ -z "$out" ]; then
    PA_LAST_SAMPLES="proc-mem unreadable"
    echo running-unknown; return 0
  fi
  pa_proc_table_busy "$out" && proc_busy=1
  for ((i = 0; i < n; i++)); do
    local u; u=$(timeout 20 npu-smi info -t usages -i "$card" 2>/dev/null) || u=""
    hbm=$(pa_hbm_of "$u"); ai=$(pa_aicore_of "$u")
    hbm_list="${hbm_list:+$hbm_list,}${hbm:-?}"
    ai_list="${ai_list:+$ai_list,}${ai:-?}"
    [ $((i + 1)) -lt "$n" ] && sleep "$nap"
  done
  PA_LAST_SAMPLES="proc_busy=$proc_busy hbm=[$hbm_list] aicore=[$ai_list]"
  pa_classify "$proc_busy" "$hbm_list" "$ai_list"
}

pa_card_busy() {  # <card> -> rc 0 = busy (keep waiting), 1 = looks free
  local state
  state=$(pa_card_state "$1" "${PA_BUSY_SAMPLES:-3}" "${PA_BUSY_SLEEP:-2}")
  PA_LAST_STATE="$state"
  [ "$state" = free ] && return 1
  return 0
}

pa_proc_table_busy() {  # <proc-mem text> -> rc 0 = a process is listed
  command grep -qE "Process id *: *[0-9]+|Process id:[0-9]+" <<< "${1:-}"
}

pa_hbm_busy() {  # <HBM usage percent> -> rc 0 = busy or unreadable
  local pct="${1:-}"
  [[ "$pct" =~ ^[0-9]+$ ]] || return 0          # unreadable -> assume busy
  [ "$pct" -ge "${PA_IDLE_HBM_PCT:-10}" ]
}
