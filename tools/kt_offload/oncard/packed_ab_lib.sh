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
pa_card_busy() {  # <card> -> rc 0 = busy (keep waiting), 1 = looks free
  local card="$1" out pct
  out=$(timeout 20 npu-smi info -t proc-mem -i "$card" 2>/dev/null) || return 0
  [ -n "$out" ] || return 0
  pa_proc_table_busy "$out" && return 0
  pct=$(timeout 20 npu-smi info -t usages -i "$card" 2>/dev/null \
        | awk -F: '/HBM Usage Rate/{gsub(/ /,"",$2); print $2; exit}')
  pa_hbm_busy "${pct:-}" && return 0
  return 1
}

pa_proc_table_busy() {  # <proc-mem text> -> rc 0 = a process is listed
  command grep -qE "Process id *: *[0-9]+|Process id:[0-9]+" <<< "${1:-}"
}

pa_hbm_busy() {  # <HBM usage percent> -> rc 0 = busy or unreadable
  local pct="${1:-}"
  [[ "$pct" =~ ^[0-9]+$ ]] || return 0          # unreadable -> assume busy
  [ "$pct" -ge "${PA_IDLE_HBM_PCT:-10}" ]
}
