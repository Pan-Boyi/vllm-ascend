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
