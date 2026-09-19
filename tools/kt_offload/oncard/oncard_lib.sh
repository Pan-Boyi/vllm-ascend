# shellcheck shell=bash
# Helpers shared by run_oncard.sh and selftest_cpu.sh. Sourced, never executed.
# Nothing here touches a device.

ONCARD_DIR="${ONCARD_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Same liveness rule as build_artifacts/card_lease.sh _alive: a zombie is dead.
oc_alive() { [ -n "${1:-}" ] && [ -d "/proc/$1" ] && ! command grep -qE '^State:[[:space:]]+Z' "/proc/$1/status" 2>/dev/null; }

oc_ms() { date +%s%3N; }

# Anomaly regex: one source of truth (oncard_common.ANOMALY_PATTERNS).
oc_anomaly_regex() { python3 "$ONCARD_DIR/oncard_common.py" --anomaly-regex; }

# oc_scan_anomaly <regex> <file|dir>...  -> prints "path:line" of the first hit, rc 0 on a hit,
# 1 on none, 2 when nothing was scannable (zero coverage is not "clean").
oc_scan_anomaly() {
  local re="$1"; shift
  local f n=0 hit
  for f in "$@"; do
    [ -e "$f" ] || continue
    n=$((n+1))
    if [ -d "$f" ]; then
      hit=$(command grep -rEn -m1 "$re" "$f" 2>/dev/null | head -n 1)
    else
      hit=$(command grep -En -m1 "$re" "$f" 2>/dev/null | head -n 1)
    fi
    if [ -n "$hit" ]; then echo "$f: $hit"; return 0; fi
  done
  [ "$n" -eq 0 ] && return 2
  return 1
}

# oc_count <regex> <file|dir>...  -> total matching lines (never piped into a rc check)
oc_count() {
  local re="$1"; shift
  local f total=0 c
  for f in "$@"; do
    [ -e "$f" ] || continue
    if [ -d "$f" ]; then c=$(command grep -rEc "$re" "$f" 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
    else c=$(command grep -Ec "$re" "$f" 2>/dev/null); c=${c:-0}; fi
    total=$((total + c))
  done
  echo "$total"
}

# oc_dmesg_new <before> <after> <out>: lines of <after> not in <before> (by content, not position).
oc_dmesg_new() { command grep -vxF -f "$1" "$2" > "$3" 2>/dev/null; [ -s "$1" ] || cp "$2" "$3"; return 0; }

# oc_dmesg_card <card> <in>: new lines attributed to this card (udevid=<card>, the rule of
# build_artifacts/card_dmesg_gate.py card_pattern), plus a count of new error lines that
# name no card at all (not ours to attribute, never silently dropped).
oc_dmesg_card() {
  local card="$1" in="$2"
  local mine other unattr
  mine=$(command grep -cE "udevid=${card}[;) ]|udevid=${card}\$" "$in" 2>/dev/null); mine=${mine:-0}
  other=$(command grep -E "udevid=[0-9]+" "$in" 2>/dev/null | command grep -cvE "udevid=${card}[;) ]|udevid=${card}\$"); other=${other:-0}
  unattr=$(command grep -iE "error|fail|exception" "$in" 2>/dev/null | command grep -cv "udevid="); unattr=${unattr:-0}
  echo "card=$card new_lines_this_card=$mine new_lines_other_cards=$other new_error_lines_without_udevid=$unattr"
  command grep -E "udevid=${card}[;) ]|udevid=${card}\$" "$in" 2>/dev/null | head -n 50
}

oc_meminfo_avail_kb() { awk '/^MemAvailable:/{print $2}' /proc/meminfo; }

# oc_watchdog <main_pid> <test_pid_file> <release_cmd...>
# Run in the background. Returns as soon as <main_pid> is gone (zombie = gone): then signals
# the test's process group (pid read from the file at that moment) and runs the release
# command. It never outlives the main shell by more than one poll (3 s).
oc_watchdog() {
  local main="$1" pidfile="$2"; shift 2
  local tp
  trap - EXIT INT TERM
  while oc_alive "$main"; do sleep "${OC_WATCHDOG_POLL_S:-3}"; done
  tp=$(cat "$pidfile" 2>/dev/null)
  [ -n "$tp" ] && kill -TERM -- "-$tp" 2>/dev/null
  "$@"
  echo "watchdog: main $main gone without cleanup; test group ${tp:-none} signalled; release ran at $(oc_ms) ms"
}

# oc_monitor <test_pid> <regex> <timeout_s> <file|dir>...
# Polls every 2 s while the test lives. On an anomaly marker: prints "ANOMALY <hit>", sends
# TERM to the test's process group at once, returns 20 (the caller does not retry).
# Timeout -> "TIMEOUT", TERM, 124. MemAvailable below ONCARD_ABORT_AVAIL_GB (default 100)
# -> "LOWMEM", TERM, 21. Normal exit of the test -> 0. Optional hook: oc_sample_hook <pid> <tick>.
oc_monitor() {
  local pid="$1" re="$2" tmo="$3"; shift 3
  local t0 tick=0 hit a
  t0=$(oc_ms)
  while oc_alive "$pid"; do
    sleep "${OC_MONITOR_POLL_S:-2}"; tick=$((tick + 1))
    if hit=$(oc_scan_anomaly "$re" "$@"); then
      echo "ANOMALY $hit"; kill -TERM -- "-$pid" 2>/dev/null; return 20
    fi
    a=$(oc_meminfo_avail_kb)
    if [ "$a" -lt $(( ${ONCARD_ABORT_AVAIL_GB:-100} * 1048576 )) ]; then
      echo "LOWMEM MemAvailable $((a / 1048576)) GiB"; kill -TERM -- "-$pid" 2>/dev/null; return 21
    fi
    if [ $(( ($(oc_ms) - t0) / 1000 )) -ge "$tmo" ]; then
      echo "TIMEOUT ${tmo}s"; kill -TERM -- "-$pid" 2>/dev/null; return 124
    fi
    declare -F oc_sample_hook >/dev/null && oc_sample_hook "$pid" "$tick"
  done
  return 0
}
