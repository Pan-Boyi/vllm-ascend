#!/usr/bin/env bash
# CPU-only end-to-end self-test of run_longctx_check.py against stub_server.py.
# Each mode has an EXPECTED verdict per item; the test fails if any differs.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
PORT=${STUB_PORT:-18766}
OUTD=${OUTD:-$D/selftest_out}
mkdir -p "$OUTD"
fail=0

start_stub() {  # $1 mode
  STUB_MODE=$1 python3 "$D/stub_server.py" "$PORT" >"$OUTD/stub_$1.log" 2>&1 &
  STUB_PID=$!
  for _ in $(seq 1 120); do
    python3 - "$PORT" <<'EOF' 2>/dev/null && return 0
import sys, urllib.request
o = urllib.request.build_opener(urllib.request.ProxyHandler({}))
o.open(f"http://127.0.0.1:{sys.argv[1]}/v1/models", timeout=1)
EOF
    sleep 0.5
  done
  echo "stub $1 did not come up"; return 1
}
stop_stub() { kill "$STUB_PID" 2>/dev/null; wait "$STUB_PID" 2>/dev/null; }

check() {  # $1 mode  $2 json-of-expected {id: verdict}  $3 expected rc
  local mode=$1 expect=$2 want_rc=$3; shift 3
  start_stub "$mode" || { fail=1; return; }
  python3 "$D/run_longctx_check.py" --port "$PORT" --ids all --timeout 60 \
      --out "$OUTD/$mode.json" "$@" >"$OUTD/run_$mode.log" 2>&1
  local rc=$?
  stop_stub
  python3 - "$OUTD/$mode.json" "$expect" "$rc" "$want_rc" "$mode" <<'EOF' || fail=1
import json, sys
path, expect, rc, want_rc, mode = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
bad = []
if rc != want_rc:
    bad.append(f"rc={rc} want {want_rc}")
if expect:
    got = {r["id"]: r["verdict"] for r in json.load(open(path))["results"]}
    for k, v in expect.items():
        if got.get(k) != v:
            bad.append(f"{k}: got {got.get(k)} want {v}")
    res = json.load(open(path))["results"]
    for r in res:
        for arm in ("full", "trunc"):
            a = r[arm]
            if a.get("prompt_tokens") is not None and a.get("server_minus_local") not in (0,):
                bad.append(f"{r['id']} {arm}: server_minus_local={a.get('server_minus_local')}")
        if r["full"].get("prompt_tokens") is not None and r["full"]["prompt_tokens"] <= 16384:
            bad.append(f"{r['id']} full prompt <=16384")
        if r["trunc"].get("prompt_tokens") is not None and r["trunc"]["prompt_tokens"] >= 16384:
            bad.append(f"{r['id']} trunc prompt >=16384")
print(f"[{mode}] {'PASS' if not bad else 'FAIL ' + '; '.join(bad)}")
sys.exit(1 if bad else 0)
EOF
}

NEM=66ebce825a08c7b9b35dfc75; MET=66ec0f3c821e116aacb19d70; BAL=66ec3a46821e116aacb1c49f
LLA=66ebc0c95a08c7b9b35de7f3; COD=6708a096bb02136c067d1789; EXI=66ebc95f5a08c7b9b35df497
ABV=66fc0152bb02136c067c8826

P=DISCRIMINATIVE_PASS; N=NO_DISCRIMINATION; W="FULL_WRONG(both_wrong)"
check oracle "{\"$NEM\":\"$P\",\"$MET\":\"$P\",\"$BAL\":\"$P\",\"$LLA\":\"$P\",\"$COD\":\"$N\",\"$EXI\":\"$P\",\"$ABV\":\"$P\"}" 0
check always_correct "{\"$NEM\":\"$N\",\"$MET\":\"$N\",\"$BAL\":\"$N\",\"$LLA\":\"$N\",\"$COD\":\"$N\",\"$EXI\":\"$N\",\"$ABV\":\"$N\"}" 0
# targets: NEM C, MET A, BAL A, LLA C, COD D, EXI B, ABV A
check always_A "{\"$NEM\":\"$W\",\"$MET\":\"$N\",\"$BAL\":\"$N\",\"$LLA\":\"$W\",\"$COD\":\"$W\",\"$EXI\":\"$W\",\"$ABV\":\"$N\"}" 0
U="INCONCLUSIVE(full:unparsed)"
check garbage "{\"$NEM\":\"$U\",\"$MET\":\"$U\",\"$BAL\":\"$U\",\"$LLA\":\"$U\",\"$COD\":\"$U\",\"$EXI\":\"$U\",\"$ABV\":\"$U\"}" 0
NU="INCONCLUSIVE(full:no_usage)"
check no_usage "{\"$NEM\":\"$NU\",\"$MET\":\"$NU\",\"$BAL\":\"$NU\",\"$LLA\":\"$NU\",\"$COD\":\"$NU\",\"$EXI\":\"$NU\",\"$ABV\":\"$NU\"}" 0
check http500 "{}" 3                                   # preflight must refuse
E="INCONCLUSIVE(full:error)"
check http500 "{\"$NEM\":\"$E\",\"$MET\":\"$E\"}" 1 --skip-preflight
# niah path (read-only call of cookbook/niah_scan.py), tiny lengths
NIAH_OUT="$OUTD/niah.json" check oracle "{\"$NEM\":\"$P\"}" 0 --niah --niah-lengths 2000,4000 --niah-depths 0,100
python3 - "$OUTD/niah.json" <<'EOF' || fail=1
import json, sys
d = json.load(open(sys.argv[1]))
ok = d["hit"] == d["total"] == 4 and d["errors"] == 0 and d["corpus"] == "real"
print(f"[niah] {'PASS' if ok else 'FAIL'} hit={d['hit']}/{d['total']} errors={d['errors']} corpus={d['corpus']}")
sys.exit(0 if ok else 1)
EOF

echo; [ $fail -eq 0 ] && echo "ALL STUB SELF-TESTS PASSED" || echo "SOME STUB SELF-TESTS FAILED"
exit $fail
