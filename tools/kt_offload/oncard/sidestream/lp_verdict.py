"""Apply the whole-network bit-identity criterion (EXPECTED 5.1) mechanically.

usage: lp_verdict.py <X.jsonl> <Y.jsonl> <label> --skip-first=N

Runs ``logprob_cmp.py`` (the same comparison every earlier run quoted) and
counts degenerate decode steps (every top-k logprob == ln(1/129280), the
signature of the 2026-09-22 side-stream bug) in each file over the same
prompts (i >= skip). Pass <=> prefill 100%, decode 100%, nonzero |dlogprob| 0,
top-k key-set differences 0, chosen-token divergences 0, degenerate steps 0 in
both files, and at least one decode position compared (0 compared is "not
looked at", never "identical").

exit 0 = all pass, 1 = some criterion failed, 3 = unreadable output.
"""
import json
import math
import os
import re
import subprocess
import sys

U = math.log(1 / 129280)
here = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--skip-first=")]
skip_arg = [a for a in sys.argv[1:] if a.startswith("--skip-first=")]
if len(args) != 3 or len(skip_arg) != 1:
    print(__doc__)
    raise SystemExit(3)
x, y, label = args
skip = int(skip_arg[0].split("=")[1])
out = subprocess.run([sys.executable, f"{here}/logprob_cmp.py", x, y, label, f"--skip-first={skip}"],
                     capture_output=True, text=True, check=False)
print(out.stdout, end="")
if out.returncode != 0:
    print(out.stderr)
    raise SystemExit(3)
res = {}
for ph in ("prefill", "decode"):
    m = re.search(rf"{ph}\s*: positions compared (\d+), bit-identical (\d+) .*?nonzero (\d+),", out.stdout)
    if not m:
        print(f"❌ 解析不出 {ph} 行")
        raise SystemExit(3)
    res[ph] = tuple(int(v) for v in m.groups())
m = re.search(r"top-k key-set differences: (\d+); chosen-token divergences: (\d+)", out.stdout)
if not m:
    print("❌ 解析不出 key-set/divergence 行")
    raise SystemExit(3)
kdiff, tdiv = int(m.group(1)), int(m.group(2))


def degen(path):
    n = d = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["i"] < skip:
            continue
        for pos, tl in enumerate(r["top_logprobs"] or []):
            if pos == 0:
                continue
            n += 1
            d += bool(tl) and all(abs(v - U) < 1e-3 for v in tl.values())
    return d, n


dx, nx = degen(x)
dy, ny = degen(y)
checks = [
    ("prefill 全同", res["prefill"][0] > 0 and res["prefill"][0] == res["prefill"][1]),
    ("decode 逐位 100%（且比较数 > 0）", res["decode"][0] > 0 and res["decode"][0] == res["decode"][1]),
    ("非零 |Δlogprob| = 0", res["prefill"][2] == 0 and res["decode"][2] == 0),
    ("top-k key-set 差 = 0", kdiff == 0),
    ("chosen-token 差 = 0", tdiv == 0),
    (f"退化步 = 0（{os.path.basename(os.path.dirname(x))} {dx}/{nx}，{os.path.basename(os.path.dirname(y))} {dy}/{ny}）",
     dx == 0 and dy == 0),
]
for name, ok in checks:
    print(f"  {'✅' if ok else '❌'} {name}")
ok = all(c for _, c in checks)
print(f"  ⇒ {'🟢 全过' if ok else '🔴 有差（按 EXPECTED：补自比地板臂再判读，截断点只许由自比地板校）'}")
sys.exit(0 if ok else 1)
