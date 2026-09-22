# SPDX-License-Identifier: Apache-2.0
"""Shared preamble for the E2'/E5 gate-clearing checks: build identity read at
run time, and a host-memory guard.

Why a guard and not a note: on 2026-09-22 a phase2 GSM8K run holds a large part
of host memory, and these checks must not compete with it. A policy is not an
enforcement (memory: 政策不等于强制) -- so the floor raises, and it is checked
before every expensive step rather than only at the start, because the number
that matters is the one at the moment of the allocation.
"""

from __future__ import annotations

import json
import os

BUILD_NOTE = (
    "本次结论测的是【当前】构建：rebase 与换包都已落地（clean 头已前进，本树 .so 换成 clean "
    "同一份）。实际的 .so md5 与树 HEAD 由 build_under_test() 当场从本次运行的 "
    "build_fingerprint.json 读出并印在下面，不复制任何记忆或任务书里的常量——那些常量今天已经"
    "过期过两次（任务书给的 .so 与 HEAD 在我开工时就已作废）。新 clean 改了 ConvertType 的 "
    "4-bit 分支，而 E2' 测的正是那条路，所以这一发打的就是新代码。跨换包的读数与旧 .so 不可比。"
)


def build_under_test(run_dir: str) -> dict:
    """The .so md5 and tree HEADs of THIS run, read from the fingerprint that
    run_oncard.sh wrote before the card was touched.

    Evaluated here rather than quoted: a build identity frozen into a constant
    goes stale silently, and on 2026-09-22 both the .so md5 and the tree HEAD
    handed to this task were already wrong when the task started.
    """
    out: dict = {"source": None}
    for cand in (os.path.join(run_dir, "build_fingerprint.json"),
                 os.path.join(run_dir, "provenance.json")):
        try:
            with open(cand) as f:
                d = json.load(f)
        except Exception:
            continue
        fp = d.get("build_fingerprint", d)
        out = {
            "source": cand,
            "so_primary_md5": fp.get("so_primary_md5"),
            "so_primary_md5_short": (fp.get("so_primary_md5") or "")[:12] or None,
            "trees": {k: {"path": t.get("path"), "head": t.get("head"), "branch": t.get("branch"),
                          "dirty_files": t.get("dirty_files")}
                      for k, t in (fp.get("trees") or {}).items()},
        }
        break
    if out.get("source"):
        print(f"[build under test] so={out.get('so_primary_md5_short')} "
              + " ".join(f"{k}={(t.get('head') or '?')[:9]}(dirty={t.get('dirty_files')})"
                         for k, t in (out.get("trees") or {}).items()), flush=True)
    else:
        print("[build under test] UNREADABLE -- this run's build identity is not recorded; "
              "any reading from it can be compared to nothing", flush=True)
    return out


# ---------------------------------------------------------------------------
# host memory
# ---------------------------------------------------------------------------


def mem_avail_gib() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    raise RuntimeError("MemAvailable not in /proc/meminfo")


def rss_gib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576.0
    return float("nan")


def hwm_gib() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1048576.0
    return float("nan")


class MemGuard:
    """Self-stop on host memory. ``floor_gib`` is the abort line; ``budget_gib``
    is what THIS process is allowed to add over its own baseline.

    Both are checked, because they answer different questions: the floor is
    about the machine (a co-tenant GSM8K run must not be pushed into swap or
    the OOM killer), the budget is about us (a check that quietly grew to tens
    of GiB is a different experiment from the one that was authorised). The
    floor reading is of the whole machine and therefore includes the co-tenant:
    it can trip for reasons that are not ours, and the message says so rather
    than blaming this process.
    """

    def __init__(self, floor_gib: float = 100.0, budget_gib: float = 10.0):
        self.floor, self.budget = float(floor_gib), float(budget_gib)
        self.base_rss = rss_gib()
        self.start_avail = mem_avail_gib()
        self.min_avail = self.start_avail
        self.max_added = 0.0
        print(f"[memguard] MemAvailable {self.start_avail:.1f} GiB at start; floor {self.floor:.0f} GiB; "
              f"our baseline RSS {self.base_rss:.2f} GiB, budget +{self.budget:.0f} GiB", flush=True)

    def check(self, stage: str) -> None:
        avail = mem_avail_gib()
        added = rss_gib() - self.base_rss
        self.min_avail = min(self.min_avail, avail)
        self.max_added = max(self.max_added, added)
        if avail < self.floor:
            raise MemoryError(
                f"[memguard] STOP at '{stage}': MemAvailable {avail:.1f} GiB < floor {self.floor:.0f} GiB. "
                f"This process has added {added:.2f} GiB over its {self.base_rss:.2f} GiB baseline "
                f"(budget {self.budget:.0f} GiB), so the shortfall may be a co-tenant's, not ours -- "
                f"the reading is of the machine. Nothing further is attempted; report to the coordinator."
            )
        if added > self.budget:
            raise MemoryError(
                f"[memguard] STOP at '{stage}': this process has added {added:.2f} GiB over its "
                f"{self.base_rss:.2f} GiB baseline, past the {self.budget:.0f} GiB budget "
                f"(MemAvailable {avail:.1f} GiB). The check grew beyond what was authorised."
            )
        print(f"[memguard] {stage}: MemAvailable {avail:.1f} GiB, ours +{added:.2f} GiB "
              f"(peak +{self.max_added:.2f}, HWM {hwm_gib():.2f} GiB)", flush=True)

    def facts(self) -> dict:
        return {"floor_gib": self.floor, "budget_gib": self.budget, "baseline_rss_gib": self.base_rss,
                "start_avail_gib": self.start_avail, "min_avail_gib": self.min_avail,
                "max_added_gib": self.max_added, "final_hwm_gib": hwm_gib()}
