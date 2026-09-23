# 会话状态快照（2026-09-23 09:00，压缩上下文前留）

> 本文只记**活的状态**：正在跑什么、还差什么、东西在哪。
> 结论与读数一律在 `MOE_OFFLOAD_PLAN.md` §4.18–§4.30，**以那里为准**。

## 正在跑（两个后台 agent，互不占卡冲突）

| 事 | 产物目录 | 状态 |
|---|---|---|
| **合版本验收**（侧流 + 事件等待 P1a 两开 vs 两关） | `oncard_runs/20260923_091123_merge/`（第一发 `082414` 因外部租户占卡 0+1 作废，见其 `VOID.txt`） | **卡 2**。步骤：交错 ≥3 轮 perf（判据写在该目录 `EXPECTED.txt`，先声明后跑）→ 整网 logprob 逐位（四臂，skip=13）→ GSM8K 50 两臂 → **B 臂 1319 全量** |
| **trace 切片 + 甘特图** | `build_artifacts/moe_offload_ref/trace_slices/`（尚未生成） | 离线。从 `/dev/shm/dsv41_prof_seam`（串行）与 `_seam_side`（侧流）各抽 2–3 个稳态 step，出 `prof/trace_slice.py` + `prof/trace_gantt_section.html`；**必须与已内联聚合读数对上**，对不上就停下报告 |

⚠️ `/dev/shm/dsv41_prof_*` 是内存盘，**机器重启就没**；P1b 那份原始 trace 已删（09-23 根盘写满时）。

## 收尾还差三件

1. **把合版本结果补进 artifact 的「八、合版本验收」那一节**（现在是占位）。
2. **把 trace 甘特图片段并进 artifact**（agent 会给 `trace_gantt_section.html`，直接贴进 `<section>` 即可）。
3. **重写 `HANDOFF_next_session.md`** —— 它现在还停在 09-22 早上的状态（讲的是"去做 rebase"），已严重过期。新交接要写：
   - 现状：rebase/换包/基线/侧流/事件等待都已完成；接缝线已收敛（削 host 侧换不来 TPOT）。
   - 下一阶段入口：**频率偏斜（M0c）** 与 **吞吐轴（L3 → MAXSEQS）**，都在 §4.18.1 记账。
   - 仍开放的待查项：§4.30 那条「A+F 的 78 µs 去哪了」。

## artifact（这一轮的技术报告页）

- 链接 **https://claude.ai/artifact/7jpcSsWTme7E5KXbeyA2zH**（私有；要给别人看需在页面 Share 菜单分享）
- 源文件 `/tmp/claude-0/-workspace-code-dsv4-single-card-vllm-project/8de95c21-8bb5-40c3-a905-17734ca6f3a1/scratchpad/moe_offload_report.html`
  🔴 **在 scratchpad 里，会随会话清理消失**。要改就改这个文件再用同一路径重新 publish（保持同一 URL）；
  若文件没了，用 `Artifact action:"read"` 把已发布版本取回来再改。
- 已发布到 v5。配色经过 CVD 校验：light `#0C93A4/#D2691E/#8E44AD`、dark `#2AA6B0/#D07C34/#A87BD0`。

## 卡与纪律（当前有效）

- **只用卡 0–2，卡 3 不许用**（用户 09-22 定，未解除）。
- 起飞前 HBM+aicore 连采 ≥3 次；`stop_by_port.sh` 带 `STOP_PGID`；错码用 `ONCARD_ERRCODES` 五码名单 + 非数字边界 + 排除埋点行。
- **收益门槛（§4.27，先声明后验）**：|A−B| ≥ 3 × 同臂地板 且 ≥ 0.3 ms/token；没测地板判 fail。
- **测量口径**：默认只跑 perf + 精度逐位，**profile 非必要不采**（一发 25 min 起）。
- 根盘 09-23 被别的租户写满过，现余量小 ⇒ 采集前先 `df -h /`。

## 本轮改动的落点（都已提交，未 push）

`vllm_ascend/kt_offload/`：`layer.py`（侧流 fill-before-fork、节点发射）、`async_wait.py`（事件等待的完成线程）、
`device_signal.py`+`poll_native.c`（计数器启动，实测无收益、默认关）、`side_stream.py`、`config.py`、`envs.py`、`seam_timing.py`。
工具：`tools/kt_offload/prof/*`（seam/devgaps/bench）、`oncard/t5_side_reuse.py`、`oncard/sidestream/*`、`oncard/RUNBOOK_*.md`。
kt 树 `worktrees/ktransformers`（HEAD `53eab3c`）：`run_inline` 与 `KT_WORKER_SPIN_MS`，**未重建 wheel、未切站点**（P2 跳过不测）。

## 10:10 追加
- **共享专家挪进窗口**：改动在独立 worktree `staging/moe-offload-shared`（分支 `dsv41-shared-window`）里做，**不在本树**。
  原因：09:48–09:52 改动曾直接写进本树，而合版本验收正从本树起服务 ⇒ A4 作废、改为重跑 A4r。
- 合版本（卡 2）：单轮 perf 达预期（A1 35.60 / B1 31.38，Δ 4.22，判据 `EXPECTED_addendum_0936.txt`）；
  按用户 09:56 决定：逐位 A4r/B4 → GSM8K 50（仅 B 臂）→ 暂停，**1319 等共享专家合第二版再跑**。

## 10:50 追加
- 第一版合版本验收完成（§4.32）：单轮 Δ4.22（稳妥 3.3–4.2）、decode 1174/1174 逐位、GSM8K 50/50；1319 并入第二版。
- §4.31 已合入本树（`3226350f2`），开关默认关，单测 595/0。
- 在跑：① t5r 替身修复 + 重跑（卡 0，`staging/moe-offload-shared`）；② 第二版验收 C（侧流+事件等待）vs D（+共享专家进窗口），**阶段一只准备，等 t5r 过了我发 GO 才上卡 2**，perf 交错 3 轮（1.4 ms 级收益单轮判不出来）。
- 用户决定：perf 判负 ⇒ 停下报告；D 臂 GSM8K 50 满分 ⇒ 直接 1319；**全部验收通过后，本轮所有优化的 env 改为默认开**（侧流 / 事件等待 / 共享专家进窗口；计数器启动 P1b 保持默认关——用户 10:55 确认）。
