# RUNBOOK — 第二版合版本验收：C（侧流 + 事件等待）vs D（C + 共享专家进窗口）

> 用户 2026-09-23 10:40 定（PLAN §4.32 末）：第二版 = C vs D，**只差一个开关**
> `VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW`（§4.31）；perf 判负 ⇒ 停下报告、不跑 GSM8K；
> D 臂 GSM8K 50 满分 ⇒ 直接 1319。
> 判据文件：`oncard_runs/20260923_105413_merge_v2/EXPECTED.txt`
> （md5 **9815c9dcfd7bb0cbd96c0158ef701ed6**，写于 10:57Z、任何一臂起服务之前，chmod a-w）。
> 读数全部**内联在本文**：run 目录是未跟踪产物，随时会消失。

## 0. 两臂开关清单（逐项）

| 项 | C 臂（第一版 = 侧流 + 事件等待） | D 臂（C + 共享专家进窗口） | 在哪里生效 |
|---|---|---|---|
| `kt_offload_config`（enabled / prefix / 100 / 96 线程 / threadpool 2 / numa [0,1]） | 同第一版 B 臂 | 同 | cfg |
| `packed_transfer` | false | false | cfg |
| `overlap_side_stream` | **true** | **true** | cfg |
| cfg 文件 | `cfg_C.json` | `cfg_D.json` | 🔴 **逐字节相同**，且与第一版 `cfg_B.json` 逐字节相同（md5 d64b673a…，`cmp -s` 断言） |
| `VLLM_ASCEND_DSV41_KT_SIDE_STREAM` | 1 | 1 | env |
| `VLLM_ASCEND_DSV41_KT_ASYNC_WAIT`（P1a） | 1 | 1 | env |
| `VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL`（P1b） | 0 | 0 | env |
| `VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW`（§4.31） | **0（显式传）** | **1** | env —— **两臂唯一差异** |
| `KT_OFFLOAD` / `EAGER` / `VLLM_USE_BREAKABLE_CUDAGRAPH` | 1 / 0 / 1 | 同 | env |
| `MAXLEN` / `MAXSEQS` / `MAXBATCH` | 8192 / 8 / 4096 | 同 | env |
| `VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG` | 30 | 30 | env |
| `SEAM_TIMING` / profile | 未设 / 不采 | 同 | — |

每臂必检（`run_arm.sh` 自做，对不上即作废 rc=5）：`vllm_ascend` 指向本树；`SIDE STREAM ENABLED` 40/40；
`ASYNC WAIT ENABLED` 40/40；`SHARED IN WINDOW for layer` **C 0 / D 40**；`DEVICE SIGNAL ENABLED` 0/0；ACL 错码 0。
起服务前（rc=3 拒飞）：本树 `git status --short` 为空；`build_fingerprint.json` 两棵树 `dirty_files` 都 0；
`git diff --name-only d1c5b7faf HEAD -- vllm_ascend/` 为空（主会话会往本树提交文档，HEAD 可以动，运行时代码不许动）。

## 1. 怎么跑

```bash
ROOT=/workspace/code/dsv4_single_card/vllm_project
T=$ROOT/worktrees/moe-offload/tools/kt_offload/oncard/sidestream
RUN=$ROOT/oncard_runs/20260923_105413_merge_v2      # cfg_C/cfg_D/cfg_v1B、prompts、EXPECTED 已就位且只读
bash $T/accept_v2.sh $RUN perf   # C1,D1,C2,D2,C3,D3 → perf_verdict.py --arms CD --signed；判负 rc=10 即停
bash $T/accept_v2.sh $RUN rest   # 仅当 VERDICT_perf 成立：C4(lp)、D4(lp→gsm50→满分才 1319)→ lp_verdict；有差补 C5/D5
```
卡：先卡 2；只在 C1 起飞前卡 2 等满 20 min 仍不干净才改卡 1（此时无读数）；之后不中途换卡；卡 3 在脚本里硬拦。

## 2. 仪器改动（本发新增，均在 `sidestream/` 下）

| 文件 | 改动 | 为什么 |
|---|---|---|
| `run_arm.sh` | `SIW`（显式传 0/1）+ 必检 `SHARED IN WINDOW for layer` 40/0 | 「不设」会继承父 shell 的残留值；开关没进进程时两臂逐位全同 + perf 无差，和「改动没用」长得一样 |
| `run_arm.sh` | `REQUIRE_CLEAN_TREE` / `RUNTIME_BASE`：起服务前记 HEAD + `git status`、核指纹 dirty_files、核运行时代码未动；停服务时再记一次 | 第一版 A4 作废的教训（另一 agent 在跑期间改了本树） |
| `run_arm.sh` | 步骤 `gsm1319if50`：同一服务 GSM8K 50 = 50/50 才跑 1319，否则 rc=7 | 用户口径「<50 停下报告」由脚本执行，不靠人看 |
| `run_arm.sh` | `DRY_RUN=1`：做完起飞前核对即退出，不碰卡 | 干跑 |
| `perf_verdict.py` | `--arms CD`、`--signed` | 🔴 旧写法判 `|Δ|`：候选臂**慢** 3×地板也打「收益成立」（下表变异臂实证）。前几发 Δ 全为正所以没咬到；第二版用带符号 |
| `lp_verdict.py`（新） | 调 `logprob_cmp.py` + 数退化步，按 5.1 六条机械判 | 判据不靠人读输出 |
| `seq_merge_v2.sh`（新） | C/D 交错；`cmp -s cfg_C cfg_D cfg_v1B`；每臂预检（与 `seq_merge.sh` 同口径）；卡 3 硬拦 | — |
| `accept_v2.sh` rest 的 `REST_OVERRIDE=<addendum>`（11:47 后加） | 用户决定跳过 perf 判定时，显式指向 run 目录内带 md5 的 addendum 放行；不伪造 `VERDICT_perf.txt` | 变异：无判定无 override rc=10、override 在 run 目录外 rc=2、addendum 与 md5 不符 rc=2、合法 override + 脏树 rc=3（拒飞） |
| `accept_v2.sh`（新） | 整段驱动，分 perf / rest 两段 | 协调方要求 perf 段结束先报一声 |

### 2.1 干跑与变异臂（阶段一，不占卡）

夹具在 scratchpad（会消失），读数内联于此；全部可在无卡机器上复验。

| 仪器 | 喂进去 | 期望 | 实测 |
|---|---|---|---|
| `perf_verdict.py --arms CD --signed` | C 31.4/31.3/31.5 · D 30.0/29.9/30.1 | 成立 | ✅ rc=0（Δ+1.400，floor 0.200） |
| 同上 | C 31.4/30.6/31.5 · D 30.0/30.9/30.1 | 判负（地板吃掉） | ✅ rc=1（Δ+0.833 < 3×0.900） |
| 同上 | D 比 C **慢** 2.0（33.4/33.3/33.5），floor 0.2 | 判负 | ✅ rc=1（Δ−2.000）；🔴 **不带 `--signed` 的旧写法 rc=0「收益成立」** |
| 同上 | 缺 D3 | 报错非判定 | ✅ rc=3 |
| 同上 `--arms CC` | 非法参数 | 报错 | ✅ rc=3 |
| `perf_verdict.py`（默认 AB，回归） | §4.28 P1a 32.36/32.51 · 31.82/31.98 | 成立 | ✅ rc=0（Δ+0.535，3×floor 0.480） |
| 同上 | §4.29 P1b 32.52/31.39 · 31.32/30.89 | 判负 | ✅ rc=1（Δ+0.850，floor 1.13） |
| `lp_verdict.py` | 第一版 A4r vs B4 真实数据，skip=13 | 全过 | ✅ rc=0（decode 1174/1174，退化 0/1174 ×2，与 RUNBOOK_merge §3 一致） |
| 同上 | B4 自身 vs 把 i=50 第 2 位一个 top-k 值 +1e-6 | 红 | ✅ rc=1（decode 逐位、非零 Δlogprob 两条红） |
| 同上 | 两份都在 i=60 第 1 位造退化步（逐位全同） | 红 | ✅ rc=1（只有退化步一条红 —— 「全同」≠「对」） |
| 同上 | skip=999（0 条可比） | 不许判全同 | ✅ rc=3 |
| `seq_merge_v2.sh`（rounds=0） | 三份 cfg 相同 | 绿 | ✅ rc=0 |
| 同上 | cfg_D 线程 96→64 | 红 | ✅ rc=2 |
| 同上 | cfg_D 末尾多一个空格（语义相同、字节不同） | 红 | ✅ rc=2 |
| 同上 | cfg_C = cfg_D 但都 ≠ 第一版 cfg_B | 红 | ✅ rc=2 |
| 同上 | 卡 3 | 红 | ✅ rc=2 |
| `seq_merge_v2.sh` + `DRY_RUN=1` | 本树有未提交改动（本仪器自己的 4 个文件） | 拒飞 | ✅ rc=3，`git status` 4 行打进臂日志，服务未起 |
| 同上，提交 e00dc3012 后 | 干净树、三轮六臂 | 绿，C 臂 SIW=0 / D 臂 SIW=1 | ✅ rc=0；每臂 HEAD e00dc3012、git-status 0 行、dirty_files vllm=0 vllm_ascend=0、运行时差 0 文件 |
| 同上 `RUNTIME_BASE=f8eff083d`（§4.31 合入之前） | vllm_ascend/ 有 6 个文件差 | 拒飞 | ✅ rc=3 |

🔴 干跑**不覆盖**的部分（只有上卡才验得到）：env 真进了服务进程（靠必检计数 40/0）、预检的「等待→放行」分支
（干跑时卡 2 已空，只走了立即放行）、`gsm1319if50` 的两个分支、`accept_v2.sh` 的换卡分支。

## 3. 读数（`oncard_runs/20260923_105413_merge_v2`，全部在卡 2 上跑；run 目录未跟踪，读数内联于此）

构建与树（每臂都盖了指纹，下表逐臂核过）：`.so` 为 85331298…；每臂起服务前和停服务时，`git status` 都是 **0 行**；
两棵树的 `dirty_files` 都是 **0**；`vllm_ascend/` 相对 d1c5b7faf 的差异是 **0 个文件**。
两个阶段的 HEAD 不同，但都只是 tools/ 下的提交：perf 段是 4d1d0eb22（主会话合入的 t5r 工具修复），逐位和 GSM8K 段是 52c0d1e05（本线提交的 REST_OVERRIDE）。

| 臂 | 就绪 | SIDE | ASYNC | SHARED IN WINDOW | DEVICE SIGNAL | ACL 错码 | 每层报告 / never-ran |
|---|---|---|---|---|---|---|---|
| C1 | 240 s | 40 | 40 | **0** | 0 | 0 | 21 / 0 |
| D1 | 355 s | 40 | 40 | **40** | 0 | 0 | 20 / 0 |
| C2 | 305 s | 40 | 40 | 0 | 0 | 0 | —（🔴 作废，见 §3.1） |
| C4 | 320 s | 40 | 40 | **0** | 0 | 0 | 16 / 0 |
| D4 | 325 s | 40 | 40 | **40** | 0 | 0 | 144 / 0 |

### 3.1 perf：只跑了一轮（用户 11:47 决定），🔴 **不是 §4.27 判定**
| 臂 | decode ms/token（TPOT 中位）均值 | 轮内 min–max | TPOT 均值 | TTFT 中位 | 输出吞吐 |
|---|---|---|---|---|---|
| **C1**（侧流 + 事件等待） | **32.02** | 31.78–32.33（σ0.28） | 31.99 | 769.3 ms | 28.63 tok/s |
| **D1**（C + 共享专家进窗口） | **30.31** | 30.12–30.60（σ0.26） | 31.56（σ2.43，有一轮 34.36 离群） | 739.4 ms | 29.17 tok/s |

**Δ = C1 − D1 = +1.71 ms/token**（带符号，D 更快）。按 §4.31 的推断，上限约 1.5、预期约 1.4，实测量级相符。
同一服务内三轮摆动：C 0.55、D 0.48 ⇒ 3×0.55 = 1.65 ≤ 1.71；Δ ≥ 0.3 也满足。
- 用户 11:47 决定 perf 只要 C1/D1 这一轮（理由是「收益符合预期」），见 `ADDENDUM_user_1147.txt`（md5 94d81de2…，只读）。
  这份 ADDENDUM 写于 C1、D1 读数都已产出之后。`EXPECTED.txt` 没改动；`VERDICT_perf.txt` **没有生成，也没有伪造**。
- 🔴 **局限**：
  - 单轮，**没有同臂跨重启的地板**。同一服务内的摆动比跨重启地板偏乐观（§4.18）。
  - 1.65 与 1.71 之间余量只有 0.06。
  - D1 的 `TPOT 均值` 那一行有一轮离群（34.36）。判读用的是中位那一行，不受影响，但如实记下。
  - 要写进收益账，还欠一发预先声明门槛的三轮交错。
- **C2 作废（原因是用户决定，不是读数，也不是外部占卡）**：
  - C2 在 11:48:41 就绪，必检全过，perf 采集到一半时被主动停掉。
  - 停法：驱动进程组 kill -TERM；服务走 `stop_by_port.sh 8041` 并带 `STOP_PGID=152986`，rc=0，端口已释放。
  - C2 只留下 warmup 产物，不进入任何计算。详见 `C2/VOID.txt`。
- rest 的放行依据是 `REST_OVERRIDE=ADDENDUM_user_1147.txt`（见 §2），驱动日志里点名了这份文件。

### 3.2 整网逐位：C4 vs D4（`lp_verdict.py`，skip=13，种子 20260923104）
| 判据（EXPECTED §5） | 要求 | C4 vs D4 |
|---|---|---|
| prefill 内部对照 | 全同 | **187/187 = 100%** |
| decode 逐位 | 100% | **1170/1170 = 100%** |
| 非零 \|Δlogprob\| | 0 | **0**（7020 个条目，max 0） |
| top-k key-set 差异 | 0 | **0** |
| chosen-token 差异 | 0 | **0** |
| 退化步 | 0 | C4 **0/1170**，D4 **0/1170** |

⇒ **全过**，所以按 EXPECTED 不补 C5/D5。截断点 13 是沿用值，没有由本发结果反推。
D4 的 `SHARED IN WINDOW` 计数是 40，说明两臂全同**不是因为开关没开**。

### 3.3 GSM8K（只跑 D 臂，与 D4 共用同一次服务）
| 集 | 分数 | err / 截断 | 墙钟 |
|---|---|---|---|
| GSM8K 50 | **50/50** | 0 / 0 | 110 s |
| GSM8K 1319（50 满分后直接跑） | **1283/1319 = 97.27%** | 0 / 0 | 3736 s |

**判读**（EXPECTED §6）：
- 多卡带是 1276–1281。1283 比带的上沿高 2 题；同代码跨重启的噪声是 11 题，差距在噪声以内。
- 与侧流单独那一版的 1275/1319 相差 8 题，同样 ≤11。
- 结论：**没有可检出的回归**。🔴 **不得读成「比多卡更好」或「比侧流版更好」**，这些差距都不可分辨。
- 这是合版本第一次有 1319 读数（第一版的 1319 当时暂缓了）。

### 3.4 作废与残留
- 作废只有 C2 一臂（用户决定）。没有外部占卡，每臂起飞前的卡 2 预检都是 4%/0% 立即放行。
- 每臂停服务后都有 `⚠ pgid still alive`，查过是 `<defunct>` 僵尸进程，不占卡（13:29 卡 2 HBM 4%、Aicore 0）。第一版也是同样现象。
- 没有采 profile。根盘余量全程 880 GiB。
