# RUNBOOK — 合版本（侧流 + P1a）验收

> 用户 2026-09-23 定的合版本内容：**开**侧流（`overlap_side_stream` + `VLLM_ASCEND_DSV41_KT_SIDE_STREAM=1`）
> 与 P1a（`VLLM_ASCEND_DSV41_KT_ASYNC_WAIT=1`）；**不开** P1b（`_DEVICE_SIGNAL`，§4.30 判机制不成立）、
> 报告线程绑核（§4.27 前节判死）、P2 内联（跳过不测）、packed（默认关）。
> 相对【现状基线】＝两个开关都关（§4.18 的串行形态）测**总收益**，再跑精度。
>
> 读数全部**内联在本文**：run 目录是未跟踪产物（k.16），随时会消失。

## 0. B 臂的完整开关清单（逐项）

| 项 | A 臂（现状基线 / 串行） | B 臂（合版本） | 在哪里生效 |
|---|---|---|---|
| `kt_offload_config.enabled` | true | true | cfg |
| `resident_strategy` / `num_resident_experts` | prefix / 100 | 同 | cfg |
| `cpuinfer_threads` / `threadpool_count` / `numa_nodes` | 96 / 2 / [0,1] | 同 | cfg |
| `packed_transfer` | **false** | **false** | cfg（合版本不开 packed） |
| `overlap_side_stream` | **false** | **true** | cfg |
| `VLLM_ASCEND_DSV41_KT_SIDE_STREAM` | **0** | **1** | env |
| `VLLM_ASCEND_DSV41_KT_ASYNC_WAIT`（P1a） | **0** | **1** | env |
| `VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL`（P1b） | 0 | **0** | env（§4.30 判 ⛔，不进合版本） |
| `VLLM_ASCEND_DSV41_KT_SEAM_TIMING` | 未设 | 未设 | env（与读数互斥，见 §4.26②） |
| 报告线程绑核 | 无 | **无** | ⛔ 不进合版本（§4.27 前节） |
| `KT_OFFLOAD` / `EAGER` / `VLLM_USE_BREAKABLE_CUDAGRAPH` | 1 / 0（图模式） / 1 | 同 | env |
| `MAXLEN` / `MAXSEQS` / `MAXBATCH` | 8192 / 8 / 4096 | 同 | env |
| `VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG` | 30 | 30 | env（两臂对称） |
| profile | **不采** | **不采** | 根盘余量（§4.30 机器级） |

🔴 **两臂的 cfg 不是逐字节相同，这是代码强制的，不是本线的选择**：
`kt_offload/config.py:_validate_preconditions` 在 `overlap_side_stream=true` 而
`VLLM_ASCEND_DSV41_KT_SIDE_STREAM` 未置 1 时**直接 raise**（起服务就失败）；而
`layer.py:221` 的 `self.overlap` **只读 cfg、不读 env** ⇒ 侧流「关」必须由 cfg 关。
⇒ 两臂的最小差 = cfg 的 `overlap_side_stream` 一行 + 两个 env，与 §4.26 侧流 A/B 的做法一致
（那一发也是 cfg_A overlap=false / cfg_B overlap=true）。P1a 侧确是纯 env（§4.28），两臂 cfg 在它上面本来就相同。
⇒ `seq_p1a.sh` 的 `cmp -s cfg_A cfg_B` 断言在本发**会误拦**，所以本发用 `seq_merge.sh`，
断言改成「**除 `overlap_side_stream` 一行外逐字节相同**，且 A=false / B=true」，
并配了三个变异臂自证有区分力（见 §5）。

## 1. 怎么跑

```bash
ROOT=/workspace/code/dsv4_single_card/vllm_project
WT=$ROOT/worktrees/moe-offload/tools/kt_offload/oncard
RUN=$ROOT/oncard_runs/$(date +%Y%m%d_%H%M%S)_merge; mkdir -p $RUN
# cfg：A overlap=false / B overlap=true，其余逐字节相同
# EXPECTED.txt：判据先写死再跑（含「不满足就判负、不许调参数重试」），跑完不许改
# ① 总收益 A/B（交错 3 轮；卡只用 0-2）
bash $WT/sidestream/seq_merge.sh $RUN 2 perf 3          # A1,B1,A2,B2,A3,B3（每臂自带占卡预检，§7）
python3 $WT/sidestream/perf_verdict.py $RUN --rounds 3  # §4.27 判定，rc 0=成立 1=判负 3=少臂
# ② 整网逐位（四臂：A 自比地板 / B 自比地板 / 跨臂）
python3 $WT/sidestream/make_prompts.py $RUN/prompts.jsonl <新种子>
ARM_OFFSET=3 bash $WT/sidestream/seq_merge.sh $RUN 2 lp 2   # A4,B4,A5,B5
python3 $WT/sidestream/logprob_cmp.py $RUN/A4/lp.jsonl $RUN/B4/lp.jsonl "A4 vs B4 (跨臂)" --skip-first=13
# ③ GSM8K 50 → ④ GSM8K 1319（本发按用户决定只做到 ③，见 §8）
```

🔴 **起飞前**：HBM + aicore **连采 ≥3 次**（判最大值与趋势，不判瞬时；口径同
`cookbook/serve_multicard.sh` ③）。单采分不开「卡是空的」和「外部租户正在加载权重」。
卡**只用 0–2**（用户 2026-09-23）。

## 2. 读数（本发：`oncard_runs/20260923_091123_merge`，卡 2）

> 出处 run 目录未跟踪（k.16），**读数全部内联在此**。构建指纹每臂已盖，
> `.so` = `85331298bdeffee02b7c42c5658030ac`（每臂 `run_arm.sh` 自检，不符即 exit 3）。

### 2.1 第一发（`20260923_082414_merge`，卡 0）🔴 **作废 —— 外部占卡，不是读数不合意**
A2 起服务失败：`ValueError: Free memory on device (96.31/123.16 GiB) on startup is less than
desired GPU memory utilization (0.9, 110.84 GiB)`（rc=4，服务从未就绪），按其 EXPECTED §3 停在 A2。
**归因证据**（本线自己的 `card_before/after`，已内联）：

| 时刻 | 卡 0 | 卡 1 | 卡 2 | 说明 |
|---|---|---|---|---|
| A1 起飞前 08:28 | 5,253 MB | 5,247 | 5,251 | 三张都空 |
| B1 停机后 09:02:53 | **16,056** | **16,061** | 5,253 | 🔴 卡 1 我们**从没碰过** ⇒ 卡 0 那 16 GB 不是我方拆解残留 |
| A2 起飞前 09:05:03 | 17,079 | 18,109 | 5,253 | 还在涨 |
| A2 探测时 09:05:53 | ≈27.5 GB（由 96.31 GiB free 反推） | — | — | 外部租户**正在加载权重** |

⇒ 整段开头的预检是绿的（0/1/2 全 4%/0%、无趋势），外部租户是**跑到一半**上来的。
**处置**：① 换到全程干净的**卡 2**；② `seq_merge.sh` 加**每臂**起飞前预检（见 §7）。
作废发次只跑完 1 轮（A1 34.64 / B1 32.26，卡 0）—— 🔴 **不进任何判定**（§4.25 跨发不得相减），
但**如实记下我已经看过它**（见 §6 对 floor2 的说明）。

### 2.2 第二发（本发）：A1 vs B1
| 臂 | 形态 | decode ms/token(TPOT 中位) 均值 | 轮内 min–max | TPOT 均值 | TTFT 中位 | 输出吞吐 |
|---|---|---|---|---|---|---|
| **A1** | 串行（现状基线，两开关全关） | **35.60** | 35.49–35.78 (σ0.16) | 35.96 | 810.8 ms | 25.70 tok/s |
| **B1** | 合版本（侧流 + P1a） | **31.38** | 30.94–31.91 (σ0.49) | 31.39 | 751.9 ms | 29.22 tok/s |

**Δ = A1 − B1 = +4.22 ms/token（B 更快）**；吞吐 25.70 → 29.22 tok/s（**+13.7%**）。
A1 = 35.60 与 §4.18 基线 34.36/34.60（卡 0）同量级；本发在**卡 2**，
🔴 **跨卡跨发不得相减**，A1 在此只作本发的对照臂。

必检（两臂全过）：`vllm_ascend` 指向本树；`SIDE STREAM ENABLED` A1 **0** / B1 **40**；
`ASYNC WAIT ENABLED` A1 **0** / B1 **40**；`DEVICE SIGNAL ENABLED` 两臂 **0**；
ACL 错码行 **0**；每层报告 A1 24 行 / B1 21 行，`never-ran` **0**。

### 2.3 判定（`EXPECTED_addendum_0936`，用户 2026-09-23 09:36 决定）
🔴 **单轮、地板为代用 —— 这不是 PLAN §4.27 判定，不得写成「§4.27 收益成立」。**

```
Δ = 4.220
floor1（同服务内三轮摆动，代用）= max(0.29, 0.97) = 0.970 ⇒ 3×floor1 = 2.910
floor2（|本发 A1 35.60 − 作废发 A1 34.64|，跨发跨卡一次观测，代用）= 0.960 ⇒ 3×floor2 = 2.880
① Δ ≥ 0.3      ✅    ② Δ ≥ 3×floor1 ✅    ③ Δ ≥ 3×floor2 ✅   ⇒ 🟢 达预期
```
达预期 ⇒ 按 addendum 停止 perf 交错（A2/B2/A3/B3 **未跑**），直接进逐位。

🔴 **这个判定的三处局限，要跟着结论一起走**：
1. **本发没有测到同臂跨重启地板**。§4.27 的地板是同臂跨轮/跨重启的摆动；单轮测不到。
   `floor1` 是**同一次服务内**的摆动（§4.18：同服务内 σ 0.06–0.19，而真正的跨重启地板 0.24–1.0 ⇒ **偏乐观约 5–10×**）。
2. `floor2` 是**跨发跨卡相减**得来的（§4.25 明令读数不得跨发相减），且是**一次观测、不是分布**。
3. 时序如实记：**A1 读数 09:29:33 已产出，addendum 写于 09:36** ⇒ floor2 是看到 A1 之后由 A1 算出来的。
   ⚪ 缓解（也如实记）：三条判据都是**机械公式**、无可调阈值，且 3×floor2=2.88 比原 §4.27 门槛更严；
   但「更严」不等于「测到了地板」—— 上面第 1 条仍然成立。
⇒ **要写进收益账，还欠一发预先声明门槛的三轮交错 A/B。**

### 2.4 与单项收益的关系（别误读）
侧流单独 −3.00（§4.25/§4.26）、P1a 单独 −0.53（§4.28），两笔**分别**结过账，合 ≈3.53。
本发 4.22 是**合版本对串行基线的总账**，取自不同时段/不同卡、且 A 臂形态与那两发都不同
⇒ 🔴 **不得用 4.22 − 3.53 = 0.69 去说「还有协同收益」**，也不得拿本发去反推单项。

## 3. 整网逐位（A4r vs B4）

口径：`make_prompts.py` 200 条、种子 **20260923824**（本发首次用；与 §4.29 那批只重合 13/197）、
`--skip-first=13`（§4.29 由各臂自身地板校出来的值，**本发直接沿用、未由本发结果反推**）。

| 判据（EXPECTED §6 5.1） | 要求 | A4r(串行) vs B4(合版本) |
|---|---|---|
| prefill 内部对照 | 全同 | **187/187 = 100.0%** |
| decode 逐位 | 100% | **1174/1174 = 100.0%** |
| 非零 \|Δlogprob\| | 0 | **0**（7044 个条目，max 0.000e+00） |
| top-k key-set 差异 | 0 | **0** |
| chosen-token 差异（token 变化） | 0 | **0** |
| 退化步（top-5 全 = ln(1/129280) = −11.7697） | 0 | A4r **0/1265**、B4 **0/1265** |

⇒ **全过**。按 `EXPECTED_addendum_0936`（跨臂全同蕴含两臂各自也无非确定性；自比地板只用来解释「有差」），
**A5/B5 未跑**。截断点 13 未被改动，也未由跨臂结果反推。

必检（两臂）：`SIDE STREAM ENABLED` A4r **0** / B4 **40**；`ASYNC WAIT ENABLED` A4r **0** / B4 **40**；
`DEVICE SIGNAL` 两臂 0；ACL 错码 0；`never-ran` 0。
🔴 这一条必检不是形式：**没有它，「两臂逐位全同」完全可能只是 B 臂根本没开开关**。

### 3.1 🔴 A4（第一次跑的串行臂）作废 —— 与本线无关的另一个 agent 改了同一棵树
另一个 agent 于 09:48–09:52 在 **worktrees/moe-offload** 里直接改了（未提交）
`vllm_ascend/ops/fused_moe/{fused_moe,routed_experts}.py`、`vllm_ascend/kt_offload/{layer,config}.py`、`envs.py`，
而 **A4 于 09:50:30 起服务** ⇒ 可能加载了改到一半的代码。串行臂改名 **A4r** 重跑（树已恢复后）。

**靠指纹判、不靠时钟**（`build_fingerprint.json` 的 `trees.vllm_ascend`）：

| 臂 | HEAD | dirty_files | 读数可用？ |
|---|---|---|---|
| A1 | 7768480c9 | **3** | ✅（3 = 当时我自己的未跟踪脚本，**无已跟踪文件被改**） |
| B1 | ff4ffb920 | **3** | ✅ 同上 |
| **A4** | ff4ffb920 | **7** | 🔴 **作废**（已跟踪文件被改） |
| A4r | ba8e36484 | **4** | ✅（4 = 我的 4 个未跟踪脚本） |
| B4 | ba8e36484 | **4** | ✅ 与 A4r **同一棵树、同一 HEAD** |

**A1/B1 为什么不受影响**（两条独立证据，不是「时间上应该没事」）：
① 上表 dirty_files：A1/B1 = 3，只含我自己的未跟踪脚本；
② A1 与 B1 的 HEAD 确实不同（7768480c9 vs ff4ffb920），但
   `git diff --name-only 7768480c9 ff4ffb920` = **只有 `SESSION_STATE_0923.md`**；
   且 `git diff --name-only df94d6400 ff4ffb920 -- vllm_ascend/` = **0 个文件**
   ⇒ 本 session 全程 **运行时代码一行没动**，A1/B1 可比。
🔴 **教训**：本线的树被别的线当工作区用过。`seq_merge.sh` 之外，A4r/B4 起服务前都记了
`git status --short` 与 HEAD，并在发现**已跟踪文件被改**时拒绝起飞。

## 4. GSM8K

**GSM8K 50（只跑 B 臂 = 合版本，用户 09:56 决定；与 B4 的逐位共用同一次服务）**
口径同 §4.1：贪心 `--temperature 0 --top-p 1.0`、`EV_THINKING=0`、`--max-tokens 4096`、`--num-threads 8`、`--limit 50`。

| 臂 | 分数 | err_rate | 截断 | 墙钟 |
|---|---|---|---|---|
| **B4（合版本：侧流 + P1a）** | **50/50 = 1.0000** | 0.0 | 0.0 | 114 s |

⇒ 过 EXPECTED §6 5.2 的「B < 50 就停下报告」。A 臂本发按用户决定**未跑**
（§4.18 的同口径 A 臂历史值是 50/50，但那是别的发次，**只作背景、不作本发对照**）。

**GSM8K 1319：按用户 2026-09-23 09:56 决定暂缓**，等共享专家改动后合第二版再跑。
⇒ 合版本目前**没有** 1319 读数；§4.26 那个 **1275/1319 = 96.664%** 是**只含侧流**那一版的，
🔴 **不得拿它代表合版本**。

## 5. 仪器：`seq_merge.sh` 的 cfg 断言与它的变异臂

断言：`strip()` 把两份 cfg 的 `overlap_side_stream` 值抹成 `X` 后必须逐字节相同，
且 A 为 `false`、B 为 `true`。三个变异臂（`rounds=0` 干跑，不起服务，秒级）：

| 变异 | 期望 | 实测 |
|---|---|---|
| cfg_B 的 `cpuinfer_threads` 96→64（除那一行外还有别的差异） | 红 rc=2 | ✅ rc=2，并打印出那一行 diff |
| cfg_A 的 `overlap_side_stream` 写成 true | 红 rc=2 | ✅ rc=2 |
| cfg_B 的 `overlap_side_stream` 写成 false | 红 rc=2 | ✅ rc=2 |
| 正确的两份 cfg | 绿 rc=0 | ✅ rc=0 |

`ARM_OFFSET` 存在的理由只有一个：臂目录名就是读数的出处，重名 = 后一发静默覆盖前一发。
脚本另加一道 `[ -e $RUN/$ARM/arm_summary.txt ] && exit 2`。

## 6. 仪器：`perf_verdict.py`（三轮交错用）与 `perf_verdict_single.py`（本发用）

判据（PLAN §4.27）由脚本机械执行，输入固定为**每一臂每一轮**，不许剔除轮次：

```
Δ = mean(A 各轮) − mean(B 各轮)
floor = max( max(A)−min(A), max(B)−min(B) )      # 本发同臂跨轮最大摆动
收益成立 ⇔ |Δ| ≥ 3×floor 且 |Δ| ≥ 0.3 ms/token
```

读数行是 `decode ms/token(TPOT 中位)` 的均值（REPEAT 轮里每轮**中位**的均值），
**不是** `TPOT 均值 ms` 那一行（那是均值的均值、带尾巴；§4.30 记了 σ 的 3.09/4.74/7.27 离群）。
§4.28 / §4.29 引的都是中位那一行，换一行就与它们不可比。

**用历史读数校验过它会红也会绿**（不是只验绿的）：

| 喂进去 | 期望 | 实测 |
|---|---|---|
| §4.29 P1b：A 32.52/31.39 · B 31.32/30.89 | 判负（Δ0.85 < 3×1.13） | ✅ rc=1，复算出 Δ+0.850 / floor 1.130 |
| §4.28 P1a：A 32.36/32.51 · B 31.82/31.98 | 成立（Δ0.535 ≥ 3×0.16） | ✅ rc=0，复算出 Δ+0.535 / floor 0.160 |
| 少一臂（B2 缺 summary） | **报错**，不是判定 | ✅ rc=**3**（与判负的 rc=1 分开：「少一臂」和「判据判负」不是一回事） |

## 7. 仪器：每臂起飞前的占卡预检（`seq_merge.sh:card_clear`）

**为什么在每一臂前、而不是整段开头一次**：见 §2.1 —— 第一发开头预检全绿，外部租户是**跑到一半**上来的。
判的是**最大值与趋势**（连采 5 × 8 s，HBM 峰值 ≤8% 且 `last ≤ first+3`），不判某一瞬间；
不干净就等（最多 20 min），仍不干净则**拒绝起飞** rc=6。代价每臂约 40 s。

| 变异 | 期望 | 实测 |
|---|---|---|
| `PF_MAXPCT=0`（把阈值调到连 4% 空卡都不合格） | 红，且**拦住起飞** | ✅ rc=**6**，`seq.txt` 写 `STOPPED at A1 (preflight)`，服务从未起 |
| 默认阈值、空卡 | 绿 | ✅ rc=0（序列 `4/0 4/0 4/0`） |
| 卡号读不出来（喂 `-i 99`） | **当成不干净**（读不到 ≠ 空闲） | ✅ rc=1 |

🔴 `set -o pipefail` 是承重的：`card_clear ... | tee` 的 `||` 否则判的是 `tee` 的退出码，
而 `tee` 几乎永远成功 ⇒ **预检红了也会照样起飞，且看起来一切正常**。

## 8. 本发没有做的事（别当成做过了）

1. **三轮交错 A/B 没做完** ⇒ **本发没有测到同臂跨重启地板**，§4.27 判定**未执行**（见 §2.3）。
   要写进收益账，还欠一发预先声明门槛的三轮交错。
2. **GSM8K 1319 没跑**（用户 09:56 决定暂缓）⇒ 合版本没有 1319 读数。
3. **A 臂 GSM8K 50 没跑**（用户 09:56 决定只跑 B 臂）。
4. **A5/B5 自比地板没跑**（addendum：跨臂全同则不补）⇒ 本发的 `skip=13` 是**沿用** §4.29 的值，
   不是由本发地板重新校出来的。跨臂 100% 全同 ⇒ 没有差异需要解释，这个沿用不影响结论；
   但若将来某一发出现差异，**必须先补自比地板**，不许由跨臂结果反推截断点。
5. **没采 profile**（根盘余量 + 用户口径）。
