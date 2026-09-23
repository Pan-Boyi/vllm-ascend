# RUNBOOK：t4p 机制代理（`t4_proxy.py`）

> **一句话**：用 8–16 层的**设备计时**在 2–4 分钟里给一个机制报「预测 ms/token」，
> 用来**排序和筛选**候选；**进收益账的数永远是整网实测**。
> 判据先声明在 [`PROXY_CRITERIA.md`](PROXY_CRITERIA.md)（跑之前写死，代码常数 `t4_model.PROXY_REL_TOL/PROXY_MIN_CALIBRATED`）。

---

## 0. 五层验证，各自能证明什么

🔴 **第 1 层（t4p）只验性能，验不了精度；第 1 层的精度那半边是 t5r。** 用户 2026-09-23 确认的口径：

| 层 | 验什么 | 时间 | 能否作为合版本依据 |
|---|---|---|---|
| `t5r`（`run_oncard.sh <card> t5r`） | **正确性**（≥8 层、连续多步 replay 不 synchronize、`--probe`、逐位对串行参照 + 变异臂） | 3–5 min | ❌ 只作 go/no-go |
| **`t4p` 代理**（本文） | **性能** | 2–4 min | ❌ 只作排序/筛选 |
| 整网 logprob 逐位（`sidestream/run_arm.sh ... lp`） | 正确性 | ~20 min | ✅ |
| 整网 perf 交错（`seq_ab.sh` / `seq_p1a.sh` / `seq_p1b.sh`） | 性能 | ~50 min | ✅ |
| GSM8K 50 → 1319 | 精度验收 | 1 h+ | ✅ 合版本一次 |

🔴 **快速正确性层有已知盲区，用之前先读这条**：2026-09-22 侧流算错的那次，
**单层 9/9 全绿也没抓住**（§4.23），直到加了 `--stress ids-delay`（在**主流**插一段延时，
把「ids 先到还是 fork 先放行」从抛硬币变成**决定**）才复现（§4.25）。
⇒ 规矩：**每引入一个新机制，必须给 t5r 补一条针对它的 stress 或变异臂**
（侧流→`ids-delay`/`early-fork`；P1a→`early-record`/`no-wait`/`no-reset`+`--stall-ms`；
P1b→`signal-first`/`early-record`/`no-wait`+`--stall-ms`）。
**没有对应臂的机制，t5r 的绿不计入 go/no-go** —— 它只证明了「没红」，没证明它**能**红。

---

## 1. 怎么跑

```bash
cd /workspace/code/dsv4_single_card/vllm_project
WT=worktrees/moe-offload/tools/kt_offload/oncard
CARD=2                      # 卡 1 不用；卡 3 不许；卡 0 常被整网 A/B 占着 —— 先看 card_lease.sh status 与 npu-smi

# P1a 那一组（serial / fork / 空机制 fork2 / p1a）
bash $WT/../oncard/run_oncard.sh $CARD t4p --arms serial,fork,fork2,p1a \
     --layers 16 --reps 60 --measured-sidestream -3.00 --measured-p1a -0.53

# P1b 那一组（p1a 与 p1b 不能同进程：它们共用 self._ext_events，工具会拒绝）
bash $WT/../oncard/run_oncard.sh $CARD t4p --arms serial,fork,fork2,p1b \
     --layers 16 --reps 60 --measured-sidestream -3.00
```

- 租约、卡就绪门（HBM+aicore **连采 3 次**）、dmesg 门、错码扫描（`ONCARD_ERRCODES` 五码 + 非数字边界 + 排掉埋点行）
  全部由 `run_oncard.sh` 做，**不要绕过它直接 `python3 t4_proxy.py`**（那样没有租约、没有 provenance、没有异常扫描）。
- 产物：`build_artifacts/moe_offload_ref/oncard_runs/<stamp>_t4p/{test.log,t4p_result.json,provenance.json,...}`。
- **发间离散度**：同一组连跑两发，第二发加 `--prev <第一发>/t4p_result.json`，它会把两发的预测并排打印。
- 不采 profile。只有当代理与整网实测**严重矛盾**、非看 device 时间线不能解释时才采，并在报告里写明为什么非看不可。

## 2. 怎么读

```
  DEV serial   median 3855.9 us ... per-layer 482.0 us  host-ahead 100%
  DEV fork     median 3062.6 us ... per-layer 382.8 us  host-ahead 100%
  预测（代理）：
    sidestream  fork - serial   -100.5 us/layer (p10 -119.9, p90 -76.7)  => -4.02 ms/token  resolved
    p1a         p1a  - fork       -1.4 us/layer (p10  -19.5, p90 +22.0)  => -0.05 ms/token  🔴 NOT RESOLVED
    null        fork2- fork       +0.1 us/layer                          => +0.00 ms/token
```

- `DEV <arm>`：一次 replay 整个 `--layers` 层栈的**设备** µs（不是墙钟）。`per-layer` 是它除以层数。
- 预测 = **配对**（同一轮内每臂各一次）逐轮差 ÷ 层数 × `--model-layers` ÷ 1000。**负数 = 省**。
- `resolved` / `NOT RESOLVED`：配对差的 p10/p90 是否与中位同侧于 0（判据 G1）。
  **NOT RESOLVED 的机制，中位数不许引** —— 它是一枚带小数点的硬币。
- `null`（`fork2 - fork`，同一配置捕获两次）＝**代理自己的地板**。整网读数都要同臂地板，代理没有理由例外。
- `PROXY USABLE: usable | ranking_only | unusable` —— 见 `PROXY_CRITERIA.md` §3，由 `t4_model.proxy_verdict()` 判。
- 两条对照任一红 ⇒ **一个预测值都不打印**。看到 `🔴 a control is RED` 就当这一发没有读数。

**两条对照的正常读数**（本文落地那几发，卡 2、32 线程、8–16 层）：

| 对照 | 判据 | 实测 |
|---|---|---|
| 阴性：`g_npu_only`（无 kt 无节点）vs eager 同样工作 | 比值 1±0.35 | **0.98**（07:12:54 那发） |
| 阳性：每层一个 500 µs 胖节点 | 每层 ≥300 µs | **559.4 µs/层** |
| host 跑在 device 前面 | 每臂 ≥90% | **100%**（每一臂） |

## 3. 标定（2026-09-23，卡 2，`--threads 32 --numa 1 --n 100 --tokens 1`）

判据原文在 `PROXY_CRITERIA.md`（起飞前写死，跑完照它判，**没有改过**）。
产物在 `build_artifacts/moe_offload_ref/oncard_runs/<stamp>_t4p/`【未跟踪，读数内联】。

**🔴 前两发的仪器是红的，读数作废，但照规矩记在这里**（`PROXY_CRITERIA` §5.3：不许把调过的那一发当第一发）：

| # | stamp | 层×轮 | 结果 |
|---|---|---|---|
| 1 | 07:08:08 | 8×24 | ❌ **仪器红**：eager 对照臂 host-ahead **12%** —— t4 那个写死 ~5 ms 的 blocker 是按**一层**的 eager 臂定的，8 层的 eager 链光下发就比它长。修法：blocker 按**实测**的最慢臂 host 下发时间定。 |
| 2 | 07:11:23 | 8×60 | ❌ **仪器红**：host-ahead **42%**。第二个坑：blocker 的单价是用「下发 10 个 mm + synchronize」的**墙钟**算的（589 µs/个），而设备实际只花 **54.5 µs/个** ⇒ 以为铺了 8.2 ms，其实不到 1 ms。修法：blocker 单价改用 `Event(enable_timing)` 量**设备**时间。 |

**下面三发两条对照全绿（判据 G0 全过），是可用的读数**：

| # | stamp | 层×轮 | 阴性对照 | 阳性对照/层 | 侧流预测 | P1a 预测 | P1b 预测 | 代理地板(null) |
|---|---|---|---|---|---|---|---|---|
| 3 | 07:12:54 | 8×60 | 0.98 | 559.4 µs | **−4.02** | −0.05 🔴未分辨 | — | +0.00 |
| 4 | 07:14:30 | 16×60 | 0.99 | 562.8 µs | **−4.06** | −0.41 🔴未分辨 | — | −0.09 |
| 5 | 07:16:36 | 16×60 | 1.02 | 560.0 µs | **−4.15** | — | **−0.88** ✅分辨 | −0.12 |

**标定表**：

| 机制 | 整网实测（同臂地板） | 代理预测 | 相对误差 | 判定 |
|---|---|---|---|---|
| 侧流 vs 串行 | **−3.00 ms/token**（地板 0.71，§4.25） | **−4.02 / −4.06 / −4.15** | +34% / +35% / +38% | ✅ 同号且 ≤50% ⇒ **CALIBRATED** |
| P1a vs 侧流 | **−0.53 ms/token**（地板 0.16，§4.28） | −0.05 / −0.41，**两发都未分辨** | — | 🔴 **代理分辨不了** |
| P1b vs 侧流 | 见 §3.1 | −0.88（已分辨） | 见 §3.1 | 见 §3.1 |

**标定结论（照 `PROXY_CRITERIA.md` 判，判据没动过）**：

1. **偏差是系统性的、单向的**：三发对侧流全部**偏高** +34%…+38%，中心约 **+35%**。
   量级上说得通：代理栈里除了 MoE 什么都没有，`min(kt, window)` 的 window 几乎全部可藏；
   整网里同一层还有别的 kernel 和调度，能藏的没那么满。
   ⇒ **可以按「代理偏高约 1/3」读方向，但不许拿 1/1.35 去「校正」代理值**——
   那是拿一个点拟合出来的因子，不是量出来的。
2. **发间离散度极小**：三发侧流预测 −4.02 / −4.06 / −4.15，极差 **0.13 ms/token（3.2%）**。
   ⇒ 代理**可复现**；不可靠的地方不在重复性，在外推。
3. **分辨率就是代理的天花板**：8 层与 16 层、60 轮下，逐轮配对差的 p10/p90 都在 **±1 ms/token** 量级
   （`null` 臂：±0.94 / ±1.75 / ±1.06）。
   ⇒ **代理分辨得了 −3 ms 量级的机制；−0.5 ms 量级的（P1a）两发都判「未分辨」**，
   而它的整网实测 −0.53 是真的 ⇒ 🔴 **代理说「没看出来」时，绝不能读成「没有收益」。**
4. **可信范围（一句话）**：
   > 对 **|Δ| ≳ 2 ms/token** 的机制：方向可信、量级可信到 **±40%**；
   > 对 **|Δ| ≲ 1 ms/token** 的机制：**只有「已分辨」那一栏可信**，未分辨就是没有答案。
5. 按判据 `proxy_verdict()`：可标定的机制只有侧流一个（P1a 未分辨、P1b 当时没实测）
   ⇒ **`unusable`（证据不足，不是「代理错了」）**。这正是 `PROXY_MIN_CALIBRATED=2` 想挡的东西：
   **一个点标不出一条线。**

### 3.1 P1b 这一栏

- 2026-09-23 05:09–06:01 的四臂 perf（`oncard_runs/20260923_044930_p1b`）：
  A（P1b 关）32.52 / 31.39，B（P1b 开）31.32 / 30.89 ⇒ A−B 中位 **0.85**（两轮同向 1.20 / 0.50），
  但**同臂跨轮地板 1.13**（A 臂自己；那一小时机器在变空，A1 32.52 → B2 30.89 单调下行）。
  按 §4.27（≥3× 地板 且 ≥0.3 ms）⇒ **判 fail —— 不是没效果，是没尺子。**
  🔴 **这个 0.85 不作为标定点**：拿一个自己都判不出收益的数去标定代理，只会把噪声标进去。
- 逐位已过（各臂自身地板把截断点校到 13 之后六对 100%）。
- 代理对 P1b 的预测是 **−0.88 ms/token（已分辨）**。两个数贴得很近是**旁证不是证据**，
  在整网 perf 用降了地板的口径重测出来之前，**这一栏留空**。

🔴 **两点标定的局限，明写**：现在真正能用的标定点只有侧流（−3.00）一个，
P1a 未分辨、P1b 没尺子。**两点已经少，一点是不能外推的**。
每多一个整网实测落地，就把它补进这张表，并重跑一发代理对齐。

## 4. 🔴 什么时候**不能**用代理

按「这个机制改的是什么」分：

1. **改的是 host 侧的串行链，不是设备时间线。**
   代理量的是「设备在这段图上花了多少 µs」。一个机制如果省的是 Python 帧、GIL 等待、
   线程握手、调度器唤醒，而设备时间线上那段等待的长度不变，**代理会报 ≈0 并且是错的**。
   例：§4.20 的 C 段（kt 的 Python 前奏 0.79 ms/token）、§4.22 F4（kt TaskQueue 线程握手 10.4 µs/层）。
   判别法：问「这个改动会不会让 `NOTIFY_WAIT`/等待节点在设备上变短」。答不出来就别用代理。

2. **臂之间图结构不同 ⇒ 捕获/回放开销本身变了。**
   代理是「同一个栈、不同捕获」的差。一个机制如果增删了图**节点数量级**（不是一两个节点），
   replay 的固定开销跟着变，差值里就混进了「图变大了」而不是「机制生效了」。
   P1b 是边缘案例（去掉 40 个 host-func 节点、加 40 个 D2H + wait/reset）——
   代理对它的数**只当排序用**。

3. **机制的收益依赖整网才有的东西。**
   代理栈里没有 attention、没有别的层的 kernel、没有调度器、没有真实的 batch 混合。
   依赖「和别的 kernel 抢 AI Core」「和别的层的 D2H 抢带宽」「prefill/decode 混跑」的机制，
   代理测不到。同理：它只有 `--tokens 1` 的 decode 形态，**prefill 的机制不要用它**。

4. **机制的量级低于本发的代理地板 / 没被分辨。**
   本发实测：8 层 60 轮时，逐轮配对差的 p10/p90 约 ±20 µs/层 ≈ **±0.9 ms/token**。
   ⇒ **代理分辨得了 −3 ms/token 量级的机制，分辨不了 −0.5 ms/token 量级的**。
   小于 ~1 ms 的机制，代理的回答只能是「没看出来」，**不能读成「没有收益」**。
   （P1a 整网实测 −0.53 ms 是真的，代理报「未分辨」—— 这正是这条的现成例子。）

5. **机制会改变数值。**
   代理**一个数值结果都不验**。本条路上每一个已知的失效形态都是「**更快且算错**」，
   在代理眼里全部长得像赢。正确性只由 t5r（带针对性变异臂）与整网逐位回答。

⇒ 总结成一句写进执行单：
**代理只用于排序与筛选；进收益账的数必须是整网实测（且按 §4.27 与同臂地板一起判）。**

## 5. perf 参数能不能从 `16 条 × 256 token × 3 轮` 砍到 `8 条 × 128 token × 3 轮`

**结论：不许当默认口径；只在筛选 ≥3 ms 量级的机制时可用，且必须和地板一起报。**
（口径：`CASES="bs1 1 8" OUT_LEN=128 REPEAT=3`，对应 `build_artifacts/perf_baseline.sh` 的
`CASES`/`OUT_LEN`/`REPEAT` 三个环境变量。省约 6.5 min/臂、4 臂 ≈ 26 min。）

**🔴 未上卡验证**。经过（照实记）：
- 07:17:58 **判据先写死**在 `oncard_runs/20260923_071758_perfcal/EXPECTED.txt`
  （顺序 old1→new1→new2→old2，两个 new 夹在两个 old 中间，好让两种口径的时间重心对齐 ——
  这一小时机器在单调变空；判据 ① 口径差 < 同发同臂地板、② 新口径请求间 σ ≤ 2× 老口径）。
- 07:18 起服务（卡 0，侧流开）；07:23:05 ready。
- **一次 perf 都没跑**：用户 07:2x 改序——先采 P1b 的 profile 回答「那 110 µs/层到底省没省」，
  口径验证往后排 ⇒ 07:23 放行 hold、收服务，把卡让给 profile 那一发。
  `EXPECTED.txt` 原样留着（判据仍然有效，**下次接着用同一份，不重写**）。

下面是**从已有 perf 产物做的分析估计**，不是实测。

已有产物（`oncard_runs/20260923_044930_p1b/*/perf/bs1_r*.json`，`oncard_runs/20260923_023718_p1a/*`）：

| 量 | 读数 |
|---|---|
| 每发 `num_prompts` / `total_output_tokens` | 16 / 4096（= 16 × 256）✅ 确认就是这个口径 |
| 请求间 TPOT 标准差 `std_tpot_ms` | 典型 **0.4–1.1 ms**，但有重尾：8 臂 24 发里出现过 **3.09 / 4.74 / 7.27** |
| 同臂跨轮中位极差（3 轮） | **0.11 – 1.06 ms**（p1b 那发 A 臂 0.11/B 臂 1.06；p1a 那发 0.21–0.75） |

推算：中位的采样标准误 ≈ `1.253 × σ / √n`。
- 现口径 n=16、σ≈0.7 ⇒ **≈0.22 ms**。
- 砍半后 n=8，且每条请求的 TPOT 从 256 个 token 的均值变成 128 个 ⇒ σ 约 ×√2 ≈ 1.0
  ⇒ **≈0.44 ms**。**采样那一份噪声翻倍**（重尾下更糟：中位取自 8 个样本，一个 4.7 ms 的离群点权重大得多）。

判读：
- 现有的同臂地板已经在 **0.11–1.06 ms** 之间摆，而 §4.27 的门槛是「≥3× 地板 **且** ≥0.3 ms」。
  地板再加 0.2–0.4 ms，**P1a 量级（0.53 ms）的机制会直接变成测不出来**；P1b 那发地板 1.13 已经把 0.85 的差吃掉了。
- ⇒ **默认口径不动**。砍半只允许用在「筛掉明显没用的候选」这一步，且报告里必须写清用了砍半口径。
- ⇒ 真正省卡时的办法是**先用 t4p 代理筛**（2–4 min，见 §0 那张表），而不是把尺子改短。

**要验的时候怎么验**（有空档就照这个跑，一发即可）：
```bash
# 同一臂、同一次起服务，先老口径后新口径，中间不停服务
D=$RUN/A1
PORT=$PORT LABEL=A1_old CASES="bs1 1 16" OUT_LEN=256 REPEAT=3 PERF_OUT=$D/perf_old bash build_artifacts/perf_baseline.sh
PORT=$PORT LABEL=A1_new CASES="bs1 1 8"  OUT_LEN=128 REPEAT=3 PERF_OUT=$D/perf_new bash build_artifacts/perf_baseline.sh
# 再跑一遍老口径，拿【同一发同一臂】的跨轮地板
PORT=$PORT LABEL=A1_old2 CASES="bs1 1 16" OUT_LEN=256 REPEAT=3 PERF_OUT=$D/perf_old2 bash build_artifacts/perf_baseline.sh
```
判据（**先写死**）：`|median(new) − median(old)| < 地板`，其中地板 = `|median(old2) − median(old)|`；
并且 `新口径自己的跨轮极差 ≤ 老口径的 1.5 倍`。两条都过才算「砍半安全」。
一条不过 ⇒ 维持现口径，把这一节的「未验证」改成「验过、不安全」。

## 6. 卡上纪律（与其它 RUNBOOK 一致）

- 起飞前 HBM + aicore **连采 ≥3 次**（`run_oncard.sh` 的 `pa_sample_card` 已经做，读 `card 2 state=free (...)` 那行确认）。
- 脚本自取租约（`card_lease.sh acquire`，`LEASE_PID=$$`）；**卡 1 不用、卡 3 不许**；卡 0 常被整网 A/B 占着，先 `status`。
- 停服务用 `stop_by_port.sh` 带 `STOP_PGID`；**不用 `pkill -f`**、不用裸 `stash`。
- 就绪失败判据只认 `^Traceback|EngineCore failed|进程退出|超时`。
- 错码用 `oncard_lib.sh` 的 `ONCARD_ERRCODES` / `ONCARD_ERRCODE_RE`（显式五码 + 非数字边界 + 排除埋点行）。
- **不采 profile**，除非代理与整网实测严重矛盾且非 device 时间线不能解释 —— 那时在报告里写明理由。
