# RUNBOOK — P1b：图里不再有 host-func 节点，启动信号是一次 8 字节计数器 D2H

> 开关：`VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL=1`（**纯 env**，默认关）。
> 前置：`packed_transfer` 或 `overlap_side_stream` 之一为 true；graph 模式；
> **与 `VLLM_ASCEND_DSV41_KT_ASYNC_WAIT`（P1a）互斥**，与 `VLLM_ASCEND_DSV41_KT_SEAM_TIMING` 互斥
> （四条都在 config 里起服务就 raise）。
> 代码：`vllm_ascend/kt_offload/device_signal.py`（协议 + 轮询线程 + 判据）、
> `vllm_ascend/kt_offload/poll_native.c`（唯一一段 C：不持 GIL 的自旋）、
> `layer.py:_launch_signal_node` / `_poll_run` / `_SignalWords` / `_StepCounter` / `_selftest_signal_path`。

## 0. 这条路在改什么

P1a：`D2H → host-func[ kt_pre | submit ]（立即返回） → ExternalEvent.wait → reset → H2D`，
省掉的是 F 段（回调返回 → 通知 device），实测 **0.53 ms/token**（§4.27）。
还留在关键路径上的两项都在**节点之前**，所以 P1a 碰不到：

- **A 段**：device 走到节点 → 回调被派发（报告送达 + 抢 GIL + 建参数元组）；
- 🔴 **每个 `HOSTFUNC_CALLBACK` 之前 35.75 µs 的设备纯空转**（§4.22 F1）。同一发 profile 里
  `DAVID_EVENT_WAIT` / `MEM_WAIT_VALUE` / `MEMCPY_ASYNC` / `EVENT_RECORD` 的前置空隙**全是 0.000 µs**
  ⇒ 空转是 host-func **这个节点类型独有**的，只有把节点拆掉才吃得到。

P1b：

```
D2H(输入)  →  D2H(8 字节步号计数器)  →  ExternalEvent.wait  →  reset  →  H2D
```

图里一个 host 任务都不剩；host 侧一条**常驻轮询线程**看到本步步号就跑 kt，再 `record` 那个 event。
计数器是一个 device 标量，**每步只加一次**（本路径上最低的那层加），所以每层的字里必须是同一个步号。

🔴 **顺序就是安全性的全部**：计数器是**同一条流上、排在输入 copy 之后的另一次 copy**。同流串行
⇒「计数器到了」蕴含「输入到了」。**绝不能**把步号塞进同一次 copy 的尾部 —— 单次 DMA 内部的写顺序
平台没有承诺。§4.25 的铁律照旧：轮询线程读的每一样东西都必须在计数器写之前写完。

**为什么自旋要用 C**：Python 线程轮询 word 会在两次检查之间持有 GIL，主线程最多等一个
`switchinterval`（默认 5 ms）。本机实测（两个不同物理核）：主线程独跑 178 ms / C 轮询在旁 164 ms /
**Python 轮询在旁 1024 ms（5.8×）** / 睡眠线程 162 ms；C 自旋看到另一线程的写中位 **0.52 µs**（p90 0.60），
两端时间戳都取在 C 里。**没有放进 kt**：那要重建 wheel + 换站点目录 + 重跑 t0/t1/t2，而这段自旋不需要
kt 的任何东西。起服务时**编译一次 .so**（源码 hash 入文件名），落到 `~/.cache` 或 `/tmp`——
`/dev/shm` 常是 `noexec`，而那只在 `dlopen` 才报，所以候选目录是一个个「编译+加载+调用」试过去的。

## 1. 它会怎么错

🔴 **主失效形态仍然是：放行早 = 更快且算错，且不报错。** 判据一律是**与开关关着的那一臂逐位相同**，
不是 TPOT。P1b 比 P1a 多一条独有的失效形态，而且没有第二道防线：

| 怎么错 | 症状 | 谁能抓住 |
|---|---|---|
| 计数器 copy 排到了输入 copy **前面** | kt 读上一步的路由；更快、算错、不报错 | 进程内判据 `signal_ops_ok`（捕获期 raise）+ 卡上 `--mutant signal-first` |
| 提前 record（kt 还没跑完） | H2D 拷到上一步的 CPU 输出 | 逐位 + `--mutant early-record` |
| `ExternalEvent.wait` 节点没进图 | 同上 | 逐位 + `--mutant no-wait` |
| host-func 节点又回来了 | 这一臂量的其实是 P1a | `signal_ops_ok` + t5r 的 `NO host-func node ran` |
| 每步加了两次 / 没加 | 每层的字都超前一步 / 永远不动 | 轮询线程的步号比对（会记 ERROR）+ 捕获期的「只有最低层加」检查 |
| 信号丢了 | **挂住**（不是算错） | 轮询线程每 30 s 打一条点名日志；🔴 **它绝不会超时放行** —— 多 record 一次会把下一次 replay 的 wait 预先点亮，把看得见的挂住变成之后每一步的静默陈旧输出 |

**仪器口径变了一处**：P1b 开着时 `HOST_FUNC_CALLS` 永远是 0，而 `host_func_report` 对「什么都没长」
一个字都不打 ⇒ **老口径会让这一臂看起来最干净**。采样器改采 `device_signal.STEPS`，日志行名变成
`kt_offload device-signal over ...`；`run_arm.sh` 两个行名都收，并在一行都没有时明说「没被看过」。

## 2. 先做 t5r（不起服务，一发约 4 分钟）

```bash
WT=worktrees/moe-offload/tools/kt_offload/oncard
# 干净臂（rc 仍是 1：Verdict 要求每发至少一个变异臂；按配对发判读）
bash $WT/run_oncard.sh $CARD t5r --arms serial,fork,p1b --layers 8 --steps 8
# 🔴 这一条是 P1b 独有的、也是最该看的
bash $WT/run_oncard.sh $CARD t5r --arms serial,fork,p1b --mutant signal-first --layers 8 --steps 8
# 共用的两条 + 阳性对照（把每次 record 压后 50 ms）
bash $WT/run_oncard.sh $CARD t5r --arms serial,fork,p1b --mutant early-record --layers 8 --steps 8 --stall-ms 50
bash $WT/run_oncard.sh $CARD t5r --arms serial,fork,p1b --mutant no-wait      --layers 8 --steps 8 --stall-ms 50
# 探针发（读回 pinned 验是不是本步的值；探针加 host 工作，所以与判决发分开跑）
bash $WT/run_oncard.sh $CARD t5r --arms serial,fork,p1b --probe --layers 8 --steps 8
```

判读：

- `serial` / `fork` / `p1b` 三臂都必须 **8/8 逐位等于 eager 串行参照**；`p1b` 臂还要
  `host_func_per_layer` **全 0**（那才叫 P1b）、`device_signal_per_layer` 每层 = steps、
  `ext_records_per_layer` 与之逐层相等、`poller_errors` 为空。
- **阳性对照**：`--stall-ms 50` 下 `p1b` 臂的 8 次 replay 必须 ≈ `steps × layers × 50 ms`；
  同发的 `fork` 臂只要 ~0.03 s。🔴 **差不出来就说明设备根本没停在那个 event 上，
  本发所有「逐位相同」都不算证据。**
- **变异臂必须红**，且 `nodes_skipped` 要证明变异**真的发生了**（`signal_first` = 层数、
  `wait` = 层数）——「安静地什么也没改」和「判据没有区分力」从 verdict 行上长得一模一样。
- 🔴 `--mutant signal-first` 故意**按生产顺序记 ops**，好让进程内判据不拦住它：这一臂要的是
  **卡上的数字**（证明两次 copy 真的是被流定序的，而不是碰巧对），判据自己的红臂在 CPU 单测里
  （`tests/ut/kt_offload/test_device_signal.py::test_the_counter_copy_must_be_behind_the_input_copy`）。
- 🔴 **「完全不 record ⇒ 挂住」这一臂不在卡上跑**：它会把设备永久留在等待里，下一臂就没卡了。
  那一臂是 CPU 单测 `test_side_stream_overlap.py::test_p1b_without_the_record_hangs`。

### 2026-09-23 首发读数（卡 0，8 层 × 8 replay；产物未跟踪 ⇒ 读数内联）

| 发次 | 结果 |
|---|---|
| 干净（`044737`） | serial / fork / **p1b 各 8/8 逐位等于 eager 串行**；p1b 臂 host-func 计数**全 0**、每层信号 8 次、signals=records、轮询线程 0 错；replay 墙钟 serial 0.035 s / fork 0.027 s / **p1b 0.024 s** |
| `--mutant early-record --stall-ms 50`（`044955`） | 变异臂 **0/8** ⇒ RED ✅ |
| `--mutant no-wait --stall-ms 50`（`045045`） | 变异臂 **1/8** ⇒ RED ✅，`nodes_skipped.wait = 8`（变异真的发生了） |
| **阳性对照**（同上两发） | 每次 record 压后 50 ms ⇒ **干净 p1b 臂 8 次 replay 用了 3.23 s**（= 8×8×50 ms，下限 2.24 s），同发 fork 臂 0.026 s ⇒ **设备确实停在那个 ExternalEvent 上** |
| `--mutant signal-first`（`044908`，**无放大器**） | 🔴 **绿 8/8** —— 且 `nodes_skipped.signal_first = 8`，变异确实发生了 |
| `--mutant signal-first`（`045213`，**带放大器**） | **0/8** ⇒ RED ✅ |

🔴 **把计数器 copy 排到前面并不会当场算错 —— 它只是开了一场竞态。** 输入 copy 就排在它后面、
几微秒就完成，而轮询线程要先被唤醒（0.5 µs）、抢 GIL、建 kt 的任务（~15 µs）才会去读 pinned，
所以正常情况下**数据先到**、这一臂看起来完全正常。这与侧流那 3.1% 整步算错是同一个形状（§4.23）。
⇒ 这一臂里在两次 copy 之间插了一段设备计算（4×16 MiB 的 add，~100 µs）把竞态判给轮询线程，
和侧流用 `--recycle inplace` / `--stress ids-delay` 是同一手法。
⇒ 🔴 **那一发「绿」本身才是要记住的东西**：顺序必须**靠构造**是对的，不能靠一发测试绿。

## 3. 整网 A/B（perf 主、逐位次）

🔴 **两臂都开侧流**，只差 `VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL` 一个开关。
🔴 **A 臂是现在的默认形态（host-func 节点还在），不是 P1a** —— 这一发要回答「把节点整个拆掉值多少」。

```bash
RUN=$PWD/oncard_runs/$(date +%Y%m%d_%H%M%S)_p1b; mkdir -p $RUN
cat > $RUN/cfg_A.json <<'JSON'
{"kt_offload_config":{"enabled":true,"resident_strategy":"prefix","num_resident_experts":100,
 "cpuinfer_threads":96,"threadpool_count":2,"numa_nodes":[0,1],"packed_transfer":false,
 "overlap_side_stream":true}}
JSON
cp $RUN/cfg_A.json $RUN/cfg_B.json
bash $WT/sidestream/seq_p1b.sh $RUN $CARD perf   # A1,B1,A2,B2 交错，每臂 CASES="bs1 1 16" REPEAT=3
```

必检（`run_arm.sh` 自己做，对不上 = 这一臂作废）：
`SIDE STREAM ENABLED` 两臂都 40 条；`DEVICE SIGNAL ENABLED` **A 臂 0 条、B 臂 40 条**；
`ASYNC WAIT ENABLED` 两臂都 0 条；ACL 错码 0（显式五码 + **非数字边界** + 排掉我们自己的埋点行）；
每层采样行不为 0 条。

判读（**PLAN §4.27，先声明后验**）：
> **收益成立 ⇔ |A−B| ≥ 3 × 同臂地板 且 |A−B| ≥ 0.3 ms/token**，地板＝**本发**同臂跨轮最大摆动。
> 只有一轮 ⇒ 没有地板 ⇒ 判 fail。🔴 跨发不得相减（§4.25：绝对值跨发差 ~1.5 ms）。

### 2026-09-23 首发整网 perf（卡 0，`oncard_runs/20260923_044930_p1b`，产物未跟踪 ⇒ 读数内联）

四臂交错 A1→B1→A2→B2，两臂 cfg 逐字节相同、都开侧流，只差 `VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL`；
每臂必检全绿（`SIDE STREAM ENABLED` 40、`DEVICE SIGNAL ENABLED` A 0 / B 40、`ASYNC WAIT` 0、
ACL 错码 0、每层采样行 A 臂 21 条 host-func / B 臂 20 条 device-signal，never-ran 0）。

| 臂 | bs1 decode ms/token（中位，n=3） |
|---|---|
| A1（P1b 关） | **32.52**（sd 0.06） |
| B1（P1b 开） | **31.32**（sd 0.56） |
| A2（P1b 关） | **31.39**（sd 0.33） |
| B2（P1b 开） | **30.89**（sd 0.53） |

- A 均值 31.955 / B 均值 31.105 ⇒ **A−B = +0.85 ms/token**（P1b 更快，两轮同向：1.20 / 0.50）
- **本发地板**＝同臂跨轮最大摆动 = max(A 1.13, B 0.43) = **1.13 ms**
- 判据（§4.27）：需 |A−B| ≥ 3×1.13 = 3.39 且 ≥ 0.3 ⇒ 🔴 **不成立，判 fail**

🔴 **不许看到结果再放宽**。这一小时里机器在变空（A1 32.52 → B2 30.89 单调下行，外部租户的负载在退），
A 臂自己就摆了 1.13 ms —— 比要判的差值还大。要拿到结论只能**降地板**：机器安静时多跑几轮交错
（4 轮以上），而不是改门槛。

逐位（perf 有结论之后再排，两臂各一发）：

```bash
python3 $WT/sidestream/make_prompts.py $RUN/prompts.jsonl <新种子>   # 位置参数，不是 --out
SIDE=1 P1B=0 bash $WT/sidestream/run_arm.sh $RUN A3 $CARD lp
SIDE=1 P1B=1 bash $WT/sidestream/run_arm.sh $RUN B3 $CARD lp
python3 $WT/sidestream/logprob_cmp.py $RUN/A3/lp.jsonl $RUN/B3/lp.jsonl "A3 vs B3" --skip-first=12
```

`--skip-first=12` **按臂校准**，不许拿结果反推截断点（§4.25/§4.26）。
⚠ 两臂各一发**没有同臂地板**：逐位全同不需要地板；一旦出现差异，**先补一发同臂算跨进程地板再判**。

### 2026-09-23 首发逐位（同一个 RUN，四臂 A3/A4/B3/B4，各 200 条 prompt / seed 20260923119）

第一发（A3 vs B3，`--skip-first=12`）**不是逐位相同**：prefill 187/188、decode 1173/1180，
非零 |Δlogprob| 6 / 37，max 1.125，top-k key-set 差 5，chosen-token 差 **0**。
按 RUNBOOK 的规矩**没有就此下结论**，而是先补地板：

| 对 | `--skip-first=12` | 无截断时有差异的请求号 |
|---|---|---|
| A3 vs A4（P1b **关** 的同臂地板） | **100%** 逐位相同 | 4, 5, 7, 8 |
| B3 vs B4（P1b **开** 的同臂地板） | 99.4%（decode 1173/1180） | 1, 2, 3, 4, **9, 10, 11, 12** |
| A4 vs B4（关 vs 开） | **100%** 逐位相同 | 1–8 |
| A3 vs B3 / A4 vs B3（关 vs 开） | 99.4% | …, 12 |

⇒ **所有差异都落在同一个请求（#12）上，而且 B 臂拿自己比自己也一样差在 #12。**
这正是 `logprob_cmp.py` 注释里写的那个坑：截断点**按臂校准**，而 P1b 这条路的热身尾巴比
关着的那一臂长一格（A 臂的自比地板到 #8 为止，B 臂到 #12）。

**按各臂自己的地板把截断点校准到 13** 之后，**六对全部 100% 逐位相同**（decode 1173/1173，
非零 |Δlogprob| **0**，top-k 差 **0**，chosen-token 差 **0**）——两条地板和四条交叉全在内。
🔴 这个 13 是从**B 臂自己的地板**（B3 vs B4 差到 #12）读出来的，不是从 A/B 结果反推的；
而且在旧截断 12 下就已经有一对关/开（A4 vs B4）是 100%。

⚠ 顺带：t5r 的 `--probe` 发（`20260923_070607_t5r`）三臂各 **64/64** 次读，
每一次 pinned 输入都是本步的值 —— 这是「轮询线程读到的确实是本步路由」的直接证据。

## 4. profile：默认不采

用户 2026-09-23 明确：**默认只跑 perf + 精度判据，profile 太慢 ⇒ 非必要不采**。
只有在 perf 与逐位互相矛盾、或者必须证明「`HOSTFUNC_CALLBACK` 节点数确实变成 0 / 35.75 µs 空转确实消失」
时才加一发，并在报告里写清为什么非采不可。
（真要看时，P1b 的 device 侧指纹是：`HOSTFUNC_CALLBACK` **0 个**、`NOTIFY_WAIT` 0 个、
每层多一次 8 字节 `MEMCPY_ASYNC` 和一对 `MEM_WAIT_VALUE`，且这些节点前置空隙应为 0。）

## 5. 没有改 kt

P1b **不需要动 ktransformers**：拆 submit/sync 早就在我们这边（`_poll_run` 自己
`moe.forward_task` + `cpu_infer.submit/sync`），自旋是本树自带的一段 C。
⇒ **不换站点目录、不重建 wheel、不用重跑 t0/t1/t2**，`tools/kt_offload/kt_build.env` 一个字没动。
（P2 的 `KT_CPUINFER_INLINE` 要 kt 侧那次重建；两件事互不依赖，见协调方 2026-09-23 的安排。）
