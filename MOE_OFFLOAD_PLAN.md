# DSV4.1-flash 单卡 MoE offload（vLLM + kt-kernel）计划与跟踪

> 👉 **2026-09-22 交接**：接手请先读 **`HANDOFF_next_session.md`**（要做什么、rebase 怎么做、风险在哪、优化队列的优先级）。用户已拍定：**做 rebase（不重新分叉）到 clean `f85267ab8`**，且**先不动** —— 由新 session 从 rebase 开始。

> ⏸ 2026-09-20 再次暂停（用户：多卡优化优先，单卡先停，卡 3 收回；本线从未上过卡、未持租约）。不占卡的准备已完成。

> ▶ 2026-09-19 恢复（多卡 fix 已合入 clean 1b1ad396b / 8b24cc594e）：不占卡部分已完成，上卡待用户确认 + flash_dev 排卡。

> 本文是这条线的**唯一跟踪文档**，由 dsv41_vllm_singlecard 会话维护。
> 状态格式：⬜ 未开始 · 🟡 进行中 · ✅ 完成(附产物/判据读数) · ⛔ 阻塞(附原因)。
> 每次更新在文末「变更记录」加一行，**带日期**。数字带出处；推断/估算明确标注。

## 0. 仓库与分支

| 仓 | 路径 | 分支 | 起点 |
|---|---|---|---|
| vllm-ascend | `worktrees/moe-offload` | `dsv41-moe-offload` | dsv41-clean `1b1ad396b`（09-19 变基；旧头备份 `dsv41-moe-offload-pre-rebase-0919`） |
| vllm | `worktrees/moe-offload-vllm` | `dsv41-moe-offload` | dsv41-clean `a9686bc6e6`（09-20 快进；我们在 vllm 侧无提交） |
| ktransformers | `worktrees/ktransformers` | `dsv41_vllm_offload_dev` | kvcache-ai upstream/main `105359b` |

- ktransformers remote：`origin`=`wenxuewuhd/ktransformers-AK`（推送目标，待用户确认），`upstream`=kvcache-ai。
  本地另有 `dsv4_950_dev`（fork，只作 cherry-pick 来源，**不推**）。
- fork 独有、A5 单卡需要的两个提交：`c81fa24`（只给下放子集分配，否则按 384 全量 OOM）、
  `05a78ed`（CCD 打散绑核 / E8M0 scale 压 1 字节 / 相位计时）。已在临时副本试摘到 105359b，无冲突。
- ⚠️ 更正：MXFP4 CPU 算子 `AMXFP4_KGroup_MOE`（`kt-kernel/operators/amx/fp4-moe.hpp`）**upstream 自 #1970 起就有**，
  不是 fork 独有（已用 `git cat-file` 核实 31985f4 与 105359b 均含）。该后端不发 AMX tile 指令，走 AVX512-BF16。
- 参考：sglang 实现 `/workspace/code/dsv4_single_card/sglang-kt-a5`（分支 `dsv4/950-kt-a5`，核心 `685a00e1f`，
  主体 `python/sglang/srt/layers/moe/kt_ep_wrapper.py`）；设计文档持久副本
  `build_artifacts/moe_offload_ref/sglang_design_doc/`；sglang 侧文档在 `/workspace/code/dsv4_single_card/*.md`。

## 1. DSV4.1 与 sglang DSV4 的差异

| | DSV4（sglang，已完成） | **DSV4.1（本线）** |
|---|---|---|
| 层 × 专家，topk | 43 × 256，top6 | **40 × 384**，top6（另 3 层 MTP，不加载） |
| hidden / inter | 4096 / 2048 | **5120 / 2304**（每专家 1.4×） |
| 每专家跨全部层 | 0.535 GiB | **0.70 GiB**（compact scale） |
| 驻留 / 下放 | 160 / 96（37.5%） | **≈100 / ≈284（≈74%）**（估算，待 profile） |
| 每 token CPU 专家数 | 2.25 | **≈4.4** |

- 驻留数估算：可用 HBM ≈107 GiB − 其余权重 10.4 GiB − KV/工作区 ≈25 GiB ⇒ ≈70 GiB ÷ 0.70 ≈ 100。
- ⇒ CPU 工作量约 DSV4 的 2.8×，按 DSV4 实测带宽外推 bs=1 CPU 下限 ≈10 ms/token（推断）。
  频率放置与"提前一层下发"从可选变为值得做。
- sglang DSV4 结果（参考上限）：bs=1 20.57 ms/tok（48.6 tok/s），GSM8K 96.74% vs 多卡 96.89%。

## 2. 主机内存账

稳态两大项：**engram 189 GiB（/dev/shm） + CPU 专家 ≈195 GiB ≈ 384 GiB（≈412 GB）**，另加：
- 进程自身（torch/vLLM/其余权重加载中转）：量级十几到几十 GiB，待实测；
- 加载峰值：kt 逐层加载有 safetensors 复制 + staging + BufferB，瞬时约该层子集的 3×（≈15 GiB，推断），
  必须移植「del + malloc_trim」，否则每专家多留 ≈5 MiB；
- pinned I/O 缓冲：很小；读 safetensors 的 page cache：可回收。

⇒ 单独跑 ≈400–420 GiB，在 755 GB（≈703 GiB）的机器上装得下。

**与 integration 四卡服务并存**：engram 表是 `/dev/shm` 上**共享的同一份**，integration 已经在用，
⇒ 本线**增量只有专家 ≈195 GiB + 进程开销**。2026-09-18 实测 `MemAvailable` ≈ 457 GiB，账面够。
待核：
- NUMA 不均：engram shm 在 node0 149 GiB、node1 40 GiB（/sys/devices/system/node/*/meminfo，2026-09-18）；
  threadpool=2 时专家每节点 ≈98 GiB，node0 余量更紧。
- integration 每 worker ~67 GB 是否含 shm 映射（决定真实余量）；判据用起跑前后 `MemAvailable` 差，不用 RSS。
- 规则：整机在飞时 available < 150 GB 不起；掉到 100 GB 以下自停。

**NUMA 放置决定（2026-09-18）**：每节点 MemTotal 377.5 GiB；engram 189 + 专家 195 = 384 GiB > 单节点 ⇒ **单节点放不下**。
- 专家：**两节点均分**（kt threadpool=2，每节点 ≈98 GiB）。专家是带宽型负载，两个节点的内存控制器都要用上；
  sglang 实测 threadpool 2 比 1 快 37%（bs=1 每层 150.6 → 94.6 µs）。
- engram：**不动**。它是逐 token 小块 gather，延迟型、带宽很小，跨 NUMA 多出的延迟相对毫秒级 step 可忽略；
  且它是 integration 在用的共享产物。只在 node0 实测放不下专家时，才找 integration 用 interleave 重铺（届时 ≈95/95）。
- 当前余量（MemFree + 可回收 page cache = FilePages − Shmem）：node0 ≈152 GiB、node1 ≈269 GiB，两边都放得下 98 GiB；
  node0 余量 ≈54 GiB 偏紧。
- ⚠️ kt 工作线程用 `hwloc_set_membind(..., HWLOC_MEMBIND_BIND)`（`kt-kernel/cpu_backend/worker_pool.h:48`）严格绑定 ⇒
  本节点不够时**不会溢出到另一节点而是 OOM**（sglang 已踩过：被杀后 eval 客户端继续输出假分数）。
  系统 `zone_reclaim_mode=0`、`numa_balancing=1`。⇒ 加载后用 `/proc/<pid>/numa_maps` 核实实际落点，并盯 node0 余量。

## 3. 集成设计（vllm-ascend）

- **切入点**：`vllm_ascend/ops/fused_moe/routed_experts.py` forward_impl —— topk 之后 / dispatch 之前分流，
  finalize 之后加回 CPU 输出。
  - 原始 topk → kt-kernel；NPU 那份把 CPU 专家改写成「合法 id + 0 权重」，形状不变（Ascend 路由 kernel 不接受 -1）。
  - 静态专家掩码：EP=1 下注入 log2phy / expert_map（Ascend `init_eplb_config` 在 ep_size==1 时返回 None，需改）。
  - `weight_loader`（`routed_experts.py:554`）对 CPU 专家返回 False 跳过；kt 自己从 safetensors 读。
  - shared expert 的 event 重叠（`fused_moe.py:357`）保持。
- **进图**：host-func 图节点，照 engram 现成做法（`vllm_ascend/models/deepseek_v4_1/engram_embedding.py:272-307`）。
  MoE 在 FULL 与 PIECEWISE 下都在图内，这是 FULL 下唯一可行的路。约束（均实际炸过）：
  - 107015：必须订阅**提交时那条流**（捕获流）；
  - 107011：重复订阅，只吞这一个码；设 `KT_EXTERNAL_NPU_REPORT_SUBSCRIBER=1` 旁路 kt 自带 poller；
  - 107030：捕获期 H2D 被拒 ⇒ 查表在 `process_weights_after_loading` 预烘、常驻设备；
  - pinned I/O 缓冲按捕获 bs 固定；event 活到进程结束；捕获后 replay 一次（vLLM 核心已有）。
- **kt-kernel 侧**：`python/experts_base.py:112` 写死 sglang 的捕获态检测 → 改为参数；透出 `numa_nodes`。
- **加载判据（dsv41-model 提醒）**：E8M0 scale 只能按字节搬。判据是字节往返逐字节比（读回应落在 117..127 且多值（全表实测）），
  不比 dtype 名；另全表扫 scale==0 字节（compact 模式遇 0 抛异常，`fp4-moe.hpp:271-287`）。
- **其他缺口**：world_size=1 的 MoE comm 路径（ALLGATHER）在 V4.1 上没跑过；profile run 强制随机路由会灌满 CPU 路径；
  MTP 专家跳过（DSpark 在 A5 被 `KvQuantSparseAttnSharedkv` 挡死）。

## 4. 里程碑与跟踪

| # | 内容 | 卡 | 判据 | 状态 |
|---|---|---|---|---|
| M0a | kt 分支摘 c81fa24 + 05a78ed；构建 wheel 到私有目录（不装全局） | 否 | `__cpu_variant__=avx512_bf16`；DSV4.1 scale 全表无 0 字节 | ✅ 5378c88/b5e0f8b（待用户 push）；wheel `build_artifacts/moe_offload_ref/kt_wheel/kt_kernel-0.7.0.post4-…whl` sha256 23f1f844…，装 `kt_site/`；scale 46080 张量 0 个 0/0xFF，范围 117..127（`m0_evidence/scale_scan.txt`） |
| M0b | 纯 CPU 单测：一层 DSV4.1 kt MXFP4 vs torch 参考，带变异对照 | 否 | rel err 在 bf16 地板内；变异臂必须红 | ✅ 083e2b8 `kt-kernel/test/dsv41_mxfp4_offload_check.py`；层 1/3/14 × T=1/4/32 PASS；24 必红臂全红；字节往返 0 差异（`m0_evidence/m0b_run3.log`）。⚠️ 未覆盖：NPU 驻留专家是 W4A8（激活 MXFP8），kt 用 bf16 激活 ⇒ 两侧数值口径不同，M1 上卡判据要分臂、M4 兜底 |
| M0c | 找 integration：GSM8K 路由 topk 频率（每层，eager 采；DP=4 各 rank 加总）＋第二类数据一份；clean 多卡 bs=1 性能基线 + 同一 profiler 脚本采算子分布 | 他们的卡 | 频率表：每层计数和 = token 数×6，id 为全局 0..383；两份 top-100 重合度；profiler 脚本可复用于单卡 | 🟡 ① 计数钩子已入本树 1c047fee7（起法见 tools/kt_offload/README.md），等 flash_dev 排多卡卡位；② integration 测，等卡 |
| M1 | vLLM 集成 eager：掩码/加载/分流/加回；单层上卡 | 1 | 见 §4.2 | ✅ **t0 PASS**（44s，单卡 ALLGATHER+W4A8 在 V4.1 上首次跑通；rel_l2 5.33e-2 = fp8 激活地板，6 变异臂红）+ **t1 PASS**（78s，57 必检 / 21 变异臂；arm A 六组逐位相同；CPU 路由占比 50–75.5%）。整网 ⬜ 等卡 |
| M2 | 图模式 + host-func 节点；单层上卡 | 1 | 见 §4.2 | ✅ **t2 PASS**（17s，14 必检；捕获时 host-func 0 次、每 replay +1、与 eager 逐位相同、三输入互异；探针恒 False 变异臂红）。⚠️ 缺口：跑的是 BREAKABLE_CUDAGRAPH=0，已验证服务用 =1 ⇒ 整网图模式补。整网 ⬜ 等卡 |
| M3 | 性能（**目标 bs=1 延迟**）：侧流、提前一层下发、频率放置、线程/NUMA 调优 | 1 | bs=1 TPOT（bs=32 附带），用同一 perf_baseline.sh / perf_bench.py 同参数，预热到 Triton 缓存不再增长；**多卡基线（最终形态 clean 1b1ad396b，4 卡图模式，2026-09-19）：bs1 TPOT 26.84 ms；bs32 54.97 ms / 563 tok/s（IN128）；IN4096 bs1 27.08 / bs32 78.60 ms**（`build_artifacts/m1_run/gsm8k_flight4_final_101009/`）。旧的 43.28 ms 是修复前，作废 | ⬜ 用户定：**先验 t0/t1/t2，再做侧流**。M3 每项须归类：①纯调度（侧流/绑核/线程）判据 = 开关两态输出**逐位相同**（sglang F1：捕获未传播到侧流 ⇒ 更快且算错，只看速度会放过）；②改数值的近似（用旧输入、kt 的 max_deferred_experts_per_token 之类）默认不做，要做先报用户并单独评精度。⚫ 「提前一层下发」性质待核：第 N+1 层 MoE 输入依赖第 N 层输出，字面理解讲不通 |
| P | **NPU 算子耗时分布**（用户要求）：服务拉起后 profiler 采 decode 稳态，按算子类型聚合 device 时间 + host-func/空隙；M2 起每个里程碑各采一次，M3 优化前后对比 | 1 | 分算子表（总时长/次数/占比）+ device-busy 与空隙占比；预热后采，多步取中位 | ⬜ |
| M4 | 精度：GSM8K 1319 全量 graph，口径同基线（thinking 关、贪心、max4096） | 1 | 逐题配对，对 96.97% 在噪声内 | ⬜ |

## 4.1 单卡起法（本线自有，2026-09-18 起）

用户规矩：**单卡改动一律在本线自己的树和脚本里**；integration 的树、脚本、多卡基线保持原样，只答复问题。
integration 交接物已入库 `tools/kt_offload/`（6ebc88ff8，见其 README）：
```
KT_OFFLOAD=1 NPUS=<一张卡> WT=moe-offload DSV41_VLLM_TREE=<ROOT>/worktrees/moe-offload-vllm \
KT_ADDITIONAL_CONFIG='{"kt_offload_config":{...}}' MAXLEN=... bash worktrees/moe-offload/tools/kt_offload/serve_v41_singlecard.sh
```
- TP1/DP1、无 --enable-expert-parallel、目前固定 --enforce-eager（M2 后在本脚本里去掉）；kt_site 追加在 PYTHONPATH 末尾；SMOKE=1 与 KT 互斥。
- 本脚本是共享 `build_artifacts/serve_v41_bringup.sh` 的副本，共享原版的后续修复不会自动同步 ⇒ 起飞前 diff。共享的 env_dsv41.sh / assert_trees.py / verify_shared_artifacts.sh 只调用不改。
- 起飞闸 `tools/kt_offload/kt_offload_gate.py`：N 从 KT_ADDITIONAL_CONFIG 读；HBM：10.4 + N×0.7004×KT_DEVICE_FACTOR ≤ 128×GPUUTIL − 25；主机：(384−N)×0.7004 + 15 + 10 + 16 对 MemAvailable，另逐 NUMA。rc 0 放行 / 1 拒绝 / 2 算不出。
- **首次上卡待回报**：KT_DEVICE_FACTOR（NZ 填充）、进程开销实测。
- M0c① 计数器也归本线：钩子进本树 routed_experts.py，用**本树**起多卡服务（offload 关、WT=moe-offload）采，卡由 flash_dev 排。M0c② 多卡性能基线 integration 照做。

## 4.2 上卡测试（`tools/kt_offload/oncard/`，ce0c49f4d，CPU 自检 43/43，未上卡）

`run_oncard.sh <card> <t0|t1|t2>`：租约随进程死、ASCEND_RT_VISIBLE_DEVICES、私有缓存 /dev/shm/dsv41_singlecard/、前后显存+dmesg、AICore 类异常立停保留现场。

| 测试 | 内容 | 墙钟（估） | HBM | 主机 |
|---|---|---|---|---|
| t0 单层 | offload 关，384 专家，world_size=1 ALLGATHER + W4A8 vs 参考 | 10–15 min | ≈14 GiB | ≈10 GiB |
| t0 三层 | 同上 L=0/20/39 | 30–45 min | ≈14 GiB | ≈10 GiB |
| t1 | offload 开单层分臂（prefix 100 + 非连续 explicit） | 25–40 min | ≈17 GiB | ≈30 GiB |
| t2 | 图模式单层捕获/replay | 5–10 min | ≈5 GiB | ≈25 GiB |

- max-model-len 硬顶 16384（N3 守卫 `ceil(floor(L/r)/8) ≤ 2048`，层 20/24/28/32/36，r=1；与 TP/DP/offload 无关）；脚本默认 8192。
- 本线不挪任何东西出图（第 3 批关），不挂任何 KV/offload connector（设计文档 §j）。

## 4.3 CPU 侧实测（2026-09-20，纯 CPU，未上卡）

产物 `build_artifacts/moe_offload_ref/cpu_bench/20260920_022930/`（33 配置，~5800 次计时），脚本 `tools/kt_offload/bench/`（c5b83c0c2、d1cdf25fe）。

- **核心曲线（实测）**：T=1、每层 kt 耗时 `T(k_cpu) = 66.8 + 43.95·k_cpu µs`（k≥2 拟合；k=1 偏高是并行度不足）。边际 43.95 µs/专家 ÷ 18.80 MB = **428 GB/s**，为本机双路上限 495 GB/s 的 86%（sglang 在 DSV4 上独立测到 431 GB/s / 87%）。
- **最优配置（实测）**：cpuinfer_threads 96 / threadpool 2 / numa[0,1] / CCD stripe（胜 pack 8–26%）/ compact scale（**输出逐位相同**，6 次 sha256 全等）。th48 只慢约 5%，共享机让核时可用。
- **bs=1 外推（推断）**：均匀路由 E[k]=4.44 ⇒ 每层 260 µs ⇒ **CPU 侧 10.4 ms/token**（与 §1 独立推断吻合）。
- 🔴 **HBM 闸把驻留数限死在 N ≤ 113**（128×0.9 − 25 − 10.4 ÷ 0.7004）⇒ 可达区间 N=100–113，CPU 10.4–10.1 ms。**反解：CPU ≤10 ms 需 N ≥ 115，已超闸；≤5 ms 需 N ≥ 294（216 GiB HBM，不可能）** ⇒ 靠加驻留压 CPU 时间这条路在单卡上关死；10 ms 是 CPU 侧硬地板，再降只能靠隐藏（侧流）或真实路由偏斜（M0c 频率表未采，偏斜列只能当上界）。
- **与多卡 bs1 TPOT 26.84 ms 的对照（条件式，单卡 NPU 侧未测）**：串行则留给 NPU 的预算 16.4 ms；完全隐藏则要求 NPU 计算 ≥10.4 ms。另有每层 host-func 往返（sglang A3 约 50 µs/层 ⇒ ≈2 ms/token，NPU 侧口径，本次未测）。
  ⚪ 协调方判断（推断）：4 卡是 DP4，每 rank 也跑整网、只是专家分片 + MC2 通信；单卡省掉通信但多算本地专家 ⇒ **单卡 bs=1 追平 26.84 ms 的可能性不大**，本线的价值主要在「1 张卡而不是 4 张」，性能目标宜按此重谈。
- **可信度**：工作集扫描当阳性对照——0.11 GiB 工作集算出 604 GB/s（超硬件上限 ⇒ 测的是缓存），主测点 14.92 GiB 未落入；k 从 1→6 近似线性（不是空转）。⚠️ 主测工作集 14.92 GiB < 生产 ~100 GiB ⇒ 曲线偏乐观，幅度未量。

## 4.4 HBM/专家数账（2026-09-20，由实测反推）

```
卡可用 128×0.9 = 115.2  −  非专家权重 10.4  −  工作区 ≈7.8（多卡实测 11.4 减去单卡不需要的 MC2 缓冲 3.6，推断）
⇒ 专家+KV 额度 ≈ 97.0 GiB ；每专家 0.7004 GiB ；KV 343.8 KiB/token（多卡实测 24.54 GiB = 74,872 token 反除）
```

| KV 目标 | KV GiB | HBM 专家数 | DDR 专家数 | DDR GiB | CPU ms/token |
|---|---|---|---|---|---|
| 16,384 token（一条满长，N3 顶） | 5.4 | 130 | 254 | 178 | 9.6 |
| 32,768 token（8×4K 并发） | 10.7 | 123 | 261 | 183 | 9.9 |
| 65,536 token | 21.5 | 107 | 277 | 194 | 10.3 |
| 闸默认（KV+工作区 25） | — | 113 | 271 | 190 | 10.1 |

⚫ 两个待上卡校准项：NZ format_cast 是否有对齐填充（>0.7004 则专家数下调）；单卡工作区实测值（服务日志打印的 KV 可用量一比即得）。

## 4.5 KV 显存回收（review 交办，排在整网 bring-up 之后）

review 拆解：每 token 计费 343.6 KiB，**98.18% 是 40 个单层 SWA 组的块占用**（页 padding 4.00× × 池块计费 4.137× × 40 组各预留 261 块）；根因 `_approximate_gcd([1,3,40], lower_bound=第一个组的 tuple 数=1)` ⇒ SWA 被切成 40 个单层组；**可回收约 10.8×**（L3 SWA block_size 32→128 得 3.75×；L4 ring 区按 max_num_seqs 定长，现占池子 4.45 GiB/18.1% 而 >99.4% 碰不到）。⚠️ 上游 `_get_packed_kv_cache_groups` 在本机被 `patch_kv_cache_utils.py` 在 import 期整组换掉 ⇒ **读运行时日志，不读上游源码**。
🔴 **本线的收益要按两根轴报**：① 并发/上下文容量（成倍，10.8× 直接体现）；② TPOT——**预计 ≤1 ms，属噪声边缘**：KV 21.5→2.0 GiB 只买到 107→135 个常驻专家、CPU 10.3→9.5 ms，因为 10 ms 里 7.8 ms 是带宽硬底。⇒ 若只为 TPOT，不值得排在侧流之前。
🔁 **此结论的前提（变了要重算）**：CPU 侧是内存带宽硬底——实测 428 GB/s = 本机双路上限 495 GB/s 的 86%，下放量按比例决定字节数。**触发重算的条件**：换搬运路径（例如权重不再每 token 从 DDR 全读、加缓存/预取/常驻重排）、权重再压缩（读的字节变少）、换主机（带宽不同）、或 kt 后端改动使 T(k) 曲线斜率变化。任一条成立 ⇒ §4.5 的收益量级与「不值得排在侧流之前」这句一起作废，要用新曲线重算。
步骤：① 读我们自己那一发服务日志（是否同样 40 个单层组、ring 区占比），判据作用在**实际分配字节**；② L3 先小用例（kernel 支持 ≠ 分配/回收路径安全）；③ 收益链每步带读数。不改 clean。

### 4.5.1 机制核实（2026-09-20，本线自己的两发日志，未上卡）

工具 `tools/kt_offload/kv_budget_singlecard.py`（只读，解析日志 + 独立重算 + 逐项断言）。
另在 `tests/ut/models/test_deepseek_v4_1_swa_block_size.py` 里**直接跑本树被 patch 过的 planner**
（`_get_kv_cache_groups_uniform_groups` + `_get_kv_cache_config_deepseek_v4_main`），不是模型的复刻。

两发（eager `bringup_0920_034907` / 图 `bringup_graph_0920_040749`）**读数完全一致**，重算逐项对上：

| 量 | 值 | 来源 |
|---|---|---|
| 组结构 | 43 组 `[128, 256, 32×40, 8]`，lcm 256 | 日志 kv_cache_utils.py:732/733，重算一致 |
| 池块字节 | (131072+16896)×3 + 3×32768 = **542,208 B** | 重算；planner 实跑同值 |
| num_blocks | **54,131** | 反解 tokens=42,733 + available 区间相交 |
| 实际分配 | **29,350,261,248 B = 27.3346 GiB** | 重算；日志 27.34(eager)/27.33(图)，两发跨 27.335 的打印边界 |
| 每请求块数 | 64 + 32 + 40×257 + 1 = **10,377** | 重算；concurrency 5.2154→日志 5.22x，token 42,733 逐个吻合 |
| **每 token 计费** | **670.7 KiB**（686,828 B） | = 10,377 × 542,208 / 8192 |
| SWA 占比 | **99.07%**（10,280/10,377 块，5.574 GB/请求） | 放大链：页 padding 4.00× × 池块计费 4.137× = 16.55× |
| ring 区 | 98,304 B/块 × 54,131 = **4.956 GiB = 池子 18.13%**；每请求只用 1 块，max_num_seqs=8 ⇒ 实际碰到 0.0148% | |
| in-flight | 2×4096=8192（async scheduling 开）。取 4096 只能算出 14.27 GiB ⇒ 判据有区分力 | |

🔴 **三条更正**（都与「单卡 655 KiB / 差 1.9×」这句有关）：
1. **655 KiB 复现不出来**。同一算法算多卡是 24.54/74,872 = 343.7 KiB（与协调方 343.8 一致），算单卡是
   27.34 GiB / 42,733 = **670.7–671.0 KiB**，不是 655（低 2.4%，疑似转抄）。倍数是 **1.95×** 不是 1.9×。
2. **这 1.95× 与「单卡」无关，是 `max-model-len` 8192 vs 16384**。SWA 每请求预留 =
   `cdiv(min(127+in_flight, L), 32)+1`，in-flight 固定 8192 ⇒ L=8192 时 257 块、L=16384 时才 261 块：
   **分母翻倍而分子几乎不动**。同一公式在 L=16384 下算出 343.7 KiB，与多卡逐位吻合。
   ⇒ 「固定预留在单卡摊得更差」这句要改成「在短 max-model-len 上摊得更差」；单卡开 16384 就和多卡一样。
3. **§4.4 的两个常数都不对，但方向相反、差点互相抵消**（数值巧合不是交叉验证）：
   卡可用不是 `128×0.9=115.2` 而是日志的 **110.84**（`0.9×123.16`，卡报 123.16 GiB 不是 128）；
   单卡工作区不是推断的 7.8 GiB 而是实测 **1.36 GiB**（激活 1.21 + non-torch 0.15；图模式再 +0.24，
   且那 0.24 是压在 util 预算**之上**的）。⇒ 专家+KV 额度 = 110.84 − 10.4 − 1.36 = **99.08 GiB**（§4.4 写 97.0）。
✅ **旁证**：日志 weights 82.14 GiB − 非专家 10.4 = 71.74 = 100 × 0.7004 × **1.0243** ⇒ §4.1 待回报的
   `KT_DEVICE_FACTOR` 就地落实为 **1.024**（实测，本发）。

### 4.5.2 与「理论 KV」的对账（回答「bf16 是不是 ×4」）

**不是 ×4，是 ×3.25。** 按 card 那张表的口径逐项重算（重算 card 侧得 890 B/token，口径对得上）：

| 项 | card（FP4 + e4m3/16） | 本发 bf16 | 倍数 |
|---|---|---|---|
| compress_kv（4 源层，3×256 + 512 元素） | 720 B | 2,560 B | 3.56× |
| index_k（4 源层，3×64 + 128 元素） | 170 B | 330 B | **1.94×** |
| 合计 | **890 B/token** | **2,890 B/token** | **3.25×** |

- 主 cache：bf16 2 B/元素 vs FP4 0.5 + 1/16 scale = 0.5625 B/元素 ⇒ 3.556×（不是 4×，FP4 自带 scale 字节）。
- index_k **不是 bf16**：本发走 fp8_e4m3(1B) + 每 state 一个 fp32 scale = 132 B/state，vs MXFP4 68 B/state ⇒ 只 1.94×。
- 定长滑窗：card 口径 40 层×128 位×512 维×1B = 2.62 MB/序列，本发 bf16 = **5.24 MB/序列**（card 表写 43 层是 V4.0）。
- 上下文容量（纯内容）：4K 11.8 MB / 8K 23.7 MB / 32K 94.7 MB / 128K 378.8 MB。

🔴 **理论内容 vs 池子实际计费（同一发，L=8192，每请求）**：

| | 理论内容 | 池子实际计费 | 倍数 |
|---|---|---|---|
| 随上下文增长（4 源层） | 23.7 MB | 52.1 MB | 2.20× |
| 定长滑窗（40 层） | 5.24 MB | **5,574 MB** | **1,063×** |

滑窗那 1,063× 拆开是：admission 公式按 `min(127+in_flight, L)` 预留 **64×** 窗口本身（in-flight 8192 主导，
不是 128 位窗口）× 页 padding 4.00× × 池块计费 4.137×。
⇒ 全发 27.33 GiB 里只有 **0.51%** 是理论上需要留的内容字节。

### 4.5.3 两条改法的收益（同一发 27.33 GiB，L=8192）

A 轴 **容量（主轴）** / B 轴 专家与 CPU 时间（保住 42,733 token 不变，省下的 GiB 换常驻专家）：

| 改法 | KV token | 倍数 | 8192 并发 | 省 GiB | ΔN | N | CPU ms/token |
|---|---|---|---|---|---|---|---|
| 基线 | 42,733 | 1.00× | 5.22x | — | — | 100 | 10.47 |
| **L3 SWA block 32→128** | **164,420** | **3.85×** | 20.07x | **20.2** | +28 | 128 | 9.70（−0.77） |
| L3' 32→64 | 84,352 | 1.97× | 10.30x | 13.5 | +18 | 118 | 9.98 |
| L4 ring 定长 | 52,201 | 1.22× | 6.37x | 4.96 | +6 | 106 | 10.31 |
| L3+L4 | 200,904 | 4.70× | 24.52x | 21.5 | +30 | 130 | 9.65 |
| L5 mnbt 4096→2048（**只改配置**） | 81,861 | 1.92× | 9.99x | 13.1 | +18 | 118 | 9.98 |
| L6 SWA 每组 5 层（改分组） | 133,240 | 3.12× | 16.26x | 18.6 | +25 | 125 | 9.79 |
| L3+L5 | 304,352 | 7.12× | 37.15x | 23.5 | +32 | 132 | 9.59 |

结论与 §4.5 开头一致并更强：**B 轴最多买到 −0.96 ms（10.47→9.51），主轴是 A 轴**。
L3 一条就把 8192-token 请求的并发从 5.22 抬到 20.07，或者在并发不变时腾出 20.2 GiB。

⛔ **L4 不可直接实现（review 的「4.45 GiB 易得」要撤）**：ring 的槽位是
`slot = block_table[req][0] × CAPACITY + pos % CAPACITY`（`vllm/models/deepseek_v4_1/compressor.py:61-79`），
`block` 是**全局池块号**，取值域 [0, num_blocks)。把 ring 区缩到 max_num_seqs 块会越界写。
要做必须先给 ring 组做块号重映射（改 triton kernel + block table），风险远高于 L3，收益只有 1.22×。⇒ 不做。

### 4.5.4 实现（L3，默认关）

- `VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE`（`vllm_ascend/envs.py`，默认 `0` = 关，行为与改动前逐字节相同）。
- `AscendDeepseekV41SWACache.__init__` 里用 `resolve_swa_block_size()` 重解析
  （`vllm_ascend/models/deepseek_v4_1/caches.py`）。写在 `self.block_size` 上而**不只是 spec 上**：
  prolog 会核 `swa_kv_cache.shape[1:] == (metadata.block_size, head_dim)`，JIT 预热按
  `swa_cache_layer.block_size` 注册 kernel —— 只改 spec 会让三者不一致而**不报错**。
- 两条硬拦截（不静默回退）：① 值必须**逐字面**出现在该 cache 自己的 backend 的
  `get_supported_kernel_block_sizes()` 里（不是「整除」：`select_common_block_size` 会悄悄退到更小的一档）；
  ② 不得超过主 cache 的 storage block size（否则 planner 的裸 `assert` 在深处炸，不点名）。
- **不是拍脑袋选的 128**：Ascend 自己的 **V4.0 路径**在同一个 `AscendDSASWABackend`、同一个
  `AscendSlidingWindowMLASpec` 上就用 `DSV4_BLOCK_SIZES[128][0][1] = 128`（`models/deepseek_v4/model.py:124`）。
  V4.1 之所以是 32，只因为上游把它写死在构造点，而 Ascend 子类只覆盖了 `get_kv_cache_spec` 没覆盖块大小。
- CPU 单测 **34 个**（17 个字节账 + 17 个开关，后者含 2 个直接跑本树 planner 的），含 **9 个变异臂**
  （6 个打坏字节账的机制常数、3 个打坏开关的判据）；`pytest tests/ut/models/test_deepseek_v4_1_swa_block_size.py
  tests/ut/kt_offload/test_kv_budget.py` 34 过。
  全树回归（tests/ut/models + kt_offload + attention）：改动前后失败集合逐条相同（30 个既有失败，
  1219→1236 通过），新增全过。

### 4.5.5 上卡验证方案（交回协调方排卡）

**1 张卡，两臂，预计 2.5–3.5 h**（含起停与 GSM8K）。两臂同配置、只差
`VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE`（A 臂不设 / B 臂 128）。基线配置照 `bringup_0920_034907`：
`MAXLEN=8192 --block-size 128 --kv-cache-dtype bfloat16 --no-enable-prefix-caching
--max-num-batched-tokens 4096 --max-num-seqs 8`，KT_OFFLOAD=1 N=100。

| # | 判据 | A 臂期望 | B 臂期望 | 怎么判 |
|---|---|---|---|---|
| ①容量 | `kv cache group sizes` | `[128,256,32×40,8]` | `[128,256,128×40,8]` | 日志行 |
| | `GPU KV cache size` | 42,733 token / 5.22x | **164,420 token / 20.07x** | 日志行 |
| | 实际分配字节 | 54,131 × 542,208 | 同样 54,131 × 542,208（池不变） | 跑 `kv_budget_singlecard.py --log <新 serve.log>`，**它自己会 raise** |
| ②正确性 | 同一批 50 题 GSM8K，贪心、`temperature=0`、同 seed | — | **逐题配对输出必须一致**（不是分数一致） | 两臂 diff 逐题生成文本 |
| | 长序列覆盖 | — | 至少 1 条 prompt+max_tokens 接近 8192 | 单独一条长请求，两臂逐 token 比 |
| ③回收路径 | 必须真的踩到块回收 | — | — | 并发 8 条 × 每条 ≥2048 token 连续跑 ≥5 min，看 `GPU KV cache usage` 上升后回落；只跑短请求 = 零覆盖 |

- 时长拆：起服务 ×2 各约 6 min（本发实测 04:07→04:14 起到就绪）；50 题 GSM8K ×2 各约 25–40 min；
  长序列 + 并发浸泡 ×2 各约 15 min；日志判据与比对 20 min。
- 失败即停保留现场（`run_oncard.sh` 那套）；起飞前 5 min 内不要紧接着上一发拆除（dmesg 闸，坑 P16）。
- ⚠️ B 臂第一次跑会**重新编译 SWA 的 triton kernel**（`_COMPUTE_SWA_INDICES_AND_LENS_KERNEL`
  按 `block_size` 特化注册），冷启动会慢；`TMPDIR=TMP=/dev/shm/dsv41_tmp`。

### 4.5.6 风险与未决

| # | 风险 | 现状 |
|---|---|---|
| R1 | `get_supported_kernel_block_sizes()` 里有 128 ≠ 分配/回收路径安全 | 已把「kernel 列了」和「planner 接受」分开验：planner 在 CPU 上实跑过 32/128 两档，页 131072 无 padding、组数/lcm/池块/num_blocks 全不变。**回收路径只能上卡验**（判据 ③）。 |
| R2 | block_size 变化影响 attention kernel 的 plan 选择？ | 未见。`resolve_dsv4_cache_dtype` 的 plan 选择看 `--kv-cache-dtype`（承重参数，指南 L301），与 SWA 块大小无关；运行期 kernel 从 `swa_metadata.block_size` 取页大小，prolog 还有显式形状核对会 raise。⚠️ triton kernel 按 block_size 特化编译 ⇒ 需要一张起得来的卡。 |
| R3 | prefix cache 关着的前提 | 本发 `enable_prefix_caching=False`。开了会走 `hash_block_size = gcd(group block sizes)`，32→128 后 gcd 仍是 8（ring 的 8 决定），lcm 仍是 256 ⇒ 调度块大小不变。**但没验过**，改法不覆盖 prefix cache 开的情形。 |
| R4 | N3 守卫（max-model-len ≤16384）与 block_size 有交互？ | **无交互**（读了源码）：`assert_candidate_mask_is_vacuous` 只吃 `max_seq_len / compress_ratio / candidate_topk_blocks / candidate_block_size`（后者来自 hf_config，是 8），没有 KV 块大小这一项。反向看 L3 还让启动 admission 更松（4.29→1.12 GiB）。 |
| R5 | L5（mnbt 4096→2048）是纯配置、1.92× 白拿 | 但 engram staging 缓冲按 mnbt 构造期分配、prefill 分块也减半 ⇒ **是性能变量**，要单独 A/B，不能和 L3 混在一发里。 |
| R6 | 收益表的 CPU 轴继承 §4.3 的「带宽硬底」前提 | §4.5 那条重算触发条件原样适用。 |

## 4.6 整网 bring-up 实测（2026-09-20，卡 3，N=100 prefix）

**产物**：`build_artifacts/moe_offload_ref/bringup_0920_034907`（eager）、`bringup_graph_0920_040749`（图）**仍在**；
`prof_0920_045212`（算子 profile）**【已记录、不可再验】原始产物已于 2026-09-21 项目清理中删除（未被 git 跟踪，不可恢复）**；`build_artifacts/m1_run/perf_singlecard_offload_N100_graph_0920_041940`（perf）。
⇒ 下表与算子级一节的数字都是当时内联的，**仍可引用，但不能再回到产物复算**。

| 项 | 实测 | 多卡基线 |
|---|---|---|
| 起服务 | 约 2 分钟（kt 每层 ~1.6 s × 40） | — |
| GSM8K 50 题 eager / graph | **50/50 / 50/50**，err 0、截断 0 | 1319 题 96.97% |
| **bs1 TPOT（图，n=3）** | **40.47 ± 0.38 ms**，吞吐 22.9 tok/s，TTFT 831 ms | **26.84 ms** |
| bs32（⚠️ 我们 MAXSEQS=8，不可比） | TPOT 105.9 ms，70.5 tok/s，TTFT 89 s | 54.97 ms / 563 tok/s（MAXSEQS 256） |
| HBM | 权重 82.14（⇒ **KT_DEVICE_FACTOR=1.024**）/ KV 27.34（42,733 token，5.22×）/ 激活 1.21 / 非 torch 0.15 / 图 0.24 | — |
| 异常 | 无 10701x、无 AICore、无 507018、无 EngineCore fatal | — |

**算子级 profile（bs=1 稳态 decode，8 s 窗口 182 token；出处曾是 `prof_0920_045212/op_breakdown_bs1.txt`，【已记录、不可再验】原始产物已于 2026-09-21 项目清理中删除（未被 git 跟踪，不可恢复））**
- 区间并集：**device 真实忙 57.1%**，每 token 忙 24.96 ms、**空隙 18.73 ms**（相加法 57.0%，说明流间重叠可忽略）。
- **>100 µs 的空隙每 token 43 个（= 40 层 + 少数）合计 17.91 ms/token** ⇒ 这就是**完全暴露的 host offload 等待**；其中 kt 纯计算约 10.4 ms，**剩余约 7.5 ms 是 D2H/H2D + 回调开销**。
- Top 算子（每 token）：QuantBatchMatmulV3 4.16 ms / GroupedMatmul 1.79 ms（驻留专家）/ MhcPre+MhcSinkhorn 1.72 ms / SparseFlashMla 0.82 ms / MoeInitRoutingV3 0.51 ms。
- ⚠️ `prof_decode_breakdown.py` 按"图 Model ID"锚定，本发无该标记 ⇒ 它拒绝出结果；上表是直接聚合 `kernel_details.csv` 得到。
- ⚠️ profile 期每 token 44.08 ms（vs 无 profile 40.47）⇒ profiling 自带约 9% 开销，绝对值别直接引。
- **怎么采（本版 vLLM）**：`VLLM_TORCH_PROFILER_DIR` **不被识别**（日志 "Unknown vLLM environment variable"，`/start_profile` 返回 404）；要用启动参数
  `EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"<dir>"}'`。⚠️ `prof_decode_capture.py` 在 404 的情况下**照样打印"✅ steady decode captured"**（它的判据不看产物）⇒ 采完必须自己数 `<dir>/*_ascend_pt` 目录个数。

🔴 **对 M3 的含义**：40.47 − 26.84 = 13.6 ms 的差，与"暴露的 host 等待 17.9 ms"同量级 ⇒ **侧流重叠是唯一的大头**，且天花板由「fork 到 join 之间 NPU 有多少活」决定：本发 NPU 每层约 0.62 ms（24.96/40），而 host 每层约 0.45 ms ⇒ 理论上可藏住大部分，但要先把 7.5 ms 的搬运/回调开销压下来。

⚠️ **eager vs graph 逐题输出**：50 题里 **34 题生成文本不同**（都是贪心）。两臂都 50/50 全对。**对照已到（flash_dev 2026-09-20）**：文本相同率强依赖输出长度 ⇒ 必须按长度档比。我们 n=50、平均输出 143/147 token、文本相同 16/50=**32.0%**、判定翻转 **0**；最接近长度的参考是 GSM8K 跨 commit（154 token）**12.9%** 相同（C-Eval 10 token→96.8%；LCB 1927 token→22.2%、判定翻转 11.4%）。⇒ 同长度档上我们**更一致**，**无异常迹象**。⚠️ 口径：参考是同模式跨重启/跨 commit，我们是跨模式（方向上应更分歧）⇒ 这条只能排除「明显更差」，**不能证明没问题**；真判据是 M4 全量 GSM8K 对 96.97%。机制侧（accuracy 推测，未验证）：分歧可能来自 batch 组成（未开 prefill_mc2 时 logprob 随同批请求变化）；我两臂并发相同，但不当已排除。⚪ 我第一次比对用了不存在的字段 `output`（两边都是 None）读成"0 处不同"，加阳性对照才发现 —— **比较字段必须先证明有区分力**。

## 4.7 M3（侧流重叠）：失败、根因与现状（2026-09-20）

**主线不受影响**：不开侧流时 eager 与 graph 各 50/50、逐题一致；bs1 TPOT 39.81±0.20 ms（A 臂 n=4）。侧流是实验性优化。

| 上卡发次 | 结果 |
|---|---|
| 整网 A/B（overlap on） | 🔴 50 题只对 34、截断 32%、token 10×；逐位 38/50 不同；**无 10701x/AICore** |
| t3 3 层一次捕获 | `capture_end` **107025** |
| t3 `--graphs per-layer`（一对 fork/join） | **107025** ⇒ 「多对才失败」**已推翻** |
| t3 `--probe-only`（标量微型图） | **PASS** ⇒ 探针是**假绿灯**，已删除 |
| eager 侧流（T=4/8 × 3 输入） | 逐位相同（eager 路径本身没错） |

- 🔴 **平台结论（收窄后）**：把工作 **fork 到另一条流**再要求它进捕获 ⇒ capture_end 107025。**不是**「host 工作不能进图」——反例：engram 的 host gather 是图内节点的**交付默认路径**（`engram_embedding.py:319`，`ENGRAM_OUT_OF_GRAPH=0`，10 数据集 + 8.86 h 浸泡）；我们 A 臂同样在捕获流上跑 host-func 且 50/50。
- 三种机制分工：`graph_task_group` 改**设备任务参数** / `ExternalEvent` 管**跨流排序** / `_launch_host_func` 让 **host 工作进图**。
- **静默失败的指纹（实测算术）**：>100µs 大空隙 43.1→15.0 段/token、**存活空隙尺寸不变**（406→379 µs）⇒ 不是"被盖住变短"而是"整段不见"；每步段数**双峰**（≈210 步 6–14 段、≈26 步 42–51 段=eager 步）。⚫ 仍有两种读法（"25–30 层丢了" vs "捕获步 40 层全丢"），**逐层 HOST_FUNC_CALLS** 才能分开（开关已就绪）。
- ⚫ **服务为何不报错的首要假设（未验证）**：捕获期 `_capturing()` 为 False ⇒ 走 eager 分支 ⇒ CPU 工作在捕获那一刻于图外执行一次、**无任何节点进图** ⇒ 既不报错又每次 replay 复用旧输出。
- ✅ **安全状态**：`overlap_side_stream=true` 在图模式下**直接拒绝启动**（不是"探针不通才拒"）；假绿灯探针已删，回归条件是**元判据**——探针必须能复现已知两种真实失效（107025 与静默不重放）。
- 🔧 **两条护栏教训**（比 M3 本身重要）：① 护栏读**代理**（"有捕获在进行"≠"这条流的工作会进图"）⇒ 在该拦的那一刻放行；② 改成"对被判断对象做实验"后，**实验用了太弱的替身**（只搬 int32 标量 + `add_`，没有真实 GMM/dispatch/kt 缓冲）⇒ 同形只同了模式，没同规模与内容，仍是假绿灯。
- **修法（已审阅，暂停实现）**：形态 C = engram 形态——**同一条捕获流**上两个 host-func 节点（①只提交 `forward_task`+`cpu_infer.submit` 立即返回 → GMM → ②`sync_cpu_only()` 只等 CPU → H2D）。绕开 107025，不需票号（本层提交→GMM→本层等待之间无别的提交 ⇒ allowance=0 即"只等本层"）。⚠️ kt 硬事实：**CPU 队列单 worker 串行** ⇒ 跨层流水不可能，可重叠窗口只有本层 GMM；**每层必须 sync**（kt 只在下次 sync 重抛任务异常）。
- 🔁 **为什么暂停**：形态 C 净收益 = GMM 窗口 0.62 ms（实测）− 多一次回调 ≈62.5 µs（**推断**）。下一发 packed A/B 反解 **C = Δ(>100µs ms/token)/80**：C≈62 µs ⇒ 做；C≥300 µs ⇒ 作废；**Δ 与 80×C 对不上 ⇒ 模型本身错、7.5 ms 归因重做**。
  > 🔴 **2026-09-21 更新（这条待办已不可按原样完成）**：第三分支实际发生了——单价模型被 profile 否掉（§4.10 / design k.5d），
  > 且那份 A/B 产物已删（design k.16）⇒ **"反解 C"这件事没有产物可依**。形态 C 现在由 **t4 实测**的
  > node_us / window_us / ceiling 决定（design k.7），`0.62 ms/层` 与 `62.5 µs` 两个数均已作废，不要再引。

## 4.8 减操作数（packed_transfer，默认关，待上卡）

每层 host-device 操作 **5→3**（合并 D2H 一次拷贝 + 每层各自输出缓冲，顺带去掉 kt 的 `layer%2` 槽位共享）；eager 路径去掉每层两次**全设备** synchronize。自己拼 `forward_task` ⇒ kt 的前提变成我们的断言（按位置传参、qlen 是**指针**、张量活到任务完成、`qlen>chunked_prefill_size` 会毁堆且我们绕过了 kt 的检查、dtype/contiguous、提交与 sync 同节点）。判据：**开关两态逐位相同 + 操作数下降**，不断言速度。
⚠️ 上卡必须显式 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`（clean 第 5 批 O15 断言：图模式漏传直接拒绝；平台 import 期会默认成 0）。

## 4.9 🔴 硬约束：不允许跨层（用户，2026-09-20）

**任何把 CPU 专家贡献推迟到下一层/下一步的机制一律不做**，包括 kt 的 `max_deferred_experts_per_token`（语义：低权重专家用本层激活算、结果加到**下一层** MoE 输出）。
- 这不是"暂缓评估"，是**禁止项**：它会改变数值（CPU 贡献晚一层进残差流，下一层 attention 看到的输入就变了）。
- ⇒ 重叠窗口的上限**只能是本层 MoE 的 NPU 时间**（实测 GroupedMatmul 44.7 µs/层），即 1.8–4.4 ms/token 的量级；**10 ms 级的"基本藏掉"在此约束下不存在**，任何声称能做到的方案先检查它是不是偷偷跨了层。
- 派生：kt 的 deferred 相关参数保持 0；packed 路径写死 `incremental=False` 与此一致。

## 4.10 🔴 2026-09-20 晚：三条数字作废（按 profile 实际次数重算）

`tools/kt_offload/profile_ops.py`（可复跑）读 armA `task_time.csv`（181 步 × 40 层）：

| 类型 | 次/层 | 每次 | ms/步 |
|---|---|---|---|
| HOSTFUNC_CALLBACK | 1.05 | 0.007 µs | ≈0 |
| MEMCPY_ASYNC（所有 owner，**上界**） | — | 1.168 µs | **0.72** |
| **NOTIFY_WAIT** | **1.05** | **371.8 µs** | **15.59（TPOT 40.36 的 39%）** |

- ⛔ **作废**：「7.5 ms 是搬运+回调开销」「每操作 62.5 µs（或 37.5）」「每层一个操作 = 2.5 ms/token」「packed 应省 5.0（或 3.0）ms」「形态 C 多一个节点花 2.5 ms」。memcpy 的设备时间比两种单价推出的小 10–17 倍 ⇒ **"固定成本×操作数"模型本身错了**（这正是 §4.7 预留的第三分支）。
- ✅ **还活着的**：packed 收益 = A/B 实测；节点成本 = t4 第 1 项（未测）。
- **真正的大头是等待**（等 CPU 算完），不是搬运。
- ⛔ **作废**：形态 C 窗口「0.62 ms/层」——那是整层**全部** device 忙时间；窗口只有驻留专家 GMM，实测 **44.7 µs/层**（旁证：sglang 同结构 80–94 µs/层、只藏住约 10%）。

## 4.11 packed_transfer A/B 实测（2026-09-20，卡 3；出处曾是 `packed_ab_0920_094053`，【已记录、不可再验】原始产物已于 2026-09-21 项目清理中删除（未被 git 跟踪，不可恢复））

> 🔴 本节数字均为当时内联，**可引用、不可再验**。**依赖这份产物的待办已回扫**（design k.8 出处状态块）：
> 长尾离线定位、从 armA `task_time` 反解每操作成本、算子级前 20 两臂对照 ⇒ **都需重跑才能回答**；
> 其中"算子级前 20"**原始产物已删、本报告未内联**。另：两臂各 50/50 是**答对题数**，
> 逐位只有 **19/50 相同**（贪心），见 design k.8 的更正。

两臂 GSM8K **各 50/50**。性能**不是干净的胜利**：

| 统计量 | A(off) | B(on) | 方向 |
|---|---|---|---|
| TPOT 中位（n=4） | 40.24±0.48 | **39.50±0.37** | B 好 0.74 ms |
| TPOT 均值（n=4） | 40.36±0.55 | **41.37±1.58** | **B 差** |
| p99（四轮） | 40.6/41.5/42.7/45.7 | **40.0/56.3/71.3/83.3** | **B 明显更差** |

⇒ **典型 token 快约 0.75 ms，但长尾恶化**（四轮里三轮 p99 翻倍）。**开启前必须定位长尾来源**；报收益时两个统计量都要给，只报中位是挑数据。
⚠️ 同配置跨发散布约 0.5 ms（armA 两次 profile：17.91 vs 18.42 ms/token 大空隙）⇒ **任何 Δ<1 ms 的结论都要 n≥4 并报散布**。

## 4.12 🔴 「本平台不支持侧流进图」被反例推翻（待对表）

sglang 在**同一 torch_npu**（2.10.0.post4）上把跨流 fork/join **捕获进图并 replay 成功**（capture 2.52 s、logprob 逐位相同、profile 在侧流上数到 41 次 NOTIFY_WAIT），实测 **−5.37 ms**（`PLAN_dsv4_a5_single_card.md:508-534`）。
一个具名可查的差异：sglang 捕获传 `auto_dispatch_capture=True`（`npu_graph_runner.py:154`），我们的 `vllm_ascend/compilation/acl_graph.py:187` **没传、默认 False**。⚫ 假设，未验证。
⇒ §4.7 的平台结论降级为「**待与参考实现逐项对表后重判**」；`overlap_side_stream` 的"直接拒绝"保留（安全），但**不得再作为"平台不支持"的依据被引用**。

## 4.13 读数必须盖构建指纹（2026-09-21）

> ⛔ **2026-09-21 撤回前半段：共享 `.so` 最终【没有换】。** 我们树里仍是 `2b58b578e4fc`，钉值未动，闸未红，**旧读数继续有效、无需「适用构建」脚注、无需重测基线**。
> **撤回理由（flash_dev 交代）**：批次 1 的代码在一棵**独立 clone** 上，而分发**只搬 `.so`**、Python 改动跟着树走 ⇒ 照原方案执行会给四条线发一个**它们源码里没有的改动**，然后跑出一发**根本没测到批次 1**的 GSM8K。
> 🔴 **值得记的是失效形态**：那第三件事会**以一份漂亮的精度报告的形式出现**——产物齐全、判据全绿、结论完全错。是 integration 当场在 5 棵树里 grep 新符号（全 0 命中）拦下来的。
> ⇒ 通则：**分发二进制产物时，必须同时验证它依赖的源码改动在被分发的树里**；只搬一半的分发，症状是「测了但什么都没测到」。
> ✅ **本节余下的「产物盖指纹」仍然要做**，理由被这次事件加强：这次来回确认之所以必要，正是因为「这批数测在哪个构建上」只存在于对话里。

共享 `vllm_ascend_C*.so` 于 2026-09-21 被重建分发（多卡批次 1：index K cache fp8 → MXFP4，`ConvertType` 新增 4-bit 分支；非 4-bit 张量行为不变）。**clean 与 moe-offload 在闸里共用同一钉值 ⇒ 我们树里的 .so 也会被换**，旧的 `2b58b578e4fc` 会被判红。
🔴 **对本线的影响不是能不能起飞，而是读数跨构建不可比**：
- 旧构建（`2b58b578e4fc`）上测的仍然有效：packed A/B 两臂（§4.11）、两份 profile、§4.10 的 NOTIFY_WAIT 371.8 µs/层、`cpu_bench`（CPU 侧不受 .so 影响，但一并标注）。
- **不得与新构建的数直接比较**；t4/t5 与形态 C 的 A/B **必须整发落在同一构建内**；换完先重测一次同配置基线（≈20 min）取代旧基线。
⇒ **产物一律盖指纹**（`so_md5` + 算子包内容指纹 + kt wheel sha256 + 两树 commit），判读脚本在 `so_md5` 不同时**拒绝给差值**并打印两边指纹。理由：现在靠"记得这批数测在旧 .so 上"防止跨构建比较，**那条防线在人脑里**，而聊天记录不会出现在报告里。

## 4.14 方法论：「只在卡上暴露」是一个独立风险类别

**判据的运行环境与被判对象的运行环境不同 ⇒ CPU 单测必然测不到。** 已知成员与我们的应对：

| 成员 | 我们的应对 |
|---|---|
| 默认工厂设备（CPU 默认 vs 加载期 NPU） | CPU 建模（`meta` 默认设备 + AST 扫描 15/15 调用点点名 device） |
| 生产日志配置（pytest 挂 root handler vs worker 只挂 file handler 且不 setLevel） | CPU 建模（子进程立起 worker 形态配置 + 三条阴性对照） |
| 设备端 dtype 支持表（fp4 视图不支持设备内拷贝/reshape） | 本线当前不适用（已查：我方路径按 int8/uint8，不建 fp4 视图）；若将来用到 ⇒ 需卡上臂 |
| ACL 捕获对跨流的限制（107025） | **卡上臂**（t3/t5；标量微型图给过假绿灯 ⇒ 替身必须与被测对象同规模同内容） |

⇒ 这类守卫**必须有一条在卡上跑的臂，或在 CPU 上把环境差异显式建模**；两者都没有就等于零覆盖。

## 4.15 仪表与 A/B 设计的三条更正（2026-09-21）

**① 🔴 「等卡退干净」的循环从没等过**（`run_packed_ab.sh:159` 小写 `process id|pid`）：
npu-smi 忙卡印 `Process id`（大写）、空卡印 `No process in device.` ⇒ 两种状态都不匹配 ⇒ **忙卡漏、空卡也不匹配，第一轮就 break**。
> **一个永不等待的等待，和一个成功的等待，输出完全一样。**

✅ 它顺带解释掉一批旧观察：前几发手工拆除时反复出现的「前端进程残留 / 租约还在」，大概率就是这个循环。
修法：两个口径都看（大写进程表 + `HBM Usage Rate`），任一为忙就继续等；**读不到当忙**（放行才是危险方向）；
**阳性对照**：服务在服务时该函数必须读到忙，读不到就在报告里写**「本发所有『卡已归还』都未经证明」**。
定格式的出处是两个**独立写的**共享脚本均按大写解析（`wait_cards_free.sh:49`、`card_owners.sh:43`）——两个证人，不是一个推论。

**② `nproc` 在本环境恒为 1**（= min(亲和性, $OMP_NUM_THREADS)，真实 256）。我方脚本未用；产物里现在**把骗人的读数和真值并排记**（`nproc_reports=1` / `cpus_online_getconf=256` / `OMP_NUM_THREADS=1`）——只记真值的话，下一个人看到 1 不知道该看哪一列。
⚠️ 外线：`tools/mooncake_installer.sh:182/242/243` 的 `make -j$(nproc)` 实际是 `-j1`（**不是错值，是慢 256 倍且完全无声**），已转交。

**③ 🔴 这台机器会自己变慢 2.5×**（同一 commit，11 小时前后 242s vs 603s）⇒ **性能 A/B 两臂必须同时段**。
我们旧设计是**顺序分块**（packed A 09:48 / B 10:28；M3 A 06:22 / B 07:03，各相隔 40 分钟）⇒ 已改为**交错轮次 A,B,A,B**，按轮配对做差，
**噪声地板 = 同臂跳轮最大摆动**；🔴 **地板没测到而又测出差值 ⇒ 判 fail**（「有差值没尺子」正是最容易被读成收益的形态）。
⇒ **连带更正 §4.11**：packed 的「中位 −0.74 ms」在 2.5× 级自漂面前**不可区分** ⇒ 结论改为
「**在能区分的设计下未测到收益；旧读数受顺序分块设计限制**」（方向不变、理由换了，默认仍关）；p99 恶化（40→83 ms）幅度大得多、可能更耐受漂移，但同样要在交错设计下复核。
⇒ **跳发读数一律不得相减**（40.47 / 40.24 / 39.74 相隔数小时）。
✅ **A/A 那一发把干扰变成了仪器**：两臂同配置 ⇒ 两臂之差就是那 20 分钟的机器漂移，直接当噪声地板打印（因此 A/A 不跳 perf）。

## 4.16 窗口执行单（A/A → t4 → t5，就绪待排卡）

`tools/kt_offload/oncard/RUNBOOK_window.md`；三件均带 `ONCARD_EXPECT_SO_MD5=2b58b578e4fc720ef30f4528c5a974fa`（.so 不是这份就 rc=3 拒绍起飞）。
① **A/A**（先校仪器）：两臂均 packed=off、**同一个启动点**（自检数启动点必须=1，没有参数漂移的入口）；三档 eval：`c8`（与 A/B 同参同位置）、`c8b`（同一次服务内第二发，把「批组成」与「换了进程」拆开）、`c1`（并发 1）。
判定表由 `aa_report.py` 打印：并发8 分歧+并发1 相同 ⇒ 批组成 ⇒ **服务层逐位判据无区分力**；两档都分歧 ⇒ 另有来源（不得归给 packed）；两档都相同 ⇒ A/B 的 31/50 是 packed 真改了数值；并发1 分歧而并发8 相同 ⇒ rc=1 先重跑。
② **t4**：node_us / window_us / ceiling（全部相对本发实测，不再引用作废单价）。
③ **t5**：`auto_dispatch_capture` 能不能救回侧流；两臂**分进程**（失败的捕获会污染同进程的后一次）；`--arm off` 预期仍报 107025 作阳性对照——**没有它，「成功」可能只是参数没生效**。

## 4.18 🟢 phase 2（2026-09-22）：rebase 到 clean `a0e673186` + 换包 + 首批卡上读数

> 维护者：`dsv41_vllm_single_card_phase2`。本节读数全部**当场内联**；run 目录都是未跟踪产物（k.16），
> 出处标「可复验」= 目录仍在且可重跑；丢了就按「已记录、不可再验」引用。

**rebase**（用户拍定目标先 `f85267ab8`、途中 clean 前进后改 `a0e673186`）：74 个提交零冲突重放；
`envs.py` 只剩我方三项；三个 clean 新提交都不碰 csrc。⚠ `69132e27c`（视觉 MoE router 融合，
`VLLM_ASCEND_DSV41_FUSED_VISION_ROUTER` 默认 "1"）**走在纯文本 decode 的每一层**（V4.1 40 层都建 `bias_vl`），
不改我方接口契约（仍返回 `(weights f32, ids i32)`，我方 route/merge 在其下游），但它调新 `.so` 的
`moe_gating_top_k_hash` ⇒ 又一条"不换包不能跑"的理由。
测试：kt_offload **297 passed / 0 skipped**；selftest_cpu 70、selftest_packed_ab 102；
models+attention 失败集合与 clean `a0e673186` **逐条相同（34）**，我方多 17 通过。
快照分支：`-pre-rebase-0922`（2dfbb3449）、`-rebased-f85267ab8`（393277913）、`-rebased-a0e673186`（ddbb42454）。

**换包**（用户授权；验收在 worktrees/ 外做，免得共享闸在验收期间对全线变红）：
- 依据：`ada1cc48e..a0e673186` csrc 零改动 ⇒ clean 的 `.so` `85331298…` 对应我方源码；旧 `.so` `2b58b578…`
  里批次 1 新符号 **0 命中**、clean 那份命中（旧 `.so` + 新代码 = 静默算错，有读数）。
- 四元组逐项与 clean 钉值相等：so `85331298bdeffee02b7c42c5658030ac` / vendor 786 文件 / 9.2.0-beta.2 /
  指纹 `b7f2b59a…4a719b2`；不重编。
- 验收（树外 `staging/moe-offload-a0e6`，卡 0）：**t0** 15/15 必检、6 变异臂全红、58 s；**t1** 57/57、21 臂、76 s；
  **t2** 14/14、1 臂、17 s；整网图模式 GSM8K **50/50**（err 0 / trunc 0 / stop 1.0，T=0 thinking 关，口径自检一致）。
  出处 `oncard_runs/20260922_115351_t0`、`_115534_t1`、`_115726_t2`、`p2_serve_0922_120351/gsm8k50`【可复验】。
- 落地：拷进 `worktrees/moe-offload`，**同一步**改共享闸钉值（build_artifacts `5c87ee2`）；旧产物备份
  `moe_offload_ref/pre_swap_artifacts_0922/`；正式树上复跑 t0（卡 1）PASS 59 s。
- 🔴 **跨这次换包的读数一律不可比**（构建指纹会拒绝给差值）。

**整网服务首读**（卡 0，图模式，KT_OFFLOAD=1 N=100 prefix，MAXLEN 32768，`p2_serve_0922_120351`【可复验】）：
- 起到就绪约 4.5 min；`candidate filter: 8 indexers, 5 take part` ⇒ **ds41/O6 路径在**；无 10701x。
- KV：available **27.40 GiB**，`GPU KV cache size 172,049 tokens / 5.25x`。⚠ 这个 token 数 = 满长并发 × MAXLEN
  （旧 42,733 = 5.22 × 8192），**容量以并发数读，两次几乎相同**；不要拿 172k 与 42.7k 比。
  `kv_budget_singlecard.py` 在新构建上对账通过：池块 **517,632 B**、167.0 KiB/token（L=32768）；
  L3 在 L=32768 下 **3.58×**（8192 下 3.85×，倍数随 MAXLEN 变）。
- **>16k 功能验通**（O15 前提：ds41 qli 在 `valid_s2%8==0` 时不钉最新块，本组读数都带着它，不是干净基线）：
  - LongBench-v2 真题 4 道（完整 22.3k–26.5k tok，服务 usage 与本地计数逐一相同）：完整臂 **4/4 对**；
    对照臂（只留尾部 12k）：**1 道判区分**（Nemotron-4 Table 2 题，截断臂原话 "I don't see any information…"），
    3 道截断后仍对 ⇒ 无区分力、不计证据。单发单选 ⇒ 那道 PASS 仍有 ~25% 蒙对概率。
  - 针扫描（真实语料填充）**9/9 命中**，实际 prompt **17,254–31,141 tok** 全部 >16384；最远一针在 31,141 tok 的深度 0。
  - 单卡 offload 下 >16k 请求耗时 **37–45 s/条**（22–26.5k prompt，max_tokens 128；首个读数，未分解 prefill/decode）。
  - 工具：`tools/kt_offload/longctx/`（`de7ee6cab` 入库）；读数 `p2_serve_0922_120351/longctx_check.json`、`niah.json`【可复验】。

**这一路抓到的四个"只在卡上暴露"的缺陷**（都已修、都补了会红的检查）：
1. `run_oncard` 起跑前的占卡采样在 `$(...)` 子 shell 里设全局量 ⇒ `set -u` 下第一发 t0 死于
   `PA_LAST_SAMPLES: unbound variable`（`fb06ddedb`）。旧自检只 `grep` 函数名 = 判据读代理。
2. launcher 的 .so 闸调用时 `$_TREE` 还是空 ⇒ **从没真比过 md5**（占位期望值让两种拒绝读起来一样）（`2b89f5699`）。
3. 树外 PYTHONPATH 改写用子串替换，`worktrees/moe-offload` 是 `moe-offload-vllm` 的前缀 ⇒ vllm 树被改没
   （`195f01808`，改按路径分量；共享 bringup:316 同病，已告知 flash_dev，未改共享件）。
4. A/B runner 不传 `WT`/vllm 树给 launcher ⇒ 会起**主 checkout 那棵树**；被租约 owner 不一致意外拦下（`70d38c240`）。

**共享闸现状**：`build_artifacts/shared_so_manifest.txt` 被 09-21 清理（`2e21d32`）删除 ⇒
`verify_shared_artifacts.sh` 对所有人 rc=1。用户决定不恢复、不改共享脚本；本线 launcher 只对这两条 ❌
**窄放行**（`dc3e7ec27`，fail-closed，放行记录落 `shared_gate_waiver.txt`）。

**新基线（A/A 缩减版：A1 + A2，用户定不跑 B 臂；卡 2，同配置两次独立起服务）**
配置：图模式、KT_OFFLOAD=1 N=100 prefix、packed 关、MAXLEN 8192 / MAXSEQS 8 / mnbt 4096、**串行（无侧流）**；
perf＝`perf_baseline.sh` bs1（并发 1、16 条、random IN 128 / OUT 256、ignore-eos、预热 1 轮丢弃）。
出处 `packed_ab/20260922_122648/armA1`、`armA2`【可复验】。

| | TPOT 中位 ms | 标准差 | 最小–最大 | TTFT 中位 ms | 吞吐 tok/s |
|---|---|---|---|---|---|
| A1（n=4，12:26 起） | **34.36** | 0.19 | 34.09–34.55 | 749.7 | 26.89 |
| A2（n=2，12:55 起） | **34.60** | 0.74 | 34.08–35.12 | 766.9 | 26.65 |

- ⇒ **bs1 TPOT ≈ 34.4–34.6 ms（≈26.8 tok/s）**。旧 40.47 ms 是换包前的构建、别的时段 ⇒ **不得相减当收益**。
- **噪声地板（跨重启，初值）**：中位差 0.24 ms（0.7%）；六轮极差 34.08–35.12 ≈ **1.0 ms（3%）** ⇒ 此后 A/B 差值 <≈1 ms 判不可分辨。
  n 小、单时段，属保守初值；每个 A/B 仍须自带交错轮次。
- **GSM8K**：A1 c8 / c8b / c1 各 **50/50**（112 s / 108 s / 280 s），平均输出 142–144 tok。
- **服务层逐位判据无区分力（半个结论）**：同进程 c8 vs c8b 文本逐位相同仅 **19/50**（只差批组成）；c8 vs c1 17/50、
  c8b vs c1 12/50、A1 c8 vs 卡 0 验收发次 19/50；答案全部一致。⇒ 服务层文本逐位一致**不能**用来判"改动是否改了数值"，
  数值判据留在 t0/t1/t2 层级。⚠ 没跑两发 c1 互比（用户为省时取消），"并发 1 是否自身确定"未测。
  旁证：旧 packed A/B 的"19/50 相同"在这把尺子下**不构成** packed 改数值的证据。

**decode profile（bs1 稳态，卡 2，同基线配置 + `KT_MOE_PHASE_TIMING=1`；`p2_prof_0922_131258/SUMMARY.md`【可复验，2.2 GB 未跟踪】）**
分析脚本入库 `tools/kt_offload/prof/{kd,nw,ktphase}.py`（区间并集 / NOTIFY_WAIT 步切分 / kt 相位解析）。
`prof_decode_breakdown.py` 在本版**结构性不可用**（`kernel_details.csv` 已无 `Model ID`/`Step Id` 列）。

| TPOT 分解（196 步） | 中位 ms | p90 | 占比 |
|---|---|---|---|
| device step 周期（profiling 下） | **38.51** | 43.47 | 100% |
| NPU 计算（区间并集） | **19.49** | 19.59 | 50.6% |
| 等 CPU（NOTIFY_WAIT） | **15.88** | 19.00 | 41.2% |
| 其余 host/launch/小空隙 | **2.81** | 6.58 | 7.3% |

- 对账缺口 0.33 ms（0.9%）；profiling 膨胀 **+10.9%**（同服务不开 profiler bs1 中位 34.50–35.04，n=3）。
- device 忙 **48.7%**；<100 µs 碎空隙合计仅 0.69 ms/token ⇒ 碎下发不是问题。
- 算子 top（ms/token）：QuantBatchMatmulV3 4.30 · GroupedMatmul 1.86 · MatMulV3 1.66 · TransposeQuantBatchMatMul 1.33 ·
  MhcSinkhorn 1.03 · MhcPre 0.88 · SparseFlashMla 0.85 · DynamicMxQuant 0.80 · Cast 0.78 · MoeInitRoutingV3 0.54。
- **NOTIFY_WAIT 每层中位 353.6 µs**（40 层求和 14.39 ms/token）；host-func 计数每层每步恰一次、无丢层。
- **驻留 GMM 窗口 42.7 µs/层**（全网 1.81 ms/token）⇒ 形态 C / 侧流在本层窗口内**最多藏住等待的约 12%**（≈1.7 ms/token 量级，推断上限）；§4.9 结论不变。
- **kt CPU 分段（首次卡上实测，`KT_MOE_PHASE_TIMING=1`，不改代码、不改数值）**：每层每 NUMA 中位 236.7 µs =
  up_gate 140.4（59%）+ down 78.2（33%）+ 其余 <20；每层取慢的 NUMA **246.1 µs**，40 层 **9.59 ms/token**（profiler 关时 8.88）。
  对 §4.3 曲线（k=4.54 预测 266.4 µs）**快 7.6%** ⇒ **CPU 地板 ≈9–10 ms 被卡上证实**。
- 🔴 **新靶子：接缝开销**。NOTIFY_WAIT 353.6 − kt 计算 246.1 = **107.5 µs/层 ≈ 4.3 ms/token（等 CPU 的约 27%）不是 kt 计算**
  （D2H 等待 + host-func 派发 + Python + kt 提交/唤醒 + 跨 NUMA 归并 + H2D）。这部分不受「不许跨层」约束的上限限制。
- ⚠ 两条：① `/stop_profile` 后 EngineCore **静默退出**（`Stop profiler while current state is RECORD`，无 traceback；产物完好）
  ⇒ 采 profile 的服务当一次性用；`stop_by_port` 需带 `STOP_PGID` 才停干净（k.13 复现）。
  ② 采集期间（13:12–13:47 之间）有**别的容器**上了卡 0–3（各 84,852 MB），不能排除与 13:28 窗口重叠；NPU 计算项极稳
  （四分位无趋势）⇒ 按"未受明显影响"用，但本发绝对值不作最终基线。

**待办**：launcher 起服务不自动盖构建指纹（本次事后补盖）。

### 4.18.1 优化队列（用户 2026-09-22 定：主指标仍是 bs1 延迟；本 session 只做 ①–③）

| # | 项 | 预计收益 | 状态 |
|---|---|---|---|
| ① | **接缝拆解**：NOTIFY_WAIT 353.6 − kt 计算 246.1 = 107.5 µs/层（≈4.3 ms/token，实测差值）逐段打时间戳，再对最大段下手 | 上限 4.3 ms | 🟡 埋点已入库，**未上卡** |
| ② | t4：节点成本 + 窗口（含 shared expert / dispatch 能否挪进窗口） | 决定 ③ 值不值 | ⬜ |
| ③ | **实现侧流**（先修 §4.19③ 的进程级缓冲守卫 → 服务里打开 → 交错 A/B + 噪声地板） | 上限 ≈ 窗口 83.2 µs/层 × 40 ≈ **3.3 ms/token**（推断，t4 实测窗口外推） | ⬜ 形态 C 已 ⛔（节点费 105.2 > 窗口 83.2） |
| ④ | L3 容量（SWA block 32→128） | 32k 下 3.58×；TPOT ≤1 ms | 📒 记账，本 session 不做 |
| ⑤ | 频率放置（M0c） | 0.9–3.1 ms（推断） | 📒 记账，本 session 不做（需多卡采频率表） |
| ⑥ | 吞吐轴 MAXSEQS 8→32 | ×1.3–1.8 吞吐（推断：均匀路由下 8 条 ≈75 tok/s → 32 条 ≈128 tok/s，单请求 TPOT ≈106→≈250 ms） | 📒 记账，本 session 不做；硬前提 ④；若主指标改成单卡吞吐应提到最前 |

① 的仪器（2026-09-22，只写代码、**没碰卡**）：`VLLM_ASCEND_DSV41_KT_SEAM_TIMING=<采样秒>` 开
（默认关；开时只读时钟，不改设备操作、不改输出字节），回调内分段见
`vllm_ascend/kt_offload/seam_timing.py`；kt 的 submit/sync 靠**替换 `wrapper.cpu_infer` 的 Python 探针**拆开
（`run_pinned_forward_sync` 本身是 Python，experts_base.py:717-772）⇒ **不改 kt C++、不重建 wheel**。
离线 `tools/kt_offload/prof/seam.py`（对账 `B+C+D+E+F ≈ T_cb`，差值 `A+F_dev = NOTIFY_WAIT − T_cb`）；
上卡执行单 `tools/kt_offload/oncard/RUNBOOK_seam.md`。**注意 device 侧的 A/F 分不开，只能作为一项**。

另记账：M4（GSM8K 1319 全量精度）待最终形态再跑；O15 等算子方。

## 4.19 🔴 2026-09-22：t4 修表后的窗口/节点成本，与「侧流捕不了」的作废

> 两条旧结论作废，都是**仪器坏了**而不是平台限制。产物：`oncard_runs/20260922_143217_t4`、
> `_143319_t5`(off)、`_143359_t5`(on)、`_143512_t5`(off_evt)、`_143552_t5`(on_evt)、`_143954_t5`(off_evtw)【可复验】。
> 代码：`6627e17f8`（只动 t4/t5 两个用例文件）。

**① t4 的图臂计时以前测的不是执行时间 ⇒ 14:15 那发全部作废**
根因：捕获跑在 torch_npu 自己的 capture stream 上，而用例在 `with torch.npu.graph(g)` **之外**取
`current_stream` ⇒ host-func 被下发到非捕获流，**节点从未进图、在捕获时就地跑掉**（"500 µs 胖节点量成 −1.3 µs"）。
生产路径是在捕获内取流的（`layer.py:347`）。修法照生产；并新增判据：**捕获期回调 0 次、每次 replay g_one +1 / g_two +2**
（没有它，"节点没进图"和"节点不要钱"读起来一样）。另加设备时间测量（blocker + `Event(enable_timing=True)`，
并证明 host 跑在 device 前面，实测 100% 样本）。容差一个没动。

| 对照 | 修前（墙钟） | 修后（设备时间） |
|---|---|---|
| 阳性对照 500 µs 胖节点 | −1.3 µs ❌ | **552.1 µs** ✅ |
| 阴性对照（无节点图臂 vs eager 纯 NPU） | ratio 0.40 ❌ | **ratio 1.02** ✅ |
| 1/2 节点可加 | 空对空 | 52.6 vs 101.3/2=50.7 ✅ |

⇒ **node_us 52.6 µs/节点、window_us 83.2 µs/层**（VERDICT PASS，10 项检查全绿）。本发 `cpu_us 288.5 µs`（t4 口径）
对 profile 的 353.6 µs 比值 1.23，同量级。
🔴 **形态 C 由此判负**：它要**两个**节点 = 105.2 µs/层 > 窗口 83.2 µs ⇒ ceiling 为负（≈−22 µs/层）。
（`t4_model.ceiling()` 只扣一个节点，打印的 "1.22 ms/step" 是单节点口径，**不要引成形态 C 的收益**。）

**② 🔴「本平台副流捕不了（107025）」作废 —— 侧流路线复活**
真因：t5 自己在捕获里抛 `no preallocated int64 id buffer`（见③），**107025 是次生**。补上缓冲后：

| 臂 | auto_dispatch_capture | 副流尾 event record | 捕获 | replay 判据 |
|---|---|---|---|---|
| off（现状参数） | False | 无 | ✅ | **全过**：捕获期 host-func 0、3 次 replay 每层各 +1、逐位等于串行臂、replay 两两不同 |
| on | True | 无 | ✅ | 全过 ⇒ **该参数在此无区分力** |
| off_evt / on_evt | – | record，**无人等** | ❌ 107025 | — |
| off_evtw | False | record + 主流 `wait_event` | ✅ | 全过 |

⇒ **107025 的真正触发条件：副流上最后一次 event record 必须被主流 join**（报错文案只说"最后一个任务不是
event record"，实测"补一个没人等的 record"反而必崩）。生产 `_join_side_stream`（`layer.py:436-438`）本来就是
record+join，一直是对的。⚠ t5 只答"能捕获+重放正确"，**没测性能**；config 里 `overlap_side_stream` 仍是拒绝态。

**③ 🔴 生产 bug（已定位，未改）**：`layer.py:401-404` 的"缓冲已准备"是**进程级** `_CAPTURE_BUFFERS_READY`，
而 `_ids_i64` 是 `_prepare_capture_buffers` 按**层实例**填的（`layer.py:416`）⇒ 一个进程里只有第一个建的层有缓冲，
第 2 层起为空。侧流要在服务里用，必须先修它。

**④ 取舍更新**：侧流的 host-func 节点开销付在**副流**上、不占主流 ⇒ 上限 ≈ 整个窗口 **83.2 µs/层 ≈ 3.3 ms/token**
（推断：按 t4 实测窗口 × 40 层，未计侧流自身的 fork/join 与事件开销，也未上服务实测）；形态 C 为负。
⇒ 队列 ③ 改为「**实现侧流**（修 ③ 的守卫 → 服务里打开 → 交错 A/B + 噪声地板）」，形态 C 记为 ⛔（实测节点费 > 窗口）。

## 5. 边界（不碰）

- 不重编 `vllm_ascend_C*.so`、不改 `vendors/`（起飞闸查全部 worktree，会挡住所有人）；要重编先找 flash_dev 改闸。
- `serve_v41_bringup.sh` 的 `MIN_EP=3`（:127）/ `DP<3`（:204）：与 integration 按「卡上专家 + 其余权重」重算，不直接放开。
- 共享脚本（`env_dsv41.sh`、`serve_v41_bringup.sh`、`assert_trees.py`、`_env92_core.sh`、`setup_cann92_env.sh`）改前找 flash_dev。
- `/dev/shm/engram_stage`、`build_artifacts/vendor92/`、`cann92/`、`dsv41-vllm-ascend-hif8/` 只读。
- 碰卡先 `build_artifacts/card_lease.sh acquire`；卡 1 不用；Triton 临时目录 `TMPDIR=TMP=/dev/shm/dsv41_tmp`（不 export TEMP）。
- 代理只给 GitHub/Claude，其它外网 unset proxy；不用 `pkill -f`。

## 6. 分工

- 协调（dsv41_vllm_singlecard）：本文维护、交叉验证、排卡、对接 flash_dev / integration / dsv41-model / dsv4_a5_single_card。
- kt-kernel agent：M0a、M0b、kt 侧改动。
- vLLM 集成 agent：M1 → M2（同一 agent 保持上下文）。
- 性能 agent：M3（M2 之后）。M0c 经 flash_dev。

## 7. 待决

| # | 问题 | 状态 |
|---|---|---|
| D1 | 用哪张卡（0-3 integration 服务；4-7 被别的容器占） | ⏸ 用户：到拉服务时再定 |
| D2 | 能否与 integration 四卡服务同机并存（见 §2，账面看可以，需实测） | ✅ 用户：必要时经 flash_dev 协调卡与时段 |
| D3 | kt push 目标 = origin(wenxuewuhd/ktransformers-AK)？push 后可否直接摘两个提交 | ✅ origin；用户已推 dsv41_vllm_offload_dev(105359b) |
| D4 | M3 优化目标：bs=1 延迟 vs bs=32 吞吐 | ✅ 先 bs=1 延迟 |

## 变更记录

- 2026-09-18：建档。调研完成（两个 agent + flash_dev / dsv41-model / dsv4_a5_single_card 回复）；
  ktransformers 分支建好；§2 更正：engram 与 integration 共享，并存时增量只有专家 ≈195 GiB。
- 2026-09-18：§2 加 NUMA 放置决定（专家两节点均分，engram 不动；单节点放不下两者）。
- 2026-09-18：用户定 D2/D3/D4（卡到时再定；并存经 flash_dev 协调；origin 已推；先 bs=1）；新增 P 项 NPU 算子耗时分布；启动 M0a/M0b、M1 设计。
- 2026-09-18：M0c 已转 integration（flash_dev 附三坑：图 replay 不跑 Python ⇒ eager 采；DP 加总核全局 id；频率依赖负载 ⇒ 驻留集由启动配置给出、可换，不做在线调整）。
- 2026-09-18：M0a/M0b 完成（已核 commit、wheel sha256、扫描与判决行）。新发现：激活精度口径差异（NPU A8 vs kt bf16）。kt 构建依赖：容器缺 pkg-config/gguf，已在私有目录绕开（`m0_evidence/kt_build.sh`）。
- 2026-09-18：M1 代码完成。设计要点：驻留集 prefix/frequency/explicit 三选一；逻辑号→紧排槽位（不走 EPLB）；ALLGATHER dispatcher 是进程级单例 ⇒ 槽位数全模型取各层驻留数最大值；分流在 `_select_experts` 之后（逻辑号、随机替换已完成），CPU 路由改写为「槽位 0 + 权重 0」；CPU 输出不再乘 1.5（router 已乘）、不含 shared。前置：TP/DP/PP=1、无 EPLB/MTP/mix_placement。
  ⚠️ 首个上卡步骤要先把 ALLGATHER + W4A8 在 V4.1 上【offload 关】单独验一遍——这条路关掉 offload 也会走、从没跑过。
  🔧 协调方失误已修：`b48ca8c3b` 用 `commit -a` 误带了 agent 未提交的代码；已拆成 6ec819fc5/548f4907b/24881294e，最终树哈希不变（3e554ad9）；旧头留在 `dsv41-moe-offload-presplit-backup`。
- 2026-09-18：系数定为 0.7004 GiB/专家（flash_dev 原 0.718 含 MTP，已更正）；integration 交付 KT_OFFLOAD=1 单卡起法 + kt_offload_gate.py（§4.1）。待看：官方 vllm-ascend releases/v0.27.1rc 的 engram_offload.py 与 MegaMoE 合入（df1286cca），review 会话出单列一节后比对设计。
- 2026-09-18：用户规矩——单卡改动只在本线自己的树/脚本；integration 撤回共享脚本改动，交接物入库 tools/kt_offload/（6ebc88ff8）；M0c① 计数钩子由 M2 agent 接入本树。
- 2026-09-18：M2 代码与计数钩子完成（50f37e662 / 1c047fee7 / kt 292be81）。核实：本树图捕获 _dummy_run 为 is_profile=False ⇒ 捕获时不随机路由（只有 profile run）。计数器修了原件 ALLGATHER 下每 rank 都数 rank0 token 的错。
- 2026-09-18：计数器加身份判据（29d3e9668，旧切片变异下新判据红、求和判据绿）；kt wheel 重建；单卡脚本 EAGER 开关。**不占卡的工作已全部完成**，等：卡（多卡 4 张采频率表 / 单卡 1 张上卡验证）、官方 v0.27.1rc engram_offload/MegaMoE 比对。
- 2026-09-19：多卡性能基线到（bs1 TPOT 43.28 ms）。官方代码 review 两条相关：① slot cache 在挂 P/D 或 offload **connector** 时每请求前 3 位出错 —— 本线 kt offload 是层内 host 计算，不走 vLLM connector（待在设计文档写明并核无间接挂载）；② engram 缺陷 N1 有遮蔽效应 ⇒ M2 验证必须多请求并发。
  ⚠️ 协调失误：flash_dev 给了卡 0 做 T0 单层用例，我没及时响应、脚本也没写好 ⇒ 窗口让给 integration。教训：卡到之前把上卡脚本备好，排卡时能直接报时长。上卡测试脚本已派 agent 编写（tools/kt_offload/oncard/）。
- 2026-09-19：⏸ 用户指示暂停，等多卡版本 fix。上卡脚本 agent 已停（无残留），未持有租约。恢复步骤：同步新 clean → 写 oncard 脚本 → 排卡 T0。
- 2026-09-19：▶ 恢复。vllm 快进 8b24cc594e；vllm-ascend 变基到 1b1ad396b（唯一冲突 envs.py 两边都保留），kt_offload UT 95 过（新 clean 上收集失败 = 反向对照），mock 回归与新 clean 失败集合逐条相同；N3 上限 16384；单卡脚本副本同步共享原版的逐卡 dmesg 闸。上卡脚本 ce0c49f4d（§4.2）。对标改为 bs1 26.84 ms。
- 2026-09-20：clean 唯一新增是 vllm 侧 a9686bc6e6（engram hash 前导以 _prefix_max 替换 AI CPU cummax，与原实现逐字节等价，修的是 AI CPU 调度段错误）⇒ 我们【快进即可，无需变基】，vllm-ascend 侧 clean 未动（仍 1b1ad396b，我们落后 0）。快进后 kt_offload UT 95 过。本线不引入任何 AI CPU 算子（kt 是 host 线程计算 + host-func 回调）。用户指示：多卡优化优先，单卡暂停。
- 2026-09-20：用户定顺序——先上卡验 t0/t1/t2，再实现侧流重叠（现仅占位开关 config.py:99 设 true 即报错）。侧流判据为开关两态逐位相同；近似类优化默认不做。
- 2026-09-20：CPU 侧实测完成（§4.3）。结论：CPU 地板 ≈10.4 ms/token，驻留数被 HBM 闸限死在 ≤113，加驻留降不下来；单卡 bs=1 追平多卡 26.84 ms 可能性不大 ⇒ 性能目标待与用户重谈。
- 2026-09-20 上卡（卡 3，三发各自持租约、用完即放）：t0/t1/t2 全 PASS（产物 oncard_runs/20260920_024812_t0、_031221_t1、_031441_t2）。
  🔴 **上卡抓到生产缺陷**：`route()` 返回 (ids, weights)，**所有调用点按 (weights, ids) 接**，生产 forward_impl 亦然 ⇒ 整网路由全错；CPU 单测按返回顺序解包，照不到调用点绑定。修：具名返回 + 入口 dtype 契约（就地 raise，不 `.to()` 静默纠正）+ 禁止位置解包的 AST 检查（165f73d2d）。**教训：同一个函数的返回顺序，测试和调用点都用同一个假设时，测试就不是独立证人。**
  ⚪ 判据变更（dc6dba7ec，红之后改的，协调方已复核）：arm B tier-B 由"逐列相对误差"改为"最大绝对偏差 / 参考最大值的 bf16 ULP"，允许量 max(2×地板, 4 ULP)。理由：T 小时列内近抵消 ⇒ 分母趋 0，同一发里该度量自己的地板读到 1.333(133%)；而两个合法仿真在 T=1 舍入一致 ⇒ 地板读 0（"没测到"不是"很紧"）。复核要点：kt 实测最大 0.72 ULP，允许量 4 ULP ⇒ **余量 5.5×，取 2 ULP 也能过 ⇒ 常数不是为通过而调**；变异臂 6/6 仍红；tier A 未改；旧比值保留为诊断。
- 2026-09-20：§4.5 KV 回收评估完成（未上卡，只读日志 + CPU）。机制在我们自己的两发上核实并逐项对账
  （43 组 / 40 个单层 SWA 组 / 池块 542,208 B / 54,131 块 / 29,350,261,248 B，token 与 concurrency 逐个吻合），
  另用本树被 patch 的 planner 在 CPU 上实跑 32 与 128 两档。三条更正：655 KiB 复现不出（实为 670.7 KiB，倍数 1.95×）；
  这 1.95× 是 max-model-len 8192 vs 16384 造成的，与单卡/多卡无关；§4.4 的 115.2 与 7.8 两个常数都不对
  （实测 110.84 与 1.36，方向相反差点抵消）。KT_DEVICE_FACTOR 就地实测为 1.024。
  与理论 KV 对账：bf16 是 ×3.25 不是 ×4（index_k 走 fp8+fp32 scale 只 ×1.94）。
  实现 L3（`VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE`，默认关）：容量 3.85× 或省 20.2 GiB；
  L4 撤回——ring 槽位用全局池块号，缩区会越界写。上卡方案见 §4.5.5（1 卡 / 2.5–3.5 h）。
- 2026-09-20：起飞闸干跑通过（N=100：HBM 80.4/90.2，主机 239.9/450.2，两 NUMA 各 99.5 够）。新增 §4.4 专家数账、§4.5 KV 回收（review 交办，排在 bring-up 之后，收益按并发轴报而非 TPOT）。确认单卡 MoE 走 ALLGATHER 且 DP=1 下集合通信全被 dp_size>1 守卫挡掉（prepare_finalize.py:471/:511/:570）⇒ 无 MC2、无 HCCL，省下 HCCL_BUFFSIZE 的 3.61 GiB。
- 2026-09-20 整网 bring-up（见 §4.6）。起飞链补丁：DP=1 下 engram 守卫用 2 个测试进程（740085d43）。
  🔴 **§4.4 的两个常数错了（已由 KV agent 在实测上更正）**：卡可用不是 128×0.9=115.2 而是 **110.84**（卡报 123.16 GiB）；单卡工作区不是推断的 7.8 而是**实测 1.36 GiB**；专家+KV 额度 = **99.08 GiB**（原写 97.0 是两个方向相反的错误差点抵消出来的巧合）。
  🔴 **"单卡每 token KV 655 KiB、比多卡差 1.9×"这条要更正**：重算是 **670.7 KiB**，且 1.95× 的来源**不是"单卡"而是 max-model-len 8192 vs 16384**（SWA 每请求预留 ~257 块与长度无关，分母翻倍分子不动）⇒ 正确说法是「**短 max-model-len 摊得差**」。
- 2026-09-20 晚：M3 侧流失败并定位到「fork 到另一条流再进捕获 ⇒ 107025」；两条护栏教训（读代理 / 替身太弱）；形态 C 已审阅但**暂停实现**等实测 C；packed_transfer（5→3 操作）就绪待上卡。kt wheel 统一到含 033ea76（sha 1b8e9b21…）。kt_offload UT 166 通过。
- 2026-09-20 晚：用户定**不允许跨层**（§4.9）。三条派生数字作废（§4.10）；packed A/B 实测中位 −0.74 ms 但 p99 恶化（§4.11）；平台结论被 sglang 反例推翻、待对表（§4.12）。
- 2026-09-21：共享 .so 重建分发（§4.13）⇒ 旧读数标注构建、产物加指纹、判读跨构建拒绝给差值；新增 §4.14「只在卡上暴露」风险类别。t4/t5 排在换 .so 之后、integration 4 卡回归之后。
- 2026-09-21：⛔ .so 不换了（撤回，见 §4.13 顶部）；旧读数无需脚注、不重测基线。产物盖指纹照做。新增通则：分发二进制前必须验证依赖的源码改动在被分发的树里——只搬一半的分发会产出「测了但什么都没测到」的漂亮报告。
- 2026-09-21：仪表三条更正（§4.15）：等卡循环从没等过（小写 grep，且解释掉一批「环境毛病」旧观察）、nproc 恒为 1、机器自漂 2.5× ⇒ A/B 改交错轮次 + 噪声地板，没尺子就判 fail；packed 结论降级为「未测到收益」。.so 分发再次推后（改成树外验收），校验值保持旧值。
- 2026-09-22：用户拍定 **rebase 到 clean `f85267ab8`**（不重新分叉；理由：保住 66 个提交里的「为什么」）并**交接给新 session**；实测代价：1 处相邻行冲突、283 条测试全过、依赖的接口在新 clean 上逐位未变。风险与队列见 `HANDOFF_next_session.md`。新事实：`INDEXER_QUANT_CACHE` 默认已为 1 且不能单独关 ⇒ 换算子包是硬前提；ds41 缺失**不报错只退回**；多卡对标 26.84→**≈22.7 ms**（四卡口径，单卡无读数）。
- 2026-09-22（phase2）：rebase 到 clean `a0e673186`（74 提交零冲突）、换包（clean 同一份四元组，树外验收 t0/t1/t2 + GSM8K 50/50 后落地并改钉值）、整网图模式首读（KV 5.25× 并发 @32k、O6 路径在、>16k 验通：LongBench-v2 1 道区分 + 针 9/9）。四个卡上暴露的缺陷已修。见 §4.18。
- 2026-09-22（phase2）：t4 修表后 node 52.6 µs / window 83.2 µs（形态 C 判负）；🔴「副流捕不了」作废——真因是我们自己的 int64 缓冲守卫 bug，现状参数下真实 3 层 fork/join 可捕获且 replay 逐位正确；107025 的真条件是「副流尾 record 必须被主流 join」。见 §4.19。
