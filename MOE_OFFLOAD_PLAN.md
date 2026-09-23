# DSV4.1-flash 单卡 MoE offload（vLLM + kt-kernel）计划与跟踪

> 👉 **2026-09-22 现状**：rebase（目标最终是 clean **`a0e673186`**，不是 `f85267ab8`）、换包、整网首读**都已完成** ⇒ 见 **§4.18**。
> `HANDOFF_next_session.md` 的 §3（rebase 怎么做）**已执行完毕，只剩坑与纪律可读，状态别照抄**。
> 两条线在岗：`dsv41_vllm_single_card_phase2`（§4.18.1 的 ①接缝拆解／②t4／③t5，用卡 0–2）；
> **`dsv41_single_card_stream_prefill`（流式 prefill + 动态热专家，用卡 3；计划另见 `STREAM_PREFILL_PLAN.md`，尚未并入本文）**。

> 本文是这条线的**唯一跟踪文档**。⚠️ 原维护者 `dsv41_vllm_singlecard` 已交接（上下文满），
> 现由在岗的两条线维护（phase2: §4.18／§4.19；stream_prefill 的计划暂放 `STREAM_PREFILL_PLAN.md`），公共节改动前先说一声。
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

- ⛔ **「max-model-len 硬顶 16384」已作废（2026-09-22）**：主干交付 O6 后守卫解除，§4.18 实测 MAXLEN 32768 起得来、
  针扫描实际 prompt **17,254–31,141 tok 全部 >16384 且 9/9 命中**。脚本默认 8192 仍是默认值，不是上限。
  （原文：N3 守卫 `ceil(floor(L/r)/8) ≤ 2048`，层 20/24/28/32/36，r=1；与 TP/DP/offload 无关。）
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

> 🔴 **2026-09-22 再更正：下面这条 09-21 的撤回【本身已经过期】。`.so` 最终【换了】** ——
> 现在树里是 **`85331298bdef`**（9.2.0-beta.2 / vendor 786 文件），钉值已改，过程与验收见 **§4.18**。
> ⇒ **「旧读数继续有效、无需重测基线」这句现在是假的**：跨这次换包的读数一律不可比，基线以 §4.18 的
> **34.4–34.6 ms** 为准。下面整段**只作为历史与教训保留**（"只搬一半的分发"那条通则仍然成立）。

> ⛔ **2026-09-21 撤回前半段（历史，已被上面推翻）：共享 `.so` 最终【没有换】。** 我们树里仍是 `2b58b578e4fc`，钉值未动，闸未红，**旧读数继续有效、无需「适用构建」脚注、无需重测基线**。
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

`tools/kt_offload/oncard/RUNBOOK_window.md`。
🔴 **2026-09-22 更正**：下面写的 `ONCARD_EXPECT_SO_MD5=2b58b578e4fc720ef30f4528c5a974fa` 是**换包前的旧 `.so`**，
照抄会让三发全部 rc=3 拒绝起飞。现值是 **`85331298bdeffee02b7c42c5658030ac`**，且**以 `RUNBOOK_window.md` 里的为准**
（那里已更新）——**不要引用本节这个常量**。
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

**decode profile（bs1 稳态，卡 2，同基线配置 + `KT_MOE_PHASE_TIMING=1`；`p2_prof_0922_131258/SUMMARY.md`）**
> 🔴 **出处降级（2026-09-23）**：这发的**原始 trace 已于 09-23 根盘写满时删除**（腾 1.7 G）⇒ 本节与 §4.22 的
> F1/F2/F3 由【可复验】降为【**已记录、不可再验**】。仍在的是聚合结果（`SUMMARY.md`、`kernel_agg.*`、
> `tpot_decomp.txt`、`kt_phase_*`、`devgaps` 相关）与本文内联的数；**要再算新的口径必须重采**。
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
| P2 | kt 内联（`KT_CPUINFER_INLINE`，微基准 0.45 ms） | 📒 **跳过不测**（用户 2026-09-23）：P1b 已证削 host 侧换不来 TPOT，P2 的 10 µs/层同族 ⇒ 预期 0；代码留树、默认关 |
| ⑥ | 吞吐轴 MAXSEQS 8→32 | ×1.3–1.8 吞吐（推断：均匀路由下 8 条 ≈75 tok/s → 32 条 ≈128 tok/s，单请求 TPOT ≈106→≈250 ms） | 📒 记账，本 session 不做；硬前提 ④；若主指标改成单卡吞吐应提到最前 |

① 的仪器（2026-09-22，只写代码、**没碰卡**）：`VLLM_ASCEND_DSV41_KT_SEAM_TIMING=<采样秒>` 开
（默认关；开时只读时钟，不改设备操作、不改输出字节），回调内分段见
`vllm_ascend/kt_offload/seam_timing.py`；kt 的 submit/sync 靠**替换 `wrapper.cpu_infer` 的 Python 探针**拆开
（`run_pinned_forward_sync` 本身是 Python，experts_base.py:717-772）⇒ **不改 kt C++、不重建 wheel**。
离线 `tools/kt_offload/prof/seam.py`（对账 `B+C+D+E+F ≈ T_cb`，差值 `A+F_dev = NOTIFY_WAIT − T_cb`）；
上卡执行单 `tools/kt_offload/oncard/RUNBOOK_seam.md`。**注意 device 侧的 A/F 分不开，只能作为一项**。

另记账：O15 等算子方。

### 4.18.2 合版本的门（用户 2026-09-22 定）

**合版本（侧流 + 接缝优化合成一个配置）的前提，三条：**
1. 侧流在**同时段交错** A/B 里测到 **>1 ms/token** 的 TPOT 收益（≤ 噪声地板 ≈1 ms 判不可分辨，不算收益）；
2. 侧流过精度三层判据（③ 同臂并发 1 两发 logprob 逐位建地板 → ② 整网 `max_tokens=1` + top-5 logprob 逐位 → ① 单层逐位，含**去 join 变异臂**）；
3. 接缝侧有实测可省量（微基准 `cpuinfer_threads` 扫描有拐点 → 整网确认）。
   ⚠ 第 3 条不成立也可只合侧流，但报告里要写明"接缝未取得收益"。

🔴 **2026-09-22 夜追加（§4.23 的教训）：单层逐位绿【不构成】整网可用的证据** ——
侧流单层 9/9 逐位绿，整网仍有 3.1% 的 decode 步整步算错；合版本前必须有**整网** logprob 逐位这一关。

**合版本的内容（用户 2026-09-22 追加，09-23 扩充）：侧流 + 回调/派发侧的优化（若有收益）+ **P2 kt 内联** 合成同一个版本，最后一起测 GSM8K。**
⇒ kt 侧只重建**一次** wheel（`run_inline` + `KT_WORKER_SPIN_MS` + P1b 若需要的轮询线程）、只重跑**一次** t0/t1/t2；
切换点只有 `tools/kt_offload/kt_build.env` 一处（§4.24）。各开关**分臂测**，不许混在一发里（混了分不清谁的功劳）。
⛔ 不进合版本：报告线程绑核（三次未站住，上卡实测反而慢 0.7–0.95 ms，§4.27 前一节）。
⇒ 只有侧流通过、回调那条没做出来时，可以先合侧流，但报告要写明"回调侧未取得收益/未实现"。

**合版本后的精度验收：GSM8K 50 题（贪心、thinking 关、同 §4.1 口径）⇒ 过了再跑 1319 全量（M4）。**
⚠ 时间差：回调那条 2026-09-22 夜仍在调研，当晚跑的 1319 是**只含侧流**的版本（用于二分定位），
合版本之后**需要再跑一次 1319**。
🔴 判读口径：**单卡 offload 没有 1319 历史基线**，唯一可比的是多卡带 **1276–1281 / 1319（96.74–97.12%）**；
同代码同包跨重启噪声实测 **11 题翻转** ⇒ 落在带内 = 无可检出回归；差 ≤11 题 = **不可分辨**，不得读成"更好/更差"。
1319 全量要数小时，只在最终形态上跑一次。

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

## 4.20 🟢 接缝拆解首发（2026-09-22，卡 2）：三分之二是 ACL 派发/回通知

> 产物 `oncard_runs/20260922_144513_seam`（未跟踪，读数内联；`/dev/shm/dsv41_prof_seam` 的原始 profile 随时会没）。
> 同发三组数：`KT_MOE_PHASE_TIMING=1` + `VLLM_ASCEND_DSV41_KT_SEAM_TIMING=5` + profiler。配置逐项同 §4.18 基线（cfg.json 直接拷 armA1）。

| 段 | 中位 µs/层 | ms/token | 占接缝 |
|---|---|---|---|
| **A+F_dev＝NW − T_cb（ACL 把回调调起 + 回调返回后通知 device）** | **78.0** | **3.83** | **67%** |
| C＝kt_pre 15.60 + submit 2.40 + between 1.45（kt 的 Python 前奏为主） | 19.5 | 0.79 | 14% |
| E＝sync − D（worker 唤醒 + 跨 NUMA 归并写回） | 18.2 | 0.73 | 13% |
| B pre 2.77 + F post 2.59 + kt_tail 2.39（我方 Python） | 7.8 | 0.31 | 5% |
| （D＝kt 计算，不属接缝） | 239.2 | 9.17 | — |

- 本发自测：NOTIFY_WAIT **353.8 µs/层**（§4.18 为 353.6，两发互证）；`NW − D` 求和 **5.73 ms/token**（§4.18 口径 4.80）。
  差异方向与埋点开销一致，但**未证明**；两发主机 loadavg 也不同（27 vs 2.5）。
- ⚠ **埋点开销"没测出来"而非"没有"**：开埋点 bs1 TPOT 35.49 ms（n=3，σ0.04），对基线 34.36/34.60 是 +0.89/+1.13 ms，
  压在 ≈1 ms 噪声地板上判不可分辨；但 3 轮全高于参照的每一轮，而账面预期只有 0.06 ms ⇒ **可能有 0.4–0.8 ms 真实开销**，
  引用本节分段时带上这条。
- **层 2 离群**：NW 1071 µs 而 cb 正常 298 ⇒ 单层 A+F 773 µs（占 0.77 ms/token）。剔除后 A+F 3.06 ms/token、中位不变。
  猜测是步首 report 线程被抢占/GIL，**未验证**——它与"A+F 是最大段"指向同一个可操作方向。
- 🔴 **两条仪器问题（修前的读数别引）**：
  ① `seam.py` 的对账判据把"各段中位之和"和"cb 中位"比 —— 中位不可加，逐次回调各段是首尾相接的、和恒等于 cb
     ⇒ rc=1（24/40 层）是判据的问题；残差正负都有（−3.2…+10.2 µs），不是时间戳放错位置的形态（那会单侧固定偏差）。
     **正确判据＝逐次回调先算残差再取中位。**
  ② `--nw-csv` 原样喂 `task_time.csv` **错位两格**：每步 42 个 NOTIFY_WAIT，前 2 个不是我们的层；corr(NW,cb) 在位移 0/1/2
     分别 −0.19 / 0.06 / **0.75**。本发用派生的 40 列 csv 重算（脚本 `make_nw_ours40.py` 在 run 目录，**未入库**）。
- **停服务时 EngineCore 是 SIGSEGV**（`!!!!!!! Segfault encountered !!!!!!!`，栈顶 `_PyObject_Free ← cfunction_call`），
  不是先前记的"静默退出"；profile 产物完整、离线 analyse 成功。采 profile 的服务仍按一次性用。

### 4.20.1 仪器修正与「线程抢占」假设的筛查结果（2026-09-22 晚）

**seam.py 两条问题已修（`dac0dc596`）并重算首发：分段数一个没变**（A+F 3.833 / NW 14.901 / D 9.172 / E 0.734 /
C 0.785 / 我方 0.216 ms per token 全部相同），变的只有：对账 rc=1（24/40 假红）⇒ **rc=0（40/40 通过）**；
NOTIFY_WAIT 对位从"手工派生 csv"变成工具内做且**自证**（秩相关 0.805，位移 0/1 为 −0.105/0.056；
Pearson 只有 0.361，被层 2 的 1071 µs 离群点压住 ⇒ 判据用**秩**相关）。⇒ §4.20 的结论不受影响。
⚠ **"逐回调算残差"在现有记录里无可算量**：各段是首尾相接的同一串时间戳，逐回调 Σ段 ≡ cb 恒等，
而记录侧只输出每层 med/p90，单次回调信息已丢。现判据是一组**必要条件**（包含关系、联合界、中段为 0 判红），
**判不出**小于 slack（本发 60–237 µs/层）的洞。要精确判，记录侧需多给 `n_unsplit` 或 `resid_us.max`，解析侧已预留。
另：`test_seam_tool.py`/`test_seam_timing.py` 的"顺序耦合"根因是那条控制臂用了会假红的中位判据
（当时机器上正跑 16 发 t4、kt 线程占满 256 核），已改为整数恒等式，变异臂更严（`98a665c6b`）。

**「kt 线程挤占 ACL report 线程」假设：微基准筛查无拐点 ⇒ 本筛法下不成立**（卡 0，`t4 --threads {96,88,80,64,32}`，
交错 3 轮 16 发，每发 ~33 s，全部 VERDICT PASS；产物 `oncard_runs/20260922_seamB1/`）：

| threads | 96 | 88 | 80 | 64 | 32 |
|---|---|---|---|---|---|
| node_us 中位 | 58.5 | 57.0 | 55.8 | 56.6 | 61.8 |
| kt 计算 µs/层 | 223–280 | 232–263 | 240–308 | 226–259 | 305–337 |

档间差 ≤6 µs 且**不单调**（80 最低、32 最高），而同档内发间极差 0.7–11.4 µs ⇒ 淹没在噪声里。代价侧符合预期
（32 档比 96 档慢 25–35%）。按约定**不上整网**，卡时省下。
🔴 **这次没有否掉真正的假设**：量 node_us 时 **kt 并没有在算**（图回放，96/192 线程只是存在且空闲）⇒ 否掉的是
"线程**存在**就挤占"，"线程**正在算**时挤占"未覆盖。整网 A+F 78 µs vs 微基准空节点 56.8 µs 的 **21 µs 差仍无解释**。
补法：整网两臂只差 `cpuinfer_threads`、用埋点直接看 A+F（约 1.5 h 卡时），排在侧流主线之后。

**由此决定的下一步**（接缝的靶子不在我方 Python：我方只占 0.31 ms）：
1. **便宜先试**：怀疑 ACL report 线程被 kt 的 96 个计算线程挤占（层 2 离群同向）⇒ 降 `cpuinfer_threads` / 给回调线程留核并绑核，
   用同一套埋点看 A+F 降不降。一发可判。
2. **根治靠侧流**：侧流不消掉这 78 µs，但把它挪到副流、让主流同时算；t4 实测窗口 83.2 µs/层与之几乎相当。
   前提是先修 §4.19③ 的进程级缓冲守卫。

## 4.21 🔴 ACL 回调派发链路（2026-09-22 夜，只读代码 + 纯 CPU 实验，未占卡）

链路（带出处）：`_launch_host_func`（`layer.py:551/634/693`）→ `torch_npu._C._launch_host_func` →
`c10_npu::launch_callback` → **`aclrtLaunchCallback`（ACL_CALLBACK_BLOCK）** ⇒ device 侧就是
`NOTIFY_WAIT` + `HOSTFUNC_CALLBACK` 这一对任务。订阅走 `torch_npu.npu._subscribe_report`；
执行回调的是 C trampoline：`PyGILState_Ensure → Py_BuildValue → PyObject_CallObject`（**seam 的 T_ENTER 打在 Python 里
⇒ A 段含「报告送达 + 抢 GIL + 建参数元组」**）。

1. **torch_npu 进程内只有一条报告线程**（首次 `_subscribe_report` 时 `pthread_create`，之后所有流复用同一 tid；
   **无线程名、SCHED_OTHER、不绑核**），循环体 `while(!stop) aclrtProcessReport(100)`。
   🔴 **100 ms 是阻塞等待上限、不是轮询间隔**（有回调立即返回；若是定时轮询则平均等 50 ms，与实测 78 µs 矛盾），
   **且没有任何 env 可调** ⇒ **不要去试**。kt 自带 poller（`ascend_callback_worker.cpp:16`，同样 100 ms）与 torch_npu 逐行等价，
   换回去也没用 ⇒ **不要浪费卡时**。
2. **所有流共用这一条回调线程**（反汇编证据：全局 tid 传给每次 `aclrtSubscribeReport`；`seam_timing.py:96-101` 的注释已依赖此性质；
   engram 走同一口径 `engram_embedding.py:484-497`）。⇒ **对侧流是新约束**：副流与主流的 host-func **不会并行执行**
   （bs1 下同一时刻只有一层在飞，影响有限，但收益模型要按此重算）。
3. 🔴 **kt worker 干完活忙等 50 ms 才睡 condvar**（`kt-kernel/cpu_backend/worker_pool.cpp:326-345` 与 `:484-500`），
   decode 每层几百 µs 就有新活 ⇒ **稳态下 kt 线程全程自旋**，且每条被 `kt_bind_worker`（`:114-143`，
   `hwloc_bitmap_singlify` + `HWLOC_CPUBIND_STRICT`）**硬绑到单个 PU**。
   🔴 **线程数更正（2026-09-22 夜，读 `experts_base.py:365-368` + `WorkerPool::init` 落实）**：`cpuinfer_threads:96` +
   `threadpool_count:2` 是**总共 96 条**（每 NUMA 48 = 1 分发器 + 47 计算），占 **96 个 PU**；机器 256 PU ⇒ **空 160 个**，
   每 NUMA 还有 **16 个整核**两条 SMT 都空。先前写的"96×2=192 条、余 64"是错的。
   ⇒ 这让绑核方案更宽裕（**不必让核**），同时**进一步削弱"抢占"假设**（空闲核这么多）；21 µs 缺口仍待卡上判。
   ⇒ **这解释了 §4.20.1 微基准为何无拐点**：那里 kt 空闲 >50 ms 已睡，量到的 56.8 µs 是「无自旋」档，
   整网 78 µs 是「96 PU 自旋」档，21 µs 的缺口与之同向（**推断，待上卡判**）。
4. **GIL 不是中位的原因，但可能是离群的原因**（纯 CPU 实测，本次）：报告线程被唤醒后执行第一条字节码的延迟，
   无竞争中位 **7.0 µs**；只要有 1 条别的 Python 线程在跑，默认 `switchinterval=5 ms` 下中位 **8.1 ms**
   （0.5 ms→1.1 ms，0.05 ms→0.12 ms）。⇒ 78 µs 不是 GIL 争用；**§4.20 层 2 的 773 µs 离群点与此同形**。
5. **sglang 走完全相同的链路**（`kt_ep_wrapper.py:98-108 / :646 / :812`）⇒ 它的 −5.37 ms **不来自派发延迟**，
   来自 `_submit_raw_npu_graph_forked`（`:653-676`）的重叠。可抄的差异只有"捕获前就订阅主流与侧流"（与延迟无关）。
6. 唯一可比的 host-func 往返实测：engram 图内 **38.1 µs/节点**（`DSV4.1_性能分析报告.md:478`，6 次/228.4 µs/步，**含 numpy gather**，那发没有 kt）。
7. **未知**（不拿推测填）：`aclrtProcessReport` 内部等在什么原语（库已 strip）；**A 与 F 分不开**；
   CANN 9.2 另有一套 `aclrtLaunchHostFunc/SubscribeHostFunc/ProcessHostFunc`（`acl_rt.h:3195-3233`，与 `aclrtLaunchCallback` 是
   **两套机制**），torch_npu **没有引入**（`nm -D` 无符号）⇒ 哪套更快、能否在捕获下用，**完全未验**；容器能否 `SCHED_FIFO` 未验。

**两条实现已入库（默认关）**：`12f7aae17`（报告线程绑核，`report_affinity.py`；`auto` 的解算读 `kt_bind_worker` 落实 stripe/singlify 规则，
挑**两条 SMT 都空**的整核、取 kt 顺序上最后一个，本机算出 PU 63/127；**绑前把模型算出的 PU 集合与 `/proc/self/task/*` 实际比对、
绑后 `sched_getaffinity` 回读**，不符即拒绝启动或标未生效；单测 32 条 + 4 个变异臂实测会红）与 kt 侧 `1a8d1df`
（`KT_WORKER_SPIN_MS`，默认仍 50 ⇒ 行为逐字节不变；已编过、env 解析测试通过）。
⚠ **kt 的 wheel 未重建** ⇒ 服务里那条改动现在是惰性的，D2 臂标「未覆盖」；重建＝换构建 ⇒ t0/t1/t2 与 D0/D1 都要在新构建上重跑。

**由此定的两条实现（默认关，代码已在做）**：① **把报告线程绑到 kt 不用的 PU**（回调里 `sched_setaffinity` + **回读断言**，
可选 `SCHED_FIFO`）——不牺牲 kt 算力，性价比最高；② **kt 自旋时长可配**（默认仍 50 ms ⇒ 行为不变），
上卡时必须同时读 A+F 与 E（自旋变短会把成本挪到唤醒）。执行单 `tools/kt_offload/oncard/RUNBOOK_dispatch.md`，
三臂同时段交错 D0/D1/D2，主判据是 `seam.json` 的每层 A+F_dev，旁证 `/proc/<pid>/task/*/stat` 的 R 态线程数（192 → 更低）。

## 4.22 🔴 接缝方案表（2026-09-22 夜，未占卡）：新靶子 30.75 µs 空转 + 拆掉 host-func 节点

> 依据的 profile 是 §4.20 那发（`/dev/shm/dsv41_prof_seam`，**未跟踪、随时消失**，读数按「已记录」引用；
> 那发带埋点，TPOT 35.49 ms，可能含 0.4–0.8 ms 埋点开销 ⇒ **绝对值不作基线，相对结构是稳的**）。
> 两个 CPU 微基准已入库可复跑：`tools/kt_offload/prof/bench/{tq_bench,merge_bench}.cpp`。

**F1 🔴 每个 host-func 节点前有 30.75 µs 的设备纯空转（账本里原本没有这一项）**
stream 244 上按「前一个 task 结束→本 task 开始」统计：`HOSTFUNC_CALLBACK` n=7984 **中位 30.75 µs**；
而 `DAVID_EVENT_WAIT`(30406)、`MEM_WAIT_VALUE`(2286)、`NOTIFY_WAIT`、`MEMCPY_ASYNC`、`EVENT_RECORD` 的前置空隙**全是 0.000**。
抽 2000 个空隙核对别的流：只有 8% 被其它流覆盖，92% 是真空转 ⇒ **30.75 × 40 ≈ 1.23 ms/token**，
**挂在 host-func 机制上、且不在 §4.20 的 4.3 ms 里**（那张表只算 `NW − T_cb`）。
⇒ **可攻击面修正为 ≈5.5 ms/token**（≈139 µs/层）。旁证：t4 实测**空**节点 52.6 µs/节点（§4.19）与「空隙 30.75 + 派发的一部分」同量级
⇒ **host-func 节点本身就是 ≈2.1 ms/token 的固定开销**。

**F2 事件等待在本平台几乎不要钱**：`DAVID_EVENT_WAIT` 中位 **0.027 µs**、`MEM_WAIT_VALUE` **0.144 µs**，前置空隙 0。

**F3 `ExternalEvent` = `aclrtValueWait`/`aclrtValueWrite`，而且这套在本树、本图里已经在跑**
（`acl_rt.h:3802/3814`、`ACL_EVENT_EXTERNAL acl_rt.h:32`；捕获侧 `attention_v1.py:1036-1038`、`mla_v1.py:1744-1747`
把 `event.wait(stream)+event.reset(stream)` 进图，回放侧 `mla_v1.py:947-970` 由 host `event.record(update_stream)` 放行；
profile 里 `MEM_WAIT_VALUE` 实打实 2286 次）⇒ **「device 等 ExternalEvent、由别的线程 record」不是猜想**。
唯一无先例的是 **record 由「非主线程、在 replay 进行中」发出**。

**F4 kt 的提交路径有一次完整线程握手，空任务往返实测 10.44 µs（p90 11.0）**
（`CPUInfer::submit` → `TaskQueue`（`new Node`+mutex+notify）→ 另一线程执行 → notify 唤醒 `sync` 的 `cv.wait`；
`cpuinfer.h:82-87`、`task_queue.cpp:41-96`；内联参照 0.04 µs）⇒ **0.42 ms/token 是纯线程跳**，落在 `submit`(2.4 µs) 与 E(18.2 µs) 两段里。
⚠ 连带更正：**E 的大头不是跨 NUMA 归并** —— `merge_results`（`moe_base.hpp:766-808`）在 qlen=1 走单线程直算，
本机实测 **1.18–1.78 µs**（≈0.07 ms/token）。「让两个 NUMA 池直接写最终 pinned 输出」**没戏**。

### 方案排序（用户定：主指标 bs1 延迟；本 session 只做前三）

| # | 方案 | 预计省 | 状态 |
|---|---|---|---|
| **P0** | **把 A 与 F 分开**（仪器，不改行为）：回调 entry/exit 再取一次 `CLOCK_REALTIME`，与 profile 的 `HOSTFUNC_CALLBACK.ts` / `NOTIFY_WAIT.ts+dur` 直接相减（profile 的 `ts` 就是 CLOCK_REALTIME µs，已核）。顺带补 §4.20.1 挂起的 `resid_us.max`/`n_unsplit` | 0（但 P1a/P1b 的取舍全靠它） | ⬜ 0 独占卡时，搭车 |
| **P1** | **拆掉 host-func 节点，device 改等 ExternalEvent**。1a：回调只 submit 立即返回，kt 算完由完成线程 record 放行（省 F 段）；1b：连启动信号也不走回调——图里多挂一次 8 字节 D2H（每步自增的 device 计数器），常驻自旋线程（放 kt 里、C++、避 GIL）轮询到就跑 kt 并 record | 1a ≈1.9 ms（若 A/F 各半，推断）；**1b 净 2.5–3.5 ms**（30.75 空转 + A + F ≈109 µs/层，减新机制 6–15 µs，推断） | ⬜ 1a ~2.5 h 卡时；1b 再 +2 h（含 kt 重建与 t0/t1/t2） |
| **P2** | **kt 内联执行**：加 `CPUInfer::run_inline`，图路径下 `submit`+`sync(0)` 不入 TaskQueue | **0.3–0.4 ms**（F4 实测基础上的推断） | ⬜ CPU 可先验，整网 A/B ~1 h |
| P3 | 回调内 Python 前奏（kt_pre 0.62 ms/token）在捕获期固化成 C++ prepared task | 0.4–0.6 ms | 📒 排第四：有「缓存指针失效＝静默算错」的尖角，等 P1 落地后再动 |
| P4 | 我方 Python 0.31 ms/token 的削减 | ≈0.2 ms | 📒 搭车做；🔴 **不得为省 2 µs 把 A3/A4 守卫改弱** |
| P5 | packed 在新尺子下重测 | 上界 ≈0.16 ms（三次 MEMCPY 合计仅 6.6 µs/层） | 📒 只作 P1b 的前置（自有缓冲配计数器最干净），不单独排 |

**P1 的判据（写进将来的 t6 用例，缺一不可）**：捕获期 record/回调计数 0、每次 replay 恰 +1、与串行臂**逐位相同**、三次 replay 互不相同；
🔴 **两条变异臂**：①「不 record」⇒ 必须挂住/超时（证明 device 真在等）；②「提前 record（数据 D2H 之前）」⇒ 输出必须与串行臂不同
（**放行早 = 更快且算错，且不报错**，这是本方案的主失效形态）。另加 profile 侧硬证据：**`HOSTFUNC_CALLBACK` 节点数变 0、30.75 µs 空隙消失**。

### 4.22.1 P0/P2 落地与三条更正（2026-09-22 夜，未占卡）

**已入库**：`e19e5465c`（P0 仪器：回调 entry/exit 各取一次 `CLOCK_REALTIME`，与 profile 的 `HOSTFUNC_CALLBACK.ts`/`NOTIFY_WAIT`
直接相减出 A 与 F；判据不过就**不出这两列并 rc=1**；顺带补上 §4.20.1 挂起的 `n_unsplit` 与 `resid_us.max`）、
kt 侧 `53eab3c`（`CPUInfer::run_inline` + `KT_CPUINFER_INLINE`，默认关）。埋点净增 **≈0.15 µs/回调**（实测 `clock_gettime` 76.5 ns/次）。
变异臂实测会红：entry 戳挪到 kt 之后 ⇒ 拒绝出 A/F 列；把 `perf_counter` 当 realtime ⇒ 报「不是同一发或不是 CLOCK_REALTIME」。

🔴 **三条更正**
1. **A 不含那 30.75 µs 空转**：profile 里 `HOSTFUNC_CALLBACK.ts == NOTIFY_WAIT.ts`、HF 节点自身时长 0.007 µs
   ⇒ `A+F ≡ NW.dur − T_cb` 是恒等式，**30.75 µs 的空转在 HF 节点之前，是账外的独立一项，只有 P1b 吃得到**。
   好处：那条 `<5 µs` 判据实际在判「两个时钟一致 + record 开销」，不会被节点时长污染。
2. 🔴 **kt 的 `submit`/`sync` pybind 绑定没有 `gil_scoped_release`** ⇒ 回调线程在整个提交+等待期间（含 kt ≈240 µs 计算）
   **一直持有 GIL**，每 token 约 9.6 ms 主线程拿不到 GIL。这与 §4.21 第 4 点（GIL 只解释离群）相关且此前没记。
   ⇒ **新候选 P6：在绑定里释放 GIL**——不让回调变快，但让主线程在 CPU 算的时候能继续干活（对 TPOT 的影响未知，**待测**）。
3. `do_numa_job` 的调用方**不参与计算但忙等**（`USE_NUMA_JOB_DIRECT_WORK` 全树未定义）⇒ P2 内联不改 NUMA 局部性，
   但**把这段自旋挪到报告线程上** ⇒ 上卡判 P2 时 **D（kt 计算）必须一起读**，D 变大＝抢核了。

**P2 微基准（`prof/bench/tq_bench.cpp` 新增内联臂，本机两轮）**：队列 `enqueue+sync` **11.26 / 12.94 µs** 中位，
`run_inline+sync` **0.101 / 0.110 µs** ⇒ **≈0.45–0.52 ms/token**（与 F4 的 10.44 µs 同量级，本轮机器更忙）。
内联 vs 队列**输出逐位相同**（kt 侧 16 项 CPU 测试），变异臂「内联但故意不等某个 NUMA 池」判红。
⚠ **未覆盖**：`packed_transfer` 路径没接内联；kt wheel **未重建** ⇒ 这条改动在服务里现在**完全不存在**。

### 看着诱人、实际没戏（附理由，别重走）
1. 让两个 NUMA 池直接写最终 pinned 输出 —— 归并只值 0.07 ms（实测 1.2–1.8 µs/次），E 的大头是线程跳（F4）。
2. 把提交点前移到 top-k 之后最早处 —— gating→三次 D2H→host-func 之间**没有空隙**（中位 0.000），三次 copy 合计 6.6 µs；30.75 µs 的空转在 copy **之后**。
3. 降 `cpuinfer_threads` 给回调线程让核 —— 微基准已无拐点（§4.20.1）；F1 给了更可能的解释（空隙是 **host-func 节点类型独有**，其它节点全 0，像是 ACL 对 host 任务的调度方式）。要做也排在 P1 之后；P1b 成了它自动没意义。
4. 靠 `graph_task_group` 省接缝 —— 它解决的是「replay 时换算子入参」，能借的只有它顺带证明的 ExternalEvent 用法（已计入 P1）。

## 4.23 🔴 侧流整网判负（2026-09-22 夜，卡 0）：更快 2.74 ms，但 3.1% 的 decode 步整步算错

> 产物 `oncard_runs/20260922_161813_sidestream`（未跟踪 ⇒ 读数内联）；两臂逐项同 §4.18 基线，只差
> `overlap_side_stream` + `VLLM_ASCEND_DSV41_KT_SIDE_STREAM`；`.so` 85331298bdef。判定见执行单 `RUNBOOK_sidestream.md`（`c879efe97`）。

| 判据 | A（侧流关） | B（侧流开） |
|---|---|---|
| prefill 位逐位（eager，侧流不参与）＝**内部对照** | — | **115/115 相同** ⇒ 差异只可能来自走图的 decode |
| **decode 位逐位** | — | **622/690 ＝ 90.1%** 🔴 |
| 选出 token 不同 | — | **27 处**（35/115 prompt，30.4%） |
| max \|Δlogprob\| | 地板 **0.000** | **8.107** |
| GSM8K 50（贪心、thinking 关） | **50/50** | **46/50** |
| bs1 TPOT 中位（3 轮） | 35.15 ms | **32.41 ms（快 2.74）** ⇒ **不是收益** |
| 必检 | SIDE STREAM 日志 0 条 | 40 条（每层一条）；两臂 `1070[12]x` 均 0 |

**失效形状（两种并存）**：① **退化步**——该步 top-5 全部 ＝ `ln(1/129280) = −11.7697`（logits 全相等、隐藏态整体坏），
句中选出 `<｜begin▁of▁sentence｜>`；计数 **A 臂 1398 位 0 个（两发都 0）/ B 臂 998 位 31 个＝3.1%**；35 条分歧里 26 条首个分歧位就是它。
② 另 9 条是 token 相同但 logprob 值不同（普通数值漂移）。单条 HTTP 400 是服务端拒了那条请求（非引擎崩），单独重放当场复现退化步。

🔴 **三条要记住的**
1. **这是"更快且算错"的实例**：−2.74 ms 落在 ">1 ms 算收益" 的判据里、也没超 t4 推断的 3.3 ms 上限 ⇒ **只看 TPOT 会把它当成成功**。
2. **GSM8K 差点没抓住**：3.1% 的步整步坏掉，50 题只从 50/50 掉到 **46/50**。**分数类判据分辨率不够**，抓住它的是 logprob 逐位比。
3. 🔴 **① 单层逐位全绿挡不住 ②**：t5 每次 replay 前后都 synchronize、只建 3 层；服务是 40 层连续 decode，
   kt 的 pinned 槽按 `layer_idx%2` 跨层复用（A6）⇒ 要查的是**跨步/跨层的槽复用**（上一步的 H2D 还没读完，下一步就写同一个槽），
   **不是**本层的 record/join（本层丢 join 会在捕获期直接 107025，① 的变异臂正是这么红的）。
   ⇒ **合版本的门（§4.18.2）加一条：单层逐位绿不构成整网可用的证据。**

**顺带确立的基础事实（以后所有精度对比都用得上）**：**并发 1 下服务是逐位确定的，但起服务后头约 6 条请求不是**
（冷启动 JIT/预热；实测 200 条里 4 条不同，全在 i=2..5）。弃前 10 条后：同进程地板与**跨进程**地板都是
**100% 逐位相同、非零差 0**。口径：`/v1/completions`、`temperature=0 top_p=1`、`logprobs=5`、`max_tokens=8`、并发 1、200 条固定批、`--skip-first=10`。
⇒ ② 这把尺子有区分力，不必退到分布比较（用户给的决策树在第 0 步就解决了）。

**按判负规则主动没做的**：1319 全量、埋点+profiler 那一发（在会算错的路径上量 A+F，量到的是"少等了多少"而不是"挪走了多少"）。
⇒ §4.20 的 A+F 78 µs/层仍是旧读数、本发未复验。⚠ 今晚**没有**量同臂 TPOT 漂移地板（A2 只跑了 logprob）⇒
A1 的 35.15 与 §4.18 的 34.36 之间那 ≈0.8 ms 只能当时段漂移，**不得相减**。

**下一步**：把 ① 扩成能复现这个失效的形态——**连续多步 replay 不 synchronize + 40 层规模**（现在的 ① 每次 replay 都 synchronize，
正好把它遮住）；复现之后再改（候选：侧流改用 packed 的每层自有缓冲；或给跨步的槽复用加事件守卫，等上一步该槽的 H2D 完成）。
在修好并重跑整网 ② 之前，`overlap_side_stream` 维持默认关 + 双开关。

## 4.24 指纹改读「真正被 import 的那个 kt 站点」（2026-09-22 夜，`fa0989b38`/`9670848ab`）

**坑**：`build_fingerprint.py` 的 `KT_WHEEL_GLOB` 写死旧 wheel 目录、与站点无关 ⇒ 切到 `kt_site_v2` 却漏改它时，
两份指纹**逐字段相同、判「同构建」、允许相减且不报错**（变异臂实测：改前 `SAME BUILD rc=0`；改后
`kt_site_sha256 differs … DIFFERENT BUILD rc=1`；同站点两采仍判同构建 ⇒ 没有过度拦截）。
**改法**：指纹从 `$KT_SITE/kt_kernel/` 整棵目录算内容哈希（排除 `__pycache__`），并单独记 `.so` 与 `experts_base.py`
（服务 import 的就是这两件）；wheel 由站点名推导、继续记但不再是唯一来源；站点解析不出/缺件 ⇒ **rc 2 且不写指纹文件**。
站点路径收敛到**一个**文件 `tools/kt_offload/kt_build.env`（四个 shell 驱动都 source 它）⇒ **切换清单五处收成一处**。
schema `/3`→`/4`（`/3` 的产物说不出自己跑在哪个站点上，一律不许与 `/4` 相减）。回归 423 passed / 0 skipped。

✅ **两条假绿灯已修**（`588e82e8c` / `15e4d091c`）：`ok()` 改成收 rc 后，102 条里**6 条在空转**——
其中 1 条**从出生起就没绿过**（`grep -c gsm8k_eval_effort.py` 把预检那行也数成发射点，写下当天就是 2）、
2 条**随文案过期**（09-21 把「整体拒绝」收窄成「只拒绝减法」时改了提示串，两条 grep 当天失效）、
1 条对着**尚未建立的 fixture 目录**跑、1 条把构建闸绑在总 rc 上（判错对象）、1 条即下面的 vendors。
🔴 **元检查的第一版自己是错的**：它用坏掉的 `ok()` 报自己的结论 ⇒ 判官坏了这件事又被同一个坏判官打成 PASS
（**判官不能给自己作证**）⇒ 改成不经 `ok()` 且判官一坏**当场退出**（后面每行都是噪声）。变异实测：把 `ok()` 改回无条件 ⇒ FAIL；
把一个调用点改回消息形 ⇒ FAIL。全树扫过，同形 `ok()` 只此一处。真实计数 `selftest_packed_ab` **108/0**、`selftest_cpu` **77/0**、pytest 423/0 skip。

**vendors 那条的事实**：**不是基准被同步成 9.2，是我们今天自己换的包**（11:33 拷入 clean 已验收的那份；
`build_artifacts/vendor92` mtime 仍 09-10，自建立起没动）⇒ 两边逐字节相同是**预期状态**，
「两个哈希必须不等」的前提本来就不对（09-21 的 759 vs 786 只是当天读数）。
改法不是放宽：区分力搬到**克隆 + 翻一个 bit**（sha 必须变、文件数与字节数不许变）＋「记录的哈希必须能被重算复现」，
两个变异臂实测都被抓；「今天两包是否相同」降级为打印出来的读数（`shared_identical=True`），不参与判定。

🔴 **原始记录（上面两条修复前的状态）**
1. `oncard/selftest_packed_ab.sh:11` 的 `ok()` **无条件打印 PASS** ⇒ `[ $r -eq 0 ]; ok "…"` 那类调用点**判据被丢掉**，
   `ok $? "…"` 还把 rc 当消息打成 `[PASS] 0` ⇒ 该自检报的「102 checks, 0 failed」**有一部分是空转的**
   （对照 `selftest_cpu.sh:18` 的 `ok()` 是收 rc 的正确版本）。修的时候要**先红后绿**、逐个调用点核语义，并加一条
   「`ok()` 会红」的元检查。
2. `oncard/fp_object_check.py` 最后一条断言「树里的算子包 ≠ 共享基准」**今天必红**：两边已逐字节相同
   （787 文件 / 135,871,736 B / sha `2eae934d033b…`）。这条断言原意是证明「这条检查分得开两者」，前提已不成立
   ⇒ 要查清基准何时被同步成 9.2，再把区分力的证明换到别的对象上（例如临时拷贝改一个字节必须被抓到），**不得靠放宽断言变绿**。

## 4.25 🟢 侧流真因找到并修好（2026-09-22 夜，`800404d73`）：−3.00 ms/token，整网逐位相同

🔴 **§4.23 猜的「跨步/跨层 pinned 槽复用」不成立**——五种形态都没复现（8 层×6 步 inplace 回收、40 层×30 步探针、
40 层×200 步 + 每 7 步插 eager prefill、realloc 回收……两臂全绿）。事后读代码也对得上：**侧流自身是串行的**
（D2H(L+2) 排在 hostfunc(L) 之后，fork(L+2) 又记在 merge(L) 的 H2D 之后）⇒ fork 形态下 kt 的 `%2` 槽本来就有序。

**真因**：`_submit_captured_forked` 里 **`fork.record(compute)` 记在前、int64 路由 id 的主流写入在后**
（`self._ids_i64[bs].copy_(topk_ids)`）⇒ fork 一 record 侧流就被放行，它读那块缓冲与主流写它之间**没有任何 event**。
第二个同类洞（一并堵上，未单独复现）：侧流 `copy_inputs_to_cpu_buffers` 直接读调用方的 `topk_weights`，
而调用方在 `route()` 返回时就换掉了最后一个引用。

**怎么把抛硬币变成决定**（复现的关键）：`--stress ids-delay` 在**主流**上插一段延时（4×16 MiB device add）。
改前 fork 臂 **0/30 逐位**、探针 **1200 个层-步全错**（1160 个读到**上一步的 ids**，40 个是开跑前残值）；serial 臂 30/30、探针全对。
⇒ 差别只有 D2H 与 host-func 挂在哪条流上。**同一个延时开关，落在 fill 之前还是 fork 之后，就是答案。**
探针原文：`probe[fork]: step 3 layer 17 … the ids buffer held the PREVIOUS step's value: [263,233,…]`。

**改法取 (a)**：侧流改用 **packed 的每层自有缓冲**，主流在 **fork 之前**填好（hidden/ids/weights，含 int64 cast）。
一次去掉一类洞：`%2` 槽（A6）在这条路上不再使用、`topk_weights` 生命周期问题消失、三次 D2H 并成一次。
`packed + 侧流` 同开由「raise」改为「接受」（原先 raise 的理由——`route()` 先判 packed 会让侧流静默失效——已不成立），
并把「开了真的在 fork」钉成 UT（分支序被断言住）。
🔴 **旧守卫失效的原因比 §4.23 记的更彻底**：`route()` 进门就清 `_pending_join` ⇒ 「提交过没 join 的层」与「已 join 的层」
长得一模一样，它**连本步都看不见**。新口径：每块 store 只属于一层（`_SIDE_BUFFER_OWNER`）+ `_side_inflight` 只由
`_join_side_stream` 清 ⇒ 跨步的「没 join 就又提交」落在这里；`side_stream.py` 补 `buffer_ownership_ok` 与 `submit_order_ok`。

**验证**：① 复现用例修后 30/30 + 探针 1200/1200，留库当回归（`t5_side_reuse.py`，`run_oncard.sh <card> t5r`）；
② 变异臂 `--mutant early-fork`（只把 `fork.record` 挪回 fill 之前）⇒ **captured 但 0/30**，探针同时指名 ids 与 weights；
③ ACL 阳性对照 `--mutant drop-join` ⇒ 107025；④ CPU 单测 **431 passed**（仿真器改成每层自有缓冲，新增
「fork 记在 fill 之前 ⇒ 又快又错，且输出正好等于上一次 replay」——与卡上探针指纹同形）。

**整网 ②（logprob 逐位，四臂交错 A1,B1,A2,B2，各 200 条、并发 1、`max_tokens=8`、top-5）**
出处 `oncard_runs/20260922_190713_sidestream_fix`【未跟踪，读数内联】：prefill 位六对两两 190/190 全同；
**skip=12 起，四个独立进程两两 1149/1149 decode 位逐位全同、退化步 0（各 0/1163）、token 变化 0**。
分歧只落在请求 10、11 两条。🔴 **§4.23 的「弃前 10 条」紧了 2 条**：A 臂自己（A1 vs A2，同配置）在 i=2..9 就有分歧，
B 臂到 i=11 ⇒ **截断点必须按臂校准**，用 A 臂标定的点套到 B 臂会把 B 的预热尾巴读成「侧流算错了」。
⇒ 口径改为 `--skip-first=12`，**依据是同臂地板、不是 A/B 结果**；并已要求一发**预先声明 skip=12 + 换 prompt 种子**的确认发。

**性能（另起四臂 A3,B3,A4,B4 只跑 perf，交错）**：A 33.54/33.71，B 30.27/30.98（bs1 TPOT 中位，n=3）
⇒ **A−B = −3.00 ms/token**（最不利配对 −2.56，最有利 −3.44）；**同臂跨轮地板 A 0.17 / B 0.71 ms** ⇒ 取 0.71。
2.56 > 1 ms 判据且 > 地板 ⇒ **算收益**。⚠ 最有利配对 3.44 略超 t4 推断上限 3.3（上限本身是推断，且 host-func 每层每步 +1、
`never-ran`=0、②逐位已过 ⇒ 无「少算了东西」的迹象）。⚠ **跨发不得相减**：§4.23 的 A 35.15/B 32.41 与本发 A≈33.6/B≈30.6
绝对值差约 1.5 ms，只有同发内的 Δ 可用。

⚠ **已知项（未动）**：packed 的 `pin_out` 是 `torch.empty`（原有性质，现在侧流也走它），kt 的 `output_cpu` 是 zeros。
本轮无 NaN/野值；改成 zeros 要单独改单独验（改了就要重跑 t5r 与一臂精度）。

## 4.26 🟢 侧流三道门全过（2026-09-22 夜）：确认发 100%、GSM8K 50 两臂 50/50、1319 = 1275/1319

**① 确认发（`oncard_runs/20260922_210518_sidestream_confirm`，判据先写死再跑）**：`EXPECTED.txt` 写于 21:05:18Z、
**在生成 prompt 批之前**，钉死 `--skip-first=12`（不许传别的值）、decode 100%、退化步 0、token 变化 0、
「不满足就判负停下，不许调 skip、不许换批再试」；跑完 `EXPECTED.txt` 未改。新批 seed `20260923`，与上一发**只 20/200 重合**。
读数：prefill **188/188**、decode **1129/1129 = 100%**、非零 |Δlogprob| **0**、token 变化 **0**、
退化步 A 0/1143 · B 0/1143、top-k key-set 差异 0、必检全绿（B 臂 SIDE STREAM 40 条、`1070[12]x` 0、`never-ran` 0）。
⇒ 口径落地：`logprob_cmp.py` 默认 `--skip-first=12`，RUNBOOK 写明**截断点按臂校准**、**不许拿结果反推截断点**。

**② GSM8K 50（同发交错 A2→B2，贪心 / `EV_THINKING=0` / max_tokens 4096，口径同 §4.1）**：
A（侧流关）**50/50**、B（侧流开）**50/50**，err 0、截断 0，各约 116 s。

**③ GSM8K 1319 全量（B 臂，侧流开）**：**1275 / 1319 = 96.664%**，err 0、截断 0、stop 1.0，59.5 min，输出 262,412 token。
必检：SIDE STREAM 40 条、`1070[12]x` 0、`never-ran` 0、host-func 119 个窗口每层计数极差最大 1（采样跨步）。
**判读**：单卡 offload **没有 1319 历史基线**，唯一可比的是多卡带 **1276–1281（96.74–97.12%）**；1275 比带底低 **1 题**，
而同代码同包跨重启噪声 **11 题** ⇒ 🔴 **不可分辨 / 无可检出回归**，不得读成「更好/更差」，也不得读成「掉出带了」。
这是**第一个单卡 offload 的 1319 数**，且取自**侧流开**的臂（A 臂 1319 按安排没跑）。

🔴 **又两个「碰巧一直对」的仪器 bug（已修 + 变异验证）**
1. `816a5afdb`：汇总用 `grep -m1 acc` 取**第一行**。`--limit 50` 时第一行恰好是最后一行 ⇒ **从没暴露**；
   1319 那发它写下 `acc=1.0000`，真实是 1275/1319。**看汇总的人分不出这两种情况。**
   改成从 `metrics.json` 读（含 err/trunc/墙钟），读不出打 `score unreadable`，**不静默给个数**。
   ⚠ `B3/arm_summary.txt` 里那行 `acc=1.0000` 是修前写的，**错的，别引用**；正确数在 `B3/gsm8k1319/metrics.json`。
2. `630ec3473`：**侧流开时 `cpu_infer` 上叠了两套仪器** —— `after_weights_loaded` 判的还是 `self.packed`，
   而侧流改用 packed 缓冲后走的是自己打时间戳的 `_kt_packed_forward` ⇒ `T_SUBMIT_IN` 取自探针（更晚）、
   时间从 `submit` 挪进 `kt_pre`，**而且探针那两帧加长的正是侧流要藏起来的那个 host-func 节点**。
   只在 `SEAM_TIMING` 开着时发生 ⇒ **本轮精度/性能读数不受影响**（那些臂都没开），但它正是要拿去看「A+F 有没有被挪走」的仪器。
   改成 `own_node = self.packed or self.overlap`，两个方向都判，改回旧写法当场变红。CPU 单测 **432 passed**。

**profile 那一发没做**（在刚发现装错的仪器上赶数，比没有数更糟）；探针已修、`seam.py` 自带 NW 对位判据 ⇒ 随时可按 `RUNBOOK_seam.md` §6 单独排。

## 4.27 判据口径更正（2026-09-23，**先声明后验**）：收益门槛与地板挂钩

**起因**：P1a 实测 A−B = **0.53 ms/token**，而当发同臂地板只有 **0.16 ms**（A 0.15 / B 0.16，机器很空）。
0.53 是地板的 3.3 倍，**测得出来**；但低于事先定的 **>1 ms** 门槛。而那个 1 ms 当初是按「地板 ≈0.7–1 ms」定的
（§4.18 的 34.08–35.12 极差、§4.25 的 B 臂 0.71）。🔴 **看到结果再放宽门槛，正是本线一路在防的事**，
所以 P1a 不按旧门槛判成收益，也不就地改门槛。

**新门槛（用户 2026-09-23 批准，此后生效；本条写在任何按它判读的发次之前）**：
> **收益成立 ⇔ |A−B| ≥ 3 × 同臂地板 且 |A−B| ≥ 0.3 ms/token**，其中地板＝**本发**同臂跨轮/跨重启的最大摆动。
> 两条都要满足；地板没测到（只有一轮）⇒ 判 fail（§4.15③ 的老规矩不变）。
> 旧的固定 1 ms 退役，理由：地板随机器负载在 0.15–0.71 ms 之间变动，固定常数在安静的机器上会把真实收益判掉、
> 在吵的机器上会把噪声判成收益。0.3 ms 的绝对下限是防止地板过小时把无意义的小差判成收益。

**按新门槛回判 P1a**：0.53 ≥ 3×0.16 = 0.48 ✅ 且 ≥ 0.3 ✅ ⇒ **算收益，但很小**。
⚠ 这是**回判**，不是当初的预先声明 ⇒ 结论标「**待一发预先声明新门槛的 A/B 复现**」，复现前不写进收益账。

**P1a 的读数说明了什么**：0.53 ms ≈ **13 µs/层**，与 F 段（回调返回→通知 device）的量级相符；
而 A 段（device→回调被调起）与 `HOSTFUNC_CALLBACK` 前那 **35.75 µs** 空转都还在 ⇒ **大头在 A 段与空转，要 P1b 才吃得到**。
⇒ 用户 2026-09-23 定：**不在 P1a 上再花卡时，直接做 P1b**。

## 4.28 🟢 P1a 完成（2026-09-23）：device 改等 ExternalEvent，0.53 ms/token，逐位 100%

形态：`D2H(fork 之前填好) → host-func[只 submit 立即返回] → ExternalEvent.wait → reset → H2D → merge`；
完成线程（`kt_offload/async_wait.py:Completer`）`cpu_infer.sync(0)` 后 `event.record(<自有 stream>)` 放行。
开关 `VLLM_ASCEND_DSV41_KT_ASYNC_WAIT`（纯 env、默认关；前置：packed 或侧流之一）。**kt 一行未改 ⇒ 不需要重跑 t0/t1/t2。**
不改 kt 的依据：`TaskQueue::sync` 判的是 `pending<=allow_n` 计数、多线程同等无碍（`task_queue.cpp:112-124`），
且图里同一时刻只有一个任务在飞 ⇒ 完成线程的 `sync(0)` 恰为「本层任务跑完」。异常仍由下次 sync 重抛，
完成线程记 error **后照样 record**（设备永远等下去会无声卡死引擎；陈旧输出至少会被「不是本步输出」的判据抓到）。

| 项 | 读数 |
|---|---|
| perf（交错 A1,B1,A2,B2，两臂都开侧流，只差这一个 env） | A 32.36/32.51 · B 31.82/31.98 ⇒ **−0.53 ms/token** |
| 同臂地板 | A 0.15 / B 0.16 ⇒ 取 **0.16** |
| §4.27 判定 | 0.53 ≥ 3×0.16=0.48 ✅ 且 ≥0.3 ✅ ⇒ **算收益但很小**；🔴 **回判非预先声明 ⇒ 标「待一发预先声明门槛的复现」** |
| 整网逐位（A3/B3，200 条，skip=12） | prefill **188/188**、decode **1150/1150**、非零 |Δlogprob| **0**、token 变化 **0**、退化步 A/B 各 0 |
| t5r（8 层 × 8 步连续 replay 不同步） | 三个干净臂各 **8/8**；`no-wait` **0/8** 红、`early-record` **0/8** 红 |
| CPU 单测 | **482 passed**（原 439 + 新 43，含 8 种事件序列红法与三条变异臂） |

⇒ 0.53 ms ≈ **13 µs/层**，与 F 段量级相符；**A 段与 `HOSTFUNC_CALLBACK` 前 35.75 µs 空转原封不动** ⇒ 大头要 P1b 才吃得到。

🔴 **三条方法论发现**
1. **加阳性对照之前，「P1a 逐位相同」没有证据力**：失效形态是「放行早＝更快且算错」，而若 wait 节点**根本不挡**，
   什么都不做也会「逐位相同且更快」。`--stall-ms 50`（每次 record 压后 50 ms）：干净臂 replay **0.027 s → 3.4 s**，
   同发未开的臂仍 0.027 s ⇒ 设备确实停在该事件上、放行者就是完成线程在 replay 期间的 record。**有了它，8/8 才是证据。**
2. 🔴 **本构建上 `reset` 不是承重的**：故意留掉 8 个 reset（`nodes_skipped.reset=8` 自证变异真生效），臂仍 8/8
   且每次 replay 照付 50 ms ⇒ **捕获进图的 `ExternalEvent.wait` 每次 replay 自行重新武装**，与 torch_npu 文档
   （`streams.py:214-217`）相反。处置：生产**照旧保留 reset**（0.144 µs 换掉「依赖未写进文档的行为」）；
   该臂判据改成**钉住这个观测**（往任何方向变＝平台变了）；CPU 模型按文档建模、比卡保守，**不一致时以卡为准**。
3. 🔴 **错码正则的坑没修完**（昨夜误判作废的真因）：`grep -v kt-seam` 只解决「扫错对象」，正则仍是无锚定六位子串
   —— 19 位 CLOCK_REALTIME 里出现 `107025` 是概率问题。收成一份共用口径（`oncard_lib.sh` 的 `ONCARD_ERRCODES` 五码
   + `ONCARD_ERRCODE_RE` 两头非数字边界），`run_oncard.sh` 与 `sidestream/run_arm.sh` 共用；自检加变异臂
   （老写法数到 3、加边界只剩 1；再给一份 `cb_ns=107015,` 证明**边界救不了「正好等于错码」的埋点值 ⇒ 两条必须一起用**），84 checks/0。

## 4.29 P1b 落地（2026-09-23）：图里已无 host-func 节点；逐位过，perf 按门槛判 fail（地板太高）

形态：每层 `D2H(输入) → D2H(8 字节步号计数器) → ExternalEvent.wait → reset → H2D`，**一个 host 任务都不剩**。
计数器是**同一条流上排在输入 copy 之后的另一次 copy**（同流串行 ⇒「计数器到了」蕴含「输入到了」），
🔴 **没有塞进同一次 copy 的尾部**（单次 DMA 内部写顺序平台没承诺）。轮询线程（C 自旋，`poll_native.c`，
起服务时 `cc` 编译一次再 ctypes 加载）看到步号就跑 kt 并 `event.record`。开关默认关。
**未改 kt** ⇒ 不重建 wheel、不换站点、**不用重跑 t0/t1/t2**；P2 的 kt 重建可单独排。
起服务时有一次全链路自检（device add → 8 字节 D2H → pinned 字 → 自旋读到），把三种「第一个 decode 步挂住」提前成启动报错。

**轮询线程不抢核的证据**（本机实测，两线程钉在不同物理核）：主线程一段纯算术 —— 无旁路 178 ms / **C 自旋 164 ms** /
睡眠线程对照 162 ms / **Python 自旋 1024 ms（5.8×）**。自旋看到写入的延迟 **中位 0.52 µs、p90 0.60**（n=300，
两端时间戳都取在 C 里；从 ctypes 返回后再取 `perf_counter` 会量成 17 µs —— 那是 GIL 交接，**假读数**）。
对比：被替掉的是 A 段派发 + **35.75 µs 设备空转**。

| 验证 | 读数 |
|---|---|
| CPU 单测 | **551 passed**；判据红臂 6 类（host-func 回来 / 两次 copy 换位 / 计数器加在 copy 之后 / wait 或 reset 缺失 / 一层两次加 / 输出 copy 跑到 wait 前） |
| t5r（8 层 × 8 步不同步） | 干净臂 **8/8**、host-func 计数全 0、signals=records；`early-record` 0/8 红、`no-wait` 1/8 红；阳性对照 `--stall-ms 50` ⇒ 干净臂 **3.23 s** vs 同发 fork 臂 0.026 s ⇒ 设备确实停在该事件上；`--probe` 三臂各 64/64 |
| 整网逐位 | 见下 |
| 整网 perf | A 32.52/31.39 · B 31.32/30.89 ⇒ **A−B = 0.85 ms**（两轮同向 1.20/0.50），**本发地板 1.13**（A 臂自己）⇒ **0.85 < 3×1.13 ⇒ §4.27 判 fail** |

**逐位（四臂 A3/A4/B3/B4，200 条 seed 20260923119）**：`skip=12` 时 A3 vs B3 只有 99.4%，**没有就此下结论**，
补地板后发现 **所有差异都落在请求 #12，且 B 臂自比（B3 vs B4）也差在 #12** ⇒ **P1b 这条路的热身尾巴比关着的臂长一格**
（A 臂自比地板到 #8、B 臂到 #12）。按**各臂自己的地板**把截断点校到 **13**：六对全部 **100% 逐位相同**
（decode 1173/1173、非零 |Δlogprob| **0**、top-k 差 **0**、chosen-token 差 **0**）。
🔴 13 是从 **B 臂自己的地板**读出来的、不是从 A/B 结果反推；且旧截断 12 下已有一对关/开（A4 vs B4）就是 100%。

🔴 **三条仪器发现**
1. **`signal-first` 变异臂第一发是绿的**（8/8），而 `nodes_skipped=8` 证明变异确实发生 —— 换顺序只是**开了一场竞态**：
   输入 copy 几微秒就完成，而轮询线程要被唤醒（0.5 µs）+ 抢 GIL + 建 kt 任务（~15 µs）才去读 ⇒ 正常情况下数据先到。
   与 §4.23 侧流那 3.1% 同形。加放大器（两次 copy 之间插 ~100 µs 设备计算）后 0/8 红。
   **那一发「绿」才是要记住的：顺序必须靠构造保证，不能靠一发测试绿。**
2. **P1b 开着时 `HOST_FUNC_CALLS` 恒为 0，而报告对「什么都没长」一个字不打** ⇒ 老口径会让这一臂看起来**最干净**。
   已改采 `device_signal.STEPS`、行名改 `device-signal over`，t5r 加判「host-func 必须全 0」，0 行时明说「没被看过」。
   同类：`SEAM_TIMING` 与 P1b 在 config 里**互斥**（它量的是不再运行的回调内部分段）。
3. 探针在 p1b 臂上**一次都没打到却报全绿**（包错对象 + 覆盖率没进判据），已修，重跑 64/64。

**下一步**：perf 判 fail 是**地板太高**（那一小时机器在变空，A1 32.52 → B2 30.89 单调下行），不是量不到差
⇒ 安静时段重跑 ≥4 轮交错；口径可先按 §4.27 之后的评估砍成 `8 条 × 128 token × 3 轮`（**须先做老/新口径对照，差值 < 本发地板**）。

## 4.30 🔴 P1b 机制不成立（2026-09-23 profile 判定）；代理层 unusable；perf 口径不许砍半

**判据先写死**（`oncard_runs/20260923_072242_p1b_prof/EXPECTED.txt`，起服务前）：G1 全机器 `HOSTFUNC_CALLBACK` 每步 0；
G2 副流等待链每层中位 ≤253.5 µs（＝参照 333.5−80）；G3 新节点前置空隙 `pregap_med` ≤5 µs。
另做**阴性对照**（拿参照当本发喂 judge）：G1/G2/G3 全红 ⇒ 判据有区分力；同时暴露 G3 的 `ms/step` 半条在参照自身上就红
⇒ **判为无区分力、不作否决依据**（追记写在同一份 EXPECTED 里，且都在看到读数之前）。

| | 侧流（参照，§4.26） | P1b |
|---|---|---|
| `HOSTFUNC_CALLBACK` | 40 次/步，前置 **35.75 µs**、1.911 ms/step | **全机器 0 次** ✅ G1 过 |
| 新节点前置空隙 | — | `pregap_med` 全 **0.00 µs** ✅（中位固定代价确实消失） |
| **副流等待链每层中位** | 335.25 µs | **335.00 µs（少 0.25 µs）** ❌ **G2 FAIL** |
| 副流 wait 合计 / idle | 31.015 / 6.845 ms/step | 31.03 / 6.56（idle 只少 0.29） |
| 主流 wait / idle | 14.178 / 5.004 | 12.311（−1.87）/ **6.34（+1.34）** |
| **step 周期** | 37.96 | **37.89（−0.07 ms）** |

⇒ **时间只是从「等待任务」挪进了「空隙」**。两种候选解释里**第二条对**：A 段本来就被主流的活盖住了一部分；
**主流的停等由 kt 什么时候算完决定，不由节点的调度代价决定** ⇒ 拆掉 host-func 节点换不来 ms/token。
与整网 perf「A−B 0.85、地板 1.13、测不出来」一致 —— **不是尺子的问题，是真没有那么多可省**。
⇒ **P1b 判 ⛔ 不进收益账、不进合版本**（代码默认关，保留作记录；`RUNBOOK_p1b.md` 内联了全部读数）。
🔴 **推论（对整条接缝线）**：P1b 从链上拿掉了 ~110 µs/层而链长**没变** ⇒ 再去削 host 侧的其它小项（P2 的 10 µs/层、
P3 的 15.6 µs/层）**预期同样换不来 TPOT**；要再进一步只能动 kt 计算本身（9.6 ms 硬底）或 NPU 侧。

**代理层（t4p）判 `unusable`**（`RUNBOOK_proxy.md` / `PROXY_CRITERIA.md`，判据与代码同一提交落地）：
能标定的机制只有侧流 1 个（预测 −4.02/−4.06/−4.15 vs 实测 −3.00 ⇒ **系统性偏高 +34…38%**，发间极差仅 0.13 ⇒ 可复现、
不可靠在外推），**P1a 的 −0.53 它报「未分辨」** ⇒ 🔴 **代理说「没看出来」不能读成「没有收益」**；分辨率 ±1 ms/token 量级。
🔴 **P1b 代理 −0.88 与整网 0.85 贴得极近是巧合**：代理看得见 wait 变短，看不见时间转成 idle —— 已作标本写进执行单 §4。
前两发因对照红（host-ahead 12% / 42%，blocker 用**墙钟**定价而非设备时间）**读数作废**。

**perf 口径不许砍半**（`16×256×3` → `8×128×3`）：中位采样标准误 ≈1.253σ/√n，现口径 n=16、σ≈0.7 ⇒ **≈0.22 ms**；
砍半 n=8 且 σ×√2 ⇒ **≈0.44 ms**（噪声翻倍，且 σ 有重尾：24 发里出现过 3.09/4.74/7.27）⇒ **P1a 量级直接测不出来**。
默认口径不动；砍半只可用于筛掉明显没用的候选并须写明。

🔴 **待查项（未查，不影响结论）：P1b 拿掉了 A+F（78 µs/层）而链长没动的那 78 µs 去哪了。**
有分解的只有串行/侧流两臂（§4.20）：A+F **78.0** / C 19.5（kt_pre 15.6 + submit 2.4 + between 1.45）/ E 18.2 /
我方 Python 7.8 ≈ 123 µs/层。P1b 那一臂**没有分解** —— `SEAM_TIMING` 与 P1b 在 config 里互斥（它量的是已不存在的回调内部分段）。
两种可能：① **A+F 本来就不在串行链上**（与别的东西重叠 ⇒ 拿掉不影响链长，和「时间从 wait 挪进 idle」一致）；
② **有别的东西补上了这 78 µs**（轮询线程唤醒 + 建 kt 任务 ≈15 µs、event record 下发、GIL 交接）。
要分清得**给轮询线程补上同一套时间戳**（现有埋点只装在回调路径上）再采一发，约 30 min。
**本轮不查**：无论哪一种，「削 host 侧换不来 TPOT」的结论都不变 —— P1b 已用最直接的方式（把节点整个删掉）证明了它。
将来真要回头做 host 侧优化，从这一条开始。

⚠ **机器级**：2026-09-23 07:43 前后根 overlay **6.0 T 100% 满、只剩 20 K**（非本线造成，本项目整棵树 ~25 G）。
已清本线 profile 原始 trace（约 1.7 G），现 9.3 G 可用，**余量仍极小**。采 profile 前先 `df -h /`。

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
- 2026-09-22（stream_prefill 建线）：本线（流式 prefill + 动态热专家，卡 3；phase2 优先用卡 0–2）的计划与读数
  **暂不并入本文**，放在 `STREAM_PREFILL_PLAN.md`（用户 2026-09-22 指示）。未跟踪的 `dynhot_plan/FEASIBILITY.md`
  已合入该文档后删除（k.16），删前逐项核对 28/28 读数已落地。本文这次只做下面的过期信息清理：
  🔴 **清掉四处危险的过期信息**：① §0 头还写着"rebase 到 `f85267ab8`、先不动"（实际已完成、目标是 `a0e673186`）；
  ② §4.13 顶部"`.so` 最终没换、旧读数继续有效、无需重测基线"——**换了**（`85331298bdef`），这句已是假的；
  ③ §4.16 的 `ONCARD_EXPECT_SO_MD5` 仍是旧 `.so`，照抄会 rc=3 全部拒飞（真值在 `RUNBOOK_window.md:12`，已核与树里 `.so` 逐字相同）；
  ④ §4.2"max-model-len 硬顶 16384"——O6 交付后已解除，§4.18 实测 31,141 tok 跑通。
  另：原维护者 `dsv41_vllm_singlecard` 已交接，文档改为两条在岗线各维护自己那节。
  🔴 **一条前提被推翻**：sglang 的"`index_select(out=)` 静默丢弃"**在本构建不成立** ⇒ 凡以它为阳性对照的判据
  （design/旧 ROADMAP 的 S2）都会把**正确实现判 fail**，已改为"目标缓冲清零 + 故意不写的臂必须红"。
- 2026-09-22（phase2）：rebase 到 clean `a0e673186`（74 提交零冲突）、换包（clean 同一份四元组，树外验收 t0/t1/t2 + GSM8K 50/50 后落地并改钉值）、整网图模式首读（KV 5.25× 并发 @32k、O6 路径在、>16k 验通：LongBench-v2 1 道区分 + 针 9/9）。四个卡上暴露的缺陷已修。见 §4.18。
- 2026-09-22（phase2）：t4 修表后 node 52.6 µs / window 83.2 µs（形态 C 判负）；🔴「副流捕不了」作废——真因是我们自己的 int64 缓冲守卫 bug，现状参数下真实 3 层 fork/join 可捕获且 replay 逐位正确；107025 的真条件是「副流尾 record 必须被主流 join」。见 §4.19。
- 2026-09-22（phase2）：接缝拆解首发——A+F（ACL 派发+回通知）78 µs/层 = 3.83 ms/token 占接缝 67%，我方 Python 仅 0.31 ms；埋点开销"没测出来但可能 0.4–0.8 ms"；seam.py 两条仪器问题待修。见 §4.20。
- 2026-09-22（phase2）：查清 ACL 回调派发链路——单条报告线程、不绑核、`aclrtProcessReport(100ms)` 是阻塞上限非轮询间隔且不可调；🔴 kt worker 忙等 50 ms 才睡 ⇒ 稳态 192 个 PU 全程自旋、挤占不绑核的报告线程（解释了微基准 56.8 vs 整网 78 µs 的缺口）；GIL 只解释离群不解释中位。见 §4.21。
- 2026-09-22（phase2）：接缝方案表——🔴 新靶子 F1「每个 host-func 节点前 30.75 µs 设备空转（其它节点类型全 0）＝1.23 ms/token」，可攻击面修正为 ≈5.5 ms；ExternalEvent 机制在本树本图已有先例（F3）；kt TaskQueue 线程握手实测 10.4 µs＝0.42 ms/token（F4）；归并只值 0.07 ms（"直接写最终缓冲"没戏）。三件事：P0 分开 A/F → P1 拆掉 host-func 节点改等事件 → P2 kt 内联。见 §4.22。
- 2026-09-22 夜（phase2）：🔴 侧流整网判负——decode 位逐位仅 90.1%、3.1% 的步整步退化（top-5 全等于 ln(1/129280)）、GSM8K 46/50，而 TPOT 快 2.74 ms ⇒ **更快且算错**的实例；单层 9/9 绿挡不住，真凶指向跨步/跨层的 pinned 槽复用。顺带确立：并发 1 下服务逐位确定（弃前 10 条冷启动请求后，跨进程地板 100% 相同）。见 §4.23。
- 2026-09-22 夜（phase2）：🟢 侧流真因＝`fork.record` 记在 ids 填入之前（不是槽复用，五种形态未复现）；改用 packed 每层自有缓冲、fill 在 fork 之前 ⇒ 整网 decode 位 1149/1149 逐位相同、退化步 0，**−3.00 ms/token**（地板 0.71）。截断点按**同臂地板**改为 skip=12。见 §4.25。
- 2026-09-22 夜（phase2）：🟢 侧流三道门全过——确认发（判据先写死、换种子）decode 100%、GSM8K 50 两臂 50/50、**1319 = 1275/1319 = 96.664%（对多卡带 1276–1281 差 1 题，噪声 11 题 ⇒ 不可分辨）**。又修两个「碰巧一直对」的仪器 bug（grep -m1 取成绩、侧流开时探针双装）。见 §4.26。
- 2026-09-23（phase2）：判据口径更正——收益门槛改为「≥3× 同臂地板 且 ≥0.3 ms」（固定 1 ms 退役，理由：地板随负载在 0.15–0.71 之间变动）；P1a 实测 0.53 ms（地板 0.16）＝13 µs/层，与 F 段量级相符 ⇒ 大头在 A 段与节点前 35.75 µs 空转，直接做 P1b。见 §4.27。
- 2026-09-23（phase2）：🟢 P1a 完成——ExternalEvent 等待，**−0.53 ms/token**（地板 0.16，按 §4.27 算收益但小、待复现）、整网 decode 1150/1150 逐位、kt 未改。三条发现：阳性对照 `--stall-ms` 才让「逐位相同」有证据力；本构建 reset 不承重（与文档相反，仍保留）；错码正则需「显式名单 + 非数字边界 + 排除埋点行」三者并用。见 §4.28。
- 2026-09-23（phase2）：P1b 落地——图里已无 host-func 节点（计数器 D2H 启动 + C 自旋轮询 + ExternalEvent），**未改 kt**；逐位按各臂自身地板校到 skip=13 后六对 100% 相同；perf A−B=0.85 但本发地板 1.13 ⇒ §4.27 判 fail（机器在变空），待安静时段 ≥4 轮重测。三条仪器发现见 §4.29。
- 2026-09-23（phase2）：🔴 P1b 机制不成立——节点确实归零、35.75 µs 中位代价确实消失，但副流链每层只少 0.25 µs、step 周期只动 0.07 ms（时间从 wait 挪进 idle）⇒ 主流停等由 kt 何时算完决定，削 host 侧换不来 TPOT；推论：P2/P3 预期同样无效。代理层判 unusable（只有 1 个标定点、分辨率 ±1 ms）；perf 口径不许砍半（噪声翻倍）。见 §4.30。
