# DSV4.1-flash 单卡 MoE offload（vLLM + kt-kernel）计划与跟踪

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
