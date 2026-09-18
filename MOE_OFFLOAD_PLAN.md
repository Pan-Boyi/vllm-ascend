# DSV4.1-flash 单卡 MoE offload（vLLM + kt-kernel）计划与跟踪

> 本文是这条线的**唯一跟踪文档**，由 dsv41_vllm_singlecard 会话维护。
> 状态格式：⬜ 未开始 · 🟡 进行中 · ✅ 完成(附产物/判据读数) · ⛔ 阻塞(附原因)。
> 每次更新在文末「变更记录」加一行，**带日期**。数字带出处；推断/估算明确标注。

## 0. 仓库与分支

| 仓 | 路径 | 分支 | 起点 |
|---|---|---|---|
| vllm-ascend | `worktrees/moe-offload` | `dsv41-moe-offload` | dsv41-clean `b7c9ca944` |
| vllm | `worktrees/moe-offload-vllm` | `dsv41-moe-offload` | dsv41-clean `dd63fc7d6` |
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
- **加载判据（dsv41-model 提醒）**：E8M0 scale 只能按字节搬。判据是字节往返逐字节比（读回应落在 119..123 且多值），
  不比 dtype 名；另全表扫 scale==0 字节（compact 模式遇 0 抛异常，`fp4-moe.hpp:271-287`）。
- **其他缺口**：world_size=1 的 MoE comm 路径（ALLGATHER）在 V4.1 上没跑过；profile run 强制随机路由会灌满 CPU 路径；
  MTP 专家跳过（DSpark 在 A5 被 `KvQuantSparseAttnSharedkv` 挡死）。

## 4. 里程碑与跟踪

| # | 内容 | 卡 | 判据 | 状态 |
|---|---|---|---|---|
| M0a | kt 分支摘 c81fa24 + 05a78ed；构建 wheel 到私有目录（`CPUINFER_PARALLEL=48`，不装全局） | 否 | `__cpu_variant__=avx512_bf16`；DSV4.1 scale 全表无 0 字节 | ⬜ 等用户 push + 批准 |
| M0b | 纯 CPU 单测：一层 DSV4.1 kt MXFP4 vs torch 参考，带变异对照 | 否 | rel err 在 bf16 地板内；变异臂必须红 | ⬜ |
| M0c | 找 integration：GSM8K 路由 topk 频率（每层）；clean 多卡性能基线补测 | 他们的卡 | 两份产物落盘 | ⬜ |
| M1 | vLLM 集成 eager（`--enforce-eager`）：掩码/加载/分流/加回。先单层小用例，再整网 | 1 | 单层 vs NPU 全专家一致；整网 GSM8K 50 题冒烟 | ⬜ |
| M2 | 图模式 FULL_AND_PIECEWISE + host-func 节点 | 1 | eager vs graph 逐 token 一致；无 10701x | ⬜ |
| M3 | 性能：侧流、提前一层下发、频率放置、线程/NUMA 调优 | 1 | bs=1 / bs=32 数字，各配多卡基线 | ⬜ |
| M4 | 精度：GSM8K 1319 全量 graph，口径同基线（thinking 关、贪心、max4096） | 1 | 逐题配对，对 96.97% 在噪声内 | ⬜ |

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
| D1 | 用哪张卡（0-3 integration 服务；4-7 被别的容器占） | ⬜ |
| D2 | 能否与 integration 四卡服务同机并存（见 §2，账面看可以，需实测） | ⬜ |
| D3 | kt push 目标 = origin(wenxuewuhd/ktransformers-AK)？push 后可否直接摘两个提交 | ⬜ |
| D4 | M3 优化目标：bs=1 延迟 vs bs=32 吞吐 | ⬜ |

## 变更记录

- 2026-09-18：建档。调研完成（两个 agent + flash_dev / dsv41-model / dsv4_a5_single_card 回复）；
  ktransformers 分支建好；§2 更正：engram 与 integration 共享，并存时增量只有专家 ≈195 GiB。
- 2026-09-18：§2 加 NUMA 放置决定（专家两节点均分，engram 不动；单节点放不下两者）。
