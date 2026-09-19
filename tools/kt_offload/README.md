# 单卡 MoE offload 工具（本线自有，不在 integration 的共享脚本里）

2026-09-18 由 dsv41-integration 交接（用户规矩：单卡改动不进多卡基线的树与脚本）。

- `serve_v41_singlecard.sh`：共享 `build_artifacts/serve_v41_bringup.sh` 的副本 + `KT_OFFLOAD=1` 分支
  （默认路径与原版逐字相同，integration 用 24 种环境组合比对过）。闸改为按本脚本所在目录找。
  ⚠️ 共享原版以后若有修复，本副本不会自动同步 —— 起飞前 diff 一次。
  2026-09-19 已同步共享原版的逐卡 dmesg 闸（`gate_cards.sh`，共享版 sha256 31b5ca8d…）；默认路径 6 格 dry-run 逐字节相同
  （`build_artifacts/moe_offload_ref/m0_evidence/serve_sync_0919_dryrun/`）。
- `oncard/`：单层上卡测试 T0/T1/T2 与外壳 `run_oncard.sh`（租约、异常立停、产物），见其 README。
- `kt_offload_gate.py`：起飞前显存 + 主机内存（逐 NUMA）两道闸；rc 0 放行 / 1 拒绝 / 2 算不出。
  `KT_DEVICE_FACTOR`、进程开销两个常数待首次上卡实测校准。
- `dsv41_expert_counter.py` / `routed_experts_counter_hook.diff` / `expert_count_report.py`：M0c① 路由频率计数器（离线测过）。
  钩子要进本树的 routed_experts.py，只在 `enable_force_load_balance=False` 时累加。

## 路由频率计数器（M0c①）—— 本树内置版

树内实现：`vllm_ascend/kt_offload/expert_counter.py`，钩子已接进 `vllm_ascend/ops/fused_moe/routed_experts.py`
（`_select_experts` 里 router 输出之后、log2phy 之前；构造时 `check_config`）。本目录的 `dsv41_expert_counter.py`
与 `routed_experts_counter_hook.diff` 是 integration 交接的原件，**不再使用**，只留作出处。与原件的差异：

- **ALLGATHER + DP>1 的行切片**：这一步 router 看到的是所有 DP rank all-gather 后的行（每 rank 补齐到
  `max_tokens_across_dp`）。原件取前 `num_actual_tokens` 行 ⇒ 每个 rank 都在数 rank0 的 token。现在取本 rank 的
  `[dp_rank × max_tokens_across_dp, + num_actual_tokens)`；行数对不上就跳过并记入 `skipped`。
  （EP=4 多卡服务在 token 数超过 MC2 容量时会走 ALLGATHER：`ascend_forward_context.py:326-330`。）
- **图模式显式报错**：`--enforce-eager` 没开时，构造 MoE 层就抛 `ExpertCounterError`；在捕获中调用也抛。不会静默只计捕获那一次。
- 开关变量登记在 `vllm_ascend/envs.py`；不设时两个调用点都被 `if _expert_counter.ENABLED:` 挡住，UT 用 AST 检查守卫。

### 多卡采集的起法（offload 关，WT=moe-offload，eager）

```bash
OUT=/dev/shm/dsv41_expert_count/$(date +%Y%m%d_%H%M)_gsm8k     # 每次一个新目录，别和旧产物混
mkdir -p "$OUT"
# 先拿租约（build_artifacts/card_lease.sh acquire），卡号按协调结果
WT=moe-offload EAGER=1 NPUS=0,1,2,3 \
VLLM_ASCEND_DSV41_EXPERT_COUNT="$OUT" \
  /workspace/code/dsv4_single_card/vllm_project/build_artifacts/serve_v41_bringup.sh
```
- 用共享的 `serve_v41_bringup.sh`（多卡默认路径），**不要**用本目录的 `serve_v41_singlecard.sh` 的 `KT_OFFLOAD=1` 分支。
- `WT=moe-offload` 过得了脚本的"不落后于 dsv41-base"闸（2026-09-18 核：本树包含 dsv41-base）。
- 必须 `EAGER=1`，否则服务起不来（这是设计：图模式下计数会漏掉全部 replay）。
- 起来之后跑评测（GSM8K 等，口径同基线）。每个 rank 每 200 步落一次盘，进程退出时再落一次：
  产物是 `$OUT/rank<dp>_pid<pid>.pt`，每个 DP rank 一个文件，不会互相覆盖。
  **评测跑完后要正常停服务**（`stop_by_port.sh`，别 `kill -9`），否则最后不足 200 步的部分不落盘。
- 第二类负载（M0c 要两份做重合度）：换一个新的 `$OUT` 目录重起一次，跑另一类数据。

### 汇总与判据

```bash
python3 tools/kt_offload/expert_count_report.py "$OUT"                    # 单份：判据 + [40,384] + 每层 top-100
python3 tools/kt_offload/expert_count_report.py "$OUT_A" --compare "$OUT_B"  # 两份 top-100 逐层重合度
```
- **判据**（任何一条不满足退出码 1）：
  1. 和判据（逐 rank、逐层、逐 bucket）：`counts.sum() == tokens × 6`。它只证明"每行数了 6 次"，
     **对数错了哪几行没有区分力**——数别的 rank 的行，和照样对。
  2. 身份判据（逐层、逐 bucket、rank 两两）：两个都有 token 的 rank，其 `[384]` 计数不能完全相同，相同则点名那对 rank。
     ALLGATHER 切片 bug（每个 rank 都数 rank0 的行）的指纹就是各 rank 计数相同；UT 里旧切片喂 4 个 rank 时
     判据 1 仍绿、判据 2 变红。输出里另有各 rank 的 token 数和"比较了多少格、多少格有重复"的摘要；
     比较格数为 0（只有一个 rank 有数据）时身份判据零覆盖，会给警告。
     ⚠️ 真实负载下两个 rank 同一步路由完全相同几乎不可能；若报红，先看是不是同一批请求被复制发到了多个 rank。
  各层 token 数应一致（每层路由同一批 token）；`skipped` 与 `steps` 原样打印，覆盖量和结论一起报。
  另核：4 个 rank 文件都在（`rank0..rank3`），`skipped` 里 `allgather rows ...` 这类原因应为 0 或极少。
- 产物 `$OUT/expert_counts_total.pt` 可以直接给 `kt_offload_config.frequency_table_path`：
  `placement.load_frequency_table` 取 `counts["all"]`（`[40,384]`，decode+prefill 合计）。只想按 decode 选驻留集时，
  另存 `{"counts": total["counts"]["decode"]}`。
- 汇总命令用 `rc=$?` 判退出码，别接管道。
