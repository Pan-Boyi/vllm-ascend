# 单卡 MoE offload 工具（本线自有，不在 integration 的共享脚本里）

2026-09-18 由 dsv41-integration 交接（用户规矩：单卡改动不进多卡基线的树与脚本）。

- `serve_v41_singlecard.sh`：共享 `build_artifacts/serve_v41_bringup.sh` 的副本 + `KT_OFFLOAD=1` 分支
  （默认路径与原版逐字相同，integration 用 24 种环境组合比对过）。闸改为按本脚本所在目录找。
  ⚠️ 共享原版以后若有修复，本副本不会自动同步 —— 起飞前 diff 一次。
- `kt_offload_gate.py`：起飞前显存 + 主机内存（逐 NUMA）两道闸；rc 0 放行 / 1 拒绝 / 2 算不出。
  `KT_DEVICE_FACTOR`、进程开销两个常数待首次上卡实测校准。
- `dsv41_expert_counter.py` / `routed_experts_counter_hook.diff` / `expert_count_report.py`：M0c① 路由频率计数器（离线测过）。
  钩子要进本树的 routed_experts.py，只在 `enable_force_load_balance=False` 时累加。
