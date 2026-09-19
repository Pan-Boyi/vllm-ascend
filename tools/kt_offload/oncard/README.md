# 单层上卡测试（T0 / T1 / T2）

本目录只在 `run_oncard.sh` 里碰卡。脚本写于 2026-09-19，**还没上过卡**：判据与 CPU 自检已验证；NPU 这一半
（`npu_moe.py` 的运行时初始化、shim 层、forward context）第一次上卡时才会真正执行，见文末「首次上卡的风险」。

## 用法

```bash
cd /workspace/code/dsv4_single_card/vllm_project/worktrees/moe-offload
bash tools/kt_offload/oncard/selftest_cpu.sh                 # 不碰卡，不拿租约；应 SELFTEST: PASS
bash tools/kt_offload/oncard/run_oncard.sh <卡> t0 [--layers 0,20,39]
bash tools/kt_offload/oncard/run_oncard.sh <卡> t1 [--sets prefix,explicit --n 100]
bash tools/kt_offload/oncard/run_oncard.sh <卡> t2 [--tokens 4 --inputs 3]
```

产物：`build_artifacts/moe_offload_ref/oncard_runs/<UTC 时间>_<test>/`：`summary.txt`（外壳的每一步，含租约拿到和释放的毫秒时刻）、
`test.log`、`t*_result.json`、`provenance.json`（两棵树的 commit 和脏文件数、kt wheel sha256、命令行、环境）、
`code/`（本次实际执行的脚本快照）、`ascend_log/`（plog）、`card_before/after.txt`、`dmesg_{before,after,new,card}.txt`、`meminfo.txt`。

## 外壳强制的规则（run_oncard.sh）

| 规则 | 做法 |
|---|---|
| 先拿租约 | `LEASE_PID=$$ build_artifacts/card_lease.sh acquire <卡> dsv41_vllm_singlecard`，拿不到 rc 4，什么都不跑。拿租约之前只做主机侧检查（npu-smi 在不在、树、kt wheel、MemAvailable ≥ 150 GiB） |
| 同生共死 | EXIT trap：停掉测试的进程组 → 等该卡进程表里不再有测试 pid（pid 没了不等于上下文释放了，最多 30 min）→ release。外壳被 SIGKILL 时由看门狗 release；看门狗每 3 s 查一次外壳，外壳一没（僵尸也算没）就收尾退出，不会比外壳多活一个周期 |
| 起跑前 | 该卡 HBM < 20% 且进程表为空；共享 `check_card_dmesg.sh <卡>` 返回 0（非 0 就不跑；`ONCARD_SKIP_DMESG_GATE=1` 可以跳过，跳过会记进 summary） |
| 单卡 | `ASCEND_RT_VISIBLE_DEVICES=<卡>`；进程里不初始化 torch.distributed，也不起 HCCL（DP=1 时 prepare/finalize 是空操作，测试里直接跳过） |
| 私有缓存 | `TMPDIR=TMP=/dev/shm/dsv41_singlecard/tmp`，`TORCH_EXTENSIONS_DIR` 和 `TRITON_CACHE_DIR` 也在 `/dev/shm/dsv41_singlecard/` 下；**不 export TEMP**；`unset LD_PRELOAD`；环境来自 `source env_dsv41.sh moe-offload <卡>`（`DSV41_VLLM_TREE=moe-offload-vllm`），kt_site 追加在 PYTHONPATH 末尾 |
| 异常立停 | 每 2 s 扫一遍 test.log 和 plog，看有没有 `oncard_common.ANOMALY_PATTERNS`（AICore / aicore exception / MTE_ERROR / EZ9999 / EE9999 / …）。命中就立刻给进程组发 TERM，不重试，现场全部保留，rc 20 |
| 其他止损 | MemAvailable < 100 GiB → rc 21；超时（t0 40 min、t1 60 min、t2 40 min，可用 `ONCARD_TIMEOUT_S` 改）→ rc 124 |
| 前后对照 | npu-smi usages/proc-mem；`dmesg \| tail -n 200` 前后按内容求差，只报 `udevid=<卡>` 的行，其他卡的行数和没写 udevid 的错误行只计数、不丢；MemAvailable 在前、在后、运行中的最小值；HBM 占用率的采样最大值；测试进程的 VmHWM |
| 快照 | 起跑前把本目录复制到 `code/`，然后 exec 那一份（运行中改源文件不会破坏这一发）。exec 不换 PID，所以租约登记的 PID 不变 |

rc：0 通过 | 1 判据 FAIL | 2 用法错误 | 3 起跑前检查没过（没碰卡）| 4 没拿到租约 | 5 卡不空闲或 dmesg 闸没过 | 20 异常特征 | 21 主机内存不足 | 124 超时。

## 判据

公共：每条必需检查打一行 PASS/FAIL；**必需检查数为 0 或变异臂数为 0 都判 FAIL**；最后一行是 `VERDICT: PASS|FAIL`。
逐位相同的判法是「字节相同**或**值相同」：+0 和 −0 算相同，同一位置都是 NaN 也算相同（`oncard_common.bit_identical`）。

### T0：offload 关，单层 384 专家全驻留（ALLGATHER，world_size=1）
- 先把 ALLGATHER + W4A8（`npu_moe_init_routing quant_mode=3`）这条路单独验一遍：它在 V4.1 上没跑过，offload 关了也要走。
- 同一输入跑两次，结果必须**逐位相同**；输出里不能有 NaN/Inf。
- 先量地板：q = rel_l2(MXFP8 激活参考, bf16 激活参考)，即 fp8 激活本身带来的误差；另报累加顺序地板。
- 判据：rel_l2(NPU, bf16 参考) ≤ 2·max(q, 2⁻⁸)，最差行也按同样方式判。
- 只报告、不作判据的：NPU 离「单边 clamp」和「对称 clamp」两种参考哪个更近（回答设计文档里的开放问题：clamp_value 是哪种语义）；NPU 是不是比 bf16 参考更靠近 MXFP8 模拟（量化口径的指纹）。
- 变异臂（改的是上传到卡上的字节，在 post-load hook 之前改；被改的专家在每一行都被路由到）：该专家 nibble 互换、该专家 w2 scale 整体 +1。两条都必须 RED。CPU 自检里已验过这条判据的两个方向：理想的 MXFP8 实现能通过，scale+1 会 RED。

### T1：offload 开，单层分臂（prefix N=100 和非连续 explicit 集合各一遍，T=1/4/32）
- explicit 集合：100 个专家，不含 0、含 383、不连续 ⇒ 每个驻留专家的槽位号都不等于它的逻辑号。
- A 臂：NPU 驻留那一半（merge 之前）和「全驻留层 + 原始逻辑号 + CPU 路由权重置 0」比，必须**逐位相同**。
- B 臂：kt 的输出（在 `sync_forward` 取出）和只算 CPU 路由的 bf16 torch 参考比，用 M0b 的两档判据。参考侧做 nibble 互换必须 RED。
- C 臂：合成输出和 NPU 全专家输出比。地板 = |NPU 只算 CPU 路由 − bf16 参考|/|NPU 全专家|，也就是这些 CPU 路由上 fp8 激活带来的误差；判据 ≤ max(1.5 × 地板, 2⁻⁸)。另报符号一致率。
- 覆盖面：打印 CPU 路由占比，0% 或 100% 判失败；至少要有一行含 ≥2 条 CPU 路由（这时 NPU kernel 会在同一行看到重复的槽位 0）；另跑一遍 `check_split` 分流契约。
- 变异臂（直接替换生产对象上的 `_slot_tables` / `_keep_tables`）：槽位表整体错一位（只在 explicit 集合上做）⇒ A 臂必须 RED；keep 表全 1（CPU 路由在 NPU 上又算一遍，算了两次）⇒ A 臂和 C 臂都必须 RED。

### T2：图模式单层（prefix N=100，捕获尺寸 T=4，3 个输入）
- 在 `torch.npu.graph` 里捕获一次 route → apply → merge，然后换 3 个不同输入各 replay 一次。
- replay 的输出必须与同一输入的 eager 输出逐位相同；3 次 replay 两两不同；replay k 不能等于 eager k−1（旧输出指纹：输出等于上一个输入的结果，说明 host-func 这一步没真正执行）。
- `HOST_FUNC_CALLS[layer]`：捕获时 +0，每次 replay 恰好 +1。
- 不能出现 10701x：外壳统计 log 和 plog 里的行数写进 summary（107011 会被我们吞掉，按设计不会外抛；出现任何一条都要人看）。
- 变异臂：把 `_capturing` 和 kt 的捕获态探针都强制返回 False 再捕获一次，必须 RED（要么捕获时抛错，要么出现上面那种旧输出指纹，要么和 eager 不一致）。这一步放在最后跑。
- 整网图模式（FULL_DECODE_ONLY / PIECEWISE × BREAKABLE、多请求并发）不在这里做，留给 M2 整网验证。

## 预计时长与资源（给排卡方；**推断，未实测**，第一发之后按 summary 里的实测值更新）

| 测试 | 墙钟（含加载） | HBM 峰值 | 主机内存峰值（本进程） | 说明 |
|---|---|---|---|---|
| t0，1 层（默认 L20） | 10–15 min | ≈ 14 GiB | ≈ 10 GiB | 读 384 专家的原始字节，共 6.72 GiB（冷盘 1–3 min）；format cast 期间输入和输出同时占卡，≈ 2 × 6.72 GiB；两条变异臂各重建一次层 |
| t0，3 层（0,20,39） | 30–45 min | ≈ 14 GiB | ≈ 10 GiB | 要设 `ONCARD_TIMEOUT_S=3600` |
| t1（两个集合 × T=1/4/32） | 25–40 min | ≈ 17 GiB（全驻留 6.72 + N=100 的 1.75，外加 cast 的临时占用） | ≈ 30 GiB（原始字节 6.72 + 驻留子集 1.75 + kt BufferB 284×17.93 MiB ≈ 5 GiB + kt 加载时的瞬时 ≈ 15 GiB） | kt 线程：`--threads 32 --numa 1`（node1 余量大，见计划 §2） |
| t2 | 5–10 min | ≈ 5 GiB | ≈ 25 GiB | 只加载 100 个驻留专家 |

整机约束：起跑要求 MemAvailable ≥ 150 GiB，运行中低于 100 GiB 就自停（计划 §2 的规则）。一次只跑一个测试，每个测试进程只用一张卡。
建议顺序：t0 → t1 → t2，前一个 PASS 再跑下一个（t1 的 A、C 两臂拿 t0 走过的那条路做参照）。

## 范围：哪些是真的，哪些是替身

- 真实对象：`AscendW4A8MXFPDSDynamicFusedMoEMethod`（其 `process_weights_after_loading`、`apply`、`apply_gmm1_act_quant`、`apply_gmm2`）、`AllGatherCommImpl.fused_experts`（`TokenDispatcherWithAllGather` → `npu_moe_init_routing` → GMM → `npu_moe_token_unpermute`）、`KTLayerOffload`（`attach_after_init` 包装 post-load hook、`route`、`merge`、captured 路径）、`plan_from_config`、kt-kernel wheel。
- 替身：层对象是一个 `nn.Module`，只带参数和 `apply()` 要读的那些属性；`moe_config` 是一个 namespace；forward context 也是 namespace，只给了 `moe_comm_method`；量化 method 的 `__init__` 只读两个配置值，这里直接设成 V4.1 的值（group_size=32，dynamic_eplb=False）；`VllmConfig()` 用默认值调 `init_ascend_config`。
  ⇒ `AscendRoutedExperts` 自己的构造和 `forward_impl` 挂点这里**没有覆盖**（forward_impl 是逐个调用照抄的）。挂点由 CPU UT 的 AST 守卫和整网冒烟（设计文档「上卡验证草案」步骤 3/4）覆盖。
- 没测的：prepare/finalize（DP=1 时是空操作）；profile run 的随机路由；多层之间 kt 缓冲按 `layer_idx % 2` 轮换使用。

## 首次上卡的风险（第一发要人盯）

1. `npu_moe.init_runtime`：`bootstrap_custom_op_env` + `import vllm_ascend.vllm_ascend_C` + `VllmConfig()` 默认值 + `init_ascend_config`。服务进程里这些由 platform 插件和 camem 完成，这里是手动照做的；默认的 ascend 配置（例如 `fusion_ops_gmmswigluquant=True`）与服务的配置是否一致没有核过。
2. forward context 用的是 namespace：MoE 这条路径只读 `moe_comm_method` 和 `use_mega_moe`，已用 grep 核过；其余字段读出来都是 None。
3. 同一层在同一个进程里先后建两个 `KTMoEWrapper`（prefix 和 explicit）：M0b 只在不同层上这样做过。如果第二个集合构造失败，改成分两发跑：`--sets prefix` 一发、`--sets explicit` 一发。
4. T0 的 2× 地板判据是按 MXFP8 的理论误差定的。如果 NPU 实际的激活量化方式和模拟不同，报告里的「口径指纹」会显示出来：先看它，不要直接放宽阈值。
5. 异常特征里的 `AICore` 按用户口径是整词匹配。如果某个 CANN 版本在正常日志里也会打印这个词，就会误停（误停方向是安全的）；遇到这种情况，把那一行原样贴出来，再改 `ANOMALY_PATTERNS`。
