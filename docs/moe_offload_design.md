# DSV4.1 单卡 MoE offload（vLLM-Ascend + kt-kernel）M1 设计

> 范围：M1 = eager（`--enforce-eager`）。M2 图模式只预留接口（§f）。
> 行号基于本分支 `dsv41-moe-offload`（vllm-ascend）与 `worktrees/moe-offload-vllm`（vllm，`dd63fc7d6`）。
> 标 ⚫ 的是依赖 kt-kernel 或上卡才能确认的点。跟踪文档是 `MOE_OFFLOAD_PLAN.md`，本文不维护状态。

代码位置一览：

| 文件 | 作用 |
|---|---|
| `vllm_ascend/kt_offload/config.py` | `additional_config.kt_offload_config`（`KTOffloadConfig`） |
| `vllm_ascend/kt_offload/placement.py` | 驻留集、逻辑号↔槽位、路由分流、分流判据 `check_split`。纯 torch，CPU 可测 |
| `vllm_ascend/kt_offload/layer.py` | 每层运行时 `KTLayerOffload`（kt wrapper、eager submit/sync）和 `create_layer_offload` |
| `vllm_ascend/ops/fused_moe/routed_experts.py` | 6 处挂点：构造 1 处（关时立即返回 None）+ 5 处 `kt_offload is not None` 守卫；UT 用 AST 检查守卫 |
| `vllm_ascend/ascend_config.py:445,1509` | 注册配置段和前置条件校验 |
| `tests/ut/kt_offload/` | CPU 单测（含变异臂） |

---

## a. 配置入口

`--additional-config '{"kt_offload_config": {...}}'`，字段定义在 `kt_offload/config.py:23-105`，挂到
`AscendConfig.kt_offload_config`（`ascend_config.py:445`），`init_ascend_config` 里调用
`_validate_preconditions`（`ascend_config.py:1509`）。

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `false` | 总开关。只有显式 `true` 才生效（`layer.py:72 offload_enabled` 用 `is True`，MagicMock/缺段都算关） |
| `weight_path` | `null`=模型目录 | kt 读 CPU 专家的 checkpoint 目录（原生 MXFP4 safetensors） |
| `method` | `"MXFP4"` | kt 后端，显式传给 `KTMoEWrapper` |
| `resident_strategy` | `"prefix"` | `prefix` / `frequency` / `explicit` |
| `num_resident_experts` | 0 | prefix/frequency 下每层驻留数 N（启用时必填） |
| `frequency_table_path` | — | `[40, 384]` 计数表（主干层 0..39 × 全局专家号），`.pt/.npy/.json` |
| `resident_experts_path` | — | explicit：JSON，`{"layers": {"0": [ids...], ...}}` 或按层的列表 |
| `cpuinfer_threads` | 0 | 启用时必填（无隐式默认，决定占用哪些主机核） |
| `threadpool_count` | 2 | kt NUMA 子池数（计划 §2 决定两节点均分） |
| `numa_nodes` | `null` | 长度必须等于 `threadpool_count` |
| `chunked_prefill_size` | 0=`max_num_batched_tokens` | kt 单次最大 token 数；超出 kt 直接 raise |
| `skip_cpu_in_profile_run` | `true` | 见 §h |
| `force_sync_submit` | `true` | eager 下设 `KT_FORCE_SYNC_SUBMIT=1`，走 kt 的同步 submit/sync |

前置条件（`config.py:107-130`）：TP=DP=PP=1、无 EPLB、`--enforce-eager`、无投机解码。
层级前置条件（`layer.py:100-116`）：层号 < `num_hidden_layers`（MTP 层拒绝）、无冗余专家、`ep_size=tp_size=1`、
`mix_placement=false`、`dynamic_eplb=false`。

**换驻留集只改配置重启**（flash_dev 要求）：三种策略都在启动时解析成每层一个 id 集合（`placement.py:331 plan_from_config`），
不假设各层相同、不假设连续；explicit 还允许各层个数不同（§b.3）。

## b. 专家掩码 / 驻留集 / 加载

### b.1 三个编号空间（`placement.py:1-20`）

- **逻辑号** 0..383：路由输出、checkpoint 键 `layers.L.ffn.experts.<id>`、kt 的 `gpu_experts_mask[id]` 都用它。
- **槽位** 0..num_slots-1：NPU 权重张量的行号 `w13_weight[slot]`。驻留专家按逻辑号**升序**紧排，所以 prefix 下槽位 = 逻辑号，
  frequency/explicit 下是一次物理重排（逻辑号 → 槽位，`LayerPlacement.slot_of`）。
- `CPU_SLOT=-1`：只在加载期出现（告诉 loader "不在本卡"），**绝不进 NPU kernel**。

### b.2 为什么不用 EPLB 的 expert_map / log2phy

- `init_eplb_config` 在 `ep_size==1` 直接返回 `None, None, None`（`eplb/core/eplb_utils.py:82-84`），`log2phy` 恒为 None。
- `TokenDispatcherWithAllGather` 在 `expert_map is None` 时取 `first_expert_idx=0, last_expert_idx=num_experts_local,
  expert_num=num_experts_local`（`token_dispatcher.py:392-410`）。它只假设"本卡槽位是 0..N-1 连续"，**不假设逻辑号连续**。
  所以不走 expert_map：路由在进入 dispatcher 之前已被改写成槽位号 ∈ [0, N)，dispatcher 看到的就是一个 N 专家的普通 MoE。
  `active_expert_range`/`first_expert_idx` 的 EP 语义完全不被触发。
- 结论：frequency 的非连续驻留集通过"逻辑号→槽位"重排容纳，dispatcher / EPLB 代码零改动。

### b.3 分配 N 个槽位

- 上游 factory 用 `expert_map_manager.local_num_experts`（ep=1 时 = 384）填 `moe_config.num_local_experts`
  （vllm `fused_moe/layer.py:344`），`RoutedExperts.__init__` 用它建权重（vllm `routed_experts.py:160`）。
- 挂点：`AscendRoutedExperts.__init__` 在调父类构造**之前**调 `create_layer_offload`（`routed_experts.py:356-366`），
  它按名字绑定参数（`layer.py:66 bind_routed_experts_args`，不按位置），把 `moe_config.num_local_experts` 改成 `num_slots`
  （`layer.py:120`）。父类构造后 `attach_after_init` 把 `layer.local_num_experts` 也改成 `num_slots`（`layer.py:150`）。
- `init_eplb` 的容量检查按 384/ep_size 算，会拒绝 N：offload 时期望值改为 `num_slots`（`routed_experts.py:477-478`）。
- **num_slots 全模型一个数**：ALLGATHER 的 `AllGatherCommImpl` 是按类型存的进程级单例，每层 `setup_moe_comm_method`
  都会用本层 `moe_config` 覆盖它（`moe_comm_method.py:55-62`），dispatcher 的 `num_local_experts` 取自最后一层
  （`moe_comm_method.py:203-207`）。所以各层槽位数必须相同：`num_slots = max(每层驻留数)`（`placement.py:317`），
  驻留少的层尾部槽位空置；空置槽位在后处理前清零（`layer.py:176 zero_unused_slots`），且不会有路由指向它们。
- 每层至少 1 个驻留专家（`placement.py:62-66`）：CPU 路由要一个"真的加载过权重"的槽位做 0 权重目标，
  未加载的字节乘 0 仍可能是 NaN。

### b.4 weight_loader 跳过 CPU 专家

- 上游 `weight_loader` 先 `_map_global_expert_id_to_local_expert_id`，得 -1 就返回 False 跳过
  （vllm `routed_experts.py:598,605`）。`AscendRoutedExperts` 覆写这个方法：offload 时返回 `placement.slot_of(id)`
  （`routed_experts.py:563-568`），CPU 专家得 -1，驻留专家落到自己的槽位。
- V4.1 的 `load_weights` 对专家参数无论成败都 `loaded_params.add(name_mapped)`（vllm `deepseek_v4_1/nvidia/model.py:769-780`），
  不会因跳过而报"权重未加载"。E8M0 scale 仍按 uint8 视图字节搬（同文件 :747 注释处），offload 不改这条路。
- **在线调整不做**：kt 在 `load_weights` 时把 `gpu_experts_mask` 的原始指针交给 C++，且子集加载后驻留专家在 CPU 侧没有
  BufferB（kt `operators/common.hpp:270-298`）——load 之后改 mask 会空指针。若将来要在线换集合，需要：
  kt 侧支持按专家增删 BufferB 并重建 mask；vLLM 侧重新加载被换入专家的 NPU 权重、在不在飞的时刻整体替换两张设备表；
  图模式下两张表本来就是常驻缓冲区，替换要原地 `copy_` 不能换对象（否则被 FULL 图烘死）。M1 不做。

## c. 路由分流（`routed_experts.py:685-695, 738-742`；`placement.py:126-141`）

顺序（forward_impl）：
1. `_select_experts`（`routed_experts.py:589-642`）：路由 op 输出权重时**已乘 routed_scaling_factor=1.5**
   （`router/fused_topk_router.py:202,224`，vision 分支 :74-75）；`log2phy` 在 ep=1 为 None；profile run 的随机替换
   （:634-640）也在这里，产出的仍是逻辑号 0..383。
2. `kt_offload.route(...)`（`layer.py:226-256`），必须在第 1 步**之后**：
   - kt 拿**原始** topk（逻辑号 + 已缩放权重）：`wrapper.submit_forward(h, topk_ids, topk_weights, stream)`。
     kt 自己按 `gpu_experts_mask` 跳过驻留专家（kt `common.hpp:302-304 should_skip_expert`）。
   - NPU 拿改写后的：`ids = safe_slot_table[ids]`，`w = w * keep_table[ids]`（两次 gather + 一次乘，表在设备上，
     int32/int64、fp32/bf16 各预烘一份）。驻留路由 → 自己的槽位、权重不变；CPU 路由 → 槽位 0、权重 0。
     形状/dtype 不变，不产生 -1。
3. `quant_method.apply` → ALLGATHER dispatch/GMM/combine。combine 用 `probs=topk_weights.float()`（`token_dispatcher.py:435`），
   0 权重行的贡献被乘掉；W4A8 路径的 `npu_swiglu_group_quant(topk_weight=None)` 不在 MLP 内乘权重，所以只在 combine 乘一次。
   - **判定基于逻辑号**：`log2phy` 在 ep=1 恒为 None，且 offload 启用时若出现 log2phy 直接拒绝（`layer.py:151-152`），
     所以分流看到的永远是 log2phy 之前的逻辑号。
   - **mix_placement 拒绝**（`layer.py:113-114`）：它会把 shared expert 以编号 384 追加进 topk（`routed_experts.py:616-631`），
     分流会把它当 routed 专家；V4.1 默认不开。
4. `finalize` 之后 `kt_offload.merge(routed_out)`（`layer.py:258-267`）：`sync_forward` 取回 CPU 输出，
   **fp32 相加后回 bf16**（少一次舍入）。这是 routed 输出的语义点：两边都已含 1.5 缩放，runner 的
   `routed_scaling_factor` 传的是 1.0（vllm `fused_moe/layer.py:131,393,418-424`，`apply_routed_scale_to_output=False`），
   shared expert 在之后由 runner 加（`fused_moe.py:347-363`，vllm `moe_runner.py:787`）。⇒ CPU 输出不再乘 1.5，也不含 shared。
- 集合判据（`placement.py:144 check_split`，按 kt 的 mask 而不是 NPU 的表判定，避免两表同错）：
  NPU 有效路由（权重≠0）与 kt 路由互补、并集覆盖全部 topk；NPU 权重 + CPU 权重 == 原权重（逐位）；NPU id ∈ [0, 驻留数)。
- 已知代价（M3 处理）：槽位 0 被当 dummy，bs=1 时 CPU 路由的 ≈4.4 行会让专家 0 在它本不被选中时多算一次，
  约多读一个专家的权重（≈18 MB/层）。候选：dummy 取本行自己的某个驻留专家；或 dispatcher 走 active_expert_range 丢行
  （需要 A5 上验证 `npu_moe_init_routing` 对越界号的行为）。sglang 同样用槽位 0（`kt_ep_wrapper.py:331-359`），
  其 GSM8K 96.74% 是"同一行专家号重复在 A5 路由 kernel 上可接受"的旁证（未单独验证，单层用例会覆盖）。

## d. kt wrapper 生命周期

- 每层一个 `KTLayerOffload`，`layer_idx` 从模块名 `...layers.<N>.ffn.experts` 解析（`layer.py:41`），与 checkpoint 的
  `layers.N` 对齐；kt 的 MXFP4 loader 会试 `model.layers.N` 去掉 `model.` 前缀的形式（kt `utils/loader.py:1220-1224`），
  V4.1 键就是 `layers.N.ffn.experts.E.w1.weight`（已核 index.json）。
- 构造时机：包装 `quant_method.process_weights_after_loading`（`layer.py:148-174`，模型 loader 按
  `module.quant_method.process_weights_after_loading(module)` 调，vllm `model_loader/utils.py:111-113`）：
  清零空槽 → 原后处理（W4A8 的 NZ format cast）→ `after_weights_loaded`：设备表预烘到权重所在设备，
  构造 `KTMoEWrapper(layer_idx, 384, 6, 5120, 2304, gpu_experts_mask=resident_mask, cpuinfer_threads, threadpool_count,
  weight_path, chunked_prefill_size, method="MXFP4", numa_nodes, swiglu_limit=10.0)` 并 `load_weights(arange(384))`
  （EPLB 关，物理=逻辑）。全驻留的层不建 wrapper。
- `swiglu_limit` 取自 `moe_config.swiglu_limit`（V4.1=10.0，vllm `deepseek_v4/nvidia/model.py:1012`）；为 None 直接拒绝
  （`layer.py:153-162`），因为 kt 的 0.0 表示不 clamp，NPU 侧却 clamp（`w4a8_mxfp4.py:258`）。
- CPUInfer 是进程级单例，第一次构造的线程/NUMA 参数生效（M0 已核）。

## e. eager 同步时序（M1）

每层：
```
router(topk, 已×1.5) ─► route: D2H(h, ids, w → pinned, non_blocking)
                              → torch.npu.synchronize()       (kt _wait_device，保证 D2H 完成)
                              → cpu_infer.submit(task)          (CPU 开算，与下面 NPU 重叠)
                        NPU: split → dispatch → GMM(驻留) → combine → finalize
merge: sync_forward → [KT_FORCE_SYNC_SUBMIT=1: cpu_infer.sync()] → H2D(output_cpu → output_gpu, non_blocking)
       → routed_out = (routed_out.f32 + cpu_out.f32).bf16      (与 H2D 同流，流序保证)
```
- `force_sync_submit=true` 让 kt 走 bypass 分支（kt `experts_base.py:73-83,846-890`）：不订阅流、不起 ACL 回调，
  merge 时只做主机侧 `cpu_infer.sync`，每层 1 次设备同步（在 submit 前）。关掉则 kt 在 sync 时再做一次
  `torch.npu.synchronize`。
- kt 的 pinned/输出缓冲按 batch size 缓存（`KExpertsCPUBuffer`，slot=layer_idx%2），eager 下 bs 变化会重分配
  （含设备上的 `output_gpu`）。merge 的结果是新张量，不写回 kt 的缓冲。

## f. M2 预留（不在 M1 实现）

照 `models/deepseek_v4_1/engram_embedding.py:270-307` 的做法：
- 捕获下 `route` 改为 `copy_inputs_to_cpu_buffers` + `torch_npu.npu._launch_host_func(stream, run_pinned_forward_sync)`，
  `merge` 改为 `copy_forward_output_to_device`（kt `experts_base.py:602,684,830`）。
- 订阅**捕获流**（107015），只吞 107011；进程设 `KT_EXTERNAL_NPU_REPORT_SUBSCRIBER=1`，kt 不再自带 poller/订阅。
- 设备表已在 `process_weights_after_loading` 预烘（107030：捕获期 H2D 被拒）——M1 已这样做。
- 所有捕获 bs 先 `KTMoEWrapper.set_capture_batch_sizes(...)` 登记（只有登记过的 bs 永久缓存 pinned 缓冲），payload 活到进程结束。
- 捕获态判断：kt 计划加 `set_capture_probe(fn)` 注入点（M0 反馈，尚未合入）；vLLM 侧注入
  `torch.npu.is_current_stream_capturing` 或 forward context 的 capturing 标志，替掉 kt 里写死的 sglang 探针
  （kt `experts_base.py:113-124`）。
- 接口：`KTLayerOffload.route/merge` 内按"是否捕获"分支即可，routed_experts.py 的挂点不变。
- 删 `config.py:125-127` 的 enforce_eager 前置条件。

## g. world_size=1 的 comm 路径

- `select_moe_comm_method`：`not enable_expert_parallel or get_ep_group().world_size == 1` → `ALLGATHER`
  （`ascend_forward_context.py:356-357`）。`setup_moe_comm_method` 在 ep_size=1 只建 AllGather（`moe_comm_method.py:61-62`）。
- 这条路在 dp=1 时 prepare/finalize 全是 no-op（`prepare_finalize.py:455-573`），`pertoken_scale=None`，于是 W4A8MXFP
  的激活量化被融合进 `npu_moe_init_routing(quant_mode=3)`（`token_dispatcher.py:381`）。
- **已知缺口**：V4.1 多卡基线走的是 MC2（EP≥3），ALLGATHER + `quant_mode=3` + A5 这个组合在 V4.1 上没跑过；
  与 offload 无关、关掉 offload 同样会走。所以上卡第一步先跑"offload 关、全 384 专家驻留"的单层（§4 步骤 1），把它和 offload 分开。

## h. profile run 强制随机路由

- `enable_force_load_balance = _EXTRA_CTX.in_profile_run`（`routed_experts.py:659`），随机替换在 `_select_experts` 里、
  分流之前（:634-640）⇒ 两边看到同一组随机逻辑号，约 3/4 路由落到 CPU。
- 处理：`skip_cpu_in_profile_run=true` 时 profile run 不提交 kt、merge 为 no-op（`layer.py:241-245`）。NPU 侧照常改写，
  dispatcher 行数仍是 T×6 ⇒ NPU 峰值显存与真实 decode/prefill 同形，**profile 不失真**。
- 不算进 profile 的：kt 的 `output_gpu` 设备缓冲（2×T×5120×bf16，T=4096 时 80 MiB）——建议 `--gpu-memory-utilization` 留出这点。
- 统计：M1 不采路由统计。以后若加计数，必须排除 `in_profile_run`。

## i. 显存与起服务

- 每驻留专家每层：w13 `[2I, H/2]` + w2 `[H, I/2]` + E8M0 scale `[2I, H/32]` + `[H, I/32]` = **18,800,640 B = 17.93 MiB**
  （`placement.py:362 resident_bytes_per_slot`，未计 NZ cast 可能的对齐填充）；40 层 = 0.70 GiB/槽。
- `N = floor((HBM 可用 − 其余权重 − KV/工作区) / (40 × 17.93 MiB))`；按计划 §1 的估算 (107 − 10.4 − 25) / 0.70 ≈ 100。
  实测做法：先 N=100 起，读 vLLM 日志的 model weights 占用和 KV 可用量再调整。
- 起服务参数：`--tensor-parallel-size 1 --data-parallel-size 1`（不加 `--enable-expert-parallel`）、`--enforce-eager`、
  `--max-model-len` 小值起、`--max-num-batched-tokens` ≤ kt `chunked_prefill_size`、`--kv-cache-dtype bfloat16 --block-size 128`、
  `--additional-config '{"kt_offload_config": {...}}'`；环境：`unset LD_PRELOAD`（jemalloc），PYTHONPATH 加 kt wheel
  目录 `build_artifacts/moe_offload_ref/kt_site`。
- **`serve_v41_bringup.sh` 需要协调（本线不改）**：
  - `:127 MIN_EP=3` 与 `:151` 的健康卡数检查、`:204` 的 `DP<3` 拒绝：offload 时每卡权重不是 `275.7/DP + 10.4`，
    而是 `10.4 + 0.70×N` GiB。建议加一个 `KT_OFFLOAD=1` 分支：`MIN_EP=1`，`:204` 改为按 `10.4 + 0.70×N` 判。
  - `:447-449` 固定传 `--data-parallel-size "$DP" --enable-expert-parallel`：offload 分支去掉 `--enable-expert-parallel`。
  - `:322 verify_shm_sharing.py --ranks "$DP"` 对 DP=1 应能直接用（待确认）。
  - `HCCL_BUFFSIZE`（MC2 用）在 ALLGATHER/单卡下无意义，保留无害。
  - 这些改动经 flash_dev（计划 §5）。

## 上卡验证草案（M1，供排卡；先拿租约 `card_lease.sh acquire`，TMPDIR=/dev/shm/dsv41_tmp）

步骤 1 —— **基线臂先行，把 ALLGATHER 缺口与 offload 分开**：单层小用例（真实 V4.1 第 L 层权重，L 取 0/20/39），
offload **关**、384 专家全驻留，走 `AscendRoutedExperts.forward_impl` 的 ALLGATHER 路径。对照：同层 torch CPU 参考
（MXFP4 反量化 + 激活按 MXFP8 量化，与 NPU 同口径）。判据先量地板：同输入跑两次看是否逐位确定，再看与参考的 rel_l2。

步骤 2 —— **单层分臂判据**（同一输入 x，T ∈ {1, 7, 256}，含随机路由与真实路由两种 topk）：
- 臂 A「NPU 驻留部分」：offload 开的 NPU 输出（merge 之前的 routed_out）vs offload 关、但把 CPU 路由权重手工置 0 的全驻留 NPU 输出。
  两者每行输入、权重、非零路由相同，**预期逐位相等**（字节相同 或 值相同）。GMM 的分组形状不同（N 组 vs 384 组），
  若不逐位，先量"只改分组形状"的地板（offload 关、同一路由把专家号做一次无关重排）再判；超地板 ⇒ 槽位映射或 dummy 行污染。
- 臂 B「CPU 部分」：kt 输出 vs bf16 torch 参考（只算 CPU 路由）；判据 rel_l2 在 M0b 实测地板内。
- 臂 C「合成」：驻留+CPU vs NPU 全专家。**不要求逐位**：CPU 部分是 bf16 激活、NPU 是 MXFP8 激活。
  先量地板 = NPU 全专家 vs bf16 参考 在 CPU 那组路由上的差（fp8 激活量化误差），判据 = 合成差 ≤ 地板 × 1.5 且高于 bf16 ULP 地板；
  另报 top-1 输出维度符号一致率。
- 变异臂（必须红）：把 `safe_slot_table` 换成恒等（frequency 集合下）、或不清 CPU 权重（双算）——臂 A/C 必须报红。
- 覆盖量打印：每次打印 CPU 路由数 / 总路由数，为 0 或等于总数都算失败（没覆盖到分流）。

步骤 3 —— **profile/起服务冒烟**：`--enforce-eager`、TP1/DP1、N=100 prefix、`max-model-len` 小值；看日志 plan 行
（`kt_offload plan: 40 layers x 384 experts, 100 NPU slots/layer`）、模型权重显存 ≈ 10.4 + 70 GiB、kt 加载后
`/proc/<pid>/numa_maps` 两节点各 ≈98 GiB、MemAvailable 前后差。

步骤 4 —— **整网 GSM8K 50 题冒烟**：贪心、thinking 关、max4096（口径同基线；harness 默认 temp=1.0 要显式改）；
对照 integration 多卡同 50 题逐题配对。再换 frequency 集合跑一次同 50 题（验证非连续集合路径）。

## 开放问题

- ⚫ NPU 驻留专家是 W4A8（激活量化为 MXFP8），kt CPU 专家是 bf16 激活（M0 反馈）⇒ "驻留+CPU" 与 "NPU 全专家" 不会逐位一致，
  CPU 部分差异应在 fp8 激活量化量级。上卡判据分臂写（见计划的上卡步骤）。
- ⚫ kt 的 swiglu clamp 语义（gate 只 clamp 上界、up 双边？）是否与 `npu_swiglu_group_quant(clamp_value)` 一致——M0b 对的是
  torch 参考，需确认参考与 NPU 算子语义一致。
- ⚫ `npu_format_cast(29)` 对 N 槽权重是否有对齐填充（影响 §i 的 17.93 MiB）。
- ALLGATHER + W4A8 `quant_mode=3` 在 A5/V4.1 未验证（§g）。
