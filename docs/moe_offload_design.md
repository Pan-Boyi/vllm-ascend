# DSV4.1 单卡 MoE offload（vLLM-Ascend + kt-kernel）M1 设计

> 范围：M1 = eager；M2 = 图模式（FULL / PIECEWISE，§f）。侧流 overlap 只留开关（M3）。
> 行号基于本分支 `dsv41-moe-offload`（vllm-ascend）与 `worktrees/moe-offload-vllm`（vllm，`dd63fc7d6`）。
> 2026-09-19 变基到 dsv41-clean `1b1ad396b` / vllm `8b24cc594e`：新基线没动 `ops/`、`quantization/`、`compilation/`、`ascend_config.py`、vllm 的 fused_moe/breakable_cudagraph，
> 所以本文 vllm-ascend 这些文件的行号仍有效；`model_runner_v1.py` 的行号整体后移（见 §j）。
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
| `overlap_side_stream` | `false` | M3 预留（sglang `KT_SIDE_STREAM` 的对应物）；设 `true` 目前直接报错 |

前置条件（`config.py` `_validate_preconditions`）：TP=DP=PP=1、无 EPLB、无投机解码。M2 起不再要求 `--enforce-eager`。
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
- **返回值按名字绑定**：`route` 返回 `NpuRouting(topk_weights, topk_ids)`（`placement.py` `NpuRouting`），
  字段顺序与本仓所有 MoE 调用点的命名一致；调用点读字段不按位置解包。入口另有 dtype 契约检查
  （`assert_routing_dtypes`）：ids 必须 int32/int64、weights 必须浮点、两者同形，不满足当场 raise，不用 `.to()` 纠正。
  2026-09-20 上卡 t1 的实证：位置解包把两者调换后，唯一的报错来自 `npu_moe_init_routing_v2`
  （`aclnnMoeInitRoutingV3 ... EZ1001: expertIdx not implemented for DT_FLOAT`），既不点名调用点也不点名参数。
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

## f. M2 图模式（`layer.py` 的 captured 路径）

每次 forward 按"当前流是否正在被捕获"二选一（`layer.py:314`，判据 `_capturing()` = `torch.npu.is_current_stream_capturing() is True`，
与 engram host gather 同一判据；照抄而非 import，因为 import `models.deepseek_v4_1` 会触发 V4.1 平台补丁）：

```
捕获中  route: copy_inputs_to_cpu_buffers(h, ids, w)      (D2H 进 kt pinned 缓冲，被捕获成 memcpy 节点)
             _launch_host_func(捕获流, _kt_host_forward)   (host-func 节点：每次 replay 在主机上跑 run_pinned_forward_sync)
       NPU: split → dispatch → GMM(驻留) → combine → finalize (普通捕获)
       merge: copy_forward_output_to_device(h)             (pinned → output_gpu 的 H2D，流序在 host-func 之后)
             routed_out = (routed_out.f32 + cpu_out.f32).bf16
非捕获  M1 的 eager 路径（prefill、未捕获的 bs、warmup）
```

- **捕获分支不受任何运行时条件控制**（`layer.py:314` 在 profile 判断之前）：捕获那一次跳过的东西，每次 replay 都没有
  （memory「运行时条件为闸的块不会进图」）。捕获用的 `_dummy_run` 传 `is_profile=False`（`model_runner_v1.py:3960` `in_profile_run=is_profile`；捕获调用在 vllm `gpu_model_runner.py:6937`，不传 is_profile ⇒ 默认 False），
  本来也不会随机路由；这条是防御，单测有 `in_profile_run=True` 的捕获臂。
- **订阅捕获流、只吞 107011**：`_subscribe_stream`（`layer.py:408`），每条流订阅一次；torch_npu ≥ 2.10 捕获时已预订阅，
  engram 也可能订阅过，重复订阅报 107011 = 已订阅。
- **`KT_EXTERNAL_NPU_REPORT_SUBSCRIBER=1`**：图模式下构造 wrapper 前设置（`layer.py:270`），kt 不起 poller、不自己订阅。
- **捕获前登记全部捕获 bs 并分配缓冲**（`_prepare_capture_buffers`，`layer.py:226`）：在 `process_weights_after_loading` 里，
  `set_capture_batch_sizes(compilation_config.cudagraph_capture_sizes)`，并对每个 bs 调一次 `KExpertsCPUBuffer.get_buffer`
  把 pinned 缓冲和设备 `output_gpu` 分配好（kt 的缓冲是类级、进程共享，只做一次）。捕获到未登记的 bs 直接报错
  （`layer.py:348` 起），不在捕获内分配。捕获 bs 大于 kt `chunked_prefill_size` 也在此拒绝。
  代价：设备 `output_gpu` = 2 × Σbs × 5120 × 2 B，pinned 约 3 倍于此；发生在 KV profile 之前，会被计入。
- **查表预烘**：M1 已在 `process_weights_after_loading` 放到设备上，捕获期没有 H2D 建表。
- **捕获态探针注入 kt**：ktransformers `292be81` 加了 `set_capture_probe(fn)`（默认仍是 sglang 探针）；我们注入 `_capturing`
  （`_install_capture_probe`，`layer.py:440`）。captured 路径本身不调 kt 的 `_wait_device`，这是第二道防线；kt 版本没有这个钩子时只警告。
- **payload 常驻**：`(wrapper, meta 张量, 流句柄, layer_idx)` 追加进 `_CAPTURED_PAYLOADS`，活到进程结束。kt 在 replay 时只读输入张量的
  形状，所以传 meta 张量，不拖住图内存池的内存。
- **host-func 异常不外抛**（ACL 侧没人接）：记 error 日志，说明本步 CPU 输出是旧值；`HOST_FUNC_CALLS[layer]` 只在成功时 +1，
  供上卡核对"replay 真的跑了节点"。
- **FULL 与 PIECEWISE**：两者都是流捕获，MoE 在两种描述符下都在图内（PIECEWISE 下 MoE 在 `moe_forward` 自定义算子里，dynamo 不追踪，
  我们的 Python 在捕获时执行一次），captured 路径与描述符无关，不需要分支。
- **捕获后要不要多 replay 一次**：不需要我们额外做。捕获那一次 host-func 节点**只被记录不执行**，捕获那一次的输出里 CPU 部分是
  pinned 缓冲的旧值；但捕获用的是 dummy run，输出被丢弃。
  - 已验证配置用的 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`：vLLM 核心捕获后已 replay 一次（vllm `compilation/breakable_cudagraph.py:398-404`，为 engram 加的）。
  - 默认的 `ACLGraphWrapper`（`vllm_ascend/compilation/acl_graph.py:164-240`）不 replay，同样只影响丢弃的 dummy 输出。
  - ⚫ 上卡核：第一个真实请求的第一步 CPU 输出不是旧值（§M2 验证 3）。
- **串行代价**：host-func 节点阻塞所在流直到 CPU 专家算完，排在它后面的驻留专家 GMM 要等（与 sglang 不开 `KT_SIDE_STREAM` 时相同）。
  M3 用 `overlap_side_stream`：fork 到侧流跑 D2H + host-func，merge 前 join（sglang `kt_ep_wrapper.py:652-679` 的做法），
  event 活到进程结束。M2 只留开关。
- 备选（M3 可比）：breakable 图的 `eager_break_during_capture` 能把整个 CPU 部分做成 eager 段，replay 时直接跑 Python 的 eager 路径；
  代价是每层两次断图。

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
- 统计：路由计数钩子（`vllm_ascend/kt_offload/expert_counter.py`，开关 `VLLM_ASCEND_DSV41_EXPERT_COUNT`，用法见 `tools/kt_offload/README.md`）
  只在 `enable_force_load_balance=False` 时累加，被排除的步数单独计；图模式下拒绝启用。

## i. 显存与主机内存账（integration 写起飞闸用）

所有系数由形状推导（`placement.py:362 resident_bytes_per_slot`），代码里没有写死的 GiB 系数。

| 项 | 公式 | DSV4.1 数值 |
|---|---|---|
| 每专家每层（卡上/盘上） | w1+w3: I8`[2×2304, 2560]` + E8M0`[2×2304, 160]`；w2: I8`[5120, 1152]` + E8M0`[5120, 72]` | **18,800,640 B**（17.93 MiB） |
| 每专家跨 40 层 | ×40（**不含 MTP**；275.7 GiB 总数含 MTP 6.72 GiB，别用 0.718） | **0.7004 GiB** |
| 卡上 | 其余权重 10.4 + 0.7004 × N | N=100 ⇒ ≈ 80.4 GiB（⚫ NZ `format_cast` 是否有对齐填充，待实测校准） |
| 主机常驻（kt BufferB） | compact scale 下与盘上同大：(384 − N) × 0.7004；关 compact（`KT_MXFP4_COMPACT_SCALES=0`）为 (384 − N) × 0.824 | N=100 ⇒ ≈ 198.9 GiB（关 compact ≈ 234.0） |
| 每 NUMA 节点 | threadpool=2 时约一半，`hwloc` 严格 BIND，本节点不够直接 OOM 不溢出 | N=100 ⇒ ≈ 99.5 GiB/节点 |
| 加载瞬时 | ≈ 单层子集 × 3（推断：safetensors 副本 + staging + BufferB） | N=100 ⇒ 284 × 17.93 MiB × 3 ≈ 14.9 GiB |
| 图模式 kt 缓冲 | 设备 2 × Σ捕获bs × 5120 × 2 B；pinned 约 3 倍 | 依捕获 bs 列表 |

- `N = floor((HBM 可用 − 10.4 − KV/工作区) / 0.7004)`；按计划 §1 的估算 (107 − 10.4 − 25) / 0.7004 ≈ 102。
  实测做法：先 N=100 起，读 vLLM 日志的 model weights 占用和 KV 可用量再调整。
- 起服务参数：`--tensor-parallel-size 1 --data-parallel-size 1`（不加 `--enable-expert-parallel`）、M1 用 `--enforce-eager`（M2 起可不加，需 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 与已验证配置一致）、
  `--max-model-len` 小值起、`--max-num-batched-tokens` ≤ kt `chunked_prefill_size`、`--kv-cache-dtype bfloat16 --block-size 128`、
  `--additional-config '{"kt_offload_config": {...}}'`；环境：`unset LD_PRELOAD`（jemalloc），PYTHONPATH 加 kt wheel
  目录 `build_artifacts/moe_offload_ref/kt_site`。
- **`serve_v41_bringup.sh` 需要协调（本线不改）**：
  - `:127 MIN_EP=3` 与 `:151` 的健康卡数检查、`:204` 的 `DP<3` 拒绝：offload 时每卡权重不是 `275.7/DP + 10.4`，
    而是 `10.4 + 0.7004×N` GiB。建议加一个 `KT_OFFLOAD=1` 分支：`MIN_EP=1`，`:204` 改为按 `10.4 + 0.7004×N` 判。
  - `:447-449` 固定传 `--data-parallel-size "$DP" --enable-expert-parallel`：offload 分支去掉 `--enable-expert-parallel`。
  - `:322 verify_shm_sharing.py --ranks "$DP"` 对 DP=1 应能直接用（待确认）。
  - `HCCL_BUFFSIZE`（MC2 用）在 ALLGATHER/单卡下无意义，保留无害。
  - 这些改动经 flash_dev（计划 §5）。

## j. 与新多卡基线（dsv41-clean `1b1ad396b`）的关系（2026-09-19 变基后核对）

**N3 候选掩码守卫 ⇒ 单卡 offload 的 max-model-len 上限 = 16384。**
- 公式（`vllm_ascend/models/deepseek_v4_1/indexer_select.py:35-56`）：放行当且仅当
  `ceil(floor(L / max(r,1)) / candidate_block_size) ≤ candidate_topk_blocks`。等价于 `L ≤ topk_blocks × block_size × r + (r − 1)`。
- 受检对象：只有读候选掩码的 indexer（`candidate_blocks` 非 None：候选源层 20 及其后的 indexer，vllm `attention.py:398-419`；
  `collect_candidate_guards` 只收 `candidate_blocks.shape[1] > 0` 的项生效）。V4.1-Flash `config.json`：`candidate_source_layer_id=20`、
  `candidate_topk_blocks=2048`、`candidate_block_size=8`、`compress_ratios[20..39]=1`、`index_source_layer_ids=[2,8,14,20,24,28,32,36]`
  ⇒ 受检的是 20/24/28/32/36，r=1 ⇒ `L ≤ 2048 × 8 × 1 + 0 = 16384`。16385 ⇒ 2049 块 > 2048 ⇒ 第 20 层先报（与多卡实测一致）。
- 检查时刻：模型构建后按 `max_model_len` 查一次，之后每步在 host 上按本步最长序列查（`model_runner_v1.py` `_v41_candidate_guards`）。
  两处都与 TP/DP/offload 无关 ⇒ 单卡同一上限。**实际能开多大还受单卡 KV 余量约束**（N=100 时卡上权重 ≈80 GiB，KV 余量远小于四卡），
  那是另一条更紧的线，上卡读 vLLM 的 KV 可用量再定；16384 只是守卫给的硬顶。
- `tools/kt_offload/serve_v41_singlecard.sh` 的 `MAXLEN` 默认 8192 ≤ 16384，不冲突。

**第 3 批（engram 移出图，`VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH`，默认 0，4 卡上崩过）与本线无关：**
本线 M2 不把任何东西挪出图 —— kt 的 CPU 专家是**图内**的 host-func 节点（D2H → host-func → H2D 都被捕获，§f），不依赖该开关，
也不依赖 engram 先订阅捕获流（`_subscribe_stream` 自己订阅、只吞 107011）。开关=1 时 engram 那条 host-func 节点消失，kt 的仍在；
开关=0（默认）两者并存。本线不打开第 3/4 批开关。第 2 批（mHC pre、逆 RoPE 走算子）默认开，与 MoE 无交集，照默认。

**没有挂任何 vLLM KV / offload connector**（官方 review：connector 下 slot cache 每请求前 3 位出错）：kt offload 是 MoE 层内的
host 计算，不经 vLLM 的 KV connector / `kv_transfer_config`。核对：`command grep -rni "connector|kv_transfer|kv-transfer|kv_offload"`
于 `vllm_ascend/kt_offload/`、`tools/kt_offload/`、`routed_experts.py` 零命中（同路径 grep `kt_offload_config` 有命中作阳性对照）；
单卡脚本不传 `--kv-transfer-config`，`env_dsv41.sh` 也没有。⚠️ 唯一入口是调用方自己在 `EXTRA_ARGS` 里加 —— 不要加。

**新基线里本线依赖的对象（逐项核过未变）**：`routed_experts.py` 的 `_select_experts` / `forward_impl` /
`enable_force_load_balance = _EXTRA_CTX.in_profile_run`；`token_dispatcher.py` ALLGATHER；`w4a8_mxfp4.py` DS method；
vllm `breakable_cudagraph.py` 捕获后 replay 一次；engram 的 host-func 订阅与 107011 处理（新基线只在前面加了 out-of-graph 分支，订阅代码未改，
M2 照抄的部分无需同步）。新增、与本线相关的只有：模型构建前的 `validate_if_v41` 拓扑校验与上面的 N3 守卫（均模型级，offload 不影响）。

## k. 侧流重叠（M3，`overlap_side_stream`，默认关）

**靶子**（本线实测，产物 `build_artifacts/moe_offload_ref/prof_0920_045212/op_breakdown_bs1.txt`）：图模式 bs=1 TPOT
40.47±0.38 ms；device 真实忙（区间并集）57.1% = 24.96 ms/token，空隙 18.73 ms，其中 >100 µs 的大空隙每 token 43 段
合计 17.91 ms ⇒ host offload 完全暴露。kt 纯计算约 10.4 ms（CPU 曲线 T(k)=66.8+43.95k µs/层），余 ≈7.5 ms 是搬运+回调。

实现（`vllm_ascend/kt_offload/layer.py`）：

```
捕获中  route: fork.record(compute); side.wait_event(fork)
             with stream(side):  [捕获传播检查] → copy_inputs_to_cpu_buffers → _launch_host_func
             记下 (join, side, compute)，(fork, join) 进 _SIDE_EVENTS（活到进程结束）
       compute: split → dispatch → GMM(驻留) → combine
       merge: join.record(side); compute.wait_event(join); copy_forward_output_to_device
eager  route: fork.record(compute); side.wait_event(fork); with stream(side): copy_inputs…
             side.synchronize()（只等这条流）→ forward_on_pinned_buffers（提交，不阻塞）
       merge: cpu_infer.sync() → copy_forward_output_to_device
```

- eager 这条路的收益来自**绕开 kt 自己的 `torch.npu.synchronize()`**：kt 的 `submit_forward` 为了确保输入 D2H 落地会做
  整卡同步（`experts_base.py` `_wait_device`），那会连驻留专家一起等，重叠归零。改成只 `side.synchronize()`。
  代价是要用 kt 的 `forward_on_pinned_buffers` + `cpu_infer.sync`，所以 `overlap_side_stream` 要求 `force_sync_submit`
  （否则 kt 自己还会再提交一次），且启用时会检查这几个方法在不在（`_overlap_supported`，`layer.py` 约 :233）。
- **捕获传播检查**（`layer.py:451` 起）：在侧流上下文里再问一次 `_capturing()`；为假就 raise。
  sglang 的 F1 失效正是"捕获没传播到侧流"——侧流上的 D2H 与 host-func **只在捕获那一次执行**、不进图，
  于是每次 replay 都复用捕获那一次的 CPU 输出：**更快而且算错**。只看 TPOT 的判据对它是零覆盖。
- **event 保活**：每对 (fork, join) 进 `_SIDE_EVENTS`，捕获图里的 record/wait 节点引用它们（sglang 丢过）。
- **陈旧 join 清理**：每次 `_submit_captured_forked` 先把上一轮的 `_pending_join` 清掉（sglang `18c8b21e3`）；
  `merge` 在 overlap 打开却没有 join 时直接 raise（不静默跳过）。
- CPU 单测 `tests/ut/kt_offload/test_side_stream_overlap.py`：除调用序列外，有一个**按完成时间调度的流模拟器**，
  把录下来的调度真的跑一遍并与串行臂**逐位**比较。四个变异臂（join 去掉 / join 记早 / event 未保活 / 捕获未传播）
  每一个都**更快且算错**，测试同时断言"更快"与"逐位不同"，把"只看时间抓不到"写成可执行的事实。

## l. 搬运与回调那 7.5 ms 的归因（测量项，**不与侧流同一个变更**）

字节账（`tools/kt_offload/transport_budget.py budget`，纯算术）：每层每 token D2H 10,312 B（hidden bf16 + ids int64 +
weights fp32）+ H2D 10,240 B = 20,552 B；40 层 = 822 KB/token ⇒ 20 GB/s 下 **41 µs/token**。
⇒ 相对 7.5 ms，**带宽只占 0.5%**（推断，链路带宽未实测）。~~同一脚本按 120 个操作反推：每个操作 62.5 µs 固定开销。~~
🔴 **撤回**（k.5d，2026-09-20 profile 实测）：memcpy 的 device 时间是 1.168 µs/次，62.5 与 37.5 µs 都只是
"某个 7.5 ms ÷ 某个操作数"，不能当作每操作成本，也不能反过来仲裁操作数。kt 侧 CPU 曲线的每层固定项另算。
⇒ 待验证的假设：这 7.5 ms 是**每操作固定成本 × 操作数**，不是字节数；能减的是操作数（例如 ids/weights 与 hidden 合并成一次拷贝、
把 H2D 并进下一层的 D2H）或回调路径本身，不是加带宽。
上卡确认办法：`transport_budget.py measure <kernel_details.csv> --steps N`，它按**区间并集**算真实忙、单列 D2H/H2D/host-func
的次数与时长、并给 >100 µs 空隙的个数与合计。**这项单独提，别和侧流绑在一个变更里**：重叠是把成本藏起来，不是去掉。

## k.2 2026-09-20 A/B 失败：已排除的、已改的、仍未定位的

**结果**（产物 `build_artifacts/moe_offload_ref/m3_ab_0920_061531/`）：B 臂（overlap=on）GSM8K 50 题 34/50，
38/50 文本与 A 不同，16 题跑到 max_tokens；TPOT 39.81→36.08 ms。**判据①不过 ⇒ 这一版不能用。**

### 已用数据排除的
- **不是"机制没发生"**：device 忙 57.8→79.9%，每 token 空隙 18.29→6.72 ms，>100 µs 大空隙 43.1→15.0 段。
- **不是两个根因**（问题①）：38 个分歧里 22 个 finish_reason=stop、16 个 length；两类的**首次分歧字符位置分布连续且重叠**
  （stop 6…397 中位 105，length 3…589 中位 192），未见分层；12 个逐位相同的题在提示长度上也不成一类。
  ⇒ 最简读法是**一个间歇性污染**，下游表现分成"还能收敛"和"永不吐 EOS"两种。判据：修好后两类必须同时消失。
- **GroupedMatmul +76% 不是取样口径问题**（问题②的一半）：两臂**每次调用**的耗时对照——SparseFlashMla 21.2→21.5 µs、
  MoeInitRoutingV3 13.2→13.3、QuantBatchMatmulV3 20.4→20.5、GMM 的 Block Num 都是 28、kernel 名一样，
  只有 GroupedMatmul 23.0→40.3 µs。⇒ batch 组成相同、不是降频（QBMM 会一起动）。
  但 **B 臂算的是错的，两臂在算不同的东西，此刻无法归因**。判据：修好后若逐位相同而 GMM 仍 +76%，才是真的重叠代价，另立一项查。
- **`+0.05 GiB` 捕获显存不是"多出的 event 节点"**（问题②的另一半，此前是未经验证的吻合，现撤回）：
  事件对象几十字节量级；我们在捕获里唯一新增的张量是 kt 的 ids cast，按捕获尺寸 [1,2,4,8]×40 层只有 ~29 KB。
  两者都解释不了 52 MB。剩下的候选是"侧流在图内存池里自己占了 segment"（未验证）。
  下次窗口的判据：两臂各 `torch.npu.memory_stats()` 捕获前后差 + segment 数；本次改动去掉了 ids cast，
  **预测这 52 MB 不会因此变化**（若变了，说明候选错了）。

### 已排除的假设（我原本的首要怀疑）
"eager 分叉去掉了 kt 的整卡同步 ⇒ 主机抢先覆盖了两层前的 pinned 输出槽"——**用模拟器测了，不成立**：
侧流的第一条指令是等 fork event，而 fork 记录在 compute 上、位置在上一层 H2D 之后，
所以 `side.synchronize()` **传递地**等到了那次 H2D。（也因此 eager 分叉根本不省时间。）

### 本轮改了三处（每处都独立成立，但**都未被证明是这次的根因**）
1. **eager 不再分叉**（与参考实现一致）：eager 走回 kt 自己的 submit/sync。理由见上（没收益），
   且它把 kt 的槽位复用/temp buffer 重分配移出了那次整卡同步的保护。
2. **陈旧 join 每次 route 都清**（不只捕获分支），对齐 sglang `18c8b21e3`。
3. **捕获期侧流上下文里不再有任何分配**：kt 的 `topk_ids.to(torch.long)` 会在"当前流"上分配，而侧流不是捕获流，
   那块内存是否来自图的私有池不该由我们假设。改为提前（捕获前）分配 int64 ids 缓冲，kt 拿到的已经是 int64，
   `.to(long)` 变成恒等 ⇒ 侧流上下文里零分配。

### 仍未定位，下一步不要再跑整网
剩下的怀疑集中在**平台是否真的把捕获传播到侧流、以及分段捕获（breakable，每层一次 capture_begin/end）下这件事是否每段都成立**。
⇒ 先跑 **T3（`tools/kt_offload/oncard/t3_side_stream.py`，单卡几分钟）**：3 层（L、L+1、L+2 共享 kt 的 `layer_idx%2` 槽）、
两个捕获尺寸交替 replay、每次 replay 逐层核 `HOST_FUNC_CALLS +1`、输出与串行臂**逐位**比。
它能把"某些层的侧流工作没进图"和"进了图但算错"分开，而整网 A/B 分不开。T3 绿了再排整网 A/B。

## k.3 2026-09-20 定位：护栏读的是代理；侧流捕获这条路在本平台不通

### ① 护栏为什么没响（定位到行）
护栏在 `vllm_ascend/kt_offload/layer.py` 的 `_submit_captured_forked` 里：侧流上下文内 `if not _capturing(): raise`，
而 `_capturing()`（同文件末尾）读的是 `torch.npu.is_current_stream_capturing()`。
**它读的是代理**：「当前有捕获在进行」≠「这条流上的工作会进图」。
T3 的产物就是证据：**护栏放行了**，随后 ACL 在 `capture_end` 拒绝（107025）。服务那一发同样放行、且没有任何东西拒绝。
⇒ 护栏从来没有判断过它声称判断的那件事。

**修法（已实现）**：`probe_side_stream_capture()` —— 启用侧流且图模式时，在 `after_weights_loaded` 里
**捕获一个微型的同形图**（fork → 侧流上一次 D2H + 一个 host-func 节点 → join → H2D），然后**用两个不同输入各重放一次**，
要求 (a) 捕获不抛错、(b) host-func 节点每次重放各跑一次、(c) 图搬回来的值随输入变化。
三条都读**被判断的那个对象**（这张图到底会不会重放侧流上的工作）。两次真实失效都会被它拦下：
T3 那种在 (a) 抛错，服务那种在 (b)/(c) 不成立。不通就**拒绝启动**并说清换回 `overlap_side_stream=false`。

**与原有变异臂的差别**（协调方要求说明）：原来的 `capture not propagated` 变异是**事后手改调度**（把 side 那一列删掉），
测的是判据能不能看出结果不对；真实失效是**平台自己说"在捕获"却不记录侧流节点**。新测试用一个这样的假平台
（`sim.capture_records_side=False`），并在同一个测试里断言**旧的代理谓词仍然返回 True** —— 非等价性因此是可执行的。

### ② 为什么 T3 报错、服务静默
- 不是上层吞错误码：`vllm/compilation/breakable_cudagraph.py` 的 `_end_segment` 没有 try/except；
  `torch.cuda.CUDAGraph` 在 `vllm_ascend/worker/model_runner_v1.py:6056` 被换成 `torch.npu.NPUGraph`，
  也就是同一个 C++ `capture_end`；那一段 shim 里唯一的 try/except 包的是 **runner 初始化**，不是捕获。
- 剩下的差异是**捕获的形状**：T3 把 3 层放进**一个**捕获（3 对 fork/join）；服务是 breakable，
  **每个注意力算子断一次图**（`vllm_ascend/ops/mla.py:207` / `ops/dsa.py:171` 上的 `@eager_break_during_capture`，
  `vllm/compilation/breakable_cudagraph.py:59` 起），所以每段里只有一层、只有一对 fork/join。
  ⇒ 待验假设：107025 与「一次捕获里多对 fork/join」有关；一对就不报错，但侧流节点没有被有效重放。
  判据（下次窗口，几十秒）：`run_oncard.sh <card> t3 -- --layers 1` 与 `--graphs per-layer`。
  现在启动探针会先拦，所以这两发要在探针失败的报文里读结论（探针本身就是同形最小用例）。

### ③ 在现有产物上验「复用旧输出」指纹（不靠推断）
从两臂的 `kernel_details.csv` 自己算（步数用 `MoeInitRoutingV3`/40）：

| | A（侧流关） | B（侧流开） |
|---|---|---|
| 步数 / 步时 | 181 / 44.32 ms | 236 / 33.94 ms |
| device 忙（并集）/步 | 25.6 ms | 27.1 ms |
| >100 µs 空隙/步 | **44.1 段 × 406 µs = 17.90 ms** | **15.3 段 × 379 µs = 5.80 ms** |
| 每步空隙段数分布 | 144/181 步正好 43 段（均匀） | **双峰**：约 210 步是 6–14 段、约 26 步是 42–51 段 |

判据与读数：
- 「重叠在所有层生效」预测 步时 ≈ 44.32 − 17.90 = **26.4 ms**、大空隙 ≈ **0** ⇒ **与观测不符**（33.94 ms / 5.80 ms）⇒ **排除**。
- 「CPU 工作在 N 层上没有发生、其余不变」解出 **N ≈ 29.8（按空隙）/ 25.5（按步时）**，两条独立读数一致。
- 且**存活的空隙尺寸和 A 一样**（379 vs 406 µs，中位 279 vs 348）⇒ 不是「被盖住变短」，是「整段不见了」。
- 每步段数**双峰**：约 26 步仍是 43 段（= eager 步，prefill/未捕获尺寸，与 A 同形），其余步只有 ~10 段。
  ⇒ **缺陷只在捕获路径**，eager 路径那一发是好的（这也说明我上一轮把 eager 分叉当首要嫌疑是错的方向）。
⇒ 指纹成立：**重放时大部分层的 CPU 工作没有执行，那些层用的是捕获那一刻的 pinned 输出。**

### ④ 段数复核（协调方要求单独报）
`@eager_break_during_capture` 的断点判据**只看有没有 breakable 捕获在进行**，不看 FULL/PIECEWISE
（`vllm/compilation/breakable_cudagraph.py:94-115`）。DSV4.1 每层一次 `dsa_forward`/`mla_forward` 都带这个装饰器
⇒ **`VLLM_USE_BREAKABLE_CUDAGRAPH=1` 下，即使 decode 走 FULL，注意力也是 eager 段，每个捕获描述符 ≈41 段**（40 断点）。
而 `VLLM_USE_BREAKABLE_CUDAGRAPH=0`（t2 那一发用的 `ACLGraphWrapper`）没有断点机制，注意力**在图内**，每个描述符 1 张图。
⇒ 两种说法各自成立于各自的开关状态；本次 A/B 用的是 `=1`。「分段捕获是最大结构差异」这条**仍然是推测**，
现在它的地位是 ② 里的待验假设，不是既成事实。

## k.4 2026-09-20 收敛：平台结论只到这一句 + 回扫被推翻的假设

### 平台结论（收窄，别外推）
**成立**：把工作 **fork 到另一条流**再 join 回去、并要求它进捕获，在这套 torch_npu 上 `capture_end` 报 **ACL 107025**。
证据三发独立：3 层一次捕获、每层一次捕获（`--graphs per-layer`）、以及整网 A/B 那一发的捕获路径。
**不成立（不要这么写）**：「host 工作不能进图」。反例都在生产路径上：
- engram 的 host gather 就是图内节点（`vllm_ascend/models/deepseek_v4_1/engram_embedding.py` 的
  `torch_npu.npu._launch_host_func(stream, _engram_host_gather, payload)`，`ENGRAM_OUT_OF_GRAPH` 默认留在图里），
  过了 10 个数据集精度 + 8.86 h 浸泡；
- 我们自己的 A 臂（overlap=off）就是在**捕获流**上跑 host-func，50/50 全对；t2 也验过每 replay 执行一次、值随输入变。
⇒ 三种机制别混：`graph_task_group` 改**设备任务参数**；`ExternalEvent` 管**跨流排序**；**host 工作进图用 `_launch_host_func`**。

### 护栏：探针已删，开关改为直接拒绝
`probe_side_stream_capture()`（上一轮写的启动探针）**是假绿灯**：它捕获同样的 fork/join 模式，但只搬一个 int32 标量、
host 侧只 `buf.add_(1)` —— 在真实层捕获全部失败的平台上它**通过**。⇒ 替身比被测对象弱，判据就会在该拦时放行
（与「读代理」同族的第二种形态）。已删除；`overlap_side_stream=true` 现在在 `init_ascend_config` 里**直接拒绝**。
若将来还要探针，元判据是：**它必须能复现已知的两种真实失效**（107025 与静默不重放），否则不算护栏。
（测试用的对象仍可直接构造 config 绕过拒绝，供上卡二分。）

### 🔴 回扫：基于「多对 fork/join 才失败」的结论一律作废
| 上一轮写的 | 现状 |
|---|---|
| 「107025 与一次捕获里多对 fork/join 有关；一对就不报错」 | **作废**（`--graphs per-layer` 一对也报 107025） |
| 「服务每段一对，所以捕获成功了、只是没重放」 | **作废**（它的前提就是上一条） |
| 「分段捕获是最大结构差异」 | **降级为无关**：段数不是原因；仍待解释的是「服务为什么没报错」 |
| 「T3 `--graphs per-layer` 是判别多对/一对的判据」 | 已执行并**否证**了那个假设；开关保留，语义改为「同形于服务的段结构」 |
| 「eager 路径在 B 臂是好的」 | **仍成立**，但理由换了：不是「段结构」，而是侧流第一条指令等的 fork event 记录在上一层 H2D 之后（模拟器验证），所以 eager 分叉根本不省时间、也没造成错 |
| 「指纹 = 重放没跑侧流工作」 | **仍成立**（§k.3 的空隙算术与双峰直方图不依赖 pair 假设） |
| 「N ≈ 25–30 层丢了 CPU 工作」 | **降为两读之一**：另一读是「捕获步里 CPU 工作全丢（40 层），剩下的 ~10 段大空隙是 breakable 在 replay 期 eager 跑注意力的开销」。两者都与现有产物一致 ⇒ 需要服务内**逐层 HOST_FUNC_CALLS** 才能分开（下次窗口几秒钟） |

**服务为什么没报 107025（新的首要假设，未验证）**：捕获期 `_capturing()` 返回 False ⇒ route 走了**eager 分支**
（当时那是"eager 分叉"版），于是 CPU 工作在捕获那一刻**在图外执行一次**、没有任何节点进图、也没有跨流捕获 ⇒
既不报错又每次 replay 复用那一次的 pinned 输出。判据：服务内打印每层每步的 `HOST_FUNC_CALLS`（图内节点才会 +1）。

## k.5 减操作数（`packed_transfer`，默认关；不占卡，已实现）

靶子：搬运+回调这一块（原记 7.5 ms/token）。🔴 **单价模型已作废（k.5d）**：不要再写 "62.5 µs × 120 个操作"，
也不要再引用由它派生的 "packed 省 3.0/5.0 ms"。packed 的收益只认 **A/B 两臂 TPOT 差**（已测，见 k.8）。

kt 的输入放在**三块独立 pinned 内存**里（`input_tensor_cpu` / `immediate_experts_ids_cpu` / `weights_cpu`），
所以一层要付**三次** host-device 拷贝。`packed_transfer=true` 改成：
- 一块合并缓冲（`placement.packed_layout`：hidden bf16 / ids int64 / weights fp32 三段，各段 512 字节对齐），
  **设备侧三次填充**（普通设备工作）+ **一次** host-device 拷贝；
- 输出也用我们自己的 pinned/设备对，**每层各一份** ⇒ 顺带去掉 kt 的 `layer_idx % 2` 槽位共享；
- CPU 任务用 kt 的 `moe.forward_task(...)` + `cpu_infer.submit/sync` 直接读我们的指针（kt 版本没有这两个入口就**拒绝启用**）；
- eager 路径不再用 kt 的 `submit_forward`/`sync_forward`（各含一次 `torch.npu.synchronize(device)`，等全设备、每层两次），
  改成一次 `event.record + event.synchronize()`（只等本流已入队的工作）。
操作数：每层 5 → **3**；40 层 200 → **120**（`placement.host_ops_per_layer`，被单测钉住）。
判据（与上卡相同）：**开关两态输出逐位相同** + 操作数下降。收益上卡再量，这里不断言速度。
变异臂（全红）：三段各拷一次（把节省撤掉）/ 复用 `layer%2` 的共享缓冲 / 漏掉输出回拷。
⚫ 待 kt 侧确认：`moe.forward_task` 的参数顺序与 `cpu_infer.sync(allow_pending)` 语义（本线是按 `experts_base.py` 现有调用抄的）。

## k.5b 每操作成本 C 要先实测（形态 C 的前置；协调方 2026-09-20 决定）

> 🔴 **这一节的单价模型已被 k.5d 的 profile 实测否掉**（memcpy 的 device 时间是 1.168 µs/次，不是 62.5 µs）。
> 下面保留原文只为记录当时的理由；三分支判定里真正落地的是第三条：模型错了，归因要重做。

**不在推断值上决定**。形态 C 的净收益 = 本层 GMM 窗口（0.62 ms，**实测**）− 多出一次回调（≈62.5 µs，**推断**：
7.5 ms ÷ 120 个操作的平均值，既没分开 D2H/H2D/host-func，也没验证"固定成本 × 操作数"这个模型本身）。
校准**不需要额外开销**：k.5 的 packed A/B（每层 5→3 个操作）上卡时同时反解 C：

```
模型："每 token 的搬运+回调 ≈ 操作数 × C"；packed 每层省 2 个操作 ⇒ 每 token 省 40×2×C
实测 C = Δ(>100µs 空隙 ms/token) / 80        （两臂同构建、同仪器、只切 packed_transfer）
```
三分支判定（**第三条最重要，别漏**）：
- C ≈ 62 µs ⇒ 形态 C 划算，实现；
- C 远大于 62 µs（例如 ≥300 µs）⇒ 形态 C 多出的那一次回调吃掉 GMM 窗口 ⇒ **方案作废**，继续减操作数；
- **Δ 与 80×C 的模型对不上（例如 Δ 远小于 80×62 µs，或随 bs 不成比例）⇒ "固定成本×操作数"模型错了**，
  那 7.5 ms 的归因要重做（分开量 D2H/H2D/host-func 各自的成本，而不是用一个平均值）。
⇒ **形态 C 暂停实现**，等这组读数。

## k.5c 服务内逐层 host-func 计数（几秒钟，同时结案"服务为什么没报 107025"）

开关：`VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=<秒>`（默认空=关，登记在 `vllm_ascend/envs.py`），
实现 `vllm_ascend/kt_offload/hostfunc_log.py`，启动时（`after_weights_loaded`）拉起一个**守护线程**按间隔采样
`layer.HOST_FUNC_CALLS` 并打印每层增量。**不需要上卡临时改代码。**
- 判据：decode 稳态下**每步每层 +1** ⇒ 均匀；某些层**恒为 0** ⇒ 那些层的 host 工作根本没进图（图 replay 不跑 Python）。
  这正好把 0920 那两种都与产物一致的读法分开（"25–30 层丢了" vs "捕获步 40 层全丢、剩下的大空隙是 breakable 在 replay 期 eager 跑注意力"）。
- 报告里"缺的层"按**层号列出**，缺席按 0 计（缺键与 0 在日志里长得一样，只有一个是结论）。
- 守护线程 + `atexit` 停止：孤儿采样器会对着别人的服务采数，而输出看起来完全正常。

## k.6 形态 C 方案（**已审阅通过，但暂停实现，等 k.5b 的 C**；2026-09-20 按 kt 回复重写）

> 🔴 **前置约束（A6，k.11）**：形态 C **必须建在 packed 的每层自有缓冲之上**。用 kt 的缓冲时第 L 与 L+2 层共用同一组 pinned 张量（`layer_idx % 2`），去掉整卡 `_wait_device` 之后L 层的输出 H2D 还在飞、L+2 层的 CPU 任务已经在改同一块内存；今天不炸只是被那次整卡同步顺带救了。

**形态**：在**正被捕获的那条流**上放**两个** `_launch_host_func` 节点，不碰侧流、不用 ExternalEvent、不用 task group。
这是 engram 已交付的形态（host 工作进图是支持的；不支持的只有"fork 到另一条流再 join 进捕获"）。

```
捕获流（每层，全部在图内）：
  合并 D2H（packed_transfer 的一次拷贝）
  节点①  moe.forward_task(...) + cpu_infer.submit(task)   ← 只提交，立即返回
  dispatch → GMM(驻留专家) → combine                        ← 与 CPU 并行的**唯一**窗口
  节点②  wrapper.sync_cpu_only()                           ← 只等 CPU 队列，不碰设备（kt 033ea76）
  H2D（我们自己的 pinned 输出）→ 与 routed_out 相加
```

### 被 kt 的两条硬事实**排除**的东西（明确不做）
- ❌ **跨层提前下发/流水**：kt 的 CPU 队列是**单 worker 串行 FIFO**（任务内部才用 WorkerPool 并行），
  第 L+1 层的任务不可能先于第 L 层完成。⇒ 可重叠窗口**只有本层 GMM 那一段**，设计里不写"跨层流水"。
- ❌ **票号/按 task 等待（kt 约 20 行 C++ + 重编 wheel）**：形态 C 的时序是「本层提交 → GMM → 本层等待」，
  中间**没有别的提交**，所以 `sync_cpu_only()` 的 allowance=0（等空队列）**恰好就是"只等本层"**。
  只有将来真做跨层提前下发才需要票号——那条已进待决，现在不引入。
- ❌ **kt 的 `layer_idx % 2` 深度-2 pinned 槽**：k.5 起我们用**每层各一份自己的缓冲**，所以"提前两层会抢槽"这条对我们不成立。
  ⚠️ 谁要是把 packed_transfer 改回 kt 的缓冲，这条约束就**立刻回来**（写在这里就是为了别再踩）。

### 必须照抄的约束（engram 的三条 + kt 的四条）
1. 缓冲在**构造期**分配（构造是唯一保证早于捕获的时点）；2. **replay 读捕获期记录的地址** ⇒ 每步新分配即被烘死；
3. 容量按 `max_num_batched_tokens` 定死，超了**拒绝**而不是重分配。以上三条 k.5 的 packed 缓冲**已经满足**。
4. `forward_task` 按位置传参、`qlen` 是**指针**；5. 张量活到任务跑完且内容不变；6. `qlen > chunked_prefill_size` 会写越界毁堆
   （kt 自己的 wrapper 会查，我们绕过了它 ⇒ 判据是我们的，已加断言）；7. dtype/contiguous 已在建缓冲时断言。
8. **每层都必须 sync**：kt 的 worker 把任务异常存起来、**只在下一次 sync 时重抛**（`task_queue.cpp:55-63`）⇒
   节点②不能省，也不能"隔几层再等一次"。
9. `incremental=False` = 覆盖写（True 是读回 output 累加，不是我们要的）。
10. `sync_cpu_only()` / `has_pending_deferred()` 在 `KTMoEWrapper.__new__` 返回的**实例**上（工厂类），不是类方法。

### 代价（要在判据里一起看，别只报收益）
节点数从每层 1 个变成 **2 个** ⇒ host-device 侧操作 3 → **4**（k.5 把 5 降到 3，形态 C 退回 4）。
🔴 **这两个数都已作废**（k.5d / §4.10）：每操作 ≈62.5 µs 是推断值，本层窗口 "≈0.62 ms/层" 也不成立——
协调方给的实测是 **GMM 44.7 µs/层**，整层 MoE 的窗口由 t4 第 2 项实测。净收益只能写成
`min(cpu_us, window_us) − node_us`，三项都要实测（t4）。
⚪ 泄漏：`ext_bindings.cpp` 的 `new Args{...}` 全仓无 `delete`，每次 `forward_task` 漏 72–80 B。
我们这条路**不新增**泄漏源（和 kt 自己的路径一样是每层每步一次），但按 29 步/s × 40 层估算约 **85 KB/s ≈ 0.3 GB/小时**，
长跑要盯住；修在上游，不在我们这条路上绕。

### 判据（CPU 先跑，卡上再跑）
- CPU：开关两态**逐位相同**（模拟器 + packed 的假 kt）；每层每 replay 节点① 和 节点② 各执行一次；
  变异臂：去掉节点②（错误被吞 + 读到未完成的输出）、把节点②挪到 H2D 之后、节点①与②之间插入别的提交（票号假设的反例）。
- 卡上：① 逐位相同（含并发）；② 空隙表：大空隙**变短**（被 GMM 盖住），段数不应大幅减少——
  **段数骤减而单段不变 = 工作没跑**，这正是 0920 的指纹，要能分开；③ 两臂各一份算子 profile；④ 无 10701x。

### 依赖与待决
- `kt_site` 已重建到含 `033ea76`（sha256 1b8e9b21…）。两臂必须同构建，不在窗口中间换（协调方保证）。
- **待决（上游）**：`ext_bindings.cpp` 的 8 处 `new Args{...}` 全仓无 `delete` ⇒ 每次 `forward_task` 漏 72–80 B。
  我们自己拼 task 的这条路**不新增泄漏源**：调用频次与 kt 自己的路径相同（每层每步一次），
  我们自己不分配任何 C++ 对象（Python 侧的 task 元组由 GC 回收）。按 29 步/s × 40 层估 ≈85 KB/s ≈ **0.3 GB/小时**，
  长跑要盯；修在上游，别在我们这条路上再漏一份。

## k.5d 🔴 单价模型被 profile 否掉（2026-09-20，armA profile 实测）

两套操作账（120 = 40×3 与 200 = 40×5）对不上，于是按协调方要求**按 profile 的实际次数重算**，不按代码推。
重算工具：`tools/kt_offload/profile_ops.py <task_time.csv> --price-us 62.5 --claimed-total-ms 7.5`
（对 `build_artifacts/moe_offload_ref/packed_ab_0920_094053/armA/prof_out/csv/*/task_time.csv`，181 个
MODEL_EXECUTE = 181 步，40 层）。实测：

| 任务类型 | 次/步 | 次/步/层 | 每次 µs | ms/步 |
|---|---|---|---|---|
| `HOSTFUNC_CALLBACK` | 41.9 | **1.05** | 0.007 | 0.000 |
| `MEMCPY_ASYNC`（所有 owner） | 615.5 | 15.4 | **1.168** | **0.719** |
| `NOTIFY_WAIT` | 41.9 | **1.05** | **371.8** | **15.59** |

结论：
1. **每层 1 次回调**被证实（41.9/步）；每层几次 memcpy 无法从这张表判定——这些类型没有 kernel 名字，
   615.5 是**所有 owner 合计**，只能当我们的上界。⇒ 120 与 200 这两个账，profile 只能确认回调那一项。
2. **62.5 µs 与 37.5 µs 都不是"一个操作的成本"**：memcpy 的 device 时间是 1.168 µs/次、合计 0.72 ms/步，
   比两种单价推出的 7.5 / 12.5 ms 小 10–17 倍。两个数都是"某个 7.5 ms 的总量 ÷ 某个操作数"得来的，
   **不能反过来仲裁操作数**。
3. 真正吃掉时间的是**等待**：`NOTIFY_WAIT` 每层 1 次、每次 371.8 µs、每步 15.59 ms（TPOT 40.36 ms 的 39%）。
   这就是每层等 CPU 专家的时间，不是搬运成本。
⇒ **k.5b 的单价模型（"每 token 搬运+回调 ≈ 操作数 × C"）作废**，连同由它派生的两个数：
   "packed 省 5.0 ms"、"形态 C 多一个节点要花 2.5 ms" —— 两者都不再成立，不要再引用。
   还活着的只有：packed 的收益 = **A/B 两臂 TPOT 差**（实测，B 臂在跑）；节点成本 = **t4 第 1 项**（实测）。
   注意 device 时间不含 host 侧下发/回调调度开销，所以"一个操作到底贵不贵"只能按**对步时的影响**测，
   这正是 t4 与 A/B 的做法。

## k.7 t4：形态 C 值不值得做（只测时间，不验证正确性）

`tools/kt_offload/oncard/t4_overlap_ceiling.py`（判定算术在 `t4_model.py`，CPU 单测
`tests/ut/kt_offload/test_t4_model.py` 20 条）。单层、T=1 decode 形态、≥20 轮**交错**采样，报中位/p10/p90。
- **三个数**：① `node_us`＝一个 host-func 节点占用计算流多久（图：不插节点 / 插 1 个空节点 / 插 2 个空节点
  即形态 C 的摆位 / 插 1 个只做 kt 提交的节点）；② `window_us`＝提交点与等待点之间本层 NPU 的实际时间
  （图里只有 NPU MoE 的那一臂）；③ `ceiling = min(cpu_us, window_us) − node_us`，按层与按步都打印。
- **eager 三臂可比**（图模式下 sglang 形态进不了图，见 k.4）：串行 / sglang 形态（侧流 fork-join）/
  形态 C 形态（同一流，提交…等待）。并额外在图里**试一次** sglang 形态，把报错原文与抛出位置作为结论打印；
  若竟然捕获成功，判为"意外"并要求回头重开 M3 的结论。
- **判据与对照**：阴性对照＝不插节点的图臂必须与 eager 的纯 NPU 臂相当（比值 ±tol，否则测量本身有问题）；
  仪器阳性对照＝故意 500 µs 的胖节点必须被量到（≥0.6×）；1 节点与 2 节点要可加；变异臂＝把胖节点的实测
  成本代进判定模型，结论必须翻成"不值得"。
- **交叉来源**：`--ab-node-us` 传 A/B 反解的每操作成本，两者**对上才算立住**，对不上打印
  `INCONSISTENT` 并明说两个数都不能用（不挑一个）。没传就打印"未交叉验证"。
- **明确不做**：不验证任何数值正确性（形态 C 的正确性靠 t2/t3 那种形状的测试）；eager 的数**不能外推**
  到图模式（捕获/replay 开销、节点成本只在图里存在）。两条都印在输出和 JSON 里。


## k.8 packed A/B 判定：无收益，默认关，长尾当缺陷记录（2026-09-20）

> 🔴 **更正（2026-09-21，我方复跑判读脚本得到）**：`packed_ab_report.py` 在这份产物上给出
> **19/50 逐位相同、31/50 不同**——两臂各 50/50 指的是**答对题数**，不是逐位一致。跑的是贪心
> （`--temperature 0 --top-p 1.0`），分歧发生在生成中途（例：`gsm8k-59` 前 266 字符相同，之后分叉，
> 最终 `\boxed{187}` 仍相同），completion tokens 7269 vs 7151。
> ⇒ **"逐位判据先过"这句不成立**，本节结论只由 TPOT 支撑。**但不能就此判 packed 数值错**：
> harness 用 8 并发（`num_threads: 8`），批组成随发变化，贪心解码在服务层**本来就可能跨发不一致**。
> 要归因必须补一条 **A/A 对照**（同构建、同配置、连跑两发 A 臂）：若 A/A 也差 ~31/50，
> 说明这条服务级判据没有区分力，逐位一致只能在层级用例（t1/t2/t3 那种形状）上判；
> 若 A/A 逐位相同而 A/B 不同，才是 packed 改了数值。⇒ **默认关这个结论不变，理由要加一条"数值未证一致"。**

两臂 GSM8K 各 50/50（**答对题数**；逐位见上方更正）。TPOT 中位 40.24 → 39.50（−0.74 ms），但**均值 40.36 → 41.37**，
p99 四轮 40.6/41.5/42.7/45.7 → 40.0/56.3/71.3/83.3（三轮翻倍），>3 ms 的等待段 6 → 125 个。
⇒ 中位的 0.74 ms 落在跨发漂移（~0.5 ms）量级内，**判为无收益**：`packed_transfer` 默认保持关闭，
不作为优化项继续投入；**长尾按缺陷记录**（嫌疑：每层自有 pinned/设备输出缓冲带来的分配或页行为、
合并缓冲在某些 bs 上的填充退化、与 kt 串行队列的交互）。派生数字见 §k.5d 的作废清单。

## k.9 生产规模 CPU 曲线（重测，2026-09-20；所有 CPU 侧数字都长在这条曲线上）

旧曲线是 3 层 / 14.92 GiB 工作集测的（ANALYSIS 自标"偏乐观、幅度未量"）。本次用
`bash tools/kt_offload/bench/run_cpu_bench.sh <outdir> prod`（40 层 × n_cpu=284 = **198.91 GiB**，
两遍 a/b，起跑前查 MemAvailable ≥150 GB）重测，并在**同一时刻、同一台机器的同一负载下**补了一个
3 层对照（`ctl_3layer_today`），否则"尾巴"分不清是工作集还是同机的其它租户（实测当时有 4 个
VLLM::Worker_DP 各占 ~65% CPU）。

T=1 每层耗时（µs，p50 / p90）：

| k_cpu | 3层(今天) p50 | prod_a p50 | prod_b p50 | 3层 p90 | prod_a p90 | prod_b p90 |
|---|---|---|---|---|---|---|
| 0 | 12.6 | 22.7 | 23.5 | 13.5 | 25.1 | 25.5 |
| 2 | 130.5 | 145.8 | 178.6 | 139.5 | **480.4** | **1691.9** |
| 4 | 216.3 | 246.2 | 274.6 | 228.1 | **519.0** | 319.5 |
| 6 | 316.8 | 354.5 | 382.8 | 328.8 | **741.8** | 700.5 |

1. **中位数**：生产足迹比同负载下的 3 层慢 **12–27%**（k=4：216 → 246/275 µs）。
2. **长尾是工作集造成的，不是同租户**：3 层对照在同一负载下 p90/p50 ≈ 1.04–1.05，40 层是 **1.3–9×**。
   对照臂把租户固定住、只改工作集，尾巴就出现/消失。
3. 按超几何分布外推（N_res=100 ⇒ E[k]=4.44，40 层）：

| 曲线 | 按 p50 | 按均值 | 按 p90 |
|---|---|---|---|
| 3 层(今天) | 9.59 ms/token | 9.71 | 10.06 |
| prod_a | 10.85 | **14.98** | 25.27 |
| prod_b | 11.91 | **14.77** | 20.22 |

4. **与卡上实测对上了**：armA profile 的 `NOTIFY_WAIT` = 每层 371.8 µs × 40 = **14.87 ms/step**，
   与本曲线按均值的 14.77–14.98 ms/token 几乎重合（两条独立来源：主机侧 kt bench 与卡上 profile）。
   ⇒ CPU 侧就是主项，且旧的 10.41 ms/token 只在"按中位、小工作集"这一读法下成立。
5. ⇒ **重叠的上限由窗口卡死**：每层 CPU ≈ 372 µs（均值）远大于本层 NPU 窗口（GMM 实测 44.7 µs/层，
   整层窗口由 t4 第 2 项实测），所以 `min(cpu, window) = window`，天花板 ≈ 40 × (window − node)。
   任何声称能藏掉 10 ms 的方案，先检查它是不是跨了层（§4.9 禁止）。

## k.10 t5：`auto_dispatch_capture` 能不能救回侧流（写好，待排窗口）

`tools/kt_offload/oncard/t5_auto_dispatch.py`。具名差异：sglang 捕获传
`auto_dispatch_capture=True`（`npu_graph_runner.py:154`），我们 `vllm_ascend/compilation/acl_graph.py:187`
用默认 False；torch_npu 里这个参数会在捕获期进入 `_GraphDispatchMode`，把算子包进
`graph_task_group_begin/end(stream)` —— 正是决定"别的流上的活怎么进图"的那套机制，而我们三发 107011/107025
全是在它关着的情况下测的。
- **真实层**捕获场景（t3 那套，≥3 层以覆盖 kt 的 `layer_idx % 2` 槽复用），只切这一个参数；
  绝不用标量微型图（0920 的假绿灯就是那样来的）。
- 臂 `off`（默认 False）＝**阳性对照**，必须抛错才说明机器状态和前三发一致；不抛错则本次比较作废。
- 臂 `on`：捕获成功**不等于**通过，另外判 replay：≥3 个不同输入、每次逐位等于串行臂、replay 之间两两不同、
  每层 host-func 每次 replay +1（捕获期 0）。0920 那次就是"捕获成功而侧流工作没进图"。
- overlap 开关在 config 里**仍然是拒绝**的；用例只在自己建的层实例上翻 `off.overlap`，不动生产捕获路径。
- 失败的捕获可能污染同进程里后一次捕获 ⇒ 默认 `--arm both` 会打印这条警告，正式判读请分两次
  （`run_oncard.sh <card> t5 --arm on` 与 `--arm off`）。



## k.11 跨层入口全清单（kt 侧提供，已对着我们分支的 kt 逐行核过）

行号：`build_artifacts/moe_offload_ref/kt_site/kt_kernel/experts_base.py`（分支 HEAD `033ea76`）。
我方已核实的事实：标志字典 `BaseMoEWrapper._layer_has_pending_deferred` 是**类属性、键是裸 layer_idx**（:411）；
提交读的是**上一层**的键（:667/:744/:819）；自己那层的键决定 drain allowance（:771/:942）；
False 只在成功提交之后写（:697/:755/:844）；pinned 槽位 = `layer_idx % buffer_depth`，`buffer_depth=2`（:259）；
输出 H2D 是 `non_blocking=True`（:916）；整卡同步在 `submit_forward` 的 `_wait_device`（:817）。

### A 组：已变成会 raise 的拦截（代码里，不是文档里）

| 判据 | 判什么 | 在哪 | 失效形态 |
|---|---|---|---|
| A1 | `max_deferred_experts_per_token == 0` | `_refuse_cross_layer`，建 wrapper 后 | deferral **能不能产生** |
| **A3** | `not has_pending_deferred(L-1)` | 每次提交前（eager 的 `route`、两个图节点回调） | kt **读到了** ⇒ 下一次提交变 `incremental=True`，静默累加 |
| **A4** | `not has_pending_deferred(L)` | 每层 drain 之后（`merge`、两个图节点回调） | 残留 True ⇒ allowance 变 1（:771/:942）**少等一个任务**，读到写了一半的 output；异常路径（`_check_qlen_fits_cpp_buffers` :593/:650/:728、`_wait_device` :817）正会留下它 |
| A5 | 我方 wrapper 的 `layer_idx` 互不重复（弱引用，只管活着的） | `_register_wrapper` | 字典是进程级裸键 ⇒ 别人的 True 被我们读成"上一层有 deferred" |
| A9 | 所有 wrapper 的 `cpu_infer` 是同一个对象 | `_register_wrapper` | 两个队列 ⇒ "串行 FIFO" 这个前提悄悄不成立 |

A1 与 A3/A4 都留着：**一个判"能不能产生"，一个判"有没有读到"**，失效点不同。
kt build 若没有 `has_pending_deferred`，A3/A4 打 warning 明说**本 build 上覆盖为零**（缺席不是证据）。
单测 `tests/ut/kt_offload/test_cross_layer_contract.py`（13 条，含"图节点里必须记日志而不是抛进 ACL"、
"弱引用不能误红已释放的 wrapper"）。

### 🔴 A6：形态 C 必须建在**我们自己的缓冲**之上

kt 的 pinned 槽位是 `layer_idx % 2`（:259）⇒ **第 L 层与第 L+2 层共用同一组 input/ids/weights/output**。
形态 C 去掉 `submit_forward` 里那次整卡 `_wait_device`（:817）之后，L 层的输出 H2D（:916，non_blocking）
可能还在飞，而 L+2 层的 CPU 任务已经在改同一块 `output_cpu`。**今天不炸，只是因为那次整卡同步顺带等掉了它。**
⇒ 形态 C 若继续用 kt 的缓冲，就**必须**自己维护"未 drain 的在飞层数 < buffer_depth"的守卫；
而我们 packed 路径**每层各自的 pinned/设备缓冲**天然没有这个共享（`_packed_buffers` 按 bs 建、按层持有）。
**结论：形态 C 建在 packed 的缓冲之上**——即便 packed 因长尾默认关（k.8），**缓冲这一半要复用**。
"不受 A6 约束"的**前提就是用我们自己的缓冲**；一旦改回 kt 的缓冲，这条约束立刻回来。

### B 组：**强制不了**，只能写进设计并在评审时逐条过（不要假装拦住了）

| | 事实 | 我们能做的 |
|---|---|---|
| B1 | 队列在飞任务数**不可查**（sync 能让它变真，但查不到） | 只能靠"每层必须 drain"的结构，不能靠运行时断言 |
| B2 | pinned 输出的 H2D 是否完成 kt **不留句柄** | 守卫只能由**我们自己 record 的 event** 提供（形态 C 的等待点就要用它） |
| B3 | 所有层的中间缓冲是**进程级同一块内存**（C++ 侧 `shared_mem_buffer` 静态全局；此条在 Python 层看不到，未由我方复核） | 今天安全只因队列串行；任何并行化都要先解决它 |
| B4 | 加载与前向**不能重叠** | 结构上保证：`after_weights_loaded` 之后才有前向 |
| B5 | 换 bs / `clear_buffer_cache()` 会换张量对象，而在飞任务持**裸指针** | 缓解：所有捕获 bs 都在 `set_capture_batch_sizes` 里登记（已做，`_prepare_capture_buffers`） |
| B6 | `incremental=True` 还有两个 C++ 直接入口（`ext_bindings.cpp:472/477`） | 只能钉死我们自己的调用点：单测按 **AST** 检查每个 `forward_task` 的 `incremental` 实参必须是字面量 `False`（字符串搜索会被解释这件事的散文骗到，已踩过） |

⚪ 两条排查用（不是判据）：`new Args{}` 无 `delete`，每次 `forward_task` 漏 72–80 B——长跑时症状**看起来像"跨层缓冲没释放"**，
别误诊；任务异常**在下一次 sync 才重抛** ⇒ 报错会记在**下一层**账上，定位时要把层号往前推一格。



## m. 方法论：判据的运行环境与被判对象的运行环境不同 ⇒ CPU 单测必然测不到

规则本身与已知成员清单由 `MOE_OFFLOAD_PLAN.md` §4.14 持有（那份是用户的清单，改由那边改；这里只记**我方每条守卫用的是哪种应对**，避免同一条结论在两处各自漂移）。

> 判据的运行环境与被判对象的运行环境不同时，CPU 单测必然测不到。
> ⇒ 这类守卫**必须有一条在卡上跑的臂**，或**在 CPU 上把环境差异显式建模**；两者都没有＝零覆盖。

| 成员 | 我方应对 | 在哪 |
|---|---|---|
| 默认工厂设备（CPU 默认 vs 加载期 NPU） | **CPU 建模**：`meta` 当默认设备 + 源码级变异（去掉 `device="cpu"` 在 CPU 默认下全绿、在 `meta` 下必须红且红因点名 meta）+ AST 扫描 15/15 调用点 | `tests/ut/kt_offload/test_load_time_default_device.py` |
| 生产日志配置（pytest 挂 root handler vs worker 只挂 file handler 且不 setLevel） | **CPU 建模**：子进程里只调 `configure_ascend_file_logging()` 立起 worker 形态，三条阴性对照（旧写法静默／`init_logger` 写法静默／caplog 看不到） | `tests/ut/kt_offload/test_production_logging.py` |
| 设备端 dtype 支持表（fp4 视图不支持设备内拷贝/reshape） | **当前不适用**：我方路径按 int8/uint8 搬字节，不建 fp4 视图；将来若用到 ⇒ 需卡上臂（CPU 上无法建模一张设备支持表） | — |
| ACL 捕获对跨流的限制（107025） | **卡上臂**：t3/t5 在**真实层**上捕获；标量微型图给过假绿灯 ⇒ 替身必须与被测对象同规模同内容（这条本身也是一条通则） | `tools/kt_offload/oncard/t3_side_stream.py`、`t5_auto_dispatch.py` |
| 构建（共享 `.so`／算子包／kt wheel 换了） | **产物指纹 + 判读拒绝**：运行期盖指纹，判读脚本在指纹不同时拒绝给差值（见 m.2） | `tools/kt_offload/build_fingerprint.py` |

判定这类风险的问法不是"这个 bug 能不能在 CPU 上复现"，而是**"我的判据跑在哪个环境里，被判对象跑在哪个环境里"**——两者不同就先假设零覆盖，再去补卡上臂或补建模。

## m.2 读数一律盖构建指纹，判读跨构建拒绝给差值

政策与背景见 `MOE_OFFLOAD_PLAN.md` §4.13（共享 `.so` 于 2026-09-21 重建分发）。这里是**落地**：

- `tools/kt_offload/build_fingerprint.py`：`capture` / `stamp <dir>` / `compare A B` / `backfill`。
  内容＝`vllm_ascend_C*.so` 的 md5（含"运行时会导入哪一份"）＋算子包 `vendors/` 的**内容**指纹
  （逐文件 md5 + 路径 + 大小的 sha256；**按内容不按 mtime**，`cp` 会刷新 mtime；符号链接按 `link:<target>` 记，
  链接改指向也算变化）＋kt wheel sha256 ＋两棵树 HEAD/脏文件数。
- 谁盖：`run_oncard.sh`（t0–t5，落 `build_fingerprint.json` 并进 `provenance.json`，**在 dry-run 之前**就盖，
  所以自检也能验它）、`run_packed_ab.sh`（落在 run 与每个 arm 目录）、`profile_ops.py`（打印本次读数所属构建）。
- 谁拦：`packed_ab_report.py` 的 **§0 闸**——两臂指纹不同就**拒绝给任何差值**（rc=1），打印两边指纹并说明"跨构建不可比"；
  `profile_ops.py --against` 同理。阻断项＝`.so` md5／算子包指纹／kt wheel sha；两棵树 commit 只**提示**
  （文档改动不该拦住一次合法比较），`strict_trees=True` 可升级为阻断。
- **缺指纹也拒绝**：「没看到不一致」不是「一致」的证据。两份都是补盖的（只知道 `.so`）时，
  未记录的项打 `ADVISORY ... unknown on both sides` 而不阻断——已知项已经把构建钉住了。
- 起飞前可选闸：`ONCARD_EXPECT_SO_MD5=<md5>`，`.so` 不是那份就**拒绝起飞**（换 `.so` 期间用，避免读数落在换的中间）。
- 单测 `tests/ut/kt_offload/test_build_fingerprint.py`（15 条）：同构建给差值／不同 `.so` 拒绝且不打印差值／
  缺指纹拒绝／单侧未记录拒绝而双侧未记录只提示／树 commit 默认只提示且可升级／按内容而非 mtime／链接改指向能看出来。
  `selftest_packed_ab.sh` 另有 7 条端到端（含"拒绝时确实一个差值都没打印"）。
- 旧产物已**事后补盖**旧 `.so` md5 `2b58b578e4fc`（`packed_ab_0920_094053`、`prof_0920_045212`、
  `m3_ab_0920_061531`、`cpu_bench/20260920_022930`、本轮 `cpu_bench/*_prod`），写成各自目录下的
  `build_fingerprint.backfilled.json`，**不改原始 csv**，并在文件里写明"补盖的是出处、不是测量"：
  算子包指纹当时没记，留空而不是拿今天的树补一个假的。



## k.12 A/A 对照（先校仪器再量；窗口里第一件）

问题：packed A/B 的服务层逐位判据给出 31/50 不同（k.8 的红条），而这有两种读法——判据本身钝，
或 packed 真改了数值。**在分开之前，这个读数两个方向都不能用。**

做法：`MODE=aa bash run_packed_ab.sh <card>`，两臂都 `packed_transfer=false`（**改动项＝零**）。
关键不是"照抄参数"，而是**同一个函数**：runner 里只有一处 GSM8K 启动点（`run_evals`），
两个模式共用，所以参数**没有漂移的入口**（自检里有一条直接数启动点个数）。并发 8 的那一档
仍在 A/B 的同一位置（服务就绪后第一件事）。

每臂三个 eval：`c8`（并发 8，与 A/B 相同）、`c8b`（**同一次服务内**的第二发并发 8）、`c1`（并发 1）。
`c8b` 是自己加的旁证：它把"批组成"和"换了一个进程"从一个解释拆成两个。

判定表（`aa_report.py` 直接打印，窗口里不用手工拼）：

| 读数 | 结论 |
|---|---|
| 并发 8 分歧 且 并发 1 逐位相同 | 原因＝**批组成** ⇒ 服务层逐位判据**无区分力**；逐位一致只能在 t1/t2/t3 层级用例上判 |
| 两档都分歧 | **不是**批组成，另有来源（per-start 状态）要查；**不得归给 packed** |
| 两档都逐位相同 | A/B 的 31/50 ＝ **packed 真改了数值**（那它就是正确性问题，不是性能问题） |
| 并发 1 分歧而并发 8 相同 | 与所有假设相反 ⇒ **rc=1，先重跑**，不解释 |

拒绝条件（都会 rc=1）：缺 `c1`／生成为空／两臂题集不同／比较自身没有区分力（一个字符的变异检不出来）／
两臂指纹不同。**"没法判"和"判成一样"必须产出不同的东西。**

七个 fixture 覆盖每条分支（含三条必须拒绝的），`selftest_packed_ab.sh` 64/64。
A/B 的 31/50 在报告第 4 节被**引用**并附出处核对结果；构建证不了相同时明写"仅作上下文引用，
不在这条边界上做减法"（算子包指纹当时没记录，见 m.2）。

窗口执行单：`tools/kt_offload/oncard/RUNBOOK_window.md`（A/A → t4 → t5，三件都带
`ONCARD_EXPECT_SO_MD5`，.so 不是那份就拒绝起飞）。


## 上卡 A/B 方案（M3 侧流，1 张卡；两臂只差一个开关）

两臂：`overlap_side_stream: false`（现状）与 `true`。**同一构建、同一配置、同一套仪器**，只切这个开关。

1. **判据①逐位相同（最重要，先跑，不过不看性能）**
   - 单条请求：同一组提示、贪心、同 seed，两臂逐 token 比对；判据用 `oncard_common.bit_identical`（字节相同 or 值相同，
     处理 ±0/NaN），不是 rel_l2。
   - **并发**：≥2 组并发请求（不同长度、含一条长 prefill 与若干 decode 混跑），跑够让同一捕获 bs 被多次 replay；
     侧流的风险正在跨请求/跨流的时序，单条比对对这类缺陷是零覆盖。
   - 另核 `HOST_FUNC_CALLS[layer]` 随 decode 步数线性增长（侧流上的 host-func 真的每次都跑）。
2. **判据② TPOT**：`perf_baseline.sh`（同脚本、同参数），n≥3 且**重复到能看出分布形状**再下结论
   （多卡在 bs=4 上栽过：第一档取样看不出双峰）。报中位数 + 全部样本 + 离散度，不只报差值。
3. **判据③ 算子级 profiling 两臂各一份**（用户点名要）。开法与坑（计划 §4.6）：
   - 本版 vLLM 不认 `VLLM_TORCH_PROFILER_DIR`，要 `EXTRA_ARGS='--profiler-config {"profiler":"torch","torch_profiler_dir":"<dir>"}'`；
   - `prof_decode_capture.py` 在 404 时照样打印 "✅ captured" ⇒ **自己数 `<dir>/*_ascend_pt`**；
   - `prof_decode_breakdown.py` 会因「无 kernel 带图 Model ID」拒绝出结果 ⇒ 直接聚合 `kernel_details.csv`
     （`transport_budget.py measure` 就是干这个的）。
4. **收益要落到空隙表上**，不只一个 TPOT 差值。两臂各重算三项并对照：
   | 量 | 现状（offload 开、侧流关） | 侧流开 | 读法 |
   |---|---|---|---|
   | 每 token 空隙合计 | 18.73 ms | ? | 总量 |
   | >100 µs 大空隙：个数 / 合计 | 43 段 / 17.91 ms | ? | **个数减少**=某些层的暴露被整段盖住 |
   | 单段大空隙的中位/分布 | ≈0.42 ms | ? | **单段变短**=部分盖住（NPU 活不够长） |
   | device 真实忙（并集）占比 | 57.1% | ? | **占比升高**=同样的活挤进更短的时间 |
   三种机制（段数少了 / 单段短了 / 整体被盖住）要能分开报，别合成一句"变快了"。
5. 失败处置：判据①不过 ⇒ 立刻停，不看性能数字。2026-09-20 就是这样：见 §k.2。
6. **前置**：整网 A/B 之前先过 T3（`run_oncard.sh <card> t3`）。2026-09-20 的教训是整网跑完也说不出是哪个环节坏了。

## 上卡验证草案（M1，供排卡；先拿租约 `card_lease.sh acquire`，TMPDIR=/dev/shm/dsv41_tmp）

> 步骤 1–2 与 M2 的单层部分已写成脚本：`tools/kt_offload/oncard/`（T0/T1/T2，外壳 `run_oncard.sh` 强制租约与异常立停；判据、时长、资源见其 README）。

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

## 上卡验证草案（M2）

前提：M1 步骤 1–2 通过。`VLLM_USE_BREAKABLE_CUDAGRAPH=1`（已验证配置），另跑一遍 `=0`（ACLGraphWrapper）。
1. **eager vs graph 逐 token 一致**：同一批提示（含 bs=1 与 bs 在捕获列表内的多个值），贪心，同 N 同驻留集；
   eager（`--enforce-eager`）与 graph（FULL_AND_PIECEWISE）逐 token 比，判据：前 256 token 逐 token 相同；
   不同则先量地板：graph 关 offload（全驻留）与 eager 关 offload 的同一对比。
2. **无 10701x**：日志与 plog 里 `107011/107015/107027/107030` 为 0（107011 被吞，只允许出现在我们吞掉的那条路径，grep 应为 0 条外抛）。
3. **CPU 输出不是捕获那一次的旧值**：在同一个捕获 bs 下连续发 ≥3 个不同输入，单层 dump `merge` 前的 `cpu_out`
   （或比较 routed_out − NPU 驻留部分）：3 个输入必须得到 3 个不同的 CPU 输出，且各自与 eager 同输入的 CPU 输出一致（bf16 地板内）。
   "多个输入同一个输出"= 图没重放 host-func 节点的指纹。另核 `HOST_FUNC_CALLS[layer]` 随 decode 步数增长（每步每层 +1）。
4. **首个请求**：服务起来后的第一个请求第一步即满足 3（捕获那一次不执行节点，这里核它不影响真实请求）。
5. **FULL 与 PIECEWISE 各一遍**：`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` 与 `PIECEWISE` 分别跑 1–3。
6. 变异臂：把 `_capturing` 恒 False（捕获时走 eager）——应在捕获时报 107027 或出现 3 的指纹；必须红。

## 开放问题

- ⚫ NPU 驻留专家是 W4A8（激活量化为 MXFP8），kt CPU 专家是 bf16 激活（M0 反馈）⇒ "驻留+CPU" 与 "NPU 全专家" 不会逐位一致，
  CPU 部分差异应在 fp8 激活量化量级。上卡判据分臂写（见计划的上卡步骤）。
- ⚫ kt 的 swiglu clamp 语义（gate 只 clamp 上界、up 双边？）是否与 `npu_swiglu_group_quant(clamp_value)` 一致——M0b 对的是
  torch 参考，需确认参考与 NPU 算子语义一致。
- ⚫ `npu_format_cast(29)` 对 N 槽权重是否有对齐填充（影响 §i 的 17.93 MiB）。
- ALLGATHER + W4A8 `quant_mode=3` 在 A5/V4.1 未验证（§g）。
- ⚫ 构建在 `build_artifacts/moe_offload_ref/kt_site` 的 kt wheel 早于 `292be81`，没有 `set_capture_probe`；图模式上卡前需要 kt agent 重建。
- ⚫ `_launch_host_func` 回调里调 kt 的 `cpu_infer.submit/sync`（阻塞主机回调线程直到 CPU 专家算完）在 ACL 回调线程上是否有超时/死锁
  问题：sglang 在 A5 上同样这样做（`_kt_npu_graph_host_forward`），旁证可行。
- kt 缓冲是 `layer_idx % 2` 两槽复用、按 bs 共享；eager 与 replay 对同一 bs 用同一组 pinned 缓冲（时间上不重叠，eager 路径自带同步）。
