# 流式 prefill + 动态热专家（单卡 MoE offload 优化）

> 维护者：`dsv41_single_card_stream_prefill`（2026-09-22 起）。**本线自己的工作文档**，
> 主线跟踪文档是 `MOE_OFFLOAD_PLAN.md`（**唯一事实源**）——本文出现的 **§4.x 一律指主线文档的节号**，
> **k.\* / m.\* 指 `docs/moe_offload_design.md`**。结论定稿后再按用户的意思决定要不要并回主线。
> **读数全部当场内联**：本文是 `build_artifacts/moe_offload_ref/dynhot_plan/FEASIBILITY.md` 的合入版，
> 那份**未被 git 跟踪的文档已删除**（k.16：未跟踪的产物会消失）。
> 状态：**设计与可行性已定，尚未开工实现**；phase2 优先用卡 0–2，本线用**卡 3**。

## 1. 为什么做这两件（它们在性能图里的位置）

§4.18 的分解：等 CPU **15.88 ms（41%）**，其中 kt 真算 246.1 µs/层、接缝 107.5 µs/层。
而 §4.3 的 CPU 曲线是**内存带宽硬底**（428 GB/s = 本机上限 86%），§4.9 禁止跨层 ⇒ 重叠窗口上限只有
驻留 GMM 的 42.7 µs/层（≈1.7 ms/token）。⇒ **这两件是仅有的两条绕开"10 ms 硬地板"的路**：
- **F1 流式 prefill**：长 prefill 时把整层专家权重灌上卡，该层 MoE **全在 NPU 上算、完全不走 CPU**；
- **F2 动态热专家**：改的是"**哪些**专家驻留"（不改驻留**数**，§b.3 的 `num_slots` 假设不用动）。

## 2. 参考实现（一手）

`/workspace/code/dsv4_single_card/pb-sglang-dsv4-cann9`，分支 `dsv4-cann9-no-patch` @ `ff46d66`，
核心 `python/sglang/srt/layers/moe/kt_stream_prefill.py`（1467 行）；910/A3 实践文档
`worktrees/ktransformers/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md`（开关 `KT_PREFILL_STREAM` /
`KT_MXFP4_DEPOOL` / `KT_DYNAMIC_RESIDENT` / `KT_SIDE_STREAM`，判据行 :294-298）。

- 它**流的是权重、不是 token**；阈值 `KT_PREFILL_STREAM_THRESHOLD=512`（单次 forward 的 token 数下限），低于它退回 hybrid。
  ⚠️ **"把长 prefill 切块让 CPU 与 NPU 分块流水"那一半在该文件里不存在**（token 分块是 sglang 的 `--chunked-prefill-size`）。
- 动态热专家**折叠在流式里**（`:1282`）：`bincount(topk_ids)` 逐层（`:1188-1191`）→ 从刚转好的整层 `index_select` 出 top-K
  （`:1253-1256`）→ 写进驻留权重本体。**全程在设备上，没有第二次从主机取。**
- 换入**执行在图外、生效在图内**：`_is_prefill() = not is_current_stream_capturing()`（`:573-577`）；改的是常驻 storage，
  图捕的是地址，replay 读到新内容。掩码**三处都要 in-place 改**：device `gpu_experts_mask`、`logical_to_gpu_index`、
  **kt 的 pinned CPU bool mask**（C++ 按指针实时读）。
- 🔴 **一条我们计划里原本没有的前置**：`_remap_resident_params_to_cache`（`:898-914`）——加载期就把驻留权重/掩码
  从"加载权重的只读区"clone 出来，否则每次改写触发**设备一致性 flush** 而拖垮同步。我们要有对应物
  （挂点就是现有的 `after_weights_loaded`）。
- 🔴 **设备侧双缓冲被它实测否掉**：注释 `:11` "measured ≤2% for twice the HBM"。它设备侧只有**一个复用 slot、串行**；
  唯一的 ping-pong 是**主机侧 pinned**（给 L+1 层预填，1 个线程 + `KT_MXFP4_COPY_THREADS=32` 并行 memcpy）。
  ⇒ **原 O6「双缓冲 13.5 GiB ⇒ 必须开 L3」这个开口关闭。**
- slot **在 KV profile 之前**预留（`maybe_reserve_slot` :916）；depool 的 slot 是**平 ND** 而非 NZ，理由正是
  "往 NZ 目的地切片拷会触发整张反格式往返/OOM"（`:844-851`）。

**不照抄的四件（理由已查明）**：① `AscendC MXFP4→W8A8` 转换器**整件删掉**（A5 原生 FP4；且写死 910B、UB 预算把维度卡在
≤4096 而我们 hidden=5120）；② **O_DIRECT 读取器**（长度不按块对齐、每线程 8/4 GiB scratch 无"装得下"检查而**短读只 `break`
静默截断** `:678-681`、W8A8 那支**缺错误路径 ⇒ 读失败后 `done_q.get()` 永久阻塞**）⇒ 我们先 mmap；
③ **掩码延迟到末层 flush**（`:1237/:1273-1286`）⇒ 中途出错会留下"权重新、mask 旧"，我们要**全有或全无**；
④ **零 assert、失败静默回退 hybrid**（`:1417`、`:1461-1467` 等 5 处）⇒ 我们要**覆盖量记账 + strict 必须 raise、零覆盖判 fail**。

🔴 **它的计算链是 W8A8，不能抄**：`init_routing(quant_mode=1)`（int8 + per-token scale）→ `grouped_matmul`
（w13 NZ int8，scale bf16 `[E,2I]`，`group_list_type=1`）→ `npu_dequant_swiglu_quant` → gmm2 →
`finalize_routing(drop_pad_mode=2)`。**MXFP4 在它那里只是存储格式，算之前一律转成 W8A8-NZ，没有 W4A8 的 gmm 路径。**
我们是 W4A8 MXFP4（block scale 每 32 个一组、生产走 `quant_mode=3`）⇒ **流式的计算链必须自己拼**，
且 `group_list_type` 不匹配是**静默算错**（它 `:1024-1032` 专门警告），必须有变异臂。

⚠️ **它文档里的读数不可作为我们的预期**：prefill 15.6 s@801 tok → 20.4 s@31,540 tok、decode ~19–20 tok/s，
是 **Atlas A3 单 die、HBM 61.3 GB、五开关全开** —— 不同机不同配置。

## 3. 卡上实验：三个关卡（2026-09-22，卡 3，共 9 发 ≈380 秒）

产物 `build_artifacts/moe_offload_ref/oncard_runs/{20260922_125758_e2p, 20260922_130518_e5, 20260922_133143_e2w}`
【可复验】；脚本 `tools/kt_offload/oncard/{e2p_slice_cast.py, e5_chunked_accum.py, e2w_write_channels.py, e_common.py}`
（commit `4428081ca`）。本构建：`.so` `85331298bdef` / 9.2.0-beta.2。**未改任何生产代码。**

开销实测（供后续发次做预算参照）：进程峰值 **+5.5 GiB**（预算 10 GiB），`MemAvailable` 最低 **276 GiB**
（从未接近 100 GiB 自停地板）；dmesg 本卡新行 0、他卡 0；每发起跑前三维判卡 + 拿租约、用完即还，**全程没碰卡 2**。

**E5 ✅ 层内分块累加 ≡ 一次全算**（关卡③关闭）：60/60 通过、12 个变异臂全红。A 臂 32 专家一次算，
B 臂 8×4 与 4×8 分块，路由改写用**生产的** `split_routing` + `LayerPlacement`。
- 数值 rel_l2 **2.32–2.34e-3**（实测地板后 tau 7.81e-3）、ulp 0.79–1.02（tau 4.0），T=1/4/32 共 6 组；
- **集合判据**：每 token 贡献专家的**多重集逐个相同**，100% 覆盖，有效路由数 = 原路由数；
- 变异臂（某块 keep 不置 0）两条判据都红。⇒ **层内分块成立，且不跨层。**

**E2′ ✅ 逐专家 cast 与整块逐字节一致**（关卡①）：C=1/4/8 的切片 cast 与 32 专家整块 cast **逐字节相同**
（w13/w2 + 两个 scale 共 12 组），每专家字节数两侧一致（11,796,480 / 5,898,240 / 737,280 / 368,640 B），
**dim 0 上无 padding**；10 个变异臂全红。
⚠️ hook 传 29、回来是 **50 = FRACTAL_NZ_C0_16** ⇒ 判据要写"**非 ND + 两臂相同**"，**写死 29 会误红**。

**E2w ✅ 写入通道闭合，但只能整块写**（关卡②）：七个候选，**唯一可用 `torch_npu.copy_memory_(dst.data, src.data)`**。

| 候选 | 结果 | 读数 / 报错 |
|---|---|---|
| C1 整张量 `Tensor.copy_` | ❌ | `copy_ do not support internal format` |
| **C2 整张量 `torch_npu.copy_memory_`** | **✅** | 四参数逐字节相同；**data_ptr 不变**；**100 专家 1.751 GiB = 2.6 ms ≈ 669 GiB/s** |
| C3 storage alias 字节切片 + `copy_` | ❌ | 同 C1 |
| C4 同上 + `copy_memory_` | ❌ | `self.storage_offset() == 0` 断言失败 |
| C5 `dst.data[i:i+1].copy_` | ❌ | 同 C1 |
| C6 `param.data = 新张量`（仅对照） | ⚠️ | 写得进去，但**四个参数地址全变** ⇒ 图重放喂旧缓冲，生产不可用 |
| C7 NZ→ND→写→转回 NZ | ❌ | **`361001`，ND 转不回内部格式** |

🔴 **作用域探针（决定了 F2 的形状）**：递给 `copy_memory_` **1 个专家的视图**（11,796,480 B），而 storage 是
**8 个专家**（94,371,840 B）⇒ **另外 7 个也全被写了**（unchanged=0/7）⇒ **它忽略视图范围、整块拷贝**。
⇒ **"按专家增量写 NZ"在本构建不存在。** C6 的价值是把两件事分开：**内部格式并非不可写，它只是不接受
`copy_` 与带偏移的视图。**

## 4. 🔴 一条前提被推翻（凡建立在它上面的判断都要重推）

sglang `935c2aa` 记录的"`index_select(out=)` **静默丢弃**（GPQA 70%→41%）"**在本构建不成立**：
ND 的 scale 上 `out=` **真的写进去了**；内部格式的权重上**直接抛异常**。**两侧都不是"静默"。**
⚠️ 连带作废：`docs/moe_offload_design.md` 与旧 ROADMAP 里"S2 必须复现这个静默 no-op，复现不出判 fail"的判据
——照它执行会把**正确的实现判 fail**。

⚪ **过程教训（比结论更值钱）**：第一版阳性对照的目标缓冲里**本来就装着要写的内容** ⇒ "0 字节变化"这个读数
对"写没写"**完全相同（零区分力）**，第一发因此读成"复现了"；改成**零初始化 + 先证明区分力**后**结论反转**。
⇒ **新的阳性对照 = 目标缓冲先清零 + 一条"故意不写"的臂必须红。**

## 5. 我们的实现形状（与参考实现的差异）

**A 流式 prefill**
- **按专家分块**，不必整层一次（分块**不需要新机制**：块内映射块内下标、块外映射 dummy 槽且权重置 0 —— 就是已有的
  `safe_slot_table`/`keep_table` 变换；combine 已乘路由权重 ⇒ 块外贡献为 0。**E5 已验证等价**）：

| 方案 | slot + 转换暂存 | 说明 |
|---|---|---|
| 整层 384 | 13.44 GiB | 装不下（见下"HBM 来源"） |
| 64 × 6 批 | 2.24 GiB | 备选 |
| **32 × 12 批** | **1.12 GiB** | 首选 |

- **不需要写 NZ**（cast 输出直接喂 GMM）⇒ E2w 的限制对 F1 无影响；slot 另建、**不复用 Parameter 的 storage**
  （驻留是 `[N,…]`、流式吃 `[C,…]`）⇒ 需要一个吃外部权重的专家函数。
- 接入点 `ops/fused_moe/routed_experts.py:forward_impl`，分叉在 **`kt_offload.route(...)` 之前**；
  走流式的层**不提交 CPU 活、也不 merge**。
- 前提（已核）：token ≥512 的 forward **不在任何捕获尺寸内**（`capture_sizes=[1,2,4,8]`）⇒ **纯 eager**，
  可自由分配/cast。**代码里要加断言。**

**B 动态热专家**——**把选择挪到 cast 之前**，绕开"按专家写 NZ"这个本构建做不到的动作：
```
流式的每个块（ND / 原始 MXFP4）
  ├─► 参与本层计算
  └─► 把选中的专家按新驻留顺序写进一块 ND staging   ← ND 可按专家写
12 块走完 → 一次 npu_format_cast                    ← E2′：切片/整块逐字节一致
          → 一次 copy_memory_ 写进驻留权重本体        ← E2w：2.6 ms / 1.75 GiB，地址不变
```
- **同样不需要第二次从主机拉**（数据来自流式那一趟）。⚠️ 选中的专家分散在不同块里 ⇒
  **ND staging 要活过整层的所有块**（同一层内，不跨层）。
- **原子步**（不抄参考实现的末层 flush）：换入点放在 **step 边界**（没有在飞的 forward），每层固定顺序
  **权重 → 槽位表 → keep 表 → 最后掩码**，全部 in-place、不换对象；40 层做完再放行下一个 step。
- 热度统计复用 `kt_offload/expert_counter.py`（已含 DP 切片与 profile-run 排除）；
  ⚠️ 它现在**图模式下会 raise** ⇒ 放宽成"只在非捕获 forward 里记"（决策点本来就在 prefill，图外）。
- ✅ kt 侧**零改动**：`skip_gpu_expert_weights` 由 `kt-kernel/python/utils/amx.py:964` 的 `bool(load_kwargs)`
  自动决定 ⇒ **按全集加载它自己就是 False**，`lacks_cpu_weights()` 恒假（原 design b.4 的"load 之后改 mask 会空指针"
  这个开口就此关闭，剩下的只有"原地改 pinned bool mask"）。

**HBM 来源（不减驻留专家）**：§4.18 的 serve.log 显示 KV 只用 27.40 GiB、日志自称"要 38.81 GiB 才用满"
⇒ **有 11.41 GiB 现在谁也没用**（被 `util 0.9` 压着）。slot 1.12 + ND staging 1.75 = **2.87 GiB**，装得下，
**驻留数 100 不动**。⚠️ 两条必须实测而非推断：① 用这块额度要动 `--gpu-memory-utilization` / `--kv-cache-memory`，
**等于吃安全余量**；② 🔴 **提高 mnbt 与 slot 抢同一块 HBM**（prefill 激活峰值 + engram staging 缓冲都按 mnbt 分配，
§4.5.6 R5）⇒ "mnbt 提到多少"不是独立旋钮，要和 slot 进同一本账。

## 6. 🔴 用户裁决（2026-09-22）

| # | 事项 | 裁决 |
|---|---|---|
| **D1** | 预取下一层**权重**算不算"跨层" | ✅ **可以预取**（只搬字节、不改数值）。但**不许在 HBM 上开 ping-pong** ⇒ 预取放**主机侧 pinned**，设备侧**单 slot**（正好与参考实现 ≤2%/2×HBM 的实测一致）；不得因此减少驻留专家数 |
| **D2** | 流式收益 ∝ mnbt，与 KV 回收方向相反 | ✅ **按提高 mnbt 的方向做**；**主指标仍是 bs1 TPOT**。L5（mnbt 4096→2048 换 1.92× KV）**记账不做** |
| **D3** | CPU 侧 BufferB 覆盖 | ✅ **放全集**。理由（用户）：动态换入**跨 request** 改热池后，**任何专家都可能被翻回来**，候选子集会在"翻到窗口外"时空指针，而失效形态是**算错不是报错** |
| **D4** | engram 重铺成 interleave | ✅ **要做**（D3 全集 ⇒ 单节点放不下 ⇒ 避不开）。但它是**唯一不可逆**的一步 ⇒ **排到最后**，且先用不拉整网的办法量出 O10 的 78 GiB 分歧 |
| **D5** | S2 的阳性对照换成什么 | ✅ 换掉（原判据已被 §4「一条前提被推翻」 推翻）：**目标缓冲先清零 + "故意不写"的臂必须红** |
| 排期 | 谁先、用哪张卡 | ✅ **phase2 优先（卡 0–2）**；本线用**卡 3**，且**先做不需要拉整网、不需要重铺 engram 的部分** |

## 7. 还没量的（按重要度）

1. 🔴 **H2D 完全没数** —— 两个功能共用这条通道；2.6 ms 的**设备内**拷贝旁边，它很可能才是瓶颈。10 分钟可测。
2. **图捕获下真跑一遍**：C2"地址不变"是**必要**条件、不是充分条件。
3. 对"**重排过顺序的 ND 缓冲**"做 cast 是否仍正确（E2′ 只证了切片与整块一致）。
4. 384 整层的 cast / 分块 / 长 prefill 都没测（E5 的 T 只到 32）。
5. `copy_memory_` 的异步与侧流语义；整块重写期间该层不可用怎么处理。
6. 🔴 **O10：两个系数打架、都没核** —— `384×0.7004 = 268.95 GiB`（NPU 侧形状推导，`placement.py:362`）
   vs kt compact 注释推 **≈191 GiB**（`common.hpp:281-287`、`amx.py:971-973`）。**差 78 GiB，正好是"node0 够不够"
   的量级** ⇒ **当场实测 `/proc/<pid>/numa_maps`，不靠任何一个外推**。它决定 D4 重铺的目标分布与闸值。

## 8. To-do（ID 固定，汇报引此号）

**P0 文档落地（0 卡）** — `T0.1` 三个实验读数内联 ✅(本节) · `T0.2` 记"静默丢弃不成立" ✅(§4「一条前提被推翻」) ·
`T0.3` 写入五条裁决 ✅(§6) · `T0.4` 改掉已被推翻的判据（design 的 S2/E2′ 相关） ·
`T0.5` HBM 账改成单 slot、关闭原 O6 开口 ✅(§2/§5) · `T0.6` 记 `_remap_resident_params_to_cache` 前置 ✅(§4.20.2)

**P1 量 78 GiB 分歧（0 卡）** — `T1.1` 只加载一层专家，读 RSS + `numa_maps` 得每专家每层真实字节 ·
`T1.2` 判 191 还是 269；阳性对照＝只给一个 node 必须 OOM 或被闸拦（**悄悄跑通 = BIND 没生效**） ·
`T1.3` 回写本节并给 D4 的目标分布与闸值

**P2 H2D 实测（卡 3，15–20 min）** — `T2.1` 租约 + 三维判卡连采 ≥3 + `TMPDIR=TMP=/dev/shm/dsv41_tmp` ·
`T2.2` mmap→pinned→设备，32×12 / 64×6 两档，中位 + p10/p90、≥20 轮**交错** ·
`T2.3` 阳性对照：**非 pinned 源必须掉速**；线程数打真值（`nproc` 恒为 1，真实 256） ·
`T2.4` 与 CPU 侧 **428 GB/s 同口径**并排 · `T2.5` 顺带量 D1/D2 的 HBM 账（11.41 GiB 能不能用、提 mnbt 涨多少） ·
`T2.6` 判 H2D 是不是真瓶颈（对照设备内 2.6 ms / 1.75 GiB）

**P3 设计 + CPU 可测部分（0 卡）** — `T3.1` F1 的 **W4A8 计算链**自己拼 + `group_list_type` 变异臂 ·
`T3.2` 分块路由改写复用 `safe_slot_table`/`keep_table`，接入点分叉在 `route()` 之前 ·
`T3.3` 断言 token≥512 不在捕获尺寸内 · `T3.4` 换入原语 + 字节定位器（mmap，**不抄 O_DIRECT**） ·
`T3.5` 掩码三元组**原子提交器**（权重→槽位表→keep 表→掩码，全有或全无） ·
`T3.6` 覆盖量记账 + `strict=1` raise，**零覆盖判 fail** · `T3.7` 两个开关做成 `KTOffloadConfig` bool 字段，
默认关，判据含"关时与改动前逐位相同" · `T3.8` `expert_counter.py` 放宽成只在非捕获 forward 记

**P4 单层上卡验 F1 计算链（卡 3，1 h）** — `T4.1` 与正常路径**逐位相同** · `T4.2` `group_list_type` 1→0 变异臂必须红 ·
`T4.3` 覆盖量记账，**零覆盖判 fail**

**等放行（不在本轮）**：S2 格式通道 · S3 动态换入小用例 · S5 整层 slot+I/O · S6 全集+频率 ·
D4 重铺窗口 · S7 整网 A/B。
