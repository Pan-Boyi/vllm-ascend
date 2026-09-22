# 单卡 MoE offload：交接给下一个 session

> 写于 2026-09-22，上一条线是 `dsv41_vllm_singlecard`（上下文将满，故交接）。
> **唯一事实源仍是 `MOE_OFFLOAD_PLAN.md`（§1–§4.17）与 `docs/moe_offload_design.md`（k.* / m.*）。本文只给"接下来做什么、怎么做、坑在哪"。**
> 状态标记：✅ 已完成且有实测 · 🟡 代码就绪待上卡 · ⬜ 未开始 · ⛔ 已否决

---

## 0. 三十秒版

单卡（TP1/DP1/EP1）+ CPU 专家 offload **已经整网跑通并有精度**：图模式 GSM8K 50 题 50/50，bs1 TPOT ≈40.3 ms（旧基线）。
差距的三分之一是**等 CPU 专家**（每层 ≈372 µs，实测）。**用户已禁止跨层** ⇒ 精确类优化的上限只有 1.8–4.4 ms/token。
现在的头等任务是 **rebase 到重构后的 clean**（用户 2026-09-22 拍定：**做 rebase，不重新分叉**），因为主干已交付 O6（>16k）与 −16.4%（四卡口径）的 decode 优化，而我们的对标值和读数已全部过期。

---

## 1. 当前状态

| | 状态 |
|---|---|
| 两棵树 | `worktrees/moe-offload`（vllm-ascend，分支 `dsv41-moe-offload` @ `d5476630c`，66 个提交）、`worktrees/moe-offload-vllm`（vllm @ `a9686bc6e6`，**我方零改动**） |
| 分叉点 / clean 头 | `1b1ad396b` / **`f85267ab8`**（落后 24 个提交；vllm 侧两边同 commit） |
| 代码 | ✅ M1 eager 集成、✅ M2 图模式 host-func、✅ 频率计数器、✅ packed_transfer（默认关）、⛔ 侧流（图模式下直接拒绝启动）、🟡 L3 SWA block_size（默认关） |
| 测试 | kt_offload 单测 266（rebase 后 283，含 SWA 判据）；`selftest_cpu` 70、`selftest_packed_ab` 98 |
| 上卡就绪未跑 | 🟡 **A/A → t4 → t5**（执行单 `tools/kt_offload/oncard/RUNBOOK_window.md`）——⚠️ rebase 后期望值与基线都要改，见 §3 |
| kt-kernel | 分支 `dsv41_vllm_offload_dev`（`worktrees/ktransformers`），已含 A5 两个 cherry-pick + `set_capture_probe` + `sync_cpu_only`；wheel 装在 `build_artifacts/moe_offload_ref/kt_site`（sha256 `1b8e9b21…`）。**用户尚未 push 过这条分支** |

---

## 2. 不可违反的约束（用户 / 协调方定的）

1. 🔴 **不允许跨层**（用户 2026-09-22 前定）：任何把 CPU 专家贡献推迟到下一层/下一步的机制一律不做，含 kt 的 `max_deferred_experts_per_token`。已做成会 raise 的硬拦截（A3/A4 读真实状态位）。
2. 🔴 **单卡的改动只在本线自己的树/脚本里**；integration 的树、共享脚本、多卡基线保持原样（只调用不改）。共享件清单见 PLAN §5。
3. 🔴 **不重编 `.so`、不动共享算子包**——起飞闸查全部注册的树，重编会挡住所有人。真要动先找 `dsv4.1_flash_dev`。
4. **碰卡先拿租约**（`build_artifacts/card_lease.sh acquire`，`LEASE_PID` 给活得够久的进程，不要 `$$`）；卡 1 不用；卡号每次去问，不引用旧约定。
5. **代理只给 GitHub/Claude**，其它外部访问 unset proxy。**不用 `pkill -f`**（会杀掉自己的 shell，栽过两次）。
6. 性能 A/B **两臂必须同时段、交错轮次**（机器会自漂 2.5×），并报噪声地板；**地板没测到而又测出差值 ⇒ 判 fail**。跨发读数**不得相减**。

---

## 3. 任务一：rebase 到 `f85267ab8`（不占卡的部分先做完）

**已实测的代价**（临时 worktree 上真跑过 `git rebase dsv41-clean` 后丢弃）：**1 处相邻行冲突**（`envs.py`），`caches.py` 自动合；我们 283 条测试全过；`models`/`attention` 的失败集合与 clean 单独跑**逐条相同**（47/47），我们多 17 个通过。我方依赖的十余个文件在新 clean 上**逐位未变**（`routed_experts.py`、`token_dispatcher.py`、`prepare_finalize.py`、`moe_comm_method.py`、`w4a8_mxfp4.py`、`acl_graph.py`、`model_runner_v1.py`、`engram_embedding.py`、`ascend_config.py`…）。

### 3.1 步骤

1. **打快照分支**：`git branch dsv41-moe-offload-pre-rebase-0922 HEAD`（先例：`-pre-rebase-0919`、`-pre-ff-0920`）。
2. `git rebase dsv41-clean`；`envs.py` 冲突取 clean 那一行 + 保留我们两项（`EXPERT_COUNT`、`KT_HOSTFUNC_LOG`、`SWA_BLOCK_SIZE` 等，按实际）。
3. **五项跟改**（缺一就会静默出错）：
   - `tools/kt_offload/kv_budget_singlecard.py:57` 写死 `P_IDX=16896` ⇒ index-K 页已降为 8,704 B；且 `DEFAULT_LOGS` 指旧 log 会一直绿着掩护它。
   - `tests/ut/models/test_deepseek_v4_1_swa_block_size.py:267/333` 与 `test_kv_budget.py:66` **手搭 spec、不调真对象** ⇒ 生产池块 542,208→517,632 它们照样绿（**假绿灯**）。改成调真的 `get_kv_cache_spec`。
   - `tools/kt_offload/serve_v41_singlecard.sh`：补 **DS41 段**（照共享 `cookbook/serve_multicard.sh:66` 的 ③.1，约 20 行，`DS41=/workspace/code/dsv4_single_card/ds41_site` 前置 PYTHONPATH；**缺了不报错、只是退回旧路径没有 O6**）＋ **显式 `VLLM_USE_BREAKABLE_CUDAGRAPH=1`**（新 O15 闸：真捕图不传就 raise；`--enforce-eager` 豁免）。
   - `RUNBOOK_window.md` 的 `ONCARD_EXPECT_SO_MD5` 与 `build_fingerprint.py` 的期望值换成**新 `.so`**，否则三发全部 rc=3 拒绝起飞。
   - **指纹要覆盖新四件**（`custom_ops_dsv41/*` 三轮子+`.run` 与 `ds41_site`）——现在只盯 `.so`+`vendors/`，**它们换了"同构建"判定发现不了**。
4. **跑测试的位置有讲究**：worktree 必须摆在 `<proj>/worktrees/<name>`，否则 `test_kv_budget.py:24` 按 `parents[1]` 找 `build_artifacts/` 会**静默 skip 14 条**（读数是"269 passed, 14 skipped"，看起来像通过）。
5. **不要从 `worktrees/clean` 的工作树拷文件**（那里现在有别人的未提交改动）；以已提交的 `f85267ab8` 为准。

### 3.2 rebase 的风险（按严重度）

1. 🔴 **旧 `.so` + 新代码 = 静默算错，不是崩**。clean 改了 csrc（`op_api_common.h` 的 4-bit nibble 分支）并默认把 `float4_e2m1fn_x2` 喂给新算子；我们的 `.so`（`2b58b578e4fc`）里那 6 个符号 0 命中。⇒ **rebase 完成后到换 `.so` 之前，绝对不要在卡上跑任何"结论性"的发次**。换完先跑 t0（单层逐位/地板判据 + 6 条变异臂，约 44 s）当 sanity。
2. 🔴 **换 `.so` 与算子包不是我们能单方面做的**：钉值按树拆在 `verify_shared_artifacts.sh:144-180`，我们那一行现在钉的是 9.1.0/759 文件那份，**且理由字串写的是"从 clean 分出时未变"——rebase 后这句为假而闸仍然绿**（它比产物、从不比源码）。必须与 `dsv4.1_flash_dev` / `dsv41-integration` 协调：换包 → 重跑 GSM8K → 再改钉值（他们的规矩）。
3. 🔴 **"先不换包"这条退路被堵死**：`INDEXER_QUANT_CACHE` 默认已是 "1"，index-K 已是 MXFP4，**不能单独关**（要连 `caches.py` 的 spec 一起改）。⇒ 要么换包，要么别 rebase。
4. **ds41 缺失的症状是"能跑但没有 O6"**（退回 v2、>16k 被守卫挡），不是报错 ⇒ 不加那 20 行 launcher 代码，会静默失去 >16k。
5. **所有旧读数作废，跨 rebase 的比较无效**：§4.6 全表（40.47 ms、device 忙 57.1%、43 段/17.91 ms）、§4.10 的 NOTIFY_WAIT 371.8 µs/层、§4.5 的 KV 绝对数（42,733 / 164,420 / 27.34 GiB / 670.7 KiB）全要重算重测。**不受影响**：§4.3 的 CPU 曲线（428 GB/s、10 ms 地板，与 `.so` 无关）与 L3 的**相对**倍数（3.85×、省 20.2 GiB）。
6. **对标值变了**：多卡 26.84 ms 作废，新的是 **≈22.7 ms（四卡、同时段 A/B、−16.4%）**；⚠️ **单卡能拿到多少没有读数**，别把四卡的降幅当成我们的。
7. **形态 C 的窗口变小了**：clean 把 NPU 侧 decode 压掉 16.4%，而重叠上限 = min(CPU/层, **本层 MoE 的 NPU 时间**) − 回调成本。§4.7/§4.9 的天花板要按新 device 时间重算。
8. 历史重写本身风险低：**这条分支从未 push 过**（本容器 ssh 推 GitHub 无权限），没有别人基于它开发；但仍要打快照分支。

---

## 4. 任务二：重测基线（换完 `.so` 之后，1 张卡约 2.5–3 h）

最小集：起服务 ×2 各 ~6 min｜`perf_baseline.sh` bs1 **交错 A,B,A,B 4 轮** ~40 min（含噪声地板）｜1 份 decode profile ~15 min｜serve.log 取新的 KV `available` 与池块 517,632｜GSM8K 50 题 ~30 min（口径校验，非精度验收）｜判读 20 min。
产出要盖构建指纹（`build_fingerprint.py`），**跨构建判读会拒绝给差值**（这是设计，不是故障）。

---

## 5. 任务三：优化队列（重测之后，按此优先级）

| 优先级 | 项 | 预计收益 | 前提 / 备注 |
|---|---|---|---|
| **1** | **L3：SWA `block_size` 32→128**（`VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE`，已实现默认关） | **容量 3.85×**（KV token 数、并发倍数）；TPOT ≤1 ms | **O6 已交付 ⇒ 现在能验证收益**（262144 起得来）。clean **没做** KV 回收（L4/L6 零命中），机制仍是我们的；《适配方案》§6.9.10 已点名解除 O6 后合入我们这个开关。⚠️ 绝对数要按新页大小重算；上卡判据必须**并发跑满让块回收真的发生**（不跑满的话"容量提升"和"分配器没压力"读数相同） |
| **2** | **t5：侧流能不能救回** | 若成功 **1.8–4.4 ms**（sglang 同版 torch_npu 实测 −5.37 ms） | 唯一可能翻盘的变量。验那条具名差异：`acl_graph.py:187` 的 `auto_dispatch_capture` 默认 False，sglang 传 True。**必须用真实层**（标量微型图给过假绿灯）；`--arm off` 报 107025 是阳性对照——**没有它，"成功"可能只是参数没生效**；捕获成功 ≠ 会重放，要 replay 逐位 + 逐层计数 |
| **3** | **t4 → 形态 C** | 0.9–2.4 ms（窗口 45–110 µs/层 − 回调成本） | t4 先测三个数（node_us / window_us / ceiling），**node ≥ window 就当场作废**。形态 C = engram 已交付的形态（同一捕获流上两个 host-func 节点），**必须建在 packed 的每层自有缓冲之上**（kt 的 `layer_idx%2` 槽位会让 L 与 L+2 撞车，今天没炸只是被那次整卡同步救了） |
| **4** | **A/A 对照**（可与上面任一发搭车） | 不给收益，**校仪器** | 两档（并发 8 + 并发 1）定"服务层逐位判据有没有区分力"，并把两臂之差当**噪声地板** |
| **5** | **频率放置（M0c）** | 0.9–3.1 ms（若偏斜成立） | 需多卡一发 eager 采频率表（代码与判据就绪，卡由 flash_dev 排）。偏斜倍数只有 sglang DSV4 的旁证，DSV4.1 未实测 |
| 6 | 吞吐轴（`MAXSEQS` 8→32） | 吞吐 ×1.3–1.8（推断） | **硬前置是 L3**（现在 KV 只够 5.22 条满长请求）。若主指标改成"一张卡的吞吐/成本"，这条应上调 |
| ⛔ | packed_transfer | 在能区分的设计下未测到收益 | 默认关；长尾（>3 ms 等待段 6→125）当**缺陷记录**，而底噪里本来就有与 packed 无关的 CPU 长尾 |
| ⛔ | 跨层 / deferred | — | 用户禁止 |
| ⛔ | 热专家动态缓存 / 权重再压缩 / 只读部分专家 / 激活降精度 | ≈0 | HBM 已顶格、无部分读接口、MXFP4+compact 已是下限、搬运只占 0.5% |

---

## 6. 交接时必须知道的坑（详见 design 的 k.* 节）

- **判据要读它所判断的那个对象**（踩过 3 次）：护栏读"当前有捕获在进行"≠"这条流的工作会进图"；探针用标量微型图（替身比被测对象弱）⇒ 在真实层全失败的平台上通过；指纹盯 `build_artifacts/vendor92` 而真正加载的是本树的包。
- **占卡判据三维**：HBM 高 + aicore 非零 = 有人在跑（等/协调，别 kill）；HBM 高 + aicore 恒 0 = 拆解残留（等它落）。**只看 HBM 分不开这两者，而该做的事相反**；连采 ≥3 次；读不到当忙。
- **租约跨容器不可见** ⇒ acquire 成功 ≠ 卡可用（原因不是"上下文没回收"）。
- **"一个永不等待的等待和一个成功的等待，输出完全一样"**：小写 `process id` 让拆除等待循环从没生效过（npu-smi 印大写，空卡印 `No process in device.`）。
- **profiler 开法**：本版 vLLM 不认 `VLLM_TORCH_PROFILER_DIR`，要用 `EXTRA_ARGS='--profiler-config {...}'`；`prof_decode_capture.py` 在 404 时**照样打印成功** ⇒ 采完自己数 `*_ascend_pt`；`prof_decode_breakdown.py` 会因"无图 Model ID"拒绝出结果 ⇒ 直接聚合 `kernel_details.csv` 并按**区间并集**算真实 device 忙。
- **未跟踪的产物会消失**（三个 run 目录已在 09-21 清理中永久丢失）⇒ 结论落地就把读数内联进文档，出处标「可复验 / 已记录不可再验」。
- **eval harness 默认 `temperature=1.0`** ⇒ 跑精度必须显式覆盖成贪心。
- **E8M0 scale 只能按字节搬**，不能 `.to()`；判据是字节往返，不是比 dtype 名。
- **kt 的任务异常要到下次 sync 才抛** ⇒ 报错记在下一层账上，定位要把层号往前推一格；kt `new Args{}` 无 delete，长跑像"跨层缓冲没释放"。

---

## 7. 对口

| 事 | 找谁 |
|---|---|
| 卡、窗口、共享脚本、闸与钉值 | `dsv4.1_flash_dev`（协调方） |
| clean 的代码/迁移/算子包细节 | `dsv41_vllm_pm`、`dsv41-integration` |
| kt-kernel 接口与重建 wheel | kt 侧 agent（本线自己的 M0 agent 线） |
| 官方代码比对（engram_offload / MegaMoE） | `dsv41_vllm_official_code_invest` |
| sglang 单卡参考实现 | `dsv4_a5_single_card`（另一台机器，Remote Control） |

---

## 8. 接手第一步建议

1. 读 `MOE_OFFLOAD_PLAN.md` §0–§2 与 §4.9–§4.17（约束、作废的数字、仪表纪律），再读本文。
2. 打快照分支 → 按 §3 做 rebase 与五项跟改 → 跑三套测试（单测 + 两个 selftest），**位置要对**。
3. 找 `dsv4.1_flash_dev` 谈换 `.so` + 算子包 + `ds41_site` 的钉值与指纹覆盖。
4. 拿到新 `.so` 后：t0 sanity（44 s）→ 重测基线（§4）→ 按 §5 的优先级往下做。
5. **不要**在换 `.so` 之前跑任何结论性的卡上发次。
