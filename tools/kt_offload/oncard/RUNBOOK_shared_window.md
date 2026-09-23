# RUNBOOK — §4.31：共享专家 MLP 挪进「主流等 CPU」的窗口

> 开关：`VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW=1`（**只认 env**，默认 `0` = 关；`true` 等别的值一律算关）。
> 代码：`vllm_ascend/kt_offload/shared_window.py`（开关 + 判据）、`ops/fused_moe/fused_moe.py:_forward_impl`
> （有共享专家那一支在 `ascend_shared_experts.forward(...)` **之后**调 `routed_experts.kt_merge`）、
> `ops/fused_moe/routed_experts.py`（`kt_shared_in_window` / `kt_merge` / `forward_impl(defer_kt_merge=)`）、
> `kt_offload/layer.py:_assert_merge_balance`（每次 route 恰好一次 merge 调用，否则下一次 route 就 raise）。
> 树：`staging/moe-offload-shared`（分支 `dsv41-shared-window`）。**本执行单只准备、没有跑过任何一发。**

## 0. 先写死的判据（看读数之前）

- **正确性 = 逐位 100%，且变异臂全红。** 共享专家读的是 MoE 层输入，不依赖 routed_out，也不依赖 CPU 那一半
  ⇒ 把 merge 挪到它后面**一位都不能变**。没有「差一点」这一档。
- 🔴 **漏 merge 是静默的**：CPU 那一半专家直接消失，张量形状对、数值看着也合理、不报错
  （§4.23 的「更快且算错」，而且特征更弱）。所以**判据永远不是 TPOT**。
- **性能只当信号。** t4p 代理 §4.30 已判 `unusable`（分辨率约 ±1 ms/token，P1a 的 −0.53 它都报「未分辨」）
  ⇒ **代理「没看出来」≠「没有收益」**，代理「看出来了」也不进收益账。收益账只收整网 A/B，门槛按 §4.27
  （|A−B| ≥ 3× 同臂地板 且 ≥ 0.3 ms/token）。
- 预期（来自离线 trace 对账，不是本执行单的读数）：共享专家 38.25 µs/层 × 40 层 ≈ 1.5 ms/token 上限；
  窗口 77 + 38 = 115 µs，远小于副流链 ~370 µs ⇒ 不会产生新的关键路径。

## 0.5 判别力前置闸（2026-09-23 加，三发被拦之后）

🔴 **替身把判据毁掉过一次，所以判别力现在是硬闸，不是打印。** 2026-09-23 的三发在跑到任何臂之前
就停了：替身 `silu(gate)*up` 是二次型（增益 O(|x|)），t5r 把 8 层串成残差栈且每层复用同一个 `shared_fn`
⇒ 幅值每层平方一次（18.8 → 6.4e2 → 4.0e5 → 3.1e11 → 9.4e22 → NaN），第 6 层起全 NaN；`bit_identical`
对 NaN 感知 ⇒ 两个参照步「逐位相同」⇒ 判别性自检开火。**减层数不是绕路**：幅值到 1e5~1e22 时，merge 的
O(|h|) 贡献低于该层输出的 1 个 bf16 ULP，**漏 merge 会被读成逐位相同 = 绿**（同一 CPU 复现里第 5 层起
确实如此）。

修法（`npu_moe._normalized`）：替身做成**齐次、非扩张**——输入先归一到 RMS 1（生产里共享专家读的本来
就是 post-attention-norm 之后的 hidden），权重 1/sqrt(fan_in)，输出按声明增益放回：

```
shared(x) = SHARED_GAIN * rms(x) * core(x/rms(x)) / rms(core(x/rms(x)))      # SHARED_GAIN = 0.5
```

于是 (a) `rms(shared(x)) == 0.5*rms(x)`，每层增长 ≤ 1+0.5+merge 份额而不是平方；(b) `shared(cx)=c·shared(x)`
⇒ `rms(merge)/rms(h)` 层间不变，merge 永远浮在 ULP 地板之上；(c) 保证来自 wrapper 而不是 core
⇒ W8A8 链和 bf16 链同样成立。

两道闸，**不满足就 raise、不跑**（判据常数：`npu_moe.MERGE_ULP_MARGIN = 8` 个 bf16 ULP，`BF16_EPS = 2^-8`）：

| 闸 | 在哪 | 查什么 |
|---|---|---|
| 1/2 主机复算（不占卡） | `npu_moe.shared_stack_cpu_check_default`，t5r 起手、`init_runtime` 之前 | 按声明的 merge 份额（0.25×rms(h)）复算 L 层 × S 步的臂算术，要求：幅值有界无 NaN/Inf、每层增长在 [1/4, 4]、**no-merge 变异在每一层每一步都逐位不同且 ≥ 8 bf16 ULP**、两个参照步不同 |
| 2/2 卡上实测 | t5r 参照步内（`merged - pre` 就是 CPU 那一半） | 用**卡上真实数字**而不是声明值：每层每步 `rms(merged-pre)/rms(h_next) ≥ 8 ULP`，且 `max|h| < 1e4` |

单测（`selftest_cpu.py`，`bash tools/kt_offload/oncard/selftest_cpu.sh`）证明这道闸**会红**：旧的二次型替身红、
纯放大（增益 40/层）替身红、merge 份额压到 2^-12 红，而修好的替身绿。

算子链：`--shared-chain auto` 现在**先试生产那条 MXFP8 链**（`shared_experts.py` 的 `QuantType.W4A8MXFP` 分支，
~367-398）：`npu_dynamic_mx_quant(x, e4m3)` → gate_up（W4A8_MXFP 线性：`npu_quant_matmul`，`scale_dtype=e8m0fnu`、
`pertoken_scale_dtype=e8m0fnu`、`x2_dtype=float4_e2m1fn_x2`、`group_sizes=[0,0,32]`，权重按
`w4a8_mxfp4.AscendW4A8MXFPDynamicLinearMethod.process_weights_after_loading` 的 format_cast/转置/scale 重排预处理）
→ `npu_swiglu_group_quant(dst_type=e4m3, quant_mode=2, clamp_value=swiglu_limit)` → down（同款线性）。
权重随机，算子序列与参数形态是服务的。救不回才退到归一化 bf16（那时读数里必须写明「非量化替身」）。

W8A8 **不在 auto 里**（生产的共享专家不走它），只能 `--shared-chain w8a8` 显式选。它在本构建上还有一道坎：
`DequantSwigluQuant` 的 tiling（CANN 9.1.0 built-in `libophost_nn.so`）要求 **`clamp_limit` 严格 > 0**
（`clamp_limit should be positive finite`），而本仓 `csrc` 只要求 `>= 0`、proto 头还写着「0.0 = 关闭 clamp」——
两边不一致，传 0.0 必挂 EZ0026；探针按 `(10.0, 7.0, 1.0)` 依次试。
🔴 **顺带的生产隐患（本执行单不改）**：`vllm_ascend/ops/fused_moe/shared_experts.py:86` 把 `swiglu_limit=None`
映射成 `0.0`，再在 `:351`（W8A8/W4A8 分支）无条件以 `swiglu_mode=1` 传下去 —— 在这台机器的 CANN 上是必挂路径，
值得单独修。

## 1. 前置（不占卡）

```bash
ROOT=/workspace/code/dsv4_single_card/vllm_project
NT=$ROOT/staging/moe-offload-shared            # 新树；不是 worktrees/moe-offload（那棵在跑合版本验收，只读）
OC=$NT/tools/kt_offload/oncard
# 1) 树必须干净：ONCARD_OUT_OF_TREE 对有未提交改动的树直接 rc=3
git -C $NT status --porcelain                  # 必须空
# 2) .so 在树里（gitignore 的，不进提交）；md5 必须是钉值
md5sum $NT/vllm_ascend/vllm_ascend_C*.so       # 85331298bdeffee02b7c42c5658030ac
#    缺了就从验收树【读】一份：cp -p $ROOT/worktrees/moe-offload/vllm_ascend/vllm_ascend_C*.so $NT/vllm_ascend/
ls $NT/vllm_ascend/_cann_ops_custom/vendors    # 算子包也要在（同样 gitignore）；🔴 缺了【不报错】，ds41 算子静默退回
#    缺了同样从验收树【读】一份：cp -a $ROOT/worktrees/moe-offload/vllm_ascend/_cann_ops_custom/vendors $NT/vllm_ascend/_cann_ops_custom/
#    干跑输出里必须是 `vendors(loaded)=2eae934d033b files=787`（与验收树 2026-09-23 干跑逐字相同）；`files=0` ⇒ 不许飞
# 3) 指向新树：run_oncard.sh 用 ONCARD_OUT_OF_TREE 把 PYTHONPATH 上的 vllm_ascend 换成新树
#    （vllm 树 DSV41_VLLM_TREE 仍是 worktrees/moe-offload-vllm，由脚本自己设，不用管）
export ONCARD_OUT_OF_TREE=$NT
export ONCARD_EXPECT_SO_MD5=85331298bdeffee02b7c42c5658030ac
CARD=0                                          # 由用户排；🔴 卡 3 永远不用，卡 2 在跑验收
# 4) 干跑：走完预检 + 指纹，停在拿租约之前（不碰卡）。确认日志里 tree= 是 $NT
ONCARD_DRY=1 ONCARD_RUNS_DIR=/tmp/oncard_dry bash $OC/run_oncard.sh $CARD t5r --shared-mlp both
```

2026-09-23 已在提交 `105f02459` 上干跑过一次（rc=0，不碰卡）：`tree=.../staging/moe-offload-shared`、
`.so md5=85331298…` ✅、`vendors(loaded)=2eae934d033b files=787`（第一次干跑是 `files=0` —— 新树建出来时没带算子包，
已从验收树拷入；这正是上面那条 🔴 要拦的情况）。

🔴 每发开头看 `🔴 OUT-OF-TREE run: vllm_ascend = .../staging/moe-offload-shared @ <提交>`；
没有这行 ⇒ 跑的是验收树的代码，这一发作废。

## 2. t5r：正确性（每发约 5–7 分钟；共 3 发 ≈ 20 min）

`--shared-mlp both` 给每个基础臂各跑两次：`#shared-after`（今天的顺序：merge → 共享 MLP）和
`#shared-before`（开关开：共享 MLP 在 fork 与 join 之间 → merge）。共享 MLP 是替身：服务的形状
（5120 → 2×2304 → 5120）、随机权重、算子链由 `npu_moe.build_shared_mlp` 探测本构建后**打印出来**
（优先 W8A8 量化 matmul 链，不行才退 bf16）。正确性不依赖用哪条链（两臂跑同一条），计时依赖。

```bash
# ① 干净臂 + 新增「漏 merge」变异臂（必须红，且必须是【逐位】红）
bash $OC/run_oncard.sh $CARD t5r --arms serial,fork,p1a --shared-mlp both --mutant no-merge  --layers 8 --steps 8
# ② 原阳性对照 + no-wait 变异臂（沿用 P1a 的口径；在 shared-before 的顺序里跑）
bash $OC/run_oncard.sh $CARD t5r --arms serial,fork,p1a --shared-mlp both --mutant no-wait   --layers 8 --steps 8 --stall-ms 50
# ③ drop-join（t5r 的默认变异臂）
bash $OC/run_oncard.sh $CARD t5r --arms serial,fork     --shared-mlp both --mutant drop-join --layers 8 --steps 8
```

判读（三发都要）：

| 项 | 要求 |
|---|---|
| 每个干净臂 × {after, before} | **8/8 逐位等于 eager 串行参照** |
| after vs before 直接比 | `arm X: the merge AFTER ... and BEFORE ... produced the same bytes` 每个基础臂一条 ✅（`step_hashes` 逐步相同） |
| `[cpu-check] ...` 两行 | 主机闸过了（幅值、增长、最小 ULP 余量）；没有这两行 ⇒ 这一发不是新工具跑的 |
| `eager serial reference: ... measured merge share X bf16 ULP at its weakest` | 卡上闸过了，记下 X（余量 ≥ 8） |
| `[shared-mlp] chain ...` | 记下是哪条链（含 `clamp_limit`）；是 `bf16` 就在读数里写明「非量化替身」 |
| `--mutant no-merge` | **RED**，并且 `mutant no-merge is red BY ITS BITS (captured=True, bit_identical=False)` ✅（在**串行**基础臂上跑：侧流开时漏 merge 也就漏了 join，捕获会先死于 107025 —— 那是 drop-join 的事，红得不是这个原因）；`guard_raised = 8/8`（生产守卫 `_assert_merge_balance` 每层都看见了） |
| `--mutant no-wait` | RED，`nodes_skipped.wait = 8` |
| `--mutant drop-join` | RED |
| 阳性对照 `--stall-ms 50` | 干净 p1a 臂 8 次 replay ≥ 0.7×8×8×50 ms = **2.24 s**；同发 fork 臂 ~0.03 s。🔴 差不出来 ⇒ 本发所有「逐位相同」不算证据 |

rc：Verdict 要求每发至少一个变异臂；干净臂全绿、变异臂全红时 rc=0。任何一条不满足 ⇒ **停**，不往下做。

## 3. t4p：性能信号（约 4–6 分钟，1 发；只作信号）

```bash
bash $OC/run_oncard.sh $CARD t4p --arms serial,fork,fork2 --shared-mlp --layers 8 --reps 24
```

`--shared-mlp` 自动加两臂 `fork_sa`（侧流 + 共享 MLP 在 merge 后）/ `fork_sb`（在 merge 前）和一个对照
`g_npu_shared`（NPU 链 + 共享 MLP、无 kt）。读这一行：

```
§4.31 shared MLP (<chain>): costs +X us/layer on its own (g_npu_shared - g_npu_only ...); fork_sb - fork_sa = −Y us/layer ...
```

- 理想是 **−Y ≈ −X**（MLP 整个藏进窗口）。机制 `shared_window` 也会出现在「预测（代理）」表里（未标定）。
- 对照红（negative / positive / host-ahead < 90%）⇒ 本发不读。
- 🔴 **`NOT resolved` 或 Y 接近 0 都不能读成「没收益」**（§4.30：代理对 P1a 的 −0.53 就是这么报的）；
  它也可能与 X 不同，因为替身不是服务的那条 MXFP 链。
- 能做到每层时间差，所以做了；做不到的部分：代理量不到服务真实的 DynamicMxQuant/SwigluGroupQuant 链。

## 4. 整网 A/B（**本执行单没有准备**，t5r 全绿后再排）

只有整网读数进收益账。现成的 `sidestream/run_arm.sh` **写死起 `worktrees/moe-offload` 的服务**
（`WT=moe-offload`、`build_fingerprint.py` 路径），不能直接指向新树 ⇒ 需要先给它加树外支持，或者等本改动
合进验收树后再跑。口径照 §4.28（P1a）：两臂**都开侧流**（和当时生产态一致的 P1a 设置），只差
`VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW` 一个 env（cfg json 两臂必须相同）。必检：
`grep -c 'SHARED IN WINDOW for layer' serve.log` 开臂 = 40、关臂 = 0；logprob 逐位 100%（skip 按同臂地板）；
perf ≥4 轮交错，按 §4.27 判。粗估 ≈ 1–1.5 h（两臂起服务 + logprob + perf）。

## 5. 仍存在的风险

- 守卫在**该层的下一次 route**才检查（merge 次数 ≠ 1 就 raise）。图模式下 replay 不跑 Python，route/merge 只在
  捕获和 eager 步里跑 ⇒ 漏 merge 在该层下一次捕获（下一个 capture size）或下一个 eager（prefill）步被抓到，
  不是当场。侧流开时另有一道：漏 merge 就漏 join，捕获当场死于 107025 / `_claim_side_buffers`。
- 在 route 与 merge 之间抛出的异常（例如捕获失败后框架重试）会让下一次 route 以「0 次 merge」raise
  —— 那是更响的失败，不是静默错，但会把原始异常的位置让给这条断言。
- 共享 MLP 挪进窗口后若 kt 反而比窗口短（kt 很快的层），节省会小于 38 µs/层；预计收益是上限。
