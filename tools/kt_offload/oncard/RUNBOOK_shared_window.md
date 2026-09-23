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
ls $NT/vllm_ascend/_cann_ops_custom/vendors    # 算子包也要在（同样 gitignore）
# 3) 指向新树：run_oncard.sh 用 ONCARD_OUT_OF_TREE 把 PYTHONPATH 上的 vllm_ascend 换成新树
#    （vllm 树 DSV41_VLLM_TREE 仍是 worktrees/moe-offload-vllm，由脚本自己设，不用管）
export ONCARD_OUT_OF_TREE=$NT
export ONCARD_EXPECT_SO_MD5=85331298bdeffee02b7c42c5658030ac
CARD=0                                          # 由用户排；🔴 卡 3 永远不用，卡 2 在跑验收
# 4) 干跑：走完预检 + 指纹，停在拿租约之前（不碰卡）。确认日志里 tree= 是 $NT
ONCARD_DRY=1 ONCARD_RUNS_DIR=/tmp/oncard_dry bash $OC/run_oncard.sh $CARD t5r --shared-mlp both
```

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
| `[shared-mlp] chain ...` | 记下是哪条链；是 `bf16` 就在读数里写明「非量化替身」 |
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
