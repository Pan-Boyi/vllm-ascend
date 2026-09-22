# 上卡执行单：侧流重叠 A/B（§4.18.1 ③；上限 ≈ 窗口 83.2 µs/层 ≈ 3.3 ms/token，推断）

**本单只回答两个问题**：
1. 侧流在**真服务**里能不能起来并且**算得对**（判据是 §3 的三层**逐位**判据，
   不是 GSM8K 分数，更不是 TPOT）；
2. 它到底省不省时间 —— 交错 A/B，差值必须大于噪声地板 **≈1 ms** 才算数。

🔴 **每一种已知的侧流失效都是「更快而且错」**（捕获不进图 ⇒ 每次 replay 复用捕获那一次的 CPU 输出；
join 丢了 ⇒ H2D 读到上一步的 pinned 输出）。所以**任何"TPOT 变好了"单独都不是证据**。

## 结论（2026-09-22 首发，整发判负）

🔴 **侧流在整网服务里算错了 —— 而且更快。不要引用 B 臂的任何 TPOT。**
（run `oncard_runs/20260922_161813_sidestream`，未跟踪产物，读数已内联在这里；`.so` 85331298bdef）

| 判据 | A 臂（侧流关） | B 臂（侧流开） |
|---|---|---|
| ③ 尺子地板（**跨进程** A1.lp vs A2.lp，并发 1，弃前 10 条） | **190/190 prompt、1128 个 decode 位、6767 个 top-5 值逐位全同（非零差 0）** | — |
| ② 整网 logprob 逐位（A1 vs B1，同批 prompt） | — | **decode 位只有 622/690 逐位相同（90.1%）；选出的 token 有 27 处不同；max \|Δlogprob\| 8.107** |
| 退化步（top-5 全等于 ln(1/129280)= −11.7697，即 logits 全相等） | 1398 个位置里 **0** 个（两发） | 998 个位置里 **31** 个（3.1%） |
| prefill 位（eager，侧流不参与）| 逐位相同 | 逐位相同（内部对照成立 ⇒ 差异只来自走图的 decode 步） |
| GSM8K 50（贪心，EV_THINKING=0） | **50/50 acc=1.0000** | **46/50 acc=0.9200** |
| bs1 TPOT 中位（3 轮） | 35.15 ms（34.75/35.29/35.41） | **32.41 ms（32.12/32.55/32.57）** |

🔴 **2026-09-22 更正：③ 那一行的「弃前 10 条」紧了 2 条，而且截断点必须按【臂】校准。** 同日重测
（`.so` 不变，四个进程）：**串行臂自己**的跨进程地板在请求 **2,3,4,6,7,8,9** 上还不一样，侧流臂在
**7,10,11** 上还不一样 —— 也就是说 A 臂的预热尾巴到第 9 条、B 臂到第 11 条。
**用 A 臂标定的截断点套到 B 臂，会把 B 更长的预热尾巴读成「侧流算错了」。**
⇒ 口径改为 **`--skip-first=12`**（`logprob_cmp.py` 的默认值已改成 12）。从第 12 条起，
两个地板 + 四个 A/B 交叉一共六对**全部 100% 逐位相同**（1149/1149 decode 位、非零差 0）。
⚠ 换 prompt 批或换配置就要**重新量各臂自己的地板**再定截断点，不许拿结果反推截断点 ——
确认发的做法是：先把判据（`--skip-first=12`、100%、退化步 0、token 变化 0）写进 run 目录的
`EXPECTED.txt`，再换种子生成新批（`make_prompts.py <out> <seed>`）重跑。

⇒ B 臂"快 2.7 ms"正是执行单开头那句话的实例：**每一种已知的侧流失效都是「更快而且错」**。
按 §5 判读表，② 有一条不逐位相同即整发判负 ⇒ **TPOT 不许引用，1319 不跑，埋点那一发不跑**。

**失效的形状**（供下一步定位，不是结论）：两种症状并存 —— 35 条分歧 prompt 里 26 条的首个分歧位是
**退化步**（该步 logits 全相等 ⇒ 那一步的隐藏态整体坏掉），另 9 条是**普通数值漂移**（token 相同、
logprob 值不同）。有一条请求还因此被服务端判 400（同一条重放时复现退化步）。
🔴 **单层 t5 逐位臂 9/9 全绿**（见 §3①）**却挡不住这个**：t5 一次只建 3 层一张图、每次 replay 前后
都 synchronize，而服务里是 40 层、连续 decode 步、kt 的 pinned 槽按 `layer_idx%2` 跨层复用
（A6：L 与 L+2 共用）。⇒ 下一步该查的是**跨步/跨层的槽复用**（上一步的 H2D 还没读完，下一步就
开始写同一个 pinned 槽），而不是本层的 record/join —— 本层的 join 若丢，ACL 在捕获期就直接 107025 了。

## 0. 这一发之前发生了什么（读数的前提）

- 侧流曾被"平台不支持（ACL 107025）"判死；2026-09-22 **作废**：那发 t5 是被我方**空的捕获缓冲**
  （`no preallocated int64 id buffer`）绊倒，107025 是次生。真触发条件是**副流上最后一次 event record
  没有被主流 join**（t5 `off_evt` 崩 / `off_evtw` 过）。生产 `_join_side_stream` 一直是 record+join。
- 缓冲 bug 已修（本单对应的提交）：`_CAPTURE_BUFFERS_READY` 只再管 kt 那半边（进程级、类变量），
  `self._ids_i64` **每个层实例都建**。CPU 单测 `tests/ut/kt_offload/test_capture_buffers.py`。
- 开关**默认仍关**：`kt_offload_config.overlap_side_stream=true` **且**
  `VLLM_ASCEND_DSV41_KT_SIDE_STREAM=1` 才放行；放行时每层打一条 `SIDE STREAM ENABLED` WARNING。

### 只能在卡上验的三件事（代码里标了"零覆盖"）

| # | 判据 | CPU 上为什么做不了 |
|---|---|---|
| Z1 | ACL 捕获真的跟着 `wait_event` 走到副流上 | 仿真器里是我们自己规定它会 |
| Z2 | **107025 阳性对照**：副流尾 record 无人 join ⇒ capture_end 必崩 | 只有真 ACL 会报这个码 |
| Z3 | 真 TPOT / NOTIFY_WAIT / A+F 的变化 | 时间不在 CPU 模型里 |

CPU 上已有的（**不要在卡上重做**）：join/record 配对、捕获期 host-func 计数为 0、每 replay 每层 +1、
A6 缓冲归属与「同槽两层不得同时在飞」、四个变异臂（丢 join / join 提前 / event 被回收 / 捕获没传播）
全部更快且错。

## 1. 起飞前

```bash
cd /workspace/code/dsv4_single_card/vllm_project
export TMPDIR=/dev/shm/dsv41_tmp TMP=/dev/shm/dsv41_tmp
export SO=85331298bdeffee02b7c42c5658030ac     # 本树（worktrees/moe-offload）期望的 .so
export CARD=?          # 0–2 里起飞当时实测空闲的那张；🔴 别用卡 3
export PORT=8041
export OWNER=dsv41_vllm_singlecard
export WT=$PWD/worktrees/moe-offload/tools/kt_offload/oncard
```

**卡空不空按三维读**（`RUNBOOK_window.md` §0 那张表，照抄不再重复）：HBM + aicore **连采 ≥3 次**；
`HBM 高 + aicore 恒 0 = 拆解残留`（等，别 kill、别 reset）。租约表跨容器不可见 ⇒ acquire 成功 ≠ 卡上没人。

## 2. 先做 Z2：107025 的阳性对照（不起服务，t5，约 2 分钟）

**先证明仪器认得出真失效，再去读 A/B。** 顺序不能反：若 `off_evt` 这一臂没崩，
说明这张卡/这个构建上"副流尾 record 无人 join"不再触发 107025 ⇒ **本单的探针失效，停下来重判**，
不要接着往下跑 A/B。

```bash
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 --arm off_evt   # 期望：capture_end 107025
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 --arm off_evtw  # 期望：捕获成功、replay 判据全过
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 --arm off       # 现状参数，期望全过
```
判读（照 §4.19 ② 那张表）：`off_evt` 崩 **且** `off_evtw`/`off` 过 ⇒ 探针有效，继续。
三臂都过或都崩 ⇒ 探针无区分力，**本单到此为止**。

**2026-09-22 首发实测（卡 0，`.so` 85331298bdef）**：三臂都如期
（`oncard_runs/20260922_160255_t5` off_evt → 107025「In the cross-stream capture scenario, the last
task on the sub stream lacks the corresponding event wait task」；`_160358_t5` off_evtw 与
`_160436_t5` off 各 12/12 replay 判据全过）。两条仪器修正一并落地：
- t5 旧判据把**所有** `off*` 臂当"必须崩"的阳性对照（来自已作废的"平台不支持"假设）⇒ 干净臂
  永远判不了 PASS。现在只有 `off_evt` 必须红，`off`/`off_evtw` 必须 captured（提交 b35a700ca）。
- 🔴 **107025 不在 `10701[0-9]` 里**。run_oncard 的 "10701x lines" 计数和下面 §4 那条
  `grep -c '10701[0-9]'` 都**漏掉 107025**；要查侧流的错码请用 `grep -cE '1070[12][0-9]'`。

## 3. 精度判据：三层，跑的顺序是 ③ → ② → ①（用户 2026-09-22 定）

③ 与 ② 要有服务，**跑在第 4 步那几臂的服务里**（③ 用 A1 的服务，② 用交错的 A/B 两臂）；
① 不占整网、可与之并行安排。本节先写判据，命令里的 `$D`/`$PORT` 就是第 4 步那套。

侧流是**纯调度改动**（同样的 kt 计算，换了一条流和一对 event）⇒ 判据做成**逐位相同**，
不是"分数差不多"。GSM8K 只当最后的冒烟（第 4 步里那一发），1319 全量留到最终形态（M4）。
🔴 服务层**文本**逐位判据已证无区分力（§4.18：同进程只换批组成，c8 vs c8b 文本逐位相同仅 19/50）
⇒ 整网这一层要比的是 **logprob 的浮点值**，不是文本。

先跑基线逐位臂做背景（两发都必须 PASS；它们不走侧流，eager 永不 fork）：
```bash
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t1   # eager offload，逐位
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t2   # 图模式，逐位
```

### ③ 先建仪器地板（不做这个，② 不可读）

同一臂（先用 A＝侧流关）、**并发 1**、同一批 prompt **跑两次**，比 logprob 浮点值是否逐位相同。

- 两次逐位相同 ⇒ 这把尺子有区分力，② 照常做。
- 两次就不同 ⇒ 🔴 **② 改成分布比较**（逐位判据在这台机器上不成立），并在报告里写明
  "只能排除大偏差，不能证明未改数值"，同时给出观察到的最大 |Δlogprob| 与分位数。
- 背景：两发**并发 1** 互比从来没做过（§4.18 里用户为省时取消），所以这一步不是形式。

### ② 整网强制单步 + logprob 逐位比（主判据）

固定一批 prompt **≈200 条**（覆盖长短：短 ≈32 tok、中 ≈512、长 ≈4000，各约 1/3；从 GSM8K +
LongBench 片段里取，**批固定存成 `$RUN/prompts.jsonl`，两臂读同一个文件**）。
🔴 **`max_tokens=1` 在本配置下对侧流没有区分力（2026-09-22 发现）**：侧流只在**捕获路径**上
fork（`eager never forks`），而本配置 `cudagraph_capture_sizes=[1,2,4,8]`（serve.log 自证）
⇒ 任何超过 8 token 的 prefill 都走 eager，`max_tokens=1` 拿到的那个 logprob **只来自 prefill**，
两臂必然相同，比了也说明不了问题。改成 **`max_tokens=8`**：位置 0（prefill、eager）当**内部对照**
（它必须两臂相同，否则是别的东西变了），位置 1.. 是**并发 1 的 decode 步**（FULL 图 ⇒ 捕获路径
⇒ 侧流真的在跑），逐位比就比这些位置；某条在位置 k 选出的 token 就不同 ⇒ 这条从 k 起不再可比
（k 本身要报出来，它就是"侧流改了数值"的直接证据）。工具：`sidestream/logprob_run.py`
（并发 1、贪心、top-5）与 `sidestream/logprob_cmp.py`（逐位 + 地板分位数），
prompt 批由 `sidestream/make_prompts.py` 定死（200 条：67 短 / 67 中 / 66 长，长的 < 4096 token
⇒ 不跨 chunk）。每条 **`max_tokens=8`**（decode 步够 7 个 ⇒ 不被自回归放大），请求 `logprobs`，
**top-k 取 5**（够分辨又不至于让响应体爆掉；k 要在报告里写清楚，两臂必须同 k）。
**并发固定 1**，两臂（A=关 / B=开）**同时段交错**，比每条每个候选 token 的 logprob 浮点值逐位相同。

⚠ **先确认这个配置支不支持返回 logprobs**（有的配置默认关）：起服务后先探一发
```bash
curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","prompt":"1+1=","max_tokens":1,"temperature":0,"logprobs":5}' | head -c 600
```
- 返回里有 `logprobs.top_logprobs` ⇒ 照上面做。
- 报 `logprobs is disabled` / `max_logprobs` 之类 ⇒ 起服务时加 `EXTRA_ARGS="--max-logprobs 5"`
  （加了就**两臂都加**，且这发与第 4 步的 TPOT 读数不可混用）。
- 仍不行 ⇒ 退路按优先级：(a) `prompt_logprobs`（chat/completions 皆可，评的是 prefill，
  对侧流同样有效，因为 MoE offload 在 prefill 也走这条路）；(b) `echo=true` + `logprobs`；
  (c) 都不行就**报告替代方案而不是降判据**，并把 ① 提为主判据。
- 判读：逐位相同 ⇒ 侧流未改数值（在 ③ 证明尺子有区分力的前提下）；
  任何一条不同 ⇒ 🔴 **侧流改了数值**，先查 host-func 计数与 join，不许拿 GSM8K 分数把它盖过去。

### ① 单层逐位臂（最强、最便宜；t5 已扩好，**与整网可并行安排**）

真实层、图捕获 + replay，侧流开的输出**逐位等于**串行臂。`--layer` 与 `--tokens` 现在都收列表，
一发扫完 3 个位置 × 3 个 token 数（每个 scenario 自带自己的串行参照，不会互相借）：

```bash
# 干净臂：9 个 scenario（层 0-2 / 20-22 / 37-39，T=1/4/32；每臂 3 个不同输入）
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 \
  --arm off --layer 0,20,39 --tokens 1,4,32 --inputs 3
# 🔴 变异臂（必须红）：主流不 wait 副流的 join
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 \
  --arm off --layer 0,20,39 --tokens 1,4,32 --inputs 3 --mutant drop-join
```
- `--layer 39` 会被**下移**成 37,38,39（3 层窗口要落在 40 层以内），输出里会明说 `shifted`；
  每个窗口都同时覆盖 kt 的两种槽位 `layer_idx%2`（A6：L 与 L+2 共用 pinned 槽）。
- 干净臂判据（t5 原有的，全部保留）：捕获期 host-func **0**、每 replay 每层 **+1**、
  每个 replay **逐位等于**串行臂、replay 两两不同、replay k ≠ 串行 k−1（陈旧输出指纹）。
- 变异臂判据：**没有任何一个臂是"captured 且逐位相同"**（`must_be_red`）。它有两种红法，都接受：
  捕获直接失败（107025：副流尾 record 无人 join），或捕获成功但输出不同。
  🔴 若变异臂和干净臂一样过 ⇒ 逐位判据在这发里**没有区分力**（多半是副流上根本没东西在跑），
  t5 会拒绝把干净臂报成 PASS ——**这一条就是"逐位相同不等于真的并行"的挡板**。
  与第 2 步 Z2 的区别：Z2 在层的工作之外**额外**补一个 record（问的是 ACL 的规则），
  ① 的变异臂动的是**本层自己的 join**（问的是我们的正确性）。
- 🔴 **一发跑 9 个 scenario 现在会在第 2 个 scenario 崩**（2026-09-22 首发实测，
  `oncard_runs/20260922_161358_t5`）：上一个 scenario 捕过图的 forked 栈还活着（多半被图里的
  host-func 回调持有），下一个 scenario 建同号层时撞上生产的"一个 layer_idx 只许一个活 wrapper"
  硬拦截（`layer.py:_register_wrapper`）。**不要**用 `_reset_wrapper_registry()` 绕 —— 那正是
  kt 按层号记的 deferred 标志会串台的地方。**按 scenario 一发一进程**（本来就是最终结论的口径）：
  ```bash
  for L in 0 20 39; do for T in 1 4 32; do
    ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 --arm off --layer $L --tokens $T --inputs 3
    ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 --arm off --layer $L --tokens $T --inputs 3 --mutant drop-join
  done; done
  ```
  **2026-09-22 首发结果：9/9 干净臂 13/13 全绿（captured 且逐位等于串行臂），9/9 变异臂全红。**
  变异臂**每一个**都是"捕获直接失败（107025）"那种红法，没有一个是"捕获成功但输出不同"
  ⇒ 这证明的是 **ACL 不让缺 join 的图过**，**不**等于"逐位判据自己能抓住少 join 的竞态"。
- 干净臂单发的 rc 仍是 1：Verdict 要求每发至少一个变异臂，而干净臂发里没有。判读要
  **按配对发一起判**（干净发 required 13/13 failed 0 + 同参数的 drop-join 发 must_be_red 红）。
- CPU 侧的同形判据已在 `tests/ut/kt_offload/test_side_stream_overlap.py`（四个变异臂全部"更快且错"）
  与 `test_t5_scenarios.py`（矩阵/下移/拒绝）⇒ 卡上这一步加的是**真 ACL + 真 kt + 真权重**。

## 4. 交错 A/B（A=侧流关，B=侧流开，**同时段 A,B,A,B**）

🔴 配置逐项同 §4.18 基线（出处 `packed_ab/20260922_122648/armA1/cfg.json`）。
🔴 **跨发读数不得相减**；每臂自带重复轮次，A 与 B 交错以吃掉时段漂移。

```bash
RUN=$PWD/oncard_runs/$(date +%Y%m%d_%H%M%S)_sidestream; mkdir -p $RUN
python3 worktrees/moe-offload/tools/kt_offload/build_fingerprint.py stamp $RUN --label sidestream | tee $RUN/fp.txt
grep -q "$SO" $RUN/build_fingerprint.json || { echo "❌ .so 不是 $SO，停"; }

cat > $RUN/cfg_A.json <<'JSON'
{"kt_offload_config":{"enabled":true,"resident_strategy":"prefix","num_resident_experts":100,
 "cpuinfer_threads":96,"threadpool_count":2,"numa_nodes":[0,1],"packed_transfer":false,
 "overlap_side_stream":false}}
JSON
sed 's/"overlap_side_stream":false/"overlap_side_stream":true/' $RUN/cfg_A.json > $RUN/cfg_B.json
```
⚠ 两臂都必须 `packed_transfer:false`：**packed 与侧流同开现在起服务就 raise**
（packed 分支在 `route()` 里优先 ⇒ 侧流根本不会被用到，"开了却没生效"）。

一轮 = 起 A → 测 → 收 → 起 B → 测 → 收，共 **两轮**（A1,B1,A2,B2）。整套已脚本化：
```bash
bash $WT/sidestream/seq_ab.sh $RUN $CARD     # A1(probe,lp,lp2,perf,gsm) B1 A2 B2，任一臂非 0 就停
```
（A1 跑两发 logprob = §3③ 的**同进程**地板；A1 vs A2 是**跨进程**地板 —— A 与 B 本来就是两个进程，
所以判 ② 用的地板必须是跨进程那个。）下面是它每臂做的事，手跑时照抄：

```bash
ARM=A1   # A1 / B1 / A2 / B2；A* 用 cfg_A.json 且不设 SIDE_STREAM，B* 用 cfg_B.json 且 =1
CFG=$RUN/cfg_${ARM:0:1}.json; D=$RUN/$ARM; mkdir -p $D
( cd /workspace/code/dsv4_single_card/vllm_project && setsid env \
    WT=moe-offload DSV41_VLLM_TREE=$PWD/worktrees/moe-offload-vllm LEASE_OWNER=$OWNER \
    KT_OFFLOAD=1 NPUS=$CARD PORT=$PORT EAGER=0 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    KT_ADDITIONAL_CONFIG="$(cat $CFG)" \
    MAXLEN=8192 MAXSEQS=8 MAXBATCH=4096 \
    VLLM_ASCEND_DSV41_KT_SIDE_STREAM=$([ "${ARM:0:1}" = B ] && echo 1 || echo 0) \
    VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=30 \
    bash worktrees/moe-offload/tools/kt_offload/serve_v41_singlecard.sh ) > $D/serve.log 2>&1 &
SERVE_PID=$!; PGID=$(ps -o pgid= -p $SERVE_PID | tr -d ' '); echo "pgid=$PGID" | tee $D/pgid.txt
```

**等就绪**：轮询 `curl -s http://127.0.0.1:$PORT/v1/models`，超时 900 s。
失败判据**只认** `^Traceback|EngineCore failed|进程退出|超时` ——🔴 不要 grep 横幅里的 ❌ / 拒绝启动
（脚本横幅自己就印这些词，已误判过两次）。

**起来后三条必检**（B 臂多前两条；对不上 = 这一臂作废，不是"读数偏小"）：
```bash
grep -m1 'vllm_ascend = ' $D/serve.log            # 必须是 worktrees/moe-offload/vllm_ascend/__init__.py
grep -c 'SIDE STREAM ENABLED' $D/serve.log        # B 臂 = 40（每层一条）；A 臂 = 0
grep -cE '1070[12][0-9]' $D/serve.log             # 必须 0（🔴 107025 不在 10701x 里，见 §2）
```
🔴 **A 臂出现 `SIDE STREAM ENABLED`，或 B 臂一条都没有 ⇒ 两臂其实是同一臂，整发作废。**
另：`VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=30` 的周期报告在 B 臂必须是 **uniform、40 层每层都在长**；
出现 `layers never ran` ⇒ 那些层的 CPU 专家根本不在图里（静默不重放），**这一臂的 TPOT 不许引用**。

每臂两个 eval（GSM8K 50 在这里只是**冒烟**，数值判据在 §3）：
```bash
( cd /workspace/code/dsv4_single_card/vllm_project && PORT=$PORT LABEL=$ARM \
    CASES="bs1 1 16" REPEAT=3 PERF_OUT=$D/perf bash build_artifacts/perf_baseline.sh ) > $D/perf.log 2>&1
# 参数逐项抄 run_packed_ab.sh:211（A/B 与 A/A 的 GSM8K 只有那一处启动点）
( cd /workspace/code/dsv4_single_card/vllm_project && EV_THINKING=0 python3 \
    build_artifacts/eval_runners/gsm8k_eval_effort.py \
    --base-url "http://127.0.0.1:$PORT/v1" --model dsv41 --out-dir $D/gsm8k50 \
    --max-tokens 4096 --temperature 0 --top-p 1.0 --num-threads 8 --limit 50 ) > $D/gsm8k.log 2>&1
```
收摊（每臂都收干净再起下一臂）：
```bash
OWNED_CARDS=$CARD STOP_PGID=$PGID bash build_artifacts/stop_by_port.sh $PORT | tee $D/stop.log
```

## 5. 判读表

| 读数 | 结论 |
|---|---|
| B 臂起不来 / 有 10701x | 侧流在服务里不可用；把报错原文与 §4.19 ② 那张表对照，**不要**回到"平台不支持" |
| B 臂 host-func 报告非 uniform | 静默不重放 ⇒ B 臂数值不可信，TPOT 不许引用 |
| §3② logprob 有一条不逐位相同 | 🔴 侧流改了数值 ⇒ 整发判负，TPOT 不许引用 |
| GSM8K B < A 明显 | 数值坏了（但**分数相同不代表没坏**：GSM8K 在这里只是冒烟，判数值靠 §3） |
| \|median(B) − median(A)\| < **1 ms** | **判不可分辨**（不是"没有收益"，也不是"有收益"） |
| B 比 A 快 >1 ms 且上面全绿 | 记为收益上限内的实测；与 t4 推断上限 3.3 ms/token 对表 |
| B 比 A 快 >3.3 ms/token | 🔴 超过窗口上限 ⇒ 多半是少算了东西（查 host-func 计数），**先别高兴** |

TPOT 取每臂 `perf/*.json` 的 **中位**；A1 vs A2 的差就是本时段的漂移，写进报告当地板
（§4.18 初值 0.24 ms 中位差 / 1.0 ms 极差）。

## 6. 第三发：埋点 + profiler（只在第 5 步判出"可分辨"之后做）

看侧流把 §4.20 的 **A+F_dev 78 µs/层（3.83 ms/token）** 挪走了多少、NOTIFY_WAIT 怎么变。

```bash
# 在 B 臂的起服务命令上再加：
#   KT_MOE_PHASE_TIMING=1  VLLM_ASCEND_DSV41_KT_SEAM_TIMING=5
#   EXTRA_ARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFDIR\"}"
```
⚠ **采 profile 的服务一次性用**：`/stop_profile` 时 EngineCore 会 **SIGSEGV**（产物完好）。
`stop_by_port.sh` 必须带 `STOP_PGID`。离线解析照 `RUNBOOK_seam.md` §6，
并注意那里记的两条仪器问题（中位不可加 / `--nw-csv` 错位两格）。
🔴 埋点本身可能有 **0.4–0.8 ms** 真实开销（§4.20，未证明）⇒ 这一发的绝对值**不得**与第 4 步的 A/B 相减。

## 7. 收尾

```bash
bash build_artifacts/card_lease.sh release $CARD $OWNER
```
报告里每个数都标出处 run 目录与"可复验／已记录不可再验"；run 目录是未跟踪产物，随时会没（k.16）⇒
结论落地时**当场内联读数**。
