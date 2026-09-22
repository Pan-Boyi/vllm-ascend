# 上卡执行单：接缝拆解（§4.18.1 ①，107.5 µs/层）

**本单只回答一个问题**：§4.18 里 `NOTIFY_WAIT 353.6 µs/层 − kt 计算 246.1 µs/层 = 107.5 µs/层
（≈4.3 ms/token）`落在哪一段。埋点只读时钟，不改任何设备操作、不改一个输出字节
（开关：`VLLM_ASCEND_DSV41_KT_SEAM_TIMING`，默认空＝关）。

🔴 **配置必须与 §4.18 那发 decode profile 逐项相同**，否则读数不能和 353.6 / 246.1 相减。
🔴 **跨构建不可比**：run 目录必须盖构建指纹（下面第 1 步），指纹不同的两发不得相减。

## 分段定义（与 `vllm_ascend/kt_offload/seam_timing.py` 的文档字符串同源）

| 段 | 含义 | 怎么来的 |
|---|---|---|
| A | device 到达 host-func 节点 → 回调开始跑 | **算出来的**：`NOTIFY_WAIT − T_cb` 的一半（与 F_device 合并，分不开） |
| B `pre` | 回调进入 → Python 前置（`_assert_no_pending`）结束 | 直接测 |
| C `kt_pre`+`submit`+`between` | kt 的 Python 前奏（`_check_qlen_fits`/`get_buffer`/建 task）、`CPUInfer.submit`、submit 到 sync 之间 | 直接测（CPUInfer 探针） |
| D | kt 计算 | **别的仪器**：`KT_MOE_PHASE_TIMING=1` → `prof/ktphase.py` |
| E | worker 唤醒 + 归并/写回 pinned 输出 | **算出来的**：`sync − D` |
| F `post` | 回调后置 → 回调返回（host 部分）；通知回 device（device 部分）算在 A+F_dev 里 | host 部分直接测 |
| `cb` | T_cb：回调进入 → 退出 | 直接测 |
| `gap` | 同一层相邻两次回调进入的间隔 | 直接测；bs1 稳态下应 ≈ TPOT（device 每层周期的 host 视角旁证） |

**唯一的跨仪器等式**：`A + F_device = NOTIFY_WAIT(每层) − T_cb(每层)`。
⇒ profile 与 `[kt-seam]` 记录**必须来自同一次服务、同一批步**（本单把 profile 放在最后 8 s，
`[kt-seam]` 则整段服务都在记；解析时取 profile 窗口附近的采样记录）。
`prof/seam.py` 会把这条差值、以及 `B+C+D+E+F ≈ T_cb` 的对账残差都打出来；**残差不为 0 就先别读表**。

## 0. 起飞前

```bash
cd /workspace/code/dsv4_single_card/vllm_project
export TMPDIR=/dev/shm/dsv41_tmp TMP=/dev/shm/dsv41_tmp
export SO=85331298bdeffee02b7c42c5658030ac     # 本树（worktrees/moe-offload）期望的 .so
export CARD=?                                  # 起飞当时实测空闲的那张
export PORT=8041
export OWNER=dsv41_vllm_singlecard
RUN=$PWD/oncard_runs/$(date +%Y%m%d_%H%M%S)_seam; mkdir -p $RUN
```

**卡空不空按三维读**（`RUNBOOK_window.md` §0 那张表，照抄不再重复）：HBM + aicore **连采 ≥3 次**；
`HBM 高 + aicore 恒 0 = 拆解残留`（等，别 kill、别 reset）。租约表跨容器不可见 ⇒ acquire 成功 ≠ 卡上没人。

## 1. 构建指纹（先盖，后跑）

```bash
python3 worktrees/moe-offload/tools/kt_offload/build_fingerprint.py stamp $RUN --label seam | tee $RUN/fp.txt
grep -q "$SO" $RUN/build_fingerprint.json || { echo "❌ .so 不是 $SO，停"; }
```

## 2. 起服务（一次性；**profile 放最后**）

⚠ 已知：`/stop_profile` 之后 EngineCore **静默退出**（`Stop profiler while current state is RECORD`，无 traceback，
产物完好）⇒ 这次服务只用一次，采完 profile 就收。

```bash
# 租约不在这里拿：launcher 自己用 LEASE_PID=$$ acquire（末尾 exec，PID 不变）。
# 在这里手动 acquire 会被 card_lease.sh 拒绝（"acquire 必须显式给 LEASE_PID"），
# 给了 shell 的 PID 又会随 shell 退出失效 —— 2026-09-22 首发实测踩到，就地删掉。
PROFDIR=/dev/shm/dsv41_prof_seam; rm -rf $PROFDIR; mkdir -p $PROFDIR
cat > $RUN/cfg.json <<'JSON'
{"kt_offload_config":{"enabled":true,"resident_strategy":"prefix","num_resident_experts":100,
 "cpuinfer_threads":96,"threadpool_count":2,"numa_nodes":[0,1],"packed_transfer":false}}
JSON
# cfg 与 §4.18 基线发次逐项相同（cpuinfer_threads 96 / threadpool 2 / numa [0,1] / N=100 prefix / packed 关）；出处 packed_ab/20260922_122648/armA1/cfg.json
( cd /workspace/code/dsv4_single_card/vllm_project && setsid env \
    WT=moe-offload DSV41_VLLM_TREE=$PWD/worktrees/moe-offload-vllm LEASE_OWNER=$OWNER \
    KT_OFFLOAD=1 NPUS=$CARD PORT=$PORT EAGER=0 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    KT_ADDITIONAL_CONFIG="$(cat $RUN/cfg.json)" \
    MAXLEN=8192 MAXSEQS=8 MAXBATCH=4096 \
    KT_MOE_PHASE_TIMING=1 \
    VLLM_ASCEND_DSV41_KT_SEAM_TIMING=5 \
    VLLM_ASCEND_DSV41_KT_HOSTFUNC_LOG=30 \
    EXTRA_ARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFDIR\"}" \
    bash worktrees/moe-offload/tools/kt_offload/serve_v41_singlecard.sh ) > $RUN/serve.log 2>&1 &
SERVE_PID=$!; PGID=$(ps -o pgid= -p $SERVE_PID | tr -d ' '); echo "pgid=$PGID" | tee $RUN/pgid.txt
```

**等就绪**：轮询 `curl -s http://127.0.0.1:$PORT/v1/models`，超时 900 s。
失败判据**只认** `^Traceback|EngineCore failed|进程退出|超时`
——🔴 **不要 grep 横幅里的 ❌ / 拒绝启动 字样**（脚本横幅里就印着这些词，已经误判过两次）。
起来后先读真对象：`grep -m1 'vllm_ascend = ' $RUN/serve.log` 必须是
`.../worktrees/moe-offload/vllm_ascend/__init__.py`。
再确认埋点真的开了：`grep -m1 'seam timing on' $RUN/serve.log`（会印层数与采样间隔）。

## 3. 先量埋点开销（不开 profiler，n=3）

```bash
( cd /workspace/code/dsv4_single_card/vllm_project && PORT=$PORT LABEL=seam_on NOTE="seam timing on" \
    CASES="bs1 1 16" REPEAT=3 PERF_OUT=$RUN/perf_seam_on bash build_artifacts/perf_baseline.sh ) > $RUN/perf.log 2>&1
```
判读：对 §4.18 基线 **34.4–34.6 ms**，噪声地板 **≈1 ms（3%）**。
- 差 <≈1 ms ⇒ **判不可分辨**（不是"证明没开销"），照常往下做。
- 差 >1 ms ⇒ 埋点本身进了账，后面所有分段数都要带这个标注；先停下来想，别直接采 profile。
- CPU 单测里同一段埋点的开销是 **≈1.5 µs/回调**（见 `tests/ut/kt_offload/test_seam_timing.py` 最后一条，
  会打印实测值）⇒ 40 层 ≈ 0.06 ms/token，账面上应当落在噪声地板以下。**账面不是读数**，所以这一步还是要跑。

## 4. 采 8 s profile（放最后）

```bash
( cd /workspace/code/dsv4_single_card/vllm_project && python3 build_artifacts/eval_runners/prof_decode_capture.py \
    --port $PORT --out $RUN/cap --bs 1 --capture-s 8 ) > $RUN/capture.log 2>&1
sleep 5; ls -d $PROFDIR/*_ascend_pt   # 空 = profile 没写出来，capture 脚本照样印 "captured"
date -u +%FT%TZ | tee $RUN/profile_window_end.txt   # 解析时用它对齐 [kt-seam] 采样
```
采完记下时刻：`[kt-seam]` 是**整段服务**都在记的，profile 只有这 8 s ⇒
解析时只取该窗口附近的采样记录（见第 6 步）。

## 5. 收服务

```bash
OWNED_CARDS=$CARD STOP_PGID=$PGID bash build_artifacts/stop_by_port.sh $PORT | tee $RUN/stop.log
bash build_artifacts/card_lease.sh release $CARD $OWNER
```
（EngineCore 可能已在 `/stop_profile` 后自己退了；仍按这条收，并确认端口空、卡的进程表与 HBM 都落回去。）

## 6. 离线解析

```bash
cd $PROFDIR/*_ascend_pt/ASCEND_PROFILER_OUTPUT   # task_time.csv / kernel_details.csv 在这
WT=/workspace/code/dsv4_single_card/vllm_project/worktrees/moe-offload/tools/kt_offload/prof
# ① kt 相位（D）：先用 kd/ktphase 的老口径拿到每层 D，写 JSON
python3 $WT/ktphase.py $RUN/serve.log <start_byte> <end_byte> $RUN/ktphase.json
# ② 接缝分解：窗口取 cap/capture.json 的 capture_start/capture_end（下面两行按它填）
python3 $WT/seam.py $RUN/serve.log --ktphase $RUN/ktphase.json \
        --nw-csv ./task_time.csv --layers 40 \
        --since 2026-09-22T15:01:48Z --until 2026-09-22T15:01:56Z \
        --json-out $RUN/seam.json
```
`seam.py` 打印：窗口里用了几条记录、每层各段中位/p90、全网 ms/token、`E = sync − D`、
`A+F_dev = NW − T_cb`、以及 `NW − D` 这条与 §4.18 的 107.5 µs 直接对照的总量。

**窗口**（09-22 入库）：`--since/--until` 收下 `2026-09-22T15:01:48Z` / `09-22 15:01:48` / `15:01:48`
三种写法，**按日志自己的时钟**（本机 = UTC）；每条记录覆盖它前面 `interval_s` 秒，重叠 ≥ 50%
（`--window-min-overlap`）才收。不给窗口就是全量，**用了几条一定印出来**。也可以 `--bytes A:B`
按字节切（与 ktphase 同一口径）。`profile_window_end.txt` **不是**窗口终点（capture 脚本在
EngineCore 死后会挂满自己的超时，首发晚了 10 min）；窗口以 `cap/capture.json` 为准。

**NOTIFY_WAIT 的对位**（09-22 入库，首发是手工脚本 `make_nw_ours40.py`）：本构建每步 **42** 个
NOTIFY_WAIT，**前 2 个不是我们的层**。现在这条规则在工具里：只用"恰好 `层数 + --nw-skip`（默认 2）个"
的步、丢掉每步前 `--nw-skip` 个；并且**自带判据**——对位后 `Spearman(NW, cb)` 必须 ≥ `--nw-align-min`
（0.5）且比其它任何位移高出 `--nw-align-margin`（0.3），否则**拒绝出 NW / A+F_dev 两列并 rc=1**。
首发实测：位移 0/1/2 = −0.105 / 0.056 / **0.805**（Spearman；Pearson 0.361，被层 2 那个 1071 µs 的
离群点压住了，所以判据用秩相关）。`--nw-median-us` 仍可给一个常数当参照（那条不做对位检查）。

**判红条件**（rc=1，09-22 改口径）：
- 分段自检失败（见下），或
- NOTIFY_WAIT 对位判据没过。

⚠ 分段自检判的是什么、不判什么（**首发 rc=1 的 24/40 层是旧判据的锅，不是机器的锅**）：
`seam_timing.record()` 里各段是首尾相接的同一串时间戳 ⇒ **逐回调 Σ段 ≡ cb，没有可查的残差**；
而记录侧只给中位/p90，"各段中位之和 vs cb 的中位"是两个不同的统计量（**中位不可加**），右偏的段会
让它系统性为正。现在这个数只作为 `resid_med` 打印，**不是判据**。判据换成逐层的**必要条件**（都不假设可加）：
- 包含关系：每段的中位与 p90 都必须 ≤ cb 的（split 时中间各段还要 ≤ `kt` 的）；
- 联合界：`med(cb) ≤ p90(pre)+p90(kt)+p90(post)`（3×10% < 50%，故中位不可能超过它）；
  split 时另有 `med(kt) ≤ Σ p90(中间 5 段)`。每层的余量 `slack` 逐行打印出来 = **这一层能查出多大的洞**；
- split 跑里中间任何一段中位为 0 ⇒ 红（那是记录侧没填中段的回调混进来了）。

**这套查不出**：小于 slack 的洞；占比不高的"未 split 回调"；以及**单次回调**残差被放大——
中位/p90 的记录里根本没有这个信息。要判到单次回调，**记录侧**得多给一项（建议：每层
`n_unsplit`，或 `resid_us` 的 max）；解析侧已经预留：记录里出现这两个字段就直接按它精确判。
（本次没改 `vllm_ascend/` 下任何文件。）

## 读完之后

哪一段最大就打哪一段，并且先问它是不是**只是把 D 记在别处**：
- `sync − D` 大 ⇒ 唤醒/跨 NUMA 归并/写回（可与 `ktphase` 的 `weight`/`down` 相位交叉核对）；
- `A+F_dev` 大 ⇒ ACL report 线程的派发与回通知（`KT_EXTERNAL_NPU_REPORT_SUBSCRIBER=1`、`_subscribe_stream`）；
- `pre`/`post` 大 ⇒ 是我们自己的 Python（`_assert_no_pending`），最容易改，也最不可能是大头。

## 不覆盖什么

`packed_transfer=true` 的 `_kt_packed_forward` 也埋了同样的段（那条路的 submit/sync 是我们自己发的，
不需要 CPUInfer 探针），但**本执行单按 §4.18 基线把 packed 关着跑**；要量 packed 另起一发，
且与本发不可混读（改动项不为零）。
