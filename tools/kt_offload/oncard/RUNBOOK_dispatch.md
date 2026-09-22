# 上卡执行单：ACL host-func 派发延迟（A+F = 78.0 µs/层 = 3.83 ms/token）

**本单只回答**：§4.20 的 `A+F_dev = NOTIFY_WAIT − T_cb = 78.0 µs/层` 里，有多少是
**「报告线程被 kt 的自旋线程挤住」**，有多少是 ACL/驱动固有的往返。

**不执行**。这份是给下一次排到卡的人贴的命令。三个臂都是**默认关**的开关，
不设开关时代码路径逐字节等于 §4.18/§4.20 基线。

## 0. 为什么值得一试（读代码得到的三条硬事实）

1. `kt-kernel/cpu_backend/worker_pool.cpp` 的 `InNumaPool::worker_thread` 与
   `NumaJobDistributor::worker_thread`：worker 干完活后**忙等 50 ms 才睡 condvar**
   （现在这个 50 是 `kt_worker_spin_ms()` 读 `KT_WORKER_SPIN_MS`，**默认仍是 50**）。
   decode 每层几百 µs 就有新活 ⇒ 稳态下 kt 线程**全程 100% 自旋**，且每条被
   `kt_bind_worker`（`hwloc_bitmap_singlify` + `HWLOC_CPUBIND_STRICT`）**硬绑到单个 PU**。
2. 🔴 **数量订正（09-22 读代码得到，与本单上一版写的「96×2=192」不符）**：
   `cpuinfer_threads:96 / threadpool_count:2` ⇒ 每个 subpool **48** 条
   （`experts_base.py:365-368`），每 subpool 里 1 条是 NumaJobDistributor、47 条是 InNumaPool
   （`worker_pool.cpp` 的 `InNumaPool` 构造从 `i=1` 起建线程，线程 0 就是 distributor）
   ⇒ **总共 96 条 kt 线程、占 96 个 PU**（每 NUMA 48 个）。机器 256 个 PU ⇒ **空闲 160 个**，
   其中**每 NUMA 有 16 个整核**（两个 SMT 都空）完全没有 kt。起飞后用第 2b 步的 `threads.txt`
   直接数（`numa_*_t_*` / `numa_*_m_*` 的条数）**当场证伪或坐实这条**。
3. 派发回调的是 torch_npu 进程内**唯一一条**报告线程，普通 `SCHED_OTHER`、**不绑核、无线程名**
   （见报告正文的链路图）。它被唤醒时要在一台 96 个 PU 在自旋的机器上找核。

⇒ §4.20.1 微基准（kt 线程**存在但空闲** ⇒ 50 ms 后已睡进 condvar ⇒ 核是空的）量到 56.8 µs，
整网（线程**正在自旋**）78 µs。**这 21 µs 差与「自旋挤占」方向一致**，本单就是去判它。

## 1. 三个臂（**同时段交错**，不是先跑完一个再跑下一个）

一臂 = 一发服务（配置只能在起服务时给）。**三臂交错轮转**（D0→D1→D2→D0→…，至少 2 轮），
因为这台机器是共用的：先跑完 D0 再跑完 D1，量到的可能是「别的租户什么时候忙」。
配置其余项**逐项同 §4.18/§4.20 基线**（`RUNBOOK_seam.md` 第 2 步的 `cfg.json`，
`cpuinfer_threads:96 / threadpool_count:2 / numa[0,1] / N=100 prefix / packed 关`），
只改下面这一列。**每臂都要盖构建指纹**（`RUNBOOK_seam.md` 第 1 步），指纹不同不得相减。

| 臂 | 改什么 | 预计 |
|---|---|---|
| **D0 对照** | 什么都不改（两个 env 都不设） | A+F ≈ 78 µs/层（复现 §4.20） |
| **D1 绑核** | `VLLM_ASCEND_DSV41_KT_REPORT_CPU=auto`（kt 仍是 96 线程，**一个核都不让**） | 若自旋挤占成立：A+F 中位/p90 下降；D（kt 计算）**不应变**——变了说明配置不只差这一项 |
| **D2 绑核+短自旋** | D1 的 env **加** `KT_WORKER_SPIN_MS=1` | A+F 再降；**代价在 E（worker 唤醒）**：kt 每层改走 condvar 唤醒，E 现在是 18.2 µs，可能把省下的吃回去 ⇒ **A+F 与 E 必须一起读** |

🔴 **D2 的前提**：`KT_WORKER_SPIN_MS` 是 kt-kernel 的 C++ 改动，**要重建并换掉 kt wheel 才存在**。
wheel 没重建之前，D2 这一臂标 **未覆盖**（设了 env 也没有任何效果，而且 `[kt] worker spin ...`
这行日志不会出现——见第 2c 步的判据），不要写成「试了没用」。重建 wheel = 换构建 ⇒
**t0/t1/t2 要重跑**，且重建后的 D0/D1 也要在新构建上重跑一遍才能与 D2 相减。

`VLLM_ASCEND_DSV41_KT_REPORT_CPU` 的取值：
- 空（默认）＝**一次 `sched_setaffinity` 都不调**；
- 具体 PU 号 ＝ 就绑它（越界、或不在本进程 cpuset 里 ⇒ **拒绝启动**）；
- `auto` ＝ 按 kt 自己的绑核规则算出 kt 占了哪些 PU，再从**两个 SMT 都空闲**的整核里挑
  **kt 顺序上最后**的那个（规则与出处写在 `vllm_ascend/kt_offload/report_affinity.py`），
  并且**在挑之前先读被判对象**：把算出来的 PU 集合与 `/proc/self/task/*/`
  里 kt 线程（`numa_*_t_*` / `numa_*_m_*`）真实的 `Cpus_allowed_list` 比对，**不一致就拒绝启动**。
  本机（2 socket × 8 CCD × 8 core × 2 SMT）96 线程下 auto 的预期结果：**PU 63（NUMA0）/ 127（NUMA1）**；
  device 属于哪个 NUMA 读不出来时（本机 8 张卡跨两个 socket ⇒ 读不出）按 NUMA 升序挑 ⇒ **63**。
  想钉到另一侧就直接写 `=127`。

可选第二个开关 `VLLM_ASCEND_DSV41_KT_REPORT_FIFO=<1..99>`（默认空＝关）：把报告线程切到 SCHED_FIFO。
容器里没有 `CAP_SYS_NICE` 会失败 ⇒ **只 warn 并继续**，日志里 `fifo_applied=` 会是 `False`。
**它是第三变量**：要开就单独起一发（D1F），不要和 D1 混在一发里读。

## 2. 起服务并读 A+F

```bash
cd /workspace/code/dsv4_single_card/vllm_project
export TMPDIR=/dev/shm/dsv41_tmp TMP=/dev/shm/dsv41_tmp
export SO=85331298bdeffee02b7c42c5658030ac
export CARD=?            # 起飞当时连采 ≥3 次实测空闲的那张（判据抄 RUNBOOK_window.md §0）；本线只用卡 0–2
export PORT=8041 OWNER=dsv41_vllm_singlecard
export ARM=D0            # D0 / D1 / D2
RUN=$PWD/oncard_runs/$(date +%Y%m%d_%H%M%S)_dispatch_$ARM; mkdir -p $RUN
python3 worktrees/moe-offload/tools/kt_offload/build_fingerprint.py stamp $RUN --label dispatch_$ARM | tee $RUN/fp.txt
grep -q "$SO" $RUN/build_fingerprint.json || echo "❌ .so 不是 $SO，停"

# 本臂要加的 env（D0 = 两行都留空）
case $ARM in
  D0) ARM_ENV="" ;;
  D1) ARM_ENV="VLLM_ASCEND_DSV41_KT_REPORT_CPU=auto" ;;
  D2) ARM_ENV="VLLM_ASCEND_DSV41_KT_REPORT_CPU=auto KT_WORKER_SPIN_MS=1" ;;   # 需要重建过的 kt wheel
esac

PROFDIR=/dev/shm/dsv41_prof_disp_$ARM; rm -rf $PROFDIR; mkdir -p $PROFDIR
cat > $RUN/cfg.json <<JSON
{"kt_offload_config":{"enabled":true,"resident_strategy":"prefix","num_resident_experts":100,
 "cpuinfer_threads":96,"threadpool_count":2,"numa_nodes":[0,1],"packed_transfer":false}}
JSON
( cd /workspace/code/dsv4_single_card/vllm_project && setsid env \
    WT=moe-offload DSV41_VLLM_TREE=$PWD/worktrees/moe-offload-vllm LEASE_OWNER=$OWNER \
    KT_OFFLOAD=1 NPUS=$CARD PORT=$PORT EAGER=0 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    KT_ADDITIONAL_CONFIG="$(cat $RUN/cfg.json)" \
    MAXLEN=8192 MAXSEQS=8 MAXBATCH=4096 \
    KT_MOE_PHASE_TIMING=1 \
    VLLM_ASCEND_DSV41_KT_SEAM_TIMING=5 \
    $ARM_ENV \
    EXTRA_ARGS="--profiler-config {\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFDIR\"}" \
    bash worktrees/moe-offload/tools/kt_offload/serve_v41_singlecard.sh ) > $RUN/serve.log 2>&1 &
SERVE_PID=$!; PGID=$(ps -o pgid= -p $SERVE_PID | tr -d ' '); echo "pgid=$PGID" | tee $RUN/pgid.txt
```

就绪判据（只认 `^Traceback|EngineCore failed|进程退出|超时`）、`vllm_ascend = ` 读真对象、
profile 采集（**一次性**：`stop_profile` 之后进程会 SIGSEGV，所以 profile 放最后）、
收服务（`stop_by_port.sh` 带 `STOP_PGID=$PGID`）：**逐条照抄 `RUNBOOK_seam.md` 第 2/4/5 步**。

### 2b. 采完 profile 前，先拍一张「核忙不忙」的证据（30 s，纯 host）

这一步是 D1/D2 的**正证据**：不拍它，A+F 降了也说不清是不是绑核绑出来的。

```bash
EPID=$(pgrep -f 'VLLM::EngineCore' | head -1)     # 不用 pkill，只读
# ① 进程里有多少条线程在跑（R 态）—— 自旋线程会全程 R
for i in 1 2 3; do
  awk '{print $3}' /proc/$EPID/task/*/stat | sort | uniq -c | tr '\n' ' '; echo; sleep 2
done | tee $RUN/thread_states.txt
# ② 每个 PU 的占用（有没有空核）
top -b -n1 -1 | head -20 > $RUN/percpu.txt
# ③ 线程名 + 当前 CPU + 绑核：kt 的是 numa_<n>_t_*/numa_<n>_m_*，报告线程是没名字的那条
for t in /proc/$EPID/task/*; do
  printf '%s %-16s cpu=%s %s\n' "${t##*/}" "$(cat $t/comm)" "$(awk '{print $39}' $t/stat)" \
     "$(awk -F'\t' '/Cpus_allowed_list/{print $2}' $t/status)"
done | sort -k2 | tee $RUN/threads.txt
grep -c 'numa_' $RUN/threads.txt        # ← 事实 2 的判据：应当是 96，不是 192
```

判读：`thread_states.txt` 的 R 态数 D0 ≈ 96 + vLLM 自己的几条；
D2（自旋 1 ms）应**明显更低**（kt 线程多数时间睡在 condvar 上）——**这是 D2 生效的独立证据**，
比 TPOT 可靠。D1 的 R 态数**不应变**（一个核都没让）。

### 2c. 开关到底生没生效（判据要读被判对象）

```bash
grep -n "report thread pinning planned\|ACL report thread:" $RUN/serve.log | tee $RUN/report_thread.txt
grep -n "\[kt\] worker spin before condvar sleep" $RUN/serve.log | tee -a $RUN/report_thread.txt
grep -n "\[kt\] NUMA . core order" $RUN/serve.log >> $RUN/report_thread.txt     # stripe/pack，auto 的前提
```

- D1/D2 必须有一行 `kt_offload ACL report thread: tid=… affinity [...] -> [63] (target PU 63,
  in_effect=True, …)`：`affinity ... -> ...` 是**绑前/绑后 `sched_getaffinity` 的回读**，
  `in_effect=False` ⇒ **这一臂作废**（绑了没生效，不能当绑核臂读）。
  那行里的 `tid=` 就是报告线程，可以回到 `threads.txt` 里核对它确实只允许一个 PU。
- D2 必须有 `[kt] worker spin before condvar sleep: 1 ms`；**没有这行 = wheel 没换** ⇒ D2 未覆盖。
- D0 里这两行都**不应该**出现（开关默认关的自证）。

## 3. 离线判读

```bash
cd $PROFDIR/*_ascend_pt/ASCEND_PROFILER_OUTPUT
WT=/workspace/code/dsv4_single_card/vllm_project/worktrees/moe-offload/tools/kt_offload/prof
python3 $WT/ktphase.py $RUN/serve.log <start_byte> <end_byte> $RUN/ktphase.json
python3 $WT/seam.py $RUN/serve.log --ktphase $RUN/ktphase.json \
        --nw-csv ./task_time.csv --layers 40 --since <cap start> --until <cap end> \
        --json-out $RUN/seam.json
```

**主判据 = `seam.json` 每层 `A+F_dev` 的中位与 p90**（D0 vs D1 vs D2 逐层配对比）。
`seam.py` 现在自带对位判据（Spearman ≥ 0.5 且优于其它位移 0.3），**判据没过就没有 A+F 这一列**，
那一臂作废，别拿 `NW` 硬读。

**同时必须读 D 与 E**（同一份 `seam.json`）：
- D（kt 计算，来自 `ktphase`）：D1 不应变；变了说明两臂差的不只是绑核。
- E = `sync − D`（worker 唤醒 + 归并/写回，§4.20 = 18.2 µs）：**D2 的代价在这里**。
  只看 A+F 会把 D2 判成净赚。判 D2 要看 `A+F_dev + E` 的合计。

**旁证（不依赖我们的埋点）**：`task_time.csv` 里 `HOSTFUNC_CALLBACK` 每次的时长与计数。

```bash
awk -F, 'NR==1{for(i=1;i<=NF;i++)h[$i]=i} NR>1 && $h["Name"]~/HOSTFUNC_CALLBACK|NOTIFY_WAIT/ {print $h["Name"], $h["Duration(us)"]}' \
   task_time.csv | sort | uniq -c | head
```

⚠ 已知仪器坑（别再踩）：`task_time.csv` 每步 **42** 个 NOTIFY_WAIT，前 2 个不是我们的层；
`seam.py` 自己对位并自证。

**次判据 bs1 TPOT**：噪声地板 ≈1 ms（§4.20）⇒ **差值小于 1 ms 判「不可分辨」**，
并且**跨发不得相减**——三臂是三发服务，TPOT 只能在同一发内部与它自己的分布比。
预期量级：A+F 每层省 20 µs × 40 层 = 0.8 ms/token，**正好在地板附近** ⇒ 指望 TPOT 判这件事会判错。

## 4. 还没有可粘命令的那条（GIL 尾巴，旧 D2）

`sys.setswitchinterval` 没有环境变量，本次**没有**为它加开关（本单的 D2 已改指自旋时长）。
纯 CPU 已量到的量级（读数内联，可复验）：GIL 无争用时「C 线程醒来 → 第一条 Python 字节码」**≈7 µs**；
只要有**一条**别的 Python 线程在跑，默认 5 ms 切换间隔下这个延迟中位 **8.1 ms**、p90 15.2 ms；
切到 0.05 ms ⇒ 中位 116 µs。⇒ **中位 78 µs 不可能是 GIL 争用**（争用一旦发生就是毫秒级），
但 §4.20 层 2 那个 **773 µs 的步首离群点正是这个形态**。要判它得另加一个默认关的开关
（`VLLM_ASCEND_DSV41_KT_SWITCH_INTERVAL_US`），判的是离群点不是中位。

## 5. 这一单不覆盖什么

- `packed_transfer=true`：两条回调路径都调了绑核（`_kt_host_forward` / `_kt_packed_forward`），
  但本单按基线把 packed 关着跑；要量 packed 另起一发。
- `--enforce-eager`：eager 路径的 host func 跑在**主线程**上，绑核开关会**跳过并 warn**
  （绑主线程是另一个实验）⇒ eager 发次读不到本单的东西。
- SCHED_FIFO（`..._KT_REPORT_FIFO`）：见 §1 末尾，要单独一发。
