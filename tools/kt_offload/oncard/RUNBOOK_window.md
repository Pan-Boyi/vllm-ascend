# 窗口执行单（A/A → t4 → t5，按此顺序，一条一条粘）

三件同构建；`.so` 不换，用现值 `2b58b578e4fc720ef30f4528c5a974fa`。
每条都带 `ONCARD_EXPECT_SO_MD5`：`.so` 不是这份就**拒绝起飞**（rc=3，什么都不碰），
所以三件不可能落在换构建的中间。

```bash
cd /workspace/code/dsv4_single_card/vllm_project
export TMPDIR=/dev/shm/dsv41_tmp TMP=/dev/shm/dsv41_tmp
export SO=2b58b578e4fc720ef30f4528c5a974fa
export CARD=3            # 按当时空闲的卡改
WT=worktrees/moe-offload/tools/kt_offload/oncard
```

## 1. A/A 对照（先校仪器，再量任何东西）

```bash
ONCARD_EXPECT_SO_MD5=$SO MODE=aa \
  AB_RUN=build_artifacts/moe_offload_ref/packed_ab_0920_094053 \
  bash $WT/run_packed_ab.sh $CARD
```
- 两臂都是 `packed_transfer=false`（**改动项＝零**），harness 参数与 A/B **同一个函数**产生
  （runner 里只有一处 GSM8K 启动点，自检里有一条数它的个数），并发 8 的那一档在 A/B 的同一位置先跑。
- 每臂三个 eval：`c8`（并发 8，与 A/B 完全相同）、`c8b`（同一次服务内的第二发并发 8）、`c1`（并发 1）。
  时间不够就 `AA_EVALS=c8,c1`（会少掉"同进程 vs 换进程"的那条旁证，但判定表仍然完整）。
- 跑完自动打印判定表（也可单独重跑：`python3 $WT/aa_report.py <run> --ab-run <A/B run>`）：

| 读数 | 结论 |
|---|---|
| 并发 8 分歧 且 并发 1 逐位相同 | 原因＝批组成 ⇒ 服务层逐位判据**无区分力**，逐位一致只能在 t1/t2/t3 层级用例上判 |
| 两档都分歧 | 不是批组成，另有来源（要查，**不得归给 packed**） |
| 两档都逐位相同 | A/B 的 31/50 ＝ packed 真改了数值 |
| 并发 1 分歧而并发 8 相同 | 与所有假设相反 ⇒ 脚本 rc=1，**不解释、先重跑** |

判据不可用（缺 eval／空字段／两臂题集不同／比较自身没有区分力）时脚本 **rc=1 并拒绝下结论**。

## 2. t4（node_us / window_us / ceiling；只测时间）

```bash
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t4 \
  --layer 20 --n 100 --reps 24 --profile-cpu-us 371.8
```
- 判定全部相对本发实测：`node ≥ window` 即形态 C 作废；否则 `ceiling = min(cpu, window) − node`。
- 有阴性对照（无节点图臂 vs eager 纯 NPU 臂）、仪器阳性对照（500 µs 胖节点必须被量到）、
  1/2 节点可加性、变异臂（把胖节点成本代进模型，结论必须翻面）。任一不过 ⇒ VERDICT: FAIL，数别用。

## 3. t5（侧流能不能救回；两臂分进程）

```bash
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 --arm on
ONCARD_EXPECT_SO_MD5=$SO bash $WT/run_oncard.sh $CARD t5 --arm off
```
- `--arm off` 是**阳性对照**：必须抛错（107025/107011），否则机器不在前三发那个状态 ⇒ 本轮比较作废。
- `--arm on` 捕获成功**不等于通过**：还要 replay ≥3 个不同输入、逐位等于串行臂、两两不同、
  每层 host-func 每 replay +1。分两次跑是因为失败的捕获可能污染同进程里后一次捕获。

## 跑完怎么读

1. A/A 先定"服务层逐位判据有没有区分力"——它决定 t4/t5 的读法，也决定 k.8 那条"数值未证一致"怎么收。
2. t5 若捕获并正确重放 ⇒ 侧流路线复活，形态 C 降为备胎；否则形态 C 由 t4 的 ceiling 决定做不做。
3. 形态 C 若要做，必须建在 packed 的每层自有缓冲之上（A6，kt 的 pinned 槽位是 `layer_idx % 2`）。
