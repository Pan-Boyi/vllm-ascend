# RUNBOOK — P1a：设备改等 ExternalEvent，不再等 host-func 回调返回

> 开关：`VLLM_ASCEND_DSV41_KT_ASYNC_WAIT=1`（**纯 env**，默认关）。
> 前置：`packed_transfer` 或 `overlap_side_stream` 之一为 true（否则 config 起服务就 raise）。
> 代码：`vllm_ascend/kt_offload/async_wait.py`（完成线程 + 判据）、
> `layer.py:_launch_cpu_node` / `_kt_packed_submit_only` / `_ext_wait` / `_ext_reset`。

## 0. 这条路在改什么，以及它怎么错

原来：`D2H → host-func[ kt_pre | submit | kt 算 | sync ] → H2D`，设备的 `NOTIFY_WAIT`
把整段连同 ACL 两头的往返（A＝报告送达+抢 GIL+建参数元组，F＝回调返回+通知 device）全包了。
现在：`D2H → host-func[ kt_pre | submit ]（立即返回） → ExternalEvent.wait → reset → H2D`，
`sync(0)` 与 `event.record()` 由**完成线程**做。省的是 F 那一条腿（§4.22 F1/F2：A+F 78 µs/层，
而 `DAVID_EVENT_WAIT` 0.027 µs、`MEM_WAIT_VALUE` 0.144 µs）。

🔴 **主失效形态：放行早 = 更快且算错，且不报错**（H2D 拷到的是上一步的 CPU 输出）。
⇒ 判据一律是**与开关关着的那一臂逐位相同**，绝不是 TPOT。三个必须能红的地方：
① 完成线程在 `sync` **之后**才 record；② `ExternalEvent.wait` 节点真的在图里；
③ `reset` 节点也真的在图里（否则第 2 次 replay 起，wait 被上一次的信号直接放过）。

🔴 **唯一无先例的一点**：record 由「非主线程、在 replay 进行中」发出。
（`ExternalEvent` 在捕获期 wait+reset、回放期由 host record，`mla_v1.py:947-970` / `:1744-1747`
与 `attention_v1.py:1036-1038` 早就在跑 —— 但那是**主线程**在两次 replay 之间做的。）
所以下面 §1 的阳性对照不是形式：它是**唯一**能回答「设备到底有没有停在这个 event 上」的检查。

## 1. 先做 t5r（不起服务，约 4 分钟一发）

```bash
WT=worktrees/moe-offload/tools/kt_offload/oncard
# 干净臂 + 阳性对照（--stall-ms 把每一次 record 压后 50 ms）
bash $WT/run_oncard.sh $CARD t5r --arms serial,fork,p1a --mutant no-wait  --layers 8 --steps 8 --stall-ms 50
bash $WT/run_oncard.sh $CARD t5r --arms serial,fork,p1a --mutant no-reset --layers 8 --steps 8 --stall-ms 50
bash $WT/run_oncard.sh $CARD t5r --arms serial,fork,p1a --mutant early-record --layers 8 --steps 8 --stall-ms 50
```
判读：
- `serial` / `fork` / `p1a` 三臂都必须 **8/8 逐位等于 eager 串行参照**，且 `p1a` 的
  `ext_records_per_layer` 每层 = steps（提交数与放行数配对）。
- **阳性对照**：`--stall-ms 50` 下 `p1a` 臂的 8 次 replay 必须 ≈ `steps × layers × 50 ms`；
  同一发里 `fork` 臂（没有 async wait）只要 ~0.03 s。🔴 **差不出来就说明设备根本没停在那个 event 上，
  本发所有"逐位相同"都不算证据。**
- **变异臂**必须红，且 `nodes_skipped` 要证明变异**真的发生了**（"安静地什么也没改"和
  "判据没有区分力"从 verdict 行上长得一模一样，所以两者分开判）。
- 干净臂单发 rc 仍是 1（Verdict 要求每发至少一个变异臂）；按**配对发**判读。

🔴 **`--mutant no-reset` 在本构建上没有区分力（2026-09-23 实测，不是推测）**：
8 个 reset 节点**确实**被留掉了（`nodes_skipped.reset = 8`），臂仍然 8/8 逐位相同，
而且在 `--stall-ms 50` 下**每一次 replay 都照付了 50 ms**（3.343 s ≈ 8×8×50 ms）
⇒ 本构建上捕获进图的 `ExternalEvent.wait` **每次 replay 自己重新武装**，reset 不是让下一次
再等的那个东西。torch_npu 的文档写的正相反（"Before reusing ExternalEvent, you need to call
reset()"），生产路径**照旧保留 reset**（0.144 µs 换掉「依赖未写进文档的行为」）。
⇒ 这一臂的判据已改成**钉住这个观测**：它往任何一个方向变都说明平台变了。
**真正能红的那一臂是 `--mutant no-wait`**（wait 节点留掉 ⇒ 0/8 逐位、0.141 s）。
reset 的红臂只在 CPU 模型里：`test_side_stream_overlap.py::test_p1a_without_the_reset_*`
（CPU 模型按 torch_npu 文档建模「信号跨 replay 留存」，比这张卡更保守 —— 两者不一致时，**卡说了算**）。

🔴 **「完全不 record ⇒ 挂住」这一臂不在卡上跑**：它会把设备永久留在等待里，下一臂就没卡了。
那一臂是 CPU 单测 `tests/ut/kt_offload/test_side_stream_overlap.py::test_p1a_without_the_record_hangs`。

## 2. 整网 A/B（perf 主、逐位次）

🔴 **两臂都开侧流**，只差 `VLLM_ASCEND_DSV41_KT_ASYNC_WAIT` 一个开关 ——
侧流的 −3.00 ms 早在 §4.26 单独结过账，把它和 P1a 混在一起量出来的是合计，不是 P1a。
⇒ `cfg_A.json` 与 `cfg_B.json` **必须逐字节相同**（都带 `overlap_side_stream:true`），
`seq_p1a.sh` 会核对，不同就拒绝跑。

```bash
RUN=$PWD/oncard_runs/$(date +%Y%m%d_%H%M%S)_p1a; mkdir -p $RUN
cat > $RUN/cfg_A.json <<'JSON'
{"kt_offload_config":{"enabled":true,"resident_strategy":"prefix","num_resident_experts":100,
 "cpuinfer_threads":96,"threadpool_count":2,"numa_nodes":[0,1],"packed_transfer":false,
 "overlap_side_stream":true}}
JSON
cp $RUN/cfg_A.json $RUN/cfg_B.json
bash $WT/sidestream/seq_p1a.sh $RUN $CARD perf        # A1,B1,A2,B2 交错，每臂 CASES="bs1 1 16" REPEAT=3
```
必检（`run_arm.sh` 自己做，对不上 = 这一臂作废）：
`SIDE STREAM ENABLED` 两臂都 40 条；`ASYNC WAIT ENABLED` **A 臂 0 条、B 臂 40 条**；
ACL 错码 0（显式五码 + **非数字边界** + 排掉 `kt-seam` 埋点行，见 `oncard_lib.sh:ONCARD_ERRCODE_RE`）；
`host-func ... never ran` 0 条。

判读：**同臂跨轮地板先算出来，再看 A−B**；差值 > 1 ms 且 > 地板才算收益。
🔴 跨发不得相减（§4.25 的教训：绝对值跨发差 ~1.5 ms）。

逐位（perf 有结论之后再排，两臂各一发）：
```bash
python3 $WT/sidestream/make_prompts.py --out $RUN/prompts.jsonl --seed <新种子>
bash $WT/sidestream/seq_p1a.sh $RUN $CARD lp          # 之后 logprob_cmp.py --skip-first=12
```
`--skip-first=12` **按臂校准**，不许拿结果反推截断点（§4.25/§4.26）。

## 3. profile：默认不采
用户 2026-09-23 明确：**默认只跑 perf + 精度判据，profile 太慢 ⇒ 非必要不采**。
只有在 perf 与逐位互相矛盾时才加 profiler，并且要在报告里写清为什么非看不可。
（真要看时，P1a 的 device 侧指纹是：`HOSTFUNC_CALLBACK` 节点仍在但变短、
`NOTIFY_WAIT` 里不再包含 kt 的计算、多出每层一对 `MEM_WAIT_VALUE`。）
