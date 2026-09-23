# RUNBOOK：§4.33「默认开」上卡冒烟

> PLAN §4.33：侧流 / 事件等待 P1a / 共享专家进窗口 **默认开**，P1b 仍默认关（用户 2026-09-23 10:40 定、10:55 确认 P1b 不动；
> 第二版验收全过后执行）。本冒烟回答两件事，**只占一张卡、起一次服务**：
> ① 什么都不设时，进程里三项的有效值真的是开（不是「代码说默认开、服务里其实没开」）；
> ② 这个「什么都不设」和第二版 D 臂（显式 `SIDE=1 P1A=1 P1B=0 SIW=1` + cfg `overlap_side_stream: true`）逐位相同。
> 读数全部**内联回本文 §4**：run 目录是未跟踪产物，随时会消失。

## 0. 语义表（代码：`vllm_ascend/kt_offload/switches.py`，唯一读取点）

| 开关 | 未设（默认） | `"0"` | `"1"` | 其它值 |
|---|---|---|---|---|
| 侧流 `VLLM_ASCEND_DSV41_KT_SIDE_STREAM` + cfg `overlap_side_stream`（默认 **true**） | = cfg（默认开） | **关**（压过 cfg，关闸） | 开，仅当 cfg 为 true（cfg false 时 `1` 不翻盘，日志注明） | 拒起 |
| P1a `VLLM_ASCEND_DSV41_KT_ASYNC_WAIT` | **自动**：packed 或侧流开 ⇒ 开；否则关（不报错）；P1b 开 ⇒ 关（让位） | 关 | 开；前提不满足 ⇒ 拒起；与 P1b 同为显式 `1` ⇒ 拒起 | 拒起 |
| 共享专家进窗口 `VLLM_ASCEND_DSV41_KT_SHARED_IN_WINDOW` | **跟随侧流** | 关 | 开（无侧流 ⇒ warning，照跑） | 拒起 |
| P1b `VLLM_ASCEND_DSV41_KT_DEVICE_SIGNAL` | 关 | 关 | 开（前提同 P1a；P1a 未设则 P1a 自动让位） | 拒起 |

eager（`enforce_eager` / cudagraph NONE）：侧流/P1a/P1b 只作用于捕获路径；另外 layer 在 eager 下把**自动**得来的 P1a 与
共享进窗口关掉、不建副流、不打 `SIDE STREAM ENABLED` ⇒ 全不设的 eager 与 §4.33 之前逐调用相同（单测
`test_switches.py::test_eager_with_every_default_runs_the_serial_path_bit_for_bit`，去掉门控的变异臂 2 条红）。

启动时每个进程打一行（`config._validate_preconditions`，INFO）：

```
kt_offload switches (PLAN §4.33; value(source[: why])): side_stream=on(default) async_wait=on(auto: side stream on) shared_in_window=on(auto: follows the side stream) device_signal=off(default)
```

每层的 `SIDE STREAM ENABLED` / `ASYNC WAIT ENABLED` / `SHARED IN WINDOW for layer` / `DEVICE SIGNAL ENABLED` 行照旧（前缀未改，
`run_arm.sh` 的 40/0 计数照旧可用；括号里改成打印有效值与来源）。

## 1. 旧验收脚本在默认翻转后的口径

- `run_arm.sh` 一直**显式**把四个开关传进服务（`SIDE P1A P1B SIW`，缺省 `SIDE`=A 臂 0/B 臂 1，其余 0）⇒ **A 臂仍是串行**：
  `SIDE=0` 在 §4.33 后是关闸（不再 raise），与 cfg 里 `overlap_side_stream` 写 true/false 无关；cfg false 也照常起。
  🔴 不许把这些缺省改成「不设」——那样 A 臂会悄悄变成合版本，而两臂逐位全同 + perf 无差正是那种错误的样子。
- 新增 `DEFAULTS=1`：四个开关一个都不传，并 `env -u` 剥掉父 shell 残留；此时显式给 `SIDE/P1A/P1B/SIW` 任一 ⇒ rc=2。
  必检期望取 `WANT_SIDE/WANT_P1A/WANT_P1B/WANT_SIW`（默认 1/1/0/1），另核启动行逐字等于 `WANT_SWITCH_LINE`（默认即上面那行），
  不等 ⇒ 臂作废 rc=5。显式臂只把启动行记进 `arm_summary.txt`，不判。
- `seq_merge_v2.sh` 的 `RUNTIME_BASE` 缺省 `d1c5b7faf`：§4.33 改了 `vllm_ascend/`，再用它跑第二版会 rc=3 拒飞——要重跑须显式给新基。
- 板上 harness（t3/t4p/t5r/t5 auto-dispatch，经 `npu_moe.make_offload` 建层）：`make_offload` 对 P1a 与共享进窗口
  `os.environ.setdefault(..., "0")`，保持 §4.33 之前「不设 = 0」的建层；调用方显式给的值照旧生效。

## 2. 怎么跑（主会话排卡后执行；本 agent 未上卡）

前提：本改动已合进 `worktrees/moe-offload`（`run_arm.sh` 核的是那棵树：`vllm_ascend` 路径、`git status` 为空、指纹 dirty_files 0/0）；
第二版产物 `oncard_runs/20260923_105413_merge_v2/{D4/lp.jsonl,prompts.jsonl,cfg_D.json}` 仍在（未跟踪，会消失；脚本缺了就拒飞）。

```bash
ROOT=/workspace/code/dsv4_single_card/vllm_project
T=$ROOT/worktrees/moe-offload/tools/kt_offload/oncard/sidestream
# 干跑（不起服务；预检仍会读一次 npu-smi）：
DRY_RUN=1 PF_N=1 PF_GAP=0 PF_WAIT=0 bash $T/smoke_defaults.sh 2 /dev/shm/dsv41_tmp/smoke_dry
# 实跑（卡 2；卡 3 硬拦）：
bash $T/smoke_defaults.sh 2          # RUN = $ROOT/oncard_runs/<ts>_defaults_on
```

`smoke_defaults.sh` 做的事：由第二版 `cfg_D.json` **去掉 `overlap_side_stream` 一项**生成 `cfg_E.json`（断言原值 true、packed false），
复制 `prompts.jsonl`，记 D4 lp 与 prompts 的 md5；占卡预检（与 `seq_merge_v2.sh:card_clear` 同口径、`pipefail`）；
`DEFAULTS=1 REQUIRE_CLEAN_TREE=1 run_arm.sh $RUN E1 <卡> probe,lp`；然后
`lp_verdict.py D4/lp.jsonl E1/lp.jsonl --skip-first=13`（与第二版 C4 vs D4 同口径）。约 5.5 min 加载 + 8 min lp。

可选第二臂（cfg 关侧流 + 全不设 = 三项皆关且能起），只看必检与启动行，不比 lp：

```bash
RUN=<同上 RUN>; python3 -c "import json;d=json.load(open('$RUN/cfg_E.json'));d['kt_offload_config']['overlap_side_stream']=False;json.dump(d,open('$RUN/cfg_G.json','w'),separators=(',',':'))"
DEFAULTS=1 WANT_SIDE=0 WANT_P1A=0 WANT_P1B=0 WANT_SIW=0 \
WANT_SWITCH_LINE='side_stream=off(cfg: overlap_side_stream=false) async_wait=off(auto: needs packed_transfer or the side stream) shared_in_window=off(auto: follows the side stream) device_signal=off(default)' \
REQUIRE_CLEAN_TREE=1 bash $T/run_arm.sh $RUN G1 2 probe
```

## 3. 判据（跑之前写定）

E1 通过 ⇔ 以下全部成立：

1. `arm_summary.txt` 必检：`vllm_ascend` 指向 `worktrees/moe-offload`；`SIDE STREAM ENABLED` **40**、`ASYNC WAIT ENABLED` **40**、
   `SHARED IN WINDOW for layer` **40**、`DEVICE SIGNAL ENABLED` **0**、ACL 错码 **0**；
   `kt_offload switches` 行 ≥1 条且**每条**含 `side_stream=on(default) async_wait=on(auto: side stream on) shared_in_window=on(auto: follows the side stream) device_signal=off(default)`。
   任一不符 ⇒ run_arm rc=5，臂作废（=「代码默认开」没有进服务）。
2. `lp_cmp_D4_E1.txt` 六条全绿：prefill 187/187、decode **1170/1170**、非零 |Δlogprob| 0、top-k key-set 差 0、chosen-token 差 0、退化步 0/0（两份）。
   依据：跨重启逐位相同是本线已立住的口径（第二版 C4 vs D4 decode 1170/1170；第一版 A4r vs B4 1174/1174）。
3. 🔴 lp 红（rc=11）时**不下结论、不调 skip、不换批**：补 F1 = 新代码 + cfg_D + 显式 `SIDE=1 P1A=1 P1B=0 SIW=1`（`run_arm.sh $RUN F1 <卡> lp`，
   需先 `cp $V2/cfg_D.json $RUN/cfg_F.json`），再比 D4 vs F1、E1 vs F1：
   E1=F1≠D4 ⇒ 新代码对「显式」也改了结果（§4.33 的实现有问题，不是默认值的问题）；E1≠F1 ⇒ 默认解析与显式不等价（查启动行）；三者两两皆异 ⇒ 重启地板，另排自比。

不做：perf、GSM8K（第二版已跑；本改动不动计算路径，只动开关解析——单测 `test_the_explicit_accepted_arm_resolves_like_the_default` 钉住「D 臂显式 = 全不设」）。

## 4. 读数

（上卡后填；未跑）
