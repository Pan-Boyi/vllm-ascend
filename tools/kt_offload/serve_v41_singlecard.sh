#!/usr/bin/env bash
# ============================================================================
#  serve_v41_bringup.sh —— DeepSeek-V4.1-Flash 首次起服务（bring-up）
#
#  目标：EP=4 / 卡 0,1,2,3（一个完整 UB 平面）/ TP=1。
#
#  📌 卡 1 的历史（留着是为了别再误读，不是为了排除它）：
#     2026-09-10 16:24 起 udevid=1 的 udis 错误逐分钟连续、无零窗口（6272 行），
#     软复位对卡 2/7 有效、对卡 1 无效 —— 那几个小时里默认卡位一度退到 0,2,3（EP=3）。
#     2026-09-11 整机重启后卡 1 三条仪器【全清】：health OK、dmesg 该卡 0 条、set_device+实算通过。
#     ⇒ 默认改回 0,1,2,3。
#     🔴 不把卡 1 写进任何黑名单，也不因为前科排除它 —— 排班按【起飞前实时查】，不按病历。
#     ⚠️ 但记住那天的教训：卡 1 的 `npu-smi -t health` 曾一度答 OK 而卡实际是坏的 ——
#        健康查询能答 ≠ 卡好了。所以下面的守卫是 health + dmesg 两条，不是一条。
#  这是「第一次前向」的必经之路，不是回归脚本 —— 参数按【先能起来】选，不按性能选。
#
#  与 serve_v4flash_regression.sh 的区别：
#    · 模型是 V4.1（hidden 5120、384 routed experts、engram 在 host）
#    · dp=4/ep=4 而不是 dp=2
#    · max-model-len 先开小（首次前向只要不炸，不要满长度 1048576）
#    · 不做门禁判定，只做起得来/起不来
#
#  ⚠️ `--kv-cache-dtype bfloat16` 是承重参数，不是可选优化：
#     A5 上它是唯一能脱离 fp8 的写法（`dsa_attn_kv_plan.py:104-115` resolve_dsv4_cache_dtype()
#     把其余取值全部塌成 auto → fp8）。不写它就走 npu_kv_quant_sparse_attn_sharedkv，
#     而那条的 cmp_ratio 白名单是 (0,1,4,128) —— **V4.1 要的 2 不在里面，会直接 raise**。
#     带上它走 A5 BF16 plan（cann_ops_transformer.sparse_flash_mla，layout PA_BBND），
#     而且它是 sparse 的不是 dense。
#  🔴 但这条 plan【存在】≠【跑得通】：代码里有，但没有任何测试或日志证明它被端到端跑过。
#     **我们会是第一个。⇒ 如果服务起不来，这条 plan 本身就是头号嫌疑人，别先查我们自己的接线。**
#
#  ⚠️ 三个【已知会挡路】的，撞到不是你的 bug：
#    1) `_select_dsv4_attn_cls` 需要 CUDA —— dsv41-model 在劫持，没好之前起不来
#    2) engram 存储层还没实现 —— dsv41-model 在写 gather kernel
#    3) HcPre 的 hidden 5120 已放开（kernel 64d07ecc7），但**要确认 .so 是新的**
#       —— 起飞前的产物新鲜度检查就是干这个的
#
#  显存预期（逐张量算出，见 V41_GATE_DESIGN.md 的分类表）：
#    EP=4: 275.7÷4 = 68.9 + 每卡各一份 10.4 =  79.3 GiB/卡  扣底噪 5.1 ⇒ 余约 43.6 GiB  ← 当前
#    EP=3: 275.7÷3 = 91.9 + 10.4           = 102.3 GiB/卡              ⇒ 余约 20.6 GiB  ← 少一张卡时的退路
#    bf16 KV 在 128K ctx 只占 0.396 GiB（3200 B/token，只有 4 个 kv-source 层随 ctx 增长），
#    EP=4 的 43.6 GiB 余量绰绰有余（EP=3 的 20.6 GiB 也吃得下）。
#    ⚠️ EP=2 是 148.2 GiB/卡，超 20 GiB，【起不来】—— 不要用两卡试
#
#  ---------------------------- SMOKE=1（撞阶梯专用）--------------------------
#  SMOKE=1 走 --load-format dummy，跳过真权重加载（省约 30~90 秒/轮）。
#  🔴 它只回答【会不会崩、崩在哪】，**不回答任何数值问题**。
#     任何精度/正确性结论不得引用 SMOKE 轮 —— 权重是随机的。
#  🔴 而且「权重加载」那一级在 SMOKE 下印【跳过】而不是【通过】：
#     被跳过的检查和通过的检查，输出必须不一样。
#  ⚪ 我们定过「改了配置就不是被验的那个配置」，也定过「撞崩溃阶梯可以、得结论不行」。
#     这条落在后者里 —— 所以它被允许，但必须一路带着标注，不能只在脚本头写一次。
#  ============================ 预期失败阶梯 ============================
#  🔴 第一次起服务【大概率会失败】，而这是正常的。
#     关键不是"有没有失败"，是【失败在第几级】—— 失败得越靠后，说明前面越多级过了。
#
#    ① 注册 / import         这里炸 = 注册没接上（model #13）
#    （原 ② 权重加载 mtp.* 这一级【已关闭】：上游 mapper 自己会丢，model 实测 in:3 → out:0）
#    ② engram _storage()     只读 mmap —— ⚠️【零 aclrtHostRegisterV2】。
#                            注册禁令因卡 3 死在真表的第一次 1 GiB 注册上而全面恢复：
#                            任何规模、任何卡都不注册。这一级炸 = 只读映射或表本身的问题。
#                            /dev/shm/engram_stage 全量已就绪（零容差校验），
#                            前置链里有 verify_shm_sharing.py --require-full。
#    ③ KV cache spec 收集     model #12/#12b 已解决
#    ④ profiling / dummy run   （model 已显式清零 warmup：新扩的池子里常有 NaN，
#                              从 warmup 飘出的 NaN 和模型里的数值问题长得一样）
#    ⑤ forward → metadata 字段映射
#       🔴 最可能炸在这一级，而且炸在【metadata 字段映射】的五个之一：
#          query_start_loc→cu_seqlens_q · seq_lens→seqused_kv · sas_metadata→metadata
#          cu_cmp_seqlen_list→seqused_cmp_kv · qli_cmp_residual_k→cmp_residual_kv
#       ⚪ 每个字段缺了都【带着字段名 raise】，不会把 None 喂进算子。
#    🟢 ⇒ 「失败在第 ⑤ 级并指名了字段」是【成功】，不是失败 —— 它意味着前四级全过了。
#
#    ⚠️ 这张阶梯自己也会过期：原「② 权重加载 mtp.*」和原「⑥ all_reduce 是 bf16」两级都已作废。
#       一张写着已修问题的清单比没有清单更糟 —— 它有名字、有数字、看起来已经查过。
#
#    ⑥ 跑起来之后的数值
#       ⚠️ 这一级的措辞被改过，改前的版本会把人送去查一个 no-op，照抄的话第一次
#          精度失败就会归错因。当前事实：
#          · TP=1 时 tensor_model_parallel_all_reduce 是 no-op
#            （vllm/distributed/parallel_state.py:737-739，world_size==1 直接 return）。
#            ⇒ 「all_reduce 是 bf16」这句在本配置下【没有对象】，不要照抄进任何结论。
#          · MoE 的归约在 token-combine：decode 走 MC2、prefill 走 AllGather。
#            两条都实测过：累加 fp32、只舍一次（逐元素 100.0000%）。
#          · 真正的缺陷是【路由权重被舍到 bf16】，已修（35fbdfeb5），
#            改前 40 层后 logits top-1 翻 13/27。
#       🔴 仍然未知、且必须在解读任何精度数字之前回答的：
#          **融合算子（npu_moe_* / FlashMLA 一族）的 MMA 累加器有多宽。**
#          fp32 ⇒ 形式差异只是求和顺序，~1e-7 可忽略；更窄 ⇒ 每加一次舍一次，量级跳到 O(1)。
#          没有这个答案，精度残差的解读会整体偏。
#  =====================================================================
#
#  用法:
#      ./build_artifacts/serve_v41_bringup.sh
#      NPUS=0,2,3 PORT=8010 MAXLEN=8192 ./build_artifacts/serve_v41_bringup.sh   # 退到 EP=3
#      EAGER=1 ...            # 关图捕获（--enforce-eager）。用来把「主机内存是谁吃的」
#                             #   分成两臂：捕获窗口 vs 加载残留。不是性能配置。
#      EXTRA_ARGS="--foo 1"   # 透传给 vllm serve，放在 --port 之前
# ============================================================================
set -o pipefail

ROOT=/workspace/code/dsv4_single_card/vllm_project
# 🔴 2026-09-11 11:09 的第一次起飞就死在这行上：它默认 integration，
#    而合好的 V4.1 代码在 base。integration 停在 1f2ab7a94，没有 attn_selector，
#    于是上游 nvidia 的 flashmla 类没被劫持，直接 raise
#    "fp8_ds_mla layout only supports fp8 kv-cache, got bfloat16"。
#    ⇒ 那次失败【测的不是我们的接线】，是一棵没有模型代码的树。不能当数据点。
#    这是今天第三个「判据/动作读了错的对象」：新鲜度检查读错树、采集守望壳采错服务、
#    现在是服务本身跑错树。⇒ 默认改成 base，并且【下面加一条会拦住它的守卫】。
WT=${WT:-base}
# ---- 选卡：不写死默认值，也不维护坏卡名单 -----------------------------------
# 🔴 两次教训都指向同一件事：任何写死的卡号都会过期。
#    09-10 写死「可用只有 0/2/3」→ 09-11 重启后卡 1 好了，那行字还在劝人绕开它。
#    09-11 改成默认 0,1,2,3 → 当天卡 3 死在真表的第一次注册上，默认值又不可用了。
#    ⇒ 稳定的是【归属】（哪几张是我们的），会变的是【健康】。
#      归属写死没问题；健康每次起飞现查，查不到 = 判据不成立 = 不选它。
#    NPUS 显式给了就按给的走（下面所有守卫照常跑）；没给就现选。
OURS="${OURS:-0 1 2 3}"            # 归属：用户 2026-09-14 口径「优先用 0-3」。4-7 不是我们的，要用须重新确认。
# ⚪ 2026-09-18 由 4 5 6 7 改来（过期默认值，整理会话发现）。只影响【不传 NPUS】的自动选卡；已验证发次都显式传 NPUS=0,1,2,3。
# 🔴 之前写的是 "4 5 6 7 0 1 2 3"（注释"现在全是我们的"），而它引用的权威
#    （env_dsv41.sh 的横幅）只说了「用卡 4 5 6 7」—— 清单【比它的出处宽】。
#    ⚠️ 一份自称有出处的清单，比它的出处宽，读的人只会看见"有出处"三个字。
#    而 stop_by_port.sh 的杀进程护栏一度从这里派生 ⇒ 它会愿意杀卡 0-3 上的实例。
#    ⇒ 卡 0 2 3 不是我们的，任何用途先过协调线；
#      卡 1 有前科（用户：「我不建议用卡1…如果实在不行，就先用卡1」）——
#      那是【用户保留的决定】，本线不自行开启。
MIN_EP="${MIN_EP:-3}"              # V4.1 每卡权重 275.7/EP + 10.4，EP=2 是 148.2 GiB，装不下
# ---- KT_OFFLOAD=1(单卡 MoE offload,09-18 加):必须显式给【一张】卡 ----------------
#   这样 MIN_EP / 下面的自动选卡段对 KT 分支不生效,也就不用改它们 —— 默认路径一个字节不动。
if [ "${KT_OFFLOAD:-0}" = "1" ]; then
  [ -n "${NPUS:-}" ] || { echo "❌ KT_OFFLOAD=1 要显式传 NPUS=<一张卡>,不走自动选卡"; exit 1; }
  [ "${SMOKE:-0}" = "1" ] && { echo "❌ KT_OFFLOAD=1 与 SMOKE=1(dummy 权重)不兼容"; exit 1; }
fi
if [ -z "${NPUS:-}" ]; then
  # LEASE-EXEMPT-BEGIN: 自动选卡是鸡生蛋——要选卡必先查卡，而租约要求先知道卡号。
  #   只做只读的 health/dmesg 查询，不起服务、不分配 HBM。
  #   ⚠️ 我们的发次一律显式传 NPUS，所以这条分支【不执行】；它只在没给 NPUS 时走。
  echo "▶ 没给 NPUS —— 现查健康自动选卡（归属 [$OURS]，不读任何坏卡名单）"
  _ok=""
  for c in $OURS; do
    h=$(timeout 25 npu-smi info -t health -i "$c" 2>&1) || {
      echo "   跳过卡 $c：健康查询失败/超时（查不到 ≠ 好的）"; continue; }
    st=$(awk -F: '/Health Status/{gsub(/ /,"",$2);print $2}' <<< "$h")
    [ "$st" = "OK" ] || { echo "   跳过卡 $c：health=$st"; continue; }
    if ! "$ROOT/build_artifacts/check_card_dmesg.sh" "$c" >/dev/null 2>&1; then
      echo "   跳过卡 $c：health 说 OK 但 dmesg 侧有驱动错误（两条路，卡 1 实证过）"; continue
    fi
    _ok="${_ok:+$_ok }$c"
  done
  # 必须同一 UB 平面（0-3 / 4-7），跨平面 HCCL 挂 EI0006。取健康卡最多的那个平面。
  _p0=""; _p1=""
  for c in $_ok; do [ "$c" -lt 4 ] && _p0="${_p0:+$_p0,}$c" || _p1="${_p1:+$_p1,}$c"; done
  _n0=$(awk -F, '{print NF}' <<< "${_p0:-}"); [ -z "$_p0" ] && _n0=0
  _n1=$(awk -F, '{print NF}' <<< "${_p1:-}"); [ -z "$_p1" ] && _n1=0
  if [ "$_n0" -ge "$_n1" ]; then NPUS="$_p0"; _n=$_n0; else NPUS="$_p1"; _n=$_n1; fi
  echo "   健康的: [$_ok]   平面0-3: ${_p0:-无}   平面4-7: ${_p1:-无}   选中: [${NPUS:-无}]"
  if [ "${_n:-0}" -lt "$MIN_EP" ]; then
    echo "❌ 同一 UB 平面内只有 ${_n:-0} 张健康卡，少于 EP 下限 $MIN_EP —— 起不来，拒绝。"
    echo "   V4.1 每卡权重 = 275.7/EP + 10.4 GiB；EP=2 是 148.2 GiB，单卡 HBM 只有 128 GiB。"
    exit 1
  fi
  # LEASE-EXEMPT-END
fi
PORT=${PORT:-8010}
MAXLEN=${MAXLEN:-8192}
MODEL=${MODEL:-/workspace/models/DeepSeek-V4.1-Flash}
DP=$(awk -F, '{print NF}' <<< "$NPUS")

# ---- 禁区与坏卡：判据构建失败一律 fail-safe ---------------------------------
# 🔴 2026-09-11 10:5x 所有权变更：用户把卡 4/5/6/7 划给我们（经协调方转达）。
#    我起飞前自己量过：四张 health=OK、HBM 各 3%、8003 的 V4.0 基线已不在监听、
#    没有任何进程绑定这些卡 —— 转达的说法和实测一致才照它办。
#    ⚠️ 又一个「写死的快照只在写下它的那一刻是对的」：这行禁区表今天已经反了一次方向。
#    ⚠️ 所有权放开【不等于】健康判据放松：平面 4-7 今天有过阵发故障
#       （4+5 一组、6+7 一组轮流，中间有过整半小时全零）。
#       ⇒ 在这批卡上，起飞前的实时 health + dmesg 检查比在 0-3 上【更】重要。
#    ⚠️ 退路：前置链拦下 4-7 里任何一张 ⇒ 退 EP=3 / 卡 0,1,2（今天全程零错误，脚本验过）。
FORBIDDEN="${FORBIDDEN:-}"
if [ -n "$FORBIDDEN" ]; then
  _bad=$(tr ',' '\n' <<< "$NPUS" | grep -E '^[0-9]+$' | grep -E "^($(tr ' ' '|' <<< "$FORBIDDEN"))$")
  [ -n "$_bad" ] && { echo "❌ 卡 [$(tr '\n' ' ' <<< "$_bad")] 在禁区表 FORBIDDEN=[$FORBIDDEN] 里"; exit 1; }
fi
# ⚠️ 不硬编码坏卡列表。2026-09-10 夜卡 1/7 硬故障，2026-09-11 凌晨复位后恢复 —— 
#    一份写死的清单只在写下它的那一刻是对的。改成【起飞前实时查健康】。
#    查不到 = 判据不成立 = 拒绝（不是放行）。
# 测试夹具：DSV41_FAKE_HEALTH="<卡>:<状态>:<错误码>" 让这个守卫的【拒绝臂】能被真实执行一次。
#   ⚠️ 它【只能把卡伪装成不健康】—— 伪装成 OK 会被拒绝，否则这个夹具本身会变成绕过守卫的后门。
#   为什么需要：我们四张卡碰巧都健康，这个守卫因此从没真正运行过。
#   "没有夹具" ≠ "不可达" —— 卡明天变 Alarm 它就会触发，而那天正好是最不该发现它有 bug 的一天。
_fake_card=""; _fake_st=""; _fake_ec=""
if [ -n "${DSV41_FAKE_HEALTH:-}" ]; then
  IFS=: read -r _fake_card _fake_st _fake_ec <<< "$DSV41_FAKE_HEALTH"
  if [ "$_fake_st" = "OK" ]; then
    echo "❌ DSV41_FAKE_HEALTH 不允许伪装成 OK —— 这个注入点只用于触发拒绝路径，不能绕过守卫"; exit 1
  fi
  echo "  ⚠️ 测试夹具生效: 卡 $_fake_card 被伪装成 $_fake_st（$_fake_ec）"
fi
tr ',' '\n' <<< "$NPUS" | grep -qE '^[0-9]+$' || { echo "❌ NPUS='$NPUS' 解析不出卡号 —— fail-safe"; exit 1; }
[ "$PORT" = "8003" ] && { echo "❌ 8003 是别的 session 的基线"; exit 1; }

# ---- 同 UB 平面：跨平面 HCCL 挂 EI0006 --------------------------------------
# ⚠️ 这条分支【今天没有夹具】，不是不可达 —— 别因为"测不到"就删掉它。
#    跨平面的合法组合需要 4-7 里有我们能用的卡，而 4/5/6 是别人的、7 我们没在用，
#    所以跨平面输入总会先被所有权检查挡下。表达式单独验过（0,2,3→1 个平面；0,4→2 个）。
#    不可达 = 死代码，该删；没夹具 = 活代码，该留并补夹具。这条属于后者。
planes=$(tr ',' '\n' <<< "$NPUS" | awk '{print ($1<4)?0:1}' | sort -u | wc -l)
[ "$planes" -ne 1 ] && { echo "❌ 卡 [$NPUS] 跨 UB 平面(0-3 / 4-7)，HCCL 会挂 EI0006"; exit 1; }

# ---- EP=2 的硬缺口，直接挡下来，别让人花 10 分钟加载再 OOM -------------------
if [ "${KT_OFFLOAD:-0}" = "1" ]; then
  # 单卡 offload:每卡权重不再是 275.7/EP + 10.4,而是 非专家 + N×每专家;另加主机内存(逐 NUMA)闸。
  # 系数与 N 都由 kt_offload_gate.py 当场从权重头与 KT_ADDITIONAL_CONFIG 算出,不写死。
  [ "$DP" -eq 1 ] || { echo "❌ KT_OFFLOAD=1 只支持单卡(DP=1),现在 NPUS=[$NPUS]"; exit 1; }
  MODEL="$MODEL" GPUUTIL="${GPUUTIL:-0.9}" python3 "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/kt_offload_gate.py" \
    || { echo "❌ KT offload 起飞前检查未过(显存/主机内存),拒绝启动"; exit 1; }
elif [ "$DP" -lt 3 ]; then
  echo "❌ EP=$DP：每卡权重 = 275.7/$DP + 10.4 = $(awk "BEGIN{printf \"%.1f\", 275.7/$DP+10.4}") GiB"
  echo "   单卡 HBM 只有 128 GiB —— 装不下。V4.1 至少要 EP=3。"; exit 1
fi


# 🔴 2026-09-14：顺序被改过 —— 【先拿租约，再碰设备】。
#   原来的顺序是 健康查询(npu-smi) → dmesg 检查 → acquire。
#   实测：xr_rid2_025007 在 02:50:07 起飞，而租约直到 02:52:43 才被取得
#   ⇒ 中间【2 分 36 秒】租约是空的 ⇒ 别人 acquire 会【合法成功】并上卡。
#   而那 2.5 分钟正好覆盖 HCCL 建链窗口（上一发就是起飞后 2~3 分钟建链失败）。
#   ⚠️ 这不是『谁该记得』的问题 —— 对方是严格照租约办事的。
#      **租约在那 2.5 分钟里没有代表真相。**
#   ⇒ 守卫：build_artifacts/check_acquire_first.py（改之前先在旧脚本上验它会红）

# ---- 占卡租约：两条线都能起服务，口头协调挡不住【同时】起 -------------------
#   同时起的症状很难看：端口占用、HBM 冲突、dmesg 归因混乱（谁的错误算谁的分不清）。
#   LEASE_PID=$$ —— 脚本末尾是 exec，PID 不变，所以租约的存活判据在服务运行期间有效。
#   判据构建失败一律拒绝（租约目录不可写 / 卡号解析不出 / 持有者进程查不了）。
LEASE_OWNER="${LEASE_OWNER:-dsv41-$WT}"
if [ "${LEASE:-1}" = "1" ]; then
  LEASE_PID=$$ "$ROOT/build_artifacts/card_lease.sh" acquire "$NPUS" "$LEASE_OWNER" "$PORT" || {
    echo "❌ 拿不到占卡租约 —— 拒绝启动（别和另一条线撞在同一批卡上）"; exit 1; }
  # 🔴 2026-09-13 实测：这条 trap 【永远不会执行】。
  #    本脚本末尾是 `exec vllm serve …`，而 exec 用新程序替换进程映像 ——
  #    bash 的 EXIT trap 随之消失。实测：`bash -c 'trap "echo X" EXIT; exec true'` 不打印任何东西。
  #    ⚠️ 它看起来完全正确（代码里明明写着释放），所以残留租约会被当成"别的线忘了清"。
  #    ⇒ exec 是刻意的（租约的存活判据要求 PID 不变），所以不动它；
  #      **释放由调用方在 stop_by_port 之后做**（probe_cross_restart.sh 已加）。
  #      留着这条 trap 只为覆盖"还没走到 exec 就退出"的那些早退路径。
  trap '"$ROOT/build_artifacts/card_lease.sh" release "$NPUS" "$LEASE_OWNER" >/dev/null 2>&1' EXIT
fi

for card in $(tr ',' '\n' <<< "$NPUS"); do
  if [ "$card" = "$_fake_card" ]; then
    st="$_fake_st"; h="Error Code : $_fake_ec"
  else
  h=$(timeout 20 npu-smi info -t health -i "$card" 2>&1) || {
    echo "❌ 卡 $card 健康查询失败/超时 —— 判据不成立，fail-safe 拒绝启动"; exit 1; }
  st=$(awk -F: '/Health Status/{gsub(/ /,"",$2);print $2}' <<< "$h")
  fi
  case "$st" in
    OK) ;;
    "") echo "❌ 卡 $card 解析不出 Health Status —— 判据不成立，拒绝"; exit 1;;
    *)  echo "❌ 卡 $card 健康状态 = $st（$(awk -F: '/Error Code/{gsub(/ /,"",$2);print $2}' <<< "$h")）"
        echo "   用非 OK 的卡起服务，得到的报错会和真实原因无关"; exit 1;;
  esac
done

# ---- dmesg 侧：health 只覆盖一个子系统，不够 --------------------------------
#   卡 1 实证：health 曾一度答 OK，而 udis 状态机从 09-10 16:24 起就没好过。
#   后来 health 能拦住它是因为它烂到连 health 都超时了 —— 那是运气不是设计。
#   ⇒ 判据覆盖的子系统数，要和结论需要的子系统数对得上。
"$ROOT/build_artifacts/check_card_dmesg.sh" "$NPUS" || {
  echo "❌ dmesg 侧检查未过，拒绝启动"; exit 1; }


# ---- 等卡空闲：只查指定卡 + timeout（全量 npu-smi 会被坏卡挂死）--------------
command -v npu-smi >/dev/null 2>&1 || { echo "❌ 没有 npu-smi，判据不成立 —— fail-safe"; exit 1; }
for card in $(tr ',' '\n' <<< "$NPUS"); do
  waited=0
  while :; do
    out=$(timeout 20 npu-smi info -t usages -i "$card" 2>/dev/null) || {
      echo "❌ 查不到卡 $card（超时）—— 判据不成立，fail-safe"; exit 1; }
    pct=$(awk -F: '/HBM Usage Rate/{gsub(/ /,"",$2); print $2; exit}' <<< "$out")
    [[ "$pct" =~ ^[0-9]+$ ]] || { echo "❌ 卡 $card 的 HBM 占用解析不出 —— fail-safe"; exit 1; }
    [ "$pct" -lt "${IDLE_PCT:-20}" ] && { echo "  卡 $card 空闲 (HBM ${pct}%)"; break; }
    [ "$waited" -ge "${WAIT_S:-300}" ] && { echo "❌ 卡 $card 等了 ${WAIT_S:-300}s 仍占 ${pct}%"; exit 1; }
    echo "  等卡 $card 释放… 当前 HBM ${pct}% (已等 ${waited}s)"; sleep 5; waited=$((waited+5))
  done
done

set +u
source "$ROOT/build_artifacts/env_dsv41.sh" "$WT" "$NPUS" || exit 1
set +u

# ---- 产物新鲜度 + 树来源：两条都必须过，否则起来的也是错的 -------------------
"$ROOT/build_artifacts/verify_shared_artifacts.sh" >/dev/null || {
  echo "❌ 共享产物检查未过 —— 多半是源码合了但 .so 没重编。跑一次看详情："
  echo "   $ROOT/build_artifacts/verify_shared_artifacts.sh"; exit 1; }
python3 "$ROOT/build_artifacts/assert_trees.py" || { echo "❌ 树来源断言失败，拒绝启动"; exit 1; }

# ---- 要跑的那棵树不能落后于 dsv41-base ---------------------------------------
#   assert_trees.py 只回答「是不是我们的树、CANN 对不对」，不回答「这棵树新不新」。
#   而落后的树【起得来】—— 它只是缺代码，症状是一个来自上游的、看起来很专业的 ValueError。
#   判据用 commit 祖先关系，不用时间戳（时间戳会被 cp/checkout 刷新）。
_TREE=$("$ROOT/build_artifacts/_resolve_tree.sh" "$WT") || { echo "❌ 解析不出树: $WT"; exit 1; }
_BASE=$(git -C "$ROOT/vllm-ascend" rev-parse dsv41-base 2>/dev/null)
_HEAD=$(git -C "$_TREE" rev-parse HEAD 2>/dev/null)
if [ -z "$_BASE" ] || [ -z "$_HEAD" ]; then
  echo "❌ 取不到 dsv41-base 或 $WT 的 HEAD —— 判据不成立，拒绝启动"; exit 1
fi
if ! git -C "$ROOT/vllm-ascend" merge-base --is-ancestor "$_BASE" "$_HEAD" 2>/dev/null; then
  echo "❌ 要跑的树 $WT ($(git -C "$_TREE" rev-parse --short HEAD)) 【落后于】 dsv41-base (${_BASE:0:9})"
  echo "   落后的树起得来，它只是缺代码 —— 2026-09-11 11:09 实证：缺 attn_selector 时"
  echo "   上游 flashmla 没被劫持，报的是 'fp8_ds_mla layout only supports fp8 kv-cache'，"
  echo "   一个看起来很专业、但和我们的接线毫无关系的 ValueError。"
  echo "   ⇒ 用 WT=base，或把该分支合到 base 之后再跑。"
  exit 1
fi
echo "  ✅ 要跑的树 $WT ($(git -C "$_TREE" rev-parse --short HEAD)) 包含 dsv41-base (${_BASE:0:9})"

# ---- engram：必须是【全量】，不是切片 ---------------------------------------
#   切片和全量在日志里长得一样，而下游 mmap 到截断张量【不报错，只算错】。
#   ⚠️ tmpfs 在内存里，重启即失效 —— 重启后这里会是空目录，也会被同一条判据拦下。
# 🔴 2026-09-15 加 SKIP_SHM_GATE（默认 0，对其它线零影响）:
#    这条闸会起 DP 个子进程、每个【映射并 touch 188.8 GiB】—— 一次巨大的
#    瞬时内存事件。实测在共享机器上连杀三发:MemFree 起飞时 160 GB,
#    serve_1.log 一行没写就被宿主低内存杀手打掉,死的正是这一级。
#    ⚠️ 跳过它【放弃的是】"engram 是全量且共享真的生效"这条保证 ——
#      而那两件事今天已被同一条闸验过多次、engram 铺设期间没有改动过。
#      所以只在【诊断飞行】里跳，起基线/发结论的飞行不许跳。
#      跳过时必须在日志里留痕，否则"没验"和"验过了"读起来一样。
if [ "${SKIP_SHM_GATE:-0}" = "1" ]; then
  echo "  ⚠️ SKIP_SHM_GATE=1：【跳过】engram 全量/共享校验。"
  echo "     放弃的保证：表是全量、N 个 rank 真的共享一份。"
  echo "     只有诊断飞行可以这么做；本次读数不得用来下'共享生效'一类的结论。"
else
python3 "$ROOT/build_artifacts/verify_shm_sharing.py" --ranks "$DP" --require-full || {
  echo "❌ engram 表检查未过（不是全量 / 共享没生效 / 目录不存在）—— 拒绝启动"
  echo "   重铺: python3 $ROOT/build_artifacts/stage_engram_shm.py --full --i-have-user-approval"
  exit 1; }
fi

# ---- dmesg 基线：三张卡是在没人看 dmesg 的时候静默死的，这次让它有 ---------
UDIS_BEFORE=$(dmesg 2>/dev/null | grep -ci "udis")
echo "  dmesg udis 基线行数 = ${UDIS_BEFORE:-读不到}"
echo "  ⚠️ 服务退出后请对比: dmesg | grep -ci udis   （涨了说明期间有新的驱动错误）"
echo "$UDIS_BEFORE" > "${UDIS_FILE:-/tmp/udis_before_$PORT}"

# ---- 起飞时刻的卡快照落盘：中途挂了要能分辨「卡没了」和「程序挂了」 ---------
#   前面的检查只是【拦不拦】，拦过就丢掉了。这里把那一刻的状态留下来当基线。
SNAP="${SNAP:-$ROOT/build_artifacts/m1_run/cards_at_launch_$PORT.txt}"
mkdir -p "$(dirname "$SNAP")"
{
  echo "# 起飞时刻卡快照  $(date -Is)  port=$PORT  NPUS=$NPUS  dp=ep=$DP"
  for c in $(tr ',' '\n' <<< "$NPUS"); do
    h=$(timeout 20 npu-smi info -t health -i "$c" 2>&1)
    u=$(timeout 20 npu-smi info -t usages -i "$c" 2>/dev/null)
    printf "card %s  health=%s  hbm=%s%%\n" "$c" \
      "$(awk -F: '/Health Status/{gsub(/ /,"",$2);print $2}' <<< "$h")" \
      "$(awk -F: '/HBM Usage Rate/{gsub(/ /,"",$2);print $2;exit}' <<< "$u")"
  done
  echo "dmesg udis 总行数 = $(dmesg 2>/dev/null | grep -ci udis)"
  grep -E "^Shmem:|^ShmemHugePages" /proc/meminfo
} | tee "$SNAP"
echo "  卡快照已写入 $SNAP"

# ---- 第一次前向的采集挂上去：成功那一刻信息最密，而那时人只会想「终于起来了」----
if [ "${CAPTURE:-1}" = "1" ]; then
  # 🔴 守望壳必须随主脚本一起死。实测泄漏（2026-09-11）：主脚本被 timeout 杀掉后，
  #    这个子壳被 reparent 到 1 并继续轮询了十几分钟 —— 期间若别人在同一端口起了服务，
  #    它会对着别人的服务跑采集。后台看门狗不设存活条件，就是一个会自己活下去的东西。
  #    脚本末尾是 exec，$$ 在 exec 后不变，所以这个判据在服务运行期间依然有效。
  _MAIN_PID=$$
  ( for _ in $(seq 1 "${CAPTURE_WAIT_S:-1800}"); do
      # ⚠️ kill -0 对【僵尸】也返回成功（pid 还在，进程已死）。实测就撞到了：
      #    主脚本变成 Z 之后守望壳还活着。所以再查一次 /proc 的 State。
      kill -0 "$_MAIN_PID" 2>/dev/null || exit 0
      grep -qE '^State:[[:space:]]+Z' "/proc/$_MAIN_PID/status" 2>/dev/null && exit 0
      curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && {
        sleep 5
        PORT="$PORT" NPUS="$NPUS" EP="$DP" LOG="${SERVE_LOG:-}" \
          "$ROOT/build_artifacts/capture_first_forward.sh" \
          > "$ROOT/build_artifacts/m1_run/capture_$PORT.log" 2>&1
        exit 0; }
      sleep 1
    done
    echo "capture: 等了 ${CAPTURE_WAIT_S:-1800}s 服务没起来，不采集" \
      > "$ROOT/build_artifacts/m1_run/capture_$PORT.log"
  ) &
  echo "  已挂上第一次前向采集（后台，最多等 ${CAPTURE_WAIT_S:-1800}s）"
  echo "  ⚠️ 它只采集不改任何东西；CAPTURE=0 可关"
fi

if [ "${SMOKE:-0}" = "1" ]; then
  echo "🟡 ============================================================"
  echo "🟡 SMOKE=1：dummy 权重（--load-format dummy）"
  echo "🟡   本轮只回答【会不会崩、崩在哪】，【不回答】任何数值问题。"
  echo "🟡   ⛔ 任何精度 / 正确性 / 显存账 结论【不得】引用本轮 —— 权重是随机的。"
  echo "🟡   ⏭  阶梯里「权重加载」那一级本轮是【跳过】，不是【通过】。"
  echo "🟡 ============================================================"
fi
echo "▶ V4.1 bring-up  worktree=$WT  卡=$NPUS  dp=ep=$DP  tp=1  port=$PORT  max-model-len=$MAXLEN${SMOKE:+  SMOKE=$SMOKE}"
echo "  预期每卡权重 ≈ $(awk "BEGIN{printf \"%.1f\", 275.7/$DP+10.4}") GiB（engram 189.1 GiB 应在 host，不占 HBM）"
echo "  ⚠️ 起飞是一次性占满四张卡的操作；异常就停，不要「再试一次看看」。"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "🟡 DRY_RUN=1 —— 前置链全部跑完且通过，【不起服务】。"
  echo "   真起: DRY_RUN=0 NPUS=$NPUS ./build_artifacts/serve_v41_bringup.sh"
  exit 0
fi

# SMOKE=1 用 --load-format dummy,而 dummy 明确拒绝 --model-loader-extra-config
# ("Model loader extra config is not supported for load format dummy")。
# 🔴 这两个开关一直是互斥的,而脚本无条件传后者 ⇒ SMOKE 这条路【从来没通过】,
#    第二十次起飞才第一次真的去走它。一个没人走过的分支,和一个能走的分支,
#    在脚本里长得一模一样。
# 🔴 2026-09-15 加 LOADER_THREADS(默认仍是 64,对其它线零影响):
#    两发连续在【权重加载完成后 ~25 秒】被宿主判定内存不足而杀掉。
#    64 个线程同时读 safetensors 是一个很大的瞬时宿主内存峰值,而这台机器
#    是共享的(同期另一个容器在跑 4×88 GB)。降线程数是拿加载时间换峰值。
#    ⚠️ 这是【假设】不是结论:证据只是"两次都死在加载后不久"。
#       所以起飞时同步采样 MemFree —— 峰值真的贴地才算证实。
LOADER_THREADS="${LOADER_THREADS:-64}"
LOADER_ARGS=()
if [ "${SMOKE:-0}" != "1" ]; then
  LOADER_ARGS=(--model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": ${LOADER_THREADS}}")
fi

# ── 🔴 两个【配置缺失】，2026-09-16 实测补上 ──────────────────────────────
# 它们不是调优项：不设的后果都被伪装成别的东西，很难从症状回溯到这里。

# ① jemalloc 必须摘掉。容器把它写进环境
#    (LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:)，
#    后果是 worker 在【365~430 个请求】之后 segfault，而 faulthandler 打的
#    Python 栈是假的(si_code=-6 SI_TKILL)，要 gdb 读 core 才看得到 libjemalloc。
#    ⚠️ 坎的位置很坑：单个压测用例最多 256 个请求，【刚好在坎下面】，
#       所以压测全绿而 GSM8K(1319 题)必炸。unset 后实测 1319 题零崩、
#       且权重加载反而更快(54~64s vs 109s)。
unset LD_PRELOAD

# ② HCCL 通信缓冲。不设就用 CANN 默认值，而 MC2 的 MoE dispatch 需要的缓冲
#    随每 rank 批量线性增长：
#        NEEDED = ep_worldsize × maxBs × Align512(h×2) × 2 × localMoeExpertNum
#      本配置(EP=4, h=5120, 384 专家/4 = 96 局部专家) ⇒ 约 7.5 MB × maxBs
#      ⇒ 2048 MB 可支持 maxBs ≈ 273；要上 512 得调到 4096。
#      (公式出自 CANN host 侧 libophost_transformer.so 的
#       "HCCL_BUFFSIZE is too SMALL, ... = %luMB" 报错串。)
#    🔴 不设的症状极具误导性：--max-num-seqs 64 时挂在 profile_run 的
#       npu_moe_distribute_dispatch_v2，算子只报
#       "NnopbaseExecutorDoTiling failed"、【不给任何维度】，
#       读起来完全像"平台不支持大批"。实测把它设成 2048 之后
#       64/128/256 全部由失败转成功(并发 128 → 1195 tok/s)。
#    ⚠️ 代价是设备内存：实测 2048 MB 的设置吃掉 3.61 GiB 的 KV 余量
#       (28.15 → 24.54 GiB，不止一个通信域各分一块)，所以不要盲目调大。
#    vllm-ascend 自己的 e2e 测试在 MoE/EP 配置下也都显式设它(512~2048)。
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-2048}"
echo "  ⚪ LD_PRELOAD 已 unset(jemalloc)；HCCL_BUFFSIZE=${HCCL_BUFFSIZE} MB"
# ─────────────────────────────────────────────────────────────────────────

if [ "${KT_OFFLOAD:-0}" = "1" ]; then
  # 单卡 MoE offload(M1):TP1/DP1、不开 EP、强制 eager、kt_offload_config 走 --additional-config。
  # kt_site 追加在 PYTHONPATH 末尾(不前置,免得它遮住 vllm/vllm_ascend)。
  export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/build_artifacts/moe_offload_ref/kt_site"
  echo "  ⚪ KT_OFFLOAD=1:单卡 offload 起法(DP=1、无 EP、--enforce-eager);kt_site 已追加到 PYTHONPATH"
  exec vllm serve "$MODEL" \
    --served-model-name dsv41 \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len "$MAXLEN" \
    --max-num-batched-tokens "${MAXBATCH:-4096}" \
    --max-num-seqs "${MAXSEQS:-8}" \
    --gpu-memory-utilization "${GPUUTIL:-0.9}" \
    --kv-cache-dtype bfloat16 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    "${LOADER_ARGS[@]}" \
    --enforce-eager \
    --additional-config "$KT_ADDITIONAL_CONFIG" \
    ${EXTRA_ARGS:-} \
    --port "$PORT"
fi

exec vllm serve "$MODEL" \
    --served-model-name dsv41 \
    --tensor-parallel-size 1 \
    --data-parallel-size "$DP" \
    --enable-expert-parallel \
    --max-model-len "$MAXLEN" \
    --max-num-batched-tokens "${MAXBATCH:-4096}" \
    --max-num-seqs "${MAXSEQS:-8}" \
    --gpu-memory-utilization "${GPUUTIL:-0.9}" \
    --kv-cache-dtype bfloat16 \
    --block-size 128 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    "${LOADER_ARGS[@]}" \
    ${SMOKE:+$([ "$SMOKE" = "1" ] && echo --load-format dummy)} \
    ${EAGER:+$([ "$EAGER" = "1" ] && echo --enforce-eager)} \
    ${EXTRA_ARGS:-} \
    --port "$PORT"
