#!/usr/bin/env bash
# ============================================================================
#  单卡起服务的两道起飞前闸（本线自有，2026-09-22）
#
#  抽成独立文件，是为了能【不起服务】就单独验：
#      bash tools/kt_offload/serve_preflight_gates.sh so
#      DS41=/workspace/code/dsv4_single_card/ds41_site \
#        bash tools/kt_offload/serve_preflight_gates.sh ds41
#      ... all   （两道都跑）
#  serve_v41_singlecard.sh 起服务前 source 本文件并调用这两个函数。
#  rc: 0 放行 / 非 0 拒绝起飞（调用方 exit）。
#
#  ① gate_so —— 本树 .so 的 md5 必须等于期望值
#     🔴 旧 .so + 新代码 = 静默算错，不是崩。clean 在 f85267ab8 改了 csrc
#        （op_api_common.h 的 4-bit nibble 分支、float4_e2m1fn_x2 喂新算子），
#        我们 rebase 到它之后，树里的 .so 仍是旧的 2b58b578e4fc…，
#        那几个符号 0 命中 ⇒ 跑得起来、数是错的。
#     期望值默认留成【待定占位】，所以 rebase 后、换 .so 前，本闸必然拒绝。
#     换完 .so 由协调方给出新 md5，改 SERVE_EXPECT_SO_MD5 的默认值（或临时用
#     env 覆盖）。写法照 oncard 的 ONCARD_EXPECT_SO_MD5（RUNBOOK_window.md）。
#     ⚠️ 期望值是【本树】的（k.14）：clean 已经换过钉值，我们没有 ⇒ 打印时
#        一定带上被检查的树路径。
#
#  ② gate_ds41 —— ds41 算子包（O6：>16k 长上下文）必须真的在 PYTHONPATH 前面
#     🔴 缺了不报错：五个候选层静默退回 v2、max-model-len 仍被 16384 守卫卡住。
#     🔴 判据读真实对象（k.6 那一族）：不查 env、不查目录存在，查
#        【解释器实际会解析到的那个文件】—— find_spec().origin，以及真 import
#        一个轻量子包看它的 __file__。CANN 自带一份同名的 cann_ops_transformer
#        （set_env.sh 会把 9.2 的 site-packages 插到 PYTHONPATH 最前），
#        装到解释器里会被它【静默遮住】：import 成功、ds41 一个都没有。
#     ⚠️ 故意【不】import cann_ops_transformer 本身：它在 import 期就 JIT
#        编译扩展（实测两个 .so），起飞前不该顺手编一份；它按 origin 判。
# ============================================================================

# 🔴 新 .so 待定：这个占位值不可能等于任何真实 md5 ⇒ 现在起飞必然 rc≠0。
#    换 .so 之后由协调方给值再改这里（同一发里 oncard 的 ONCARD_EXPECT_SO_MD5
#    也要一起改，两处期望值不许只改一处）。
SERVE_EXPECT_SO_MD5="${SERVE_EXPECT_SO_MD5:-PENDING_NEW_SO_AFTER_f85267ab8}"

gate_so() {
  local tree="$1"
  [ -n "$tree" ] || { echo "🔴 gate_so: 没给树路径 —— 判据不成立，拒绝起飞"; return 2; }
  local sos=()
  while IFS= read -r f; do sos+=("$f"); done < <(ls -1 "$tree"/vllm_ascend/vllm_ascend_C*.so 2>/dev/null)
  if [ "${#sos[@]}" -eq 0 ]; then
    echo "🔴 gate_so: $tree/vllm_ascend 下没有 vllm_ascend_C*.so —— 读不到就当不匹配（fail-safe）"
    return 3
  fi
  if [ "${#sos[@]}" -gt 1 ]; then
    echo "🔴 gate_so: 找到 ${#sos[@]} 份 .so，说不清哪份会被 import：${sos[*]}"
    return 3
  fi
  local got
  got=$(md5sum "${sos[0]}" | awk '{print $1}')
  echo "  gate_so: tree=$tree"
  echo "           so=${sos[0]}"
  echo "           md5=$got  期望=$SERVE_EXPECT_SO_MD5（本树期望值；别的树可能不同）"
  if [ "$got" != "$SERVE_EXPECT_SO_MD5" ]; then
    echo "🔴 拒绝起飞：.so 不是期望的那份。"
    echo "   旧 .so + 新代码 = 【静默算错】，不是崩：f85267ab8 改了 csrc"
    echo "   （4-bit nibble 分支 / float4_e2m1fn_x2 喂新算子），旧 .so 里那些符号 0 命中，"
    echo "   服务照样起来、GSM8K 照样出分，而那个分数什么都没测到。"
    if [ "$SERVE_EXPECT_SO_MD5" = "PENDING_NEW_SO_AFTER_f85267ab8" ]; then
      echo "   ⚠️ 期望值仍是【待定占位】：rebase 到 f85267ab8 之后、换 .so 之前，"
      echo "      本闸【必然】拒绝，这是设计。换包要与 dsv4.1_flash_dev 协调"
      echo "      （换包 → 重跑 GSM8K → 再改钉值），拿到新 md5 后改本文件默认值。"
    fi
    return 1
  fi
  echo "  ✅ gate_so: .so 与本树期望值一致"
  return 0
}

gate_ds41() {
  # DS41 未设 ⇒ 按 clean 的口径默认打开（cookbook/serve_multicard.sh ③.1）；
  # DS41=0 / DS41="" 是显式关闭（单变量对照），此时只告警不拒绝。
  DS41="${DS41-/workspace/code/dsv4_single_card/ds41_site}"
  export DS41
  [ "$DS41" = "0" ] && DS41=""
  if [ -z "${DS41:-}" ]; then
    echo "⚠️  gate_ds41: DS41 已显式关闭 —— 本发没有 ds41 算子：五个候选层走 v2、"
    echo "    max-model-len 受 16384 守卫约束。这是对照口径，不是生产口径（指南 §3.9）。"
    return 0
  fi
  [ -d "$DS41" ] || { echo "🔴 gate_ds41: DS41=$DS41 不是目录 —— ds41 算子包必装（指南 §3.9）"; return 1; }
  export PYTHONPATH="$DS41:$PYTHONPATH"
  python3 - <<'PYCHK' || return 1
import importlib.util as u
import os
import sys

ds41 = os.path.realpath(os.environ["DS41"])
bad = []
for name in ("cann_ops_transformer", "cannbotdsl", "ops"):
    try:
        spec = u.find_spec(name)
    except Exception as e:                       # noqa: BLE001
        bad.append(f"{name}: find_spec 抛 {e!r}")
        continue
    origin = getattr(spec, "origin", None) if spec else None
    if not origin or not os.path.realpath(origin).startswith(ds41):
        bad.append(f"{name}: 解析到 {origin}")
    else:
        print(f"  ✅ gate_ds41: {name} -> {origin}")

# 真 import 一个轻量子包：find_spec 只说"会解析到哪"，import 还会走
# __init__ 里的路径拼装。cann_ops_transformer 本身 import 期会 JIT 编译扩展，
# 起飞前不顺手编，所以它只按 origin 判。
try:
    import cannbotdsl
    where = os.path.realpath(cannbotdsl.__file__)
    print(f"  ✅ gate_ds41: import cannbotdsl 的 __file__ = {where}")
    if not where.startswith(ds41):
        bad.append(f"cannbotdsl.__file__ = {where}")
except Exception as e:                           # noqa: BLE001
    bad.append(f"import cannbotdsl 失败：{e!r}")

if bad:
    print("🔴 拒绝起飞：ds41 算子包不是解释器实际会用的那一份 ——")
    for b in bad:
        print("   - " + b)
    print("   多半是被 CANN 自带的同名包遮住了（set_env.sh 会把它的 site-packages")
    print("   插到 PYTHONPATH 最前）。这条失败是【静默】的：服务照样起来，只是")
    print("   五个候选层退回 v2、>16k 被守卫挡住，症状伪装成「算子不支持」。")
    sys.exit(1)
PYCHK
  echo "  ✅ gate_ds41: ds41 包在 PYTHONPATH 前面（DS41=$DS41）"
  return 0
}

# ── gate_shared：共享闸的【窄放行】（用户 2026-09-22 授权）───────────────────
#   现状：build_artifacts 仓 2e21d32（09-21 清理）删了 shared_so_manifest.txt ⇒
#   verify_shared_artifacts.sh 对所有人 rc=1，而它报的只有两条 ❌，判的都是
#   build_artifacts/ 那份【共享】.so 的出处。本线这一发加载的是 $_TREE 自己的 .so，
#   由其后必跑的 gate_so（md5 期望值）判；共享 .so 本发根本不 import。
#   用户决定：不恢复 manifest、不改共享脚本，只在本线 launcher 放行【这两条且仅这两条】。
#   🔴 fail-closed：rc 不是 1、没有 ❌、或出现白名单之外任何一条 ❌ ⇒ 照旧拒绝。
#   🔴 放行的前提是 gate_so 在起服务前一定会跑（launcher 里它无条件 `|| exit 1`）；
#      放行后设 SHARED_GATE_WAIVED=1，launcher 据此再断言一次 gate_so 已通过。
#   白名单是逐字子串（取自 verify_shared_artifacts.sh 的原文输出）：
GATE_SHARED_WAIVABLE=(
  "取不到 base 的 csrc 最新提交或 manifest 的 built_from"
  "build_artifacts/shared_so_manifest.txt —— .so 没有出处"
)
gate_shared() {  # <共享闸完整输出文件> <它的 rc>
  local out="$1" rc="$2" line w hit n=0
  SHARED_GATE_WAIVED=0
  [ "$rc" = 0 ] && { echo "  ✅ gate_shared: 共享闸 rc=0"; return 0; }
  [ -r "$out" ] || { echo "🔴 gate_shared: 读不到共享闸输出 —— 拒绝"; return 1; }
  [ "$rc" = 1 ] || { echo "🔴 gate_shared: 共享闸 rc=$rc（不是 1）—— 不在放行范围，拒绝"; return 1; }
  while IFS= read -r line; do
    n=$((n + 1)); hit=0
    for w in "${GATE_SHARED_WAIVABLE[@]}"; do
      [[ "$line" == *"$w"* ]] && { hit=1; break; }
    done
    [ "$hit" = 1 ] || { echo "🔴 gate_shared: 白名单之外的 ❌ —— 拒绝："; echo "     $line"; return 1; }
  done < <(grep '❌' "$out")
  [ "$n" -ge 1 ] || { echo "🔴 gate_shared: rc=1 却没有任何 ❌ 行 —— 解析不了就不放行"; return 1; }
  SHARED_GATE_WAIVED=1
  echo "=============================================================================="
  echo "  🟠 共享闸【窄放行】：$n 条 ❌ 全部属于「共享 .so 出处未验」"
  echo "     （shared_so_manifest.txt 缺失，build_artifacts 2e21d32 删除；用户 2026-09-22 授权）"
  echo "     本发【不加载】共享 .so；本发实际加载的 .so 由其后的 gate_so 按 md5 判定。"
  echo "     其余共享判据（各树钉值、覆盖数）已在同一份输出里通过。产物里请连这一行一起引。"
  echo "=============================================================================="
  return 0
}

# 直接跑本文件时当独立小脚本用（不起服务，可在 CPU 上验）
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  _rc=0
  case "${1:-all}" in
    so)   gate_so "${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}" || _rc=$? ;;
    ds41) gate_ds41 || _rc=$? ;;
    shared) gate_shared "${2:?共享闸输出文件}" "${3:?共享闸 rc}" || _rc=$? ;;
    all)  gate_so "${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}" || _rc=$?
          gate_ds41 || _rc=$? ;;
    *)    echo "用法: $0 [so|ds41|all] [tree] | shared <输出文件> <rc>"; _rc=2 ;;
  esac
  exit $_rc
fi
