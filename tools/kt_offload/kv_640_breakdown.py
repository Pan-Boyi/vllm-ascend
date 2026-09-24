#!/usr/bin/env python3
"""把「27.40 GiB / 44,872 token = 640.3 KiB/token」逐字节还原。只读、不上卡、不起服务。

为什么是独立一个脚本、而不是塞进 kv_budget_singlecard.py：
  那个脚本的职责是**判据**（重算 == 日志，对不上就 raise），它的 `--log` 前面有一道
  构建指纹闸。本脚本的职责是**拆解**——把 640.3 KiB 拆成「真内容 / admission 超额 /
  页 padding / 池块计费」四类并给残差，以及把 640.3 vs §4.4 的 343.8 拆成两个因子。
  两件事的失败模式不同，混在一起会让判据的绿灯含义变模糊。

几何与页字节全部取自 kv_budget_singlecard 的 real_specs()/real_planner()
（真实 cache 对象 + 本树 patch 过的 planner），本脚本自己不写任何页字节常数。

用法：
    PYTHONPATH=<worktrees/moe-offload-vllm>:<本树> LD_PRELOAD= \
        python3 tools/kt_offload/kv_640_breakdown.py

🔴 每个输出数只在它自己的操作点上成立。本脚本默认的操作点是
   `p2_prof_0922_131258` 那一发：max_model_len 8192、--block-size 128、
   --kv-cache-dtype bfloat16、mnbt 4096（async scheduling 开 ⇒ in-flight 8192）、
   SWA block_size 32、index-K MXFP4（68 B/state）。换任何一项都要重跑，
   **尤其是 max_model_len**——每 token 计费对它近似反比（见 §6 的拆解）。
"""
from __future__ import annotations

import importlib.util
import os
import sys

GiB = 1 << 30
KiB = 1024

_TOOL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "kv_budget_singlecard.py")


def _load_kvb():
    spec = importlib.util.spec_from_file_location("kv_budget_singlecard", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    # 必须先进 sys.modules：该模块里有 @dataclass，dataclasses 会回查
    # sys.modules[cls.__module__]。
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


kvb = _load_kvb()
cdiv = kvb.cdiv

# ---- 操作点（这一发的启动参数；日志 serve.log:119 的 non-default args）----
L = 8192            # --max-model-len
MNBT = 4096         # --max-num-batched-tokens
INFLIGHT = 2 * MNBT  # async scheduling 开 ⇒ max_in_flight_tokens = 2×mnbt
SWA_BS = 32         # 上游构造点写死（attention.py:456），本树开关默认 0=不改
NL = 40             # num_hidden_layers
SW = 128            # sliding_window
HD = 512            # head_dim
# 日志读数（p2_prof_0922_131258/serve.log:488,489）
LOG_TOKENS = 44_872
LOG_AVAIL_GIB = 27.40
LOG_CONC = 5.48


def main() -> int:
    P_MLA, P_IDX, P_RING, N_RING = kvb.P_MLA, kvb.P_IDX, kvb.P_RING, kvb.N_RING
    SBS = kvb.STORAGE_BS
    IDX_STATE = P_IDX // SBS          # index-K 每 state 字节（MXFP4 时 68）
    lay = kvb.build_layout(L, INFLIGHT, swa_bs=SWA_BS)
    BPB, BPR = lay.bytes_per_block, lay.blocks_per_req
    charged = BPR * BPB
    bt = charged / L

    print("=== 1. 池块字节与每请求块数（页字节取自真实 spec；已被本树 planner 复核）")
    print(f"  P_MLA={P_MLA:,}  P_IDX={P_IDX:,}({IDX_STATE} B/state)  "
          f"P_RING={P_RING:,}×{N_RING}  storage_block={SBS}")
    print(f"  池块 = ({P_IDX:,}+{P_MLA:,})×{lay.num_tuple_slots} + "
          f"{lay.ring_bytes_per_block:,} = {BPB:,} B")
    print(f"  blocks/req = {cdiv(L,SBS)} + {cdiv(L,2*SBS)} + "
          f"{NL}×{lay.swa_blocks_per_req} + 1 = {BPR:,}")
    print(f"  ⇒ 每 token 计费 = {BPR:,}×{BPB:,}/{L} = {bt:,.1f} B = "
          f"{bt/KiB:.2f} KiB    【日志 640.3】")

    print("\n=== 2. 反解 num_blocks（tokens 是整数，能把它钉死）")
    cands = [nb for nb in range(1, 1 << 20) if int(nb / BPR * L) == LOG_TOKENS]
    assert cands, "反解不出 num_blocks ⇒ 几何判错了"
    nb = cands[0]
    alloc = nb * BPB
    print(f"  能重现 tokens={LOG_TOKENS:,} 的 num_blocks = {cands}")
    print(f"  分配 = {nb:,}×{BPB:,} = {alloc:,} B = {alloc/GiB:.4f} GiB  "
          f"【日志 {LOG_AVAIL_GIB}】")
    print(f"  concurrency = {nb/BPR:.4f} → 打印 {round(nb/BPR,2)}x  【日志 {LOG_CONC}】")

    print(f"\n=== 3. 640.3 KiB 逐项还原（每请求，L={L}）")
    c_kv = 3 * (L // 2) * HD * 2 + L * HD * 2       # compress_kv：3 个 ratio2 + 1 个 ratio1
    c_idx = 3 * (L // 2) * IDX_STATE + L * IDX_STATE
    c_swa = NL * SW * HD * 2                        # 定长滑窗真窗口
    c_ring = N_RING * P_RING                        # ring 每请求 1 块 ×3 层
    real = c_kv + c_idx + c_swa + c_ring
    swa_admit = NL * lay.swa_blocks_per_req * SWA_BS * HD * 2
    swa_pad = NL * lay.swa_blocks_per_req * P_MLA
    swa_chg = NL * lay.swa_blocks_per_req * BPB
    g01_real = 64 * (P_MLA + P_IDX) + 32 * 3 * (P_MLA + P_IDX)
    g01_chg = (64 + 32) * BPB
    items = [
        ("A1 真内容 compress_kv（4 源层 bf16 2 B/elem）", c_kv),
        (f"A2 真内容 index_k（4 源层 MXFP4 {IDX_STATE} B/state）", c_idx),
        (f"A3 真内容 定长滑窗（{NL} 层×{SW} 位×{HD}×2 B）", c_swa),
        ("A4 真内容 ring（3 层×8 槽×1024×4 B）", c_ring),
        (f"B  SWA admission 超额（窗口 {SW}→"
         f"{lay.swa_blocks_per_req*SWA_BS} token/层）", swa_admit - c_swa),
        (f"C  SWA 页 padding {SWA_BS*HD*2:,}→{P_MLA:,}"
         f"（×{P_MLA/(SWA_BS*HD*2):.2f}）", swa_pad - swa_admit),
        (f"D  SWA 池块计费 {P_MLA:,}→{BPB:,}（×{BPB/P_MLA:.4f}）", swa_chg - swa_pad),
        ("E  g0/g1 池块计费（139,776 / 419,328 → 池块）", g01_chg - g01_real),
        ("F  ring 组按 1 整池块计费", BPB - c_ring),
    ]
    tot = 0
    for name, v in items:
        tot += v
        print(f"  {name:54} {v:>15,}  {v/L:>10,.1f} B/tok  {100*v/charged:>6.3f}%")
    print(f"  {'合计':54} {tot:>15,}  {tot/L:>10,.1f} B/tok  "
          f"{100*tot/charged:>6.3f}%")
    print(f"  残差 = {charged - tot:,} B")
    assert charged == tot, "还原对不上 ⇒ 少算了一类"
    print(f"  ⇒ 真内容 A = {real:,} B = {real/L:,.1f} B/token = {100*real/charged:.3f}%")
    print(f"  ⇒ 整池 {alloc/GiB:.2f} GiB 里真正装内容的上限 = "
          f"{nb/BPR:.4f}×{real:,} = {nb/BPR*real/1e6:.1f} MB")

    print("\n=== 4. 与 model card 那张理论表逐行对照（随上下文增长的部分）")
    print(f"  {'行':42} {'card(FP4)':>10} {'本发':>10} {'倍数':>7}")
    for name, card, ours in [
            ("compress_kv L2/8/14（ratio2，3×256 elem）", 432, 3 * 256 * 2),
            ("compress_kv L20（ratio1，512 elem）", 288, 512 * 2),
            ("index_k     L2/8/14（3×64 elem）", 102, 3 * IDX_STATE // 2),
            ("index_k     L20（128 elem）", 68, IDX_STATE)]:
        print(f"  {name:42} {card:>10} {ours:>10} {ours/card:>6.3f}×")
    print(f"  {'合计 B/token':42} {890:>10} {(c_kv+c_idx)//L:>10} "
          f"{(c_kv+c_idx)/L/890:>6.3f}×")
    print(f"  定长滑窗：card 43 层×{SW}×{HD}×1 B = {43*SW*HD/1e6:.2f} MB/序列；"
          f"本发 {NL} 层 bf16 = {c_swa/1e6:.2f} MB/序列")

    print("\n=== 5. 640.3 vs §4.4 的 343.8：两个因子")
    P_IDX_FP8 = 16_896   # rebase 前 index-K 的页（fp8 132 B/state × 128）

    def pt(Lx, p_idx, inflight=INFLIGHT, swa_bs=SWA_BS, slots=3):
        bpb = (p_idx + P_MLA) * slots + N_RING * P_RING
        swab = cdiv(min(SW - 1 + inflight, Lx), swa_bs) + 1
        bpr = cdiv(Lx, SBS) + cdiv(Lx, 2 * SBS) + NL * swab + 1
        return bpr * bpb / Lx, bpr, bpb

    a, _, _ = pt(8192, P_IDX)
    b, bpr_b, bpb_b = pt(16384, P_IDX)
    c, bpr_c, bpb_c = pt(16384, P_IDX_FP8)
    print(f"  L=8192  MXFP4: {a/KiB:7.2f} KiB/token  ← 本发实测 640.3")
    print(f"  L=16384 MXFP4: {b/KiB:7.2f} KiB/token  (blk/req {bpr_b:,}, 池块 {bpb_b:,})")
    print(f"  L=16384 fp8  : {c/KiB:7.2f} KiB/token  (blk/req {bpr_c:,}, 池块 {bpb_c:,})"
          f"  ← §4.4 的 343.8")
    print(f"  拆解: max_model_len ×{a/b:.4f} × index-K 量化 ×{b/c:.4f} = ×{a/c:.4f}"
          f"   （实测比 {640.3/343.8:.4f}，残差 {100*(a/c-640.3/343.8)/(640.3/343.8):+.2f}%）")
    for nbm in range(1, 1 << 20):
        if int(nbm / bpr_c * 16384) == 74_872:
            break
    print(f"  多卡那一发的反解：num_blocks={nbm:,} ⇒ 分配 {nbm*bpb_c/GiB:.4f} GiB"
          f"（指南记 24.54）、tokens {int(nbm/bpr_c*16384):,}（记 74,872）"
          f" —— 一个自由度同时对上两条读数")
    print(f"  同一 num_blocks 若换成 MXFP4 池块：{nbm*bpb_b/GiB:.4f} GiB ≠ 24.54"
          f" ⇒ 那一发确是 fp8 index-K 构建")

    print(f"\n=== 6. 可回收空间（同 {alloc/GiB:.2f} GiB，同 L={L}）")
    base_t = lay.tokens(alloc)
    levers = [
        ("基线（现状）", {}),
        ("L3  SWA block 32→128", dict(swa_bs=128)),
        ("L3' SWA block 32→64", dict(swa_bs=64)),
        ("L5  mnbt 4096→2048（in-flight 4096）", dict(inflight=MNBT)),
        ("L6  SWA 每组 5 层", dict(swa_layers_per_group=5)),
        ("L6' SWA 每组 40 层（单组）", dict(swa_layers_per_group=40)),
        ("L3+L5", dict(swa_bs=128, inflight=MNBT)),
        ("L3+L6（每组 4 层）", dict(swa_bs=128, swa_layers_per_group=4)),
        ("L3+L6（每组 5 层）", dict(swa_bs=128, swa_layers_per_group=5)),
        ("L3+L5+L6", dict(swa_bs=128, inflight=MNBT, swa_layers_per_group=5)),
    ]
    print(f"  {'改法':36} {'池块 B':>10} {'blk/req':>8} {'B/token':>10} "
          f"{'KV token':>11} {'倍数':>7}")
    for name, kw in levers:
        kw = dict(kw)
        l2 = kvb.build_layout(L, kw.pop("inflight", INFLIGHT), **kw)
        t = l2.tokens(alloc)
        print(f"  {name:36} {l2.bytes_per_block:>10,} {l2.blocks_per_req:>8,} "
              f"{l2.bytes_per_token():>10,.0f} {t:>11,} {t/base_t:>6.2f}×")
    cd = (swa_pad - swa_admit) + (swa_chg - swa_pad)
    print(f"  上界：消掉 C+D（{100*cd/charged:.2f}%）⇒ ×{charged/(charged-cd):.1f}；"
          f"再消掉 B ⇒ ×{charged/(real + g01_chg - g01_real + BPB - c_ring):.1f}；"
          f"全消到纯内容 A ⇒ ×{charged/real:.1f}（不可达：B 是 chunked prefill 的"
          f"正确性需要，E 是跨组共用块号池的代价）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
