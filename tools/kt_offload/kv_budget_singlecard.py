#!/usr/bin/env python3
"""单卡 DSV4.1 KV cache 字节账：从我们自己那两发服务日志反查机制。

判据作用在**实际分配的字节**上：脚本先从 serve.log 里解析出运行时读数
（Available KV cache memory / GPU KV cache size / Maximum concurrency /
kv cache group sizes / kv lcm block sizes / weights|activation|non-torch|graph），
再从配置独立重算 (组结构 × 池块数 × 池块字节)，然后逐项断言重算值 == 日志值。
对不上就 raise —— 机制判错的唯一出口。

⚠️ 上游 `_get_packed_kv_cache_groups` 在本机被 `patch_kv_cache_utils.py` 在
import 期整组换掉。本脚本的公式照的是**本树 patch 后的实现**
（vllm_ascend/patch/platform/patch_kv_cache_utils.py:486
`_get_kv_cache_config_deepseek_v4_main`），每个常量旁边标了出处，并用日志读数
反证。不起服务、不碰卡、只读。

用法（PYTHONPATH=<worktrees/moe-offload-vllm>:<本树> LD_PRELOAD=）：
    python3 tools/kt_offload/kv_budget_singlecard.py --planner-only
    python3 tools/kt_offload/kv_budget_singlecard.py --log <serve.log> [--log ...]
页字节取自真实 cache 对象；每次运行先与本树真实 planner 逐项核对（不依赖任何
log）。--log 必须显式传，且 log 必须带与当前构建相同的构建指纹，否则拒绝。
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass, field

cdiv = lambda a, b: -(-a // b)  # noqa: E731
GiB = 1 << 30
KiB = 1024

# --------------------------------------------------------------- 日志来源 ----
# 🔴 没有默认日志。原先默认读 build_artifacts/moe_offload_ref/bringup_0920_*
# 两发（旧构建：.so 2b58b578e4fc、index-K fp8），它们把写死的 P_IDX=16896 一直
# 对账成绿的——旧 log 描述的是旧构建，拿它判今天的代码，绿灯不说明任何事。
# 现在 --log 必须显式传，且 log 所在目录（或其上层）必须有 build_fingerprint.json
# （build_fingerprint.py stamp 写的），与此刻 capture() 的构建判为同构建才对账。
# 旧的两发没有指纹 ⇒ 会被拒绝；它们的读数已内联在 MOE_OFFLOAD_PLAN §4.5。
LOG_FINGERPRINT_TOOL = "build_fingerprint.py"   # 与本脚本同目录


# ----------------------------------------------------- 模型侧固定几何 ----
# hf_config（/workspace/models/DeepSeek-V4.1-Flash）。这些是模型常量，不随
# 启动参数变；日志里不打印，所以标成模型出处而不是实测。
N_LAYERS = 40           # num_hidden_layers（无 MTP：speculative_config=None）
HEAD_DIM = 512          # head_dim —— 主 MLA 的 state 维
IDX_HEAD = 128          # index_head_dim
SW = 128                # sliding_window（滑窗长度，card 里的「128 位置」）
KV_SRC = (2, 8, 14, 20)  # kv_source_layer_ids
RATIO = {2: 2, 8: 2, 14: 2, 20: 1}  # compress_ratios

# ------------------------------------------------------- 页字节（每层） ----
# 🔴 不写常数。STORAGE_BS / P_MLA / P_IDX / RING_BS / P_RING 与 SWA 页都在下方
# 「真实对象与真实 planner」一节里由 real_pages() 从 worker 真正产出 spec 的类
# 取出（模块加载时算一次）。写死过的 P_IDX=16896 在 rebase 到 f85267ab8
# （index-K 改 MXFP4，页 8,704）之后仍在旧 log 上对账通过，就是反例。
# SWA：上游 vllm/models/deepseek_v4_1/attention.py:456 构造点写死 32 —— 这是输入，
#   不是页字节。
SWA_BS_SHIPPED = 32


def swa_page(block_size: int) -> int:
    """SWA 每块真实字节（padding 前），取自真实 AscendDeepseekV41SWACache 的 spec。"""
    return _swa_page_real(block_size)


def swa_page_padded(block_size: int) -> int:
    """SWA 页被抬到 full-MLA 组里「不小于自己的最近一档」。

    patch_kv_cache_utils.py:331-337 `size_to_candidate`。候选集 = full-MLA 组
    (group[0]) 的页字节集合 = {P_IDX, P_MLA}。
    """
    real = swa_page(block_size)
    candidates = [p for p in (P_IDX, P_MLA) if p >= real]
    if not candidates:
        raise ValueError(
            f"SWA block_size={block_size} 的页 {real} B 超过 full-MLA 最大页 "
            f"{max(P_IDX, P_MLA)} B；patch_kv_cache_utils.py:328 的 "
            "`assert max(sm_page_sizes) <= max(all_page_sizes)` 会先炸。")
    return min(candidates)


# ================================================ 真实对象与真实 planner ====
# 页字节不许写成常数：2026-09-22 rebase 到 f85267ab8 后 index-K 行从 fp8+fp32
# scale（132 B/state，页 16,896）变成 MXFP4（68 B/state，页 8,704），而这里写死
# 的 16896 在默认的两份旧 log 上照样对账通过——旧 log 描述的正是旧构建。
# 下面从 worker 里真正产出 spec 的那几个类取页字节，并用本树 patch 后的 planner
# 本身复算 Layout 的每一项。

FLIGHT_CACHE_BLOCK = 128          # --block-size 128（本线所有发次）
FLIGHT_KV_DTYPE = "bfloat16"      # --kv-cache-dtype bfloat16（A5 bf16 KV 路径）


def _flight_config(max_model_len: int = 8192, inflight: int = 8192,
                   cache_block_size: int = FLIGHT_CACHE_BLOCK,
                   cache_dtype: str = FLIGHT_KV_DTYPE):
    """cache 构造器与 planner 读到的那一片 VllmConfig（其余字段它们不读）。"""
    from types import SimpleNamespace
    cc = SimpleNamespace(block_size=cache_block_size, cache_dtype=cache_dtype,
                         num_gpu_blocks_override=None)
    return SimpleNamespace(
        cache_config=cc,
        model_config=SimpleNamespace(max_model_len=max_model_len),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        max_in_flight_tokens=inflight,
        num_speculative_tokens=0,
        compilation_config=SimpleNamespace(static_forward_context={}))


def real_specs(swa_bs: int = 32, cfg=None) -> dict:
    """V4.1 的 spec 字典，逐个取自真实 cache 对象的 get_kv_cache_spec。

    index K / ring / SWA 走真实 __init__；主 MLA 层在 CPU 上造不起来，只在裸
    实例上设它的 get_kv_cache_spec 读的四个字段，算术是方法自己的。
    需要 PYTHONPATH=<vllm 树>:<本树>（与跑单测相同）。
    """
    try:
        # 顺序有讲究：先 caches / 模型 attention，再 dsa_v1。直接先 import
        # vllm_ascend.attention.dsa_v1 会撞上 vllm_ascend.device.device_op 的
        # 循环导入（单测里由 conftest 的导入顺序绕开，脚本独立跑时会撞）。
        import torch
        from vllm.config import set_current_vllm_config
        from vllm.models.deepseek_v4_1.attention import \
            _indexer_k_cache_head_dim
        from vllm_ascend.models.deepseek_v4_1 import caches
        from vllm_ascend.models.deepseek_v4_1.attention import \
            AscendDeepseekV41Attention
        from vllm_ascend.attention.dsa_v1 import AscendDSASWABackend
    except ImportError as e:
        raise SystemExit(
            f"导入真实 cache 类失败（{e}）。本脚本的页字节只从真实对象取、没有"
            "常数退路 ⇒ 请带 PYTHONPATH=<worktrees/moe-offload-vllm>:<本树> "
            "LD_PRELOAD= 运行。") from e

    cfg = cfg or _flight_config()
    cc = cfg.cache_config
    specs = {}
    with set_current_vllm_config(cfg):
        for layer in KV_SRC:
            ratio = RATIO[layer]
            attn = object.__new__(AscendDeepseekV41Attention)
            torch.nn.Module.__init__(attn)
            attn.is_kv_source = True
            attn.head_dim = HEAD_DIM
            attn.compress_ratio = ratio
            attn.kv_cache_torch_dtype = torch.bfloat16
            specs[f"L{layer}.attn"] = attn.get_kv_cache_spec(cfg)
            idx = caches.AscendDeepseekV41IndexerCache(
                head_dim=_indexer_k_cache_head_dim(IDX_HEAD, False),
                dtype=torch.uint8, prefix=f"L{layer}.idx", cache_config=cc,
                compress_ratio=ratio)
            specs[f"L{layer}.idx"] = idx.get_kv_cache_spec(cfg)
            if ratio > 1:
                ring = caches.AscendDeepseekV41CompressorStateCache(
                    state_dim=2 * HEAD_DIM, dtype=torch.float32,
                    prefix=f"L{layer}.ring")
                specs[f"L{layer}.ring"] = ring.get_kv_cache_spec(cfg)
        for layer in range(40):   # 真实层数，刻意不用 N_LAYERS（它是被核对的模型）
            swa = caches.AscendDeepseekV41SWACache(
                head_dim=HEAD_DIM, window_size=SW, dtype=torch.bfloat16,
                prefix=f"L{layer}.swa", cache_config=cc,
                backend_cls=AscendDSASWABackend, block_size=swa_bs)
            specs[f"L{layer}.swa"] = swa.get_kv_cache_spec(cfg)
    return specs


def real_planner(max_model_len: int, inflight: int, swa_bs: int,
                 avail_bytes: int) -> dict:
    """本树 patch 后的 planner（worker 调的那个路由入口）在真实 spec 上的输出。"""
    import vllm_ascend.patch.platform.patch_kv_cache_utils as P
    from vllm.v1.kv_cache_interface import CircularBufferSpec

    cfg = _flight_config(max_model_len, inflight)
    specs = real_specs(swa_bs, cfg)
    groups = P._ascend_get_kv_cache_groups(cfg, specs)
    num_blocks, tensors = P._get_kv_cache_config_deepseek_v4_main(
        cfg, groups, avail_bytes)
    uniform = [g for g in groups
               if not isinstance(g.kv_cache_spec, CircularBufferSpec)]
    swa = [g for g in uniform if g.layer_names[0].endswith(".swa")]
    return dict(
        specs=specs,
        group_block_sizes=[g.kv_cache_spec.block_size for g in groups],
        bytes_per_block=P._ascend_pool_bytes_per_block(groups),
        num_blocks=num_blocks,
        backing_size=tensors[0].size,
        swa_groups=len(swa),
        swa_blocks_per_req=swa[0].kv_cache_spec.max_memory_usage_pages(cfg),
        blocks_per_req=(sum(g.kv_cache_spec.max_memory_usage_pages(cfg)
                            for g in uniform)
                        + sum(1 for g in groups
                              if isinstance(g.kv_cache_spec, CircularBufferSpec))),
    )


def real_pages() -> dict:
    """页字节，取自真实对象（ratio-1 源层 L20 与 ratio-2 源层 L2）。

    同一类的各层必须同页（MLA spec 不能 page-pad，planner 也按一份页字节算
    tuple）；不同就 raise，而不是挑一个。
    """
    specs = real_specs(SWA_BS_SHIPPED)
    out = {}
    for kind in ("attn", "idx"):
        pages = {specs[f"L{l}.{kind}"].page_size_bytes for l in KV_SRC}
        if len(pages) != 1:
            raise AssertionError(f"{kind} 各源层页字节不一：{sorted(pages)}")
        out[kind] = pages.pop()
    storage = {specs[f"L{l}.attn"].storage_block_size for l in KV_SRC}
    if len(storage) != 1:
        raise AssertionError(f"主 MLA storage_block_size 不一：{sorted(storage)}")
    rings = [specs[k] for k in specs if k.endswith(".ring")]
    if len({(r.block_size, r.page_size_bytes) for r in rings}) != 1:
        raise AssertionError("各 ring 的 (block_size, page) 不一")
    out.update(storage_bs=storage.pop(), ring_bs=rings[0].block_size,
               ring=rings[0].page_size_bytes, n_ring=len(rings))
    return out


_SWA_PAGE_CACHE: dict[int, int] = {}


def _swa_page_real(block_size: int) -> int:
    if block_size not in _SWA_PAGE_CACHE:
        _SWA_PAGE_CACHE[block_size] = real_specs(block_size)["L0.swa"].page_size_bytes
    return _SWA_PAGE_CACHE[block_size]


_PAGES = real_pages()
STORAGE_BS = _PAGES["storage_bs"]   # 128：主 MLA 的 storage_block_size
P_MLA = _PAGES["attn"]              # 131,072 @f85267ab8：128 × 512 × bf16
P_IDX = _PAGES["idx"]               # 8,704 @f85267ab8：128 × (64 E2M1 + 4 E8M0)；旧构建 16,896
RING_BS = _PAGES["ring_bs"]         # 8
P_RING = _PAGES["ring"]             # 32,768：8 × 1024 × fp32
# N_RING 按模型几何算（ratio-1 层没有 ring），再与真实对象数出来的个数核对。
N_RING = sum(1 for l in KV_SRC if RATIO[l] > 1)
if N_RING != _PAGES["n_ring"]:
    raise AssertionError(f"ring 层数：模型几何 {N_RING} != 真实对象 {_PAGES['n_ring']}")


def check_against_planner(max_model_len: int = 8192, inflight: int = 8192,
                          swa_bs: int = SWA_BS_SHIPPED,
                          avail_bytes: int = 29350261248) -> Layout:
    """build_layout 的每一项 == 真实 planner 在真实 spec 上的输出，否则 raise。

    这是不依赖任何 serve.log 的判据：log 只描述它那一发的构建，而这条比的是
    本树此刻的代码。avail_bytes 只是输入（默认取 09-20 那发的 backing 大小）。
    """
    lay = build_layout(max_model_len, inflight, swa_bs=swa_bs)
    real = real_planner(max_model_len, inflight, swa_bs, avail_bytes)
    got = dict(group_block_sizes=lay.group_block_sizes,
               bytes_per_block=lay.bytes_per_block,
               num_blocks=lay.num_blocks(avail_bytes),
               backing_size=lay.allocated(avail_bytes),
               swa_groups=lay.swa_groups,
               swa_blocks_per_req=lay.swa_blocks_per_req,
               blocks_per_req=lay.blocks_per_req)
    fails = [f"{k}: 本脚本 {v!r} != 真实 planner {real[k]!r}"
             for k, v in got.items() if v != real[k]]
    if fails:
        raise AssertionError(
            f"L={max_model_len} in-flight={inflight} swa_bs={swa_bs}: 字节账与"
            "本树真实 planner 对不上 ⇒\n    - " + "\n    - ".join(fails))
    return lay


# ============================================================ 日志解析 ====

_PAT = {
    "avail_gib": re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB"),
    "tokens": re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens"),
    "conc": re.compile(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x"),
    "groups": re.compile(r"kv cache group sizes \[([^\]]+)\]"),
    "lcm": re.compile(r"kv lcm block sizes (\d+)"),
    "mem": re.compile(
        r"Actual usage: ([\d.]+) GiB for weights, ([\d.]+) GiB for peak activation, "
        r"([\d.]+) GiB for non-torch memory, ([\d.]+) GiB for NPU graph memory"),
    "util": re.compile(r"Desired GPU memory utilization is \(([\d.]+), ([\d.]+) GiB\)"),
    "args": re.compile(r"non-default args: (\{.*\})"),
}


@dataclass
class LogReading:
    path: str
    avail_gib: float
    tokens: int
    max_model_len: int
    conc: float
    group_block_sizes: list[int]
    lcm: int
    weights_gib: float
    act_gib: float
    nontorch_gib: float
    graph_gib: float
    util: float
    util_gib: float
    args: dict = field(default_factory=dict)


def parse_log(path: str) -> LogReading:
    text = open(path, encoding="utf-8", errors="replace").read()

    def one(key, why):
        m = _PAT[key].search(text)
        if m is None:
            raise SystemExit(f"{path}: 找不到 {why}（{_PAT[key].pattern!r}）；"
                             "这一发不能当判据来源。")
        return m

    avail = float(one("avail_gib", "Available KV cache memory").group(1))
    tok = int(one("tokens", "GPU KV cache size").group(1).replace(",", ""))
    mc = one("conc", "Maximum concurrency")
    mem = one("mem", "worker.py 的显存分项")
    util = one("util", "Desired GPU memory utilization")
    groups = [int(x) for x in one("groups", "kv cache group sizes").group(1).split(",")]
    args_m = _PAT["args"].search(text)
    args = {}
    if args_m:
        # 日志里是 python repr，不是 JSON
        try:
            import ast
            args = ast.literal_eval(args_m.group(1))
        except (ValueError, SyntaxError):
            args = {}
    return LogReading(
        path=path,
        avail_gib=avail,
        tokens=tok,
        max_model_len=int(mc.group(1).replace(",", "")),
        conc=float(mc.group(2)),
        group_block_sizes=groups,
        lcm=int(one("lcm", "kv lcm block sizes").group(1)),
        weights_gib=float(mem.group(1)),
        act_gib=float(mem.group(2)),
        nontorch_gib=float(mem.group(3)),
        graph_gib=float(mem.group(4)),
        util=float(util.group(1)),
        util_gib=float(util.group(2)),
        args=args,
    )


# ======================================================== 机制重算 ====

@dataclass
class Layout:
    """一个 kv_cache_config 的几何 + 分配量。全部按字节算。"""
    max_model_len: int
    inflight: int
    swa_bs: int
    swa_groups: int
    num_tuple_slots: int
    bytes_per_tuple: int
    ring_bytes_per_block: int
    bytes_per_block: int
    swa_blocks_per_req: int          # 每组
    blocks_per_req: int
    group_block_sizes: list[int]

    @property
    def lcm(self) -> int:
        return math.lcm(*self.group_block_sizes)

    def num_blocks(self, avail_bytes: int) -> int:
        # patch_kv_cache_utils.py:506
        return avail_bytes // self.bytes_per_block

    def allocated(self, avail_bytes: int) -> int:
        """真正写进 KVCacheTensor 的 backing_size（patch:511）。"""
        nb = self.num_blocks(avail_bytes)
        return self.bytes_per_tuple * self.num_tuple_slots * nb + self.ring_bytes_per_block * nb

    def concurrency(self, avail_bytes: int) -> float:
        return self.num_blocks(avail_bytes) / self.blocks_per_req

    def tokens(self, avail_bytes: int) -> int:
        return int(self.concurrency(avail_bytes) * self.max_model_len)

    def bytes_per_token(self) -> float:
        """每 token 计费 = 每请求块数 × 池块字节 / max_model_len。

        与可用显存无关 —— 这就是「固定预留在短 max-model-len 上摊得更差」的
        那个量。
        """
        return self.blocks_per_req * self.bytes_per_block / self.max_model_len


def swa_admission_blocks(max_model_len: int, inflight: int, swa_bs: int) -> int:
    """SlidingWindowSpec.max_admission_blocks_per_request（vllm 侧未被 patch）。

    vllm/v1/kv_cache_interface.py:717-742:
        num_tokens = min(sliding_window - 1 + extra_retained + in_flight, L)
        return cdiv(num_tokens, block_size) + 1
    extra_retained_tokens=0（无 multi-module spec decode）。
    """
    num_tokens = min(SW - 1 + inflight, max_model_len)
    return cdiv(num_tokens, swa_bs) + 1


def build_layout(max_model_len: int, inflight: int, swa_bs: int = SWA_BS_SHIPPED,
                 swa_layers_per_group: int = 1, ring_capped_seqs: int | None = None,
                 mnbt: int = 4096) -> Layout:
    """按本树 patch 后的实现重算几何。

    swa_layers_per_group：`_approximate_gcd([1,3,40], lower_bound=1)=1` 的结果是
    1 ⇒ 40 个单层组。传 >1 只用于「如果分组不这样」的对照，不是现状。
    ring_capped_seqs：L4 假设（ring 区按并发数定长而不是 num_blocks）。
    """
    swa_groups = cdiv(N_LAYERS, swa_layers_per_group)
    # num_tuple_slots = 所有非 ring 组里「同一页字节的层数」的最大值（patch:424）
    # group1（L2/L8/L14 的 MLA + idx）每档 3 层；group0 每档 1 层；
    # SWA 子组每组 swa_layers_per_group 层，单一页字节。
    num_tuple_slots = max(3, swa_layers_per_group)
    bytes_per_tuple = P_IDX + P_MLA                 # sum(page_sizes of group[0])
    ring_bpb = N_RING * P_RING
    if ring_capped_seqs is None:
        bytes_per_block = bytes_per_tuple * num_tuple_slots + ring_bpb
    else:
        # ring 区不再随 num_blocks 走 ⇒ 不进分母
        bytes_per_block = bytes_per_tuple * num_tuple_slots

    swa_bpr = swa_admission_blocks(max_model_len, inflight, swa_bs)
    blocks_per_req = (
        cdiv(max_model_len, STORAGE_BS)        # group0：MLA r1 + idx，block 128
        + cdiv(max_model_len, STORAGE_BS * 2)  # group1：MLA r2 + idx，block 256
        + swa_groups * swa_bpr
        + (0 if ring_capped_seqs is not None else 1)  # ring 组每请求 1 块
    )
    gbs = ([STORAGE_BS, STORAGE_BS * 2] + [swa_bs] * swa_groups + [RING_BS])
    return Layout(
        max_model_len=max_model_len, inflight=inflight, swa_bs=swa_bs,
        swa_groups=swa_groups, num_tuple_slots=num_tuple_slots,
        bytes_per_tuple=bytes_per_tuple, ring_bytes_per_block=ring_bpb,
        bytes_per_block=bytes_per_block, swa_blocks_per_req=swa_bpr,
        blocks_per_req=blocks_per_req, group_block_sizes=gbs,
    )


# ==================================================== 判据：对账日志 ====

def solve_avail_bytes(reading: LogReading, layout: Layout) -> tuple[int, int, int]:
    """从 (tokens, max_model_len) 反解 num_blocks 与可用字节区间。

    日志只把 available 打印到 0.01 GiB（27.34），精度不足以定 num_blocks；
    但 token 数是整数、由 ``int(conc*L)`` 得到，能把 num_blocks 收到 1~2 个值。
    反过来 ``num_blocks = available // bytes_per_block`` 又把 available 圈进一个
    区间 —— 两边一交，才是「重现了日志」。

    返回 (num_blocks, avail_lo, avail_hi)，avail 区间为 [lo, hi)。
    """
    npr = layout.blocks_per_req
    L = layout.max_model_len
    lo = cdiv(reading.tokens * npr, L)
    hi = cdiv((reading.tokens + 1) * npr, L) + 1   # 稍放宽再逐个验
    cands = [nb for nb in range(max(lo - 2, 1), hi) if int(nb / npr * L) == reading.tokens]
    if not cands:
        raise AssertionError(
            f"{reading.path}: 反解不出 num_blocks —— token 数 {reading.tokens} 与"
            f" blocks/req={npr} 不相容 ⇒ 机制判错了（组结构或 SWA 预留公式）。")
    bpb = layout.bytes_per_block
    return cands[0], min(cands) * bpb, (max(cands) + 1) * bpb


def check(reading: LogReading) -> tuple[Layout, int]:
    """重算并逐项断言 == 日志读数。返回 (现状 layout, 可用字节)。"""
    args = reading.args
    mnbt = args.get("max_num_batched_tokens", 4096)
    print(f"\n=== {reading.path}")
    print(f"  日志读数: max_model_len={reading.max_model_len} block_size="
          f"{args.get('block_size')} max_num_seqs={args.get('max_num_seqs')} "
          f"mnbt={mnbt} kv_dtype={args.get('kv_cache_dtype')} "
          f"prefix_cache={args.get('enable_prefix_caching')} "
          f"eager={args.get('enforce_eager', False)}")
    print(f"           Available KV={reading.avail_gib} GiB  tokens="
          f"{reading.tokens:,}  conc={reading.conc}x  groups="
          f"{len(reading.group_block_sizes)}  lcm={reading.lcm}")
    print(f"           weights={reading.weights_gib} act={reading.act_gib} "
          f"nontorch={reading.nontorch_gib} graph={reading.graph_gib} "
          f"util={reading.util}→{reading.util_gib} GiB")

    # in-flight token 数不在日志里（max_in_flight_tokens 是 VllmConfig 的属性），
    # 但它只有两个可能值：mnbt（async 关）或 2*mnbt（async 开，pp=1）。
    # 用「哪个能重现日志的 token 数」把它定下来 —— 判据求值，不是假设。
    fitted = []
    for k, why in ((2, "async scheduling 开（max_concurrent_batches=2）"),
                   (1, "async scheduling 关")):
        lay = build_layout(reading.max_model_len, k * mnbt)
        try:
            nb, a_lo, a_hi = solve_avail_bytes(reading, lay)
        except AssertionError:
            continue
        # 打印值是 available 四舍五入到 0.01 GiB ⇒ 真值落在 ±0.005 GiB 里。
        # 与 num_blocks 反解出的区间必须有交集。
        p_lo = (reading.avail_gib - 0.005) * GiB
        p_hi = (reading.avail_gib + 0.005) * GiB
        if max(a_lo, p_lo) < min(a_hi, p_hi):
            fitted.append((k, why, lay, nb * lay.bytes_per_block))
    if len(fitted) != 1:
        raise AssertionError(
            f"{reading.path}: in-flight 取 mnbt / 2*mnbt 中有 {len(fitted)} 个能同时"
            f"重现 tokens={reading.tokens:,} 与 Available={reading.avail_gib} GiB"
            "；这条判据没有区分力，不能据此定机制。")
    k, why, layout, avail = fitted[0]
    nb = layout.num_blocks(avail)

    print(f"  重算: in-flight={k}×{mnbt}={layout.inflight}  [{why}]")
    print(f"        num_tuple_slots={layout.num_tuple_slots}  "
          f"bytes_per_tuple={layout.bytes_per_tuple}  ring/block="
          f"{layout.ring_bytes_per_block}  ⇒ 池块 {layout.bytes_per_block} B")
    print(f"        SWA 组数={layout.swa_groups}（每组 1 层）× 每组每请求 "
          f"{layout.swa_blocks_per_req} 块  ⇒ blocks/req={layout.blocks_per_req}")
    print(f"        num_blocks={nb:,}  实际分配 {layout.allocated(avail):,} B = "
          f"{layout.allocated(avail)/GiB:.4f} GiB")

    # --- 断言 ---------------------------------------------------------
    fails = []
    got_groups = layout.group_block_sizes
    if got_groups != reading.group_block_sizes:
        fails.append(f"组 block_size 序列 {got_groups[:4]}…({len(got_groups)}) != "
                     f"日志 {reading.group_block_sizes[:4]}…"
                     f"({len(reading.group_block_sizes)})")
    if layout.lcm != reading.lcm:
        fails.append(f"lcm {layout.lcm} != 日志 {reading.lcm}")
    # 实际分配 = num_blocks × 池块字节（floor 后），必须 ≤ 打印的 available，
    # 且差额小于一个池块（否则就是多算/少算了一档）。
    # 实际分配 = num_blocks × 池块字节，num_blocks = available // 池块字节 ⇒
    #   alloc ∈ (available − 池块字节, available]。
    # available 真值只知道落在打印值 ±0.005 GiB 里，所以判据是区间相交。
    alloc = layout.allocated(avail)
    a_lo = (reading.avail_gib - 0.005) * GiB
    a_hi = (reading.avail_gib + 0.005) * GiB
    if not (a_lo - layout.bytes_per_block < alloc <= a_hi):
        fails.append(f"分配字节 {alloc:,} B ({alloc/GiB:.4f} GiB) 不落在日志 "
                     f"available {reading.avail_gib}±0.005 GiB 允许的 "
                     f"({(a_lo-layout.bytes_per_block)/GiB:.4f}, "
                     f"{a_hi/GiB:.4f}] GiB 里")
    if layout.tokens(avail) != reading.tokens:
        fails.append(f"token 数 {layout.tokens(avail):,} != 日志 {reading.tokens:,}")
    if abs(round(layout.concurrency(avail), 2) - reading.conc) > 5e-3:
        fails.append(f"concurrency {layout.concurrency(avail):.4f} != 日志 "
                     f"{reading.conc}")
    # 显存分项自洽：util_gib − weights − act − nontorch == Available KV。
    # 图显存**不在**这条等式里：KV 池在图捕获之前就定好了，那 0.24 GiB 是压在
    # gpu_memory_utilization 预算之上的额外占用（两发的 available 一致、图那发
    # 多用了 0.24 GiB，就是这个意思）。
    resid = (reading.util_gib - reading.weights_gib - reading.act_gib
             - reading.nontorch_gib)
    if abs(resid - reading.avail_gib) > 0.015:
        fails.append(f"显存分项残差 {resid:.2f} GiB != Available KV "
                     f"{reading.avail_gib} GiB")
    if fails:
        raise AssertionError(f"{reading.path}: 机制对不上 ⇒\n    - "
                             + "\n    - ".join(fails))
    print(f"  ✅ 5 项判据全过（组序列/lcm/分配字节/token 数/concurrency）"
          f"+ 显存分项残差 {resid:.2f} GiB")
    return layout, avail


def per_group_table(layout: Layout, avail: int) -> None:
    bpb = layout.bytes_per_block
    L = layout.max_model_len
    rows = [
        ("g0    MLA r1(L20) + idxK(L20)", 2, STORAGE_BS, P_MLA + P_IDX,
         cdiv(L, STORAGE_BS), 1),
        ("g1    MLA r2(L2,8,14)×3 + idxK×3", 6, STORAGE_BS * 2,
         3 * (P_MLA + P_IDX), cdiv(L, STORAGE_BS * 2), 1),
        (f"g2..g{1+layout.swa_groups}  SWA × {layout.swa_groups} 组 × 1 层",
         N_LAYERS, layout.swa_bs, swa_page_padded(layout.swa_bs),
         layout.swa_blocks_per_req, layout.swa_groups),
        (f"g{2+layout.swa_groups}   compressor ring × {N_RING}", N_RING, RING_BS,
         P_RING, 1, 1),
    ]
    tot = sum(bpr * ng * bpb for _, _, _, _, bpr, ng in rows)
    print(f"\n  {'组':34} {'层':>3} {'blk':>4} {'页 B':>9} {'块/请求':>8}"
          f" {'池块数':>8} {'B/请求':>14} {'B/token':>9} {'占比':>7}")
    for name, nl, bs, pg, bpr, ng in rows:
        by = bpr * ng * bpb
        print(f"  {name:34} {nl:>3} {bs:>4} {pg:>9} {bpr:>8} {bpr*ng:>8} "
              f"{by:>14,} {by/L:>9.0f} {100*by/tot:>6.2f}%")
    print(f"  {'合计':34} {'':>3} {'':>4} {'':>9} {'':>8} "
          f"{layout.blocks_per_req:>8} {tot:>14,} {tot/L:>9.0f} {100.0:>6.2f}%")
    print(f"  ⇒ 每 token 计费 = {tot/L:,.0f} B = {tot/L/KiB:.2f} KiB"
          f"   （= blocks/req × 池块字节 / max_model_len，与可用显存无关）")
    nb = layout.num_blocks(avail)
    ring_bytes = layout.ring_bytes_per_block * nb
    print(f"  ⇒ ring 区 = {layout.ring_bytes_per_block} B/块 × {nb:,} 块 = "
          f"{ring_bytes/GiB:.4f} GiB = {100*layout.ring_bytes_per_block/bpb:.2f}% "
          f"的池子；每请求只用 1 块 ⇒ max_num_seqs=8 时实际会碰到 "
          f"{8*layout.ring_bytes_per_block/GiB:.6f} GiB（{100*8/nb:.4f}%）")
    pad = swa_page_padded(layout.swa_bs) / swa_page(layout.swa_bs)
    pool = bpb / swa_page_padded(layout.swa_bs)
    print(f"  ⇒ SWA 每块放大链：页 padding {pad:.2f}× "
          f"({swa_page(layout.swa_bs)}→{swa_page_padded(layout.swa_bs)} B) × "
          f"池块计费 {pool:.3f}× ({swa_page_padded(layout.swa_bs)}→{bpb} B) "
          f"= {pad*pool:.2f}×")


# ============================================ 与「理论 KV」的对账 ====

CARD_BYTES_PER_TOKEN = 890.0  # model card 的合计（FP4 + e4m3/16）


def theory_table() -> None:
    """model card 的 890 B/token vs 本发 bf16 的同口径数，以及与实际计费的差。"""
    # card：主 cache FP4(0.5 B) + 每 16 通道一个 e4m3 scale(1/16 B) = 0.5625 B/elem
    card_main_per_elem = 0.5 + 1 / 16
    # index_k：MXFP4，E8M0 per 32 ⇒ 0.5 + 1/32
    card_idx_per_elem = 0.5 + 1 / 32
    # 本发：主 cache bf16 = 2 B/elem，无 scale（A5 bf16 KV 路径 scale_dim=0）
    ours_main_per_elem = 2.0
    # 本树 index_k：每 state 字节 = 真实 spec 的页 / storage block
    # （f85267ab8 起 MXFP4：64 packed E2M1 + 4 E8M0 = 68；旧构建 fp8+fp32 scale = 132）
    def ours_idx_state_bytes():
        return P_IDX // STORAGE_BS

    def per_token(main_state_bytes: float, idx_state_bytes: float) -> float:
        # 每层每 token 的 state 数 = 1/ratio
        return sum((main_state_bytes + idx_state_bytes) / RATIO[l] for l in KV_SRC)

    card = per_token(HEAD_DIM * card_main_per_elem, IDX_HEAD * card_idx_per_elem)
    ours = per_token(HEAD_DIM * ours_main_per_elem, ours_idx_state_bytes())
    card_main = sum(HEAD_DIM * card_main_per_elem / RATIO[l] for l in KV_SRC)
    ours_main = sum(HEAD_DIM * ours_main_per_elem / RATIO[l] for l in KV_SRC)
    card_idx = card - card_main
    ours_idx = ours - ours_main

    print("\n=== 理论 KV（随上下文增长的部分）：card 口径 vs 本发 bf16 口径")
    print(f"  {'项':28} {'card(FP4)':>12} {'本发(bf16)':>12} {'倍数':>7}")
    print(f"  {'compress_kv（4 源层）':28} {card_main:>12.0f} {ours_main:>12.0f}"
          f" {ours_main/card_main:>6.2f}×")
    print(f"  {'index_k（4 源层）':28} {card_idx:>12.0f} {ours_idx:>12.0f}"
          f" {ours_idx/card_idx:>6.2f}×")
    print(f"  {'合计 B/token':28} {card:>12.0f} {ours:>12.0f} {ours/card:>6.2f}×")
    assert abs(card - CARD_BYTES_PER_TOKEN) < 0.6, (
        f"重算 card 口径得 {card:.1f} B/token，与 card 写的 "
        f"{CARD_BYTES_PER_TOKEN} 差太多 ⇒ 我理解的 card 元素/量化口径是错的")
    print(f"  （重算 card = {card:.0f} B/token，与 card 表的 890 B 一致 ⇒ 口径对得上）")
    print(f"  🔸 **不是 ×4，是 ×{ours/card:.2f}**：主 cache bf16 2 B vs FP4+1/16 "
          f"scale 0.5625 B ⇒ ×{ours_main_per_elem/card_main_per_elem:.3f}；"
          f"index_k 本树 {ours_idx_state_bytes()} B/state（取自真实 spec）"
          f"vs card MXFP4（{IDX_HEAD*card_idx_per_elem:.0f} B/state）⇒ "
          f"×{ours_idx/card_idx:.2f}。两项加权后 {ours/card:.2f}×。")
    print(f"  上下文 ×  容量/序列（纯内容，本发 bf16 口径）:", end="")
    for ctx in (4096, 8192, 32768, 131072):
        print(f"  {ctx//1024}K={ours*ctx/1e6:.1f} MB", end="")
    print()

    # 定长滑窗
    card_swa = N_LAYERS * SW * HEAD_DIM * 1      # card 口径：fp8
    ours_swa = N_LAYERS * SW * HEAD_DIM * 2      # 本发 bf16
    print(f"\n  定长滑窗 cache（与上下文无关）: card 口径 {N_LAYERS}层×{SW}位×"
          f"{HEAD_DIM}维×1B = {card_swa/1e6:.2f} MB/序列；"
          f"本发 bf16 = {ours_swa/1e6:.2f} MB/序列（×2）")
    print("  ⚠️ card 表里写 43 层是 V4.0；V4.1 是 40 层（+3 层 MTP 不加载）。")
    return ours, ours_swa


def reconcile_theory_vs_charged(layout: Layout, avail: int, ours_bt: float,
                                ours_swa_bytes: float) -> None:
    L = layout.max_model_len
    bpb = layout.bytes_per_block
    # 全局/持久部分（g0+g1）按池计费
    glob_blocks = cdiv(L, STORAGE_BS) + cdiv(L, STORAGE_BS * 2)
    glob_charged = glob_blocks * bpb
    # SWA 部分按池计费
    swa_charged = layout.swa_groups * layout.swa_blocks_per_req * bpb
    # SWA「真正被预留」的内容字节（admission 公式的 token 数）
    swa_admit_tokens = min(SW - 1 + layout.inflight, L)
    swa_content = N_LAYERS * swa_admit_tokens * HEAD_DIM * 2
    print("\n=== 理论 vs 实际计费（同一发，max_model_len="
          f"{L}，每请求）")
    print(f"  ① 随上下文增长部分（4 个源层）")
    print(f"     理论内容      {ours_bt*L:>16,.0f} B  ({ours_bt:.0f} B/token)")
    print(f"     池子实际计费  {glob_charged:>16,} B  "
          f"({glob_charged/L:,.0f} B/token) ⇒ {glob_charged/(ours_bt*L):.2f}×")
    print(f"  ② 定长滑窗部分（40 层）")
    print(f"     理论内容(窗口 {SW} 位) {ours_swa_bytes:>11,.0f} B")
    print(f"     admission 公式预留的内容: 40 层 × min({SW}-1+in-flight"
          f"{layout.inflight}, {L})={swa_admit_tokens} token × {HEAD_DIM}×2 = "
          f"{swa_content:,} B  ⇒ {swa_content/ours_swa_bytes:.0f}× 窗口本身")
    print(f"     再 ×页 padding {swa_page_padded(layout.swa_bs)/swa_page(layout.swa_bs):.2f}"
          f" ×池块计费 {bpb/swa_page_padded(layout.swa_bs):.3f}")
    print(f"     池子实际计费  {swa_charged:>16,} B  ⇒ "
          f"{swa_charged/ours_swa_bytes:,.0f}× 理论窗口、"
          f"{swa_charged/swa_content:.2f}× admission 内容")
    tot = glob_charged + swa_charged + bpb
    print(f"  ③ 合计 {tot:,} B/请求 = {tot/L/KiB:.2f} KiB/token；"
          f"其中滑窗占 {100*swa_charged/tot:.2f}%")
    print(f"  ⇒ 全发 {avail/GiB:.2f} GiB 里，只有 "
          f"{100*(ours_bt*L+ours_swa_bytes)/tot:.2f}% 是理论上需要留的内容字节。")


# =================================================== 两条改法的收益 ====

# §4.4 / 起飞闸 tools/kt_offload/kt_offload_gate.py 的系数
EXPERT_GIB = 0.7004          # 每专家跨全部层（compact scale），flash_dev 核过
DEVICE_FACTOR = 1.024        # NZ 填充，本发实测（见 main 里的反证）
NON_EXPERT_WEIGHTS = 10.4    # 非专家权重
TOTAL_EXPERTS = 384


def cpu_ms_per_token(n_resident: int) -> float:
    """§4.3 实测曲线：T(k)=66.8+43.95k µs/层 × 40 层，k=6(1−N/384)。"""
    k = 6 * (1 - n_resident / TOTAL_EXPERTS)
    return (66.8 + 43.95 * k) * N_LAYERS / 1000.0


def lever_table(base: Layout, avail: int, reading: LogReading) -> None:
    mnbt = reading.args.get("max_num_batched_tokens", 4096)
    seqs = reading.args.get("max_num_seqs", 8)
    L = base.max_model_len
    base_tokens = base.tokens(avail)
    base_bt = base.bytes_per_token()

    # 非 KV 的一切（权重 + 激活 + non-torch + 图）都不变 ⇒ 省下的 KV 字节
    # 一比一变成专家预算。
    pool_for_experts_and_kv = (reading.util_gib - NON_EXPERT_WEIGHTS
                               - reading.act_gib - reading.nontorch_gib
                               - reading.graph_gib)
    n_base = round((reading.weights_gib - NON_EXPERT_WEIGHTS)
                   / (EXPERT_GIB * DEVICE_FACTOR))

    print(f"\n=== 改法收益（基线：{base_tokens:,} token / {avail/GiB:.2f} GiB / "
            f"{base_bt/KiB:.1f} KiB/token / N={n_base}）")
    print(f"  专家+KV 总额度 = util {reading.util_gib} − 非专家权重 "
          f"{NON_EXPERT_WEIGHTS} − 激活 {reading.act_gib} − non-torch "
          f"{reading.nontorch_gib} − 图 {reading.graph_gib} = "
          f"{pool_for_experts_and_kv:.2f} GiB")

    levers = [
        ("基线（现状）", dict()),
        ("L3  SWA block_size 32→128", dict(swa_bs=128)),
        ("L3' SWA block_size 32→64", dict(swa_bs=64)),
        ("L4  ring 区按 max_num_seqs 定长", dict(ring_capped_seqs=seqs)),
        ("L3+L4", dict(swa_bs=128, ring_capped_seqs=seqs)),
        ("L5  mnbt 4096→2048（只改配置）", dict(inflight=2 * 2048)),
        ("L6  SWA 每组 5 层（改分组）", dict(swa_layers_per_group=5)),
        ("L3+L5", dict(swa_bs=128, inflight=2 * 2048)),
        ("L3+L4+L6", dict(swa_bs=128, ring_capped_seqs=seqs,
                          swa_layers_per_group=5)),
    ]
    print(f"\n  A 轴 —— 容量（同样 {avail/GiB:.2f} GiB 可用显存）")
    print(f"  {'改法':32} {'池块 B':>9} {'blk/req':>8} {'KV token':>10} "
          f"{'倍数':>6} {'B/token':>9} {'8192 并发':>9}")
    for name, kw0 in levers:
        kw = dict(kw0)          # 不要就地 pop：同一个 levers 还要走 B 轴一遍
        lay = build_layout(kw.pop("max_model_len", L),
                           kw.pop("inflight", base.inflight), **kw)
        t = lay.tokens(avail)
        print(f"  {name:32} {lay.bytes_per_block:>9,} {lay.blocks_per_req:>8,} "
              f"{t:>10,} {t/base_tokens:>5.2f}× {lay.bytes_per_token()/KiB:>8.1f}K "
              f"{lay.concurrency(avail):>8.2f}x")

    print(f"\n  B 轴 —— 省下的 GiB ⇒ 常驻专家 ⇒ CPU 时间（保住基线 "
          f"{base_tokens:,} token 不变）")
    print(f"  {'改法':32} {'KV GiB':>8} {'省':>7} {'ΔN':>5} {'N':>5} "
          f"{'CPU ms/token':>13} {'Δms':>7}")
    for name, kw0 in levers:
        kw = dict(kw0)
        ring_seqs = kw.get("ring_capped_seqs")
        lay = build_layout(kw.pop("max_model_len", L),
                           kw.pop("inflight", base.inflight), **kw)
        # 保住同样的 token 数需要多少块 → 多少字节
        need_blocks = cdiv(base_tokens * lay.blocks_per_req, L)
        need_bytes = need_blocks * lay.bytes_per_block
        if ring_seqs is not None:
            # ring 区被移出分母后，要单独按并发数计一份
            need_bytes += ring_seqs * lay.ring_bytes_per_block
        kv_gib = need_bytes / GiB
        saved = avail / GiB - kv_gib
        dn = int(saved / (EXPERT_GIB * DEVICE_FACTOR))
        n = min(n_base + dn, TOTAL_EXPERTS)
        # 闸也要放得下：专家 + KV ≤ 额度
        n = min(n, int((pool_for_experts_and_kv - kv_gib) / (EXPERT_GIB * DEVICE_FACTOR)))
        print(f"  {name:32} {kv_gib:>8.2f} {saved:>7.2f} {dn:>5} {n:>5} "
              f"{cpu_ms_per_token(n):>13.2f} {cpu_ms_per_token(n)-cpu_ms_per_token(n_base):>7.2f}")
    print(f"  （CPU 曲线 T(k)=66.8+43.95k µs/层 × {N_LAYERS} 层，"
          f"k=6(1−N/{TOTAL_EXPERTS})，§4.3 实测；N 上限还受闸约束）")


class StaleLogError(RuntimeError):
    """这份 log 不能证明自己测的是当前构建 ⇒ 不拿它对账。"""


def _fingerprint_module():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        LOG_FINGERPRINT_TOOL)
    spec = importlib.util.spec_from_file_location("build_fingerprint", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def require_log_build(path: str, current: dict | None = None,
                      strict_trees: bool = False) -> list[str]:
    """log 的构建指纹必须存在且与当前构建相同，否则 raise StaleLogError。

    "没看到指纹" 不是 "构建相同" 的证据（m.2）。current 缺省时现场 capture()。
    返回 ADVISORY 行（例如树 HEAD 不同但不阻断；--strict-trees 升级为阻断）。
    """
    bf = _fingerprint_module()
    fp = bf.load(path)
    if fp is None:
        raise StaleLogError(
            f"{path}: 旁边（及上层目录）没有 {bf.FILENAME} ⇒ 证明不了它测的是当前"
            "构建，拒绝对账。旧构建的 log 会把过期的页字节对账成绿的（P_IDX=16896"
            " 就是这样在 rebase 后仍然绿着）。新发次请在起服务的目录上"
            " `build_fingerprint.py stamp <dir>`。")
    cur = current if current is not None else bf.capture(label="kv_budget_check")
    ok, reasons = bf.same_build(fp, cur, strict_trees)
    if not ok:
        raise StaleLogError(
            f"{path}: 与当前构建不是同一构建，拒绝对账。\n  log: {bf.short(fp)}"
            f"\n  now: {bf.short(cur)}\n  " + "\n  ".join(reasons))
    return [r for r in reasons if r.startswith("ADVISORY ")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append", default=None,
                    help="serve.log；必须显式给，且必须带同构建指纹")
    ap.add_argument("--planner-only", action="store_true",
                    help="只做与真实 planner 的核对（不读任何 log）")
    ap.add_argument("--strict-trees", action="store_true",
                    help="log 与当前树 HEAD 不同也拒绝（默认只提示）")
    args = ap.parse_args()

    print(f"页字节（取自真实对象）：主 MLA {P_MLA} / index-K {P_IDX} / ring "
          f"{P_RING}×{N_RING} / storage block {STORAGE_BS}")
    for L in (8192, 16384):
        for bs in (SWA_BS_SHIPPED, 128):
            lay = check_against_planner(L, 8192, bs)
            print(f"  ✅ 真实 planner 核对 L={L} swa_bs={bs}: 池块 "
                  f"{lay.bytes_per_block:,} B  blocks/req {lay.blocks_per_req:,}")
    if args.planner_only:
        return 0
    if not args.log:
        print("❌ 没给 --log。本脚本没有默认日志：旧构建的 log 会把过期常数对账成"
              "绿的。给一份带同构建指纹的 serve.log，或用 --planner-only。",
              file=sys.stderr)
        return 2
    logs = args.log
    for path in logs:
        try:
            for adv in require_log_build(path, strict_trees=args.strict_trees):
                print(f"  {adv}")
        except StaleLogError as e:
            print(f"❌ {e}", file=sys.stderr)
            return 3

    first = None
    for path in logs:
        reading = parse_log(path)
        layout, avail = check(reading)
        if first is None:
            first = (reading, layout, avail)
            per_group_table(layout, avail)

    reading, layout, avail = first
    # 设备系数的独立反证：weights = 非专家 10.4 + N × 0.7004 × factor
    n_cfg = reading.args.get("additional_config", {}).get(
        "kt_offload_config", {}).get("num_resident_experts")
    if n_cfg:
        implied = (reading.weights_gib - NON_EXPERT_WEIGHTS) / (n_cfg * EXPERT_GIB)
        print(f"\n  旁证：日志 weights {reading.weights_gib} GiB − 非专家 "
              f"{NON_EXPERT_WEIGHTS} = {reading.weights_gib-NON_EXPERT_WEIGHTS:.2f}"
              f"，除 N={n_cfg} × {EXPERT_GIB} ⇒ 设备系数 {implied:.4f}"
              f"（脚本用 {DEVICE_FACTOR}）")

    ours_bt, ours_swa = theory_table()
    reconcile_theory_vs_charged(layout, avail, ours_bt, ours_swa)
    lever_table(layout, avail, reading)
    return 0


if __name__ == "__main__":
    sys.exit(main())
