"""DSV4.1 专家路由频率计数(单卡 MoE offload 线 M0c 用;飞行树诊断,不进 dsv41-clean)。

开启:VLLM_ASCEND_DSV41_EXPERT_COUNT=<输出目录>。不设时 record() 立即返回。
产出:<目录>/rank<dp>_pid<pid>.pt,内含
  counts[bucket]  [40, 384] int64  —— bucket ∈ decode / prefill
  tokens[bucket]  [40]      int64  —— 该层计入的 token 数(判据:counts[b][L].sum() == tokens[b][L] * topk)
  skipped         dict             —— 被排除的调用数(按原因),覆盖量要和结论一起报

🔴 四个坑(09-18 读 routed_experts.py 得出):
  · 计的是 router 输出的【全局逻辑专家号】0..383,在 log2phy / mix_placement 追加共享专家【之前】
    ⇒ 四个 DP rank 的计数可直接相加
  · enable_force_load_balance=True(profile/dummy run、图捕获)时 topk 是随机排列 ⇒ 不计,记入 skipped
  · DP 补齐:batch 被补到各 rank 最大 token 数,补齐行也会路由 ⇒ 只计前 num_actual_tokens 行
  · 图模式 replay 不执行 Python ⇒ 本计数器只在 eager(--enforce-eager)下有效;图模式下只会在捕获时跑
decode/prefill 按 vLLM v1 的 batch 顺序(decode 在前)切分:[0:num_decode_tokens) 为 decode。
"""
import atexit
import os
import re

import torch

_DIR = os.getenv("VLLM_ASCEND_DSV41_EXPERT_COUNT", "")
ENABLED = bool(_DIR)
_N_LAYERS, _N_EXPERTS, _DUMP_EVERY = 40, 384, 200
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_S = None


def _state(device):
    global _S
    if _S is None:
        _S = {"counts": {b: torch.zeros(_N_LAYERS, _N_EXPERTS, dtype=torch.int64, device=device)
                         for b in ("decode", "prefill")},
              "tokens": {b: [0] * _N_LAYERS for b in ("decode", "prefill")},
              "skipped": {}, "steps": 0, "rank": None}
        os.makedirs(_DIR, exist_ok=True)
        atexit.register(dump)
    return _S


def _skip(s, why):
    s["skipped"][why] = s["skipped"].get(why, 0) + 1


def _split():
    """(num_decode_tokens, num_actual_tokens) —— 纯 Python 属性,不触发设备同步。"""
    from vllm.forward_context import get_forward_context
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        md = next((m for m in md.values() if getattr(m, "num_actual_tokens", None) is not None), None)
    if md is None:
        return None
    n_act = int(md.num_actual_tokens)
    n_dec = getattr(md, "num_decode_tokens", None)
    if n_dec is None:
        n_dec = getattr(md, "num_decodes", None)
    return (None if n_dec is None else int(n_dec)), n_act


def record(layer, topk_ids, force_load_balance):
    if not ENABLED:
        return
    s = _state(topk_ids.device)
    m = _LAYER_RE.search(getattr(layer, "layer_name", "") or "")
    if m is None or "dspark" in (getattr(layer, "layer_name", "") or ""):
        _skip(s, "非主干层(无层号或 dspark)"); return
    L = int(m.group(1))
    if L >= _N_LAYERS:
        _skip(s, f"层号越界 {L}"); return
    if force_load_balance:
        _skip(s, "force_load_balance(随机 topk,profile/dummy)"); return
    try:
        sp = _split()
    except Exception as e:  # noqa: BLE001
        _skip(s, f"取元数据失败 {type(e).__name__}"); return
    if sp is None:
        _skip(s, "无 attn 元数据(dummy?)"); return
    n_dec, n_act = sp
    ids = topk_ids[:n_act]
    if ids.numel() and int(ids.shape[1]) != 6:
        _skip(s, f"topk 宽度不是 6({int(ids.shape[1])})"); return
    if n_dec is None:
        _skip(s, "元数据无 decode 计数"); return
    for b, lo, hi in (("decode", 0, min(n_dec, n_act)), ("prefill", min(n_dec, n_act), n_act)):
        if hi > lo:
            flat = ids[lo:hi].reshape(-1).to(torch.int64)
            s["counts"][b][L].index_add_(0, flat, torch.ones_like(flat))
            s["tokens"][b][L] += hi - lo
    if L == 0:
        s["steps"] += 1
        if s["steps"] % _DUMP_EVERY == 0:
            dump()


def dump():
    s = _S
    if s is None:
        return
    if s["rank"] is None:
        try:
            from vllm.distributed import get_dp_group
            s["rank"] = get_dp_group().rank_in_group
        except Exception:  # noqa: BLE001
            s["rank"] = "x"
    out = {"counts": {b: t.detach().cpu() for b, t in s["counts"].items()},
           "tokens": {b: torch.tensor(v) for b, v in s["tokens"].items()},
           "skipped": dict(s["skipped"]), "steps": s["steps"], "topk": 6}
    p = os.path.join(_DIR, f"rank{s['rank']}_pid{os.getpid()}.pt")
    torch.save(out, p + ".tmp")
    os.replace(p + ".tmp", p)
