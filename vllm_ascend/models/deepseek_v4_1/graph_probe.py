# SPDX-License-Identifier: Apache-2.0
"""看见图段内部 —— 把探针做成图里的一条 kernel，而不是一句 Python。

🔴 为什么需要这个模块（2026-09-15）：

vLLM 的 breakable cudagraph 在 replay 时是
``BreakableCUDAGraphCapture.replay(): for r in self.segments: r()``，
segments 里混着**捕获好的图段**和 ``add_eager`` 记下的 **Python callable**。
⇒ **图段里的 Python 一行都不执行**。所有写在图段里的 tap，在图模式的 dump 里
**一个键都不会出现** —— 而"缺的键"读起来和"没差异"一模一样。

实测代价：逐层比对一度把第一个分歧点定在压缩器，而那其实是
**"第一个分歧点在图模式下唯一还看得见的那一格"**。真正的分歧发生在
``_o_proj`` → MHC → MoE 这一段，它整段在图里，我们整天对它是瞎的。

**但 kernel 会重放。** 所以：
```
捕获前  ：分配一块静态 buffer（地址固定，图捕获要求的正是这个）
图段里  ：buf.copy_(x)        ← 这是一条 kernel，会被录进图，每次 replay 都执行
runner  ：读 buf              ← runner 的 Python 每步都跑
```

⚠️ 三条必须守住的：
① **buffer 必须在捕获【之前】分配**。捕获期新分配会落进 graph memory pool，
   地址不稳。本模块按 (name, shape, dtype) 懒分配 —— 第一次调用发生在
   warmup（``_dummy_run`` 不捕获的那一步），正好在捕获之前。
② **形状一旦定下就不能变**。变了说明这个探针点在不同 forward 上形状不同，
   那它本来就不该用一块静态 buffer —— 直接拒绝并记账，不要悄悄重分配。
③ **读出来的是"最后一次写入"**。同名探针被多层/多步复用时，读到的是最后
   那一次。所以名字里要带层号；不带层号的名字只在"只有一个写者"时才有意义。
   本模块**不替调用方判断这件事**，但会记下每个名字被写了多少次，
   让读数的人看得见分母。
"""
from __future__ import annotations

import torch

#: name -> 静态 buffer
_BUFS: dict[str, torch.Tensor] = {}

#: name -> 这一轮被写了几次（由 runner 在读完后清零）
_WRITES: dict[str, int] = {}

#: 出错/拒绝的记账。**无条件可读**，包括全零 —— 否则"探针没装上"和
#: "探针装上了但这一层没差异"读数一样。
_STATS: dict = {"allocated": 0, "copies": 0, "shape_changed": 0,
                "alloc_failed": 0, "read": 0, "filtered_other_bucket": 0, "adopted": 0,
               "call_site_errors": 0}


def stats() -> dict:
    """本模块的覆盖量。无条件可读。"""
    return dict(_STATS)


def probe(name: str, x: torch.Tensor) -> None:
    """把 ``x`` 的当前值抄进一块静态 buffer —— 作为图里的一条 kernel。

    在图段里调用：捕获时这句 copy 被录进图，replay 时它照常执行。
    在 eager 下调用：就是一次普通的 copy，语义相同。

    不抛异常：探针不许杀 run。但**每一种失败都计数**，
    并且形状变化会被单独记成一格（那是"这个探针点不该用静态 buffer"的信号，
    不是噪声）。
    """
    if not isinstance(x, torch.Tensor):
        return
    # 🔴 按形状分桶(2026-09-15 实测)：一个名字一块 buffer 时,T=8 的请求先来、
    #    把 buffer 定成 8 行,之后 T=5 的请求形状对不上 ⇒ 被 shape_changed 拒绝
    #    ⇒ buffer 里一直留着 T=8 那道题的数据,而 dump 照样把它记下来。
    #    于是"这道题的探针"其实是"上一道题的残留",比出来 absmax=448,
    #    看起来像一个惊人的缺陷 —— 而它只是两边比的不是同一个东西。
    #    ⚠️ 守卫本身是对的(它拒绝了、也记了账),坏在【那个账我从没读过】。
    #    形状进名字,读数的人就不可能把两个形状混着比。
    name = f"{name}.T{int(x.shape[0])}"
    buf = _BUFS.get(name)
    if buf is None:
        try:
            # 🔴 empty_like 而不是 clone：clone 会多一次拷贝，而且我们要的是
            #    "一块地址固定的空间"，不是"这一刻的值"。
            _BUFS[name] = torch.empty_like(x)
            _STATS["allocated"] += 1
        except Exception:                                    # noqa: BLE001
            _STATS["alloc_failed"] += 1
            return
        buf = _BUFS[name]
    if buf.shape != x.shape or buf.dtype != x.dtype:
        # 形状/dtype 变了 ⇒ 这个点在不同 forward 上不是同一个东西。
        # 不重分配（捕获后重分配地址会变，下游图段会读到错的地方），
        # 也不静默截断 —— 记一笔，这一次不采。
        _STATS["shape_changed"] += 1
        return
    buf.copy_(x)
    _STATS["copies"] += 1
    _WRITES[name] = _WRITES.get(name, 0) + 1


def adopt(name: str, buf: torch.Tensor, rows: "int | None" = None) -> None:
    """认领一块【调用方已经在捕获前分配好】的张量作为探针 buffer。

    🔴 为什么需要它:probe() 会在第一次调用时 empty_like 一块 buffer,
    而如果那第一次调用发生在【捕获中】(比如 engram 的捕获路线,它本来就
    只在捕获期执行),那块 buffer 就落进了图内存池,地址不稳 ——
    正是本模块第一条注释要避免的事。
    ⇒ 这种场合由调用方在捕获前分配好,这里只登记,不分配、不拷贝。
    名字里仍带形状桶,规则和 probe() 一致。
    """
    # 🔴 桶号必须是【本次 forward 的 token 数】,不是探针自己的 dim0。
    #    第一版用了 buf.shape[0](=5,探针的槽位数),而 runner 的
    #    drain(only_rows=num_tokens_padded) 只收 .T{1,2,4,8} ——
    #    5 永远不在里面 ⇒ 这两个键【每一步都被静默过滤】,
    #    产物上表现为"键不存在",而键不存在读起来和"两臂没差异"一样。
    key = f"{name}.T{int(rows if rows is not None else buf.shape[0])}"
    if key not in _BUFS:
        _BUFS[key] = buf
        _STATS["adopted"] = _STATS.get("adopted", 0) + 1
    _WRITES[key] = _WRITES.get(key, 0) + 1


def note_call_site_error() -> None:
    """调用点自己出错(取层号失败等)。和 probe() 内部的失败分开记。"""
    _STATS["call_site_errors"] = _STATS.get("call_site_errors", 0) + 1


def drain(only_rows: "int | None" = None) -> "list[tuple[str, torch.Tensor, int]]":
    """把这一轮写过的探针读出来：(名字, buffer, 本轮写入次数)。

    由 model_runner 在 ``run_model()`` 之后调用 —— 那里的 Python 每步都跑，
    两种模式都跑。写入次数一并交出去，因为"读到的是最后一次写入"这件事
    必须对读数的人可见。
    """
    # 🔴 遍历 _BUFS，不是 _WRITES。
    #    `_WRITES[name] += 1` 这句本身就是【图段里的 Python】——
    #    copy kernel 会重放，这个计数器不会。第一版按 _WRITES 找本轮写过谁,
    #    于是 replay 时它永远是空的 ⇒ 探针一个都读不出来,
    #    而症状和"探针没装上"完全一样。
    #    ⇒ buffer 是捕获期分配、整轮存在的,它才是可靠的清单。
    # 🔴 只交出【本次 forward 那个桶】(2026-09-15)。
    #    分桶解决了"形状混着比",但没解决"归属":图模式下 T8 桶既可能是
    #    T=8 那道题的 prefill,也可能是 T=5 被 padding 到 8 —— 最后写入者胜。
    #    而一次 forward 里探针写的就是这一个桶,这个请求的落盘也发生在同一次
    #    forward 结束时 ⇒ 按桶过滤,归属就自动是对的,不需要再加任何机制。
    #    only_rows=None 表示不过滤(单测/eager 早期路径)。
    out = []
    for name, buf in _BUFS.items():
        if only_rows is not None and not name.endswith(f".T{int(only_rows)}"):
            _STATS["filtered_other_bucket"] = \
                _STATS.get("filtered_other_bucket", 0) + 1
            continue
        # 写入次数在 replay 下【拿不到】(那行 Python 不执行)，
        # 所以交出去的是 -1 而不是 0 —— 0 会被读成"这一轮没写过"。
        n = _WRITES.get(name, -1)
        out.append((name, buf, n))
        _STATS["read"] += 1
    _WRITES.clear()
    return out


def reset_for_test() -> None:
    _BUFS.clear()
    _WRITES.clear()
    for k in _STATS:
        _STATS[k] = 0
