# SPDX-License-Identifier: Apache-2.0
"""The engram table, never allocated and never registered.

Upstream's ``ParallelEngramEmbedding.__init__`` allocates the whole shard up
front, and on this platform both of its branches are impossible::

    kwargs = {"device": "cpu", "pin_memory": True} if cpu_offload else {}

``cpu_offload=True`` takes ``aclrtMallocHostWithCfg`` -- the pinned-host path,
which is banned, and which at TP=1 (no sharding, so no reduction per rank)
wants 188.83 GiB per rank -- 755.3 GiB across four ranks, against a 755 GiB
host that already has 188.8 GiB of it in /dev/shm. ``False`` puts the same
188.83 GiB in HBM. The first real launch died here, and it is worth being
precise
about how: it died of ``rtsMallocHost ... out of memory``. **The banned path was
stopped by the machine running out of memory, not by us.** On a larger host it
would have succeeded and then done the thing that took a card down.

So this does two separate jobs, and the second is not a consequence of the
first:

1. The tables are never allocated. The Parameters are created on the ``meta``
   device -- correct shape and dtype, zero bytes -- because the real data is
   already staged in ``/dev/shm`` and is read from there.
2. The banned path is made unreachable and loud. ``_storage()`` raises, and
   construction refuses if anything managed to allocate real storage. A policy
   that is only written down is enforced by whatever happens to fail first.
"""

from __future__ import annotations

import contextlib

import torch

from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout
from vllm_ascend.models.deepseek_v4_1.engram_staging import (DEFAULT_STAGE_DIR,
                                                             staged_file_name)


#: Upstream logs this from inside ``if cpu_offload:`` regardless of what the
#: allocation actually did, so with the allocation diverted it describes an
#: intention rather than an event.
_STALE_OFFLOAD_LOG = "offloaded to pinned host memory"


class _DropStaleOffloadLine:
    """Filter out upstream's now-untrue INFO line.

    It reports "94.42 GiB per rank", which over four ranks and two layers is
    755.4 GiB against a 755 GiB host -- a reader would conclude host memory had
    just been exhausted, when the measured figure is VmPin = 0. Left in place it
    is worse than no log line, because the number is specific enough to be
    believed.
    """

    def filter(self, record) -> bool:
        return _STALE_OFFLOAD_LOG not in record.getMessage()


@contextlib.contextmanager
def _upstream_offload_line_suppressed():
    import logging

    logger = logging.getLogger(
        "vllm.models.deepseek_v4_1.common.engram")
    drop = _DropStaleOffloadLine()
    logger.addFilter(drop)
    try:
        yield
    finally:
        logger.removeFilter(drop)


@contextlib.contextmanager
def tables_not_allocated():
    """Force ``torch.empty`` onto ``meta`` for the duration.

    Scoped to upstream's ``__init__``, whose only two ``torch.empty`` calls are
    the table and its scales. Narrow and ugly, but the alternative is copying
    upstream's constructor here so the two allocations can be dropped, and a
    copied constructor drifts silently while this fails loudly if the shape of
    that code changes.
    """
    real_empty = torch.empty

    def empty_on_meta(*sizes, **kwargs):
        kwargs.pop("pin_memory", None)
        kwargs["device"] = "meta"
        return real_empty(*sizes, **kwargs)

    torch.empty = empty_on_meta
    try:
        yield
    finally:
        torch.empty = real_empty


def _read_only_table(path: str, rows: int, width: int):
    """A read-only view of one staged table. Never writable: registration is
    what needed write access, and registration is gone."""
    import mmap
    import os

    import numpy as np

    size = os.path.getsize(path)
    if size != rows * width:
        raise ValueError(
            f"{path}: expected {rows:,} x {width} = {rows * width:,} bytes, "
            f"found {size:,}. The config's shapes and the file's size are the "
            "same fact by two routes.")
    handle = os.open(path, os.O_RDONLY)
    try:
        mapping = mmap.mmap(handle, size, flags=mmap.MAP_SHARED,
                            prot=mmap.PROT_READ)
    finally:
        os.close(handle)
    return np.frombuffer(mapping, dtype=np.uint8).reshape(rows, width), mapping


class AscendStagedEngramEmbedding:
    """Mixin over upstream's ``ParallelEngramEmbedding``.

    Mixed in ahead of it so these win the MRO::

        class AscendEngramEmbedding(AscendStagedEngramEmbedding,
                                    ParallelEngramEmbedding):
            pass
    """

    #: Set once construction has bound the staged tables.
    _staged_tables: tuple | None = None

    def __init__(self, *args, **kwargs) -> None:
        with tables_not_allocated(), _upstream_offload_line_suppressed():
            super().__init__(*args, **kwargs)

        self._assert_nothing_was_allocated()
        self._layer_id = None          # bound by the owner; see bind_layer()

        # vllm.logger: a bare module logger is silent once vLLM has configured
        # logging, and this line is the only evidence that the table was left
        # unallocated.
        from vllm.logger import logger
        logger.info(
            "Engram table not allocated: %d rows x %d stays in the staged "
            "read-only mapping, 0 bytes resident and 0 bytes pinned per rank. "
            "Upstream's \"%s\" line is suppressed here because the allocation "
            "it describes does not happen.",
            getattr(self, "part_num_embeddings", -1),
            getattr(self, "dim", -1), _STALE_OFFLOAD_LOG)

        self._allocate_staging_at_construction()

    def _allocate_staging_at_construction(self) -> None:
        """Size the per-step buffers here, not on first lookup.

        Three constraints and only one place satisfies all of them.

        *Before graph capture.* Replay reads the device addresses recorded at
        capture, so a buffer allocated later -- on the first lookup, say -- is a
        different address than the one the graph replays against, and nothing
        reports it. Construction is the only point guaranteed to precede
        capture.

        *Once per worker.* Each rank builds its own layer, so this runs exactly
        once each without a registry to keep in step.

        *Sized to the batch, not the sequence.* ``lookup`` receives one row per
        token in the forward, so the bound is
        ``scheduler_config.max_num_batched_tokens`` -- 4096 here -- not
        ``max_model_len``. Those differ by a factor of two and the wrong one is
        not free in either direction: too large wastes about 52 MB a layer, too
        small raises at the first batch that exceeds it. The tight one is taken
        because ``lookup`` already refuses an oversized batch by name, so a
        mistake here surfaces as that error rather than as an overrun. It also
        matches what rope_dsv4 sizes its own per-step buffers with, which is the
        same quantity for the same reason.
        """
        from vllm.config import get_current_vllm_config
        from vllm.platforms import current_platform

        from vllm_ascend.models.deepseek_v4_1.engram_layout import EngramLayout

        vllm_config = get_current_vllm_config()
        if vllm_config is None:
            raise RuntimeError(
                "engram staging cannot be sized: no vllm config is current "
                "during construction. Deferring the allocation to the first "
                "lookup would place it after graph capture, and replay reads "
                "the addresses recorded at capture -- which fails silently, "
                "not loudly.")
        layout = EngramLayout.from_config(vllm_config.model_config.hf_text_config)
        self.allocate_staging(
            layout, vllm_config.scheduler_config.max_num_batched_tokens,
            current_platform.device_type)
        self.bind_staged_tables(layout, self._own_layer_id(layout))

    def _own_layer_id(self, layout) -> int:
        """Which engram layer this instance is, from its own row count.

        Upstream's constructor takes ``num_embeddings, dim, head_sizes,
        block_size, cpu_offload`` -- no prefix and no layer id -- so there is
        nothing to read the identity off directly. The row count supplies it:
        the two engram layers have different vocabularies (384006168 and
        384016682 in this checkpoint), and the layout carries the pairing.

        Ambiguity refuses rather than picking the first match. Two layers with
        equal row counts would make this a coin flip, and the wrong table maps
        cleanly, gathers cleanly, and returns embeddings from the other layer --
        no error at any point.
        """
        rows = getattr(self, "num_embeddings", None)
        matches = [lid for lid, n in zip(layout.layer_ids, layout.num_embeddings)
                   if n == rows]
        if len(matches) == 1:
            return matches[0]
        raise RuntimeError(
            f"cannot tell which engram layer this is: {rows} rows matches "
            f"{matches or 'none'} of {dict(zip(layout.layer_ids, layout.num_embeddings))}. "
            "The identity comes from the row count because upstream's "
            "constructor carries neither a prefix nor a layer id. Binding the "
            "wrong layer's table would map, gather and return embeddings from "
            "the other layer without erroring anywhere.")

    # -- guards --------------------------------------------------------------

    def _assert_nothing_was_allocated(self) -> None:
        """The interception, not the intention.

        Checked rather than trusted because the failure mode of the allocation
        succeeding is not an error -- it is 188.83 GiB of pinned host memory per
        rank and, on a host large enough to provide it, the registration that
        took a card down.
        """
        for name in ("weight", "weight_scale_inv"):
            param = getattr(self, name, None)
            if param is None:
                continue
            if param.device.type != "meta":
                raise RuntimeError(
                    f"engram {name} was allocated on {param.device} with "
                    f"{param.numel() * param.element_size() / 1024**3:.1f} GiB "
                    "of real storage. It must not be: the table is read from "
                    "the staged mapping, and the pinned-host path this would "
                    "otherwise take is the one that took an NPU down. Note "
                    "that running out of memory here is not the guard -- a "
                    "larger host would have let it through.")

    def _storage(self):
        raise RuntimeError(
            "engram _storage() is the UVA / host-registration entry point and "
            "must not be reached. The device does not address the engram "
            "tables on this platform: the CPU gathers the rows a step needs "
            "out of a read-only mapping and one H2D copy carries them. If this "
            "raised during a forward pass, something is still on upstream's "
            "direct-read path -- see engram_layout for the replacement.")

    # -- the staged tables ---------------------------------------------------

    def bind_staged_tables(self, layout: EngramLayout, layer_id: int,
                           stage_dir: str = DEFAULT_STAGE_DIR) -> None:
        """Map this layer's staged weight and scale tables, read-only."""
        rows = dict(zip(layout.layer_ids, layout.num_embeddings))[layer_id]
        weight, wmap = _read_only_table(
            f"{stage_dir}/{staged_file_name(f'layers.{layer_id}.engram.embed.weight')}",
            rows, layout.dim)
        scale, smap = _read_only_table(
            f"{stage_dir}/{staged_file_name(f'layers.{layer_id}.engram.embed.scale')}",
            rows, layout.n_scales)
        self._staged_tables = (weight, scale, wmap, smap)
        self._layer_id = layer_id

    def staged_tables(self):
        if self._staged_tables is None:
            raise RuntimeError(
                f"engram tables for layer {self._layer_id} are not bound; call "
                "bind_staged_tables() after construction. They are not "
                "allocated by the constructor on purpose -- see this module's "
                "docstring.")
        return self._staged_tables[0], self._staged_tables[1]

    # -- the lookup path -----------------------------------------------------

    def allocate_staging(self, layout: EngramLayout, max_tokens: int,
                         device) -> None:
        """Pre-allocate the per-step buffers, once, at the maximum batch.

        Allocated once and reused so the device addresses stay put: ACL graph
        replay requires the addresses recorded at capture, and a fresh buffer
        per step desynchronises it.

        The host side is **not** pinned. Pinning would make the H2D faster and
        ``aclrtMallocHost`` at this size (about 104 MB at 8192 tokens) is not
        the call that took a card down -- that was ``aclrtHostRegisterV2``. But
        the two are close enough in this area that turning it on should be a
        decision someone makes deliberately rather than something that arrives
        inside an optimisation, so it is left off and noted here.
        """
        import numpy as np

        (wshape, sshape) = layout.staging_shapes(max_tokens)
        self._layout = layout
        self._max_tokens = max_tokens
        self._host_weight = np.empty(wshape, dtype=np.uint8)
        self._host_scale = np.empty(sshape, dtype=np.uint8)
        self._dev_weight = torch.empty(wshape, dtype=torch.uint8, device=device)
        self._dev_scale = torch.empty(sshape, dtype=torch.uint8, device=device)

        # 捕获路线的 pinned 缓冲：【捕获前】分配一次，地址此后不再变动。
        # 惰性分配在这里是错的 —— NPU 没有 eager warmup forward，第一次 forward
        # 就在捕获里，届时任何分配/H2D 都会被 ACL 拒（107030）。
        self._pin_ready = False
        if _capture_route_enabled():
            self._allocate_pinned_for_capture(wshape, sshape)

    def _allocate_pinned_for_capture(self, wshape, sshape) -> None:
        """捕获期可用的 host 缓冲。异步拷贝只有对 pinned 内存才是可捕获的。

        🔴 和被禁的那个调用的关系，按【产物】核过，不是按名字猜的（2026-09-14）：
            nm -D --undefined-only libtorch_npu.so
              U aclrtFreeHost
              U aclrtMallocHost          ← pinned 分配走这个，【已链接】
            aclrtHostRegisterV2 只以【字符串】出现，且旁边就是
              "Failed to find function aclrtHostRegisterV2"
            ⇒ 它是 dlsym 动态查找的另一个特性（UVA / host register），
              不在 pin_memory 这条路上。**HostRegisterV2 的禁令不受影响。**
        ⚠️ 残余不确定性：dlsym 的调用者无法只靠链接关系确定。所以这条路默认关着，
           打开它是一次显式的决定（见 _CAPTURE_ROUTE_ENV），不是优化里顺带发生的事。

        ids 两种 dtype 各备一份：在捕获里做 dtype 转换要么带分配、要么要额外 kernel，
        而这两个缓冲一共只有几 MB（8192 token × n_hash_cols × 12 B）。
        """
        # 🔴 device="cpu" 必须显式写。第一版没写，实测(xr_hostfunc2_100807)当场死在
        #    `RuntimeError: Only dense CPU tensors can be pinned` —— 因为 worker 里
        #    默认设备是 npu，torch.empty(..., pin_memory=True) 就去 pin 了一个 NPU 张量。
        #    ⚠️ "pinned host buffer" 这个名字读起来已经说了它在 host 上，而代码没说。
        rows = wshape[0]
        self._pin_ids64 = torch.empty(rows, dtype=torch.int64,
                                      device="cpu", pin_memory=True)
        self._pin_ids32 = torch.empty(rows, dtype=torch.int32,
                                      device="cpu", pin_memory=True)
        self._pin_weight = torch.empty(wshape, dtype=torch.uint8,
                                       device="cpu", pin_memory=True)
        self._pin_scale = torch.empty(sshape, dtype=torch.uint8,
                                      device="cpu", pin_memory=True)
        for _name, _t in (("ids64", self._pin_ids64), ("ids32", self._pin_ids32),
                          ("weight", self._pin_weight), ("scale", self._pin_scale)):
            # 记账位要有区分力：is_pinned() 为假时这条路【静默退化】成普通 pageable，
            # 而 pageable 的异步拷贝在捕获期是不可捕获的 —— 那会变成下一个静默失效。
            if _t.device.type != "cpu" or not _t.is_pinned():
                raise RuntimeError(
                    f"engram capture buffer {_name} is on {_t.device} "
                    f"pinned={_t.is_pinned()}; the capture route needs "
                    "dense pinned CPU memory or its async copies are not "
                    "capturable.")
        self._pin_ids64_np = self._pin_ids64.numpy()
        self._pin_ids32_np = self._pin_ids32.numpy()
        self._pin_weight_np = self._pin_weight.numpy()
        self._pin_scale_np = self._pin_scale.numpy()
        self._pin_ready = True

        from vllm.logger import logger
        logger.info(
            "engram capture route ON: pinned staging allocated "
            "(ids %d rows, weight %s, scale %s). The host gather will be "
            "submitted as a graph node via _launch_host_func.",
            rows, tuple(wshape), tuple(sshape))

    def _staging_host_views(self, rows: int):
        """The host rows this forward will gather into.

        🔴 One buffer, reused every forward, is the hazard: the H2D issued out
        of it may still be in flight when the next forward's numpy gather
        rewrites those bytes. Under ``"double"`` the forwards alternate
        between two buffers, so the bytes a copy is reading are not the bytes
        the next gather writes.

        That is the one remedy here that **changes no timing**: no wait, no
        synchronise, nothing added to the forward that other ranks are not
        also doing. After a flight died in the MoE all-to-all with a
        device-wide sync in the forward, that property is the point, not a
        detail -- it makes the hypothesis testable without the test itself
        being able to break collectives.

        Allocated lazily so the second buffer costs nothing unless asked for.
        """
        if not _double_buffered():
            return self._host_weight[:rows], self._host_scale[:rows]
        import numpy as np

        if getattr(self, "_host_weight_alt", None) is None:
            self._host_weight_alt = np.empty_like(self._host_weight)
            self._host_scale_alt = np.empty_like(self._host_scale)
            self._staging_parity = 0
        self._staging_parity ^= 1
        if self._staging_parity:
            return self._host_weight_alt[:rows], self._host_scale_alt[:rows]
        return self._host_weight[:rows], self._host_scale[:rows]

    def _submit_under_capture(self, indices: torch.Tensor, out: torch.Tensor,
                              rows: int, weight_table, scale_table) -> None:
        """把 host gather 作为【图节点】提交，一次图都不切。

        三步各自换成可被捕获的那一种：

          ① hash id 的 D2H  →  异步拷进捕获前就固定的 pinned buffer
          ② numpy gather    →  _launch_host_func，录成图节点，重放时在 host 执行
          ③ staged rows H2D →  从同一块 pinned buffer 异步拷回

        ⚠️ **host func 会阻塞本流**：ACL 的语义是回调未返回、本流不许继续。
           另一条线实测（43 层 × 每层一个回调）：每步 8.58 ms 花在
           HOSTFUNC_CALLBACK + NOTIFY_WAIT 上，占 token 时间的 41.3%，零重叠；
           他们把提交 fork 到侧流、在 combine 后 join，才拿到 −5.97 ms/token。
           ⇒ **"进得了图" 和 "变快了" 是两件事。** 本函数只做前者；侧流重叠是下一步，
             而且在量到串行化【之前】不该先写，否则又是一个没有对照的优化。

        ⚪ **行数是 token 数的纯函数**，与 hash 内容无关：
           rows = num_tokens * part_n_hash_cols（:458），part_n_hash_cols 在
           __init__ 里定死（common/engram.py:649，TP=1 下 = n_hash_cols），
           而 gather_into_staging 的 docstring 明写 "No de-duplication"。
           ⇒ 同一张捕获图里 num_tokens 是桶大小、恒定 ⇒ rows 恒定 ⇒ 可以烘进图。
           🔴 **触发条件**：哪天有人给 gather 加了去重，rows 就变成【内容相关】，
             而这里烘进图的 [:rows] 会静默地切错 —— 那时这条路必须重新判，
             不是调参数能补的。

        ⚪ 这里【绕开】了 _staging_host_views 的双缓冲：捕获把地址烘进图里，
           而双缓冲每次换地址 —— 两者不兼容。捕获路线用固定的那一块。
        """
        import torch_npu

        flat = indices.reshape(-1)
        if flat.numel() != rows:
            raise AssertionError(
                f"{flat.numel()} ids for {rows} staging rows -- the row "
                "numbering the gather recomputes would not match. "
                "(EngramLayout.row_index is the only definition of it.)")
        if flat.dtype == torch.int64:
            pin_ids, ids_np = self._pin_ids64, self._pin_ids64_np
        elif flat.dtype == torch.int32:
            pin_ids, ids_np = self._pin_ids32, self._pin_ids32_np
        else:
            raise TypeError(
                f"hash ids are {flat.dtype}; the pinned id buffers are int32 "
                "and int64. Converting here would allocate inside the "
                "capture, which is the thing this path exists to avoid.")

        stream = torch.npu.current_stream(indices.device)
        _ensure_subscribe_report(stream)

        pin_ids[:rows].copy_(flat, non_blocking=True)

        # 🔴 payload 必须活到进程结束：节点在【重放】时才读它。
        #    只有捕获期的调用会走到这里（重放不执行 Python），
        #    所以它的上界是 (被捕获的形状数 × warmup 次数)，不随请求增长。
        payload = (weight_table, scale_table, ids_np,
                   self._pin_weight_np, self._pin_scale_np, rows)
        _CAPTURED_PAYLOADS.append(payload)
        if len(_CAPTURED_PAYLOADS) == 1:
            # 🔴 与 "callback EXECUTED" 对称的证据行。两条一起才分得开三种去向：
            #   无 SUBMITTED              ⇒ 这两层根本没走到捕获路径
            #   有 SUBMITTED 无 EXECUTED  ⇒ 录进去了但没被投递，或图没被重放
            #   两条都有                  ⇒ 路线真的跑过
            # ⚪ _CAPTURED_PAYLOADS 原先【只被写、从不被读】—— 一个没有读者的
            #    记账位，记的恰好是我们最想知道的那个数。
            from vllm.logger import logger
            logger.info(
                "engram host gather SUBMITTED to the graph (first time), "
                "rows=%d. Pair this with the EXECUTED line: submitted without "
                "executed means the node was recorded but never delivered.",
                rows)
        torch_npu.npu._launch_host_func(stream, _engram_host_gather, payload)

        self._dev_weight[:rows].copy_(self._pin_weight[:rows],
                                      non_blocking=True)
        self._dev_scale[:rows].copy_(self._pin_scale[:rows],
                                     non_blocking=True)

        self._dequantize_staged(rows, out)

    def lookup(self, indices: torch.Tensor, out: torch.Tensor,
               background: bool = False) -> None:
        """Gather on the host, copy once, let the kernel read HBM.

        ``indices`` is ``[num_tokens, n_hash_cols]`` of table rows. Flattened
        row-major it is already in staging order: upstream numbers its rows
        ``token * LOCAL_HEADS + head`` and, with the hash columns undivided,
        that is exactly ``EngramLayout.row_index``. The two agreeing is checked
        rather than assumed -- see ``assert_head_sharding_is_degenerate``.
        """
        import numpy as np

        from vllm_ascend.models.deepseek_v4_1.engram_layout import \
            gather_into_staging

        num_tokens = indices.shape[0]
        rows = num_tokens * self.part_n_hash_cols
        if not rows:
            return

        layout = getattr(self, "_layout", None)
        if layout is None:
            raise RuntimeError(
                "engram staging buffers are not allocated; call "
                "allocate_staging() after construction. The tables are not "
                "device-addressable on this platform, so there is no path that "
                "reads them without this.")
        layout.assert_head_sharding_is_degenerate(
            self.part_n_hash_cols, self.n_hash_cols, self.head_start)
        if num_tokens > self._max_tokens:
            raise ValueError(
                f"{num_tokens} tokens exceeds the {self._max_tokens} the "
                "staging buffers were sized for. They are allocated once so "
                "their device addresses stay stable for graph replay, so this "
                "cannot grow on demand.")

        weight_table, scale_table = self.staged_tables()

        # 🔴 捕获中 ⇒ 【选路】，不是拒绝。
        #    同一个谓词，之前拿来 raise，现在拿来选提交方式。原先那条拒绝
        #    仍然是默认行为：捕获路线没打开、或 pinned 缓冲没就绪时，照旧拒绝。
        #    （拒绝的理由一个字没变，见 _refuse_under_capture 的 docstring。）
        if _stream_is_capturing():
            if not (_capture_route_enabled()
                    and getattr(self, "_pin_ready", False)):
                _refuse_under_capture(num_tokens)
            self._submit_under_capture(indices, out, rows,
                                       weight_table, scale_table)
            return

        # The one synchronising transfer on this path: the hash ids have to
        # reach the host to be gathered. Small (24 int32 per token) but it is a
        # sync point, and the way to remove it is to hash on the host from the
        # token ids the runner already holds, rather than to make it faster.
        row_ids = indices.reshape(-1).to("cpu", torch.int64).numpy()

        from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

        # 🔴 Here, at the copy, not at the parse. The launch that set this
        # variable and learned nothing had only vLLM's "Unknown environment
        # variable" warning to go on, which proves it was set and not that
        # this line ran.
        announce_once("engram_h2d",
                      "engram staging H2D mode %r reached the copy "
                      "(rows=%d).", _h2d_mode(), rows)
        host_weight, host_scale = self._staging_host_views(rows)
        gather_into_staging(weight_table, scale_table, row_ids,
                            host_weight, host_scale)

        # ⚠️ The host source is **one buffer, reused every forward**, and the
        # next forward's gather_into_staging overwrites it. Whether that is
        # safe depends entirely on whether this H2D may still be in flight
        # when that happens:
        #
        #   CUDA, pageable source   safe -- the runtime stages the copy, so
        #                           the source may be touched immediately
        #   CUDA, pinned source     UNSAFE -- and this buffer would be pinned
        #                           the day someone takes the speed
        #   torch_npu              not established. The transfer goes through
        #                           copy_between_host_and_device_opapi and
        #                           nothing read here says which of the two it
        #                           behaves like.
        #
        # 🔴 If it is the unsafe one, the device gets a mix of this forward's
        # rows and the previous forward's, which is a forward whose result
        # depends on how many requests came before it -- and deterministically
        # so, since the sequence of requests fixes the sequence of overwrites.
        # That is the shape of the trajectory integration measured: the Nth
        # request reproducibly differs from the first, across independent runs.
        #
        # Not asserted, and not fixed by default: the switch is the one-
        # variable test. If waiting removes the trajectory, this was it, and
        # the default flips on that measurement rather than on this comment.
        non_blocking = not _blocking_h2d()
        self._dev_weight[:rows].copy_(
            torch.from_numpy(host_weight), non_blocking=non_blocking)
        self._dev_scale[:rows].copy_(
            torch.from_numpy(host_scale), non_blocking=non_blocking)
        if _h2d_mode() == "sync":
            _synchronize_current_stream()

        self._dequantize_staged(rows, out)

    def _dequantize_staged(self, rows: int, out: torch.Tensor) -> None:
        """Read the staged rows out of HBM and write the embedding.

        The seam the device-side kernel binds to. It reads
        ``self._dev_weight[:rows]`` and ``self._dev_scale[:rows]``, which are
        contiguous, so it needs no hash ids, no vocabulary offset and -- this
        is the load-bearing one -- no ``cols // QUANT_BLOCK`` scale gather: the
        scales are ``n_scales`` contiguous bytes per row, read with
        ``tl.arange`` and broadcast.
        """
        flat = _flatten_out_rows(out, rows)
        if envs_dsv41_engram_dequant_torch():
            dequantize_staged_torch(
                self._dev_weight[:rows], self._dev_scale[:rows], flat,
                quant_block=self._layout.quant_block)
            return

        launch = _staged_dequant_launcher()
        launch(self._dev_weight[:rows], self._dev_scale[:rows], flat,
               dim=self._layout.dim, quant_block=self._layout.quant_block)


#: How the staging H2D is made safe against its host source being rewritten.
#: Four settings rather than a flag, because the first version bundled two
#: independent changes -- ordering the copy and draining it -- into one, and
#: the launch that used it died somewhere neither of them is.
#:
#:   "0" / ""   today: non_blocking=True, nothing waits
#:   "block"    non_blocking=False. Orders the copy; does not prove it drained
#:   "sync"     "block" plus a **current-stream** synchronise
#:   "double"   two host buffers, used alternately. Removes the reuse entirely
#:              and changes no timing at all, so it cannot desynchronise ranks
#:   "1"        alias for "sync" -- what the first flight used
H2D_MODES = ("0", "", "block", "sync", "double", "1")


def _h2d_mode() -> str:
    from vllm_ascend import envs

    mode = str(getattr(envs, "VLLM_ASCEND_DSV41_ENGRAM_BLOCKING_H2D", "")
               or "").strip()
    if mode not in H2D_MODES:
        raise ValueError(
            f"VLLM_ASCEND_DSV41_ENGRAM_BLOCKING_H2D={mode!r} is not one of "
            f"{H2D_MODES}. Refusing rather than falling back to the default: "
            "a typo that silently selects today's behaviour makes the "
            "experiment read as 'the change had no effect'.")
    return "sync" if mode == "1" else mode


def _blocking_h2d() -> bool:
    """Whether the copy itself waits."""
    return _h2d_mode() in ("block", "sync")


def _double_buffered() -> bool:
    """Whether the host source alternates instead of being reused."""
    return _h2d_mode() == "double"


def _synchronize_current_stream() -> None:
    """Wait for the staging copy before its host source can be rewritten.

    ⚠️ ``non_blocking=False`` alone is not obviously enough: it orders the
    copy on the stream, and what this needs is for the copy to have *drained*
    before the next forward's numpy gather writes the same host bytes. Those
    are two different claims, which is why they are two different settings.

    🔴 Scoped to the current stream, not the device. The first version called
    ``torch.npu.synchronize()``, which is a **device-wide** host block placed
    inside the forward of a model running EP=4 collectives -- and that flight
    died in ``libccl_kernel.so`` with an aicpu exception, in the MoE
    all-to-all, without entering this function's own copy at all. Whether the
    sync caused it is not established; what is established is that the
    instrument was far broader than the thing it needed to wait for, and an
    instrument that heavy has no business being the first rung of the
    experiment.
    """
    stream = None
    for module in (getattr(torch, "npu", None), getattr(torch, "cuda", None)):
        getter = getattr(module, "current_stream", None)
        if getter is None:
            continue
        try:
            stream = getter()
        except Exception:                                   # noqa: BLE001
            stream = None
        break
    if stream is None:
        return
    try:
        stream.synchronize()
    except Exception:                                       # noqa: BLE001
        pass


# 🔴 被捕获的图【重放的是设备操作】。engram 的 host gather 里没有设备操作，
#    所以三步要各自变成【可被捕获的那一种】才进得去图：
#
#      ① hash id 的 D2H   →  异步拷进【捕获前就分配好的 pinned buffer】
#      ② numpy gather     →  torch_npu.npu._launch_host_func，录成【图节点】
#      ③ staged rows H2D  →  从同一块 pinned buffer 异步拷回
#
#    ⚠️ 现在失败的是 ①：`indices.to("cpu", ...)` 是【同步且带分配】的拷贝，
#       CANN 在捕获期拒绝它（107030）。异步 + pinned + 地址预先固定才是可捕获的形状。
#
# ⚪ 默认【关】。打开它要显式设 VLLM_ASCEND_DSV41_ENGRAM_CAPTURE_ROUTE=1。
#    偏严的错会被后续工作发现，偏松的错会发一张假的安全证明 —— 所以默认留在拒绝那一侧。
_CAPTURE_ROUTE_ENV = "VLLM_ASCEND_DSV41_ENGRAM_CAPTURE_ROUTE"


def _capture_route_enabled() -> bool:
    import os
    return os.environ.get(_CAPTURE_ROUTE_ENV, "0") == "1"


# 🔴 捕获图里保留着引用这些对象的节点，回调在【重放】时才执行。
#    payload 被 GC 掉 ⇒ 重放时读到已释放的对象 ⇒ 静默错值。整个进程都要留着。
_CAPTURED_PAYLOADS: list = []
# 🔴 "回调一次都没跑" 和 "跑了但结果不对" 在产物上长得一样。记账位无条件存在。
_HOST_GATHER_CALLS = [0]
_HOST_GATHER_ERRORS: list = []
_REPORT_SUBSCRIBED: set = set()


def _ensure_subscribe_report(stream) -> None:
    """回调必须先订阅本流，否则【不会被投递】——而且不报错。

    🔴 torch_npu >= 2.10 在 NPUGraph 内部已经预先订阅了捕获流，再订一次会得到
    **107011**。只吞这一个错误码：把整个 except 吞掉会让真正的订阅失败也消失，
    表现为"回调从不执行、零报错、只是数值不对"（另一条线实测踩过）。
    """
    import torch_npu

    key = int(stream.npu_stream)
    if key in _REPORT_SUBSCRIBED:
        return
    try:
        torch_npu.npu._subscribe_report(stream)
    except RuntimeError as exc:                              # noqa: BLE001
        if "107011" not in str(exc):
            raise
    _REPORT_SUBSCRIBED.add(key)


def _engram_host_gather(payload) -> None:
    """图节点的回调体：在 host 上把这一步要的行 gather 进 pinned 输出缓冲。

    🔴 异常绝不能逃出去。ACL 这一侧没有地方接，抛出去等于"什么都没发生"，
    而下游会照常读到【上一次】的内容 —— 一个成功启动、零报错的错答案。
    ⇒ 记进 ``_HOST_GATHER_ERRORS`` 并打日志，让它有产物可查。
    """
    import numpy as np

    from vllm_ascend.models.deepseek_v4_1.engram_layout import         gather_into_staging

    try:
        weight_table, scale_table, ids_np, host_weight, host_scale, rows = payload
        row_ids = ids_np[:rows]
        if row_ids.dtype != np.int64:
            row_ids = row_ids.astype(np.int64)
        gather_into_staging(weight_table, scale_table, row_ids,
                            host_weight[:rows], host_scale[:rows])
        _HOST_GATHER_CALLS[0] += 1
        if _HOST_GATHER_CALLS[0] == 1:
            # 🔴 "回调一次都没被投递" 是这条路最可能的静默失效，而它的产物是
            #    【什么都没有】—— 没有报错、没有缺键，只有陈旧的 staged_rows。
            #    ⇒ 第一次执行必须留下一行，否则"成功"和"从没跑过"读起来一样。
            #    只印一次：这是每步都会走的路径。
            from vllm.logger import logger
            logger.info(
                "engram host gather callback EXECUTED (first time), "
                "rows=%d. This is the line that distinguishes 'the graph "
                "node ran' from 'the node was never delivered'.", rows)
    except BaseException as exc:                             # noqa: BLE001
        _HOST_GATHER_ERRORS.append(repr(exc))
        # vllm.logger，理由同 __init__ 里那条：裸模块 logger 在 vLLM 配置完日志后是哑的，
        # 而这条是这次失效【唯一】的证据行。
        from vllm.logger import logger
        logger.error(
            "engram host gather callback raised inside a graph node: %r. "
            "The staged rows for this step are NOT this step's rows -- the "
            "device will read whatever the buffer held before. This is a "
            "wrong answer, not a slowdown.", exc)


def host_gather_accounting() -> tuple:
    """(提交进图的次数, 回调执行次数, 回调内异常数)。无条件可读，包括全为零。

    🔴 三个数缺一不可：只有 (执行, 异常) 时，"从没提交" 和 "提交了没被投递"
    读数完全一样 —— 而这两者的去向相反。
    """
    return (len(_CAPTURED_PAYLOADS), _HOST_GATHER_CALLS[0],
            len(_HOST_GATHER_ERRORS))


def _refuse_under_capture(num_tokens: int) -> None:
    """Say who is incompatible with graph capture, in this frame.

    🔴 This path cannot be captured, and not for one removable reason. A
    captured graph replays **device** operations; this lookup does three
    things none of which are device operations:

        1. D2H of the hash ids (the one synchronising transfer here)
        2. ``gather_into_staging`` -- a numpy gather over host-resident tables
        3. H2D of the staged rows

    Upstream's ``EngramEmbedding.lookup`` is a single triton kernel over
    device-resident tables (``common/engram.py:716``) and captures fine. The
    host gather is this line's own adaptation, forced by the tables not being
    device-addressable on A5, so the incompatibility is ours to own and ours
    to state.

    ⚠️ Without this, the failure arrives as ``rtMemcpy: operation not
    permitted when a stream is capturing`` from inside
    ``copy_between_host_and_device_opapi`` -- a message that names the copy
    and says nothing about which design decision made the copy unreplayable.
    It reads as a driver or a torch_npu problem, which is the wrong team.

    What to do about it is a configuration choice, not a fix here: run with
    ``--enforce-eager`` (how every token this line has produced so far was
    produced), or put this lookup in an eager segment -- upstream's
    ``staged_rows`` exists to "keep lookup results alive across breakable
    graph segments" (``common/engram.py:936``), which is that shape.
    ``VLLM_USE_BREAKABLE_CUDAGRAPH=1`` is the switch; ⚠️ unverified on
    Ascend, where vllm_ascend keeps it opt-in on purpose, and
    ``DeepseekV41ForCausalLM`` is **not** in upstream's auto-enable set
    (``config/vllm.py:77``), so it is never turned on for us by default.

    🔴 **Do NOT silence this by switching the capture mode to relaxed**
    (``capture_error_mode="relaxed"`` in ``torch_npu/npu/graphs.py:619``,
    ``ACL_MODEL_RI_CAPTURE_MODE_RELAXED`` in ``acl_rt.h:405``). Measured
    2026-09-14: the CANN 9.1 runtime string this error comes from is only
    the first half of ``operation not permitted when a stream is capturing
    and the specified capture mode is not relaxed``; the sibling string in
    the same ``.so`` says ``the operation has been converted to a
    synchronous operation``. So relaxed does not make the copy capturable --
    it runs it **once, synchronously, during capture**, and records no node.
    Replay then never performs it, and every decode step reads whatever the
    warmup batch (``num_tokens=8``) left in ``staged_rows``. That failure is
    a silent wrong answer with a successful startup and no error at all,
    which is strictly worse than this refusal. ⚠️ The error text itself
    names ``relaxed`` and therefore invites exactly this mistake.

    🔵 The shape that does work (another line's measured implementation,
    SGLang + torch_npu 2.10 + CANN 9.1, DeepSeek MoE CPU offload): do not
    refuse and do not break the graph -- **route**. Under capture, submit the
    host work as a graph node via ``torch_npu.npu._launch_host_func`` (all of
    ``_launch_host_func`` / ``_subscribe_report`` / ``_unsubscribe_report``
    are present in this container). Their predicate is the same
    ``is_current_stream_capturing()`` used here, but it selects a path
    instead of raising. Three conditions they had to satisfy: the stream must
    be subscribed before capture (a second subscribe raises 107011 on
    torch_npu >= 2.10 -- swallow only that code); every H2D must be hoisted
    to a pre-capture point because capture rejects it (107030); and any event
    referenced by a captured record/wait node must outlive the process.
    ⚠️ A host func **blocks its stream** until it returns -- they measured
    41.3% of each token in ``NOTIFY_WAIT`` before forking to a side stream.
    """
    if not _stream_is_capturing():
        return
    raise RuntimeError(
        f"engram lookup was reached for {num_tokens} tokens while the stream "
        "is capturing, and it cannot be captured: it copies hash ids to the "
        "host, gathers from host-resident tables with numpy, and copies the "
        "staged rows back. None of the three is a device operation, so there "
        "is nothing for a graph to replay -- this is a property of the host "
        "gather, not a call that needs fixing. Upstream's lookup is one "
        "triton kernel over device-resident tables; ours is not, because the "
        "tables are not device-addressable on A5. Run with --enforce-eager, "
        "or arrange for this lookup to fall in an eager segment "
        "(VLLM_USE_BREAKABLE_CUDAGRAPH=1 -- unverified on Ascend).")


def _stream_is_capturing() -> bool:
    """Whether a graph capture is in progress, or ``False`` if unaskable.

    ⚠️ Deliberately permissive on the third state. A predicate that is not
    available must not turn into a refusal of a path that works today: every
    token this line has produced came through here with capture off. The cost
    of answering "no" wrongly is the CANN error we already get; the cost of
    answering "yes" wrongly is a server that will not start.
    """
    for module in (getattr(torch, "npu", None), getattr(torch, "cuda", None)):
        predicate = getattr(module, "is_current_stream_capturing", None)
        if predicate is None:
            continue
        try:
            # 🔴 ``is True``, not ``bool(...)``. A test double left on
            # torch.npu makes the predicate return a Mock, and ``bool(Mock)``
            # is True -- so the probe reports "capturing" and every caller
            # stands down, silently. Measured: eight engram tests and ten
            # slot-mapping tests failed only inside the full suite, for that
            # reason, and had been on the books as unexplained "existing
            # failures". Anything that is not the literal True reads as not
            # capturing, which is the safe direction: saying "yes" wrongly
            # silences a check, saying "no" wrongly costs at worst the CANN
            # error we would have got anyway.
            return predicate() is True
        except Exception:
            return False
    return False


def _flatten_out_rows(out: torch.Tensor, rows: int) -> torch.Tensor:
    """The caller's per-token view, as the per-gathered-row view the kernel takes.

    Upstream allocates ``staged_rows`` as ``[max_tokens, n_hash_cols, dim]`` and
    hands over ``[num_tokens, n_hash_cols, dim]``; the kernel indexes one
    gathered row per program and wants ``[num_tokens * n_hash_cols, dim]``.
    Both are right about their own axis, and the two disagree about what a row
    is -- which is why the kernel's guard fired with 192 against 8 rather than
    with anything about shapes.

    The flattening is not a choice. ``EngramLayout.row_index`` is
    ``token * n_hash_cols + col``, which *is* the row-major order of the first
    two axes, so the same memory answers both descriptions. That identity is
    the contract the CPU gather already relies on.

    ``view`` rather than ``reshape``, deliberately: reshape on a
    non-contiguous tensor returns a **copy**, the kernel writes into the copy,
    and the caller reads back whatever its buffer held before -- no error
    anywhere. view raises instead, which is the outcome worth having.
    """
    if out.ndim == 2:
        if out.shape[0] != rows:
            raise ValueError(
                f"the output buffer has {out.shape[0]} rows and the gather "
                f"produced {rows}. A two-dimensional buffer is already in "
                "gathered-row order, so these have to match.")
        return out
    if out.ndim == 3:
        if out.shape[0] * out.shape[1] != rows:
            raise ValueError(
                f"the output buffer is {tuple(out.shape)}, whose first two "
                f"axes give {out.shape[0] * out.shape[1]} rows, but the gather "
                f"produced {rows}. The buffer is [num_tokens, n_hash_cols, "
                "dim]; a mismatch means the caller and the gather disagree "
                "about how many hash columns a token has.")
        if not out.is_contiguous():
            raise ValueError(
                f"the output buffer {tuple(out.shape)} is not contiguous, so "
                "it cannot be viewed as gathered rows. Reshaping it would copy, "
                "the kernel would write into the copy, and the caller would "
                "read back its previous contents with nothing raised.")
        return out.view(rows, out.shape[-1])
    raise ValueError(
        f"the output buffer has {out.ndim} dimensions; this takes "
        "[rows, dim] or [num_tokens, n_hash_cols, dim].")


#: Where the kernel lives. The package path is where it belongs; the
#: build_artifacts path is where it is today, under a directory named for the
#: UVA design that this replaced.
_KERNEL_LOCATIONS = (
    ("vllm_ascend.ops.triton.engram_dequant_staging", None),
    ("engram_dequant_staging",
     "/workspace/code/dsv4_single_card/vllm_project/build_artifacts/npu_view/"
     "engram_dequant_staging.py"),
)


def _staged_dequant_launcher():
    """Resolve the kernel, preferring a real package location.

    Verification status, stated here because binding is not endorsing:

    - The arithmetic is settled **for the torch route**, in the gate:
      ``test_torch_dequant_matches_an_independent_computation`` checks it
      against a separately written expression, and
      ``test_each_scale_covers_its_own_block_only`` checks the broadcast
      positionally -- that one targets the maxdiff-28 failure by construction
      rather than by a tolerance.
    - The arithmetic is also checked against **golden's own rows** -- real fp8
      values, real ue8m0 scales, the bf16 the reference model produced, byte
      for byte including a negative zero that a value comparison would call
      equal. That comparison lives in ``build_artifacts/npu_view`` and is now
      invoked from the gate by
      ``tests/ut/models/test_deepseek_v4_1_engram_dequant_golden.py``.
    - ⚠️ This line twice claimed more than it could show. It first cited a
      test by a name that was not the file's, and then said no such file
      existed -- a conclusion from grepping for the *string* when the question
      was whether a *file* was there. Both times the substance was the same:
      the check was real and the gate never ran it. A docstring reporting
      verification status is a claim, and "exists" is not the property that
      matters; "runs" is.
    - ⚠️ The **kernel** route -- the one launched by default -- has no
      in-gate check of its arithmetic. What the torch tests above cover is the
      fallback that ``VLLM_ASCEND_DSV41_ENGRAM_DEQUANT_TORCH`` selects, not the
      Triton kernel.
    - The kernel has run on hardware. Triton compiles at first launch, and
      ``/root/.triton/cache`` holds two compilations of ``engram_dequant_staging``
      (2026-09-11 10:48:43 and 11:35:35).
    - Its file header still says "nothing here has run". That is stale -- it
      was written before the run -- and it is worth knowing that reading it is
      what produced the opposite conclusion here first. The compile cache is
      the artefact that settles it; a comment is not.

    Still open: performance at production scale, and getting that run's output
    written down somewhere other than a chat log.
    """
    import importlib

    errors = []
    for module_name, path in _KERNEL_LOCATIONS:
        try:
            if path is None:
                module = importlib.import_module(module_name)
            else:
                import importlib.util
                import os
                if not os.path.isfile(path):
                    errors.append(f"{path}: not present")
                    continue
                spec = importlib.util.spec_from_file_location(module_name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            return module.launch
        except Exception as exc:                      # noqa: BLE001
            errors.append(f"{module_name}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "the staged-dequantise kernel could not be loaded from any known "
        "location:\n  " + "\n  ".join(errors)
        + "\n\nEverything ahead of it is wired: this step's rows are already "
        "in HBM, laid out as EngramLayout describes. Only the kernel is "
        "missing.")


# -- a torch route through the same arithmetic -------------------------------
#
# Exists to answer one question: whether a triton-ascend compiler assertion is
# coming from this kernel. That assertion is an MLIR abort(), not an exception
# -- the worker vanishes with exit code None, no traceback, nothing in dmesg --
# so it cannot be caught, narrowed by a stack, or attributed by reading a log.
# The only cheap way to attribute it is to take one Triton kernel out of the
# picture and see whether the process still dies.
#
# This is not a fallback anybody should run in production: it materialises the
# dequantised rows through several ordinary torch ops where the kernel does one
# pass. It is a bisection instrument.


def envs_dsv41_engram_dequant_torch() -> bool:
    from vllm_ascend import envs

    return envs.VLLM_ASCEND_DSV41_ENGRAM_DEQUANT_TORCH == "1"


def dequantize_staged_torch(staging_weight: torch.Tensor,
                            staging_scale: torch.Tensor,
                            out: torch.Tensor,
                            quant_block: int) -> None:
    """The staging kernel's arithmetic, in torch.

    Same contract: row r of the weight and scale buffers is row r of the
    output. The scales are read contiguously and broadcast over their block --
    never indexed by ``cols // quant_block``, which is the form that reads the
    same values from HBM in a 2-D block and the wrong ones everywhere else.
    """
    rows, dim = staging_weight.shape
    n_scales = dim // quant_block
    if staging_scale.shape != (rows, n_scales):
        raise ValueError(
            f"scale buffer is {tuple(staging_scale.shape)}, expected "
            f"({rows}, {n_scales}) for dim={dim} and quant_block={quant_block}")

    values = staging_weight.view(torch.float8_e4m3fn).to(torch.float32)
    # ue8m0 is a power of two, so the byte *is* the fp32 exponent field.
    scales = (staging_scale.to(torch.int32) << 23).view(torch.float32)

    scaled = values.view(rows, n_scales, quant_block) * scales.unsqueeze(-1)
    out.copy_(scaled.view(rows, dim).to(out.dtype))
