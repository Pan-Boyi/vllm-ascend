# SPDX-License-Identifier: Apache-2.0
"""The q-norm / RoPE / SWA-insert prolog, decomposed.

Upstream fuses five steps into one CUDA op per cache dtype, and all three of
those symbols are absent on Ascend. Reached from ``_prepare_and_attn_eager`` on
every forward, so this is not a dormant path.

🔴 **The contract this was first written from was the wrong model's.** It came
from ``deepseek_v4/cpu/cpu_sparse.py:426`` -- a **V4.0** file -- describing the
same fused operator:

    Q side does per-head weight-free RMSNorm + GPT-J RoPE (zero-filling padded
    head slots); KV side does GPT-J RoPE + UE8M0 FP8 quant + paged insert.

That sentence is true of V4.0 and false of V4.1, and the difference is a
single argument the prose does not mention. The operator is shared and its
signature ends ``bool apply_q_norm=True``
(``csrc/libtorch_stable/torch_bindings.cpp:441``). V4.0 omits it everywhere and
takes the norm; **V4.1's attention path passes ``False``** at all three dtype
branches (``deepseek_v4_1/attention.py:871``, ``:892``, ``:907``), and its own
comment for the uint8 branch says only "Q side: GPT-J RoPE, zero-filling the
padding head slots" -- no norm. The reference agrees
(``dsv41_golden/ref/model.py:770-772``: ``wq_b`` then ``apply_rotary_emb``,
nothing between).

⚠️ **Not "every V4.1 call site"**, which is what this said first. V4.1's DSpark
path omits the argument at all six of its sites
(``deepseek_v4_1/{nvidia,amd}/dspark.py``) and therefore takes ``True``, the
same as V4.0. Whether that is deliberate upstream is not this line's question;
what matters here is that DSpark calls ``torch.ops._C`` directly rather than
through ``_fused_qnorm_rope_kv_insert``, so it does not reach this function --
on Ascend it would fail on the missing symbol, loudly. The scope of the claim
is **the attention path**, and it is written down because the first version of
this paragraph generalised one call site's argument to a whole model, which is
the same mistake in the other direction as the one it describes.

So the q side here does **RoPE only**. Measured cost of getting it wrong: q
left the projection matching the reference to the floor (2.6e-4) and reached
the operator 1000x off (3.1e-1), because a per-(token, head) RMS factor
ranging 0.21-1.23 had been applied to it.

⚠️ The lesson is not "read the contract" -- the contract was read. It is that
**a prose contract for a shared operator describes one caller's arguments**,
and the argument that differed was the one prose omits.

Two things follow that a reading of the name would have got wrong. The q norm
takes no learned weight -- the op is passed ``eps`` and nothing else, and the
layer's own ``q_norm`` is an ``RMSNorm(q_lora_rank)`` applied much earlier, to
``qr``, not to this per-head ``q``. And the kv side does **not** norm: ``kv``
arrives already normed from ``_split_qkv_and_norm``, which is why the op's name
says ``kv_rope`` and not ``kv_norm_rope``.

⚠️ Unverified, and stated rather than assumed:

* **Numerical equivalence with the fused kernel has never been checked by
  anyone.** The decomposition is transcribed from the descriptions above, not
  from a comparison against the op's output.
* **Where bf16 rounding falls.** The fused kernel's internal precision is not
  visible from any of these call sites. This computes in float32 and rounds
  once, at the end. If the kernel rounds between the norm and the rotation --
  as the indexer's K writer does, where it is load-bearing -- the two differ by
  a few ULP in a way no shape or dtype check would show.

Torch first, deliberately. Every step here has a named Ascend operator, and the
operator path is worth taking once these values are known to be right; taking
it first would mean debugging a decomposition and a binding at the same time.
"""

from __future__ import annotations

import torch

from vllm_ascend.models.deepseek_v4_1.rope_widen import cos_sin_for_slice


def rms_norm_weightless(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row RMS norm with no learned weight, in float32.

    🔴 **V4.1 does not take this path.** This is the operator's
    ``apply_q_norm=True`` branch, which is V4.0's: V4.1 passes ``False`` at
    every call site. Kept so that branch has something to point at, and
    labelled so the next reader does not take it for a general helper the way
    this module once did -- which cost q a factor of 0.21-1.23 per row, all
    the way to the attention operator.

    "Per-head" for ``q`` of ``[tokens, heads, head_dim]`` is the same statement
    as "per-row" over the last axis -- which is why this takes no axis
    argument: there is no other axis it could be.
    """
    acc = x.to(torch.float32)
    variance = acc.pow(2).mean(dim=-1, keepdim=True)
    return acc * torch.rsqrt(variance + eps)


def rope_tail(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
              rope_head_dim: int) -> torch.Tensor:
    """GPT-J rotation over the trailing ``rope_head_dim``, interleaved.

    ``cos``/``sin`` are per-token and already widened to ``rope_head_dim``;
    they broadcast across any axes between the token axis and the last one, so
    the same function serves ``q`` ``[T, H, D]`` and ``kv`` ``[T, D]``.

    The NoPE part is the *leading* ``head_dim - rope_head_dim``. Rotating the
    whole row instead is a change no shape check sees.
    """
    head_dim = x.shape[-1]
    if rope_head_dim > head_dim:
        raise ValueError(
            f"rope width {rope_head_dim} exceeds the head width {head_dim}")
    nope = head_dim - rope_head_dim
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    # clone, not just to(float32): `to` returns the *same* tensor when the
    # dtype already matches, and the in-place write below would then rotate the
    # caller's input. Upstream rewrites q in place but states "kv is unchanged;
    # attention reads kv solely via swa_kv_cache" -- so mutating kv here would
    # be wrong, and silently so, since the rotated values are what gets written
    # to the cache either way.
    out = x.to(torch.float32).clone()
    tail = out[..., nope:]
    pairs = tail.reshape(*tail.shape[:-1], rope_head_dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    # cos/sin are widened per channel; one value per pair is enough and both
    # channels of a pair carry the same one, so either index works. Taking [0]
    # rather than slicing keeps the pair axis aligned with `even`/`odd`.
    c = cos.reshape(*cos.shape[:-1], rope_head_dim // 2, 2)[..., 0]
    s = sin.reshape(*sin.shape[:-1], rope_head_dim // 2, 2)[..., 0]
    rotated = torch.stack((even * c - odd * s, odd * c + even * s), dim=-1)
    out[..., nope:] = rotated.reshape(*tail.shape)
    return out


_CLAMP_SLOT_CHECK = {"ran": 0, "skipped_under_capture": 0, "violations": 0}


def clamp_slot_check_accounting() -> dict:
    """无条件可读，包括全为零。"""
    return dict(_CLAMP_SLOT_CHECK)


def _payload_that_leaves_invalid_rows_untouched(current: torch.Tensor,
                                                rows: torch.Tensor,
                                                valid: torch.Tensor):
    """有效行写新值，无效行【写回它原本的值】—— 对读侧是真正的 no-op。

    🔴 上一版把无效行导向 block 0 并【写垃圾进去】。那是错的，而且三档判据
    全绿：捕获台架 OK、等价性 5 正例逐位相同、还专门写了守卫。**守卫守错了一侧** ——
    它检查"有没有有效 token 【写】到 block 0"，而真正的危险是"有没有层【读】
    block 0"。实测(xr_sanity5 的 dump)：``read.ori_win_left = ori_win_right = -1``
    在 **ratio-2 的 layer_02** 上就成立 ⇒ 那一层按 ``[0, seqused)`` 全范围读，
    block table 里的 null 填充会让它读到 block 0。
    ⇒ 写进去的 padding 垃圾会被读进注意力，静默错值，没有任何检查会响。

    改成写回原值之后，**读侧看不到任何变化**，所以那一层读不读 block 0 都无所谓。

    ⚠️ **仍然剩一个【弱】依赖，写出来而不是假装没有**：
    无效行被 clamp 到槽 0。若某个【有效】行也落在槽 0，两者是重复索引，
    谁最后写由实现定 ⇒ 有效行可能被写回的旧值盖掉。
    ⇒ 依赖的是「没有有效行落在槽 0」（比上一版的「没人读 block 0」弱得多，
      而且**可检查**）。由 ``_assert_clamp_slot_unused`` 在非捕获期查并记账。

    ⚫ **边界检查这一格在 NPU 上未验**（2026-09-14）。CPU 实测 ``index_put_``
    对 block/offset 越界都抛 ``IndexError``（而 ``npu_scatter_nd_update_`` 的
    offset 轴是**静默写进邻块**，block 轴是忽略 —— 同一接口的两个轴失效模式相反）。
    这是保留本实现而不换用 ``npu_scatter_nd_update_`` 的理由之一，但
    **NPU 上 ``index_put_`` 是否同样抛，没有实测** ⇒ 不要读成"(b) 在 NPU 上有边界检查"。

    ⚠️ 顺序前提：``current`` 的 gather 必须严格早于 scatter。同一个流内按下发顺序
    成立，捕获后的图保留同一条流的依赖。**若哪天有人把其中一个挪到别的流，这个前提
    就没了**，而失效形态是"无效行把一个刚写对的槽改回旧值" —— 无声。
    """
    return torch.where(valid.unsqueeze(-1), rows.to(current.dtype), current)


def _assert_clamp_slot_unused(valid: torch.Tensor,
                              slot_mapping: torch.Tensor) -> None:
    """查上面那个弱依赖：有没有【有效】行落在 clamp 目标（槽 0）。

    捕获期跳过（检查本身要读设备值），但**跳过要记账**，否则"跳过了"和
    "检查通过了"读数一样。
    """
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _stream_is_capturing

    if _stream_is_capturing():
        _CLAMP_SLOT_CHECK["skipped_under_capture"] += 1
        return
    _CLAMP_SLOT_CHECK["ran"] += 1
    if slot_mapping.ndim == 2:
        at0 = torch.logical_and(slot_mapping[:, 0] == 0, slot_mapping[:, 1] == 0)
    else:
        at0 = slot_mapping == 0
    if bool(torch.logical_and(valid, at0).any()):
        _CLAMP_SLOT_CHECK["violations"] += 1
        from vllm.logger import logger
        logger.warning(
            "scatter_rows: a VALID row targets slot 0, which is also where "
            "invalid (-1) rows are clamped. Their write-back of the old value "
            "may race with the valid write. Accounting: %s",
            dict(_CLAMP_SLOT_CHECK))


def scatter_rows(cache: torch.Tensor, rows: torch.Tensor,
                 slot_mapping: torch.Tensor, block_size: int) -> None:
    """Write one row per token into a paged ``[blocks, block_size, dim]`` cache.

    Two slot-mapping shapes are legal, and which one arrives is decided by the
    cache dtype rather than by the caller (``dsa_v1.py:782``)::

        FP8_ATTENTION and not a5_bf16_kv   ->  [num_tokens]      flat slots
        otherwise                          ->  [num_tokens, 2]   (block, offset)

    On A5 with a bf16 KV cache -- what we run -- the second is the default, not
    an alternative some plan happens to use.

    Both are accepted and dispatched on shape, because both are valid inputs
    that raise nothing when read as the other: a ``[T, 2]`` mapping read flat
    is ``2T`` slots with every offset treated as a slot index. The rows land
    somewhere real and belong to other tokens, so the values stay in range and
    the error surfaces only as accuracy. Which branch ran is logged once for
    that reason.

    ``-1`` means "do not write", which is how padded and already-cached tokens
    arrive. Filtered, not clamped: a clamp sends those rows to slot 0 of block
    0, over a real token's entry.
    """
    if slot_mapping.ndim == 2 and slot_mapping.shape[-1] == 2:
        blocks, offsets = slot_mapping[:, 0], slot_mapping[:, 1]
        valid = blocks >= 0
        b_i = torch.where(valid, blocks, blocks.new_zeros(())).to(torch.long)
        o_i = torch.where(valid, offsets, offsets.new_zeros(())).to(torch.long)
        _log_slot_form("(block, offset)", int(blocks.numel()))
        if blocks.numel() == 0:
            return
        # Indexed directly rather than flattened to block * block_size +
        # offset. The flattening is correct but needs block_size, and the two
        # numbers here are already the two the paged write wants.
        _assert_clamp_slot_unused(valid, slot_mapping)
        payload = _payload_that_leaves_invalid_rows_untouched(
            cache[b_i, o_i], rows, valid)
        cache.index_put_((b_i, o_i), payload)
        return

    if slot_mapping.ndim != 1:
        raise ValueError(
            f"slot_mapping is {tuple(slot_mapping.shape)}; this takes flat "
            "[num_tokens] slots or [num_tokens, 2] (block, offset) pairs.")
    valid = slot_mapping >= 0
    slots = torch.where(valid, slot_mapping, slot_mapping.new_zeros(())).to(torch.long)
    _log_slot_form("flat", int(slots.numel()))
    if slots.numel() == 0:
        return
    flat_cache = cache.view(-1, cache.shape[-1])
    _assert_clamp_slot_unused(valid, slot_mapping)
    payload = _payload_that_leaves_invalid_rows_untouched(
        flat_cache[slots], rows, valid)
    flat_cache.index_copy_(0, slots, payload)


_LOGGED_SLOT_FORMS: set[str] = set()


def _log_slot_form(form: str, rows: int) -> None:
    """Say which mapping form was taken, once per form, and how many rows.

    Both forms are legal and neither errors when read as the other, so the only
    way to know which one ran is to record it.

    Logged *before* the empty-batch return, and with the row count, because the
    first version sat after it: profile and dummy runs arrive with every slot
    at -1, so the function ran and said nothing. That made one line answer
    "which branch" while being read as an answer to "did this run at all" --
    and those have different silences.
    """
    if form in _LOGGED_SLOT_FORMS:
        return
    _LOGGED_SLOT_FORMS.add(form)
    # vllm.logger, not logging.getLogger(__name__): vLLM configures its own
    # logging and a bare module logger produces no output at all once that has
    # happened. This line exists to prove which branch ran, and a silent
    # instrument proves nothing -- "the log is dumb" and "the path was not
    # taken" look identical in the output.
    from vllm.logger import logger

    logger.info(
        "DSV4.1 SWA cache insert is using the %s slot mapping form "
        "(%d rows to write this call).", form, rows)


def qnorm_rope_kv_insert(*, q: torch.Tensor, kv: torch.Tensor,
                         positions: torch.Tensor, cos_sin_cache: torch.Tensor,
                         swa_kv_cache: torch.Tensor,
                         slot_mapping: torch.Tensor, eps: float,
                         rope_head_dim: int,
                         tap=None) -> torch.Tensor:
    """The bf16 branch, decomposed. Returns the rotated, normed ``q``.

    ``q`` is ``[tokens, heads, head_dim]`` and ``kv`` is ``[tokens, head_dim]``;
    both are rotated over the same trailing slice at the same positions, so the
    cos/sin pair is built once.

    ⚠️ ``eps`` is **unused on this path** and is kept for signature parity with
    the operator, which also takes it and also ignores it once
    ``apply_q_norm`` is false. Its only consumer here was the norm that no
    longer runs. Said out loud because an unused parameter is exactly where
    the next reader puts a norm back.

    ``tap`` -- optional ``(key, tensor) -> None`` -- is offered two values
    that exist only inside this function: the rotation's input on the q side
    (its trailing ``rope_head_dim``) and its output on the kv side. Passed in
    rather than imported here because this module has no layer identity and
    the caller does.
    """
    cos, sin = cos_sin_for_slice(cos_sin_cache, positions, rope_head_dim)
    # 🔴 No norm on the q side. V4.1 passes ``apply_q_norm=False`` at every
    # call site of the fused op (attention.py:881 and :901); the argument
    # defaults to True and V4.0 takes the default. This function used to run
    # the norm, because it was transcribed from V4.0's description of the
    # same operator -- see the module docstring.
    q_normed = q
    if tap is not None:
        # 🔵 Only the slice the rotation touches. The leading
        # ``head_dim - rope_head_dim`` dims pass through unchanged, so they
        # are already in ``sa_in.q`` and recording them again would be the
        # same numbers at eight times the size (measured on this model:
        # [T, 64, 512] against [T, 64, 64]).
        tap("rope.q.in", q_normed[..., q_normed.shape[-1] - rope_head_dim:])
    q_out = rope_tail(q_normed, cos, sin, rope_head_dim)
    # kv arrives normed; the op's name says kv_rope, not kv_norm_rope.
    kv_out = rope_tail(kv, cos, sin, rope_head_dim)
    if tap is not None:
        # 🔴 The value that actually goes to the cache. Without it the kv
        # path had no post-rope key at all: ``sa_in.kv`` is this function's
        # *argument*, which upstream's contract keeps unrotated, so over the
        # rope slice it is a stage below golden's key of the same name and
        # the two are not comparable there. Measured on flight-24 --
        # ``read.ori_kv.rows`` against the rotation of the pre-rope tensor
        # gives E_err/E_gold = 1e-6 -- so this is the tensor golden's
        # ``rope.kv.out`` should be compared with, and it is named for it.
        tap("rope.kv.out", kv_out)
    scatter_rows(swa_kv_cache, kv_out, slot_mapping, swa_kv_cache.shape[1])
    return q_out.to(q.dtype)
