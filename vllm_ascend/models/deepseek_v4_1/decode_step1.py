# SPDX-License-Identifier: Apache-2.0
"""Record the **first decode step**, so the two arms can be compared there.

Why this step and no other. Integration's same-side comparison on the
existing dumps is negative and complete: DP3's req_000 (group A) against
req_001 (group B), same process, same rank, adjacent, 923 shared keys,
923/923 covered, and **zero real model tensors differ**. Prefill is still
bit-identical, and it stayed bit-identical across both the A-2 fix and the
scale root-cause fix. Meanwhile the two arms already disagree numerically at
decode step 1 -- ' p' at -1.032261 against -0.433779, which is 0.6 logprob
and not last-bit jitter -- while the text only diverges at position 6.

So the divergence is born in one forward, and this is that forward.

🔵 What makes it cheap: the comparison is **same-side**. Two requests in one
server on one rank, no golden, no cross-side alignment, no G-1. The criterion
is the one that proved prefill identical -- key-by-key, and among the keys
that differ, the first in forward order is the source.

⚠️ Grouping, established by integration over 6/6 requests: req1 DP1#1,
req2 DP2#1, req3 DP3#1, req5 DP0#1 are all group A; req4 DP3#2 and req6
DP0#2 are all group B. The grouping is by **which prefill this is for that
rank**, and it does not group by rank identity. So this is the old
odd/even phenomenon, not a new one -- which says the quantity to look for is
one that **carries state across requests**.

🔴 Four candidates, and this records all four rather than the two with the
higher prior: the ring buffer's state, the engram hash cache, the
compressor's partial-group state, and the indexer's single-query branch.
Integration explicitly refused to rank them, because there is no decode-side
reading of any of them and a ranking would be prior rather than measurement.
That refusal is right, and ordering them here would quietly undo it.
"""

from __future__ import annotations

_SEEN_PREFILL: set = set()
_RECORDED_DECODE: dict = {}
_MAX_STEPS: "int | None" = None
_ARM_CALLS = 0
_ARMED_DECODE = 0
_ARMED_DECODE_AT_LAST_FLUSH = 0
_NO_IDENTITY_REPORTED = False

#: 🔴 由 model_runner 每次 forward 前供给(2026-09-15)。
#: 原来只有下面那条【扫 attn_metadata 找 req_ids_tensor】的路,而
#: AscendDSAReqMetadata 上【没有这个字段】—— 它是照上游 GPU 的 metadata 形状
#: 写的。于是 request_identity() 在 A5 上恒为 None ⇒ _SEEN_PREFILL 永远是空集
#: ⇒ is_first_decode_step 永远为假 ⇒ **整套 decode step-1 录制从未激活过**,
#: 而产物上表现为"只有 prefill 的 dump",和"decode 确实没问题"读起来一样。
#: ⇒ 改成问【生产的解析链】:runner 自己持有 input_batch.req_ids,
#:   那是同一个对象在 prefill 和 decode 两次 forward 上都成立的身份。
_RUNTIME_IDENTITY: "str | None" = None
#: 身份来源的记账。三个都要可读,否则"没人喂"和"喂了但是 None"读数一样。
_IDENTITY_SOURCE = {"runner": 0, "metadata": 0, "none": 0}


def set_runtime_identity(ident: "str | None") -> None:
    """由 model_runner 在每次 forward 之前设置本次 forward 的请求身份。

    传 None 表示"这一次说不出身份"(dummy run、批里不止一个请求),
    那就退回扫 metadata 那条路 —— 不编一个出来。
    """
    global _RUNTIME_IDENTITY
    _RUNTIME_IDENTITY = ident


def request_identity() -> str | None:
    """A name for the request this forward belongs to, or None.

    🔴 By identity, never by arrival order. "The forward after the prefill"
    is only the same request while nothing interleaves, and when something
    does interleave the mistake is silent: a decode step of one request gets
    recorded under another's dump, with every shape and every key name
    correct.

    ⚠️ Returns None rather than a guess. A synthesised identity would make
    this function total and its callers wrong.
    """
    if _RUNTIME_IDENTITY is not None:
        _IDENTITY_SOURCE["runner"] += 1
        return _RUNTIME_IDENTITY

    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except Exception:                                        # noqa: BLE001
        _IDENTITY_SOURCE["none"] += 1
        return None
    metadata = getattr(context, "attn_metadata", None)
    if not isinstance(metadata, dict):
        _IDENTITY_SOURCE["none"] += 1
        return None
    for entry in metadata.values():
        req = getattr(entry, "req_metadata", None)
        ids = getattr(req, "req_ids_tensor", None)
        if ids is not None and ids.numel() == 1:
            _IDENTITY_SOURCE["metadata"] += 1
            return f"req:{int(ids.flatten()[0])}"
    _IDENTITY_SOURCE["none"] += 1
    return None


def note_prefill(identity: str | None) -> None:
    """This request's prefill was recorded; its next single-row forward is
    decode step 1."""
    if identity is not None:
        _SEEN_PREFILL.add(identity)


def is_first_decode_step(identity: str | None, rows: int | None,
                         seq_len: int | None) -> bool:
    """Whether this forward is the first decode step of a recorded prefill.

    All four conditions are required and each excludes a real case:

    · a known identity        -- otherwise this cannot be attributed at all
    · whose prefill we saw    -- a request whose prefill was not recorded has
                                 nothing on the other side to compare against
    · exactly one query row   -- two rows is a chunked prefill or a batched
                                 decode, neither of which is this step
    · not already recorded    -- step 2 also has one row, and recording it
                                 under the same name would overwrite step 1
                                 with something that looks just like it

    🔴 2026-09-15:从"只录第 1 步"放宽到"录前 N 步"(DSV41_DECODE_STEPS,
    默认 1 = 原行为)。理由不是想多要数据:唯一一道【两臂各自都稳定】、
    因而能当判据的题(The capital of Japan is),在 decode 第 1 步上两臂
    【逐位相同】,分歧发生在第 2 步及以后 —— 只录第 1 步的话,这道唯一可信的
    题产出的永远是一张全绿的表。
    每一步各自落一个 dump 目录(_flush_and_rotate 每次 forward 都轮转),
    所以"第 2 步覆盖第 1 步"这个原来的理由在按步计数下不再成立。
    """
    global _ARM_CALLS
    _ARM_CALLS += 1
    if identity is None or rows != 1 or seq_len is None:
        return False
    if identity not in _SEEN_PREFILL:
        return False
    if _RECORDED_DECODE.get(identity, 0) >= _max_decode_steps():
        return False
    return True


def _max_decode_steps() -> int:
    global _MAX_STEPS
    if _MAX_STEPS is None:
        import os
        try:
            _MAX_STEPS = max(1, int(os.getenv("DSV41_DECODE_STEPS", "1")))
        except ValueError:
            _MAX_STEPS = 1
    return _MAX_STEPS


def decode_step_index(identity: str | None) -> int:
    """本请求已经录了几步 —— 调用方用它给 dump 命名/记账。"""
    return _RECORDED_DECODE.get(identity, 0)


def note_decode_recorded(identity: str | None) -> None:
    global _ARMED_DECODE
    if identity is not None:
        _RECORDED_DECODE[identity] = _RECORDED_DECODE.get(identity, 0) + 1
    _ARMED_DECODE += 1


def report_decode_step1_coverage() -> None:
    """Print what this accounting bit holds, every flush, **including zero**.

    🔴 Two lessons, both paid for on the routing tap, and they are different
    failures:

    · that counter existed for three flights with **no caller anywhere**, so
      "it never ran" and "nobody printed it" produced the same output
    · when a caller was added, the line said "this dump" while the counter
      was process-cumulative, so every reading was a running total wearing a
      per-dump label

    So: a reader that is actually called, and two numbers whose labels each
    match their own scope.
    """
    global _ARMED_DECODE_AT_LAST_FLUSH
    from vllm.logger import logger

    this_dump = _ARMED_DECODE - _ARMED_DECODE_AT_LAST_FLUSH
    _ARMED_DECODE_AT_LAST_FLUSH = _ARMED_DECODE
    # 🔴 这条自检原本在【身份恒为 None】这条路上是瞎的(2026-09-15 实测):
    #    判据是"prefills seen 比 decode dumps 多出 1 以上",而 0 和 0 正好通过。
    #    整套录制从未激活,产物上只有 prefill,读起来和"decode 没问题"一样。
    #    ⇒ 分母必须和失败数并排打印,且【零覆盖要失败】,不能只是"correct"。
    blind = _ARM_CALLS > 0 and not _SEEN_PREFILL
    logger.info(
        "DSV4.1 decode step-1 recording: armed %d time(s) this dump "
        "(%d since this process started; %d prefills seen, %d arming checks; "
        "身份来源 runner=%d metadata=%d none=%d)%s",
        this_dump, _ARMED_DECODE, len(_SEEN_PREFILL), _ARM_CALLS,
        _IDENTITY_SOURCE["runner"], _IDENTITY_SOURCE["metadata"],
        _IDENTITY_SOURCE["none"],
        " -- 🔴 ZERO COVERAGE: arming was checked %d time(s) and NOT ONE "
        "prefill was ever attributed to a request. That is not 'no decode "
        "yet' -- it means request_identity() never resolved, so decode "
        "step-1 recording is switched off entirely and the dump holds only "
        "prefills. Check who feeds set_runtime_identity()." % _ARM_CALLS
        if blind else
        "" if this_dump else
        " -- ZERO this dump. That is correct for a dump that IS the prefill: "
        "the decode step comes after it. It is a defect only if a whole "
        "request produced no decode dump, which shows up as prefills seen "
        "exceeding decode dumps by more than one.")


def reset_for_test() -> None:
    global _ARM_CALLS, _ARMED_DECODE, _ARMED_DECODE_AT_LAST_FLUSH
    global _NO_IDENTITY_REPORTED
    global _RUNTIME_IDENTITY
    _SEEN_PREFILL.clear()
    global _MAX_STEPS
    _RECORDED_DECODE.clear()
    _MAX_STEPS = None
    _RUNTIME_IDENTITY = None
    for _k in _IDENTITY_SOURCE:
        _IDENTITY_SOURCE[_k] = 0
    _ARM_CALLS = 0
    _ARMED_DECODE = 0
    _ARMED_DECODE_AT_LAST_FLUSH = 0
    _NO_IDENTITY_REPORTED = False
