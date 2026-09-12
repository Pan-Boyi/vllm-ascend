# SPDX-License-Identifier: Apache-2.0
"""NPU-side tensor dumps for the CPU-golden comparison.

Off unless ``DSV41_GOLDEN_DUMP_DIR`` is set, so importing this costs nothing.

Two things this module refuses to do, both learned the hard way on this port:

**It will not write a tensor without an identity header.** A dump of 56 argument
slots was produced on 2026-09-12 and two inferences were drawn from it; both had
to be withdrawn, because the artifact never recorded *which layer* it came from.
The bytes were fine. What was missing was the one line that said what they were
a dump of. So the header is written first, it is structured rather than free
text, and a tensor whose header cannot be built is not written at all.

**It will not silently truncate.** The host carries a 188 GiB resident mapping
and HBM sits at 89%; an unbounded dump turns a diagnostic run into an OOM, and
an OOM reads like "the change broke startup". Both caps refuse and say so.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_DIR = os.environ.get("DSV41_GOLDEN_DUMP_DIR", "")
_MAX_FILE_MB = float(os.environ.get("DSV41_GOLDEN_DUMP_MAX_FILE_MB", "512"))
_MAX_TOTAL_MB = float(os.environ.get("DSV41_GOLDEN_DUMP_MAX_TOTAL_MB", "4096"))
_RUN = os.environ.get("DSV41_GOLDEN_DUMP_RUN", "")

_written_mb = 0.0
_seq = 0
_manifest_done = False
_refusals: list[str] = []


def enabled() -> bool:
    return bool(_DIR)


def _logger():
    from vllm.logger import logger
    return logger


def _run_manifest() -> None:
    """Record what this whole run was, once, beside the tensors."""
    global _manifest_done
    if _manifest_done:
        return
    _manifest_done = True
    import subprocess
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def _git(*a):
        try:
            return subprocess.run(("git", "-C", here) + a, capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    man = {
        "这是什么的现场": "NPU 侧张量 dump，用于与 CPU golden 双向对照",
        "轮次": _RUN or "未命名(请设 DSV41_GOLDEN_DUMP_RUN=#N)",
        "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "工作区是否干净": _git("status", "--porcelain") == "",
        # 配置分支 = 这一发实际生效的开关，不是"应该生效的"
        "开关": {k: v for k, v in sorted(os.environ.items())
                 if k.startswith(("VLLM_ASCEND_DSV41", "DSV41_"))},
        "上限_单份MB": _MAX_FILE_MB,
        "上限_总量MB": _MAX_TOTAL_MB,
    }
    os.makedirs(_DIR, exist_ok=True)
    with open(os.path.join(_DIR, "RUN_MANIFEST.json"), "w") as f:
        json.dump(man, f, ensure_ascii=False, indent=1)
    _logger().info("【GOLDEN-DUMP】现场清单 -> %s/RUN_MANIFEST.json (commit=%s 干净=%s)",
                   _DIR, man["commit"][:9], man["工作区是否干净"])


def dump(name: str, tensor: Any, *, layer: str, call_idx: int,
         group_key: str | None = None, num_tokens: int | None = None,
         num_requests: int | None = None, note: str = "") -> str | None:
    """Write one tensor plus its identity header. Returns the path, or None.

    `name` is the tensor's name in the golden's vocabulary -- it is what makes
    two dumps comparable, so it is required and never derived from a variable
    name at the call site.
    """
    global _written_mb, _seq
    if not _DIR:
        return None
    import numpy as np
    import torch
    if not isinstance(tensor, torch.Tensor):
        _refuse(f"{layer}/{name}: 不是张量({type(tensor).__name__})")
        return None
    _run_manifest()
    _seq += 1
    seq = _seq
    try:
        host = tensor.detach().to("cpu").contiguous()
    except Exception as e:
        _refuse(f"{layer}/{name}: 取不回 host ({e!r})")
        return None
    mb = host.numel() * host.element_size() / 1024 / 1024
    if mb > _MAX_FILE_MB:
        _refuse(f"{layer}/{name}: 单份 {mb:.1f} MB > 上限 {_MAX_FILE_MB} MB "
                f"(shape={tuple(host.shape)} dtype={host.dtype}) —— 拒绝，不截断")
        return None
    if _written_mb + mb > _MAX_TOTAL_MB:
        _refuse(f"{layer}/{name}: 总量将达 {_written_mb + mb:.1f} MB > 上限 "
                f"{_MAX_TOTAL_MB} MB —— 拒绝，不截断")
        return None

    header = {
        "这是什么的现场": "NPU 侧单个张量的 dump",
        "轮次": _RUN or "未命名",
        "时间": time.strftime("%Y-%m-%d %H:%M:%S"),
        "张量名": name,
        "层名": layer,
        "第几次调用": call_idx,
        "落盘序号": seq,
        "cache_group_key": group_key,
        "R(请求数)": num_requests,
        "num_tokens": num_tokens,
        "dtype": str(host.dtype),
        "shape": list(host.shape),
        "字节数": int(host.numel() * host.element_size()),
        "备注": note,
    }
    # bf16/fp16 numpy 没有原生 dtype：存 uint16 原始字节 + 在头里写明真 dtype，
    # 不静默转 fp32 —— 转过的数和原数在对照里是两个东西。
    raw_dtype = None
    if host.dtype in (torch.bfloat16, torch.float16):
        raw_dtype = str(host.dtype)
        arr = host.view(torch.uint16).numpy()
        header["落盘方式"] = f"uint16 原始字节，真 dtype={raw_dtype}（未转 fp32）"
    else:
        arr = host.numpy()
        header["落盘方式"] = "原 dtype 直存"

    base = f"{seq:04d}_{_safe(layer)}_{_safe(name)}_call{call_idx}"
    os.makedirs(_DIR, exist_ok=True)
    npy = os.path.join(_DIR, base + ".npy")
    # 头先写：没有头的张量不允许存在
    with open(os.path.join(_DIR, base + ".json"), "w") as f:
        json.dump(header, f, ensure_ascii=False, indent=1)
    np.save(npy, arr)
    _written_mb += mb
    _logger().info("【GOLDEN-DUMP #%d】%s/%s shape=%s dtype=%s %.2f MB "
                   "(累计 %.1f/%.0f MB) -> %s",
                   seq, layer, name, tuple(host.shape), host.dtype, mb,
                   _written_mb, _MAX_TOTAL_MB, base)
    return npy


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(s))[:80]


def _refuse(msg: str) -> None:
    _refusals.append(msg)
    _logger().error("【GOLDEN-DUMP 拒绝】%s", msg)
    if _DIR:
        try:
            os.makedirs(_DIR, exist_ok=True)
            with open(os.path.join(_DIR, "REFUSALS.txt"), "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass
