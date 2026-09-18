# SPDX-License-Identifier: Apache-2.0
"""Make upstream's V4.1 decoder layer build the Ascend attention class.

``DeepseekV4DecoderLayer.__init__`` picks its attention class by calling the
module-level ``_select_dsv4_attn_cls`` and instantiating the result inline::

    self.attn = _select_dsv4_attn_cls(vllm_config)(...)

That selector only ever returns CUDA classes, and because it is a module-level
function rather than a method, subclassing the decoder layer cannot replace it.
So the name is rebound in the module's globals, which is where the call looks
it up.

Rebinding rather than forking upstream's ``model.py``: if upstream drifts, the
attribute disappears or its signature changes and
``test_upstream_selector_contract`` goes red, whereas a forked model file
would drift silently.
"""

from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig

#: Name rebound in upstream's module. Pinned by the contract test.
SELECTOR_NAME = "_select_dsv4_attn_cls"


def _upstream_module() -> Any:
    import vllm.models.deepseek_v4_1.nvidia.model as upstream

    return upstream


def select_ascend_attn_cls(vllm_config: VllmConfig) -> type:
    """Ascend replacement for upstream's CUDA-only selector.

    Upstream branches on ``AttentionBackendEnum`` and CUDA device capability to
    pick between FlashMLA and two FlashInfer classes. None of those apply here;
    the per-layer choice that does matter on Ascend -- C4 / C128 / SWA by
    compression ratio -- is made by ``vllm_ascend.models.layer.attention``,
    below this point.
    """
    from vllm_ascend.models.deepseek_v4_1.attention import \
        AscendDeepseekV41Attention

    return AscendDeepseekV41Attention


def apply_attn_selector_patch() -> None:
    """Rebind the selector. Idempotent."""
    upstream = _upstream_module()
    # Checked before reading it: a bare getattr would raise Python's own terse
    # AttributeError and this explanation would never be reached.
    if not hasattr(upstream, SELECTOR_NAME):
        raise AttributeError(
            f"upstream no longer defines {SELECTOR_NAME!r}; the decoder layer "
            "must be picking its attention class some other way now, and this "
            "patch is silently doing nothing. Re-read "
            "vllm/models/deepseek_v4_1/nvidia/model.py before adapting it.")
    if getattr(upstream, SELECTOR_NAME) is select_ascend_attn_cls:
        return
    setattr(upstream, SELECTOR_NAME, select_ascend_attn_cls)
