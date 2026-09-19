# SPDX-License-Identifier: Apache-2.0
"""Engram prepared outside the captured graph (F10, option b).

Enabled by ``VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH=1`` (default off; off leaves
every engram path exactly as before).

Before every model forward -- graph replay, graph capture or eager -- the model
runner calls ``prepare`` (through the model's
``prepare_engram_out_of_graph``). Outside any graph it

1. runs upstream's n-gram hash on the device, slot-cache history write
   included, with the arguments upstream's model prologue would pass
   (``vllm/models/deepseek_v4_1/nvidia/model.py``, the ``engram_hash`` call);
2. copies the ids into a persistent device buffer (``hash_buf``) and D2H's the
   real tokens' ids into a pinned host buffer -- synchronously, which is where
   the host waits;
3. has each engram layer gather its rows on the host and H2D them into its
   persistent staging buffer, zeroing the padded rows
   (``AscendStagedEngramEmbedding.stage_out_of_graph``).

Inside the forward, ``OutOfGraphEngramHash`` stands in for upstream's
``NgramHashState``: it returns a view of ``hash_buf`` (the layers only take
its shape) and computes nothing, and each layer's ``lookup`` only dequantises
the staging buffer. Every tensor the forward reads for engram is a view of a
buffer allocated at construction, so the FULL-graph invariant holds and there
is no host-function node.

The ids are always the device's. The runner's CPU copy of ``input_ids`` holds
placeholders for the previous step's samples under async scheduling, so it is
refused explicitly (``_check_device_ids``).

What this does not remove (see the batch 3 report):

* PIECEWISE dummy attention metadata (``_should_build_dummy_attn_metadata``):
  upstream's prologue still gates the hash call and every layer's engram
  injection on ``attn_metadata`` being a dict, so a capture without metadata
  would record a graph with no engram at all.
* vLLM's replay-after-capture: no longer needed by engram under this flag (no
  host node), and harmless -- the replay dequantises the same staged rows --
  but the flag-off path needs it, and it lives in vLLM core.
"""

from __future__ import annotations

from itertools import islice

import torch
from torch import nn


def enabled() -> bool:
    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        out_of_graph_enabled

    return out_of_graph_enabled()


class OutOfGraphEngramHash(nn.Module):
    """Upstream's ``NgramHashState`` as the model forward sees it under the flag.

    ``inner`` is the real hash state; only ``prepare`` calls it.
    """

    def __init__(self, inner: nn.Module, max_tokens: int,
                 engram_layers: list) -> None:
        super().__init__()
        self.inner = inner
        num_layers, max_ngram = inner.multipliers.shape
        cols = (max_ngram - 1) * inner.primes.shape[-1]
        bad = [i for i, _ in engram_layers if not 0 <= i < num_layers]
        if bad or not engram_layers:
            raise RuntimeError(
                f"engram layer hash indices {[i for i, _ in engram_layers]} "
                f"against {num_layers} hashed layers.")
        #: (layer_hash_index, embed_tokens) of every local engram layer.
        self._engram_layers = list(engram_layers)
        device = inner.token_map.device
        # Allocated here, at model construction, i.e. before any capture.
        self.hash_buf = torch.zeros((max_tokens, num_layers, cols),
                                    dtype=torch.int32, device=device)
        # device="cpu" is explicit: the worker's default device is the NPU.
        self.host_ids = torch.empty((max_tokens, num_layers, cols),
                                    dtype=torch.int32, device="cpu",
                                    pin_memory=True)
        if not self.host_ids.is_pinned():
            raise RuntimeError(
                "engram out-of-graph id buffer is not pinned; the id D2H "
                "needs pinned memory.")
        #: Tokens staged by ``stage`` for the forward about to run.
        self._staged_tokens: int | None = None

    # -- what upstream's model prologue reads --------------------------------

    @property
    def lookback_depth(self) -> int:
        return self.inner.lookback_depth

    @property
    def use_slot_cache(self) -> bool:
        return self.inner.use_slot_cache

    def ensure_cache(self) -> bool:
        return self.inner.ensure_cache()

    def forward(self, input_ids: torch.Tensor, *unused_args) -> torch.Tensor:
        """Return this step's ids, computed before the forward.

        Allocates nothing and launches nothing: the result is a view of the
        persistent ``hash_buf``. The check below runs in eager and at capture
        (graph replay runs no Python).
        """
        num_tokens = input_ids.shape[0]
        staged, self._staged_tokens = self._staged_tokens, None
        if staged != num_tokens:
            raise RuntimeError(
                f"engram forward for {num_tokens} tokens, but {staged} were "
                "staged before it. With VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH=1 "
                "the model runner must call prepare_engram_out_of_graph() "
                "before every forward.")
        return self.hash_buf[:num_tokens]

    # -- the out-of-graph half -----------------------------------------------

    def stage(self, hashes: torch.Tensor, num_actual_tokens: int) -> None:
        """Publish this step's device-computed ids and stage every layer."""
        num_tokens = hashes.shape[0]
        if (num_tokens > self.hash_buf.shape[0]
                or tuple(hashes.shape[1:]) != tuple(self.hash_buf.shape[1:])):
            raise ValueError(
                f"engram hashes {tuple(hashes.shape)} do not fit the "
                f"persistent buffer {tuple(self.hash_buf.shape)}.")
        if hashes.device != self.hash_buf.device:
            raise ValueError(
                f"engram hashes are on {hashes.device}, not {self.hash_buf.device}: "
                "they must be the device hash's output.")
        num_actual_tokens = int(num_actual_tokens)
        if not 0 <= num_actual_tokens <= num_tokens:
            raise ValueError(
                f"{num_actual_tokens} real tokens in a batch of {num_tokens}.")
        self.hash_buf[:num_tokens].copy_(hashes)
        host = self.host_ids[:num_actual_tokens]
        # Synchronous D2H of the device-computed ids: the host waits here for
        # everything queued before, the hash included.
        host.copy_(self.hash_buf[:num_actual_tokens])
        for layer_index, embed in self._engram_layers:
            embed.stage_out_of_graph(host[:, layer_index], num_actual_tokens,
                                     num_tokens)
        self._staged_tokens = num_tokens


def _engram_layers(inner_model) -> list:
    out = []
    for layer in islice(inner_model.layers, inner_model.start_layer,
                        inner_model.end_layer):
        engram = getattr(layer, "engram", None)
        if engram is not None:
            out.append((engram.layer_hash_index, engram.embed_tokens))
    return out


def _inner(model):
    """``AscendDeepseekV41ForCausalLM`` -> its ``DeepseekV4Model``."""
    return model.language_model.model


def install(model) -> OutOfGraphEngramHash | None:
    """Swap the model's hash state for the out-of-graph stand-in.

    Called at model construction, so every buffer exists before capture.
    """
    inner = _inner(model)
    hash_state = getattr(inner, "engram_hash", None)
    if hash_state is None:
        return None
    if isinstance(hash_state, OutOfGraphEngramHash):
        return hash_state
    layers = _engram_layers(inner)
    missing = [i for i, embed in layers
               if not hasattr(embed, "stage_out_of_graph")]
    if missing:
        raise RuntimeError(
            f"engram layers {missing} are not the Ascend staged embedding; "
            "the out-of-graph route needs stage_out_of_graph().")
    sizes = {int(embed._max_tokens) for _, embed in layers}
    if len(sizes) != 1:
        raise RuntimeError(f"engram staging buffers disagree on size: {sizes}")
    proxy = OutOfGraphEngramHash(hash_state, sizes.pop(), layers)
    inner.engram_hash = proxy
    return proxy


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (a.device == b.device
            and a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr())


def _check_device_ids(input_ids: torch.Tensor, host_input_ids,
                      hash_buf: torch.Tensor) -> None:
    """The hash must read the device's token ids, never the runner's CPU copy.

    Under async scheduling the CPU copy holds placeholders where the previous
    step's samples go; only the device copy has the real ids.
    """
    if host_input_ids is not None and _same_storage(input_ids, host_input_ids):
        raise AssertionError(
            "engram out-of-graph hash was given the runner's CPU input_ids. "
            "Under async scheduling they hold placeholder tokens; the hash "
            "must run on the device input_ids.")
    if input_ids.device != hash_buf.device:
        raise AssertionError(
            f"engram out-of-graph hash was given input_ids on "
            f"{input_ids.device}; the hash runs on {hash_buf.device}, on the "
            "device copy of the ids.")


def prepare(model, input_ids, positions, host_input_ids=None) -> None:
    """Hash, D2H, host gather and H2D for the coming forward.

    Mirrors the gate and the arguments of upstream's model prologue; where the
    prologue would skip engram this skips too, and the stand-in then refuses
    a forward that does reach it.
    """
    from vllm.forward_context import (get_forward_context,
                                      is_forward_context_available)
    from vllm.models.deepseek_v4_1.common.mm_preprocess import \
        image_sentinel_mask

    from vllm_ascend.models.deepseek_v4_1.engram_embedding import \
        _stream_is_capturing

    inner = _inner(model)
    proxy = getattr(inner, "engram_hash", None)
    if proxy is None:
        return
    if not isinstance(proxy, OutOfGraphEngramHash):
        raise RuntimeError(
            "VLLM_ASCEND_DSV41_ENGRAM_OUT_OF_GRAPH=1 but the model was built "
            "without it: the hash state was not replaced at construction.")
    proxy._staged_tokens = None
    if _stream_is_capturing():
        raise RuntimeError(
            "engram prepare ran inside a graph capture; it synchronises, so "
            "it must run before the captured forward.")
    if input_ids is None or not is_forward_context_available():
        return
    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, list):
        raise NotImplementedError(
            "engram out-of-graph preparation does not support micro-batched "
            "(DBO) attention metadata.")
    if not isinstance(attn_metadata, dict) or not proxy.ensure_cache():
        return
    if not proxy.use_slot_cache:
        raise NotImplementedError(
            "engram out-of-graph preparation passes no lookback ids and "
            "relies on the V1 slot cache.")
    _check_device_ids(input_ids, host_input_ids, proxy.hash_buf)

    swa_metadata = attn_metadata[inner.engram_swa_prefix]
    image_mask = image_sentinel_mask(input_ids)
    # One row per query_start_loc row, padding requests included: the hash
    # kernel reads row S-1 for padded lanes. Mirrors upstream after N8
    # (vllm 8b24cc594e); test_prepare_passes_the_prologue_arguments pins it.
    num_reqs = swa_metadata.query_start_loc.numel() - 1
    lookback_token_ids = input_ids.new_full((num_reqs, proxy.lookback_depth),
                                            -1)
    hashes = proxy.inner(
        input_ids,
        positions,
        swa_metadata.query_start_loc,
        image_mask,
        lookback_token_ids,
        image_sentinel_mask(lookback_token_ids),
        swa_metadata.slot_mapping,
        swa_metadata.block_table,
    )
    proxy.stage(hashes, swa_metadata.num_actual_tokens)
