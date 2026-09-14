# SPDX-License-Identifier: Apache-2.0
"""Where the alignment dump is taken from, and why it costs nothing when off.

The tensors golden compares are module inputs and outputs, so they are read
with torch forward hooks rather than with calls placed in the forward path.
That is not a style preference -- it is what makes "off" provable:

* when the dump directory is unset, :func:`install_alignment_taps` registers
  **no hooks at all**, so no tap function is entered, nothing is branched on
  per layer, and the module graph is byte-identical to the untapped one;
* the alternative -- ``if recording: tap(...)`` at thirty points -- is off by
  early return, which is a claim about what the branch does rather than about
  whether it runs.

Not every protocol tensor is a module boundary. ``sa_in.*``, ``hc_*``,
``moe_gate.*`` and ``pre_mix.*`` are intermediates inside functions this line
owns, and those are tapped inline in their own files. The split is by what the
value *is*, not by convenience: a hook can only see what crosses a module
edge, and pretending otherwise would mean hooking a module that does not
compute the thing whose name it carries.

⚠️ One such trap is already visible. ``attn_norm`` and ``ffn_norm`` are real
submodules, but the norm they are named after is computed *inside*
``mhc_pre_delayed_tilelang``, which only borrows their weights -- a hook on
either would never fire, and "no rows recorded" reads exactly like "this
layer did not run".
"""

from __future__ import annotations

import re
from typing import Any

import torch

#: Per-layer dump tags. The only naming rule this module knows,
#: and it is used to *exclude* other layers -- never to decide
#: what counts as a tag.
_LAYER_TAG = re.compile(r"layer_\d+")

#: Protocol key -> which module edge, and for an input, **which parameter**.
#:
#: 🔴 "the module's first positional input" was wrong and shipped wrong.
#: ``DeepseekV4Attention.forward(self, positions, hidden_states, ...)`` takes
#: positions first, so ``attn.in`` recorded an int64 [T] position vector
#: under a key golden uses for a bf16 [T, 5120] hidden state. Both sides had
#: a tensor called ``attn.in``; they were different tensors; nothing said so,
#: because a key name is not a contract.
#:
#: Bound by parameter name and resolved once at install, so a signature that
#: moves fails loudly instead of recording the neighbour.
_BOUNDARY_TAPS: tuple[tuple[str, str, str, "str | None"], ...] = (
    ("attn", "attn.in", "in", "hidden_states"),
    # 🔵 Recorded in its own right, not kept for compatibility. positions is
    # one of the inputs that reaches layer 0, and the earliest divergence is
    # *at* layer 0 -- so "input_ids matched but positions did not" produces
    # exactly the symptom of "the embedding is wrong", and only this key
    # tells the two apart.
    ("attn", "attn.positions", "in", "positions"),
    ("attn", "attn.out", "out", None),
    ("ffn", "moe.in", "in", "hidden_states"),
    ("ffn", "moe.out", "out", None),
)

#: What each key **is**, declared here so a dump can say it rather than
#: leaving the reader to infer it from a name two teams chose separately.
#: Two keys have already turned out to mean different things on the two
#: sides; once that happens to one key, every other key is unverified too.
KEY_SEMANTICS: dict = {
    "attn.in": "the attention module's input hidden state. ⚠️ Before "
               "fc8536f05 this key held `positions` instead -- an int64 [T] "
               "vector under a name golden uses for a bf16 [T, 5120] hidden "
               "state. Dumps from before and after that commit are two "
               "different keys and must not be compared",
    "attn.positions": "the attention module's `positions` argument: the "
                      "position of each row within its own request",
    "attn.out": "the attention module's output, before the residual add",
    "moe.in": "the FFN/MoE module's input hidden state",
    "moe.out": "the FFN/MoE module's output",
    "layer.in": "the decoder layer's `residual` argument: the hc_mult-way "
                "residual streams [T, 4, 5120]. Absent on layer 0, which "
                "receives a 2-D embedding and builds the streams itself",
    "pre_mix.in": "the decoder layer's `pre_mix` argument [T, 4] fp32; "
                  "absent on layer 0, where it is None",
    "input_ids": "the token ids the model was handed this forward",
    "engram_hash_ids":
        "the engram n-gram hash ids, computed once per forward and handed to "
        "every engram layer. 🔵 The tables are gathered *by these*: a wrong "
        "id returns a different row of a 189 GiB table and the layer then "
        "computes correctly on the wrong weights",
    "positions": "the position of each row in its own request",
    "embed.out": "the embedding output, i.e. what the first decoder layer "
                 "receives",
    "qkv_a.out": "the fused q-lora/KV projection's output, [T, q_lora_rank + "
                 "head_dim], before either RMSNorm. The first point after "
                 "attn.in on this path",
    "split.qr": "the q-lora activation after q_norm, i.e. wq_b's input. The "
                "kv half of the same split is sa_in.kv. ⚠️ Absent when the "
                "fused query-quant path is taken, because qr is then a "
                "QuantizedActivation and not a tensor",
    "q.wq_b_out": "q after the wq_b projection, [T, heads, head_dim], before "
                  "the weight-free per-head RMSNorm. ⚠️ That norm is NOT "
                  "q_norm: q_norm is an RMSNorm(q_lora_rank) applied earlier "
                  "to qr, and this one takes no learned weight at all",
    "rope.q.in": "the rotation's input on the q side: the trailing "
                 "rope_head_dim of the normed q, [T, heads, rope_head_dim]. "
                 "🔵 Only the slice the rotation touches -- the leading dims "
                 "pass through unchanged and are already in sa_in.q, so "
                 "recording them again would be the same numbers at eight "
                 "times the size. Named to match golden's rope.kv.in",
    "read.compressor.*.latent":
        "🔴 an **instrument read-back**, not a model tensor -- hence the "
        "`read.` prefix. It is the latent this compressor call was handed, "
        "recorded before the call. Rows whose companion `.slots` entry is "
        "negative were never written by whoever produced it and are "
        "**uninitialised memory**. ⚠️ Do not key a detector on a particular "
        "value: measured across three flights the residue is a different bit "
        "pattern each time (per-row |max| ran 5e33, 8e37, 1.6e38, and one "
        "NaN). There is no value to recognise -- that is the whole of what is "
        "wrong with it. Use `.slots`, which says which rows to drop and is "
        "true every run",
    "read.compressor.*.latent_out":
        "the latent a compressor call **produced**, recorded after the call. "
        "🔴 The distinction is not pedantry: `latent_out` is an output "
        "parameter filled by the operator, so recording it beforehand records "
        "the allocator's leftovers -- and because the caching allocator "
        "usually hands back a recently freed block, two calls in three came "
        "back holding the *previous* call's latent and looked like real "
        "values. That is the dangerous half; obvious garbage is the safe half",
    "read.compressor.*.slots":
        "the flat destination slots this compressor call was handed, taken "
        "before the operator runs. Also the mask for the `.latent` key above: "
        "negative means that row was never written. ⚠️ The `<NN>` in these "
        "keys is the call's index **within one forward** and restarts each "
        "forward; it used to be a process-level counter, which gave the same "
        "tensor a different name in every dump",
    "moe_gate.out": "the router's decision, under golden's own key name so the "
                    "two dumps need no mapping: `.0` is the normalised "
                    "top-k weights [T, topk] and `.1` the expert ids "
                    "[T, topk].\n"
                    "🔴 **How to compare `.1`, and it is not elementwise.** "
                    "Compare the ids as a SET per token: topk's return order "
                    "is not a difference, and comparing by position "
                    "manufactures a routing mismatch out of an ordering.\n"
                    "🔴 **And never without the margin.** golden's layer_00 "
                    "routing margins are 0.05159 / 0.10506 / 0.00314 / "
                    "0.00126 / 0.01951 for the five tokens: on tokens 2 and 3 "
                    "the 6th and 7th experts are all but tied, so two correct "
                    "implementations can pick different 6th experts. An id "
                    "difference where the margin is ~1e-3 is ⚫ undecidable, "
                    "not a defect -- and the two readings lead to opposite "
                    "next steps",
    "moe.experts.out": "the routed experts' aggregate, before the shared "
                       "expert is added. golden has no counterpart; this key "
                       "exists so our own chain can contradict itself, which "
                       "is how the q chain's defect was found",
    "moe.shared.out": "the shared expert's output. Every token goes through "
                      "it and it is added separately from the routed "
                      "aggregate (ref/model.py:889-891), so it is the one "
                      "part of moe.out that does not depend on routing at all",
    "rope.kv.out": "🔵 the **post-RoPE** kv, i.e. the rows written to the SWA "
                   "cache. This is the key to compare with golden's "
                   "rope.kv.out; sa_in.kv is a stage below it and is not "
                   "comparable over [448:512]. ⚠️ On this plan there is no "
                   "quantisation after the rotation, so unlike golden there "
                   "is no separate quant.window.in/out between this and the "
                   "cache -- this value and the cache row are the same "
                   "numbers (measured, E_err/E_gold = 1e-6). ⚠️ **float32**: "
                   "rope_tail computes in fp32 and the cast to the cache's "
                   "bf16 happens in the scatter, so this key is the unrounded "
                   "value and the cache row is its bf16 rounding. 🔴 It is "
                   "the SLIDING-WINDOW stream on every layer, ratio-2 "
                   "included -- the same rows as "
                   "read.write.ori_kv.latent_pre_rope, one stage later. The "
                   "compressed stream never passes through here, and this "
                   "side never concatenates the two the way golden's "
                   "sa_in.kv does",
    "sa_in.q": "the query handed to the sparse-attention operator, after "
               "rope and quantisation, sliced to n_local_heads",
    "sa_in.kv": "🔴 the **pre-RoPE**, post-kv_norm kv -- ✅ measured, not "
                "inferred (flight-24; see read.ori_kv's block above for the "
                "numbers and the method). golden's key of the same name is "
                "post-rope AND post-quant, so over [448:512] the two are NOT "
                "COMPARABLE and a difference there is a stage mismatch; over "
                "[0:448] they are the same stage. Detail: the argument this "
                "layer was handed, not what the operator reads. Upstream's "
                "contract is 'kv is unchanged; attention reads kv solely via "
                "swa_kv_cache' (qkv_prolog.py:80-83), so the rotation is "
                "applied to a separate kv_out that goes to the cache and the "
                "argument stays pre-rope. ⚠️ Two consequences: golden's key "
                "of "
                "the same name has the compressed stream concatenated and "
                "this one does not, AND if golden records the value the "
                "operator reads then golden's is post-rope while this one is "
                "not -- a difference whose angle grows exactly with position "
                "and which inverse-rotation cancels. That signature is what "
                "a stage mismatch looks like; it is also what a missing "
                "rotation looks like, and only the write side tells them "
                "apart (compare read.ori_kv.rows against "
                "read.write.ori_kv.latent_pre_rope)",
    "sa_in.attn_sink": "the per-head attention sink (no token axis)",
    "sa_in.softmax_scale": "the softmax scale (scalar)",
    "attn.o_raw": "the operator's output before inverse rope",
    "attn.o_unroped": "the operator's output after inverse rope",
    "read.sa_in.cmp_kv_rows":
        "the compressed entries this request hands the operator, read from "
        "the block its table names. 🔵 golden's sa_in.kv is [original rows] "
        "++ [these]; on this side they are two separate arguments, so the "
        "relation is: golden sa_in.kv == ours sa_in.kv ++ ours "
        "read.sa_in.cmp_kv_rows",
    # ⚠️ What is actually stored in the KV caches, because a comparison
    # against golden that does not know this produces a number that looks
    # like a finding. Measured against upstream's own docstring
    # (fused_compress_quant_cache.py:231-238) plus this line's plan:
    #
    #   ✅ measured  RoPE is applied **to the rows in the cache**. Grade
    #                   earned on flight-24, from this line's own dump and
    #                   without reference to golden: over [448:512],
    #                   ``read.ori_kv.rows`` against
    #                   ``rope(read.write.ori_kv.latent_pre_rope)`` gives
    #                   E_err/E_gold = 0.000001, against the unrotated
    #                   tensor 0.544731, with the [0:448] scale reading
    #                   0.000000 (the rotation does not touch it). The
    #                   criterion needed no cos/sin reconstruction: a
    #                   rotation leaves the leading dims untouched and
    #                   preserves the tail's norm, and those two properties
    #                   are enough.
    #                   ⚠️ Before that it carried a ✅ whose actual basis was
    #                   the operator's *signature*. Two faults at once: the
    #                   grade said measured, so nobody re-checked; and the
    #                   subject was ``rope_quant_insert``'s cache rows, so
    #                   anyone who did re-check would have checked the wrong
    #                   object. A day passed without rope being reconsidered.
    #   🔴 SCOPE  the line above is about **cache rows**, and it does not
    #                   transfer to ``sa_in.kv``. That key is the pre-rope
    #                   argument (see its own entry), and reading this line
    #                   as covering it is how a stage mismatch between the
    #                   two dumps looked, for one day, like a rope defect.
    #   ✅ established  On the A5 bf16 plan nothing is quantised: the cache
    #                   dtype is bfloat16 and fp8_scale is None. The uint8
    #                   fp8_ds_mla layout is a different plan.
    #   ✅ established  Row layout is [448 NoPE | 64 RoPE] within the 512, in
    #                   that order -- upstream's words for the bfloat16 case.
    #   ✅ established  Addressing is PA_BBND: a row is at
    #                   block * block_size + offset, flat.
    #   ✅ established  golden's ``sa_in.kv`` is **post-rope and
    #                   post-quant**: golden measured its own chain,
    #                   ``rope.kv.out == quant.window.in`` and
    #                   ``quant.window.out == sa_in.kv[..., -64:]``. Ours is
    #                   pre-rope. ⇒ Over **[448:512] the two keys are not
    #                   comparable at all**, and a difference there is a
    #                   stage mismatch, not a finding. Over [0:448] they are
    #                   the same stage -- the rotation does not reach it --
    #                   which is what makes "the kv projection is clean,
    #                   0.099% error energy" a real result.
    "read.write.ori_kv.latent_pre_rope":
        "🔴 the **sliding-window** stream's row, pre-RoPE: the `kv` argument "
        "of _fused_qnorm_rope_kv_insert, post-kv_norm, which is what gets "
        "rotated into the SWA cache. ⚠️ 'latent' here means the MLA latent "
        "row (512 wide), **not** the compressed-stream latent -- that one is "
        "read.compressor.*.latent_out and comes from an entirely different "
        "path (kv_score -> fused_save_compress_norm). The name has misled a "
        "reader already; kept because renaming mid-investigation costs more "
        "than it saves, and corrected here instead",
    "read.ori_kv": "rows of the original-stream KV cache: post-RoPE, bf16, "
                   "[448 NoPE | 64 RoPE], addressed block*page+offset. "
                   "The page is taken from THIS tensor's own shape, never "
                   "written down: the serve log has four different block "
                   "sizes (SWA 32, compressed 128 and 256, compressor ring "
                   "8) and each is correct in its own group, so a number "
                   "recorded here would be wrong in three of them",
    "read.cmp_kv": "rows of the compressed-stream KV cache; same layout as "
                   "read.ori_kv and the same rule about the page. Its "
                   "spec block_size is the physical page MULTIPLIED BY the "
                   "compression ratio -- this port's convention, the "
                   "opposite of upstream, which keeps block_size at the "
                   "config value and lets the storage page shrink by the "
                   "ratio. So a block_size // ratio read anywhere in this "
                   "tree cannot be judged without first knowing which "
                   "convention its dividend came from: same field name, "
                   "two meanings, and the two versions of the code look "
                   "identical",
    "read.write.ori_kv.latent_pre_rope":
        "⚠️ the KV latent handed to the original-stream cache insert, "
        "**before** rope. The cached row is this after rope, so comparing "
        "the two directly compares across a rotation. Rope is orthogonal, so "
        "a content comparison at equal positions survives it; a positional "
        "one does not",
    "read.write.ori_kv.slots":
        "the slot mapping that insert was given, in the form it was given "
        "(the caller's, before any flattening this line does elsewhere)",
    "read.sa_in.cmp_kv_rows.count":
        "how many rows the line above took, since 'the first "
        "seqused_cmp_kv of the first block' is an assumption about layout",
}


#: Protocol key -> the decoder layer's *argument* carrying it. These are the
#: keys the injector replaces, so they have to be the tensors the layer was
#: actually handed, not anything reconstructed from them.
#:
#: 🔴 Bound by parameter **name**, resolved once at install time. The call site
#: passes all nine positionally (``model.py:626-636``), so ``args[6]`` would
#: work today and would silently record the wrong tensor the day an argument
#: is inserted ahead of it. Resolving the name and failing loudly if it is
#: gone is the difference between a wrong dump and no dump.
_ARG_TAPS: tuple[tuple[str, str], ...] = (
    ("residual", "layer.in"),
    ("pre_mix", "pre_mix.in"),
)

#: Arguments that belong to the forward rather than to a layer: the model
#: computes them once and hands the same tensor to every layer. Recorded
#: under the ``prologue`` tag, once, from whichever layer sees them first.
#:
#: 🔵 ``engram_hashes`` is here because of what it decides. The engram tables
#: are gathered **by these ids**: a wrong id returns a different row of a
#: 189 GiB table, and the layer then computes correctly on the wrong weights.
#: The symptom of that is structural, present from the first layer, identical
#: on every request and independent of cross-request state -- which is the
#: complete description of the defect currently being hunted. It has never
#: been recorded.
_FORWARD_ARG_TAPS: tuple[tuple[str, str], ...] = (
    ("engram_hashes", "engram_hash_ids"),
)


def _arg_positions(layer: torch.nn.Module) -> dict[str, int]:
    """Where each tapped argument sits in this layer's positional signature."""
    import inspect

    parameters = list(
        inspect.signature(type(layer).forward).parameters)
    positions = {}
    for name, key in _ARG_TAPS:
        if name not in parameters:
            raise RuntimeError(
                f"{type(layer).__name__}.forward has no {name!r} parameter, so "
                f"{key!r} cannot be recorded from the tensor the layer is "
                "handed. Recording it from anywhere else would be recording a "
                "reconstruction: the injector would then be replacing "
                "something the layer never received. Its parameters are "
                f"{parameters}.")
        # -1 for self, which a hook's args tuple does not carry.
        positions[key] = parameters.index(name) - 1
    return positions


def _named_argument(module: torch.nn.Module, parameter: str, args: tuple,
                    kwargs: "dict | None"):
    """One argument of this call, found by name in args or kwargs.

    ⚠️ Returns ``None`` when it is not there, which is a real answer: a
    pipeline rank or a code path that does not pass it has none. What it
    must never do is return the *neighbouring* argument, which is what a
    fixed index does the day the signature or the call style changes.
    """
    if kwargs and parameter in kwargs:
        return kwargs[parameter]
    try:
        index = _parameter_index(module, parameter, parameter)
    except Exception:                                        # noqa: BLE001
        return None
    return args[index] if 0 <= index < len(args) else None


def _parameter_index(module: torch.nn.Module, parameter: str,
                     key: str) -> int:
    """Where ``parameter`` sits in this module's positional signature."""
    import inspect

    names = list(inspect.signature(type(module).forward).parameters)
    if parameter not in names:
        raise RuntimeError(
            f"{type(module).__name__}.forward has no {parameter!r} parameter, "
            f"so {key!r} cannot be recorded from the tensor it names. Its "
            f"parameters are {names}. Recording a neighbour instead is how "
            "this key came to hold an int64 position vector on one side and "
            "a bf16 hidden state on the other.")
    return names.index(parameter) - 1          # -1 for self


def _first_tensor(value: Any) -> torch.Tensor | None:
    """The tensor a hook actually got, out of whatever shape it arrived in."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _armed_recorder():
    """The recorder, or ``None`` when this forward must not be recorded.

    🔴 The single door. Every path that writes a tensor goes through this one
    call, because the previous arrangement -- each writer fetching the
    recorder and checking ``_ARMED`` itself -- had exactly the failure this
    project keeps producing: a new writer that does not know about the check.

    Measured twice. First ``_record_prologue`` was a second writer of the same
    shape and was caught by a text match. Then ``_record_with`` -- the inline
    ``tap`` API -- had a *different* shape, so the same text match did not
    reach it, and a decode step's tap called ``record()`` against a recorder
    that had no sequence length yet:

        RuntimeError: set_seq_len() must be called before record(),
                      normally from the prologue hook

    Enumerating writers by pattern finds the writers that look like the ones
    you already know about. One door, plus a test that walks this module's
    AST for ``recorder.record*`` calls and fails on any that did not come
    through it, finds the ones that do not.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    if not _ARMED:
        return None
    return get_recorder()


def _record(tag_layer: int, key: str, tensor: torch.Tensor) -> None:
    """One tensor into the recorder, addressed by global sequence position.

    The row mapping is rebuilt here rather than cached because it belongs to
    the forward, not to the tap: a cached one would survive into the next
    step and scatter this step's rows onto last step's positions.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import current_row_to_pos

    recorder = _armed_recorder()
    if recorder is None:
        return
    positions = _current_positions()
    if positions is None:
        return
    row_to_pos = current_row_to_pos(positions)
    if row_to_pos is None:
        # Profile run, dummy run or graph capture. Reported by the resolver,
        # not swallowed here.
        return
    if not bool((row_to_pos >= 0).any()):
        # 🔴 Every row is padding. Recording this creates a buffer whose
        # coverage is zero everywhere, and the flush then reports the whole
        # key as missing -- which is what stopped a dump flight: the run had
        # armed on a real request and then met a forward whose rows all
        # mapped to nothing.
        #
        # Skipping is the honest outcome: the key is simply not in this dump,
        # and golden_coverage lists it as missing. Creating an empty buffer
        # instead would make "this forward had nothing to contribute" and
        # "the mapping is broken" the same artifact.
        _report_all_padding(tag_layer, key, row_to_pos.numel())
        return
    rows = tensor.reshape(row_to_pos.numel(), -1) if tensor.dim() > 2 else tensor
    recorder.record_layer(tag_layer, key, rows.reshape(tensor.shape), row_to_pos)


#: The positions argument of the decoder layer currently executing. Captured
#: by a pre-hook on the layer itself rather than looked up elsewhere.
_CURRENT_POSITIONS: torch.Tensor | None = None


#: Protocol key -> the parameter it comes from, for the keyword lookup.
_ARG_NAME_BY_KEY: dict = {key: name for name, key in _ARG_TAPS}


#: Which forward-level arguments have been recorded this forward. Cleared at
#: the flush, so each forward records its own rather than the first one's.
_FORWARD_RECORDED: set = set()


#: Whether this forward's recording is armed. False until the sequence length
#: is known, because every ``record()`` scatters into a [seq_len, ...] buffer
#: and raises without one.
_ARMED = False


def _arm_recording(_module: torch.nn.Module, _args: tuple) -> None:
    """Fix the sequence length for this forward, or disarm it and say so.

    🔴 This used to live in the prologue hook and to ``return`` when the length
    could not be determined. The result was recording left half-armed:
    ``set_seq_len`` never ran, every later tap called ``record()``, and the run
    died forty taps away with

        RuntimeError: set_seq_len() must be called before record(),
                      normally from the prologue hook

    -- a message that names the hook which did *not* fail. It cost three
    launches once, and another the first time a request arrived through
    ``/v1/completions``, where the prologue's own precondition happened not to
    hold. The fix is not a better message: "cannot size the buffer" has to
    disable recording **here**, where it is discovered, instead of arming a
    trap for the next forty taps.

    Installed on the inner model, which always runs -- ``embed_tokens`` does
    not exist on a non-first pipeline rank -- and which is strictly earlier
    than every tap, the ordering the recorder needs.
    """
    global _ARMED

    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    recorder = get_recorder()
    if recorder is None:
        _ARMED = False
        return
    try:
        seq_len = _single_request_seq_len()
        rows = _single_request_query_rows()
    except ValueError as exc:
        _ARMED = False
        _report_disarmed(str(exc))
        return
    if seq_len is not None and rows is not None and rows != seq_len:
        # 🔴 This forward cannot fill the buffer, so it must not open one.
        #
        # Recording scatters rows into a [seq_len, ...] buffer and the flush
        # requires every position written exactly once. A decode step has one
        # row against a seq_len of prompt+N; a chunked prefill or a
        # prefix-cache hit has some of them. All three are *normal* and all
        # three produce an "incomplete coverage" dump, which is the artifact
        # a broken row mapping produces -- so "nothing to contribute here"
        # and "the mapping is wrong" were the same output again.
        #
        # Measured: a warm-up request recorded its prefill correctly and then
        # its first decode armed at seq_len=202 (= 201 + 1, which only a
        # decode can be), recorded one row, and the flush raised. The
        # assertion was firing on an object its premise does not cover.
        _ARMED = False
        _report_disarmed(
            f"this forward supplies {rows} query rows against a sequence "
            f"length of {seq_len}, so it cannot write every position exactly "
            "once. That is the normal shape of a decode step, a chunked "
            "prefill or a prefix-cache hit -- not a defect. Only a forward "
            "that covers the whole sequence is recorded.")
        return
    if seq_len is None:
        # ⚠️ The precondition is "the recorder can size its buffer", not "the
        # forward context told us the length". A recorder constructed with a
        # seq_len already can, and disarming it would refuse to record a run
        # that is perfectly recordable -- the opposite failure to the one
        # this function exists for, and just as silent.
        if getattr(recorder, "seq_len", None) is not None:
            _ARMED = True
            return
        _ARMED = False
        _report_disarmed(
            "no single request's sequence length was available from the "
            "forward context and the recorder has none of its own. "
            + _why_no_seq_len())
        return
    recorder.set_seq_len(seq_len)
    _ARMED = True
    _report_armed(seq_len)


def _flush_and_rotate(_module: torch.nn.Module, _args: tuple,
                      _output) -> None:
    """Write this prefill's dump and open the next one.

    🔴 Flushing here rather than at process exit is what makes two dumps in
    one server possible, and two dumps in one server is the comparison that
    needs no reference: same weights, same prompt, one request before a
    different one has run and one after. A correct forward makes those
    byte-identical, and they are already known to produce different text.

    Only when this forward actually recorded something. A decode step
    disarms, and flushing there would write an empty dump over a real one --
    the same failure as the trace's sentinel, where a well-formed artifact
    says nothing and reads as if it did.
    """
    global _ARMED

    if not _ARMED:
        return
    _ARMED = False
    # Per forward, so the next one records its own.
    _FORWARD_RECORDED.clear()
    # 🔴 And the compressor's call index, for the same reason and with a
    # sharper symptom: it is part of a **key name**, so letting it run
    # process-wide made req_000 end at index 3 and req_001 start at 48 -- the
    # same tensor under a different name in two dumps, which a key-by-key
    # comparison reads as two unrelated keys rather than as a pair.
    from vllm_ascend.models.deepseek_v4_1.compressor_insert import \
        reset_write_call_index
    reset_write_call_index()
    from vllm_ascend.models.deepseek_v4_1.debug_align import rotate_recorder

    rotate_recorder()


def _report_all_padding(layer: int, key: str, rows: int) -> None:
    from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

    announce_once(
        "align_all_padding",
        "a forward reached the taps with all %d of its rows mapping to no "
        "position (first seen at layer %d, key %r). Nothing was recorded for "
        "it. If a whole key is missing from this dump, this is why -- not a "
        "broken row mapping on the rows that do exist.", rows, layer, key)


def _record_model_inputs(module: torch.nn.Module, args: tuple,
                         kwargs: "dict | None") -> None:
    """input_ids and positions, from the model's own arguments.

    🔴 They used to be taken from hooks on ``embed_tokens``, and flight-22
    came back with a prologue section holding only ``engram_hash_ids``: the
    section existed, three of its four keys did not, and nothing said so.

    ``embed_tokens`` is not on the path when the caller supplies
    ``inputs_embeds`` -- ``model.py:541-545`` takes it in preference and never
    calls the embedding -- which is exactly what a multimodal wrapper does.
    The hooks were installed and simply never fired, so neither the install
    report nor the flush report had anything to say.

    ⚠️ So these are read where they are **arguments**, not where some module
    happens to consume them. An argument is present whatever the path taken
    with it; a module's being called is a property of the path.
    """
    for name in ("input_ids", "positions"):
        value = _named_argument(module, name, args, kwargs)
        if isinstance(value, torch.Tensor):
            _record_prologue(name, value)


def _report_armed(seq_len: int) -> None:
    """Say, once, that recording did arm.

    🔴 Needed as much as the disarm line, and for a reason that bit here
    already: the first forward of a server is a profile run with no request,
    so it disarms, and "disarmed" would be the only thing in the log whether
    or not a real forward ever armed. One line each makes "never armed" and
    "armed on the real forward" different readings instead of the same one.
    """
    from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

    announce_once(
        "align_armed",
        "alignment recording armed at seq_len=%d. If a 'recording is "
        "disabled' line appeared before this one it was a profile or dummy "
        "forward, not this run's request.", seq_len)


def _why_no_seq_len() -> str:
    """Which step of the lookup came up empty.

    🔴 "No sequence length" has four different causes needing four different
    next actions: no forward context at all (profile run), metadata that is
    not the per-layer dict this reads, a dict whose entries carry no
    req_metadata, or entries whose seq_lens is empty. Reporting only the
    conclusion makes a flight that disarms cost a second flight to find out
    why -- and this path has already been debugged twice from the outside.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except Exception:                                        # noqa: BLE001
        return "There is no forward context here at all (profile run)."
    metadata = getattr(context, "attn_metadata", None)
    if metadata is None:
        return "The forward context carries attn_metadata=None."
    if not isinstance(metadata, dict):
        return (f"attn_metadata is {type(metadata).__name__}, not the "
                "per-layer dict this reads; on this path the length lives "
                "somewhere else.")
    if not metadata:
        return "attn_metadata is an empty dict."
    with_req = [k for k, v in metadata.items()
                if getattr(v, "req_metadata", None) is not None]
    if not with_req:
        return (f"none of the {len(metadata)} attn_metadata entries carries "
                f"req_metadata (keys: {sorted(metadata)[:4]}).")
    shapes = []
    for key in with_req[:4]:
        lens = getattr(getattr(metadata[key], "req_metadata", None),
                       "seq_lens", None)
        shapes.append((key, None if lens is None else tuple(lens.shape)))
    return ("attn_metadata entries carry req_metadata but no usable "
            f"seq_lens: {shapes}.")


def _report_disarmed(reason: str) -> None:
    from vllm_ascend.models.deepseek_v4_1.switch_log import announce_once

    announce_once(
        "align_disarmed",
        "alignment recording is disabled for this forward -- %s. Nothing will "
        "be written, so the dump will be short rather than wrong. Said here, "
        "where it is discovered: the previous behaviour left recording "
        "half-armed and failed forty taps later with a message naming the "
        "hook that did not fail.", reason)


def _capture_positions(_module: torch.nn.Module, args: tuple,
                       kwargs: dict | None = None) -> None:
    """Remember this layer's ``positions`` for the boundary hooks below.

    The attention metadata does **not** carry positions -- checked, not
    assumed -- so a tap that looked for them there would find None on every
    forward and record nothing, which reads exactly like "this model never
    ran". The decoder layer's own signature has them as the second positional
    argument, which is the only place they exist without deriving them a
    second time from query_start_loc and seq_lens.
    """
    global _CURRENT_POSITIONS
    # Assigned unconditionally, including to None: leaving the previous
    # forward's positions in place is the failure mode this guards. A stale
    # position vector is in range and wrong, so every row lands somewhere
    # real and the coverage assertion may not even fire.
    #
    # 🔴 By name, and accepting keyword arguments. ``args[1]`` was the whole
    # of this, and it is None the moment a caller passes positions by
    # keyword -- which silently disables every position-sharded tap,
    # including the prologue's. The dump then simply has no prologue file,
    # and "the embedding was never recorded" reads the same as "there was
    # nothing to record".
    _CURRENT_POSITIONS = _named_argument(_module, "positions", args, kwargs)
    _record_model_inputs(_module, args, kwargs)


def current_positions() -> torch.Tensor | None:
    """The positions of the layer currently being tapped, for inline taps.

    A public name for what the hooks use, so a tap site in another module
    does not reach into a private one -- the private name is free to change
    with the hook, and a cross-module use of it would not be found when it
    does.
    """
    return _CURRENT_POSITIONS


def _current_positions() -> torch.Tensor | None:
    """The positions of the layer whose submodule is being tapped."""
    return _CURRENT_POSITIONS


def install_alignment_taps(model: torch.nn.Module) -> list:
    """Register the boundary hooks, or none at all.

    Returns the handles so a caller can remove them; an empty list is the
    normal case and is what makes the off state provable -- there is nothing
    to enter, not merely nothing to do.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    if get_recorder() is None:
        return []

    handles: list = []
    inner = _find_inner_model(model)
    if inner is None:
        _report_nothing_found(model)
        return []
    # 🔴 First, and on the model rather than on a layer. The layer-level
    # capture below runs only once the first decoder layer starts, which is
    # *after* embed_tokens -- so a prologue tap reading positions would get
    # either None or, worse, the previous step's, which is in range and wrong.
    # This also clears the value, so a forward that never reaches the capture
    # cannot leave the last one visible to the next.
    # Before the positions capture and before every tap: nothing records
    # until the buffer can be sized.
    handles.append(inner.register_forward_pre_hook(_arm_recording))
    handles.append(inner.register_forward_pre_hook(_capture_positions,
                                                   with_kwargs=True))
    # And after everything, where "this request's tensors are all in" is true.
    handles.append(inner.register_forward_hook(_flush_and_rotate))
    # Counted apart from the positions capture, which always registers. Using
    # len(handles) would have made the "nothing was tapped" branch below
    # unreachable -- a guard that cannot fire, which is worse than none
    # because it reads as coverage.
    taps_installed = 0
    embed = getattr(inner, "embed_tokens", None)
    if embed is not None and type(embed).__name__ != "PPMissingLayer":
        # 🔴 The prologue is not optional plumbing. AlignRecorder.record()
        # raises until set_seq_len() has been called, so without this hook the
        # very first tap -- sa_in.q -- fails and the whole dump is empty. That
        # is what happened on the first real run: six taps were entered, none
        # recorded, and the error correctly said so.
        #
        # It lives on embed_tokens because that is the one module that runs
        # once per forward, before any layer, and whose output *is* a protocol
        # key. A separate "prologue" module does not exist to hook.
        handles.append(embed.register_forward_pre_hook(_prologue_pre))
        handles.append(embed.register_forward_hook(_prologue_post))
        taps_installed += 2
    else:
        # 🔴 Said out loud. The prologue is the only observation point before
        # layer 0, and a dump without it is missing the one section that
        # could show whether the embedding is right. "No prologue hooks were
        # installed" and "the prologue had nothing to record" produce the
        # same artifact -- an absent prologue.safetensors -- and only this
        # line tells them apart.
        from vllm.logger import logger

        logger.error(
            "DSV4.1 alignment: no prologue hooks installed. %s.embed_tokens "
            "is %s, so input_ids and embed.out will be absent from every "
            "dump this process writes. That is the section before layer 0, "
            "and its absence is not evidence that there was nothing to "
            "record.", type(inner).__name__,
            "missing" if embed is None else type(embed).__name__)
    for layer_id, layer in _decoder_layers(model):
        # First, so the boundary hooks below it in the same layer see this
        # layer's positions rather than the previous layer's.
        handles.append(layer.register_forward_pre_hook(_capture_positions,
                                                       with_kwargs=True))
        # The layer's own inputs, before its submodules see anything. These
        # are what the injector replaces, so they are read from the argument
        # the layer was handed.
        #
        # ⚠️ Layer 0 is a different shape of thing and is left to produce
        # nothing here: it receives ``x`` as 2-D [T, 5120] and builds the
        # 4-stream tensor itself (``model.py:303-305``), and its ``pre_mix``
        # is None. The prologue's ``embed.out`` is what layer 0 was actually
        # handed. Recording an expanded [T, 4, 5120] for it would be
        # equivalent *today* -- expand loses nothing -- and that is exactly
        # the trap: equivalence is a property of the current code, not of the
        # interface, so the day that line stops being a broadcast nothing
        # goes red and the injector quietly compares something else.
        handles.append(
            layer.register_forward_pre_hook(
                _make_argument_tap(layer_id, _arg_positions(layer)),
                with_kwargs=True))
        taps_installed += 1
        for attribute, key, side, parameter in _BOUNDARY_TAPS:
            module = getattr(layer, attribute, None)
            if module is None:
                continue
            handles.append(_attach(module, layer_id, key, side, parameter))
            taps_installed += 1
        taps_installed += _attach_moe_internals(layer, layer_id, handles)

    # Said out loud, because the two failures so far were both diagnosed by
    # inferring "the hooks were never installed" from "the dump is empty" --
    # two steps away from the fact. Zero handles with a recorder present is a
    # defect, not a quiet no-op, so it is reported as one.
    from vllm.logger import logger

    if taps_installed:
        logger.info(
            "DSV4.1 alignment: installed %d taps (%d decoder layers, "
            "prologue %s)", taps_installed,
            sum(1 for _ in _decoder_layers(model)),
            "yes" if embed is not None else "MISSING")
    else:
        logger.error(
            "DSV4.1 alignment: found %s (it has embed_tokens and layers) but "
            "installed zero taps -- it holds no decoder layers this "
            "recognises. Every inline tap will reach record() before "
            "set_seq_len() and raise.", type(inner).__name__)
    return handles


#: Submodules of the FFN whose outputs bracket the MoE's own steps. Golden
#: records only ``moe.in`` and ``moe.out``, so when those two disagree the
#: segment between them is a black box -- the same shape as the q chain,
#: where the defect was found by two of **our** keys contradicting each other
#: rather than by comparing with the reference.
#:
#: Measured on flight xr_a2b_011352: moe.in agrees with golden (RMS 0.1263 vs
#: 0.1264) and moe.out is ~5e6x too large *and* pointing elsewhere (per-row
#: cosine 0.29-0.34), so it is not a missing scalar factor. Magnitude and
#: direction are separable by these three:
#:
#:   moe.router.out   which experts, with what weights -- wrong experts move
#:                    direction, unnormalised weights move magnitude
#:   moe.experts.out  the routed experts' aggregate, before the shared expert
#:   moe.shared.out   the shared expert, which every token goes through and
#:                    which is added separately (ref/model.py:889-891)
_MOE_TAPS: tuple[tuple[str, str], ...] = (
    # 🔵 golden's own key name, so the two dumps need no mapping: the gate
    # returns (weights, indices) and _flatten_outputs suffixes them, giving
    # moe_gate.out.0 (F32 [T, topk] weights) and moe_gate.out.1 (I64 [T, topk]
    # expert ids) -- exactly what golden records.
    ("gate", "moe_gate.out"),
    ("experts", "moe.experts.out"),
    ("shared_experts", "moe.shared.out"),
)


def _attach_moe_internals(layer: torch.nn.Module, layer_id: int,
                          handles: list) -> int:
    """Hook the FFN's own submodules, and say which ones were not there.

    ⚠️ Attached by attribute name on the module this build actually has, and
    an absent name is reported rather than skipped: "this build fuses the
    gate into the experts" and "the tap table is stale" produce the same
    empty section otherwise.
    """
    ffn = getattr(layer, "ffn", None) or getattr(layer, "moe", None)
    if ffn is None:
        return 0
    attached = 0
    for attribute, key in _MOE_TAPS:
        module = getattr(ffn, attribute, None)
        if module is None or not isinstance(module, torch.nn.Module):
            _report_missing_moe_part(attribute, type(ffn).__name__)
            continue
        handles.append(module.register_forward_hook(
            _make_moe_tap(layer_id, key)))
        attached += 1
    return attached


_REPORTED_MISSING_MOE: set = set()


def _report_missing_moe_part(attribute: str, owner: str) -> None:
    if attribute in _REPORTED_MISSING_MOE:
        return
    _REPORTED_MISSING_MOE.add(attribute)
    from vllm.logger import logger

    logger.warning(
        "DSV4.1 alignment: %s has no %r submodule, so moe.%s will be absent "
        "from every dump. That may be right -- a fused build has no separate "
        "gate -- but absent-because-fused and absent-because-this-table-is-"
        "stale produce the same empty section, so it is said here.",
        owner, attribute, attribute)


def _make_moe_tap(layer_id: int, key: str):
    """Record whatever this submodule returned, under one key per element.

    The gate returns ``(weights, indices)`` and the experts return a tensor;
    rather than branch on which, every tensor in the returned structure is
    recorded under a suffixed key. A branch would have to be right about a
    shape this module does not own.
    """

    def _tap(_module: torch.nn.Module, _args, output) -> None:
        for suffix, tensor in _flatten_outputs(output):
            if not isinstance(tensor, torch.Tensor):
                continue
            name = f"{key}{suffix}"
            positions = _current_positions()
            rows = tensor.shape[0] if tensor.ndim else 0
            if positions is not None and rows == positions.shape[0]:
                _record(layer_id, name, tensor)
            else:
                # No token axis (or a fused layout this does not understand):
                # recorded whole rather than scattered, because a wrong
                # scatter is silent and an unscattered tensor is not.
                tap_whole(layer_id, name, tensor)

    return _tap


def _flatten_outputs(output, prefix: str = ""):
    """``(suffix, value)`` for every tensor in a returned structure."""
    if isinstance(output, torch.Tensor):
        yield prefix, output
    elif isinstance(output, (tuple, list)):
        for i, item in enumerate(output):
            yield from _flatten_outputs(item, f"{prefix}.{i}")
    elif isinstance(output, dict):
        for name, item in output.items():
            yield from _flatten_outputs(item, f"{prefix}.{name}")


def _make_argument_tap(layer_id: int, positions: dict[str, int]):
    """Record the layer's own inputs, by name, from the args it was given."""

    # ⚠️ with_kwargs. The caller passes some of these by keyword, and a hook
    # that only sees positional arguments finds nothing and records nothing
    # -- silently, which is the same failure that left the prologue empty all
    # day. Caught here by a test whose double passes engram_hashes by
    # keyword, which is how the real model passes it.
    def _tap(_module: torch.nn.Module, args: tuple,
             kwargs: dict | None = None) -> None:
        if layer_id == 0:
            # 🔵 What layer 0 was handed *is* the embedding output, whichever
            # path produced it -- the embedding module, a multimodal wrapper's
            # inputs_embeds, or anything else. Recorded here rather than at
            # embed_tokens for that reason: the argument exists on every path,
            # the module call does not.
            x = _named_argument(_module, "x", args, kwargs)
            if isinstance(x, torch.Tensor):
                _record_prologue("embed.out", x)
        for name, key in _FORWARD_ARG_TAPS:
            # No skip and no marking here any more: _record_prologue is the
            # door and owns the once-per-forward guard for every prologue
            # key. Two places deciding "already recorded" is how the other
            # three keys ended up with no such decision at all.
            value = _named_argument(_module, name, args, kwargs)
            if isinstance(value, torch.Tensor):
                # Under the prologue tag: the model computes it once for the
                # whole forward, so tagging it to whichever layer happened to
                # be first would make it look layer-specific and would put it
                # somewhere golden does not compare.
                _record_prologue(key, value)
        for key, index in positions.items():
            value = (args[index] if index < len(args) else None)
            if value is None and kwargs:
                value = kwargs.get(_ARG_NAME_BY_KEY.get(key, ""))
            # None is the honest answer for layer 0's residual and pre_mix,
            # and it is recorded as absence rather than as a fabricated
            # identity: "layer 0 has no pre_mix" is a fact the dump has to be
            # able to show, and an identity would erase it.
            if isinstance(value, torch.Tensor):
                _record(layer_id, key, value)

    return _tap


def _prologue_pre(_module: torch.nn.Module, args: tuple) -> None:
    """Fix the sequence length, then record input_ids.

    The order matters and is the whole point of this hook: every record()
    scatters into a [seq_len, ...] buffer, so seq_len has to be known before
    the first one. Set from the request's own total length rather than from
    the number of rows in this chunk -- under chunked prefill those differ,
    and sizing the buffer by the chunk would drop every later row off the end.
    """
    input_ids = _first_tensor(args)
    if input_ids is not None:
        _record_prologue("input_ids", input_ids)


def _prologue_post(_module: torch.nn.Module, _args: tuple,
                   output: Any) -> None:
    tensor = _first_tensor(output)
    if tensor is not None:
        _record_prologue("embed.out", tensor)


def _record_prologue(key: str, tensor: torch.Tensor) -> None:
    """One prologue tensor, once per forward, scattered by position.

    Tagged "prologue" rather than by layer: golden keeps the model-level
    tensors in their own file, and a per-layer tag here would put them in
    layer_00 where nothing compares them.

    🔴 **Once per forward, and the guard lives here** -- not at the call
    sites, because counting the call sites is what failed. Every prologue key
    is a property of the forward, not of whoever noticed it, and each one had
    more writers than anybody had counted:

        input_ids   the embed_tokens pre-hook, and the model's own arguments
        embed.out   the embed_tokens post-hook, and layer 0's ``x``
        positions   the model's arguments -- and ``_capture_positions`` is
                    registered on the inner model **and on every layer**
                    (install_alignment_taps :615 and :657), so 40 layers each
                    recorded it again: 41 writers for one key

    Flight-23 died on it: ``position coverage is not exactly 1``, with
    ``duplicated`` listing *every* position of ``input_ids`` and
    ``positions``. The second writer of each was added for a real reason (the
    embedding is bypassed when a caller supplies ``inputs_embeds``, so an
    argument-side reader is needed) -- the mistake was adding it without
    asking how many writers the key already had.

    ⚠️ ``embed.out`` has two writers as well and did *not* appear in that
    report. That is luck about ordering, not a difference in kind.

    The guard is here rather than at each site so that a fourth writer needs
    no change: this is the only door into the prologue section.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import current_row_to_pos

    if key in _FORWARD_RECORDED:
        return
    recorder = _armed_recorder()
    if recorder is None:
        return
    positions = _current_positions()
    if positions is None:
        return
    row_to_pos = current_row_to_pos(positions)
    if row_to_pos is None:
        return
    if not bool((row_to_pos >= 0).any()):
        # Same reason as the layer path: an all-padding forward would create
        # a buffer nothing scatters into, and the flush then reports the key
        # as missing everywhere -- which is the artifact a broken mapping
        # produces. The prologue had this guard missing while the layer path
        # had it.
        _report_all_padding(-1, key, row_to_pos.numel())
        return
    # Marked only after the write actually happened. Marking on entry would
    # let an early return -- not armed yet, no positions yet -- consume the
    # key and lock out the writer that could have recorded it, which is the
    # same "absent and silent" state this whole section exists to end.
    _FORWARD_RECORDED.add(key)
    recorder.record("prologue", key, tensor, row_to_pos)


def _single_request_seq_len() -> int | None:
    """The one request's total sequence length, or None if there is no batch.

    None on a profile or dummy run -- there is nothing to size a buffer with,
    and guessing would produce a dump shaped like a real one.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return None
    metadata = getattr(context, "attn_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for entry in metadata.values():
        req = getattr(entry, "req_metadata", None)
        seq_lens = getattr(req, "seq_lens", None)
        if seq_lens is None or seq_lens.numel() == 0:
            continue
        if seq_lens.numel() != 1:
            raise ValueError(
                f"alignment recording needs a single request, got "
                f"{seq_lens.numel()} sequence lengths. positions are "
                "per-request, so two requests both start at 0 and their rows "
                "scatter onto each other -- silently, since every value stays "
                "in range.")
        return int(seq_lens[0])
    return None


def _single_request_query_rows() -> int | None:
    """How many query rows this forward carries, or ``None`` if unavailable.

    From ``query_start_loc[-1]``, which is the one place the count exists
    without re-deriving it: it is the cumulative row boundary the whole
    attention path already agrees on.
    """
    from vllm.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except Exception:                                        # noqa: BLE001
        return None
    metadata = getattr(context, "attn_metadata", None)
    if not isinstance(metadata, dict):
        return None
    for entry in metadata.values():
        req = getattr(entry, "req_metadata", None)
        starts = getattr(req, "query_start_loc", None)
        if starts is None or starts.numel() == 0:
            continue
        return int(starts[-1])
    return None


def _attach(module: torch.nn.Module, layer_id: int, key: str,
            side: str, parameter: "str | None" = None) -> Any:
    index = None if parameter is None else _parameter_index(module, parameter,
                                                            key)

    def _pre(_mod: torch.nn.Module, args: tuple) -> None:
        # By position resolved from the name, never "the first one". The
        # first positional argument of this module is `positions`, and
        # recording it under `attn.in` is how two sides ended up with the
        # same key holding different tensors.
        if index is None or index >= len(args):
            return
        tensor = args[index]
        if isinstance(tensor, torch.Tensor):
            _record(layer_id, key, tensor)

    def _post(_mod: torch.nn.Module, _args: tuple, output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is not None:
            _record(layer_id, key, tensor)

    if side == "in":
        return module.register_forward_pre_hook(_pre)
    return module.register_forward_hook(_post)


def _find_inner_model(model: torch.nn.Module) -> torch.nn.Module | None:
    """The module that actually holds ``embed_tokens`` and ``layers``.

    Searched by capability, not by path. ``model.model`` was hard-coded here
    and found nothing, because V4.1-Flash is a multimodal checkpoint and the
    real path is ``model.language_model.model`` -- one wrapper deeper.

    🔴 That wrapper has now broken three different assumptions, each one
    written for the shape of a text-only model, and each failing silently:

        the config      use_compress read hf_config where compress_ratios
                        lives under hf_text_config
        the request     mm_req_doc_ranges is ``{0: []}`` for a text-only
                        request -- a truthy dict of empty lists -- which put
                        every text prefill through the vision index path and
                        was the aicore exception's root cause
        the module tree this function, returning zero taps

    So the fix is not to hard-code one more level. A path that names
    ``language_model`` would work today and go silently empty behind the next
    wrapper, which is the same failure with a different address. Breadth-first
    so the outermost match wins, and depth-limited so a pathological tree
    cannot make model construction crawl.
    """
    queue = [(model, 0)]
    while queue:
        module, depth = queue.pop(0)
        # ``layers`` alone, not layers *and* embed_tokens. A pipeline rank
        # that is not the first has a PPMissingLayer where the embedding
        # would be, and requiring both would find nothing there and tap
        # nothing -- silently, which is the failure this function exists to
        # stop. The prologue is looked up separately and is allowed to be
        # absent for exactly that reason.
        if getattr(module, "layers", None) is not None:
            return module
        if depth < 4:
            queue.extend((child, depth + 1) for child in module.children())
    return None


def _report_nothing_found(model: torch.nn.Module) -> None:
    from vllm.logger import logger

    logger.error(
        "DSV4.1 alignment: a recorder is active but no module under %s has "
        "decoder layers, so zero taps were installed. Every "
        "inline tap will reach record() before set_seq_len() and raise. "
        "Children searched: %s", type(model).__name__,
        [name for name, _ in model.named_children()])


def _decoder_layers(model: torch.nn.Module):
    """``(layer_index, module)`` for each decoder layer, or nothing.

    Indices come from the layer's position in ``layers``, offset by
    ``start_layer`` so they are global under pipeline parallelism rather than
    local to this rank -- golden's tags are global.
    """
    inner = _find_inner_model(model) or model
    layers = getattr(inner, "layers", None)
    if layers is None:
        return
    start = int(getattr(inner, "start_layer", 0))
    for offset, layer in enumerate(layers):
        if isinstance(layer, torch.nn.Module) and type(layer).__name__ != \
                "PPMissingLayer":
            yield start + offset, layer


def expected_keys(dump_dir: str, layers: tuple[int, ...]) -> set[str]:
    """What golden will compare, from golden's own manifest.

    Read from the producer's declaration rather than discovered, because the
    two ways of discovering it have opposite blind spots and both were hit
    within a day:

        scanning the directory   picks up things that are not outputs
                                 (golden's routing_plan fixture, 8 extra keys)
        scanning by name pattern misses anything the pattern does not cover
                                 (this function's own hand-written tag list --
                                 a new protocol tag would silently drop a
                                 whole file from the comparison)

    A manifest has neither, because the side that produced the tensors is the
    side that says which ones they are. Returned as ``tag/key`` so the caller
    can difference in both directions: keys never recorded are a comparison
    that cannot run, keys recorded that golden does not compare are bytes and
    seconds spent for nothing -- and they also make the tensor count disagree,
    which is the number everyone reads first.
    """
    import json
    import os

    path = os.path.join(dump_dir, "manifest.json")
    if not os.path.exists(path):
        raise ValueError(
            f"{path} does not exist, so there is no declaration of what this "
            "dump contains. Falling back to a scan here would reintroduce "
            "exactly the blind spot the manifest removes -- and it would do "
            "so silently, since a scan always returns something.")
    declared = json.loads(open(path).read()).get("tensors")
    if not isinstance(declared, dict) or not declared:
        raise ValueError(
            f"{path} declares no 'tensors'. An empty expected set makes every "
            "coverage comparison vacuous: the difference in both directions "
            "is empty, which reads as full agreement.")

    # An exclusion rule, not an allow-list. Only the per-layer tags are
    # filtered -- by which layers were asked for -- and everything else the
    # manifest declares is kept whatever it is called. An allow-list of
    # {prologue, epilogue, layer_NN} would be this function's own version of
    # the blind spot above: a protocol that grows a new top-level tag would
    # drop it silently, and a dropped tag looks exactly like a tag with
    # nothing in it.
    keep = {f"layer_{i:02d}" for i in layers}
    wanted = {f"{tag}/{key}"
              for tag, keys in declared.items()
              if not _LAYER_TAG.fullmatch(tag) or tag in keep
              for key in keys}
    if not wanted:
        raise ValueError(
            f"{path} declares tags {sorted(declared)}; none survived selecting "
            f"layers {sorted(layers)}. Comparing against an empty set would "
            "report agreement against nothing.")
    return wanted



def golden_layers(golden_dir: str) -> tuple[int, ...]:
    """Which layers golden actually declares, from its own manifest.

    Not a configured range and not a scan of our own output. A layer that we
    failed to record entirely has no tag of ours to be discovered from, so
    deriving the layer list from what we produced would make exactly the
    omission we are looking for invisible -- the missing row cannot show
    itself in a hand-built enumeration of the rows that are present.
    """
    import json
    import os

    path = os.path.join(golden_dir, "manifest.json")
    declared = json.loads(open(path).read()).get("tensors")
    if not isinstance(declared, dict):
        raise ValueError(
            f"{path} declares no 'tensors', so there is no list of layers to "
            "compare against.")
    return tuple(sorted(int(tag.split("_")[1]) for tag in declared
                        if _LAYER_TAG.fullmatch(tag)))


#: The prologue keys the hooks declare they will write, each with the one
#: reason it may legitimately be absent. A key with no such reason that is
#: missing is a defect, and the two must not share an artifact.
PROLOGUE_KEYS: tuple[tuple[str, str], ...] = (
    ("input_ids", "only a pipeline rank that does not hold the embedding"),
    ("positions", "only a pipeline rank that does not hold the embedding"),
    ("embed.out", "only a pipeline rank that does not hold the embedding"),
    ("engram_hash_ids", "only a model with no engram layers"),
)


def declared_coverage(recorded: set[str],
                      layer_tags: "list | None" = None) -> dict:
    """Per key: declared and recorded, declared and missing, or exempt.

    🔴 The gap this closes. ``coverage_report`` answers "does this dump hold
    what golden compares", and it answers "not_checked" whenever no golden
    directory is set -- which is most runs. Between those two there was a
    fourth state nobody reported: the section exists, the keys are missing,
    and nothing says so. That is how ``input_ids`` / ``positions`` /
    ``embed.out`` stayed absent for a day while every flight looked fine.

    So this asks a question that needs no golden at all: **for every key the
    hooks in this module declare they will write, was it written?** One line
    per key, three states, and the exemptions named rather than assumed --
    ``layer.in`` and ``pre_mix.in`` are genuinely absent on the first layer
    (it receives a 2-D embedding and builds the streams itself), and saying
    "missing" there would train the reader to skip the report.

    Scope is stated, not implied: only the hook-declared keys. The inline
    taps (``sa_in.*``, ``read.*``) fire on selected layers by design, so
    "absent" carries no information about them and listing them would drown
    the keys where it does.

    ``layer_tags`` is the set of layers this forward actually had, supplied by
    the caller. Without it the layers are inferred from ``recorded``, and a
    layer whose every key was refused would then not appear at all.
    """
    # 🔴 The universe of layers comes from the caller when it can say, not
    # from the recorded set. Deriving it from what survived means a layer all
    # of whose keys were refused disappears from the report entirely -- the
    # report's own subject list taken from the thing it is judging, which is
    # how "nothing was recorded" ends up reading as "nothing was missing".
    if layer_tags is None:
        layer_tags = [entry.split("/", 1)[0] for entry in recorded]
    layer_tags = sorted({tag for tag in layer_tags
                         if str(tag).startswith("layer_")})
    first_layer = layer_tags[0] if layer_tags else None

    prologue = {}
    for key, exemption in PROLOGUE_KEYS:
        prologue[key] = ("recorded" if f"prologue/{key}" in recorded else
                         f"MISSING -- expected unless {exemption}")

    per_layer = {}
    for key in [k for _, k, _, _ in _BOUNDARY_TAPS] + [k for _, k in _ARG_TAPS]:
        exempt = ([first_layer] if key in {k for _, k in _ARG_TAPS}
                  and first_layer is not None else [])
        wanted = [tag for tag in layer_tags if tag not in exempt]
        missing = [tag for tag in wanted if f"{tag}/{key}" not in recorded]
        per_layer[key] = {
            "recorded_in": len(wanted) - len(missing),
            "of": len(wanted),
            "missing": missing,
            "exempt": exempt,
        }

    problems = [f"prologue/{key}" for key, state in prologue.items()
                if state != "recorded"]
    problems += [f"{tag}/{key}" for key, row in per_layer.items()
                 for tag in row["missing"]]
    return {
        "state": "ok" if not problems else "incomplete",
        "scope": ("the keys the hooks in align_taps declare they will write. "
                  "Inline taps (sa_in.*, read.*) fire on selected layers by "
                  "design and are not judged here."),
        "layers_seen": len(layer_tags),
        "prologue": prologue,
        "per_layer": per_layer,
        "problems": problems,
    }


def coverage_report(recorded: set[str], golden_dir: str | None) -> dict:
    """Did this run record everything golden will compare, and nothing else.

    🔴 ``AlignRecorder``'s flush already asserts position coverage -- every
    token of every recorded key written exactly once. What it cannot see is a
    key that was **never recorded at all**: there is no buffer, so there is
    no coverage vector, so there is nothing for that assertion to look at.
    A whole tensor silently absent and a whole tensor perfectly recorded
    produce the same green flush.

    Both directions are reported because they mean different things and only
    one of them is obvious:

        missing   golden compares it, we never wrote it
                  -> that comparison cannot run, and its absence looks
                     exactly like a comparison that passed
        extra     we wrote it, golden does not compare it
                  -> bytes and seconds for nothing, and it moves the tensor
                     count, which is the number everyone reads first

    ⚠️ Three states. Without a golden directory this is **not checked**, and
    that is returned as its own state rather than as two empty differences --
    empty differences read as full agreement, which is the one reading that
    must not be available to something that did not run.
    """
    if not golden_dir:
        return {
            "state": "not_checked",
            "reason": ("VLLM_ASCEND_DSV41_GOLDEN_DIR is not set, so there is "
                       "no declaration of what this dump should contain. "
                       "This is not a pass."),
            "recorded_count": len(recorded),
        }
    try:
        wanted = expected_keys(golden_dir, golden_layers(golden_dir))
    except Exception as exc:                       # noqa: BLE001
        return {
            "state": "not_checked",
            "reason": (f"golden's manifest could not be read ({exc!r}), so "
                       "the expected set is unknown. This is not a pass."),
            "recorded_count": len(recorded),
        }

    missing = sorted(wanted - recorded)
    extra = sorted(recorded - wanted)
    return {
        "state": "ok" if not missing else "incomplete",
        "missing": missing,
        "extra": extra,
        "expected_count": len(wanted),
        "recorded_count": len(recorded),
    }


def alignment_active() -> bool:
    """Whether anything should be recorded at all.

    Called at each inline tap site; the tap function itself is not entered
    when this is False, which is the same guarantee the hooks give
    structurally. Kept as one call rather than an attribute lookup so the tap
    sites read the same way the hook installer does.
    """
    from vllm_ascend.models.deepseek_v4_1.debug_align import get_recorder

    return get_recorder() is not None


def layer_index(prefix: str, hf_text_config) -> int:
    """The layer number golden tags with, out of the module prefix.

    Uses the project's own extractor rather than a split on ".", because MTP
    layers keep their original namespace and are addressed after the main
    layers -- a naive parse tags them as 0..n and silently overwrites the
    real layer 0.
    """
    from vllm_ascend.utils import extract_dsv4_layer_index

    return extract_dsv4_layer_index(hf_text_config, prefix)


def tap(layer_id: int, key: str, tensor: torch.Tensor,
        positions: torch.Tensor) -> None:
    """One position-sharded tensor, from inside a forward that owns it.

    For values that are not module boundaries -- ``sa_in.*``, ``hc_*``,
    ``moe_gate.*``, ``pre_mix.*``. A hook cannot see these: they never cross a
    module edge, and hooking the module whose name they carry is how you get a
    tap that never fires (see this module's docstring on attn_norm).
    """
    _record_with(layer_id, key, tensor, positions)


def tap_whole(layer_id: int, key: str, tensor: torch.Tensor) -> None:
    """One tensor that is not sharded by position.

    ``sa_in.attn_sink`` is per head and ``sa_in.softmax_scale`` is a scalar --
    neither has a token axis, so scattering them by row would be meaningless
    and the coverage assertion would never be satisfied.
    """
    recorder = _armed_recorder()
    if recorder is None:
        return
    recorder.record_whole(f"layer_{layer_id:02d}", key, tensor)


def _record_with(layer_id: int, key: str, tensor: torch.Tensor,
                 positions: torch.Tensor) -> None:
    from vllm_ascend.models.deepseek_v4_1.debug_align import current_row_to_pos

    recorder = _armed_recorder()
    if recorder is None:
        return
    row_to_pos = current_row_to_pos(positions)
    if row_to_pos is None:
        return
    recorder.record_layer(layer_id, key, tensor, row_to_pos)
