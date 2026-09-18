# SPDX-License-Identifier: Apache-2.0
"""A block-scale parameter must be declared in a dtype the loader's bytes survive.

The defect this pins: the routed experts' scale parameters were declared
``torch.float8_e8m0fnu`` while the loader hands over a ``uint8`` **view** of the
checkpoint's bytes. One byte wide either way, every shape agreeing, nothing
downstream complaining -- and every scale in the model rewritten to the same
constant.

The full calibration, measured on CPU against
``layers.0.ffn.experts.0.w1.scale`` (``(2304, 160) float8_e8m0fnu``, bytes
119..121)::

    source dtype              target dtype        bytes after copy_
    ------------------------  ------------------  ---------------------------
    e8m0 (as safetensors)     uint8               BAD: all 0
    e8m0 (as safetensors)     float8_e8m0fnu      ok: 119..122   preserved
    uint8 view                uint8               ok: 119..122   preserved
    uint8 view                float8_e8m0fnu      BAD: all 134    constant

Which row is live was settled by two independent things: this table, and a
runtime read of a pre-fix flight (arm 0ecd4f2bb) reporting
``w13_weight_scale ... dtype=uint8 ... min=134 median=134 max=134`` over
70,778,880 elements, read direct. Only the last row produces 134, so the loader
passes the uint8 view -- and therefore ``uint8`` is the declaration that keeps
the bytes.

The assertion is behavioural, not ``dtype is uint8``. Restating the fix would
pass for a parameter declared in some third dtype that also destroys the bytes;
what matters is that a byte written by the loader can be read back. So the test
performs the loader's own copy and compares bytes.

The parameters are selected by a conjunction, not by name alone: an earlier
version of a sibling check keyed on "the name contains scale" and went red on
seven parameters that are legitimately floating point -- a criterion wider than
the defect fails on correct code, and the repair for that is to loosen it, which
is how a criterion loses its power.
"""

from __future__ import annotations

import pytest
import torch

GROUP_SIZE = 32
#: Bytes an E8M0 scale actually takes in this checkpoint (measured on
#: layers.0.ffn.experts.0.w1.scale), recorded for context. The property below is
#: NOT restricted to this range: a declaration has to store whatever byte the
#: loader hands it, and restricting the test to 119..124 let torch.int8 pass --
#: it represents those values exactly and only breaks at 128 and above. A
#: negative control that the hazard survives is not a negative control.
OBSERVED = range(119, 125)


def _methods():
    """The MoE schemes whose scale parameters are MX block scales."""
    from vllm_ascend.quantization.methods.w4a8.w4a8_mxfp4 import (
        AscendW4A8MXFPDSDynamicFusedMoEMethod, AscendW4A8MXFPDynamicFusedMoEMethod)
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import (
        AscendW8A8MXFP8DynamicFusedMoEMethod)
    return [AscendW4A8MXFPDSDynamicFusedMoEMethod,
            AscendW4A8MXFPDynamicFusedMoEMethod,
            AscendW8A8MXFP8DynamicFusedMoEMethod]


def _scale_params(cls):
    """(name, tensor) for every declared parameter that is an MX block scale.

    Two conditions, both required: the name says scale, AND the last dimension is
    the operand's divided by the group size. The second is what separates a block
    scale from the per-tensor and per-token scales that are legitimately float.
    """
    inst = cls.__new__(cls)          # no __init__: it wants a live vllm config
    inst.group_size = GROUP_SIZE
    params = cls.get_dynamic_quant_param(
        inst, num_experts=2, intermediate_size_per_partition=64,
        hidden_sizes=128, params_dtype=torch.bfloat16)
    out = []
    for name, t in params.items():
        if "scale" not in name:
            continue
        if t.shape[-1] not in (128 // GROUP_SIZE, 64 // GROUP_SIZE):
            continue                  # not divided by the group size: not a block scale
        out.append((name, t))
    return out


def test_block_scale_parameters_survive_the_loaders_byte_copy():
    seen = 0
    for cls in _methods():
        for name, param in _scale_params(cls):
            seen += 1
            # What the loader does: a uint8 view of the checkpoint's bytes,
            # copied into the declared parameter.
            src = torch.randint(0, 256, param.shape, dtype=torch.uint8)
            dst = torch.empty(param.shape, dtype=param.dtype)
            dst.copy_(src)
            back = dst.view(torch.uint8)
            assert torch.equal(back, src), (
                f"{cls.__name__}.{name} is declared {param.dtype}, and the loader's bytes do "
                f"not survive being copied into it: {src.flatten()[:4].tolist()} went in, "
                f"{back.flatten()[:4].tolist()} came out. A one-byte dtype that reinterprets "
                f"rather than stores rewrites every scale in the model, and nothing downstream "
                f"can tell -- the widths and shapes all agree.")
    assert seen >= 6, (
        f"only {seen} block-scale parameters were examined; this check is meant to cover the "
        f"w13/w2 pair of three MoE methods. A drop in coverage looks exactly like a pass.")


@pytest.mark.parametrize("bad", [torch.float8_e8m0fnu, torch.float8_e4m3fn])
def test_the_check_fails_for_a_dtype_that_reinterprets(bad):
    """The assertion above must be able to go red, including on the exact pre-fix dtype.

    float8_e8m0fnu is what w4a8_mxfp4.py declared before 0608ac1bd. e4m3 is here so the control is
    not pinned to the single value that happened to be wrong.
    """
    src = torch.randint(0, 256, (8, 4), dtype=torch.uint8)
    dst = torch.empty((8, 4), dtype=bad)
    dst.copy_(src)
    assert not torch.equal(dst.view(torch.uint8), src), (
        f"{bad} preserved the bytes, so it would not have been caught -- and the test above "
        f"cannot distinguish a safe declaration from an unsafe one")


@pytest.mark.parametrize("safe", [torch.uint8, torch.int8])
def test_these_dtypes_do_preserve_the_bytes(safe):
    """int8 is here because an earlier version of this file asserted it was a hazard.

    It is not: copying uint8 200 into int8 wraps to -56, and viewing that as uint8 gives 200 back.
    Two's complement round-trips every byte. The earlier version drew its test bytes from the
    range this checkpoint happens to hold (119..124), where int8 is exact anyway, so the mistake
    was invisible in both directions -- the control passed for the wrong reason and would have
    failed a correct declaration.

    Recorded as a positive list rather than deleted, so that declaring the parameter int8 is not
    later "fixed" on the strength of a wrong memory.
    """
    src = torch.randint(0, 256, (8, 4), dtype=torch.uint8)
    dst = torch.empty((8, 4), dtype=safe)
    dst.copy_(src)
    assert torch.equal(dst.view(torch.uint8), src), f"{safe} did not round-trip a byte"
