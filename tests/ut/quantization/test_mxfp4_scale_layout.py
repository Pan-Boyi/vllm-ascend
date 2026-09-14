# SPDX-License-Identifier: Apache-2.0
"""A layout change moves bytes; it must not change them.

Needs no flight, no reference and no operator experiment -- one function's
input against its output. That is what makes it worth doing first: kernel has
shown both operators *use* the E8M0 scale exactly, so the open half is whether
the scale we hand them is the one the checkpoint held, and this answers the
part of that which can be answered on the ground.

⚠️ A wrong scale leaves no downstream trace: the blocks are cut along the
reduction dimension and the GEMM's sum flattens the structure, so the output
carries nothing that says which block scale was applied.
"""

import pytest
import torch

from vllm_ascend.quantization import scale_layout_check as slc


class _Layer:
    pass


def _layer(**scales):
    layer = _Layer()
    for name, tensor in scales.items():
        setattr(layer, name, torch.nn.Parameter(tensor, requires_grad=False))
    return layer


def test_a_reshape_and_transpose_pass():
    slc.reset_reports_for_test()
    data = torch.randint(115, 125, (8, 6), dtype=torch.uint8)
    layer = _layer(weight_scale=data.clone())
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    layer.weight_scale.data = layer.weight_scale.data.reshape(8, 3, 2).transpose(0, 1)
    slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ))


def test_declared_padding_is_subtracted_not_tolerated():
    """⚠️ "differs by N zeros because we padded" and "differs" must not
    produce the same output."""
    slc.reset_reports_for_test()
    data = torch.randint(115, 125, (4, 3), dtype=torch.uint8)
    layer = _layer(weight_scale=data.clone())
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    layer.weight_scale.data = torch.nn.functional.pad(
        layer.weight_scale.data, (0, 1), value=0).reshape(4, 2, 2).transpose(0, 1)
    slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ),
                                      padded={"weight_scale": (4, 0)})

    # And undeclared padding is caught.
    slc.reset_reports_for_test()
    with pytest.raises(ValueError, match="counts do not reconcile"):
        slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ))


@pytest.mark.parametrize("mutation,match", [
    (lambda t: t + 1, "byte histogram changed"),
    (lambda t: t * 2, "byte histogram changed"),
    (lambda t: torch.zeros_like(t), "byte histogram changed"),
    (lambda t: t.flip(0) + 0 * t, None),           # a permutation: allowed
])
def test_a_transform_that_edits_values_is_caught(mutation, match):
    """🔴 The mutation arm. Each of these is a plausible way to get the
    exponent bias wrong -- off by one, doubled, dropped -- and each leaves the
    shape correct, which is all any shape check would look at."""
    slc.reset_reports_for_test()
    data = torch.randint(115, 125, (6, 4), dtype=torch.uint8)
    layer = _layer(weight_scale=data.clone())
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    layer.weight_scale.data = mutation(layer.weight_scale.data)
    if match is None:
        slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ))
    else:
        with pytest.raises(ValueError, match=match):
            slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ))


def test_the_report_prints_the_byte_range(monkeypatch):
    """🔴 Because the other half of the question is whether those bytes were
    right on arrival, which no invariant on this function can decide -- the
    checkpoint's are 119..124, and only a reader comparing them can tell."""
    said = []
    monkeypatch.setattr("vllm.logger.logger.info",
                        lambda fmt, *a: said.append(fmt % a))
    slc.reset_reports_for_test()
    data = torch.full((4, 2), 121, dtype=torch.uint8)
    layer = _layer(weight_scale=data.clone())
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)
    slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ),
                                      what="w1")
    assert len(said) == 1
    assert "min=121 median=121 max=121" in said[0]
    assert "119..124" in said[0]


def test_a_declared_copy_multiplies_every_bucket_and_adds_no_new_value():
    """🔵 The real path's shape: the tile scales are broadcast from
    [out/block, in/block] to [out, in/group] with repeat_interleave
    (w8a8_mxfp8.py:548-550). A copy multiplies every bucket by one factor and
    introduces no byte value that was not already there."""
    slc.reset_reports_for_test()
    data = torch.randint(115, 125, (4, 3), dtype=torch.uint8)
    layer = _layer(weight_scale=data.clone())
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    layer.weight_scale.data = layer.weight_scale.data.repeat_interleave(32, dim=0)
    slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ),
                                      expanded={"weight_scale": 32})


def test_a_broadcast_that_averages_instead_of_copying_is_caught():
    """🔴 The failure this exists for, and the reason the invariant is exact
    rather than a tolerance: an averaging or requantising 'broadcast' keeps
    the count right and produces byte values that were never in the input."""
    slc.reset_reports_for_test()
    # ⚠️ Built, not sampled. With random bytes the mean can land on a value
    # that was already present, and then the test passes or fails depending
    # on the seed -- which is how it behaved until it was run alongside other
    # tests and the global random state moved. The property being tested is
    # "averaging invents a value", so the data has to guarantee it: the mean
    # of 115 and 125 is 120, and 120 is not in the input.
    data = torch.tensor([[115, 115, 115], [125, 125, 125],
                         [115, 115, 115], [125, 125, 125]], dtype=torch.uint8)
    assert 120 not in set(data.flatten().tolist())
    layer = _layer(weight_scale=data.clone())
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    # Same element count as a x32 copy, but the values are smeared.
    averaged = data.to(torch.float32).mean(dim=0, keepdim=True).round()
    assert int(averaged.flatten()[0]) == 120
    layer.weight_scale.data = averaged.to(torch.uint8).repeat(4 * 32, 1)
    with pytest.raises(ValueError, match="are NEW"):
        slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ),
                                          expanded={"weight_scale": 32})
