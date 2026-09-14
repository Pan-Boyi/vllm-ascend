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


@pytest.fixture(autouse=True)
def _armed(monkeypatch):
    """🔴 Armed explicitly. The criterion reports instead of raising until a
    flight has shown it clean on the device -- the last instrument here that
    could stop a launch did stop one, taking four workers down for eight
    printed bytes. A test that asserts a refusal has to switch that on, and
    one that asserts the default has to switch it off."""
    monkeypatch.setenv("VLLM_ASCEND_DSV41_SCALE_LAYOUT_STRICT", "1")
    slc._FAILED.clear()


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


def test_bytes_are_compared_not_decoded_values():
    """🔴 The mistake this module exists to catch, nearly made by the module.

    The DS scheme declares w13_weight_scale as float8_e8m0fnu and views it as
    uint8 inside the transform (w4a8_mxfp4.py:329-334, :357-361). A snapshot
    that converted with .to(int64) would read the *decoded exponent* before
    and the *stored byte* after, and compare two different quantities.
    """
    raw = torch.tensor([119, 121, 124, 120], dtype=torch.uint8)
    as_e8m0 = raw.view(torch.float8_e8m0fnu)
    assert torch.equal(slc._as_bytes(as_e8m0).bytes, slc._as_bytes(raw).bytes)
    # And the decoded values are nothing like the bytes, which is the point.
    decoded = as_e8m0.to(torch.float32)
    assert not torch.equal(decoded.to(torch.int64),
                           raw.to(torch.int64)), "otherwise this proves nothing"


def test_the_ds_scheme_is_the_one_instrumented():
    """⚠️ ds_w4a8_moe is registered on the SUBCLASS, which overrides
    process_weights_after_loading. Instrumenting the base class's version --
    which was one keystroke away -- would have watched a method that does not
    run on this path, for the fourth time in this investigation."""
    import inspect

    from vllm_ascend.quantization.methods.w4a8 import w4a8_mxfp4

    ds = w4a8_mxfp4.AscendW4A8MXFPDSDynamicFusedMoEMethod
    base = w4a8_mxfp4.AscendW4A8MXFPDynamicFusedMoEMethod
    assert "process_weights_after_loading" in vars(ds), (
        "the subclass overrides it; if that stops being true the base "
        "class's version becomes the live one and the instruments move")
    source = inspect.getsource(ds.process_weights_after_loading)
    assert "verify_scale_layout_preserved" in source
    assert "report_scale_bytes" in source
    assert ds is not base


def test_the_criterion_reports_without_raising_by_default(monkeypatch):
    """The default: loud, and the launch continues."""
    monkeypatch.delenv("VLLM_ASCEND_DSV41_SCALE_LAYOUT_STRICT", raising=False)
    said = []
    monkeypatch.setattr("vllm.logger.logger.error",
                        lambda fmt, *a: said.append(fmt % a))
    slc.reset_reports_for_test()
    data = torch.randint(115, 125, (6, 4), dtype=torch.uint8)
    layer = _layer(weight_scale=data.clone())
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    layer.weight_scale.data = layer.weight_scale.data + 1
    slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ))
    assert len(said) == 1
    assert "NOT raising" in said[0] and "byte histogram changed" in said[0]


def test_an_unreadable_tensor_does_not_stop_anything(monkeypatch):
    """🔴 The failure this module caused, made impossible.

    On the device these tensors are in Ascend's internal format after
    process_weights_after_loading, and PTA refuses reshape/clone/.cpu() on
    one. That raised inside weight loading, all four workers failed to start,
    and a whole-network flight was lost -- for a line whose entire job was to
    print eight bytes.
    """
    monkeypatch.delenv("VLLM_ASCEND_DSV41_SCALE_LAYOUT_STRICT", raising=False)
    warned = []
    monkeypatch.setattr("vllm.logger.logger.warning",
                        lambda fmt, *a: warned.append(fmt % a))
    slc.reset_reports_for_test()
    slc._FAILED.clear()

    # ⚠️ Patched at the real failure point. A stand-in object that is not a
    # torch.Tensor never reaches the reading code at all -- it is skipped by
    # the isinstance guard, so a test built that way exercises nothing and
    # passes for the wrong reason.
    # Signature copied from scale_layout_check._readable, not simplified:
    # a double that drops a parameter goes green on a call the production
    # code can no longer make.
    def _unreadable(_data, view_as=None):
        raise RuntimeError(
            "Current device only support aclnn operator, and current operator "
            "...::clone do not support internal format")

    monkeypatch.setattr(slc, "_readable", _unreadable)
    layer = _layer(weight_scale=torch.zeros(4, 2, dtype=torch.uint8))

    # Returns None instead of raising, and says why exactly once.
    assert slc.snapshot_scales(layer, ("weight_scale", )) is None
    assert len(warned) == 1 and "was skipped" in warned[0]
    assert "internal format" in warned[0]
    # The reading entry point is equally harmless, and does not repeat itself.
    slc.report_scale_bytes(layer, ("weight_scale", ), where="x")
    assert len(warned) == 2, "once per entry point, not once per call"


def test_a_missing_snapshot_does_not_become_a_false_verdict():
    """If the "before" could not be read, there is nothing to compare -- and
    silence is the right answer, not a passed check."""
    slc.reset_reports_for_test()
    layer = _layer(weight_scale=torch.zeros(4, 2, dtype=torch.uint8))
    slc.verify_scale_layout_preserved(None, layer, ("weight_scale", ))
    slc.verify_scale_layout_preserved({}, layer, ("weight_scale", ))


def test_the_guard_lets_the_criterion_speak(monkeypatch):
    """🔴 A guard must not share the failure mode of what it guards.

    The net that stops a broken *reading* from killing a launch must not also
    swallow the criterion's *verdict* -- the first version of it did, leaving
    the check armed and mute, which is the worst of the three states.
    """
    monkeypatch.setenv("VLLM_ASCEND_DSV41_SCALE_LAYOUT_STRICT", "1")
    slc.reset_reports_for_test()
    slc._FAILED.clear()
    data = torch.randint(115, 125, (6, 4), dtype=torch.uint8)
    layer = _layer(weight_scale=data.clone())
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    layer.weight_scale.data = layer.weight_scale.data + 1
    with pytest.raises(slc.ScaleLayoutRefusal):
        slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ))
    # And an instrument failure in the same call still does not propagate.
    assert slc.snapshot_scales(object(), ("weight_scale", )) == {}


def test_two_arms_read_by_different_routes_are_not_graded(monkeypatch):
    """🔴 A pair read two different ways cannot be graded either way.

    ``npu_format_cast`` is a fallback for tensors this host cannot read
    directly, and it is documented to be able to change *bytes*, not only
    layout. So when the "before" reading came out direct and the "after" one
    came through the cast, any difference between them is equally explained
    by the cast -- and so is any *agreement*.

    ⚠️ Of the three possible answers here the dangerous one is "OK": a clean
    pass is what a reader will quote, and it would be quoting the instrument,
    not the transform. The check has to stand down and say the two arms were
    read differently, which is why this asserts on the *absence* of the OK
    line as well as on the presence of the warning.
    """
    monkeypatch.setenv("VLLM_ASCEND_DSV41_SCALE_LAYOUT_STRICT", "1")
    slc.reset_reports_for_test()
    slc._FAILED.clear()
    said = []
    monkeypatch.setattr("vllm.logger.logger.warning",
                        lambda fmt, *a: said.append(("warn", fmt % a)))
    monkeypatch.setattr("vllm.logger.logger.info",
                        lambda fmt, *a: said.append(("info", fmt % a)))

    data = torch.randint(115, 125, (6, 4), dtype=torch.uint8)
    layer = _layer(weight_scale=data.clone())

    real = slc._readable
    routes = iter(["direct", "via npu_format_cast to ND"])

    def _changing_route(tensor, view_as=None):
        flat, _ = real(tensor, view_as=view_as)
        return flat, next(routes)

    monkeypatch.setattr(slc, "_readable", _changing_route)
    before = slc.snapshot_scales(layer, ("weight_scale", ))
    # The bytes are byte-for-byte identical; only the route differs.
    slc.verify_scale_layout_preserved(before, layer, ("weight_scale", ))

    warnings = [text for kind, text in said if kind == "warn"]
    assert any("NOT CHECKED" in text for text in warnings), said
    assert any("cannot be graded" in text for text in warnings), said
    # 🔴 The wrong answer this exists to prevent.
    assert not any("layout OK" in text for kind, text in said
                   if kind == "info"), said


def test_the_storage_format_is_reported_and_never_raises():
    """Named beside every reading, because shape and dtype cannot tell a
    tensor this host can read from one it cannot."""
    fmt = slc._storage_format(torch.zeros(4, 2, dtype=torch.uint8))
    assert isinstance(fmt, str) and fmt


def test_every_reading_carries_its_route():
    """The route is computed at the only place it can be known; a reading
    that drops it becomes indistinguishable from a direct one downstream --
    which is how the first version of the scale line printed nothing about
    having gone through a conversion."""
    reading = slc._as_bytes(torch.zeros(4, 2, dtype=torch.uint8))
    assert reading.how == "direct"
    assert reading.fmt
    assert reading.bytes.numel() == 8


class _ReshapesButCannotBeCopied(torch.Tensor):
    """Copied from the device's actual behaviour, not from what is convenient.

    🔴 On a FRACTAL_NZ tensor ``reshape(-1)`` **succeeds**; the step that
    raises is the host copy (``copy_``). Measured at L0 on card 4. A double
    that refuses the reshape instead would let a guard wrapped around only
    the reshape pass this test -- which is exactly the bug that shipped.
    """

    @staticmethod
    def __new__(cls, data):
        return torch.Tensor._make_subclass(cls, data, False)

    def to(self, *args, **kwargs):  # noqa: A003
        raise RuntimeError(
            "Current device only support aclnn operator, and current operator "
            "at_npu::native::NPUNativeFunctions::copy_ do not support "
            "internal format")


def _fake_torch_npu(monkeypatch, seen):
    import torch_npu

    def _cast(tensor, fmt):
        seen.append((tensor.dtype, tuple(tensor.shape), fmt))
        return torch.as_tensor(bytearray(range(tensor.nbytes)),
                               dtype=torch.uint8).view(tensor.dtype).reshape(
                                   tensor.shape)

    monkeypatch.setattr(torch_npu, "npu_format_cast", _cast)


def test_the_fallback_fires_on_a_copy_failure_not_only_a_reshape_failure(
        monkeypatch):
    """🔴 A guard whose trigger point is not the failure point is dead code.

    The first version wrapped the reshape alone. On the device the reshape
    succeeds, so it returned happily, labelled the reading "direct", and
    raised from outside itself one line later -- never once taking the
    fallback it existed to provide, and mislabelling the route as well.
    """
    seen = []
    _fake_torch_npu(monkeypatch, seen)
    hostile = _ReshapesButCannotBeCopied(
        torch.randint(115, 125, (4, 4), dtype=torch.uint8))
    # It reshapes fine -- which is why guarding the reshape proved nothing.
    assert hostile.reshape(-1).numel() == 16

    flat, how = slc._readable(hostile)
    assert how == "via npu_format_cast to ND", (
        "the route has to be reported as the conversion it was")
    assert flat.numel() == 16
    assert seen, "the fallback never ran"


def test_the_dtype_view_is_applied_after_the_cast_not_before(monkeypatch):
    """🔴 Order is not free here.

    ``.view(torch.uint8)`` on an internal-format tensor leaves a format
    descriptor that no longer matches its storage, and the cast then refuses
    it: "Parameter c0 of aclnnNpuFormatCast has incorrect value 16. It should
    be 32" -- NZ counts its inner block in elements, so halving the element
    size halves it. Measured at L0 on bf16 and fp16.
    """
    seen = []
    _fake_torch_npu(monkeypatch, seen)
    hostile = _ReshapesButCannotBeCopied(
        torch.randint(115, 125, (4, 4), dtype=torch.uint8)).view(torch.int8)

    slc._readable(hostile, view_as=torch.uint8)

    assert seen, "the fallback never ran"
    cast_dtype, cast_shape, fmt = seen[0]
    assert cast_dtype == torch.int8, (
        f"npu_format_cast was handed {cast_dtype}; it must receive the "
        "tensor's own dtype, with the uint8 view applied afterwards")
    assert cast_shape == (4, 4) and fmt == 2
