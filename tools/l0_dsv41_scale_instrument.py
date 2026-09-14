"""L0: the scale instrument, on the device, on the formats that killed a flight.

Run it before any change to scale_layout_check.py reaches a whole-network
flight::

    source build_artifacts/env_dsv41.sh model <card>
    python tools/l0_dsv41_scale_instrument.py     # exit 0 = nothing raised

🔴 Worth its own file because the CPU unit tests cannot see what this sees.
On 2026-09-14 it found two defects that all 15 of them passed over:
``reshape(-1)`` succeeds on a FRACTAL_NZ tensor (so a guard wrapped around
the reshape was dead code on the only failure it was written for), and
``.view(torch.uint8)`` before the ND cast breaks the format descriptor.

Answers two things the CPU tests cannot:
  1. does anything in this module raise on an internal-format tensor?
  2. is the npu_format_cast fallback byte-preserving, or does the reading
     route itself change the numbers it reports?
(2) is integration's open question, and it is answerable on the ground.
"""
import os, sys, traceback
import torch, torch_npu

torch.npu.set_device(0)  # env_dsv41.sh already masks to the chosen card
from vllm_ascend.quantization import scale_layout_check as slc

FMT = {2: "ND", 3: "NC1HWC0", 4: "FRACTAL_Z", 29: "FRACTAL_NZ"}
fails = []


def step(label, fn):
    try:
        out = fn()
        print(f"  ✅ {label}: {out}")
        return out
    except Exception as exc:
        print(f"  🔴 {label} RAISED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        fails.append(label)
        return None


print("=== 1. plain ND uint8 on device ===")
nd = torch.randint(115, 125, (64, 160), dtype=torch.uint8).npu()
step("_storage_format", lambda: slc._storage_format(nd))
step("_readable", lambda: slc._readable(nd)[1])
step("_as_bytes", lambda: (slc._as_bytes(nd).how, slc._as_bytes(nd).fmt))

print("=== 2. float8_e8m0fnu, the dtype the DS scheme actually declares ===")
if hasattr(torch, "float8_e8m0fnu"):
    e8 = nd.view(torch.float8_e8m0fnu)
    r = step("_as_bytes on e8m0", lambda: slc._as_bytes(e8))
    if r is not None:
        same = torch.equal(r.bytes, slc._as_bytes(nd).bytes)
        print(f"  {'✅' if same else '🔴'} e8m0 view reads the same bytes as uint8: {same}")
        if not same:
            fails.append("e8m0 bytes differ from uint8 bytes")
else:
    print("  ⚠️ torch.float8_e8m0fnu absent in this build -- not tested")

print("=== 3. internal format: the thing that killed xr_sc_032634 ===")
for dtype in (torch.bfloat16, torch.float16, torch.int8):
    for target in (29, 4):
        try:
            t = torch.randn(64, 160).to(dtype).npu() if dtype.is_floating_point \
                else torch.randint(-8, 8, (64, 160), dtype=dtype).npu()
            internal = torch_npu.npu_format_cast(t, target)
        except Exception as exc:
            print(f"  ⚠️ cannot even build {dtype}/{FMT.get(target)}: {exc}")
            continue
        got = slc._storage_format(internal)
        print(f"  {dtype} -> {FMT.get(target)}: stored as {got}")
        if got in ("ND", "host/ND"):
            print("    ⚠️ cast did not take; this arm proves nothing")
            continue
        step(f"_readable on {dtype}/{got}", lambda i=internal: slc._readable(i)[1])
        step(f"_as_bytes on {dtype}/{got}", lambda i=internal: slc._as_bytes(i).how)

print("=== 4. ⭐ is the fallback route byte-preserving? ===")
print("  (ND -> internal -> ND, compared against the bytes we started with)")
for dtype, target in ((torch.int8, 29), (torch.float16, 29), (torch.bfloat16, 29)):
    try:
        src = torch.randint(-120, 120, (64, 160), dtype=torch.int8).npu()
        src = src if dtype == torch.int8 else src.view(torch.uint8)[:, :160].view(
            torch.uint8).to(torch.uint8).view(torch.uint8)
        base = torch.randint(0, 255, (64, 160), dtype=torch.uint8).npu()
        orig_bytes = base.cpu().clone()
        wearing = base.view(dtype) if dtype != torch.int8 else base.view(torch.int8)
        internal = torch_npu.npu_format_cast(wearing, target)
        if slc._storage_format(internal) in ("ND", "host/ND"):
            print(f"  ⚠️ {dtype}: cast did not take, skipped")
            continue
        back = torch_npu.npu_format_cast(internal, 2)
        back_bytes = back.view(torch.uint8).cpu()
        same = torch.equal(orig_bytes.flatten(), back_bytes.flatten())
        n_diff = int((orig_bytes.flatten() != back_bytes.flatten()).sum())
        print(f"  {'✅' if same else '🔴'} {dtype} via {FMT[target]}: round trip "
              f"byte-identical={same}, differing bytes={n_diff}/{orig_bytes.numel()}")
        if not same:
            fails.append(f"format_cast round trip changed bytes for {dtype}")
    except Exception as exc:
        print(f"  ⚠️ {dtype}: {type(exc).__name__}: {exc}")

print("=== 5. the public entry points, end to end, on device ===")


class L:
    pass


layer = L()
layer.weight_scale = torch.nn.Parameter(nd.clone(), requires_grad=False)
before = step("snapshot_scales", lambda: slc.snapshot_scales(layer, ("weight_scale",)))
layer.weight_scale.data = layer.weight_scale.data.reshape(64, 80, 2).transpose(0, 1)
step("verify_scale_layout_preserved",
     lambda: slc.verify_scale_layout_preserved(before, layer, ("weight_scale",),
                                               what="l0"))
step("report_scale_bytes",
     lambda: slc.report_scale_bytes(layer, ("weight_scale",), where="l0",
                                    weights=("weight_scale",)))

print()
print("=== VERDICT ===")
print("🔴 FAILURES: " + ", ".join(fails) if fails else
      "✅ nothing in the instrument raised on device, on any format tested")
sys.exit(1 if fails else 0)
