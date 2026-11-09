# Mainline single-card port validation

## Revisions and scope

- vLLM Ascend base: `3725f6b36fd18aff3c15bcea6843bbe5133675cf`.
- vLLM: `24420b7a341af37a9bc3186e6fc6062512e0aa58`, matching the base's
  `.github/vllm-main-verified.commit`.
- Expert-offload reference: `59ce1e18a4f93c20a872e4450bd14e891c556cb7`.
- The port retains the native mainline V4.1 model, attention, packed A5 cache
  and Engram host-memory implementation. It adds no DSpark or Engram SSD
  execution path.

## CPU validation

The tests used Python 3.12.13, PyTorch 2.10.0 CPU, the pinned upstream vLLM
source, Transformers 5.19.0, Tokenizers 0.23.2 and Hugging Face Hub 1.31.0.
The container exposed no NPU devices. Explicit CPU platform metadata was used
for tests that construct a default `VllmConfig`.

| Coverage | Result |
| --- | --- |
| `tests/ut/kt_offload` | 1037 passed, 20 skipped |
| Complete configuration, MoE and dispatcher regression tests | 402 passed, 11 skipped |
| Total executed CPU tests | 1439 passed, 31 skipped |

The skipped tests require NUMA syscalls unavailable under the container's
seccomp policy, optional real-checkpoint fixtures or other hardware paths.
Mock-based NUMA and
placement tests passed. No skipped test is counted as hardware evidence.

The MoE tests include compact resident weight loading, CPU-route exclusion,
empty input, immediate/deferred output merging, streamed-prefill execution,
shared-expert milestones, and the disabled feature path. Configuration tests
cover nested coercion, factory validation, defaults and unsupported settings.
Streaming tests verify raw E8M0 scale bytes while preserving mainline parameter
dtypes. Worker tests verify current scheduler metadata reaches both runner
selections before model dispatch.

Zero-copy startup tests use the real NUMA fraction helper and derive its memory
requirement from the exact checkpoint tensors selected for placement, including
resident experts. That placement budget does not cover later Engram or CPU
kernel allocations; a separate per-node host-memory check is still required.

`bash format.sh ci` passed, including Ruff, spelling, Markdown, shell,
workflow, secret and repository policy checks. A separate CPU-only wheel
build confirmed that all 23 runtime Python modules and `poll_native.c` are
packaged; the C source bytes match the checkout. This packaging artifact was
not installed as an NPU build.

## Checkpoint inspection

Read-only inspection of the original MXFP checkpoint found compatible model
geometry and native mainline loader mappings. The model has 40 layers, hidden
size 5120, 384 routed experts, top-k 6 and intermediate size 2304. Both Engram
prime-bucket sums match the stored tables (384006168 and 384016682 rows).
Mainline tokenizer normalization produces the configured compressed vocabulary
size of 99092 from the original vocabulary of 129280.

This confirms configuration and storage compatibility, not successful tensor
loading or model execution. The launch command explicitly selects
`deepseek_v4_fp8` quantization and Engram CPU offload.

## NPU validation status

Full model loading, generation accuracy, eager/graph parity, streaming parity,
HBM and host-memory peaks, and performance remain unverified on this port.
They require an unreserved healthy card and a mainline-compatible CANN 9.2.1 /
torch-npu 2.10.0.post7 environment. CPU tests and the reference fork's
historical results do not establish those outcomes.
