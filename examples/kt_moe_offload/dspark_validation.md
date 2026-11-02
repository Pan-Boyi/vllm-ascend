# Single-card DSpark validation and CPU kernel dependency

The DSpark runtime is in
[vllm-ascend PR #2](https://github.com/wenxuewuhd/vllm-ascend/pull/2), targeting
`dsv41-moe-offload-pr` at `59ce1e18a4f93c20a872e4450bd14e891c556cb7`.
Runtime validation used vllm-ascend `8147c5f4e131939878ff30086f69d5d885142431`
and vLLM `a9686bc6e6809a49cb3e3c592a337d0721c3619f`, CANN 9.2.0-beta.2,
single-card text generation, bf16 KV, target graphs, and eager draft execution.
On 2026-11-02, all 4,025 tracked files in the later remote runtime snapshot
matched the pre-documentation PR head. Diagnostic instrumentation and benchmark
controllers are separate from the runtime changes.

## Adapter scope and current verification

The Ascend drafter inherits upstream's forward and checkpoint loader. Its
constructor binds only three hardware entry points; loading the target model
alone does not bind the drafter. The mHC collapse and context-KV insert use the
target's existing helpers. Context insertion rotates and writes KV directly,
without allocating or rotating a discarded dummy query. Expert placement is
enforced at the existing offload boundary rather than scanned again after
model construction.

The 2026-11-02 simplification passed 59 isolated CPU tests with real PyTorch
on macOS, including context-insert equivalence, padding, logical/physical KV
indices, persistent buffers, proposer input preparation, drafter-only binding,
and draft/target offload boundaries. Six checkpoint-dependent tests were
skipped because the model index is absent. These checks stub vLLM/hardware
imports; they do not replace full vLLM integration or native NPU validation.
The historical results below predate this simplification. A fresh NPU run of
the simplified adapter is still pending.

## CPU expert weight reuse

The companion change is
[ktransformers-AK PR #1](https://github.com/wenxuewuhd/ktransformers-AK/pull/1), targeting
[`wenxuewuhd/ktransformers-AK:dsv41-singlecard-offload`](https://github.com/wenxuewuhd/ktransformers-AK/tree/dsv41-singlecard-offload),
from `Pan-Boyi/ktransformers:codex/dspark-expert-weight-reuse`.
The implementation is in `kt-kernel/operators/amx/fp4-moe.hpp`, rather than in
vllm-ascend. It adds a templated GEMV for M=2..6 and dispatches those sizes to
it; each weight group is loaded and decoded outside the token loop. M=1 and
M>6 retain the existing paths. Each token keeps its original K-group
accumulation order. M counts tokens routed to one expert, not the configured K.

Build and install the companion branch using step 6 of the README. Set
`KT_SITE` to that installation and verify the imported `kt_kernel` extension
comes from it. No additional runtime switch is required. The existing kernel
remains sufficient for functional DSpark execution; this dependency supplies
the CPU optimization.

The original 2026-10-29 measurement used a newer upstream kt base,
`105359b4bf70073af9fab992ad29920666acccc9`, with the offload changes applied.
The submitted kernel patch is ported onto the recipe's existing
`dsv41-singlecard-offload` branch at `32c0e52196ab04407791bf487c389c1af2c8a04e`.
The measured baseline header also has a pre-existing AVX512BF16 guard that the
target branch lacks. The optimization does not change that guard. Historical
results therefore describe the measured build, not a fresh binary built from
the companion PR head. The companion PR includes the regression source and
build command; a native rerun of the packaged submission remains pending.

## Correctness results

- The DSpark attention-index fix passed 280 targeted unit tests and one native
  A5 attention regression. Shuffled KV pages expose the former double mapping
  of physical slots; request-relative logical indices match the CPU reference.
- The original CPU weight-reuse kernel passed 483 cases covering M=1..8,
  real projection dimensions, short N, empty workers, multiple scale layouts,
  strided zero-copy weights, and serial/threaded execution. All 2,086,560 float
  outputs were finite and bitwise identical to the original GEMV. An additional
  run with multiple allowed CPUs also passed.
- In the same-K baseline-before / optimized / baseline-after service check,
  six prompts produced identical IDs at all 1,536 output positions and equal
  native acceptance counters.
- Changing K or switching DSpark on/off did not preserve all generated token
  sequences in the later samples. The same-K kernel comparison does not prove
  equivalence across speculative configurations.

## Asynchronous scheduling and K2/K5

On 2026-10-29, runtime configuration checks confirmed
`async_scheduling=True` when neither scheduler flag was supplied. The target
used graphs and the draft used eager execution with all 128 experts in each
draft layer in HBM. K5 and K2 were both exercised with concurrency one.

K2 requires a target verification graph for three tokens; K5 requires six.
`[1,2,3,6]` retains both within Ascend's four-size limit. The earlier K5-only
run used `[1,2,4,6]`. A five-entry list can drop size three during platform
normalization, so it is not a working substitute for the K2 configuration.

## Measured performance and limits

For the CPU optimization, the 2026-10-29 same-K run measured:

| Input tokens | Combined baseline decode TPS | Optimized decode TPS | Change |
| --- | ---: | ---: | ---: |
| 1024 | 32.539 | 32.648 | +0.34% |
| 4096 | 30.554 | 32.271 | +5.62% |

The 4096-token before/after baselines moved from 28.505 to 32.921 TPS, and the
optimized run was below the after baseline. End-to-end output TPS decreased
by 5.54% and 3.95%, respectively. The warm-cache single-thread microkernel
improved about 1.76–3.89x for M=2..6, but the service comparison does not show
a stable whole-model gain. DRAM traffic was not measured.

The optimized-kernel LongBench samples used the first five qualifying source
records per dataset, a 4096-token body plus the official instruction/chat
template, temperature zero, seed zero, natural EOS, and at most 256 output
tokens. Timing excludes the first emitted token batch.

| Dataset | K | Accepted / proposed | Mean acceptance length | Decode TPS | TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MultiNews | 5 | 29.10% | 2.455 | 32.13 | 31.12 |
| MultiNews | 2 | 54.40% | 2.088 | 32.13 | 31.12 |
| VCSUM | 5 | 25.71% | 2.285 | 33.52 | 29.84 |
| VCSUM | 2 | 47.94% | 1.959 | 31.76 | 31.49 |

Acceptance is A/T, where A is the native accepted-draft count and T is the
proposed-draft count. Mean acceptance length is `1 + A/R`, where R counts
verification rounds. The additional target token per round is not included
in A/T. Counters can include a final step truncated by EOS or the output cap.

These are small sequential samples on a shared host. Each K2/K5 pair had zero
of five identical full output sequences. There was no OFF control in this
LongBench experiment, so it establishes neither a DSpark acceleration ratio
nor a complete LongBench quality score. A later controlled experiment did not
complete its final OFF bracket and supplies no replacement controlled speedup.
Multi-card and multimodal end-to-end generation remain unverified.
