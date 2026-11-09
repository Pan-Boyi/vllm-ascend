# Single-card DeepSeek V4.1 on Ascend

This adapter runs routed experts across one NPU and the host CPU using
`kt-kernel`. The configured resident experts stay in HBM; the remaining
experts execute on the CPU. The feature is disabled by default.
While enabled, all execution paths retain FP32 router probabilities.

The implementation is based on upstream vLLM Ascend, with the expert-offload
runtime adapted from `wenxuewuhd/vllm-ascend` at
`59ce1e18a4f93c20a872e4450bd14e891c556cb7`. It uses the mainline V4.1 model,
attention, cache and Engram implementations. Engram stays in host RAM through
the existing `--engram-config '{"cpu_offload":true}'` option. This port adds
neither DSpark execution nor an Engram SSD store. Speculative decoding is
rejected while expert offload is enabled.

## Environment

- Ascend 950PR with 128 GiB HBM and an x86 host suitable for the selected
  kt-kernel CPU backend. Do not infer support for another host architecture
  from NPU compatibility alone.
- Build vLLM Ascend normally. Use the upstream vLLM revision recorded in
  `.github/vllm-main-verified.commit`, and the CANN, torch-npu and Triton
  versions required by this checkout.
- Install a kt-kernel build exposing `KTMoEWrapper` with MXFP4 expert loading,
  Ascend pinned buffers and the `swiglu_limit` argument. Optional zero-copy and
  dynamic placement require the corresponding kt-kernel capabilities; the
  adapter validates them when requested.
- Use the original MXFP checkpoint. The default resident count is 100 experts
  per target layer. Recalculate HBM and per-NUMA host-memory requirements
  before changing it. The original checkpoint contains approximately 189 GiB
  of Engram tables, in addition to CPU expert weights and loading workspace.

The reference CPU kernel predates the excluded DSpark work:
[`wenxuewuhd/ktransformers-AK` at `32c0e521`](https://github.com/wenxuewuhd/ktransformers-AK/tree/32c0e52196ab04407791bf487c389c1af2c8a04e).
Build its `kt-kernel` package following that revision's instructions, and
install the resulting wheel with `--no-deps` in the selected serving
environment. Optional CPU weight-reuse changes from later DSpark work are not
required by this adapter.

## Start the service

Check device processes and host memory before selecting an idle card:

```bash
npu-smi info
free -h

export MODEL=/path/to/DeepSeek-V4.1-Flash
export CARD=1
export PORT=18141
bash examples/kt_moe_offload/serve.sh
```

The script uses TP=DP=PP=1, Model Runner V2, native packed A5 caches, target
decode graphs, a maximum model length of 8192 and one sequence at a time.
The CPU expert dependency is imported only when offload is enabled.

`KT_CONFIG` can supply a complete `--additional-config` JSON object. For
example, a kernel with zero-copy support can avoid a private copy of the
CPU gate/up weights:

```bash
export KT_CONFIG='{
  "enable_fused_mc2": 0,
  "multistream_engram_overlap": false,
  "kt_offload_config": {
    "enabled": true,
    "num_resident_experts": 100,
    "cpuinfer_threads": 96,
    "threadpool_count": 2,
    "numa_nodes": [0, 1],
    "zerocopy_w13": true
  }
}'
bash examples/kt_moe_offload/serve.sh
```

Streaming prefill additionally accepts `stream_prefill: true` and
`stream_prefill_chunk_experts: 32`. The provider allocates its shared staging
slot before KV profiling. `dynamic_resident: true` requires streaming,
`max_num_seqs=1`, and a kernel that loads the complete expert set with
`KT_FULLSET_LOAD=1`. Keep these options disabled during the first correctness
run, then compare each option with the same prompts and environment.

## Validate

```bash
curl --fail http://127.0.0.1:18141/health
curl --fail http://127.0.0.1:18141/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41-singlecard","messages":[{"role":"user","content":"What is 17 + 25?"}],"temperature":0,"max_tokens":128}'

TORCH_DEVICE_BACKEND_AUTOLOAD=0 pytest -q tests/ut/kt_offload \
  tests/ut/ops/test_kt_offload_main.py

python examples/kt_moe_offload/smoke_test.py --output /tmp/dsv41-smoke.json
```

Record the exact source and dependency revisions, launch command, device,
memory peak, generated outputs and request timings. The historical performance
numbers from the reference fork do not validate this mainline port.
