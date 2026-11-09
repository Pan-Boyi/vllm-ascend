#!/usr/bin/env bash
# Run from a main-compatible vLLM Ascend environment with kt-kernel installed.
set -euo pipefail

: "${MODEL:?Set MODEL to the original DeepSeek-V4.1-Flash MXFP checkpoint}"
: "${CARD:?Set CARD to an idle physical NPU ID}"

PORT=${PORT:-18141}
RESIDENT=${RESIDENT:-100}
CPU_THREADS=${CPU_THREADS:-96}
MAXLEN=${MAXLEN:-8192}
MAXBATCH=${MAXBATCH:-4096}
MAXSEQS=${MAXSEQS:-1}

export ASCEND_RT_VISIBLE_DEVICES="$CARD"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-10}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}

# Start with fixed resident experts. Streaming, dynamic placement and zero-copy
# are opt-in, after validating the CPU kernel and the host memory budget.
KT_CONFIG=${KT_CONFIG:-$(python3 - "$RESIDENT" "$CPU_THREADS" <<'PY'
import json
import sys

print(json.dumps({
    "enable_fused_mc2": 0,
    "multistream_engram_overlap": False,
    "kt_offload_config": {
        "enabled": True,
        "num_resident_experts": int(sys.argv[1]),
        "cpuinfer_threads": int(sys.argv[2]),
        "threadpool_count": 2,
        "numa_nodes": [0, 1],
    },
}))
PY
)}

exec vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --served-model-name dsv41-singlecard \
    --tensor-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len "$MAXLEN" \
    --max-num-batched-tokens "$MAXBATCH" \
    --max-num-seqs "$MAXSEQS" \
    --gpu-memory-utilization "${GPUUTIL:-0.90}" \
    --block-size 128 \
    --quantization deepseek_v4_fp8 \
    --engram-config '{"cpu_offload":true}' \
    --tokenizer-mode deepseek_v41 \
    --reasoning-parser deepseek_v41 \
    --tool-call-parser deepseek_v41 \
    --enable-auto-tool-choice \
    --safetensors-load-strategy lazy \
    --no-enable-prefix-caching \
    --no-swa-bounded-replay \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
    --additional-config "$KT_CONFIG" \
    "$@"
