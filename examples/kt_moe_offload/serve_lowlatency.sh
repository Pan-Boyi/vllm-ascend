#!/usr/bin/env bash
# DeepSeek-V4.1-Flash, one Ascend card — LOW LATENCY profile.
#
# Streaming prefill and dynamic resident experts ON, one sequence at a time.
# This is the configuration the time-to-first-token result was measured on.
#
#   MODEL=... CANN=... VLLM_TREE=... ASCEND_TREE=... DS41_SITE=... KT_SITE=... \
#     bash serve_lowlatency.sh
#
# Measured on this configuration (one card, DeepSeek-V4.1-Flash, MXFP4):
#   TTFT      78.8 s -> 21.9 s  (3.60x), three interleaved paired rounds,
#                                all three differences the same sign
#   TPOT      22.6 ms at 4096 in / 128 out, 23.0 ms at 1024 in / 256 out
#             (5 and 3 requests at concurrency 1, spread 0.22 and 0.15 ms).
#             Judged indistinguishable between the copy and zero-copy paths --
#             zero-copy buys TTFT, not TPOT.
#
#   🔴 TPOT is 3.7x worse under --enforce-eager (85-98 ms measured). Every
#      number in this file, and every accuracy result quoted below, was taken
#      with graphs captured. _common.sh sets that up; do not add --enforce-eager
#      back to compare against anything here.
#
#   TTFT      barely moves with prompt length -- 19.8 s at 1024 tokens, 21.4 s at
#             4096, 21.9 s at 8184. Streaming prefill passes the expert weights
#             once per prefill, and that fixed cost dominates. Do not extrapolate
#             TTFT from tokens.
#   Accuracy  token ids bit-identical to the copy path over 30 prompts, with a
#             negative control that goes fully red; 20 LongBench-v2 prompts of
#             8184+ tokens produced byte-identical text
#
#   SWA block  turning the override on changes nothing the model emits: with
#              streaming prefill and dynamic resident experts both on, and only
#              VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE differing (0 vs 128), the same
#              30 prompts produced bit-identical token ids, 30/30.
#
# Start-up is ~400 s: the weights are faulted in and placed per NUMA node once.
# A run that looks stuck at "loading weights" is usually still doing that. The
# first launch also spends a few minutes compiling the ds41 operators, which the
# engine builds on first use; _common.sh gives this tree its own cache directory
# so later launches reuse them.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MAXLEN=${MAXLEN:-8192}
MAXSEQS=${MAXSEQS:-1}
MAXBATCH=${MAXBATCH:-4096}

# Same SWA block override as the throughput profile: 171,988 -> 615,442 KV tokens
# for the same 27.39 GiB pool (3.58x), measured with max_model_len pinned and
# only this changed. Fail-closed -- a value not verbatim among the backend's
# supported kernel block sizes, or larger than the main cache's, refuses to start.
#
# Measured at this profile's max_model_len of 8192: 44,867 -> 172,657 KV tokens,
# concurrency 5.48x -> 21.08x (3.85x).
#
# 🔴 At 8192 with one sequence that buys nothing you can measure -- the pool is
#    far larger than one request needs either way. What it buys is headroom to
#    RAISE max_model_len. The gain itself depends on max_model_len: 3.85x here,
#    3.58x at 32768, so do not carry either number to a third setting.
export VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE=${VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE:-128}

# zerocopy_node_fraction is pinned rather than derived. Deriving it reads the
# nodes' free memory at start-up, which drifts by tens of GiB within one night
# on a shared host -- two launches then run different placements and are no
# longer comparable. [1.0, 1.0] is full locality: each TP part keeps its own
# bytes on its own node.
KT_CONFIG=${KT_CONFIG:-$(cat <<EOF
{"kt_offload_config": {
  "enabled": true,
  "resident_strategy": "prefix",
  "num_resident_experts": $RESIDENT,
  "cpuinfer_threads": 96,
  "threadpool_count": 2,
  "numa_nodes": [0, 1],
  "packed_transfer": false,
  "stream_prefill": true,
  "dynamic_resident": true,
  "zerocopy_w13": true,
  "zerocopy_node_fraction": [1.0, 1.0],
  "zerocopy_streaming_node": 1
}}
EOF
)}

echo "  profile=low-latency  card=$CARD port=$PORT max_model_len=$MAXLEN max_num_seqs=$MAXSEQS"
exec vllm serve "$MODEL" "${COMMON_ARGS[@]}" \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens "$MAXBATCH" \
  --max-num-seqs "$MAXSEQS" \
  --additional-config "$KT_CONFIG" \
  ${EXTRA_ARGS:-} \
  --port "$PORT"
