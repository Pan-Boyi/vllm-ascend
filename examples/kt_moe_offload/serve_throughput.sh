#!/usr/bin/env bash
# DeepSeek-V4.1-Flash, one Ascend card — THROUGHPUT profile.
#
# Streaming prefill and dynamic resident experts OFF, batch of 8. Those two are
# what buy time-to-first-token on long prompts; with short prompts and many of
# them they do nothing (prefill streaming only engages above 512 tokens), so
# this profile drops them and keeps the weight zero-copy.
#
#   MODEL=... CANN=... VLLM_TREE=... ASCEND_TREE=... DS41_SITE=... KT_SITE=... \
#     bash serve_throughput.sh
#
# Measured on this configuration (one card):
#   GSM8K   1319 items, 1276/1319 (96.74%) with zero-copy off, 1278/1319 (96.89%)
#           with it on; 10 paired flips, under the 11-flip floor this suite shows
#           across plain restarts => indistinguishable. Both inside the documented
#           acceptance range. (measured without the SWA block override)
#   GPQA    Diamond 198 items, 76.26%, zero truncations and zero errors, against
#           a reference of 76.26% +/- 2.20 on the same harness and sampling.
#           (measured with the SWA block override on, max_model_len 32768)
#   Throughput  96.3 tok/s at 8 concurrent, server saturated (requests queueing).
#
# 🔵 Higher batch gives more: 132.0 tok/s at 16 and 160.5 tok/s at 32 were
#    measured on a synthetic fixed load, still rising at 32. Those settings have
#    NO accuracy validation -- 8 is what the numbers above were taken at.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MAXLEN=${MAXLEN:-32768}
MAXSEQS=${MAXSEQS:-8}
MAXBATCH=${MAXBATCH:-4096}

# The sliding-window KV groups are built at block size 32; 128 is what Ascend's
# own V4.0 path already uses on this same backend. Measured here, with
# max_model_len pinned and only this changed: 171,988 -> 615,442 KV tokens,
# concurrency 5.25x -> 18.78x (3.58x). Unset or 0 leaves the cache byte-identical.
# It is fail-closed: a value that is not verbatim among the backend's supported
# kernel block sizes, or larger than the main cache's, refuses to start.
export VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE=${VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE:-128}

KT_CONFIG=${KT_CONFIG:-$(cat <<EOF
{"kt_offload_config": {
  "enabled": true,
  "resident_strategy": "prefix",
  "num_resident_experts": $RESIDENT,
  "cpuinfer_threads": 96,
  "threadpool_count": 2,
  "numa_nodes": [0, 1],
  "packed_transfer": false,
  "stream_prefill": false,
  "dynamic_resident": false,
  "zerocopy_w13": true,
  "zerocopy_node_fraction": [1.0, 1.0],
  "zerocopy_streaming_node": 1
}}
EOF
)}

echo "  profile=throughput  card=$CARD port=$PORT max_model_len=$MAXLEN max_num_seqs=$MAXSEQS swa_block=$VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE"
exec vllm serve "$MODEL" "${COMMON_ARGS[@]}" \
  --max-model-len "$MAXLEN" \
  --max-num-batched-tokens "$MAXBATCH" \
  --max-num-seqs "$MAXSEQS" \
  --additional-config "$KT_CONFIG" \
  ${EXTRA_ARGS:-} \
  --port "$PORT"
