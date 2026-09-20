#!/usr/bin/env bash
# Driver for kt_cpu_bench.py.  Pure CPU, no NPU, no card lease.
# One process per pool configuration: CPUInfer is a process-level singleton, so
# cpuinfer_threads / threadpool_count / numa_nodes / KT_CPU_BIND can only be
# changed by restarting.
#
#   bash run_cpu_bench.sh <outdir> <phase>
#     phase = sweep | curve | compact
set -u

OUT="${1:?usage: run_cpu_bench.sh <outdir> <phase>}"
PHASE="${2:?usage: run_cpu_bench.sh <outdir> <phase>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/workspace/code/dsv4_single_card/vllm_project
KT_SITE="$ROOT/build_artifacts/moe_offload_ref/kt_site"
mkdir -p "$OUT"

run() {  # run <tag> <extra args...>   (env prefixed by caller via RUNENV)
  local tag="$1"; shift
  echo "=== $tag  $(date -Is)  MemAvail $(awk '/MemAvailable/{print $2/1e6" GB"}' /proc/meminfo)  load $(cut -d' ' -f1 /proc/loadavg)"
  env -u LD_PRELOAD PYTHONPATH="$KT_SITE" ${RUNENV:-} \
    python3 "$HERE/kt_cpu_bench.py" --tag "$tag" --out "$OUT/$tag.json" "$@" \
    >"$OUT/$tag.log" 2>&1
  local rc=$?
  echo "    rc=$rc"
  command grep -E '^(T=|\[load\] total)' "$OUT/$tag.log" || true
  return $rc
}

case "$PHASE" in
sweep)
  # k_cpu=4, T=1 working point (PLAN §1: DSV4.1 E[k_cpu] ~= 4.4).
  # 2 layers x 284 experts = 9.95 GiB resident -> 20x L3, no cache reuse.
  COMMON="--layers 1,3 --n-cpu 284 --kcpu 0,1,4 --tokens 1 --reps 40 --warmup 10"
  for th in 8 16 32 48; do
    RUNENV="" run "tp1n0_th${th}" $COMMON --threads $th --threadpool 1 --numa 0
  done
  RUNENV="" run "tp1n1_th32" $COMMON --threads 32 --threadpool 1 --numa 1
  for th in 8 16 32 48 64 96; do
    RUNENV="" run "tp2_th${th}" $COMMON --threads $th --threadpool 2 --numa 0,1
  done
  ;;
bind)
  COMMON="--layers 1,3 --n-cpu 284 --kcpu 0,1,4 --tokens 1 --reps 40 --warmup 10 --threadpool 2 --numa 0,1"
  for th in 32 64; do
    RUNENV="KT_CPU_BIND=pack"   run "bind_pack_th${th}"   $COMMON --threads $th
    RUNENV="KT_CPU_BIND=stripe" run "bind_stripe_th${th}" $COMMON --threads $th
  done
  ;;
compact)
  # KT_MXFP4_COMPACT_SCALES 1 vs 0: speed A/B *and* bit-exactness of the output.
  COMMON="--layers 1,3 --n-cpu 284 --kcpu 0,1,4 --tokens 1 --reps 40 --warmup 10 --threadpool 2 --numa 0,1 --threads ${TH:-64}"
  RUNENV="KT_MXFP4_COMPACT_SCALES=1" run "compact1" $COMMON --dump-out "$OUT/out_compact1.npy"
  RUNENV="KT_MXFP4_COMPACT_SCALES=0" run "compact0" $COMMON --dump-out "$OUT/out_compact0.npy"
  ;;
curve)
  # The core curve: k_cpu 0..6 x T in {1,4,32}, best pool configuration.
  # 3 layers x 284 experts = 14.9 GiB resident -> 30x L3.
  run "curve" --layers 1,3,5 --n-cpu 284 --kcpu 0,1,2,3,4,5,6 --tokens 1,4,32 \
      --reps 60 --warmup 15 --threads "${TH:-64}" --threadpool "${TP:-2}" --numa "${NUMA:-0,1}"
  ;;
prod)
  # Production-scale recalibration (2026-09-20). The curve above was measured
  # on 3 layers = 14.92 GiB resident; production holds ~199 GiB (40 layers x
  # 284 CPU experts), which is 13x the working set and a different page/TLB
  # and DDR regime. ANALYSIS.txt called its own numbers "optimistic by an
  # unquantified amount"; this phase quantifies it by running the same points
  # against the real footprint, twice, so the two passes bound the noise.
  # Every CPU-side figure downstream (E[T(k)], ms/token, any overlap ceiling)
  # must be rebuilt on this curve, not on the 3-layer one.
  ALL="$(python3 -c 'print(",".join(str(i) for i in range(40)))')"
  avail=$(awk '/MemAvailable/{print int($2/1e6)}' /proc/meminfo)
  echo "MemAvailable ${avail} GB (need >= 150 to start, the bench self-stops below 100)"
  [ "$avail" -ge 150 ] || { echo "refusing to start"; exit 3; }
  for pass in a b; do
    run "prod_${pass}" --layers "$ALL" --n-cpu 284 --kcpu 0,1,2,3,4,5,6 --tokens 1,4,32 \
        --reps "${REPS:-60}" --warmup "${WARMUP:-15}" --threads "${TH:-96}" \
        --threadpool "${TP:-2}" --numa "${NUMA:-0,1}" || exit $?
  done
  ;;
*) echo "unknown phase $PHASE"; exit 2;;
esac
