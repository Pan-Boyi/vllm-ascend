# Single-card DeepSeek V4.1 Engram on SSD

The optional `ssd` backend uses verified, read-only mappings of the four native
MXFP8/E8M0 Engram tables in the DeepSeek V4.1 Flash checkpoint. It preserves the
mainline model, attention, hash calculation, encoded-row staging and device
dequantization. The complete table is never allocated in HBM or pinned host RAM.
The default backend remains `uva`.

## Prepare and serve

Use the dependencies and expert-offload configuration in [the deployment
guide](README.md). Prepare an empty directory on SSD:

```bash
python3 examples/kt_moe_offload/prepare_engram_ssd.py \
  --checkpoint /models/DeepSeek-V4.1-Flash --output /ssd/engram

# Optional full verification; reads all source and destination table bytes.
python3 examples/kt_moe_offload/prepare_engram_ssd.py \
  --checkpoint /models/DeepSeek-V4.1-Flash --output /ssd/engram --verify
```

Preparation streams with an 8 MiB buffer, computes SHA256, verifies independent
destination readback, publishes read-only files without replacing existing
paths, and publishes the manifest last. The tested Flash tables require about
188.83 GiB of additional disk space. Keep sufficient disk and host-memory
headroom. Interrupted output must be inspected; preparation never silently
overwrites incomplete or complete artifacts. The manifest format is compatible
with the reference single-card SSD preparation at `b0075aa7`.

Mount the checkpoint and prepared directory read-only inside the serving
container. Merge these fields into the existing expert-offload additional config:

```json
{
  "engram_host_backend": "ssd",
  "engram_ssd_path": "/ssd/engram",
  "multistream_engram_overlap": false
}
```

Keep `--engram-config '{"cpu_offload":true}'`, TP=DP=PCP=1 and Model Runner V2.
Start with eager execution and a small `--max-num-batched-tokens`, then compare
`FULL_DECODE_ONLY` with capture size `[1]` under the same workload. Hashing,
SSD row gathering and row transfers run outside graph capture and refresh the
fixed output buffers before every replay. Engram side-stream overlap, PIECEWISE
graphs, RFork loading and conversion to INT8/BF16 are unsupported in SSD mode.

Startup checks the current checkpoint index/headers and the exact opened table
descriptors against the manifest without scanning all table payloads. Writable,
replaced, truncated or symlink-substituted tables are rejected. Source files
must remain unchanged while the service runs. SSD mode rejects tmpfs and ramfs;
also record the actual block-device backing, since a disk filesystem alone does
not establish that the device is an SSD.

## Memory and measurement

This is ordinary read-only `mmap`. Linux page cache participates; there is no
`O_DIRECT`, full-table pinning, hard page-cache limit or asynchronous SSD
prefetch. The bounded per-step buffers do not bound the filesystem page cache.
Recalculate loading, per-NUMA, cgroup and runtime reserves separately from the
full-table RAM backend. CPU expert weights still occupy host memory.

Preparation and full verification warm the cache. Record table page residency
before startup and before/after requests, process disk reads and major faults,
host/cgroup memory peaks, hardware/software identities and the exact launch
arguments. Preserve generated text, prompt/output token counts and finish
reasons. Compare eager and graph output token IDs, and report TTFT, client TPOT
and output throughput separately. Cache hits alone do not establish physical
SSD throughput. Do not drop global caches to manufacture a cold measurement.
