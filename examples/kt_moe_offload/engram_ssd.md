# DeepSeek-V4.1 Engram on SSD

The optional `ssd` backend stores the four Engram weight/scale tables in ordinary
files and maps them read-only. Requested rows use the existing CPU gather, pinned
staging buffers, H2D copies and dequantization path. The default `ram` backend and
`/dev/shm/engram_stage` directory remain unchanged.

## Prepare the tables

Run from the repository root on Linux, with a local safetensors checkpoint and
an empty directory on a disk-backed filesystem:

```bash
export MODEL=/path/to/DeepSeek-V4.1-Flash
export SSD_STAGE=/path/on/ssd/engram

python3 examples/kt_moe_offload/prepare_engram_ssd.py \
  --checkpoint "$MODEL" --output "$SSD_STAGE"
```

This replaces the RAM table staging step in the [deployment guide](README.md).
The four tables require approximately 188.83 GiB for the tested Flash checkpoint.
The tool streams checkpoint byte ranges using an 8 MiB buffer, computes SHA256,
independently reads each destination back, checks checkpoint stability and
publishes read-only files without overwriting existing paths. The manifest is
published last, after every table has passed verification. Interrupted output
remains incomplete and must be inspected before preparing a new directory.

An explicit full verification reads both the tables and checkpoint ranges:

```bash
python3 examples/kt_moe_offload/prepare_engram_ssd.py \
  --checkpoint "$MODEL" --output "$SSD_STAGE" --verify
```

Preparation and full verification warm the filesystem page cache. Keep them
outside performance measurement and record the cache state separately.

## Select the backend

After configuring the other dependencies and environment in the deployment guide:

```bash
export VLLM_ASCEND_DSV41_ENGRAM_BACKEND=ssd
export VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR="$SSD_STAGE"
bash examples/kt_moe_offload/serve_lowlatency.sh
```

Use the path visible inside the serving container and mount the table directory
read-only. `ssd` requires an explicit absolute directory and rejects tmpfs/ramfs.
Identify the physical backing device separately: a disk-backed filesystem alone
does not prove that the files are on an SSD.

The manifest binds table names, shapes, dtypes and file identities to the current
checkpoint index and safetensors headers. Startup validates metadata and the
exact opened descriptor before mapping; it does not hash the table payload.
Replacing or copying files invalidates their recorded identity. Prepare a new
verified artifact directory rather than editing the manifest to accept them.
Read-only file permissions and container mounts protect already-bound mappings
from writers; metadata checks do not make external writes or truncation safe.

The backend is fixed for each embedding instance before graph capture. Changing
the environment does not rebind existing mappings; restart the service to switch
backends. Neither setting enables DSpark or changes graph configuration.

For the legacy backend, unset both new variables. An explicit absolute RAM stage
directory can also be selected with `VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR` while
`VLLM_ASCEND_DSV41_ENGRAM_BACKEND=ram`.

## Interpret measurements

This is ordinary read-only `mmap`, with Linux page cache participating in reads.
It does not use `O_DIRECT`, pin the complete tables or impose a hard page-cache
budget. A startup with zero resident table pages is not evidence that every
subsequent lookup reaches the physical SSD: verification requests and the
original throughput calibration can populate the cache.

Compare backends with the same checkpoint, prompts, output count, concurrency,
graph settings and CPU/NUMA placement. Record table page residency before startup,
before the throughput client and afterward, plus other NPU processes and host
pressure throughout calibration and formal requests. Keep client TPOT and native
decode throughput separate, and retain the original client's SSE text-chunk
length warnings while checking actual generated-token counters independently.
