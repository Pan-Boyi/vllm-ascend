"""E-ENG-1: is CANN's ElasticBuffer engram path byte-transparent?

The engram tables are 189 GiB of fp8 and will not fit on a card, so they live in
host memory and get gathered per row.  CANN ships an operator for exactly that
shape of problem -- ``ElasticBuffer.engram_write`` / ``engram_fetch`` -- but its
host storage accepts only bfloat16/float16/float32.  Storing bf16 instead would
double the table to 393 GB of *pinned*, non-reclaimable memory on a machine
shared with three other workstreams, which is not a thing we can do.

The proposal (AD2) is to hand the operator our fp8 bytes wearing a bf16 costume:
a ``[N, 256]`` fp8 table viewed as ``[N, 128]`` bf16 has byte-identical layout,
passes the dtype check, and -- if the operator only moves bytes -- comes back
unchanged, to be dequantized on device against scales that stay in HBM.

Reading the operator's own source says it should work:

    elastic_buffer.py     engram_write: torch._check(storage.dtype in
                          (bfloat16, float16, float32)) -- we pass bf16, fine.
                          Also, undocumented: size(1) % 128 == 0. 128 is.
    elastic_buffer.cpp    EngramWrite: memcpy_s over a `const uint8_t*`.
                          engramDtype_ is assigned and never read again.
    elastic_buffer.cpp    EngramFetch: dtypeEnum only allocates the output
                          tensor; it is NOT among aclnnEngramFetch's arguments.

What none of that covers is the AIV binary behind ``aclnnEngramFetch``.  Hence
this test, which checks the one thing source review cannot: that bytes survive
the round trip -- including bit patterns that are NaN, Inf or denormal *when read
as bf16*, since a dtype-aware normalisation anywhere in the path would corrupt
exactly those and nothing else.

Comparison is on bytes, never on values.  NaN != NaN, so a value comparison would
report success precisely where the interesting failure lives.

Three things must hold, and any one failing sinks AD2:

  1. Round-tripped bytes are identical, hostile patterns included.
  2. It holds for indices on a *remote* rank as well as the local one -- those
     are different code paths (local memcpy vs RDMA read, ElasticBuffer.md:196).
  3. Out-of-range indices still return zeros rather than raising
     (ElasticBuffer.md:230) -- our own range check has to be built on top of that
     behaviour, and nothing in _check_engram_fetch_ready validates the range.

    ASCEND_RT_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 e_eng_1_bit_fidelity.py
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist

NUM_ENTRIES = 4096          # per rank; small, this is a fidelity test not a benchmark
FP8_HIDDEN = 256            # engram_head_dim
BF16_HIDDEN = FP8_HIDDEN // 2   # the costume: 128 bf16 == 256 bytes == one fp8 row
ROW_BYTES = FP8_HIDDEN

# 16-bit patterns that are pathological *read as bf16*, little-endian byte pairs.
# If anything in the path is dtype-aware -- a normalisation, a NaN canonicalise, a
# denormal flush -- these are where it shows, and only here.
HOSTILE_U16 = [
    0x7F80,  # +Inf
    0xFF80,  # -Inf
    0x7FC0,  # quiet NaN
    0xFFC0,  # negative quiet NaN
    0x7FA0,  # signalling NaN
    0x7FFF,  # NaN, all mantissa bits set
    0x8000,  # negative zero
    0x0001,  # smallest denormal
    0x007F,  # largest denormal
    0x8001,  # negative denormal
    0x0080,  # smallest normal
    0x7F7F,  # largest finite
    0xFF7F,  # most negative finite
    0x0000,  # zero
    0xAAAA,  # alternating bits
    0x5555,  # alternating bits, inverted
]


def build_storage_bytes(rank: int) -> torch.Tensor:
    """A [NUM_ENTRIES, ROW_BYTES] uint8 table whose every byte is predictable.

    Row 0 is all 256 byte values, so no value goes untested.  Row 1 is the hostile
    bf16 patterns.  The rest encode (rank, row, col) so a wrong row or a
    wrong rank's row is identifiable rather than merely unequal.
    """
    t = torch.empty((NUM_ENTRIES, ROW_BYTES), dtype=torch.uint8)

    t[0] = torch.arange(256, dtype=torch.uint8)

    pair = torch.tensor(
        [[p & 0xFF, (p >> 8) & 0xFF] for p in HOSTILE_U16], dtype=torch.uint8
    ).reshape(-1)                                    # 32 bytes
    t[1] = pair.repeat(ROW_BYTES // pair.numel())

    rows = torch.arange(2, NUM_ENTRIES, dtype=torch.int64)
    cols = torch.arange(ROW_BYTES, dtype=torch.int64)
    # rank in the high nibble, row and col mixed into the rest
    t[2:] = (((rank & 0xF) << 4) ^ (rows[:, None] * 31 + cols[None, :] * 17)).to(
        torch.uint8
    )
    return t


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    import torch_npu  # noqa: F401  registers the npu backend
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", rank=rank, world_size=world)
    group = dist.group.WORLD

    from cann_ops_transformer.ops import ElasticBuffer

    def log(*a):
        if rank == 0:
            print(*a, flush=True)

    log(f"world_size={world}  ASCEND_RT_VISIBLE_DEVICES="
        f"{os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")
    log(f"per-rank entries={NUM_ENTRIES}  fp8 row={FP8_HIDDEN}B  "
        f"declared as bf16 hidden={BF16_HIDDEN}")
    log(f"128-alignment check: {BF16_HIDDEN} % 128 == {BF16_HIDDEN % 128}  "
        f"(undocumented constraint in elastic_buffer.py)")

    # -- the costume -------------------------------------------------------
    src_bytes = build_storage_bytes(rank)                    # [N, 256] uint8
    storage = src_bytes.view(torch.bfloat16)                 # [N, 128] bf16, no copy
    assert storage.shape == (NUM_ENTRIES, BF16_HIDDEN), storage.shape
    assert storage.dtype == torch.bfloat16
    assert storage.is_contiguous() and storage.is_cpu
    n_nan = int(torch.isnan(storage).sum())
    n_inf = int(torch.isinf(storage).sum())
    log(f"storage carries {n_nan} NaN and {n_inf} Inf bf16 values -- deliberately: "
        f"a dtype-aware path corrupts these and only these")

    num_cpu_bytes = ElasticBuffer.get_engram_storage_size_hint(
        NUM_ENTRIES, BF16_HIDDEN, torch.bfloat16
    )
    log(f"num_cpu_bytes hint = {num_cpu_bytes} "
        f"({num_cpu_bytes / 2**20:.1f} MiB, 2MB-aligned)")

    eb = ElasticBuffer(group, num_cpu_bytes=num_cpu_bytes)
    eb.engram_write(storage)
    log("engram_write OK")

    # -- indices: local rows, remote rows, and out-of-range ----------------
    # Global index space is [0, world*NUM_ENTRIES); rank r owns [r*N, (r+1)*N).
    interesting = [0, 1, 2, 3, NUM_ENTRIES - 1]              # incl. the two special rows
    local = [rank * NUM_ENTRIES + i for i in interesting]
    remote = [((rank + 1) % world) * NUM_ENTRIES + i for i in interesting]
    oob = [world * NUM_ENTRIES, world * NUM_ENTRIES + 12345]
    idx_list = local + remote + oob
    idx = torch.tensor(idx_list, dtype=torch.int32).npu()

    fetched = eb.engram_fetch(idx)()
    assert fetched.shape == (len(idx_list), BF16_HIDDEN), fetched.shape
    assert fetched.dtype == torch.bfloat16, fetched.dtype
    got = fetched.cpu().view(torch.uint8)                    # [n, 256] -- BYTES
    log(f"engram_fetch OK -> {tuple(fetched.shape)} {fetched.dtype} "
        f"(as bytes {tuple(got.shape)})")

    # every rank's expected table, recomputed locally
    tables = {r: build_storage_bytes(r) for r in range(world)}

    results, failures = [], []
    for k, gi in enumerate(idx_list):
        kind = "oob" if gi >= world * NUM_ENTRIES else (
            "local" if gi // NUM_ENTRIES == rank else "remote")
        if kind == "oob":
            ok = bool((got[k] == 0).all())
            detail = "all zeros" if ok else f"nonzero: {got[k][:8].tolist()}"
        else:
            want = tables[gi // NUM_ENTRIES][gi % NUM_ENTRIES]
            ok = bool(torch.equal(got[k], want))
            if ok:
                detail = "byte-identical"
            else:
                diff = (got[k] != want).nonzero().flatten()
                detail = (f"{diff.numel()}/{ROW_BYTES} bytes differ at "
                          f"{diff[:8].tolist()}; got {got[k][diff[:4]].tolist()} "
                          f"want {want[diff[:4]].tolist()}")
        results.append({"global_index": gi, "kind": kind, "ok": ok, "detail": detail})
        if not ok:
            failures.append((gi, kind, detail))

    n_local = sum(r["kind"] == "local" and r["ok"] for r in results)
    n_remote = sum(r["kind"] == "remote" and r["ok"] for r in results)
    n_oob = sum(r["kind"] == "oob" and r["ok"] for r in results)
    tot = {k: sum(r["kind"] == k for r in results) for k in ("local", "remote", "oob")}

    # Row 1 is the hostile row; call it out separately, it is the whole point.
    hostile = [r for r in results if r["global_index"] % NUM_ENTRIES == 1]

    if rank == 0:
        print()
        for r in results:
            flag = "ok  " if r["ok"] else "FAIL"
            print(f"  {flag} idx {r['global_index']:>6} {r['kind']:>6}  {r['detail']}")
        print()
        print(f"local  rows byte-identical : {n_local}/{tot['local']}")
        print(f"remote rows byte-identical : {n_remote}/{tot['remote']}   "
              f"(RDMA path, different code from local)")
        print(f"out-of-range -> zeros      : {n_oob}/{tot['oob']}")
        print(f"hostile-pattern row        : "
              f"{sum(r['ok'] for r in hostile)}/{len(hostile)}  "
              f"<-- NaN/Inf/denormal bit patterns")
        verdict = not failures
        print()
        print(f"E-ENG-1 byte fidelity: {'PASS' if verdict else 'FAIL'}")
        if verdict:
            print("=> ElasticBuffer's engram path is byte-transparent; AD2 (fp8 bytes in a")
            print("   bf16 costume) is viable. 203 GB pinned instead of 393 GB, and the")
            print("   official AIV gather kernel does the work -- no Triton needed.")
        else:
            print("=> AD2 does not hold. Something in the path interprets the dtype.")
        out = os.environ.get("E_ENG_1_JSON")
        if out:
            with open(out, "w") as f:
                json.dump({"world_size": world, "num_entries": NUM_ENTRIES,
                           "fp8_hidden": FP8_HIDDEN, "bf16_hidden": BF16_HIDDEN,
                           "num_cpu_bytes": num_cpu_bytes,
                           "n_nan_in_storage": n_nan, "n_inf_in_storage": n_inf,
                           "results": results, "pass": verdict}, f, indent=2)
            print(f"wrote {out}")

    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
