# SPDX-License-Identifier: Apache-2.0
"""Engram gather on NPU, checked byte-for-byte against the CPU reference.

Runs against a small synthetic table in shared memory rather than the real
94 GiB one: the gather does not care how many rows follow, and the bytes are
chosen to include the patterns that a dtype-aware path anywhere would corrupt.

Needs an NPU, so it runs in a subprocess -- registering host memory mutates
process-global device state, and testing several registrations in one
interpreter lets an earlier one supply the answer for a later one.
"""

import subprocess
import sys
import textwrap

import pytest

DIM = 256
BLOCK = 32
NUM_ROWS = 512

_SCRIPT = textwrap.dedent('''
    import ctypes, mmap, os, struct, sys
    import torch, torch_npu
    import vllm_ascend.ops  # noqa: F401  (import order: see engram_gather docstring)
    from vllm_ascend.models.deepseek_v4_1.engram_gather import (
        check_in_range, gather, register_host_table, unregister_host_table)

    DIM, BLOCK, NUM_ROWS = {dim}, {block}, {rows}
    torch.npu.set_device(0)

    # Every byte value except 0x7F and 0xFF, which are e4m3's NaNs. Golden
    # measured the real tables: 254 of 256 values occur and those two are
    # exactly the pair that does not, because quantisation clamps to +/-448.
    # Planting them would test a case the data cannot produce, and NaN compares
    # unequal to itself, so the only thing such a row measures is which NaN
    # payload each side happens to emit.
    NAN_BYTES = (0x7F, 0xFF)
    REALISTIC = [b for b in range(256) if b not in NAN_BYTES]
    # 0x80 is fp8 negative zero: it does occur (about one row in 1150) and is
    # the one pattern where bytes may differ while the value is identical.
    PATTERNS = [0x00, 0x80, 0x01, 0x81, 0x08, 0x88,
                0x55, 0xAA, 0x0F, 0x8F, 0x70, 0xF0, 0x77, 0xF7]
    tbl = bytearray(NUM_ROWS * DIM)
    tbl[:DIM] = bytes((REALISTIC * 2)[:DIM])
    tbl[DIM:DIM + len(PATTERNS)] = bytes(PATTERNS)
    for r in range(2, NUM_ROWS):
        tbl[r*DIM:(r+1)*DIM] = bytes(REALISTIC[(r * 13 + j) % len(REALISTIC)]
                                     for j in range(DIM))
    assert not (set(tbl) & set(NAN_BYTES)), "synthetic table must contain no NaN"
    assert 0x80 in set(tbl), "synthetic table must exercise negative zero"

    # ue8m0 scales: exponents around 1.0, plus 0x80 (-0 territory is in the
    # values, not here) -- one per BLOCK values along each row.
    NS = DIM // BLOCK
    scl = bytearray(NUM_ROWS * NS)
    for r in range(NUM_ROWS):
        for k in range(NS):
            scl[r*NS + k] = 127 + ((r + k) % 7) - 3

    def shm(name, payload):
        path = "/dev/shm/" + name
        with open(path, "wb") as f: f.write(bytes(payload))
        fd = os.open(path, os.O_RDWR)
        mm = mmap.mmap(fd, len(payload), mmap.MAP_SHARED)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        ctypes.CDLL("libc.so.6").mlock(ctypes.c_void_p(addr), ctypes.c_size_t(len(payload)))
        return mm, fd, path, addr

    mm_t, fd_t, p_t, a_t = shm("engram_test_tbl.bin", tbl)
    mm_s, fd_s, p_s, a_s = shm("engram_test_scl.bin", scl)
    t_table = register_host_table(a_t, len(tbl))
    t_scale = register_host_table(a_s, len(scl))

    ids = torch.tensor([0, 1, 17, 511, 2, 300, 1, 0], dtype=torch.int64)
    check_in_range(ids, NUM_ROWS)
    got = gather(t_table, t_scale, ids.npu(), DIM, NUM_ROWS, BLOCK).cpu()
    torch.npu.synchronize()

    # CPU reference over exactly the same bytes, the way golden does it.
    tb = torch.frombuffer(bytearray(tbl), dtype=torch.uint8).view(NUM_ROWS, DIM)
    sb = torch.frombuffer(bytearray(scl), dtype=torch.uint8).view(NUM_ROWS, NS)
    fp8 = tb[ids].view(torch.float8_e4m3fn)
    sc = sb[ids].view(torch.float8_e8m0fnu)
    ref = (fp8.float().unflatten(-1, (-1, BLOCK)) * sc.float().unsqueeze(-1)
           ).flatten(-2).to(torch.bfloat16)

    # G-3's criterion: an element passes if its BYTES match or its VALUE does.
    # One condition covers both hazards -- NaN (bytes match, values would not)
    # and signed zero (value matches, bytes would not).
    assert ref.shape == got.shape, (ref.shape, got.shape)
    assert ref.dtype == got.dtype, (ref.dtype, got.dtype)
    byte_diff = got.view(torch.uint8) != ref.view(torch.uint8)
    value_same = (ref == got).view(torch.uint8).repeat_interleave(2, -1).bool()
    real = int((byte_diff & ~value_same).sum())
    forgiven = int((byte_diff & value_same).sum())
    assert real == 0, "%d bytes differ in value" % real
    print("OK rows=%d dim=%d signed_zero=%d" % (ids.numel(), DIM, forgiven))

    try:
        check_in_range(torch.tensor([NUM_ROWS]), NUM_ROWS)
    except IndexError:
        print("OK range check refuses out-of-range")
    else:
        raise AssertionError("out-of-range id was accepted")

    unregister_host_table(t_table); unregister_host_table(t_scale)
    mm_t.close(); os.close(fd_t); os.unlink(p_t)
    mm_s.close(); os.close(fd_s); os.unlink(p_s)
''')


def _npu_available() -> bool:
    r = subprocess.run(
        [sys.executable, "-c",
         "import torch, torch_npu; print(torch.npu.device_count())"],
        capture_output=True, text=True, timeout=600)
    return r.returncode == 0 and r.stdout.strip().isdigit() and int(r.stdout) > 0


@pytest.mark.skipif(not _npu_available(), reason="no NPU available")
def test_gather_matches_the_cpu_reference_byte_for_byte():
    script = _SCRIPT.format(dim=DIM, block=BLOCK, rows=NUM_ROWS)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        pytest.fail(f"subprocess failed ({result.returncode}):\n"
                    f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    assert "OK rows=" in result.stdout
    assert "OK range check refuses out-of-range" in result.stdout


_MIXIN_SCRIPT = textwrap.dedent('''
    import ctypes, mmap, os
    import torch, torch_npu
    import vllm_ascend.ops  # noqa
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29611")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (init_distributed_environment,
                                  initialize_model_parallel)
    from vllm.models.deepseek_v4_1.common.engram import ParallelEngramEmbedding
    from vllm_ascend.models.deepseek_v4_1.engram_gather import (
        AscendParallelEngramEmbedding, register_host_table)

    DIM, BLOCK, NUM_ROWS, COLS = {dim}, {block}, {rows}, 4
    torch.npu.set_device(0)

    NAN_BYTES = (0x7F, 0xFF)
    REALISTIC = [b for b in range(256) if b not in NAN_BYTES]
    tbl = bytearray(NUM_ROWS * DIM)
    for r in range(NUM_ROWS):
        tbl[r*DIM:(r+1)*DIM] = bytes(REALISTIC[(r*13 + j) % len(REALISTIC)]
                                     for j in range(DIM))
    NS = DIM // BLOCK
    scl = bytes((127 + ((r + k) % 5) - 2) for r in range(NUM_ROWS) for k in range(NS))

    def shm(name, payload):
        path = "/dev/shm/" + name
        with open(path, "wb") as f: f.write(bytes(payload))
        fd = os.open(path, os.O_RDWR)
        mm = mmap.mmap(fd, len(payload), mmap.MAP_SHARED)
        a = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        ctypes.CDLL("libc.so.6").mlock(ctypes.c_void_p(a), ctypes.c_size_t(len(payload)))
        return mm, fd, path, a

    mm_t, fd_t, p_t, a_t = shm("eng_mix_tbl.bin", tbl)
    mm_s, fd_s, p_s, a_s = shm("eng_mix_scl.bin", scl)

    class AscendEngramEmbedding(AscendParallelEngramEmbedding, ParallelEngramEmbedding):
        pass

    # Construction reads torch.cuda.get_device_properties for _num_sms, which
    # is a stub on NPU. _torch_cuda_wrapper is what redirects it; production
    # enters it in the model runner, so this is not test-only scaffolding --
    # engram is the first thing that construction gap actually reaches.
    from vllm_ascend.worker.model_runner_v1 import _torch_cuda_wrapper

    cfg = VllmConfig()
    with set_current_vllm_config(cfg):
        init_distributed_environment(world_size=1, rank=0,
                                     distributed_init_method="env://",
                                     local_rank=0, backend="hccl")
        initialize_model_parallel(tensor_model_parallel_size=1)
        with _torch_cuda_wrapper():
            emb = AscendEngramEmbedding(num_embeddings=NUM_ROWS, dim=DIM,
                                        head_sizes=[NUM_ROWS // COLS] * COLS,
                                        cpu_offload=False)
    # TP=1: no sharding, no all_gather -- the whole table is local
    assert emb.part_n_hash_cols == COLS, emb.part_n_hash_cols
    assert emb.head_start == 0, emb.head_start

    emb.bind_host_tables(register_host_table(a_t, len(tbl)),
                         register_host_table(a_s, len(scl)))

    T = 5
    ids = torch.arange(T * COLS, dtype=torch.int64).reshape(T, COLS).npu() % NUM_ROWS
    out = torch.zeros(T, COLS, DIM, dtype=torch.bfloat16, device="npu")
    emb.lookup(ids, out)
    torch.npu.synchronize()

    tb = torch.frombuffer(bytearray(tbl), dtype=torch.uint8).view(NUM_ROWS, DIM)
    sb = torch.frombuffer(bytearray(scl), dtype=torch.uint8).view(NUM_ROWS, NS)
    flat = ids.cpu().reshape(-1)
    ref = ((tb[flat].view(torch.float8_e4m3fn).float().unflatten(-1, (-1, BLOCK))
            * sb[flat].view(torch.float8_e8m0fnu).float().unsqueeze(-1))
           .flatten(-2).to(torch.bfloat16).view(T, COLS, DIM))
    got = out.cpu()
    byte_diff = got.view(torch.uint8) != ref.view(torch.uint8)
    value_same = (ref == got).view(torch.uint8).repeat_interleave(2, -1).bool()
    real = int((byte_diff & ~value_same).sum())
    assert real == 0, "%d bytes differ in value" % real
    print("OK mixin lookup T=%d cols=%d" % (T, COLS))

    emb.release_host_tables()
    mm_t.close(); os.close(fd_t); os.unlink(p_t)
    mm_s.close(); os.close(fd_s); os.unlink(p_s)
''')


@pytest.mark.skipif(not _npu_available(), reason="no NPU available")
def test_mixin_drives_upstreams_embedding_class():
    """The override surface is exactly _storage and lookup, nothing else."""
    script = _MIXIN_SCRIPT.format(dim=DIM, block=BLOCK, rows=NUM_ROWS)
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        pytest.fail(f"subprocess failed ({result.returncode}):\n"
                    f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
    assert "OK mixin lookup" in result.stdout
