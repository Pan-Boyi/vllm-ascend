# SPDX-License-Identifier: Apache-2.0
"""Byte locator + host gather -- CPU only, no NPU.

Oracle: ``safe_open().get_tensor(key)`` bytes, compared byte-for-byte with what
the locator (its own header parser + its own mmap) hands back. Every oracle has
mutation arms that break the locator one way and must turn it red.

Two fixtures: a synthetic two-shard checkpoint (always runs, and can be cut
short) and the real checkpoint, sampled.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
import torch
from safetensors.torch import save_file

from vllm_ascend.kt_offload import weight_source as ws
from vllm_ascend.kt_offload.weight_source import (
    DEFAULT_COPY_THREADS,
    E8M0_HI,
    E8M0_LO,
    ByteSourceError,
    Coverage,
    WeightSource,
    ZeroCoverageError,
    e8m0_stats,
    scan,
)

# Points at a real checkpoint; the tests that use it skip when it is absent.
# Override with KT_TEST_CKPT to run them against a checkpoint elsewhere.
REAL_CKPT = os.environ.get("KT_TEST_CKPT", "/workspace/models/DeepSeek-V4.1-Flash")
needs_real = pytest.mark.skipif(
    not os.path.exists(os.path.join(REAL_CKPT, "model.safetensors.index.json")),
    reason=f"real checkpoint not at {REAL_CKPT}",
)

# synthetic checkpoint
SYN_LAYERS, SYN_EXPERTS = 2, 5
# Distinct sizes per (proj, kind) so a key mix-up changes the length, not
# only the content.
SYN_SHAPES = {
    ("w1", "weight"): (6, 10),
    ("w3", "weight"): (6, 10),
    ("w2", "weight"): (12, 3),
    ("w1", "scale"): (6, 2),
    ("w3", "scale"): (6, 2),
    ("w2", "scale"): (12, 1),
}


def _syn_tensor(g, proj, kind):
    shape = SYN_SHAPES[(proj, kind)]
    if kind == "scale":
        return torch.randint(E8M0_LO, E8M0_HI + 1, shape, generator=g, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return torch.randint(-128, 128, shape, generator=g, dtype=torch.int8)


@pytest.fixture(scope="module")
def syn_ckpt(tmp_path_factory):
    d = tmp_path_factory.mktemp("syn_ckpt")
    g = torch.Generator().manual_seed(7)
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    names = list(shards)
    wmap = {}
    for layer in range(SYN_LAYERS):
        for e in range(SYN_EXPERTS):
            for proj in ws.PROJS:
                for kind in ws.KINDS:
                    k = ws.expert_key(layer, e, proj, kind)
                    # layer 1 straddles both shards, like a real split point
                    fn = names[0] if (layer == 0 or e < 2) else names[1]
                    shards[fn][k] = _syn_tensor(g, proj, kind)
                    wmap[k] = fn
    # a non-expert neighbour with a 2-byte dtype, so offsets are not all
    # multiples of the expert record sizes
    shards[names[0]]["layers.0.ffn.gate.weight"] = torch.randn(3, 5, generator=g).to(torch.bfloat16)
    wmap["layers.0.ffn.gate.weight"] = names[0]
    for fn, tensors in shards.items():
        save_file(tensors, str(d / fn))
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}))
    return str(d)


@pytest.fixture
def syn(syn_ckpt):
    src = WeightSource(syn_ckpt, num_experts=SYN_EXPERTS)
    yield src
    src.close()


def _oracle_bytes(path, key):
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key).contiguous().view(-1).view(torch.uint8).clone()


# byte round-trip (synthetic, full coverage)
def test_syn_roundtrip_all_keys(syn):
    rep = scan(syn, range(SYN_LAYERS), range(SYN_EXPERTS))
    n = SYN_LAYERS * SYN_EXPERTS * 3 * 2
    print(rep.coverage, rep.e8m0)
    assert rep.mismatches == []
    assert rep.coverage.requested_tensors == rep.coverage.produced_tensors == n
    assert rep.coverage.produced_triples == SYN_LAYERS * SYN_EXPERTS * 3
    assert rep.e8m0 is not None and rep.e8m0.ok()
    assert rep.ok


def test_syn_gather_layout_matches_oracle(syn):
    experts = [3, 0, 4]  # caller order, not sorted: records follow it
    layout = syn.plan_chunk(1, experts)
    staging = torch.zeros(layout.nbytes + 11, dtype=torch.uint8)  # slack must stay untouched
    staging[layout.nbytes :] = 0xAB
    res = syn.gather(1, experts, staging)
    assert res.coverage.produced_tensors == len(experts) * 6
    assert res.coverage.produced_triples == len(experts) * 3
    assert res.coverage.produced_bytes == layout.nbytes
    assert torch.all(staging[layout.nbytes :] == 0xAB)
    views = layout.views(staging)
    for gname, members in ws.GROUPS:
        for i, e in enumerate(experts):
            want = torch.cat([_oracle_bytes(syn.locate(1, e, p, k).path, ws.expert_key(1, e, p, k)) for p, k in members])
            assert torch.equal(views[gname][i], want), (gname, e)


def test_syn_threads_do_not_change_bytes(syn_ckpt):
    outs = []
    for t in (1, 3, DEFAULT_COPY_THREADS):
        src = WeightSource(syn_ckpt, num_experts=SYN_EXPERTS, copy_threads=t)
        layout = src.plan_chunk(1, range(SYN_EXPERTS))
        buf = torch.zeros(layout.nbytes, dtype=torch.uint8)
        src.gather(1, range(SYN_EXPERTS), buf)
        src.close()
        outs.append(buf)
    assert all(torch.equal(outs[0], o) for o in outs[1:])


def test_default_copy_threads_is_8():
    # measured faster than the reference implementation's 32 on this host
    assert DEFAULT_COPY_THREADS == 8
    assert WeightSource.__init__.__kwdefaults__["copy_threads"] == 8


# mutation arms -- each must turn an oracle red
@pytest.mark.parametrize("delta", [+1, -1])
def test_mut_offset_shift_is_red(syn, monkeypatch, delta):
    orig = ws._Shard.span

    def shifted(self, key):
        s = orig(self, key)
        return ws.ByteSpan(s.key, s.path, s.start + delta, s.nbytes, s.dtype, s.shape)

    monkeypatch.setattr(ws._Shard, "span", shifted)
    rep = scan(syn, range(SYN_LAYERS), range(SYN_EXPERTS))
    assert not rep.ok
    # every tensor must be caught, not just some: a shift that only some
    # keys notice would mean the oracle is looking at the wrong bytes
    assert len(rep.mismatches) == rep.coverage.requested_tensors


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.locate_key("layers.0.ffn.expert.0.w1.weight"),  # expert (no s)
        lambda s: s.locate_key("layers.0.ffn.experts.0.w1.weights"),
        lambda s: s.locate(0, 0, "w4"),
        lambda s: s.locate(0, 0, "w1", "scales"),
        lambda s: s.locate(0, True, "w1"),  # bool is an int; must not mean expert 1
        lambda s: s.locate(SYN_LAYERS, 0, "w1"),
    ],
)
def test_mut_misspelled_key_raises(syn, call):
    # returning None would be the silent-skip failure; only an exception passes
    with pytest.raises(KeyError):
        call(syn)


def test_mut_short_read_raises(syn, monkeypatch):
    orig = WeightSource._source_view
    monkeypatch.setattr(WeightSource, "_source_view", lambda self, span: orig(self, span)[:-1])
    with pytest.raises(ByteSourceError, match="read"):
        syn.read(syn.locate(0, 0, "w1"))
    layout = syn.plan_chunk(0, [0, 1])
    with pytest.raises(ByteSourceError):
        syn.gather(0, [0, 1], torch.zeros(layout.nbytes, dtype=torch.uint8))


def test_mut_truncated_file_raises(syn_ckpt, tmp_path):
    d = tmp_path / "trunc"
    shutil.copytree(syn_ckpt, d)
    fn = d / "model-00002-of-00002.safetensors"
    size = fn.stat().st_size
    with open(fn, "r+b") as f:
        f.truncate(size - 1)
    src = WeightSource(str(d), num_experts=SYN_EXPERTS)
    # the tensor at the very end of the file must be refused, not shortened
    last = max(
        (k for k in src.weight_map if src.weight_map[k] == fn.name),
        key=lambda k: src._shard(fn.name).header[k]["data_offsets"][1],
    )
    with pytest.raises(ByteSourceError, match="past end"):
        src.locate_key(last)


def test_mut_wrong_expert_is_red(syn, monkeypatch):
    orig = WeightSource.locate

    def off_by_one(self, layer, expert, proj, kind="weight"):
        return orig(self, layer, (expert + 1) % SYN_EXPERTS, proj, kind)

    monkeypatch.setattr(WeightSource, "locate", off_by_one)
    rep = scan(syn, range(SYN_LAYERS), range(SYN_EXPERTS))
    assert not rep.ok and len(rep.mismatches) == rep.coverage.requested_tensors


def test_mut_expert_out_of_range_raises(syn):
    with pytest.raises(KeyError):
        syn.locate(0, SYN_EXPERTS, "w1")


def test_mut_e8m0_value_cast_is_red(syn):
    span = syn.locate(0, 0, "w1", "scale")
    good = e8m0_stats(syn.read(span))
    assert good.ok()
    from safetensors import safe_open

    with safe_open(span.path, framework="pt") as f:
        typed = f.get_tensor(span.key)
    assert typed.dtype == torch.float8_e8m0fnu
    # the forbidden move: a value cast. 2**(x-127) for x<=127 is 0 or 1.
    assert not e8m0_stats(typed.to(torch.uint8)).ok()
    with pytest.raises(TypeError):
        e8m0_stats(typed)  # typed E8M0 is refused outright


def test_mut_e8m0_zero_byte_is_red(syn):
    b = syn.read(syn.locate(1, 2, "w2", "scale")).clone()
    b[len(b) // 2] = 0
    st = e8m0_stats(b)
    assert st.zeros == 1 and not st.ok()


def test_mut_staging_too_small_raises(syn):
    layout = syn.plan_chunk(0, [1, 2])
    with pytest.raises(ByteSourceError, match="staging"):
        syn.gather(0, [1, 2], torch.zeros(layout.nbytes - 1, dtype=torch.uint8))
    with pytest.raises(TypeError):
        syn.gather(0, [1, 2], torch.zeros(layout.nbytes, dtype=torch.int8))


def test_zero_coverage_is_an_error(syn):
    with pytest.raises(ZeroCoverageError):
        syn.gather(0, [], torch.zeros(16, dtype=torch.uint8))
    with pytest.raises(ZeroCoverageError):
        scan(syn, [], range(SYN_EXPERTS))
    with pytest.raises(ZeroCoverageError):
        Coverage(requested_tensors=6).require_complete("x")
    with pytest.raises(ZeroCoverageError, match="5 of 6"):
        Coverage(requested_tensors=6, produced_tensors=5).require_complete("x")


def test_mut_dropped_segment_is_red(syn, monkeypatch):
    # a worker that silently skips one copy (the reference's `break`) must
    # show up as partial coverage, not a green gather
    orig = WeightSource.read
    calls = {"n": 0}

    def flaky(self, span):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ByteSourceError("injected worker failure")
        return orig(self, span)

    monkeypatch.setattr(WeightSource, "read", flaky)
    layout = syn.plan_chunk(0, [0, 1, 2])
    with pytest.raises(ByteSourceError, match="injected"):
        syn.gather(0, [0, 1, 2], torch.zeros(layout.nbytes, dtype=torch.uint8))


# real checkpoint, sampled
REAL_LAYERS = (0, 3, 20, 39)
REAL_EXPERTS = (0, 5, 191, 383)
# 3 x 5,898,240 (packed FP4 as I8) + 3 x 368,640 (E8M0) per expert per layer
REAL_EXPERT_BYTES = 18_800_640


@pytest.fixture(scope="module")
def real():
    src = WeightSource(REAL_CKPT)
    yield src
    src.close()


@needs_real
def test_real_roundtrip_sampled(real):
    rep = scan(real, REAL_LAYERS, REAL_EXPERTS)
    n = len(REAL_LAYERS) * len(REAL_EXPERTS)
    print(rep.coverage, rep.e8m0)
    assert rep.mismatches == []
    assert rep.coverage.produced_tensors == rep.coverage.requested_tensors == n * 6
    assert rep.coverage.produced_triples == n * 3
    assert rep.coverage.produced_bytes == n * REAL_EXPERT_BYTES
    # E8M0: no 0x00 byte, every byte in 117..127
    st = rep.e8m0
    assert st is not None and st.nbytes == n * 3 * 368_640
    assert st.zeros == 0 and E8M0_LO <= st.lo and st.hi <= E8M0_HI, st
    assert rep.ok


@needs_real
def test_real_spans_and_dtypes(real):
    for proj, shape in (("w1", (2304, 2560)), ("w3", (2304, 2560)), ("w2", (5120, 1152))):
        s = real.locate(3, 5, proj)
        assert (s.dtype, s.shape, s.nbytes) == ("I8", shape, 5_898_240)
    for proj in ws.PROJS:
        s = real.locate(3, 5, proj, "scale")
        assert s.dtype == "F8_E8M0" and s.nbytes == 368_640


@needs_real
def test_real_gather_chunk(real):
    experts = [383, 7, 191, 0, 64, 5, 300, 1]
    layout = real.plan_chunk(20, experts)
    assert layout.nbytes == len(experts) * REAL_EXPERT_BYTES
    staging = torch.zeros(layout.nbytes, dtype=torch.uint8)
    res = real.gather(20, experts, staging)
    assert res.coverage.produced_triples == len(experts) * 3
    assert res.coverage.produced_bytes == layout.nbytes
    views = layout.views(staging)
    assert views["w13_weight"].shape == (len(experts), 2 * 5_898_240)
    assert views["w13_scale"].shape == (len(experts), 2 * 368_640)
    for gname, members in ws.GROUPS:
        for i in (0, 3, len(experts) - 1):
            e = experts[i]
            want = torch.cat([_oracle_bytes(real.locate(20, e, p, k).path, ws.expert_key(20, e, p, k)) for p, k in members])
            assert torch.equal(views[gname][i], want), (gname, e)
    sc = e8m0_stats(torch.cat([views["w13_scale"].reshape(-1), views["w2_scale"].reshape(-1)]))
    assert sc.ok(), sc


@needs_real
@pytest.mark.parametrize("delta", [+1, -1])
def test_real_mut_offset_shift_is_red(real, monkeypatch, delta):
    orig = ws._Shard.span

    def shifted(self, key):
        s = orig(self, key)
        return ws.ByteSpan(s.key, s.path, s.start + delta, s.nbytes, s.dtype, s.shape)

    monkeypatch.setattr(ws._Shard, "span", shifted)
    rep = scan(real, (3,), (5, 383))
    assert not rep.ok and len(rep.mismatches) == rep.coverage.requested_tensors == 12


@needs_real
def test_real_mut_wrong_expert_and_key_are_red(real, monkeypatch):
    with pytest.raises(KeyError):
        real.locate_key("layers.3.ffn.experts.5.w1.wieght")
    with pytest.raises(KeyError):
        real.locate(3, 384, "w1")
    orig = WeightSource.locate
    monkeypatch.setattr(WeightSource, "locate", lambda self, l, e, p, k="weight": orig(self, l, e + 1, p, k))
    rep = scan(real, (3,), (5,))
    assert not rep.ok and len(rep.mismatches) == rep.coverage.requested_tensors == 6
