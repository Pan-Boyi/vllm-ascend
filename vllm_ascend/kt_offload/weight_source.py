# SPDX-License-Identifier: Apache-2.0
"""Byte locator + host gather for streaming routed-expert weights.

Pure host code: no NPU, no torch_npu, no kt-kernel. Two jobs:

* **locate** ``(layer, expert, proj, kind)`` -> ``ByteSpan`` = shard file +
  absolute byte interval, from the checkpoint index plus the shard's own
  safetensors header. The header is parsed here, *not* taken from
  ``safe_open``: the unit test uses ``safe_open().get_tensor()`` as the
  oracle, and an oracle that shares the code under test cannot go red.
* **gather** a chunk of experts of one layer into a caller-provided uint8
  staging buffer, by bytes, with a small thread pool.

mmap rather than an O_DIRECT reader: here every source view is length-checked
against the header before it is copied, and any worker exception propagates
out of ``gather`` instead of leaving the caller blocked on a queue.

Why bytes and not dtypes: the scales are E8M0 (``F8_E8M0``). ``.to(uint8)``
on an E8M0 tensor converts the *value* 2**(x-127), which for this checkpoint
(x in 117..127) is 0 or 1 -- every scale is destroyed without an error. The
only safe move is a uint8 *view*; this module never materialises a typed
tensor at all.

Every entry point returns a ``Coverage``. A request that produces nothing is
an error (``ZeroCoverageError``), never a pass: a fallback that reads "no work
done" looks exactly like "work done correctly".
"""

from __future__ import annotations

import json
import math
import mmap
import os
import struct
import warnings
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

# Checkpoint layout (DeepSeek-V4.1-Flash).
PROJS = ("w1", "w3", "w2")
KINDS = ("weight", "scale")
NUM_EXPERTS = 384
NUM_LAYERS = 40

# Expected range of this checkpoint's E8M0 scale bytes. 0x00 = 2**-127 would
# be a hole left by a mis-located or unwritten read, which is why it is checked.
E8M0_LO = 117
E8M0_HI = 127

# mmap -> staging copy threads. 8 measured fastest on this host and matches
# the H2D leg, so the two-stage pipeline is balanced; 32 and above measured
# *slower* here, whatever a single-NUMA box would want. Do not raise it
# without re-running the read bench.
DEFAULT_COPY_THREADS = 8

# Staging groups: how a chunk is laid out in the caller's buffer. Each group
# is ``C`` equal per-expert records in the caller's expert order. w13 packs w1
# then w3 per expert because a row-major concat along dim 0 is a plain byte
# concat -- the same ``w13[e] = cat(w1[e], w3[e])`` the fused-MoE loader
# builds (shard 0 = w1, shard 1 = w3) -- so no reshuffle is needed later.
GROUPS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("w13_weight", (("w1", "weight"), ("w3", "weight"))),
    ("w2_weight", (("w2", "weight"),)),
    ("w13_scale", (("w1", "scale"), ("w3", "scale"))),
    ("w2_scale", (("w2", "scale"),)),
)

# safetensors dtype -> itemsize. Only used to cross-check the header's own
# byte interval against its shape; an unknown dtype is an error, not a guess.
_ITEMSIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}  # fmt: skip

# safetensors caps the JSON header at 100 MB; a larger length prefix means
# the file is not a safetensors file (or the prefix was read at a wrong
# offset), and json.loads on gigabytes of tensor data is a hang, not an error.
_MAX_HEADER = 100 * 1024 * 1024


class ZeroCoverageError(RuntimeError):
    """An entry point produced nothing. Never a silent pass."""


class ByteSourceError(RuntimeError):
    """The bytes on disk do not match what the header promises (truncation, overlap)."""


def expert_key(layer: int, expert: int, proj: str, kind: str = "weight", prefix: str = "layers") -> str:
    if proj not in PROJS:
        raise KeyError(f"proj must be one of {PROJS}, got {proj!r}")
    if kind not in KINDS:
        raise KeyError(f"kind must be one of {KINDS}, got {kind!r}")
    # bool is an int subclass; True would silently mean expert 1.
    for name, v in (("layer", layer), ("expert", expert)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            raise KeyError(f"{name} must be a non-negative int, got {v!r}")
    return f"{prefix}.{layer}.ffn.experts.{expert}.{proj}.{kind}"


@dataclass(frozen=True)
class ByteSpan:
    """Where one tensor's bytes live: ``path[start : start + nbytes]``."""

    key: str
    path: str
    start: int  # absolute file offset (8 + header_len + data_offsets[0])
    nbytes: int
    dtype: str
    shape: tuple[int, ...]


@dataclass
class Coverage:
    """What an entry point actually produced, counted where it was produced."""

    requested_tensors: int = 0
    produced_tensors: int = 0
    produced_bytes: int = 0
    # (layer, expert, proj) triples for which every requested kind was produced.
    triples: set[tuple[int, int, str]] = field(default_factory=set)

    @property
    def produced_triples(self) -> int:
        return len(self.triples)

    def require_complete(self, what: str) -> Coverage:
        if self.produced_tensors == 0:
            raise ZeroCoverageError(f"{what}: zero coverage (requested {self.requested_tensors} tensors)")
        if self.produced_tensors != self.requested_tensors:
            raise ZeroCoverageError(
                f"{what}: produced {self.produced_tensors} of {self.requested_tensors} requested tensors"
            )
        return self

    def __str__(self) -> str:
        return (
            f"coverage tensors={self.produced_tensors}/{self.requested_tensors} "
            f"triples={self.produced_triples} bytes={self.produced_bytes}"
        )


@dataclass(frozen=True)
class Segment:
    """One tensor's place in the staging buffer."""

    group: str
    index: int  # position of the expert in the chunk (not its logical id)
    layer: int
    expert: int
    proj: str
    kind: str
    dst_offset: int
    span: ByteSpan


@dataclass(frozen=True)
class ChunkLayout:
    """Byte layout of one chunk in a staging buffer; computed, never guessed."""

    layer: int
    experts: tuple[int, ...]
    segments: tuple[Segment, ...]
    # group -> (offset of the group, bytes per expert record)
    groups: dict[str, tuple[int, int]]
    nbytes: int

    def views(self, staging: torch.Tensor) -> dict[str, torch.Tensor]:
        """``group -> uint8 [C, record_bytes]`` views into ``staging``."""
        flat = staging.view(-1)
        c = len(self.experts)
        return {g: flat[off : off + c * rec].view(c, rec) for g, (off, rec) in self.groups.items()}


@dataclass
class GatherResult:
    layout: ChunkLayout
    coverage: Coverage


def _uint8_view(buf) -> torch.Tensor:
    """``torch.frombuffer`` over the read-only mapping (see ``_Shard.view``)."""
    return torch.frombuffer(buf, dtype=torch.uint8)


# Read-only is intended here, so torch's "buffer is not writable" warning is
# noise. Registered once at import: warnings filter state is process-global and
# unlocked, and gather() runs on a thread pool, so catch_warnings would swallow
# other threads' warnings.
warnings.filterwarnings("ignore", message="The given buffer is not writable",
                        category=UserWarning)


class _Shard:
    """One safetensors file: parsed header + a lazily created private mmap."""

    def __init__(self, path: str):
        self.path = path
        self.file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ByteSourceError(f"{path}: shorter than the 8-byte header length prefix")
            (hlen,) = struct.unpack("<Q", raw)
            if hlen > _MAX_HEADER or 8 + hlen > self.file_size:
                raise ByteSourceError(f"{path}: header length {hlen} impossible for file size {self.file_size}")
            hdr = f.read(hlen)
        if len(hdr) != hlen:
            raise ByteSourceError(f"{path}: short header read {len(hdr)} < {hlen}")
        self.data_start = 8 + hlen
        self.header: dict = json.loads(hdr)
        self._mm: mmap.mmap | None = None
        self._base: torch.Tensor | None = None
        self._lock = threading.Lock()

    def span(self, key: str) -> ByteSpan:
        meta = self.header.get(key)
        if meta is None:
            raise KeyError(f"{key!r} is in the index but not in the header of {self.path}")
        dtype = meta["dtype"]
        shape = tuple(int(s) for s in meta["shape"])
        begin, end = (int(x) for x in meta["data_offsets"])
        if dtype not in _ITEMSIZE:
            raise ByteSourceError(f"{key}: unknown safetensors dtype {dtype!r}")
        expect = math.prod(shape) * _ITEMSIZE[dtype]
        if end - begin != expect:
            raise ByteSourceError(f"{key}: data_offsets span {end - begin} B != shape {shape} x {dtype} = {expect} B")
        start = self.data_start + begin
        # Truncation guard: a file cut short must fail here, not hand back
        # fewer bytes than the header promises.
        if begin < 0 or start + expect > self.file_size:
            raise ByteSourceError(
                f"{key}: bytes [{start}, {start + expect}) run past end of {self.path} ({self.file_size} B)"
            )
        return ByteSpan(key, self.path, start, expect, dtype, shape)

    def view(self, span: ByteSpan) -> torch.Tensor:
        """uint8 view of the span straight off the page cache (no copy)."""
        if self._base is None:
            with self._lock:
                if self._base is None:
                    with open(self.path, "rb") as f:
                        # ACCESS_READ, not ACCESS_COPY. kt maps this same
                        # checkpoint read-only; a writable private mapping turns
                        # a stray write into a silent COW that diverges from what
                        # kt reads. Read-only makes it a SIGSEGV instead.
                        self._mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    if len(self._mm) != self.file_size:
                        raise ByteSourceError(f"{self.path}: file changed size since the header was parsed")
                    # One frombuffer per shard, then slice: per-span it would
                    # run 40 x 384 x 4 times for the same bytes.
                    self._base = _uint8_view(self._mm)
        if len(self._mm) != self.file_size:
            raise ByteSourceError(f"{self.path}: file changed size since the header was parsed")
        return self._base[span.start : span.start + span.nbytes]


class WeightSource:
    """Locate and gather routed-expert bytes of one checkpoint.

    Instances hold open mmaps; they are meant to live for the process.
    """

    def __init__(
        self,
        ckpt_dir: str,
        *,
        prefix: str = "layers",
        num_experts: int = NUM_EXPERTS,
        copy_threads: int = DEFAULT_COPY_THREADS,
    ):
        if copy_threads < 1:
            raise ValueError(f"copy_threads must be >= 1, got {copy_threads}")
        self.ckpt_dir = ckpt_dir
        self.prefix = prefix
        self.num_experts = num_experts
        self.copy_threads = copy_threads
        with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._shards: dict[str, _Shard] = {}
        self._shards_lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None

    # -- locate ---------------------------------------------------------------
    def _shard(self, fname: str) -> _Shard:
        with self._shards_lock:
            s = self._shards.get(fname)
            if s is None:
                s = self._shards[fname] = _Shard(os.path.join(self.ckpt_dir, fname))
            return s

    def locate_key(self, key: str) -> ByteSpan:
        """Raises ``KeyError`` for an unknown key; there is no ``None`` path."""
        fname = self.weight_map.get(key)
        if fname is None:
            raise KeyError(f"{key!r} not in the checkpoint index")
        return self._shard(fname).span(key)

    def locate(self, layer: int, expert: int, proj: str, kind: str = "weight") -> ByteSpan:
        if isinstance(expert, int) and not isinstance(expert, bool) and expert >= self.num_experts:
            raise KeyError(f"expert {expert} >= num_experts {self.num_experts}")
        return self.locate_key(expert_key(layer, expert, proj, kind, self.prefix))

    def read(self, span: ByteSpan) -> torch.Tensor:
        """uint8 mmap view of exactly ``span.nbytes`` bytes. Do not write into it."""
        v = self._source_view(span)
        if v.numel() != span.nbytes:
            raise ByteSourceError(f"{span.key}: read {v.numel()} B, header says {span.nbytes} B")
        return v

    def _source_view(self, span: ByteSpan) -> torch.Tensor:
        # Separate hook so a test can inject a short read and prove the
        # length check in read() is live.
        return self._shard(os.path.basename(span.path)).view(span)

    # -- layout + gather --------------------------------------------------------
    def plan_chunk(self, layer: int, experts: Sequence[int]) -> ChunkLayout:
        experts = tuple(experts)
        if not experts:
            raise ZeroCoverageError(f"layer {layer}: empty expert chunk")
        if len(set(experts)) != len(experts):
            raise ValueError(f"layer {layer}: duplicate expert ids in chunk {experts}")
        segs: list[Segment] = []
        groups: dict[str, tuple[int, int]] = {}
        off = 0
        for gname, members in GROUPS:
            g_off, rec = off, None
            for i, e in enumerate(experts):
                r = 0
                for proj, kind in members:
                    span = self.locate(layer, e, proj, kind)
                    segs.append(Segment(gname, i, layer, e, proj, kind, off, span))
                    off += span.nbytes
                    r += span.nbytes
                # [C, rec] views need equal records; a mixed-shape chunk means
                # the checkpoint is not the one this layout was written for.
                if rec is None:
                    rec = r
                elif r != rec:
                    raise ByteSourceError(f"layer {layer} {gname}: expert {e} record {r} B != {rec} B")
            groups[gname] = (g_off, rec)
        return ChunkLayout(layer, experts, tuple(segs), groups, off)

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.copy_threads, thread_name_prefix="kt_wsrc")
        return self._pool

    def gather(self, layer: int, experts: Sequence[int], staging: torch.Tensor) -> GatherResult:
        """Copy ``experts`` of ``layer`` into ``staging`` (uint8, CPU, contiguous).

        Raises instead of truncating when the buffer is too small, and
        re-raises any worker failure (no hang, no partial success).
        """
        layout = self.plan_chunk(layer, experts)
        if staging.dtype != torch.uint8 or staging.device.type != "cpu" or not staging.is_contiguous():
            raise TypeError(
                f"staging must be a contiguous CPU uint8 tensor, got {staging.dtype} on {staging.device}"
            )
        if staging.numel() < layout.nbytes:
            raise ByteSourceError(f"staging holds {staging.numel()} B, chunk needs {layout.nbytes} B")
        flat = staging.view(-1)
        cov = Coverage(requested_tensors=len(layout.segments))
        lock = threading.Lock()

        def copy_one(seg: Segment) -> None:
            src = self.read(seg.span)
            flat[seg.dst_offset : seg.dst_offset + seg.span.nbytes].copy_(src)
            with lock:
                cov.produced_tensors += 1
                cov.produced_bytes += seg.span.nbytes

        if self.copy_threads == 1:
            for seg in layout.segments:
                copy_one(seg)
        else:
            # list() drains the iterator so the first worker exception is
            # raised here rather than lost with an unread future.
            list(self._executor().map(copy_one, layout.segments))
        _fill_triples(cov, layout.segments, KINDS)
        cov.require_complete(f"gather layer {layer} experts {len(layout.experts)}")
        return GatherResult(layout, cov)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def _fill_triples(cov: Coverage, segments: Iterable[Segment], kinds: Sequence[str]) -> None:
    seen: dict[tuple[int, int, str], set[str]] = {}
    for s in segments:
        seen.setdefault((s.layer, s.expert, s.proj), set()).add(s.kind)
    cov.triples |= {t for t, ks in seen.items() if set(kinds) <= ks}


# ---------------------------------------------------------------------------
# Oracles (used by the unit test on a sample and by a host runner on everything)
# ---------------------------------------------------------------------------


@dataclass
class E8M0Stats:
    nbytes: int
    zeros: int
    lo: int
    hi: int

    def ok(self, lo: int = E8M0_LO, hi: int = E8M0_HI) -> bool:
        return self.nbytes > 0 and self.zeros == 0 and lo <= self.lo and self.hi <= hi


def e8m0_stats(u8: torch.Tensor) -> E8M0Stats:
    """Stats over raw E8M0 bytes. Takes uint8 only: a typed E8M0 tensor here
    means someone already went through ``.to()`` and lost the values."""
    if u8.dtype != torch.uint8:
        raise TypeError(f"expected raw uint8 bytes, got {u8.dtype}")
    if u8.numel() == 0:
        raise ZeroCoverageError("e8m0_stats on zero bytes")
    return E8M0Stats(
        u8.numel(), int((u8 == 0).sum()), int(u8.min()), int(u8.max())
    )


@dataclass
class ScanReport:
    coverage: Coverage
    mismatches: list[str] = field(default_factory=list)
    e8m0: E8M0Stats | None = None

    @property
    def ok(self) -> bool:
        return (
            not self.mismatches
            and self.coverage.produced_tensors > 0
            and self.coverage.produced_tensors == self.coverage.requested_tensors
            and (self.e8m0 is None or self.e8m0.ok())
        )


def scan(
    src: WeightSource,
    layers: Iterable[int],
    experts: Iterable[int],
    *,
    projs: Sequence[str] = PROJS,
    kinds: Sequence[str] = KINDS,
) -> ScanReport:
    """Byte round-trip: located bytes == ``safe_open().get_tensor(key)`` bytes.

    The oracle is the safetensors library's own reader; the locator parses
    headers itself, so the two only agree if the offsets are right.

    The reference is addressed from the LOOP VARIABLES, never from the span the
    locator handed back: fetching ``get_tensor(span.key)`` would move both sides
    together, so a locator that resolved (layer, e, proj) to the wrong tensor
    would still pass. The key equality is therefore asserted explicitly, and the
    shard file is taken from the index by the expected key, not ``span.path``.

    A triple that cannot be read at all (offset past the end of the file, short
    read) is recorded as a mismatch and the scan continues: the report is meant
    to name every bad triple, and an exception on the first one would hide the
    rest -- while still never passing, since a mismatch is a failure.
    """
    from safetensors import safe_open

    experts = list(experts)
    handles: dict[str, object] = {}
    rep = ScanReport(Coverage())
    zeros, lo, hi, nb = 0, 255, 0, 0
    segs: list[Segment] = []
    for layer in layers:
        for e in experts:
            for proj in projs:
                for kind in kinds:
                    rep.coverage.requested_tensors += 1
                    want_key = expert_key(layer, e, proj, kind, src.prefix)
                    want_file = src.weight_map.get(want_key)
                    if want_file is None:
                        rep.mismatches.append(f"{want_key}: not in the checkpoint index")
                        continue
                    want_path = os.path.join(src.ckpt_dir, want_file)
                    try:
                        span = src.locate(layer, e, proj, kind)
                        if span.key != want_key:
                            rep.mismatches.append(
                                f"{want_key}: locator resolved {span.key!r} instead")
                            continue
                        got = src.read(span)
                    except (ByteSourceError, KeyError, ValueError, OSError) as exc:
                        rep.mismatches.append(f"{want_key}: unreadable ({type(exc).__name__}: {exc})")
                        continue
                    if want_path not in handles:
                        handles[want_path] = safe_open(want_path, framework="pt")
                    want = handles[want_path].get_tensor(want_key).contiguous().view(-1).view(torch.uint8)
                    if got.numel() != want.numel() or not torch.equal(got, want):
                        rep.mismatches.append(span.key)
                        continue
                    rep.coverage.produced_tensors += 1
                    rep.coverage.produced_bytes += span.nbytes
                    segs.append(Segment("", 0, layer, e, proj, kind, 0, span))
                    if kind == "scale":
                        st = e8m0_stats(got)
                        zeros, nb = zeros + st.zeros, nb + st.nbytes
                        lo, hi = min(lo, st.lo), max(hi, st.hi)
    _fill_triples(rep.coverage, segs, kinds)
    if nb:
        rep.e8m0 = E8M0Stats(nb, zeros, lo, hi)
    if rep.coverage.requested_tensors == 0:
        raise ZeroCoverageError("scan: nothing requested")
    return rep


def full_scan(ckpt_dir: str, *, layers: int = NUM_LAYERS, experts: int = NUM_EXPERTS) -> ScanReport:
    """Every layer x expert x (w1,w3,w2) x (weight,scale). Host runner only:
    hundreds of GiB, read twice (locator + oracle); the unit test samples
    instead."""
    return scan(WeightSource(ckpt_dir), range(layers), range(experts))


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True, help="checkpoint directory")
    ap.add_argument("--full", action="store_true", help="scan all layers x experts (slow)")
    ap.add_argument("--layers", type=int, nargs="*", default=[0, 20, 39])
    ap.add_argument("--experts", type=int, nargs="*", default=[0, 191, 383])
    a = ap.parse_args(argv)
    rep = full_scan(a.ckpt) if a.full else scan(WeightSource(a.ckpt), a.layers, a.experts)
    print(rep.coverage)
    print(f"mismatches={len(rep.mismatches)} {rep.mismatches[:8]}")
    print(f"e8m0={rep.e8m0} ok={rep.e8m0.ok() if rep.e8m0 else None}")
    print("VERDICT", "PASS" if rep.ok else "FAIL")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
