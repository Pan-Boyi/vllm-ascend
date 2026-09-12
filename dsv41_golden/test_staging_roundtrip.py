"""The staging-layout round trip: CPU writes, device reads, do they agree on the order?

AD2'' is void (registering on the real table killed a card), so the gather moved to
the CPU: it reads rows out of a read-only mmap and packs them into two staging
tensors that go to the device in one H2D.  The device then does a purely contiguous
read -- no indirection, no hash ids on card.

    staging_weight [num_tokens*24, 256] uint8      row = token_idx * 24 + head
    staging_scale  [num_tokens*24,   8] uint8

🔴 That removed a silent defect and introduced a different one.  The CPU writes row
``token*24 + head``; the device recomputes the same row from its program id.  **If
the two sides disagree about that formula, both still touch valid bytes** -- the CPU
wrote real rows, the device read real rows, they are just paired up wrong.  Nothing
raises.  The only symptom is "accuracy is off".

So the criterion cannot be "did it crash" or even "is the error small".  It has to
be: **is every row in the right place**, judged against a reference that does not go
through staging at all.

Why the negative arms are shaped the way they are: an ordering bug **preserves the
multiset of rows** and only moves them; a corrupted value does not.  That is a
decidable difference, so the test can say *which* of the two happened instead of
reporting one number.  A criterion that only prints a distance sends the reader to
audit the dequantize when the real fault is an index formula.

⚠️ This guard is written **before** the kernel exists, on purpose.  A test written
afterwards tends to encode whatever the kernel already does.

    python test_staging_roundtrip.py [-v]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "ref"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from compare_dumps import permutation_signature          # noqa: E402
from engram_lookup_ref import lookup_raw                 # noqa: E402

N_HASH_COLS = 24        # LOCAL_HEADS == TOTAL_HEADS == 24 only because TP=1
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok, detail))
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))


def build_staging(rows: torch.Tensor, head_major: bool = False):
    """Pack [T, 24, D] gathered rows into the flat staging tensor the device reads.

    ``head_major`` is the wrong layout, kept here as the negative arm rather than as
    a comment: the contract is only testable if the other choice is expressible.
    """
    T, H, D = rows.shape
    out = torch.empty(T * H, D, dtype=rows.dtype)
    for t in range(T):
        for h in range(H):
            out[(h * T + t) if head_major else (t * H + h)] = rows[t, h]
    return out


def device_read(staging: torch.Tensor, T: int, H: int, row_offset: int = 0):
    """What the device reconstructs: program id -> row = token*H + head."""
    D = staging.shape[-1]
    out = torch.empty(T, H, D, dtype=staging.dtype)
    for t in range(T):
        for h in range(H):
            out[t, h] = staging[(t * H + h + row_offset) % staging.shape[0]]
    return out


def verdict(name: str, ref: torch.Tensor, got: torch.Tensor, expect_ok: bool,
            expect_word: str = ""):
    """Compare, and require the REASON to be the right one, not just the verdict."""
    flat_r = ref.reshape(-1, ref.shape[-1])
    flat_g = got.reshape(-1, got.shape[-1])
    identical = torch.equal(flat_r, flat_g)
    if identical:
        check(name, expect_ok, "" if expect_ok else "expected a failure, got a match")
        return
    perm = permutation_signature(flat_r, flat_g)
    if perm and perm.get("is_permutation"):
        reason = (f"ROW ORDER MISMATCH: every row is present, {perm['moved']}/"
                  f"{perm['rows']} are in the wrong slot (slot "
                  f"{perm['first_moved_to']} holds slot "
                  f"{perm['first_came_from']}'s row). The two sides disagree about "
                  f"row = token*24 + head; do not look at the dequantize.")
    else:
        reason = ("VALUES DIFFER: the row multiset is not preserved, so this is not "
                  "a reordering -- the bytes themselves are wrong.")
    ok = (not expect_ok) and (expect_word in reason)
    check(name, ok, reason if not ok else reason[:72] + "...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--tokens", type=int, default=8)
    a = ap.parse_args()

    import torch as _t
    from ckpt import Checkpoint
    from golden import MODEL_DIR

    ck = Checkpoint(MODEL_DIR)
    g = _t.Generator().manual_seed(7)
    # Deliberate repeats: hash ids repeat naturally, and the CPU gather therefore
    # copies the same table row into several staging slots.  A layout bug that only
    # shows on distinct rows would be missed by an all-unique input.
    ids = _t.randint(0, 384006168, (1, a.tokens, N_HASH_COLS), generator=g)
    ids[0, 1, 3] = ids[0, 0, 0]
    ids[0, 2, 7] = ids[0, 0, 0]
    n_rep = ids.numel() - int(_t.unique(ids).numel())
    print(f"{a.tokens} tokens x {N_HASH_COLS} cols, {n_rep} repeated ids "
          f"(repeats are the normal case, not a corner)")
    if n_rep < 2:
        print("NOTHING CHECKED -- the input must contain repeats; regenerate")
        sys.exit(1)

    # the independent reference: straight per-hash-id gather, never goes via staging
    fp8_ref, _ = lookup_raw(ck, 1, ids)
    ref = fp8_ref[0]                                     # [T, 24, D]
    print(f"reference rows {tuple(ref.shape)} {ref.dtype}\n")

    print("1. the honest round trip must be bit-identical")
    st = build_staging(ref)
    verdict("CPU token-major -> device token-major", ref, device_read(st, *ref.shape[:2]),
            expect_ok=True)

    print("\n2. CPU writes head-major, device reads token-major")
    st_bad = build_staging(ref, head_major=True)
    verdict("head-major write -> named as ROW ORDER", ref,
            device_read(st_bad, *ref.shape[:2]), expect_ok=False,
            expect_word="ROW ORDER MISMATCH")

    print("\n3. device's row formula off by one")
    verdict("row off by one -> named as ROW ORDER", ref,
            device_read(st, *ref.shape[:2], row_offset=1), expect_ok=False,
            expect_word="ROW ORDER MISMATCH")

    print("\n4. a corrupted BYTE must NOT be called an ordering bug")
    st_val = st.clone()
    st_val[5] = st_val[5].view(torch.uint8).add(1).view(st_val.dtype)
    verdict("byte corruption -> named as VALUES DIFFER", ref,
            device_read(st_val, *ref.shape[:2]), expect_ok=False,
            expect_word="VALUES DIFFER")

    ck.close()
    print()
    bad = [n for n, ok, _ in results if not ok]
    print(f"{len(results) - len(bad)}/{len(results)} arms behaved as specified")
    if bad:
        print("FAILED: " + ", ".join(bad))
        sys.exit(1)
    print("the two ordering faults are named as ordering, the value fault is not")
    sys.exit(0)


if __name__ == "__main__":
    main()
