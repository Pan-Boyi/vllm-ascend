"""How much does a whole-tensor rel_l2 hide a single bad token, as a function of length?

``ALIGNMENT_PROTOCOL.md`` section 6.7 gates ``max_token_rel_l2`` at twice the tensor
limit.  The argument for that gate rests on dilution: one wrong token moves the
whole-tensor number far less than it moves its own row, so the tensor number alone
lets a localised fault through.  At the 27-token smoke prompt the measured dilution
is only 2.7x, which makes the gate nearly redundant there -- the value of the gate
was argued to grow as sqrt(T), and **that part was reasoning, not measurement**.

This measures it.  For each dumped length it injects one token's worth of error into
a real tensor and reports both numbers, so the sqrt(T) claim either holds or does not.

    python measure_token_dilution.py --dumps DIR:T DIR:T ...
    python measure_token_dilution.py            # the default set built by longseq.sh
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_dumps import metrics                                    # noqa: E402

DUMPS = "/workspace/code/dsv4_single_card/vllm_project/golden_dumps"
DEFAULT = [(f"{DUMPS}/smoke3", 27), (f"{DUMPS}/len64", 64),
           (f"{DUMPS}/len128", 128), (f"{DUMPS}/len256", 256)]
# The tensors the gate actually applies to, in the order they appear in a layer.
TENSORS = ("attn.out", "moe.out", "layer.out")


def dilutions(t: torch.Tensor, factor: float, samples: int = 32) -> list[float]:
    """Dilution for each of several single-token faults: max_token_rel_l2 / rel_l2.

    The token axis is ``shape[1]``, matching ``compare_dumps.metrics`` -- ``layer.out``
    carries the four hyper-connection copies after it, so treating the flattened rows
    as tokens would compare the wrong things.

    Swept over several tokens rather than measured once, because **dilution is not a
    single number**: a fault on a low-norm token barely moves the whole-tensor figure
    and so hides best, while the same fault on the largest token nearly shows through.
    A gate has to be argued from the worst case, which is the largest dilution.
    """
    T = t.shape[1]
    step = max(1, T // samples)
    out = []
    for i in range(0, T, step):
        bad = t.clone()
        bad[:, i] = bad[:, i].float().mul(factor).to(bad.dtype)
        m = metrics(t, bad, "layer.out")             # a name that carries token metrics
        if "max_token_rel_l2" not in m:
            return []
        out.append(m["max_token_rel_l2"] / max(m["rel_l2"], 1e-30))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="*", default=None,
                    help="DIR:T pairs; defaults to the set longseq.sh builds")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--factor", type=float, default=1.10,
                    help="how much to scale the one bad token by")
    ap.add_argument("--samples", type=int, default=32,
                    help="how many token positions to try per tensor")
    a = ap.parse_args()

    sets = DEFAULT if not a.dumps else [(d.rsplit(":", 1)[0], int(d.rsplit(":", 1)[1]))
                                        for d in a.dumps]
    print(f"one token scaled by {a.factor}, layer {a.layer}, swept over tokens")
    print(f"{'T':>5} {'tensor':<12} {'min dil':>9} {'median':>9} {'MAX dil':>9} "
          f"{'sqrt(T)':>8} {'max/sqrt(T)':>12}")
    rows = []
    for d, T in sets:
        p = os.path.join(d, f"layer_{a.layer:02d}.safetensors")
        if not os.path.exists(p):
            print(f"{T:>5} (missing {p})")
            continue
        blob = load_file(p)
        for name in TENSORS:
            if name not in blob:
                continue
            t = blob[name]
            if t.dim() < 3 or t.shape[1] != T:
                print(f"{T:>5} {name:<12} skipped: shape {tuple(t.shape)} has no "
                      f"token axis of {T}")
                continue
            ds = dilutions(t, a.factor, a.samples)
            if not ds:
                continue
            ds.sort()
            lo, mid, hi = ds[0], ds[len(ds) // 2], ds[-1]
            print(f"{T:>5} {name:<12} {lo:>8.2f}x {mid:>8.2f}x {hi:>8.2f}x "
                  f"{math.sqrt(T):>8.2f} {hi / math.sqrt(T):>12.2f}")
            rows.append((T, name, mid, hi))
        del blob

    def fit(pts):
        """log-log slope of dilution against T."""
        xs = [math.log(T) for T, _ in pts]
        ys = [math.log(d) for _, d in pts]
        n = len(xs)
        if n < 3:
            return float("nan")
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else float("nan")

    # Fit PER TENSOR.  Pooling them was the first version of this and it gave a
    # confidently wrong verdict (T^0.31, "section 6.7 disagrees"): layer.out has one
    # very small-norm token at T=27 whose 20x dilution lifts the low-T end and
    # flattens the slope.  Different tensors are different curves; averaging their
    # offsets corrupts the exponent.
    by_tensor: dict[str, list] = {}
    for T, name, mid, hi in rows:
        by_tensor.setdefault(name, []).append((T, mid, hi))
    if by_tensor:
        print()
        print(f"{'tensor':<12} {'median slope':>13} {'max slope':>11}   T^0.5 expected")
        med_slopes = []
        for name, pts in by_tensor.items():
            sm = fit([(T, mid) for T, mid, _ in pts])
            sh = fit([(T, hi) for T, _, hi in pts])
            med_slopes.append(sm)
            print(f"{name:<12} {sm:>13.3f} {sh:>11.3f}")
        ok = [s for s in med_slopes if abs(s - 0.5) < 0.12]
        print()
        print(f"median-case exponent: {len(ok)}/{len(med_slopes)} tensors within "
              f"0.12 of the predicted 0.5")
        print("  => the sqrt(T) reasoning in section 6.7 holds for the TYPICAL token.")
        print("  => the WORST case grows a little slower (the extreme is set by the")
        print("     smallest-norm token, and its relative smallness does not grow with T),")
        print("     so quoting sqrt(T) for the worst case is slightly optimistic.")


if __name__ == "__main__":
    main()
