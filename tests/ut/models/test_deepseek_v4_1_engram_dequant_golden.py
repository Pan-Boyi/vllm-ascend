# SPDX-License-Identifier: Apache-2.0
"""Run the golden dequantisation comparison inside the gate.

This existed and never ran. ``build_artifacts/npu_view/
test_engram_dequant_reference.py`` checks the dequantisation arithmetic against
golden's authoritative dump -- the real fp8 rows, their ue8m0 scales, and the
bf16 the reference model produced -- byte for byte, on the CPU, with no card.
It is better evidence than the synthetic checks beside it in this directory,
and it sat outside ``tests/ut`` where nothing would ever run it.

It is invoked rather than ported. Two copies of a byte-comparison drift, and
the one that drifts is the copy nobody runs -- which is how this started.

A missing dump **skips loudly** rather than passing. "The data is not here" and
"the arithmetic is right" are different answers, and a gate that returns the
second for the first is worse than one that has no check at all: it reports
coverage it does not have.
"""

import os
import subprocess
import sys

import pytest

_ROOT = "/workspace/code/dsv4_single_card/vllm_project"
SCRIPT = f"{_ROOT}/build_artifacts/npu_view/test_engram_dequant_reference.py"
DUMP = f"{_ROOT}/golden_dumps/engram_lookup_ref"


def _missing() -> str | None:
    if not os.path.exists(SCRIPT):
        return f"script not found: {SCRIPT}"
    if not os.path.isdir(DUMP):
        return f"golden dump not found: {DUMP}"
    return None


def test_the_dequantisation_matches_goldens_own_rows():
    """The arithmetic, against the model's real output rather than a construction.

    The synthetic tests next door check the same expression two ways, which
    catches a transcription error but cannot catch a shared misreading of what
    the data looks like. This one reads golden's bytes.
    """
    reason = _missing()
    if reason:
        pytest.skip(f"cannot evaluate: {reason}")
    proc = subprocess.run([sys.executable, SCRIPT], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, (
        f"golden dequantisation comparison failed:\n{proc.stdout}\n{proc.stderr}")
    # The script prints one line per check. A run that compared nothing also
    # exits 0, so the count is part of the criterion.
    checks = [ln for ln in proc.stdout.splitlines()
              if ln.strip().startswith(("ok ", "MISMATCH"))]
    assert len(checks) >= 4, (
        f"only {len(checks)} comparisons ran; the dump has two engram layers "
        f"and several properties each. Output:\n{proc.stdout}")


def test_the_negative_zero_case_is_actually_present():
    """The hazard the byte comparison exists for has to be in the data.

    golden's manifest records a negative zero in both the fp8 input and the
    bf16 output. torch.equal reports 0.0 and -0.0 as equal, so a sign error
    producing +0.0 would pass a value comparison -- and if the dump ever stops
    containing one, the byte comparison is still correct and no longer
    load-bearing. Then this says so.
    """
    reason = _missing()
    if reason:
        pytest.skip(f"cannot evaluate: {reason}")
    import json

    with open(f"{DUMP}/engram_lookup_ref.json") as fh:
        manifest = json.load(fh)
    blob = json.dumps(manifest)
    assert "neg_zero" in blob, (
        "golden's manifest no longer records a negative-zero count. The byte "
        "comparison is still right, but it is no longer covering the case it "
        "was written for -- check what the dump changed to.")
