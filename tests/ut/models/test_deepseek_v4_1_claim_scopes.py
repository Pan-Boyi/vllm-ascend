# SPDX-License-Identifier: Apache-2.0
"""Every graded claim names the object it is about.

🔴 Four times in one day a claim's *subject* was narrower than its *wording*,
and three of those were in one comment block:

    the rope ✅          evidence: rope_quant_insert's signature
                         subject: cache rows
                         read as: covering sa_in.kv
                         cost: four independent readings read as "rope is not
                               applied", and a correct rope nearly rewritten

    "nothing is         evidence: one cache's dtype, one argument
     quantised" ✅       subject: the KV cache write path
                         read as: the whole model
                         cost: an entire quantisation family ruled out, while
                               the MoE weights are MXFP4 with E8M0 scales

    the prose contract   subject: another *model's* call site
    the substitution     subject: the kinds of substitution I had thought of
    table

Every one of those four pieces of evidence is true. What carried the error was
the wording, and the wording is what the next reader quotes. Rules did not stop
it -- this is the fourth occurrence -- so it is a check instead.
"""

import inspect
import re

from vllm_ascend.models.deepseek_v4_1 import align_taps

#: A graded claim opens with one of these.
GRADES = ("✅", "📖")

#: And must carry one of these, naming what it is about.
SCOPE_MARKER = "SCOPE"


def _claim_blocks() -> list:
    """``(line number, first line, whole block)`` for every graded claim.

    A block runs from its grade marker to the next grade marker or the end of
    the comment run, so a SCOPE note that belongs to the claim above it is
    inside that claim's block and not the next one's.
    """
    source = inspect.getsource(align_taps).splitlines()
    blocks, current = [], None
    for number, line in enumerate(source, start=1):
        stripped = line.strip().lstrip("#").strip()
        opens = any(stripped.startswith(g) for g in GRADES)
        if opens:
            if current is not None:
                blocks.append(current)
            current = [number, stripped, [stripped]]
            continue
        if current is None:
            continue
        if not line.strip().startswith("#"):
            blocks.append(current)
            current = None
            continue
        current[2].append(stripped)
    if current is not None:
        blocks.append(current)
    return blocks


def test_there_are_graded_claims_to_check():
    """Zero coverage would read exactly like every claim passing."""
    blocks = _claim_blocks()
    assert len(blocks) >= 4, (
        f"only {len(blocks)} graded claims found; the markers were probably "
        "renamed and this check has quietly stopped checking anything")


def test_every_graded_claim_names_its_subject():
    """🔴 The check that four incidents' worth of rules did not achieve.

    A claim graded as established must say which path or tensor it is
    established *for*. Without it the sentence is loose in exactly the
    direction that gets quoted.
    """
    missing = []
    for number, first, block in _claim_blocks():
        if not any(SCOPE_MARKER in line for line in block):
            missing.append(f"align_taps.py:{number}  {first[:88]}")
    assert not missing, (
        "these graded claims do not name the object they are about, and a "
        "claim whose subject is narrower than its wording is what has gone "
        "wrong four times:\n  " + "\n  ".join(missing))
