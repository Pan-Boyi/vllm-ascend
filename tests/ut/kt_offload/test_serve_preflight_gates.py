# SPDX-License-Identifier: Apache-2.0
"""The two start-up gates of the single-card launcher -- CPU only, no service.

``tools/kt_offload/serve_preflight_gates.sh`` is a separate file precisely so
these can be exercised without starting anything:

* ``gate_so``   the tree's ``vllm_ascend_C*.so`` must be the expected build.
  Old ``.so`` + new code is a silent wrong answer, not a crash, so the
  expectation defaults to a placeholder and the gate must refuse until the
  operator package is swapped.
* ``gate_ds41`` the ds41 operator package (O6, >16k) must be the one the
  interpreter really resolves. Missing it raises nothing: the candidate
  layers fall back to v2 and >16k stays blocked by the guard.

Every criterion here has both arms: a refusal and the case that must pass,
because a gate that refuses everything and one that works read the same.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
GATES = ROOT / "tools/kt_offload/serve_preflight_gates.sh"
DS41_SITE = pathlib.Path("/workspace/code/dsv4_single_card/ds41_site")
PLACEHOLDER = "PENDING_NEW_SO_AFTER_f85267ab8"


def run(args, env=None):
    e = {"PATH": "/usr/local/python3.12.13/bin:/usr/bin:/bin", "HOME": "/root"}
    e.update(env or {})
    return subprocess.run(["bash", str(GATES), *args], capture_output=True,
                          text=True, env=e, timeout=300)


def tree_so():
    sos = sorted((ROOT / "vllm_ascend").glob("vllm_ascend_C*.so"))
    assert len(sos) == 1, f"树里有 {len(sos)} 份 .so：{sos}"
    return sos[0]


# -- the .so gate ------------------------------------------------------------


def test_the_placeholder_expectation_refuses_this_tree():
    """The state this was written in: rebased onto f85267ab8, still carrying
    the old ``.so``. The gate must refuse, and say why."""
    r = run(["so"])
    assert r.returncode == 1, r.stdout + r.stderr
    assert "拒绝起飞" in r.stdout
    assert "静默算错" in r.stdout
    assert PLACEHOLDER in r.stdout
    assert str(ROOT) in r.stdout          # names the tree it checked (k.14)


def test_the_gate_passes_on_the_matching_md5():
    """Positive control: a gate that refuses everything is not a gate."""
    md5 = hashlib.md5(tree_so().read_bytes()).hexdigest()  # noqa: S324
    r = run(["so"], {"SERVE_EXPECT_SO_MD5": md5})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "与本树期望值一致" in r.stdout


def test_one_flipped_character_is_refused():
    """Mutation arm on the comparison itself."""
    md5 = hashlib.md5(tree_so().read_bytes()).hexdigest()  # noqa: S324
    wrong = ("b" if md5[0] == "a" else "a") + md5[1:]
    r = run(["so"], {"SERVE_EXPECT_SO_MD5": wrong})
    assert r.returncode == 1
    assert md5 in r.stdout and wrong in r.stdout


def test_a_tree_without_a_so_is_refused_not_skipped(tmp_path):
    """Unreadable must not read as "matches"; fail-safe is to refuse."""
    (tmp_path / "vllm_ascend").mkdir()
    r = run(["so", str(tmp_path)], {"SERVE_EXPECT_SO_MD5": PLACEHOLDER})
    assert r.returncode == 3
    assert "没有 vllm_ascend_C" in r.stdout


# -- the ds41 gate -----------------------------------------------------------


@pytest.mark.skipif(not DS41_SITE.is_dir(), reason=f"{DS41_SITE} 不在")
def test_ds41_resolves_inside_the_site_dir():
    r = run(["ds41"], {"DS41": str(DS41_SITE)})
    assert r.returncode == 0, r.stdout + r.stderr
    # the criterion reads the resolved files, not the env var
    assert f"cann_ops_transformer -> {DS41_SITE}" in r.stdout
    assert f"import cannbotdsl 的 __file__ = {DS41_SITE}" in r.stdout


def test_a_wrong_ds41_dir_is_refused_and_names_what_it_resolved_to(tmp_path):
    """Positive control for the test above, and the failure mode it guards:
    with ds41 not in front, ``cann_ops_transformer`` resolves to CANN's own
    copy and nothing raises at import time -- the run just silently loses O6.
    """
    r = run(["ds41"], {"DS41": str(tmp_path)})
    assert r.returncode == 1
    assert "拒绝起飞" in r.stdout
    assert "cann_ops_transformer: 解析到" in r.stdout


def test_ds41_can_be_turned_off_on_purpose_and_says_so():
    r = run(["ds41"], {"DS41": "0"})
    assert r.returncode == 0
    assert "显式关闭" in r.stdout and "16384" in r.stdout


# -- the launcher really calls them -----------------------------------------


def test_the_launcher_calls_both_gates_and_sets_the_graph_flag():
    """A gate nobody calls is the defect it claims to fix, under a name that
    says it is fixed. Checked on the launcher's source."""
    src = (ROOT / "tools/kt_offload/serve_v41_singlecard.sh").read_text()
    assert "serve_preflight_gates.sh" in src
    assert "gate_so \"$_TREE\" || exit 1" in src
    assert "gate_ds41 || exit 1" in src
    assert 'export VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-1}"' in src
    # ...and before the two exec lines, not after them
    assert src.index("gate_so") < src.index("exec vllm serve")
