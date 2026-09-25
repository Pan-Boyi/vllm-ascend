# SPDX-License-Identifier: Apache-2.0
"""Our log lines must actually come out in a worker-shaped process.

The worker only calls ``configure_ascend_file_logging()``, so a ``vllm_ascend.*``
logger keeps the root's WARNING level and its INFO never reaches the console;
only ``from vllm.logger import logger`` does. Under pytest every logger name is
visible to caplog, so the check runs in a subprocess that stands the worker's
configuration up, with the old spelling alongside as the arm that must stay
silent.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODULES = ("layer", "placement", "hostfunc_log", "expert_counter", "numa_placement")

PROBE = textwrap.dedent(
    """
    import json, logging, sys, tempfile

    import vllm  # vLLM's own logging config is applied on import
    import vllm_ascend.logger as al

    al._LOG_DIR = tempfile.mkdtemp(prefix="kt_probe_")
    al.configure_ascend_file_logging()   # exactly what the worker process calls
    # configure_ascend_logging() is deliberately NOT called: the worker does not.

    import importlib

    out = {}
    for name in __MODULES__:
        mod = importlib.import_module("vllm_ascend.kt_offload." + name)
        out[name] = mod.logger.name
        mod.logger.info("KTPROBE-INFO %s", name)
        mod.logger.warning("KTPROBE-WARNING %s", name)

    # the old spelling, on a fresh name so nothing else has configured it
    old = logging.getLogger("vllm_ascend.kt_offload.old_spelling")
    old.info("KTPROBE-INFO old_spelling")
    old.warning("KTPROBE-WARNING old_spelling")
    from vllm.logger import init_logger

    # init_logger under the same name: same gate, no handler and no level
    init_logger("vllm_ascend.kt_offload.init_logger_spelling").info("KTPROBE-INFO init_logger_spelling")

    sys.stderr.flush()
    sys.stdout.flush()
    print("KTPROBE-NAMES " + json.dumps(out))
    """
)


@pytest.fixture(scope="module")
def probe() -> dict:
    """Run the probe once; hand back its output and the logger names it saw."""
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(REPO.parent / "moe-offload-vllm"))
    env["VLLM_LOGGING_LEVEL"] = "INFO"  # production value; named, not assumed
    proc = subprocess.run(
        [sys.executable, "-c", PROBE.replace("__MODULES__", repr(MODULES))],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    text = proc.stdout + proc.stderr
    marker = "KTPROBE-NAMES "
    assert marker in proc.stdout, f"the probe did not finish (rc={proc.returncode}):\n{text[-4000:]}"
    names = json.loads(proc.stdout.split(marker, 1)[1].splitlines()[0])
    return {"text": text, "names": names, "rc": proc.returncode}


@pytest.mark.parametrize("module", MODULES)
def test_our_info_lines_reach_the_console_in_a_worker_shaped_process(probe, module):
    assert f"KTPROBE-INFO {module}" in probe["text"], (
        f"kt_offload.{module}'s INFO did not come out; its logger is "
        f"{probe['names'][module]!r}\n--- probe output ---\n{probe['text'][-3000:]}"
    )


@pytest.mark.parametrize("module", MODULES)
def test_our_warnings_reach_the_console_too(probe, module):
    assert f"KTPROBE-WARNING {module}" in probe["text"], probe["text"][-3000:]


def test_the_old_spelling_is_still_silent(probe):
    """Positive control: without it, a probe that prints everything would pass."""
    assert "KTPROBE-INFO old_spelling" not in probe["text"], (
        "logging.getLogger(__name__) now delivers INFO in the probe, so the probe no "
        "longer reproduces the production configuration and the arms above prove nothing"
    )


def test_init_logger_would_not_have_fixed_it(probe):
    """init_logger changes neither the handler nor the level: same name, same silence."""
    assert "KTPROBE-INFO init_logger_spelling" not in probe["text"], (
        "init_logger under a vllm_ascend.* name delivered INFO; re-read "
        "vllm/logger.py:204-214 before trusting the docstring above"
    )


def test_the_old_spelling_did_reach_the_ascend_file_log(probe):
    """WARNING/ERROR on the old spelling goes to the ascend file log, not the console."""
    assert "KTPROBE-WARNING old_spelling" not in probe["text"], probe["text"][-2000:]


def test_every_logger_in_this_package_is_one_that_comes_out():
    """Covers the next module someone adds, and reports its own coverage."""
    import ast

    pkg = REPO / "vllm_ascend" / "kt_offload"
    files = sorted(p for p in pkg.glob("*.py") if p.name != "__init__.py")
    assert len(files) >= 5, f"the scan only found {len(files)} modules in {pkg}"
    bad, logging_users = [], 0
    for path in files:
        src = path.read_text()
        tree = ast.parse(src)
        uses_logger = any(
            isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name)
            and n.value.id == "logger"
            and n.attr in {"info", "warning", "error", "debug", "info_once", "warning_once", "exception"}
            for n in ast.walk(tree)
        )
        if not uses_logger:
            continue
        logging_users += 1
        if "from vllm.logger import logger" not in src:
            bad.append(path.name)
    assert not bad, (
        "these modules log through a logger that is silent in the worker process: "
        f"{bad}; use `from vllm.logger import logger`"
    )
    assert logging_users >= 4, f"only {logging_users} modules log at all; the scan is looking in the wrong place"


def test_caplog_cannot_see_our_records(caplog):
    """vLLM's dictConfig sets ``propagate: False`` on the ``vllm`` logger, so our
    records never reach the root handler pytest installs -- which is why the other
    tests here attach a handler to the logger itself.
    """
    import logging

    from vllm_ascend.kt_offload import layer as kl

    with caplog.at_level(logging.DEBUG):
        kl.logger.info("KTPROBE-caplog")
    assert "KTPROBE-caplog" not in caplog.text, (
        "caplog now sees our records; if the vllm logger's propagate changed, the "
        "our_logs fixture in test_graph_path.py can be simplified"
    )
