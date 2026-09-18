"""Retry bishengir-compile when it fails to create its temporary directory.

Inside vLLM worker processes, triton-ascend's bishengir-compile intermittently
fails with "Failed to generate temporary directory", which kills the worker and
hangs the DP group; the same compile succeeds in a standalone process. The root
cause is not known. Kernels with a batch-dependent constexpr can trigger a new
compile at any point during serving, so precompiling cannot rule it out.

On that failure: log the worker's temp-dir state, retry up to 3 times with
backoff, then once more with a fresh explicit TMPDIR. If all attempts fail the
original ``check=True`` semantics are kept (CalledProcessError is raised).
Disable with VLLM_ASCEND_DSV41_TRITON_COMPILE_RETRY=0.
"""
import os
import subprocess as _real_sp
import tempfile
import time

from vllm.logger import init_logger

# Must live under the "vllm." logger hierarchy to be emitted.
logger = init_logger("vllm.dsv41.triton_compile_retry")
_MARK = b"Failed to generate temporary directory"


def _probe() -> str:
    """Temp-dir state of this process, for the failure log."""
    parts = []
    try:
        d = tempfile.mkdtemp(prefix="dsv41_probe_")
        os.rmdir(d)
        parts.append("mkdtemp=OK")
    except OSError as e:  # noqa: PERF203
        parts.append(f"mkdtemp=FAIL errno={e.errno}({os.strerror(e.errno)})")
    try:
        st = os.statvfs(tempfile.gettempdir())
        parts.append(f"tmp={tempfile.gettempdir()} free={st.f_bavail * st.f_frsize / 2**30:.1f}GiB "
                     f"free_inodes={st.f_favail}")
    except OSError as e:
        parts.append(f"statvfs FAIL {e}")
    try:
        parts.append(f"fd={len(os.listdir('/proc/self/fd'))}")
        with open("/proc/self/status") as f:
            thr = next((l.split()[1] for l in f if l.startswith("Threads:")), "?")
        parts.append(f"threads={thr}")
    except OSError:
        pass
    parts.append("TMP*=" + (",".join(f"{k}={os.environ[k]}" for k in
                                     ("TMPDIR", "TMP", "TEMP") if k in os.environ) or "<unset>"))
    return " | ".join(parts)


_ORIG_RUN = _real_sp.run


def _as_bytes(x) -> bytes:
    if x is None:
        return b""
    return x.encode("utf-8", "replace") if isinstance(x, str) else bytes(x)


def _is_bishengir(cmd) -> bool:
    return isinstance(cmd, (list, tuple)) and bool(cmd) and str(cmd[0]).endswith("bishengir-compile")


def _run(cmd, *args, **kwargs):
    # Only bishengir-compile is intercepted; every other call passes through unchanged.
    if not _is_bishengir(cmd):
        return _ORIG_RUN(cmd, *args, **kwargs)
    check = kwargs.pop("check", False)
    r = _ORIG_RUN(cmd, *args, **kwargs)
    # Leave successful calls untouched and normalise to bytes before matching:
    # triton-ascend's capability probes run `bishengir-compile --help` with
    # text=True, and any exception here would make them report False and change
    # the compile options.
    if r.returncode != 0 and _MARK in _as_bytes(r.stderr) + _as_bytes(r.stdout):
        name = os.path.basename(str(cmd[1])) if len(cmd) > 1 else "?"
        logger.warning("bishengir-compile could not create its temp dir (pid=%d, input=%s): %s",
                       os.getpid(), name, _probe())
        for i in range(3):
            time.sleep(1.0 * (2 ** i))
            r = _ORIG_RUN(cmd, *args, **kwargs)
            if r.returncode == 0:
                logger.warning("bishengir-compile succeeded on retry %d", i + 1)
                break
        else:
            env = dict(kwargs.get("env") or os.environ)
            fresh = tempfile.mkdtemp(prefix="dsv41_tmpdir_")
            env.update(TMPDIR=fresh, TMP=fresh, TEMP=fresh)
            r = _ORIG_RUN(cmd, *args, **dict(kwargs, env=env))
            if r.returncode == 0:
                logger.warning("bishengir-compile succeeded with a fresh TMPDIR=%s after 3 retries", fresh)
            else:
                logger.error("bishengir-compile failed on every retry: %s", _probe())
    if check and r.returncode != 0:
        raise _real_sp.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    return r


_run._dsv41_retry = True


def install() -> None:
    # Patch subprocess itself, not triton.backends.ascend.compiler.subprocess: Triton loads its
    # backend via spec_from_file_location, so the imported compiler module is not the one it uses.
    # Both copies share the same subprocess module.
    if os.getenv("VLLM_ASCEND_DSV41_TRITON_COMPILE_RETRY", "1") == "0":
        return
    if getattr(_real_sp.run, "_dsv41_retry", False):
        return
    _real_sp.run = _run
    logger.warning("Installed bishengir-compile retry hook on subprocess.run (pid=%d)", os.getpid())


install()
