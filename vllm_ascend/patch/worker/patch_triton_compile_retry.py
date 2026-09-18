"""DSV4.1: worker 内 Triton 编译故障的现场取证 + 重试。

🔴 现象:在 vLLM worker 进程里,triton-ascend 调 bishengir-compile 编新特化时间歇报
   "Failed to generate temporary directory" ×5 → "Failed to run BiShengIR pipeline",
   worker 死、整个 DP 组卡死。同一特化在独立进程里编译【总是成功】(09-18 实测:
   _write_hash_cache_kernel 的半截缓存目录在独立进程里被补全,缓存键与 worker 完全一致)。
   已排除:并发(独立 16 并发全绿)、/tmp 条目数、fd 上限(120/524288)、cwd 被删、
   TMPDIR/TMP/TEMP(全链路零写者,带阳性对照)、umask/rlimit(与 shell 逐项相同)、
   cgroup pids(上限 629145,峰值 2912,撞限 0 次)。根因未定位。
   而 _compute_slot_mapping_fused_groups_adaptive_kernel 把 NUM_REQS 作 constexpr,
   任何没见过的批内请求数都会触发编译 ⇒ 这是一颗随时会响的雷,预编译穷举不干净。

这里做两件事,且【分开记账】,好区分机制:
  ① 失败现场,在【worker 进程内】亲自建一次临时目录,记 errno/磁盘/fd/线程;
  ② 按原样重试(最多 3 次,退避);全失败再用一个全新的显式 TMPDIR 试最后一次。
  最终仍失败 ⇒ 照原语义抛 CalledProcessError(不吞错)。
关闭:VLLM_ASCEND_DSV41_TRITON_COMPILE_RETRY=0
"""
import os
import subprocess as _real_sp
import tempfile
import time

from vllm.logger import init_logger

logger = init_logger("vllm.dsv41.triton_compile_retry")  # 必须在 vllm.* 下,否则不输出
_MARK = b"Failed to generate temporary directory"
_STATS = {"hit": 0, "plain_retry_ok": 0, "fresh_tmpdir_ok": 0, "gave_up": 0}


def _probe() -> str:
    parts = []
    try:
        d = tempfile.mkdtemp(prefix="dsv41_probe_")
        os.rmdir(d)
        parts.append("worker内mkdtemp=OK")
    except OSError as e:  # noqa: PERF203
        parts.append(f"worker内mkdtemp=FAIL errno={e.errno}({os.strerror(e.errno)})")
    try:
        st = os.statvfs(tempfile.gettempdir())
        parts.append(f"tmp={tempfile.gettempdir()} 余量={st.f_bavail * st.f_frsize / 2**30:.1f}GiB "
                     f"inode余={st.f_favail}")
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
                                     ("TMPDIR", "TMP", "TEMP") if k in os.environ) or "<无>"))
    return " | ".join(parts)


_ORIG_RUN = _real_sp.run


def _is_bishengir(cmd) -> bool:
    return isinstance(cmd, (list, tuple)) and bool(cmd) and str(cmd[0]).endswith("bishengir-compile")


def _run(cmd, *args, **kwargs):
    # 只拦 bishengir-compile;其余 subprocess.run 原样放行(含 check 语义)。
    if not _is_bishengir(cmd):
        return _ORIG_RUN(cmd, *args, **kwargs)
    check = kwargs.pop("check", False)
    r = _ORIG_RUN(cmd, *args, **kwargs)
    out = (r.stderr or b"") + (r.stdout or b"")
    if r.returncode != 0 and _MARK in out:
        _STATS["hit"] += 1
        name = os.path.basename(str(cmd[1])) if len(cmd) > 1 else "?"
        logger.warning("DSV4.1 triton 编译临时目录故障 #%d pid=%d 输入=%s 现场: %s",
                       _STATS["hit"], os.getpid(), name, _probe())
        for i in range(3):
            time.sleep(1.0 * (2 ** i))
            r = _ORIG_RUN(cmd, *args, **kwargs)
            if r.returncode == 0:
                _STATS["plain_retry_ok"] += 1
                logger.warning("DSV4.1 triton 编译:原样重试第 %d 次成功 %s", i + 1, _STATS)
                break
        else:
            env = dict(kwargs.get("env") or os.environ)
            fresh = tempfile.mkdtemp(prefix="dsv41_tmpdir_")
            env.update(TMPDIR=fresh, TMP=fresh, TEMP=fresh)
            r = _ORIG_RUN(cmd, *args, **dict(kwargs, env=env))
            if r.returncode == 0:
                _STATS["fresh_tmpdir_ok"] += 1
                logger.warning("DSV4.1 triton 编译:原样重试 3 次失败,显式全新 TMPDIR=%s 成功 %s",
                               fresh, _STATS)
            else:
                _STATS["gave_up"] += 1
                logger.error("DSV4.1 triton 编译:全部重试失败 %s 现场: %s", _STATS, _probe())
    if check and r.returncode != 0:
        raise _real_sp.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
    return r


_run._dsv41_retry = True


def install() -> None:
    # 🔴 不能去改 triton.backends.ascend.compiler.subprocess:Triton 用
    #    spec_from_file_location + exec_module 按路径【另起一个模块对象】加载后端,
    #    import 得到的那个不是后端实际用的那个(09-18 实测:钩子"已安装",出事时栈里
    #    走的仍是真的 subprocess.run)。两个副本共用同一个 subprocess 模块 ⇒ 包在这里。
    if os.getenv("VLLM_ASCEND_DSV41_TRITON_COMPILE_RETRY", "1") == "0":
        return
    if getattr(_real_sp.run, "_dsv41_retry", False):
        return
    _real_sp.run = _run
    logger.warning("DSV4.1 triton 编译重试钩子已安装(subprocess.run 级) pid=%d", os.getpid())


install()
