# SPDX-License-Identifier: Apache-2.0
"""Pin the ACL report thread to a PU kt's spinning workers do not own. Off by default.

Why this exists. Every ``_launch_host_func`` node is delivered by ONE thread
per process: torch_npu creates it on the first ``_subscribe_report`` and every
stream reuses it (no thread name, ``SCHED_OTHER``, unbound; its loop is
``while (!stop) aclrtProcessReport(100)``, where 100 ms is a blocking bound,
not a poll interval, and no env tunes it). Meanwhile kt's workers spin for
50 ms after finishing a job before they sleep on their condvar
(``kt-kernel/cpu_backend/worker_pool.cpp`` ``InNumaPool::worker_thread`` and
``NumaJobDistributor::worker_thread``), and decode hands them new work every
few hundred µs -- so in steady state every kt worker is a runnable spinner
hard-bound to a single PU (``kt_bind_worker``, ``hwloc_bitmap_singlify`` +
``HWLOC_CPUBIND_STRICT``). The unbound report thread has to find a CPU among
those; this switch moves it onto one kt does not own, without taking any core
away from kt (kt normally leaves PUs free).

Everything here is off unless ``VLLM_ASCEND_DSV41_KT_REPORT_CPU`` is set.

Two halves, deliberately separated:

  * resolving the target PU runs at weight-load time on the main thread, where
    a refusal is a refused start-up rather than a warning inside a graph node;
  * applying it runs inside the first host-func callback, because that is the
    only code of ours that ever executes ON the report thread.

The criterion reads the judged object twice: the kt PU set the model predicts
is compared against the affinities kt's own threads actually have
(``/proc/self/task/*/``), and the bind is followed by ``sched_getaffinity``
read-back. A bind that cannot be verified is reported as not in effect.
"""

from __future__ import annotations

import ctypes
import os
import re
from dataclasses import dataclass, field

from vllm.logger import logger

# kt names its threads "numa_<n>_t_<i>" (InNumaPool) and "numa_<n>_m_<i>"
# (NumaJobDistributor), in worker_pool.cpp.
KT_THREAD_NAME = re.compile(r"^numa_(\d+)_[tm]_(\d+)$")


class ReportAffinityError(RuntimeError):
    """The requested pinning cannot be resolved; start-up must refuse."""


# -- topology ---------------------------------------------------------------


@dataclass(frozen=True)
class Topology:
    """Cores of the machine, each as its sorted PU list, plus grouping.

    ``core_numa`` / ``core_l3`` are keyed by core id, and a core's id is its
    lowest PU: hwloc enumerates objects inside a cpuset in logical order,
    which for a sane x86 topology is ascending first-PU order, and that is the
    order ``kt_bind_worker`` indexes into. This identity is the one modelling
    assumption here, and it is cross-checked against kt's real threads before
    anything is pinned (``check_model_against_kt``).
    """

    core_pus: dict[int, tuple[int, ...]]
    core_numa: dict[int, int]
    core_l3: dict[int, int]

    def cores_of_numa(self, numa: int) -> list[int]:
        return sorted(c for c, n in self.core_numa.items() if n == numa)

    def numa_of_pu(self, pu: int) -> int | None:
        for core, pus in self.core_pus.items():
            if pu in pus:
                return self.core_numa[core]
        return None

    def all_pus(self) -> set[int]:
        return {pu for pus in self.core_pus.values() for pu in pus}


def parse_cpu_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(out)


def read_topology(sysfs: str = "/sys") -> Topology:
    """Build the core/L3/NUMA grouping from sysfs (the same facts hwloc reads)."""
    node_dir = os.path.join(sysfs, "devices/system/node")
    cpu_dir = os.path.join(sysfs, "devices/system/cpu")
    pu_numa: dict[int, int] = {}
    for name in os.listdir(node_dir):
        m = re.fullmatch(r"node(\d+)", name)
        if not m:
            continue
        with open(os.path.join(node_dir, name, "cpulist")) as fh:
            for pu in parse_cpu_list(fh.read()):
                pu_numa[pu] = int(m.group(1))

    core_pus: dict[int, tuple[int, ...]] = {}
    pu_core: dict[int, int] = {}
    core_l3: dict[int, int] = {}
    for name in sorted(os.listdir(cpu_dir)):
        m = re.fullmatch(r"cpu(\d+)", name)
        if not m:
            continue
        pu = int(m.group(1))
        sib_path = os.path.join(cpu_dir, name, "topology/thread_siblings_list")
        if not os.path.exists(sib_path):
            continue
        with open(sib_path) as fh:
            siblings = tuple(parse_cpu_list(fh.read()))
        core = siblings[0]
        core_pus[core] = siblings
        pu_core[pu] = core

    for core in core_pus:
        cache_root = os.path.join(cpu_dir, f"cpu{core}/cache")
        l3_id = core  # no L3 exposed: every core is its own domain
        if os.path.isdir(cache_root):
            for entry in sorted(os.listdir(cache_root)):
                level = os.path.join(cache_root, entry, "level")
                shared = os.path.join(cache_root, entry, "shared_cpu_list")
                if not (os.path.exists(level) and os.path.exists(shared)):
                    continue
                with open(level) as fh:
                    if fh.read().strip() != "3":
                        continue
                with open(shared) as fh:
                    l3_id = min(parse_cpu_list(fh.read()))
                break
        core_l3[core] = l3_id

    core_numa = {core: pu_numa.get(core, -1) for core in core_pus}
    return Topology(core_pus=core_pus, core_numa=core_numa, core_l3=core_l3)


# -- kt's binding rule, modelled ---------------------------------------------


def kt_threads_per_numa(cpuinfer_threads: int, threadpool_count: int, numa_nodes: list[int] | None) -> dict[int, int]:
    """How many kt threads each NUMA node gets, and hence which indices it binds.

    kt splits ``cpuinfer_threads`` over ``threadpool_count`` subpools with the
    remainder on the first subpools (``kt-kernel/python/experts_base.py``), and
    ``WorkerPool::init`` gives subpool *j* the start index ``sum(threads of
    earlier subpools on the SAME numa node)``. Inside a subpool, worker 0 is the
    NumaJobDistributor thread (bound with the start index) and workers 1..t-1
    are the InNumaPool threads (bound with ``i + start``), so a node with T
    threads in total occupies exactly the indices 0..T-1 of its own order.
    """
    if cpuinfer_threads <= 0 or threadpool_count <= 0:
        raise ReportAffinityError(
            f"kt_offload report pinning: cpuinfer_threads={cpuinfer_threads} / "
            f"threadpool_count={threadpool_count} is not a kt configuration."
        )
    nodes = list(numa_nodes) if numa_nodes is not None else list(range(threadpool_count))
    if len(nodes) != threadpool_count:
        raise ReportAffinityError(
            f"kt_offload report pinning: numa_nodes={nodes} does not match threadpool_count={threadpool_count}."
        )
    counts: dict[int, int] = {}
    for j in range(threadpool_count):
        per = cpuinfer_threads // threadpool_count + (1 if j < cpuinfer_threads % threadpool_count else 0)
        counts[nodes[j]] = counts.get(nodes[j], 0) + per
    return counts


def kt_core_order(topo: Topology, numa: int, striped: bool = True) -> list[int]:
    """The core order ``kt_bind_worker`` indexes into, for one NUMA node.

    Striped (kt's default, ``KT_CPU_BIND`` unset): the node's L3 domains are
    walked slot by slot, so worker *i* lands on domain ``i % ndomains``, slot
    ``i / ndomains`` -- one CCD per consecutive worker, because a single CCD
    saturates well below the node's memory bandwidth. With one domain or none,
    kt falls back to the dense order. Pack mode (``KT_CPU_BIND=pack``) is the
    dense order: core *i* of the node.
    """
    cores = topo.cores_of_numa(numa)
    if not striped:
        return cores
    domains: dict[int, list[int]] = {}
    for core in cores:
        domains.setdefault(topo.core_l3[core], []).append(core)
    ordered_domains = [sorted(domains[k]) for k in sorted(domains)]
    if len(ordered_domains) <= 1:
        return cores
    out: list[int] = []
    deepest = max(len(d) for d in ordered_domains)
    for slot in range(deepest):
        for dom in ordered_domains:
            if slot < len(dom):
                out.append(dom[slot])
    return out


def kt_bound_pus(topo: Topology, counts: dict[int, int], striped: bool = True) -> tuple[set[int], list[str]]:
    """PUs kt's workers are hard-bound to, and the indices that stay unbound.

    ``kt_bind_worker`` copies the core's cpuset and calls
    ``hwloc_bitmap_singlify``, which keeps the lowest bit, so a worker with
    ``pu_slot == 0`` gets the core's FIRST PU. Over-subscription
    (``index >= cores``) walks onto the core's SMT siblings, but only in
    striped mode; pack mode leaves such a worker unbound, which is a hole in
    any prediction and is reported rather than hidden.
    """
    pus: set[int] = set()
    unbound: list[str] = []
    for numa, count in sorted(counts.items()):
        order = kt_core_order(topo, numa, striped)
        if not order:
            unbound.append(f"numa {numa}: no cores in the topology")
            continue
        for index in range(count):
            if striped:
                core = order[index % len(order)]
                slot = index // len(order)
                core_pus = topo.core_pus[core]
                pus.add(core_pus[slot % len(core_pus)] if slot else core_pus[0])
            else:
                if index >= len(order):
                    unbound.append(f"numa {numa}: worker index {index} has no core (pack mode leaves it unbound)")
                    continue
                pus.add(topo.core_pus[order[index]][0])
    return pus, unbound


def observed_kt_pus(proc_task: str = "/proc/self/task") -> tuple[set[int], int]:
    """The PUs kt's own threads are currently bound to, read off /proc.

    Returns the union of their single-PU affinities and the number of kt
    threads seen. This is the judged object: the model above is a prediction
    about these threads, and it is only allowed to pick a PU once the two
    agree.
    """
    pus: set[int] = set()
    seen = 0
    if not os.path.isdir(proc_task):
        return pus, 0
    for tid in os.listdir(proc_task):
        comm_path = os.path.join(proc_task, tid, "comm")
        status_path = os.path.join(proc_task, tid, "status")
        try:
            with open(comm_path) as fh:
                comm = fh.read().strip()
            if not KT_THREAD_NAME.match(comm):
                continue
            with open(status_path) as fh:
                for line in fh:
                    if line.startswith("Cpus_allowed_list:"):
                        pus.update(parse_cpu_list(line.split(":", 1)[1]))
                        break
        except OSError:
            continue
        seen += 1
    return pus, seen


def check_model_against_kt(predicted: set[int], observed: set[int], observed_threads: int) -> str:
    """Refuse when the prediction and kt's real threads disagree.

    Returns a one-line note for the log. No kt thread visible (an older kt
    build without thread names, or a pool that has not been built yet) is a
    warning, not a pass: the note says the model is unverified.
    """
    if observed_threads == 0:
        return "kt threads not visible in /proc (no names?): the modelled PU set is UNVERIFIED"
    if observed != predicted:
        only_pred = sorted(predicted - observed)[:8]
        only_obs = sorted(observed - predicted)[:8]
        raise ReportAffinityError(
            "kt_offload report pinning: the modelled kt PU set does not match what kt's threads are "
            f"actually bound to ({observed_threads} threads). Predicted-only {only_pred}, "
            f"observed-only {only_obs}. Refusing to pick a 'free' PU from a model that is wrong about "
            "which PUs are busy -- set VLLM_ASCEND_DSV41_KT_REPORT_CPU to an explicit PU number instead."
        )
    return f"modelled kt PU set matches all {observed_threads} kt threads"


# -- choosing the PU ---------------------------------------------------------


def choose_report_pu(
    topo: Topology,
    kt_pus: set[int],
    allowed: set[int],
    device_numa: int | None = None,
) -> tuple[int, str]:
    """Pick a PU on a core kt owns no part of; refuse rather than guess.

    Only a core whose PUs are ALL free is eligible: an SMT sibling of a
    spinning worker shares the core's front end, which is what we are trying
    to get away from. Among those, the one kt's own order would reach LAST is
    taken, so raising ``cpuinfer_threads`` later collides with this PU last of
    all. The NUMA node of the device is preferred when it is known; nodes are
    otherwise tried in ascending order.
    """
    nodes = sorted({n for n in topo.core_numa.values() if n >= 0})
    if device_numa is not None and device_numa in nodes:
        nodes = [device_numa] + [n for n in nodes if n != device_numa]
    tried: list[str] = []
    for numa in nodes:
        order = kt_core_order(topo, numa)
        free = [
            core for core in order if not (set(topo.core_pus[core]) & kt_pus) and set(topo.core_pus[core]) <= allowed
        ]
        if not free:
            tried.append(f"numa {numa}: no core is both fully free of kt and allowed by this process's cpuset")
            continue
        core = free[-1]
        why = (
            f"numa {numa}, core {core} (PUs {list(topo.core_pus[core])}): last core in kt's own stripe order "
            f"that has no kt worker on it; kt holds {len(kt_pus)} PU(s)"
            + (" (device NUMA)" if numa == device_numa else "")
        )
        return topo.core_pus[core][0], why
    raise ReportAffinityError(
        "kt_offload report pinning: auto found no fully idle core to put the ACL report thread on "
        f"({'; '.join(tried)}). Lower cpuinfer_threads, or name a PU explicitly. Not falling back to an "
        "SMT sibling of a spinning kt worker: that would be a silent downgrade of the thing being tested."
    )


# -- the plan, resolved once at start-up -------------------------------------


@dataclass
class Plan:
    pu: int
    why: str
    fifo_prio: int | None = None
    # filled in by the callback
    applied: bool = False
    verified: bool = False
    fifo_applied: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"pu={self.pu} applied={self.applied} verified={self.verified} "
            f"fifo={self.fifo_prio} fifo_applied={self.fifo_applied} why={self.why}"
            + ("; " + "; ".join(self.notes) if self.notes else "")
        )


_PLAN: Plan | None = None
_RESOLVED = False


def _env(name: str) -> str:
    from vllm_ascend import envs

    return str(getattr(envs, name) or "").strip()


def _parse_fifo(value: str) -> int | None:
    if not value or value == "0":
        return None
    try:
        prio = int(value)
    except ValueError as exc:
        raise ReportAffinityError(
            f"VLLM_ASCEND_DSV41_KT_REPORT_FIFO={value!r} is not an integer priority (1-99, empty = off)."
        ) from exc
    if not 1 <= prio <= 99:
        raise ReportAffinityError(f"VLLM_ASCEND_DSV41_KT_REPORT_FIFO={prio} is outside the SCHED_FIFO range 1-99.")
    return prio


def build_plan(
    spec: str,
    cfg,
    *,
    fifo: str = "",
    topo: Topology | None = None,
    allowed: set[int] | None = None,
    device_numa: int | None = None,
    observed: tuple[set[int], int] | None = None,
) -> Plan | None:
    """Turn the env values into a Plan, or None when the switch is off."""
    spec = (spec or "").strip()
    if not spec:
        return None
    prio = _parse_fifo(fifo)
    topo = topo if topo is not None else read_topology()
    allowed = allowed if allowed is not None else set(os.sched_getaffinity(0))

    if spec.lower() != "auto":
        try:
            pu = int(spec)
        except ValueError as exc:
            raise ReportAffinityError(
                f"VLLM_ASCEND_DSV41_KT_REPORT_CPU={spec!r} is neither 'auto' nor a PU number."
            ) from exc
        if pu not in topo.all_pus():
            raise ReportAffinityError(f"VLLM_ASCEND_DSV41_KT_REPORT_CPU={pu} is not a PU of this machine.")
        if pu not in allowed:
            raise ReportAffinityError(
                f"VLLM_ASCEND_DSV41_KT_REPORT_CPU={pu} is outside this process's cpuset "
                f"({len(allowed)} PU(s) allowed); the bind would fail."
            )
        plan = Plan(pu=pu, why="explicit VLLM_ASCEND_DSV41_KT_REPORT_CPU", fifo_prio=prio)
        # An explicit PU is the operator's call, so a collision with kt is a
        # note, not a refusal -- but it must be said out loud.
        try:
            counts = kt_threads_per_numa(cfg.cpuinfer_threads, cfg.threadpool_count, cfg.numa_nodes)
            kt_pus, _ = kt_bound_pus(topo, counts, striped=kt_striped())
        except ReportAffinityError:
            kt_pus = set()
        if pu in kt_pus:
            plan.notes.append(f"PU {pu} is also modelled as a kt worker PU: the report thread will share it")
        return plan

    counts = kt_threads_per_numa(cfg.cpuinfer_threads, cfg.threadpool_count, cfg.numa_nodes)
    striped = kt_striped()
    kt_pus, unbound = kt_bound_pus(topo, counts, striped=striped)
    if unbound:
        raise ReportAffinityError(
            "kt_offload report pinning: kt leaves some workers unbound with this configuration "
            f"({unbound[0]}), so which PUs are busy is not predictable. Refusing 'auto'."
        )
    obs_pus, obs_n = observed if observed is not None else observed_kt_pus()
    note = check_model_against_kt(kt_pus, obs_pus, obs_n)
    pu, why = choose_report_pu(topo, kt_pus, allowed, device_numa)
    plan = Plan(pu=pu, why=f"auto: {why} [KT_CPU_BIND={'stripe' if striped else 'pack'}]", fifo_prio=prio)
    plan.notes.append(note)
    return plan


def kt_striped() -> bool:
    """kt's own reading of ``KT_CPU_BIND`` (``worker_pool.cpp``)."""
    value = os.environ.get("KT_CPU_BIND")
    if value is None:
        return True
    return value not in ("pack", "dense", "0")


def resolve_once(cfg) -> Plan | None:
    """Resolve the plan on the main thread, at weight-load time. Idempotent.

    A bad or unsatisfiable request raises here, where it refuses the start-up;
    inside the callback there is nowhere for an exception to go.
    """
    global _PLAN, _RESOLVED
    if _RESOLVED:
        return _PLAN
    _RESOLVED = True
    _PLAN = build_plan(
        _env("VLLM_ASCEND_DSV41_KT_REPORT_CPU"),
        cfg,
        fifo=_env("VLLM_ASCEND_DSV41_KT_REPORT_FIFO"),
        device_numa=device_numa_hint(),
    )
    if _PLAN is not None:
        logger.info("kt_offload report thread pinning planned: %s", _PLAN.summary())
    return _PLAN


def reset_for_tests() -> None:
    global _PLAN, _RESOLVED, _APPLIED
    _PLAN, _RESOLVED, _APPLIED = None, False, False


def device_numa_hint() -> int | None:
    """NUMA node of the NPU this worker drives, or None when it cannot be read.

    Best effort on purpose: it only orders the candidate nodes. Every Huawei
    NPU function reporting the same node is the only case that can be answered
    without mapping logical device ids to BDFs, which needs npu-smi and can
    hang; a mixed machine simply returns None and the choice falls back to
    ascending node order (and says so in the log).
    """
    nodes: set[int] = set()
    root = "/sys/bus/pci/devices"
    try:
        entries = os.listdir(root)
    except OSError:
        return None
    for name in entries:
        try:
            with open(os.path.join(root, name, "vendor")) as fh:
                if fh.read().strip().lower() != "0x19e5":  # Huawei
                    continue
            with open(os.path.join(root, name, "numa_node")) as fh:
                node = int(fh.read().strip())
        except (OSError, ValueError):
            continue
        if node >= 0:
            nodes.add(node)
    return next(iter(nodes)) if len(nodes) == 1 else None


# -- applying it, on the report thread itself --------------------------------

_APPLIED = False


# Indirected so a test can substitute a fake kernel: pinning is exactly the
# kind of thing whose "it worked" must come from a read-back, and the read-back
# has to be exercisable without a second CPU.
def _set_affinity(pus: set[int]) -> None:
    os.sched_setaffinity(0, pus)


def _get_affinity() -> set[int]:
    return set(os.sched_getaffinity(0))


def apply_in_callback() -> None:
    """Bind THIS thread as planned. Runs on the report thread; never raises.

    The eager path calls the same host function inline on the main thread
    (layer.py route()), and pinning the main thread to one PU is a different
    (and unwanted) experiment, so the main thread is skipped. With
    ``--enforce-eager`` the switch therefore does nothing, and says so once.
    """
    global _APPLIED
    if _APPLIED or _PLAN is None:
        return
    import threading

    if threading.current_thread() is threading.main_thread():
        logger.warning(
            "kt_offload: the host func ran on the main thread (eager path), so there is no ACL report "
            "thread to pin; VLLM_ASCEND_DSV41_KT_REPORT_CPU has no effect on this run."
        )
        _APPLIED = True
        _PLAN.notes.append("skipped: host func ran on the main thread (eager path)")
        return
    _APPLIED = True
    plan = _PLAN
    tid = _gettid()
    before = sorted(_get_affinity())
    try:
        _set_affinity({plan.pu})
    except OSError as exc:
        plan.notes.append(f"sched_setaffinity({plan.pu}) failed: {exc!r}")
        logger.warning(
            "kt_offload: could not pin the ACL report thread (tid %s) to PU %d: %r. Continuing unpinned.",
            tid,
            plan.pu,
            exc,
        )
    else:
        plan.applied = True
    after = sorted(_get_affinity())
    plan.verified = after == [plan.pu]
    if plan.applied and not plan.verified:
        plan.notes.append(f"read-back says {after}, not [{plan.pu}]: NOT in effect")
        logger.warning(
            "kt_offload: pinned the ACL report thread (tid %s) to PU %d but sched_getaffinity reads back "
            "%s; treating the pinning as NOT in effect.",
            tid,
            plan.pu,
            after,
        )
    if plan.fifo_prio is not None:
        plan.fifo_applied = _set_fifo(plan.fifo_prio, plan)
    logger.info(
        "kt_offload ACL report thread: tid=%s affinity %s -> %s (target PU %d, in_effect=%s, "
        "fifo=%s in_effect=%s) -- %s",
        tid,
        before,
        after,
        plan.pu,
        plan.verified,
        plan.fifo_prio,
        plan.fifo_applied,
        plan.why,
    )


def _gettid() -> int:
    try:
        import threading

        return threading.get_native_id()
    except Exception:  # noqa: BLE001 - diagnostics only
        return -1


class _SchedParam(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]


SCHED_FIFO = 1


def _set_fifo(prio: int, plan: Plan) -> bool:
    """SCHED_FIFO for this thread; a container without CAP_SYS_NICE just warns.

    Verified by reading the policy back (``sched_getscheduler``): a call that
    returns 0 and a policy that did not change would otherwise be recorded as
    a success.
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        param = _SchedParam(prio)
        rc = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
        if rc != 0:
            err = ctypes.get_errno()
            plan.notes.append(f"sched_setscheduler(SCHED_FIFO,{prio}) failed: errno {err} ({os.strerror(err)})")
            logger.warning(
                "kt_offload: SCHED_FIFO(%d) for the ACL report thread failed: errno %d (%s). This usually "
                "means the container has no CAP_SYS_NICE. Continuing with SCHED_OTHER.",
                prio,
                err,
                os.strerror(err),
            )
            return False
        policy = libc.sched_getscheduler(0)
        if policy != SCHED_FIFO:
            plan.notes.append(f"sched_getscheduler reads back policy {policy}, not SCHED_FIFO: NOT in effect")
            logger.warning(
                "kt_offload: sched_setscheduler returned 0 but the policy reads back as %d, not SCHED_FIFO; "
                "treating it as not in effect.",
                policy,
            )
            return False
        return True
    except Exception as exc:  # noqa: BLE001 - never break a graph node over a knob
        plan.notes.append(f"SCHED_FIFO attempt raised {exc!r}")
        logger.warning("kt_offload: SCHED_FIFO for the ACL report thread raised %r; continuing.", exc)
        return False


def status() -> dict:
    """What actually happened, for logging and diagnostics."""
    if _PLAN is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "pu": _PLAN.pu,
        "why": _PLAN.why,
        "applied": _PLAN.applied,
        "verified": _PLAN.verified,
        "fifo_prio": _PLAN.fifo_prio,
        "fifo_applied": _PLAN.fifo_applied,
        "notes": list(_PLAN.notes),
    }
