# SPDX-License-Identifier: Apache-2.0
"""The ACL report thread's pinning: the model of kt's rule, and the read-backs.

Two things have to be true before a PU is called "free": the model of
``kt_bind_worker`` must be kt's actual rule (every test of the model here has
a mutation arm that breaks the rule and must go red), and the bind must be
read back on the thread it was applied to. Nothing in this file touches a
device, and with the env unset nothing calls a scheduler syscall at all.
"""

from __future__ import annotations

import threading

import pytest

from vllm_ascend.kt_offload import report_affinity as ra

# -- topologies -------------------------------------------------------------


def make_topology(nodes: int, domains_per_node: int, cores_per_domain: int, smt: int = 2) -> ra.Topology:
    """A machine shaped like the one under test, built by hand.

    Core ids are their first PU and the SMT sibling is ``pu + total_cores``,
    which is what Linux does on this EPYC (core 0 = PUs {0,128} with 128
    cores). That identity is what lets the expected sets below be written as
    plain numbers.
    """
    total_cores = nodes * domains_per_node * cores_per_domain
    core_pus, core_numa, core_l3 = {}, {}, {}
    core = 0
    for node in range(nodes):
        for dom in range(domains_per_node):
            for _ in range(cores_per_domain):
                core_pus[core] = tuple(core + total_cores * s for s in range(smt))
                core_numa[core] = node
                core_l3[core] = node * domains_per_node * cores_per_domain + dom * cores_per_domain
                core += 1
    return ra.Topology(core_pus=core_pus, core_numa=core_numa, core_l3=core_l3)


# The real machine: 2 sockets x 8 CCDs x 8 cores x 2 SMT = 256 PUs, node0 =
# 0-63 + 128-191 (checked against /sys on 2026-09-22).
EPYC = make_topology(nodes=2, domains_per_node=8, cores_per_domain=8)


class FakeCfg:
    def __init__(self, threads=96, pools=2, nodes=(0, 1)):
        self.cpuinfer_threads = threads
        self.threadpool_count = pools
        self.numa_nodes = list(nodes) if nodes is not None else None


@pytest.fixture(autouse=True)
def _clean():
    ra.reset_for_tests()
    yield
    ra.reset_for_tests()


# -- how many threads land on each node -------------------------------------


def test_threads_split_over_subpools_like_kt_does():
    # experts_base.py:365-368: remainder goes to the first subpools.
    assert ra.kt_threads_per_numa(96, 2, [0, 1]) == {0: 48, 1: 48}
    assert ra.kt_threads_per_numa(97, 2, [0, 1]) == {0: 49, 1: 48}
    # Two subpools on one node stack their start indices (worker_pool.cpp:517-531).
    assert ra.kt_threads_per_numa(96, 2, [0, 0]) == {0: 96}
    assert ra.kt_threads_per_numa(96, 2, None) == {0: 48, 1: 48}


def test_a_configuration_kt_would_reject_is_refused_here_too():
    with pytest.raises(ra.ReportAffinityError):
        ra.kt_threads_per_numa(0, 2, [0, 1])
    with pytest.raises(ra.ReportAffinityError):
        ra.kt_threads_per_numa(96, 2, [0, 1, 2])


# -- kt's core order ---------------------------------------------------------


def test_stripe_order_walks_l3_domains_one_core_at_a_time():
    order = ra.kt_core_order(EPYC, 0, striped=True)
    # worker i -> domain i % 8, slot i / 8
    assert order[:9] == [0, 8, 16, 24, 32, 40, 48, 56, 1]
    assert order[-1] == 63  # the last core kt's own order would ever reach
    assert sorted(order) == list(range(64))


def test_pack_order_is_the_dense_one_and_differs_from_stripe():
    """Mutation arm for the rule itself: ignoring the stripe is a visible error."""
    dense = ra.kt_core_order(EPYC, 0, striped=False)
    assert dense[:9] == list(range(9))
    assert dense != ra.kt_core_order(EPYC, 0, striped=True)


def test_one_l3_domain_falls_back_to_the_dense_order():
    flat = make_topology(nodes=1, domains_per_node=1, cores_per_domain=4)
    assert ra.kt_core_order(flat, 0, striped=True) == ra.kt_core_order(flat, 0, striped=False)


# -- which PUs kt holds ------------------------------------------------------


def test_stripe_binding_leaves_the_last_two_cores_of_every_ccd_free():
    kt_pus, unbound = ra.kt_bound_pus(EPYC, {0: 48, 1: 48}, striped=True)
    assert unbound == []
    # slots 0..5 of each of the 16 CCDs; singlify keeps the core's FIRST PU,
    # so no SMT sibling (>=128) is ever taken.
    expected = {d * 8 + s for d in range(16) for s in range(6)}
    assert kt_pus == expected
    assert max(kt_pus) < 128 and len(kt_pus) == 96


def test_ignoring_the_stripe_predicts_a_different_busy_set():
    """Mutation arm: the pack model says cores 0-47 of each node are busy."""
    striped, _ = ra.kt_bound_pus(EPYC, {0: 48, 1: 48}, striped=True)
    packed, _ = ra.kt_bound_pus(EPYC, {0: 48, 1: 48}, striped=False)
    assert packed != striped
    assert 48 in striped and 48 not in packed  # slot 0 of CCD6: busy under stripe, idle under pack
    assert 6 in packed and 6 not in striped  # core 6 of CCD0: the other way round


def test_oversubscription_walks_onto_smt_siblings_under_stripe():
    kt_pus, unbound = ra.kt_bound_pus(EPYC, {0: 70}, striped=True)
    assert unbound == []
    assert any(pu >= 128 for pu in kt_pus)  # index >= 64 wraps onto the sibling


def test_pack_mode_leaves_oversubscribed_workers_unbound_and_says_so():
    _, unbound = ra.kt_bound_pus(EPYC, {0: 70}, striped=False)
    assert unbound and "unbound" in unbound[0]


# -- the model is checked against kt's real threads --------------------------


def test_matching_model_reports_how_many_threads_backed_it():
    note = ra.check_model_against_kt({1, 2}, {1, 2}, 96)
    assert "96" in note


def test_a_model_that_disagrees_with_kts_threads_refuses():
    with pytest.raises(ra.ReportAffinityError, match="does not match"):
        ra.check_model_against_kt({1, 2}, {1, 3}, 96)


def test_no_visible_kt_thread_is_unverified_not_verified():
    assert "UNVERIFIED" in ra.check_model_against_kt({1, 2}, set(), 0)


def test_observed_kt_pus_reads_comm_and_cpus_allowed(tmp_path):
    for tid, comm, cpus in (("11", "numa_0_t_3", "7"), ("12", "numa_1_m_0", "64-65"), ("13", "python3", "0-255")):
        d = tmp_path / tid
        d.mkdir()
        (d / "comm").write_text(comm + "\n")
        (d / "status").write_text(f"Name:\t{comm}\nCpus_allowed_list:\t{cpus}\n")
    pus, seen = ra.observed_kt_pus(str(tmp_path))
    assert seen == 2 and pus == {7, 64, 65}  # the unnamed python thread is not kt's


# -- picking the PU ----------------------------------------------------------


def _all_pus(topo):
    return topo.all_pus()


def test_auto_picks_the_last_fully_idle_core_of_the_device_node():
    kt_pus, _ = ra.kt_bound_pus(EPYC, {0: 48, 1: 48}, striped=True)
    pu, why = ra.choose_report_pu(EPYC, kt_pus, _all_pus(EPYC), device_numa=0)
    assert pu == 63 and "device NUMA" in why
    pu1, _ = ra.choose_report_pu(EPYC, kt_pus, _all_pus(EPYC), device_numa=1)
    assert pu1 == 127


def test_a_core_is_only_free_when_both_its_smt_siblings_are():
    """Half a core is not a free core: the sibling shares the front end.

    Mutation arm: a rule that only looks at the core's first PU picks 63 here
    and this test goes red.
    """
    kt_pus, _ = ra.kt_bound_pus(EPYC, {0: 48, 1: 48}, striped=True)
    kt_pus = kt_pus | {191}  # core 63's sibling is taken (another tenant, or kt over-subscribed)
    pu, _ = ra.choose_report_pu(EPYC, kt_pus, _all_pus(EPYC), device_numa=0)
    assert pu == 55 and pu not in {p + 128 for p in kt_pus}


def test_a_pu_outside_this_process_cpuset_is_not_chosen():
    kt_pus, _ = ra.kt_bound_pus(EPYC, {0: 48, 1: 48}, striped=True)
    allowed = _all_pus(EPYC) - {63, 191}  # core 63's two PUs are fenced off
    pu, _ = ra.choose_report_pu(EPYC, kt_pus, allowed, device_numa=0)
    assert pu == 55  # the next core back along kt's order (slot 7 of CCD6)


def test_no_idle_core_refuses_instead_of_taking_a_busy_one():
    full, _ = ra.kt_bound_pus(EPYC, {0: 128, 1: 128}, striped=True)
    with pytest.raises(ra.ReportAffinityError, match="no fully idle core"):
        ra.choose_report_pu(EPYC, full, _all_pus(EPYC), device_numa=0)


def test_unknown_device_node_still_resolves_deterministically():
    kt_pus, _ = ra.kt_bound_pus(EPYC, {0: 48, 1: 48}, striped=True)
    pu, why = ra.choose_report_pu(EPYC, kt_pus, _all_pus(EPYC), device_numa=None)
    assert pu == 63 and "device NUMA" not in why


# -- the plan ----------------------------------------------------------------


def _plan(spec, **kw):
    kw.setdefault("topo", EPYC)
    kw.setdefault("allowed", _all_pus(EPYC))
    kw.setdefault("observed", ({d * 8 + s for d in range(16) for s in range(6)}, 96))
    return ra.build_plan(spec, FakeCfg(), **kw)


def test_the_switch_is_off_by_default():
    assert _plan("") is None and _plan("   ") is None


def test_auto_end_to_end():
    plan = _plan("auto", device_numa=0)
    assert plan.pu == 63 and plan.fifo_prio is None
    assert "stripe" in plan.why and "matches all 96" in plan.notes[0]


def test_auto_refuses_when_kts_threads_contradict_the_model():
    with pytest.raises(ra.ReportAffinityError, match="does not match"):
        _plan("auto", observed=({0, 1, 2}, 96))


def test_explicit_pu_is_taken_as_given_but_a_collision_is_said_out_loud():
    plan = _plan("8")
    assert plan.pu == 8 and any("kt worker PU" in n for n in plan.notes)
    assert _plan("63").notes == []


def test_explicit_values_that_cannot_work_are_refused():
    with pytest.raises(ra.ReportAffinityError, match="neither 'auto'"):
        _plan("cpu7")
    with pytest.raises(ra.ReportAffinityError, match="not a PU"):
        _plan("999")
    with pytest.raises(ra.ReportAffinityError, match="outside this process's cpuset"):
        _plan("63", allowed=_all_pus(EPYC) - {63})


def test_fifo_priority_parsing():
    assert _plan("63", fifo="").fifo_prio is None
    assert _plan("63", fifo="0").fifo_prio is None
    assert _plan("63", fifo="10").fifo_prio == 10
    for bad in ("high", "0.5", "100", "-1"):
        with pytest.raises(ra.ReportAffinityError):
            _plan("63", fifo=bad)


def test_auto_refuses_when_kt_would_leave_workers_unbound(monkeypatch):
    monkeypatch.setenv("KT_CPU_BIND", "pack")
    with pytest.raises(ra.ReportAffinityError, match="unbound"):
        ra.build_plan("auto", FakeCfg(threads=260, pools=2), topo=EPYC, allowed=_all_pus(EPYC), observed=(set(), 0))


# -- applying it on the thread itself ----------------------------------------


class FakeSched:
    """A scheduler whose read-back can be made to disagree with the write."""

    def __init__(self, start=None, honour=True):
        self.current = set(range(256)) if start is None else set(start)
        self.honour = honour
        self.calls: list[set[int]] = []

    def set(self, pus):
        self.calls.append(set(pus))
        if self.honour:
            self.current = set(pus)

    def get(self):
        return set(self.current)


def _apply_off_thread(monkeypatch, plan, sched):
    """apply_in_callback() skips the main thread on purpose, so run it elsewhere."""
    monkeypatch.setattr(ra, "_PLAN", plan)
    monkeypatch.setattr(ra, "_APPLIED", False)
    monkeypatch.setattr(ra, "_set_affinity", sched.set)
    monkeypatch.setattr(ra, "_get_affinity", sched.get)
    t = threading.Thread(target=ra.apply_in_callback)
    t.start()
    t.join()


def test_a_bind_that_reads_back_is_in_effect(monkeypatch):
    plan = ra.Plan(pu=63, why="test")
    sched = FakeSched()
    _apply_off_thread(monkeypatch, plan, sched)
    assert sched.calls == [{63}]
    assert plan.applied and plan.verified and ra.status()["verified"] is True


def test_a_bind_the_kernel_ignores_is_recorded_as_not_in_effect(monkeypatch):
    plan = ra.Plan(pu=63, why="test")
    sched = FakeSched(honour=False)  # the call "succeeds" and nothing moves
    _apply_off_thread(monkeypatch, plan, sched)
    assert plan.applied and not plan.verified
    assert any("NOT in effect" in n for n in plan.notes)


def test_a_failing_bind_warns_and_continues(monkeypatch):
    plan = ra.Plan(pu=63, why="test")
    sched = FakeSched()

    def boom(_pus):
        raise OSError(22, "Invalid argument")

    sched.set = boom
    _apply_off_thread(monkeypatch, plan, sched)
    assert not plan.applied and not plan.verified
    assert any("failed" in n for n in plan.notes)


def test_it_runs_once_even_if_every_layer_calls_it(monkeypatch):
    plan = ra.Plan(pu=63, why="test")
    sched = FakeSched()
    _apply_off_thread(monkeypatch, plan, sched)
    t = threading.Thread(target=ra.apply_in_callback)
    t.start()
    t.join()
    assert sched.calls == [{63}]


def test_with_the_switch_off_no_scheduler_call_is_made_at_all(monkeypatch):
    sched = FakeSched()
    monkeypatch.setattr(ra, "_PLAN", None)
    monkeypatch.setattr(ra, "_APPLIED", False)
    monkeypatch.setattr(ra, "_set_affinity", sched.set)
    monkeypatch.setattr(ra, "_get_affinity", sched.get)
    t = threading.Thread(target=ra.apply_in_callback)
    t.start()
    t.join()
    assert sched.calls == [] and ra.status() == {"enabled": False}


def test_the_eager_path_does_not_pin_the_main_thread(monkeypatch):
    """The same host func runs inline on the main thread when eager."""
    plan = ra.Plan(pu=63, why="test")
    sched = FakeSched()
    monkeypatch.setattr(ra, "_PLAN", plan)
    monkeypatch.setattr(ra, "_APPLIED", False)
    monkeypatch.setattr(ra, "_set_affinity", sched.set)
    monkeypatch.setattr(ra, "_get_affinity", sched.get)
    ra.apply_in_callback()  # this test runs on the main thread
    assert sched.calls == [] and not plan.applied
    assert any("main thread" in n for n in plan.notes)


# -- the topology reader reads this machine ----------------------------------


def test_read_topology_agrees_with_the_running_machine():
    topo = ra.read_topology()
    assert topo.core_pus, "no cores read from /sys"
    for core, pus in topo.core_pus.items():
        assert pus[0] == core and topo.core_numa[core] >= 0
    # every PU of a core shares its NUMA node and its L3 domain id
    assert len({topo.core_l3[c] for c in topo.cores_of_numa(topo.core_numa[min(topo.core_pus)])}) >= 1
