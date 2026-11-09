# SPDX-License-Identifier: Apache-2.0
"""Which forward ends a prefill: ``prefill_window.observe``'s derivation.

Expectations come from the DEFINITION (a request is prefilling while
``num_computed_tokens < prompt_len``; the step that takes it to ``prompt_len``
is its last chunk), never read back off ``observe``, and the scheduler outputs
are the real dataclasses rather than stand-ins for the thing under test.
"""

from __future__ import annotations

import ast
import pathlib
from types import SimpleNamespace

import pytest

from vllm_ascend.kt_offload import prefill_window
from vllm_ascend.kt_offload.prefill_window import PrefillWindow

#: The repo root, for the source-level checks at the bottom of this file.
ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Both runner versions receive their schedule through this common worker.
WORKER = "vllm_ascend/worker/worker.py"


def new_req(req_id: str, prompt_len: int, computed: int = 0):
    """``NewRequestData`` as the scheduler emits it for a request's first step."""
    from vllm.v1.core.sched.output import NewRequestData

    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(range(prompt_len)),
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=computed,
        lora_request=None,
    )


def sched(new=(), cached=(), scheduled=None, finished=()):
    """A ``SchedulerOutput``. ``cached`` is ``[(req_id, num_computed_tokens), ...]``."""
    from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

    cached = list(cached)
    cached_data = CachedRequestData(
        req_ids=[r for r, _ in cached],
        resumed_req_ids=set(),
        new_token_ids=[[] for _ in cached],
        all_token_ids={},
        new_block_ids=[None for _ in cached],
        num_computed_tokens=[n for _, n in cached],
        num_output_tokens=[0 for _ in cached],
    )
    scheduled = dict(scheduled or {})
    return SchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=cached_data,
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
    )


def prefill_steps(req_id: str, prompt_len: int, budget: int):
    """The steps the scheduler produces for a chunked prefill, as
    ``[(scheduler_output, is_last_by_definition), ...]``. The truth column is
    computed here from the definition, not read off the code under test."""
    out, computed, first = [], 0, True
    while computed < prompt_len:
        n = min(budget, prompt_len - computed)
        step = (
            sched(new=[new_req(req_id, prompt_len, computed)], scheduled={req_id: n})
            if first
            else sched(cached=[(req_id, computed)], scheduled={req_id: n})
        )
        out.append((step, computed + n >= prompt_len))
        computed += n
        first = False
    return out


@pytest.fixture(autouse=True)
def armed():
    prefill_window.arm()
    yield


def test_only_the_last_chunk_of_a_prompt_is_marked_last():
    got = []
    for step, _ in prefill_steps("A", prompt_len=100, budget=30):
        got.append(prefill_window.observe(step))
    assert [w is not None for w in got] == [True] * 4
    assert [w.last for w in got] == [False, False, False, True]
    assert [w.first for w in got] == [True, False, False, False]
    assert [(w.computed, w.scheduled) for w in got] == [(0, 30), (30, 30), (60, 30), (90, 10)]
    assert {w.key for w in got} == {"A"} and [w.prompt_len for w in got] == [100] * 4
    assert [w.seq for w in got] == [1, 2, 3, 4], "each step must publish a fresh window"
    assert prefill_window.current() is got[-1]


def test_equal_sized_chunks_still_have_exactly_one_last():
    """The case the 'short last chunk' heuristic gets wrong."""
    steps = prefill_steps("A", prompt_len=4096, budget=2048)
    assert {s.num_scheduled_tokens["A"] for s, _ in steps} == {2048}, "every chunk is the same size"
    got = [prefill_window.observe(s) for s, _ in steps]
    assert [w.last for w in got] == [truth for _, truth in steps] == [False, True]


def test_remaining_is_what_the_prompt_has_left_after_this_forward():
    """``remaining`` from the definition; ``last`` is exactly ``remaining == 0``."""
    steps = prefill_steps("A", prompt_len=100, budget=30)
    got = [prefill_window.observe(s) for s, _ in steps]
    assert [w.remaining for w in got] == [70, 40, 10, 0]
    assert [w.last for w in got] == [w.remaining == 0 for w in got] == [False, False, False, True]


def test_the_short_last_chunk_is_visible_one_forward_early():
    """A tail of ``prompt_len mod MAXBATCH`` tokens is below
    ``stream_prefill_min_tokens`` and takes the hybrid path, never reaching the
    fold. The forward BEFORE it can see that coming, via ``remaining``."""
    maxbatch, min_tokens = 4096, 512  # max_num_batched_tokens / config default
    steps = prefill_steps("A", prompt_len=2 * maxbatch + 40, budget=maxbatch)
    got = [prefill_window.observe(s) for s, _ in steps]
    assert [w.scheduled for w in got] == [4096, 4096, 40]
    assert [w.remaining for w in got] == [4136, 40, 0]
    assert [w.last for w in got] == [False, False, True]
    # the last chunk does not stream, so "last" is not where a commit can land
    assert got[-1].scheduled < min_tokens
    # ... and "the rest of this prefill cannot stream" picks out the one before it
    assert [0 < w.remaining < min_tokens for w in got] == [False, True, False]


def test_a_prompt_whose_tail_is_long_enough_keeps_the_commit_on_the_last_chunk():
    """The other half of the control: r >= min_tokens, and nothing moves forward."""
    maxbatch, min_tokens = 4096, 512
    steps = prefill_steps("A", prompt_len=2 * maxbatch + 512, budget=maxbatch)
    got = [prefill_window.observe(s) for s, _ in steps]
    assert [w.remaining for w in got] == [4608, 512, 0]
    assert [0 < w.remaining < min_tokens for w in got] == [False, False, False]
    # the tail is exactly the threshold, and the gate is ``<``, so it streams
    assert got[-1].scheduled == min_tokens and not got[-1].scheduled < min_tokens


def test_a_decode_step_has_no_window():
    """Past the prompt there is nothing to accumulate and nothing to commit."""
    prefill_window.observe(sched(new=[new_req("A", 8)], scheduled={"A": 8}))
    after = prefill_window.observe(sched(cached=[("A", 8)], scheduled={"A": 1}))
    assert after is None and prefill_window.current() is None


def test_two_prefills_in_one_step_have_no_window(monkeypatch):
    """bs>1 is refused at start-up; if one ever gets here it is not attributable."""
    said = []
    monkeypatch.setattr(prefill_window.logger, "warning", lambda fmt, *a: said.append(fmt % a if a else fmt))
    step = sched(new=[new_req("A", 100), new_req("B", 100)], scheduled={"A": 30, "B": 30})
    assert prefill_window.observe(step) is None
    assert len(said) == 1 and "2 requests are prefilling" in said[0], said


def test_one_request_prefilling_beside_a_decoder_is_still_attributable():
    """Only the PREFILLING requests decide; a request past its prompt is not one."""
    prefill_window.observe(sched(new=[new_req("D", 8)], scheduled={"D": 8}))
    step = sched(new=[new_req("A", 100)], cached=[("D", 8)], scheduled={"A": 30, "D": 1})
    w = prefill_window.observe(step)
    assert w is not None and w.key == "A" and w.first and not w.last


def test_a_prefix_cache_hit_is_not_a_first_chunk():
    """``first`` is "nothing computed yet", which a cached prefix is not."""
    w = prefill_window.observe(sched(new=[new_req("A", 100, computed=64)], scheduled={"A": 36}))
    assert w is not None and w.first is False and w.last is True and w.computed == 64


def test_a_request_whose_prompt_length_was_never_seen_is_not_guessed(monkeypatch):
    """A cached request this process never saw as new: no length, so no window."""
    said = []
    monkeypatch.setattr(prefill_window.logger, "warning", lambda fmt, *a: said.append(fmt % a if a else fmt))
    assert prefill_window.observe(sched(cached=[("A", 10)], scheduled={"A": 30})) is None
    assert said and "no prompt length on record" in said[0]


def test_a_finished_request_is_forgotten():
    steps = prefill_steps("A", prompt_len=60, budget=30)
    for step, _ in steps:
        prefill_window.observe(step)
    assert "A" in prefill_window._prompt_len
    prefill_window.observe(sched(finished=["A"]))
    assert "A" not in prefill_window._prompt_len
    assert prefill_window.current() is None


def test_the_same_id_reused_takes_the_new_prompt_length():
    prefill_window.observe(sched(new=[new_req("A", 60)], scheduled={"A": 60}))
    # the step that finishes the old A and schedules a new request under the same id
    w = prefill_window.observe(sched(new=[new_req("A", 10)], scheduled={"A": 5}, finished=["A"]))
    assert w is not None and w.prompt_len == 10 and w.last is False


def test_mismatched_cached_lists_are_refused():
    """The two lists are paired by position; a length mismatch would pair them wrongly."""
    step = sched(cached=[("A", 1), ("B", 2)], scheduled={"A": 1})
    step.scheduled_cached_reqs.num_computed_tokens = [1]
    with pytest.raises(RuntimeError, match="parallel lists"):
        prefill_window.observe(step)


def test_the_end_of_a_prefill_is_reported_once():
    """Reported on the step AFTER the last prefill chunk, and only then. It has
    to fire from the per-step hook, not from a layer: the consumer must also see
    a prefill whose every forward was too short to stream."""
    ended = []
    prefill_window.on_prefill_end(ended.append)
    for step, _ in prefill_steps("A", prompt_len=60, budget=30):
        prefill_window.observe(step)
    assert ended == [], "not while the request is still prefilling"
    prefill_window.observe(sched(cached=[("A", 60)], scheduled={"A": 1}))  # first decode step
    assert ended == ["A"]
    prefill_window.observe(sched(cached=[("A", 61)], scheduled={"A": 1}))
    assert ended == ["A"], "one report per prefill, not one per step after it"


def test_a_prefill_that_only_ever_takes_the_hybrid_path_is_reported_too():
    """Every forward of this prompt is below the streaming gate, so nothing in
    the fold runs and this hook is the only signal the request went by."""
    ended = []
    prefill_window.on_prefill_end(ended.append)
    prefill_window.observe(sched(new=[new_req("A", 8)], scheduled={"A": 8}))  # < min_tokens
    prefill_window.observe(sched(cached=[("A", 8)], scheduled={"A": 1}))
    assert ended == ["A"]


def test_a_prefill_that_gives_way_to_another_request_is_reported():
    """The abandoned case: no decode step, another request simply takes over."""
    prefill_window.observe(prefill_steps("A", prompt_len=60, budget=30)[0][0])
    ended = []
    prefill_window.on_prefill_end(ended.append)
    prefill_window.observe(sched(new=[new_req("B", 30)], scheduled={"B": 30}))
    assert ended == ["A"], "A stopped prefilling, whatever the reason"
    prefill_window.observe(sched(cached=[("B", 30)], scheduled={"B": 1}))
    assert ended == ["A", "B"]


def test_the_same_hook_registered_twice_reports_once():
    """``install_resident_fold`` can run more than once in a process; the warning must not double."""
    ended = []
    prefill_window.on_prefill_end(ended.append)
    prefill_window.on_prefill_end(ended.append)
    prefill_window.observe(sched(new=[new_req("A", 8)], scheduled={"A": 8}))
    prefill_window.observe(sched(finished=["A"]))
    assert ended == ["A"]


def test_an_unfinished_prefill_is_not_reported():
    """Positive control: the report means "it ended", not "we stopped looking"."""
    ended = []
    prefill_window.on_prefill_end(ended.append)
    for step, _ in prefill_steps("A", prompt_len=90, budget=30)[:2]:
        prefill_window.observe(step)
    assert ended == []


def test_unarmed_observe_derives_nothing():
    """With dynamic_resident off the worker hook must cost a bool test."""
    prefill_window.reset()
    assert prefill_window.armed() is False
    step = sched(new=[new_req("A", 100)], scheduled={"A": 30})
    assert prefill_window.observe(step) is None
    assert prefill_window.current() is None and prefill_window._prompt_len == {}
    # ... and it still counts as "the hook is installed", which is a different
    # question from "the feature is on".
    assert prefill_window.require() is None


def test_require_refuses_when_nothing_ever_published():
    prefill_window.reset()
    with pytest.raises(RuntimeError, match="nothing has published a prefill window"):
        prefill_window.require()
    prefill_window.publish(None)
    assert prefill_window.require() is None  # the hook ran; this step just has no window


def test_publish_and_current_round_trip():
    w = PrefillWindow(key="A", first=True, last=False, computed=0, scheduled=4, prompt_len=9, seq=7)
    prefill_window.publish(w)
    assert prefill_window.current() is w and prefill_window.require() is w
    assert str(w) == "request A: tokens 0..4 of 9 (first) [step 7]"


def test_reset_forgets_everything():
    prefill_window.on_prefill_end(lambda key: None)
    prefill_window.observe(sched(new=[new_req("A", 100)], scheduled={"A": 30}))
    prefill_window.reset()
    assert prefill_window.current() is None and prefill_window.armed() is False
    assert prefill_window._prompt_len == {}
    assert prefill_window._end_hooks == [] and prefill_window._prefilling is None
    with pytest.raises(RuntimeError):
        prefill_window.require()


def _execute_model_source() -> str:
    """Read the common worker method without importing its NPU dependencies."""
    text = (ROOT / WORKER).read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == "NPUWorker":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "execute_model":
                    return ast.get_source_segment(text, fn)
    raise AssertionError(f"{WORKER} has no NPUWorker.execute_model to check")


def _worker_hook_problems(src: str) -> list[str]:
    """The schedule is published once, unconditionally, before runner dispatch."""
    fn = ast.parse(src).body[0]
    calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)]
    observes = [node for node in calls if ast.unparse(node.func) == "prefill_window.observe"]
    executes = [node for node in calls if ast.unparse(node.func) == "self.model_runner.execute_model"]
    problems = []
    if len(observes) != 1:
        return [f"{len(observes)} observe calls, expected exactly one"]
    if len(executes) != 1:
        return [f"{len(executes)} model-runner dispatch calls, expected exactly one"]
    observe, execute = observes[0], executes[0]
    if len(observe.args) != 1 or ast.unparse(observe.args[0]) != "scheduler_output" or observe.keywords:
        problems.append("observe does not receive this step's scheduler_output")
    top_level = {
        id(node)
        for statement in fn.body
        if not isinstance(statement, (ast.If, ast.For, ast.While, ast.Try, ast.With))
        for node in ast.walk(statement)
    }
    if id(observe) not in top_level:
        problems.append("observe is conditional instead of covering both runners")
    if id(execute) not in top_level:
        problems.append("model-runner dispatch is conditional")
    if observe.lineno >= execute.lineno:
        problems.append("observe runs after model-runner dispatch")
    return problems


def test_the_common_worker_publishes_before_either_runner():
    assert _worker_hook_problems(_execute_model_source()) == []


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda src: src.replace("prefill_window.observe(scheduler_output)", "pass", 1), "0 observe calls"),
        (
            lambda src: src.replace("prefill_window.observe(scheduler_output)", "prefill_window.observe(None)", 1),
            "this step's scheduler_output",
        ),
        (
            lambda src: src.replace(
                "        prefill_window.observe(scheduler_output)",
                "        if self.use_v2_model_runner:\n            prefill_window.observe(scheduler_output)",
                1,
            ),
            "conditional",
        ),
        (
            lambda src: src.replace("        prefill_window.observe(scheduler_output)\n", "", 1).replace(
                "        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)",
                "        output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)\n"
                "        prefill_window.observe(scheduler_output)",
                1,
            ),
            "after model-runner dispatch",
        ),
    ],
)
def test_worker_hook_mutations_are_red(mutation, expected):
    original = _execute_model_source()
    mutated = mutation(original)
    assert mutated != original, "mutation did not apply to the current worker"
    assert any(expected in problem for problem in _worker_hook_problems(mutated))


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_worker_hook_publishes_current_window_before_dispatch(use_v2_model_runner):
    """Execute the worker's actual hook/dispatch statements with either runner."""
    fn = ast.parse(_execute_model_source()).body[0]
    start = next(
        index
        for index, node in enumerate(fn.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "prefill_window.observe"
    )
    end = next(
        index
        for index, node in enumerate(fn.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "self.model_runner.execute_model"
    )
    dispatch = compile(ast.Module(body=fn.body[start : end + 1], type_ignores=[]), WORKER, "exec")
    seen = []
    intermediate = object()
    result = object()

    def execute(schedule, intermediate_tensors):
        window = prefill_window.require()
        seen.append((schedule, intermediate_tensors, window.key, window.scheduled))
        return result

    worker = SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner, model_runner=SimpleNamespace(execute_model=execute)
    )
    for request_id, prompt_len in (("A", 8), ("B", 16)):
        schedule = sched(new=[new_req(request_id, prompt_len)], scheduled={request_id: prompt_len})
        namespace = dict(
            prefill_window=prefill_window, self=worker, scheduler_output=schedule, intermediate_tensors=intermediate
        )
        exec(dispatch, namespace)
        assert namespace["output"] is result
        assert seen[-1] == (schedule, intermediate, request_id, prompt_len)


def test_the_install_arms_the_derivation():
    """``dynamic_resident`` is what switches ``observe`` on; nothing else does."""
    import inspect

    from vllm_ascend.kt_offload import layer as kl

    src = inspect.getsource(kl.install_resident_fold)
    assert "prefill_window.arm()" in src
    assert "prefill_window.arm()" not in inspect.getsource(kl.install_stream_provider)


def test_the_short_chunk_heuristic_would_disagree_with_the_definition():
    """Positive control: a case where the heuristic and the definition differ.
    If they never did, reading the scheduler output would be ceremony."""
    steps = prefill_steps("A", prompt_len=4096, budget=2048)
    budget = 2048
    heuristic = [s.num_scheduled_tokens["A"] < budget for s, _ in steps]
    truth = [t for _, t in steps]
    assert heuristic == [False, False] and truth == [False, True]
    got = [prefill_window.observe(s).last for s, _ in steps]
    assert got == truth != heuristic


def test_a_prompt_the_heuristic_gets_right_too():
    """The other half of the control: where they agree, the derivation agrees as well."""
    steps = prefill_steps("A", prompt_len=100, budget=30)
    heuristic = [s.num_scheduled_tokens["A"] < 30 for s, _ in steps]
    truth = [t for _, t in steps]
    assert heuristic == truth == [False, False, False, True]
    assert [prefill_window.observe(s).last for s, _ in steps] == truth


def test_scheduler_output_is_the_real_dataclass():
    """The fixtures above must be the objects production passes, not substitutes."""
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

    step = sched(new=[new_req("A", 4)], scheduled={"A": 4})
    assert isinstance(step, SchedulerOutput) and isinstance(step.scheduled_new_reqs[0], NewRequestData)
    assert step.scheduled_new_reqs[0].prompt_len == 4
    assert not isinstance(step, SimpleNamespace)
