import json
import multiprocessing as mp
import os
import signal
import time

import pytest

from agents.base import build_agent
from core.config import JudgeConfig, ModelConfig, RunConfig
from core.events import EventWatcher
from core.manager import RuntimeManager
from core.task_ids import hash_prompt
from core.tasks import TaskDef, run_task

FAKE_MODEL_KWARGS = {"responses": ["final_answer('done')"]}


def setup_basics(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")


def test_run_fake_task(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent", "tools": ["fake_search"],'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model",
            model_id="fake",
            api_key_env="FAKE_KEY",
            extra_kwargs={"responses": ["final_answer('done')"]},
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        repetitions_per_task=1,
    )

    import tests.fakes.presets
    import tests.fakes.tools

    with open(task_file) as f:
        raw_task = json.load(f)
        taskdef = TaskDef(**raw_task)

    log_path = tmp_path / "logs"
    queue = mp.Queue()

    run_task(taskdef, conf, log_path, queue)

    data = json.loads((log_path / f"{taskdef.task_id:03d}.jsonl").read_text())

    print(data["error"])
    assert data["success"] is True


def test_run_fake_task_with_a_fixed_id_records_prompt_md5(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        json.dumps(
            {
                "task_id": "SFB-0001",
                "task": "say hi",
                "prompt_md5": hash_prompt("say hi"),
                "agent_id": "fake_agent",
                "failure_modes": {"quantitative": ["correctness"]},
            }
        )
        + "\n"
    )
    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model", model_id="fake", api_key_env="FAKE_KEY", extra_kwargs=FAKE_MODEL_KWARGS
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        repetitions_per_task=1,
    )

    import tests.fakes.presets
    import tests.fakes.tools

    taskdef = TaskDef(**json.loads(task_file.read_text()))
    log_path = tmp_path / "logs"

    run_task(taskdef, conf, log_path, mp.Queue())

    data = json.loads((log_path / "SFB-0001.jsonl").read_text())
    assert data["success"] is True
    assert data["prompt_md5"] == hash_prompt("say hi")


def test_single_task_pipeline(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent", "tools": ["fake_search"],'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model",
            model_id="fake",
            api_key_env="FAKE_KEY",
            extra_kwargs={"responses": ["final_answer('done')"]},
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        max_concurrent=2,
        repetitions_per_task=1,
    )

    RuntimeManager(conf).run()

    data = json.loads((tmp_path / "logs" / "results" / "001.jsonl").read_text().strip())
    print(data["error"])
    assert data["success"] is True


def test_run_task_records_the_judge_score_and_justification(tmp_path, monkeypatch):
    """A `judge:rubric` validator grades the trace and its verdict lands in the result metadata."""
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    from evaluation.judge import JudgeVerdict, LiteLLMJudge

    seen = {}

    def fake_judge(self, rubric, trace):
        seen["rubric"] = rubric
        seen["trace"] = trace
        return JudgeVerdict(passed=True, justification="the trace supports it", raw_reply="{}")

    monkeypatch.setattr(LiteLLMJudge, "judge", fake_judge)

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model",
            model_id="fake",
            api_key_env="FAKE_KEY",
            extra_kwargs={"responses": ["final_answer('done')"]},
        ),
        judge=JudgeConfig(model_id="fake-judge", api_key_env="JUDGE_KEY"),
        task_file=task_file,
        log_path=tmp_path / "logs",
        repetitions_per_task=1,
    )

    import tests.fakes.presets  # noqa: F401
    import tests.fakes.tools  # noqa: F401

    task = TaskDef(
        task_id=1,
        task="say hi",
        agent_id="fake_agent",
        failure_modes={"quantitative": ["correctness"]},
        validators=[
            {"name": "judge:rubric", "kwargs": {"rubric": "Does the trace say hi?"}, "failure_mode": "correctness"}
        ],
    )

    output_dir = tmp_path / "logs" / "results"
    run_task(task, conf, output_dir, mp.Queue())

    result = json.loads((output_dir / "001.jsonl").read_text().strip())
    check = result["check_results"][0]

    assert check["passed"] is True
    assert check["justification"] == "the trace supports it"
    assert check["metadata"] == {"rubric": "Does the trace say hi?", "judge_model": "fake-judge"}
    # The judge is sent the rendered trace, not just the final answer.
    assert seen["rubric"] == "Does the trace say hi?"
    assert "Task Report" in seen["trace"]


def test_full_pipeline(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent", "tools": ["fake_search"],'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
        '{"task_id": 2, "task": "say bye", "agent_id": "fake_agent", "tools": ["fake_search"],'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model",
            model_id="fake",
            api_key_env="FAKE_KEY",
            extra_kwargs={"responses": ["final_answer('done')"]},
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        max_concurrent=2,
        repetitions_per_task=1,
    )

    RuntimeManager(conf).run()

    for task_id in (1, 2):
        data = json.loads((tmp_path / "logs" / "results" / f"{task_id:03d}.jsonl").read_text())
        print(data["error"])
        assert data["success"] is True


def test_pipeline_with_pressure(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi",  "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}  }\n'
        '{"task_id": 2, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
        '{"task_id": 3, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
        '{"task_id": 4, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
        '{"task_id": 5, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
        '{"task_id": 6, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
        '{"task_id": 7, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model",
            model_id="fake",
            api_key_env="FAKE_KEY",
            extra_kwargs={"responses": ["final_answer('done')"]},
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        max_concurrent=4,
        repetitions_per_task=1,
    )

    RuntimeManager(conf).run()

    for task_id in (1, 2, 3, 4, 5, 6, 7):
        data = json.loads((tmp_path / "logs" / "results" / f"{task_id:03d}.jsonl").read_text())
        print(data["error"])
        assert data["success"] is True


def test_success_task_writes_result_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n',
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model",
            model_id="fake",
            api_key_env="FAKE_KEY",
            extra_kwargs={"responses": ["""```python\nfinal_answer("hi")\n```"""]},
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        max_concurrent=4,
        task_timeout_s=1,
        repetitions_per_task=1,
    )

    RuntimeManager(conf).run()

    result = json.loads((tmp_path / "logs" / "results" / "001.jsonl").read_text())
    assert result["success"] is True

    summary_lines = (tmp_path / "logs" / "aggregate_results.jsonl").read_text().strip().splitlines()
    assert len(summary_lines) == 1
    entry = json.loads(summary_lines[0])
    assert entry["id"] == 1
    assert entry["status"] == "success"
    assert entry["time_elapsed"]
    assert entry["error"] == ""
    assert entry["checks"] == []


def test_timedout_tasks(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "hang too long", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model",
            model_id="fake",
            api_key_env="FAKE_KEY",
            extra_kwargs={"responses": ["thanks for bearing with the wait"], "timeout": 99.0},
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        max_concurrent=4,
        task_timeout_s=3,
        repetitions_per_task=1,
    )

    RuntimeManager(conf).run()

    assert (tmp_path / "logs" / "results" / "001.1.partial.json").exists()

    summary_lines = (tmp_path / "logs" / "aggregate_results.jsonl").read_text().strip().splitlines()
    assert len(summary_lines) == 1
    entry = json.loads(summary_lines[0])
    assert entry["id"] == 1
    assert entry["status"] == "terminated"
    assert entry["time_elapsed"]
    assert entry["last_event"]
    # killed while waiting on a model call that never returned: that counts against the model
    assert entry["last_event"]["event_type"] == "model_call_start"
    assert (entry["failure_kind"], entry["error_type"], entry["repetition"]) == ("model", "task_timeout", 1)
    scores = json.loads((tmp_path / "logs" / "scores.json").read_text())
    assert (scores["model_failures"], scores["overall"]["score"]) == (1, 0.0)


def test_timedout_repetitions_keep_their_own_partial_trace(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "hang too long", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model",
            model_id="fake",
            api_key_env="FAKE_KEY",
            extra_kwargs={"responses": ["thanks for bearing with the wait"], "timeout": 99.0},
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        max_concurrent=4,
        task_timeout_s=3,
        repetitions_per_task=3,
    )

    RuntimeManager(conf).run()

    results = tmp_path / "logs" / "results"
    assert sorted(path.name for path in results.glob("*.partial.json")) == [
        "001.1.partial.json",
        "001.2.partial.json",
        "001.3.partial.json",
    ]

    # Events carry no repetition number, but the run summary prints the last event of each killed
    # repetition: every file must end with the event reported for its own repetition.
    summary = (tmp_path / "logs" / "run_summary.log").read_text().splitlines()
    for repetition in (1, 2, 3):
        events = json.loads((results / f"001.{repetition}.partial.json").read_text())
        assert events and all(event["task_id"] == 1 for event in events)
        reported = [line for line in summary if f"task    1.{repetition} " in line]
        assert len(reported) == 1
        assert f"'timestamp': {events[-1]['timestamp']!r}" in reported[0]


def test_task_killed_before_its_first_event_is_still_reported(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]} }\n'
    )
    taskdef = TaskDef(**json.loads(task_file.read_text()))
    conf = _resume_conf(tmp_path, task_file)

    def killed_while_building(*args, **kwargs):
        # the manager's SIGTERM arrives while the agent is still being built, before any event exists
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(10)
        raise AssertionError("the SIGTERM handler did not end the task")

    monkeypatch.setattr("core.tasks.build_agent", killed_while_building)
    queue = mp.Queue()
    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(SystemExit):
            run_task(taskdef, conf, tmp_path, queue)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    message = queue.get(timeout=5)
    assert message["kind"] == "killed"
    assert message["to_log"]["last_event"] is None
    # nothing ran yet, so the kill is not held against the model
    assert message["to_log"]["failure_kind"] == "infrastructure"
    assert json.loads((tmp_path / "001.1.partial.json").read_text()) == []


def _resume_conf(tmp_path, task_file, **overrides):
    settings = {
        "model": ModelConfig(
            provider="fake_model", model_id="fake", api_key_env="FAKE_KEY", extra_kwargs=FAKE_MODEL_KWARGS
        ),
        "task_file": task_file,
        "log_path": tmp_path / "logs",
        "restarting": True,
        "repetitions_per_task": 1,
    }
    settings.update(overrides)
    return RunConfig(**settings)


def test_interrupted_run_resumes_and_finishes(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")
    monkeypatch.delenv("NO_REPRODUCIBILITY_GUARANTEES", raising=False)

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
        '{"task_id": 2, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )
    conf = _resume_conf(tmp_path, task_file)

    # an interrupted first run: the manifest is written, one result made it out
    interrupted = RuntimeManager(conf)
    run_task(
        TaskDef(task_id=1, task="say hi", agent_id="fake_agent", failure_modes={"quantitative": ["correctness"]}),
        conf,
        interrupted.results_dir,
        mp.Queue(),
    )

    resumed = RuntimeManager(conf)

    assert [t.task_id for t in resumed._pending] == [2]
    # the checkpoint's manifest is reused rather than replaced
    assert resumed.manifest_sha256 == interrupted.manifest_sha256

    resumed.run()

    assert (tmp_path / "logs" / "results" / "001.jsonl").exists()
    assert (tmp_path / "logs" / "results" / "002.jsonl").exists()


def test_resume_refuses_a_modified_task_set(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")
    monkeypatch.delenv("NO_REPRODUCIBILITY_GUARANTEES", raising=False)

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )
    conf = _resume_conf(tmp_path, task_file)

    RuntimeManager(conf)  # writes the run manifest
    (conf.log_path / "results" / "001.jsonl").write_text("{}")

    # a second task is added behind the manifest's back
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
        '{"task_id": 2, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )

    with pytest.raises(SystemExit):
        RuntimeManager(conf)


def test_resume_override_allows_a_modified_task_set(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")
    monkeypatch.setenv("NO_REPRODUCIBILITY_GUARANTEES", "true")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )
    conf = _resume_conf(tmp_path, task_file)

    RuntimeManager(conf)
    (conf.log_path / "results" / "001.jsonl").write_text("{}")
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
        '{"task_id": 2, "task": "say bye", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )

    manager = RuntimeManager(conf)  # no SystemExit thanks to the override

    assert [t.task_id for t in manager._pending] == [2]


def test_read_file_is_opt_in_for_agents(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    model = ModelConfig(provider="fake_model", model_id="fake", api_key_env="FAKE_KEY", extra_kwargs=FAKE_MODEL_KWARGS)

    def build(extra_tools, closed_book: bool = False):
        return build_agent(
            "code_agent",
            model,
            EventWatcher(task_id=1, sink=lambda event: None),
            {},
            extra_tools,
            closed_book=closed_book,
        )

    # not enabled by default ...
    assert "read_file" not in build([]).agent.tools

    # ... but available when an agent definition or task asks for it ...
    open_agent = build(["read_file"])
    assert "read_file" in open_agent.agent.tools
    assert "web_search" in open_agent.agent.tools

    # ... while a closed-book run gets no tools at all: smolagents keeps only the
    # agent's own Python execution (code agents) and its built-in `final_answer`
    closed_agent = build(["read_file"], closed_book=True)
    assert set(closed_agent.agent.tools) == {"final_answer"}

    # the tool-calling agent is left with nothing but `final_answer` as well
    closed_tool_agent = build_agent(
        "tool_agent", model, EventWatcher(task_id=1, sink=lambda event: None), {}, [], closed_book=True
    )
    assert set(closed_tool_agent.agent.tools) == {"final_answer"}


def test_closed_book_run_completes_without_tools(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi from memory", "agent_id": "fake_agent",'
        ' "extra_tools": [{"tool_name": "read_file"}],'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model", model_id="fake", api_key_env="FAKE_KEY", extra_kwargs=FAKE_MODEL_KWARGS
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        max_concurrent=1,
        repetitions_per_task=1,
        closed_book=True,
    )

    RuntimeManager(conf).run()

    result = json.loads((tmp_path / "logs" / "results" / "001.jsonl").read_text())
    assert result["success"] is True


def test_run_writes_overall_and_per_failure_mode_scores(tmp_path, monkeypatch):
    """A finished run writes scores.json and appends the score block to run_summary.log."""
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")

    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]},'
        ' "validators": [{"name": "content:contains_str", "kwargs": {"expected": "done"},'
        ' "failure_mode": "correctness"}]}\n'
    )

    conf = RunConfig(
        model=ModelConfig(
            provider="fake_model", model_id="fake", api_key_env="FAKE_KEY", extra_kwargs=FAKE_MODEL_KWARGS
        ),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        repetitions_per_task=1,
    )

    RuntimeManager(conf).run()

    scores = json.loads((tmp_path / "logs" / "scores.json").read_text())
    assert scores["overall"]["score"] == 1.0
    assert scores["failure_modes"]["correctness"] == {
        "score": 1.0,
        "pass_rate": 1.0,
        "passed": 1,
        "total": 1,
        "tasks": 1,
        "runs": 1,
        "excluded": 0,
    }
    # each check carries the failure mode it grades into the run records
    record = json.loads((tmp_path / "logs" / "aggregate_results.jsonl").read_text().splitlines()[0])
    assert [check["failure_mode"] for check in record["checks"]] == ["correctness"]
    # A mode no task declares is still reported, with no runs behind it.
    assert scores["failure_modes"]["sycophancy"] == {
        "score": None,
        "pass_rate": None,
        "passed": 0,
        "total": 0,
        "tasks": 0,
        "runs": 0,
        "excluded": 0,
    }

    summary = (tmp_path / "logs" / "run_summary.log").read_text()
    assert "overall" in summary
    assert "by failure mode" in summary


def _one_check_task(tmp_path):
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "fake_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]},'
        ' "validators": [{"name": "content:contains_str", "kwargs": {"expected": "done"},'
        ' "failure_mode": "correctness"}]}\n'
    )
    return task_file


def test_a_run_that_fails_because_of_the_model_scores_zero(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")
    model = ModelConfig(
        provider="fake_model",
        model_id="fake",
        api_key_env="FAKE_KEY",
        extra_kwargs={**FAKE_MODEL_KWARGS, "error": "context_length"},
    )

    RuntimeManager(_resume_conf(tmp_path, _one_check_task(tmp_path), model=model)).run()

    record = json.loads((tmp_path / "logs" / "aggregate_results.jsonl").read_text())
    assert (record["status"], record["failure_kind"], record["repetition"]) == ("failed", "model", 1)
    assert record["error_type"] == "BadRequestError"  # the provider's context-length error
    scores = json.loads((tmp_path / "logs" / "scores.json").read_text())
    assert (scores["model_failures"], scores["unscored_runs"]) == (1, 0)
    assert scores["overall"]["score"] == 0.0
    correctness = scores["failure_modes"]["correctness"]
    assert (correctness["score"], correctness["total"]) == (0.0, 1)


def test_an_error_while_building_the_agent_is_not_held_against_the_model(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    task_file = _one_check_task(tmp_path)
    taskdef = TaskDef(**json.loads(task_file.read_text()))

    def broken_build(*args, **kwargs):
        raise ValueError("no tool is registered under 'fake_search'")

    monkeypatch.setattr("core.tasks.build_agent", broken_build)
    queue = mp.Queue()
    run_task(taskdef, _resume_conf(tmp_path, task_file), tmp_path, queue)

    message = queue.get(timeout=5)
    to_log = message["to_log"]
    assert (to_log["status"], to_log["failure_kind"], to_log["error_type"]) == (
        "failed",
        "infrastructure",
        "ValueError",
    )


def test_an_infrastructure_failure_is_left_out_and_run_again_on_resume(tmp_path, monkeypatch):
    setup_basics(tmp_path, monkeypatch)
    monkeypatch.setenv("ENABLE_TEST_FAKES", "1")
    task_file = _one_check_task(tmp_path)
    out_of_credits = ModelConfig(
        provider="fake_model",
        model_id="fake",
        api_key_env="FAKE_KEY",
        extra_kwargs={**FAKE_MODEL_KWARGS, "error": "rate_limit"},
    )

    RuntimeManager(_resume_conf(tmp_path, task_file, model=out_of_credits)).run()

    scores = json.loads((tmp_path / "logs" / "scores.json").read_text())
    assert scores["overall"]["score"] is None
    assert scores["unscored_runs"] == 1
    [failure] = scores["infrastructure_failures"]
    assert (failure["task_id"], failure["repetition"], failure["error_type"]) == (1, 1, "RateLimitError")

    # once the provider is back, resuming runs the repetition again and its new record replaces the failure
    monkeypatch.setenv("NO_REPRODUCIBILITY_GUARANTEES", "true")  # the model configuration changed
    resumed = RuntimeManager(_resume_conf(tmp_path, task_file))
    assert [(task.task_id, task.repetition) for task in resumed._pending] == [(1, 1)]
    resumed.run()

    scores = json.loads((tmp_path / "logs" / "scores.json").read_text())
    assert (scores["runs"], scores["unscored_runs"], scores["infrastructure_failures"]) == (1, 0, [])
    assert scores["overall"]["score"] == 1.0
