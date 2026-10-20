import json
import multiprocessing as mp

import pytest

from agents.base import build_agent
from core.config import JudgeConfig, ModelConfig, RunConfig
from core.events import EventWatcher
from core.manager import RuntimeManager
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
        validators=[{"name": "judge:rubric", "kwargs": {"rubric": "Does the trace say hi?"}}],
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

    assert (tmp_path / "logs" / "results" / "001.partial.json").exists()

    summary_lines = (tmp_path / "logs" / "aggregate_results.jsonl").read_text().strip().splitlines()
    assert len(summary_lines) == 1
    entry = json.loads(summary_lines[0])
    assert entry["id"] == 1
    assert entry["status"] == "terminated"
    assert entry["time_elapsed"]
    assert entry["last_event"]


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
        ' "validators": [{"name": "content:contains_str", "kwargs": {"expected": "done"}}]}\n'
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
    assert scores["failure_modes"]["correctness"] == {"score": 1.0, "pass_rate": 1.0, "passed": 1, "total": 1}
    # A mode no task declares is still reported, with no runs behind it.
    assert scores["failure_modes"]["sycophancy"] == {"score": None, "pass_rate": None, "passed": 0, "total": 0}

    summary = (tmp_path / "logs" / "run_summary.log").read_text()
    assert "overall" in summary
    assert "by failure mode" in summary
