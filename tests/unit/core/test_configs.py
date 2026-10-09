import pytest
from pydantic import ValidationError

from core.config import ModelConfig, RunConfig
from core.manifest import build_run_signature
from core.tasks import TaskDef


def _task_file(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"task_id": 1, "task": "say hi", "agent_id": "code_agent",'
        ' "failure_modes": {"quantitative": ["correctness"]}}\n'
    )
    return path


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")


def _model(**overrides):
    settings = {"provider": "litellm", "model_id": "fake", "api_key_env": "FAKE_KEY"}
    settings.update(overrides)
    return ModelConfig(**settings)


def _tasks():
    return [TaskDef(task_id=1, task="say hi", agent_id="code_agent", failure_modes={"quantitative": ["correctness"]})]


def _conf(tmp_path, monkeypatch, **overrides):
    monkeypatch.setenv("FAKE_KEY", "x")
    settings = {"model": _model(), "task_file": _task_file(tmp_path), "log_path": tmp_path / "logs"}
    settings.update(overrides)
    return RunConfig(**settings)


def test_model_max_context_defaults_to_128k():
    assert _model().model_max_context == 128_000


def test_model_max_context_can_be_extended():
    assert _model(model_max_context=1_000_000).model_max_context == 1_000_000


def test_model_max_context_must_be_positive():
    with pytest.raises(ValidationError):
        _model(model_max_context=0)


def test_run_output_settings_have_defaults(tmp_path, monkeypatch):
    conf = _conf(tmp_path, monkeypatch)
    assert conf.artifact_spill is True
    assert conf.tool_output_max_chars == 20_000


def test_tool_output_max_chars_must_be_positive(tmp_path, monkeypatch):
    with pytest.raises(ValidationError):
        _conf(tmp_path, monkeypatch, tool_output_max_chars=0)


def test_run_signature_tracks_the_context_window(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    task_file = _task_file(tmp_path)
    tasks = _tasks()

    small = build_run_signature(RunConfig(model=_model(), task_file=task_file, log_path=tmp_path / "logs"), tasks)
    large = build_run_signature(
        RunConfig(model=_model(model_max_context=1_000_000), task_file=task_file, log_path=tmp_path / "logs"), tasks
    )

    assert small["model"]["model_max_context"] == 128_000
    assert large["model"]["model_max_context"] == 1_000_000
    assert small != large


def test_run_signature_tracks_artifact_settings(tmp_path, monkeypatch):
    baseline = build_run_signature(_conf(tmp_path, monkeypatch), _tasks())
    assert baseline["artifact_spill"] is True
    assert baseline["tool_output_max_chars"] == 20_000

    changed = build_run_signature(
        _conf(tmp_path, monkeypatch, artifact_spill=False, tool_output_max_chars=1000), _tasks()
    )
    assert changed["artifact_spill"] is False
    assert changed["tool_output_max_chars"] == 1000
    assert changed != baseline
