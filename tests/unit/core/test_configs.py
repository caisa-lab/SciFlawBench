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


def test_model_max_context_defaults_to_auto_detection():
    assert _model().model_max_context is None


def test_model_max_context_can_be_extended():
    assert _model(model_max_context=1_000_000).model_max_context == 1_000_000


def test_model_max_context_must_be_positive():
    with pytest.raises(ValidationError):
        _model(model_max_context=0)


def test_max_new_tokens_has_a_default_and_must_be_positive():
    assert _model().max_new_tokens == 16_384
    assert _model(max_new_tokens=None).max_new_tokens is None
    with pytest.raises(ValidationError):
        _model(max_new_tokens=0)


def test_legacy_extra_kwargs_max_tokens_is_adopted():
    assert _model(extra_kwargs={"max_tokens": 2048}).max_new_tokens == 2048


def test_conflicting_max_tokens_is_rejected():
    with pytest.raises(ValidationError):
        _model(max_new_tokens=4096, extra_kwargs={"max_tokens": 2048})


def test_memory_offload_defaults(tmp_path, monkeypatch):
    conf = _conf(tmp_path, monkeypatch)
    assert conf.memory_offload is True
    assert conf.memory_digest_chars == 200
    assert conf.memory_index_max_chars == 4_000


def test_memory_limits_must_not_be_negative(tmp_path, monkeypatch):
    with pytest.raises(ValidationError):
        _conf(tmp_path, monkeypatch, memory_digest_chars=-1)


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

    assert small["model"]["model_max_context"] is None
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


def test_run_signature_tracks_generation_and_memory_settings(tmp_path, monkeypatch):
    baseline = build_run_signature(_conf(tmp_path, monkeypatch), _tasks())
    assert baseline["model"]["max_new_tokens"] == 16_384
    assert baseline["memory_offload"] is True

    for overrides in ({"memory_offload": False}, {"memory_digest_chars": 50}, {"memory_index_max_chars": 100}):
        assert build_run_signature(_conf(tmp_path, monkeypatch, **overrides), _tasks()) != baseline

    model = _model(max_new_tokens=1024)
    assert build_run_signature(_conf(tmp_path, monkeypatch, model=model), _tasks()) != baseline
