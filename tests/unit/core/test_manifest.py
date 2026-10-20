import json

import pytest
from pydantic import ValidationError

from core.config import JudgeConfig, ModelConfig, RunConfig
from core.manifest import (
    MANIFEST_FILENAME,
    MANIFEST_SHA_FILENAME,
    OVERRIDE_ENV_VAR,
    build_run_manifest,
    build_run_signature,
    canonical_json,
    hash_rows,
    load_manifest_sha256,
    load_run_manifest,
    manifest_sha256,
    reproducibility_override,
    save_run_manifest,
    validate_resume,
)
from core.tasks import TaskDef
from evaluation.judge import judge_template_sha256

TASK_LINES = [
    '{"task_id": 1, "task": "say hi", "agent_id": "code_agent", "failure_modes": {"quantitative": ["correctness"]}}',
    '{"task_id": 2, "task": "say bye", "agent_id": "code_agent", "failure_modes": {"quantitative": ["correctness"]}}',
]

_FM = {"quantitative": ["correctness"]}


def _task_file(tmp_path, lines=None):
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(lines if lines is not None else TASK_LINES) + "\n")
    return path


def _tasks():
    return [TaskDef(task_id=1, task="say hi", agent_id="code_agent", failure_modes={"quantitative": ["correctness"]})]


def _conf(tmp_path, task_file, **overrides):
    settings = {
        "model": ModelConfig(provider="litellm", model_id="fake", api_key_env="FAKE_KEY"),
        "task_file": task_file,
        "log_path": tmp_path / "logs",
    }
    settings.update(overrides)
    return RunConfig(**settings)


def _make_manifest(tmp_path, conf, tasks):
    return build_run_manifest(conf=conf, tasks=tasks, run_signature=build_run_signature(conf, tasks))


def _judge(model_id: str) -> JudgeConfig:
    return JudgeConfig(model_id=model_id, api_key_env="JUDGE_KEY")


def test_canonical_json_ignores_key_order():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_hash_rows_is_content_sensitive_but_order_stable():
    rows = [{"id": 1, "task": "a", "agent_id": "code_agent"}]

    assert hash_rows(rows) == hash_rows([{"agent_id": "code_agent", "task": "a", "id": 1}])
    assert hash_rows(rows) != hash_rows([{"id": 1, "task": "b", "agent_id": "code_agent"}])


def test_save_run_manifest_writes_checksum_and_is_immutable(tmp_path):
    manifest = {"hello": "world"}

    path = save_run_manifest(tmp_path, manifest)

    assert path == tmp_path / MANIFEST_FILENAME
    assert load_run_manifest(tmp_path) == manifest
    assert load_manifest_sha256(tmp_path) == manifest_sha256(manifest)
    assert (tmp_path / MANIFEST_SHA_FILENAME).exists()

    with pytest.raises(FileExistsError):
        save_run_manifest(tmp_path, manifest)


def test_manifest_records_provenance_and_signature(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    conf = _conf(tmp_path, _task_file(tmp_path))

    manifest = _make_manifest(tmp_path, conf, _tasks())

    assert set(manifest) >= {
        "created_at",
        "git",
        "environment",
        "task_set",
        "model",
        "tools",
        "prompts",
        "budget",
        "dependencies",
        "run_signature",
    }
    assert manifest["task_set"]["n_tasks"] == 1
    assert manifest["task_set"]["file_sha256"] is not None
    assert manifest["run_signature"] == build_run_signature(conf, _tasks())


def test_manifest_never_leaks_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "super-secret-value")
    conf = _conf(tmp_path, _task_file(tmp_path))

    manifest = _make_manifest(tmp_path, conf, _tasks())

    assert "super-secret-value" not in canonical_json(manifest)
    assert "FAKE_KEY" not in canonical_json(manifest)


def test_run_signature_ignores_scheduling_knobs(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    task_file = _task_file(tmp_path)
    tasks = _tasks()

    baseline = build_run_signature(_conf(tmp_path, task_file, max_concurrent=2), tasks)
    rescheduled = build_run_signature(
        _conf(tmp_path, task_file, max_concurrent=8, logging_level=10, generate_trace_reports=True), tasks
    )

    assert baseline == rescheduled


def test_run_signature_changes_with_the_task_set(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    conf = _conf(tmp_path, _task_file(tmp_path))
    tasks = _tasks()

    extended = build_run_signature(
        conf,
        [*tasks, TaskDef(task_id=2, task="say bye", agent_id="code_agent", failure_modes=_FM)],
    )

    assert extended != build_run_signature(conf, tasks)


def test_run_signature_records_an_absent_judge_as_none(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    conf = _conf(tmp_path, _task_file(tmp_path))

    assert build_run_signature(conf, _tasks())["judge"] is None


def test_run_signature_changes_with_the_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setenv("JUDGE_KEY", "judge-secret-value")
    task_file = _task_file(tmp_path)
    tasks = _tasks()

    first = build_run_signature(_conf(tmp_path, task_file, judge=_judge("judge-one")), tasks)
    second = build_run_signature(_conf(tmp_path, task_file, judge=_judge("judge-two")), tasks)

    assert first["judge"]["model_id"] == "judge-one"
    assert first["judge"] != second["judge"]
    assert first["task_set_sha256"] == second["task_set_sha256"]  # only the judge changed


def test_judge_fingerprint_pins_the_prompt_template(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setenv("JUDGE_KEY", "judge-secret-value")
    conf = _conf(tmp_path, _task_file(tmp_path), judge=_judge("judge-one"), extra_kwargs={"temperature": 0.0})

    fingerprint = build_run_signature(conf, _tasks())["judge"]

    assert fingerprint["prompt_sha256"] == judge_template_sha256()
    assert fingerprint["extra_kwargs"] == {}


def test_manifest_pins_the_judge_without_leaking_its_key(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setenv("JUDGE_KEY", "judge-secret-value")
    conf = _conf(tmp_path, _task_file(tmp_path), judge=_judge("judge-one"))

    manifest = _make_manifest(tmp_path, conf, _tasks())

    assert manifest["judge"] == build_run_signature(conf, _tasks())["judge"]
    assert "judge-secret-value" not in canonical_json(manifest)
    assert "JUDGE_KEY" not in canonical_json(manifest)


def test_validate_resume_reports_a_changed_judge(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setenv("JUDGE_KEY", "judge-secret-value")
    task_file = _task_file(tmp_path)
    tasks = _tasks()
    save_run_manifest(tmp_path, _make_manifest(tmp_path, _conf(tmp_path, task_file, judge=_judge("judge-one")), tasks))

    resumed_signature = build_run_signature(_conf(tmp_path, task_file, judge=_judge("judge-two")), tasks)

    problems = validate_resume(tmp_path, resumed_signature)

    assert any("run signature differs" in problem for problem in problems)


def test_validate_resume_accepts_matching_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    conf = _conf(tmp_path, _task_file(tmp_path))
    signature = build_run_signature(conf, _tasks())
    save_run_manifest(tmp_path, _make_manifest(tmp_path, conf, _tasks()))

    assert validate_resume(tmp_path, signature) == []


def test_validate_resume_reports_signature_change(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    conf = _conf(tmp_path, _task_file(tmp_path))
    signature = build_run_signature(conf, _tasks())
    save_run_manifest(tmp_path, _make_manifest(tmp_path, conf, _tasks()))

    problems = validate_resume(tmp_path, {**signature, "closed_book": True})

    assert any("run signature differs" in problem for problem in problems)


def test_validate_resume_detects_a_tampered_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    conf = _conf(tmp_path, _task_file(tmp_path))
    signature = build_run_signature(conf, _tasks())
    save_run_manifest(tmp_path, _make_manifest(tmp_path, conf, _tasks()))

    path = tmp_path / MANIFEST_FILENAME
    data = json.loads(path.read_text())
    data["created_at"] = "2000-01-01T00:00:00+00:00"
    path.write_text(json.dumps(data))

    problems = validate_resume(tmp_path, signature)

    assert any("recorded checksum" in problem for problem in problems)


def test_validate_resume_without_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    conf = _conf(tmp_path, _task_file(tmp_path))

    problems = validate_resume(tmp_path, build_run_signature(conf, _tasks()))

    assert any("no run_manifest.json" in problem for problem in problems)


def test_reproducibility_override_reads_the_env_var(monkeypatch):
    monkeypatch.delenv(OVERRIDE_ENV_VAR, raising=False)
    assert reproducibility_override() is False

    monkeypatch.setenv(OVERRIDE_ENV_VAR, "true")
    assert reproducibility_override() is True


def test_task_def_validation_error_is_not_swallowed(tmp_path, monkeypatch):
    """Sanity check that the helpers above run against real TaskDef validation."""
    with pytest.raises(ValidationError):
        TaskDef(task_id="0" * 32, task="mismatched", agent_id="code_agent", failure_modes=_FM)
