import hashlib
import json

import pytest
from pydantic import ValidationError

from core.config import ModelConfig, RunConfig
from core.manager import RuntimeManager
from core.task_ids import TASK_ID_PLACEHOLDER, hash_prompt, task_id_slug
from core.tasks import TaskDef


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


_FM = {"quantitative": ["correctness"]}


def test_task_id_slug_keeps_legacy_int_padding():
    assert task_id_slug(1) == "001"
    assert task_id_slug(42) == "042"


def test_task_id_slug_passes_string_ids_through():
    tid = _hash("prompt")
    assert task_id_slug(tid) == tid


def test_taskdef_accepts_a_hash_task_id():
    tid = _hash("some prompt")
    task = TaskDef(task_id=tid, task="some prompt", agent_id="code_agent", failure_modes=_FM)
    assert task.task_id == tid


def _make_run_config(tmp_path, task_file_content: str) -> RunConfig:
    task_file = tmp_path / "tasks.jsonl"
    task_file.write_text(task_file_content)
    return RunConfig(
        model=ModelConfig(provider="litellm", model_id="fake", api_key_env="FAKE_KEY"),
        task_file=task_file,
        log_path=tmp_path / "logs",
        restarting=True,
        max_concurrent=2,
        repetitions_per_task=1,
    )


def test_load_tasks_keeps_hash_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    prompt = "read the csv and average a column"
    tid = _hash(prompt)
    line = json.dumps({"task_id": tid, "task": prompt, "agent_id": "code_agent", "failure_modes": _FM})

    manager = RuntimeManager(_make_run_config(tmp_path, line + "\n"))

    assert [(t.task_id, t.repetition) for t in manager._pending] == [(tid, 1)]


def test_load_completed_recovers_hash_ids_from_result_filenames(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    prompt = "read the csv and average a column"
    tid = _hash(prompt)
    line = json.dumps({"task_id": tid, "task": prompt, "agent_id": "code_agent", "failure_modes": _FM})

    conf = _make_run_config(tmp_path, line + "\n")
    RuntimeManager(conf)  # a first run writes the run manifest

    results_dir = conf.log_path / "results"
    (results_dir / f"{tid}.jsonl").write_text("{}")

    manager = RuntimeManager(conf)

    assert manager.load_completed() == {(tid, 1)}
    assert manager._pending == []


def test_legacy_integer_ids_are_allowed():
    # ad-hoc/legacy tasks do not follow the hash convention and are not checked
    task = TaskDef(task_id=1, task="say hi", agent_id="code_agent", failure_modes=_FM)

    assert task.task_id == 1


def test_hash_id_mismatch_is_rejected():
    with pytest.raises(ValidationError):
        TaskDef(task_id=_hash("one prompt"), task="a different prompt", agent_id="code_agent", failure_modes=_FM)


def test_placeholder_is_kept_and_rendered_at_run_time():
    template = f"Read `data/tasks/assets/{TASK_ID_PLACEHOLDER}/measurements.csv`"
    tid = _hash(template)
    task = TaskDef(task_id=tid, task=template, agent_id="code_agent", failure_modes=_FM)

    # the stored prompt keeps the placeholder...
    assert TASK_ID_PLACEHOLDER in task.task
    # ...and the harness resolves it to the real id when running
    assert task.rendered_task == f"Read `data/tasks/assets/{tid}/measurements.csv`"


def test_hash_prompt_matches_taskdef_convention():
    assert hash_prompt("abc") == _hash("abc")


def test_fixed_id_with_prompt_md5_is_accepted_and_rendered():
    template = f"Read `data/tasks/assets/{TASK_ID_PLACEHOLDER}/measurements.csv`"
    task = TaskDef(
        task_id="SFB-0042", task=template, prompt_md5=_hash(template), agent_id="code_agent", failure_modes=_FM
    )

    assert task.task_id == "SFB-0042"
    assert task.rendered_task == "Read `data/tasks/assets/SFB-0042/measurements.csv`"


def test_fixed_id_with_a_stale_prompt_md5_is_rejected():
    with pytest.raises(ValidationError, match="prompt_md5"):
        TaskDef(
            task_id="SFB-0042",
            task="an edited prompt",
            prompt_md5=_hash("the original prompt"),
            agent_id="code_agent",
            failure_modes=_FM,
        )


@pytest.mark.parametrize("task_id", ["0042", 42, "", "-0042", "SFB 0042", "SFB/0042"])
def test_fixed_id_must_start_with_a_letter_and_be_file_safe(task_id):
    # a digit-only id would be read back from its result file name as an integer
    with pytest.raises(ValidationError, match="not a valid fixed id"):
        TaskDef(task_id=task_id, task="p", prompt_md5=_hash("p"), agent_id="code_agent", failure_modes=_FM)


def test_load_completed_recovers_fixed_ids_from_result_filenames(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_KEY", "x")
    prompt = "read the csv and average a column"
    line = json.dumps(
        {
            "task_id": "SFB-0042",
            "task": prompt,
            "prompt_md5": _hash(prompt),
            "agent_id": "code_agent",
            "failure_modes": _FM,
        }
    )

    conf = _make_run_config(tmp_path, line + "\n")
    RuntimeManager(conf)  # a first run writes the run manifest

    (conf.log_path / "results" / "SFB-0042.jsonl").write_text("{}")

    manager = RuntimeManager(conf)

    assert manager.load_completed() == {("SFB-0042", 1)}
    assert manager._pending == []
