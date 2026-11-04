import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.task_ids import hash_prompt
from core.tasks import TaskDef
from evaluation.base import kwargs_problem
from evaluation.definitions import verifier_registry

REPO_ROOT = Path(__file__).resolve().parents[3]


def _task(name, kwargs):
    return TaskDef(
        task_id=hash_prompt("p"),
        task="p",
        agent_id="code_agent",
        failure_modes={"quantitative": ["correctness"], "qualitative": []},
        validators=[{"name": name, "kwargs": kwargs, "failure_mode": "correctness"}],
    )


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("content:numeric_match", {"expected": "1", "tol": 0.1, "index": 0}),
        ("content:scientific_notation", {"expected": "1e5", "rel_tol": 0.1, "sig_figs": 2}),
        ("judge:rubric", {"rubric": "r"}),
        ("format:json_output", {}),
    ],
)
def test_kwargs_a_verifier_takes_load(name, kwargs):
    assert _task(name, kwargs).validators[0].kwargs == kwargs


@pytest.mark.parametrize(
    ("name", "kwargs", "reason"),
    [
        pytest.param("content:scientific_notation", {"expected": "1e5", "tol": 0.1}, "tol", id="misspelt"),
        pytest.param("format:json_output", {"expected": {}}, "expected", id="takes none"),
        pytest.param("content:numeric_match", {"tol": 0.1}, "expected", id="missing"),
        pytest.param("judge:rubric", {}, "rubric", id="missing rubric"),
        pytest.param("judge:rubric", {"rubric": "r", "context": "c"}, "context", id="context is not a kwarg"),
        pytest.param("content:numeric_match", {"got": "1", "expected": "1"}, "got", id="got is not a kwarg"),
    ],
)
def test_a_check_that_could_never_run_is_refused_at_load(name, kwargs, reason):
    with pytest.raises(ValidationError, match=reason):
        _task(name, kwargs)


def test_a_verifier_taking_anything_accepts_anything():
    assert kwargs_problem(lambda got, **kwargs: None, {"whatever": 1}) is None


@pytest.mark.parametrize("task_file", ["data/tasks/v0/tasks.jsonl", "data/example/tasks.jsonl"])
def test_every_check_in_the_task_files_can_run(task_file):
    tasks = [
        json.loads(line) for line in (REPO_ROOT / task_file).read_text(encoding="utf-8").splitlines() if line.strip()
    ]

    assert tasks
    for task in tasks:
        for validator in task["validators"]:
            assert kwargs_problem(verifier_registry.get(validator["name"]), validator["kwargs"]) is None, task[
                "task_id"
            ]
