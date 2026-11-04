import json
import sys
from pathlib import Path

import pytest

import utils.build_task as build_task_module
from agents.definitions import agent_registry
from core.task_ids import hash_prompt
from core.tasks import TaskDef
from evaluation.definitions import verifier_registry
from evaluation.scoring import ALL_FAILURE_MODES
from tools.definitions import tool_registry
from utils.build_task import (
    AGENTS,
    ALL_TOOLS,
    DEFAULT_TOOLS,
    FAILURE_MODES,
    OPT_IN_TOOLS,
    REPO_ROOT,
    VALIDATORS,
    append_task,
    build_task,
    check_task,
    check_task_file,
    normalize_sources,
    prompt_hash,
    read_tasks,
)

NOTEBOOK = REPO_ROOT / "utils" / "build_task.ipynb"
CHECK = {"name": "content:numeric_match", "kwargs": {"expected": "1.0"}, "failure_mode": "correctness"}


def _task(**overrides):
    """A valid task with a fixed id, with any field replaced."""
    task = build_task("What is 1?", "code_agent", ["correctness"], [], [], [CHECK], task_id="SFB-0001")
    task.update(overrides)
    return task


# The harness's own lists.


def test_the_lists_come_from_the_harness():
    assert list(ALL_FAILURE_MODES) == FAILURE_MODES
    # other tests can register fake agents and tools after this module was imported, so these are subsets
    assert {"code_agent", "tool_agent"} <= set(AGENTS) <= set(agent_registry.names())
    assert {"judge:rubric", "content:numeric_match"} <= set(VALIDATORS) <= set(verifier_registry.names())
    assert set(ALL_TOOLS) <= set(tool_registry.names())
    assert set(DEFAULT_TOOLS) | set(OPT_IN_TOOLS) == set(ALL_TOOLS)
    assert not set(DEFAULT_TOOLS) & set(OPT_IN_TOOLS)
    assert "web_search" in DEFAULT_TOOLS
    assert "read_file" in OPT_IN_TOOLS


# Building a task.


def test_the_id_defaults_to_the_md5_of_the_stripped_prompt():
    task = build_task("  What is 1?\n", "code_agent", ["correctness"], [], [], [CHECK])

    assert task["task"] == "What is 1?"
    assert task["task_id"] == hash_prompt("What is 1?")
    assert "prompt_md5" not in task


def test_a_fixed_id_comes_with_its_prompt_hash():
    task = _task()

    assert task["task_id"] == "SFB-0001"
    assert task["prompt_md5"] == hash_prompt("What is 1?")


def test_bare_tool_names_and_sources_are_normalised():
    task = build_task(
        "What is 1?",
        "code_agent",
        ["correctness"],
        [],
        ["read_file", {"tool_name": "numerics"}],
        [CHECK],
        ["https://arxiv.org/abs/1", {"url": " https://arxiv.org/abs/2 ", "note": " seen "}],
    )

    assert task["extra_tools"] == [{"tool_name": "read_file"}, {"tool_name": "numerics"}]
    assert task["sources"] == [
        {"url": "https://arxiv.org/abs/1", "note": ""},
        {"url": "https://arxiv.org/abs/2", "note": "seen"},
    ]
    assert normalize_sources(None) == []


@pytest.mark.parametrize("task_id", [None, "SFB-0001"])
def test_a_built_task_loads_in_the_harness(task_id):
    task = build_task("What is 1?", "code_agent", ["correctness"], [], [], [CHECK], task_id=task_id)

    assert check_task(task) == []
    TaskDef.model_validate(task)


# Checking a task.


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"task": " "}, "the prompt is empty"),
        ({"task_id": "0042", "prompt_md5": "x"}, "a fixed task_id must start with a letter"),
        ({"agent_id": "nobody"}, "agent_id must be one of"),
        ({"failure_modes": {"quantitative": ["nope"], "qualitative": []}}, "unknown failure mode(s) ['nope']"),
        ({"failure_modes": {"quantitative": ["sycophancy"], "qualitative": []}}, "wrong family"),
        ({"failure_modes": {"quantitative": [], "qualitative": []}}, "declare at least one failure mode"),
        ({"extra_tools": [{"tool_name": "nope"}]}, "unknown tool 'nope'"),
        ({"extra_tools": [{"tool_name": "web_search"}]}, "web_search is already available"),
        ({"extra_tools": ["read_file", "read_file"]}, "read_file is listed twice"),
        ({"validators": []}, "no validators"),
        (
            {"validators": [{"name": "content:numeric_match", "kwargs": {}}]},
            "each failure mode needs exactly one check",
        ),
        ({"validators": [{**CHECK, "name": "content:nope"}]}, "unknown verifier 'content:nope'"),
        ({"validators": [{**CHECK, "kwargs": {"expected": "1", "tolerance": 0.1}}]}, "can't run with its kwargs"),
        ({"validators": [{**CHECK, "kwargs": {}}]}, "can't run with its kwargs"),
        ({"sources": "https://arxiv.org"}, "sources must be a list"),
        ({"sources": ["https://arxiv.org"]}, "must be an object"),
        ({"sources": [{"url": "arxiv.org", "note": ""}]}, "must start with http"),
        ({"sources": [{"url": "https://arxiv.org", "note": 1}]}, "note must be a string"),
    ],
)
def test_check_task_reports_each_problem(overrides, expected):
    problems = check_task(_task(**overrides))

    assert any(expected in problem for problem in problems), problems


def test_check_task_wants_labels_in_the_order_of_the_modes():
    syc = {"name": "judge:rubric", "kwargs": {"rubric": "r"}, "failure_mode": "sycophancy"}
    modes = {"quantitative": ["correctness"], "qualitative": ["sycophancy"]}

    assert check_task(_task(failure_modes=modes, validators=[CHECK, syc])) == []
    assert any("exactly one check" in p for p in check_task(_task(failure_modes=modes, validators=[syc, CHECK])))


def test_a_prompt_with_the_placeholder_needs_its_assets_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(build_task_module, "REPO_ROOT", tmp_path)
    task = _task(task="Read data/tasks/assets/{TASK_ID}/a.csv")

    assert any("does not exist" in problem for problem in check_task(task))

    (tmp_path / "data" / "tasks" / "assets" / "SFB-0001").mkdir(parents=True)
    assert check_task(task) == []


# Task files.


def test_read_tasks_of_a_missing_file_is_empty(tmp_path):
    assert read_tasks(tmp_path / "missing.jsonl") == []


def test_append_task_writes_one_line_per_task_and_creates_the_folder(tmp_path):
    path = tmp_path / "new" / "tasks.jsonl"

    assert append_task(_task(), path) == 1
    assert append_task(_task(task_id="SFB-0002", task="What is 2?", prompt_md5=hash_prompt("What is 2?")), path) == 2

    assert [task["task_id"] for task in read_tasks(path)] == ["SFB-0001", "SFB-0002"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_append_task_keeps_non_ascii_text(tmp_path):
    path = tmp_path / "tasks.jsonl"
    prompt = "What is the mass of the Λ(1405)?"

    append_task(_task(task=prompt, prompt_md5=hash_prompt(prompt)), path)

    assert prompt in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param({"task": "What is 2?", "prompt_md5": hash_prompt("What is 2?")}, id="same-id"),
        pytest.param({"task_id": "SFB-0002"}, id="same-prompt"),
    ],
)
def test_append_task_refuses_a_duplicate(tmp_path, duplicate):
    path = tmp_path / "tasks.jsonl"
    append_task(_task(), path)

    with pytest.raises(ValueError, match="already in"):
        append_task(_task(**duplicate), path)

    assert len(read_tasks(path)) == 1


def test_prompt_hash_prefers_the_recorded_hash():
    assert prompt_hash({"task_id": "SFB-0001", "prompt_md5": "abc"}) == "abc"
    assert prompt_hash({"task_id": "abc"}) == "abc"


def test_check_task_file_accepts_a_valid_file(tmp_path):
    path = tmp_path / "tasks.jsonl"
    append_task(_task(), path)

    rows, duplicates, problems = check_task_file(path)

    assert (len(rows), duplicates, problems) == (1, [], [])


def test_check_task_file_reports_problems_by_line(tmp_path):
    path = tmp_path / "tasks.jsonl"
    edited = _task(task="What is 1, really?")
    no_agent = _task(task_id="SFB-0002", task="What is 2?", prompt_md5=hash_prompt("What is 2?"), agent_id="nobody")
    path.write_text(json.dumps(edited) + "\n" + json.dumps(no_agent) + "\n", encoding="utf-8")

    _, _, problems = check_task_file(path)

    assert any(p.startswith("line 1: prompt_md5 does not match") for p in problems)
    assert any(p.startswith("line 2: agent_id") for p in problems)


def test_check_task_file_reports_duplicate_ids(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(_task()) + "\n" + json.dumps(_task()) + "\n", encoding="utf-8")

    _, duplicates, _ = check_task_file(path)

    assert "SFB-0001" in duplicates


# The notebook.


def _code_cells():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def test_the_notebook_keeps_no_copy_of_the_module():
    source = "\n".join(_code_cells())

    assert "def " not in source
    assert "QUANTITATIVE_FAILURE_MODES =" not in source


@pytest.mark.parametrize("working_dir", [".", "utils"])
def test_the_notebook_runs_from_setup_to_recheck(tmp_path, monkeypatch, capsys, working_dir):
    monkeypatch.chdir(Path(REPO_ROOT) / working_dir)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "build_task", raising=False)
    fields = {
        "OUTPUT_FILE": tmp_path / "tasks.jsonl",
        "PROMPT": "What is 1?",
        "TASK_ID": "SFB-0001",
        "QUANT_FAILURE_MODES": ["correctness"],
        "VALIDATOR_DEFS": [CHECK],
        "SOURCES": ["https://arxiv.org/abs/1"],
    }

    namespace = {}
    for source in _code_cells():
        exec(compile(source, "build_task.ipynb", "exec"), namespace)
        if "# EDIT ME" in source:
            namespace.update(fields)

    output = capsys.readouterr().out
    assert "Looks good" in output
    assert "appended task #1" in output
    assert "All tasks are valid." in output
    assert [task["task_id"] for task in read_tasks(fields["OUTPUT_FILE"])] == ["SFB-0001"]
