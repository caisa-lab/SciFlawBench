import json
import subprocess
from collections import Counter

import pytest
import yaml

from core.task_ids import hash_prompt
from evaluation.scoring import ALL_FAILURE_MODES
from utils.build_task import REPO_ROOT
from utils.task_stats import (
    BAR_WIDTH,
    END_MARKER,
    START_MARKER,
    main,
    read_task_set,
    render,
    update_readme,
)

WORKFLOW = REPO_ROOT / ".github" / "workflows" / "task-stats.yml"
RUBRIC = {"name": "judge:rubric", "kwargs": {"rubric": "r"}}
NUMBER = {"name": "content:numeric_match", "kwargs": {"expected": "1"}}


def _task(prompt="What is 1?", quantitative=("correctness",), qualitative=(), validators=None):
    if validators is None:
        validators = [{**NUMBER, "failure_mode": mode} for mode in quantitative]
        validators += [{**RUBRIC, "failure_mode": mode} for mode in qualitative]
    return {
        "task_id": hash_prompt(prompt),
        "task": prompt,
        "failure_modes": {"quantitative": list(quantitative), "qualitative": list(qualitative)},
        "validators": validators,
    }


def _row(block, title, label):
    """The row of one chart with this label."""
    lines = block[block.index(title) :].splitlines()
    return next(line for line in lines if line.strip().startswith(f"{label} ") or line.strip() == label)


def _count(row):
    return int(row.removesuffix("  ← needs tasks").split()[-1])


# The block.


def test_the_headline_counts_the_task_set():
    tasks = [_task(), _task("Read {TASK_ID}/a.csv", qualitative=("sycophancy",))]

    headline = render(tasks).splitlines()[0]

    assert "**2 accepted tasks**" in headline
    assert f"2 of {len(ALL_FAILURE_MODES)} failure modes covered" in headline
    assert "3 validators" in headline
    assert "1 tasks with files" in headline


def test_every_failure_mode_is_shown_even_with_no_tasks():
    block = render([_task()])

    for mode in ALL_FAILURE_MODES:
        assert mode in block
    assert _count(_row(block, "Tasks per failure mode", "code_safety")) == 0
    assert _row(block, "Tasks per failure mode", "code_safety").endswith("0  ← needs tasks")
    assert "█" not in _row(block, "Tasks per failure mode", "code_safety")


def test_only_a_failure_mode_with_no_tasks_is_marked():
    block = render([_task()])

    assert block.count("needs tasks") == len(ALL_FAILURE_MODES) - 1
    assert "needs tasks" not in _row(block, "Prompt length", "200+")


def test_modes_are_ranked_within_their_family():
    tasks = [_task("a", ("correct_tool_calls",)), _task("b", ("correct_tool_calls", "correctness")), _task("c")]

    block = render(tasks)

    assert block.index("correct_tool_calls ") < block.index("code_safety ")
    assert block.index("Quantitative") < block.index("Qualitative")


def test_the_longest_bar_has_the_full_width_and_a_small_count_still_shows():
    tasks = [_task(str(n)) for n in range(100)] + [_task("x", ("correct_tool_calls",))]

    block = render(tasks)

    assert _row(block, "Tasks per failure mode", "correctness").count("█") == BAR_WIDTH
    assert _row(block, "Tasks per failure mode", "correct_tool_calls").count("█") == 1


def test_validators_per_task_has_one_row_per_count_top_to_bottom():
    tasks = [_task("a"), _task("b", ("correctness", "correct_tool_calls", "code_safety"))]

    block = render(tasks)
    rows = block[block.index("Validators per task") :].splitlines()[1:4]

    assert [row.split()[0] for row in rows] == ["1", "2", "3"]
    assert [_count(row) for row in rows] == [1, 0, 1]


@pytest.mark.parametrize(
    ("words", "label"),
    [(1, "0-49"), (49, "0-49"), (50, "50-99"), (99, "50-99"), (100, "100-149"), (199, "150-199"), (200, "200+")],
)
def test_prompt_length_lands_in_its_bin(words, label):
    block = render([_task(" ".join(["word"] * words))])

    assert _count(_row(block, "Prompt length", label)) == 1


def test_prompt_length_has_one_row_per_bin_top_to_bottom():
    block = render([_task()])
    rows = block[block.index("Prompt length") :].splitlines()[1:6]

    assert [row.split()[0] for row in rows] == ["0-49", "50-99", "100-149", "150-199", "200+"]


@pytest.mark.parametrize(
    ("task", "label"),
    [
        (_task(), "automatic validators only"),
        (_task(quantitative=(), qualitative=("sycophancy",)), "judge validators only"),
        (_task(qualitative=("sycophancy",)), "judge and automatic validators"),
        (_task(validators=[]), "no validators"),
    ],
)
def test_each_task_is_counted_by_how_it_is_graded(task, label):
    assert _count(_row(render([task]), "How tasks are graded", label)) == 1


def test_no_validators_is_only_shown_when_a_task_has_none():
    assert "no validators" not in render([_task()])


def test_the_table_lists_each_modes_validators_by_count():
    tasks = [_task("a"), _task("b"), _task("c", validators=[{**RUBRIC, "failure_mode": "correctness"}])]

    table = render(tasks)

    assert "| `correctness` | `content:numeric_match` 2, `judge:rubric` 1 |" in table
    assert "| `code_safety` | none yet |" in table


def test_it_says_validators_and_never_checks():
    assert "check" not in render([_task(), _task("b", qualitative=("sycophancy",))]).casefold()


def test_a_tasks_text_never_reaches_the_block():
    prompt = "SECRET ``` <!-- task-stats:end --> <img src=x>"

    block = render([_task(prompt, qualitative=("sycophancy",))])

    for text in ("SECRET", "<img", END_MARKER):
        assert text not in block
    assert block.count("```") == 2


def test_an_empty_task_set_has_a_block():
    assert render([]) == "No accepted tasks yet."


# The real task set.


def test_the_charts_add_up_on_the_real_task_set():
    tasks = read_task_set()
    block = render(tasks)
    plots = block[block.index("```text") : block.rindex("```")]
    charts = plots.split("\n\n")
    graded = next(chart for chart in charts if chart.startswith("How tasks are graded"))
    per_task = next(chart for chart in charts if chart.startswith("Validators per task"))
    lengths = next(chart for chart in charts if chart.startswith("Prompt length"))

    assert tasks
    for chart in (graded, per_task, lengths):
        assert sum(_count(row) for row in chart.splitlines()[1:]) == len(tasks)
    counts = Counter(mode for task in tasks for group in task["failure_modes"].values() for mode in group)
    for mode in ALL_FAILURE_MODES:
        assert _count(_row(block, "Tasks per failure mode", mode)) == counts[mode]


def test_the_example_tasks_do_not_count(tmp_path):
    (tmp_path / "data" / "example").mkdir(parents=True)
    (tmp_path / "data" / "example" / "tasks.jsonl").write_text(json.dumps(_task()) + "\n", encoding="utf-8")
    (tmp_path / "data" / "tasks" / "v0").mkdir(parents=True)
    (tmp_path / "data" / "tasks" / "v0" / "tasks.jsonl").write_text(json.dumps(_task("b")) + "\n", encoding="utf-8")

    assert [task["task"] for task in read_task_set(tmp_path)] == ["b"]


def test_every_version_counts(tmp_path):
    for version in ("v0", "v1"):
        (tmp_path / "data" / "tasks" / version).mkdir(parents=True)
        path = tmp_path / "data" / "tasks" / version / "tasks.jsonl"
        path.write_text(json.dumps(_task(version)) + "\n", encoding="utf-8")

    assert [task["task"] for task in read_task_set(tmp_path)] == ["v0", "v1"]


# The README.


def test_only_what_is_between_the_markers_changes():
    readme = f"before\n{START_MARKER}\nold\n{END_MARKER}\nafter\n"

    assert update_readme(readme, "new") == f"before\n{START_MARKER}\nnew\n{END_MARKER}\nafter\n"


def test_updating_twice_changes_nothing_more():
    readme = f"before\n{START_MARKER}\n{END_MARKER}\nafter\n"

    once = update_readme(readme, "new")

    assert update_readme(once, "new") == once


@pytest.mark.parametrize("readme", ["no markers", START_MARKER, END_MARKER])
def test_a_readme_without_both_markers_stops_it(readme):
    with pytest.raises(SystemExit, match="needs both"):
        update_readme(readme, "new")


def test_the_real_readme_has_the_markers():
    assert update_readme((REPO_ROOT / "README.md").read_text(encoding="utf-8"), "new")


def _repo_with_tasks(path):
    (path / "data" / "tasks" / "v0").mkdir(parents=True)
    (path / "data" / "tasks" / "v0" / "tasks.jsonl").write_text(json.dumps(_task()) + "\n", encoding="utf-8")
    (path / "README.md").write_text(f"{START_MARKER}\n{END_MARKER}\n", encoding="utf-8")


def test_running_it_writes_the_readme(tmp_path):
    _repo_with_tasks(tmp_path)

    main([], repo=tmp_path)

    assert "**1 accepted tasks**" in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_print_shows_the_block_and_writes_nothing(tmp_path, capsys):
    _repo_with_tasks(tmp_path)

    main(["--print"], repo=tmp_path)

    assert "**1 accepted tasks**" in capsys.readouterr().out
    assert "accepted" not in (tmp_path / "README.md").read_text(encoding="utf-8")


# The workflow.


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_the_workflow_only_runs_after_a_merge_that_changes_tasks():
    triggers = _workflow()[True]  # YAML reads the key `on` as true

    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"] == {"branches": ["main"], "paths": ["data/tasks/**"]}


def test_the_workflow_can_only_write_contents_and_uses_no_secrets():
    assert _workflow()["permissions"] == {"contents": "write"}
    assert "secrets." not in WORKFLOW.read_text(encoding="utf-8")


def _git(path, *args):
    return subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True).stdout


def _commit_step():
    steps = _workflow()["jobs"]["update"]["steps"]
    return next(step["run"] for step in steps if step["name"] == "Commit if the numbers changed")


@pytest.fixture
def clone(tmp_path):
    """A clone of a shared remote whose main has a README, as the workflow's checkout would be."""
    _git(tmp_path, "init", "--quiet", "--bare", "--initial-branch=main", "origin.git")
    _git(tmp_path, "clone", "--quiet", "origin.git", "work")
    work = tmp_path / "work"
    _git(work, "switch", "--quiet", "-c", "main")
    _git(work, "config", "user.name", "someone")
    _git(work, "config", "user.email", "someone@example.org")
    (work / "README.md").write_text("old\n", encoding="utf-8")
    _git(work, "add", ".")
    _git(work, "commit", "--quiet", "-m", "Start")
    _git(work, "push", "--quiet", "origin", "main")
    return work


def test_the_commit_step_does_nothing_when_the_numbers_did_not_change(clone):
    head = _git(clone, "rev-parse", "HEAD")

    subprocess.run(["bash", "-e", "-c", _commit_step()], cwd=clone, check=True, capture_output=True)

    assert _git(clone, "rev-parse", "HEAD") == head


def test_the_commit_step_commits_only_the_readme_and_pushes(clone):
    (clone / "README.md").write_text("new\n", encoding="utf-8")
    (clone / "other.txt").write_text("stray\n", encoding="utf-8")

    subprocess.run(["bash", "-e", "-c", _commit_step()], cwd=clone, check=True, capture_output=True)

    assert _git(clone, "show", "--name-only", "--format=%s", "origin/main").split("\n")[:3] == [
        "Update the task stats in the README",
        "",
        "README.md",
    ]
    assert _git(clone, "log", "-1", "--format=%an", "origin/main").strip() == "github-actions[bot]"


def test_the_commit_step_survives_another_merge_landing_first(clone, tmp_path):
    _git(tmp_path, "clone", "--quiet", "origin.git", "other")
    other = tmp_path / "other"
    _git(other, "config", "user.name", "someone")
    _git(other, "config", "user.email", "someone@example.org")
    (other / "merged.txt").write_text("merged\n", encoding="utf-8")
    _git(other, "add", ".")
    _git(other, "commit", "--quiet", "-m", "Another merge")
    _git(other, "push", "--quiet", "origin", "HEAD:main")
    (clone / "README.md").write_text("new\n", encoding="utf-8")

    subprocess.run(["bash", "-e", "-c", _commit_step()], cwd=clone, check=True, capture_output=True)

    assert _git(clone, "log", "-2", "--format=%s", "origin/main").split("\n")[:2] == [
        "Update the task stats in the README",
        "Another merge",
    ]
