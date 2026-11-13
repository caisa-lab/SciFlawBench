"""
Write the task set's statistics into the README, between its task-stats markers.

Run it from the repo root with `python -m utils.task_stats`, or add `--print` to see the block without writing it. The
`task-stats` workflow runs it after each merge that changes `data/tasks/`. Only numbers and the harness's own names
reach the README, never a task's text.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluation.scoring import QUALITATIVE_FAILURE_MODES, QUANTITATIVE_FAILURE_MODES

REPO_ROOT = Path(__file__).resolve().parents[1]
START_MARKER = "<!-- task-stats:start -->"
END_MARKER = "<!-- task-stats:end -->"
BAR_WIDTH = 30
PROMPT_LENGTH_BINS = ((0, 49), (50, 99), (100, 149), (150, 199), (200, None))


def read_task_set(repo: Path = REPO_ROOT) -> list[dict]:
    """Every accepted task, across the versions in `data/tasks/`. The example tasks don't count."""
    tasks = []
    for path in sorted((repo / "data" / "tasks").glob("*/tasks.jsonl")):
        tasks += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tasks


def _modes(task: dict) -> list[str]:
    modes = task.get("failure_modes") or {}
    return [*modes.get("quantitative", []), *modes.get("qualitative", [])]


def _bar(count: int, largest: int) -> str:
    """A bar scaled to the largest count of its chart. Any count above zero shows at least one block."""
    return "█" * max(1, round(count / largest * BAR_WIDTH))


def _chart_rows(rows: list[tuple[str, int]], width: int, largest: int, zero_note: str = "") -> list[str]:
    lines = []
    for label, count in rows:
        value = f"{_bar(count, largest)} {count}" if count else f"0{zero_note}"
        lines.append(f"  {label:<{width}}  {value}")
    return lines


def _chart(title: str, rows: list[tuple[str, int]]) -> list[str]:
    width = max(len(label) for label, _ in rows)
    largest = max(count for _, count in rows)
    return [title, *_chart_rows(rows, width, largest)]


def _ranked(modes: tuple | list, counts: Counter) -> list[tuple[str, int]]:
    return sorted(((mode, counts[mode]) for mode in modes), key=lambda row: (-row[1], row[0]))


def _failure_mode_chart(tasks: list[dict]) -> list[str]:
    counts = Counter(mode for task in tasks for mode in _modes(task))
    quantitative = _ranked(QUANTITATIVE_FAILURE_MODES, counts)
    qualitative = _ranked(QUALITATIVE_FAILURE_MODES, counts)
    width = max(len(mode) for mode, _ in [*quantitative, *qualitative])
    largest = max(count for _, count in [*quantitative, *qualitative])
    note = "  ← needs tasks"
    lines = ["Tasks per failure mode", "", "Quantitative", *_chart_rows(quantitative, width, largest, note)]
    return [*lines, "", "Qualitative", *_chart_rows(qualitative, width, largest, note)]


def _grading_chart(tasks: list[dict]) -> list[str]:
    counts = Counter()
    for task in tasks:
        names = [validator["name"] for validator in task.get("validators") or []]
        judged = sum(name.startswith("judge:") for name in names)
        if not names:
            counts["no validators"] += 1
        elif judged == len(names):
            counts["judge validators only"] += 1
        elif judged:
            counts["judge and automatic validators"] += 1
        else:
            counts["automatic validators only"] += 1
    labels = ["judge and automatic validators", "judge validators only", "automatic validators only"]
    labels += ["no validators"] if counts["no validators"] else []
    return _chart("How tasks are graded", [(label, counts[label]) for label in labels])


def _validators_per_task_chart(tasks: list[dict]) -> list[str]:
    counts = Counter(len(task.get("validators") or []) for task in tasks)
    return _chart("Validators per task", [(str(n), counts[n]) for n in range(min(counts), max(counts) + 1)])


def _prompt_length_chart(tasks: list[dict]) -> list[str]:
    lengths = [len(str(task.get("task") or "").split()) for task in tasks]
    rows = []
    for low, high in PROMPT_LENGTH_BINS:
        label = f"{low}-{high}" if high is not None else f"{low}+"
        rows.append((label, sum(low <= length and (high is None or length <= high) for length in lengths)))
    return _chart("Prompt length (words)", rows)


def _validators_by_mode_table(tasks: list[dict]) -> list[str]:
    used: dict[str, Counter] = {}
    for task in tasks:
        for validator in task.get("validators") or []:
            used.setdefault(validator.get("failure_mode"), Counter())[validator["name"]] += 1
    lines = ["| Failure mode | Validators |", "|---|---|"]
    for mode in [*QUANTITATIVE_FAILURE_MODES, *QUALITATIVE_FAILURE_MODES]:
        names = sorted(used.get(mode, Counter()).items(), key=lambda row: (-row[1], row[0]))
        text = ", ".join(f"`{name}` {count}" for name, count in names) or "none yet"
        lines.append(f"| `{mode}` | {text} |")
    return lines


def render(tasks: list[dict]) -> str:
    """
    Build the README block.

    Args:
        tasks (list[dict]): the accepted tasks

    Returns (str): the block, in Markdown
    """
    if not tasks:
        return "No accepted tasks yet."
    all_modes = [*QUANTITATIVE_FAILURE_MODES, *QUALITATIVE_FAILURE_MODES]
    covered = {mode for task in tasks for mode in _modes(task)} & set(all_modes)
    validators = sum(len(task.get("validators") or []) for task in tasks)
    with_files = sum("{TASK_ID}" in str(task.get("task") or "") for task in tasks)
    charts = [
        _failure_mode_chart(tasks),
        _grading_chart(tasks),
        _validators_per_task_chart(tasks),
        _prompt_length_chart(tasks),
    ]
    plots = "\n\n".join("\n".join(chart) for chart in charts)
    table = "\n".join(_validators_by_mode_table(tasks))
    return (
        f"**{len(tasks)} accepted tasks** · {len(covered)} of {len(all_modes)} failure modes covered · "
        f"{validators} validators · {with_files} tasks with files\n\n"
        f"```text\n{plots}\n```\n\n"
        f"<details>\n<summary>Which validators grade each failure mode</summary>\n\n{table}\n\n</details>"
    )


def update_readme(readme: str, block: str) -> str:
    """
    Replace what is between the markers, and nothing else.

    Args:
        readme (str): the README
        block (str): the new block

    Returns (str): the updated README
    """
    before, start, rest = readme.partition(START_MARKER)
    _, end, after = rest.partition(END_MARKER)
    if not start or not end:
        raise SystemExit(f"The README needs both {START_MARKER} and {END_MARKER}.")
    return f"{before}{start}\n{block}\n{end}{after}"


def main(argv: list[str] | None = None, repo: Path = REPO_ROOT) -> None:
    parser = argparse.ArgumentParser(description="Write the task set's statistics into the README.")
    parser.add_argument("--print", action="store_true", help="show the block without writing it")
    args = parser.parse_args(argv)

    block = render(read_task_set(repo))
    if args.print:
        print(block)
        return
    path = repo / "README.md"
    path.write_text(update_readme(path.read_text(encoding="utf-8"), block), encoding="utf-8")


if __name__ == "__main__":
    main()
