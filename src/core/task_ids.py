import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

# A task id may be a plain integer (legacy/ad-hoc tasks) or the 32-character
# prompt hash used by SciFlawBench. Both must round-trip through file names.
TaskId = int | str

# Where the task id goes inside a prompt that references its assets folder.
TASK_ID_PLACEHOLDER = "{TASK_ID}"

# Keys emitted first (in this order) when a task line is rewritten.
_LEADING_KEYS = ("task_id", "task")


def hash_prompt(task: str) -> str:
    """The canonical 32-character task id for a prompt (md5 of the prompt text)."""
    return hashlib.md5(task.encode("utf-8")).hexdigest()


def task_id_slug(task_id: TaskId) -> str:
    """
    Render a task id as a filesystem-safe string used for log/report file names.

    Integer ids keep the historical zero-padded 3-digit form (`1 -> "001"`) so
    existing logs and tooling stay compatible. String ids (e.g. the 32-character
    prompt hash used by SciFlawBench) are used verbatim.
    """
    return f"{task_id:03d}" if isinstance(task_id, int) else str(task_id)


def parse_task_id_slug(slug: str) -> TaskId:
    """Inverse of :func:`task_id_slug`: `"001" -> 1`, a 32-character hash stays a string."""
    try:
        return int(slug)
    except ValueError:
        return slug


def find_repo_root(start: Path) -> Path | None:
    """Walk up from `start` looking for the directory holding `.git` or `pyproject.toml`."""
    start = Path(start).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def default_assets_dir(start: Path) -> Path:
    """`<repo-root>/data/tasks/assets`, falling back to `<cwd>/data/tasks/assets`."""
    return (find_repo_root(start) or Path.cwd()) / "data" / "tasks" / "assets"


def render_task(task: str, task_id: TaskId) -> str:
    """Resolve `{TASK_ID}` in a stored prompt to the concrete id (what the agent sees)."""
    return task.replace(TASK_ID_PLACEHOLDER, str(task_id))


def authored_template(task: str, recorded_id: TaskId) -> str:
    """
    Recover the authored prompt by turning an embedded id back into the placeholder.

    Handles prompts written before the placeholder convention (i.e. with the id
    already substituted in). A no-op when the id does not appear in the text.
    """
    return task.replace(str(recorded_id), TASK_ID_PLACEHOLDER)


def render_task_line(obj: dict) -> str:
    """Serialise a task object as one JSON line, with `task_id`/`task` first."""
    ordered = {key: obj[key] for key in _LEADING_KEYS if key in obj}
    ordered.update({key: value for key, value in obj.items() if key not in _LEADING_KEYS})
    return json.dumps(ordered, ensure_ascii=False)


@dataclass
class TaskIdChange:
    """A single task whose id was stale, and how its assets folder moved."""

    line: int
    old_id: str
    new_id: str
    renamed: tuple[Path, Path] | None = None


@dataclass
class RecomputeReport:
    """Outcome of :func:`recompute_task_ids`."""

    task_file: Path
    total: int = 0
    changes: list[TaskIdChange] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    applied: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.changes)


def recompute_task_ids(
    task_file: Path,
    assets_dir: Path | None = None,
    *,
    apply: bool = True,
) -> RecomputeReport:
    """
    Recompute every task id in `task_file` from its prompt.

    When `apply` is true the file is rewritten, and (if `assets_dir` is given)
    each stale task's assets folder is renamed from the old id to the new one.

    Nothing is written if any line fails to parse, or if a rename would clobber an
    existing folder, so the task file is never left pointing at a missing folder.
    """
    report = RecomputeReport(task_file=task_file)

    raw_lines = task_file.read_text().splitlines()
    out_lines: list[str] = []

    for lineno, line in enumerate(raw_lines, start=1):
        if not line.strip():
            out_lines.append(line)
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            report.errors.append(f"line {lineno}: invalid JSON ({exc})")
            continue
        if "task_id" not in obj or "task" not in obj:
            report.errors.append(f"line {lineno}: missing 'task_id' or 'task'")
            continue

        old_id = str(obj["task_id"])
        template = authored_template(obj["task"], old_id)
        new_id = hash_prompt(template)

        obj["task_id"] = new_id
        obj["task"] = template
        report.total += 1

        if old_id != new_id:
            report.changes.append(TaskIdChange(line=lineno, old_id=old_id, new_id=new_id))

        out_lines.append(render_task_line(obj))

    if report.errors or not apply:
        return report

    if assets_dir is not None:
        # validate every move before touching the filesystem, so we never
        # half-rename a set of folders and then bail out
        moves: list[tuple[TaskIdChange, Path, Path]] = []
        for change in report.changes:
            source = assets_dir / change.old_id
            destination = assets_dir / change.new_id
            if not source.is_dir():
                continue
            if destination.exists():
                report.errors.append(f"cannot rename {source.name} -> {destination.name}: destination already exists")
                continue
            moves.append((change, source, destination))

        if report.errors:
            return report

        for change, source, destination in moves:
            source.rename(destination)
            change.renamed = (source, destination)

    task_file.write_text("\n".join(out_lines) + "\n")
    report.applied = True
    return report
