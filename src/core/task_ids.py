import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# A task id may be a plain integer (legacy/ad-hoc tasks), the 32-character
# prompt hash used by SciFlawBench, or a fixed id recorded with `prompt_md5`.
# All must round-trip through file names.
TaskId = int | str

# A fixed id starts with a letter, so it is never read back from a file name as an
# integer, and uses only characters that are safe in file names, e.g. "SFB-0042".
FIXED_ID_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9._-]*")

# Where the task id goes inside a prompt that references its assets folder.
TASK_ID_PLACEHOLDER = "{TASK_ID}"

# Keys emitted first (in this order) when a task line is rewritten.
_LEADING_KEYS = ("task_id", "task")


def hash_prompt(task: str) -> str:
    """The md5 of a prompt: the task's id by default, or its `prompt_md5` when the id is fixed."""
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
    """A single task whose id (or, for a fixed id, `prompt_md5`) was stale, and how its assets folder moved."""

    line: int
    old_id: str
    new_id: str
    renamed: tuple[Path, Path] | None = None
    field: str = "task_id"  # "prompt_md5" when the task keeps a fixed id and only its hash changed


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
    A task with a fixed id (one that records `prompt_md5`) keeps its id and its
    folder; only its `prompt_md5` is recomputed.

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
        new_hash = hash_prompt(template)

        obj["task"] = template
        report.total += 1

        if "prompt_md5" in obj:
            # a fixed id stays; only the recorded hash follows the prompt
            old_hash = str(obj["prompt_md5"])
            obj["prompt_md5"] = new_hash
            if old_hash != new_hash:
                report.changes.append(TaskIdChange(line=lineno, old_id=old_hash, new_id=new_hash, field="prompt_md5"))
        else:
            obj["task_id"] = new_hash
            if old_id != new_hash:
                report.changes.append(TaskIdChange(line=lineno, old_id=old_id, new_id=new_hash))

        out_lines.append(render_task_line(obj))

    if report.errors or not apply:
        return report

    if assets_dir is not None:
        # validate every move before touching the filesystem, so we never
        # half-rename a set of folders and then bail out
        moves: list[tuple[TaskIdChange, Path, Path]] = []
        for change in report.changes:
            if change.field != "task_id":
                continue  # a fixed id keeps its folder
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
