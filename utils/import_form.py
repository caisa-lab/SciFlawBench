"""
Turn Google Form submissions into task lines on a local branch, for a maintainer to read and push.

Run it from the repo root; `docs/form-import.md` covers the setup:

    uv run --with google-api-python-client --with google-auth-oauthlib python -m utils.import_form --sheet <id>

It signs in to Google with exactly two scopes, and nothing broader: `spreadsheets.readonly` and `drive.readonly`.
`drive.readonly` covers the whole Drive of the account that signs in, so run it from the lab account that owns the form
and not from a personal one. The sign-in stays in a per-user config directory, outside the repo.

`--from-csv` reads a CSV export of the sheet and a local folder of files instead, so everything except the Google calls
can run without Google, and `--dry-run` only reports. The Google libraries are imported only when the script talks to
Google. Problems are reported by task id and never quote a personal detail. The script never pushes.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

from pydantic import ValidationError

from core.task_ids import FIXED_ID_PATTERN, TASK_ID_PLACEHOLDER, hash_prompt
from core.tasks import TaskDef
from evaluation.scoring import QUANTITATIVE_FAILURE_MODES
from utils.build_task import FAILURE_MODES, REPO_ROOT, append_task, build_task, check_task, prompt_hash, read_tasks

TASK_FILE = "data/tasks/v0/tasks.jsonl"
ASSETS_DIR = "data/tasks/assets"
REJECTED_FILE = "data/tasks/rejected_ids.txt"
AGENT_ID = "code_agent"

# The form's questions, by their titles on the form; case and spacing don't matter. "Task ID" is added by the script.
PERSONAL_COLUMNS = ("Full Name", "Email", "Institution", "ORCID", "OpenReview")
TASK_COLUMNS = ("Task ID", "Task Description", "Failure Modes", "Validation", "Sources", "Data (Optional)")
FILES_COLUMN = "Data (Optional)"
MAX_FAILURE_MODES = 4

# Each failure mode's validation is imported as a rubric in the author's own words, for a reviewer to refine.
RUBRIC_VALIDATOR = "judge:rubric"
HEADING = re.compile(r"#+\s*(.+?)\s*:?")
URL = re.compile(r"https?://\S+")

# Only text, so the leak check can read everything that gets committed. PDFs and the like hide details in metadata.
TEXT_SUFFIXES = {".csv", ".dat", ".json", ".md", ".py", ".txt"}
MAX_TASK_BYTES = 20 * 1024 * 1024
SAFE_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
DRIVE_FILE_ID = re.compile(r"[?&]id=([\w-]+)")
ORCID = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dXx]")
NO_ANSWER = {"-", "n/a", "na", "nil", "no", "none"}

# Commits carry this identity, so no maintainer or author details reach the git history.
COMMIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "SciFlawBench import",
    "GIT_AUTHOR_EMAIL": "import@sciflawbench.invalid",
    "GIT_COMMITTER_NAME": "SciFlawBench import",
    "GIT_COMMITTER_EMAIL": "import@sciflawbench.invalid",
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]
CONFIG_DIR = Path.home() / ".config" / "sciflawbench"


def _title_key(title: str) -> str:
    return " ".join(str(title).split()).casefold()


def required_columns() -> list[str]:
    """Every column the response sheet must have, so a renamed form question stops the import."""
    return [*PERSONAL_COLUMNS, *TASK_COLUMNS]


def parse_rows(values: list[list[str]]) -> list[dict[str, str]]:
    """
    Turn sheet values into one dict per submission, keyed by column title. Titles are matched ignoring case and
    spacing; other columns, such as the timestamp, keep their own titles.

    Args:
        values (list[list[str]]): the header row and the rows; the Sheets API leaves out empty cells at a row's end

    Returns (list[dict[str, str]]): the submissions
    """
    if not values:
        raise SystemExit("The response sheet is empty.")
    header, *rows = values
    keys = [_title_key(title) for title in header]
    wanted = {_title_key(column): column for column in required_columns()}
    missing = [column for key, column in wanted.items() if key not in keys]
    if missing:
        raise SystemExit(f"The response sheet lacks these columns: {missing}")
    doubled = [column for key, column in wanted.items() if keys.count(key) > 1]
    if doubled:
        raise SystemExit(f"The response sheet has more than one column for each of these: {doubled}")
    titles = [wanted.get(key, title) for key, title in zip(keys, header, strict=True)]
    return [dict(zip(titles, row, strict=False)) for row in rows]


def _box(row: dict[str, str], column: str) -> str:
    return str(row.get(column) or "").strip()


def failure_mode_name(label: str) -> str:
    """The harness name of a failure mode as the form writes it: `Correct tool calls` becomes `correct_tool_calls`."""
    return "_".join(re.split(r"[\s_-]+", label.strip().casefold()))


def ticked_modes(answer: str) -> tuple[list[str], list[str]]:
    """
    Read the failure modes ticked on the form, which the sheet joins with commas.

    Args:
        answer (str): the Failure Modes answer

    Returns (tuple[list[str], list[str]]): the known modes, quantitative first, and the problems
    """
    modes = [failure_mode_name(label) for label in answer.split(",") if label.strip()]
    problems = []
    unknown = sum(mode not in FAILURE_MODES for mode in modes)
    if unknown:
        problems.append(f"{unknown} of the ticked failure modes are not ones the harness knows")
    if not 1 <= len(modes) <= MAX_FAILURE_MODES:
        problems.append(f"tick between 1 and {MAX_FAILURE_MODES} failure modes; {len(modes)} are ticked")
    known = [mode for mode in modes if mode in FAILURE_MODES]
    return sorted(known, key=lambda mode: mode not in QUANTITATIVE_FAILURE_MODES), problems


def parse_validation(answer: str) -> tuple[dict[str, str], list[str]]:
    """
    Split the Validation answer into one description per failure mode. The form asks for each as a `# MODE` line
    followed by how to tell whether the agent made that mistake. Problems number the sections, never quoting them.

    Args:
        answer (str): the Validation answer

    Returns (tuple[dict[str, str], list[str]]): each failure mode's description, in the order written, and the problems
    """
    sections, problems = [], []
    for line in answer.splitlines():
        heading = HEADING.fullmatch(line.strip())
        if heading:
            sections.append((failure_mode_name(heading.group(1)), []))
        elif sections:
            sections[-1][1].append(line)
        elif line.strip() and not problems:
            problems.append("validation: start each failure mode with a line such as '# Correctness'")

    descriptions = {}
    for number, (mode, lines) in enumerate(sections, start=1):
        description = "\n".join(lines).strip()
        if mode not in FAILURE_MODES:
            problems.append(f"validation: the heading of section {number} is not a failure mode")
        elif mode in descriptions:
            problems.append(f"validation: {mode} is described twice")
        elif not description:
            problems.append(f"validation: {mode} has no description")
        else:
            descriptions[mode] = description
    return descriptions, problems


def _first_url(text: str) -> str:
    """The first link in `text`, without punctuation that follows it, keeping brackets that are part of it."""
    match = URL.search(text)
    if not match:
        return ""
    url = match.group().rstrip(".,;:>")
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip(".,;:")
    return url


def parse_sources(answer: str) -> tuple[list[dict], list[str]]:
    """
    Split the Sources answer into sources. The form asks for each as a `* SOURCE / URL` line with `- DESCRIPTION`
    lines below it. The first link on a `*` line is the source's URL, and the rest of that line and the lines below
    are its note. A line with a link that isn't a `-` line also starts a source, so a plain list of links works too.

    Args:
        answer (str): the Sources answer

    Returns (tuple[list[dict], list[str]]): the sources as {"url", "note"}, and the problems
    """
    entries, problems = [], []
    for line in answer.splitlines():
        text = line.strip()
        if not text:
            continue
        if text[0] in "*•" or (URL.search(text) and (text[0] not in "-–" or not entries)):
            entries.append((text.lstrip("*•-– ").strip(), []))
        elif entries:
            entries[-1][1].append(text.lstrip("-–• ").strip())
        elif not problems:
            problems.append("sources: some text comes before the first source")

    sources = []
    for number, (head, notes) in enumerate(entries, start=1):
        url = _first_url(head)
        if not url:
            problems.append(f"sources: source {number} has no link starting with http:// or https://")
            continue
        title = head.replace(url, "", 1).strip(" /|-–—:,;.()<>")
        note = " ".join(note for note in notes if note)
        sources.append({"url": url, "note": ": ".join(part for part in (title, note) if part)})
    if not entries:
        problems.append("sources: give at least one source, as '* URL' with '- what it confirms' below")
    return sources, problems


def row_to_task(row: dict[str, str]) -> tuple[dict, list[str]]:
    """
    Build the task line of one submission. The personal columns are never read here.

    Each ticked failure mode gets a `judge:rubric` validator holding the author's description of it from the
    Validation answer. A reviewer may swap one for an automatic validator, such as a numeric match, before the PR.

    Args:
        row (dict[str, str]): the submission

    Returns (tuple[dict, list[str]]): the task, and the problems found in its answers
    """
    modes, problems = ticked_modes(_box(row, "Failure Modes"))
    descriptions, validation_problems = parse_validation(_box(row, "Validation"))
    sources, source_problems = parse_sources(_box(row, "Sources"))
    problems += validation_problems + source_problems

    undescribed = [mode for mode in modes if mode not in descriptions]
    if undescribed:
        problems.append(f"these ticked failure modes have no validation: {undescribed}")
    unticked = [mode for mode in descriptions if mode not in modes]
    if unticked:
        problems.append(f"these failure modes have a validation but are not ticked: {unticked}")

    validators = [
        {"name": RUBRIC_VALIDATOR, "kwargs": {"rubric": descriptions[mode]}, "failure_mode": mode}
        for mode in modes
        if mode in descriptions
    ]
    task = build_task(
        _box(row, "Task Description"),
        AGENT_ID,
        [mode for mode in modes if mode in QUANTITATIVE_FAILURE_MODES],
        [mode for mode in modes if mode not in QUANTITATIVE_FAILURE_MODES],
        ["read_file"] if _box(row, FILES_COLUMN) else [],
        validators,
        sources,
        task_id=_box(row, "Task ID"),
    )
    return task, problems


def clean_file_name(name: str) -> tuple[str, str]:
    """
    Split an uploaded file's name, which Google writes as `data - Name Surname.csv`.

    Args:
        name (str): the name in Drive

    Returns (tuple[str, str]): the name for the repo, and the uploader's Google account name
    """
    stem, suffix = os.path.splitext(name)
    kept, separator, uploader = stem.rpartition(" - ")
    return (kept + suffix, uploader) if separator else (name, "")


def fetch_files(row: dict[str, str], fetch) -> tuple[dict[str, bytes], list[str], list[str]]:
    """
    Get a submission's files and clean their names. A file that can't be read stops the task.

    Args:
        row (dict[str, str]): the submission
        fetch (Callable[[dict[str, str]], list[tuple[str, bytes]]]): gives the name and contents of a row's files

    Returns (tuple[dict[str, bytes], list[str], list[str]]): cleaned name to contents, the uploader names, the problems
    """
    if not _box(row, FILES_COLUMN):
        return {}, [], []
    try:
        uploads = fetch(row)
    except Exception as error:
        return {}, [], [f"its files could not be read ({type(error).__name__})"]
    if not uploads:
        return {}, [], ["the form lists files for it, but none could be found"]

    files, uploaders, problems = {}, [], []
    for index, (name, content) in enumerate(uploads, start=1):
        cleaned, uploader = clean_file_name(name)
        if not uploader:
            problems.append(f"file {index}: its name does not end in ' - <account name>', so it cannot be cleaned")
        elif not SAFE_FILE_NAME.fullmatch(cleaned):
            problems.append(f"file {index}: use only letters, digits, '.', '_' and '-' in the file name")
        elif cleaned in files:
            problems.append(f"file {index}: {cleaned} was uploaded twice")
        else:
            files[cleaned] = content
        if uploader:
            uploaders.append(uploader)
    if sum(len(content) for content in files.values()) > MAX_TASK_BYTES:
        problems.append(f"its files are over the limit of {MAX_TASK_BYTES // (1024 * 1024)} MB per task")
    return files, uploaders, problems


def check_files(prompt: str, files: dict[str, bytes]) -> list[str]:
    """
    Find the problems in a submission's cleaned files.

    Args:
        prompt (str): the task's prompt
        files (dict[str, bytes]): cleaned name to contents

    Returns (list[str]): the problems found
    """
    problems = []
    if files and TASK_ID_PLACEHOLDER not in prompt:
        problems.append(f"files were uploaded, but the prompt does not refer to {ASSETS_DIR}/{TASK_ID_PLACEHOLDER}/")
    if TASK_ID_PLACEHOLDER in prompt and not files:
        problems.append(f"the prompt refers to {ASSETS_DIR}/{TASK_ID_PLACEHOLDER}/, but no file was uploaded")
    for name, content in files.items():
        if os.path.splitext(name)[1].lower() not in TEXT_SUFFIXES:
            problems.append(f"{name}: only these file types are accepted: {sorted(TEXT_SUFFIXES)}")
        else:
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                problems.append(f"{name}: not a UTF-8 text file")
        if not re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w-])", prompt):
            problems.append(f"{name}: the prompt does not mention this file")
    named = {match.rstrip(".") for match in re.findall(rf"{re.escape(TASK_ID_PLACEHOLDER)}/([\w.-]+)", prompt)}
    for name in sorted(named - set(files)):
        problems.append(f"the prompt mentions {name}, but no such file was uploaded")
    return problems


def personal_terms(row: dict[str, str], uploaders: list[str]) -> list[str]:
    """
    The details of one person that must not reach the repo. The institution is left out, since a prompt may rightly
    name one, and so are one-word names, which `name_parts` handles because a word like "Li" can mean anything.

    Args:
        row (dict[str, str]): the submission
        uploaders (list[str]): the Google account names cut from the file names

    Returns (list[str]): the terms to search for
    """
    names = [_box(row, "Full Name"), *uploaders]
    terms = [name for name in names if len(name.split()) > 1]
    terms.append(_box(row, "Email"))
    orcid = _box(row, "ORCID")
    match = ORCID.search(orcid)
    terms.append(match.group() if match else orcid)
    openreview = _box(row, "OpenReview")
    if openreview.casefold() not in NO_ANSWER:
        handle = re.search(r"(?:~|id=)([^/&\s]+)", openreview)
        terms += [openreview, handle.group(1) if handle else ""]
    return [term for term in terms if term]


def name_parts(row: dict[str, str], uploaders: list[str]) -> list[str]:
    """The single words of a person's names: finding one may be chance, so it only asks a maintainer to look."""
    return [word for name in (_box(row, "Full Name"), *uploaders) for word in name.split() if len(word) > 1]


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_accents.split()).casefold()


def contains_personal_detail(text: str, terms: list[str]) -> bool:
    """Whether `text` holds any term as a whole word, ignoring case, accents and spacing, since names vary in all."""
    if not terms:
        return False
    pattern = "|".join(re.escape(_normalize(term)) for term in terms)
    return bool(re.search(rf"(?<!\w)(?:{pattern})(?!\w)", _normalize(text)))


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    environment = {**os.environ, **env} if env else None
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True, env=environment).stdout


def _lines_at(repo: Path, ref: str, path: str) -> list[str]:
    result = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=repo, capture_output=True, text=True)
    return result.stdout.splitlines() if result.returncode == 0 else []


def _task_contents(repo: Path, ref: str) -> set[str]:
    """The task lines, rejected ids and asset files of `ref`, comparable however the commits were merged."""
    assets = _git(repo, "ls-tree", "-r", ref, "--", ASSETS_DIR).splitlines()
    return {*_lines_at(repo, ref, TASK_FILE), *_lines_at(repo, ref, REJECTED_FILE), *assets}


def _exists(repo: Path, ref: str) -> bool:
    return (
        subprocess.run(["git", "rev-parse", "--verify", "--quiet", ref], cwd=repo, capture_output=True).returncode == 0
    )


def _holds_unmerged_work(repo: Path, ref: str, base: str) -> bool:
    return not _task_contents(repo, ref) <= _task_contents(repo, base)


def prepare_branch(repo: Path, branch: str, base: str) -> None:
    """
    Switch to the import branch, continuing from whatever isn't merged yet, so maintainers can import in turn.

    A local branch holding unmerged work is kept and brought up to date with the pushed one; otherwise the pushed one
    is continued. Only when neither holds anything unmerged does it restart from `base`. PRs are squash-merged, so
    commits can't tell what is merged: this compares the contents, and never moves a branch holding unmerged work.

    Args:
        repo (Path): the repo
        branch (str): the import branch
        base (str): the branch it starts from, such as `origin/main`; the pushed branch is looked for on its remote
    """
    tracked = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if tracked or _git(repo, "status", "--porcelain", "--", "data/tasks"):
        raise SystemExit("Commit or set aside your changes first: the import switches branches.")

    shared = f"{base.split('/')[0]}/{branch}" if "/" in base else ""
    shared_work = bool(shared) and _exists(repo, f"refs/remotes/{shared}") and _holds_unmerged_work(repo, shared, base)
    if _exists(repo, f"refs/heads/{branch}") and _holds_unmerged_work(repo, branch, base):
        _git(repo, "switch", "--quiet", branch)
        if shared_work:
            try:
                _git(repo, "merge", "--quiet", "--ff-only", shared)
            except subprocess.CalledProcessError:
                raise SystemExit(
                    f"Your {branch} and {shared} have both changed. Nothing was imported: sort out which to keep first."
                ) from None
    else:
        _git(repo, "switch", "--quiet", "--no-track", "-C", branch, shared if shared_work else base)


def _loader_problems(task: dict) -> list[str]:
    try:
        TaskDef.model_validate(task)
    except ValidationError as error:
        return [f"the harness refuses it: {detail['msg']}" for detail in error.errors()]
    return []


def staged_problems(repo: Path, terms: list[str]) -> list[str]:
    """
    Check what is about to be committed, which also covers renames, mode changes and links.

    Args:
        repo (Path): the repo, with the task staged
        terms (list[str]): the personal details to search for

    Returns (list[str]): the problems found
    """
    problems = []
    for line in _git(repo, "diff", "--cached", "--raw", "--no-renames").splitlines():
        status, path = line.split("\t", 1)
        if not path.startswith("data/tasks/"):
            problems.append(f"{path} is staged, but it is outside data/tasks/")
        if status.split()[1] == "120000":
            problems.append(f"{path} is a symbolic link")
        if any(part.startswith(".") for part in Path(path).parts):
            problems.append(f"{path} is a hidden file")
    if contains_personal_detail(_git(repo, "diff", "--cached", "--text", "--unified=0"), terms):
        problems.append("a personal detail of a contributor appears in the staged changes")
    return problems


def import_submission(
    row: dict[str, str], fetch, repo: Path, terms: list[str], parts: list[str], *, dry_run=False
) -> tuple[list[str], list[str]]:
    """
    Check one submission and, if it is valid, commit its task line and files.

    Args:
        row (dict[str, str]): the submission
        fetch (Callable[[dict[str, str]], list[tuple[str, bytes]]]): gives the name and contents of a row's files
        repo (Path): the repo, on the import branch
        terms (list[str]): the details of everyone in the sheet, which stop a task
        parts (list[str]): single words of their names, which only raise a warning
        dry_run (bool): whether to stop before writing anything

    Returns (tuple[list[str], list[str]]): the problems, when nothing was written, and the warnings
    """
    task_id = _box(row, "Task ID")
    if not FIXED_ID_PATTERN.fullmatch(task_id):
        return [f"{task_id!r} is not a valid task id"], []
    assets = repo / ASSETS_DIR / task_id
    if assets.exists():
        return [f"{ASSETS_DIR}/{task_id} already exists"], []

    task, problems = row_to_task(row)
    files, uploaders, file_problems = fetch_files(row, fetch)
    problems += file_problems + check_files(task["task"], files)

    terms = [*terms, *personal_terms(row, uploaders)]
    parts = [*parts, *name_parts(row, uploaders)]
    texts = {"the task line": json.dumps(task, ensure_ascii=False)}
    texts |= {name: f"{name}\n{content.decode('utf-8', errors='replace')}" for name, content in files.items()}
    warnings = []
    for where, text in texts.items():
        if contains_personal_detail(text, terms):
            problems.append(f"a personal detail of a contributor appears in {where}")
        elif contains_personal_detail(text, parts):
            warnings.append(f"part of a contributor's name may appear in {where}; please read it")

    if any(prompt_hash(known) == prompt_hash(task) for known in read_tasks(repo / TASK_FILE)):
        problems.append(f"the same prompt is already in {TASK_FILE}")

    # check_task wants the folder of a prompt with {TASK_ID} to exist
    assets.mkdir(parents=True)
    problems += check_task(task) + _loader_problems(task)
    if problems or dry_run or not files:
        shutil.rmtree(assets)
    if problems or dry_run:
        return problems, warnings

    append_task(task, repo / TASK_FILE)
    for name, content in files.items():
        (assets / name).write_bytes(content)
    _git(repo, "add", "--", TASK_FILE, *([f"{ASSETS_DIR}/{task_id}"] if files else []))
    problems = staged_problems(repo, terms)
    if problems:
        _git(repo, "reset", "--quiet")
        _git(repo, "checkout", "--quiet", "--", TASK_FILE)
        shutil.rmtree(assets, ignore_errors=True)
        return problems, warnings
    _git(repo, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", f"Add task {task_id}", env=COMMIT_IDENTITY)
    return [], warnings


def rejected_ids(repo: Path) -> set[str]:
    """The ids a maintainer rejected: one per line in `data/tasks/rejected_ids.txt`, with `#` starting a comment."""
    path = repo / REJECTED_FILE
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return {line.split("#")[0].strip() for line in lines} - {""}


def import_rows(
    rows: list[dict[str, str]], fetch, repo: Path, *, dry_run=False
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    """
    Import every new submission. Rejected ids and ids already imported are skipped, and an id already imported with a
    different prompt is reported, so an answer edited after the import can't overwrite the task.

    Args:
        rows (list[dict[str, str]]): the submissions
        fetch (Callable[[dict[str, str]], list[tuple[str, bytes]]]): gives the name and contents of a row's files
        repo (Path): the repo, on the import branch
        dry_run (bool): whether to only report

    Returns (tuple[list[str], dict[str, list[str]], dict[str, list[str]]]): the ids imported, or those that would be on
        a dry run, then the problems and the warnings by task id or sheet row
    """
    imported, reports, warnings = [], {}, {}
    known = {task.get("task_id"): task for task in read_tasks(repo / TASK_FILE)}
    rejected = rejected_ids(repo)
    terms = sorted({term for row in rows for term in personal_terms(row, [])})
    parts = sorted({part for row in rows for part in name_parts(row, [])})
    for number, row in enumerate(rows, start=2):
        task_id = _box(row, "Task ID")
        if not task_id:
            reports[f"sheet row {number}"] = ["it has no task id, so the Apps Script did not run on it"]
        elif task_id in known:
            if known[task_id].get("prompt_md5") != hash_prompt(_box(row, "Task Description")):
                reports[task_id] = ["it is already imported with a different prompt; the task was not changed"]
        elif task_id not in rejected:
            problems, notes = import_submission(row, fetch, repo, terms, parts, dry_run=dry_run)
            if notes:
                warnings[task_id] = notes
            if problems:
                reports[task_id] = problems
            else:
                imported.append(task_id)
    return imported, reports, warnings


def new_task_lines(repo: Path, branch: str, base: str) -> list[str]:
    """The task lines on `branch` that `base` does not have, to run the new tasks without the whole set."""
    known = set(_lines_at(repo, base, TASK_FILE))
    return [line for line in _lines_at(repo, branch, TASK_FILE) if line not in known]


def local_source(csv_file: Path, files_dir: Path | None):
    """
    Read a CSV export of the sheet, with each submission's files in `<files_dir>/<task id>/`.

    Args:
        csv_file (Path): the export
        files_dir (Path | None): the folder of files

    Returns (tuple): the submissions, and the function giving a row's files
    """
    with csv_file.open(encoding="utf-8", newline="") as handle:
        rows = parse_rows(list(csv.reader(handle)))

    def fetch(row: dict[str, str]) -> list[tuple[str, bytes]]:
        folder = (files_dir or Path()) / _box(row, "Task ID")
        paths = sorted(folder.iterdir()) if folder.is_dir() else []
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise OSError("only plain files can be imported")
        return [(path.name, path.read_bytes()) for path in paths]

    return rows, fetch


def google_source(sheet_id: str, tab: str, client_file: Path, token_file: Path):
    """
    Read the response sheet through Google, signing in with read-only access to Sheets and Drive.

    Args:
        sheet_id (str): the id of the response spreadsheet
        tab (str): the tab holding the responses
        client_file (Path): the OAuth client file from Google Cloud
        token_file (Path): where the sign-in is kept, outside the repo

    Returns (tuple): the submissions, and the function giving a row's files
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    credentials = Credentials.from_authorized_user_file(token_file, SCOPES) if token_file.exists() else None
    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
        except RefreshError:
            credentials = None
    if not credentials or not credentials.valid:
        credentials = InstalledAppFlow.from_client_secrets_file(client_file, SCOPES).run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.touch(mode=0o600)
        token_file.write_text(credentials.to_json(), encoding="utf-8")

    sheets = build("sheets", "v4", credentials=credentials)
    response = sheets.spreadsheets().values().get(spreadsheetId=sheet_id, range=tab).execute()
    return parse_rows(response.get("values", [])), drive_fetch(build("drive", "v3", credentials=credentials))


def drive_fetch(drive):
    """
    The function giving a row's files from Drive.

    Args:
        drive: the Drive service

    Returns (Callable[[dict[str, str]], list[tuple[str, bytes]]]): gives the name and contents of a row's files
    """

    def fetch(row: dict[str, str]) -> list[tuple[str, bytes]]:
        files = []
        for file_id in DRIVE_FILE_ID.findall(_box(row, FILES_COLUMN)):
            name = drive.files().get(fileId=file_id, fields="name").execute()["name"]
            files.append((name, drive.files().get_media(fileId=file_id).execute()))
        return files

    return fetch


def main(argv: list[str] | None = None, repo: Path = REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description="Import Google Form submissions as commits on a local branch.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sheet", help="the id of the response spreadsheet, from its URL")
    source.add_argument("--from-csv", type=Path, help="read a CSV export of the sheet instead of Google")
    parser.add_argument("--files-dir", type=Path, help="with --from-csv, the folder holding <task id>/<uploaded files>")
    parser.add_argument("--dry-run", action="store_true", help="only report: switch no branch and write nothing")
    parser.add_argument("--tab", default="Responses", help="the tab holding the form responses")
    parser.add_argument("--branch", default="new-tasks", help="the import branch")
    parser.add_argument("--base", default="origin/main", help="the branch it starts from; fetch this first")
    parser.add_argument("--client-file", type=Path, default=CONFIG_DIR / "google_client.json")
    parser.add_argument("--token-file", type=Path, default=CONFIG_DIR / "google_token.json")
    parser.add_argument("--export", type=Path, help="also write the tasks that are new on the branch to this file")
    args = parser.parse_args(argv)

    if args.from_csv:
        rows, fetch = local_source(args.from_csv, args.files_dir)
    else:
        rows, fetch = google_source(args.sheet, args.tab, args.client_file, args.token_file)

    if not args.dry_run:
        prepare_branch(repo, args.branch, args.base)
    imported, reports, warnings = import_rows(rows, fetch, repo, dry_run=args.dry_run)

    if args.dry_run:
        print(f"Would import {len(imported)} task(s): {imported}. Nothing was written.")
    else:
        print(f"Imported {len(imported)} task(s) onto {args.branch}: {imported}. Nothing was pushed.")
    for key, notes in warnings.items():
        for note in notes:
            print(f"{key}: {note}")
    for key, problems in reports.items():
        print(f"{key} was not written:")
        for problem in problems:
            print(f"  - {problem}")
    if args.export and not args.dry_run:
        lines = new_task_lines(repo, args.branch, args.base)
        args.export.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        print(f"Wrote the {len(lines)} new task(s) to {args.export}")
    return 1 if reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
