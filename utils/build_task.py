"""
Build SciFlawBench task lines and check them against the rules in `docs/task-authoring.md`.

These are the functions behind `utils/build_task.ipynb`. The failure modes, agents, verifiers and tools come from the
harness itself, so they cannot drift from it.
"""

import json
from pathlib import Path
from typing import Any

from agents.definitions import DEFAULT_CODING_AGENT, agent_registry
from core.task_ids import FIXED_ID_PATTERN, TASK_ID_PLACEHOLDER, find_repo_root, hash_prompt
from evaluation.base import kwargs_problem
from evaluation.definitions import verifier_registry
from evaluation.scoring import ALL_FAILURE_MODES, QUALITATIVE_FAILURE_MODES, QUANTITATIVE_FAILURE_MODES
from tools.definitions import tool_registry

REPO_ROOT = find_repo_root(Path(__file__)) or Path.cwd()

FAILURE_MODES = list(ALL_FAILURE_MODES)
AGENTS = agent_registry.names()
VALIDATORS = verifier_registry.names()

# The tools every default agent already carries, and the opt-in ones a task has to request through `extra_tools`.
DEFAULT_TOOLS = [tool.tool_name for tool in DEFAULT_CODING_AGENT.tools if not isinstance(tool, str)]
ALL_TOOLS = tool_registry.names()
OPT_IN_TOOLS = [tool for tool in ALL_TOOLS if tool not in DEFAULT_TOOLS]


def display_path(path) -> str:
    """`path` relative to the repo when possible, else as given."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def normalize_sources(sources) -> list[dict]:
    """Bookkeeping references as {"url", "note"} dicts; a bare string becomes a url with an empty note."""
    items = []
    for source in sources or []:
        if isinstance(source, dict):
            items.append({"url": str(source.get("url", "")).strip(), "note": str(source.get("note", "")).strip()})
        else:
            items.append({"url": str(source).strip(), "note": ""})
    return items


def build_task(
    prompt, agent_id, quant_failure_modes, qual_failure_modes, extra_tools, validators, sources=None, task_id=None
) -> dict[str, Any]:
    """
    Build one harness task, shaped like a task-file line.

    Args:
        prompt (str): the exact text handed to the agent
        agent_id (str): the agent preset, `code_agent` or `tool_agent`
        quant_failure_modes (list[str]): the quantitative failure modes the task probes
        qual_failure_modes (list[str]): the qualitative failure modes the task probes
        extra_tools (list[dict | str]): opt-in tools, as `{"tool_name": ...}` or a bare name
        validators (list[dict]): one `{"name", "kwargs", "failure_mode"}` check per failure mode
        sources (list[dict | str] | None): where the ground truth was verified
        task_id (str | None): a fixed id that stays the same when the prompt is edited; md5(prompt) when empty

    Returns (dict[str, Any]): the task
    """
    prompt = str(prompt).strip()
    prompt_md5 = hash_prompt(prompt)
    # a fixed id stays the same when the prompt is edited; prompt_md5 then guards the prompt instead
    if task_id:
        ids = {"task_id": str(task_id).strip(), "task": prompt, "prompt_md5": prompt_md5}
    else:
        ids = {"task_id": prompt_md5, "task": prompt}
    return {
        **ids,
        "agent_id": agent_id,
        "failure_modes": {
            "quantitative": list(quant_failure_modes),
            "qualitative": list(qual_failure_modes),
        },
        "extra_tools": [dict(tool) if isinstance(tool, dict) else {"tool_name": tool} for tool in extra_tools],
        "validators": [dict(validator) for validator in validators],
        "sources": normalize_sources(sources),
    }


def check_task(task) -> list[str]:
    """
    Find the problems in one task, reporting all of them at once where the harness stops at the first.

    Args:
        task (dict): a task, shaped like a task-file line

    Returns (list[str]): the problems found, empty when the task is valid
    """
    problems = []

    prompt = str(task.get("task") or "")
    if not prompt.strip():
        problems.append("the prompt is empty")

    if "prompt_md5" in task and not FIXED_ID_PATTERN.fullmatch(str(task.get("task_id") or "")):
        problems.append("a fixed task_id must start with a letter and use only letters, digits, '.', '_' and '-'")

    if task.get("agent_id") not in AGENTS:
        problems.append(f"agent_id must be one of {AGENTS}")

    modes = task.get("failure_modes") or {}
    quantitative = list(modes.get("quantitative") or [])
    qualitative = list(modes.get("qualitative") or [])
    unknown = [mode for mode in quantitative + qualitative if mode not in FAILURE_MODES]
    if unknown:
        problems.append(f"unknown failure mode(s) {unknown}; allowed: {FAILURE_MODES}")
    misplaced = [mode for mode in quantitative if mode in QUALITATIVE_FAILURE_MODES] + [
        mode for mode in qualitative if mode in QUANTITATIVE_FAILURE_MODES
    ]
    if misplaced:
        problems.append(f"failure mode(s) {misplaced} are listed under the wrong family (quantitative/qualitative)")
    if not quantitative and not qualitative:
        problems.append("declare at least one failure mode (quantitative or qualitative)")

    seen_tools = []
    for index, tool in enumerate(task.get("extra_tools") or [], start=1):
        name = tool.get("tool_name") if isinstance(tool, dict) else tool
        if name not in ALL_TOOLS:
            problems.append(f"extra_tools[{index}]: unknown tool {name!r}")
        elif name in DEFAULT_TOOLS:
            problems.append(f"extra_tools[{index}]: {name} is already available, and duplicate names crash the run")
        elif name in seen_tools:
            problems.append(f"extra_tools[{index}]: {name} is listed twice")
        seen_tools.append(name)

    validators = task.get("validators") or []
    if not validators:
        problems.append("no validators: nothing checks the answer, so the task scores nothing")
    labels = [validator.get("failure_mode") if isinstance(validator, dict) else None for validator in validators]
    if validators and labels != quantitative + qualitative:
        problems.append(
            "each failure mode needs exactly one check, labelled with it in 'failure_mode' and listed in the order "
            f"of the modes (quantitative first): modes {quantitative + qualitative}, check labels {labels}"
        )
    for index, validator in enumerate(validators, start=1):
        name = validator.get("name") if isinstance(validator, dict) else validator
        if name not in VALIDATORS:
            problems.append(f"validators[{index}]: unknown verifier {name!r}; allowed: {VALIDATORS}")
        elif isinstance(validator, dict):
            problem = kwargs_problem(verifier_registry.get(name), validator.get("kwargs") or {})
            if problem:
                problems.append(f"validators[{index}]: {name} can't run with its kwargs: {problem}")

    if TASK_ID_PLACEHOLDER in prompt:
        assets = REPO_ROOT / "data" / "tasks" / "assets" / str(task.get("task_id"))
        if not assets.is_dir():
            problems.append(f"the prompt uses {TASK_ID_PLACEHOLDER} but {display_path(assets)} does not exist")

    sources = task.get("sources") or []
    if not isinstance(sources, list):
        problems.append("sources must be a list of {'url', 'note'} objects")
    else:
        for index, source in enumerate(sources, start=1):
            if not isinstance(source, dict):
                problems.append(f"sources[{index}]: must be an object with 'url' and 'note'")
                continue
            url = str(source.get("url") or "")
            if not url.startswith(("http://", "https://")):
                problems.append(f"sources[{index}]: url {url!r} must start with http:// or https://")
            if not isinstance(source.get("note", ""), str):
                problems.append(f"sources[{index}]: note must be a string")

    return problems


def read_tasks(path) -> list[dict]:
    """Every task already in `path` (an empty list when the file does not exist)."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_hash(task) -> str:
    """The md5 recorded for a task's prompt: `prompt_md5` for a fixed id, else the id itself."""
    return task.get("prompt_md5") or task.get("task_id")


def append_task(task, path) -> int:
    """
    Append a task as one JSON line, refusing a duplicate id or prompt.

    Args:
        task (dict): the task to append
        path (Path | str): the task file, created if missing

    Returns (int): the number of tasks now in the file
    """
    path = Path(path)
    rows = read_tasks(path)
    if any(row.get("task_id") == task["task_id"] or prompt_hash(row) == prompt_hash(task) for row in rows):
        raise ValueError(f"a task with the same id or prompt is already in {display_path(path)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(task, ensure_ascii=False) + "\n")
    return len(rows) + 1


def check_task_file(path) -> tuple[list[dict], list[str], list[str]]:
    """
    Check every task in a task file with the same rules as `check_task`.

    Args:
        path (Path | str): the task file

    Returns (tuple[list[dict], list[str], list[str]]): the tasks, the duplicate ids, and the problems found
    """
    rows = read_tasks(path)
    ids = [row.get("task_id") for row in rows] + [prompt_hash(row) for row in rows if "prompt_md5" in row]
    duplicates = sorted({value for value in ids if value and ids.count(value) > 1})

    problems = []
    for index, row in enumerate(rows, start=1):
        for problem in check_task(row):
            problems.append(f"line {index}: {problem}")
        if prompt_hash(row) != hash_prompt(str(row.get("task") or "")):
            field = "prompt_md5" if "prompt_md5" in row else "task_id"
            problems.append(f"line {index}: {field} does not match md5(task)")
    return rows, duplicates, problems
