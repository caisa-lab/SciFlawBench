import dataclasses
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from agents.base import build_agent
from core.config import RunConfig
from core.events import AgentEvent, EventWatcher
from core.task_ids import TaskId, hash_prompt, render_task, task_id_slug
from evaluation.base import VerificationResult, VerifierContext, VerifierDef, run_check, wants_context
from evaluation.definitions import verifier_registry
from evaluation.judge import LiteLLMJudge
from evaluation.print_report import render_trace_markdown, save_markdown_report
from evaluation.scoring import FailureModes
from tools.artifacts import ArtifactStore
from tools.base import ToolDef

logger = logging.getLogger(__file__)


if os.environ.get("ENABLE_TEST_FAKES") == "1":
    import tests.fakes.presets
    import tests.fakes.tools


class TaskDef(BaseModel):
    """
    Working definition for tasks to be passed through the runtime manager and dispatched to a task runner
    (not necessarily for the evaluator itself)

    TODO: ground truth should be passed through here as well to allow for quantitative checks to be run by the
    task runners upon getting the solution to be included in the logs
    TODO: David agrees, maybe also a parser or specifications for universal parsers for quantitative checks
    """

    task_id: TaskId
    task: str
    agent_id: str
    extra_tools: list[ToolDef | str] = Field(default_factory=list)  # TODO: why allow strings?
    validators: list[VerifierDef] = Field(default_factory=list)
    failure_modes: FailureModes  # what the task is designed to probe; drives per-mode scoring
    closed_book: bool = False  # for the individual task flag

    repetition: int = 1  # for multiple runs of the same tasks

    @property
    def rendered_task(self) -> str:
        """The prompt handed to the agent: `{TASK_ID}` resolved to this task's id."""
        return render_task(self.task, self.task_id)

    @field_validator("extra_tools", mode="before")
    @classmethod
    def normalize_tools(cls, v):
        if not isinstance(v, list):
            return v
        return [{"tool_name": t} if isinstance(t, str) else t for t in v]

    @model_validator(mode="after")
    def verify_task_id_matches_prompt(self):
        """
        Guard against modified samples: a hash-style id must equal `md5(task)`.

        The prompt is hashed *as stored* (i.e. with any `{TASK_ID}` placeholder
        still in place), so editing a prompt without recomputing its id is
        rejected at load time. Integer ids (legacy/ad-hoc tasks) are left alone.
        """
        if isinstance(self.task_id, str):
            expected = hash_prompt(self.task)
            if self.task_id != expected:
                raise ValueError(
                    f"task_id {self.task_id!r} does not match md5(task) {expected!r}: "
                    "the task text was modified without updating its id"
                )
        return self


class TaskResult(BaseModel):
    """
    Final result that gets dumped into the log file

    (FOR NOW: really only gets used in run task but perhaps later we use it for passing around result objects and
    unloading the tasks I feel it's worth keeping it around)
    """

    task_id: TaskId
    task: str
    repetition: int
    output: Any
    success: bool
    error: str
    full_trace: list[dict]
    failure_modes: FailureModes | None = None
    check_results: list[VerificationResult] = Field(default_factory=list)


def verifier_wants_context(name: str) -> bool:
    """Whether the verifier registered under `name` accepts the optional `context` argument."""
    try:
        return wants_context(verifier_registry.get(name))
    except KeyError:
        return False  # unknown names are reported as a failed check, not here


def build_verifier_context(
    task: TaskDef, prompt: str, output: Any, run_config: RunConfig, events: list[AgentEvent], success: bool = True
) -> VerifierContext | None:
    """
    Assemble what context-aware verifiers (the LLM-as-a-judge ones) need to do their job.

    The trace is rendered, and the judge client built, only when at least one of the task's
    verifiers asks for context, so ordinary tasks pay nothing for this.

    Args:
        task (TaskDef): the task that just ran
        prompt (str): the prompt the agent was given
        output (Any): the agent's final answer
        run_config (RunConfig): the run configuration, which carries the judge settings
        events (list[AgentEvent]): the events recorded during the run
        success (bool): whether the agent finished without raising

    Returns (VerifierContext | None): the context, or None when no verifier asks for one
    """
    if not any(verifier_wants_context(verifier.name) for verifier in task.validators):
        return None

    judge: LiteLLMJudge | None = None
    judge_error: str | None = None
    if run_config.judge is None:
        judge_error = "the run config has no 'judge' section"
    else:
        try:
            judge = LiteLLMJudge(run_config.judge)
        except Exception as exc:  # building the client must not take down the task itself
            judge_error = f"{type(exc).__name__}: {exc}"

    trace = render_trace_markdown(
        {
            "task_id": task.task_id,
            "task": prompt,
            "output": output,
            "success": success,
            "full_trace": [dataclasses.asdict(event) for event in events],
        }
    )

    return VerifierContext(trace=trace, task_id=task.task_id, task=prompt, judge=judge, judge_error=judge_error)


def run_task(task: TaskDef, run_config: RunConfig, output_dir: Path, res_queue: mp.Queue):
    """
    The target function actually run by the runtime manager to launch subprocesses which complete provision and
    complete the agentic tasks

    Args:
        task (TaskDef): necessary information to run the given task
        run_config (RunConfig): configuration of the harness for this run
        output_dir (Path): path to the log directory where the result json file is written
        res_queue (mp.Queue): queue in which to signal that the task has finished running so the runtimme manager can
        clean up

    """

    start_time = time.time()
    events: list[AgentEvent] = []
    watcher = EventWatcher(task_id=task.task_id, sink=events.append)
    model_conf = run_config.model

    # the prompt the agent sees: {TASK_ID} resolved to this task's assets folder
    prompt = task.rendered_task

    # this dictionary will get passed into the shared log file
    to_log = {"id": task.task_id, "task": prompt, "failure_modes": task.failure_modes.model_dump()}

    import signal

    def handle_sigterm(
        *args,
    ):  # this function runs if this task ever gets terminated by the manager  # TODO: unused args?
        to_log["last_event"] = dataclasses.asdict(events[-1])  # type: ignore
        to_log["status"] = "terminated"
        to_log["time_elapsed"] = time.time() - start_time  # type: ignore

        res_queue.put({"task_id": task.task_id, "repetition": task.repetition, "kind": "killed", "to_log": to_log})
        try:
            partial_path = output_dir / f"{task_id_slug(task.task_id)}.partial.json"
            partial_path.write_text(json.dumps([dataclasses.asdict(e) for e in events]))
        except Exception:
            pass

        raise SystemExit(1)

    signal.signal(signal.SIGTERM, handle_sigterm)

    tool_overrides = {t.tool_name: t for t in run_config.tool_configs}
    is_closed_book = run_config.closed_book or task.closed_book

    # Per-task artifact store: oversized tool outputs are spilled next to this task's traces
    # and reports (`<output_dir>/<task_id>_artifacts/<repetition>/`), so the agent can grep or
    # page through them later instead of losing the content to a plain truncation. A
    # closed-book run has no tools, so it produces no artifacts and gets no store.
    artifact_store = None
    if run_config.artifact_spill and not is_closed_book:
        artifact_store = ArtifactStore(output_dir / f"{task_id_slug(task.task_id)}_artifacts")
        artifact_store.set_scope(f"rep{task.repetition}")

    try:
        built_agent = build_agent(
            task.agent_id,
            model_conf,
            watcher,
            tool_overrides,
            task.extra_tools,
            closed_book=is_closed_book,
            artifact_store=artifact_store,
            tool_output_max_chars=run_config.tool_output_max_chars,
        )
        out = built_agent.watcher("agent", built_agent.definition.name, built_agent.agent.run, prompt)
        success = True
        error_str = ""
        verifier_context = build_verifier_context(task, prompt, out, run_config, events, success)
        verifier_results = [
            run_check(verifier_registry.get(verifier.name), out, context=verifier_context, **verifier.kwargs)
            for verifier in task.validators
        ]
    except Exception:
        out = None
        success = False
        error_str = traceback.format_exc()
        verifier_results = []

    result = TaskResult(
        task_id=task.task_id,
        repetition=task.repetition,
        task=prompt,
        output=out,
        success=success,
        error=error_str,
        full_trace=[dataclasses.asdict(event) for event in events],
        failure_modes=task.failure_modes,
        check_results=verifier_results,
    )

    if not output_dir.exists():
        # parents=True: the runtime manager passes `<log_path>/results`, which may not exist yet
        output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{task_id_slug(task.task_id)}.jsonl"

    result = result.model_dump()

    with open(out_file, "a") as f:
        json.dump(result, f)
        f.write("\n")

    # the JSONL above is what resuming reads; this indented copy is only for humans
    trace_dir = output_dir / f"{task_id_slug(task.task_id)}_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / f"{task_id_slug(task.task_id)}.{task.repetition}.json"
    trace_file.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    to_log["checks"] = [res.model_dump() for res in verifier_results]
    to_log["status"] = "success" if success else "failed"
    to_log["time_elapsed"] = time.time() - start_time
    to_log["error"] = error_str

    res_queue.put(
        {
            "task_id": task.task_id,
            "repetition": task.repetition,
            "kind": "task_finished",
            "success": success,
            "to_log": to_log,
        }
    )

    if run_config.generate_trace_reports:
        report_dir = output_dir / f"{task_id_slug(task.task_id)}_reports/"
        if not report_dir.exists():
            os.mkdir(report_dir)

        report_file = report_dir / f"{task_id_slug(task.task_id)}.{task.repetition}_report.md"
        save_markdown_report(result, report_file)


def completed_repetitions(result_file: Path) -> set[int]:
    """
    Repetition numbers already recorded in a per-task result file.

    Lines are read defensively: a partially written trailing line (the process was
    killed mid-write) is skipped rather than counted, so an interrupted run does not
    silently lose a repetition. Each record's own `repetition` field is preferred;
    the line index is only used as a fallback for records that lack it.
    """
    repetitions: set[int] = set()
    with open(result_file, encoding="utf-8") as fh:
        for index, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            repetition = record.get("repetition") if isinstance(record, dict) else None
            repetitions.add(repetition if isinstance(repetition, int) else index)
    return repetitions
