import dataclasses
import fcntl
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import openai
from pydantic import BaseModel, Field, field_validator, model_validator

from agents.base import build_agent
from core.config import DEFAULT_MAX_TRACE_TOKENS, RunConfig
from core.events import AgentEvent, EventType, EventWatcher
from core.task_ids import FIXED_ID_PATTERN, TaskId, hash_prompt, render_task, task_id_slug
from evaluation.base import VerificationResult, VerifierContext, VerifierDef, run_check, wants_context
from evaluation.definitions import verifier_registry
from evaluation.judge import LiteLLMJudge
from evaluation.print_report import render_trace_markdown, save_markdown_report
from evaluation.scoring import INFRASTRUCTURE_FAILURE, MODEL_FAILURE, FailureModes
from models.memory import MemoryOffloader
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
    prompt_md5: str | None = None  # md5(task) when the task keeps a fixed id; None when the id is the md5
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
        Guard against modified samples: a hash-style id, or `prompt_md5` when given, must equal `md5(task)`.

        The prompt is hashed *as stored* (i.e. with any `{TASK_ID}` placeholder
        still in place), so editing a prompt without recomputing its hash is
        rejected at load time. A task with `prompt_md5` keeps a fixed id (see
        `FIXED_ID_PATTERN`) that does not change when the prompt is edited.
        Integer ids (legacy/ad-hoc tasks) without `prompt_md5` are left alone.
        """
        expected = hash_prompt(self.task)
        if self.prompt_md5 is not None:
            if not isinstance(self.task_id, str) or not FIXED_ID_PATTERN.fullmatch(self.task_id):
                raise ValueError(
                    f"task_id {self.task_id!r} is not a valid fixed id: with prompt_md5 the id must start with a "
                    "letter and use only letters, digits, '.', '_' and '-' (e.g. 'SFB-0042')"
                )
            if self.prompt_md5 != expected:
                raise ValueError(
                    f"prompt_md5 {self.prompt_md5!r} does not match md5(task) {expected!r}: "
                    "the task text was modified without updating prompt_md5"
                )
        elif isinstance(self.task_id, str) and self.task_id != expected:
            raise ValueError(
                f"task_id {self.task_id!r} does not match md5(task) {expected!r}: "
                "the task text was modified without updating its id"
            )
        return self

    @model_validator(mode="after")
    def verify_one_check_per_failure_mode(self):
        """
        Each declared failure mode is graded by exactly one check, which names it in `failure_mode`.

        The checks follow the order in which the modes are declared (quantitative first), so the label and
        the position must agree; per-mode scores are then built from each mode's own check. A task without
        checks has nothing to score and is left alone.
        """
        if not self.validators:
            return self
        modes = self.failure_modes.all()
        labels = [validator.failure_mode for validator in self.validators]
        if labels != modes:
            raise ValueError(
                "each failure mode needs exactly one check, labelled with it in 'failure_mode' and listed in the "
                f"order the modes are declared (quantitative first): modes {modes}, check labels {labels}"
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
    prompt_md5: str | None = None  # which version of the prompt ran, for a task with a fixed id
    repetition: int
    output: Any
    success: bool
    error: str
    full_trace: list[dict]
    failure_modes: FailureModes | None = None
    check_results: list[VerificationResult] = Field(default_factory=list)
    # for a run that raised or was killed: "model" (it scores 0) or "infrastructure" (left out of the scores and run
    # again on resume), and the class of the error that decided it
    failure_kind: str | None = None
    error_type: str | None = None


#: Errors that are not the model's fault. litellm's errors subclass the matching openai ones, so these cover both
#: providers.
_INFRASTRUCTURE_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    openai.APIConnectionError,  # includes timeouts
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,  # the provider does not know the model or the endpoint
    openai.RateLimitError,
)

#: litellm's errors for a request it refuses to send. They are looked up only when litellm is loaded (importing it
#: takes a second), since none of them can have been raised otherwise.
_LITELLM_INFRASTRUCTURE_ERRORS = ("UnsupportedParamsError", "BudgetExceededError")

#: The `code` of an OpenAI-style 400 for a parameter or value the model does not accept.
_UNSUPPORTED_PARAMETER_CODES = {"unsupported_parameter", "unsupported_value"}


def classify_failure(exc: BaseException) -> tuple[str, str]:
    """
    Whether a run that raised failed because of the model or because of the infrastructure around it.

    The decision rests on exception types (and a provider's error `code`), never on messages. An error is the
    infrastructure's when it, or any error in its chain of causes, is a connection, authentication, rate-limit,
    unknown-model or provider (5xx) error, a request the provider rejects for an unsupported parameter (litellm's
    `UnsupportedParamsError`, or an OpenAI 400 coded `unsupported_parameter` or `unsupported_value`), or a litellm
    budget error. Anything else is the model's, on purpose also a `ContextWindowExceededError` (the model filled its
    context) and a provider refusing the conversation's content (`ContentPolicyViolationError`,
    `BlockedPiiEntityError`), which a re-run would only repeat. Providers' errors usually arrive wrapped (smolagents
    raises `AgentGenerationError` from them), so the whole chain is searched.

    Args:
        exc (BaseException): the exception the run raised

    Returns (tuple[str, str]): `INFRASTRUCTURE_FAILURE` or `MODEL_FAILURE`, and the class name of the error that
    decided it: the matching error for the infrastructure, the innermost error of the chain for the model
    """
    litellm_exceptions = sys.modules.get("litellm.exceptions")
    litellm_errors = tuple(
        error
        for error in (getattr(litellm_exceptions, name, None) for name in _LITELLM_INFRASTRUCTURE_ERRORS)
        if isinstance(error, type)
    )
    seen: set[int] = set()
    current: BaseException | None = exc
    innermost = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        innermost = current
        if (
            isinstance(current, _INFRASTRUCTURE_ERRORS + litellm_errors)
            or (isinstance(current, openai.APIStatusError) and current.status_code >= 500)
            or (isinstance(current, openai.BadRequestError) and current.code in _UNSUPPORTED_PARAMETER_CODES)
        ):
            return INFRASTRUCTURE_FAILURE, type(current).__name__
        current = current.__cause__ or current.__context__
    return MODEL_FAILURE, type(innermost).__name__


def timeout_failure_kind(last_event: AgentEvent | None) -> str:
    """
    Whose fault a run killed at `task_timeout_s` is, judged by the last event it recorded.

    A kill inside a tool call (the last event is a `tool_call_start`) is a hung tool or network call, and a kill
    before the agent recorded any event is a hung setup: both are the infrastructure's. Any other kill, e.g. inside
    a model call that never returned or in the agent's own code, is the model's.

    Args:
        last_event (AgentEvent | None): the run's last event, or None when it recorded none

    Returns (str): `INFRASTRUCTURE_FAILURE` or `MODEL_FAILURE`
    """
    if last_event is None or last_event.event_type == EventType.ToolCallStart:
        return INFRASTRUCTURE_FAILURE
    return MODEL_FAILURE


def verifier_wants_context(name: str) -> bool:
    """Whether the verifier registered under `name` accepts the optional `context` argument."""
    try:
        return wants_context(verifier_registry.get(name))
    except KeyError:
        return False  # unknown names are reported as a failed check, not here


def trace_token_budget(run_config: RunConfig) -> int:
    """
    Token budget for rendering a run's trace: the judge's `max_trace_tokens`, or its default without a judge.

    The judge and the per-task trace reports use the same budget, so a report shows what the judge read.
    """
    return run_config.judge.max_trace_tokens if run_config.judge is not None else DEFAULT_MAX_TRACE_TOKENS


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
        },
        max_tokens=trace_token_budget(run_config),
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
    to_log = {
        "id": task.task_id,
        "repetition": task.repetition,
        "task": prompt,
        "failure_modes": task.failure_modes.model_dump(),
    }

    import signal

    def handle_sigterm(
        *args,
    ):  # this function runs if this task ever gets terminated by the manager  # TODO: unused args?
        to_log["last_event"] = dataclasses.asdict(events[-1]) if events else None  # type: ignore
        to_log["status"] = "terminated"
        to_log["failure_kind"] = timeout_failure_kind(events[-1] if events else None)
        to_log["error_type"] = "task_timeout"
        to_log["time_elapsed"] = time.time() - start_time  # type: ignore

        res_queue.put({"task_id": task.task_id, "repetition": task.repetition, "kind": "killed", "to_log": to_log})
        try:
            partial_path = output_dir / f"{task_id_slug(task.task_id)}.{task.repetition}.partial.json"
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
    memory_offloader = None
    if run_config.artifact_spill and not is_closed_book:
        artifact_store = ArtifactStore(output_dir / f"{task_id_slug(task.task_id)}_artifacts")
        artifact_store.set_scope(f"rep{task.repetition}")
        # Messages evicted by context trimming go to `memory.md` in that same scope
        if run_config.memory_offload:
            memory_offloader = MemoryOffloader(
                artifact_store,
                digest_chars=run_config.memory_digest_chars,
                index_max_chars=run_config.memory_index_max_chars,
            )

    failure_kind = error_type = None
    agent_running = False
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
            memory_offloader=memory_offloader,
        )
        agent_running = True
        out = built_agent.watcher("agent", built_agent.definition.name, built_agent.agent.run, prompt)
        agent_running = False
        success = True
        error_str = ""
        verifier_context = build_verifier_context(task, prompt, out, run_config, events, success)
        verifier_results = [
            run_check(
                verifier_registry.get(verifier.name), out, context=verifier_context, **verifier.kwargs
            ).model_copy(update={"failure_mode": verifier.failure_mode})
            for verifier in task.validators
        ]
    except Exception as exc:
        out = None
        success = False
        error_str = traceback.format_exc()
        verifier_results = []
        # Only the agent run itself can fail because of the model; building the agent and grading its answer are
        # the harness's part, so an error there is not held against the model.
        if agent_running:
            failure_kind, error_type = classify_failure(exc)
        else:
            failure_kind, error_type = INFRASTRUCTURE_FAILURE, type(exc).__name__

    result = TaskResult(
        task_id=task.task_id,
        prompt_md5=task.prompt_md5,
        repetition=task.repetition,
        task=prompt,
        output=out,
        success=success,
        error=error_str,
        full_trace=[dataclasses.asdict(event) for event in events],
        failure_modes=task.failure_modes,
        check_results=verifier_results,
        failure_kind=failure_kind,
        error_type=error_type,
    )

    if not output_dir.exists():
        # parents=True: the runtime manager passes `<log_path>/results`, which may not exist yet
        output_dir.mkdir(parents=True, exist_ok=True)

    out_file = output_dir / f"{task_id_slug(task.task_id)}.jsonl"

    result = result.model_dump()

    append_result_line(out_file, result)

    # the JSONL above is what resuming reads; this indented copy is only for humans
    trace_dir = output_dir / f"{task_id_slug(task.task_id)}_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file = trace_dir / f"{task_id_slug(task.task_id)}.{task.repetition}.json"
    trace_file.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    to_log["checks"] = [res.model_dump() for res in verifier_results]
    to_log["status"] = "success" if success else "failed"
    to_log["time_elapsed"] = time.time() - start_time
    to_log["error"] = error_str
    to_log["failure_kind"] = failure_kind
    to_log["error_type"] = error_type

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
        save_markdown_report(result, report_file, max_tokens=trace_token_budget(run_config))


def append_result_line(result_file: Path, record: dict) -> None:
    """
    Append one record as a JSON line to a per-task result file.

    Repetitions of a task run in parallel and can finish at the same moment. A record holds the full trace and is
    often 50-120 KB, so a buffered write reaches the file in several pieces that can interleave with another
    process's; the exclusive lock keeps every line whole.
    """
    line = json.dumps(record) + "\n"
    with open(result_file, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # released when the file is closed, after its buffer is flushed
        fh.write(line)


def completed_repetitions(result_file: Path) -> set[int]:
    """
    Repetition numbers already recorded in a per-task result file.

    Lines are read defensively: a partially written trailing line (the process was
    killed mid-write) is skipped rather than counted, so an interrupted run does not
    silently lose a repetition. Each record's own `repetition` field is preferred;
    the line index is only used as a fallback for records that lack it. A repetition
    that failed because of the infrastructure is not counted, so resuming runs it again.
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
            if isinstance(record, dict) and record.get("failure_kind") == INFRASTRUCTURE_FAILURE:
                continue
            repetition = record.get("repetition") if isinstance(record, dict) else None
            repetitions.add(repetition if isinstance(repetition, int) else index)
    return repetitions
