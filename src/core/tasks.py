import dataclasses
import json
import logging
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agents.base import build_agent
from core.config import RunConfig
from core.events import AgentEvent, EventWatcher
from evaluation.base import VerificationResult, VerifierDef, run_check
from evaluation.definitions import verifier_registry
from evaluation.print_report import save_markdown_report
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

    task_id: int
    task: str
    agent_id: str
    extra_tools: list[ToolDef | str] = Field(default_factory=list)  # TODO: why allow strings?
    validators: list[VerifierDef] = Field(default_factory=list)
    closed_book: bool = False  # for the individual task flag

    repetition: int = 1  # for multiple runs of the same tasks

    @field_validator("extra_tools", mode="before")
    @classmethod
    def normalize_tools(cls, v):
        if not isinstance(v, list):
            return v
        return [{"tool_name": t} if isinstance(t, str) else t for t in v]


class TaskResult(BaseModel):
    """
    Final result that gets dumped into the log file

    (FOR NOW: really only gets used in run task but perhaps later we use it for passing around result objects and
    unloading the tasks I feel it's worth keeping it around)
    """

    task_id: int
    task: str
    repetition: int
    output: Any
    success: bool
    error: str
    full_trace: list[dict]
    check_results: list[VerificationResult] = Field(default_factory=list)


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

    # this dictionary will get passed into the shared log file
    to_log = {"id": task.task_id, "task": task.task}

    import signal

    def handle_sigterm(
        *args,
    ):  # this function runs if this task ever gets terminated by the manager  # TODO: unused args?
        to_log["last_event"] = dataclasses.asdict(events[-1])  # type: ignore
        to_log["status"] = "terminated"
        to_log["time_elapsed"] = time.time() - start_time  # type: ignore

        res_queue.put({"task_id": task.task_id, "repetition": task.repetition, "kind": "killed", "to_log": to_log})
        try:
            partial_path = output_dir / f"{task.task_id:03d}.partial.json"
            partial_path.write_text(json.dumps([dataclasses.asdict(e) for e in events]))
        except Exception:
            pass

        raise SystemExit(1)

    signal.signal(signal.SIGTERM, handle_sigterm)

    tool_overrides = {t.tool_name: t for t in run_config.tool_configs}
    is_closed_book = run_config.closed_book or task.closed_book

    try:
        built_agent = build_agent(
            task.agent_id,
            model_conf,
            watcher,
            tool_overrides,
            task.extra_tools,
            closed_book=is_closed_book
        )
        out = built_agent.watcher("agent", built_agent.definition.name, built_agent.agent.run, task.task)
        success = True
        error_str = ""
        verifier_results = [
            run_check(verifier_registry.get(verifier.name), out, **verifier.kwargs) for verifier in task.validators
        ]
    except Exception:
        out = None
        success = False
        error_str = traceback.format_exc()
        verifier_results = []

    result = TaskResult(
        task_id=task.task_id,
        repetition=task.repetition,
        task=task.task,
        output=out,
        success=success,
        error=error_str,
        full_trace=[dataclasses.asdict(event) for event in events],
        check_results=verifier_results,
    )

    if not output_dir.exists():
        os.mkdir(output_dir)

    out_file = output_dir / f"{task.task_id:03d}.jsonl"

    result = result.model_dump()

    with open(out_file, "a") as f:
        json.dump(result, f)
        f.write("\n")

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
        report_dir = output_dir / f"{task.task_id:03d}_reports/"
        if not report_dir.exists():
            os.mkdir(report_dir)

        report_file = report_dir / f"{task.task_id:03d}.{task.repetition}_report.md"
        save_markdown_report(result, report_file)
