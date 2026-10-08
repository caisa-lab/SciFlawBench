import json
import logging
import multiprocessing as mp
import os
import queue as q
import time
from pathlib import Path
from typing import Any

from core.config import ModelConfig, RunConfig
from core.manifest import (
    OVERRIDE_ENV_VAR,
    build_run_manifest,
    build_run_signature,
    load_run_manifest,
    manifest_sha256,
    reproducibility_override,
    save_run_manifest,
    validate_resume,
)
from core.task_ids import parse_task_id_slug, task_id_slug
from core.tasks import TaskDef, completed_repetitions, run_task
from evaluation.scoring import ScoreBoard, aggregate_scores, format_scoreboard, scoreboard_to_dict

mp.set_start_method(
    "spawn", force=True
)  # IMPORTANT: this means it will not just fork the process, which means slightly
# slower start times but ultimately saves from pain when it comes to possible
# deadlocks with open file descriptors and networking (although im willing to
# remove if we promise to be careful about not doing any of that stuff with
# the main process)

logger = logging.getLogger(__file__)


class RuntimeManager:
    """
    This is the class that orchestrates running all of the tasks from the tasklist through the run method. It is
    configured mainly through the config.json file that holds all the necessary information needed to provision a test
    """

    full_conf: RunConfig
    model_conf: ModelConfig
    ...

    def __init__(self, conf: RunConfig):
        # TODO: set attributes at the top as well for easier IDE navigation
        self.full_conf = conf

        logging.basicConfig(level=conf.logging_level)
        logger.info("Initializing runtime manager for current run")
        self.model_conf = conf.model

        # `restarting` means "resume in place": use `log_path` as the run directory
        # instead of creating a fresh timestamped one.
        if conf.restarting:
            self.log_path = conf.log_path
        else:
            self.log_path = conf.log_path / time.strftime("%Y-%m-%d %H:%M:%S")

        self.task_file = conf.task_file
        self.max_concurrent = conf.max_concurrent
        self.task_timeout_s = conf.task_timeout_s
        self.repetitions = conf.repetitions_per_task

        self.results_dir = self.log_path / "results"
        os.makedirs(self.results_dir, exist_ok=True)

        self.shared_results_jsonl = self.log_path / "aggregate_results.jsonl"
        self.run_summary_file = self.log_path / "run_summary.log"
        self.score_file = self.log_path / "scores.json"

        # Reproducibility: every run records an immutable manifest, and resuming is
        # only allowed while the manifest's checksums still match this configuration.
        self.tasks = self.load_tasks()
        self.run_signature = build_run_signature(conf, self.tasks)
        self.manifest = self.prepare_manifest()
        self.manifest_sha256 = manifest_sha256(self.manifest) if self.manifest else None

        self._result_queue = mp.Queue()
        self._active: dict[tuple[int | str, int], dict] = {}  # TODO: make this a data model?
        self._pending = self.pending_tasks()
        logger.info("Runtime manager initialized")

    def load_tasks(self) -> list[TaskDef]:
        """
        Read every task defined in the task file, unfiltered.

        Filtering out work that has already been recorded is :meth:`pending_tasks`'
        job. Returns a list of task definitions (see tasks.py) read from the task
        file specified through the configuration file.
        """
        tasks = []

        with open(self.task_file) as f:
            for line_num, raw_line in enumerate(f):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    tasks.append(TaskDef(**json.loads(line)))
                except Exception as e:
                    logger.error(f"Failed to load task on line {line_num + 1}: got the following error: {e}")
                    exit(1)
        return tasks

    def pending_tasks(self) -> list[TaskDef]:
        """
        The `(task, repetition)` runs still outstanding, in task-file order.

        When resuming in place, repetitions already recorded under `results/` are
        skipped. Those result files were validated against the run manifest before
        this point (see :meth:`prepare_manifest`), so only matching work is skipped.
        """
        completed = self.load_completed()
        pending = []
        for task in self.tasks:
            for i in range(1, self.repetitions + 1):
                if (task.task_id, i) not in completed:
                    pending.append(task.model_copy(update={"repetition": i}))

        total = len(self.tasks) * self.repetitions
        if completed:
            logger.info(
                f"Checkpoint: {total - len(pending)} of {total} run(s) already recorded, {len(pending)} remaining."
            )
        return pending

    def prepare_manifest(self) -> dict[str, Any] | None:
        """
        Load the run manifest when resuming, or create it for a fresh run.

        Resuming verifies the manifest's checksums against this run's signature and
        aborts unless they match, so a checkpoint can never be continued with
        different tasks, assets, prompts, tools, model settings or budget. Set
        `NO_REPRODUCIBILITY_GUARANTEES=true` to downgrade this to a warning.
        """
        existing = load_run_manifest(self.log_path)
        if existing is not None:
            problems = validate_resume(self.log_path, self.run_signature)
            if problems and not reproducibility_override():
                logger.error("Refusing to resume: the checkpoint does not match this run.")
                for problem in problems:
                    logger.error(f"   - {problem}")
                logger.error(f"   Set {OVERRIDE_ENV_VAR}=true to resume anyway (results may not be reproducible).")
                raise SystemExit(1)
            for problem in problems:
                logger.warning(f"Ignoring checkpoint problem ({OVERRIDE_ENV_VAR} is set): {problem}")
            logger.info(f"Resuming checkpoint; manifest sha256 {manifest_sha256(existing)[:12]}...")
            return existing

        if self.load_completed():
            problem = "checkpoint holds results but no run manifest, so its checksums cannot be verified."
            if not reproducibility_override():
                logger.error(f"Refusing to resume: {problem}")
                logger.error(f"   Set {OVERRIDE_ENV_VAR}=true to resume anyway (results may not be reproducible).")
                raise SystemExit(1)
            logger.warning(f"Ignoring unverifiable checkpoint ({OVERRIDE_ENV_VAR} is set): {problem}")

        manifest = build_run_manifest(conf=self.full_conf, tasks=self.tasks, run_signature=self.run_signature)
        save_run_manifest(self.log_path, manifest)
        logger.info(f"Wrote run manifest (sha256 {manifest_sha256(manifest)[:12]}...).")
        return manifest

    def load_completed(self) -> set[tuple[int | str, int]]:
        """
        Scan the run's `results/` directory for runs that are already recorded.

        Returns (Set[Tuple[int | str, int]]): completed `(task_id, repetition)` pairs.
        Task ids are recovered from the result file names, which are the task id
        slugs: zero-padded integers (`"001"`) for legacy ids, or the raw
        32-character hash for SciFlawBench ids.
        """
        res: set[tuple[int | str, int]] = set()
        if not self.results_dir.is_dir():
            logger.info(f"loaded no completed items from {self.log_path}")
            return res

        for path in self.results_dir.glob("*.jsonl"):
            task_id = parse_task_id_slug(path.stem)
            for repetition in completed_repetitions(path):
                res.add((task_id, repetition))
        return res

    def run(self) -> None:
        """
        This function starts the main loop that launches subprocesses to run tasks. it will try to launch subprocesses
        until it's less than the current maximum concurrent and while there are still tasks to be run (in pending).
        """

        while self._pending or self._active:
            while self._pending and len(self._active) < self.max_concurrent:
                task = self._pending.pop(0)
                repetition = task.repetition
                proc = self._spawn_task(
                    task=task, conf=self.full_conf, output_dir=self.results_dir, res_queue=self._result_queue
                )

                self._active[(task.task_id, repetition)] = {"proc": proc, "started": time.time()}

                if self.repetitions <= 1:
                    logger.info(f"Task id - ({task_id_slug(task.task_id)}) started")
                else:
                    logger.info(f"Task id - ({task_id_slug(task.task_id)}.{task.repetition}) started")

            self._drain_results()
            self._check_timeouts()

        self._drain_remaining()
        self.write_scores()

    def load_records(self) -> list[dict[str, Any]]:
        """
        Every run record written to `aggregate_results.jsonl` so far, defensively parsed.

        A partially written trailing line (the manager was interrupted mid-write) is skipped
        rather than crashing the aggregation.
        """
        records: list[dict[str, Any]] = []
        if not self.shared_results_jsonl.is_file():
            return records
        with open(self.shared_results_jsonl, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    def write_scores(self) -> ScoreBoard:
        """
        Aggregate the run's recorded results into overall and per-failure-mode scores.

        The scoreboard is written to `scores.json` next to the run's logs and appended to
        `run_summary.log`, so a finished run reports both its overall score and one score per
        failure mode, each restricted to the tasks that declare that mode.
        """
        board = aggregate_scores(self.load_records())
        self.score_file.write_text(json.dumps(scoreboard_to_dict(board), indent=2) + "\n", encoding="utf-8")
        with open(self.run_summary_file, "a", encoding="utf-8") as f:
            f.write("\n" + format_scoreboard(board) + "\n")
        logger.info("Wrote scores to %s", self.score_file)
        return board

    def _spawn_task(self, task: TaskDef, conf: RunConfig, output_dir: Path, res_queue: mp.Queue) -> mp.Process:
        """
        simply spawns a subprocess which actually runs the task with the agent setup and model configuration specified

        Args:
            task (TaskDef): definition of the task (includes the agentic preset to be run)
            conf (ModelConfig): configuration struct containing what is needed to provision a fresh model
            output_dir (Path): path to the run's `results` directory, where the task writes its result file
            res_queue (mp.Queue): queue used to track state of active processes and when they finish

        Returns (mp.Process): a process class handler class which will be tracked through the _active queue
        """
        p = mp.Process(
            target=run_task,
            args=(
                task,
                conf,
                output_dir,
                res_queue,
            ),
        )
        p.start()
        return p

    def _drain_results(self, timeout: float = 3.0):
        """
        Function run at the end of the spawning loop which basically just checks for finished processes and reaps them
        upon having completed. Adding important results to the shared results file

        Args:
            timeout (float): amount of seconds to wait while accessing the result queue
        """
        try:
            msg = self._result_queue.get(timeout=timeout)
        except q.Empty:
            return

        # reap the finished process
        task_id = msg["task_id"]
        repetition = msg["repetition"]

        entry = self._active.pop((task_id, repetition), None)
        if entry:
            entry["proc"].join(timeout=5)

        self._handle_message(msg)

    def _check_timeouts(self):
        """
        Another function run at the end of the spawning loop which basically just checks the active processes and kills
        them if they don't complete in the specified amount of seconds. For the moment this waits 15 minutes on any
        given process but this can be configured fairly easily if we find that we need different time scales
        """
        now = time.time()
        to_kill: list[tuple[int | str, int]] = []

        for (task_id, repitition), entry in self._active.items():
            if now - entry["started"] > self.task_timeout_s:
                entry["proc"].terminate()
                entry["proc"].join(timeout=10)  # 10 seconds for the process to clean up after itself
                if entry["proc"].is_alive():
                    entry["proc"].kill()
                    entry["proc"].join(timeout=5)
                to_kill.append((task_id, repitition))
                logger.info(f"Task: {task_id_slug(task_id)}.{repitition} timed out...")

        # update dictionary state associated with killed tasks
        for task_id, rep in to_kill:
            del self._active[(task_id, rep)]

    def _handle_message(self, msg: dict[str, Any]):
        """
        takes a message in and handles logging according to what the message content is

        Args:
            msg (Dict[str, Any]): the message being sent by the subprocess to be logged
        """
        task_id = msg["task_id"]
        repetition = msg["repetition"]
        log = msg["to_log"]
        log["manifest_sha256"] = self.manifest_sha256

        match msg["kind"]:
            case "task_finished":
                if msg["success"]:
                    logger.info(f"Task: {task_id_slug(task_id)}.{repetition} completed successfully!")
                else:
                    err = log["error"]
                    logger.info(
                        f"Task: {task_id_slug(task_id)}.{repetition} completed execution with following errors:\n {err}"
                    )
            case "killed":
                logger.info(f"Task: {task_id_slug(task_id)}.{repetition} reaped. Killed by timeout.")
            case _:
                logger.info(f"Task: {task_id_slug(task_id)}.{repetition} finished with undefined state...")

        with open(self.shared_results_jsonl, "a") as f:
            f.write(json.dumps(log) + "\n")

        with open(self.run_summary_file, "a") as f:
            line = f"[{log['status'].upper():9}] task {log['id']:>4}.{repetition}  {log['time_elapsed']:.1f}s"
            if log["status"] == "success":
                checks = log.get("checks", [])
                passed = sum(int(check["passed"]) for check in checks)
                line += f"  Checks:     {passed}/{len(checks)}"
                if checks:
                    line += f"  Score: {passed / len(checks):.2f}"
            else:
                line += f"  Last Event: {log.get('last_event', '')}"
            f.write(line + "\n")

    def _drain_remaining(self):
        """
        This function runs after the main loop has been completed and handles draining
        any remaining results in the result queue (e.g. handling logging task info)
        """

        while True:
            try:
                msg = self._result_queue.get(timeout=3.0)
            except q.Empty:
                break

            self._handle_message(msg)
