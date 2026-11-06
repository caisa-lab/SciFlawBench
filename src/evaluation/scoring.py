from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

#: Failure modes graded by deterministic checks or a trace rubric.
QUANTITATIVE_FAILURE_MODES: tuple[str, ...] = (
    "correctness",
    "correct_tool_calls",
    "code_safety",
    "robustness_against_adversarial_inputs",
)

#: Failure modes graded by a checklist-style rubric.
QUALITATIVE_FAILURE_MODES: tuple[str, ...] = (
    "sycophancy",
    "planning",
    "reasoning",
    "uncertainty_awareness",
    "implicit_domain_knowledge",
)

#: Every canonical failure mode, quantitative first.
ALL_FAILURE_MODES: tuple[str, ...] = QUANTITATIVE_FAILURE_MODES + QUALITATIVE_FAILURE_MODES

#: The family a mode belongs to (`"quantitative"` or `"qualitative"`).
FAILURE_MODE_FAMILY: dict[str, str] = {
    **{mode: "quantitative" for mode in QUANTITATIVE_FAILURE_MODES},
    **{mode: "qualitative" for mode in QUALITATIVE_FAILURE_MODES},
}

#: The canonical modes of each family, keyed by family name.
FAILURE_MODES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "quantitative": QUANTITATIVE_FAILURE_MODES,
    "qualitative": QUALITATIVE_FAILURE_MODES,
}

#: The `failure` of a run that raised or was killed because of the model: it scores 0.
MODEL_FAILURE = "model"

#: The `failure` of a run that raised because the provider or the connection to it failed: it is left out of the
#: scores and listed, so it can be re-run.
INFRASTRUCTURE_FAILURE = "infrastructure"


class FailureModes(BaseModel):
    """
    The failure modes a task is designed to probe, split by family.

    Both keys are always present; an empty list means the task probes no failure mode of that
    family. Every entry must be a canonical mode for its family, and a task must declare at least
    one mode across the two lists, so it is always clear what an item measures.
    """

    quantitative: list[str] = Field(default_factory=list)
    qualitative: list[str] = Field(default_factory=list)

    @field_validator("quantitative", "qualitative", mode="before")
    @classmethod
    def _default_to_list(cls, v: Any) -> Any:
        return [] if v is None else v

    @model_validator(mode="after")
    def _validate_modes(self) -> "FailureModes":
        self.quantitative = _canonicalize(self.quantitative, "quantitative")
        self.qualitative = _canonicalize(self.qualitative, "qualitative")
        if not self.quantitative and not self.qualitative:
            raise ValueError(
                "a task must declare at least one failure mode (see docs/failure-modes.md); "
                "got empty 'quantitative' and 'qualitative' lists"
            )
        return self

    def all(self) -> list[str]:
        """Every mode this task declares, quantitative first."""
        return [*self.quantitative, *self.qualitative]


def _canonicalize(modes: Sequence[str], family: str) -> list[str]:
    """Validate `modes` against the canonical list for `family`, de-duplicated, order preserved."""
    allowed = FAILURE_MODES_BY_FAMILY[family]
    unknown = [mode for mode in modes if mode not in allowed]
    if unknown:
        raise ValueError(
            f"unknown {family} failure mode(s) {unknown}; allowed values are {list(allowed)} "
            "(see docs/failure-modes.md)"
        )
    seen: dict[str, None] = {}
    for mode in modes:
        seen[mode] = None
    return list(seen)


@dataclass(frozen=True)
class ModeScore:
    """Aggregate score for one bucket: the whole run, a family, or a single failure mode."""

    label: str
    family: str | None  # "quantitative" / "qualitative", or None for the overall bucket
    score: float | None  # mean over tasks of each task's mean run score, or None when the bucket has no scored runs
    passed: int  # scored runs that passed every check in the bucket
    total: int  # scored runs in the bucket (a run that failed because of the model is scored, as 0)
    tasks: int = 0  # tasks with at least one scored run in the bucket
    runs: int = 0  # every recorded run of the bucket's tasks: the scored ones plus the excluded ones
    excluded: int = 0  # runs left out: infrastructure failures, and runs without checks
    pass_rate: float | None = None  # fraction of the tasks whose mean over their scored runs is 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": None if self.score is None else round(self.score, 6),
            "pass_rate": None if self.pass_rate is None else round(self.pass_rate, 6),
            "passed": self.passed,
            "total": self.total,
            "tasks": self.tasks,
            "runs": self.runs,
            "excluded": self.excluded,
        }


@dataclass(frozen=True)
class ScoreBoard:
    """Overall, per-family and per-failure-mode scores for a set of recorded runs."""

    overall: ModeScore
    families: dict[str, ModeScore] = field(default_factory=dict)
    failure_modes: dict[str, ModeScore] = field(default_factory=dict)
    runs: int = 0  # every record seen (a repetition recorded twice counts once), including ones with nothing to score
    unscored_runs: int = 0  # records left out of every score: infrastructure failures and runs without checks
    model_failures: int = 0  # runs that raised or were killed because of the model, scored 0
    # the runs to re-run: {"task_id", "repetition", "error_type", "error"} for each infrastructure failure
    infrastructure_failures: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "unscored_runs": self.unscored_runs,
            "model_failures": self.model_failures,
            "infrastructure_failures": self.infrastructure_failures,
            "overall": self.overall.as_dict(),
            "families": {name: score.as_dict() for name, score in self.families.items()},
            "failure_modes": {name: score.as_dict() for name, score in self.failure_modes.items()},
        }


def _passed(check: Any) -> bool:
    """Whether a single check result passed, accepting either a mapping or a model."""
    if isinstance(check, Mapping):
        return bool(check.get("passed"))
    return bool(getattr(check, "passed", False))


def record_check_results(record: Mapping[str, Any]) -> list[Any]:
    """
    The checks recorded for a run.

    The runtime manager logs them under `checks` (see `aggregate_results.jsonl`), while a per-task
    `TaskResult` stores the same list under `check_results`; both shapes are accepted.
    """
    checks = record.get("checks")
    if checks is None:
        checks = record.get("check_results")
    return list(checks or [])


def task_score(check_results: Iterable[Any] | None) -> float | None:
    """
    Score one task run as the fraction of its checks that passed.

    Args:
        check_results (Iterable | None): the run's checks (mappings or result models)

    Returns (float | None): a value in `[0, 1]`, or `None` when the run has no checks to score
    """
    results = list(check_results or [])
    if not results:
        return None
    return _fraction_passed(results)


def _fraction_passed(results: list[Any]) -> float:
    """The fraction of a non-empty list of checks that passed."""
    return sum(1.0 if _passed(check) else 0.0 for check in results) / len(results)


def checks_by_failure_mode(check_results: Iterable[Any]) -> dict[str, list[Any]]:
    """
    The recorded checks grouped by the failure mode each one grades (its `failure_mode`).

    Empty for runs recorded before checks named their failure mode; those are scored the old way.
    """
    grouped: dict[str, list[Any]] = {}
    for check in check_results:
        mode = check.get("failure_mode") if isinstance(check, Mapping) else getattr(check, "failure_mode", None)
        if mode:
            grouped.setdefault(mode, []).append(check)
    return grouped


def record_failure_modes(record: Mapping[str, Any]) -> set[str]:
    """
    The failure modes a recorded run is tagged with, flattened across both families.

    Records without a `failure_modes` block (legacy runs, killed tasks) contribute to the overall
    score but to no failure mode.
    """
    declared = record.get("failure_modes") or {}
    if not isinstance(declared, Mapping):
        return set()
    modes: set[str] = set()
    for family in ("quantitative", "qualitative"):
        for mode in declared.get(family) or []:
            modes.add(mode)
    return modes


def aggregate_scores(records: Iterable[Mapping[str, Any]]) -> ScoreBoard:
    """
    Aggregate recorded runs into overall, per-family and per-failure-mode scores.

    A run is one record carrying a checks list (under `checks` or `check_results`) and, optionally,
    a `failure_modes` block, its task's `id`, its `repetition` and, for a run that raised or was
    killed, a `failure_kind` saying whose fault that was. Per-failure-mode buckets only include runs
    whose task declares that mode, so a mode is never diluted by tasks that do not probe it.

    The overall score of a run is the fraction of all its checks that passed. When the checks name
    the failure mode they grade (one check per mode), each mode is scored from its own check and each
    family from the checks of its modes; runs recorded without these labels give every declared mode
    and family the run's overall score, as before. A run that failed because of the model scores 0
    overall and for every mode and family it declares. A run that failed because of the
    infrastructure is left out of every score and listed in `infrastructure_failures`; any other
    run without checks is left out too, since there is nothing to grade (as are runs recorded before
    `failure_kind` existed that raised or were killed).

    Each task counts once: a bucket's score is the mean over its tasks of each task's mean over its
    runs. A repetition recorded more than once (re-run after resuming) counts with its last record.

    Args:
        records (Iterable[Mapping]): the run records (as written to `aggregate_results.jsonl`)

    Returns (ScoreBoard): the aggregated scores
    """
    overall = _Accumulator(label="overall", family=None)
    families = {name: _Accumulator(label=name, family=name) for name in FAILURE_MODES_BY_FAMILY}
    modes = {mode: _Accumulator(label=mode, family=FAILURE_MODE_FAMILY[mode]) for mode in ALL_FAILURE_MODES}

    runs = 0
    unscored = 0
    model_failures = 0
    infrastructure_failures: list[dict[str, Any]] = []
    for index, record in enumerate(_last_record_of_each_repetition(records)):
        runs += 1
        # A record without an id (legacy runs, tests) is its own task.
        task = record.get("id", f"record {index}")
        checks = record_check_results(record)
        score = task_score(checks)
        if score is None:
            if record.get("failure_kind") == MODEL_FAILURE:
                model_failures += 1
                for bucket in _declared_buckets(record, overall, families, modes):
                    bucket.add(task, 0.0)
                continue
            unscored += 1
            for bucket in _declared_buckets(record, overall, families, modes):
                bucket.exclude()
            if record.get("failure_kind") == INFRASTRUCTURE_FAILURE:
                infrastructure_failures.append(
                    {
                        "task_id": record.get("id"),
                        "repetition": record.get("repetition"),
                        "error_type": record.get("error_type"),
                        "error": _last_line(record),
                    }
                )
            continue

        by_mode = checks_by_failure_mode(checks)
        if not by_mode:
            for bucket in _declared_buckets(record, overall, families, modes):
                bucket.add(task, score)
            continue

        overall.add(task, score)
        for family, family_modes in FAILURE_MODES_BY_FAMILY.items():
            family_checks = [check for mode in family_modes for check in by_mode.get(mode, [])]
            if family_checks:
                families[family].add(task, _fraction_passed(family_checks))
        for mode, mode_checks in by_mode.items():
            # A mode from an unknown/legacy source is ignored rather than crashing the summary.
            if mode in modes:
                modes[mode].add(task, _fraction_passed(mode_checks))

    return ScoreBoard(
        overall=overall.build(),
        families={name: acc.build() for name, acc in families.items()},
        failure_modes={mode: acc.build() for mode, acc in modes.items()},
        runs=runs,
        unscored_runs=unscored,
        model_failures=model_failures,
        infrastructure_failures=infrastructure_failures,
    )


def _last_record_of_each_repetition(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The records to score: a repetition recorded more than once (re-run after resuming) keeps only its last."""
    latest: dict[Any, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        repetition = record.get("repetition")
        # Records written before they named their repetition cannot be matched up, so each one is kept.
        key = (record.get("id"), repetition) if "id" in record and repetition is not None else index
        latest.pop(key, None)
        latest[key] = record
    return list(latest.values())


def _declared_buckets(
    record: Mapping[str, Any],
    overall: "_Accumulator",
    families: dict[str, "_Accumulator"],
    modes: dict[str, "_Accumulator"],
) -> list["_Accumulator"]:
    """The overall bucket and the bucket of every family and mode the run's task declares."""
    declared = record_failure_modes(record)
    buckets = [overall]
    buckets += [families[family] for family, members in FAILURE_MODES_BY_FAMILY.items() if declared & set(members)]
    # A mode from an unknown/legacy source is ignored rather than crashing the summary.
    buckets += [modes[mode] for mode in declared if mode in modes]
    return buckets


def _last_line(record: Mapping[str, Any]) -> str:
    """
    The last line of a run's error (the exception and its message), shortened for the score summary; for a run
    killed at the timeout, the event it was killed in.
    """
    lines = [line.strip() for line in str(record.get("error") or "").splitlines() if line.strip()]
    if lines:
        return lines[-1][:200]
    last_event = record.get("last_event")
    if isinstance(last_event, Mapping):
        return f"killed during {last_event.get('event_type')}"
    return ""


@dataclass
class _Accumulator:
    """The run scores of one scoring bucket, kept per task so that every task counts once."""

    label: str
    family: str | None
    excluded: int = 0
    _by_task: dict[Any, list[float]] = field(default_factory=dict)

    def add(self, task: Any, score: float) -> None:
        self._by_task.setdefault(task, []).append(score)

    def exclude(self) -> None:
        self.excluded += 1

    def build(self) -> ModeScore:
        task_means = [sum(scores) / len(scores) for scores in self._by_task.values()]
        runs = [score for scores in self._by_task.values() for score in scores]
        return ModeScore(
            label=self.label,
            family=self.family,
            score=sum(task_means) / len(task_means) if task_means else None,
            passed=sum(score >= 1.0 for score in runs),
            total=len(runs),
            tasks=len(task_means),
            runs=len(runs) + self.excluded,
            excluded=self.excluded,
            pass_rate=sum(mean >= 1.0 for mean in task_means) / len(task_means) if task_means else None,
        )


def _format_score(score: float | None) -> str:
    return "  n/a" if score is None else f"{score:.3f}"


def format_scoreboard(board: ScoreBoard) -> str:
    """Render a scoreboard as a short, aligned, human-readable block for the run summary."""
    lines = [
        "Scores",
        "=" * 72,
        f"  runs recorded: {board.runs} ({board.model_failures} failed because of the model and scored 0;"
        f" {board.unscored_runs} excluded from scores, {len(board.infrastructure_failures)} of them"
        " infrastructure failures)",
        "  scores average each task's runs first; [passed/scored runs, tasks]",
        f"  {'overall':<40} {_format_score(board.overall.score)}   {_format_counts(board.overall)}",
        "",
        "  by family",
    ]
    for name, score in board.families.items():
        lines.append(f"    {name:<38} {_format_score(score.score)}   {_format_counts(score)}")

    lines.append("")
    lines.append("  by failure mode")
    for name, score in board.failure_modes.items():
        lines.append(f"    {name:<38} {_format_score(score.score)}   {_format_counts(score)}")

    if board.infrastructure_failures:
        lines.append("")
        lines.append("  infrastructure failures, left out of the scores (resume the run to re-run them)")
        for failure in board.infrastructure_failures:
            lines.append(
                f"    task {failure['task_id']}.{failure['repetition']}: {failure['error_type']}: {failure['error']}"
            )

    return "\n".join(lines)


def _format_counts(score: ModeScore) -> str:
    excluded = f", {score.excluded} excluded" if score.excluded else ""
    return f"[{score.passed}/{score.total}, {score.tasks} tasks{excluded}]"


def scoreboard_to_dict(board: ScoreBoard) -> dict[str, Any]:
    """Scoreboard as a plain, JSON-serialisable dict (what gets written to `scores.json`)."""
    return board.as_dict()
