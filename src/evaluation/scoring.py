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
    "aesthetic_quality",
    "lost_context_on_multi_agent_tasks",
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
    score: float | None  # mean task score, or None when the bucket has no scored runs
    passed: int  # scored runs whose task passed every check
    total: int  # scored runs in the bucket

    @property
    def pass_rate(self) -> float | None:
        """Fraction of runs in the bucket whose task passed every check."""
        return None if self.total == 0 else self.passed / self.total

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": None if self.score is None else round(self.score, 6),
            "pass_rate": None if self.pass_rate is None else round(self.pass_rate, 6),
            "passed": self.passed,
            "total": self.total,
        }


@dataclass(frozen=True)
class ScoreBoard:
    """Overall, per-family and per-failure-mode scores for a set of recorded runs."""

    overall: ModeScore
    families: dict[str, ModeScore] = field(default_factory=dict)
    failure_modes: dict[str, ModeScore] = field(default_factory=dict)
    runs: int = 0  # every record seen, including ones with nothing to score
    unscored_runs: int = 0  # records with no checks at all (killed or crashed before verification)

    def as_dict(self) -> dict[str, Any]:
        return {
            "runs": self.runs,
            "unscored_runs": self.unscored_runs,
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
    return sum(1.0 if _passed(check) else 0.0 for check in results) / len(results)


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
    a `failure_modes` block. Runs with no checks are counted in `runs`/`unscored_runs` but excluded
    from every score, since there is nothing to grade. Per-failure-mode buckets only include runs
    whose task declares that mode, so a mode is never diluted by tasks that do not probe it.

    Args:
        records (Iterable[Mapping]): the run records (as written to `aggregate_results.jsonl`)

    Returns (ScoreBoard): the aggregated scores
    """
    overall = _Accumulator(label="overall", family=None)
    families = {name: _Accumulator(label=name, family=name) for name in FAILURE_MODES_BY_FAMILY}
    modes = {mode: _Accumulator(label=mode, family=FAILURE_MODE_FAMILY[mode]) for mode in ALL_FAILURE_MODES}

    runs = 0
    unscored = 0
    for record in records:
        runs += 1
        score = task_score(record_check_results(record))
        if score is None:
            unscored += 1
            continue

        passed = score >= 1.0
        overall.add(score, passed)

        declared = record_failure_modes(record)
        for family, family_modes in FAILURE_MODES_BY_FAMILY.items():
            if declared.intersection(family_modes):
                families[family].add(score, passed)
        for mode in declared:
            # A mode from an unknown/legacy source is ignored rather than crashing the summary.
            if mode in modes:
                modes[mode].add(score, passed)

    return ScoreBoard(
        overall=overall.build(),
        families={name: acc.build() for name, acc in families.items()},
        failure_modes={mode: acc.build() for mode, acc in modes.items()},
        runs=runs,
        unscored_runs=unscored,
    )


@dataclass
class _Accumulator:
    """Running mean of task scores for one scoring bucket."""

    label: str
    family: str | None
    total: int = 0
    passed: int = 0
    _score_sum: float = 0.0

    def add(self, score: float, passed: bool) -> None:
        self.total += 1
        self._score_sum += score
        if passed:
            self.passed += 1

    def build(self) -> ModeScore:
        score = None if self.total == 0 else self._score_sum / self.total
        return ModeScore(label=self.label, family=self.family, score=score, passed=self.passed, total=self.total)


def _format_score(score: float | None) -> str:
    return "  n/a" if score is None else f"{score:.3f}"


def format_scoreboard(board: ScoreBoard) -> str:
    """Render a scoreboard as a short, aligned, human-readable block for the run summary."""
    lines = [
        "Scores",
        "=" * 72,
        f"  runs recorded: {board.runs} ({board.unscored_runs} with no checks, excluded from scores)",
        f"  {'overall':<40} {_format_score(board.overall.score)}   [{board.overall.passed}/{board.overall.total}]",
        "",
        "  by family",
    ]
    for name, score in board.families.items():
        lines.append(f"    {name:<38} {_format_score(score.score)}   [{score.passed}/{score.total}]")

    lines.append("")
    lines.append("  by failure mode")
    for name, score in board.failure_modes.items():
        lines.append(f"    {name:<38} {_format_score(score.score)}   [{score.passed}/{score.total}]")

    return "\n".join(lines)


def scoreboard_to_dict(board: ScoreBoard) -> dict[str, Any]:
    """Scoreboard as a plain, JSON-serialisable dict (what gets written to `scores.json`)."""
    return board.as_dict()
