import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.scoring import (
    ALL_FAILURE_MODES,
    QUALITATIVE_FAILURE_MODES,
    QUANTITATIVE_FAILURE_MODES,
    FailureModes,
    aggregate_scores,
    format_scoreboard,
    record_check_results,
    record_failure_modes,
    scoreboard_to_dict,
    task_score,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_TASKS = REPO_ROOT / "data" / "example" / "tasks.jsonl"


def _record(*passed: bool, quantitative=(), qualitative=()):
    """A minimal run record like the ones written to `aggregate_results.jsonl`."""
    return {
        "failure_modes": {"quantitative": list(quantitative), "qualitative": list(qualitative)},
        "checks": [{"passed": p} for p in passed],
    }


def _labelled(*checks, quantitative=(), qualitative=()):
    """A run record whose checks name the failure mode they grade; `checks` are (mode, passed) pairs."""
    return {
        "failure_modes": {"quantitative": list(quantitative), "qualitative": list(qualitative)},
        "checks": [{"passed": passed, "failure_mode": mode} for mode, passed in checks],
    }


# FailureModes validation.


def test_failure_modes_reject_unknown_names():
    with pytest.raises(ValidationError):
        FailureModes(quantitative=["not_a_real_mode"])


def test_failure_modes_reject_a_qualitative_name_in_the_quantitative_list():
    with pytest.raises(ValidationError):
        FailureModes(quantitative=["sycophancy"])


def test_failure_modes_require_at_least_one_mode():
    with pytest.raises(ValidationError):
        FailureModes(quantitative=[], qualitative=[])


def test_failure_modes_default_both_families_to_empty_lists_when_absent():
    modes = FailureModes(qualitative=["sycophancy"])
    assert modes.quantitative == []
    assert modes.all() == ["sycophancy"]


def test_failure_modes_deduplicate_preserving_order():
    modes = FailureModes(quantitative=["correctness", "correctness", "code_safety"])
    assert modes.quantitative == ["correctness", "code_safety"]


# Task scoring.


def test_task_score_is_the_fraction_of_checks_that_pass():
    assert task_score([{"passed": True}, {"passed": True}]) == 1.0
    assert task_score([{"passed": True}, {"passed": False}]) == 0.5
    assert task_score([{"passed": False}]) == 0.0


def test_task_score_is_none_without_checks():
    assert task_score([]) is None
    assert task_score(None) is None


def test_record_failure_modes_flattens_both_families():
    record = _record(True, quantitative=["correctness"], qualitative=["planning"])
    assert record_failure_modes(record) == {"correctness", "planning"}


def test_record_check_results_reads_either_key():
    assert record_check_results({"checks": [{"passed": True}]}) == [{"passed": True}]
    assert record_check_results({"check_results": [{"passed": True}]}) == [{"passed": True}]
    assert record_check_results({}) == []


def test_record_failure_modes_is_empty_for_legacy_records():
    assert record_failure_modes({}) == set()


# Aggregation.


def test_overall_score_is_the_mean_task_score():
    board = aggregate_scores([_record(True), _record(True, False), _record(False, qualitative=["sycophancy"])])

    assert board.overall.score == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert board.overall.passed == 1
    assert board.overall.total == 3
    assert board.runs == 3


def test_per_failure_mode_scores_only_count_tasks_that_declare_the_mode():
    records = [
        _record(True, quantitative=["correctness"]),
        _record(False, quantitative=["correctness"]),
        _record(True, quantitative=["code_safety"]),
    ]
    board = aggregate_scores(records)

    assert board.failure_modes["correctness"].score == pytest.approx(0.5)
    assert board.failure_modes["correctness"].total == 2
    assert board.failure_modes["code_safety"].score == pytest.approx(1.0)
    assert board.failure_modes["code_safety"].total == 1


def test_every_canonical_failure_mode_appears_even_with_no_runs():
    board = aggregate_scores([_record(True, quantitative=["correctness"])])

    assert set(board.failure_modes) == set(ALL_FAILURE_MODES)
    assert board.failure_modes["sycophancy"].score is None
    assert board.failure_modes["sycophancy"].total == 0


def test_family_scores_only_count_tasks_in_that_family():
    records = [
        _record(True, quantitative=["correctness"]),
        _record(False, qualitative=["planning"]),
    ]
    board = aggregate_scores(records)

    assert board.families["quantitative"].score == pytest.approx(1.0)
    assert board.families["quantitative"].total == 1
    assert board.families["qualitative"].score == pytest.approx(0.0)
    assert board.families["qualitative"].total == 1


def test_runs_without_checks_are_excluded_but_still_counted():
    board = aggregate_scores(
        [
            _record(True, quantitative=["correctness"]),
            {"failure_modes": {"quantitative": ["correctness"]}},  # e.g. a killed task: no checks
        ]
    )

    assert board.runs == 2
    assert board.unscored_runs == 1
    assert board.overall.total == 1
    assert board.failure_modes["correctness"].total == 1


def test_modes_and_families_are_listed_in_canonical_order():
    board = aggregate_scores([])

    assert list(board.families) == ["quantitative", "qualitative"]
    assert list(board.failure_modes) == list(ALL_FAILURE_MODES)


def test_failure_modes_line_up_with_the_documented_families():
    assert set(QUANTITATIVE_FAILURE_MODES).isdisjoint(QUALITATIVE_FAILURE_MODES)
    assert set(ALL_FAILURE_MODES) == set(QUANTITATIVE_FAILURE_MODES) | set(QUALITATIVE_FAILURE_MODES)


# Rendering.


def test_scoreboard_serialises_to_json():
    board = aggregate_scores([_record(True, quantitative=["correctness"])])

    payload = scoreboard_to_dict(board)
    reloaded = json.loads(json.dumps(payload))

    assert reloaded["overall"]["score"] == 1.0
    assert reloaded["failure_modes"]["correctness"]["total"] == 1
    assert reloaded["failure_modes"]["sycophancy"]["score"] is None


def test_format_scoreboard_reports_every_mode():
    board = aggregate_scores([_record(False, quantitative=["correctness"])])
    text = format_scoreboard(board)

    assert "overall" in text
    assert "by failure mode" in text
    for mode in ALL_FAILURE_MODES:
        assert mode in text


# The shipped example task file.


def test_example_tasks_all_declare_valid_failure_modes():
    tasks = [json.loads(line) for line in EXAMPLE_TASKS.read_text().splitlines() if line.strip()]

    assert tasks, "the example task file should not be empty"
    for index, task in enumerate(tasks, start=1):
        assert "failure_modes" in task, f"task {index} in {EXAMPLE_TASKS.name} has no failure_modes"
        modes = FailureModes(**task["failure_modes"])  # raises on unknown names / no modes
        assert modes.all(), f"task {index} declares no failure mode"


# One labelled check per failure mode.


def test_each_mode_is_scored_from_its_own_check():
    # the number is right, but the run goes along with a false premise
    record = _labelled(
        ("correctness", True), ("sycophancy", False), quantitative=["correctness"], qualitative=["sycophancy"]
    )

    board = aggregate_scores([record])

    assert board.overall.score == 0.5
    assert (board.failure_modes["correctness"].score, board.failure_modes["correctness"].pass_rate) == (1.0, 1.0)
    assert (board.failure_modes["sycophancy"].score, board.failure_modes["sycophancy"].pass_rate) == (0.0, 0.0)
    assert board.families["quantitative"].score == 1.0
    assert board.families["qualitative"].score == 0.0


def test_mode_scores_average_over_every_task_that_declares_the_mode():
    records = [
        _labelled(
            ("correctness", True), ("sycophancy", False), quantitative=["correctness"], qualitative=["sycophancy"]
        ),
        _labelled(("correctness", False), quantitative=["correctness"]),
        _labelled(("sycophancy", True), qualitative=["sycophancy"]),
    ]

    board = aggregate_scores(records)

    correctness, sycophancy = board.failure_modes["correctness"], board.failure_modes["sycophancy"]
    assert (correctness.score, correctness.passed, correctness.total) == (0.5, 1, 2)
    assert (sycophancy.score, sycophancy.passed, sycophancy.total) == (0.5, 1, 2)
    # the sycophancy-only task has no quantitative check, so it is not in that family
    assert (board.families["quantitative"].score, board.families["quantitative"].total) == (0.5, 2)


def test_runs_without_labels_are_scored_as_before_next_to_labelled_runs():
    records = [
        _record(True, False, quantitative=["correctness"], qualitative=["sycophancy"]),
        _labelled(("correctness", True), quantitative=["correctness"]),
    ]

    board = aggregate_scores(records)

    # the unlabelled run gives each of its modes its task score, 0.5
    assert board.failure_modes["sycophancy"].score == 0.5
    assert board.failure_modes["correctness"].score == 0.75


def test_example_tasks_grade_each_mode_with_one_labelled_check():
    from core.tasks import TaskDef

    for line in EXAMPLE_TASKS.read_text().splitlines():
        if line.strip():
            TaskDef.model_validate(json.loads(line))  # raises if the checks and modes do not match one to one


# Failed runs, re-runs and per-task averages.


def _run(task, repetition, *checks, failure=None, error_type=None, error="", quantitative=("correctness",), **modes):
    """A run record of `task`; `checks` are (mode, passed) pairs, and a run that failed has none."""
    record = _labelled(*checks, quantitative=quantitative, qualitative=modes.get("qualitative", ()))
    record.update(
        {"id": task, "repetition": repetition, "failure_kind": failure, "error_type": error_type, "error": error}
    )
    return record


def test_each_task_counts_once_whatever_its_number_of_runs():
    records = [
        _run("a", 1, ("correctness", True)),
        _run("a", 2, ("correctness", True)),
        _run("a", 3, ("correctness", True)),
        _run("b", 1, ("correctness", False)),
    ]

    board = aggregate_scores(records)

    # task a scores 1 and task b scores 0; a's three runs do not outweigh b's one
    correctness = board.failure_modes["correctness"]
    assert (correctness.score, correctness.passed, correctness.total, correctness.tasks) == (0.5, 3, 4, 2)
    assert board.overall.score == 0.5


def test_a_run_that_failed_because_of_the_model_scores_zero_for_every_declared_mode():
    records = [
        _run("a", 1, ("correctness", True), ("sycophancy", True), qualitative=["sycophancy"]),
        _run("a", 2, failure="model", qualitative=["sycophancy"]),  # raised, or was killed at the timeout
    ]

    board = aggregate_scores(records)

    assert board.model_failures == 1
    assert board.unscored_runs == 0
    for bucket in (board.overall, board.failure_modes["correctness"], board.failure_modes["sycophancy"]):
        assert (bucket.score, bucket.passed, bucket.total, bucket.tasks) == (0.5, 1, 2, 1)
    assert board.families["qualitative"].score == 0.5


def test_a_run_that_failed_because_of_the_infrastructure_is_left_out_and_listed():
    error = "Traceback (most recent call last):\n  ...\nopenai.RateLimitError: Error code: 429 - no credits\n"
    records = [
        _run("a", 1, ("correctness", True)),
        _run("a", 2, failure="infrastructure", error_type="RateLimitError", error=error),
    ]

    board = aggregate_scores(records)

    assert (board.runs, board.unscored_runs, board.model_failures) == (2, 1, 0)
    assert board.infrastructure_failures == [
        {
            "task_id": "a",
            "repetition": 2,
            "error_type": "RateLimitError",
            "error": "openai.RateLimitError: Error code: 429 - no credits",
        }
    ]
    correctness = board.failure_modes["correctness"]
    assert (correctness.score, correctness.total, correctness.runs, correctness.excluded) == (1.0, 1, 2, 1)
    assert "task a.2: RateLimitError: openai.RateLimitError" in format_scoreboard(board)


def test_a_repetition_run_again_counts_with_its_last_record():
    records = [
        _run("a", 1, failure="infrastructure", error="openai.APIConnectionError: Connection error."),
        _run("a", 1, ("correctness", False)),
    ]

    board = aggregate_scores(records)

    assert (board.runs, board.unscored_runs, board.infrastructure_failures) == (1, 0, [])
    assert board.overall.score == 0.0


def test_the_pass_rate_is_the_fraction_of_tasks_that_pass_on_average():
    records = [
        _run("a", 1, ("correctness", True)),
        _run("a", 2, ("correctness", True)),
        _run("b", 1, ("correctness", True)),
        _run("b", 2, ("correctness", False)),
        _run("b", 3, failure="model"),
    ]

    correctness = aggregate_scores(records).failure_modes["correctness"]

    assert correctness.score == pytest.approx((1.0 + 1 / 3) / 2)
    # task a passes on average, task b does not
    assert correctness.pass_rate == 0.5
    assert (correctness.passed, correctness.total, correctness.tasks) == (3, 5, 2)


def test_records_from_before_failure_kinds_are_left_out_as_before():
    # a run recorded before this change that crashed: no checks, no failure_kind
    legacy = {"id": "a", "failure_modes": {"quantitative": ["correctness"]}, "status": "failed", "error": "boom"}

    board = aggregate_scores([_run("a", 1, ("correctness", True)), legacy])

    assert (board.model_failures, board.unscored_runs, board.infrastructure_failures) == (0, 1, [])
    assert board.overall.score == 1.0
