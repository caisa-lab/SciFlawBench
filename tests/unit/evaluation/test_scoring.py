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
