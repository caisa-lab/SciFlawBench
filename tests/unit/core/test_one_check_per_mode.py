import pytest
from pydantic import ValidationError

from core.task_ids import hash_prompt
from core.tasks import TaskDef

MODES = {"quantitative": ["correctness"], "qualitative": ["sycophancy"]}
NUMBER = {"name": "content:numeric_match", "kwargs": {"expected": "1"}}
RUBRIC = {"name": "judge:rubric", "kwargs": {"rubric": "r"}}


def _task(validators):
    return TaskDef(
        task_id=hash_prompt("p"), task="p", agent_id="code_agent", failure_modes=MODES, validators=validators
    )


def test_one_labelled_check_per_mode_in_order_is_accepted():
    task = _task([{**NUMBER, "failure_mode": "correctness"}, {**RUBRIC, "failure_mode": "sycophancy"}])

    assert [validator.failure_mode for validator in task.validators] == ["correctness", "sycophancy"]


@pytest.mark.parametrize(
    "validators",
    [
        pytest.param([NUMBER, RUBRIC], id="no labels"),
        pytest.param([{**NUMBER, "failure_mode": "correctness"}, RUBRIC], id="a label missing"),
        pytest.param(
            [{**RUBRIC, "failure_mode": "sycophancy"}, {**NUMBER, "failure_mode": "correctness"}], id="wrong order"
        ),
        pytest.param(
            [
                {**NUMBER, "failure_mode": "correctness"},
                {**NUMBER, "failure_mode": "correctness"},
                {**RUBRIC, "failure_mode": "sycophancy"},
            ],
            id="two checks for one mode",
        ),
        pytest.param([{**NUMBER, "failure_mode": "correctness"}], id="a mode without a check"),
        pytest.param(
            [{**NUMBER, "failure_mode": "correctness"}, {**RUBRIC, "failure_mode": "reasoning"}],
            id="an undeclared mode",
        ),
    ],
)
def test_checks_that_do_not_match_the_modes_one_to_one_are_refused(validators):
    with pytest.raises(ValidationError, match="exactly one check"):
        _task(validators)


def test_a_task_without_checks_is_left_alone():
    assert _task([]).validators == []
