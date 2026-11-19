import json

import pytest

from evaluation.scoring import aggregate_scores, scoreboard_to_dict
from utils.rescore_votes import decide, main, read_records, rescore, resolve_judges

JUDGES = ["p/glm-flash", "q/gpt-mini", "r/deepseek-flash"]


def _panel_check(*verdicts, mode="reasoning"):
    """A judge:rubric check graded by the panel, with one vote per judge (None: the judge gave no verdict)."""
    votes = []
    for judge, verdict in zip(JUDGES, verdicts, strict=True):
        vote = {"judge_model": judge, "passed": verdict}
        vote.update({"justification": "ok"} if verdict is not None else {"error": "RuntimeError: down"})
        votes.append(vote)
    counted = [v for v in verdicts if v is not None]
    return {
        "passed": 2 * sum(counted) > len(counted),
        "details": "panel",
        "metadata": {"rubric": "R", "judge_model": "majority of ...", "votes": votes, "tie": False},
        "failure_mode": mode,
    }


def _record(task_id, *checks, modes=("reasoning",)):
    return {
        "id": task_id,
        "repetition": 1,
        "status": "success",
        "failure_modes": {
            "quantitative": [m for m in modes if m == "correctness"],
            "qualitative": [m for m in modes if m != "correctness"],
        },
        "checks": list(checks),
    }


def _run(tmp_path, records):
    (tmp_path / "aggregate_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return tmp_path


RECORDS = [
    _record("A", _panel_check(True, True, False)),  # 2 of 3 pass
    _record("B", _panel_check(False, True, False)),  # 1 of 3
    _record(
        "C",
        {"passed": True, "details": "numeric", "metadata": {}, "failure_mode": "correctness"},
        _panel_check(True, None, False),  # one judge gave no verdict: 1-1 tie
        modes=("correctness", "reasoning"),
    ),
]


def test_the_majority_of_all_judges_reproduces_the_recorded_scores():
    recorded = scoreboard_to_dict(aggregate_scores(RECORDS))
    rescored = scoreboard_to_dict(aggregate_scores(rescore(RECORDS, JUDGES, "majority")))

    assert rescored == recorded


@pytest.mark.parametrize(
    ("judges", "rule", "expected"),
    [
        (["p/glm-flash"], "majority", [True, False, True]),  # one judge alone
        (["q/gpt-mini"], "majority", [True, True, False]),  # its missing verdict fails the check
        (["p/glm-flash", "q/gpt-mini"], "majority", [True, False, True]),  # 1-1 ties fail; C counts glm only
        (JUDGES, "unanimous", [False, False, False]),
        (["p/glm-flash", "r/deepseek-flash"], "majority", [False, False, False]),
    ],
)
def test_other_judges_and_rules_decide_each_panel_check(judges, rule, expected):
    rescored = rescore(RECORDS, judges, rule)

    assert [record["checks"][-1]["passed"] for record in rescored] == expected
    assert rescored[2]["checks"][0]["passed"] is True  # a check without votes keeps its result


def test_a_check_no_chosen_judge_answered_fails():
    votes = [{"judge_model": "q/gpt-mini", "passed": None, "error": "down"}]
    assert decide(votes, ["q/gpt-mini"], "majority") is False
    assert decide(votes, ["q/gpt-mini"], "unanimous") is False


def test_judges_are_named_by_their_id_or_a_unique_part_of_it():
    assert resolve_judges(["glm", "q/gpt-mini"], JUDGES) == ["p/glm-flash", "q/gpt-mini"]
    with pytest.raises(ValueError, match="matches 2 judges"):
        resolve_judges(["flash"], JUDGES)
    with pytest.raises(ValueError, match="matches no judge"):
        resolve_judges(["claude"], JUDGES)


def test_the_command_prints_and_writes_the_rescored_scores(tmp_path, capsys):
    run = _run(tmp_path, RECORDS)
    out = tmp_path / "glm.json"

    assert main([str(run), "--judges", "glm", "--out", str(out)]) == 0

    printed = capsys.readouterr().out
    assert printed.startswith("Rescored from the stored votes: majority of p/glm-flash")
    written = json.loads(out.read_text())
    assert written["rescored"] == {"judges": ["p/glm-flash"], "rule": "majority"}
    assert written == {
        "rescored": written["rescored"],
        **scoreboard_to_dict(aggregate_scores(rescore(RECORDS, ["p/glm-flash"], "majority"))),
    }
    assert not (run / "scores.json").exists()  # the run's own scores are never written


def test_the_command_lists_the_judges_and_refuses_runs_without_votes(tmp_path, capsys):
    panel = tmp_path / "panel"
    panel.mkdir()
    assert main([str(_run(panel, RECORDS)), "--list"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{judge}  (3 checks)" for judge in sorted(JUDGES)]

    single = tmp_path / "single"
    single.mkdir()
    _run(
        single,
        [_record("A", {"passed": True, "details": "judge", "metadata": {"rubric": "R"}, "failure_mode": "reasoning"})],
    )
    assert main([str(single)]) == 1
    assert "not judged by a panel" in capsys.readouterr().err


def test_a_cut_off_last_line_is_skipped(tmp_path):
    run = _run(tmp_path, RECORDS)
    with open(run / "aggregate_results.jsonl", "a") as f:
        f.write('{"id": "D", "checks": [')

    assert [record["id"] for record in read_records(run)] == ["A", "B", "C"]
