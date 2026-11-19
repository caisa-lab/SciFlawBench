"""
Score a run judged by a panel again from its stored votes, with no new judge calls.

A run judged by a panel of `judges` stores every judge's vote with each `judge:rubric` check (`metadata.votes`, see
`evaluation.judge.JudgePanel`), and its scores use their majority. This recomputes the scores with other judges or
another rule, for example one judge alone or a stricter rule, and prints them in the format of `run_summary.log`:

    python -m utils.rescore_votes logs/<run> --list                    # the judges in the run's votes
    python -m utils.rescore_votes logs/<run>                           # majority of all judges, as in scores.json
    python -m utils.rescore_votes logs/<run> --judges glm              # one judge alone
    python -m utils.rescore_votes logs/<run> --judges glm gpt-5-mini   # two judges: a tie fails, so both must pass
    python -m utils.rescore_votes logs/<run> --rule unanimous          # every judge must pass
    python -m utils.rescore_votes logs/<run> --judges glm --out scores_glm.json

A judge is named by its model id or any unique part of it. A judge that gave no verdict on a check (its call failed
twice) is left out of that check's vote; if no chosen judge answered, the check fails, as for a single judge whose
call fails. Checks without votes (graded by a single judge, or not by a judge) keep their recorded result, and the
run's own `scores.json` is never changed. Scores stay pass or fail per check, so the mean of the votes is not offered.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evaluation.scoring import aggregate_scores, format_scoreboard, record_check_results, scoreboard_to_dict

RULES = ("majority", "unanimous")


def read_records(run_dir: Path) -> list[dict[str, Any]]:
    """The run's records from `aggregate_results.jsonl`, the file its scores are built from."""
    path = run_dir / "aggregate_results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: give the run's log directory (logs/<timestamp>)")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut off by an interrupted run, skipped as by the runtime manager
        if isinstance(record, dict):
            records.append(record)
    return records


def _votes(check: Any) -> list[dict[str, Any]] | None:
    metadata = check.get("metadata") if isinstance(check, Mapping) else None
    votes = (metadata or {}).get("votes")
    return votes if isinstance(votes, list) else None


def judges_in(records: Iterable[Mapping[str, Any]]) -> Counter:
    """How many checks each judge voted on."""
    counts: Counter = Counter()
    for record in records:
        for check in record_check_results(record):
            for vote in _votes(check) or []:
                counts[vote["judge_model"]] += 1
    return counts


def resolve_judges(names: Sequence[str], available: Iterable[str]) -> list[str]:
    """The judges' full model ids, from exact ids or unique parts of them."""
    available = sorted(set(available))
    resolved = []
    for name in names:
        matches = [model for model in available if model == name] or [model for model in available if name in model]
        if len(matches) != 1:
            found = "no judge" if not matches else f"{len(matches)} judges ({', '.join(matches)})"
            raise ValueError(f"{name!r} matches {found}; the run's judges are: {', '.join(available)}")
        resolved.append(matches[0])
    return resolved


def decide(votes: Sequence[Mapping[str, Any]], judges: Sequence[str], rule: str) -> bool:
    """Whether the chosen judges pass a check under `rule`; judges that gave no verdict are left out."""
    counted = [vote["passed"] for vote in votes if vote["judge_model"] in judges and vote["passed"] is not None]
    if not counted:
        return False
    if rule == "unanimous":
        return all(counted)
    return 2 * sum(counted) > len(counted)  # a tie fails, as in JudgePanel


def rescore(records: Iterable[Mapping[str, Any]], judges: Sequence[str], rule: str) -> list[dict[str, Any]]:
    """Copies of the records whose panel checks pass or fail by the chosen judges and rule."""
    out = []
    for record in records:
        checks = []
        for check in record_check_results(record):
            votes = _votes(check)
            checks.append(check if votes is None else {**check, "passed": decide(votes, judges, rule)})
        key = next((k for k in ("checks", "check_results") if k in record), None)
        out.append(record if key is None else {**record, key: checks})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="the run's log directory, which holds aggregate_results.jsonl")
    parser.add_argument("--judges", nargs="+", help="judges to count (model ids or unique parts); default: all")
    parser.add_argument("--rule", choices=RULES, default="majority", help="majority (default) or unanimous")
    parser.add_argument("--out", type=Path, help="also write the scores, as in scores.json, to this file")
    parser.add_argument("--list", action="store_true", help="list the judges in the run's votes and exit")
    args = parser.parse_args(argv)

    records = read_records(args.run_dir)
    counts = judges_in(records)
    if not counts:
        print(f"{args.run_dir} has no judge votes: it was not judged by a panel", file=sys.stderr)
        return 1
    if args.list:
        for model, n in sorted(counts.items()):
            print(f"{model}  ({n} checks)")
        return 0
    try:
        judges = resolve_judges(args.judges, counts) if args.judges else sorted(counts)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    board = aggregate_scores(rescore(records, judges, args.rule))
    print(f"Rescored from the stored votes: {args.rule} of {', '.join(judges)}\n")
    print(format_scoreboard(board))
    if args.out:
        result = {"rescored": {"judges": judges, "rule": args.rule}, **scoreboard_to_dict(board)}
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
