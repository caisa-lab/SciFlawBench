"""
Recompute SciFlawBench task ids (and rename their assets folders) after prompts change.

A task's `task_id` is the md5 of its stored prompt, and tasks that use local files
keep them in `data/tasks/assets/<task_id>/`. Editing a prompt therefore invalidates
both the id and the folder name; this tool fixes both, in place.

    python src/make_task_ids.py data/example/tasks.jsonl
    python src/make_task_ids.py data/tasks/v0/tasks.jsonl --check
    python src/make_task_ids.py data/tasks/v0/tasks.jsonl --skip-assets
"""

import argparse
import sys
from pathlib import Path

from core.task_ids import default_assets_dir, recompute_task_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task_files", nargs="+", type=Path, help="one or more .jsonl task files to fix")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report stale ids without writing anything; exit 1 if any are stale",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="assets folder holding per-task folders (default: <repo>/data/tasks/assets)",
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="recompute ids only; never touch assets folders",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = 0

    for task_file in args.task_files:
        if not task_file.is_file():
            print(f"error: {task_file}: no such file", file=sys.stderr)
            status = 1
            continue

        assets_dir = None
        if not args.skip_assets:
            assets_dir = args.assets_dir or default_assets_dir(task_file.resolve().parent)

        report = recompute_task_ids(task_file, assets_dir, apply=not args.check)

        for error in report.errors:
            print(f"error: {task_file}: {error}", file=sys.stderr)

        verb = "would change" if args.check else "changed"
        print(f"{task_file}: {report.total} task(s), {len(report.changes)} id(s) {verb}")

        for change in report.changes:
            print(f"  line {change.line}: {change.old_id} -> {change.new_id}")
            if change.renamed:
                print(f"      renamed {change.renamed[0]} -> {change.renamed[1]}")

        if report.applied:
            print(f"  wrote {task_file}")

        if report.errors or (args.check and report.changed):
            status = 1

    return status


if __name__ == "__main__":
    raise SystemExit(main())
