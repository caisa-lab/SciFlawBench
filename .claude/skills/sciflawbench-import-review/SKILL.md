---
name: sciflawbench-import-review
description: For SciFlawBench maintainers in Claude Code. Imports new Google Form submissions with utils/import_form.py, reviews each new task, converts its validators to the harness format, checks them, and prepares the pull request that adds the tasks to data/tasks/v0/tasks.jsonl on main. Use when a maintainer asks to import, review or convert form submissions, or to open the new-tasks PR.
---

# Import, review and convert form submissions

Authors submit tasks through the Google Form. `utils/import_form.py` turns each valid submission into one commit on the shared `new-tasks` branch:
- the task line is appended to `data/tasks/v0/tasks.jsonl`;
- the task's files go to `data/tasks/assets/<id>/`;
- each failure mode gets a `judge:rubric` validator in the author's own words.

The importer already checks each row, scans for personal details, and commits under a fixed identity. Your job is what comes after it:
- reviewing each task;
- converting validators where an automatic one fits;
- checking that the validators work;
- preparing the pull request to main.

Never build task lines yourself from a pasted submission: that would skip the importer's safeguards. For setup and the importer's options, read [`docs/form-import.md`](../../../docs/form-import.md). Don't copy its commands from memory.

## Rules that always apply

**Personal details never reach the repository, GitHub or your messages.** The form collects each author's name, email, institution, ORCID and OpenReview handle.
- **Never open** the response sheet, a CSV export of it, the private Authors tab, or the files in `~/.config/sciflawbench/`. Pass the sheet id or the CSV path to the importer and let it read them.
- **Never list or open the folder of uploads** used with `--from-csv`. Google's file names carry the uploader's account name. Read a task's files only in `data/tasks/assets/<id>/` once the import has written them.
- **Name submissions by task id only,** in messages, commits and the PR.
- **When you find a personal detail** in a prompt, a file or a source note, give its place (file and line), never the detail itself. Fix it before anything is pushed.
- **Stage files by explicit path.** Never use `git add -A` or `git add .`.

**Ask the maintainer before you:**
- import for real, after showing the dry run;
- run any agent or judge, which costs money;
- push;
- open the PR;
- delete a branch.

Never merge.

**Use the caisa-lab remote.** Find it with `git remote -v`: it is the remote whose URL holds `caisa-lab/SciFlawBench`. It may not be called `origin`. Fetch from it before importing and again before pushing, and pass it to the importer as `--base <remote>/main`.

**`new-tasks` is shared.** The next maintainer's import continues from it. Never rebase or force-push it. To bring it up to date, merge main into it.

**Write multi-line text with quoted heredocs (`<<'EOF'`).** Prompts hold `$`, backticks and backslashes, and an unquoted heredoc would run or change them. Edit task lines with a script, never by retyping JSON.

**Keep the author's science.** Never change an answer, a tolerance or a trap on your own. Propose the change, explain it, and record every change in the PR.

## 1. Check the setup

- **The working tree is clean.** The importer refuses to switch branches over uncommitted changes. If the maintainer has work in progress, ask them to commit it or move it elsewhere.
- **The Google files exist.**
  - Run `test -f ~/.config/sciflawbench/google_client.json` and `test -f ~/.config/sciflawbench/google_token.json`, without reading either file.
  - Without a client file, point the maintainer to the setup steps in `docs/form-import.md`.
  - Without a token, or with an expired one (it lasts about 7 days), the first run opens a browser to sign in and would hang your command. Ask the maintainer to run that first import in their own terminal.
- **You have the spreadsheet id** from the maintainer. It is in the sheet's URL.
- **Or a CSV export.** With `--from-csv <export> --files-dir <uploads>` instead of `--sheet`, skip the Google checks. You still pass both paths to the importer without opening either.

Run every importer command with a timeout, using the Bash tool's own timeout (macOS has no `timeout` command). If it waits for a sign-in, stop and hand it to the maintainer.

If uv tries to change `uv.lock` when you run a command, add `--frozen` to it.

## 2. Dry run

```bash
git fetch <remote>
uv run --with google-api-python-client --with google-auth-oauthlib python -m utils.import_form --sheet <id> --base <remote>/main --dry-run
```

From a CSV export, which needs no Google packages:

```bash
uv run python -m utils.import_form --from-csv <export> --files-dir <uploads> --base <remote>/main --dry-run
```

Report to the maintainer:
- the ids that would be imported;
- the ones refused, each with the importer's reasons;
- its warnings.

A refused submission is fixed in its sheet row and imported on a later run, or rejected (step 6). Then ask whether to import.

## 3. Import

Before importing, note the commit `new-tasks` will continue from:
- `<remote>/new-tasks` if it has been pushed;
- `<remote>/main` if it hasn't.

Run the same command without `--dry-run`. It switches to `new-tasks` and adds one commit per task; it never pushes.

**Read every new prompt and file for personal details now, before any push** (check 1 of step 4).
- The import commits hold whatever the files hold, and a fix in a later commit would still show in the branch's history once pushed.
- If you find one, don't push. Ask the maintainer to fix it at the source:
  - replace the uploaded file in Google Drive with a cleaned copy (Manage versions keeps its link);
  - or correct the sheet row;
  - or reject the task.
- Then, with the maintainer's yes, reset `new-tasks` to the commit you noted (`git reset --hard <commit>`) and import again. This drops only the unpushed commits.
- Until then you may go on reviewing. Write each task's edit as a script in a scratch folder outside the repository, since the reset drops review commits too, and the scripts let you replay them.

`docs/form-import.md` says to push `new-tasks` straight away, so the next maintainer continues from it. Once the new tasks are clean, ask the maintainer whether to push now.

The importer's warnings tell you what to read first:
- a file it renamed;
- the path line it added to a prompt (`The file is at data/tasks/assets/{TASK_ID}/<name>.`);
- a file the prompt doesn't name;
- part of a name found in a task. "file 1" means the task's first upload. With several files, read them all.

## 4. Review each task

For each new task, read its line and every file in `data/tasks/assets/<id>/` in full. The importer's scan finds only the exact details the author typed, not a nickname, initials or a signature.

Then check, and note each finding as blocking or advisory:

1. **Personal details,** as above: signatures, "by …" lines and names in file headers. Authors of a cited paper are fine. Do this check before any push (step 3).
2. **The checks for authors.** Apply checks 3 to 8 of [`skills/sciflawbench-task-check/SKILL.md`](../../../skills/sciflawbench-task-check/SKILL.md):
   - each validation section grades on its own;
   - each failure mode fits the task;
   - the prompt: self-contained, one exact answer form, needs a tool, nothing gives the trap away, no shortcut;
   - the files;
   - the sources;
   - personal details.
3. **The answer.** Open the sources and recompute the answer where you can, and say what you checked. A wrong answer is blocking.
4. **Overlap.** Look for the same object, value or paper in an existing task in `data/tasks/v0/tasks.jsonl`.
5. **Leaks.** The prompt has no link or hint that leads to the benchmark's own pages ("sciflawbench").

## 5. Convert the validators

Each failure mode keeps exactly one validator, with its `failure_mode` unchanged and in the same order (quantitative first). Never add `code_safety`.

**When to use an automatic validator.** Replace a rubric with one only when the prompt states the exact form the validator reads. The kwargs below are those in `src/evaluation/definitions.py`; check there if in doubt.

| The answer is | Validator and kwargs |
| --- | --- |
| One number | `content:numeric_match`: `expected` as a string, `tol` (absolute), `index` |
| One number in a range the author states ("accept 6.50 to 6.54") | `content:numeric_within_range`: `minimum`, `maximum`, `index` |
| A very large or small number written as `4.2e8` or `4.2 × 10^8` | `content:scientific_notation`: `expected` as a string, `rel_tol` |
| Several values | `content:json_output`: `expected` holds only the graded fields, `tol` for its numbers (default `0.0`). The answer may sit in one code block, and extra fields are ignored |
| A list of arXiv papers | `content:paper_json_match`: `expected` as a list of `{"arxiv_id", "title"}`, `match_mode` |
| A name or string | `content:exact_str_match` or `content:contains_str`. `contains_str` also passes an answer that lists every option, so avoid it when the options are few |
| Anything else | `judge:rubric` |

**How `numeric_match` reads numbers.** It takes runs of plain digits with an optional `-` and decimal point, then picks one by `index`: `0` is the first, `-1` (the default) the last.
- `421,603,913` is read as 421, 603 and 913.
- `4.216e8` is read as 4.216 and 8.
- Numbers in names and dates count: "DES DR2", "2021".
- A Unicode minus (`−300`) is fine: the harness turns it into `-` before any validator sees it.
- `tol` is compared in floating point, so `expected` 6.52 with `tol` 0.02 refuses 6.54. Use `numeric_within_range` for a range the author gives, or add a small margin to `tol`.

So the prompt must ask for the number first, as a plain number, for example "Final answer: the ratio as a plain number, then one sentence". Then use `index: 0`.

**Rubrics that stay.** Rewrite the author's words so the judge can grade them alone:

> Does the trace show that …? Answer true only if …; answer false if ….

- Keep every value, fact and near miss the author gave. State the value.
- Grade one property per rubric.
- For `correct_tool_calls`, name the tool, what it must read, and that the value must come from the tool's output.

**Tools.** Add the opt-in tools an author asked for in a source note (`numerics`, `significance`, `cosmology`, `sky`, `json_answer_tool`) to `extra_tools`. Never list a default tool. `read_file` is already there when files were uploaded.

**Prompt edits.** Avoid them, and prefer changing only the validators.
- A later import compares each task's prompt with the sheet's. An edited prompt is reported there as a conflict on every run, and the import exits with 1, until the same edit is made in the sheet row.
- If an edit is needed, for example to state the answer format, ask the maintainer. Then make the edit in the task line, and have the maintainer make the same edit in the row's Task Description cell.
- Run `uv run python src/make_task_ids.py data/tasks/v0/tasks.jsonl` to update `prompt_md5`; the fixed id stays. Changing a validator leaves `prompt_md5` as it is.

**Edit a line with a script,** keeping its key order and writing UTF-8:

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

path = Path("data/tasks/v0/tasks.jsonl")
lines = path.read_text(encoding="utf-8").splitlines()
for number, line in enumerate(lines):
    task = json.loads(line)
    if task["task_id"] == "SFB-0052":
        task["validators"][0] = {
            "name": "content:numeric_within_range",
            "kwargs": {"minimum": 6.5, "maximum": 6.54, "index": 0},
            "failure_mode": "correctness",
        }
        lines[number] = json.dumps(task, ensure_ascii=False)
path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
EOF
```

**Check each automatic validator by hand** on three answers, at no cost:
- a right answer in the prompt's stated form;
- the trap answer;
- a right answer written another way, such as with units, a thousands separator or a comment that holds numbers.

It must pass the first and fail the second. You must also know what it does with the third, which here fails because 24 comes first:

```bash
uv run python - <<'EOF'
from evaluation.base import run_check
from evaluation.definitions import verifier_registry

check = verifier_registry.get("content:numeric_within_range")
kwargs = {"minimum": 6.5, "maximum": 6.54, "index": 0}
for answer in ["6.52, the mean of the 24 readings", "7.9", "The mean of the 24 readings is 6.52"]:
    print(run_check(check, answer, **kwargs).passed, answer)
EOF
```

**Commit each task's review on its own,** staging only that task's files:
- message: `Review SFB-0052: <what changed>`;
- end it with the attribution line your setup uses.

Review commits carry the maintainer's own git identity, like any other commit they push. Only the import uses the fixed `SciFlawBench import` identity.

## 6. Validate, and reject what can't be fixed

Run these on the branch:

```bash
uv run python - <<'EOF'
from utils.build_task import check_task_file

_, duplicates, problems = check_task_file("data/tasks/v0/tasks.jsonl")
print("duplicate ids:", duplicates or "none")
print("\n".join(problems) or "no problems")
EOF
uv run python src/make_task_ids.py data/tasks/v0/tasks.jsonl --check
uv run pytest -q tests/unit/core/test_validator_kwargs.py tests/unit/utils/test_build_task.py tests/unit/utils/test_import_form.py
```

**To reject a task,** after the maintainer agrees:
- add its id to `data/tasks/rejected_ids.txt`, which holds ids only, with no reason and no name;
- remove its line and its `data/tasks/assets/<id>/` folder;
- commit.

The id is never reused. The maintainers tell the author by email; you write nothing to them.

## 7. Run the new tasks on a model

Ask the maintainer first:
- which model to run;
- how many repetitions (1 for a first look, 3 to see how much results vary);
- the cost.

The judges in `examples/config.json` need `OPENROUTER_API_KEY`.

Write the reviewed task lines and a run config to a scratch folder outside the repository. Commit the review first, since the export reads committed lines:

```bash
uv run python - <<'EOF'
import json
from pathlib import Path

from utils.import_form import new_task_lines

out = Path("/tmp/sfb-review")  # any folder outside the repository
out.mkdir(exist_ok=True)
lines = new_task_lines(Path.cwd(), "new-tasks", "<remote>/main")
(out / "new.jsonl").write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
config = json.loads(Path("examples/config.json").read_text(encoding="utf-8"))
config |= {"task_file": str(out / "new.jsonl"), "repetitions_per_task": 1}
(out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
print(len(lines), "tasks")
EOF
uv run python src/main.py --config /tmp/sfb-review/config.json
```

Read the run in `logs/<timestamp>/`. Per-task scores are in `scores.json`. Each validator's verdict, and the judges' votes and justifications, are under `check_results` in `results/<id>.jsonl`.

For each task, answer the questions in `docs/form-import.md`, "Reviewing the new tasks":
- Does the prompt state one exact answer format?
- Is each rubric clear to the judge?
- Did you read every failure? If the answer was right but a validator failed, the validator is wrong.
- If no run passed a validator, did a right answer pass when checked by hand?

Fix and re-check as in step 5. Never commit `logs/`.

## 8. Prepare the pull request

1. Fetch the remote. If main has moved, run `git merge --no-edit <remote>/main` into `new-tasks`, then run step 6 again.
2. Write the PR text to a file outside the repository and show it to the maintainer:
   - **Title:** `Tasks: add SFB-0052 to SFB-0058 from the form`. For two tasks, or ids with gaps, list them: `Tasks: add SFB-0052, SFB-0054 and SFB-0055 from the form`.
   - **For each task:** its id, its failure modes, its validators (automatic or rubric), what the review changed and why, and the model run's result per validator ("not run yet" if step 7 was skipped).
   - **Also list:** the rejected ids, and the checks you ran.
   - **Leave out:** authors' names or any other personal detail.
3. After the maintainer says yes:
   - `git push <remote> new-tasks`;
   - `gh pr create --base main --head new-tasks --title "…" --body-file <file>`, with `-R caisa-lab/SciFlawBench` if the remote isn't the default.

PRs are squash-merged. After the merge, ask the maintainer before deleting the pushed `new-tasks`; the next import then starts it again from main. The `task stats` workflow updates the README's numbers on main by itself.

## Report back

End with:
- what was imported, changed, rejected and run;
- what is waiting for the maintainer: a sign-in, a sheet edit, a rejection, the push, the PR.

Name every task by its id only.
