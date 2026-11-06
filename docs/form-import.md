# Importing tasks from the Google Form

For maintainers. Authors submit tasks through a Google Form; a maintainer imports them as commits on a branch, reviews them, and opens a PR. Passing the import only means a task is usable; whether it is a good task is for the reviewer to decide. Authors' personal details stay on the Google account and never reach GitHub.

```
Author fills in the form and uploads any files
        │
        ▼
utils/form_script.gs, inside Google, on each submission
  • gives the submission the next id, starting at SFB-0052
  • copies the author's details to the private Authors tab
  • emails the id to the author and tells the maintainers
        │
        ▼
utils/import_form.py, run by a maintainer
  • reads new submissions and downloads their files (read-only)
  • checks that each task is usable, and reports the rest without writing them
  • adds one commit per valid task to the new-tasks branch; it never pushes
        │
        ▼
The maintainer pushes the branch, reviews the tasks and opens a PR to main
```

## Setting up the form

Do this once, signed in to the Google account that will own the form. It has to be a normal Drive, because shared drives don't allow uploads. Turn on 2-step verification, and make sure more than one person can recover the account, since it holds authors' personal details.

1. At <https://script.google.com>, create a project, paste in [`utils/form_script.gs`](../utils/form_script.gs), run `setup` and allow what it asks for. Its log gives the links to the form and the spreadsheet.
2. Open the form. At the end of **Sources and files**, add a **File upload** question titled exactly `Files`, which Apps Script can't create. Leave "Allow only specific file types" off, and set 5 files and 10 MB per file. Uploads count against the account's Drive storage, so in **Settings → Responses** set the total to 10 GB at most.
3. Publish the form so that anyone with the link can respond. Uploading needs a Google account.
4. In the script's **Project Settings → Script properties**, set `MAINTAINERS` to the comma-separated addresses to notify.

The importer reads questions by their titles, so a renamed question stops it. To change a question, change `setup`, `import_form.py` and the form together.

Before the first real use, send a fake submission with answers such as `1/2`, `007`, `18.40` and `=2+2` and a prompt with apostrophes, backslashes, line breaks and `Λ`, and check that the import gives them back exactly as typed.

Then clean up before sharing the link, so the first real task gets `SFB-0052`:

1. Delete the fake responses in the form, their rows in the **Responses** and **Authors** tabs, and their uploads in Drive.
2. In **Script properties**, delete `NEXT_TASK_NUMBER`.
3. Remove any fake tasks from your local `new-tasks`.

## Giving the importer read-only access

The importer asks for exactly two scopes, and nothing broader: `spreadsheets.readonly` and `drive.readonly`. `drive.readonly` covers the whole Drive of the account that signs in, so sign in with the lab account that owns the form, not a personal one. Do this once per maintainer, in the [Google Cloud console](https://console.cloud.google.com):

1. Create a project and enable the Google Sheets API and the Google Drive API.
2. Configure the OAuth consent screen as external, and add the account you'll sign in with as a test user.
3. Create an OAuth client of type **Desktop app**, and save its file as `~/.config/sciflawbench/google_client.json`.

The first run opens a browser to sign in, and keeps the sign-in in `~/.config/sciflawbench/google_token.json`. While the consent screen is in testing, the sign-in lasts about 7 days.

Neither file belongs in the repo, and neither do the spreadsheet's contents.

## Importing

Run this from the repo root, with `<id>` taken from the spreadsheet's URL:

```bash
git fetch origin
uv run --with google-api-python-client --with google-auth-oauthlib python -m utils.import_form --sheet <id>
```

Then push the branch straight away, so the next person can continue from it. The import itself never pushes, and the PR to main still waits for the review:

```bash
git push origin new-tasks
```

The Google libraries are not project dependencies. If your remote for this repo isn't called `origin`, add `--base <remote>/main`, and fetch and push there instead. `--export new.jsonl` also writes just the new tasks to a file, so you can run them on a model without running everything.

To work without Google, use a CSV export of the sheet, with each submission's files in `<folder>/<task id>/`. Add `--dry-run` to only see the report; it switches no branch and writes nothing:

```bash
python -m utils.import_form --from-csv responses.csv --files-dir uploads --dry-run
```

Running the import again is safe. It skips ids already in the task file and ids in [`data/tasks/rejected_ids.txt`](../data/tasks/rejected_ids.txt). If an id is already in the task file with a different prompt, for example because the response was edited after the import, it reports a conflict and changes nothing.

A submission is reported and not written when:

| What it checks | Reported when |
|---|---|
| Validators | A box the validator needs is empty, or one it doesn't use is filled in; a correct answer doesn't fit the validator; a failure mode has two validators |
| The task | [`check_task`](../utils/build_task.py) or the harness's `TaskDef` loader refuses it, or its prompt is already in the task file |
| Files | The type isn't `.csv`, `.dat`, `.json`, `.md`, `.py` or `.txt`; a file isn't UTF-8 text or can't be read; the name isn't in the prompt, or could leave the task's folder; the prompt names a file nobody uploaded; or the files come to over 20 MB |
| Personal details | The name, email, ORCID or OpenReview handle of **anyone in the sheet**, or the uploader's Google account name, appears in the task line, a file name or a file's contents. Case, accents and spacing are ignored |
| What is staged | Just before committing, the staged changes are scanned for the same details, and for symbolic links, hidden files and anything outside `data/tasks/` |

A single word of a person's name, such as a surname on its own, doesn't stop a task, because it may be chance: the importer warns you to read that task. Reports never quote the detail they found. Commits use a fixed `SciFlawBench import` identity, so no author or maintainer details reach the git history.

**File names.** Google renames uploads to `data - Name Surname.csv`, using the name on the uploader's Google account, which can differ from the name typed in the form. The importer keeps what comes before the last ` - `, so that file becomes `data.csv` in `data/tasks/assets/<id>/`, and searches for the cut name along with the typed details.

Uploaded files are never run by the importer, and **a person still has to read them**: the leak check only finds the exact details the author gave, not a nickname or a shortened name.

## Several maintainers

Several people can run the import, one after another. Each needs the spreadsheet id, the client file and a sign-in. Ids can't clash, because the script in Google assigns them.

Fetch first: the import continues from the pushed `new-tasks`, so it only adds what is new. If your local branch and the pushed one have both changed, it stops and imports nothing, so you can decide which to keep. Someone who imports without pushing leaves the next person to import the same tasks again.

## Reviewing the new tasks

The importer can't tell whether a validator suits its prompt, so a person has to, before the PR. Use `--export new.jsonl`, run the new tasks on a model from the `new-tasks` branch, and go through each task:

1. Does the prompt state one exact answer format?
2. Did you read every failure? If the model's answer is right but a validator failed, the validator is wrong.
3. If no run passed a validator, did you confirm by hand that a right answer would pass? Otherwise nothing shows the validator can pass at all.

## After the review

- **If a validator should read another number:** the form's numeric validators read the last number in the answer. For another one, add `index` to the validator's kwargs in the task line, such as `0` for the first; this leaves `prompt_md5` as it is.
- **To fix a submission:** correct its row in the sheet, and run the import again if it wasn't written. Once it's on the branch, edit the line and update `prompt_md5` with `python src/make_task_ids.py`.
- **To reject a submission:** add its id to [`data/tasks/rejected_ids.txt`](../data/tasks/rejected_ids.txt), which holds ids only, so it isn't imported again. If it's on the branch, remove its line and its asset folder too. Its id isn't reused.
- **After the merge:** PRs are squash-merged, so `new-tasks` can't simply continue. Delete the pushed branch, or your next push will be refused. The next run restarts it from main, but only if the working tree is clean and every task line, rejected id and asset on it is already in main. Otherwise it stays where it is, so a run never throws away unmerged tasks.
