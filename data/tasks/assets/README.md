# Assets

Local files that SciFlawBench tasks depend on live here: JSON files, CSV files, and anything else a task needs to read at runtime.

Every task that needs a local file gets its **own folder inside `assets/`, named after the task's `task_id`**:

```
assets/
└── 40897e4550180c736a05c904f440566c/     # the task's task_id
    └── measurements.csv
```

The folder name is the id, so it is unique and stable, and a task can address its files through that id.

## How a task points at its files

Prompts are written with a `{TASK_ID}` placeholder wherever the id should appear (the folder name *is* the id), and are stored **with the placeholder still in place**:

```
A CSV file with instrument readings is provided at
`data/tasks/assets/{TASK_ID}/measurements.csv`. ...
```

When the task runs, the harness substitutes the task's real `task_id` for the placeholder, so the agent sees a concrete path:

```
... `data/tasks/assets/40897e4550180c736a05c904f440566c/measurements.csv` ...
```

## How `task_id` is derived

`task_id` is the **32-character MD5 hash of the stored prompt text** — including the `{TASK_ID}` placeholder:

```python
import hashlib

prompt = "A CSV file is provided at `data/tasks/assets/{TASK_ID}/measurements.csv`. ..."
task_id = hashlib.md5(prompt.encode("utf-8")).hexdigest()   # 32 hex chars -> the task_id field
```

Tasks that need no local files simply hash their plain prompt (there is no placeholder). Either way the prompt is stored verbatim, so the id can always be recomputed from the file.

## Integrity check

Because the prompt is stored unmodified, `md5(task)` acts as a checksum for the sample. When a task file is loaded the harness recomputes it and **refuses to run if it does not match the recorded `task_id`**, so editing a prompt without recomputing its id is caught immediately.

## Fixing a stale id

After editing a prompt, recompute its id and rename its assets folder in one step:

```bash
python src/make_task_ids.py data/example/tasks.jsonl
```

It rewrites the task file in place and renames `assets/<old id>/` to the new id (use `--skip-assets` to leave folders alone, or `--assets-dir DIR` if your assets live elsewhere).

Use `--check` to verify without writing anything:

```bash
python src/make_task_ids.py data/tasks/v0/tasks.jsonl --check
```

## Notes

- Keep big/binary assets out of Git when you can. Instead, document the source and how to obtain the file here.
- Give each asset a clear name, and note here which task ids use it.
- Never rename an assets folder by hand: its name must match the task's `task_id`, which is derived
  from the prompt. If you edit a prompt, recompute the id and rename the folder to match.
- If an asset is shared by several tasks, note that here rather than duplicating it into every folder.
