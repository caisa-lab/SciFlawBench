# Assets

Local files that SciFlawBench tasks depend on live here: JSON files, CSV files, and anything else a task needs to read at runtime.

Every task that needs a local file gets its **own folder inside `assets/`, named after the task's `task_id`**:

```
assets/
└── SFB-9010/                             # the task's task_id
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
... `data/tasks/assets/SFB-9010/measurements.csv` ...
```

## Task ids and `prompt_md5`

Tasks in the official set have a **fixed id** of the form `SFB-NNNN` (for example `SFB-9010`). The id is a name: it is assigned once and does not change when the prompt is edited, so the assets folder `assets/SFB-9010/` stays the same too.

The hash of the prompt is stored separately in `prompt_md5`. It is the **32-character MD5 hash of the stored prompt text**, including the `{TASK_ID}` placeholder:

```python
import hashlib

prompt = "A CSV file is provided at `data/tasks/assets/{TASK_ID}/measurements.csv`. ..."
prompt_md5 = hashlib.md5(prompt.encode("utf-8")).hexdigest()   # 32 hex chars -> the prompt_md5 field
```

A task without a fixed id uses this hash as its `task_id` itself (`task_id = md5(task)`), and its folder is then `assets/<md5>/`. Tasks that need no local files simply hash their plain prompt (there is no placeholder). Either way the prompt is stored verbatim, so the hash can always be recomputed from the file.

## Integrity check

Because the prompt is stored unmodified, `md5(task)` acts as a checksum for the sample. When a task file is loaded the harness recomputes it and **refuses to run if it does not match the recorded `prompt_md5`** (or `task_id` for a task without a fixed id), so editing a prompt without recomputing its hash is caught immediately.

## Fixing a stale hash

After editing a prompt, recompute its hash in one step:

```bash
python src/make_task_ids.py data/tasks/v0/tasks.jsonl
```

For a task with a fixed id it only updates `prompt_md5`; the id and the folder stay. For a task without a fixed id it rewrites the task file in place and renames `assets/<old id>/` to the new id (use `--skip-assets` to leave folders alone, or `--assets-dir DIR` if your assets live elsewhere).

Use `--check` to verify without writing anything:

```bash
python src/make_task_ids.py data/tasks/v0/tasks.jsonl --check
```

## Notes

- Keep big/binary assets out of Git when you can. Instead, document the source and how to obtain the file here.
- Give each asset a clear name, and note here which task ids use it.
- Never rename an assets folder by hand: its name must match the task's `task_id`, which is either its fixed id (for example `SFB-9010`) or, for a task without one, the MD5 hash of the prompt. If you edit the prompt of a task without a fixed id, recompute the id and rename the folder to match.
- If an asset is shared by several tasks, note that here rather than duplicating it into every folder.
