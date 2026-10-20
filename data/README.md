# Data

This folder contains the task data used by the SciFlawBench harness, along with any supporting local files that tasks depend on.

Pass a task file to the harness through the `task_file` field of `config.json` (see the top-level `README.md`). Task files are newline-delimited JSON (`.jsonl`), with one task object per line.

## Layout

```
data/
├── example/     # Self-contained example task file for demos, smoke tests and quick local runs
└── tasks/       # The official SciFlawBench sample tasks, versioned over time
    ├── v0/      # Version 0 of the official task set
    └── assets/  # One folder per task id, holding that task's local files
```

### `example/`

A small, illustrative task file (`tasks.jsonl`) that mirrors the official task format. Use it to try the harness end to end without pulling a full benchmark version.

### `tasks/`

Holds the samples that make up the **official SciFlawBench**, split into one subfolder per benchmark version (`v0`, `v1`, ...).

### `tasks/assets/`

Local, non-code files that tasks refer to (datasets, JSON, CSV, and similar structured data). Each task that needs a file gets its own subfolder **named after its `task_id`**, and can address it in the prompt as `data/tasks/assets/<task_id>/<file>`. See [`tasks/assets/README.md`](tasks/assets/README.md) for the naming/hashing convention.
