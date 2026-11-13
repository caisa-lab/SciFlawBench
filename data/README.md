# Data

This folder contains the task data used by the SciFlawBench harness, along with any supporting local files that tasks depend on.

Pass a task file to the harness through the `task_file` field of `config.json` (see the top-level `README.md`). Task files are newline-delimited JSON (`.jsonl`), with one task object per line.

## Layout

```
data/
├── LICENSE      # CC BY 4.0, for everything in this folder (see "License" below)
├── example/     # Self-contained example task file for demos, smoke tests and quick local runs
└── tasks/       # The official SciFlawBench sample tasks, versioned over time
    ├── v0/      # Version 0 of the official task set
    ├── assets/  # One folder per task id, holding that task's local files
    └── rejected_ids.txt  # Ids reviewed and not kept, so the form import skips them
```

### `example/`

A small, illustrative task file (`tasks.jsonl`) that mirrors the official task format. Use it to try the harness end to end without pulling a full benchmark version.

### `tasks/`

Holds the samples that make up the **official SciFlawBench**, split into one subfolder per benchmark version (`v0`, `v1`, ...).

### `tasks/rejected_ids.txt`

The ids of form submissions that were reviewed and not kept, one per line. It holds ids only. See [`../docs/form-import.md`](../docs/form-import.md).

### `tasks/assets/`

Local, non-code files that tasks refer to (datasets, JSON, CSV, and similar structured data). Each task that needs a file gets its own subfolder **named after its `task_id`** (for example `SFB-0042`), and can address it in the prompt as `data/tasks/assets/<task_id>/<file>`. See [`tasks/assets/README.md`](tasks/assets/README.md) for the naming/hashing convention.

## License

The tasks and their files in this folder are licensed under the [Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/); the full text is in [`LICENSE`](LICENSE). The harness code is licensed separately, under the MIT License (see [`../LICENSE`](../LICENSE)).

You may copy, adapt and redistribute the tasks, including commercially, as long as you:

* credit "SciFlawBench maintainers and contributors" and link to <https://github.com/caisa-lab/SciFlawBench>;
* link to the license; and
* say whether you changed anything.

Some task files are copies of third-party material, such as an official data product or a published abstract. Those files keep their original terms and are not covered by this license; each task's `sources` names where they come from.
