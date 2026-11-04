# Tasks — v0

Version 0 of the official SciFlawBench task set.

This folder holds the v0 task file, `tasks.jsonl`. A task file is newline-delimited JSON (`.jsonl`) with one task object per line; see `../../example/README.md` for the field reference.

To run the v0 version of SciFlawBench, point `task_file` in `config.json` at it:

```json
{
  "task_file": "data/tasks/v0/tasks.jsonl"
}
```
