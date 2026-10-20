# Example tasks

This folder holds a small, self-contained example task file, `tasks.jsonl`, used for demonstrations, smoke tests, and quick local runs of the harness.

## Task file format

Each line of `tasks.jsonl` is a single JSON object describing one task. The four fields required for the harness to run a task are:

| Field           | Type                           | Description                                                                                                        |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| `task_id`       | `str`                          | 32-character MD5 hash of the `task` text.                                                                          |
| `task`          | `str`                          | The prompt handed to the agent.                                                                                    |
| `agent_id`      | `"code_agent" \| "tool_agent"` | Which registered agent should attempt the task.                                                                    |
| `failure_modes` | `object`                       | The failure modes the task is designed to probe, split into `quantitative` and `qualitative` lists. See [Failure modes](#failure-modes). |

Optional fields let a task do more than the minimum:

- `extra_tools` — additional tool definitions made available for the task, given either as `{"tool_name": ..., "kwargs": {...}}` objects or as bare tool-name strings. Only tools the agent does not already have: smolagents rejects duplicate tool names, so declaring a default tool here fails the run. See the tool list in [`../../examples/README.md`](../../examples/README.md#tool-overrides).
- `validators` — the correctness criteria applied to the agent's output. **This is where a task's expected answer lives**: there is no separate reference-answer field, so every task should carry the expected values in its validator kwargs.
- `closed_book` — run just this task with no harness tools.
- `repetition` — set by the harness when the run is dispatched; **do not author it by hand**.

### Failure modes

`failure_modes` records what a task is designed to catch. It is what lets the harness report a score per failure mode, and it uses the same two families as
[`../../docs/failure-modes.md`](../../docs/failure-modes.md):

```json
{ "failure_modes": { "quantitative": ["correctness"], "qualitative": [] } }
```

- Both keys are always present; an empty list means the task probes no mode of that family.
- Every entry must be one of the canonical names for its family — the quantitative and qualitative modes listed in [`../../docs/failure-modes.md`](../../docs/failure-modes.md).
- A task must declare **at least one** mode.

The declared modes are copied into every result record. At the end of a run the harness averages its task scores (see [Validators](#validators)) into `scores.json` and appends a score block to
`run_summary.log`: the overall score plus one score per failure mode, each measured only over the tasks that declare it.

### Validators

Each entry of `validators` is a `{"name": ..., "kwargs": {...}}` object naming a registered verifier:

```json
{"validators": [{"name": "content:numeric_match", "kwargs": {"expected": "18.4", "tol": 0.05}}]}
```

Every verifier receives the agent's final answer as its first argument and returns a pass/fail plus a human-readable detail string; a verifier that raises counts as failed. The results are stored as `check_results` in the task's result record and summarised as `Checks: passed/total` in `run_summary.log`. A task's **score** is the fraction of its checks that pass (`1.0` when every check passes); those task scores are what the harness averages into the overall and per-failure-mode scores written to `scores.json` (see [Failure modes](#failure-modes)).

| Verifier | `kwargs` | What it checks |
| -------- | -------- | -------------- |
| `format:json_output` | – | The answer parses as JSON |
| `content:json_output` | `expected` (object) | The answer parses as JSON and every field of `expected` is present with exactly that value |
| `content:contains_str` | `expected` (string), `case_sensitive` (`false`) | `expected` occurs somewhere in the answer |
| `content:exact_str_match` | `expected` (string), `case_sensitive` (`false`) | The answer equals `expected` once whitespace is stripped |
| `content:numeric_match` | `expected` (string), `tol` (`0.0`), `index` (`-1`) | The `index`-th number found in the answer is within `tol` of `expected` |
| `content:numeric_within_range` | `minimum`, `maximum`, `index` (`-1`) | The `index`-th number found in the answer lies inside `[minimum, maximum]` |
| `content:scientific_notation` | `expected` (string), `rel_tol` (`0.01`), `sig_figs` (`null`), `index` (`-1`) | The `index`-th scientific-notation value in the answer matches `expected`, either within `rel_tol` or exactly at `sig_figs` significant figures |
| `content:paper_json_match` | `expected` (list of `{"arxiv_id", "title"}`), `match_mode` (`"all"`) | The JSON answer's `papers` list contains all (or, with `match_mode: "any"`, at least one) of the expected arXiv ids |
| `judge:rubric` | `rubric` (string) | An LLM judge is shown the rubric and the run's **trace** and decides whether the trace satisfies the rubric. Requires a `judge` section in the run config; the verdict and the judge's justification are recorded in the check |

> `index: -1` means "the last number in the answer"; use `0`, `1`, ... to pick a specific one. A task passes its correctness check when **all** of its validators pass — note that the per-run `status`/`success` flag only records whether the agent finished without raising, so read `check_results` (or the `Checks: n/m` summary) to judge correctness.
>
> `judge:rubric` is the one content-free verifier: instead of matching strings or numbers it asks a model to grade the whole trace against a rubric, so it can check things a regex cannot ("did the agent actually read the abstract it cited?"). It needs a `judge` section in the run config, is sent the rubric plus the run's trace, and returns the judge's score and justification. See [Judge settings](../../examples/README.md#judge-settings). `tasks.jsonl` contains a worked example (task `07032b5c991008f5e7dad4e25734de90`) that checks a numeric answer with `content:numeric_match` and the trace's reasoning with `judge:rubric`.

## Task ids and local files

`task_id` is the **32-character MD5 hash of the `task` text**. Prompts that need a local file reference it through a `{TASK_ID}` placeholder, which stays in the stored prompt; the harness substitutes the real id when the task runs. The file itself lives in `../tasks/assets/<task_id>/`. For example:

```json
{
  "task_id": "40897e4550180c736a05c904f440566c",
  "task": "... `data/tasks/assets/{TASK_ID}/measurements.csv` ...",
  "agent_id": "code_agent",
  "extra_tools": [{ "tool_name": "read_file" }]
}
```

> IMPORTANT: A task must declare any tool it needs that the agent does not already have. The default `code_agent` / `tool_agent` ship with `web_search`, `wikipedia_search`, `visit_webpage`, `calculator`, `current_time` and `arxiv_search`. On the other hand, `read_file` and `json_answer_tool` are opt-in and have to be listed in `extra_tools`. Listing a tool the agent already has is an error: smolagents rejects duplicate tool names, so only genuinely missing tools belong in `extra_tools`. See the [tool list](../../examples/README.md#tool-overrides) for every registered tool.

Prompts that use no local files simply hash their plain text. Because the stored prompt is what gets hashed, the harness verifies `md5(task) == task_id` at load time and refuses any sample whose id no longer matches its prompt. See [`../tasks/assets/README.md`](../tasks/assets/README.md) for the full convention.

## Usage

Point the harness at this file from your run configuration (start from [`../../examples/config.json`](../../examples/config.json)):

```json
{
  "task_file": "data/example/tasks.jsonl"
}
```

then run:

```bash
python src/main.py --config examples/config.json
```
