# Example Configuration

This folder holds a ready-to-edit run configuration for the harness, and this file documents every argument the configuration file and the runner accept.

## Running the harness

```bash
python src/main.py --config examples/config.json
```

| Flag | Required | Default | Description |
| ---- | -------- | ------- | ----------- |
| `--config PATH` | yes | – | Path to the JSON configuration file to run |
| `--dry` | no | off | Print the fully resolved configuration (defaults applied, secrets excluded) and exit without running anything |
| `--show_trace` | no | off | Force `generate_trace_reports` on for this run |
| `--closed_book` | no | off | Force `closed_book` on for this run |
| `--help` | no | – | Show every option and exit |

The two mode flags are applied to the parsed file *before* validation, so `--show_trace` / `--closed_book` are just conveniences for one-off runs; setting the same keys in the file has the identical effect.

## Configuration file

The file is a single JSON object. Only `model` and `task_file` are required. Unknown keys are **silently ignored** (the models are plain pydantic models), so a typo in a key name will not raise an error — it will simply do nothing.

### Run settings (top level)

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `model` | object | *required* | Model configuration, see [Model settings](#model-settings) |
| `judge` | object \| null | `null` | LLM-as-a-judge model used by the `judge:*` validators, see [Judge settings](#judge-settings) |
| `task_file` | string | *required* | Path to the `.jsonl` task file. Checked at load time: it must exist and end in `.jsonl`, or the config is rejected |
| `tool_configs` | array | `[]` | Per-run tool overrides, see [Tool overrides](#tool-overrides) |
| `log_path` | string | `"logs/"` | Where the run lives. A fresh run creates `<log_path>/<timestamp>/`; with `restarting` the harness uses `<log_path>` itself |
| `max_concurrent` | int | `3` | Maximum number of task subprocesses running at once |
| `repetitions_per_task` | int | `3` | How many times each task is run. Every repetition is an independent agent run and is recorded separately |
| `task_timeout_s` | int | `900` | Wall-clock budget per task before the runtime manager terminates that subprocess |
| `closed_book` | bool | `false` | Run the whole benchmark closed-book: the agent is given no harness tools at all (no search, no webpage visiting, no local file reading, no calculator) and may only use its own knowledge. Python execution and `final_answer` remain available |
| `logging_level` | int | `20` | Standard-library logging level: `10` DEBUG, `20` INFO, `30` WARNING, `40` ERROR |
| `restarting` | bool | `false` | Resume in place: treat `log_path` as an existing run directory instead of creating a new timestamped one. Set `log_path` to the interrupted run's directory. See [Reproducibility and resuming](../README.md#reproducibility-and-resuming) |
| `generate_trace_reports` | bool | `false` | Also write a human-readable Markdown report per task/repetition under `results/<task_id>_reports/` |

Both `closed_book` (here) and `restarting` interact with per-task and per-run state: a task may also declare `closed_book` itself. See [Per-task fields](#per-task-fields).

### Model settings

`model` is an object with the following keys:

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `provider` | `"litellm"` \| `"openai_server"` \| `"hf_api"` \| `"fake_model"` | *required* | Which client to build (see [Provider notes](#provider-notes)) |
| `model_id` | string | *required* | Provider-specific model identifier, e.g. `openrouter/qwen/qwen3.7-flash` (LiteLLM), `gpt-4o-mini` (`openai_server`), or a Hub model id (`hf_api`) |
| `api_key_env` | string | *required* | **Name** of the environment variable that holds the API key — never the key itself. The value is read from the environment (a `.env` file in the working directory is loaded first) and the config is rejected if that variable is unset |
| `api_base` | string \| null | `null` | Base URL of the endpoint. Needed for self-hosted or non-default OpenAI-compatible servers |
| `extra_kwargs` | object | `{}` | Extra keyword arguments forwarded verbatim to the smolagents model constructor: `temperature`, `max_tokens`, request timeouts, provider routing, and so on |
| `code_block_tags` | `[string, string]` \| null | `null` | Rarely needed: the opening/closing fence pair used to extract code blocks from a model's response, for models whose output does not use the default Markdown fences |

#### Provider notes

- **`litellm`** — routes to most hosted providers (OpenAI, Anthropic, OpenRouter, Google, ...) and to local servers, using `provider/model` style ids. The most portable choice.
- **`openai_server`** — any OpenAI-compatible `/v1` endpoint: OpenAI itself, vLLM, TGI, llama.cpp servers, and similar. Set `api_base` when the server is not the default.
- **`hf_api`** — the Hugging Face Inference API, authenticated with your HF token.
- **`fake_model`** — test-only scripted model used by the smoke tests; its behaviour comes from `extra_kwargs`. Pair it with `ENABLE_TEST_FAKES=1` (see [Environment variables](#environment-variables)).

### Judge settings

`judge` configures the model that grades run traces for the `judge:*` validators. It is optional and only needed by a run whose tasks use them. Judging always goes through [LiteLLM](https://docs.litellm.ai/), so there is no `provider` key.

| Key | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `model_id` | string | *required* | LiteLLM model string for the judge, e.g. `openrouter/qwen/qwen3.7-flash` or `gpt-4o-mini` |
| `api_key_env` | string | *required* | **Name** of the environment variable that holds the judge's API key; read at config load time exactly like the model's |
| `api_base` | string \| null | `null` | Base URL of the judge endpoint, for self-hosted or non-default OpenAI-compatible servers |
| `extra_kwargs` | object | `{}` | Extra keyword arguments forwarded to the LiteLLM completion call: `temperature`, `max_tokens`, `timeout`, provider routing, ... |

The judge is sent three things: the task's **rubric**, the run's **trace** (rendered exactly like the markdown trace reports, see `generate_trace_reports`), and a **hardcoded output specification** that demands a JSON object with `evaluation` (the score) and `justification`. Both are recorded per task in the run's `check_results`, so every judged run keeps the reason for its verdict. The trace is sent whole: the harness never truncates or compresses it, leaving oversized
requests to the API.

The entire judge configuration — model, endpoint and extra kwargs — plus a hash of the hardcoded judge prompt template is written into the run manifest and is part of the run signature, so a run
can only be resumed while it is still graded by the same judge. The judge's API key is never written anywhere.

### Tool overrides

`tool_configs` is a list of objects with a `tool_name` and the `kwargs` to build that tool with:

```json
{
  "tool_configs": [
    { "tool_name": "web_search", "kwargs": { "max_results": 5 } },
    { "tool_name": "wikipedia_search", "kwargs": { "operator": "you@example.com" } }
  ]
}
```

Each entry **replaces** the keyword arguments of that tool for the whole run (rather than merging with the defaults), for every agent that has the tool. `tool_name` must be one of the registered tools:

| Tool | Options (defaults) | Notes |
| ---- | ------------------ | ----- |
| `web_search` | `max_results` (`8`), `rate_limit` (`1.0`, requests per second), `engine` (`"duckduckgo"`) | SerpAPI is used instead of DuckDuckGo when `SERPAPI_KEY` is set **and** `engine` is not `"duckduckgo"` |
| `wikipedia_search` | `operator` (`"OPERATOR EMAIL NOT SET"`) | The operator email is placed in the request `User-Agent`, as the Wikimedia API policy asks |
| `arxiv_search` | `operator` (`"OPERATOR EMAIL NOT SET"`) | Same as above, for the arXiv API |
| `visit_webpage` | – | Fetches a URL and returns the page as Markdown |
| `read_file` | `max_rows` (`50`), `max_chars` (`20000`) | **Not enabled by default**; add it to a task through `extra_tools` |
| `calculator` | – | Evaluates a `sympy` expression in a resource-limited subprocess |
| `json_answer_tool` | `required_fields` (`[]`) | **Not enabled by default**; validates a JSON answer against required keys. Usually added per task |
| `current_time` | – | Returns the current local time |

The default agents (`code_agent`, `tool_agent`) start with `web_search`, `wikipedia_search`, `visit_webpage`, `calculator`, `current_time` and `arxiv_search`. Tools listed in a task's `extra_tools` are appended on top of that list.

### Environment variables

| Variable | Required | Description |
| -------- | -------- | ----------- |
| *the name in `api_key_env`* | yes | API key for the configured provider. Read from the environment when the config is validated; a `.env` file in the working directory is loaded automatically |
| *the name in `judge.api_key_env`* | only with a `judge` | API key for the judge model, read the same way |
| `SERPAPI_KEY` | no | When set, `web_search` may use SerpAPI instead of DuckDuckGo (better results, needs a SerpAPI account) |
| `NO_REPRODUCIBILITY_GUARANTEES` | no | Set to `true` to downgrade resume/manifest mismatches from a hard refusal to a warning. Results produced under it are no longer verifiable |
| `ENABLE_TEST_FAKES` | no | Set to `1` to register the test-only fake presets and tools used by the smoke tests |

Never commit `.env` files or keys: `.env` is already listed in `.gitignore`. See [`../SECURITY.md`](../SECURITY.md) for what the harness does and does not protect against.

## Per-task fields

The task file referenced by `task_file` has its own schema, documented in [`../data/example/README.md`](../data/example/README.md). The fields a task can set are:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `task_id` | string | The 32-character MD5 of the `task` text. Verified at load time; modified prompts are rejected |
| `task` | string | The prompt handed to the agent. Local files are referenced through the `{TASK_ID}` placeholder, which the harness resolves |
| `agent_id` | `"code_agent"` \| `"tool_agent"` | Which registered agent attempts the task |
| `extra_tools` | array | Additional tool definitions (or bare tool-name strings) made available for this task only. Only tools the agent does not already have — declaring a default tool produces a duplicate name, which smolagents rejects |
| `validators` | array | The correctness criteria applied to the agent's output — this is where a task's expected answer lives, e.g. `{"name": "content:numeric_match", "kwargs": {"expected": "18.4"}}`. See the [verifier reference](../data/example/README.md#validators) for every registered verifier and its kwargs |
| `failure_modes` | object | The failure modes the task probes, as `{"quantitative": [...], "qualitative": [...]}`. Required, and at least one mode must be listed. Drives per-failure-mode scoring; see the [failure-mode reference](../docs/failure-modes.md) |
| `closed_book` | bool | Run just this task closed-book. It is combined with the run-level flag using OR, so either one is enough to enable closed-book for this task |
| `repetition` | int | Set by the harness when the run is dispatched; do not author it by hand |

## Complete example

```json
{
  "model": {
    "provider": "litellm",
    "model_id": "openrouter/qwen/qwen3.7-flash",
    "api_key_env": "OPENROUTER_API_KEY",
    "extra_kwargs": { "temperature": 0.2 }
  },
  "judge": {
    "model_id": "openrouter/qwen/qwen3.7-flash",
    "api_key_env": "OPENROUTER_API_KEY"
  },
  "task_file": "data/example/tasks.jsonl",
  "tool_configs": [{ "tool_name": "web_search", "kwargs": { "max_results": 5 } }],
  "log_path": "logs/",
  "max_concurrent": 3,
  "repetitions_per_task": 3,
  "task_timeout_s": 900,
  "generate_trace_reports": true
}
```

## Where the output goes

```
logs/<timestamp>/
├── run_manifest.json        # immutable record of everything that shaped the run
├── run_manifest.sha256      # checksum of the manifest above
├── aggregate_results.jsonl  # one line per finished run
├── run_summary.log          # human-readable progress log (ends with the score block)
├── scores.json              # overall, per-family and per-failure-mode scores
└── results/
    ├── <task_id>.jsonl      # one line per repetition
    ├── <task_id>_traces/<task_id>.<repetition>.json         # same record, indented for reading
    └── <task_id>_reports/<task_id>.<repetition>_report.md   # with generate_trace_reports
```

See [Reproducibility and resuming](../README.md#reproducibility-and-resuming) for how the manifest gates resuming a run.
