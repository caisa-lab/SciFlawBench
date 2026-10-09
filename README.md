<div align="center">

# SciFlawBench

**🔬 An agentic benchmark for measuring whether AI agents can reliably complete real scientific tasks. 🔭**

</div>

---

## What is SciFlawBench?

SciFlawBench measures whether AI agents can reliably complete **real scientific tasks** — data analysis, literature synthesis, experimental design, computational modelling, lab-protocol execution — while operating with tools: web search, arXiv and Wikipedia lookup, webpage visiting, a sandboxed
Python interpreter and local file readers.

What separates it from a general capability benchmark is the **flaw** framing. Every item is written by a practising scientist and built so that a plausible-looking agent fails in a specific, diagnosable way. Each item is labelled with the **failure modes** it probes, catalogued in [`docs/failure-modes.md`](docs/failure-modes.md).

### How a run works

```
task file (.jsonl) -> one process per task -> agent + tools -> verifiers -> results + scores
```

Each task runs in its own process under a wall-clock timeout and writes its own result file. Verifiers grade either the final answer (deterministic
checks) or the whole run trace (an LLM judge), and the harness aggregates every recorded run into the overall and per-failure-mode scores — see [Scoring](#scoring).

## Repository overview

### Source

```
src/
├── core/
│   ├── config.py               # base types used for parsing the config (w/ pydantic)
│   ├── events.py               # defines event watchers and base event primitives
│   ├── manager.py              # runtime manager code that handles dispatching subprocesses
│   ├── registry.py             # registry code which is reused for agent and tool registry (handles mapping string -> obj factory)
│   └── tasks.py                # task primitives and the run_task function used to execute a task
├── agents/
│   ├── prompts/
│   │   ├── code_agent.yaml        # default prompt associated with coding agent
│   │   └── tool_agent.yaml        # default prompt associated with tool calling agent
│   ├── prompts.py              # loads the prompt templates
│   ├── definitions.py          # definitions of base agents to be used in testing and agent registry
│   └── base.py                 # contains the build_agent function
├── evaluation/
│   ├── print_report.py         # renders run traces and results as Markdown reports
│   ├── definitions.py          # definitions of verifiers and the verifier registry
│   ├── judge.py                # LLM-as-a-judge client, judge prompt and reply parsing
│   ├── scoring.py              # failure-mode taxonomy and overall/per-mode score aggregation
│   └── base.py                 # defines basic necessary structures for validation code
├── models/
│   └── base.py                 # defines how to build a model and model wrapper
├── tools/
│   ├── misc.py                 # miscellaneous custom tool classes that get registered
│   ├── definitions.py          # tool definitions and the tool registry
│   ├── searchtools.py          # contains the searchtools available to the agents: (arxiv, SerpAPI, wikipedia)
│   └── base.py                 # the wrapper used by every tool
└── main.py                     # CLI entry point for running the harness
```

### Data

All benchmark data (task files and the local files tasks depend on) lives under `data/`:

```
data/
├── example/      # self-contained example tasks.jsonl for demos, smoke tests and quick runs
└── tasks/        # the official SciFlawBench task sets, versioned over time
    ├── v0/       # version 0 of the official task set
    └── assets/   # one folder per task id, holding that task's local files
```

See [`data/README.md`](data/README.md) for the full description.

### Examples

A ready-to-edit run configuration lives under `examples/`:

```
examples/
├── config.json   # a minimal run configuration (model + task file)
└── README.md     # reference for every config key and runner flag
```

Start from [`examples/config.json`](examples/config.json), and see [`examples/README.md`](examples/README.md) for the full argument reference.

### Authoring helpers

```
utils/
└── build_task.ipynb   # fill in one submitted item and append it to a task file as a JSONL line
```

[`utils/build_task.ipynb`](utils/build_task.ipynb) turns one submitted item into a task line: fill in the fields and it validates the item, computes the `task_id`, and appends it to a task file. See [`docs/task-authoring.md`](docs/task-authoring.md#5-from-submission-to-harness-task).

## Requirements

- **Python >= 3.12**
- API credentials for at least one model provider (LiteLLM covers most hosted providers), or a local OpenAI-compatible server

## Installation

### Installing through pip

```bash
git clone https://github.com/caisa-lab/SciFlawBench.git && cd SciFlawBench
python -m venv .venv && source .venv/bin/activate   # optional, but recommended
pip install .                                       # add -e for an editable install
```

### Installing with uv

```bash
git clone https://github.com/caisa-lab/SciFlawBench.git && cd SciFlawBench
uv sync                                             # runtime dependencies
uv sync --all-extras && pre-commit install          # developer setup: dev tools + git hooks
```

Developer setup (test suite, linting, git hooks) is described in [`CONTRIBUTING.md`](CONTRIBUTING.md#developer-setup).

## Quick start

Every run argument lives in a single JSON file. A minimal configuration only needs to choose a model and a task file:

```json
{
  "model": {
    "provider": "litellm",
    "model_id": "openrouter/qwen/qwen3.7-flash",
    "api_key_env": "OPENROUTER_API_KEY"
  },
  "task_file": "data/example/tasks.jsonl"
}
```

[`examples/README.md`](examples/README.md) is the full reference: every key accepted in the config file (run settings, model settings, per-tool overrides), the registered tools and their options, the environment variables the harness reads, and every runner flag.

1. Copy and edit the example config ([`examples/config.json`](examples/config.json)) with your
   model access settings and task file.

2. Export the API key environment variable named by `api_key_env` (or put it in a `.env` file,
   which is loaded automatically).

3. Run the harness with your config path:

```bash
python src/main.py --config my-run.json
```

4. Useful runner flags:

```bash
python src/main.py --config my-run.json --dry          # print the resolved config and exit
python src/main.py --config my-run.json --show_trace   # per-task markdown trace reports
python src/main.py --config my-run.json --closed_book  # closed-book mode: the agent gets no tools
```

Run `python src/main.py --help` for the complete list.

## Scoring

Each task is scored by the fraction of its validators that pass: `1.0` when every check passes, `0.0` when none do, and a task with no checks at all has no score. At the end of a run the harness aggregates those task scores into `scores.json` and appends a matching block to `run_summary.log`:

- an **overall** score — the mean task score across every recorded run;
- a **per failure mode** score for every mode the tasks declare in their `failure_modes` field, measured only over the tasks that declare that mode; and
- a score for each **family** (`quantitative`, `qualitative`).

Runs killed or crashed before verification (no checks) are counted but excluded from the scores. The failure-mode taxonomy itself is documented in [`docs/failure-modes.md`](docs/failure-modes.md).

## Reproducibility and Resuming

Every run records an immutable manifest next to its logs, and an interrupted run can be resumed only while that manifest still matches the current configuration.

### The run directory

```
logs/<timestamp>/            # the `log_path` itself when `restarting` is set
├── run_manifest.json        # immutable record of everything that shapes the run
├── run_manifest.sha256      # checksum of the manifest above
├── aggregate_results.jsonl  # one line per finished run (each links to the manifest sha)
├── run_summary.log          # human-readable progress log (ends with the score block)
├── scores.json              # overall, per-family and per-failure-mode scores
└── results/                 # per-task results: one file per task id, one line per repetition
```

The manifest captures the task set (file and content hashes), the local assets it depends on, the agent prompt templates, the model configuration, the tool configuration, the judge configuration (for runs whose tasks are graded by an LLM judge), the budget (repetitions, timeout) and environment provenance (git revision, dependency versions, platform).

### Resuming an interrupted run

Point `log_path` at the interrupted run's directory and set `restarting` to `true`:

```json
{
  "log_path": "logs/<timestamp>",
  "restarting": true
}
```

Repetitions already recorded under `results/` are skipped and everything else is re-run. Before that, the harness recomputes this run's *signature* and compares it with the manifest. If anything that affects results has changed — the task set, the local assets, the prompts, the model or tool configuration, the budget — the run is **refused**:

```
Refusing to resume: the checkpoint does not match this run.
   - run signature differs from the manifest (changed: task_set_sha256).
```

Set `NO_REPRODUCIBILITY_GUARANTEES=true` to downgrade those checks to warnings, e.g. when you knowingly want to continue after changing something (the resulting numbers are then no longer directly comparable). A checkpoint that holds results but no manifest cannot be verified and is refused for the same reason — rerunning from scratch is the safe option.

## Contributing

The project grows along two tracks:

| Track | Who it is for | Where it starts |
| ----- | ------------- | --------------- |
| **1. Benchmark items** — scientific tasks with a ground truth and failure-mode labels | Practising scientists | [`CONTRIBUTING.md`](CONTRIBUTING.md) Track 1 · [`docs/task-authoring.md`](docs/task-authoring.md) |
| **2. Harness code** — the runner, tools, verifiers and scoring under `src/` | Software engineers | [`CONTRIBUTING.md`](CONTRIBUTING.md) Track 2 |

Items are submitted through two Google Forms (annotator registration, then task submission); both are linked in [`docs/task-authoring.md`](docs/task-authoring.md#4-how-to-submit), together with the rules an item must satisfy.

## Where to read more

| Document | What it covers |
| -------- | -------------- |
| [`examples/README.md`](examples/README.md) | every configuration key, every runner flag, tool overrides, environment variables and judge settings |
| [`data/README.md`](data/README.md) | task-file layout, task ids and the asset convention |
| [`data/example/README.md`](data/example/README.md) | the per-task fields and the verifier reference |
| [`docs/README.md`](docs/README.md) | index of the documentation set |
| [`docs/task-authoring.md`](docs/task-authoring.md) | how to author and submit an item |
| [`docs/failure-modes.md`](docs/failure-modes.md) | every failure mode, and the check that implements it |
| [`docs/tools.md`](docs/tools.md) | canonical tool names, tool-surface rules and code-executor limits |
| [`SECURITY.md`](SECURITY.md) | what the harness does and does not protect against |
| [Google Colab](https://colab.research.google.com/drive/1ctDfb7he22O-ipqqhIxSmfM40fEOTWXI?usp=sharing) | run the harness end to end without installing anything |

## Citation

TBD

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
