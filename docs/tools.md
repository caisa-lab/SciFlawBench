# SciFlawBench Tools and Scope Reference

This is the canonical list of tools an item may use, how the harness builds an agent's tool surface, and what the sandboxed code executor can and cannot do. It complements [`task-authoring.md`](task-authoring.md) (item authoring) and
[`failure-modes.md`](failure-modes.md) (what items probe).

The single source of truth for what exists is the tool registry in [`../src/tools/definitions.py`](../src/tools/definitions.py); this document is kept in sync with it.

## 1. Canonical tool names

Every tool named in an item submission must match a name from this table. Do not invent ad hoc names — if you need a tool that is missing, ask the organizers to add it (see [section 5](#5-tools-requested-but-not-implemented-yet)).

| Tool | Options (defaults) | Default tool? | Notes |
| ---- | ------------------ | ------------- | ----- |
| `web_search` | `max_results` (`8`), `rate_limit` (`1.0`, requests/second), `engine` (`"duckduckgo"`) | yes | SerpAPI replaces DuckDuckGo when `SERPAPI_KEY` is set **and** `engine` is not `"duckduckgo"` |
| `wikipedia_search` | `operator` (`"OPERATOR EMAIL NOT SET"`) | yes | The operator email is sent in the request `User-Agent`, as the Wikimedia API policy asks |
| `arxiv_search` | `operator` (`"OPERATOR EMAIL NOT SET"`) | yes | Same, for the arXiv API. Fuzzy-matches titles/abstracts and returns ids, links and abstracts |
| `visit_webpage` | – | yes | Fetches a URL and returns the page as Markdown (20 s timeout, truncated to 40 000 characters) |
| `calculator` | – | yes | Evaluates a `sympy` expression in a subprocess limited to 5 CPU-seconds, 1 GiB and a 10 s wall clock; rejects expressions longer than 500 characters |
| `current_time` | – | yes | Returns the current local time |
| `read_file` | `max_rows` (`50`), `max_chars` (`20000`) | **no** — opt-in | Renders CSV/TSV/PSV as tables, pretty-prints JSON, summarises JSONL, lists directories, and returns other text files as-is. Refuses binaries and files over 8 MB |
| `json_answer_tool` | `required_fields` (`[]`) | **no** — opt-in | Validates that a proposed final answer is JSON containing the required keys. It does **not** submit the answer: the agent still has to call `final_answer` |

The default agents (`code_agent` for code-generating agents, `tool_agent` for tool-calling agents) start with `web_search`, `wikipedia_search`, `visit_webpage`, `calculator`, `current_time` and `arxiv_search`. `final_answer` is supplied by the framework and is always available.

## 2. Declaring tools on an item

A submission's `tools` array lists the tools the agent may or must use. In the harness this becomes:

* the agent's **default** tool set (from the chosen `agent_id`), plus
* anything listed in the task's **`extra_tools`**.

```json
{
  "agent_id": "code_agent",
  "extra_tools": [
    { "tool_name": "read_file" },
    { "tool_name": "json_answer_tool", "kwargs": { "required_fields": ["papers"] } }
  ]
}
```

Rules to remember:

* **Only list tools the agent does not already have.** `extra_tools` is *appended* to the defaults, and smolagents rejects duplicate tool names (`ValueError: ... should have a unique name!`), so listing `web_search` on a `code_agent` task fails the run rather than being a harmless no-op.
* `extra_tools` entries may be bare strings (`"read_file"`) or objects with `kwargs`.
* An item that must read a supplied file needs `read_file`: the code executor cannot open files by itself (see [section 4](#4-what-the-code-executor-can-do)).
* A run-wide override of a tool's options is configured in the run config's `tool_configs` (see [`../examples/README.md`](../examples/README.md#tool-overrides)), not per item.

### "Must use tool X"

A submission can require a particular tool call, but the harness has no deterministic tool-call verifier yet: the check is a `judge:rubric` validator whose rubric names the required call and its parameters, applied to the run's trace (which contains every tool name, input and output). If your item needs a hard, mechanical assertion on the invocation log, say so in the issue tracker.

## 3. Tool versioning rules

* If an item depends on a specific tool or package version (e.g. `blastp`, `rdkit==2023.09`), pin the version in the item's `notes` field.
* Items requiring Python 2.7 or other unmaintained/end-of-life packages are not accepted.
* Where a tool has multiple major versions with different behaviour (e.g. different default coordinate epochs), state which version the ground truth assumes.

## 4. What the code executor can do

`code_agent` runs the Python it writes inside smolagents' local executor. That executor is a *containment* for the tool surface, not a security sandbox, and it is deliberately narrow:

* **Imports are restricted** to a small allowlist of standard-library modules: `collections`, `datetime`, `itertools`, `math`, `queue`, `random`, `re`, `stat`, `statistics`, `time`, `unicodedata`.
* `open()`, `eval`-style builtins and shell access are **not** available.
* Consequently `numpy`, `pandas`, `matplotlib`, `astropy`, `rdkit`, `Bio`/`BioPython` and friends **cannot be imported today**.

Practical consequences for item design:

* An item that needs to read a supplied file must include the `read_file` tool, and should be written so the arithmetic is doable with the standard-library modules above (or a tool).
* An item whose ground truth depends on a domain library is **not runnable yet**, even if the tool is listed in this document's future-work section. Note the dependency in your submission so the organizers can prioritise it.
* Items whose deliverable is a rendered figure cannot be visually graded by the harness today; phrase the rubric around the computation, the values and the text of the trace (see [`failure-modes.md`](failure-modes.md#9-aesthetic-quality-qualitative)).

## 5. Tools requested but not implemented yet

These names appear in contributor material and in annotations, but are **not** registered in the harness. Listing them in an item will make the run fail with an unknown-tool error. Ask the organizers to add them (or pin an equivalent implemented tool) before submitting.

| Requested name | Status | Workaround today |
| -------------- | ------ | ---------------- |
| `python_execution` | Not a tool: it is the agent's built-in capability | Nothing to declare — code execution is always available to `code_agent` |
| `astropy`, `matplotlib`, `numpy`, `pandas` | Not importable (executor allowlist) | Re-express the computation with standard-library modules, or request a dedicated tool |
| `astroquery_simbad` | Not implemented | `visit_webpage` against the SIMBAD web form / `web_search` |
| `blast`, `primer3`, `rdkit`, `BLAST+`, `ASE`, `PyMOL`, `scikit-bio`, `LAMMPS` | Not implemented | None — request them |
| `wget`, `curl` | Not implemented | `visit_webpage` for pages; ask for a fetch-to-disk tool if an item needs files on disk |

## 6. Domain and scope

| Domain / Subfield | Example topics |
| ----------------- | -------------- |
| Physics | data analysis, simulation, experimental design, lab-protocol execution, literature synthesis |
| General scientific literature synthesis | multi-paper claim reconciliation, meta-analysis, citation verification |

**Out of scope:** pure trivia recall with no reasoning or tool use (e.g. "What year was X discovered?"), non-scientific general coding tasks, purely subjective or ethical debates with no verifiable answer, and tasks requiring proprietary or paywalled data that a reviewer cannot access.

Contributors in the computational sciences regularly reach for domain tools such as RDKit, BLAST/BLAST+, AlphaFold/ColabFold, ASE, PyMOL, scikit-bio, astropy and LAMMPS. An item that needs one of these is welcome — it just has to be flagged, because the tool has to exist in the harness before the item can run.
