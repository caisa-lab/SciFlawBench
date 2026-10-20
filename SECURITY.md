# Security Policy

## Overview

SciFlawBench Harness is the evaluation harness for **SciFlawBench**, an agentic scientific-flaw benchmark built on [smolagents](https://github.com/huggingface/smolagents). It runs a language model as an agent (a `CodeAgent` or a `ToolCallingAgent`) against tasks read from a `.jsonl` file, across several providers (LiteLLM, OpenAI-compatible servers, the Hugging Face Inference API), with tools such as web search, Wikipedia/arXiv lookup, webpage visiting, a `sympy` calculator and a local file reader. Tasks are fanned out over multiple OS processes (`spawn`), per-task traces are written to disk, and every run records a reproducibility manifest.

It is research software, not a hardened multi-tenant service. The agent **writes and executes Python code itself** and reads content from the open web. Please follow the recommendations below o keep your runtime and your local environment safe.

**Golden rule:** run the harness somewhere an agent that can execute arbitrary Python and read arbitrary local files cannot reach anything you would not be happy handing to a third party.

## Execution of model-generated code

The core of the benchmark is smolagents' `CodeAgent`, which writes Python and executes it locally through smolagents' `LocalPythonExecutor`. That executor is **not a security boundary**: smolagents documents it as a way to restrict the surface of tools available to the model, not as a way to isolate malicious code. `CodeAgent` is built in `src/agents/base.py` without `additional_authorized_imports`, so model code may import only smolagents' base allowlist of standard-library modules (see `BASE_BUILTIN_MODULES` in `smolagents/utils.py`). That is a convenience restriction, **not** a containment guarantee.

Consequences and mitigations:

- Treat every task execution as untrusted code. Each task runs in its own `spawn`ed OS process   (`src/core/manager.py`), which limits the blast radius across tasks but does **not** make arbitrary code execution safe — the subprocess shares your user, filesystem and network access.
- The `task_timeout_s` budget and the SIGTERM handling in `src/core/tasks.py` bound runaway work. They are a scheduling control, not a sandbox.
- For untrusted tasks, run inside a container/VM under a dedicated unprivileged user with no access to your data, or replace the executor with a real sandbox (e.g. smolagents' E2B or Docker executor).

Note that **closed-book mode is not a safe mode**: `closed_book` (global option in `config.json`, and a per-task flag) removes the harness's tools from the agent, but the agent still writes and executes Python.

## Local file access (`read_file`)

The optional `read_file` tool (`src/tools/filetools.py`) resolves whatever path the model asks for — absolute, `~`-expanded, or relative to the process working directory — and returns the contents to the model. It is **not** confined to the task's assets folder: the `data/tasks/assets/<task_id>/` layout is a convention for task authors, not an enforced boundary.

- The tool refuses binary files and files larger than 8 MB, and truncates rows and characters (`max_rows`, `max_chars`). These are robustness limits that stop a stray path from blowing up a run; they are **not** access control.
- Anything the harness process can read, the agent can read and, because tool output is sent back in the next model request (see *Data leaving your machine*), have forwarded to whichever model provider you configured. Keep secrets, keys and sensitive datasets out of reach of the process.
- Paths are resolved relative to the working directory, so prefer running from a directory that contains only the task file, its assets and its logs.

## Untrusted web content and prompt injection

Several tools reach out to third parties:

- `web_search` uses DuckDuckGo by default, and SerpAPI when `SERPAPI_KEY` is set (`src/tools/definitions.py`).
- `wikipedia_search` and `arxiv_search` query the Wikipedia and arXiv APIs (`src/tools/searchtools.py`).
- `visit_webpage` (`VisitWebpageTool`) performs an in-memory HTTP GET with a 20-second timeout and truncates the page to 40 000 characters before converting it to Markdown. The harness itself never writes fetched content to disk.

Everything these tools return is untrusted and flows straight into the model's context. This is a classic *prompt-injection* surface: a page or a search snippet can contain instructions that steer the agent into reading local files, calling other tools, or generating code that does something other than the task. There is no sanitisation layer in between.

On shared clusters, remember that the **agent**, not you, chooses which URLs are requested, and that query strings — including anything the model decides to put in them — leave your machine.

## Data leaving your machine

Every model request carries the task prompt plus the accumulated agent history, which includes tool results: the contents of files read with `read_file`, fetched web pages, search results and calculator output. Third-party search and API endpoints additionally receive the queries themselves. If your task set or assets are confidential, either run against a provider you are
permitted to send them to, or run closed-book with no local-file and no web tools.

Tasks graded by an LLM judge (`judge:rubric` validators) send an extra copy of the whole run trace — including any local file contents and fetched web pages it contains — to the model configured in the run's `judge` section, which is usually a different (third-party) provider from the one being benchmarked. The harness does not truncate that trace. Only enable judging on data you are willing to hand to that provider.

## The calculator tool

`CalculatorTool` (`src/tools/misc.py`) evaluates a model-supplied expression with `sympy` in a separate Python subprocess, capped at 5 CPU seconds and 1 GiB of address space (`resource.setrlimit`) plus a 10-second wall-clock timeout, and rejects expressions longer than 500 characters. These limits bound resource exhaustion. `sympy.sympify` is a full expression
parser and is **not** a general-purpose sandbox, so treat the expression as untrusted input.

## Task files, assets and ids are *integrity* controls

SciFlawBench uses `task_id = md5(task)` and verifies it when the task file is loaded (`src/core/tasks.py`); `python src/make_task_ids.py <task-file>` re-derives ids (and renames asset folders) after a prompt is edited. This detects **accidental** modification of a task set. It is not an authenticity or tamper-proofing mechanism: MD5 is collision-broken and the check is computed locally from the very file it validates, so anyone able to edit that file can recompute a matching id.

Run manifests (`src/core/manifest.py`) are stronger. They are SHA-256 based, written once per run and never overwritten, and a resumed run is refused unless its whole run signature still matches the manifest. `NO_REPRODUCIBILITY_GUARANTEES=true` downgrades that refusal to a warning; it is an explicit escape hatch, and runs produced under it are no longer verifiable.

`tool_configs`, per-task `extra_tools` and `validators` are read from `config.json` and the task file and determine which tools and verifiers an agent gets. **Task files are trust-bearing input:** review third-party task sets before running them.

## Credentials

- API keys are read from environment variables only. `config.json` stores the *name* of the variable (`api_key_env`), never a key value; `python-dotenv` loads a local `.env` if present.
- `.env` is listed in `.gitignore`. **Never commit it** and never paste keys into task files, configs or issue reports. On a cluster, prefer the scheduler's secret/environment mechanism.
- The harness does not write resolved secrets into the run manifest, the aggregate results file or the per-task traces. If you find a code path that logs, serialises or otherwise persists a resolved secret value, that is a vulnerability — please report it (see below).

## What ends up in the logs

Per-task traces under `logs/<timestamp>/results/` hold the full event stream: prompts, model requests and responses, tool inputs and outputs, the code the agent executed, and any file contents or web pages it pulled in. The manifest records provenance metadata (git commit and branch, platform, hostname, CPU count, installed dependency versions). Logs are gitignored, but
**review and sanitise them before sharing or publishing them** — they can easily contain private data the agent read, and they identify the machine that produced them.

## Recommended isolation

1. Run under a dedicated unprivileged user, or inside a container/VM.
2. Expose only the task file, its assets and a writable log directory to that environment. Do not mount `$HOME`, SSH keys, credential stores or production data.
3. Export only the model-provider keys you need, and leave `SERPAPI_KEY` unset if you do not want third-party search.
4. Review the task set and the agent prompts (`src/agents/prompts/`) before running it.
5. Treat the run directory as disposable and its contents as sensitive.

## Research software disclaimer

SciFlawBench Harness is research software developed at the University of Bonn, primarily intended to run on university HPC clusters. It is provided "as is", without warranty of any kind and with no security guarantees. Always review the code, task sets and assets you download or execute before running them somewhere that has access to sensitive or shared resources.

## Reporting a vulnerability

If you believe you have found a security issue in the harness itself — a resolved secret leaking into the manifest, logs or traces, a code path in a tool that escapes its intended containment, a way for fetched web content to be written to disk, or similar — please report it privately rather than opening a public issue:

- Preferred: GitHub's private vulnerability reporting (**Security** tab → "Report a vulnerability") on the repository, or
- Email the maintainers listed in [`CODEOWNERS`](CODEOWNERS).

Please include the affected file/component, a minimal reproduction (a task definition or config that triggers it, where applicable), and the impact as you see it.

## What's out of scope

- Vulnerabilities in third-party dependencies (`smolagents`, `litellm`, `sympy`, `serpapi`, `wikipedia-api`, `requests`, model-provider SDKs, ...) — please report those upstream. We are happy to take a heads-up when one of them affects how this project uses it, but the fix belongs in that project.
- The general fact that model-generated code execution and untrusted tool content are inherently risky — that is a documented property of this architecture (see above), not a new finding, unless you have found a way to escalate it beyond what is described here.
- The behaviour of the models being benchmarked (e.g. a model producing a harmful or incorrect *answer* to a task) — that is a model-behaviour question, not a harness security issue.
- A model failing, refusing, or being unable to use a tool. That is a benchmark result, not a vulnerability.

## Response expectations

This is a small research project without a dedicated security team. Reports are acknowledged on a best-effort basis; there is no formal SLA. Fixes for genuine harness-level issues (secret leakage, containment bypass, credential handling) are prioritised over feature work.
