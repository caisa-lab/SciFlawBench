# Contributing to SciFlawBench

Thank you for taking the time to contribute to our project!

There are two ways to contribute, with different requirements:

| Track | Who it is for | Where it starts |
| ----- | ------------- | --------------- |
| **1. Benchmark items** — tasks that measure whether an AI agent can complete real scientific work, together with their ground truth and the flaws they are designed to catch | Practising scientists and computational scientists with 3+ years of research experience | [Track 1](#track-1--contributing-benchmark-items) · [`docs/task-authoring.md`](docs/task-authoring.md) |
| **2. Harness code** — the evaluation harness in `src/` that runs items against models | Software engineers | [Track 2](#track-2--harness-development) |

---

# Track 1 — Contributing benchmark items

You do not need any AI/ML experience: you need to be an expert in your domain and able to state one
unambiguous ground truth and defend it against a sceptical reviewer.

## 1. Register and submit

1. **Register as an annotator** — <https://docs.google.com/forms/d/159NWfHjjZRg4pP5HzAyBStZldaby0kF1AMuI3y1mPIY>
2. **Submit your items** with the **SciFlawBench Task submission form** — <https://docs.google.com/forms/d/e/1FAIpQLSdZKM8HSoGEXnV9oTWezMQhHekC2rcyH8r6VqzFjZ_DdsYAPg/closedform>

Single items are welcome. If you submit a batch, include a dataset description (coverage, methods,
AI use, quality control), described in
[`docs/task-authoring.md`](docs/task-authoring.md#9-review-dataset-description-and-ai-disclosure).

## 2. What makes a good item

* It is **realistic** — a task a competent scientist would actually face, not trivia.
* It is **self-contained** — everything the agent needs is in the prompt or an attached file.
* It has **one unambiguous ground truth**, or an explicit checklist rubric.
* It has **no shortcut** — it cannot be passed by guessing, dumping every candidate answer, or doing
  nothing.
* It is **labelled honestly**: only the failure modes the item genuinely probes are set to `true`.

## 3. What items are designed to catch

Every item is labelled with the flaws it probes — 4 quantitative and 7 qualitative flags:

| Quantitative | Qualitative |
| ------------ | ----------- |
| `correctness`, `correct_tool_calls`, `code_safety`, `robustness_against_adversarial_inputs` | `sycophancy`, `planning`, `reasoning`, `uncertainty_awareness`, `aesthetic_quality`, `lost_context_on_multi_agent_tasks`, `implicit_domain_knowledge` |

Most items should set only **1–3** of these to `true`; more than 3–4 usually means the item is doing too
much. Each flag — what it measures, good and bad designs, a worked example, and the check that
implements it in the harness — is documented in [`docs/failure-modes.md`](docs/failure-modes.md).

## 4. Using tools

Items may hand the agent tools: web search, arXiv/Wikipedia lookup, webpage visiting, a calculator, a
file reader and a JSON-answer validator. The rules that trip people up:

* Tool names must be **canonical** — the full list is in [`docs/tools.md`](docs/tools.md). Never invent
  a name.
* An item that must read a supplied file needs `read_file`: the code executor cannot open files on its
  own, and cannot import `numpy`, `pandas`, `astropy`, `matplotlib` or similar. Design the arithmetic so
  it is doable with the standard library or a tool.
* Domain tools (`rdkit`, `blast`, `astropy`, ...) are **not implemented yet** — name them in the item's
  `notes` and open an issue so they can be added before the item runs.
* Requirements like "must call tool X with parameter Y" are checked by a rubric applied to the run's
  trace, since the harness has no mechanical tool-call verifier yet.

## 5. Before you submit

Work through the [self-review checklist](docs/task-authoring.md#10-self-review-checklist). In short:
self-contained; one answer or an explicit rubric; honest labels; canonical tools; no shortcuts;
disambiguated entities; units with tolerances; and a reference a reviewer can check.

Accepted items are checked by a domain expert and an agentic-evaluation reviewer, who comment on or
flag issues rather than editing your item. See
[`docs/task-authoring.md`](docs/task-authoring.md) for how to submit an item and how
your submission becomes a harness task.

## 6. Reporting a flawed item or a missing capability

If an item is unsolvable, mislabelled, ambiguous, or needs a tool or a check that does not exist yet,
open an issue using the **task or verifier issue** template. The failure modes you cannot currently
express matter as much as the items themselves — they shape what the harness supports next.

---

# Track 2 — Harness development

### Developer setup

Follow the instructions in the README to get the repository running on your machine.

For developer packages to be added as well, make sure to

```bash
pip install ."[dev]"
```

from whatever environment manager you are using instead of just using '.'


### Issues and discussions

For any bugs or general fixes to the harness itself, make sure to check the issues section. If you see any open issues related to what you are suggesting, upvote that issue and perhaps add a comment under this issue for further clarification on your specific perspective on the isssue. Otherwhise, open a new issue according to our templates to specify what needs to be fixed or added.

For additional feature requests, please put these under the discussions section before opening a related issue so that maintainers can have a dialogue about implementation before they are fully done.


### Style considerations

Ruff linting and formatting has been setup on the repository so please try and match things according to that. to check if your code is in compliance before making a commit simply run the following:
    ```bash
    ruff check .
    ```
Ensure that there are no errors before a commit is made

For docstrings, the following style is preferred:

    ```python
    """
    <description>

    Args:
        <arg1_name> (<arg1_type>): ...
        <arg2_name> (<arg2_type>): ...
        ...

    Returns (<return_type>): ...

    """
    ```

On the commit messages, try to preface the message with one of the following tags:
    a) "Fix": fix for unintended or innefficient behaviour,
    b) "Feat": New feature or added functioniality,
    c) "Breaking": Break to the previous api such that downstream users may need to update their configurations

### Testing policy

This project uses pytest as the platform on which tests are written. once the environment is sourced, running the test
suite can be accomplished with the following command:  `pytest`

To run a specific test file that you may be developing it is also possible to specify by using:

```python
pytest /path/to/your/file <options>
```

Ensure that for any major feature addition that there are unit tests that make sure it has its intended behaviour just from the perspective the class by itself.

If the addition is more structural or would generally benefit from having more involved tests, add an integration test with either the fake classes as can be seen in `tests/integration/smoke` or with real api calls by copying the patterns in `tests/integration/live`. If the test will ultimately require API usage which may be limited, mark it with live to prevent running them by mistake. When needed, the live suite of tests can be run using the 'live' argument to pytest.


### Checklist before making a pull request

1. There are no ruff formatting or linting errors
2. Any major changed/additional functions are properly documented with updated docstrings in the proper style
3. Any additional **new** code paths should have a test checking that they function as intended
4. Running pytest causes no errors

### Where to read more

| Document | What it covers |
| -------- | -------------- |
| [`README.md`](README.md) | installing the harness, the run configuration and how to run a benchmark |
| [`examples/README.md`](examples/README.md) | every run-configuration key, every runner flag, and the LLM-judge settings |
| [`docs/README.md`](docs/README.md) | index of the documentation set |
| [`docs/task-authoring.md`](docs/task-authoring.md) | contributing benchmark items: schema, scope, review process |
| [`docs/failure-modes.md`](docs/failure-modes.md) | the failure modes items are labelled with, and how each is checked |
| [`docs/tools.md`](docs/tools.md) | canonical tool names, tool surface rules, executor limits |
| [`data/README.md`](data/README.md) | task-file layout, task ids and the asset convention |
| [`data/example/README.md`](data/example/README.md) | task-file fields and the verifier reference |
| [`SECURITY.md`](SECURITY.md) | what the harness does and does not protect against, and how to report a vulnerability |

Questions, tool requests and proposals for new failure modes belong in the issue tracker; the
maintainers are listed in [`CODEOWNERS`](CODEOWNERS).
