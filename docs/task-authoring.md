# Authoring SciFlawBench Items

This is the detailed reference for **contributing benchmark items** (tasks, their ground truth and their labels). The summary lives in [`../CONTRIBUTING.md`](../CONTRIBUTING.md); the label catalogue is in [`failure-modes.md`](failure-modes.md) and the tool list in [`tools.md`](tools.md).

## Contents

1. [What SciFlawBench measures](#1-what-sciflawbench-measures)
2. [Who should contribute](#2-who-should-contribute)
3. [Scope](#3-scope)
4. [How To Submit](#4-how-to-submit)
5. [From submission to harness task](#5-from-submission-to-harness-task)
6. [Formatting rules](#6-formatting-rules)
7. [Answers: format, tolerance and rubrics](#7-answers-format-tolerance-and-rubrics)
8. [Evidence and verifiability](#8-evidence-and-verifiability)
9. [Review, dataset description and AI disclosure](#9-review-dataset-description-and-ai-disclosure)
10. [Self-review checklist](#10-self-review-checklist)
11. [FAQ](#11-faq)

## 1. What SciFlawBench measures

SciFlawBench measures whether AI agents can reliably complete real scientific tasks — data analysis, literature synthesis, experimental design, computational modelling, lab-protocol execution — while operating with tools (search, code execution, databases, domain software). Each item may require the agent to plan, call tools with correct parameters, write and run code, and produce a deliverable.

What distinguishes SciFlawBench from a general capability benchmark is its focus on **flaws**: the item is designed so that a plausible-looking agent fails in a specific, diagnosable way. That is what the **failure-mode** labels record. See [`failure-modes.md`](failure-modes.md).

## 2. Who should contribute

We are looking for:

* **Practising scientists** — chemists, molecular/cell biologists, physicists, materials scientists, geoscientists — who regularly design experiments or analyse data.
* **Computational scientists / research software engineers** who use domain tools regularly (RDKit, BLAST/BLAST+, AlphaFold/ColabFold, ASE, PyMOL, scikit-bio, astropy, LAMMPS, ...).

Domain experts with **3+ years of research experience** are the target audience: contributors must be able to state **one unambiguous ground truth** and defend it against a sceptical reviewer, and to solve their own proposed item correctly and explain why the answer is right.

No AI/ML experience is required — you only need to be an expert in your domain; we handle the agent harness.

## 3. Scope

| Domain / Subfield | Example topics |
| ----------------- | -------------- |
| Physics | data analysis, simulation, experimental design, lab-protocol execution, literature synthesis |
| General scientific literature synthesis | multi-paper claim reconciliation, meta-analysis, citation verification |

**Out of scope**

* pure trivia recall with no reasoning or tool use ("What year was X discovered?");
* non-scientific general coding tasks;
* purely subjective or ethical debates with no verifiable answer;
* tasks requiring proprietary or paywalled data that a reviewer cannot access.

## 4. How To Submit

Submissions are collected through Google Forms, which define the fields you fill in (your identity, the domain, the task text, its ground truth, the tools it needs and the failure-mode flags it probes).

1. **Register as an annotator** — <https://docs.google.com/forms/d/159NWfHjjZRg4pP5HzAyBStZldaby0kF1AMuI3y1mPIY>
2. **Submit your items** — <https://docs.google.com/forms/d/e/1FAIpQLSdZKM8HSoGEXnV9oTWezMQhHekC2rcyH8r6VqzFjZ_DdsYAPg/closedform>

Register first, then submit single items or a curated batch; a batch should come with a dataset description (see [section 9](#9-review-dataset-description-and-ai-disclosure)). Questions, tool requests and proposals for new failure modes belong in the issue tracker.

## 5. From submission to harness task

Accepted items are converted into harness tasks. [`../utils/build_task.ipynb`](../utils/build_task.ipynb) turns one submission row into a task line,
checks it against these rules, and appends it to a task file.

A finished task looks like this:

```json
{
  "task_id": "40897e4550180c736a05c904f440566c",
  "task": "A CSV file with instrument readings is provided at `data/tasks/assets/{TASK_ID}/measurements.csv`. Each row has a `sample_id`, a `temperature`, and a `value` column. Read the file and compute the mean of the `value` column. Report the mean, rounded to two decimal places, as a single number.",
  "agent_id": "code_agent",
  "failure_modes": { "quantitative": ["correctness"], "qualitative": [] },
  "extra_tools": [{ "tool_name": "read_file" }],
  "validators": [{ "name": "content:numeric_match", "kwargs": { "expected": "18.4", "tol": 0.05 } }]
}
```

Three things follow from this mapping and are worth repeating:

* **There is no `ground_truth` field in the task file.** The expected answer lives in the validator kwargs. An `answer`/`ground_truth` key written into a task line is silently ignored by the loader — if a check does not reference your ground truth, nothing will catch a wrong answer.
* **`task_id` is `md5(task)`.** Never edit a prompt without re-deriving its id; the run refuses to start on a mismatch. Repair ids (and rename their asset folders) with `python src/make_task_ids.py <task-file>` (`--check` for a no-write).
* **`failure_modes` is required and drives scoring.** It is the list of the modes you set to `true`, split into `quantitative` and `qualitative` (at least one mode in total). The harness averages each task's score into an overall score and one score per mode, each restricted to the tasks that declare it — see [Scoring](../README.md#scoring).

Local data files for an item live in `data/tasks/assets/<task_id>/` and are addressed in the prompt as `data/tasks/assets/{TASK_ID}/<file>`; the harness substitutes the real id at run time. See [`../data/README.md`](../data/README.md). If files are required for a task, make sure to create them in the appropriate asset folder with the correct filenames and folder name.

## 6. Formatting rules

* **Self-contained.** Include any tables, sequences or file contents inline or as an attachment that a sandboxed agent could fetch. The agent cannot browse your filesystem.
* **Machine-checkable ground truth** where possible: an exact string, a numeric value with tolerance, or a short rubric of required elements.
* **Canonical tool names only** — from [`tools.md`](tools.md). Do not invent names.
* **All 11 failure-mode booleans present**, unused ones `false`.
* **No shortcuts.** Reject designs an agent could pass by dumping every possible answer, doing nothing, or exploiting a pattern in the ground truth.
* **Pinned versions** for anything whose behaviour is version-dependent (state the version in `notes`).

## 7. Answers: format, tolerance and rubrics

* Keep answers short and machine-checkable: a value with tolerance, a canonical name, or a yes/no plus a justification rubric.
* Use standard terminology and nomenclature (IUPAC names, standard gene symbols, SI units).
* For numerical answers state units, tolerance (e.g. ±5%) and significant figures.
* For open-ended deliverables (plans, protocols, figures, reports) give an explicit **checklist rubric** — a list of required elements — instead of a single string, so two reviewers grade it the same way.
* Rubrics are graded by an LLM judge (`judge:rubric`), which sees the run's trace. Phrase each rubric as an answerable question with an explicit pass condition.

## 8. Evidence and verifiability

Provide, where possible, a reference a reviewer can check:

* **Preferred** — peer-reviewed papers, official database entries (UniProt, PDB, NCBI, PubChem), official tool documentation.
* **Acceptable** — well-known textbooks, established technical resources (e.g. NEB or Sigma-Aldrich bulletins).
* **Avoid** — personal blogs, unmoderated forums, social media, or AI-generated summaries as the sole source.

If an item deliberately withholds part of the reference from the agent (for example an implicit-domain-knowledge check), say so in `notes` and keep the hidden reference with the organizers, not in the prompt.

## 9. Review, dataset description and AI disclosure

Every item is checked by **two reviewers** — a domain expert and an agentic-evaluation reviewer —before acceptance. Reviewers comment on or flag issues rather than editing your item directly.

### Dataset description

Each contributor or group submits a description covering:

* **Coverage** — which domains, subfields and failure modes your items represent.
* **Methods** — how the items were created (from personal research experience, adapted from real protocols/papers, or AI-assisted and then verified).
* **AI use disclosure** — which AI tools helped draft the items, how, and confirmation that a human domain expert independently verified the ground truth and the labels.
* **Quality control** — the review and verification you applied.
* **Challenges** — notable difficulties and how you resolved them.

The description should be detailed enough that another expert could reconstruct a comparable set.

## 10. Self-review checklist

Before submitting, verify **all** of the following:

- [ ] The item is self-contained.
- [ ] There is exactly one correct answer, or an explicit rubric.
- [ ] The item is realistic — it reflects a genuine scientific/agentic task.
- [ ] The domain/subfield label is correct.
- [ ] Every failure-mode flag marked `true` is genuinely probed by the item; everything else is
      `false`.
- [ ] Tool names are canonical, tools the agent lacks are declared, and versions are pinned where it
      matters.
- [ ] The item cannot be passed by a shortcut (empty response, listing all answers, guessing).
- [ ] Entities are disambiguated (full names, not bare acronyms unless universally standard).
- [ ] Numbers have units, tolerance and significant figures.
- [ ] A reviewer can verify the ground truth from the reference you supplied.
- [ ] Any AI assistance is disclosed and human-verified.

## 11. FAQ

**Do all 11 failure-mode flags need to be considered?**

- Yes — all must be present, but most items should set only 1–3 to `true`. More than 3–4 usually means the item is doing too much; split it.

**Can one item test a quantitative and a qualitative mode?**

- Yes — that is common and encouraged, as long as each `true` flag is independently justified in `notes` and has a matching validator.

**What if the ground truth is not a single value?**

- Provide an explicit checklist rubric, and implement it as a `judge:rubric` validator.

**My item needs a multi-agent handoff.**

- The harness builds single agents today, so, for now, avoid designing items that require multi-agent handoffs.

**My item needs a tool or library that does not exist yet.**

- List it in `notes`, open an issue, and expect a delay while the tool is added — see [`tools.md`](tools.md#5-tools-requested-but-not-implemented-yet).

**Can I use AI to draft items?**

- Yes, with disclosure (section 9). Every ground truth and every label must be independently verified by a human domain expert; AI-drafted items without that verification are rejected.

**How is authorship credit assigned?**

- Contributors with at least a threshold number of accepted items after peer review are invited as co-authors on the resulting dataset paper; the threshold is announced with the release timeline.
