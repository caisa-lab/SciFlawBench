---
name: Task / Verifier issue
about: A specific benchmark task, verifier, or ground-truth answer looks wrong
title: "[Task]: "
labels: task-data
assignees: ''
---

## Task ID(s) affected
`task_id`:

## What's wrong
- [ ] Ground truth / expected answer appears incorrect
- [ ] Task wording is ambiguous or leaks the answer's cardinality/shape
- [ ] Verifier is producing a false pass
- [ ] Verifier is producing a false fail
- [ ] Task consistently times out / gets killed
- [ ] Other:

## Task definition
```json
paste the relevant task JSONL entry here
```

## Model output vs. expected
**Model's final answer:**
```

```
**Expected (`answer` / `checks`):**
```

```

## Verifier result
```json
paste the relevant portion of the TaskResult / verification field
```

## Why this seems wrong
Explain the discrepancy — e.g. the verifier's tolerance is too strict, the extraction regex grabbed the wrong number, the task instructions don't match what the checks actually validate, etc.

## Suggested fix
If you have one — a corrected ground truth, a verifier kwarg change, a rewritten task prompt, etc.
