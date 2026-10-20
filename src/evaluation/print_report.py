import json
import re
from pathlib import Path
from typing import Any

TOOL_ICONS = {
    "web_search": "🔍",
    "wikipedia_search": "📚",
    "arxiv_search": "📄",
    "calculator": "🧮",
    "visit_webpage": "🌐",
    "read_file": "📂",
    "current_time": "🕒",
    "json_answer_tool": "📦",
}


def extract_concise_trace(events: list[dict]) -> tuple:
    # TODO: This has to be improved or the CodingAgent, and also checked for other than search and claculator tool
    """
    Extracts a concise trace from the list of AgentEvent objects, summarizing tool and model calls, duraion, and tokens.

    Args:
        events: List[Dict]: list of dumped agent events

    Returns: A tuple containing the concise trace, token counts, and duration of the events.
    """

    trace = []
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    pending_tool = None

    start_ts = events[0].get("timestamp", 0.0) if events else 0.0
    end_ts = events[-1].get("timestamp", 0.0) if events else 0.0
    duration = round(end_ts - start_ts, 3) if end_ts and start_ts else 0.0

    for e in events:
        tool, payload = e.get("event_type", ""), e.get("payload", {})

        if tool == "tool_call_start":
            inputs = payload.get("kwargs") or (payload.get("args", [])[0] if payload.get("args") else {})
            pending_tool = {"tool_name": payload.get("name"), "inputs": inputs}
        elif tool == "tool_call_end" and pending_tool:
            pending_tool["result"] = payload.get("result")
            trace.append(pending_tool)
            pending_tool = None
        elif tool == "model_call_end":
            res = payload.get("result", {}) or {}
            content = res.get("content", "") if isinstance(res, dict) else str(res)
            trace.append({"event_type": tool, "content": content})

            usage = res.get("token_usage", {}) or (res.get("raw") or {}).get("usage") or {}
            tokens["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            tokens["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0

    if pending_tool:
        trace.append(pending_tool)

    return trace, tokens, duration


def mermaid_schema(trace, status):
    steps = []
    for x in trace:
        has_error = False
        if x.get("content"):  # omit empty thoughts
            steps.append(("💭 Thought", None, has_error))
        elif "tool_name" in x:
            name = str(x["tool_name"]).replace('"', "'")
            label = f"{TOOL_ICONS[name]} {name}" if name in TOOL_ICONS else name
            inp_key = json.dumps(x.get("inputs"), sort_keys=True)
            res = str(x.get("result") or "").lower()
            has_error = "error" in res
            steps.append((label, inp_key, has_error))

    # Collapse identical consecutive calls
    merged = []
    for label, inp, has_error in steps:
        if inp is not None and merged and merged[-1][:3] == [label, inp, has_error]:
            merged[-1][3] += 1
        else:
            merged.append([label, inp, has_error, 1])

    # Node labels are quoted so emoji, spaces and "×" are valid mermaid
    lines = ["flowchart LR", '    n0(["Start"])']
    for i, (label, _, has_error, count) in enumerate(merged, start=1):
        suffix = f" ×{count}" if count > 1 else ""
        error = " ❌" if has_error else ""
        lines.append(f'    n{i}["{label}{suffix}{error}"]')
    end = len(merged) + 1
    lines.append(f'    n{end}(["{status}"])')
    lines.append("    " + " --> ".join(f"n{i}" for i in range(end + 1)))
    lines.append(f"    class n{end} {'passed' if status == 'Passed' else 'failed'}")
    lines.append("    classDef passed fill:#d4edda,stroke:#28a745,color:#155724")
    lines.append("    classDef failed fill:#f8d7da,stroke:#dc3545,color:#721c24")
    return "```mermaid\n" + "\n".join(lines) + "\n```"


def get_preview(tool, text):
    preview = ""
    if "search" in tool:
        pattern = r"\[.*?\]\(https?://(?:www\.)?([^\)]+)\)"
        urls = [f"{u.rstrip('/')}" for u in re.findall(pattern, str(text or ""))]
        preview = "**URLs Found:**\n" + ", ".join(urls) + "\n\n" if urls else ""
    if "visit" in tool and text:
        first_lines = "\n> ".join([line.strip() for line in text.split("\n") if line.strip()][:5])
        preview = f"**Snippet**\n> {first_lines}...\n\n"

    details = (
        f"<details>\n<summary>Click to expand output ({tool})</summary>\n\n```text\n{preview}\n```\n\n</details>\n"
    )

    return f"{preview}{details}"


def render_judge_section(checks: list[dict]) -> list[str]:
    """Markdown lines for the LLM-judge checks among `checks` (empty when there are none)."""
    judged = [c for c in checks if "rubric" in (c.get("metadata") or {})]
    if not judged:
        return []

    lines = ["## Judge Evaluation\n"]
    for i, check in enumerate(judged, start=1):
        meta = check["metadata"]
        verdict = "✅ Passed" if check.get("passed") else "❌ Failed"
        lines.append(f"### Judge {i}: {verdict}\n")
        lines.append(f"**Judge model:** `{meta.get('judge_model', 'unavailable')}`\n")
        rubric = "\n> ".join(str(meta["rubric"]).splitlines())
        lines.append(f"**Rubric:**\n\n> {rubric}\n")
        # without a justification the judge never answered, so `details` holds the reason
        lines.append(f"**Justification:** {check.get('justification') or check.get('details', '')}\n")
    lines.append("\n---\n")
    return lines


def render_validation_lines(checks: list[dict]) -> list[str]:
    """One summary line per check; judge checks defer to the judge section for their rationale."""
    if not checks:
        return ["**Validation:** None\n"]

    lines = [f"**Validation:** {sum(bool(c.get('passed')) for c in checks)}/{len(checks)} checks passed\n"]
    judge_index = 0
    for check in checks:
        mark = "✅" if check.get("passed") else "❌"
        if "rubric" in (check.get("metadata") or {}):
            judge_index += 1
            lines.append(f"- {mark} LLM judge {judge_index} (see Judge Evaluation)")
        else:
            lines.append(f"- {mark} {check.get('details', '')}")
    lines.append("")
    return lines


def render_trace_markdown(data: dict[str, Any]) -> str:
    """
    Render a run's trace as markdown.

    Used both for the per-task trace reports and as the trace handed to the LLM judge, so the
    two always agree on what a trace looks like. When `check_results` is absent from `data`
    (which is the case while the run is still being verified) the validation line is omitted
    rather than printing a misleading "None".

    Args:
        data (dict): a task result dict (task_id, task, output, success, full_trace, ...)

    Returns (str): the markdown report
    """
    # Top summary section
    trace, tokens, duration = extract_concise_trace(data.get("full_trace", []))
    status = "Passed" if data.get("success") else "Failed"
    token_str = f"in {tokens['input_tokens']:,} | out {tokens['output_tokens']:,}"
    checks = data.get("check_results")
    validation_lines = []
    if checks is not None:
        validation_lines = render_validation_lines(checks)
    diagram_block = mermaid_schema(trace, status)
    legend = " | ".join(f"{icon} {name}" for name, icon in TOOL_ICONS.items())

    md_lines = [
        f"# Task Report #{data.get('task_id', 'N/A')}\n",
        f"**Task:** {data.get('task', '')}\n",
        f"**Final Output:** `{data.get('output', '')}`\n",
        f"**Status:** {status}&emsp;&emsp;&emsp;",
        f"**Duration:** {duration:.2f}&emsp;&emsp;&emsp;",
        f"**Tokens:** {token_str}\n",
        *validation_lines,
        "\n---\n",
        *render_judge_section(checks or []),
        "## Execution Trace\n",
        f"**Legend:** 💭 Thought | {legend}\n",
        diagram_block,
        "\n---\n",
    ]

    for i, step in enumerate(trace):
        if i > 0:
            md_lines.append("\n---")

        if "content" in step:
            content = str(step.get("content") or "").removeprefix("Thought:").strip()
            if not content:
                md_lines.append("\n> **Model Thought**\n> (No content)\n")
            else:
                body = re.sub(
                    r"<code>\s*([\s\S]*?)\s*</code>",
                    r"\n<details>\n<summary>Click to expand code</summary>\n\n```python\n\1\n```\n</details>\n",
                    content,
                )
                md_lines.append(f"\n> **Model Thought**\n{body}\n")

        elif "tool_name" in step:
            tool = step.get("tool_name", "Tool")
            inputs = json.dumps(step.get("inputs", {}), indent=2)
            result = step.get("result", "")
            output_section = f"**Output:** `{result}`\n" if tool == "calculator" else get_preview(tool, result)

            block = f"\n> **Tool Call:** `{tool}`\n\n```json\n{inputs}\n```\n\n{output_section}"
            md_lines.append(block)

    return "\n".join(md_lines)


def save_markdown_report(data: dict[str, Any], output_path: Path):
    """Write the markdown trace report for one run to `output_path`."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(render_trace_markdown(data))
