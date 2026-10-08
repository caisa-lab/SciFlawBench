import json
import re
from pathlib import Path
from typing import Any

TOOL_ICONS = {
    "web_search": "🔍",
    "wiki_search": "📚",
    "calculator": "🧮",
    "visit_webpage": "🌐",
    "json_answer": "📦",
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
            steps.append(("💭", None, has_error))
        elif "tool_name" in x:
            label = TOOL_ICONS.get(x["tool_name"], x["tool_name"])
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

    # Build diagram
    nodes = ["Start"]
    for label, _, has_error, count in merged:
        suffix = f" ×{count}" if count > 1 else ""
        error = "❌ " if has_error else ""
        nodes.append(f"{label}{suffix}{error}")
    nodes.append(f"{status}")

    flow = " -> ".join(nodes)
    return f"```mermaid\n    {flow}\n```"


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
        validation_str = checks[0].get("details", "None") if checks else "None"
        validation_lines = [f"**Validation:** {validation_str}\n"]
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

            block = f"\n> **Tool Call:** `{tool}`\n\n>`{inputs}`\n\n{output_section}"
            md_lines.append(block)

    return "\n".join(md_lines)


def save_markdown_report(data: dict[str, Any], output_path: Path):
    """Write the markdown trace report for one run to `output_path`."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(render_trace_markdown(data))
