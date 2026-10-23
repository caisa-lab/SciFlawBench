import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from models.context import CHARS_PER_TOKEN, message_role, message_text

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

#: A shortened tool output or execution log keeps at least this many characters.
MIN_ITEM_CHARS = 500
#: Share of a shortened item taken from its start; the rest comes from its end, where results and errors usually are.
HEAD_SHARE = 0.7
#: Characters set aside per shortened item for its omission marker.
MARKER_RESERVE = 64
#: Steps always kept at each end of a run whose middle has to be dropped to fit the budget.
EDGE_STEPS = 3

# A step whose model output contains code ran in the code executor, so its observation holds execution logs.
_CODE_PATTERN = re.compile(r"<code>|```")


def extract_concise_trace(events: list[dict]) -> tuple:
    """
    Extracts a concise trace from the list of AgentEvent objects, summarizing tool and model calls, duraion, and tokens.

    A tool call that raised is kept with its `error`. The observation that follows a code step (the code agent's
    execution logs) is read from the next model call's logged messages and attached to that step as `observation`;
    if context trimming dropped it from those messages, the step is flagged with `observation_missing` instead.

    Args:
        events: List[Dict]: list of dumped agent events

    Returns: A tuple containing the concise trace, token counts, and duration of the events.
    """

    trace = []
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    pending_tool = None
    last_model_step = None

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
        elif tool == "error" and pending_tool:
            pending_tool["error"] = payload.get("error")
            trace.append(pending_tool)
            pending_tool = None
        elif tool == "model_call_start" and last_model_step is not None:
            _attach_observation(last_model_step, payload)
        elif tool == "model_call_end":
            res = payload.get("result", {}) or {}
            content = res.get("content", "") if isinstance(res, dict) else str(res)
            last_model_step = {"event_type": tool, "content": content}
            trace.append(last_model_step)

            usage = res.get("token_usage", {}) or (res.get("raw") or {}).get("usage") or {}
            tokens["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
            tokens["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0

    if pending_tool:
        trace.append(pending_tool)

    return trace, tokens, duration


def _attach_observation(step: dict, payload: dict) -> None:
    """
    Attach the execution logs of a code step, read from the next model call's newly logged messages.

    Only the last tool response among those messages is used, so a log is never attached twice. Context trimming
    can drop it from the logged messages; the step is then flagged rather than given a wrong or older log.
    """
    if "observation" in step or not _CODE_PATTERN.search(str(step.get("content") or "")):
        return
    responses = [m for m in payload.get("new_messages") or [] if message_role(m) in ("tool-response", "tool")]
    if responses:
        step["observation"] = message_text(responses[-1]).removeprefix("Observation:").strip()
    elif payload.get("context_trimmed"):
        step["observation_missing"] = True


def tool_failed(step: dict) -> bool:
    """A tool call failed if it raised, or if it returned an `Error...` message instead of a result."""
    return "error" in step or str(step.get("result") or "").lstrip().startswith("Error")


def mermaid_schema(trace, status):
    steps = []
    for x in trace:
        has_error = False
        if x.get("content"):  # omit empty thoughts
            steps.append(("💭 Thought", None, has_error))
        elif "tool_name" in x:
            name = str(x["tool_name"]).replace('"', "'")
            label = f"{TOOL_ICONS[name]} {name}" if name in TOOL_ICONS else name
            inp_key = json.dumps(x.get("inputs"), sort_keys=True, default=str)
            has_error = tool_failed(x)
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
    """A short summary shown above a search or webpage output: the URLs found, or the first lines of the page."""
    preview = ""
    if "search" in tool:
        pattern = r"\[.*?\]\(https?://(?:www\.)?([^\)]+)\)"
        urls = [f"{u.rstrip('/')}" for u in re.findall(pattern, str(text or ""))]
        preview = "**URLs Found:**\n" + ", ".join(urls) + "\n\n" if urls else ""
    if "visit" in tool and text:
        first_lines = "\n> ".join([line.strip()[:200] for line in str(text).split("\n") if line.strip()][:5])
        preview = f"**Snippet**\n> {first_lines}...\n\n"
    return preview


def shorten(text: str, cap: int | None) -> str:
    """Cut `text` to about `cap` characters, keeping its head and tail around an omission marker."""
    if cap is None or len(text) <= cap:
        return text
    head = int(cap * HEAD_SHARE)
    tail = cap - head
    return f"{text[:head]}\n[... {len(text) - cap:,} characters omitted ...]\n{text[len(text) - tail :]}"


def _fence(text: str) -> str:
    """A code fence longer than any run of backticks in `text`, so the text cannot close it early."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _details_block(summary: str, shown: str, source: str) -> str:
    """A collapsible text block with `shown` (possibly shortened); the fence is sized from the full `source`."""
    fence = _fence(source)
    return f"<details>\n<summary>{summary}</summary>\n\n{fence}text\n{shown}\n{fence}\n\n</details>\n"


def _step_items(step: dict) -> list[str]:
    """The large, shortenable parts of a step: a tool's output (or error) and a code step's execution logs."""
    if "tool_name" in step:
        value = step.get("error") if "error" in step else step.get("result")
        return ["" if value is None else str(value)]
    if "observation" in step:
        return [str(step["observation"])]
    return []


def render_step(step: dict, cut: Callable[[str], str]) -> str:
    """Markdown for one trace step; `cut` shortens its large parts (tool output, execution logs)."""
    if "tool_name" in step:
        tool = step.get("tool_name") or "Tool"
        inputs = json.dumps(step.get("inputs", {}), indent=2, default=str)
        (text,) = _step_items(step)
        shown = cut(text)
        if "error" in step:
            output_section = _details_block(f"Tool call failed ({tool})", shown, text)
        elif tool == "calculator":
            output_section = f"**Output:** `{shown}`\n"
        else:
            output_section = get_preview(tool, text) + _details_block(f"Click to expand output ({tool})", shown, text)
        return f"\n> **Tool Call:** `{tool}`\n\n```json\n{inputs}\n```\n\n{output_section}"

    content = str(step.get("content") or "").removeprefix("Thought:").strip()
    if not content:
        return "\n> **Model Thought**\n> (No content)\n"
    body = re.sub(
        r"<code>\s*([\s\S]*?)\s*</code>",
        r"\n<details>\n<summary>Click to expand code</summary>\n\n```python\n\1\n```\n</details>\n",
        content,
    )
    block = f"\n> **Model Thought**\n{body}\n"
    if "observation" in step:
        logs = str(step["observation"])
        block += "\n" + _details_block("Click to expand execution logs", cut(logs), logs)
    elif step.get("observation_missing"):
        block += "\n*Execution logs not recorded: context trimming dropped them from the logged messages.*\n"
    return block


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


def _render_header(data: dict[str, Any], trace: list[dict], tokens: dict, duration: float) -> str:
    """Everything above the steps: task, final output, status, checks, judge section and the flowchart."""
    status = "Passed" if data.get("success") else "Failed"
    token_str = f"in {tokens['input_tokens']:,} | out {tokens['output_tokens']:,}"
    checks = data.get("check_results")
    validation_lines = []
    if checks is not None:
        validation_lines = render_validation_lines(checks)
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
        mermaid_schema(trace, status),
        "\n---\n",
    ]
    return "\n".join(md_lines)


def _assemble(header: str, trace: list[dict], cut: Callable[[str], str], dropped: int = 0) -> str:
    """The full markdown; with `dropped` > 0 the steps after the first EDGE_STEPS are preceded by a gap marker."""
    md_lines = [header]
    for i, step in enumerate(trace):
        if i > 0:
            md_lines.append("\n---")
        if dropped and i == EDGE_STEPS:
            md_lines.append(f"\n> [{dropped:,} steps omitted to fit the trace budget]\n\n---")
        md_lines.append(render_step(step, cut))
    return "\n".join(md_lines)


def fair_share_cap(lengths: list[int], room: int, reserve: int = MARKER_RESERVE) -> int | None:
    """
    The largest per-item cap `C` with `sum(min(l, C) + reserve for each l > C) <= room`.

    Returns None when every item fits whole (no cap needed); otherwise C, which may be 0 or negative when even
    empty items do not fit. Items shorter than C are kept whole, so only the largest ones are shortened.
    """
    if sum(lengths) <= room:
        return None
    ordered = sorted(lengths)
    whole = 0  # total length of the items kept whole
    for k, length in enumerate(ordered):
        shortened = len(ordered) - k  # this item and every longer one are shortened
        cap = (room - whole) // shortened - reserve
        if cap < length:
            return cap
        whole += length
    return None  # unreachable: the items would all fit


def _layout(header: str, trace: list[dict], budget_chars: int, dropped: int) -> tuple[list[dict], int | None]:
    """The steps kept when `dropped` middle steps are left out, and the item cap that then fits the budget."""
    kept = trace if not dropped else trace[:EDGE_STEPS] + trace[EDGE_STEPS + dropped :]
    fixed = len(_assemble(header, kept, lambda text: "", dropped))
    lengths = [len(item) for step in kept for item in _step_items(step)]
    return kept, fair_share_cap(lengths, budget_chars - fixed)


def render_trace_markdown(data: dict[str, Any], max_tokens: int | None = None) -> str:
    """
    Render a run's trace as markdown.

    Used both for the per-task trace reports and as the trace handed to the LLM judge, so the
    two always agree on what a trace looks like. When `check_results` is absent from `data`
    (which is the case while the run is still being verified) the validation line is omitted
    rather than printing a misleading "None".

    With `max_tokens`, the rendering is kept within that many estimated tokens (CHARS_PER_TOKEN characters
    each). The task, final output, status, checks, model thoughts with their code and tool inputs are always
    kept. Tool outputs and execution logs share the remaining room: while everything fits nothing is shortened;
    otherwise the largest items are cut to a common cap, keeping their head and tail around an
    `[... N characters omitted ...]` marker. If even a cap of MIN_ITEM_CHARS does not fit, the oldest steps after
    the first EDGE_STEPS are dropped behind an `[N steps omitted ...]` marker.

    Args:
        data (dict): a task result dict (task_id, task, output, success, full_trace, ...)
        max_tokens (int | None): token budget for the whole rendering; None renders everything in full

    Returns (str): the markdown report
    """
    trace, tokens, duration = extract_concise_trace(data.get("full_trace", []))
    header = _render_header(data, trace, tokens, duration)
    if max_tokens is None:
        return _assemble(header, trace, lambda text: text)

    budget_chars = max_tokens * CHARS_PER_TOKEN
    dropped = 0
    kept, cap = _layout(header, trace, budget_chars, dropped)
    droppable = len(trace) - 2 * EDGE_STEPS
    if cap is not None and cap < MIN_ITEM_CHARS and droppable > 0:

        def fits(steps_dropped: int) -> bool:
            steps_cap = _layout(header, trace, budget_chars, steps_dropped)[1]
            return steps_cap is None or steps_cap >= MIN_ITEM_CHARS

        # drop the fewest oldest middle steps that let every remaining item keep at least MIN_ITEM_CHARS
        low, high = 1, droppable
        while low < high:
            middle = (low + high) // 2
            if fits(middle):
                high = middle
            else:
                low = middle + 1
        dropped = low
        kept, cap = _layout(header, trace, budget_chars, dropped)

    final_cap = None if cap is None else max(cap, MIN_ITEM_CHARS)
    return _assemble(header, kept, lambda text: shorten(text, final_cap), dropped)


def save_markdown_report(data: dict[str, Any], output_path: Path, max_tokens: int | None = None):
    """Write the markdown trace report for one run to `output_path`, within `max_tokens` when given."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(render_trace_markdown(data, max_tokens=max_tokens))
