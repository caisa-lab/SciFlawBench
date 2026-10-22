import dataclasses
from types import SimpleNamespace

from smolagents import ChatMessage, MessageRole

from core.config import DEFAULT_MAX_TRACE_TOKENS
from core.events import EventWatcher
from core.tasks import trace_token_budget
from evaluation.print_report import (
    EDGE_STEPS,
    MARKER_RESERVE,
    MIN_ITEM_CHARS,
    extract_concise_trace,
    fair_share_cap,
    mermaid_schema,
    render_trace_markdown,
    save_markdown_report,
    shorten,
)
from models.base import WrappedModel
from models.context import estimate_tokens

CODE_STEP = "Thought: compute it\n<code>\nprint(1)\n</code>"


def ev(kind, payload):
    return {"event_type": kind, "task_id": "demo", "payload": payload, "timestamp": 0.0}


def tool_call(name, result, **kwargs):
    return [
        ev("tool_call_start", {"name": name, "args": [], "kwargs": kwargs}),
        ev("tool_call_end", {"name": name, "result": result}),
    ]


def model_step(content, new_messages=(), trimmed=False):
    return [
        ev("model_call_start", {"name": "m", "new_messages": list(new_messages), "context_trimmed": trimmed}),
        ev("model_call_end", {"name": "m", "result": {"content": content}}),
    ]


def observation(text):
    return {"role": "tool-response", "content": [{"type": "text", "text": f"Observation:\n{text}"}]}


def run(events, output="final answer 0.863"):
    return {"task_id": "demo", "task": "the task", "output": output, "success": True, "full_trace": events}


# --- tool outputs reach the trace -----------------------------------------------------------------


def test_read_file_output_is_in_the_rendered_trace():
    # the reproduction from the issue
    events = tool_call("read_file", "data.csv: 2 column(s) x 3 row(s)\nx | y\n1 | 42\n2 | 43\n3 | 44", path="data.csv")

    md = render_trace_markdown(run(events))

    assert "42" in md


def test_every_tool_output_is_rendered_not_only_search_and_webpages():
    events = tool_call("numerics", '{"solution": [0.8, 1.4]}') + tool_call("current_time", "2026-10-22 12:00")

    md = render_trace_markdown(run(events), max_tokens=DEFAULT_MAX_TRACE_TOKENS)

    assert '{"solution": [0.8, 1.4]}' in md
    assert "2026-10-22 12:00" in md


def test_search_and_webpage_keep_their_preview_and_also_show_the_full_output():
    page = "\n".join(f"line {i} of the page" for i in range(10))
    events = tool_call("visit_webpage", page, url="https://example.org") + tool_call(
        "web_search", "[Result](https://example.org/paper) snippet", query="q"
    )

    md = render_trace_markdown(run(events))

    assert "**Snippet**" in md and "**URLs Found:**" in md
    assert "line 9 of the page" in md  # beyond the five-line snippet


def test_output_containing_backticks_cannot_close_its_block_early():
    events = tool_call("read_file", "```\n# not a heading\n```")

    md = render_trace_markdown(run(events))

    assert "````text\n```\n# not a heading\n```\n````" in md


# --- the budget -----------------------------------------------------------------------------------


def test_trace_below_the_budget_is_not_shortened():
    events = tool_call("read_file", "a" * 300, path="small") + tool_call("read_file", "b" * 5000, path="large")

    md = render_trace_markdown(run(events), max_tokens=DEFAULT_MAX_TRACE_TOKENS)

    assert "a" * 300 in md and "b" * 5000 in md
    assert "omitted" not in md


def test_oversized_trace_stays_within_the_budget_and_keeps_the_final_output():
    events = (
        tool_call("read_file", "s" * 300, path="small")
        + tool_call("read_file", "g" * 2_500_000, path="giant")
        + tool_call("read_file", "m" * 40_000, path="medium")
    )

    md = render_trace_markdown(run(events), max_tokens=20_000)

    assert estimate_tokens(md) <= 20_000
    assert "final answer 0.863" in md
    assert "characters omitted" in md


def test_only_the_largest_outputs_are_shortened():
    events = (
        tool_call("read_file", "s" * 300, path="small")
        + tool_call("read_file", "t" * 3000, path="medium")
        + tool_call("read_file", "g" * 2_500_000, path="giant")
    )

    md = render_trace_markdown(run(events), max_tokens=20_000)

    assert "s" * 300 in md and "t" * 3000 in md  # small items are kept whole
    assert "g" * 100_000 not in md
    assert md.count("characters omitted") == 1


def test_shortened_output_keeps_its_head_and_its_tail():
    text = "HEAD-OF-OUTPUT " + "x" * 500_000 + " RESULT-AT-THE-END"
    events = tool_call("read_file", text, path="big")

    md = render_trace_markdown(run(events), max_tokens=10_000)

    assert "HEAD-OF-OUTPUT" in md and "RESULT-AT-THE-END" in md
    assert "characters omitted" in md


def test_without_a_budget_nothing_is_shortened():
    events = tool_call("read_file", "g" * 2_000_000, path="giant")

    md = render_trace_markdown(run(events))

    assert "g" * 2_000_000 in md


def test_very_long_runs_drop_the_oldest_middle_steps():
    events = [e for i in range(2000) for e in tool_call("read_file", f"output {i} " + "z" * 5000, path=f"file-{i}")]

    md = render_trace_markdown(run(events), max_tokens=100_000)

    assert estimate_tokens(md) <= 100_000
    assert "steps omitted to fit the trace budget" in md
    for i in range(EDGE_STEPS):
        assert f'"path": "file-{i}"' in md  # the first steps are kept
    assert '"path": "file-1999"' in md  # and so is the last one
    assert '"path": "file-1000"' not in md
    assert "final answer 0.863" in md


def test_shortened_items_keep_at_least_the_floor():
    events = [e for i in range(200) for e in tool_call("read_file", f"output {i} " + "z" * 5000, path=f"file-{i}")]

    md = render_trace_markdown(run(events), max_tokens=60_000)

    # every kept output shows at least MIN_ITEM_CHARS characters of its own text
    assert md.count("z" * int(MIN_ITEM_CHARS * 0.7 - 20)) >= EDGE_STEPS


def test_shorten_keeps_seventy_percent_head_and_thirty_percent_tail():
    text = "".join(chr(65 + i % 26) for i in range(10_000))

    out = shorten(text, 1000)

    assert out.startswith(text[:700]) and out.endswith(text[-300:])
    assert "[... 9,000 characters omitted ...]" in out
    assert len(out) <= 1000 + MARKER_RESERVE
    assert shorten("short", 1000) == "short" and shorten("short", None) == "short"


def test_fair_share_cap():
    assert fair_share_cap([100, 200, 300], room=1000) is None  # everything fits
    cap = fair_share_cap([100, 200, 10_000], room=1000)
    assert cap == 1000 - 100 - 200 - MARKER_RESERVE  # only the giant is cut
    two = fair_share_cap([5000, 10_000], room=2000)
    assert two == 2000 // 2 - MARKER_RESERVE  # both are cut to the same cap


def test_save_markdown_report_uses_the_same_budget(tmp_path):
    data = run(tool_call("read_file", "g" * 1_000_000, path="giant"))
    path = tmp_path / "report.md"

    save_markdown_report(data, path, max_tokens=5_000)

    assert path.read_text(encoding="utf-8") == render_trace_markdown(data, max_tokens=5_000)


def test_reports_and_the_judge_share_the_judges_budget():
    assert trace_token_budget(SimpleNamespace(judge=SimpleNamespace(max_trace_tokens=1234))) == 1234
    assert trace_token_budget(SimpleNamespace(judge=None)) == DEFAULT_MAX_TRACE_TOKENS


# --- execution logs ---------------------------------------------------------------------------------


def test_execution_logs_are_attached_to_their_code_step():
    events = (
        model_step("Thought: first\n<code>\nprint('a')\n</code>", [{"role": "user", "content": "task"}])
        + model_step("Thought: second\n<code>\nprint('b')\n</code>", [observation("Execution logs:\nFIRST-LOGS")])
        + model_step("Thought: done\n<code>\nfinal_answer(1)\n</code>", [observation("Execution logs:\nSECOND-LOGS")])
    )

    md = render_trace_markdown(run(events))

    assert md.count("FIRST-LOGS") == 1 and md.count("SECOND-LOGS") == 1
    assert md.index("print('a')") < md.index("FIRST-LOGS") < md.index("print('b')") < md.index("SECOND-LOGS")


def test_observations_of_tool_calling_steps_are_not_duplicated():
    # a step without code is a tool-calling step: its tool outputs are already rendered from the tool events
    events = (
        model_step("calling web_search")
        + tool_call("web_search", "TOOL-OUTPUT", query="q")
        + model_step("done", [observation("TOOL-OUTPUT")])
    )

    md = render_trace_markdown(run(events))

    assert md.count("TOOL-OUTPUT") == 1


def test_a_log_dropped_by_trimming_is_flagged_not_replaced():
    events = model_step(CODE_STEP) + model_step(CODE_STEP, [{"role": "assistant", "content": "x"}], trimmed=True)

    trace, _, _ = extract_concise_trace(events)

    assert trace[0].get("observation_missing") is True
    assert "Execution logs not recorded" in render_trace_markdown(run(events))


class _CodeModel:
    """Answers every call with a numbered code step."""

    def __init__(self):
        self.calls = 0

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self.calls += 1
        return ChatMessage(
            role=MessageRole.ASSISTANT, content=f"Thought: step {self.calls}\n<code>\nprint({self.calls})\n</code>"
        )


def _conversation(steps):
    """A fresh message list for `steps` finished code steps, as smolagents rebuilds it before every call."""
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="system prompt"),
        ChatMessage(role=MessageRole.USER, content="New task: solve it"),
    ]
    for i in range(1, steps + 1):
        messages.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=f"Thought: step {i}\n<code>\nprint({i})\n</code>")
        )
        messages.append(ChatMessage(role=MessageRole.TOOL_CALL, content="Calling tools: python_interpreter"))
        messages.append(
            ChatMessage(role=MessageRole.TOOL_RESPONSE, content=f"Observation:\nExecution logs:\nLOG-{i} " + "o" * 2000)
        )
    return messages


def _logged_run(model_max_context):
    events = []
    model = WrappedModel(_CodeModel(), EventWatcher(task_id=1, sink=events.append), model_max_context=model_max_context)
    for steps in range(8):
        model.generate(_conversation(steps))
    return [dataclasses.asdict(event) for event in events]


def test_untrimmed_run_shows_every_execution_log_once():
    md = render_trace_markdown(run(_logged_run(model_max_context=None)))

    for i in range(1, 8):
        assert md.count(f"LOG-{i} ") == 1


def test_trimmed_run_shows_every_execution_log_once():
    # with a tiny window the trimmer drops whole messages, so the conversation stops growing between calls
    events = _logged_run(model_max_context=1500)
    starts = [e["payload"] for e in events if e["event_type"] == "model_call_start"]
    assert sum(p["context_trimmed"] for p in starts) >= 5
    assert starts[-1]["message_count"] == starts[-2]["message_count"]

    trace, _, _ = extract_concise_trace(events)
    md = render_trace_markdown(run(events))

    code_steps = [step for step in trace if "content" in step][:-1]  # the last step has no following call
    for i, step in enumerate(code_steps, start=1):
        assert step["observation"].startswith(f"Execution logs:\nLOG-{i} ")
        assert md.count(f"LOG-{i} ") == 1


# --- the failure marker -------------------------------------------------------------------------------


def test_a_successful_read_of_a_file_mentioning_error_is_not_marked_failed():
    events = tool_call("read_file", "kappa_err   1-sigma error on kappa", path="voids.py")

    assert "❌" not in mermaid_schema(extract_concise_trace(events)[0], "Passed")


def test_an_error_message_from_a_tool_is_marked_failed():
    events = tool_call("numerics", "Error: matrix must be square, got 2x3")

    assert "❌" in mermaid_schema(extract_concise_trace(events)[0], "Passed")


def test_a_tool_that_raised_is_kept_and_marked_failed():
    events = [
        ev("tool_call_start", {"name": "visit_webpage", "args": [], "kwargs": {"url": "https://x"}}),
        ev("error", {"name": "visit_webpage", "error": "Traceback ... ConnectionError: refused"}),
    ]

    trace, _, _ = extract_concise_trace(events)
    md = render_trace_markdown(run(events))

    assert trace[0]["error"].endswith("ConnectionError: refused")
    assert "Tool call failed (visit_webpage)" in md and "ConnectionError: refused" in md
    assert "❌" in md


def test_the_judge_is_given_the_trace_rendered_within_its_budget(monkeypatch):
    from core import tasks
    from core.events import AgentEvent, EventType
    from core.task_ids import hash_prompt

    seen = {}

    def fake_render(data, max_tokens=None):
        seen["max_tokens"] = max_tokens
        return "rendered"

    monkeypatch.setattr(tasks, "render_trace_markdown", fake_render)
    task = tasks.TaskDef(
        task_id=hash_prompt("p"),
        task="p",
        agent_id="code_agent",
        failure_modes={"qualitative": ["reasoning"]},
        validators=[{"name": "judge:rubric", "kwargs": {"rubric": "r"}}],
    )
    run_config = SimpleNamespace(judge=SimpleNamespace(max_trace_tokens=4321, model_id="j"))
    events = [AgentEvent(event_type=EventType.AgentStart, task_id="demo", payload={}, timestamp=0.0)]

    context = tasks.build_verifier_context(task, "p", "answer", run_config, events)

    assert context.trace == "rendered"
    assert seen["max_tokens"] == 4321
