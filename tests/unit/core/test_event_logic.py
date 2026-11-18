import dataclasses
import json
import threading

import pytest
from smolagents import Model, Tool
from smolagents.models import ChatMessage, MessageRole

from core.events import AgentEvent, EventType, EventWatcher, event_record, failed_tool_calls
from models.base import WrappedModel
from tools.base import WrappedTool


def test_emit_calls_sink_with_event():
    captured: list[AgentEvent] = []
    watcher = EventWatcher(task_id=1, sink=captured.append)

    watcher._emit(EventType("tool_call_start"), {"name": "web_search"})

    assert len(captured) == 1
    assert captured[0].task_id == 1
    assert captured[0].payload["name"] == "web_search"


def test_watcher_call_with_fake_tool():

    class FakeSearchTool(Tool):
        name = "fake_search"
        description = "a fake search tool for testing"
        inputs = {"query": {"type": "string", "description": "search query"}}
        output_type = "string"

        def forward(self, query: str) -> str:
            return f"results for: {query}"

    captured: list[AgentEvent] = []
    watcher = EventWatcher(task_id=1, sink=captured.append)
    test_tool = WrappedTool(wrapped_tool=FakeSearchTool(), watcher=watcher)

    result = test_tool.forward(query="python testing")

    assert result == "results for: python testing"
    event_types = [e.event_type for e in captured]
    assert event_types == [EventType.ToolCallStart, EventType.ToolCallEnd]
    assert captured[0].payload["kwargs"] == {"query": "python testing"}
    assert captured[1].payload["result"] == "results for: python testing"


def test_watcher_call_with_fake_model():

    class FakeModel(Model):
        def generate(
            self,
            messages,
            stop_sequences=None,
            response_format=None,
            tools_to_call_from=None,
            **kwargs,
        ) -> ChatMessage:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content="fake response",
            )

    captured: list[AgentEvent] = []
    watcher = EventWatcher(task_id=1, sink=captured.append)
    test_model = WrappedModel(wrapped_model=FakeModel(), watcher=watcher)

    result = test_model.generate(messages="python testing")

    assert isinstance(result, ChatMessage)
    event_types = [e.event_type for e in captured]
    assert event_types == [EventType.ModelCallStart, EventType.ModelCallEnd]
    print(captured)
    assert captured[0].payload["new_messages"] == "python testing"
    assert captured[1].payload["result"].role == MessageRole.ASSISTANT
    assert captured[1].payload["result"].content == "fake response"


def test_watcher_records_error_and_reraises():
    captured: list[AgentEvent] = []
    watcher = EventWatcher(task_id=1, sink=captured.append)

    class BrokenTool(Tool):
        name = "broken"
        description = "always fails"
        inputs = {}
        output_type = "string"

        def forward(self):
            raise ValueError("simulated failure")

    test_tool = WrappedTool(wrapped_tool=BrokenTool(), watcher=watcher)

    with pytest.raises(ValueError, match="simulated failure"):
        test_tool.forward()

    assert captured[-1].event_type == EventType.Errored
    assert "simulated failure" in captured[-1].payload["error"]


def _event(kind, name):
    return AgentEvent(event_type=EventType(kind), task_id=1, payload={"name": name}, timestamp=0.0)


def test_failed_tool_calls_counts_the_errors_of_each_tool():
    events = [
        _event("model_call_start", "fake-model"),
        _event("tool_call_start", "web_search"),
        _event("error", "web_search"),
        _event("tool_call_start", "web_search"),
        _event("tool_call_end", "web_search"),
        _event("tool_call_start", "visit_webpage"),
        _event("error", "visit_webpage"),
        _event("tool_call_start", "web_search"),
        _event("error", "web_search"),
        _event("error", "fake-model"),  # the model's error, not a tool's
    ]

    assert failed_tool_calls(events) == {"visit_webpage": 1, "web_search": 2}


def test_failed_tool_calls_is_empty_when_every_tool_call_worked():
    events = [_event("tool_call_start", "web_search"), _event("tool_call_end", "web_search")]
    assert failed_tool_calls(events) == {}


def test_a_tool_that_raises_is_counted_through_the_watcher():
    class BrokenSearchTool(Tool):
        name = "broken_search"
        description = "a search tool that always fails"
        inputs = {"query": {"type": "string", "description": "search query"}}
        output_type = "string"

        def forward(self, query: str) -> str:
            raise RuntimeError("No results found.")

    captured: list[AgentEvent] = []
    tool = WrappedTool(wrapped_tool=BrokenSearchTool(), watcher=EventWatcher(task_id=1, sink=captured.append))

    with pytest.raises(RuntimeError):
        tool.forward(query="GW190521")

    assert failed_tool_calls(captured) == {"broken_search": 1}


def test_event_record_is_asdict_for_an_ordinary_event():
    event = AgentEvent(EventType.ToolCallEnd, "SFB-0001", {"name": "web_search", "result": ["a", ("b", 1)]}, 1.0)
    assert event_record(event) == dataclasses.asdict(event)


def test_event_record_keeps_a_stand_in_for_a_value_that_cannot_be_copied():
    """A code agent can pass anything to a tool, e.g. a module; the record must still be written (issue #68)."""
    events: list[AgentEvent] = []
    watcher = EventWatcher(task_id="SFB-0000", sink=events.append)
    lock = threading.Lock()
    watcher("tool", "json_answer_tool", lambda **kwargs: "ok", answer=json, extra={"lock": lock, "n": [1, 2]})

    with pytest.raises(TypeError):
        dataclasses.asdict(events[0])  # what crashed the run before
    record = event_record(events[0])
    kwargs = record["payload"]["kwargs"]
    assert kwargs["answer"] == repr(json)
    assert kwargs["extra"] == {"lock": repr(lock), "n": [1, 2]}
    assert record["payload"]["name"] == "json_answer_tool"
    json.dumps(record)  # the record can be written
