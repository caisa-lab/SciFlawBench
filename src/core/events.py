import copy
import dataclasses
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal


class EventType(StrEnum):
    """
    Enum used to track the events we're interested in (listed down below)
    """

    ToolCallStart = "tool_call_start"
    ToolCallEnd = "tool_call_end"
    ModelCallStart = "model_call_start"
    ModelCallEnd = "model_call_end"
    AgentStart = "agent_call_start"
    AgentEnd = "agent_call_end"
    Errored = "error"


@dataclass
class AgentEvent:
    """
    dataclass containing the full information captured by our agent events listed above.
    """

    event_type: EventType
    task_id: int | str
    payload: dict[str, Any]
    timestamp: float


def event_record(event: AgentEvent) -> dict[str, Any]:
    """The event as a plain dict, like `dataclasses.asdict`, with a stand-in for any value that can't be copied.

    Payloads hold the raw arguments and results of the agent's calls. In a code agent these come from the agent's own
    Python code and can be anything, such as a module or an open file, which `asdict` can't copy (issue #68). Such a
    value is kept as its `repr`, so the run's record is still written.
    """
    try:
        return dataclasses.asdict(event)
    except Exception:
        return {field.name: _copyable(getattr(event, field.name)) for field in dataclasses.fields(event)}


def _copyable(value: Any) -> Any:
    """`value` copied the way `dataclasses.asdict` copies it, with the `repr` of any part that can't be copied."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _copyable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {_copyable(key): _copyable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copyable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copyable(item) for item in value)
    try:
        return copy.deepcopy(value)
    except Exception:
        return repr(value)


class EventWatcher:
    """
    A watcher class which wraps around all agent, tool, and model calls which will basically just store a bunch of agent
    events into the sink (function) its given
    """

    def __init__(self, task_id: int | str, sink: Callable):
        self.task_id = task_id
        self._sink = sink

    def _emit(self, event_type: EventType, payload: dict[str, Any]):
        """
        Hidden method used in the call which makes the dataclass that gets stored in the sink and actually pushes it
        through to the sink

        Args:
            event_type (EventType): enum value specifying what kindo of event just took place
            payload (Dict[str, Any]): the actually interesting data associated with this event
        """
        ev = AgentEvent(event_type=event_type, task_id=self.task_id, payload=payload, timestamp=time.time())
        self._sink(ev)

    def __call__(self, kind: Literal["agent", "model", "tool"], name: str, fn: Callable, *args, **kwargs) -> Any:
        """
        the wrapper endpoint through which you can log one of these events, basically allows a passthrough of any of the
        event functions we are interested in and stores all the information we need to know about how it was used in the
        sink via the _emit call

        Args:
            kind (Literal): one of the listed literals representing the different kinds of things were interested in
            name (str): name of the tool, model, agent etc. were interested in calling
            fn (Callable): the function to be run
            args: args to be passed through to the run function
            kwargs: kwargs to be passed through to the run function

        Returns (Any): result of the wrapped function
        """

        start_payload = kwargs.pop("start_payload", None)
        start_load = start_payload if start_payload is not None else {"args": args, "kwargs": kwargs}
        self._emit(event_type=EventType(f"{kind}_call_start"), payload={"name": name, **start_load})
        try:
            result = fn(*args, **kwargs)
            self._emit(event_type=EventType(f"{kind}_call_end"), payload={"name": name, "result": result})
            return result
        except Exception:
            self._emit(event_type=EventType("error"), payload={"name": name, "error": traceback.format_exc()})
            raise


def failed_tool_calls(events: list[AgentEvent]) -> dict[str, int]:
    """How many calls of each tool raised an error during one run: {tool name: failed calls}, tools that never failed
    left out.

    A tool's error is told apart from the model's or the agent's by its name, which is the name of a tool that was
    called. A tool that reports a problem in its answer instead of raising (e.g. "Error searching for Arxiv papers")
    is not counted.
    """
    tools = {event.payload.get("name") for event in events if event.event_type == EventType.ToolCallStart}
    failed = Counter(
        event.payload.get("name")
        for event in events
        if event.event_type == EventType.Errored and event.payload.get("name") in tools
    )
    return dict(sorted(failed.items()))
