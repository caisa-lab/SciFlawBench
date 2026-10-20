from smolagents import ChatMessage, MessageRole, Model

from core.events import EventWatcher
from models.base import WrappedModel
from models.context import (
    CONTEXT_TRIM_NOTE,
    estimate_tokens,
    message_text,
    trim_messages_to_budget,
)


class Msg:
    def __init__(self, role, text):
        self.role = role
        self.content = [{"type": "text", "text": text}]


def _build_conversation(steps=6, obs_chars=4000):
    messages = [Msg("system", "System prompt"), Msg("user", "New task: solve it")]
    for i in range(steps):
        messages.append(Msg("assistant", f"thought {i} " + "A" * 200))
        messages.append(Msg("tool-response", "Observation:\n" + "O" * obs_chars))
    return messages


def _count(text):
    return max(1, len(text) // 4)


def _total(messages):
    return sum(_count(message_text(m)) for m in messages)


def test_estimate_tokens_uses_char_ratio():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100


def test_no_trimming_when_under_budget():
    messages = _build_conversation(steps=1, obs_chars=100)
    before = [message_text(m) for m in messages]
    out = trim_messages_to_budget(messages, 10**6, _count)
    assert [message_text(m) for m in out] == before


def test_trims_oldest_observations_first():
    messages = _build_conversation()
    out = trim_messages_to_budget(messages, 3000, _count)
    assert _total(out) <= 3000
    # System prompt and task are preserved verbatim.
    assert message_text(out[0]) == "System prompt"
    assert message_text(out[1]).startswith("New task")
    # The most recent observation is never truncated.
    assert "O" * 100 in message_text(out[-1])


def test_truncated_observation_carries_note():
    messages = _build_conversation()
    out = trim_messages_to_budget(messages, 3000, _count)
    assert any(CONTEXT_TRIM_NOTE in message_text(m) for m in out)


def test_handles_dict_messages():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "New task"},
        {"role": "assistant", "content": "a" * 100},
        {"role": "tool-response", "content": "Observation\n" + "o" * 8000},
        {"role": "assistant", "content": "b" * 100},
        {"role": "tool-response", "content": "Observation\n" + "o" * 8000},
    ]
    out = trim_messages_to_budget(messages, 1500, _count)
    assert _total(out) <= _total(messages)
    assert out[0]["content"] == "sys"


class _FakeInner(Model):
    def __init__(self):
        super().__init__()
        self.model_id = "fake"
        self.seen = None

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self.seen = messages
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")


def _wrapped(model_max_context=None, reserve_tokens=0):
    events: list = []
    inner = _FakeInner()
    model = WrappedModel(
        inner,
        EventWatcher(task_id=1, sink=events.append),
        model_max_context=model_max_context,
        reserve_tokens=reserve_tokens,
    )
    return model, inner, events


def test_wrapped_model_trims_over_budget_messages():
    model, inner, events = _wrapped(model_max_context=3000)
    model.generate(_build_conversation(steps=6, obs_chars=4000))
    assert _total(inner.seen) <= 3000
    start_payload = events[0].payload
    assert start_payload["context_trimmed"] is True
    assert start_payload["context_budget"] == 3000
    assert start_payload["prompt_tokens_estimate"] <= 3000


def test_wrapped_model_without_budget_leaves_messages_untouched():
    model, inner, events = _wrapped(model_max_context=None)
    messages = _build_conversation(steps=2, obs_chars=4000)
    model.generate(messages)
    assert len(inner.seen) == len(messages)
    assert events[0].payload["context_trimmed"] is False
    assert events[0].payload["context_budget"] is None


def test_wrapped_model_reserves_room_for_the_response():
    model, _, events = _wrapped(model_max_context=1000, reserve_tokens=200)
    model.generate(_build_conversation(steps=6, obs_chars=4000))
    assert events[0].payload["context_budget"] == 800
