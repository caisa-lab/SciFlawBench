from smolagents import ChatMessage, MessageRole, Model

from core.events import EventWatcher
from models.base import WrappedModel
from models.context import (
    CONTEXT_TRIM_NOTE,
    MEMORY_INDEX_MARKER,
    compute_input_budget,
    message_text,
    resolve_context_window,
    trim_messages_to_budget,
)
from models.memory import MemoryOffloader
from tools.artifacts import ArtifactStore


class Msg:
    def __init__(self, role, text):
        self.role = role
        self.content = [{"type": "text", "text": text}]


def _count(text):
    return max(1, len(text) // 4)


def _total(messages):
    return sum(_count(message_text(m)) for m in messages)


def _conversation(steps=6, obs_chars=4000):
    messages = [Msg("system", "System prompt"), Msg("user", "New task: solve it")]
    for i in range(steps):
        messages.append(Msg("assistant", f"result = web_search(query='q{i}')"))
        messages.append(Msg("tool-response", f"Observation {i}:\n" + f"fact-{i} " * (obs_chars // 8)))
    return messages


def _offloader(tmp_path, **kwargs):
    store = ArtifactStore(tmp_path / "artifacts")
    store.set_scope("rep1")
    return MemoryOffloader(store, **kwargs), store


def test_evicted_observation_is_written_verbatim_and_replaced_by_pointer(tmp_path):
    offloader, store = _offloader(tmp_path)
    original = "Observation:\n" + "needle-in-haystack " * 400
    pointer = offloader.offload(original, fallback=CONTEXT_TRIM_NOTE, role="tool-response", label="web_search")

    assert "[block 1]" in pointer and "memory.md" in pointer and len(pointer) < len(original)
    assert original in store.read("memory.md", max_chars=0)["text"]


def test_short_text_is_not_offloaded(tmp_path):
    offloader, store = _offloader(tmp_path)
    assert offloader.offload("tiny", fallback="NOTE", role="tool-response") == "NOTE"
    assert not offloader.has_blocks


def test_same_eviction_reuses_its_block(tmp_path):
    offloader, _ = _offloader(tmp_path)
    text = "x" * 3000
    first = offloader.offload(text, fallback="NOTE")
    assert offloader.offload(text, fallback="NOTE") == first
    assert offloader.snapshot()["blocks"] == 1


def test_disabled_offloader_returns_the_fallback(tmp_path):
    offloader, _ = _offloader(tmp_path, enabled=False)
    assert offloader.offload("x" * 3000, fallback="NOTE") == "NOTE"
    assert MemoryOffloader(None).offload("x" * 3000, fallback="NOTE") == "NOTE"


def test_index_is_capped_and_keeps_recent_blocks(tmp_path):
    offloader, _ = _offloader(tmp_path, index_max_chars=600)
    for i in range(20):
        offloader.offload(f"block {i} " * 300, fallback="NOTE", label="web_search")
    index = offloader.index_text()
    assert len(index) <= 600 + 200  # the collapse line may overshoot slightly, never unboundedly
    assert "[block 20]" in index


def test_trimmer_offloads_instead_of_losing_the_text(tmp_path):
    offloader, store = _offloader(tmp_path)
    out = trim_messages_to_budget(_conversation(), 3000, _count, offloader=offloader)

    assert _total(out) <= 3000 + _count(MEMORY_INDEX_MARKER) + 400  # index sits in the task message
    assert offloader.has_blocks
    memory = store.read("memory.md", max_chars=0)["text"]
    assert "fact-0 " in memory  # the oldest observation survives on disk
    # the index lives in the protected task message, system prompt untouched
    assert MEMORY_INDEX_MARKER in message_text(out[1])
    assert message_text(out[0]) == "System prompt"


def test_index_injection_is_idempotent(tmp_path):
    offloader, _ = _offloader(tmp_path)
    messages = _conversation()
    trim_messages_to_budget(messages, 3000, _count, offloader=offloader)
    trim_messages_to_budget(messages, 3000, _count, offloader=offloader)
    assert message_text(messages[1]).count(MEMORY_INDEX_MARKER) == 1


def test_eviction_label_names_the_tool_behind_the_observation(tmp_path):
    offloader, store = _offloader(tmp_path)
    trim_messages_to_budget(_conversation(), 3000, _count, offloader=offloader)
    assert "web_search" in store.read("memory.md", max_chars=0)["text"]


def test_budget_subtracts_response_allowance_and_reserve():
    assert compute_input_budget(100_000, 16_384, 512) == 83_104
    assert compute_input_budget(None, 16_384) == 0
    assert compute_input_budget(100_000) == 100_000 - 512


def test_budget_keeps_half_the_window_when_the_response_swallows_it():
    assert compute_input_budget(8_000, 8_000, 512) == 4_000
    assert compute_input_budget(1_000, 5_000, 512) == 1_024


def test_explicit_window_always_wins():
    assert resolve_context_window("litellm", "anything", explicit=32_000) == 32_000


def test_unknown_model_falls_back_to_the_default_window():
    assert resolve_context_window("fake_model", "x") == 256_000
    assert resolve_context_window("litellm", "no-such-model-anywhere") == 256_000


def test_vllm_window_is_read_from_the_server(monkeypatch):
    monkeypatch.setattr("models.context._probe_server_window", lambda base, model, key: 40_960)
    assert resolve_context_window("openai_server", "m", api_base="http://localhost:8000/v1") == 40_960


class _Inner(Model):
    def __init__(self):
        super().__init__()
        self.model_id = "fake"
        self.seen = None

    def generate(self, messages, stop_sequences=None, response_format=None, tools_to_call_from=None, **kwargs):
        self.seen = messages
        return ChatMessage(role=MessageRole.ASSISTANT, content="ok")


def test_wrapped_model_offloads_and_reports_blocks(tmp_path):
    offloader, store = _offloader(tmp_path)
    events: list = []
    inner = _Inner()
    model = WrappedModel(
        inner,
        EventWatcher(task_id=1, sink=events.append),
        model_max_context=4000,
        reserve_tokens=0,
        max_new_tokens=1000,
        offloader=offloader,
    )
    model.generate(_conversation())

    payload = events[0].payload
    assert payload["context_budget"] == 3000
    assert payload["context_trimmed"] is True
    assert payload["memory_blocks"] >= 1
    assert (store.root / "rep1" / "memory.md").is_file()
    assert any(MEMORY_INDEX_MARKER in message_text(m) for m in inner.seen)


def test_build_model_caps_each_generation_step(monkeypatch):
    import models.base as base
    from core.config import ModelConfig

    captured = {}

    class _Recorder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("FAKE_KEY", "x")
    monkeypatch.setattr(base, "LiteLLMModel", _Recorder)
    conf = ModelConfig(
        provider="litellm", model_id="m", api_key_env="FAKE_KEY", max_new_tokens=2048, model_max_context=20_000
    )
    model = base.build_model(conf, EventWatcher(task_id=1, sink=lambda event: None))

    assert captured["max_tokens"] == 2048
    assert model._context_budget == 20_000 - 2048 - 512
