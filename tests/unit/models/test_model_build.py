import pytest
from smolagents import OpenAIServerModel

from core.config import ModelConfig
from core.events import EventWatcher
from models.base import build_model


def _build(monkeypatch, model_id: str, **kwargs):
    monkeypatch.setenv("TEST_MODEL_API_KEY", "dummy")
    conf = ModelConfig(
        provider="openai_server",
        model_id=model_id,
        api_key_env="TEST_MODEL_API_KEY",
        api_base="http://localhost:1",
        **kwargs,
    )
    return build_model(conf, EventWatcher(task_id=1, sink=lambda event: None))._wrapped


@pytest.mark.parametrize(
    ("model_id", "supports_stop", "expected"),
    [
        ("gpt-4o", None, True),  # unset: smolagents decides from the name
        ("gpt-5-mini", None, False),
        ("gpt-4o", False, False),  # a model smolagents doesn't know rejects stop
        ("gpt-5-mini", True, True),  # explicit override in the other direction
    ],
)
def test_supports_stop_override(monkeypatch, model_id, supports_stop, expected):
    model = _build(monkeypatch, model_id, supports_stop=supports_stop)

    assert model.supports_stop_parameter is expected
    assert isinstance(model, OpenAIServerModel)
    assert type(model).__name__ == OpenAIServerModel.__name__  # the override keeps the class name
