import pytest
from smolagents import OpenAIServerModel

from core.config import MAX_TOKENS_PARAMS, ModelConfig
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


@pytest.mark.parametrize(
    ("overrides", "sent"),
    [
        ({}, {"max_tokens": 16_384}),  # default: unchanged
        ({"max_tokens_param": "max_completion_tokens"}, {"max_completion_tokens": 16_384}),
        ({"max_new_tokens": 2048, "max_tokens_param": "max_completion_tokens"}, {"max_completion_tokens": 2048}),
        ({"max_new_tokens": None}, {}),  # no cap: neither name is sent
        ({"max_new_tokens": None, "max_tokens_param": "max_completion_tokens"}, {}),
        ({"extra_kwargs": {"max_completion_tokens": 4096}}, {"max_completion_tokens": 4096}),  # legacy key adopted
    ],
)
def test_cap_is_sent_under_max_tokens_param(monkeypatch, overrides, sent):
    model = _build(monkeypatch, "gpt-6-astra", **overrides)

    assert {key: value for key, value in model.kwargs.items() if key in MAX_TOKENS_PARAMS} == sent


def test_input_budget_reserve_does_not_depend_on_the_param_name(monkeypatch):
    monkeypatch.setenv("TEST_MODEL_API_KEY", "dummy")
    budgets = []
    for param in MAX_TOKENS_PARAMS:
        conf = ModelConfig(
            provider="openai_server",
            model_id="gpt-6-astra",
            api_key_env="TEST_MODEL_API_KEY",
            api_base="http://localhost:1",
            model_max_context=20_000,
            max_new_tokens=2048,
            max_tokens_param=param,
        )
        budgets.append(build_model(conf, EventWatcher(task_id=1, sink=lambda event: None))._context_budget)

    assert budgets == [20_000 - 2048 - 512] * 2
