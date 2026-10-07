from smolagents import (
    ChatMessage,
    InferenceClientModel,
    LiteLLMModel,
    Model,
    OpenAIServerModel,
    Tool,
)

from core.config import ModelConfig
from core.events import EventWatcher


class WrappedModel(Model):
    """
    A wrapper around the basic model class that smolagents uses which is connected to an event watcher that logs what it
    gets used for
    """

    def __init__(self, wrapped_model: Model, watcher: EventWatcher):
        super().__init__()

        self._wrapped = wrapped_model
        self._watcher = watcher

        self._last_logged_len = 0

    def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        """
        the exposed generate function so that smolagents knows how to use this (wrapped with the watcher so we can get
        the input/results out as well)

        note that this will try to save only message deltas on each run
        """
        new_messages = messages[self._last_logged_len :]
        self._last_logged_len = len(messages)
        kwargs["start_payload"] = {
            "new_messages": new_messages,
            "message_count": len(messages),
            "stop_sequences": stop_sequences,
            "response_format": response_format,
            "tools_to_call_from": [getattr(t, "name", t) for t in tools_to_call_from] if tools_to_call_from else [],
        }

        return self._watcher(
            "model",
            getattr(self._wrapped, "model_id", self._wrapped.__class__.__name__),
            self._wrapped.generate,
            messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )


def build_model(conf: ModelConfig, watcher: EventWatcher) -> WrappedModel:
    """
    builds a wrapped model from the provided model config

    Args:
        conf (ModelConfig): configuration needed to provision the model instance
        watcher (EventWatcher): provided Event watcher which wraps around all model calls and records inputs/outputs

    Returns (WrappedModel): wrapped model object to be used by the smolagent
    """

    match conf.provider.lower():
        case "litellm":
            import litellm

            litellm.suppress_debug_info = True

            model = LiteLLMModel(
                model_id=conf.model_id, api_base=conf.api_base, api_key=conf.api_key, **conf.extra_kwargs
            )
        case "openai_server":
            model = OpenAIServerModel(
                model_id=conf.model_id, api_base=conf.api_base, api_key=conf.api_key, **conf.extra_kwargs
            )
        case "hf_api":
            model = InferenceClientModel(model_id=conf.model_id, api_key=conf.api_key, **conf.extra_kwargs)
        case "fake_model":  # this branch is only for testing
            from tests.fakes.models import ScriptedModel

            model = ScriptedModel(**conf.extra_kwargs)
        case _:
            raise ValueError(f"Got an unsupported model Provider: {conf.provider}")

    if conf.supports_stop is not None:
        # smolagents reads stop support from a property, so swap in a subclass that reports the configured value;
        # without stop support it drops the parameter and cuts the output at the stop sequences itself
        model.__class__ = type(
            model.__class__.__name__,
            (model.__class__,),
            {"supports_stop_parameter": property(lambda self: conf.supports_stop)},
        )

    return WrappedModel(wrapped_model=model, watcher=watcher)
