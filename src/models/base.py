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
from models.context import (
    CONTEXT_TRIM_RESERVE,
    compute_input_budget,
    count_message_tokens,
    estimate_tokens,
    resolve_context_window,
    trim_messages_to_budget,
)
from models.memory import MemoryOffloader


class WrappedModel(Model):
    """
    A wrapper around the basic model class that smolagents uses which is connected to an event watcher that logs what it
    gets used for

    It also performs context hygiene: when `model_max_context` is set, the messages handed to
    the underlying model are trimmed to fit `model_max_context - max_new_tokens - reserve_tokens`
    (old observations are evicted oldest-first) so a long run never overflows the model's context.
    When a `MemoryOffloader` is given, evicted messages are spilled to the task's memory file
    and replaced by a pointer, instead of being lost.
    """

    def __init__(
        self,
        wrapped_model: Model,
        watcher: EventWatcher,
        model_max_context: int | None = None,
        reserve_tokens: int = CONTEXT_TRIM_RESERVE,
        max_new_tokens: int | None = None,
        offloader: MemoryOffloader | None = None,
    ):
        super().__init__()

        self._wrapped = wrapped_model
        self._watcher = watcher

        self._last_logged_len = 0

        self._offloader = offloader
        # Budget for the *input*: the window minus the response allowance and the estimation reserve.
        self._context_budget = compute_input_budget(model_max_context, max_new_tokens, reserve_tokens)

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
        messages, trimmed = self._apply_context_budget(messages)

        start = min(self._last_logged_len, len(messages))
        new_messages = messages[start:]
        self._last_logged_len = len(messages)
        kwargs["start_payload"] = {
            "new_messages": new_messages,
            "message_count": len(messages),
            "prompt_tokens_estimate": count_message_tokens(messages, estimate_tokens),
            "context_budget": self._context_budget or None,
            "context_trimmed": trimmed,
            "memory_blocks": self._offloader.snapshot()["blocks"] if self._offloader else 0,
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

    def _apply_context_budget(self, messages: list[ChatMessage]) -> tuple[list[ChatMessage], bool]:
        """
        Trim `messages` to the configured input budget, returning `(messages, was_trimmed)`.

        When no budget is configured the list is returned untouched. Trimming never raises:
        a failure to estimate tokens leaves the request as it was, so a broken token counter
        cannot take a run down.
        """
        if not self._context_budget or not messages:
            return messages, False
        before = count_message_tokens(messages, estimate_tokens)
        if before <= self._context_budget:
            return messages, False
        try:
            messages = trim_messages_to_budget(messages, self._context_budget, offloader=self._offloader)
        except Exception:  # noqa: BLE001 - trimming is best-effort, never fatal
            return messages, False
        return messages, True


def build_model(conf: ModelConfig, watcher: EventWatcher, offloader: MemoryOffloader | None = None) -> WrappedModel:
    """
    builds a wrapped model from the provided model config

    Args:
        conf (ModelConfig): configuration needed to provision the model instance
        watcher (EventWatcher): provided Event watcher which wraps around all model calls and records inputs/outputs
        offloader (MemoryOffloader | None): where context evicted by the trimmer is spilled; None just truncates it

    Returns (WrappedModel): wrapped model object to be used by the smolagent
    """

    # `max_new_tokens` caps every generation step (it is also reserved out of the input budget below)
    extra_kwargs = dict(conf.extra_kwargs)
    if conf.max_new_tokens and conf.provider.lower() != "fake_model":
        extra_kwargs["max_tokens"] = conf.max_new_tokens

    match conf.provider.lower():
        case "litellm":
            import litellm

            litellm.suppress_debug_info = True

            model = LiteLLMModel(model_id=conf.model_id, api_base=conf.api_base, api_key=conf.api_key, **extra_kwargs)
        case "openai_server":
            model = OpenAIServerModel(
                model_id=conf.model_id, api_base=conf.api_base, api_key=conf.api_key, **extra_kwargs
            )
        case "hf_api":
            model = InferenceClientModel(model_id=conf.model_id, api_key=conf.api_key, **extra_kwargs)
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

    context_window = resolve_context_window(
        conf.provider.lower(), conf.model_id, conf.api_base, conf.api_key, explicit=conf.model_max_context
    )

    return WrappedModel(
        wrapped_model=model,
        watcher=watcher,
        model_max_context=context_window,
        reserve_tokens=CONTEXT_TRIM_RESERVE,
        max_new_tokens=conf.max_new_tokens,
        offloader=offloader,
    )
