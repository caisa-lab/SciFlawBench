import httpx
import litellm.exceptions
import openai
import pytest
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import AgentGenerationError, AgentMaxStepsError

from core.events import AgentEvent, EventType
from core.tasks import classify_failure, timeout_failure_kind

REQUEST = httpx.Request("POST", "http://provider/v1/chat/completions")


def _status_error(cls, status, message="error", body=None):
    return cls(message, response=httpx.Response(status, request=REQUEST), body=body)


def _wrapped(error):
    """`error` as smolagents raises it from a model call: an AgentGenerationError caused by the provider's error."""
    try:
        try:
            raise error
        except Exception as exc:
            raise AgentGenerationError(f"Error in generating model output:\n{exc}", AgentLogger(LogLevel.OFF)) from exc
    except AgentGenerationError as wrapped:
        return wrapped


def _litellm(cls, **kwargs):
    return cls("provider error", llm_provider="openai", model="gpt-5.5", **kwargs)


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        (_status_error(openai.RateLimitError, 429, "You have no credits remaining."), "RateLimitError"),
        (openai.APIConnectionError(request=REQUEST), "APIConnectionError"),
        (openai.APITimeoutError(request=REQUEST), "APITimeoutError"),
        (_status_error(openai.AuthenticationError, 401), "AuthenticationError"),
        (_status_error(openai.NotFoundError, 404, "The model does not exist."), "NotFoundError"),
        (_status_error(openai.InternalServerError, 500), "InternalServerError"),
        (_status_error(openai.APIStatusError, 503, "Service Unavailable"), "APIStatusError"),
        (
            _status_error(openai.BadRequestError, 400, "not supported", {"code": "unsupported_parameter"}),
            "BadRequestError",
        ),
        (_litellm(litellm.exceptions.RateLimitError), "RateLimitError"),
        (_litellm(litellm.exceptions.ServiceUnavailableError), "ServiceUnavailableError"),
        (
            litellm.exceptions.UnsupportedParamsError("stop is not supported", llm_provider="openai"),
            "UnsupportedParamsError",
        ),
        (ConnectionResetError("connection reset by peer"), "ConnectionResetError"),
    ],
)
def test_provider_and_connection_errors_are_the_infrastructures(error, error_type):
    assert classify_failure(error) == ("infrastructure", error_type)
    assert classify_failure(_wrapped(error)) == ("infrastructure", error_type)


@pytest.mark.parametrize(
    ("error", "error_type"),
    [
        # the model filled its context
        (_litellm(litellm.exceptions.ContextWindowExceededError), "ContextWindowExceededError"),
        (
            _status_error(openai.BadRequestError, 400, "maximum context length", {"code": "context_length_exceeded"}),
            "BadRequestError",
        ),
        # the provider refuses the conversation's content: a re-run would only repeat it
        (_litellm(litellm.exceptions.ContentPolicyViolationError), "ContentPolicyViolationError"),
        (
            litellm.exceptions.BlockedPiiEntityError("an entity was blocked", guardrail_name="pii"),
            "BlockedPiiEntityError",
        ),
        # the type decides, not the message
        (_status_error(openai.BadRequestError, 400, "Unsupported parameter: 'stop'"), "BadRequestError"),
        (ValueError("could not parse the model output"), "ValueError"),
    ],
)
def test_other_errors_are_the_models(error, error_type):
    assert classify_failure(error) == ("model", error_type)
    assert classify_failure(_wrapped(error)) == ("model", error_type)


def test_running_out_of_steps_is_the_models():
    error = AgentMaxStepsError("Reached max steps.", AgentLogger(LogLevel.OFF))

    assert classify_failure(error) == ("model", "AgentMaxStepsError")


def _event(event_type):
    return AgentEvent(event_type=event_type, task_id=1, payload={}, timestamp=0.0)


@pytest.mark.parametrize(
    ("last_event", "kind"),
    [
        (_event(EventType.ModelCallStart), "model"),  # a model call that never returned
        (_event(EventType.ModelCallEnd), "model"),  # e.g. the agent's own code running on
        (_event(EventType.ToolCallEnd), "model"),
        (_event(EventType.ToolCallStart), "infrastructure"),  # a hung tool or network call
        (None, "infrastructure"),  # killed before the agent recorded anything
    ],
)
def test_a_timeout_is_judged_by_its_last_event(last_event, kind):
    assert timeout_failure_kind(last_event) == kind
