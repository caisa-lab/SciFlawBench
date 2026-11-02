import itertools
import time

import httpx
import openai
from smolagents import ChatMessage, MessageRole, Model

_REQUEST = httpx.Request("POST", "http://fake-provider/v1/chat/completions")

#: Errors a scripted model can raise instead of answering, as a provider would.
FAKE_ERRORS = {
    # the provider refuses the call: the infrastructure's fault
    "rate_limit": lambda: openai.RateLimitError(
        "You have no credits remaining.", response=httpx.Response(429, request=_REQUEST), body=None
    ),
    # the conversation outgrew the model: the model's fault
    "context_length": lambda: openai.BadRequestError(
        "This model's maximum context length is 128000 tokens.",
        response=httpx.Response(400, request=_REQUEST),
        body={"code": "context_length_exceeded"},
    ),
}


class ScriptedModel(Model):
    def __init__(self, responses: list[str], timeout: float = 0.5, error: str | None = None):
        super().__init__()
        self._responses = itertools.cycle(responses)
        self.timeout = timeout
        self.error = error

    def generate(
        self,
        messages,
        stop_sequences=None,
        response_format=None,
        tools_to_call_from=None,
        **kwargs,
    ) -> ChatMessage:
        time.sleep(self.timeout)
        if self.error:
            raise FAKE_ERRORS[self.error]()
        return ChatMessage(role=MessageRole.ASSISTANT, content=next(self._responses))
