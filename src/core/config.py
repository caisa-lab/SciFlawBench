import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from tools.base import ToolDef


def read_env_api_key(api_key_env: str) -> str:
    """
    Read an API key out of the environment, loading a local `.env` file first.

    Only the *name* of the variable is ever stored in a config file; the value itself is
    resolved here and kept as a private attribute so it cannot be serialised into a
    config dump, a manifest or a log.

    Args:
        api_key_env (str): name of the environment variable holding the key

    Returns (str): the resolved key value
    """
    load_dotenv()
    value = os.environ.get(api_key_env)
    if not value:
        raise ValueError(f"Env var '{api_key_env}' is not set")
    return value


class ModelConfig(BaseModel):
    """
    class which stores all the information needed to provision a model (gets read from the config)

    also acts as a typing mechanism through pydantic to verify things were correctly specified
    """

    provider: Literal["litellm", "openai_server", "hf_api", "fake_model"]
    model_id: str
    api_key_env: str
    api_base: str | None = None
    extra_kwargs: dict = Field(default_factory=dict)

    # this is kept in the model config because it generally is a model dependant field to be configured
    code_block_tags: tuple[str, str] | None = None

    # Context window of the model, in tokens. Left unset it is auto-detected (vLLM `/models`, then
    # LiteLLM's model table) and falls back to 256_000. Before every model call the conversation is
    # trimmed to `window - max_new_tokens - 512` so a long run never overflows the model's context.
    # Set it explicitly (e.g. 1_000_000) to override detection.
    model_max_context: int | None = None

    # Cap on the tokens generated per model call (one agent step); also reserved out of the input
    # budget. None leaves the provider default and reserves nothing. A `max_tokens` given in
    # `extra_kwargs` is adopted here for backwards compatibility.
    max_new_tokens: int | None = 16_384

    # set to False for models that reject the `stop` parameter but are missing from smolagents' hard-coded list
    # (o3, o4, gpt-5*, grok-*); None leaves the decision to smolagents
    supports_stop: bool | None = None

    _api_key: str = PrivateAttr()

    @field_validator("model_max_context", "max_new_tokens")
    @classmethod
    def token_limits_are_positive(cls, v: int | None) -> int | None:
        """Reject a non-positive limit: it would silently disable trimming or generation."""
        if v is not None and v <= 0:
            raise ValueError("token limits must be a positive number of tokens")
        return v

    @model_validator(mode="after")
    def reconcile_max_tokens(self) -> "ModelConfig":
        """Keep `max_new_tokens` and a legacy `extra_kwargs['max_tokens']` from disagreeing."""
        legacy = self.extra_kwargs.get("max_tokens")
        if legacy is None:
            return self
        if "max_new_tokens" not in self.model_fields_set:
            self.max_new_tokens = int(legacy)
        elif self.max_new_tokens != legacy:
            raise ValueError("set either max_new_tokens or extra_kwargs.max_tokens, not both with different values")
        return self

    @model_validator(mode="after")
    def resolve_api_key(self) -> "ModelConfig":
        """
        Function to get the api_key from the environment (needs api_key_env to be specified in the config)
        """
        self._api_key = read_env_api_key(self.api_key_env)
        return self

    @property
    def api_key(self) -> str:
        """
        Method to expose the api key parameter through code to classes that have the model config

        Returns (str): the raw api_key
        """
        return self._api_key


class JudgeConfig(BaseModel):
    """
    Configuration for the LLM-as-a-judge model used by the `judge:*` validators.

    Judging always goes through LiteLLM, so there is no `provider` field: `model_id` is a
    LiteLLM model string (e.g. `openrouter/qwen/qwen3.7-flash`, `gpt-4o-mini`, ...). The whole
    judge configuration is recorded in the run manifest and is part of the run signature, so
    changing the judge invalidates a resume.
    """

    model_id: str
    api_key_env: str
    api_base: str | None = None
    extra_kwargs: dict = Field(default_factory=dict)

    _api_key: str = PrivateAttr()

    @model_validator(mode="after")
    def resolve_api_key(self) -> "JudgeConfig":
        """
        Function to get the judge's api_key from the environment (needs api_key_env to be specified in the config)
        """
        self._api_key = read_env_api_key(self.api_key_env)
        return self

    @property
    def api_key(self) -> str:
        """
        Method to expose the api key parameter to the judge client

        Returns (str): the raw api_key
        """
        return self._api_key


class RunConfig(BaseModel):
    """
    class which stores all the information needed to provision a benchmark run (also gets read from the config)
    """

    model: ModelConfig
    judge: JudgeConfig | None = None  # required by the `judge:*` validators; pinned in the manifest
    task_file: Path
    tool_configs: list[ToolDef] = Field(default_factory=list)
    log_path: Path = Path("logs/")
    max_concurrent: int = 3  # default max concurrent task running processes
    closed_book: bool = False  # gives the agent no tools at all: it may only use its own knowledge

    repetitions_per_task: int = 3
    logging_level: int = 20
    task_timeout_s: int = 60 * 15  # 15 minute timeout for tasks before they get killed by the runtime manager
    restarting: bool = False  # if you want to restart on a specific dir specify the log_path and set to True
    generate_trace_reports: bool = False

    # Oversized tool outputs are spilled to a per-task artifact directory and the observation
    # carries a preview plus instructions for retrieving the rest (see tools/artifacts.py).
    # Set `artifact_spill` to False to fall back to plain in-place truncation.
    artifact_spill: bool = True
    # Characters a tool may return before its output is spilled to disk.
    tool_output_max_chars: int = 20_000

    # When context trimming evicts a message, its full text is written to `memory.md` in the task's
    # artifact scope and replaced by a digest plus a pointer; an index of the blocks is kept in the
    # task message. Needs `artifact_spill`; otherwise evicted text is replaced by a plain note.
    memory_offload: bool = True
    memory_digest_chars: int = 200  # inline preview of an evicted message
    memory_index_max_chars: int = 4_000  # cap on the block index injected into the task message

    @field_validator("tool_output_max_chars")
    @classmethod
    def tool_output_max_chars_is_positive(cls, v: int) -> int:
        """Reject a non-positive cap: it would disable output capping entirely."""
        if v <= 0:
            raise ValueError("tool_output_max_chars must be a positive number of characters")
        return v

    @field_validator("memory_digest_chars", "memory_index_max_chars")
    @classmethod
    def memory_limits_are_non_negative(cls, v: int) -> int:
        """A zero digest/index is allowed (it just omits them); a negative one is a mistake."""
        if v < 0:
            raise ValueError("memory limits must not be negative")
        return v

    @field_validator("task_file")
    @classmethod
    def task_path_exists_and_is_jsonl(cls, v: Path) -> Path:
        """
        just checks that the task_file field exists, and is a valid jsonl file
        """
        if not v.is_file():
            raise ValueError(f"Tasks file not found: {v}")
        if not v.suffix == ".jsonl":
            raise ValueError(f"Tasks file not jsonl: {v}")
        return v
