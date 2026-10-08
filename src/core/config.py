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

    _api_key: str = PrivateAttr()

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
