import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    passed: bool
    details: str


class VerifierDef(BaseModel):
    name: str
    kwargs: dict = Field(default_factory=dict)


def run_check(f: Callable, got: Any, **kwargs) -> VerificationResult:
    # a code agent's final_answer can return any python object (tuple, float, list, dict, ...); the
    # verifiers all expect the answer as text, so serialise it the way the agent would have printed it
    if not isinstance(got, str):
        try:
            got = json.dumps(got, ensure_ascii=False) if isinstance(got, (list, dict)) else str(got)
        except TypeError:
            got = str(got)
    try:
        return f(got, **kwargs)
    except Exception as e:
        return VerificationResult(passed=False, details=f"Failed to run verifier with following exception: {e}")
