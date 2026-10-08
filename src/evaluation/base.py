import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Outcome of one check run against an agent's output."""

    passed: bool
    details: str

    # Set by the LLM-as-a-judge verifier: the judge's rationale for its verdict.
    justification: str | None = None

    # Extra provenance for the check (the rubric, which judge model graded it, ...).
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierDef(BaseModel):
    name: str
    kwargs: dict = Field(default_factory=dict)


@dataclass
class VerifierContext:
    """
    Everything a verifier may need beyond the agent's final answer.

    Only verifiers that declare a `context` parameter receive it (see :func:`wants_context`),
    so the plain content verifiers keep their simple `(got, **kwargs)` signature.
    """

    trace: str = ""  # the run's trace, rendered like the markdown trace reports
    task_id: int | str = ""
    task: str = ""
    judge: Any = None  # a judge client, or None when no judge is configured
    judge_error: str | None = None  # why there is no judge client, if that is the case


def wants_context(f: Callable) -> bool:
    """Whether a verifier declared the optional `context` parameter."""
    try:
        return "context" in inspect.signature(f).parameters
    except (TypeError, ValueError):
        return False


def run_check(f: Callable, got: str, context: VerifierContext | None = None, **kwargs) -> VerificationResult:
    """
    Run one verifier against an agent's output, never raising.

    Args:
        f (Callable): the verifier function
        got (str): the agent's final answer
        context (VerifierContext | None): extra material, but only handed to verifiers that ask for it
        kwargs (dict): the verifier's configured kwargs

    Returns (VerificationResult): the verdict, or a failed result carrying the exception
    """
    try:
        if wants_context(f):
            # A verifier that declares `context` always gets one: an empty context lets the
            # judge verifiers report "no judge configured" instead of raising a TypeError.
            return f(got, context=context if context is not None else VerifierContext(), **kwargs)
        return f(got, **kwargs)
    except Exception as e:
        return VerificationResult(passed=False, details=f"Failed to run verifier with following exception: {e}")
