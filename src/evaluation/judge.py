import hashlib
import json
import re
from dataclasses import dataclass

from core.config import JudgeConfig

#: The part of the judge prompt that demands a machine-readable reply. Hardcoded on purpose:
#: the harness depends on this exact contract to record the judge's score and justification.
JUDGE_OUTPUT_SPECIFICATION = """## Output format

Reply with a single JSON object and nothing else. The object must have exactly these two keys:

  "evaluation":    the verdict, either true or false. Use a JSON boolean (true/false), not a
                   string, and not 1/0.
  "justification": a short justification for the verdict (one or two sentences) referring to
                   what the trace actually shows.

Example reply:

{"evaluation": true, "justification": "The trace shows the agent fetched the abstract and reported its score."}

Do not wrap the object in a code fence, do not add any text before or after it, and do not add
any further keys."""

#: Prompt skeleton: the rubric, the trace, then the hardcoded output specification.
JUDGE_PROMPT_TEMPLATE = """You are grading the trace of an AI agent that attempted a task. Judge only what the trace
shows; do not reward a plausible answer that the trace does not actually support.

## Rubric

{rubric}

## Trace

{trace}

{specification}"""

# Values a judge may use for the verdict when it does not emit a real JSON boolean.
_TRUE_VERDICTS = {"true", "yes", "y", "pass", "passed", "satisfied", "1"}
_FALSE_VERDICTS = {"false", "no", "n", "fail", "failed", "unsatisfied", "0"}


class JudgeReplyError(ValueError):
    """Raised when a judge reply cannot be read as the required JSON object."""


@dataclass
class JudgeVerdict:
    """A parsed judge reply."""

    passed: bool
    justification: str
    raw_reply: str


def judge_template_sha256() -> str:
    """
    SHA-256 over the judge prompt template and output specification.

    Recorded in the run manifest: if the harness ever changes how judges are prompted, a
    resumed run can no longer pretend its grades came from the same procedure.

    Returns (str): hex digest of the two hardcoded strings
    """
    payload = JUDGE_PROMPT_TEMPLATE + "\x00" + JUDGE_OUTPUT_SPECIFICATION
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_judge_prompt(rubric: str, trace: str) -> str:
    """
    Build the judge prompt: rubric + trace + the hardcoded output specification.

    Args:
        rubric (str): what the judge has to decide about the trace
        trace (str): the formatted trace of the run being graded

    Returns (str): the full prompt sent to the judge model
    """
    return JUDGE_PROMPT_TEMPLATE.format(rubric=rubric.strip(), trace=trace, specification=JUDGE_OUTPUT_SPECIFICATION)


def _extract_json_object(reply: str) -> dict:
    """Pull a JSON object out of a reply, tolerating code fences and surrounding prose."""
    text = reply.strip()

    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise JudgeReplyError(f"reply contains no JSON object: {text[:200]!r}") from None
        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise JudgeReplyError(f"reply is not valid JSON ({exc}): {text[:200]!r}") from exc

    if not isinstance(data, dict):
        raise JudgeReplyError(f"reply JSON is a {type(data).__name__}, expected an object")
    return data


def _coerce_verdict(value: object) -> bool:
    """Interpret the `evaluation` field: a boolean, 1/0, or a yes/no-style string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    if text in _TRUE_VERDICTS:
        return True
    if text in _FALSE_VERDICTS:
        return False
    raise JudgeReplyError(f"could not read 'evaluation' value {value!r} as a boolean")


def parse_judge_reply(reply: str) -> tuple[bool, str]:
    """
    Read a judge reply into `(verdict, justification)`.

    Args:
        reply (str): the raw text the judge model returned

    Returns (tuple[bool, str]): whether the trace satisfied the rubric, and the judge's rationale

    Raises:
        JudgeReplyError: when the reply is not a JSON object with the two required keys
    """
    data = _extract_json_object(reply)

    missing = [key for key in ("evaluation", "justification") if key not in data]
    if missing:
        raise JudgeReplyError(f"reply is missing the required key(s) {missing} (got keys {sorted(data)})")

    passed = _coerce_verdict(data["evaluation"])
    justification = str(data["justification"]).strip() or "(the judge provided no justification)"
    return passed, justification


class LiteLLMJudge:
    """
    Judge client: sends rubric + trace to a LiteLLM-served model and parses the verdict.

    The model, endpoint and extra kwargs come from the run config's `judge` section, which is
    pinned in the run manifest, so every task in a run is graded by the same judge.

    The trace is sent whole: the harness does not truncate, compress or otherwise limit it, and
    leaves any handling of oversized requests to the API / provider.
    """

    def __init__(self, conf: JudgeConfig):
        self.conf = conf

    @property
    def model_id(self) -> str:
        """The LiteLLM model string this judge calls."""
        return self.conf.model_id

    def judge(self, rubric: str, trace: str) -> JudgeVerdict:
        """
        Grade one trace against one rubric.

        Args:
            rubric (str): the question the judge has to answer about the trace
            trace (str): the formatted trace to grade, sent in full

        Returns (JudgeVerdict): the parsed verdict, justification and raw reply

        Raises:
            JudgeReplyError: when the model's reply does not follow the required JSON format
        """
        prompt = build_judge_prompt(rubric, trace)
        raw_reply = self._complete(prompt)
        passed, justification = parse_judge_reply(raw_reply)
        return JudgeVerdict(passed=passed, justification=justification, raw_reply=raw_reply)

    def _complete(self, prompt: str) -> str:
        """One LiteLLM completion. Imported lazily so importing this module stays cheap."""
        import litellm

        litellm.suppress_debug_info = True

        response = litellm.completion(
            model=self.conf.model_id,
            messages=[{"role": "user", "content": prompt}],
            api_key=self.conf.api_key,
            api_base=self.conf.api_base,
            **self.conf.extra_kwargs,
        )
        return response.choices[0].message.content or ""
