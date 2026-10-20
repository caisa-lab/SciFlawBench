"""Tests for the LLM-as-a-judge evaluator (no live API calls)."""

import json

import pytest

from core.config import JudgeConfig
from evaluation.base import VerifierContext, run_check, wants_context
from evaluation.definitions import verifier_registry
from evaluation.judge import (
    JUDGE_OUTPUT_SPECIFICATION,
    JudgeReplyError,
    JudgeVerdict,
    LiteLLMJudge,
    build_judge_prompt,
    judge_template_sha256,
    parse_judge_reply,
)


class FakeJudge:
    """Stand-in for `LiteLLMJudge` that never touches the network."""

    model_id = "fake/judge"

    def __init__(self, passed=True, justification="looks fine", error=None):
        self.passed = passed
        self.justification = justification
        self.error = error
        self.calls = []

    def judge(self, rubric, trace):
        self.calls.append((rubric, trace))
        if self.error is not None:
            raise self.error
        return JudgeVerdict(passed=self.passed, justification=self.justification, raw_reply="{}")


@pytest.fixture
def judge_conf(monkeypatch):
    monkeypatch.setenv("JUDGE_KEY", "judge-secret")
    return JudgeConfig(model_id="openrouter/qwen/qwen3.7-flash", api_key_env="JUDGE_KEY")


def test_judge_config_reads_key_from_env_and_hides_it(judge_conf):
    assert judge_conf.api_key == "judge-secret"
    assert "judge-secret" not in str(judge_conf.model_dump())


def test_judge_config_rejects_missing_env_var(monkeypatch):
    monkeypatch.delenv("ABSENT_JUDGE_KEY", raising=False)
    with pytest.raises(ValueError, match="ABSENT_JUDGE_KEY"):
        JudgeConfig(model_id="m", api_key_env="ABSENT_JUDGE_KEY")


def test_parse_judge_reply_accepts_plain_json():
    assert parse_judge_reply('{"evaluation": true, "justification": "it did"}') == (True, "it did")
    assert parse_judge_reply('{"evaluation": false, "justification": "it did not"}') == (False, "it did not")


def test_parse_judge_reply_tolerates_fences_and_surrounding_prose():
    fenced = '```json\n{"evaluation": true, "justification": "ok"}\n```'
    assert parse_judge_reply(fenced) == (True, "ok")
    prose = 'Sure, here it is: {"evaluation": false, "justification": "nope"} Hope that helps.'
    assert parse_judge_reply(prose) == (False, "nope")


@pytest.mark.parametrize(
    "raw_verdict, expected",
    [(1, True), (0, False), ("true", True), ("False", False), ("yes", True), ("no", False)],
)
def test_parse_judge_reply_coerces_non_boolean_verdicts(raw_verdict, expected):
    reply = json.dumps({"evaluation": raw_verdict, "justification": "j"})
    assert parse_judge_reply(reply)[0] is expected


def test_parse_judge_reply_fills_in_a_missing_justification():
    passed, justification = parse_judge_reply('{"evaluation": true, "justification": "   "}')
    assert passed is True
    assert justification


@pytest.mark.parametrize(
    "reply",
    [
        "no json at all",
        "{not valid json}",
        '["a", "list"]',
        '{"justification": "missing the verdict"}',
        '{"evaluation": true}',
        '{"evaluation": "maybe", "justification": "unclear"}',
    ],
)
def test_parse_judge_reply_rejects_malformed_replies(reply):
    with pytest.raises(JudgeReplyError):
        parse_judge_reply(reply)


def test_build_judge_prompt_carries_rubric_trace_and_output_spec():
    prompt = build_judge_prompt("Is the sky blue?", "TRACE-MARKER")
    assert "Is the sky blue?" in prompt
    assert "TRACE-MARKER" in prompt
    assert JUDGE_OUTPUT_SPECIFICATION in prompt
    assert '"evaluation"' in prompt and '"justification"' in prompt


def test_judge_template_sha256_is_stable_and_hex():
    digest = judge_template_sha256()
    assert digest == judge_template_sha256()
    assert len(digest) == 64


def test_judge_client_sends_rubric_and_trace_and_parses_verdict(monkeypatch, judge_conf):
    calls = {}

    class FakeResponse:
        class Choice:
            class Message:
                content = '{"evaluation": true, "justification": "the trace shows it"}'

            message = Message()

        choices = [Choice()]

    def fake_completion(**kwargs):
        calls.update(kwargs)
        return FakeResponse()

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)

    verdict = LiteLLMJudge(judge_conf).judge("Is it right?", "TRACE")

    assert verdict.passed is True
    assert verdict.justification == "the trace shows it"
    assert calls["model"] == "openrouter/qwen/qwen3.7-flash"
    assert calls["api_key"] == "judge-secret"
    assert "TRACE" in calls["messages"][0]["content"]


def test_judge_client_sends_the_whole_trace(monkeypatch, judge_conf):
    """No client-side truncation: whatever the harness renders is what the judge receives."""
    sent = {}

    class FakeResponse:
        class Choice:
            class Message:
                content = '{"evaluation": true, "justification": "ok"}'

            message = Message()

        choices = [Choice()]

    def fake_completion(**kwargs):
        sent["content"] = kwargs["messages"][0]["content"]
        return FakeResponse()

    import litellm

    monkeypatch.setattr(litellm, "completion", fake_completion)

    huge_trace = "TRACE-START" + ("x" * 200_000) + "TRACE-END"
    LiteLLMJudge(judge_conf).judge("rubric", huge_trace)

    assert "TRACE-START" in sent["content"]
    assert "TRACE-END" in sent["content"]
    assert huge_trace in sent["content"]


def test_judge_rubric_verifier_records_score_and_justification():
    judge = FakeJudge(passed=True, justification="the abstract is about neural networks")
    context = VerifierContext(trace="TRACE", task_id="abc", task="t", judge=judge)

    result = run_check(verifier_registry.get("judge:rubric"), "answer", context=context, rubric="RUBRIC")

    assert result.passed is True
    assert result.justification == "the abstract is about neural networks"
    assert result.metadata == {"rubric": "RUBRIC", "judge_model": "fake/judge"}
    assert judge.calls == [("RUBRIC", "TRACE")]


def test_judge_rubric_verifier_reports_a_failing_verdict():
    context = VerifierContext(judge=FakeJudge(passed=False, justification="the trace does not show it"))

    result = run_check(verifier_registry.get("judge:rubric"), "answer", context=context, rubric="RUBRIC")

    assert result.passed is False
    assert "the trace does not show it" in result.details


def test_judge_rubric_verifier_fails_cleanly_without_a_judge():
    result = run_check(
        verifier_registry.get("judge:rubric"),
        "answer",
        context=VerifierContext(judge_error="the run config has no 'judge' section"),
        rubric="RUBRIC",
    )

    assert result.passed is False
    assert "judge" in result.details.lower()


def test_judge_rubric_verifier_fails_cleanly_when_the_judge_errors():
    context = VerifierContext(judge=FakeJudge(error=RuntimeError("api down")))

    result = run_check(verifier_registry.get("judge:rubric"), "answer", context=context, rubric="RUBRIC")

    assert result.passed is False
    assert "api down" in result.details


def test_judge_rubric_verifier_fails_cleanly_on_a_malformed_reply():
    context = VerifierContext(judge=FakeJudge(error=JudgeReplyError("not JSON")))

    result = run_check(verifier_registry.get("judge:rubric"), "answer", context=context, rubric="RUBRIC")

    assert result.passed is False
    assert "format" in result.details


def test_only_context_aware_verifiers_receive_the_context():
    assert wants_context(verifier_registry.get("judge:rubric"))
    assert not wants_context(verifier_registry.get("content:contains_str"))

    # A plain verifier must still run happily when a context is available.
    result = run_check(
        verifier_registry.get("content:contains_str"),
        "hello",
        context=VerifierContext(trace="irrelevant"),
        expected="ell",
    )
    assert result.passed is True


def test_verifier_without_context_gets_a_failed_result():
    result = run_check(verifier_registry.get("judge:rubric"), "answer", rubric="RUBRIC")
    assert result.passed is False
    assert "unavailable" in result.details
