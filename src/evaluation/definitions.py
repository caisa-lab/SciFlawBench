import json
import math
import re

from core.registry import Registry
from evaluation.base import VerificationResult, VerifierContext
from evaluation.judge import JudgeReplyError

verifier_registry = Registry("verifiers")


@verifier_registry.register("content:contains_str")
def contains_str(got: str, expected: str, case_sensitive: bool = False) -> VerificationResult:
    passed = expected in got if case_sensitive else expected.lower() in got.lower()
    return VerificationResult(passed=passed, details=f"Got: {got} Expected: {expected}")


@verifier_registry.register("content:exact_str_match")
def exact_str_match(got: str, expected: str, case_sensitive: bool = False) -> VerificationResult:
    passed = expected.strip() == got.strip() if case_sensitive else expected.strip().lower() == got.strip().lower()
    return VerificationResult(passed=passed, details=f"Got: {got} Expected: {expected}")


@verifier_registry.register("content:numeric_match")
def single_numeric_match(got: str, expected: str, tol: float = 0.0, index: int = -1) -> VerificationResult:
    matches = re.findall(r"-?\d+\.?\d*", got)
    if not matches:
        return VerificationResult(passed=False, details=f"No numeric types included in answer - Got: {got}")
    try:
        got_val = float(matches[index])
    except IndexError:
        return VerificationResult(
            passed=False,
            details=f"Index {index} out of range for expected: {expected} and \
                                  received strings: {got}",
        )
    try:
        expected_val = float(expected)
    except ValueError:
        return VerificationResult(passed=False, details=f"Expected value: {expected} is not a numeric type")

    passed = abs(got_val - expected_val) <= tol
    return VerificationResult(
        passed=passed,
        details=f"Expected value: {expected_val} Got value: {got_val} diff \
            {abs(got_val - expected_val)} and tol: {tol}",
    )


@verifier_registry.register("content:numeric_within_range")
def numeric_within_range(got: str, minimum: float, maximum: float, index: int = -1) -> VerificationResult:
    matches = re.findall(r"-?\d+\.?\d*", got)
    if not matches:
        return VerificationResult(passed=False, details=f"No numeric types included in answer - Got: {got}")
    try:
        got_val = float(matches[index])
    except IndexError:
        return VerificationResult(passed=False, details=f"Index {index} out of range in received string: {got}")

    passed = got_val <= maximum and got_val >= minimum
    return VerificationResult(passed=passed, details=f"Got value: {got_val} minimum: {minimum} and maximum: {maximum}")


@verifier_registry.register("content:paper_json_match")
def paper_list_match(
    got: str, expected: list[dict], match_mode: str = "all"
) -> VerificationResult:  # TODO: make match_mode an StrEnum or give literal type hints
    try:
        parsed = json.loads(got)
        papers = parsed.get("papers", [])
    except (json.JSONDecodeError, AttributeError):
        return VerificationResult(passed=False, details="could not parse 'papers' field from output")

    got_ids = {p.get("arxiv_id", "") for p in papers if isinstance(p, dict)}
    expected_ids = {p["arxiv_id"] for p in expected}

    if not expected_ids:
        passed = len(got_ids) == 0
        return VerificationResult(passed=passed, details=f"expected no papers, got {len(got_ids)}")

    matched = got_ids & expected_ids

    passed = expected_ids.issubset(got_ids) if match_mode == "all" else len(matched) > 0

    return VerificationResult(passed=passed, details=f"matched {len(matched)}/{len(expected_ids)} expected papers")


_SCI_RE = re.compile(
    r"""
    (?<![\w.])                          # not preceded by a word char or dot (avoid grabbing mid-number/mid-identifier)
    (?P<coeff>-?(?:\d+\.\d*|\.\d+|\d+)) # 6, 6.022, or .5 — decimal point optional on either side
    \s*
    (?:
        [eE](?P<exp1>[+-]?\d+)                          # 6.022e23, 6.022E+23
      | (?:x|×|\\times|\\cdot)\s*10\s*\^?\s*
        \{?(?P<exp2>[+-]?\d+)\}?                         # 6.022 x 10^23, \times 10^{-11}, × 10 23
    )
    (?![\w.])                            # not immediately followed by a word char or dot
    """,
    re.VERBOSE,
)

_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")


def _extract_scientific_values(text: str) -> list[float]:
    """Finds values written as '6.022e23', '6.022E+23', '6.022 x 10^23',
    '6.022 × 10^23', '6.022 \\times 10^{23}', or '6.022 × 10²³'."""
    text = text.replace("−", "-").translate(_SUPERSCRIPT_MAP)
    values = []
    for m in _SCI_RE.finditer(text):
        coeff = float(m.group("coeff"))
        exp = m.group("exp1") or m.group("exp2")
        values.append(coeff * (10 ** int(exp)))
    return values


def _round_sig_figs(x: float, sig: int) -> float:
    if x == 0:
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


@verifier_registry.register("content:scientific_notation")
def scientific_notation_numeric(
    got: str,
    expected: str,
    rel_tol: float = 0.01,
    sig_figs: int | None = None,
    index: int = -1,
) -> VerificationResult:
    got_values = _extract_scientific_values(got)
    print(got_values)
    if not got_values:
        return VerificationResult(passed=False, details=f"No scientific-notation value found in output - Got: {got}")
    try:
        got_val = got_values[index]
    except IndexError:
        return VerificationResult(
            passed=False,
            details=f"Index {index} out of range for {len(got_values)} scientific-notation values found in: {got}",
        )

    expected_values = _extract_scientific_values(expected)
    if expected_values:
        expected_val = expected_values[0]
    else:
        try:
            expected_val = float(expected)  # allow plain-float expected values too, e.g. "6.022e23" written as-is
        except ValueError:
            return VerificationResult(passed=False, details=f"Expected value could not be parsed: {expected!r}")

    if sig_figs is not None:
        passed = _round_sig_figs(got_val, sig_figs) == _round_sig_figs(expected_val, sig_figs)
        details = f"Got: {got_val:.4e} Expected: {expected_val:.4e} (compared at {sig_figs} sig figs)"
    else:
        bound = abs(expected_val) * rel_tol
        diff = abs(got_val - expected_val)
        passed = diff <= bound
        details = f"Got: {got_val:.4e} Expected: {expected_val:.4e} diff: {diff:.4e} rel_tol: {rel_tol}"

    return VerificationResult(passed=passed, details=details)


@verifier_registry.register("content:json_output")
def verify_json_content(got: str, expected: dict) -> VerificationResult:

    try:
        got_json = json.loads(got)
    except json.JSONDecodeError as e:
        return VerificationResult(passed=False, details=f"Failed to parse got: {got} \n as json: \n{str(e)}")

    for field, value in expected.items():
        if field not in got_json:
            return VerificationResult(passed=False, details=f"got: {got}, but expected field {field} is not included")
        if value != got_json[field]:
            return VerificationResult(
                passed=False, details=f"got: {got}, but field {field} does not match expectation:{value}"
            )
    return VerificationResult(passed=True, details="Received Valid Json and the expected fields match in value")


@verifier_registry.register("format:json_output")
def json_output(got: str) -> VerificationResult:
    try:
        json.loads(got)
    except json.JSONDecodeError as e:
        return VerificationResult(passed=False, details=f"Failed to parse as json: {e}")
    return VerificationResult(passed=True, details="Valid Json provided")


@verifier_registry.register("judge:rubric")
def judge_rubric(got: str, rubric: str, context: VerifierContext) -> VerificationResult:
    """
    Grade the whole run with an LLM judge: does the trace satisfy `rubric`?

    The judge is a LiteLLM-served model configured through the run config's `judge` section; it
    is sent the rubric, the run's trace and a hardcoded output specification, and must reply
    with a JSON object holding `evaluation` (the score) and `justification`. Both end up in the
    check's metadata, so every judged run records why it was passed or failed.

    Args:
        got (str): the agent's final answer (not sent to the judge on its own; it is part of the trace)
        rubric (str): the question the judge has to answer about the trace
        context (VerifierContext): supplies the trace and the configured judge client

    Returns (VerificationResult): the judge's verdict, with its justification attached
    """
    if context is None or context.judge is None:
        reason = context.judge_error if context is not None and context.judge_error else "no judge model is configured"
        return VerificationResult(
            passed=False,
            details=f"LLM judge unavailable: {reason}. Add a 'judge' section to the run config to use judge:rubric.",
            metadata={"rubric": rubric},
        )

    judge_model = getattr(context.judge, "model_id", "unknown")
    metadata = {"rubric": rubric, "judge_model": judge_model}

    try:
        verdict = context.judge.judge(rubric, context.trace)
    except JudgeReplyError as exc:
        return VerificationResult(
            passed=False,
            details=f"LLM judge ({judge_model}) reply did not follow the required format: {exc}",
            metadata=metadata,
        )
    except Exception as exc:
        return VerificationResult(
            passed=False,
            details=f"LLM judge ({judge_model}) call failed: {type(exc).__name__}: {exc}",
            metadata=metadata,
        )

    return VerificationResult(
        passed=verdict.passed,
        details=f"LLM judge ({judge_model}) {'passed' if verdict.passed else 'failed'}: {verdict.justification}",
        justification=verdict.justification,
        metadata=metadata,
    )
