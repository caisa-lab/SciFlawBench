import json

import pytest

from evaluation.base import run_check
from evaluation.definitions import verifier_registry

def test_basic_logic_contains_str():
    result = run_check(verifier_registry.get("content:contains_str"), "Hello", expected="hell", case_sensitive=False )
    assert result.passed

    result = run_check(verifier_registry.get("content:contains_str"), "Hello", expected="Hell", case_sensitive=True )
    assert result.passed

    result = run_check(verifier_registry.get("content:contains_str"), "Hello", expected="hell", case_sensitive=True )
    assert not result.passed

    result = run_check(verifier_registry.get("content:contains_str"), "Hello", expected="akl;wejklr;", case_sensitive=True )
    assert not result.passed
    

def test_basic_logic_exact_str_match():
    result = run_check(verifier_registry.get("content:exact_str_match"), "Hello", expected="hello", case_sensitive=False )
    assert result.passed

    result = run_check(verifier_registry.get("content:exact_str_match"), "Hello", expected="Hello", case_sensitive=True )
    assert result.passed

    result = run_check(verifier_registry.get("content:exact_str_match"), "Hello", expected="hello", case_sensitive=True )
    assert not result.passed

    result = run_check(verifier_registry.get("content:exact_str_match"), "Hello", expected="alkjdsf;", case_sensitive=True )
    assert not result.passed

def test_basic_logic_numeric_match():
    result = run_check(verifier_registry.get("content:numeric_match"), "10 -19 80", expected="10", index=0)
    assert result.passed

    result = run_check(verifier_registry.get("content:numeric_match"), "10 -19 80", expected="11", index=0)
    assert not result.passed

    result = run_check(verifier_registry.get("content:numeric_match"), "10 -19 80", expected="-19", index=1)
    assert result.passed

    result = run_check(verifier_registry.get("content:numeric_match"), "10 -19 80", expected="80" )
    assert result.passed

    result = run_check(verifier_registry.get("content:numeric_match"), "10 -19 80", expected="83", tol=6.0 )
    assert result.passed

    result = run_check(verifier_registry.get("content:numeric_match"), "10 -19 80", expected="83", tol=2.0 )
    assert not result.passed


def test_basic_logic_numeric_range():
    result = run_check(verifier_registry.get("content:numeric_within_range"), "10 -19 80", minimum=5, maximum=15, index=0)
    assert result.passed

    result = run_check(verifier_registry.get("content:numeric_within_range"), "10 -19 80", minimum=15, maximum=30, index=0)
    assert not result.passed

    result = run_check(verifier_registry.get("content:numeric_within_range"), "10 -19 80", minimum=-19, maximum=15, index=1)
    assert result.passed

    result = run_check(verifier_registry.get("content:numeric_within_range"), "10 -19 80", minimum=80, maximum=90)
    assert result.passed

def test_basic_logic_scientific_notation():
    result = run_check(verifier_registry.get("content:scientific_notation"), "1.0x10^1, -1.9e10 1.9", expected="1e1",index=0)
    assert result.passed

    result = run_check(verifier_registry.get("content:scientific_notation"), "1.0x10^2 -1.9x10^0 1.9", expected="-1.9e0", index=1)
    assert result.passed
    
    result = run_check(verifier_registry.get("content:scientific_notation"), "jakl;dsfjal;fds 1.38x10^4", expected="1.38e4")
    assert result.passed

    result = run_check(verifier_registry.get("content:scientific_notation"), "jakl;dsfjal;fds 1.38x10^4", expected="1.38e5")
    assert not result.passed

    result = run_check(verifier_registry.get("content:scientific_notation"), "jakl;dsfjal;fds 1.38 x 10^4", expected="1.38e4")
    assert result.passed

    result = run_check(verifier_registry.get("content:scientific_notation"), "jakl;dsfjal;fds 1.38e4", expected="1.38e4")
    assert result.passed

def test_basic_json_ouptut():
    result = run_check(verifier_registry.get("content:json_output"), "{}", expected={})
    assert result.passed

    result = run_check(verifier_registry.get("content:json_output"), "{\"number\": 1}", expected={'number': 1})
    assert result.passed

    result = run_check(verifier_registry.get("content:json_output"), "{\"number\": 1}", expected={'num': 1})
    assert not result.passed

    result = run_check(verifier_registry.get("content:json_output"), "{\"number\": 1}", expected={'number': 10})
    assert not result.passed


def test_non_string_outputs_are_graded():
    # a code agent may hand back a python object rather than text, e.g. final_answer((2.7, 3.2))
    numeric = verifier_registry.get("content:numeric_match")

    result = run_check(numeric, (2.7, 3.2), expected="3.2", tol=0.05, index=1)
    assert result.passed

    result = run_check(numeric, (3.1, 2.8), expected="2.7", tol=0.05, index=0)
    assert not result.passed
    assert "Got value: 3.1" in result.details

    result = run_check(numeric, 2.7, expected="2.7", tol=0.0)
    assert result.passed

    result = run_check(numeric, [2.7, 3.2], expected="2.7", tol=0.0, index=0)
    assert result.passed

    result = run_check(verifier_registry.get("format:json_output"), {"papers": []})
    assert result.passed


def test_unicode_minus_is_a_negative_sign():
    numeric = verifier_registry.get("content:numeric_match")

    result = run_check(numeric, "(\u22120.54, \u22121.39)", expected="-0.54", tol=0.001, index=0)
    assert result.passed

    result = run_check(numeric, "(\u22120.821, \u22120.65)", expected="-0.65", tol=0.001, index=1)
    assert result.passed

    # an en dash marks a range, not a sign: the upper end must stay positive
    result = run_check(numeric, "2.8\u20134.2", expected="4.2", tol=0.001, index=1)
    assert result.passed


@pytest.mark.parametrize(
    ("got", "tol", "passed"),
    [
        pytest.param('{"mass": 12.3, "spin": 0.7}', 0.0, True, id="exact"),
        pytest.param('{"mass": 12.31, "spin": 0.7}', 0.0, False, id="no tolerance by default"),
        pytest.param('{"mass": 12.34, "spin": 0.7}', 0.05, True, id="within the tolerance"),
        pytest.param('{"mass": 12.36, "spin": 0.7}', 0.05, False, id="outside the tolerance"),
        pytest.param('{"mass": 12.3, "spin": 0.9}', 0.05, False, id="every field is graded"),
        pytest.param('{"mass": "12.3", "spin": 0.7}', 0.05, False, id="a number written as text"),
        pytest.param('{"mass": 12.3, "spin": 0.7, "justification": "x"}', 0.05, True, id="extra field"),
    ],
)
def test_json_output_grades_numbers_within_its_tolerance(got, tol, passed):
    verifier = verifier_registry.get("content:json_output")

    assert run_check(verifier, got, expected={"mass": 12.3, "spin": 0.7}, tol=tol).passed is passed


@pytest.mark.parametrize(
    ("expected", "got", "tol", "passed"),
    [
        pytest.param({"a": [1.0, 2.0]}, '{"a": [1.04, 1.96]}', 0.05, True, id="inside a list"),
        pytest.param({"a": [1.0, 2.0]}, '{"a": [1.0]}', 0.05, False, id="a shorter list"),
        pytest.param({"a": {"b": 1.0}}, '{"a": {"b": 1.04}}', 0.05, True, id="inside an object"),
        pytest.param(
            {"a": {"b": 1.0}}, '{"a": {"b": 1.0, "c": 2}}', 0.05, False, id="a nested object is compared whole"
        ),
        pytest.param({"a": "x"}, '{"a": "x"}', 0.05, True, id="text stays exact"),
        pytest.param({"a": "x"}, '{"a": "y"}', 0.05, False, id="other text"),
        pytest.param({"a": True}, '{"a": false}', 1.0, False, id="a boolean is not a number"),
    ],
)
def test_json_output_tolerance_reaches_inside_lists_and_objects(expected, got, tol, passed):
    verifier = verifier_registry.get("content:json_output")

    assert run_check(verifier, got, expected=expected, tol=tol).passed is passed


@pytest.mark.parametrize(
    "expected",
    [{"a": 1}, {"a": 1.0}, {"a": True}, {"a": None}, {"a": [1, "x", {"b": 2}]}, {"a": {"b": [1.5]}}],
)
def test_json_output_without_a_tolerance_matches_plain_equality(expected):
    verifier = verifier_registry.get("content:json_output")

    for got in ({"a": 1}, {"a": 1.0}, {"a": True}, {"a": None}, {"a": [1, "x", {"b": 2}]}, {"a": {"b": [1.5]}}):
        assert run_check(verifier, json.dumps(got), expected=expected).passed is (got["a"] == expected["a"])


@pytest.mark.parametrize(
    "got",
    ['```json\n{"a": 1}\n```', '```\n{"a": 1}\n```', '  ```JSON\n{"a": 1}\n```\n'],
)
def test_json_output_accepts_one_surrounding_code_block(got):
    assert run_check(verifier_registry.get("content:json_output"), got, expected={"a": 1}).passed


@pytest.mark.parametrize(
    "got",
    [
        'Here it is: {"a": 1}',
        'Here it is:\n```json\n{"a": 1}\n```',
        '```json\n{"a": 1}\n```\nThat is all.',
        '```json\n{"a": 1}\n```\n```json\n{"a": 1}\n```',
    ],
)
def test_json_output_still_refuses_text_around_the_json(got):
    assert not run_check(verifier_registry.get("content:json_output"), got, expected={"a": 1}).passed


def test_a_format_check_still_wants_bare_json():
    assert not run_check(verifier_registry.get("format:json_output"), '```json\n{"a": 1}\n```').passed


def test_paper_json_match_accepts_one_surrounding_code_block():
    got = '```json\n{"papers": [{"arxiv_id": "2602.02275"}]}\n```'

    assert run_check(
        verifier_registry.get("content:paper_json_match"), got, expected=[{"arxiv_id": "2602.02275"}]
    ).passed
