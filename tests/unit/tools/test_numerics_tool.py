import json

import numpy as np
import pytest

from tools.numerics import NumericsTool

tool = NumericsTool()


def run(operation, **arguments):
    out = tool.forward(operation, arguments)
    assert not out.startswith("Error"), out
    return json.loads(out)


def test_solve_and_inverse():
    assert run("solve", matrix=[[2, 1], [1, 3]], vector=[3, 5])["solution"] == pytest.approx([0.8, 1.4])
    assert np.allclose(run("inverse", matrix=[[2, 1], [1, 3]])["inverse"], [[0.6, -0.2], [-0.2, 0.4]])


def test_symmetric_eigenvalues():
    assert run("symmetric_eigenvalues", matrix=[[2, 1], [1, 2]])["eigenvalues"] == pytest.approx([1.0, 3.0])


def test_line_fit_with_sigma():
    x = [0.0, 1.0, 2.0, 3.0]
    result = run("least_squares_fit", design=[[1, xi] for xi in x], data=[1.0, 3.0, 5.0, 7.0], sigma=[0.5] * 4)
    assert result["coefficients"] == pytest.approx([1.0, 2.0])
    assert result["chi2"] == pytest.approx(0.0, abs=1e-12)
    assert result["dof"] == 2


def test_template_amplitude_with_full_covariance():
    # one-parameter template fit, checked against the generalised-least-squares formula
    t = np.array([-1.0, -0.6, -0.2, 0.1, 0.05])
    y = 0.8 * t + np.array([0.02, -0.01, 0.03, 0.0, -0.02])
    c = 0.01 * (0.5 * np.eye(5) + 0.5 * np.exp(-np.abs(np.subtract.outer(range(5), range(5)))))
    result = run("least_squares_fit", design=t[:, None].tolist(), data=y.tolist(), covariance=c.tolist())
    ci = np.linalg.inv(c)
    fisher = t @ ci @ t
    assert result["coefficients"][0] == pytest.approx((t @ ci @ y) / fisher)
    assert result["errors"][0] == pytest.approx(fisher**-0.5)


def test_unweighted_fit_says_errors_assume_unit_uncertainty():
    assert "note" in run("least_squares_fit", design=[[1], [1], [1]], data=[1.0, 2.0, 3.0])


def test_sample_summary_is_seeded_and_reports_fractions():
    args = dict(distribution="normal", parameters={"mean": 0, "std": 1}, n=200_000, seed=3, thresholds=[0, 1.96])
    first, second = run("sample_summary", **args), run("sample_summary", **args)
    assert first == second
    assert first["mean"] == pytest.approx(0.0, abs=0.01)
    assert first["fraction_below"]["0"] == pytest.approx(0.5, abs=0.01)
    assert first["fraction_below"]["1.96"] == pytest.approx(0.975, abs=0.005)


def test_loguniform_range_and_returned_samples():
    out = run(
        "sample_summary", distribution="loguniform", parameters={"low": 0.2, "high": 10}, n=1000, return_samples=10
    )
    assert out["min"] >= 0.2 and out["max"] <= 10 and len(out["samples"]) == 10


@pytest.mark.parametrize(
    "operation, arguments, message",
    [
        ("solve", {"matrix": [["1", 0], [0, 1]], "vector": [1, 1]}, "numbers"),
        ("solve", {"matrix": [[True, 0], [0, 1]], "vector": [1, 1]}, "booleans"),
        ("solve", {"matrix": [[1, 0], [0]], "vector": [1, 1]}, "rectangular"),
        ("solve", {"matrix": [[0, 0], [0, 0]], "vector": [1, 1]}, "singular"),
        ("inverse", {"matrix": np.eye(31).tolist()}, "larger than"),
        ("symmetric_eigenvalues", {"matrix": [[1, 2], [0, 1]]}, "symmetric"),
        (
            "least_squares_fit",
            {"design": [[1], [1]], "data": [1, 2], "covariance": [[1, 2], [2, 1]]},
            "positive-definite",
        ),
        ("sample_summary", {"distribution": "cauchy", "parameters": {}, "n": 10}, "distribution must be"),
        ("sample_summary", {"distribution": "normal", "parameters": {"mean": 0, "std": 1}, "n": 10**7}, "n must be"),
        ("solve", {"matrix": [[1]], "vector": [1], "extra": 1}, "unexpected keyword"),
        (
            "sample_summary",
            {"distribution": "lognormal", "parameters": {"mean": 1000, "sigma": 1}, "n": 10},
            "not finite",
        ),
        ("inverse", {"matrix": [[1e-320]]}, "not finite"),
        ("solve", {"matrix": [[10**400]], "vector": [1]}, "too large"),
        ("sample_summary", {"distribution": "normal", "parameters": {"mean": 10**400, "std": 1}, "n": 10}, "too large"),
        ("unknown_op", {}, "unknown operation"),
    ],
)
def test_bad_input_returns_an_error_message(operation, arguments, message):
    out = tool.forward(operation, arguments)
    assert out.startswith("Error") and message in out


def test_registered_in_the_tool_registry():
    from core.events import EventWatcher
    from tools.definitions import tool_registry

    numerics = tool_registry.create("numerics", watcher=EventWatcher("t", sink=lambda e: None))
    assert numerics.name == "numerics"
