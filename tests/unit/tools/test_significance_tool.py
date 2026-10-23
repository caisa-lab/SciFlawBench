import json
import math

import pytest

from tools.significance import MAX_VALUES, SignificanceTool

tool = SignificanceTool()


def run(operation, **arguments):
    out = tool.forward(operation, arguments)
    assert not out.startswith("Error"), out
    return json.loads(out)


def test_p_to_sigma_one_and_two_sided():
    assert run("p_to_sigma", p=0.0455, sided="two")["sigma"] == pytest.approx(2.0, abs=1e-4)
    assert run("p_to_sigma", p=0.0013499, sided="one")["sigma"] == pytest.approx(3.0, abs=1e-4)
    assert run("p_to_sigma", p=2.8665e-7, sided="one")["sigma"] == pytest.approx(5.0, abs=1e-4)


def test_sigma_to_p_one_and_two_sided():
    assert run("sigma_to_p", sigma=3, sided="two")["p_value"] == pytest.approx(0.0026998, rel=1e-4)
    assert run("sigma_to_p", sigma=5, sided="one")["p_value"] == pytest.approx(2.8665e-7, rel=1e-4)
    assert run("sigma_to_p", sigma=-2, sided="two")["p_value"] == run("sigma_to_p", sigma=2, sided="two")["p_value"]


@pytest.mark.parametrize("sided", ["one", "two"])
def test_p_and_sigma_round_trip(sided):
    for p in (0.3, 1e-3, 1e-12, 1e-200):
        sigma = run("p_to_sigma", p=p, sided=sided)["sigma"]
        assert run("sigma_to_p", sigma=sigma, sided=sided)["p_value"] == pytest.approx(p, rel=1e-6)


def test_chi2_probability():
    out = run("chi2_probability", chi2=12.9, dof=2)
    assert out["p_value"] == pytest.approx(math.exp(-12.9 / 2), rel=1e-9)  # chi2 with 2 dof is exponential
    assert out["sigma_two_sided"] == pytest.approx(3.1595, abs=1e-3)
    assert out["chi2_per_dof"] == pytest.approx(6.45)
    assert run("chi2_probability", chi2=0, dof=3)["p_value"] == 1.0


def test_chi2_quantile():
    assert run("chi2_quantile", probability=0.6827, dof=1)["chi2"] == pytest.approx(1.0, abs=1e-3)
    assert run("chi2_quantile", probability=0.6827, dof=2)["chi2"] == pytest.approx(2.30, abs=1e-2)
    assert run("chi2_quantile", probability=0.9545, dof=2)["chi2"] == pytest.approx(6.18, abs=1e-2)


def test_cdf_and_quantile_of_each_distribution():
    normal = {"mean": 0, "std": 1}
    assert run("cdf", distribution="normal", parameters=normal, values=[0])["cdf"] == [0.5]
    assert run("quantile", distribution="normal", parameters=normal, probabilities=[0.975])["quantile"][0] == (
        pytest.approx(1.959964, abs=1e-6)
    )
    lognormal = {"mean": 1, "sigma": 0.5}
    assert run("quantile", distribution="lognormal", parameters=lognormal, probabilities=[0.5])["quantile"][0] == (
        pytest.approx(math.e)
    )
    assert run("quantile", distribution="uniform", parameters={"low": 2, "high": 4}, probabilities=[0, 0.25, 1])[
        "quantile"
    ] == pytest.approx([2, 2.5, 4])
    loguniform = {"low": 1, "high": 100}
    assert run("quantile", distribution="loguniform", parameters=loguniform, probabilities=[0.5])["quantile"][0] == (
        pytest.approx(10)
    )
    assert run("cdf", distribution="exponential", parameters={"scale": 2}, values=[2])["cdf"][0] == pytest.approx(
        1 - math.exp(-1)
    )
    assert run("cdf", distribution="poisson", parameters={"lam": 3}, values=[0])["cdf"][0] == pytest.approx(
        math.exp(-3)
    )
    assert run("cdf", distribution="chi2", parameters={"dof": 2}, values=[12.9])["cdf"][0] == pytest.approx(
        1 - math.exp(-6.45)
    )
    assert run("quantile", distribution="student_t", parameters={"dof": 1000}, probabilities=[0.975])["quantile"][
        0
    ] == pytest.approx(1.962, abs=1e-3)


def test_lists_are_accepted_up_to_the_limit():
    values = list(range(MAX_VALUES))
    out = run("cdf", distribution="normal", parameters={"mean": 0, "std": 1}, values=values)
    assert len(out["cdf"]) == MAX_VALUES


@pytest.mark.parametrize(
    "operation, arguments, message",
    [
        ("p_to_sigma", {"p": 0, "sided": "two"}, "strictly between 0 and 1"),
        ("p_to_sigma", {"p": 1, "sided": "one"}, "strictly between 0 and 1"),
        ("p_to_sigma", {"p": "0.05", "sided": "two"}, "finite number"),
        ("p_to_sigma", {"p": True, "sided": "two"}, "finite number"),
        ("p_to_sigma", {"p": 0.05}, "missing 1 required positional argument: 'sided'"),
        ("p_to_sigma", {"p": 0.05, "sided": "both"}, 'sided must be "one" or "two"'),
        ("sigma_to_p", {"sigma": 50, "sided": "one"}, "smallest representable"),
        ("sigma_to_p", {"sigma": 10**400, "sided": "one"}, "too large"),
        ("sigma_to_p", {"sigma": float("nan"), "sided": "one"}, "finite number"),
        ("chi2_probability", {"chi2": -1, "dof": 2}, "must not be negative"),
        ("chi2_probability", {"chi2": 1e6, "dof": 1}, "smallest representable"),
        ("chi2_probability", {"chi2": 3, "dof": 0}, "dof must be an integer"),
        ("chi2_probability", {"chi2": 3, "dof": 1.5}, "dof must be an integer"),
        ("chi2_probability", {"chi2": 3, "dof": True}, "dof must be an integer"),
        ("chi2_quantile", {"probability": 1.2, "dof": 2}, "strictly between 0 and 1"),
        ("cdf", {"distribution": "cauchy", "parameters": {}, "values": [0]}, "distribution must be one of"),
        ("cdf", {"distribution": "normal", "parameters": {"mean": 0}, "values": [0]}, "needs parameters"),
        ("cdf", {"distribution": "normal", "parameters": {"mean": 0, "std": 0}, "values": [0]}, "std must be positive"),
        ("cdf", {"distribution": "normal", "parameters": {"mean": 0, "std": 1}, "values": []}, "non-empty list"),
        ("cdf", {"distribution": "normal", "parameters": {"mean": 0, "std": 1}, "values": 0.5}, "non-empty list"),
        (
            "cdf",
            {"distribution": "normal", "parameters": {"mean": 0, "std": 1}, "values": [0] * (MAX_VALUES + 1)},
            "at most",
        ),
        ("cdf", {"distribution": "lognormal", "parameters": {"mean": 800, "sigma": 1}, "values": [1]}, "between -700"),
        ("cdf", {"distribution": "loguniform", "parameters": {"low": 0, "high": 1}, "values": [1]}, "0 < low < high"),
        ("cdf", {"distribution": "chi2", "parameters": {"dof": -1}, "values": [1]}, "dof must be an integer"),
        (
            "quantile",
            {"distribution": "normal", "parameters": {"mean": 0, "std": 1}, "probabilities": [1]},
            "not finite",
        ),
        (
            "quantile",
            {"distribution": "normal", "parameters": {"mean": 0, "std": 1}, "probabilities": [1.5]},
            "between 0 and 1",
        ),
        ("p_to_sigma", {"p": 0.05, "sided": "two", "extra": 1}, "unexpected keyword"),
        ("unknown_op", {}, "unknown operation"),
    ],
)
def test_bad_input_returns_an_error_message(operation, arguments, message):
    out = tool.forward(operation, arguments)
    assert out.startswith("Error") and message in out


def test_arguments_must_be_a_dict():
    assert tool.forward("p_to_sigma", [0.05, "two"]).startswith("Error: arguments must be a dict")


def test_registered_in_the_tool_registry():
    from core.events import EventWatcher
    from tools.definitions import tool_registry

    significance = tool_registry.create("significance", watcher=EventWatcher("t", sink=lambda e: None))
    assert significance.name == "significance"


def test_description_says_how_to_read_the_output():
    # descriptions are wrapped over several lines, so compare with the whitespace collapsed
    assert "JSON string (parse it with json.loads)" in " ".join(SignificanceTool.description.split())
