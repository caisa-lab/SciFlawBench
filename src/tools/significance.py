"""
A scipy-backed tool for statistical significance.

It converts between p-values, Gaussian-equivalent significances and chi-squared values, and gives exact
cumulative probabilities and quantiles of common distributions. scipy runs here, in harness code: the agent
passes numbers and names from a fixed list and gets JSON back. It never handles scipy objects, files or code,
so the code executor's restrictions are unaffected.

scipy is imported inside the functions that use it, not at module level: the tool registry imports this module on
every harness start, and importing scipy takes one to two seconds that only runs using this tool should pay.
"""

import json
import math

from smolagents import Tool

MAX_VALUES = 1_000  # values or probabilities in one call, and numbers returned in one list
MAX_DOF = 1_000_000

SIDES = ("one", "two")

DISTRIBUTIONS = {
    "normal": ("mean", "std"),
    "lognormal": ("mean", "sigma"),
    "uniform": ("low", "high"),
    "loguniform": ("low", "high"),
    "exponential": ("scale",),
    "poisson": ("lam",),
    "chi2": ("dof",),
    "student_t": ("dof",),
}


class SignificanceError(ValueError):
    """Invalid input; reported to the agent as an error message."""


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignificanceError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:  # a Python int beyond the float range
        raise SignificanceError(f"{name} is too large to represent") from exc
    if not math.isfinite(number):
        raise SignificanceError(f"{name} must be a finite number")
    return number


def _numbers(values, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise SignificanceError(f"{name} must be a non-empty list of numbers")
    if len(values) > MAX_VALUES:
        raise SignificanceError(f"{name} accepts at most {MAX_VALUES} values")
    return [_number(value, name) for value in values]


def _probability(value, name: str, closed: bool = False) -> float:
    p = _number(value, name)
    if not (0 <= p <= 1 if closed else 0 < p < 1):
        raise SignificanceError(f"{name} must lie {'between 0 and 1' if closed else 'strictly between 0 and 1'}")
    return p


def _dof(value, name: str = "dof") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_DOF:
        raise SignificanceError(f"{name} must be an integer between 1 and {MAX_DOF}")
    return value


def _sided(value) -> str:
    if value not in SIDES:
        raise SignificanceError('sided must be "one" or "two"')
    return value


def _distribution(distribution, parameters):
    """The frozen scipy distribution for a name and its parameters, after validating both."""
    from scipy import stats

    if distribution not in DISTRIBUTIONS:
        raise SignificanceError(f"distribution must be one of {sorted(DISTRIBUTIONS)}")
    if not isinstance(parameters, dict) or set(parameters) != set(DISTRIBUTIONS[distribution]):
        raise SignificanceError(f"{distribution} needs parameters {list(DISTRIBUTIONS[distribution])}")
    if distribution in ("chi2", "student_t"):
        return (stats.chi2 if distribution == "chi2" else stats.t)(df=_dof(parameters["dof"]))
    p = {name: _number(value, name) for name, value in parameters.items()}
    if distribution == "normal":
        if p["std"] <= 0:
            raise SignificanceError("std must be positive")
        return stats.norm(loc=p["mean"], scale=p["std"])
    if distribution == "lognormal":
        if p["sigma"] <= 0:
            raise SignificanceError("sigma must be positive")
        if abs(p["mean"]) > 700:
            raise SignificanceError("mean (of the log) must lie between -700 and 700")
        return stats.lognorm(s=p["sigma"], scale=math.exp(p["mean"]))
    if distribution == "uniform":
        if p["low"] >= p["high"]:
            raise SignificanceError("low must be below high")
        if not math.isfinite(p["high"] - p["low"]):
            raise SignificanceError("high - low is too large to represent")
        return stats.uniform(loc=p["low"], scale=p["high"] - p["low"])
    if distribution == "loguniform":
        if not 0 < p["low"] < p["high"]:
            raise SignificanceError("need 0 < low < high")
        return stats.loguniform(p["low"], p["high"])
    if distribution == "exponential":
        if p["scale"] <= 0:
            raise SignificanceError("scale must be positive")
        return stats.expon(scale=p["scale"])
    if not 0 <= p["lam"] <= 1e9:
        raise SignificanceError("lam must lie between 0 and 1e9")
    return stats.poisson(mu=p["lam"])


def p_to_sigma(p, sided) -> dict:
    from scipy import stats

    p_value, sided = _probability(p, "p"), _sided(sided)
    sigma = stats.norm.isf(p_value / 2 if sided == "two" else p_value)
    return {"sigma": float(sigma), "p_value": p_value, "sided": sided}


def sigma_to_p(sigma, sided) -> dict:
    from scipy import stats

    z, sided = _number(sigma, "sigma"), _sided(sided)
    p_value = 2 * stats.norm.sf(abs(z)) if sided == "two" else stats.norm.sf(z)
    if p_value == 0:
        raise SignificanceError("the p-value is below the smallest representable number (|sigma| above about 37)")
    return {"p_value": float(p_value), "sigma": z, "sided": sided}


def chi2_probability(chi2, dof) -> dict:
    from scipy import stats

    value, dof = _number(chi2, "chi2"), _dof(dof)
    if value < 0:
        raise SignificanceError("chi2 must not be negative")
    p_value = stats.chi2.sf(value, dof)
    if p_value == 0:
        raise SignificanceError("the p-value is below the smallest representable number")
    return {
        "p_value": float(p_value),
        "sigma_two_sided": float(stats.norm.isf(p_value / 2)),
        "chi2_per_dof": value / dof,
    }


def chi2_quantile(probability, dof) -> dict:
    from scipy import stats

    q, dof = _probability(probability, "probability"), _dof(dof)
    return {"chi2": float(stats.chi2.ppf(q, dof))}


def cdf(distribution, parameters, values) -> dict:
    frozen = _distribution(distribution, parameters)
    points = _numbers(values, "values")
    return {"cdf": [float(c) for c in frozen.cdf(points)]}


def quantile(distribution, parameters, probabilities) -> dict:
    frozen = _distribution(distribution, parameters)
    qs = [_probability(q, "probabilities", closed=True) for q in _numbers(probabilities, "probabilities")]
    if distribution == "poisson" and 0 in qs:
        # scipy returns -1 here, one below the support; every probability above 0 has a proper quantile
        raise SignificanceError("the poisson quantile needs probabilities above 0")
    return {"quantile": [float(x) for x in frozen.ppf(qs)]}


OPERATIONS = {
    "p_to_sigma": p_to_sigma,
    "sigma_to_p": sigma_to_p,
    "chi2_probability": chi2_probability,
    "chi2_quantile": chi2_quantile,
    "cdf": cdf,
    "quantile": quantile,
}


class SignificanceTool(Tool):
    name = "significance"

    description = f"""Statistical significance and distribution functions, computed with scipy. Returns a JSON string
    (parse it with json.loads). Pass numbers, lists of numbers and names from the lists below. Operations and their
    arguments:
    - p_to_sigma: p (0 < p < 1), sided ("one" or "two") -> the Gaussian-equivalent significance sigma
    - sigma_to_p: sigma, sided ("one" or "two") -> the p-value
    - chi2_probability: chi2 (>= 0), dof (integer >= 1) -> p-value of a chi-squared at least this large, its
      two-sided Gaussian-equivalent sigma, and chi2 per degree of freedom
    - chi2_quantile: probability (0 < probability < 1), dof -> the chi-squared value with that cumulative probability
    - cdf: distribution, parameters, values (<= {MAX_VALUES}) -> P(X <= value) for each value
    - quantile: distribution, parameters, probabilities (<= {MAX_VALUES}, each from 0 to 1, above 0 for poisson) ->
      the value with that cumulative probability
    Distributions and their parameters: normal (mean, std), lognormal (mean, sigma of the log), uniform and
    loguniform (low, high), exponential (scale), poisson (lam), chi2 (dof), student_t (dof)."""

    inputs = {
        "operation": {"type": "string", "description": f"one of {', '.join(OPERATIONS)}"},
        "arguments": {"type": "object", "description": "the operation's arguments, as a dict of names to values"},
    }
    output_type = "string"

    def forward(self, operation: str, arguments: dict) -> str:
        if operation not in OPERATIONS:
            return f"Error: unknown operation {operation!r}; available: {', '.join(OPERATIONS)}"
        if not isinstance(arguments, dict):
            return "Error: arguments must be a dict of argument names to values"
        try:
            result = OPERATIONS[operation](**arguments)
        except TypeError as exc:  # missing or unexpected argument names
            return f"Error: {exc}"
        except SignificanceError as exc:
            return f"Error: {exc}"
        try:
            return json.dumps(result, allow_nan=False)
        except ValueError:  # inf or nan in the result, e.g. the quantile at probability 1 of an unbounded distribution
            return "Error: result is not finite; check the inputs (e.g. a quantile at probability 0 or 1)"
