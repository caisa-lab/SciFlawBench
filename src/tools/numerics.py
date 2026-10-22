"""
A numpy-backed numerics tool.

numpy runs here, in harness code: the agent passes numbers and (nested) lists of numbers and gets
JSON back. It never handles numpy objects, files or code, so the code executor's restrictions
are unaffected.
"""

import json
import math

import numpy as np
from smolagents import Tool

MAX_VALUES = 1_000_000  # numbers accepted in one argument
MAX_DIM = 500  # matrix side for solve / eigenvalues / covariance
MAX_INVERSE_DIM = 30  # inverse returns N*N numbers, kept within MAX_RETURNED
MAX_FIT_PARAMS = 30  # fit returns a K*K covariance, kept within MAX_RETURNED
MAX_SAMPLES = 1_000_000
MAX_RETURNED = 1_000  # numbers returned in one list
MAX_LIST_OPTIONS = 20  # percentiles / thresholds

DISTRIBUTIONS = {
    "normal": ("mean", "std"),
    "uniform": ("low", "high"),
    "lognormal": ("mean", "sigma"),
    "loguniform": ("low", "high"),
    "exponential": ("scale",),
    "poisson": ("lam",),
}


class NumericsError(ValueError):
    """Invalid input; reported to the agent as an error message."""


def _array(value, name: str, ndim: int) -> np.ndarray:
    """Validate a number or nested list of numbers and return it as a float array of `ndim` dimensions."""
    stack, count = [value], 0
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            raise NumericsError(f"{name} must contain numbers, not booleans")
        if isinstance(item, (int, float)):
            count += 1
            if count > MAX_VALUES:
                raise NumericsError(f"{name} has more than {MAX_VALUES} values")
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            raise NumericsError(f"{name} must be a number or a (nested) list of numbers")
    try:
        array = np.array(value, dtype=float)
    except OverflowError as exc:  # a Python int beyond the float range
        raise NumericsError(f"{name} contains a number too large to represent") from exc
    except ValueError as exc:
        raise NumericsError(f"{name} must be rectangular (all rows the same length)") from exc
    if array.ndim != ndim:
        raise NumericsError(f"{name} must have {ndim} dimension(s), got {array.ndim}")
    if array.size == 0:
        raise NumericsError(f"{name} is empty")
    if not np.all(np.isfinite(array)):
        raise NumericsError(f"{name} contains inf or nan")
    return array


def _square(value, name: str, max_dim: int) -> np.ndarray:
    matrix = _array(value, name, 2)
    if matrix.shape[0] != matrix.shape[1]:
        raise NumericsError(f"{name} must be square, got {matrix.shape[0]}x{matrix.shape[1]}")
    if matrix.shape[0] > max_dim:
        raise NumericsError(f"{name} is larger than {max_dim}x{max_dim}")
    return matrix


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericsError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:  # a Python int beyond the float range
        raise NumericsError(f"{name} is too large to represent") from exc
    if not math.isfinite(number):
        raise NumericsError(f"{name} must be a finite number")
    return number


def _small_list(value, name: str, low: float, high: float) -> list[float]:
    values = [] if value is None else _array(value, name, 1).tolist()
    if len(values) > MAX_LIST_OPTIONS:
        raise NumericsError(f"{name} accepts at most {MAX_LIST_OPTIONS} values")
    if any(not low <= v <= high for v in values):
        raise NumericsError(f"{name} values must lie between {low} and {high}")
    return values


def solve(matrix, vector) -> dict:
    a = _square(matrix, "matrix", MAX_DIM)
    b = _array(vector, "vector", 1)
    if b.shape[0] != a.shape[0]:
        raise NumericsError("vector length must equal the matrix size")
    return {"solution": np.linalg.solve(a, b).tolist()}


def inverse(matrix) -> dict:
    return {"inverse": np.linalg.inv(_square(matrix, "matrix", MAX_INVERSE_DIM)).tolist()}


def symmetric_eigenvalues(matrix) -> dict:
    a = _square(matrix, "matrix", MAX_DIM)
    if not np.allclose(a, a.T, rtol=1e-10, atol=1e-12 * float(np.max(np.abs(a)))):
        raise NumericsError("matrix must be symmetric")
    return {"eigenvalues": np.linalg.eigvalsh(a).tolist()}


def least_squares_fit(design, data, sigma=None, covariance=None) -> dict:
    x = _array(design, "design", 2)
    y = _array(data, "data", 1)
    m, k = x.shape
    if y.shape[0] != m:
        raise NumericsError("data length must equal the number of design rows")
    if k > MAX_FIT_PARAMS:
        raise NumericsError(f"design has more than {MAX_FIT_PARAMS} columns")
    if m <= k:
        raise NumericsError("need more data points than fit parameters")
    if sigma is not None and covariance is not None:
        raise NumericsError("give either sigma or covariance, not both")

    note = None
    if covariance is not None:
        c = _square(covariance, "covariance", MAX_DIM)
        if c.shape[0] != m:
            raise NumericsError("covariance size must equal the data length")
        try:
            np.linalg.cholesky(c)
        except np.linalg.LinAlgError as exc:
            raise NumericsError("covariance must be symmetric positive-definite") from exc
        ci_x, ci_y = np.linalg.solve(c, x), np.linalg.solve(c, y)
    else:
        if sigma is not None:
            s = _array(sigma, "sigma", 1)
            if s.shape[0] != m or np.any(s <= 0):
                raise NumericsError("sigma must have one positive value per data point")
            w = 1.0 / s**2
        else:
            w = np.ones(m)
            note = "no sigma or covariance given: errors assume unit uncertainty on each data point"
        ci_x, ci_y = x * w[:, None], y * w

    fisher = x.T @ ci_x
    param_cov = np.linalg.inv(fisher)
    coefficients = param_cov @ (x.T @ ci_y)
    residual = y - x @ coefficients
    chi2 = float(residual @ (np.linalg.solve(c, residual) if covariance is not None else residual * w))
    result = {
        "coefficients": coefficients.tolist(),
        "errors": np.sqrt(np.diag(param_cov)).tolist(),
        "covariance": param_cov.tolist(),
        "chi2": chi2,
        "dof": m - k,
    }
    if note:
        result["note"] = note
    return result


def sample_summary(distribution, parameters, n, seed=0, percentiles=None, thresholds=None, return_samples=0) -> dict:
    if distribution not in DISTRIBUTIONS:
        raise NumericsError(f"distribution must be one of {sorted(DISTRIBUTIONS)}")
    if not isinstance(parameters, dict) or set(parameters) != set(DISTRIBUTIONS[distribution]):
        raise NumericsError(f"{distribution} needs parameters {list(DISTRIBUTIONS[distribution])}")
    p = {name: _number(value, name) for name, value in parameters.items()}
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_SAMPLES:
        raise NumericsError(f"n must be an integer between 1 and {MAX_SAMPLES}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise NumericsError("seed must be a non-negative integer")
    if (
        isinstance(return_samples, bool)
        or not isinstance(return_samples, int)
        or not 0 <= return_samples <= MAX_RETURNED
    ):
        raise NumericsError(f"return_samples must be an integer between 0 and {MAX_RETURNED}")
    pct = _small_list([5, 16, 50, 84, 95] if percentiles is None else percentiles, "percentiles", 0, 100)
    cuts = _small_list(thresholds, "thresholds", -math.inf, math.inf)

    rng = np.random.default_rng(seed)
    if distribution == "normal":
        if p["std"] <= 0:
            raise NumericsError("std must be positive")
        x = rng.normal(p["mean"], p["std"], n)
    elif distribution == "uniform":
        if p["low"] >= p["high"]:
            raise NumericsError("low must be below high")
        x = rng.uniform(p["low"], p["high"], n)
    elif distribution == "lognormal":
        if p["sigma"] <= 0:
            raise NumericsError("sigma must be positive")
        x = rng.lognormal(p["mean"], p["sigma"], n)
    elif distribution == "loguniform":
        if not 0 < p["low"] < p["high"]:
            raise NumericsError("need 0 < low < high")
        x = np.exp(rng.uniform(math.log(p["low"]), math.log(p["high"]), n))
    elif distribution == "exponential":
        if p["scale"] <= 0:
            raise NumericsError("scale must be positive")
        x = rng.exponential(p["scale"], n)
    else:
        if not 0 <= p["lam"] <= 1e9:
            raise NumericsError("lam must lie between 0 and 1e9")
        x = rng.poisson(p["lam"], n).astype(float)

    result = {
        "n": n,
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if n > 1 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
        "percentiles": {f"{q:g}": float(v) for q, v in zip(pct, np.percentile(x, pct), strict=True)} if pct else {},
    }
    if cuts:
        result["fraction_below"] = {f"{t:g}": float(np.mean(x < t)) for t in cuts}
    if return_samples:
        result["samples"] = x[:return_samples].tolist()
    return result


OPERATIONS = {
    "solve": solve,
    "inverse": inverse,
    "symmetric_eigenvalues": symmetric_eigenvalues,
    "least_squares_fit": least_squares_fit,
    "sample_summary": sample_summary,
}


class NumericsTool(Tool):
    name = "numerics"

    description = f"""Runs a fixed numerical operation with numpy and returns JSON. Pass numbers and (nested) lists of
    numbers only. Operations and their arguments:
    - solve: matrix (NxN, N <= {MAX_DIM}), vector (N) -> solution of matrix @ x = vector
    - inverse: matrix (NxN, N <= {MAX_INVERSE_DIM}) -> inverse
    - symmetric_eigenvalues: matrix (symmetric NxN, N <= {MAX_DIM}) -> eigenvalues, ascending
    - least_squares_fit: design (MxK, K <= {MAX_FIT_PARAMS}), data (M), and optionally sigma (M) or covariance
      (MxM, M <= {MAX_DIM}) -> coefficients, 1-sigma errors, parameter covariance, chi2, dof
    - sample_summary: distribution ({", ".join(DISTRIBUTIONS)}), parameters (normal: mean, std; uniform and
      loguniform: low, high; lognormal: mean, sigma of the log; exponential: scale; poisson: lam), n (<= {MAX_SAMPLES}),
      seed (default 0), optional percentiles (default [5, 16, 50, 84, 95]), thresholds (-> fraction_below each) and
      return_samples (first <= {MAX_RETURNED} draws) -> summary statistics"""

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
        except NumericsError as exc:
            return f"Error: {exc}"
        except np.linalg.LinAlgError as exc:
            return f"Error: linear algebra failed ({exc}); the matrix may be singular"
        try:
            return json.dumps(result, allow_nan=False)
        except ValueError:  # inf or nan in the result: the computation overflowed
            return "Error: result is not finite; the inputs may be too large"
