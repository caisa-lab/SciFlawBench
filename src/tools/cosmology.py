"""
An astropy-backed cosmology tool.

It gives distances, times and the expansion rate at given redshifts in a flat ΛCDM or flat w0waCDM
cosmology. astropy runs here, in harness code: the agent passes numbers and gets JSON back. It never handles
astropy objects, files or code, so the code executor's restrictions are unaffected. Nothing used here
downloads data or writes files, and astropy's internet access is switched off before astropy is first used.

astropy is imported when the tool is first used, not at module level: the tool registry imports this module on
every harness start, and importing astropy takes one to two seconds that only runs using this tool should pay.
"""

import json
import math

from smolagents import Tool

MAX_REDSHIFTS = 1_000
MAX_REDSHIFT = 1_500.0

REQUIRED = ("H0", "Om0")
OPTIONAL = {"w0": -1.0, "wa": 0.0, "Tcmb0": 0.0}
LIMITS = {"H0": (0.0, 200.0), "Om0": (0.0, 1.0), "w0": (-3.0, 1.0), "wa": (-5.0, 5.0), "Tcmb0": (0.0, 10.0)}


class CosmologyError(ValueError):
    """Invalid input; reported to the agent as an error message."""


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CosmologyError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:  # a Python int beyond the float range
        raise CosmologyError(f"{name} is too large to represent") from exc
    if not math.isfinite(number):
        raise CosmologyError(f"{name} must be a finite number")
    return number


def _model(cosmology):
    """A flat ΛCDM model, or flat w0waCDM when w0 or wa differ from -1 and 0, from validated parameters."""
    if not isinstance(cosmology, dict):
        raise CosmologyError("cosmology must be a dict of parameters, e.g. {'H0': 70, 'Om0': 0.3}")
    missing = [name for name in REQUIRED if name not in cosmology]
    unknown = sorted(set(cosmology) - set(REQUIRED) - set(OPTIONAL))
    if missing or unknown:
        raise CosmologyError(f"cosmology needs {list(REQUIRED)}, optionally {list(OPTIONAL)}; got {sorted(cosmology)}")
    p = {**OPTIONAL, **{name: _number(value, name) for name, value in cosmology.items()}}
    for name, (low, high) in LIMITS.items():
        if not low <= p[name] <= high or (name == "H0" and p[name] == 0):
            raise CosmologyError(
                f"{name} must lie between {low:g} and {high:g}" + (" (and above 0)" if name == "H0" else "")
            )

    from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
    from astropy.utils.data import conf as astropy_data_conf

    # cosmology needs no remote data; make sure nothing in astropy tries to fetch any
    astropy_data_conf.allow_internet = False
    if p["w0"] == -1.0 and p["wa"] == 0.0:
        return FlatLambdaCDM(H0=p["H0"], Om0=p["Om0"], Tcmb0=p["Tcmb0"])
    return Flatw0waCDM(H0=p["H0"], Om0=p["Om0"], w0=p["w0"], wa=p["wa"], Tcmb0=p["Tcmb0"])


def _redshifts(values) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values:
        raise CosmologyError("redshifts must be a non-empty list of numbers")
    if len(values) > MAX_REDSHIFTS:
        raise CosmologyError(f"redshifts accepts at most {MAX_REDSHIFTS} values")
    zs = [_number(z, "redshifts") for z in values]
    if any(not 0 <= z <= MAX_REDSHIFT for z in zs):
        raise CosmologyError(f"redshifts must lie between 0 and {MAX_REDSHIFT:g}")
    return zs


def _values(quantity) -> list[float]:
    return [float(v) for v in quantity.value]


def distances(cosmology, redshifts) -> dict:
    model, zs = _model(cosmology), _redshifts(redshifts)
    return {
        "comoving_distance_mpc": _values(model.comoving_distance(zs)),
        "angular_diameter_distance_mpc": _values(model.angular_diameter_distance(zs)),
        "luminosity_distance_mpc": _values(model.luminosity_distance(zs)),
        "h": model.h,
    }


def times(cosmology, redshifts) -> dict:
    model, zs = _model(cosmology), _redshifts(redshifts)
    return {"lookback_time_gyr": _values(model.lookback_time(zs)), "age_gyr": _values(model.age(zs))}


def hubble(cosmology, redshifts) -> dict:
    model, zs = _model(cosmology), _redshifts(redshifts)
    return {"H_km_s_mpc": _values(model.H(zs)), "E": [float(e) for e in model.efunc(zs)]}


OPERATIONS = {"distances": distances, "times": times, "hubble": hubble}


class CosmologyTool(Tool):
    name = "cosmology"

    description = f"""Distances, times and expansion rate in a flat cosmology, computed with astropy. Returns a JSON
    string (parse it with json.loads). Every operation takes `cosmology`, a dict with H0 (km/s/Mpc) and Om0,
    optionally w0 and wa (flat w0waCDM; defaults -1 and 0, i.e. flat ΛCDM) and Tcmb0 (K; default 0, which leaves out
    radiation), and `redshifts`, a list
    of up to {MAX_REDSHIFTS} values from 0 to {MAX_REDSHIFT:g}. Operations:
    - distances -> comoving, angular-diameter and luminosity distance in Mpc (not Mpc/h) at each redshift, and h
    - times -> lookback time and age of the universe in Gyr at each redshift
    - hubble -> H(z) in km/s/Mpc and E(z) = H(z)/H0 at each redshift"""

    inputs = {
        "operation": {"type": "string", "description": f"one of {', '.join(OPERATIONS)}"},
        "arguments": {
            "type": "object",
            "description": "the operation's arguments: {'cosmology': {...}, 'redshifts': [...]}",
        },
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
        except CosmologyError as exc:
            return f"Error: {exc}"
        try:
            return json.dumps(result, allow_nan=False)
        except ValueError:  # inf or nan in the result
            return "Error: result is not finite; check the cosmological parameters"
