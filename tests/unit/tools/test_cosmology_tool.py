import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.cosmology import MAX_REDSHIFTS, CosmologyTool

tool = CosmologyTool()
C_KM_S = 299_792.458
HUBBLE_TIME_GYR_PER_UNIT = 977.792221  # 1 / (1 km/s/Mpc) in Gyr

LCDM = {"H0": 70, "Om0": 0.3}


def run(operation, **arguments):
    out = tool.forward(operation, arguments)
    assert not out.startswith("Error"), out
    return json.loads(out)


def e_of_z(z, om, w0=-1.0, wa=0.0):
    """Flat CPL expansion rate without radiation, written out independently of astropy."""
    de = (1 + z) ** (3 * (1 + w0 + wa)) * math.exp(-3 * wa * z / (1 + z))
    return math.sqrt(om * (1 + z) ** 3 + (1 - om) * de)


def simpson(f, a, b, n=4000):
    h = (b - a) / n
    total = f(a) + f(b) + 4 * sum(f(a + (2 * k - 1) * h) for k in range(1, n // 2 + 1))
    total += 2 * sum(f(a + 2 * k * h) for k in range(1, n // 2))
    return total * h / 3


def comoving_distance(z, h0, om, w0=-1.0, wa=0.0):
    return C_KM_S / h0 * simpson(lambda x: 1 / e_of_z(x, om, w0, wa), 0, z)


def test_distances_match_an_independent_integration():
    zs = [0.1, 0.5, 0.9, 2.0]
    out = run("distances", cosmology=LCDM, redshifts=zs)

    for z, dc, da, dl in zip(
        zs,
        out["comoving_distance_mpc"],
        out["angular_diameter_distance_mpc"],
        out["luminosity_distance_mpc"],
        strict=True,
    ):
        assert dc == pytest.approx(comoving_distance(z, 70, 0.3), rel=1e-6)
        assert da == pytest.approx(dc / (1 + z), rel=1e-9)  # flat: transverse comoving distance = comoving distance
        assert dl == pytest.approx(dc * (1 + z), rel=1e-9)
    assert out["h"] == pytest.approx(0.7)


def test_distances_are_in_mpc_not_mpc_over_h():
    # with H0 = 100 the numbers equal h^-1 Mpc; with H0 = 70 they are larger by 1/0.7
    per_h = run("distances", cosmology={"H0": 100, "Om0": 0.3}, redshifts=[0.9])["comoving_distance_mpc"][0]
    mpc = run("distances", cosmology=LCDM, redshifts=[0.9])["comoving_distance_mpc"][0]
    assert per_h == pytest.approx(2137.3, abs=0.1)
    assert mpc == pytest.approx(per_h / 0.7, rel=1e-9)


def test_times_match_the_closed_form_age_and_an_independent_lookback_time():
    out = run("times", cosmology=LCDM, redshifts=[0, 1])
    t_hubble = HUBBLE_TIME_GYR_PER_UNIT / 70
    age0 = t_hubble * 2 / (3 * math.sqrt(0.7)) * math.asinh(math.sqrt(0.7 / 0.3))
    lookback1 = t_hubble * simpson(lambda x: 1 / ((1 + x) * e_of_z(x, 0.3)), 0, 1)

    assert out["age_gyr"][0] == pytest.approx(age0, rel=1e-5)
    assert out["lookback_time_gyr"] == pytest.approx([0.0, lookback1], rel=1e-5)
    assert out["age_gyr"][1] == pytest.approx(age0 - lookback1, rel=1e-5)


def test_hubble_matches_the_cpl_formula():
    cosmology = {"H0": 67.5, "Om0": 0.31, "w0": -0.8, "wa": -0.6}
    zs = [0, 0.5, 2]
    out = run("hubble", cosmology=cosmology, redshifts=zs)

    assert out["E"] == pytest.approx([e_of_z(z, 0.31, -0.8, -0.6) for z in zs], rel=1e-9)
    assert out["H_km_s_mpc"] == pytest.approx([67.5 * e_of_z(z, 0.31, -0.8, -0.6) for z in zs], rel=1e-9)


def test_w0wa_distances_match_an_independent_integration():
    cosmology = {"H0": 70, "Om0": 0.3, "w0": -0.9, "wa": 0.2}
    dc = run("distances", cosmology=cosmology, redshifts=[1.0])["comoving_distance_mpc"][0]
    assert dc == pytest.approx(comoving_distance(1.0, 70, 0.3, -0.9, 0.2), rel=1e-6)


def test_explicit_lambda_values_give_the_lambda_cdm_result():
    explicit = run("distances", cosmology={**LCDM, "w0": -1, "wa": 0}, redshifts=[1])
    assert explicit == run("distances", cosmology=LCDM, redshifts=[1])


def test_tcmb0_adds_radiation_only_when_given():
    without = run("hubble", cosmology=LCDM, redshifts=[1000])["E"][0]
    with_radiation = run("hubble", cosmology={**LCDM, "Tcmb0": 2.7255}, redshifts=[1000])["E"][0]
    assert without == pytest.approx(e_of_z(1000, 0.3), rel=1e-9)
    assert with_radiation > 1.1 * without  # radiation matters at z = 1000


def test_lists_are_accepted_up_to_the_limit():
    zs = [i / MAX_REDSHIFTS for i in range(MAX_REDSHIFTS)]
    assert len(run("distances", cosmology=LCDM, redshifts=zs)["comoving_distance_mpc"]) == MAX_REDSHIFTS


@pytest.mark.parametrize(
    "operation, arguments, message",
    [
        ("distances", {"cosmology": {"H0": 70}, "redshifts": [1]}, "cosmology needs"),
        ("distances", {"cosmology": {**LCDM, "Ok0": 0.1}, "redshifts": [1]}, "cosmology needs"),
        ("distances", {"cosmology": [70, 0.3], "redshifts": [1]}, "must be a dict"),
        ("distances", {"cosmology": {"H0": 0, "Om0": 0.3}, "redshifts": [1]}, "H0 must lie"),
        ("distances", {"cosmology": {"H0": -70, "Om0": 0.3}, "redshifts": [1]}, "H0 must lie"),
        ("distances", {"cosmology": {"H0": 70, "Om0": 1.5}, "redshifts": [1]}, "Om0 must lie"),
        ("distances", {"cosmology": {**LCDM, "w0": -4}, "redshifts": [1]}, "w0 must lie"),
        ("distances", {"cosmology": {**LCDM, "wa": 9}, "redshifts": [1]}, "wa must lie"),
        ("distances", {"cosmology": {**LCDM, "Tcmb0": 300}, "redshifts": [1]}, "Tcmb0 must lie"),
        ("distances", {"cosmology": {"H0": "70", "Om0": 0.3}, "redshifts": [1]}, "finite number"),
        ("distances", {"cosmology": {"H0": True, "Om0": 0.3}, "redshifts": [1]}, "finite number"),
        ("distances", {"cosmology": {"H0": 10**400, "Om0": 0.3}, "redshifts": [1]}, "too large"),
        ("distances", {"cosmology": LCDM, "redshifts": [-0.1]}, "between 0 and 1500"),
        ("distances", {"cosmology": LCDM, "redshifts": [2000]}, "between 0 and 1500"),
        ("distances", {"cosmology": LCDM, "redshifts": []}, "non-empty list"),
        ("distances", {"cosmology": LCDM, "redshifts": 1.0}, "non-empty list"),
        ("distances", {"cosmology": LCDM, "redshifts": [float("nan")]}, "finite number"),
        ("distances", {"cosmology": LCDM, "redshifts": [0.5] * (MAX_REDSHIFTS + 1)}, "at most"),
        ("distances", {"cosmology": LCDM}, "missing 1 required positional argument"),
        ("distances", {"cosmology": LCDM, "redshifts": [1], "extra": 1}, "unexpected keyword"),
        ("comoving_volume", {}, "unknown operation"),
    ],
)
def test_bad_input_returns_an_error_message(operation, arguments, message):
    out = tool.forward(operation, arguments)
    assert out.startswith("Error") and message in out


def test_arguments_must_be_a_dict():
    assert tool.forward("distances", [LCDM, [1]]).startswith("Error: arguments must be a dict")


def test_tool_opens_no_network_connection_and_writes_no_files(tmp_path):
    # a fresh interpreter with an empty home directory and sockets that refuse to connect
    script = """
import json, socket, sys
def refuse(*args, **kwargs):
    raise RuntimeError("network access attempted")
socket.socket.connect = refuse
socket.create_connection = refuse
from tools.cosmology import CosmologyTool
tool = CosmologyTool()
args = {"cosmology": {"H0": 70, "Om0": 0.3, "w0": -0.9, "wa": 0.1, "Tcmb0": 2.7255}, "redshifts": [0.1, 1, 10]}
for operation in ("distances", "times", "hubble"):
    out = tool.forward(operation, args)
    assert not out.startswith("Error"), out
    json.loads(out)
print("ok")
"""
    home = tmp_path / "home"
    home.mkdir()
    src = Path(__file__).resolve().parents[3] / "src"
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "PYTHONPATH": str(src)}
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert list(home.rglob("*")) == []  # nothing was written under the home directory


def test_astropy_is_imported_only_when_the_tool_is_used():
    # The registry imports every tool module at harness start, so astropy must load on first use instead, with its
    # internet access already off. A fresh interpreter is needed: this test process has already imported astropy.
    code = (
        "import sys\n"
        "import tools.definitions\n"
        "assert 'astropy' not in sys.modules, 'astropy was imported at harness start'\n"
        "from tools.cosmology import CosmologyTool\n"
        "out = CosmologyTool().forward('hubble', {'cosmology': {'H0': 70, 'Om0': 0.3}, 'redshifts': [1]})\n"
        "assert not out.startswith('Error'), out\n"
        "from astropy.utils.data import conf\n"
        "assert conf.allow_internet is False\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_registered_in_the_tool_registry():
    from core.events import EventWatcher
    from tools.definitions import tool_registry

    cosmology = tool_registry.create("cosmology", watcher=EventWatcher("t", sink=lambda e: None))
    assert cosmology.name == "cosmology"


def test_description_says_how_to_read_the_output():
    # descriptions are wrapped over several lines, so compare with the whitespace collapsed
    assert "JSON string (parse it with json.loads)" in " ".join(CosmologyTool.description.split())
